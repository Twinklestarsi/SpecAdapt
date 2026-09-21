#!/usr/bin/env python3
"""Run Veri-Sure's non-ablation flow for every row in the C-first CSV.

The CSV intentionally contains repeated case IDs for different AREA/TIMING
objectives.  Every row therefore gets its own case directory, token ledger,
equivalence directory, and (when requested) DC report.

The upstream Veri-Sure CLI is invoked without ``--max-completion-tokens`` by
default.  Provider-reported usage is read from the patched CLI's
``token_usage.json``; no token estimate is made when a provider omits usage.
Golden RTL is never included in model prompts.  JasperGold is an external
post-generation check and DC metrics are kept separate from native Veri-Sure
simulation status.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INPUT_CSV = ROOT / "paper/C_RTLgen/ICCAD26-C-RTLgen-sub/doc/merged_token_usage_by_project_cfirst_rtl_direct_agents.csv"
PILOT_ROOT = ROOT / "runs/phase6/nvdla22_njg_20260908/pilot"
PILOT_MANIFEST = PILOT_ROOT / "pilot_manifest.json"
DEFAULT_RUN_ROOT = ROOT / "runs/external_spec_agents/20260920_verisure_csv_full"
ADAPTER = ROOT / "experiments/external_spec_agents/verisure_adapter.py"
VERISURE_PYTHON = ROOT / "external_agents/verisure/.venv/bin/python"
IVERILOG = ROOT / ".conda-env/bin/iverilog"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = reader.fieldnames or []
    required = {"objective", "project", "case_id"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def load_cases(input_csv: Path, run_root: Path) -> list[dict[str, Any]]:
    rows = read_rows(input_csv)
    manifest = read_json(PILOT_MANIFEST)
    by_id = {str(item["id"]): item for item in manifest.get("cases", [])}
    cases: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        source_id = str(row.get("case_id") or "").strip()
        objective = str(row.get("objective") or "").strip().upper()
        project = str(row.get("project") or "").strip()
        item = by_id.get(source_id)
        if item is None:
            raise KeyError(f"CSV row {row_number}: case_id is absent from pilot manifest: {source_id}")
        spec = (PILOT_ROOT / str(item.get("spec_path") or "")).resolve()
        golden = (PILOT_ROOT / str(item.get("golden_rtl_path") or "")).resolve()
        if not spec.is_file() or not golden.is_file():
            raise FileNotFoundError(f"CSV row {row_number}: missing spec/golden: {spec} / {golden}")
        top = str(item.get("golden_top_module") or "").strip()
        if not top:
            text = golden.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"^\s*module\s+([A-Za-z_]\w*)\b", text, re.MULTILINE)
            if not m:
                raise RuntimeError(f"CSV row {row_number}: cannot determine golden top: {golden}")
            top = m.group(1)
        case_key = safe_name(f"row_{row_number:02d}__{objective}__{project}__{source_id}")
        case = {
            "id": case_key,
            "csv_row": row_number,
            "source_case_id": source_id,
            "objective": objective,
            "project": project,
            "benchmark": source_id,
            "spec_path": str(spec),
            "golden_path": str(golden),
            "top_module": top,
            "golden_top_module": top,
            "design_type": str(item.get("design_type") or ""),
            "source_group": item.get("source_group"),
            "csv_fields": row,
            # Full mode is deliberate; no testbench/golden RTL is passed to
            # the model.  Equivalence is a later independent gate.
            "ablation": False,
            "golden_used_for_generation": False,
            "golden_used_for_verisure_simulation": False,
            "spec_sha256": sha256(spec),
            "golden_sha256": sha256(golden),
        }
        case_path = run_root / "cases" / f"{case_key}.json"
        write_json(case_path, case)
        case["case_path"] = str(case_path)
        cases.append(case)
    return cases


def native_status(log_path: Path) -> str:
    if not log_path.is_file():
        return "unknown"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # CLI prints a terminal standalone PASS/FAIL after all agent output.
    matches = re.findall(r"(?m)^(PASS|FAIL)\s*$", text)
    if matches:
        return matches[-1].lower()
    if "SIMULATION PASSED" in text:
        return "pass"
    return "fail" if "error" in text.lower() or "FAIL" in text else "unknown"


def syntax_check(candidate: Path, top: str, out_dir: Path) -> dict[str, Any]:
    log_path = out_dir / "syntax.log"
    command = [str(IVERILOG), "-g2012", "-s", top, "-t", "null", str(candidate)]
    started = time.time()
    try:
        cp = subprocess.run(command, cwd=str(out_dir), capture_output=True, text=True, timeout=120, check=False)
        output = (cp.stdout or "") + (cp.stderr or "")
        log_path.write_text(output, encoding="utf-8")
        return {
            "status": "passed" if cp.returncode == 0 else "failed",
            "returncode": cp.returncode,
            "elapsed_seconds": time.time() - started,
            "log_path": str(log_path),
        }
    except Exception as exc:  # noqa: BLE001
        log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {
            "status": "error",
            "returncode": None,
            "elapsed_seconds": time.time() - started,
            "log_path": str(log_path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_jg(candidate: Path, case: dict[str, Any], out_dir: Path, timeout: Optional[int]) -> dict[str, Any]:
    try:
        from module5.jg_verifier import verify_rtl_equivalence_with_jg

        return verify_rtl_equivalence_with_jg(
            candidate,
            Path(str(case["golden_path"])),
            out_dir,
            ROOT / ".env",
            golden_top=str(case["top_module"]),
            design_type=str(case.get("design_type") or ""),
            verification_timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "equivalent": False,
            "error": f"JG invocation: {type(exc).__name__}: {exc}",
        }


def run_dc(candidate: Path, case: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    try:
        from module5.dc_runner import run_dc_for_verilog

        return run_dc_for_verilog(
            candidate,
            benchmark=str(case["benchmark"]),
            goal=str(case["objective"]),
            stem=str(case["id"]),
            output_root=out_dir,
            top_module=str(case["top_module"]),
            max_cores=1,
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"DC invocation: {type(exc).__name__}: {exc}"}


def token_fields(generation: dict[str, Any]) -> dict[str, Any]:
    usage = generation.get("usage") if isinstance(generation, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "request_count": usage.get("request_count"),
        "token_source": usage.get("source"),
        "token_usage_path": usage.get("path"),
    }


def one_case(
    case: dict[str, Any],
    run_root: Path,
    config_path: Path,
    *,
    force: bool,
    verify_equivalence: bool,
    run_ppa: bool,
    verification_timeout: Optional[int],
    process_timeout: float,
) -> dict[str, Any]:
    out = run_root / "rows" / str(case["id"])
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "case_result.json"
    if result_path.is_file() and not force:
        try:
            return read_json(result_path)
        except Exception:
            pass
    started = time.time()
    generation_dir = out / "generation"
    adapter_log = out / "adapter.log"
    command = [
        str(VERISURE_PYTHON), str(ADAPTER),
        "--case-json", str(case["case_path"]),
        "--output-dir", str(generation_dir),
        "--config-json", str(config_path),
    ]
    proc: dict[str, Any]
    try:
        with adapter_log.open("w", encoding="utf-8") as log:
            cp = subprocess.run(
                command,
                cwd=str(ROOT),
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=process_timeout,
                check=False,
            )
        proc = {"returncode": cp.returncode, "timed_out": False}
    except subprocess.TimeoutExpired:
        proc = {"returncode": None, "timed_out": True}
    except Exception as exc:  # noqa: BLE001
        proc = {"returncode": None, "timed_out": False, "error": f"{type(exc).__name__}: {exc}"}

    adapter_result_path = generation_dir / "result.json"
    generation = read_json(adapter_result_path) if adapter_result_path.is_file() else {
        "status": "error", "rtl_path": None, "error": "adapter did not write result.json", "usage": {}
    }
    candidate_raw = generation.get("rtl_path")
    candidate = Path(str(candidate_raw)).resolve() if candidate_raw else None
    native = native_status(generation_dir / "verisure_run.log")
    result: dict[str, Any] = {
        "schema_version": "verisure_csv_full_case_result_v1",
        "agent": "xyjoey/Veri-Sure",
        "mode": "full_pipeline",
        "token_limit": "unset",
        "max_completion_tokens": None,
        "case_id": case["id"],
        "source_case_id": case["source_case_id"],
        "csv_row": case["csv_row"],
        "objective": case["objective"],
        "project": case["project"],
        "top_module": case["top_module"],
        "golden_path": case["golden_path"],
        "golden_used_for_generation": False,
        "golden_used_for_verisure_simulation": False,
        "process": proc,
        "generation": generation,
        "native_verisure_status": native,
        "syntax": {},
        "equivalence": {"status": "skipped", "reason": "no candidate"},
        "synthesis": {"status": "skipped", "reason": "no candidate"},
        "ppa": {},
        "paths": {"output_dir": str(out), "adapter_log": str(adapter_log), "result_path": str(result_path)},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    }
    result.update(token_fields(generation))
    if candidate is None or not candidate.is_file():
        result["terminal_status"] = "generation_failed"
        result["error"] = generation.get("error") or "candidate RTL missing"
        result["elapsed_seconds"] = time.time() - started
        write_json(result_path, result)
        return result
    result["candidate_path"] = str(candidate)
    result["candidate_sha256"] = sha256(candidate)
    result["golden_sha256"] = sha256(Path(str(case["golden_path"])))
    syn = syntax_check(candidate, str(case["top_module"]), out)
    result["syntax"] = syn
    if syn.get("status") != "passed":
        result["terminal_status"] = "candidate_syntax_failed"
        result["elapsed_seconds"] = time.time() - started
        write_json(result_path, result)
        return result

    if verify_equivalence:
        result["equivalence"] = run_jg(candidate, case, out / "verification", verification_timeout)
    else:
        result["equivalence"] = {"status": "skipped", "reason": "disabled by command line"}

    if run_ppa:
        result["synthesis"] = run_dc(candidate, case, out / "dc")
        metrics = result["synthesis"].get("metrics") if isinstance(result["synthesis"], dict) else {}
        metrics = metrics if isinstance(metrics, dict) else {}
        result["ppa"] = {
            "area": metrics.get("total_cell_area"),
            "timing_ps": metrics.get("delay_ps", metrics.get("data_arrival_time_ps")),
            "delay_ps": metrics.get("delay_ps", metrics.get("data_arrival_time_ps")),
            "slack_ps": metrics.get("slack_ps"),
            "verification_status": result["equivalence"].get("status"),
            "verified_equivalent": bool(result["equivalence"].get("equivalent", False)),
        }
    eq_pass = str(result["equivalence"].get("status", "")).lower() in {"passed", "pass", "success"} and bool(result["equivalence"].get("equivalent", False))
    dc_metrics = result.get("ppa") or {}
    if run_ppa and (dc_metrics.get("area") is not None or dc_metrics.get("timing_ps") is not None):
        result["terminal_status"] = "ppa_complete_verified" if eq_pass else "ppa_complete_unverified"
    else:
        result["terminal_status"] = "candidate_syntax_passed_verified" if eq_pass else "candidate_syntax_passed"
    result["elapsed_seconds"] = time.time() - started
    write_json(result_path, result)
    return result


def value(result: dict[str, Any], key: str) -> Any:
    v = result.get(key)
    return "" if v is None else v


def write_reports(run_root: Path, cases: list[dict[str, Any]], input_csv: Path, config: dict[str, Any]) -> None:
    results: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: int(item["csv_row"])):
        path = run_root / "rows" / str(case["id"]) / "case_result.json"
        if path.is_file():
            try:
                results.append(read_json(path))
            except Exception:
                pass
    all_fields = list(read_rows(input_csv)[0].keys())
    total_tokens = sum(int(r["total_tokens"]) for r in results if isinstance(r.get("total_tokens"), int))
    native_pass = sum(r.get("native_verisure_status") == "pass" for r in results)
    syntax_pass = sum((r.get("syntax") or {}).get("status") == "passed" for r in results)
    eq_pass = sum(bool((r.get("equivalence") or {}).get("equivalent", False)) for r in results)
    area_count = sum((r.get("ppa") or {}).get("area") is not None for r in results)
    timing_count = sum((r.get("ppa") or {}).get("timing_ps") is not None for r in results)
    summary = {
        "schema_version": "verisure_csv_full_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_csv": str(input_csv),
        "input_csv_sha256": sha256(input_csv),
        "row_count": len(cases),
        "rows_present": len(results),
        "agent": "xyjoey/Veri-Sure",
        "mode": "full_pipeline",
        "token_limit": "unset",
        "max_completion_tokens": None,
        "config": config,
        "golden_passed_to_model": False,
        "results": results,
        "counts": {
            "native_pass": native_pass,
            "syntax_pass": syntax_pass,
            "equivalence_pass": eq_pass,
            "area_measured": area_count,
            "timing_measured": timing_count,
            "total_tokens": total_tokens,
        },
    }
    write_json(run_root / "summary.json", summary)
    csv_fields = [
        "csv_row", "objective", "project", "source_case_id", "case_id", "top_module",
        "native_verisure_status", "adapter_status", "syntax_status", "equivalence_status",
        "equivalent", "terminal_status", "area", "timing_ps", "ppa_verified_equivalent",
        "prompt_tokens", "completion_tokens", "total_tokens", "request_count", "token_source",
        "elapsed_seconds", "candidate_path", "golden_path", "result_path", "error",
    ] + [f"baseline_{field}" for field in all_fields]
    with (run_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            eq = r.get("equivalence") or {}
            ppa = r.get("ppa") or {}
            row = {
                "csv_row": r.get("csv_row"),
                "objective": r.get("objective"),
                "project": r.get("project"),
                "source_case_id": r.get("source_case_id"),
                "case_id": r.get("case_id"),
                "top_module": r.get("top_module"),
                "native_verisure_status": r.get("native_verisure_status"),
                "adapter_status": (r.get("generation") or {}).get("status"),
                "syntax_status": (r.get("syntax") or {}).get("status"),
                "equivalence_status": eq.get("status"),
                "equivalent": eq.get("equivalent"),
                "terminal_status": r.get("terminal_status"),
                "area": ppa.get("area"),
                "timing_ps": ppa.get("timing_ps"),
                "ppa_verified_equivalent": ppa.get("verified_equivalent"),
                "prompt_tokens": r.get("prompt_tokens"),
                "completion_tokens": r.get("completion_tokens"),
                "total_tokens": r.get("total_tokens"),
                "request_count": r.get("request_count"),
                "token_source": r.get("token_source"),
                "elapsed_seconds": r.get("elapsed_seconds"),
                "candidate_path": r.get("candidate_path"),
                "golden_path": r.get("golden_path"),
                "result_path": (r.get("paths") or {}).get("result_path"),
                "error": r.get("error") or eq.get("error") or (r.get("synthesis") or {}).get("error"),
            }
            baseline = next((c.get("csv_fields", {}) for c in cases if c.get("csv_row") == r.get("csv_row")), {})
            for field in all_fields:
                row[f"baseline_{field}"] = baseline.get(field, "")
            writer.writerow(row)
    lines = [
        "# Veri-Sure × C-first CSV 全量实验",
        "",
        f"- 输入：`{input_csv}`，共 {len(cases)} 行；相同 case 的 AREA/TIMING 行独立运行。",
        "- 模式：Veri-Sure full pipeline（Architect → Contract → Verifier → Coder → Simulation → Debugger），没有使用 ablation。",
        "- Token：客户端 `max_completion_tokens` 未设置；记录 provider 返回的 prompt/completion/total token，不做字符估算。",
        "- 并行：所有 CSV 行一次提交，每行使用独立输出目录。",
        "- Golden RTL：不进入模型提示；候选语法通过后单独尝试 JasperGold 等价性检查。",
        f"- 汇总：结果 {len(results)}/{len(cases)}；native 仿真 PASS {native_pass}；语法通过 {syntax_pass}；JG 等价通过 {eq_pass}；Area {area_count}；Timing {timing_count}；总 token {total_tokens}。",
        "- 注意：`ppa_complete_unverified` 表示 DC 有数值但 golden RTL 等价性未通过或未完成，不能当作已验证功能结果。",
        "",
        "## 逐行结果",
        "",
        "| 行 | Objective | Project | Case | Native | Syntax | JG | Area | Timing (ps) | Total tok. | Status |",
        "|---:|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for r in results:
        eq = r.get("equivalence") or {}
        ppa = r.get("ppa") or {}
        lines.append(
            f"| {r.get('csv_row')} | {r.get('objective')} | {r.get('project')} | {r.get('source_case_id')} | "
            f"{r.get('native_verisure_status', '—')} | {(r.get('syntax') or {}).get('status', '—')} | "
            f"{eq.get('status', '—')} | {value(ppa, 'area')} | {value(ppa, 'timing_ps')} | "
            f"{value(r, 'total_tokens')} | {r.get('terminal_status', '—')} |"
        )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--sim-max-retry", type=int, default=4)
    ap.add_argument("--debug-max-trials", type=int, default=30)
    ap.add_argument("--process-timeout", type=float, default=1800)
    ap.add_argument("--verification-timeout", type=int, default=600)
    ap.add_argument("--skip-equivalence", action="store_true")
    ap.add_argument("--skip-ppa", action="store_true")
    ap.add_argument("--row", type=int, action="append", default=[], help="Run only selected 1-based CSV row(s); repeatable")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    all_cases = load_cases(input_csv, run_root)
    cases = all_cases
    if args.row:
        wanted_rows = set(args.row)
        cases = [case for case in cases if int(case["csv_row"]) in wanted_rows]
        if not cases:
            raise SystemExit(f"No selected CSV rows found: {sorted(wanted_rows)}")
    config = {
        "model": args.model,
        "temperature": 0.0,
        "top_p": 1.0,
        "sim_max_retry": args.sim_max_retry,
        "debug_max_trials": args.debug_max_trials,
        "process_timeout": args.process_timeout,
        # Explicit null documents the no-client-cap policy.  Adapter omits
        # the CLI option and the patched CLI passes None to OpenAIConfig.
        "max_completion_tokens": None,
        "ablation": False,
    }
    config_path = run_root / "verisure_config.json"
    write_json(config_path, config)
    write_json(
        run_root / "input_manifest.json",
        {"input_csv": str(input_csv), "input_csv_sha256": sha256(input_csv), "row_count": len(all_cases), "cases": all_cases},
    )
    print(f"Veri-Sure full CSV run root: {run_root}", flush=True)
    print(f"Rows: {len(cases)}; workers: {args.max_workers}; token_limit=unset; JG={'off' if args.skip_equivalence else 'on'}; DC={'off' if args.skip_ppa else 'on'}", flush=True)
    max_workers = max(1, min(int(args.max_workers), len(cases)))
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                one_case,
                case,
                run_root,
                config_path,
                force=args.force,
                verify_equivalence=not args.skip_equivalence,
                run_ppa=not args.skip_ppa,
                verification_timeout=args.verification_timeout,
                process_timeout=args.process_timeout,
            ): case
            for case in cases
        }
        for future in concurrent.futures.as_completed(futures):
            case = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "csv_row": case["csv_row"], "case_id": case["id"], "source_case_id": case["source_case_id"],
                    "objective": case["objective"], "project": case["project"], "terminal_status": "runner_exception",
                    "error": f"{type(exc).__name__}: {exc}", "total_tokens": None,
                }
                write_json(run_root / "rows" / str(case["id"]) / "case_result.json", result)
            results.append(result)
            print(
                f"[row {case['csv_row']}] {case['objective']} {case['project']} {case['source_case_id']} -> "
                f"{result.get('terminal_status')} tokens={result.get('total_tokens')}",
                flush=True,
            )
    write_reports(run_root, all_cases, input_csv, config)
    print(f"Reports: {run_root / 'summary.md'} and {run_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
