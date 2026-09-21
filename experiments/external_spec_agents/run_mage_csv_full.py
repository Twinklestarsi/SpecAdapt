#!/usr/bin/env python3
"""Run MAGE's complete spec-to-RTL flow on rows from a CSV manifest.

Each CSV row is an independent experiment.  In particular, rows with the same
case_id but different AREA/TIMING objectives are deliberately kept separate.
The MAGE run receives only the natural-language specification: the pilot's
golden RTL is retained as provenance and for the post-generation DC reference
metadata, but is not passed to MAGE's prompts or simulator.

The default configuration leaves LlamaIndex ``max_tokens`` unset.  Provider
usage is collected by the adapter shared with the NVDLA runner and persisted
both per request and per CSV row.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_mage_nvdla22 as base


INPUT_CSV = ROOT / "paper/C_RTLgen/ICCAD26-C-RTLgen-sub/doc/merged_token_usage_by_project_cfirst_rtl_direct_agents.csv"
PILOT_ROOT = ROOT / "runs/phase6/nvdla22_njg_20260908/pilot"
PILOT_MANIFEST = PILOT_ROOT / "pilot_manifest.json"
DEFAULT_RUN_ROOT = ROOT / "runs/external_spec_agents/20260920_mage_csv_full_12x1"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = reader.fieldnames or []
    required = {"objective", "project", "case_id"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError(f"Input CSV has no data rows: {path}")
    return rows


def load_cases(input_csv: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(input_csv)
    manifest = base.read_json(PILOT_MANIFEST)
    manifest_by_id = {str(item["id"]): item for item in manifest.get("cases", [])}
    cases: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        source_case_id = str(row.get("case_id") or "").strip()
        objective = str(row.get("objective") or "").strip().upper()
        project = str(row.get("project") or "").strip()
        manifest_item = manifest_by_id.get(source_case_id)
        if manifest_item is None:
            raise KeyError(f"CSV row {row_number}: case_id not in pilot manifest: {source_case_id}")
        spec = (PILOT_ROOT / str(manifest_item.get("spec_path", ""))).resolve()
        golden = (PILOT_ROOT / str(manifest_item.get("golden_rtl_path", ""))).resolve()
        if not spec.is_file() or not golden.is_file():
            raise FileNotFoundError(
                f"CSV row {row_number}: missing spec/golden for {source_case_id}: {spec} / {golden}"
            )
        top = str(manifest_item.get("golden_top_module") or "") or base.extract_module_name(golden)
        if not top:
            raise RuntimeError(f"CSV row {row_number}: cannot determine top module for {source_case_id}")
        # The row number is part of the key so duplicated case IDs never share
        # MAGE output, logs, token totals, or DC report directories.
        row_id = base.safe_name(f"row_{row_number:02d}__{objective}__{project}__{source_case_id}")
        cases.append(
            {
                "case_id": row_id,
                "source_case_id": source_case_id,
                "csv_row": row_number,
                "objective": objective,
                "project": project,
                "csv_fields": row,
                "spec_path": str(spec),
                "golden_rtl_path": str(golden),
                "top_module": top,
                "design_type": manifest_item.get("design_type"),
                "family_id": manifest_item.get("family_id"),
                "source_group": manifest_item.get("source_group"),
                "spec_sha256": base.sha256(spec),
                "golden_sha256": base.sha256(golden),
            }
        )
    return cases


def result_path(run_root: Path, case: dict[str, Any]) -> Path:
    return run_root / "round_01" / base.safe_name(str(case["case_id"])) / "result.json"


def annotate_result(result: dict[str, Any], case: dict[str, Any], run_root: Path) -> dict[str, Any]:
    """Add the source CSV identity and baseline fields to a MAGE result."""

    result["schema_version"] = "mage_csv_full_case_result_v1"
    result["case_id"] = case["case_id"]
    result["source_case_id"] = case["source_case_id"]
    result["csv_row"] = case["csv_row"]
    result["objective"] = case["objective"]
    result["project"] = case["project"]
    result["csv_fields"] = case["csv_fields"]
    result["verification_mode"] = "self_generated_tb_only"
    result["golden_used_for_generation"] = False
    result["golden_used_for_mage_simulation"] = False
    result["token_limit"] = "unset"
    result["max_tokens"] = None
    result["result_path"] = str(result_path(run_root, case))
    return result


def metric(result: dict[str, Any], goal: str) -> Any:
    return base.metric(result, goal)


def token_value(result: dict[str, Any], key: str) -> Any:
    return base.token_value(result, key)


def fmt(value: Any) -> str:
    return base.fmt(value)


def iter_results(run_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("round_01/**/result.json")):
        try:
            results.append(base.read_json(path))
        except Exception:
            continue
    return sorted(results, key=lambda item: int(item.get("csv_row", 0)))


def write_reports(
    run_root: Path,
    cases: list[dict[str, Any]],
    input_csv: Path,
    model: str,
    tokenizer_note: str,
) -> None:
    results = iter_results(run_root)
    all_fields = list(read_csv_rows(input_csv)[0].keys())
    summary = {
        "schema_version": "mage_csv_full_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_csv": str(input_csv),
        "input_csv_sha256": base.sha256(input_csv),
        "row_count": len(cases),
        "rows_present": len(results),
        "agent": "stable-lab/MAGE",
        "mode": "full_pipeline",
        "model": model,
        "token_limit": "unset",
        "max_tokens": None,
        "temperature": 0.0,
        "top_p": 1.0,
        "verification_mode": "self_generated_tb_only",
        "golden_passed_to_mage": False,
        "token_accounting": tokenizer_note,
        "results": results,
    }
    base.write_json(run_root / "summary.json", summary)

    csv_fields = [
        "csv_row", "objective", "project", "case_id", "source_case_id",
        "top_module", "candidate_top_module", "mode", "verification_mode",
        "terminal_status", "mage_is_pass", "syntax_status", "area", "timing_ps",
        "prompt_tokens", "completion_tokens", "total_tokens", "request_count",
        "token_source", "token_limit", "elapsed_seconds", "candidate_path",
        "result_path",
    ] + [f"baseline_{field}" for field in all_fields]
    with (run_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            row = {
                "csv_row": result.get("csv_row"),
                "objective": result.get("objective"),
                "project": result.get("project"),
                "case_id": result.get("case_id"),
                "source_case_id": result.get("source_case_id"),
                "top_module": result.get("top_module"),
                "candidate_top_module": result.get("candidate_top_module"),
                "mode": result.get("mode"),
                "verification_mode": result.get("verification_mode"),
                "terminal_status": result.get("terminal_status"),
                "mage_is_pass": (result.get("mage_return") or {}).get("is_pass"),
                "syntax_status": (result.get("syntax") or {}).get("status"),
                "area": metric(result, "area"),
                "timing_ps": metric(result, "timing"),
                "prompt_tokens": token_value(result, "prompt_tokens"),
                "completion_tokens": token_value(result, "completion_tokens"),
                "total_tokens": token_value(result, "total_tokens"),
                "request_count": token_value(result, "request_count"),
                "token_source": token_value(result, "source"),
                "token_limit": result.get("token_limit", "unset"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "candidate_path": result.get("candidate_path"),
                "result_path": result.get("result_path"),
            }
            for field in all_fields:
                row[f"baseline_{field}"] = (result.get("csv_fields") or {}).get(field, "")
            writer.writerow(row)

    full_ok = sum(str(item.get("terminal_status")) == "ppa_complete" for item in results)
    syntax_ok = sum((item.get("syntax") or {}).get("status") == "passed" for item in results)
    area_ok = sum(metric(item, "area") is not None for item in results)
    timing_ok = sum(metric(item, "timing") is not None for item in results)
    total_tokens = sum(
        int(token_value(item, "total_tokens"))
        for item in results
        if isinstance(token_value(item, "total_tokens"), int)
    )
    total_requests = sum(
        int(token_value(item, "request_count"))
        for item in results
        if isinstance(token_value(item, "request_count"), int)
    )

    lines = [
        "# MAGE × C-first CSV 全量模式实验",
        "",
        f"- 输入 CSV：`{input_csv}`（{len(cases)} 行；AREA/TIMING 重复 case 按行独立运行）",
        f"- Agent：`stable-lab/MAGE`；模型：`{model}`；模式：`full_pipeline`",
        "- 全量流程：MAGE `TopAgent.run`，包含自生成 testbench、RTL 生成、仿真审查/修正和编辑器链路；本轮没有使用 ablation。",
        "- 数据隔离：没有把 pilot 的 golden RTL 或 golden TB 传给 MAGE；因此验证口径是 MAGE 自生成 testbench，不能视作 golden 功能等价证明。",
        "- Token：客户端 `max_tokens` 未设置（`token_limit=unset`）；下方记录 provider 返回的每次 prompt/completion/total token。流程级重试/候选数仍按运行器边界限制。",
        f"- 总体：{len(results)}/{len(cases)} 行已产生结果；完整 PPA {full_ok}，语法通过 {syntax_ok}，Area DC {area_ok}，Timing DC {timing_ok}；总 token {total_tokens}，provider 请求 {total_requests}。",
        "- 指标：Area 为 DC `total_cell_area`；Timing 为 DC `data_arrival_time_ps`/`delay_ps`，单位 ps。",
        "",
        "## 逐行结果",
        "",
        "| CSV 行 | Objective | Project | Case | MAGE | Syntax | Area | Timing (ps) | Prompt tok. | Completion tok. | Total tok. | Requests | Status |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.get('csv_row')} | {result.get('objective')} | {result.get('project')} | "
            f"{result.get('source_case_id')} | "
            f"{'pass' if (result.get('mage_return') or {}).get('is_pass') else 'fail'} | "
            f"{(result.get('syntax') or {}).get('status', '—')} | {fmt(metric(result, 'area'))} | "
            f"{fmt(metric(result, 'timing'))} | {fmt(token_value(result, 'prompt_tokens'))} | "
            f"{fmt(token_value(result, 'completion_tokens'))} | {fmt(token_value(result, 'total_tokens'))} | "
            f"{fmt(token_value(result, 'request_count'))} | {result.get('terminal_status', '—')} |"
        )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N rows for a smoke test")
    parser.add_argument("--start-row", type=int, default=1, help="1-based inclusive CSV row to start (default: 1)")
    parser.add_argument("--end-row", type=int, default=0, help="1-based inclusive CSV row to end (default: last)")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional override; omit for no client-side cap")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_csv = args.input_csv.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if not input_csv.is_file():
        raise SystemExit(f"Input CSV not found: {input_csv}")
    run_root.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = str(ROOT / ".conda-env" / "bin") + os.pathsep + os.environ.get("PATH", "")
    all_cases = load_cases(input_csv)
    if args.start_row < 1:
        raise SystemExit("--start-row must be >= 1")
    end_row = args.end_row or len(all_cases)
    if end_row < args.start_row:
        raise SystemExit("--end-row must be >= --start-row")
    cases = all_cases[args.start_row - 1 : end_row]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No CSV rows selected")

    base.write_json(
        run_root / "input_manifest.json",
        {"source_csv": str(input_csv), "source_csv_sha256": base.sha256(input_csv), "cases": cases},
    )
    tokenizer_note = base.patch_unknown_tokenizer()
    base.install_token_counter_adapter()
    from mage.gen_config import set_exp_setting

    set_exp_setting(temperature=args.temperature, top_p=args.top_p)
    # None is intentional: LlamaIndex's OpenAI wrapper omits max_tokens from
    # the request in this mode, so the provider's own context limit is used.
    llm = base.build_llm(args.model, args.max_tokens, args.temperature, args.top_p)
    base.write_json(
        run_root / "run_config.json",
        {
            "agent": "stable-lab/MAGE",
            "mode": "full_pipeline",
            "model": args.model,
            "max_tokens": args.max_tokens,
            "token_limit": "unset" if args.max_tokens is None else args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "rounds": 1,
            "case_count": len(cases),
            "input_csv": str(input_csv),
            "golden_passed_to_mage": False,
            "verification_mode": "self_generated_tb_only",
            "token_accounting": tokenizer_note,
            "flow_caps": {
                "sim_max_retry": 1,
                "rtl_max_candidates": 1,
                "rtl_syntax_trials": 2,
                "editor_trials": 1,
                "tb_json_trials": 2,
                "tb_display_queue": False,
            },
        },
    )

    print(f"MAGE full CSV run root: {run_root}", flush=True)
    print(f"Rows: {len(cases)}; token_limit={'unset' if args.max_tokens is None else args.max_tokens}", flush=True)
    for case in cases:
        print(
            f"[row {case['csv_row']}/{len(cases)}] {case['objective']} {case['project']} {case['source_case_id']}",
            flush=True,
        )
        try:
            result = base.run_one(
                llm,
                case,
                1,
                run_root,
                args.model,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.force,
                False,
                True,
            )
            result = annotate_result(result, case, run_root)
            base.write_json(result_path(run_root, case), result)
            print(
                f"  -> {result.get('terminal_status')} tokens={token_value(result, 'total_tokens')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            result = annotate_result(
                {
                    "schema_version": "mage_csv_full_case_result_v1",
                    "agent": "stable-lab/MAGE",
                    "mode": "full_pipeline",
                    "round": 1,
                    "case_id": case["case_id"],
                    "model": args.model,
                    "top_module": case["top_module"],
                    "spec_path": case["spec_path"],
                    "golden_rtl_path": case["golden_rtl_path"],
                    "terminal_status": "runner_exception",
                    "token_usage": {
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "request_count": None,
                        "source": "unavailable",
                        "requests": [],
                    },
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                case,
                run_root,
            )
            base.write_json(result_path(run_root, case), result)
            print(f"  -> runner_exception: {type(exc).__name__}: {exc}", flush=True)
        finally:
            write_reports(run_root, cases, input_csv, args.model, tokenizer_note)

    write_reports(run_root, cases, input_csv, args.model, tokenizer_note)
    print(f"Reports: {run_root / 'summary.md'} and {run_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
