#!/usr/bin/env python3
"""Run the three existing RTL agents on nvdla22 and synthesize without JG.

The nvdla22 pilot manifest already contains the natural-language specs and
reference RTL closures used for provenance.  This runner deliberately does
not import or call JasperGold: it performs candidate generation, an Icarus
syntax gate, and independent DC AREA/TIMING runs.  A bounded thread pool lets
the three agents/cases make progress in parallel while keeping the remote
EDA/model load explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[2]
# When this file is invoked by pathname, Python puts only
# ``experiments/external_spec_agents`` on sys.path.  Add the repository root
# explicitly so the local ``module5`` DC wrapper is importable from worker
# threads as well as from an interactive shell.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NVDLA_ROOT = ROOT / "runs/phase6/nvdla22_njg_20260908/pilot"
PILOT_MANIFEST = NVDLA_ROOT / "pilot_manifest.json"
CONFIG = ROOT / "runs/external_spec_agents/20260915_nvdla22_no_jg/config.json"
RUN_ROOT = ROOT / "runs/external_spec_agents/20260915_nvdla22_no_jg"
CASE_ROOT = RUN_ROOT / "cases"

AGENTS = ("verisure", "ace_rtl", "spec2rtl_mini")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")


def extract_interface_ports(golden: Path, top: str) -> list[dict[str, Any]]:
    """Extract only the public top-level declaration for Mini's contract.

    nvdla22's conservative specs are not always interface-complete.  The
    declaration is public benchmark metadata, not behavioral RTL context; it
    is stored in the case manifest and never appended to an agent prompt.
    """

    text = golden.read_text(encoding="utf-8", errors="replace")
    module = re.search(rf"(?ms)^\s*module\s+{re.escape(top)}\b.*?^\s*endmodule\b", text)
    if not module:
        return []
    body = module.group(0)
    decl = re.compile(
        r"^\s*(input|output|inout)\s+"
        r"(?:(wire|reg|logic)\s+)?"
        r"(?:\[\s*([^]]+)\s*:\s*([^]]+)\s*\]\s+)?"
        r"([A-Za-z_]\w*)\s*;",
        re.IGNORECASE | re.MULTILINE,
    )
    ports: list[dict[str, Any]] = []
    seen: set[str] = set()
    for direction, typ, hi, lo, name in decl.findall(body):
        if name in seen:
            continue
        seen.add(name)
        if hi is not None and lo is not None and hi.isdigit() and lo.isdigit():
            width = abs(int(hi) - int(lo)) + 1
            rng = f"[{hi}:{lo}]"
        else:
            width, rng = 1, "[0:0]"
        ports.append({
            "name": name,
            "direction": direction.lower(),
            "width": width,
            "range": rng,
            "type": (typ or ("wire" if direction.lower() == "input" else "reg")).lower(),
        })
    # Some of the official-source cases (notably csrng) use an ANSI-style
    # parameterized header, so the declarations do not end in semicolons and
    # the non-ANSI expression above intentionally finds nothing.  Parse just
    # the header as a conservative fallback.  Widths containing parameters
    # are kept as symbolic ranges but represented as width 1 for Mini's plan;
    # this avoids fabricating a numeric width while still preserving every
    # public port name and direction.
    if not ports:
        start = module.start()
        terminator = text.find(");", start)
        if terminator >= 0:
            segment = text[start : terminator + 2]
            header_open = segment.rfind("(")
            if header_open >= 0:
                header = segment[header_open + 1 : -2]
                # Remove comments before splitting declarations.  Splitting
                # on commas is safe after tracking bracket/parenthesis depth,
                # because ranges and parameter expressions may contain them.
                header = re.sub(r"//[^\n]*", "", header)
                header = re.sub(r"/\*.*?\*/", "", header, flags=re.S)
                chunks: list[str] = []
                chunk_start = 0
                square_depth = paren_depth = 0
                for idx, char in enumerate(header):
                    if char == "[":
                        square_depth += 1
                    elif char == "]":
                        square_depth = max(0, square_depth - 1)
                    elif char == "(":
                        paren_depth += 1
                    elif char == ")":
                        paren_depth = max(0, paren_depth - 1)
                    elif char == "," and square_depth == 0 and paren_depth == 0:
                        chunks.append(header[chunk_start:idx])
                        chunk_start = idx + 1
                chunks.append(header[chunk_start:])
                ansi_decl = re.compile(
                    r"^\s*(input|output|inout)\b"
                    r"(?:\s+(wire|reg|logic)\b)?\s*"
                    r"(?P<ranges>(?:\[[^\]]+\]\s*)*)"
                    r"(?P<name>[A-Za-z_]\w*)\s*$",
                    re.IGNORECASE | re.DOTALL,
                )
                for chunk in chunks:
                    match = ansi_decl.match(chunk.strip())
                    if not match:
                        continue
                    direction = match.group(1).lower()
                    typ = (match.group(2) or ("wire" if direction == "input" else "reg")).lower()
                    name = match.group("name")
                    if name in seen:
                        continue
                    seen.add(name)
                    ranges = re.findall(r"\[\s*([^]]+)\s*:\s*([^]]+)\s*\]", match.group("ranges"))
                    width = 1
                    range_text: list[str] = []
                    all_numeric = True
                    for hi, lo in ranges:
                        range_text.append(f"[{hi}:{lo}]")
                        if hi.strip().isdigit() and lo.strip().isdigit():
                            width *= abs(int(hi.strip()) - int(lo.strip())) + 1
                        else:
                            all_numeric = False
                    if not ranges:
                        rng = "[0:0]"
                    else:
                        rng = "".join(range_text)
                    ports.append({
                        "name": name,
                        "direction": direction,
                        "width": width if all_numeric else 1,
                        "range": rng,
                        "type": typ,
                    })
    return ports


def materialize_cases() -> list[Path]:
    """Create compact case JSONs with absolute, auditable input paths."""

    manifest = read_json(PILOT_MANIFEST)
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    stub_root = CASE_ROOT / "ace_tb_stubs"
    stub_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for item in manifest.get("cases", []):
        case_id = str(item["id"])
        spec = (NVDLA_ROOT / str(item["spec_path"])).resolve()
        golden = (NVDLA_ROOT / str(item["golden_rtl_path"])).resolve()
        if not spec.is_file() or not golden.is_file():
            raise FileNotFoundError(f"missing nvdla22 input for {case_id}: {spec} / {golden}")
        top = str(item.get("top_module") or item.get("golden_top_module") or "TopModule")
        interface_ports = extract_interface_ports(golden, top)
        # ACE's VerilogEval adapter requires a *_test.sv file even when the
        # agent is run without testbench access.  Keep the stub behavior-only:
        # it contains no expected values and no DUT/reference instantiation.
        stub = stub_root / f"{safe(case_id)}.sv"
        if not stub.exists():
            stub.write_text(
                "`timescale 1ns/1ps\n"
                "module tb;\n"
                "  // TopModule placeholder: no functional checks are run.\n"
                "  initial begin\n"
                "    $finish;\n"
                "  end\n"
                "endmodule\n",
                encoding="utf-8",
            )
        record = {
            "id": case_id,
            "benchmark": case_id,
            "spec_path": str(spec),
            "golden_path": str(golden),
            "top_module": top,
            "design_type": item.get("design_type"),
            "source_group": item.get("source_group"),
            "testbench_path": str(stub),
            "ablation": True,
            "source_sha256": item.get("source_sha256"),
            "closure_sha256": item.get("closure_sha256"),
            "spec_sha256": sha256(spec),
            "golden_sha256": sha256(golden),
            "interface_ports": interface_ports,
            "interface_source": "public_top_module_declaration" if interface_ports else "spec_parser",
        }
        case_path = CASE_ROOT / f"{safe(case_id)}.json"
        case_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(case_path)
    if len(paths) != int(manifest.get("case_count", len(paths))):
        raise RuntimeError(f"materialized {len(paths)} cases, manifest says {manifest.get('case_count')}")
    (CASE_ROOT / "index.json").write_text(
        json.dumps({"case_count": len(paths), "cases": [str(p) for p in paths]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return sorted(paths)


def adapter_command(agent: str, case_path: Path, output_dir: Path) -> list[str]:
    if agent == "verisure":
        return [
            str(ROOT / "external_agents/verisure/.venv/bin/python"),
            str(ROOT / "experiments/external_spec_agents/verisure_adapter.py"),
            "--case-json", str(case_path), "--output-dir", str(output_dir), "--config-json", str(CONFIG),
        ]
    if agent == "ace_rtl":
        return [
            str(ROOT / ".conda-env/bin/python"),
            str(ROOT / "experiments/external_spec_agents/ace_adapter.py"),
            "--case-json", str(case_path), "--output-dir", str(output_dir), "--config-json", str(CONFIG),
        ]
    if agent == "spec2rtl_mini":
        return [
            str(ROOT / ".conda-env/bin/python"),
            str(ROOT / "experiments/external_spec_agents/spec2rtl_mini_adapter.py"),
            "--case-json", str(case_path), "--output-dir", str(output_dir), "--config-json", str(CONFIG),
        ]
    raise ValueError(agent)


def run_process(command: list[str], log_path: Path, timeout: float) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        return {
            "returncode": completed.returncode,
            "timed_out": False,
            "elapsed_seconds": round(time.time() - started, 3),
            "log_path": str(log_path),
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": None,
            "timed_out": True,
            "elapsed_seconds": round(time.time() - started, 3),
            "log_path": str(log_path),
        }


def syntax_check(rtl: Path, top: str, output_dir: Path) -> dict[str, Any]:
    log_path = output_dir / "syntax.log"
    command = [str(ROOT / ".conda-env/bin/iverilog"), "-g2012", "-s", top, "-t", "null", str(rtl)]
    try:
        completed = subprocess.run(
            command,
            cwd=str(output_dir),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        log_path.write_text(output, encoding="utf-8")
        return {"status": "passed" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "log_path": str(log_path)}
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        log_path.write_text(str(output), encoding="utf-8")
        return {"status": "timeout", "returncode": None, "log_path": str(log_path)}


def _dc_metrics(dc: dict[str, Any]) -> dict[str, Any]:
    metrics = dc.get("metrics") if isinstance(dc, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    return {
        "area": metrics.get("total_cell_area"),
        "timing_ps": metrics.get("data_arrival_time_ps", metrics.get("delay_ps")),
        "slack_ps": metrics.get("slack_ps"),
        "dynamic_power": metrics.get("dynamic_power"),
        "leakage_power": metrics.get("leakage_power"),
        "dc_status": dc.get("status") if isinstance(dc, dict) else None,
        "report_dir": dc.get("report_dir") if isinstance(dc, dict) else None,
    }


def run_dc(rtl: Path, case_id: str, agent: str, out: Path, goal: str, top: str) -> dict[str, Any]:
    from module5.dc_runner import run_dc_for_verilog

    try:
        dc = run_dc_for_verilog(
            rtl,
            benchmark=case_id,
            goal=goal,
            stem=f"{case_id}__{agent}__{goal}",
            output_root=out / "dc" / goal,
            top_module=top,
            max_cores=1,
        )
        metrics = _dc_metrics(dc)
        # `run_dc_task` may leave partial reports after an error/timeout.  Do
        # not promote those stale values to PPA: require its explicit success
        # banner plus the requested metric before marking this objective done.
        ok = bool(dc.get("success")) and (
            metrics.get("area") is not None or metrics.get("timing_ps") is not None
        )
        return {"status": "passed" if ok else "failed", "raw": dc, "metrics": metrics}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"DC invocation: {type(exc).__name__}: {exc}", "metrics": {}}


def run_case(agent: str, case_path: Path, force: bool) -> dict[str, Any]:
    case = read_json(case_path)
    case_id = str(case["id"])
    out = RUN_ROOT / agent / safe(case_id)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.is_file() and not force:
        return read_json(result_path)

    result: dict[str, Any] = {
        "schema_version": "nvdla22_external_agent_no_jg_v1",
        "agent": agent,
        "case_id": case_id,
        "benchmark": case_id,
        "top_module": case.get("top_module"),
        "design_type": case.get("design_type"),
        "source_group": case.get("source_group"),
        "spec_path": case.get("spec_path"),
        "golden_path": case.get("golden_path"),
        "golden_used_for_generation": False,
        "verification": {"status": "not_run", "reason": "JG skipped by user request"},
        "generation": {},
        "syntax": {},
        "synthesis": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    generation_dir = out / "generation"
    generation_dir.mkdir(parents=True, exist_ok=True)
    # Avoid consuming an adapter result left by an earlier interrupted run.
    # The adapter owns this private per-case directory, so removing only its
    # result marker is sufficient and keeps prior attempt logs for audit.
    stale_adapter_result = generation_dir / "result.json"
    if stale_adapter_result.exists():
        stale_adapter_result.unlink()
    cfg = read_json(CONFIG)
    timeout = float(cfg.get("process_timeout", 1500)) + 120
    proc = run_process(adapter_command(agent, case_path, generation_dir), out / "adapter.log", timeout)
    result["generation_process"] = proc
    adapter_result = generation_dir / "result.json"
    generation = read_json(adapter_result) if adapter_result.is_file() else {
        "status": "error", "rtl_path": None, "error": "adapter did not write result.json", "agent": agent,
    }
    if proc.get("timed_out") or proc.get("returncode") not in (0, None):
        # A non-zero adapter process is a generation failure even if a stale
        # candidate happened to be left in the output directory.
        generation = {
            "status": "error",
            "rtl_path": None,
            "error": f"adapter process failed (returncode={proc.get('returncode')}, timed_out={proc.get('timed_out')})",
            "agent": agent,
            "model": cfg.get("model"),
            "case_id": case_id,
        }
    result["generation"] = generation
    # Adapters expose provider usage under ``token_usage``.  Keep the
    # historical ``usage`` spelling as a fallback so older adapter outputs
    # remain readable by this runner.
    if isinstance(generation, dict):
        result["token_usage"] = generation.get("token_usage", generation.get("usage"))
        if "usage" not in generation and "token_usage" in generation:
            generation["usage"] = generation["token_usage"]
    else:
        result["token_usage"] = None
    rtl_raw = generation.get("rtl_path")
    rtl = Path(str(rtl_raw)).resolve() if rtl_raw else None
    if rtl is None or not rtl.is_file():
        result["terminal_status"] = "generation_failed"
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result
    result["candidate_path"] = str(rtl)
    result["candidate_sha256"] = sha256(rtl)
    top_for_agent = "TopModule" if agent == "ace_rtl" else str(case.get("top_module") or "TopModule")
    result["syntax"] = syntax_check(rtl, top_for_agent, out)
    if result["syntax"].get("status") != "passed":
        result["terminal_status"] = "candidate_syntax_failed"
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result

    # ACE candidates are normalized to TopModule; the other adapters preserve
    # the case's documented top module name.
    result["synthesis"] = {}
    for goal in ("area", "timing"):
        result["synthesis"][goal] = run_dc(rtl, case_id, agent, out, goal, top_for_agent)
    completed = [v for v in result["synthesis"].values() if v.get("status") == "passed"]
    result["terminal_status"] = "ppa_complete" if completed else "synthesis_failed"
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _value(result: Optional[dict[str, Any]], goal: str, key: str) -> Any:
    if not result:
        return None
    block = (result.get("synthesis") or {}).get(goal) or {}
    if block.get("status") != "passed":
        return None
    return ((block.get("metrics") or {}).get(key))


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _pair(area: Any, timing: Any) -> str:
    return f"{_fmt(area)} / {_fmt(timing)}"


def _best_lower(options: dict[str, Any]) -> tuple[Any, str]:
    """Return the smallest numeric value and its source label(s)."""
    usable = {
        label: value
        for label, value in options.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not usable:
        return None, "—"
    best = min(usable.values())
    sources = [label for label, value in usable.items() if value == best]
    return best, " = ".join(sources)


def _relative_improvement(baseline: Any, candidate: Any) -> str:
    """Format lower-is-better improvement over a baseline as a percentage."""
    if not isinstance(baseline, (int, float)) or isinstance(baseline, bool):
        return "—"
    if not isinstance(candidate, (int, float)) or isinstance(candidate, bool):
        return "—"
    if baseline == 0:
        return "0.00%" if candidate == 0 else "—"
    return f"{(baseline - candidate) / baseline * 100:.2f}%"


def write_summary(results: list[dict[str, Any]], case_paths: list[Path]) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    by_key = {(str(r.get("agent")), str(r.get("case_id"))): r for r in results}
    area_summary = read_json(NVDLA_ROOT.parent / "area/phase6_summary.json")
    timing_summary = read_json(NVDLA_ROOT.parent / "timing/phase6_summary.json")
    area_cases = {str(c["benchmark"]): c for c in area_summary.get("cases", [])}
    timing_cases = {str(c["benchmark"]): c for c in timing_summary.get("cases", [])}
    rows: list[dict[str, Any]] = []
    for case_path in case_paths:
        case = read_json(case_path)
        case_id = str(case["id"])
        cfa = ((area_cases.get(case_id) or {}).get("routes") or {}).get("c_first") or {}
        rda = ((area_cases.get(case_id) or {}).get("routes") or {}).get("rtl_direct") or {}
        cft = ((timing_cases.get(case_id) or {}).get("routes") or {}).get("c_first") or {}
        rdt = ((timing_cases.get(case_id) or {}).get("routes") or {}).get("rtl_direct") or {}
        pipeline_area, pipeline_area_source = _best_lower({
            "C-first": cfa.get("median_area"),
            "RTL-direct": rda.get("median_area"),
        })
        pipeline_timing, pipeline_timing_source = _best_lower({
            "C-first": cft.get("median_delay_ps"),
            "RTL-direct": rdt.get("median_delay_ps"),
        })
        row: dict[str, Any] = {
            "case_id": case_id,
            "pipeline_c_first": {"area": cfa.get("median_area"), "timing_ps": cft.get("median_delay_ps")},
            "pipeline_rtl_direct": {"area": rda.get("median_area"), "timing_ps": rdt.get("median_delay_ps")},
            "pipeline_best": {
                "area": pipeline_area,
                "area_source": pipeline_area_source,
                "timing_ps": pipeline_timing,
                "timing_source": pipeline_timing_source,
            },
        }
        for agent in AGENTS:
            r = by_key.get((agent, case_id))
            row[agent] = {
                "area": _value(r, "area", "area"),
                "timing_ps": _value(r, "timing", "timing_ps"),
                "status": (r or {}).get("terminal_status", "not_run"),
            }
        rows.append(row)

    summary = {
        "schema_version": "nvdla22_external_agent_no_jg_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_count": len(rows),
        "agents": list(AGENTS),
        "verification_mode": "none",
        "model": read_json(CONFIG).get("model"),
        "rows": rows,
        "raw_result_paths": [str(RUN_ROOT / r.get("agent", "") / safe(str(r.get("case_id", ""))) / "result.json") for r in results],
    }
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# nvdla22 三个 RTL Agent → DC（跳过 JG）对比",
        "",
        f"生成时间：{summary['created_at']}；case 数：{len(rows)}；模型：{summary['model']}",
        "",
        "## 口径",
        "",
        "- 本轮只做 agent 生成、Icarus 语法检查和 DC AREA/TIMING；JG 明确未运行。PPA 是未做功能等价证明的 provisional 指标。",
        "- Pipeline 两列取已有 nvdla22 no-JG 结果，分别保留 C-first 与 RTL-direct 路线；不是本轮重新生成。",
        "- Area 表的数值来自 DC `total_cell_area`；Timing 表的数值来自 DC `data_arrival_time_ps`（ps，越小越好）。`—` 表示该目标没有 DC 指标。",
        "- ACE-RTL 使用只结束仿真的最小 testbench stub 以满足其 VerilogEval 输入接口；stub 不包含期望值，也不构成功能验证。",
        "",
        "## 汇总",
        "",
        "| Agent | 生成候选 | 语法通过 | Area DC 成功 | Timing DC 成功 | 任一 PPA 成功 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for agent in AGENTS:
        rs = [r for r in results if r.get("agent") == agent]
        generated = sum(bool((r.get("candidate_path"))) for r in rs)
        syntax = sum((r.get("syntax") or {}).get("status") == "passed" for r in rs)
        area_ok = sum(((r.get("synthesis") or {}).get("area") or {}).get("status") == "passed" for r in rs)
        timing_ok = sum(((r.get("synthesis") or {}).get("timing") or {}).get("status") == "passed" for r in rs)
        any_ok = sum(r.get("terminal_status") == "ppa_complete" for r in rs)
        lines.append(f"| {agent} | {generated}/{len(rs)} | {syntax}/{len(rs)} | {area_ok}/{len(rs)} | {timing_ok}/{len(rs)} | {any_ok}/{len(rs)} |")
    lines.extend(["", "## Area 优化逐案例对齐", ""])
    lines.extend([
        "| Case | Pipeline C-first（area） | Pipeline RTL-direct（area） | C/RTL 内部最优 | Veri-Sure（area） | ACE-RTL（area） | Spec2RTL Mini（area） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        cells = [
            row["case_id"],
            _fmt(row["pipeline_c_first"]["area"]),
            _fmt(row["pipeline_rtl_direct"]["area"]),
            _fmt(row["pipeline_best"]["area"]),
        ]
        for agent in AGENTS:
            cells.append(_fmt(row[agent]["area"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "### Area 与 C/RTL 内部最优比较",
        "",
        "相对百分比按 `(C/RTL 内部最优 - Agent) / C/RTL 内部最优` 计算；Area 越小越好，正数表示 agent 节省面积，负数表示变差。",
        "",
        "| Case | C/RTL 内部最优 | Veri-Sure 相对值 | ACE-RTL 相对值 | Spec2RTL Mini 相对值 | 全局最佳（值；相对 C/RTL 内部最优） |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        baseline = row["pipeline_best"]["area"]
        agent_values = {agent: row[agent]["area"] for agent in AGENTS}
        candidates = {"C/RTL 内部最优": baseline, **{
            {"verisure": "Veri-Sure", "ace_rtl": "ACE-RTL", "spec2rtl_mini": "Spec2RTL Mini"}[agent]: value
            for agent, value in agent_values.items()
        }}
        best_value, best_source = _best_lower(candidates)
        best_pct = _relative_improvement(baseline, best_value)
        best_cell = "—" if best_value is None else f"{best_source}（{_fmt(best_value)}；{best_pct}）"
        lines.append("| " + " | ".join([
            row["case_id"],
            _fmt(baseline),
            _relative_improvement(baseline, agent_values["verisure"]),
            _relative_improvement(baseline, agent_values["ace_rtl"]),
            _relative_improvement(baseline, agent_values["spec2rtl_mini"]),
            best_cell,
        ]) + " |")
    lines.extend([
        "",
        "## Timing 优化逐案例对齐",
        "",
        "| Case | Pipeline C-first（timing_ps） | Pipeline RTL-direct（timing_ps） | C/RTL 内部最优 | Veri-Sure（timing_ps） | ACE-RTL（timing_ps） | Spec2RTL Mini（timing_ps） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        cells = [
            row["case_id"],
            _fmt(row["pipeline_c_first"]["timing_ps"]),
            _fmt(row["pipeline_rtl_direct"]["timing_ps"]),
            _fmt(row["pipeline_best"]["timing_ps"]),
        ]
        for agent in AGENTS:
            cells.append(_fmt(row[agent]["timing_ps"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "### Timing 与 C/RTL 内部最优比较",
        "",
        "相对百分比按 `(C/RTL 内部最优 - Agent) / C/RTL 内部最优` 计算；Timing 越小越好，正数表示 agent 缩短关键路径，负数表示变差。",
        "",
        "| Case | C/RTL 内部最优（timing_ps） | Veri-Sure 相对值 | ACE-RTL 相对值 | Spec2RTL Mini 相对值 | 全局最佳（值；相对 C/RTL 内部最优） |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        baseline = row["pipeline_best"]["timing_ps"]
        agent_values = {agent: row[agent]["timing_ps"] for agent in AGENTS}
        candidates = {"C/RTL 内部最优": baseline, **{
            {"verisure": "Veri-Sure", "ace_rtl": "ACE-RTL", "spec2rtl_mini": "Spec2RTL Mini"}[agent]: value
            for agent, value in agent_values.items()
        }}
        best_value, best_source = _best_lower(candidates)
        best_pct = _relative_improvement(baseline, best_value)
        best_cell = "—" if best_value is None else f"{best_source}（{_fmt(best_value)}；{best_pct}）"
        lines.append("| " + " | ".join([
            row["case_id"],
            _fmt(baseline),
            _relative_improvement(baseline, agent_values["verisure"]),
            _relative_improvement(baseline, agent_values["ace_rtl"]),
            _relative_improvement(baseline, agent_values["spec2rtl_mini"]),
            best_cell,
        ]) + " |")
    lines.extend([
        "",
        "## 失败状态",
        "",
        "未形成某个目标 PPA 的原因写在对应 agent/case 的 `result.json`：生成失败、语法失败或 DC 失败均不填估计值。",
        "",
        "## 原始证据",
        "",
        f"- `runs/external_spec_agents/20260915_nvdla22_no_jg/` 下按 agent/case 保存 adapter 日志、候选 RTL、语法日志和 DC 报告。",
        f"- Pipeline 原始基线：`{NVDLA_ROOT.parent / 'nvdla22_njg_20260908_area_timing_summary.md'}`。",
    ])
    (RUN_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global RUN_ROOT, CASE_ROOT, CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=list(AGENTS))
    parser.add_argument("--case", action="append", default=[], help="case id; repeatable")
    parser.add_argument("--workers", type=int, default=3, help="global concurrent agent/DC tasks")
    parser.add_argument("--force", action="store_true", help="ignore existing result.json")
    parser.add_argument(
        "--config-json",
        type=Path,
        default=CONFIG,
        help="shared agent configuration (model, timeouts, and provider token limits)",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=RUN_ROOT,
        help="directory for this batch's cases, results, and summary",
    )
    args = parser.parse_args()
    RUN_ROOT = args.run_root.expanduser().resolve()
    CASE_ROOT = RUN_ROOT / "cases"
    CONFIG = args.config_json.expanduser().resolve()
    case_paths = materialize_cases()
    wanted = set(args.case)
    if wanted:
        case_paths = [p for p in case_paths if read_json(p).get("id") in wanted or p.stem in wanted]
    tasks = [(agent, p) for agent in args.agents for p in case_paths]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(run_case, agent, case_path, args.force): (agent, case_path) for agent, case_path in tasks}
        for future in as_completed(futures):
            agent, case_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # keep batch moving and leave an auditable row
                result = {
                    "schema_version": "nvdla22_external_agent_no_jg_v1",
                    "agent": agent,
                    "case_id": read_json(case_path).get("id", case_path.stem),
                    "terminal_status": "runner_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                # Keep thread-level failures visible in the batch stream and
                # in a per-case marker; otherwise a future exception would be
                # indistinguishable from an adapter-generated failure.
                failed_out = RUN_ROOT / agent / safe(str(result["case_id"]))
                failed_out.mkdir(parents=True, exist_ok=True)
                (failed_out / "runner_exception.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
            results.append(result)
            print(json.dumps({
                "agent": agent,
                "case": result.get("case_id"),
                "status": result.get("terminal_status"),
                **({"error": result.get("error")} if result.get("error") else {}),
            }, ensure_ascii=False), flush=True)
    write_summary(results, case_paths)
    print(json.dumps({"summary": str(RUN_ROOT / 'summary.md'), "rows": len(case_paths), "results": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
