#!/usr/bin/env python3
"""Run external spec-to-RTL agents and the shared JG/DC PPA gate.

This runner keeps generation, equivalence, and synthesis results separate so
an agent that produced syntactically valid but incorrect RTL cannot receive a
PPA score.  It is resumable: a complete per-case result is reused unless
``--force`` is supplied.
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
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CASE_ROOT = ROOT / "runs/external_spec_agents/20260912/cases"
RUN_ROOT = ROOT / "runs/external_spec_agents/20260912/results"
CONFIG = ROOT / "runs/external_spec_agents/20260912/config.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")


def adapter_command(agent: str, case_path: Path, output_dir: Path, config_path: Path) -> list[str]:
    if agent == "verisure":
        return [
            str(ROOT / "external_agents/verisure/.venv/bin/python"),
            str(ROOT / "experiments/external_spec_agents/verisure_adapter.py"),
            "--case-json", str(case_path), "--output-dir", str(output_dir), "--config-json", str(config_path),
        ]
    if agent == "ace_rtl":
        return [
            str(ROOT / ".conda-env/bin/python"),
            str(ROOT / "experiments/external_spec_agents/ace_adapter.py"),
            "--case-json", str(case_path), "--output-dir", str(output_dir), "--config-json", str(config_path),
        ]
    if agent == "spec2rtl_mini":
        return [
            str(ROOT / ".conda-env/bin/python"),
            str(ROOT / "experiments/external_spec_agents/spec2rtl_mini_adapter.py"),
            "--case-json", str(case_path), "--output-dir", str(output_dir), "--config-json", str(config_path),
        ]
    raise ValueError(agent)


def run_process(command: list[str], cwd: Path, log_path: Path, timeout: float, env_extra: Optional[dict[str, str]] = None) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    if env_extra:
        env.update(env_extra)
    started = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            cp = subprocess.run(command, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
        return {"returncode": cp.returncode, "timed_out": False, "elapsed_seconds": time.time() - started}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "timed_out": True, "elapsed_seconds": time.time() - started}


def syntax_check(candidate: Path, top: str, output_dir: Path) -> dict[str, Any]:
    log = output_dir / "syntax.log"
    command = [str(ROOT / ".conda-env/bin/iverilog"), "-g2012", "-s", top, "-t", "null", str(candidate)]
    cp = subprocess.run(command, cwd=str(output_dir), capture_output=True, text=True, timeout=120, check=False)
    log.write_text((cp.stdout or "") + (cp.stderr or ""), encoding="utf-8")
    return {"status": "passed" if cp.returncode == 0 else "failed", "returncode": cp.returncode, "log_path": str(log)}


def run_case(agent: str, case_path: Path, config_path: Path, *, execute_ppa: bool, force: bool, verification_timeout: Optional[int]) -> dict[str, Any]:
    case = read_json(case_path)
    case_id = str(case["id"])
    objective = str(case["objective"])
    stem = f"{case_id}__{objective}"
    out = RUN_ROOT / agent / stem
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "case_result.json"
    if result_path.is_file() and not force:
        return read_json(result_path)

    result: dict[str, Any] = {
        "schema_version": "external_spec_agent_case_result_v1",
        "agent": agent,
        "case_id": case_id,
        "benchmark": case.get("benchmark", case_id),
        "objective": objective,
        "comparison_class": case.get("comparison_class"),
        "source_metrics": case.get("source_metrics", {}),
        "source_manifest": case.get("source_manifest"),
        "generation": {},
        "syntax": {},
        "verification": {},
        "synthesis": {},
    }
    result["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    adapter_log = out / "adapter.log"
    cfg = read_json(config_path)
    timeout = float(cfg.get("process_timeout", 1500)) + 120
    proc = run_process(adapter_command(agent, case_path, out / "generation", config_path), ROOT, adapter_log, timeout)
    result["generation_process"] = proc
    adapter_result_path = out / "generation" / "result.json"
    if adapter_result_path.is_file():
        generation = read_json(adapter_result_path)
    else:
        generation = {"status": "error", "rtl_path": None, "error": "adapter did not write result.json"}
    result["generation"] = generation
    rtl_raw = generation.get("rtl_path")
    rtl = Path(rtl_raw).resolve() if rtl_raw else None
    golden = Path(str(case["golden_path"])).resolve()
    top = "TopModule" if agent == "ace_rtl" else str(case.get("top_module") or "TopModule")
    if rtl is None or not rtl.is_file():
        result["terminal_status"] = "generation_failed"
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result
    result["candidate_path"] = str(rtl)
    result["candidate_sha256"] = sha256(rtl)
    result["golden_sha256"] = sha256(golden)
    syn = syntax_check(rtl, top, out)
    result["syntax"] = syn
    if syn.get("status") != "passed":
        result["terminal_status"] = "candidate_syntax_failed"
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result
    if not execute_ppa:
        result["terminal_status"] = "candidate_syntax_passed"
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result

    try:
        from module5.jg_verifier import verify_rtl_equivalence_with_jg
        jg = verify_rtl_equivalence_with_jg(
            rtl, golden, out / "verification", ROOT / ".env",
            golden_top=str(case.get("top_module") or "TopModule"),
            design_type=str(case.get("design_type") or ""),
            verification_timeout=verification_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        jg = {"status": "error", "equivalent": False, "error": f"JG invocation: {type(exc).__name__}: {exc}"}
    result["verification"] = jg
    if str(jg.get("status", "")).lower() not in {"passed", "pass", "success"} or not bool(jg.get("equivalent", False)):
        result["terminal_status"] = "verification_failed"
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result

    try:
        from module5.dc_runner import run_dc_for_verilog
        dc = run_dc_for_verilog(
            rtl, benchmark=str(case.get("benchmark", case_id)), goal=objective,
            stem=stem, output_root=out / "dc", top_module=top, max_cores=1,
        )
    except Exception as exc:  # noqa: BLE001
        dc = {"status": "error", "error": f"DC invocation: {type(exc).__name__}: {exc}"}
    result["synthesis"] = dc
    metrics = dc.get("metrics") if isinstance(dc, dict) else None
    if not isinstance(metrics, dict):
        metrics = {}
    result["ppa"] = {
        "area": metrics.get("total_cell_area"),
        "delay_ps": metrics.get("delay_ps", metrics.get("data_arrival_time_ps")),
        # Compatibility alias for consumers of the earlier report schema.
        "timing_ps": metrics.get("delay_ps", metrics.get("data_arrival_time_ps")),
        # Compatibility field; slack is a feasibility diagnostic only.
        "slack_ps": metrics.get("slack_ps"),
        "dynamic_power": metrics.get("dynamic_power"),
        "leakage_power": metrics.get("leakage_power"),
    }
    result["terminal_status"] = "ppa_complete" if result["ppa"]["area"] is not None or result["ppa"]["delay_ps"] is not None else "synthesis_failed"
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def write_summary(results: list[dict[str, Any]], agents: list[str]) -> None:
    summary = {
        "schema_version": "external_spec_agent_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agents": agents,
        "case_count": len(results),
        "status_counts": {},
        "results": results,
    }
    counts: dict[str, int] = {}
    for r in results:
        status = str(r.get("terminal_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    summary["status_counts"] = counts
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# External spec-to-RTL agent results", "", f"Cases: {len(results)}", "", "| Agent | Case | Objective | Status | Area | Delay (ps) | JG |", "|---|---|---|---|---:|---:|---|"]
    for r in results:
        ppa = r.get("ppa", {}) or {}
        v = r.get("verification", {}) or {}
        lines.append(f"| {r.get('agent')} | {r.get('case_id')} | {r.get('objective')} | {r.get('terminal_status')} | {ppa.get('area','')} | {ppa.get('delay_ps','')} | {v.get('status','')} |")
    (RUN_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", choices=["verisure", "ace_rtl", "spec2rtl_mini"], default=["verisure", "ace_rtl"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--execute-ppa", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verification-timeout", type=int, default=None)
    args = ap.parse_args()
    cases = sorted(CASE_ROOT.glob("*.json"))
    cases = [p for p in cases if p.name != "index.json"]
    if args.case:
        wanted = set(args.case)
        cases = [p for p in cases if p.stem in wanted or read_json(p).get("id") in wanted]
    if args.limit > 0:
        cases = cases[:args.limit]
    all_results: list[dict[str, Any]] = []
    for agent in args.agents:
        for case in cases:
            print(f"[{agent}] {case.name}", flush=True)
            all_results.append(run_case(agent, case, args.config.resolve(), execute_ppa=args.execute_ppa, force=args.force, verification_timeout=args.verification_timeout))
    write_summary(all_results, args.agents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
