#!/usr/bin/env python3
"""Apply the shared JasperGold/DC gate to already generated candidates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CASE_ROOT = ROOT / "runs/external_spec_agents/20260912/cases"
RUN_ROOT = ROOT / "runs/external_spec_agents/20260912/results"


def run(agent: str, case_path: Path, timeout: int | None) -> dict:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    stem = f"{case['id']}__{case['objective']}"
    out = RUN_ROOT / agent / stem
    result_path = out / "case_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rtl_raw = result.get("candidate_path") or (result.get("generation") or {}).get("rtl_path")
    if not rtl_raw or not Path(rtl_raw).is_file():
        result["terminal_status"] = "generation_failed"
        return result
    rtl = Path(rtl_raw).resolve()
    golden = Path(str(case["golden_path"])).resolve()
    top = "TopModule" if agent == "ace_rtl" else str(case.get("top_module") or "TopModule")
    try:
        from module5.jg_verifier import verify_rtl_equivalence_with_jg
        jg = verify_rtl_equivalence_with_jg(
            rtl, golden, out / "verification", ROOT / ".env",
            golden_top=str(case.get("top_module") or "TopModule"),
            design_type=str(case.get("design_type") or ""),
            verification_timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        jg = {"status": "error", "equivalent": False, "error": f"JG invocation: {type(exc).__name__}: {exc}"}
    result["verification"] = jg
    if str(jg.get("status", "")).lower() not in {"passed", "pass", "success"} or not bool(jg.get("equivalent", False)):
        result["terminal_status"] = "verification_failed"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result
    try:
        from module5.dc_runner import run_dc_for_verilog
        dc = run_dc_for_verilog(
            rtl, benchmark=str(case.get("benchmark", case["id"])), goal=str(case["objective"]),
            stem=stem, output_root=out / "dc", top_module=top, max_cores=1,
        )
    except Exception as exc:  # noqa: BLE001
        dc = {"status": "error", "error": f"DC invocation: {type(exc).__name__}: {exc}"}
    result["synthesis"] = dc
    metrics = dc.get("metrics") if isinstance(dc, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
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
    result["ppa_only_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", choices=["verisure", "ace_rtl"], required=True)
    ap.add_argument("--case", action="append", default=[], help="case stem or id; repeatable")
    ap.add_argument("--all", action="store_true", help="process every materialized case")
    ap.add_argument("--verification-timeout", type=int, default=None)
    args = ap.parse_args()
    if not args.case and not args.all:
        raise SystemExit("provide --case or --all")
    cases = sorted(p for p in CASE_ROOT.glob("*.json") if p.name != "index.json")
    if args.case:
        wanted = set(args.case)
        cases = [p for p in cases if p.stem in wanted or json.loads(p.read_text(encoding="utf-8")).get("id") in wanted]
    for agent in args.agents:
        for case_path in cases:
            stem = f"{json.loads(case_path.read_text(encoding='utf-8'))['id']}__{json.loads(case_path.read_text(encoding='utf-8'))['objective']}"
            result_path = RUN_ROOT / agent / stem / "case_result.json"
            if not result_path.is_file():
                print(json.dumps({"agent": agent, "case": case_path.stem, "terminal_status": "generation_not_ready"}, ensure_ascii=False), flush=True)
                continue
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("terminal_status") == "ppa_complete":
                print(json.dumps({"agent": agent, "case": case_path.stem, "terminal_status": "ppa_complete", "reused": True}, ensure_ascii=False), flush=True)
                continue
            print(json.dumps(run(agent, case_path, args.verification_timeout), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
