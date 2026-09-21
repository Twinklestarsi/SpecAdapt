#!/usr/bin/env python3
"""Create a compact, auditable report from external-agent case results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = ROOT / "runs/external_spec_agents/20260912/cases"
RUN_ROOT = ROOT / "runs/external_spec_agents/20260912/results"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value):
    return value if isinstance(value, (int, float)) else None


def build(agent: str, case_path: Path) -> dict:
    case = load(case_path)
    stem = f"{case['id']}__{case['objective']}"
    result_path = RUN_ROOT / agent / stem / "case_result.json"
    result = load(result_path) if result_path.is_file() else {}
    source = case.get("source_metrics") or {}
    ppa = result.get("ppa") or {}
    verification = result.get("verification") or {}
    area = number(ppa.get("area"))
    timing = number(ppa.get("delay_ps", ppa.get("timing_ps")))
    source_area = number(source.get("area"))
    source_timing = number(source.get("delay_ps", source.get("timing_ps")))
    return {
        "agent": agent,
        "case_id": case["id"],
        "benchmark": case.get("benchmark", case["id"]),
        "objective": case["objective"],
        "comparison_class": case.get("comparison_class"),
        "status": result.get("terminal_status", "not_started"),
        "generation_status": (result.get("generation") or {}).get("status"),
        "syntax_status": (result.get("syntax") or {}).get("status"),
        "jg_status": verification.get("status"),
        "equivalent": verification.get("equivalent"),
        "area": area,
        "delay_ps": timing,
        # Compatibility alias for consumers of the previous report schema.
        "timing_ps": timing,
        "slack_ps": number(ppa.get("slack_ps")),
        "dynamic_power": ppa.get("dynamic_power"),
        "leakage_power": ppa.get("leakage_power"),
        "source_area": source_area,
        "source_delay_ps": source_timing,
        # Compatibility alias for consumers of the previous report schema.
        "source_timing_ps": source_timing,
        "area_ratio_vs_cfirst": area / source_area if area is not None and source_area else None,
        "delay_ratio_vs_cfirst": timing / source_timing if timing is not None and source_timing else None,
        "timing_ratio_vs_cfirst": timing / source_timing if timing is not None and source_timing else None,
        "result_path": str(result_path),
    }


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["verisure", "ace_rtl"])
    ap.add_argument("--output", type=Path, default=RUN_ROOT / "ppa_summary")
    args = ap.parse_args()
    cases = sorted(p for p in CASE_ROOT.glob("*.json") if p.name != "index.json")
    rows = [build(agent, case) for agent in args.agents for case in cases]
    payload = {
        "schema_version": "external_spec_agent_ppa_summary_v1",
        "agents": args.agents,
        "case_count": len(cases),
        "rows": rows,
        "status_counts": {agent: dict(Counter(r["status"] for r in rows if r["agent"] == agent)) for agent in args.agents},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    md_path = args.output.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# External spec-to-RTL agent PPA summary",
        "",
        f"Cases: {len(cases)}; agents: {', '.join(args.agents)}",
        "",
        "| Agent | Case | Obj. | Class | Status | JG | Area | Delay (ps) | C-first area | C-first delay (ps) |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join([
            row["agent"], row["case_id"], row["objective"], row.get("comparison_class") or "",
            row["status"], str(row.get("jg_status") or ""), fmt(row["area"]), fmt(row["delay_ps"]),
            fmt(row["source_area"]), fmt(row["source_delay_ps"]),
        ]) + " |")
    lines += ["", "## Completed PPA counts", ""]
    for agent in args.agents:
        counts = payload["status_counts"][agent]
        lines.append(f"- `{agent}`: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    lines += ["", "`Area` is DC `total_cell_area`; `Delay` is the public `delay_ps` field (DC `data_arrival_time_ps`, critical-path delay, lower is better). Slack is retained only as a feasibility diagnostic. The C-first columns are the historical metrics recorded in the materialized case manifest.", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
