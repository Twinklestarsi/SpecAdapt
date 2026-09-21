from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from run_dc import ROOT as PROJECT_ROOT
from run_dc import parse_reports, run_dc_task


def run_dc_for_verilog(
    verilog_path: str | Path,
    *,
    benchmark: str,
    goal: str,
    stem: str,
    output_root: str | Path | None = None,
    top_module: str = "",
    max_cores: int = 1,
) -> Dict[str, Any]:
    verilog_path = Path(verilog_path).resolve()
    goal = goal.lower()
    metric_name = goal.upper()
    output_root = Path(output_root) if output_root else PROJECT_ROOT / "module5_dc_runs"
    output_root.mkdir(parents=True, exist_ok=True)

    task = {
        "v_path": verilog_path,
        "metric": metric_name,
        "subcategory": "module5",
        "benchmark": benchmark,
        "stem": stem,
        "top_module": top_module,
        "max_cores": max(1, int(max_cores)),
    }
    result = run_dc_task(task, output_root=output_root, goal=goal)
    out_dir = output_root / metric_name / "module5" / benchmark / stem
    metrics = parse_reports(out_dir) if out_dir.exists() else {}
    result["report_dir"] = str(out_dir)
    result["metrics"] = metrics
    return result
