#!/usr/bin/env python3
"""Build a per-case Markdown record table for the nvdla22 path runs."""

from __future__ import annotations

import json
from datetime import datetime
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CFIRST_ROOT = ROOT / "runs/external_spec_agents/20260915_cfirst_nvdla22_no_jg"
PHASE_ROOT = ROOT / "runs/phase6/nvdla22_njg_20260908"
REPORT = CFIRST_ROOT / "cfirst_phase6_all_paths.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_value(value: object) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def route_metric(
    route: dict | None,
    metric: str,
    *,
    new_run: bool = False,
    best_value: float | None = None,
) -> str:
    if not route:
        return "—"
    if new_run:
        value = route.get("area" if metric == "area" else "timing_ps")
    else:
        value = route.get("area" if metric == "area" else "data_arrival_time_ps")
    rendered = format_value(value)
    if (
        value is not None
        and best_value is not None
        and math.isclose(float(value), float(best_value), rel_tol=1e-12, abs_tol=1e-12)
    ):
        return f"**{rendered}**"
    return rendered


def route_value(route: dict | None, metric: str, *, new_run: bool = False) -> float | None:
    if not route:
        return None
    if new_run:
        key = "timing_ps" if metric == "timing" else "area"
    else:
        key = "data_arrival_time_ps" if metric == "timing" else "area"
    value = route.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def build() -> str:
    new_results = read_json(CFIRST_ROOT / "summary.json")["results"]
    new_by = {(str(row["objective"]).lower(), str(row["case_id"])): row for row in new_results}
    old_by: dict[tuple[str, str], dict] = {}
    old_summaries: dict[str, dict] = {}
    for objective in ("area", "timing"):
        summary = read_json(PHASE_ROOT / objective / "phase6_summary.json")
        old_summaries[objective] = summary
        for case in summary["cases"]:
            benchmark = case["benchmark"]
            pair_path = PHASE_ROOT / objective / "pairs" / f"{benchmark}__{objective}__r000" / "pair_manifest.json"
            old_by[(objective, benchmark)] = read_json(pair_path)["runs"]

    lines = [
        "# nvdla22 C-first 与 phase6 双路径 AREA/TIMING",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "以下仅保留面积和时序数值，按 case 对齐本次 C-first、phase6 c_first、phase6 rtl_direct。",
        "缺失指标以 — 表示；三组数据均为 verification_mode=none 的 DC 结果。每行最小数值代表该行最优，已加粗；并列最优会同时加粗。",
    ]

    for objective in ("area", "timing"):
        lines.extend(
            [
                "",
                f"## {objective.upper()}",
                "",
                (
                    "| Case | 本次 C-first area | phase6 c_first area | phase6 rtl_direct area |"
                    if objective == "area"
                    else "| Case | 本次 C-first timing_ps | phase6 c_first timing_ps | phase6 rtl_direct timing_ps |"
                ),
                "|---|---:|---:|---:|",
            ]
        )
        for case in old_summaries[objective]["cases"]:
            benchmark = case["benchmark"]
            current = new_by.get((objective, benchmark), {})
            runs = old_by[(objective, benchmark)]
            current_value = route_value(current, objective, new_run=True)
            c_first_value = route_value(runs.get("c_first"), objective)
            rtl_direct_value = route_value(runs.get("rtl_direct"), objective)
            available = [value for value in (current_value, c_first_value, rtl_direct_value) if value is not None]
            best_value = min(available) if available else None
            lines.append(
                "| "
                + " | ".join(
                    [
                        benchmark,
                        route_metric(current, objective, new_run=True, best_value=best_value),
                        route_metric(runs.get("c_first"), objective, best_value=best_value),
                        route_metric(runs.get("rtl_direct"), objective, best_value=best_value),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## 数据来源",
            "",
            f"- 本次 C-first：{CFIRST_ROOT / 'summary.json'}",
            f"- phase6 AREA：{PHASE_ROOT / 'area' / 'phase6_summary.json'} 及对应 area/pairs/*/pair_manifest.json",
            f"- phase6 TIMING：{PHASE_ROOT / 'timing' / 'phase6_summary.json'} 及对应 timing/pairs/*/pair_manifest.json",
            "",
            "注意：本表不将缺失指标补成 0；若 TIMING 出现 0.0，请结合对应 RTL 进一步审计。",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    REPORT.write_text(build(), encoding="utf-8")
    print(REPORT)
