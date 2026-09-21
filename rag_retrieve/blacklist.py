"""
Build harmful-transform blacklists for Module 4 / Module 5 handoff.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

SEED_BLACKLIST = {
    "area": {
        "CARRY_SAVE_REWRITE",
        "PIPELINE_STAGE_INSERT",
        "LOOP_PIPELINING",
    },
    "timing": {
        "PIPELINE_STAGE_INSERT",
    },
}

MIN_SAMPLES = 3
"""Fewer than this many measurements is an anecdote, not a verdict."""

AREA_BAN_THRESHOLD_PCT = -5.0
AREA_WATCH_THRESHOLD_PCT = -5.0

# The legacy timing rule was `mean < 0.0`, a hair trigger that only looked
# reasonable while blank cells flooded the pool with zeros.  On measurement-only
# rows the timing means split cleanly into a regressing group (-86, -63, -33,
# -20, -16, -13, -8.5 ps) and a noise floor (-2.8 .. -0.5 ps), so a magnitude
# threshold now applies to timing exactly as it always did to area.
TIMING_BAN_THRESHOLD_PS = -5.0
TIMING_WATCH_THRESHOLD_PS = -5.0


def _load_rows(csv_path: str | Path) -> List[Dict[str, Any]]:
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> Optional[float]:
    """Parse a metric cell, keeping "no measurement" distinct from "zero".

    The previous implementation used ``float(cell or 0.0)``, which turned every
    blank cell into a real 0.0 sample.  In the v6 knowledge base that is 3563 of
    4710 rows (no-op applications and unpaired runs), plus every timing row's
    empty area column -- enough zeros to drag any transform's median to exactly
    0.00 regardless of how it actually behaves.
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_measurement(row: Dict[str, Any]) -> bool:
    """Keep only rows that carry a real before/after synthesis comparison."""

    if str(row.get("no_op", "")).strip().lower() == "true":
        return False
    pairing = str(row.get("pairing_status", "")).strip().lower()
    if pairing and pairing != "exact":
        return False
    return True


def _row_objective(row: Dict[str, Any]) -> Optional[str]:
    """Objective this row was measured under, or None on the legacy schema."""

    objective = str(row.get("objective", "")).strip().lower()
    return objective if objective in {"area", "timing"} else None


def _first_metric(row: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _collect(rows: List[Dict[str, Any]]) -> tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    area_stats: Dict[str, List[float]] = defaultdict(list)
    timing_stats: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        if not _is_measurement(row):
            continue
        transform = str(row.get("transform_name") or "").strip()
        if not transform:
            continue
        objective = _row_objective(row)
        area_gain = _float_or_none(row.get("area_improvement_pct"))
        # Timing blacklist decisions are based on delay reduction only.  Do
        # not substitute slack: slack depends on the chosen clock constraint
        # and can hide the actual critical-path delay.
        timing_gain = _float_or_none(row.get("delay_improvement_ps"))
        if timing_gain is None:
            delay = _first_metric(row, "delay_ps", "data_arrival_time_ps", "data_arrival_time")
            baseline_delay = _first_metric(
                row,
                "baseline_delay_ps", "baseline_data_arrival_time_ps", "baseline_data_arrival_time",
            )
            if delay is not None and baseline_delay is not None:
                timing_gain = baseline_delay - delay
        # On the legacy schema (no `objective` column) every row counts for both
        # metrics, exactly as before.  On v6+ a row only votes for the objective
        # it was actually optimised and measured under.
        if area_gain is not None and objective in (None, "area"):
            area_stats[transform].append(area_gain)
        if timing_gain is not None and objective in (None, "timing"):
            timing_stats[transform].append(timing_gain)

    return area_stats, timing_stats


def _classify(
    stats: Dict[str, List[float]],
    *,
    ban_threshold: float,
    watch_threshold: float,
    mean_key: str,
    median_key: str,
    ban_reason: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split transforms into "ban" and "watch" using median vs mean.

    The ban test is on the **median**, not the mean.  These distributions carry
    a handful of -300%..-700% outliers -- one pathological benchmark is enough
    to push a mean below the threshold while the transform is neutral or
    slightly positive on every other sample.  Under the old mean rule that
    banned 24 of 28 area transforms outright, including ALGEBRAIC_SIMPLIFY,
    whose absence is what cost the C-first route on cf20.

    A transform whose mean is bad but whose median is fine is not exonerated
    either: it is high variance.  Those go on a watchlist and reach Module 5
    marked ``risk=high`` instead of being removed from the action space.
    """

    banned: List[Dict[str, Any]] = []
    watched: List[Dict[str, Any]] = []
    for transform, values in stats.items():
        if len(values) < MIN_SAMPLES:
            continue
        mean = sum(values) / len(values)
        median = statistics.median(values)
        record = {
            "transform_name": transform,
            mean_key: round(mean, 6),
            median_key: round(median, 6),
            "positive_rate": round(
                sum(1 for value in values if value > 0.0) / len(values), 6
            ),
            "samples": len(values),
        }
        if median <= ban_threshold:
            banned.append({**record, "reason": ban_reason})
        elif mean <= watch_threshold:
            watched.append(
                {**record, "reason": "high_variance_regression", "risk": "high"}
            )
    return banned, watched


def build_blacklist(csv_path: str | Path) -> Dict[str, Any]:
    rows = _load_rows(csv_path)
    area_stats, timing_stats = _collect(rows)

    derived_area, watch_area = _classify(
        area_stats,
        ban_threshold=AREA_BAN_THRESHOLD_PCT,
        watch_threshold=AREA_WATCH_THRESHOLD_PCT,
        mean_key="avg_area_improvement_pct",
        median_key="median_area_improvement_pct",
        ban_reason="negative_area_median",
    )
    derived_timing, watch_timing = _classify(
        timing_stats,
        ban_threshold=TIMING_BAN_THRESHOLD_PS,
        watch_threshold=TIMING_WATCH_THRESHOLD_PS,
        mean_key="avg_delay_improvement_ps",
        median_key="median_delay_improvement_ps",
        ban_reason="negative_timing_median",
    )

    area_names = set(SEED_BLACKLIST["area"]) | {item["transform_name"] for item in derived_area}
    timing_names = set(SEED_BLACKLIST["timing"]) | {item["transform_name"] for item in derived_timing}

    # A seed ban always wins over a watchlist entry: never advertise a transform
    # as merely risky when the final list forbids it.
    watch_area = [item for item in watch_area if item["transform_name"] not in area_names]
    watch_timing = [item for item in watch_timing if item["transform_name"] not in timing_names]

    return {
        "csv_path": str(Path(csv_path).resolve()),
        "criterion": {
            "ban": "median",
            "watch": "mean",
            "min_samples": MIN_SAMPLES,
            "area_ban_threshold_pct": AREA_BAN_THRESHOLD_PCT,
            "timing_ban_threshold_ps": TIMING_BAN_THRESHOLD_PS,
            "timing_metric": "delay_improvement_ps",
            "timing_direction": "positive_delay_reduction_is_better",
            "slack_role": "diagnostic_only",
            "measurement_rows_only": True,
            "objective_scoped": True,
        },
        "measurement_counts": {
            "total_rows": len(rows),
            "area_measurements": sum(len(values) for values in area_stats.values()),
            "timing_measurements": sum(len(values) for values in timing_stats.values()),
        },
        "seed_blacklist": {key: sorted(values) for key, values in SEED_BLACKLIST.items()},
        "derived": {
            "area": sorted(derived_area, key=lambda item: item["median_area_improvement_pct"]),
            "timing": sorted(derived_timing, key=lambda item: item["median_delay_improvement_ps"]),
        },
        "watchlist": {
            "area": sorted(watch_area, key=lambda item: item["avg_area_improvement_pct"]),
            "timing": sorted(watch_timing, key=lambda item: item["avg_delay_improvement_ps"]),
        },
        "final": {
            "area": sorted(area_names),
            "timing": sorted(timing_names),
        },
    }


def watchlist_names(blacklist_payload: Dict[str, Any], objective: str) -> set[str]:
    """Transforms allowed through, but forced to ``risk=high`` downstream."""

    entries = (blacklist_payload.get("watchlist") or {}).get(objective, [])
    return {
        str(item.get("transform_name") or "")
        for item in entries
        if item.get("transform_name")
    }


def write_blacklist(output_path: str | Path, payload: Dict[str, Any]) -> None:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build harmful transform blacklists from rag_knowledge_base.csv.")
    parser.add_argument(
        "--csv-path",
        default="LLM_DC_LOG/rag_knowledge_base.csv",
        help="Path to rag_knowledge_base.csv",
    )
    parser.add_argument(
        "--output",
        default="rag_retrieve/indices/module4_transform_blacklist.json",
        help="Output blacklist JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_blacklist(args.csv_path)
    write_blacklist(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "area_count": len(payload["final"]["area"]),
                "timing_count": len(payload["final"]["timing"]),
                "area_watchlist_count": len(payload["watchlist"]["area"]),
                "timing_watchlist_count": len(payload["watchlist"]["timing"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
