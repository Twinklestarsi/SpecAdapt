#!/usr/bin/env python3
"""
Extract area or timing metrics from the AREA baseline DC directory.

Usage:
  python3 extract_dc_metrics_baseline.py --area
  python3 extract_dc_metrics_baseline.py --timing

Outputs are written under ``<project>/LLM_DC_LOG``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from extract_dc_metrics import DEFAULT_OUTPUT_DIR, extract_area, extract_timing
from project_paths import PROJECT_ROOT


DEFAULT_AREA_BASELINE_ROOT = PROJECT_ROOT / "LLM_V2V_COLLECTED_DC_AREA_BASELINE"
DEFAULT_TIMING_BASELINE_ROOT = PROJECT_ROOT / "LLM_V2V_COLLECTED_DC_TIMING_BASELINE"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract area or timing metrics from AREA baseline DC results."
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--area", action="store_true", help="Extract area metrics")
    mode.add_argument("--timing", action="store_true", help="Extract timing metrics")
    ap.add_argument(
        "--area-root",
        type=Path,
        default=DEFAULT_AREA_BASELINE_ROOT,
        help=f"AREA baseline DC root (default: {DEFAULT_AREA_BASELINE_ROOT})",
    )
    ap.add_argument(
        "--timing-root",
        type=Path,
        default=DEFAULT_TIMING_BASELINE_ROOT,
        help=f"TIMING baseline DC root (default: {DEFAULT_TIMING_BASELINE_ROOT})",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = ap.parse_args()

    if args.area:
        extract_area([args.area_root], args.output_dir, "area_metrics_baseline.csv")
    else:
        extract_timing([args.timing_root], args.output_dir, "timing_metrics_baseline.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
