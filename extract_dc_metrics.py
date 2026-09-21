#!/usr/bin/env python3
"""
Extract area or timing metrics from Design Compiler report directories.

By default, the script scans the AREA and TIMING DC collections under the
current project root.

Usage:
  python3 extract_dc_metrics.py --area
  python3 extract_dc_metrics.py --timing

Outputs are written under ``<project>/LLM_DC_LOG``.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
DEFAULT_DC_ROOTS = [
    ROOT / "LLM_V2V_COLLECTED_DC_AREA",
    ROOT / "LLM_V2V_COLLECTED_DC_TIMING",
]
DEFAULT_OUTPUT_DIR = ROOT / "LLM_DC_LOG"
INVALID_REPORT_MARKERS = (
    "Current design is not defined",
    "Can't find design",
    "Cannot find the design",
    "No files or designs were specified",
    "Premature end-of-file",
)


def _to_float(value: str) -> float | str:
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return value


def _search(text: str, pattern: str, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def _is_invalid_report(text: str) -> bool:
    return any(marker in text for marker in INVALID_REPORT_MARKERS)


def _discover_reports(dc_root: Path, report_name: str) -> list[tuple[Path, dict[str, str]]]:
    found: list[tuple[Path, dict[str, str]]] = []
    pattern = f"*/*/*/*/syn_output/{report_name}"
    for report_path in sorted(dc_root.glob(pattern)):
        try:
            rel = report_path.relative_to(dc_root)
        except ValueError:
            continue

        parts = rel.parts
        if len(parts) < 6:
            continue

        found.append(
            (
                report_path,
                {
                    "dc_collection": dc_root.name,
                    "metric": parts[0],
                    "subcategory": parts[1],
                    "benchmark": parts[2],
                    "transform": parts[3],
                    "report_path": str(report_path),
                },
            )
        )
    return found


def parse_area_report_text(text: str) -> dict[str, object]:
    row: dict[str, object] = {}

    patterns = {
        "design": r"^\s*Design\s*:\s*(.+)$",
        "version": r"^\s*Version\s*:\s*(.+)$",
        "date": r"^\s*Date\s*:\s*(.+)$",
        "number_of_ports": r"^\s*Number of ports:\s*(.+)$",
        "number_of_nets": r"^\s*Number of nets:\s*(.+)$",
        "number_of_cells": r"^\s*Number of cells:\s*(.+)$",
        "number_of_combinational_cells": r"^\s*Number of combinational cells:\s*(.+)$",
        "number_of_sequential_cells": r"^\s*Number of sequential cells:\s*(.+)$",
        "number_of_macros_black_boxes": r"^\s*Number of macros/black boxes:\s*(.+)$",
        "number_of_buf_inv": r"^\s*Number of buf/inv:\s*(.+)$",
        "number_of_references": r"^\s*Number of references:\s*(.+)$",
        "combinational_area": r"^\s*Combinational area:\s*(.+)$",
        "buf_inv_area": r"^\s*Buf/Inv area:\s*(.+)$",
        "noncombinational_area": r"^\s*Noncombinational area:\s*(.+)$",
        "macro_black_box_area": r"^\s*Macro/Black Box area:\s*(.+)$",
        "net_interconnect_area": r"^\s*Net Interconnect area:\s*(.+)$",
        "total_cell_area": r"^\s*Total cell area:\s*(.+)$",
        "total_area": r"^\s*Total area:\s*(.+)$",
    }

    for key, pattern in patterns.items():
        value = _search(text, pattern, re.MULTILINE)
        if value is None:
            continue
        row[key] = _to_float(value)

    return row


def parse_area_report(report_path: Path) -> dict[str, object]:
    text = report_path.read_text(encoding="utf-8", errors="replace")
    return parse_area_report_text(text)


def parse_timing_report_text(text: str) -> dict[str, object]:
    row: dict[str, object] = {}

    simple_patterns = {
        "design": r"^\s*Design\s*:\s*(.+)$",
        "version": r"^\s*Version\s*:\s*(.+)$",
        "date": r"^\s*Date\s*:\s*(.+)$",
        "wire_load_model_mode": r"^\s*Wire Load Model Mode:\s*(.+)$",
        "startpoint": r"^\s*Startpoint:\s*(.+)$",
        "endpoint": r"^\s*Endpoint:\s*(.+)$",
        "path_group": r"^\s*Path Group:\s*(.+)$",
        "path_type": r"^\s*Path Type:\s*(.+)$",
    }

    for key, pattern in simple_patterns.items():
        value = _search(text, pattern, re.MULTILINE)
        if value is not None:
            row[key] = value

    op_match = re.search(
        r"^\s*Operating Conditions:\s*(.+?)\s+Library:\s*(.+)$",
        text,
        re.MULTILINE,
    )
    if op_match:
        row["operating_conditions"] = op_match.group(1).strip()
        row["library"] = op_match.group(2).strip()

    arrival = _search(text, r"data arrival time\s+([-\d.]+)")
    if arrival is not None:
        arrival_value = _to_float(arrival)
        row["data_arrival_time"] = arrival_value
        # ``delay_ps`` is the public timing metric.  Keep the original DC
        # spelling as an alias so old CSV consumers remain readable.
        if isinstance(arrival_value, (int, float)):
            row["data_arrival_time_ps"] = abs(float(arrival_value))
            row["delay_ps"] = abs(float(arrival_value))

    required = _search(text, r"data required time\s+([-\d.]+)")
    if required is not None:
        row["data_required_time"] = _to_float(required)

    slack_match = re.search(r"slack\s*\((MET|VIOLATED)\)\s*([-\d.]+)", text)
    if slack_match:
        row["slack_status"] = slack_match.group(1)
        row["slack_ps"] = _to_float(slack_match.group(2))
    elif "No paths." in text:
        row["slack_status"] = "NO_PATHS"

    return row


def parse_timing_report(report_path: Path) -> dict[str, object]:
    text = report_path.read_text(encoding="utf-8", errors="replace")
    return parse_timing_report_text(text)


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["no_data"])
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_area(
    dc_roots: list[Path],
    output_dir: Path,
    output_name: str = "area_metrics.csv",
) -> Path:
    rows: list[dict[str, object]] = []
    skipped_invalid = 0
    skipped_unparseable = 0
    for dc_root in dc_roots:
        if not dc_root.is_dir():
            continue
        for report_path, meta in _discover_reports(dc_root, "area.rpt"):
            text = report_path.read_text(encoding="utf-8", errors="replace")
            if _is_invalid_report(text):
                skipped_invalid += 1
                continue
            row = dict(meta)
            row.update(parse_area_report_text(text))
            if "total_cell_area" not in row:
                skipped_unparseable += 1
                continue
            rows.append(row)

    output_path = output_dir / output_name
    _write_csv(rows, output_path)
    print(f"AREA_ROWS={len(rows)}")
    print(f"AREA_SKIPPED_INVALID={skipped_invalid}")
    print(f"AREA_SKIPPED_UNPARSEABLE={skipped_unparseable}")
    print(f"AREA_OUTPUT={output_path}")
    return output_path


def extract_timing(
    dc_roots: list[Path],
    output_dir: Path,
    output_name: str = "timing_metrics.csv",
) -> Path:
    rows: list[dict[str, object]] = []
    skipped_invalid = 0
    skipped_unparseable = 0
    no_paths = 0
    for dc_root in dc_roots:
        if not dc_root.is_dir():
            continue
        for report_path, meta in _discover_reports(dc_root, "timing.rpt"):
            text = report_path.read_text(encoding="utf-8", errors="replace")
            if _is_invalid_report(text):
                skipped_invalid += 1
                continue
            row = dict(meta)
            row.update(parse_timing_report_text(text))
            if row.get("slack_status") == "NO_PATHS":
                no_paths += 1
            elif "delay_ps" not in row:
                skipped_unparseable += 1
                continue
            rows.append(row)

    output_path = output_dir / output_name
    _write_csv(rows, output_path)
    print(f"TIMING_ROWS={len(rows)}")
    print(f"TIMING_NO_PATHS={no_paths}")
    print(f"TIMING_SKIPPED_INVALID={skipped_invalid}")
    print(f"TIMING_SKIPPED_UNPARSEABLE={skipped_unparseable}")
    print(f"TIMING_OUTPUT={output_path}")
    return output_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract area or timing metrics from DC report directories."
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--area", action="store_true", help="Extract area metrics")
    mode.add_argument("--timing", action="store_true", help="Extract timing metrics")
    ap.add_argument(
        "--area-root",
        type=Path,
        default=DEFAULT_DC_ROOTS[0],
        help=f"AREA DC root (default: {DEFAULT_DC_ROOTS[0]})",
    )
    ap.add_argument(
        "--timing-root",
        type=Path,
        default=DEFAULT_DC_ROOTS[1],
        help=f"TIMING DC root (default: {DEFAULT_DC_ROOTS[1]})",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = ap.parse_args()

    dc_roots = [args.area_root, args.timing_root]
    if args.area:
        extract_area(dc_roots, args.output_dir)
    else:
        extract_timing(dc_roots, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
