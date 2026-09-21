#!/usr/bin/env python3
"""
Clean invalid Design Compiler metrics rows from existing CSV logs.

Default inputs are read from ``<project>/LLM_DC_LOG``.

By default, cleaned files are written alongside the inputs with a "_clean" suffix.
Use --in-place to overwrite the original CSVs after creating backups.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

from project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
DEFAULT_LOG_DIR = ROOT / "LLM_DC_LOG"
DEFAULT_AREA_INPUT = DEFAULT_LOG_DIR / "area_metrics.csv"
DEFAULT_TIMING_INPUT = DEFAULT_LOG_DIR / "timing_metrics.csv"
DEFAULT_SUMMARY = DEFAULT_LOG_DIR / "dc_clean_summary.json"

INVALID_REPORT_MARKERS = (
    "Current design is not defined",
    "Can't find design",
    "Cannot find the design",
    "No files or designs were specified",
    "Premature end-of-file",
)

AREA_TOTAL_RE = re.compile(r"^\s*Total cell area:\s*(.+)$", re.MULTILINE)
TIMING_ARRIVAL_RE = re.compile(r"data\s+arrival\s+time\s+([-+\d.eE]+)", re.IGNORECASE)
TIMING_SLACK_RE = re.compile(r"slack\s*\((MET|VIOLATED)\)\s*([-\d.]+)")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A cleaned timing row may add the explicit delay aliases even when the
    # input CSV predates them.  Extend the header deterministically rather than
    # silently dropping the primary metric (or failing on an extra key).
    merged_fieldnames = list(fieldnames)
    for row in rows:
        for key in row:
            if key not in merged_fieldnames:
                merged_fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=merged_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_report_text(report_path: str) -> str | None:
    path = Path(report_path)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _classify_area_report(text: str | None) -> str:
    if text is None:
        return "MISSING_REPORT"
    if any(marker in text for marker in INVALID_REPORT_MARKERS):
        return "INVALID_REPORT"
    if AREA_TOTAL_RE.search(text):
        return "VALID"
    return "UNPARSEABLE"


def _classify_timing_report(text: str | None) -> tuple[str, str | None, str | None]:
    if text is None:
        return "MISSING_REPORT", None, None
    if any(marker in text for marker in INVALID_REPORT_MARKERS):
        return "INVALID_REPORT", None, None
    arrivals = [float(value) for value in TIMING_ARRIVAL_RE.findall(text)]
    if arrivals:
        # Delay is a non-negative critical-path quantity.  Use the worst path
        # when a report contains more than one path, matching run_dc.py.
        delay_value = str(max(abs(value) for value in arrivals))
        slack = TIMING_SLACK_RE.search(text)
        if slack:
            return slack.group(1), delay_value, slack.group(2)
        return "MEASURED", delay_value, None
    if "No paths." in text:
        return "NO_PATHS", None, None
    return "UNPARSEABLE", None, None


def _derive_output_path(input_path: Path, in_place: bool) -> Path:
    if in_place:
        return input_path
    return input_path.with_name(f"{input_path.stem}_clean{input_path.suffix}")


def _backup_file(path: Path, suffix: str) -> Path:
    backup = path.with_name(f"{path.name}{suffix}")
    shutil.copy2(path, backup)
    return backup


def clean_area_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    cleaned: list[dict[str, str]] = []
    removed_examples: list[dict[str, str]] = []
    counts = {
        "input_rows": len(rows),
        "kept_rows": 0,
        "removed_invalid_report": 0,
        "removed_missing_report": 0,
        "removed_unparseable": 0,
    }

    for row in rows:
        status = _classify_area_report(_load_report_text(row.get("report_path", "")))
        if status == "VALID":
            cleaned.append(row)
            continue

        if status == "INVALID_REPORT":
            counts["removed_invalid_report"] += 1
        elif status == "MISSING_REPORT":
            counts["removed_missing_report"] += 1
        else:
            counts["removed_unparseable"] += 1

        if len(removed_examples) < 10:
            removed_examples.append(
                {
                    "benchmark": row.get("benchmark", ""),
                    "transform": row.get("transform", ""),
                    "report_path": row.get("report_path", ""),
                    "reason": status,
                }
            )

    counts["kept_rows"] = len(cleaned)
    counts["removed_rows"] = counts["input_rows"] - counts["kept_rows"]
    return cleaned, {"counts": counts, "examples": removed_examples}


def clean_timing_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    cleaned: list[dict[str, str]] = []
    removed_examples: list[dict[str, str]] = []
    counts = {
        "input_rows": len(rows),
        "kept_rows": 0,
        "kept_no_paths": 0,
        "kept_met": 0,
        "kept_violated": 0,
        "kept_measured": 0,
        "removed_invalid_report": 0,
        "removed_missing_report": 0,
        "removed_unparseable": 0,
    }

    for row in rows:
        status, delay_value, slack_value = _classify_timing_report(_load_report_text(row.get("report_path", "")))
        if status in {"MET", "VIOLATED", "MEASURED", "NO_PATHS"}:
            clean_row = dict(row)
            clean_row["slack_status"] = status
            clean_row["slack_ps"] = slack_value if slack_value is not None else ""
            if delay_value is not None:
                clean_row["data_arrival_time"] = delay_value
                clean_row["data_arrival_time_ps"] = delay_value
                clean_row["delay_ps"] = delay_value
            cleaned.append(clean_row)
            if status == "NO_PATHS":
                counts["kept_no_paths"] += 1
            elif status == "MET":
                counts["kept_met"] += 1
            elif status == "VIOLATED":
                counts["kept_violated"] += 1
            elif status == "MEASURED":
                counts["kept_measured"] += 1
            continue

        if status == "INVALID_REPORT":
            counts["removed_invalid_report"] += 1
        elif status == "MISSING_REPORT":
            counts["removed_missing_report"] += 1
        else:
            counts["removed_unparseable"] += 1

        if len(removed_examples) < 10:
            removed_examples.append(
                {
                    "benchmark": row.get("benchmark", ""),
                    "transform": row.get("transform", ""),
                    "report_path": row.get("report_path", ""),
                    "reason": status,
                }
            )

    counts["kept_rows"] = len(cleaned)
    counts["removed_rows"] = counts["input_rows"] - counts["kept_rows"]
    return cleaned, {"counts": counts, "examples": removed_examples}


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean invalid DC metric rows from CSV logs.")
    ap.add_argument("--area-input", type=Path, default=DEFAULT_AREA_INPUT)
    ap.add_argument("--timing-input", type=Path, default=DEFAULT_TIMING_INPUT)
    ap.add_argument("--in-place", action="store_true", help="Overwrite the input CSVs")
    ap.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Backup suffix when using --in-place (default: %(default)s)",
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Summary JSON output path (default: %(default)s)",
    )
    args = ap.parse_args()

    area_output = _derive_output_path(args.area_input, args.in_place)
    timing_output = _derive_output_path(args.timing_input, args.in_place)

    area_fieldnames, area_rows = _read_csv(args.area_input)
    timing_fieldnames, timing_rows = _read_csv(args.timing_input)

    if args.in_place:
        area_backup = _backup_file(args.area_input, args.backup_suffix)
        timing_backup = _backup_file(args.timing_input, args.backup_suffix)
    else:
        area_backup = None
        timing_backup = None

    cleaned_area, area_summary = clean_area_rows(area_rows)
    cleaned_timing, timing_summary = clean_timing_rows(timing_rows)

    _write_csv(area_output, area_fieldnames, cleaned_area)
    _write_csv(timing_output, timing_fieldnames, cleaned_timing)

    summary = {
        "area": {
            "input": str(args.area_input),
            "output": str(area_output),
            "backup": str(area_backup) if area_backup else None,
            **area_summary,
        },
        "timing": {
            "input": str(args.timing_input),
            "output": str(timing_output),
            "backup": str(timing_backup) if timing_backup else None,
            **timing_summary,
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"AREA_OUTPUT={area_output}")
    print(f"TIMING_OUTPUT={timing_output}")
    print(f"SUMMARY_OUTPUT={args.summary}")
    print(f"AREA_KEPT={area_summary['counts']['kept_rows']}")
    print(f"AREA_REMOVED={area_summary['counts']['removed_rows']}")
    print(f"TIMING_KEPT={timing_summary['counts']['kept_rows']}")
    print(f"TIMING_REMOVED={timing_summary['counts']['removed_rows']}")
    print(f"TIMING_NO_PATHS={timing_summary['counts']['kept_no_paths']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
