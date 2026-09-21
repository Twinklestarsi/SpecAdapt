"""Normalize historical optimization runs into auditable RAG samples.

The loader deliberately ignores JasperGold for now.  Every row is marked
``verification_status=unknown`` and correctness is not used as a filter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from rag_retrieve.path_utils import project_root, rebase_legacy_path, stored_path


RAG_FIELDNAMES = [
    "sample_id", "objective", "subcategory", "benchmark", "transform_name",
    "transform_parameters", "dc_collection", "backend", "library",
    "clock_period_ps", "constraint_id", "sdc_path",
    "original_c_path", "optimized_c_path", "original_c_hash", "optimized_c_hash",
    "transform_requested", "transform_applied", "source_changed", "no_op",
    "apply_failure_reason", "transform_summary", "pairing_status",
    "verification_status", "evidence_level", "attribution_weight",
    "total_cell_area", "baseline_area", "area_gain", "area_improvement_pct",
    "delay_ps", "baseline_delay_ps", "delay_improvement_ps",
    "slack_ps", "baseline_slack", "slack_improvement_ps", "slack_status",
    "objective_gain", "number_of_ports", "number_of_cells",
    "number_of_combinational_cells", "number_of_sequential_cells",
    "area_report_path", "timing_report_path", "raw_duplicate_count",
]

_NUMERIC_OUTPUT_FIELDS = {
    "clock_period_ps", "attribution_weight", "total_cell_area", "baseline_area",
    "area_gain", "area_improvement_pct", "slack_ps", "baseline_slack",
    "delay_ps", "baseline_delay_ps", "delay_improvement_ps",
    "slack_improvement_ps", "objective_gain", "number_of_ports", "number_of_cells",
    "number_of_combinational_cells", "number_of_sequential_cells",
    "raw_duplicate_count",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value in (None, "", "undefined", "nan", "NaN"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_float(row: Mapping[str, Any] | None, *keys: str) -> float | None:
    """Return the first finite value under ``keys`` without mixing metrics."""

    if not row:
        return None
    for key in keys:
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _csv_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.10g}"


def _tri_state(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def _objective(collection: str) -> str | None:
    upper = (collection or "").upper()
    if "DC_AREA" in upper:
        return "area"
    if "DC_TIMING" in upper:
        return "timing"
    return None


def _is_baseline(collection: str) -> bool:
    return "BASELINE" in (collection or "").upper()


def _sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_original_c(root: Path, objective: str, subcategory: str, benchmark: str, stem: str = "") -> Path:
    directory = root / f"CDFG_{objective.upper()}" / subcategory / benchmark
    preferred = directory / f"{stem}.c" if stem else Path()
    if stem and preferred.is_file():
        return preferred
    candidates = sorted(path for path in directory.glob("*.c") if path.is_file())
    return candidates[0] if candidates else preferred


def load_transform_metadata(root: str | Path) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    base = project_root(root)
    output: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for objective in ("area", "timing"):
        optimized_root = base / f"optimized_output_{objective.upper()}_LLM"
        if not optimized_root.is_dir():
            continue
        for metadata_path in sorted(optimized_root.glob("*/*/*_transforms.json")):
            relative = metadata_path.relative_to(optimized_root)
            if len(relative.parts) < 3:
                continue
            subcategory, benchmark = relative.parts[0], relative.parts[1]
            source_stem = metadata_path.name[: -len("_transforms.json")]
            original_c = _first_original_c(base, objective, subcategory, benchmark, source_stem)
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for transform_name, raw in payload.items():
                item = dict(raw or {})
                output_file = Path(str(item.get("output_file") or ""))
                optimized_c = optimized_root / output_file
                if not optimized_c.is_file():
                    optimized_c = metadata_path.parent / output_file.name
                if not optimized_c.is_file():
                    optimized_c = base / f"optimized_{objective.upper()}" / subcategory / benchmark / output_file.name
                if not optimized_c.is_file():
                    optimized_c = base / f"CDFG_HLS_{objective.upper()}" / subcategory / benchmark / output_file.name
                output[(objective, subcategory, benchmark, str(transform_name))] = {
                    **item,
                    "metadata_path": metadata_path,
                    "original_c_path": original_c,
                    "optimized_c_path": optimized_c,
                }
    return output


def _normalize_transform(
    raw_transform: str,
    design: str,
    objective: str,
    subcategory: str,
    benchmark: str,
    metadata: Mapping[Tuple[str, str, str, str], Dict[str, Any]],
) -> str:
    candidates = [
        key[3]
        for key in metadata
        if key[:3] == (objective, subcategory, benchmark)
        and (raw_transform == key[3] or raw_transform.endswith("_" + key[3]))
    ]
    if candidates:
        return max(candidates, key=len)
    prefix = design[:-4] if design.endswith("_opt") else design
    if prefix and raw_transform.startswith(prefix + "_"):
        return raw_transform[len(prefix) + 1 :]
    return raw_transform


def _metric_groups(
    rows: Sequence[Dict[str, str]],
    metadata: Mapping[Tuple[str, str, str, str], Dict[str, Any]],
) -> tuple[
    Dict[Tuple[str, str, str, str], List[Dict[str, str]]],
    Dict[Tuple[str, str, str], List[Dict[str, str]]],
]:
    optimized: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    baselines: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        objective = _objective(row.get("dc_collection", ""))
        if objective is None:
            continue
        subcategory = row.get("subcategory", "")
        benchmark = row.get("benchmark", "")
        if _is_baseline(row.get("dc_collection", "")):
            baselines[(objective, subcategory, benchmark)].append(row)
            continue
        transform = _normalize_transform(
            row.get("transform", ""), row.get("design", ""), objective,
            subcategory, benchmark, metadata,
        )
        optimized[(objective, subcategory, benchmark, transform)].append(row)
    return optimized, baselines


def _single(rows: Sequence[Dict[str, str]]) -> Dict[str, str] | None:
    return rows[0] if len(rows) == 1 else None


_CLOCK_RE = re.compile(r"create_clock\b.*?-period\s+([0-9]+(?:\.[0-9]+)?)")


def _constraint_from_report(report_path: str, objective: str, root: Path) -> tuple[float, str]:
    report = rebase_legacy_path(report_path, root)
    run_dir = report.parent.parent
    sdc_candidates = sorted(run_dir.glob("*.sdc")) if run_dir.is_dir() else []
    for sdc in sdc_candidates:
        try:
            match = _CLOCK_RE.search(sdc.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if match:
            return float(match.group(1)), stored_path(sdc, root)
    return (5000.0 if objective == "area" else 1000.0), ""


def _pairing_status(primary_baselines: Sequence[Dict[str, str]]) -> str:
    if len(primary_baselines) == 1:
        return "exact"
    if not primary_baselines:
        return "missing"
    return "ambiguous"


def _sample_id(parts: Iterable[Any]) -> str:
    payload = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def build_normalized_samples(root: str | Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base = project_root(root)
    metadata = load_transform_metadata(base)
    area_rows = _read_csv(base / "LLM_DC_LOG" / "area_metrics_merged.csv")
    timing_rows = _read_csv(base / "LLM_DC_LOG" / "timing_metrics_merged.csv")
    area_opt, area_base = _metric_groups(area_rows, metadata)
    timing_opt, timing_base = _metric_groups(timing_rows, metadata)
    keys = sorted(set(area_opt) | set(timing_opt))

    normalized: List[Dict[str, Any]] = []
    for objective, subcategory, benchmark, transform in keys:
        key4 = (objective, subcategory, benchmark, transform)
        key3 = (objective, subcategory, benchmark)
        area_candidates = area_opt.get(key4, [])
        timing_candidates = timing_opt.get(key4, [])
        area_row = _single(area_candidates)
        timing_row = _single(timing_candidates)
        baseline_area_candidates = area_base.get(key3, [])
        baseline_timing_candidates = timing_base.get(key3, [])
        baseline_area_row = _single(baseline_area_candidates)
        baseline_timing_row = _single(baseline_timing_candidates)
        primary_baselines = baseline_area_candidates if objective == "area" else baseline_timing_candidates
        primary_row = area_row if objective == "area" else timing_row

        meta = metadata.get(key4, {})
        original_c = Path(meta.get("original_c_path") or _first_original_c(base, objective, subcategory, benchmark))
        optimized_c = Path(meta.get("optimized_c_path")) if meta.get("optimized_c_path") else Path("/__missing_optimized_c__")
        original_hash = _sha256(original_c)
        optimized_hash = _sha256(optimized_c)
        actual_changed: bool | None = None
        if original_hash and optimized_hash:
            actual_changed = original_hash != optimized_hash
        elif "differs" in meta:
            actual_changed = bool(meta.get("differs"))
        applied: bool | None = bool(meta.get("applied")) if "applied" in meta else None
        no_op = applied is False or actual_changed is False

        optimized_area = _float(area_row.get("total_cell_area")) if area_row else None
        baseline_area = _float(baseline_area_row.get("total_cell_area")) if baseline_area_row else None
        area_gain = baseline_area - optimized_area if baseline_area is not None and optimized_area is not None else None
        area_pct = (
            area_gain / baseline_area * 100.0
            if area_gain is not None and baseline_area not in (None, 0.0)
            else None
        )
        # Timing is evaluated as critical-path delay (data arrival time), where
        # lower is better.  Slack remains available only as a secondary
        # constraint diagnostic and is never used for objective gain.
        delay = _first_float(timing_row, "delay_ps", "data_arrival_time_ps", "data_arrival_time", "timing_ps")
        baseline_delay = _first_float(
            baseline_timing_row,
            "delay_ps", "data_arrival_time_ps", "data_arrival_time", "timing_ps",
        )
        delay_gain = (
            baseline_delay - delay
            if delay is not None and baseline_delay is not None else None
        )
        slack = _float(timing_row.get("slack_ps")) if timing_row else None
        baseline_slack = _float(baseline_timing_row.get("slack_ps")) if baseline_timing_row else None
        slack_gain = slack - baseline_slack if slack is not None and baseline_slack is not None else None
        objective_gain = area_pct if objective == "area" else delay_gain

        report_source = primary_row or area_row or timing_row or {}
        clock_period_ps, sdc_path = _constraint_from_report(report_source.get("report_path", ""), objective, base)
        collection = report_source.get("dc_collection", "")
        library = (timing_row or baseline_timing_row or {}).get("library", "")
        constraint_id = f"synopsys_dc:{library or 'unknown'}:clock_{clock_period_ps:g}ps"
        duplicate_count = max(len(area_candidates), len(timing_candidates), 1)
        status = _pairing_status(primary_baselines)
        if len(area_candidates) > 1 or len(timing_candidates) > 1:
            status = "duplicate_metric"

        row: Dict[str, Any] = {
            "sample_id": _sample_id((objective, subcategory, benchmark, transform, constraint_id, original_hash, optimized_hash)),
            "objective": objective,
            "subcategory": subcategory,
            "benchmark": benchmark,
            "transform_name": transform,
            "transform_parameters": "",
            "dc_collection": collection,
            "backend": "synopsys_dc",
            "library": library,
            "clock_period_ps": clock_period_ps,
            "constraint_id": constraint_id,
            "sdc_path": sdc_path,
            "original_c_path": stored_path(original_c, base) if original_c else "",
            "optimized_c_path": stored_path(optimized_c, base) if optimized_c.is_file() else "",
            "original_c_hash": original_hash,
            "optimized_c_hash": optimized_hash,
            "transform_requested": "true",
            "transform_applied": _tri_state(applied),
            "source_changed": _tri_state(actual_changed),
            "no_op": "true" if no_op else "false",
            "apply_failure_reason": str(meta.get("summary") or "") if applied is False else "",
            "transform_summary": str(meta.get("summary") or ""),
            "pairing_status": status,
            "verification_status": "unknown",
            "evidence_level": "benchmark",
            "attribution_weight": 1.0,
            "total_cell_area": optimized_area,
            "baseline_area": baseline_area,
            "area_gain": area_gain,
            "area_improvement_pct": area_pct,
            "delay_ps": delay,
            "baseline_delay_ps": baseline_delay,
            "delay_improvement_ps": delay_gain,
            "slack_ps": slack,
            "baseline_slack": baseline_slack,
            "slack_improvement_ps": slack_gain,
            "slack_status": (timing_row or {}).get("slack_status", ""),
            "objective_gain": objective_gain,
            "number_of_ports": _float((area_row or {}).get("number_of_ports")),
            "number_of_cells": _float((area_row or {}).get("number_of_cells")),
            "number_of_combinational_cells": _float((area_row or {}).get("number_of_combinational_cells")),
            "number_of_sequential_cells": _float((area_row or {}).get("number_of_sequential_cells")),
            "area_report_path": stored_path((area_row or {}).get("report_path", ""), base),
            "timing_report_path": stored_path((timing_row or {}).get("report_path", ""), base),
            "raw_duplicate_count": duplicate_count,
        }
        normalized.append(row)

    # Preserve requested transforms that never reached DC.  They carry no PPA
    # gain, but are essential for applicability/no-op statistics.
    metric_keys = set(keys)
    for key4, meta in sorted(metadata.items()):
        if key4 in metric_keys:
            continue
        objective, subcategory, benchmark, transform = key4
        original_c = Path(meta.get("original_c_path")) if meta.get("original_c_path") else Path("/__missing_original_c__")
        optimized_c = Path(meta.get("optimized_c_path")) if meta.get("optimized_c_path") else Path("/__missing_optimized_c__")
        original_hash = _sha256(original_c)
        optimized_hash = _sha256(optimized_c)
        actual_changed: bool | None = None
        if original_hash and optimized_hash:
            actual_changed = original_hash != optimized_hash
        elif "differs" in meta:
            actual_changed = bool(meta.get("differs"))
        applied: bool | None = bool(meta.get("applied")) if "applied" in meta else None
        no_op = applied is False or actual_changed is False
        clock_period_ps = 5000.0 if objective == "area" else 1000.0
        constraint_id = f"not_evaluated:clock_{clock_period_ps:g}ps"
        normalized.append({
            "sample_id": _sample_id(("metadata_only", objective, subcategory, benchmark, transform, original_hash, optimized_hash)),
            "objective": objective,
            "subcategory": subcategory,
            "benchmark": benchmark,
            "transform_name": transform,
            "transform_parameters": "",
            "dc_collection": "",
            "backend": "not_evaluated",
            "library": "",
            "clock_period_ps": clock_period_ps,
            "constraint_id": constraint_id,
            "sdc_path": "",
            "original_c_path": stored_path(original_c, base) if original_c.is_file() else "",
            "optimized_c_path": stored_path(optimized_c, base) if optimized_c.is_file() else "",
            "original_c_hash": original_hash,
            "optimized_c_hash": optimized_hash,
            "transform_requested": "true",
            "transform_applied": _tri_state(applied),
            "source_changed": _tri_state(actual_changed),
            "no_op": "true" if no_op else "false",
            "apply_failure_reason": str(meta.get("summary") or "") if applied is False else "",
            "transform_summary": str(meta.get("summary") or ""),
            "pairing_status": "no_metric",
            "verification_status": "unknown",
            "evidence_level": "benchmark",
            "attribution_weight": 1.0,
            "total_cell_area": None,
            "baseline_area": None,
            "area_gain": None,
            "area_improvement_pct": None,
            "delay_ps": None,
            "baseline_delay_ps": None,
            "delay_improvement_ps": None,
            "slack_ps": None,
            "baseline_slack": None,
            "slack_improvement_ps": None,
            "slack_status": "",
            "objective_gain": None,
            "number_of_ports": None,
            "number_of_cells": None,
            "number_of_combinational_cells": None,
            "number_of_sequential_cells": None,
            "area_report_path": "",
            "timing_report_path": "",
            "raw_duplicate_count": 0,
        })

    summary = {
        "sample_count": len(normalized),
        "objective_counts": {
            objective: sum(1 for row in normalized if row["objective"] == objective)
            for objective in ("area", "timing")
        },
        "pairing_counts": dict(sorted(
            (status, sum(1 for row in normalized if row["pairing_status"] == status))
            for status in {row["pairing_status"] for row in normalized}
        )),
        "applied_counts": dict(sorted(
            (state, sum(1 for row in normalized if row["transform_applied"] == state))
            for state in {row["transform_applied"] for row in normalized}
        )),
        "no_op_count": sum(1 for row in normalized if row["no_op"] == "true"),
        "metric_sample_count": sum(1 for row in normalized if row["pairing_status"] == "exact"),
        "metadata_only_sample_count": sum(1 for row in normalized if row["pairing_status"] == "no_metric"),
        "metadata_record_count": len(metadata),
        "verification_status": "unknown",
    }
    return normalized, summary


def write_samples_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAG_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for field in _NUMERIC_OUTPUT_FIELDS:
                row[field] = _csv_number(row.get(field))
            writer.writerow({field: row.get(field, "") for field in RAG_FIELDNAMES})


def load_manifest_samples(manifest_path: str | Path, root: str | Path) -> List[Dict[str, Any]]:
    """Load a new batch expressed in the normalized public ingest schema.

    The manifest may contain ``samples`` inline or point ``samples_file`` to a
    JSON list / CSV table.  Batch defaults are copied onto rows when absent.
    """
    base = project_root(root)
    manifest_file = Path(manifest_path).expanduser().resolve()
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    data_root = Path(payload.get("data_root") or manifest_file.parent)
    if not data_root.is_absolute():
        data_root = (manifest_file.parent / data_root).resolve()
    raw_samples = payload.get("samples")
    if raw_samples is None:
        samples_file = Path(str(payload.get("samples_file") or ""))
        if not samples_file.is_absolute():
            samples_file = data_root / samples_file
        if samples_file.suffix.lower() == ".csv":
            raw_samples = _read_csv(samples_file)
        else:
            raw_samples = json.loads(samples_file.read_text(encoding="utf-8"))
    if not isinstance(raw_samples, list):
        raise ValueError(f"manifest samples must be a list: {manifest_file}")

    defaults = {
        key: payload.get(key)
        for key in ("objective", "subcategory", "backend", "library", "clock_period_ps", "constraint_id")
        if payload.get(key) not in (None, "")
    }
    rows: List[Dict[str, Any]] = []
    for raw in raw_samples:
        row = {**defaults, **dict(raw)}
        objective = str(row.get("objective") or "").lower()
        if objective not in {"area", "timing"}:
            raise ValueError(f"new sample has invalid objective: {row.get('objective')!r}")
        for field in ("original_c_path", "optimized_c_path", "sdc_path", "area_report_path", "timing_report_path"):
            value = str(row.get(field) or "")
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = data_root / path
            row[field] = stored_path(path, base)
        original = resolve_manifest_path(row.get("original_c_path", ""), base)
        optimized = resolve_manifest_path(row.get("optimized_c_path", ""), base)
        row["original_c_hash"] = row.get("original_c_hash") or _sha256(original)
        row["optimized_c_hash"] = row.get("optimized_c_hash") or _sha256(optimized)
        if row.get("source_changed") in (None, "") and row["original_c_hash"] and row["optimized_c_hash"]:
            row["source_changed"] = _tri_state(row["original_c_hash"] != row["optimized_c_hash"])
        row.setdefault("transform_requested", "true")
        row.setdefault("transform_applied", "unknown")
        changed = str(row.get("source_changed") or "unknown").lower()
        applied = str(row.get("transform_applied") or "unknown").lower()
        row["no_op"] = str(row.get("no_op") or ("true" if changed == "false" or applied == "false" else "false")).lower()
        row.setdefault("verification_status", "unknown")
        row.setdefault("evidence_level", "benchmark")
        row.setdefault("attribution_weight", 1.0)
        row.setdefault("pairing_status", "exact")
        baseline_area = _float(row.get("baseline_area"))
        optimized_area = _float(row.get("total_cell_area", row.get("optimized_area")))
        area_gain = _float(row.get("area_gain"))
        if area_gain is None and baseline_area is not None and optimized_area is not None:
            area_gain = baseline_area - optimized_area
        area_pct = _float(row.get("area_improvement_pct"))
        if area_pct is None and area_gain is not None and baseline_area not in (None, 0.0):
            area_pct = area_gain / baseline_area * 100.0
        baseline_delay = _first_float(
            row,
            "baseline_delay_ps", "baseline_data_arrival_time_ps", "baseline_data_arrival_time",
            "baseline_timing_ps", "baseline_timing",
        )
        delay = _first_float(
            row,
            "delay_ps", "data_arrival_time_ps", "data_arrival_time", "timing_ps", "timing",
            "optimized_delay_ps", "optimized_timing_ps", "optimized_timing",
        )
        delay_gain = _float(row.get("delay_improvement_ps"))
        if delay_gain is None and delay is not None and baseline_delay is not None:
            delay_gain = baseline_delay - delay
        baseline_slack = _float(row.get("baseline_slack"))
        slack = _float(row.get("slack_ps", row.get("optimized_slack")))
        slack_gain = _float(row.get("slack_improvement_ps"))
        if slack_gain is None and slack is not None and baseline_slack is not None:
            slack_gain = slack - baseline_slack
        row.update({
            "total_cell_area": optimized_area,
            "baseline_area": baseline_area,
            "area_gain": area_gain,
            "area_improvement_pct": area_pct,
            "delay_ps": delay,
            "baseline_delay_ps": baseline_delay,
            "delay_improvement_ps": delay_gain,
            "slack_ps": slack,
            "baseline_slack": baseline_slack,
            "slack_improvement_ps": slack_gain,
            "objective_gain": area_pct if objective == "area" else delay_gain,
        })
        clock = _float(row.get("clock_period_ps")) or (5000.0 if objective == "area" else 1000.0)
        row["clock_period_ps"] = clock
        row.setdefault("constraint_id", f"{row.get('backend', 'unknown')}:clock_{clock:g}ps")
        row["sample_id"] = row.get("sample_id") or _sample_id((
            payload.get("batch_id", manifest_file.stem), objective, row.get("subcategory"),
            row.get("benchmark"), row.get("transform_name"), row.get("constraint_id"),
            row.get("original_c_hash"), row.get("optimized_c_hash"),
        ))
        rows.append({field: row.get(field, "") for field in RAG_FIELDNAMES})
    return rows


def resolve_manifest_path(value: str | Path, root: Path) -> Path:
    if not value:
        return Path()
    return rebase_legacy_path(value, root)
