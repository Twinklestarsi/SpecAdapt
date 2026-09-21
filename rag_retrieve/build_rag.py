"""Single, versioned entry point for rebuilding the historical RAG."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections import Counter
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rag_retrieve.cdfg_index import build_joined_index
from rag_retrieve.defaults import ACTIVE_RAG_MANIFEST, versioned_rag_artifacts
from rag_retrieve.path_utils import project_root
from rag_retrieve.region_index import build_historical_region_index
from rag_retrieve.sample_loader import (
    build_normalized_samples,
    load_manifest_samples,
    write_samples_csv,
)
from rag_retrieve.transform_stats import summarize_transform_rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _atomic_publish(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _validate(
    rows: List[Dict[str, Any]],
    joined: Dict[str, Any],
    region_index: Dict[str, Any],
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    ids = [str(row.get("sample_id") or "") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_sample_ids")
    invalid_objectives = sorted({str(row.get("objective")) for row in rows if row.get("objective") not in {"area", "timing"}})
    if invalid_objectives:
        errors.append(f"invalid_objectives:{invalid_objectives}")
    for entry in joined.get("entries", []):
        objective = entry.get("cdfg_record", {}).get("objective")
        if any(row.get("objective") != objective for row in entry.get("rag_rows", [])):
            errors.append("objective_leak_in_joined_index")
            break
    # 检测索引里残留的旧绝对路径。标记串通过 VIVADO_LEGACY_PATH_MARKER 配置；
    # 未设置时跳过这项检查（从零构建的索引不会有历史遗留路径）。
    legacy_marker = os.environ.get("VIVADO_LEGACY_PATH_MARKER", "").strip()
    if legacy_marker:
        old_paths = [text for text in _walk_strings({"joined": joined, "region": region_index}) if legacy_marker in text]
        if old_paths:
            errors.append(f"legacy_absolute_paths:{len(old_paths)}")
    if any(entry.get("top_transforms") and not entry.get("transform_evidence") for entry in region_index.get("entries", [])):
        errors.append("benchmark_priors_copied_into_region_top_transforms")
    region_groups = region_index.get("region_groups", [])
    if any(group.get("top_transforms") and not group.get("transform_evidence") for group in region_groups):
        errors.append("empty_composite_region_evidence")
    group_ids = [str((entry.get("group") or {}).get("group_id") or "") for entry in region_groups]
    if any(not group_id for group_id in group_ids):
        errors.append("missing_composite_region_id")
    if len(group_ids) != len(set(group_ids)):
        errors.append("duplicate_composite_region_ids")
    atomic_region_keys = {
        (
            str(entry.get("objective") or ""),
            str(entry.get("subcategory") or ""),
            str(entry.get("benchmark") or ""),
            str(entry.get("graph_function") or ""),
            str((entry.get("region") or {}).get("region_id") or ""),
        )
        for entry in region_index.get("entries", [])
    }
    for entry in region_groups:
        group = entry.get("group") or {}
        members = group.get("member_regions") or []
        if len(members) < 2 or int(group.get("member_region_count") or 0) != len(members):
            errors.append("invalid_composite_region_members")
            break
        if group.get("member_semantics") != "alternative_candidates_not_individually_verified":
            errors.append("invalid_composite_region_member_semantics")
            break
        if any(
            (
                str(entry.get("objective") or ""),
                str(entry.get("subcategory") or ""),
                str(entry.get("benchmark") or ""),
                str(member.get("graph_function") or ""),
                str(member.get("region_id") or ""),
            ) not in atomic_region_keys
            for member in members
        ):
            errors.append("missing_composite_region_member")
            break
        if any(summary.get("objective") != entry.get("objective") for summary in entry.get("transform_evidence", [])):
            errors.append("objective_leak_in_composite_region")
            break
        if any(
            "region_group" not in (summary.get("evidence_levels") or [])
            for summary in entry.get("transform_evidence", [])
        ):
            errors.append("invalid_composite_region_evidence_level")
            break
    pairing_counts = Counter(str(row.get("pairing_status") or "") for row in rows)
    if pairing_counts.get("exact", 0) == 0:
        errors.append("no_exactly_paired_samples")
    if pairing_counts.get("missing", 0) or pairing_counts.get("ambiguous", 0) or pairing_counts.get("duplicate_metric", 0):
        warnings.append(f"non_exact_pairing:{dict(pairing_counts)}")
    local_count = int(region_index.get("summary", {}).get("region_entries_with_local_evidence", 0))
    group_local_count = int(region_index.get("summary", {}).get("composite_regions_with_local_evidence", 0))
    if local_count == 0 and group_local_count == 0:
        warnings.append("no_region_local_evidence; retrieval will use discounted benchmark priors")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "unique_sample_ids": len(set(ids)),
            "joined_entry_count": len(joined.get("entries", [])),
            "region_entry_count": len(region_index.get("entries", [])),
            "region_entries_with_local_evidence": local_count,
            "composite_region_count": len(region_groups),
            "composite_regions_with_local_evidence": group_local_count,
            "legacy_absolute_path_count": len(old_paths),
            "pairing_counts": dict(sorted(pairing_counts.items())),
        },
    }


def build_rag(
    *,
    root: str | Path,
    version: str = "v3",
    input_manifests: List[str] | None = None,
    dry_run: bool = False,
    cdfg_area_root: str | Path | None = None,
    cdfg_timing_root: str | Path | None = None,
    activate: bool = True,
) -> Dict[str, Any]:
    base = project_root(root)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", version):
        raise ValueError(f"unsafe version name: {version!r}")
    rows, source_summary = build_normalized_samples(base)
    added_counts: Dict[str, int] = {}
    for manifest in input_manifests or []:
        additional = load_manifest_samples(manifest, base)
        rows.extend(additional)
        added_counts[str(Path(manifest).resolve())] = len(additional)

    deduplicated: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        deduplicated[str(row.get("sample_id") or "")] = row
    rows = sorted(deduplicated.values(), key=lambda row: (
        str(row.get("objective")), str(row.get("subcategory")),
        str(row.get("benchmark")), str(row.get("transform_name")), str(row.get("sample_id")),
    ))

    artifact_set = versioned_rag_artifacts(base, version)
    outputs = artifact_set.versioned_paths()
    active_manifest_path = base / ACTIVE_RAG_MANIFEST
    area_root = Path(cdfg_area_root) if cdfg_area_root else base / "CDFG_AREA"
    timing_root = Path(cdfg_timing_root) if cdfg_timing_root else base / "CDFG_TIMING"
    if not area_root.is_absolute():
        area_root = base / area_root
    if not timing_root.is_absolute():
        timing_root = base / timing_root
    cdfg_roots = {
        "area": area_root.resolve(),
        "timing": timing_root.resolve(),
    }
    temp_parent = base / "rag_retrieve"
    with tempfile.TemporaryDirectory(prefix=".rag_build_", dir=temp_parent) as temp_name:
        temp = Path(temp_name)
        csv_path = temp / "knowledge.csv"
        write_samples_csv(csv_path, rows)
        joined = build_joined_index(
            csv_path=csv_path,
            cdfg_roots=cdfg_roots,
            project_root=base,
        )
        joined["summary"]["csv_path"] = str(outputs["knowledge_base"].relative_to(base))
        region_index = build_historical_region_index(joined, root=base)
        transform_stats = {
            "schema_version": 3,
            "area": summarize_transform_rows(rows, "area"),
            "timing": summarize_transform_rows(rows, "timing"),
        }
        validation = _validate(rows, joined, region_index)
        knowledge_summary = {
            **source_summary,
            "schema_version": 3,
            "version": version,
            "sample_count_after_manifest_merge": len(rows),
            "input_manifests": added_counts,
            "verification_status": "unknown",
        }
        report = {
            "schema_version": 4,
            "version": version,
            "dry_run": dry_run,
            "activate": activate,
            "source_summary": knowledge_summary,
            "joined_summary": joined.get("summary", {}),
            "region_summary": region_index.get("summary", {}),
            "validation": validation,
            "outputs": {key: str(path.relative_to(base)) for key, path in outputs.items()},
            "active_manifest": str(active_manifest_path.relative_to(base)),
        }
        active_manifest = {
            "schema_version": 1,
            "version": version,
            "artifacts": {
                key: str(path.relative_to(base)) for key, path in outputs.items()
            },
        }
        temp_files = {
            "knowledge_base": csv_path,
            "knowledge_summary": temp / "knowledge_summary.json",
            "joined_index": temp / "joined.json",
            "region_index": temp / "regions.json",
            "transform_stats": temp / "transform_stats.json",
            "build_report": temp / "report.json",
            "active_manifest": temp / "latest_rag.json",
        }
        _write_json(temp_files["knowledge_summary"], knowledge_summary)
        _write_json(temp_files["joined_index"], joined)
        _write_json(temp_files["region_index"], region_index)
        _write_json(temp_files["transform_stats"], transform_stats)
        _write_json(temp_files["build_report"], report)
        _write_json(temp_files["active_manifest"], active_manifest)
        if not validation["ok"]:
            raise RuntimeError("RAG validation failed: " + "; ".join(validation["errors"]))
        if not dry_run:
            for key, target in outputs.items():
                _atomic_publish(temp_files[key], target)
            if activate:
                # Publish the pointer last: readers either see the previous complete
                # release or this newly completed one, never a partial rebuild.
                _atomic_publish(temp_files["active_manifest"], active_manifest_path)
        return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild a versioned historical RAG from raw data.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--version", default="v3")
    parser.add_argument("--input-manifest", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cdfg-area-root", type=Path, default=None)
    parser.add_argument("--cdfg-timing-root", type=Path, default=None)
    parser.add_argument(
        "--no-activate", action="store_true",
        help="Publish versioned artifacts without changing latest_rag.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_rag(
        root=args.project_root or project_root(), version=args.version,
        input_manifests=args.input_manifest, dry_run=args.dry_run,
        cdfg_area_root=args.cdfg_area_root,
        cdfg_timing_root=args.cdfg_timing_root,
        activate=not args.no_activate,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
