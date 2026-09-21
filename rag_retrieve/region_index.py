"""Build an evidence-aware historical region index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from rag_retrieve.path_utils import project_root, resolve_stored_path, stored_path
from rag_retrieve.region_extract import extract_regions_from_graph
from rag_retrieve.transform_region_mapper import map_transform_rows_to_regions
from rag_retrieve.transform_stats import summarize_transform_rows
from rag_retrieve.defaults import DEFAULT_HISTORICAL_REGION_INDEX, DEFAULT_JOINED_INDEX


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _context_values(rows: List[Dict[str, Any]], key: str) -> List[Any]:
    values = {row.get(key) for row in rows if row.get(key) not in (None, "")}
    return sorted(values, key=str)


def _entry_has_evidence_level(entry: Dict[str, Any], level: str) -> bool:
    return any(
        level in (summary.get("evidence_levels") or [])
        for summary in entry.get("transform_evidence", [])
    )


def _effective_transform_summaries(
    rows: List[Dict[str, Any]],
    objective: str,
) -> List[Dict[str, Any]]:
    """Keep only summaries backed by at least one usable metric sample."""

    return [
        summary
        for summary in summarize_transform_rows(rows, objective)
        if int(summary.get("raw_effective_sample_count") or 0) > 0
    ]


def build_historical_region_index(
    joined_index: Dict[str, Any],
    root: str | Path | None = None,
) -> Dict[str, Any]:
    base = project_root(root)
    entries: List[Dict[str, Any]] = []
    composite_region_groups: List[Dict[str, Any]] = []
    benchmark_priors: Dict[str, List[Dict[str, Any]]] = {}
    region_count = 0
    graph_count = 0
    mapping_totals = {
        "row_count": 0,
        "mapped_row_count": 0,
        "atomic_low_confidence_mapped_row_count": 0,
        "composite_mapped_row_count": 0,
        "region_attributed_row_count": 0,
        "benchmark_only_row_count": 0,
        "mapping_failures": {},
        "source_location_attribution": {
            "rows_with_reliable_column_spans": 0,
            "mapped_rows_with_column_match": 0,
            "attributed_rows_with_column_match": 0,
        },
        "ambiguity_resolution": {
            "broad_candidate_row_count": 0,
            "unique_top_low_confidence_row_count": 0,
            "exact_top_tie_composite_row_count": 0,
            "unresolved_ambiguous_row_count": 0,
            "low_confidence_attribution_weight": 0.25,
        },
    }

    for item in joined_index.get("entries", []):
        cdfg_record = item.get("cdfg_record", {})
        rag_rows = [dict(row) for row in item.get("rag_rows", [])]
        objective = cdfg_record.get("objective", "area")
        subcategory = cdfg_record.get("subcategory", "")
        benchmark = cdfg_record.get("benchmark", "")
        prior_key = f"{objective}:{subcategory}:{benchmark}"
        benchmark_priors[prior_key] = summarize_transform_rows(rag_rows, objective)

        source_path = resolve_stored_path(cdfg_record.get("c_path", ""), base)
        source_lines: List[str] = []
        if source_path.is_file():
            source_lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()

        extracted: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
        all_region_dicts: List[Dict[str, Any]] = []
        for graph in cdfg_record.get("graphs", []):
            dot_value = graph.get("dot_path")
            if not dot_value:
                continue
            dot_path = resolve_stored_path(dot_value, base)
            if not dot_path.is_file():
                continue
            graph_count += 1
            regions = extract_regions_from_graph(
                dot_path=dot_path,
                graph_function=graph.get("function_name"),
                source_lines=source_lines,
            )
            for region in regions:
                region_dict = region.to_dict()
                all_region_dicts.append(region_dict)
                extracted.append((graph, region_dict))

        by_region, by_composite_group, _benchmark_only, mapping_summary = map_transform_rows_to_regions(
            rag_rows, all_region_dicts, base
        )
        for metric in (
            "row_count",
            "mapped_row_count",
            "atomic_low_confidence_mapped_row_count",
            "composite_mapped_row_count",
            "region_attributed_row_count",
            "benchmark_only_row_count",
        ):
            mapping_totals[metric] += int(mapping_summary.get(metric, 0))
        for reason, count in mapping_summary["mapping_failures"].items():
            mapping_totals["mapping_failures"][reason] = mapping_totals["mapping_failures"].get(reason, 0) + int(count)
        for metric, count in mapping_summary.get("source_location_attribution", {}).items():
            mapping_totals["source_location_attribution"][metric] = (
                mapping_totals["source_location_attribution"].get(metric, 0) + int(count)
            )
        for metric, count in mapping_summary.get("ambiguity_resolution", {}).items():
            if metric == "low_confidence_attribution_weight":
                mapping_totals["ambiguity_resolution"][metric] = float(count)
                continue
            mapping_totals["ambiguity_resolution"][metric] = (
                mapping_totals["ambiguity_resolution"].get(metric, 0) + int(count)
            )

        backends = _context_values(rag_rows, "backend")
        constraint_ids = _context_values(rag_rows, "constraint_id")
        clock_periods = _context_values(rag_rows, "clock_period_ps")
        for graph, region in extracted:
            region_count += 1
            local_rows = by_region.get(str(region.get("region_id")), [])
            local_evidence = _effective_transform_summaries(local_rows, objective)
            entries.append(
                {
                    "objective": objective,
                    "subcategory": subcategory,
                    "benchmark": benchmark,
                    "benchmark_dir": cdfg_record.get("benchmark_dir", ""),
                    "source_c_path": cdfg_record.get("c_path", ""),
                    "graph_function": graph.get("function_name", ""),
                    "graph_dot_path": stored_path(resolve_stored_path(graph.get("dot_path", ""), base), base),
                    "region": region,
                    "transform_evidence": local_evidence,
                    "top_transforms": local_evidence,
                    "benchmark_prior_key": prior_key,
                    "backends": backends,
                    "constraint_ids": constraint_ids,
                    "clock_periods_ps": clock_periods,
                    "warnings": item.get("warnings", []),
                }
            )

        for group_payload in by_composite_group.values():
            group = dict(group_payload.get("group") or {})
            local_rows = list(group_payload.get("rows") or [])
            local_evidence = _effective_transform_summaries(local_rows, objective)
            composite_region_groups.append(
                {
                    "objective": objective,
                    "subcategory": subcategory,
                    "benchmark": benchmark,
                    "benchmark_dir": cdfg_record.get("benchmark_dir", ""),
                    "source_c_path": cdfg_record.get("c_path", ""),
                    "group": group,
                    "transform_evidence": local_evidence,
                    "top_transforms": local_evidence,
                    "benchmark_prior_key": prior_key,
                    "backends": backends,
                    "constraint_ids": constraint_ids,
                    "clock_periods_ps": clock_periods,
                    "warnings": item.get("warnings", []),
                }
            )

    summary = {
        "schema_version": 4,
        "source_joined_index": joined_index.get("summary", {}),
        "historical_region_count": region_count,
        "historical_graph_count": graph_count,
        "historical_entry_count": len(entries),
        "historical_composite_region_count": len(composite_region_groups),
        "benchmark_prior_count": len(benchmark_priors),
        "region_entries_with_local_evidence": sum(bool(entry["transform_evidence"]) for entry in entries),
        "region_entries_with_high_confidence_evidence": sum(
            _entry_has_evidence_level(entry, "region") for entry in entries
        ),
        "region_entries_with_low_confidence_evidence": sum(
            _entry_has_evidence_level(entry, "region_low_confidence") for entry in entries
        ),
        "composite_regions_with_local_evidence": sum(
            bool(group["transform_evidence"]) for group in composite_region_groups
        ),
        "mapping": mapping_totals,
        "path_format": "project_relative",
    }
    return {
        "summary": summary,
        "benchmark_priors": benchmark_priors,
        "entries": entries,
        "region_groups": composite_region_groups,
    }


def write_region_index(output_path: str | Path, payload: Dict[str, Any]) -> None:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an evidence-aware historical region index.")
    parser.add_argument("--joined-index", default=str(DEFAULT_JOINED_INDEX))
    parser.add_argument("--output", default=str(DEFAULT_HISTORICAL_REGION_INDEX))
    parser.add_argument("--project-root", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_historical_region_index(_load_json(args.joined_index), root=args.project_root)
    write_region_index(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
