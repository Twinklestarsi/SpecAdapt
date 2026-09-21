"""
Build a CDFG corpus index and join it with the current RAG CSV.

This solves the first integration problem for Module 4:
existing CDFG artifacts exist on disk, but the RAG layer does not yet
know how to address them as structured retrieval data.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from rag_retrieve.schema import (
    CDFGBenchmarkRecord,
    CDFGGraphStats,
    CDFGRAGLinkedRecord,
    RAGCSVRow,
)
from rag_retrieve.defaults import get_active_rag_artifacts
from rag_retrieve.path_utils import stored_path

_NODE_RE = re.compile(
    r'^\s*(n\d+)\s+\[label="((?:\\.|[^"\\])*)"(?:,\s*.*)?\];\s*$'
)
_EDGE_RE = re.compile(r'^\s*(n\d+)\s*->\s*(n\d+)\s+\[color="([^"]+)"\];\s*$')

_NUMERIC_FIELDS = {
    "total_cell_area",
    "baseline_area",
    "area_improvement_pct",
    "area_gain",
    "delay_ps",
    "baseline_delay_ps",
    "delay_improvement_ps",
    "slack_ps",
    "baseline_slack",
    "slack_improvement_ps",
    "objective_gain",
    "clock_period_ps",
    "attribution_weight",
    "number_of_ports",
    "number_of_cells",
    "number_of_combinational_cells",
    "number_of_sequential_cells",
    "raw_duplicate_count",
}


def _project_root(start: Path | None = None) -> Path:
    return (start or Path(__file__).resolve().parent.parent).resolve()


def _pick_first(paths: Sequence[Path], suffixes: Tuple[str, ...], root: Path) -> str:
    for path in sorted(paths):
        if path.suffix.lower() in suffixes:
            return stored_path(path, root)
    return ""


def _normalize_opcode(label: str) -> str:
    text = label.strip()
    if not text:
        return "unknown"
    if text.startswith("ARG "):
        return "arg"
    if " = " in text:
        text = text.split(" = ", 1)[1].strip()
    return text.split(None, 1)[0].lower()


def _parse_cdfg_dot(dot_path: Path, project_root: Path | None = None) -> CDFGGraphStats:
    node_count = 0
    edge_count = 0
    data_edge_count = 0
    control_edge_count = 0
    opcode_histogram: Dict[str, int] = defaultdict(int)

    with dot_path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            node_match = _NODE_RE.match(raw_line)
            if node_match:
                node_count += 1
                opcode_histogram[_normalize_opcode(node_match.group(2))] += 1
                continue

            edge_match = _EDGE_RE.match(raw_line)
            if edge_match:
                edge_count += 1
                color = edge_match.group(3)
                if color == "black":
                    data_edge_count += 1
                elif color == "gray50":
                    control_edge_count += 1

    function_name = dot_path.name.replace(".cdfg.dot", "")
    root = _project_root(project_root)
    png_path = stored_path(dot_path.with_suffix(".png"), root)
    return CDFGGraphStats(
        function_name=function_name,
        dot_path=stored_path(dot_path, root),
        png_path=png_path if dot_path.with_suffix(".png").exists() else "",
        node_count=node_count,
        edge_count=edge_count,
        data_edge_count=data_edge_count,
        control_edge_count=control_edge_count,
        opcode_histogram=dict(sorted(opcode_histogram.items())),
    )


def scan_cdfg_root(root: Path, objective: str, project_root: Path | None = None) -> List[CDFGBenchmarkRecord]:
    records: List[CDFGBenchmarkRecord] = []
    base = _project_root(project_root)
    if not root.exists():
        return records

    for subcategory_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        subcategory = subcategory_dir.name
        for benchmark_dir in sorted(path for path in subcategory_dir.iterdir() if path.is_dir()):
            top_level_files = [path for path in benchmark_dir.iterdir() if path.is_file()]
            cdfg_dirs = sorted(
                path
                for path in benchmark_dir.iterdir()
                if path.is_dir() and path.name.startswith("cdfg_")
            )
            cdfg_dir = cdfg_dirs[0] if cdfg_dirs else None
            graph_paths = []
            if cdfg_dir:
                graph_paths = sorted((cdfg_dir / "cdfg_graph").glob("*.cdfg.dot"))

            records.append(
                CDFGBenchmarkRecord(
                    objective=objective,
                    subcategory=subcategory,
                    benchmark=benchmark_dir.name,
                    benchmark_dir=stored_path(benchmark_dir, base),
                    c_path=_pick_first(top_level_files, (".c",), base),
                    cpp_path=_pick_first(top_level_files, (".cpp",), base),
                    verilog_path=_pick_first(top_level_files, (".v",), base),
                    cdfg_dir=stored_path(cdfg_dir, base) if cdfg_dir else "",
                    graphs=[_parse_cdfg_dot(dot_path, base) for dot_path in graph_paths],
                )
            )

    return records


def load_rag_csv(csv_path: Path) -> Dict[Tuple[str, str, str], List[RAGCSVRow]]:
    grouped: Dict[Tuple[str, str, str], List[RAGCSVRow]] = defaultdict(list)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized: Dict[str, Any] = {}
            for key, value in row.items():
                if key in _NUMERIC_FIELDS:
                    normalized[key] = float(value) if value not in ("", None) else None
                else:
                    normalized[key] = value

            record = RAGCSVRow(**normalized)
            grouped[(record.objective, record.benchmark, record.subcategory)].append(record)
    return grouped


def join_cdfg_with_rag(
    cdfg_records: Iterable[CDFGBenchmarkRecord],
    rag_rows: Dict[Tuple[str, str, str], List[RAGCSVRow]],
) -> List[CDFGRAGLinkedRecord]:
    linked: List[CDFGRAGLinkedRecord] = []
    for record in cdfg_records:
        key = (record.objective, record.benchmark, record.subcategory)
        rows = list(rag_rows.get(key, []))
        warnings: List[str] = []
        if not rows:
            warnings.append("missing_rag_rows")
        if not record.graphs:
            warnings.append("missing_cdfg_graphs")
        linked.append(CDFGRAGLinkedRecord(cdfg_record=record, rag_rows=rows, warnings=warnings))
    return linked


def build_joined_index(
    csv_path: Path,
    cdfg_roots: Dict[str, Path],
    project_root: Path | None = None,
) -> Dict[str, Any]:
    project = _project_root(project_root or csv_path.parent.parent)
    all_records: List[CDFGBenchmarkRecord] = []
    for objective, cdfg_root in cdfg_roots.items():
        all_records.extend(scan_cdfg_root(cdfg_root, objective=objective, project_root=project))

    rag_rows = load_rag_csv(csv_path)
    linked_records = join_cdfg_with_rag(all_records, rag_rows)

    unique_cdfg_keys = {(item.cdfg_record.objective, item.cdfg_record.benchmark, item.cdfg_record.subcategory) for item in linked_records}
    unique_csv_keys = set(rag_rows)
    matched_keys = unique_cdfg_keys & unique_csv_keys

    summary = {
        "csv_path": stored_path(csv_path, project),
        "cdfg_roots": {name: stored_path(path, project) for name, path in cdfg_roots.items()},
        "cdfg_record_count": len(linked_records),
        "cdfg_unique_benchmarks": len(unique_cdfg_keys),
        "rag_unique_benchmarks": len(unique_csv_keys),
        "matched_unique_benchmarks": len(matched_keys),
        "records_missing_rag_rows": sum(1 for item in linked_records if "missing_rag_rows" in item.warnings),
        "records_missing_cdfg_graphs": sum(1 for item in linked_records if "missing_cdfg_graphs" in item.warnings),
    }

    return {
        "summary": summary,
        "entries": [item.to_dict() for item in linked_records],
    }


def build_default_joined_index(project_root: Path | None = None) -> Dict[str, Any]:
    root = _project_root(project_root)
    csv_path = get_active_rag_artifacts(root).knowledge_base
    cdfg_roots = {
        "area": root / "CDFG_AREA",
        "timing": root / "CDFG_TIMING",
    }
    return build_joined_index(csv_path=csv_path, cdfg_roots=cdfg_roots)


def write_joined_index(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
