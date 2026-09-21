"""
Build query-side CDFG artifacts for Module 4 retrieval.

This module is for query inputs only:
Module 3 generated C files are converted into temporary CDFG artifacts
under /tmp (or another caller-specified root), then normalized into a
record compatible with the historical CDFG index schema.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rag_retrieve.cdfg_index import _parse_cdfg_dot
from rag_retrieve.schema import QueryCDFGRecord
from rag_retrieve.path_utils import project_root, resolve_stored_path, stored_path
from vcpp_cdfg import process_verilog_file_cdfg


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)


def extract_query_cdfg(
    c_path: str | Path,
    query_root: str | Path = "/tmp/module4_queries",
    benchmark: str | None = None,
) -> QueryCDFGRecord:
    root = project_root()
    source_c_path = resolve_stored_path(c_path, root)
    if not source_c_path.exists():
        raise FileNotFoundError(f"C file not found: {source_c_path}")
    if source_c_path.suffix.lower() != ".c":
        raise ValueError(f"Expected a .c file, got: {source_c_path}")

    benchmark_name = benchmark or source_c_path.stem
    safe_benchmark = _sanitize_name(benchmark_name)
    query_root_path = Path(query_root).resolve() / safe_benchmark
    query_root_path.mkdir(parents=True, exist_ok=True)

    copied_c_path = query_root_path / f"{safe_benchmark}.c"
    shutil.copy2(source_c_path, copied_c_path)

    ok = process_verilog_file_cdfg(str(copied_c_path))
    if not ok:
        raise RuntimeError(f"CDFG extraction failed for {source_c_path}")

    cdfg_dir = query_root_path / f"cdfg_{safe_benchmark}"
    graph_dir = cdfg_dir / "cdfg_graph"
    graphs = [_parse_cdfg_dot(dot_path) for dot_path in sorted(graph_dir.glob("*.cdfg.dot"))]

    return QueryCDFGRecord(
        benchmark=benchmark_name,
        source_c_path=stored_path(source_c_path, root),
        query_root=stored_path(query_root_path, root),
        copied_c_path=stored_path(copied_c_path, root),
        cdfg_dir=stored_path(cdfg_dir, root),
        graphs=graphs,
    )


def write_query_record(output_path: str | Path, record: QueryCDFGRecord) -> None:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract query-side CDFG artifacts from a Module 3 C file."
    )
    parser.add_argument("c_path", help="Path to the input .c file")
    parser.add_argument(
        "--query-root",
        default="/tmp/module4_queries",
        help="Directory for temporary query-side artifacts",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Optional benchmark name override",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for the normalized query record",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    record = extract_query_cdfg(
        c_path=args.c_path,
        query_root=args.query_root,
        benchmark=args.benchmark,
    )
    if args.output:
        write_query_record(args.output, record)
    print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
