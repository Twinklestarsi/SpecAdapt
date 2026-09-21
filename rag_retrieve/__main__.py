"""
CLI for building the initial Module 4 CDFG/RAG joined index.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_retrieve.cdfg_index import (
    build_default_joined_index,
    build_joined_index,
    write_joined_index,
)
from rag_retrieve.defaults import get_active_rag_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a joined index from CDFG corpora and the active versioned RAG."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to the repository root inferred from this package.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Knowledge-base CSV override. Defaults to the release selected by latest_rag.json.",
    )
    parser.add_argument(
        "--cdfg-area-root",
        type=Path,
        default=None,
        help="Path to CDFG_AREA. Defaults to <project-root>/CDFG_AREA.",
    )
    parser.add_argument(
        "--cdfg-timing-root",
        type=Path,
        default=None,
        help="Path to CDFG_TIMING. Defaults to <project-root>/CDFG_TIMING.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. If omitted, only the summary is printed.",
    )
    return parser.parse_args()


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        from rag_retrieve.build_rag import main as build_main

        del sys.argv[1]
        return build_main()
    args = _parse_args()
    if args.csv_path or args.cdfg_area_root or args.cdfg_timing_root:
        project_root = (args.project_root or Path(__file__).resolve().parent.parent).resolve()
        active_rag = get_active_rag_artifacts(project_root)
        payload = build_joined_index(
            csv_path=(args.csv_path or active_rag.knowledge_base).resolve(),
            cdfg_roots={
                "area": (args.cdfg_area_root or project_root / "CDFG_AREA").resolve(),
                "timing": (args.cdfg_timing_root or project_root / "CDFG_TIMING").resolve(),
            },
        )
    else:
        payload = build_default_joined_index(project_root=args.project_root)

    if args.output:
        write_joined_index(args.output.resolve(), payload)

    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
