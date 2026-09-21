"""
Baseline CDFG matcher for Module 4.

This is a deterministic first-pass matcher:
- no LLM
- no MCTS
- explainable scoring from graph size + opcode distribution
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from rag_retrieve.transform_stats import summarize_transform_rows
from rag_retrieve.defaults import DEFAULT_JOINED_INDEX


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _histogram_cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    dot = sum(a.get(key, 0) * b.get(key, 0) for key in keys)
    norm_a = math.sqrt(sum(a.get(key, 0) ** 2 for key in keys))
    norm_b = math.sqrt(sum(b.get(key, 0) ** 2 for key in keys))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _relative_similarity(a: int, b: int) -> float:
    scale = max(abs(a), abs(b), 1)
    return max(0.0, 1.0 - abs(a - b) / scale)


def _score_graph_pair(query_graph: Dict[str, Any], candidate_graph: Dict[str, Any]) -> Dict[str, float]:
    opcode_score = _histogram_cosine(
        query_graph.get("opcode_histogram", {}),
        candidate_graph.get("opcode_histogram", {}),
    )
    node_score = _relative_similarity(query_graph.get("node_count", 0), candidate_graph.get("node_count", 0))
    edge_score = _relative_similarity(query_graph.get("edge_count", 0), candidate_graph.get("edge_count", 0))
    data_edge_score = _relative_similarity(
        query_graph.get("data_edge_count", 0),
        candidate_graph.get("data_edge_count", 0),
    )
    control_edge_score = _relative_similarity(
        query_graph.get("control_edge_count", 0),
        candidate_graph.get("control_edge_count", 0),
    )

    total_score = (
        0.55 * opcode_score
        + 0.15 * node_score
        + 0.15 * edge_score
        + 0.075 * data_edge_score
        + 0.075 * control_edge_score
    )
    return {
        "total_score": total_score,
        "opcode_score": opcode_score,
        "node_score": node_score,
        "edge_score": edge_score,
        "data_edge_score": data_edge_score,
        "control_edge_score": control_edge_score,
    }


def _select_query_graph(query_record: Dict[str, Any]) -> Dict[str, Any]:
    graphs = query_record.get("graphs", [])
    if not graphs:
        raise ValueError("Query record has no graphs")
    return max(graphs, key=lambda item: (item.get("node_count", 0), item.get("edge_count", 0)))


def _top_transforms(rag_rows: Iterable[Dict[str, Any]], objective: str, limit: int = 3) -> List[Dict[str, Any]]:
    return summarize_transform_rows(rag_rows, objective)[:limit]


def match_query_to_index(
    query_record: Dict[str, Any],
    joined_index: Dict[str, Any],
    top_k: int = 5,
    objective: str | None = None,
    subcategory: str | None = None,
    backend: str | None = None,
    clock_period_ps: float | None = None,
    min_similarity: float = 0.55,
) -> Dict[str, Any]:
    if objective not in {"area", "timing"}:
        raise ValueError("objective is required and must be 'area' or 'timing'")
    query_graph = _select_query_graph(query_record)
    candidates: List[Dict[str, Any]] = []

    for entry in joined_index.get("entries", []):
        cdfg_record = entry.get("cdfg_record", {})
        if objective and cdfg_record.get("objective") != objective:
            continue
        if subcategory and cdfg_record.get("subcategory") != subcategory:
            continue
        filtered_rows = [
            row for row in entry.get("rag_rows", [])
            if (not backend or row.get("backend") == backend)
            and (
                clock_period_ps is None
                or abs(float(row.get("clock_period_ps") or 0.0) - float(clock_period_ps)) < 1e-6
            )
        ]
        if not filtered_rows:
            continue

        graphs = cdfg_record.get("graphs", [])
        if not graphs:
            continue

        best_graph = None
        best_score = None
        for candidate_graph in graphs:
            score = _score_graph_pair(query_graph, candidate_graph)
            if best_score is None or score["total_score"] > best_score["total_score"]:
                best_score = score
                best_graph = candidate_graph

        if best_score is None or best_graph is None:
            continue
        if best_score["total_score"] < min_similarity:
            continue

        candidates.append(
            {
                "objective": cdfg_record.get("objective"),
                "subcategory": cdfg_record.get("subcategory"),
                "benchmark": cdfg_record.get("benchmark"),
                "benchmark_dir": cdfg_record.get("benchmark_dir"),
                "matched_graph_function": best_graph.get("function_name"),
                "matched_graph_path": best_graph.get("dot_path"),
                "score": round(best_score["total_score"], 6),
                "score_breakdown": {key: round(value, 6) for key, value in best_score.items()},
                "graph_stats": {
                    "query_node_count": query_graph.get("node_count", 0),
                    "candidate_node_count": best_graph.get("node_count", 0),
                    "query_edge_count": query_graph.get("edge_count", 0),
                    "candidate_edge_count": best_graph.get("edge_count", 0),
                },
                "top_transforms": _top_transforms(
                    filtered_rows,
                    objective=cdfg_record.get("objective", "area"),
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            item["score"],
            len(item.get("top_transforms", [])),
        ),
        reverse=True,
    )
    diverse = []
    seen_benchmarks = set()
    for candidate in candidates:
        if candidate["benchmark"] in seen_benchmarks:
            continue
        seen_benchmarks.add(candidate["benchmark"])
        diverse.append(candidate)
        if len(diverse) >= top_k:
            break
    return {
        "query_benchmark": query_record.get("benchmark"),
        "query_source_c_path": query_record.get("source_c_path"),
        "query_graph_function": query_graph.get("function_name"),
        "query_node_count": query_graph.get("node_count", 0),
        "query_edge_count": query_graph.get("edge_count", 0),
        "filters": {
            "objective": objective,
            "subcategory": subcategory,
            "top_k": top_k,
            "backend": backend,
            "clock_period_ps": clock_period_ps,
            "min_similarity": min_similarity,
        },
        "retrieval_status": "ok" if diverse else "no_reliable_match",
        "matches": diverse,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match a query CDFG against the historical Module 4 index.")
    parser.add_argument("query_json", help="Path to a query CDFG JSON file")
    parser.add_argument(
        "--index-json",
        default=str(DEFAULT_JOINED_INDEX),
        help="Joined historical CDFG/RAG index (default: active RAG release)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to return")
    parser.add_argument("--objective", required=True, choices=["area", "timing"], help="Required objective filter")
    parser.add_argument("--subcategory", default=None, help="Optional subcategory filter")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--clock-period-ps", type=float, default=None)
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = match_query_to_index(
        query_record=_load_json(args.query_json),
        joined_index=_load_json(args.index_json),
        top_k=args.top_k,
        objective=args.objective,
        subcategory=args.subcategory,
        backend=args.backend,
        clock_period_ps=args.clock_period_ps,
        min_similarity=args.min_similarity,
    )
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
