"""Region reranking with strict context filters and calibrated abstention."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from rag_retrieve.transform_stats import as_float, weighted_median
from rag_retrieve.defaults import DEFAULT_HISTORICAL_REGION_INDEX


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _histogram_cosine(a: Dict[Any, int], b: Dict[Any, int]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    dot = sum(float(a.get(key, 0)) * float(b.get(key, 0)) for key in keys)
    norm_a = math.sqrt(sum(float(a.get(key, 0)) ** 2 for key in keys))
    norm_b = math.sqrt(sum(float(b.get(key, 0)) ** 2 for key in keys))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _relative_similarity(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1.0)
    return max(0.0, 1.0 - abs(a - b) / scale)


def _type_score(query_type: str, candidate_type: str) -> float:
    if query_type == candidate_type:
        return 1.0
    pairs = {query_type, candidate_type}
    if any("hybrid" in item or "mixed" in item for item in pairs):
        return 0.55
    if all("arith" in item or "reduction" in item for item in pairs):
        return 0.75
    if any("state" in item for item in pairs) and any("memory" in item for item in pairs):
        return 0.72
    if any("bit" in item for item in pairs) and any("reduction" in item for item in pairs):
        return 0.58
    return 0.2


def _motif_set(features: Dict[str, Any]) -> Set[str]:
    return {
        key for key in (
            "has_add_chain", "has_const_mult_like", "has_priority_select",
            "has_reduction_like", "has_state_update", "has_bit_reorg",
            "has_loop_like", "has_array_access",
        ) if features.get(key)
    }


def _jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _average_relative(query: Dict[str, Any], candidate: Dict[str, Any], keys: Iterable[str]) -> float:
    keys = tuple(keys)
    return sum(
        _relative_similarity(float(query.get(key, 0.0)), float(candidate.get(key, 0.0)))
        for key in keys
    ) / len(keys)


def _score_region_pair(query_region: Dict[str, Any], candidate_entry: Dict[str, Any]) -> Dict[str, float]:
    query = query_region.get("features", {})
    candidate_region = candidate_entry.get("region", {})
    candidate = candidate_region.get("features", {})
    type_score = _type_score(query_region.get("region_type", ""), candidate_region.get("region_type", ""))
    opcode_score = _histogram_cosine(query.get("opcode_histogram", {}), candidate.get("opcode_histogram", {}))
    structural_score = _average_relative(query, candidate, (
        "node_count", "edge_count", "graph_depth", "branch_count", "phi_count",
        "load_count", "store_count", "constant_count", "expensive_operator_count",
    ))
    topology_score = _average_relative(query, candidate, (
        "data_edge_count", "control_edge_count", "memory_edge_count", "max_fanin",
        "max_fanout", "avg_fanin", "avg_fanout", "dependency_density", "depth_per_node",
    ))
    motif_score = _jaccard(_motif_set(query), _motif_set(candidate))
    bitwidth_score = _histogram_cosine(query.get("bitwidth_histogram", {}), candidate.get("bitwidth_histogram", {}))
    constant_score = _jaccard(set(query.get("unique_constants", [])), set(candidate.get("unique_constants", [])))
    total = (
        0.20 * type_score + 0.25 * opcode_score + 0.18 * structural_score
        + 0.14 * topology_score + 0.08 * motif_score
        + 0.10 * bitwidth_score + 0.05 * constant_score
    )
    return {
        "total_score": total,
        "type_score": type_score,
        "opcode_score": opcode_score,
        "structural_score": structural_score,
        "topology_score": topology_score,
        "motif_score": motif_score,
        "bitwidth_score": bitwidth_score,
        "constant_score": constant_score,
    }


def _score_region_group_pair(
    query_region: Dict[str, Any],
    group_entry: Dict[str, Any],
    atomic_entry_lookup: Dict[tuple[str, str, str, str, str], Dict[str, Any]],
) -> tuple[Dict[str, float], Dict[str, Any]] | None:
    """Score a composite group by its most similar member, without duplicating evidence."""

    members = (group_entry.get("group") or {}).get("member_regions", [])
    best: tuple[Dict[str, float], Dict[str, Any]] | None = None
    for member in members:
        lookup_key = (
            str(group_entry.get("objective") or ""),
            str(group_entry.get("subcategory") or ""),
            str(group_entry.get("benchmark") or ""),
            str(member.get("graph_function") or ""),
            str(member.get("region_id") or ""),
        )
        atomic_entry = atomic_entry_lookup.get(lookup_key)
        if atomic_entry is None:
            continue
        score = _score_region_pair(query_region, atomic_entry)
        if best is None or score["total_score"] > best[0]["total_score"]:
            best = (score, member)
    return best


def _local_evidence_level(evidence: List[Dict[str, Any]]) -> str:
    levels = {
        str(level)
        for summary in evidence
        for level in (summary.get("evidence_levels") or [])
    }
    if "region" in levels:
        return "region"
    if "region_low_confidence" in levels:
        return "region_low_confidence"
    return "region"


def _evidence_factor(evidence_level: str) -> float:
    return {
        "region": 1.0,
        "region_group": 0.60,
        "region_low_confidence": 0.40,
        "benchmark": 0.25,
    }.get(str(evidence_level), 0.25)


def _transform_applicable(name: str, features: Dict[str, Any]) -> tuple[bool, str]:
    upper = name.upper()
    op_hist = features.get("opcode_histogram", {}) or {}
    if "ENCODE_ONEHOT_TO_BINARY" in upper or "FSM_REENCODE" in upper:
        region_hint = str(
            features.get("proposal_type")
            or features.get("region_type_guess")
            or features.get("region_type")
            or ""
        ).lower()
        comparison_count = int(op_hist.get("icmp", 0)) + int(op_hist.get("fcmp", 0))
        control_evidence = (
            int(features.get("branch_count", 0) or 0) > 0
            or bool(features.get("has_priority_select"))
        )
        region_evidence = any(
            token in region_hint for token in ("select", "state", "control", "fsm")
        )
        if comparison_count < 1 or not control_evidence or not region_evidence:
            return False, "requires_encoded_state_or_control_comparisons"
        if int(features.get("constant_count", 0) or 0) < 2:
            return False, "requires_encoding_constants"
    if "SERIALIZE_PARALLELISM" in upper:
        active_ops = sum(
            int(op_hist.get(op, 0) or 0)
            for op in (
                "add", "sub", "mul", "sdiv", "udiv", "and", "or", "xor",
                "shl", "lshr", "ashr", "icmp", "select", "phi", "br",
            )
        )
        if int(features.get("node_count", 0) or 0) < 3 or active_ops < 1:
            return False, "requires_nontrivial_parallelizable_structure"
    if "LOOP" in upper or "UNROLL" in upper:
        if not features.get("has_loop_like"):
            return False, "requires_loop"
    if any(token in upper for token in ("ARRAY", "MEMORY", "BUFFER")):
        if not features.get("has_array_access") and not features.get("load_count") and not features.get("store_count"):
            return False, "requires_memory_or_array_access"
    if "CONST" in upper and not features.get("constant_count"):
        return False, "requires_constants"
    if any(token in upper for token in ("BITWIDTH", "TYPE_NARROW")) and not features.get("max_bitwidth"):
        return False, "requires_typed_values"
    if any(token in upper for token in ("RESOURCE_SHARE", "COMMON_SUBEXPR", "TIME_MULTIPLEX")):
        reusable = sum(int(op_hist.get(op, 0)) for op in ("add", "sub", "mul", "sdiv", "udiv"))
        if reusable < 2:
            return False, "requires_repeated_operators"
    return True, ""


def _aggregate_transforms(
    matches: Iterable[Dict[str, Any]],
    objective: str,
    query_features: Dict[str, Any],
    min_effective_samples: int = 1,
    limit: int = 8,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    totals: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "gains": [], "weights": [], "samples": 0.0, "raw_samples": 0,
        "positive": 0.0, "negative": 0.0, "applicability": [], "stddev": [],
        "benchmarks": set(), "evidence_levels": set(),
    })
    rejected: Dict[str, str] = {}
    for match in matches:
        similarity = float(match.get("score", 0.0))
        evidence_level = match.get("evidence_level", "benchmark")
        evidence_factor = _evidence_factor(str(evidence_level))
        for evidence in match.get("transform_evidence", []):
            name = str(evidence.get("transform_name") or "")
            applicable, reason = _transform_applicable(name, query_features)
            if not applicable:
                rejected[name] = reason
                continue
            raw_samples = int(evidence.get("raw_effective_sample_count", 0))
            effective_samples = float(evidence.get("effective_sample_count", 0.0))
            if raw_samples < min_effective_samples:
                rejected[name] = "insufficient_samples"
                continue
            gain = float(evidence.get("median_gain", 0.0))
            support_weight = similarity * evidence_factor * max(math.log1p(effective_samples), 0.5)
            bucket = totals[name]
            bucket["gains"].append(gain)
            bucket["weights"].append(support_weight)
            bucket["samples"] += effective_samples
            bucket["raw_samples"] += raw_samples
            bucket["positive"] += support_weight * float(evidence.get("positive_rate", 0.0))
            bucket["negative"] += support_weight * float(evidence.get("regression_rate", 0.0))
            bucket["applicability"].append(float(evidence.get("applicability_rate", 0.0)))
            bucket["stddev"].append(float(evidence.get("gain_stddev", 0.0)))
            bucket["benchmarks"].add(str(match.get("benchmark") or ""))
            bucket["evidence_levels"].add(evidence_level)

    ranked: List[Dict[str, Any]] = []
    gain_scale = 10.0 if objective == "area" else 500.0
    for name, payload in totals.items():
        total_weight = sum(payload["weights"])
        expected_gain = weighted_median(payload["gains"], payload["weights"])
        mean_gain = (
            sum(value * weight for value, weight in zip(payload["gains"], payload["weights"])) / total_weight
            if total_weight else 0.0
        )
        positive_rate = payload["positive"] / total_weight if total_weight else 0.0
        regression_rate = payload["negative"] / total_weight if total_weight else 0.0
        signal = positive_rate - regression_rate
        priority = total_weight * (0.65 * signal + 0.35 * math.tanh(expected_gain / gain_scale))
        ranked.append({
            "transform_name": name,
            "priority_score": round(priority, 6),
            "aggregated_weight": round(priority, 6),
            "expected_metric_gain": round(expected_gain, 6),
            "median_gain": round(expected_gain, 6),
            "mean_gain": round(mean_gain, 6),
            "positive_rate": round(positive_rate, 6),
            "regression_rate": round(regression_rate, 6),
            "gain_stddev": round(sum(payload["stddev"]) / len(payload["stddev"]), 6) if payload["stddev"] else 0.0,
            "applicability_rate": round(sum(payload["applicability"]) / len(payload["applicability"]), 6) if payload["applicability"] else 0.0,
            "samples": round(payload["samples"], 6),
            "raw_samples": payload["raw_samples"],
            "independent_benchmarks": len(payload["benchmarks"]),
            "supporting_benchmarks": sorted(payload["benchmarks"]),
            "evidence_levels": sorted(payload["evidence_levels"]),
        })
    ranked.sort(key=lambda item: (item["priority_score"], item["expected_metric_gain"], item["raw_samples"]), reverse=True)
    rejected_rows = [{"transform_name": name, "reason": reason} for name, reason in sorted(rejected.items()) if name not in totals]
    return ranked[:limit], rejected_rows


def _select_diverse(candidates: List[Dict[str, Any]], limit: int, max_per_benchmark: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)
    for candidate in candidates:
        benchmark = str(candidate.get("benchmark") or "")
        if counts[benchmark] >= max_per_benchmark:
            continue
        selected.append(candidate)
        counts[benchmark] += 1
        if len(selected) >= limit:
            break
    return selected


def rerank_query_regions(
    query_regions_payload: Dict[str, Any],
    historical_region_index: Dict[str, Any],
    objective: str,
    top_k_regions: int = 5,
    subcategory: str | None = None,
    backend: str | None = None,
    clock_period_ps: float | None = None,
    min_similarity: float = 0.55,
    min_effective_samples: int = 1,
    max_per_benchmark: int = 1,
) -> Dict[str, Any]:
    if objective not in {"area", "timing"}:
        raise ValueError(f"objective must be 'area' or 'timing', got: {objective!r}")
    results = {
        "benchmark": query_regions_payload.get("benchmark"),
        "source_c_path": query_regions_payload.get("source_c_path"),
        "filters": {
            "objective": objective, "subcategory": subcategory, "backend": backend,
            "clock_period_ps": clock_period_ps, "top_k_regions": top_k_regions,
            "min_similarity": min_similarity, "min_effective_samples": min_effective_samples,
            "max_per_benchmark": max_per_benchmark, "strict_objective": True,
            "strict_backend": bool(backend), "strict_constraint": clock_period_ps is not None,
        },
        "graphs": [],
    }
    historical_entries = historical_region_index.get("entries", [])
    historical_groups = historical_region_index.get("region_groups", [])
    benchmark_priors = historical_region_index.get("benchmark_priors", {})
    atomic_entry_lookup = {
        (
            str(entry.get("objective") or ""),
            str(entry.get("subcategory") or ""),
            str(entry.get("benchmark") or ""),
            str(entry.get("graph_function") or ""),
            str((entry.get("region") or {}).get("region_id") or ""),
        ): entry
        for entry in historical_entries
    }

    for graph in query_regions_payload.get("graphs", []):
        graph_result = {"graph_function": graph.get("graph_function"), "dot_path": graph.get("dot_path"), "regions": []}
        for query_region in graph.get("regions", []):
            candidates: List[Dict[str, Any]] = []
            for entry in historical_entries:
                if entry.get("objective") != objective:
                    continue
                if subcategory and entry.get("subcategory") != subcategory:
                    continue
                if backend and entry.get("backends") and backend not in entry.get("backends", []):
                    continue
                if clock_period_ps is not None and entry.get("clock_periods_ps"):
                    if not any(abs(float(value) - float(clock_period_ps)) < 1e-6 for value in entry.get("clock_periods_ps", [])):
                        continue
                score = _score_region_pair(query_region, entry)
                if score["total_score"] < min_similarity:
                    continue
                local = entry.get("transform_evidence", [])
                evidence_level = _local_evidence_level(local) if local else "benchmark"
                evidence = local or benchmark_priors.get(entry.get("benchmark_prior_key", ""), [])
                candidate_region = entry.get("region", {})
                candidates.append({
                    "objective": entry.get("objective"), "subcategory": entry.get("subcategory"),
                    "benchmark": entry.get("benchmark"), "benchmark_dir": entry.get("benchmark_dir"),
                    "graph_function": entry.get("graph_function"), "graph_dot_path": entry.get("graph_dot_path"),
                    "historical_region_id": candidate_region.get("region_id"),
                    "historical_region_type": candidate_region.get("region_type"),
                    "score": round(score["total_score"], 6),
                    "score_breakdown": {key: round(value, 6) for key, value in score.items()},
                    "evidence_level": evidence_level,
                    "transform_evidence": evidence,
                    "top_transforms": evidence,
                })
            for entry in historical_groups:
                if entry.get("objective") != objective:
                    continue
                if subcategory and entry.get("subcategory") != subcategory:
                    continue
                if backend and entry.get("backends") and backend not in entry.get("backends", []):
                    continue
                if clock_period_ps is not None and entry.get("clock_periods_ps"):
                    if not any(abs(float(value) - float(clock_period_ps)) < 1e-6 for value in entry.get("clock_periods_ps", [])):
                        continue
                group_score = _score_region_group_pair(
                    query_region, entry, atomic_entry_lookup
                )
                if group_score is None:
                    continue
                score, best_member = group_score
                if score["total_score"] < min_similarity:
                    continue
                group = entry.get("group", {})
                evidence = entry.get("transform_evidence", [])
                if not evidence:
                    continue
                candidates.append({
                    "objective": entry.get("objective"), "subcategory": entry.get("subcategory"),
                    "benchmark": entry.get("benchmark"), "benchmark_dir": entry.get("benchmark_dir"),
                    "graph_function": ",".join(group.get("graph_functions", [])),
                    "graph_dot_path": "",
                    "historical_region_id": group.get("group_id"),
                    "historical_region_type": group.get("group_type", "composite_region"),
                    "historical_region_kind": "composite_region",
                    "member_region_ids": group.get("member_region_ids", []),
                    "member_region_count": group.get("member_region_count", 0),
                    "best_matching_member_region_id": best_member.get("region_id"),
                    "best_matching_member_region_type": best_member.get("region_type"),
                    "score": round(score["total_score"], 6),
                    "score_breakdown": {key: round(value, 6) for key, value in score.items()},
                    "evidence_level": "region_group",
                    "transform_evidence": evidence,
                    "top_transforms": evidence,
                })
            # Structural similarity remains primary.  Evidence quality breaks
            # exact ties so a benchmark prior cannot hide a composite group
            # whose best member is the same atomic region.
            candidates.sort(
                key=lambda item: (
                    float(item["score"]),
                    _evidence_factor(str(item["evidence_level"])),
                ),
                reverse=True,
            )
            top_matches = _select_diverse(candidates, top_k_regions, max_per_benchmark)
            recommended, applicability_rejections = _aggregate_transforms(
                top_matches, objective, query_region.get("features", {}), min_effective_samples
            )
            status = "ok" if top_matches and recommended else "no_reliable_match"
            graph_result["regions"].append({
                "query_region_id": query_region.get("region_id"),
                "query_region_type": query_region.get("region_type"),
                "query_anchor_node_ids": query_region.get("anchor_node_ids", []),
                "query_anchor_labels": query_region.get("anchor_labels", []),
                "query_source_anchor": query_region.get("source_anchor", {}),
                "query_node_ids": query_region.get("node_ids", []),
                "query_edge_count": query_region.get("edge_count", 0),
                "query_region_features": query_region.get("features", {}),
                "retrieval_status": status,
                "abstention_reason": "" if status == "ok" else "no sufficiently similar, applicable evidence",
                "top_matches": top_matches,
                "recommended_transforms": recommended,
                "rejected_transforms": applicability_rejections,
            })
        results["graphs"].append(graph_result)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank query regions against the v3 historical index.")
    parser.add_argument("query_regions_json")
    parser.add_argument("--region-index", default=str(DEFAULT_HISTORICAL_REGION_INDEX))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--objective", required=True, choices=["area", "timing"])
    parser.add_argument("--subcategory", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--clock-period-ps", type=float, default=None)
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--min-effective-samples", type=int, default=1)
    parser.add_argument("--max-per-benchmark", type=int, default=1)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = rerank_query_regions(
        query_regions_payload=_load_json(args.query_regions_json),
        historical_region_index=_load_json(args.region_index), objective=args.objective,
        top_k_regions=args.top_k, subcategory=args.subcategory, backend=args.backend,
        clock_period_ps=args.clock_period_ps, min_similarity=args.min_similarity,
        min_effective_samples=args.min_effective_samples, max_per_benchmark=args.max_per_benchmark,
    )
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
