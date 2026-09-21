"""
Convert rerank results into executable Module 4 transform plans.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from rag_retrieve.blacklist import build_blacklist, watchlist_names
from rag_retrieve.defaults import DEFAULT_KNOWLEDGE_BASE, DEFAULT_TRANSFORM_BLACKLIST


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _region_score(region_result: Dict[str, Any]) -> float:
    top_matches = region_result.get("top_matches", [])
    if not top_matches:
        return 0.0
    return float(top_matches[0].get("score", 0.0))


def _filter_region_recommendations(
    region_result: Dict[str, Any],
    harmful_names: Set[str],
    watchlist_names: Set[str] | None = None,
) -> Dict[str, Any]:
    """Drop banned transforms; keep watchlisted ones but mark them risky.

    A watchlisted transform has a bad *mean* but an acceptable *median* -- it
    usually does nothing and occasionally regresses badly.  Removing it costs
    real wins (ALGEBRAIC_SIMPLIFY is the clearest case); keeping it unmarked
    hides the variance from the planner.  So it stays in the action space with
    ``risk_flag`` set, which Module 4.5 turns into ``expected_risk="high"``.
    """

    watchlist_names = watchlist_names or set()
    accepted = []
    rejected = []
    for rec in region_result.get("recommended_transforms", []):
        name = rec["transform_name"]
        weight = float(rec.get("priority_score", rec.get("aggregated_weight", 0.0)))
        metric = float(rec.get("expected_metric_gain", rec.get("median_gain", 0.0)))
        if name in harmful_names:
            rejected.append({"transform_name": name, "reason": "blacklisted", **rec})
            continue
        if name in watchlist_names:
            rec = {**rec, "risk_flag": "high_variance_regression"}
        if metric <= 0.0:
            rejected.append({"transform_name": name, "reason": "non_positive_expected_gain", **rec})
            continue
        if weight <= 0.0:
            rejected.append({"transform_name": name, "reason": "non_positive_weight", **rec})
            continue
        accepted.append(rec)

    return {
        "accepted": accepted,
        "rejected": rejected,
    }


def _calibrated_confidence(rec: Dict[str, Any], supporting_matches: List[Dict[str, Any]]) -> tuple[float, Dict[str, float]]:
    if not supporting_matches:
        return 0.0, {
            "similarity": 0.0, "benchmark_diversity": 0.0, "sample_strength": 0.0,
            "stability": 0.0, "evidence_quality": 0.0,
        }
    similarity = sum(float(item.get("score", 0.0)) for item in supporting_matches) / len(supporting_matches)
    diversity = min(len({item.get("benchmark") for item in supporting_matches}) / 3.0, 1.0)
    raw_samples = max(float(rec.get("raw_samples", rec.get("samples", 0.0))), 0.0)
    sample_strength = min(math.log1p(raw_samples) / math.log(4.0), 1.0)
    stability = max(0.0, 1.0 - float(rec.get("regression_rate", 0.0)))
    evidence_levels = set(rec.get("evidence_levels", []))
    evidence_quality = 1.0 if "region" in evidence_levels else 0.55
    confidence = (
        0.30 * similarity + 0.20 * diversity + 0.20 * sample_strength
        + 0.20 * stability + 0.10 * evidence_quality
    )
    components = {
        "similarity": round(similarity, 6),
        "benchmark_diversity": round(diversity, 6),
        "sample_strength": round(sample_strength, 6),
        "stability": round(stability, 6),
        "evidence_quality": round(evidence_quality, 6),
    }
    return round(min(max(confidence, 0.0), 1.0), 6), components


def _infer_problem_hypothesis(region_result: Dict[str, Any]) -> str:
    region_type = region_result.get("query_region_type", "")
    features = region_result.get("query_region_features", {})
    flags = []
    if features.get("has_add_chain"):
        flags.append("adder-chain hotspot")
    if features.get("has_priority_select"):
        flags.append("priority-select control hotspot")
    if features.get("has_state_update"):
        flags.append("state-update path")
    if features.get("has_bit_reorg"):
        flags.append("bit-reorganization path")
    if features.get("has_reduction_like"):
        flags.append("reduction-like logic")
    if not flags:
        if region_type == "arith_region":
            flags.append("arithmetic hotspot")
        elif region_type == "select_region":
            flags.append("selection/control hotspot")
        elif region_type == "state_region":
            flags.append("state update hotspot")
        elif region_type == "bit_region":
            flags.append("bit manipulation hotspot")
        elif region_type == "reduction_region":
            flags.append("reduction-tree hotspot")
        elif region_type == "memory_region":
            flags.append("memory-access hotspot")
        else:
            flags.append("mixed structural hotspot")
    return ", ".join(flags)


def _region_execution_hint(region_result: Dict[str, Any], objective: str) -> str:
    region_type = region_result.get("query_region_type", "")
    if objective == "timing":
        if "select" in region_type:
            return "Prefer reducing control depth and simplifying selection logic locally."
        if "state" in region_type:
            return "Prefer shortening state-update dependency paths and simplifying memory-adjacent logic."
        if "arith" in region_type:
            return "Prefer shortening critical arithmetic/dataflow paths before broad rewrites."
        if "reduction" in region_type:
            return "Prefer flattening reduction depth or simplifying converging arithmetic logic."
        return "Prefer local timing-oriented simplification with minimal functional disturbance."
    if "select" in region_type:
        return "Prefer local restructuring that reduces redundant control/select logic."
    if "state" in region_type:
        return "Prefer local simplification around state transitions and redundant load/store logic."
    if "arith" in region_type:
        return "Prefer area-reducing arithmetic simplification and sharing opportunities."
    if "bit" in region_type:
        return "Prefer compact bit-level rewrites and removal of redundant bit manipulation."
    return "Prefer local area-reducing cleanup and avoid global rewrites first."


def _build_recommended_transform_entries(
    region_result: Dict[str, Any],
    accepted_recommendations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    top_matches = region_result.get("top_matches", [])
    enriched = []
    for rec in accepted_recommendations:
        name = rec["transform_name"]
        supporting_matches = []
        for match in top_matches:
            matched_transform = None
            for transform in match.get("transform_evidence", match.get("top_transforms", [])):
                if transform["transform_name"] == name:
                    matched_transform = transform
                    break
            if matched_transform is None:
                continue
            supporting_matches.append(
                {
                    "benchmark": match["benchmark"],
                    "objective": match["objective"],
                    "subcategory": match["subcategory"],
                    "historical_region_id": match["historical_region_id"],
                    "score": match["score"],
                    "historical_metric": matched_transform,
                }
            )

        confidence, confidence_components = _calibrated_confidence(rec, supporting_matches)
        reason = "Supported by diverse, similar historical evidence with positive median objective gain."
        if not supporting_matches:
            reason = "Retained after filtering, but direct per-match evidence is limited."

        enriched.append(
            {
                "transform_name": name,
                "priority_score": rec.get("priority_score", rec.get("aggregated_weight", 0.0)),
                "expected_metric_gain": rec.get("expected_metric_gain", rec.get("median_gain", 0.0)),
                "confidence": confidence,
                "confidence_components": confidence_components,
                "samples": rec.get("samples", 0),
                "raw_samples": rec.get("raw_samples", 0),
                "positive_rate": rec.get("positive_rate", 0.0),
                "regression_rate": rec.get("regression_rate", 0.0),
                "gain_stddev": rec.get("gain_stddev", 0.0),
                "evidence_levels": rec.get("evidence_levels", []),
                "why": reason,
                "supporting_matches": supporting_matches[:3],
                "memory_score": rec.get("memory_score", 0.0),
                "memory_success_rate": rec.get("memory_success_rate", 0.0),
                "memory_evidence_count": rec.get("memory_evidence_count", 0),
                "memory_evidence": rec.get("memory_evidence", []),
                "risk_flag": rec.get("risk_flag", ""),
            }
        )
    return enriched


def build_transform_plan(
    rerank_payload: Dict[str, Any],
    blacklist_payload: Dict[str, Any],
) -> Dict[str, Any]:
    objective = rerank_payload["filters"]["objective"]
    harmful_names = set(blacklist_payload["final"][objective])
    watched_names = watchlist_names(blacklist_payload, objective)

    per_region_candidates = []
    global_scores: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "combined_score": 0.0,
            "weighted_gain_sum": 0.0,
            "gain_weight": 0.0,
            "supporting_regions": [],
            "supporting_benchmarks": set(),
            "samples": 0,
        }
    )

    for graph in rerank_payload.get("graphs", []):
        for region in graph.get("regions", []):
            filtered = _filter_region_recommendations(
                region, harmful_names, watched_names
            )
            accepted_enriched = _build_recommended_transform_entries(region, filtered["accepted"])
            source_anchor = dict(region.get("query_source_anchor", {}) or {})
            if not source_anchor:
                source_anchor = {
                    "function": graph.get("graph_function", ""),
                    "anchor_node_ids": region.get("query_anchor_node_ids", []),
                    "anchor_labels": region.get("query_anchor_labels", []),
                }
            source_anchor.update(
                {
                    "function": source_anchor.get("function") or graph.get("graph_function", ""),
                    "anchor_node_ids": source_anchor.get("anchor_node_ids", region.get("query_anchor_node_ids", [])),
                    "anchor_labels": source_anchor.get("anchor_labels", region.get("query_anchor_labels", [])),
                    "region_node_count": len(region.get("query_node_ids", [])),
                    "region_edge_count": region.get("query_edge_count", 0),
                }
            )
            region_entry = {
                "query_region_id": region["query_region_id"],
                "query_region_type": region["query_region_type"],
                "region_score": round(_region_score(region), 6),
                "region_features": dict(region.get("query_region_features", {})),
                "source_anchor": source_anchor,
                "problem_hypothesis": _infer_problem_hypothesis(region),
                "execution_hint": _region_execution_hint(region, objective),
                "accepted_transforms": filtered["accepted"],
                "recommended_transforms_llm": accepted_enriched,
                "rejected_transforms": filtered["rejected"],
                "top_matches": region.get("top_matches", []),
            }
            per_region_candidates.append(region_entry)

            top_match_benchmarks = [item["benchmark"] for item in region.get("top_matches", [])[:3]]
            for rec in filtered["accepted"]:
                name = rec["transform_name"]
                weight = float(rec.get("aggregated_weight", 0.0))
                metric = float(rec.get("expected_metric_gain", rec.get("median_gain", 0.0)))
                payload = global_scores[name]
                payload["combined_score"] += weight
                metric_weight = max(float(rec.get("raw_samples", rec.get("samples", 0.0))), 1.0)
                payload["weighted_gain_sum"] += metric * metric_weight
                payload["gain_weight"] += metric_weight
                payload["supporting_regions"].append(region["query_region_id"])
                payload["supporting_benchmarks"].update(top_match_benchmarks)
                payload["samples"] += int(rec.get("samples", 0))

    global_plan = sorted(
        (
            {
                "transform_name": name,
                "combined_score": round(payload["combined_score"], 6),
                "expected_metric_gain": round(
                    payload["weighted_gain_sum"] / payload["gain_weight"] if payload["gain_weight"] else 0.0,
                    6,
                ),
                "supporting_regions": sorted(set(payload["supporting_regions"])),
                "supporting_benchmarks": sorted(payload["supporting_benchmarks"]),
                "samples": payload["samples"],
            }
            for name, payload in global_scores.items()
        ),
        key=lambda item: (item["combined_score"], item["expected_metric_gain"], item["samples"]),
        reverse=True,
    )

    llm_regions = []
    for region in per_region_candidates:
        llm_regions.append(
            {
                "region_id": region["query_region_id"],
                "region_type": region["query_region_type"],
                "priority": region["region_score"],
                "problem_hypothesis": region["problem_hypothesis"],
                "source_anchor": region["source_anchor"],
                "execution_hint": region["execution_hint"],
                "recommended_transforms": region["recommended_transforms_llm"],
                "rejected_transforms": region["rejected_transforms"],
                "supporting_matches": region["top_matches"][:3],
                "region_features": region["region_features"],
            }
        )

    llm_ready_payload = {
        "benchmark": rerank_payload["benchmark"],
        "objective": objective,
        "source_c_path": rerank_payload["source_c_path"],
        "must_preserve_behavior": True,
        "harmful_blacklist": sorted(harmful_names),
        "risk_watchlist": sorted(watched_names),
        "global_plan": global_plan[:10],
        "regions": llm_regions,
        "llm_execution_hint": {
            "edit_scope": "Prefer local edits around recommended regions before global rewrites.",
            "selection_rule": f"Prioritize transforms with positive {objective} evidence and reject any blacklisted transform.",
            "avoid": [
                "global rewrite without local evidence",
                "any transform present in harmful_blacklist",
            ],
        },
    }

    return {
        "benchmark": rerank_payload["benchmark"],
        "source_c_path": rerank_payload["source_c_path"],
        "objective": objective,
        "filters": rerank_payload["filters"],
        "harmful_blacklist": sorted(harmful_names),
        "risk_watchlist": sorted(watched_names),
        "per_region_candidates": per_region_candidates,
        "global_plan": global_plan,
        "llm_ready_payload": llm_ready_payload,
    }


def write_plan(output_path: str | Path, payload: Dict[str, Any]) -> None:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Module 4 transform plan from rerank results.")
    parser.add_argument("rerank_json", help="Path to rerank JSON")
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_KNOWLEDGE_BASE),
        help="Active RAG CSV; used only when --blacklist-json is explicitly empty",
    )
    parser.add_argument(
        "--blacklist-json",
        default=str(DEFAULT_TRANSFORM_BLACKLIST),
        help="Prebuilt blacklist JSON matched to the active RAG release",
    )
    parser.add_argument("--output", required=True, help="Output plan JSON path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rerank_payload = _load_json(args.rerank_json)
    blacklist_payload = _load_json(args.blacklist_json) if args.blacklist_json else build_blacklist(args.csv_path)
    plan = build_transform_plan(rerank_payload, blacklist_payload)
    write_plan(args.output, plan)
    print(
        json.dumps(
            {
                "benchmark": plan["benchmark"],
                "objective": plan["objective"],
                "harmful_blacklist_count": len(plan["harmful_blacklist"]),
                "global_plan_count": len(plan["global_plan"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
