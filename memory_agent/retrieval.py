"""Structured retrieval and recommendation over Memory Agent records."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from memory_agent.policy_refiner import policy_matches
from memory_agent.sqlite_store import SQLiteMemoryStore
from memory_agent.validators import normalize_objective, validate_feature_profile


FEATURE_WEIGHTS = {
    "architecture_pattern": 0.20,
    "suggested_subcategory": 0.15,
    "is_sequential": 0.15,
    "has_fsm": 0.10,
    "num_clock_domains": 0.15,
    "complexity": 0.10,
    "data_widths": 0.05,
    "optimization_target": 0.10,
}


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return {str(item).lower() for item in value}


def _numeric_similarity(left: Any, right: Any) -> float:
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / (1.0 + abs(a - b))


def _width_similarity(left: Any, right: Any) -> float:
    a, b = _as_set(left), _as_set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def feature_similarity(
    query: Dict[str, Any],
    candidate: Dict[str, Any],
    objective: str = "",
    candidate_objective: str = "",
) -> Dict[str, Any]:
    query = validate_feature_profile(query)
    candidate = validate_feature_profile(candidate)
    breakdown: Dict[str, float] = {}
    query_weight = 0.0
    available_weight = 0.0
    weighted_score = 0.0

    for field, weight in FEATURE_WEIGHTS.items():
        if field == "optimization_target":
            left, right = objective, candidate_objective
        else:
            left, right = query.get(field), candidate.get(field)
        if left in (None, "", []):
            continue
        query_weight += weight
        if right in (None, "", []):
            breakdown[field] = 0.0
            continue
        available_weight += weight
        if field == "num_clock_domains":
            score = _numeric_similarity(left, right)
        elif field == "data_widths":
            score = _width_similarity(left, right)
        else:
            score = float(str(left).lower() == str(right).lower())
        breakdown[field] = round(score, 6)
        weighted_score += weight * score

    score = weighted_score / query_weight if query_weight else 0.0
    return {
        "score": round(score, 6),
        "breakdown": breakdown,
        "compared_weight": round(available_weight, 6),
        "query_weight": round(query_weight, 6),
    }


def _tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    return set(re.findall(r"[a-z0-9_]+", text))


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class MemoryRetriever:
    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store = store

    def retrieve_similar_by_latent(
        self,
        query_vector: Any,
        *,
        protocol_sha256: str,
        objective: str = "",
        model_alias: str = "",
        output_dim: int = 0,
        top_k: int = 5,
        min_similarity: float = 0.80,
        exclude_spec_sha256: str = "",
    ) -> Dict[str, Any]:
        """Cosine retrieval over stored spec latents (revise plan C5).

        The structured counterpart is :meth:`retrieve_similar_tasks`, which can
        only compare the features Module 1 named. This one compares the spec
        embeddings, so two specs describing the same behaviour in different
        wording still rank as neighbours.

        ``memory_agent.latent_retrieval`` is imported lazily because it needs
        numpy: the rest of the Memory Agent -- and therefore the whole pipeline
        that imports it -- must keep working in an environment without the ML
        stack installed.
        """
        from memory_agent import latent_retrieval

        neighbors = latent_retrieval.find_similar_specs(
            self.store,
            query_vector,
            protocol_sha256=protocol_sha256,
            model_alias=model_alias,
            output_dim=output_dim,
            top_k=top_k,
            min_similarity=min_similarity,
            exclude_spec_sha256=exclude_spec_sha256,
        )
        evidence = latent_retrieval.c_first_win_rate(
            self.store,
            neighbors,
            objective=normalize_objective(objective) or "AREA",
        )
        return {
            "schema_version": latent_retrieval.LATENT_RETRIEVAL_VERSION,
            "protocol_sha256": protocol_sha256,
            "neighbor_count": len(neighbors),
            **evidence.to_dict(),
        }

    def retrieve_similar_tasks(
        self,
        features: Dict[str, Any],
        objective: str = "",
        top_k: int = 5,
        exclude_task_id: str = "",
    ) -> List[Dict[str, Any]]:
        objective = normalize_objective(objective)
        candidates: Dict[str, Dict[str, Any]] = {}
        for row in self.store.feature_rows():
            if exclude_task_id and row["task_id"] == exclude_task_id:
                continue
            candidate_objective = (
                row.get("optimization_target")
                or row.get("run_objective")
                or ""
            )
            similarity = feature_similarity(
                features, row["features"], objective, candidate_objective
            )
            item = {
                "task_id": row["task_id"],
                "run_id": row.get("run_id", ""),
                "benchmark": row["benchmark"],
                "path": row.get("run_path", ""),
                "objective": candidate_objective,
                "features": row["features"],
                "similarity": similarity,
                "source_path": row.get("source_path", ""),
            }
            current = candidates.get(row["task_id"])
            if current is None or similarity["score"] > current["similarity"]["score"]:
                candidates[row["task_id"]] = item

        ranked = sorted(
            candidates.values(),
            key=lambda item: item["similarity"]["score"],
            reverse=True,
        )[:max(0, top_k)]

        for item in ranked:
            item["evidence"] = self.store.rows(
                """
                SELECT e.objective, e.path, e.correctness_status, e.synthesis_status,
                       e.area_gain_pct, e.timing_gain_ps, e.slack_status,
                       e.token_usage_json, e.failure_reason
                FROM evaluations e
                WHERE e.task_id = ?
                ORDER BY e.timestamp DESC
                LIMIT 10
                """,
                (item["task_id"],),
            )
            for evidence in item["evidence"]:
                evidence["token_usage"] = _safe_json(
                    evidence.pop("token_usage_json", "{}")
                )
        return ranked

    def recommend_path(
        self,
        features: Dict[str, Any],
        objective: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        objective = normalize_objective(objective)
        if int(features.get("num_clock_domains") or 0) >= 2:
            return {
                "recommended_path": "rtl_direct",
                "objective": objective,
                "path_statistics": {},
                "similar_tasks": [],
                "has_sufficient_evidence": True,
                "decision_source": "hard_constraint",
                "reason": "Multi-clock designs must preserve clock-domain semantics.",
            }
        similar = self.retrieve_similar_tasks(features, objective, top_k=top_k)
        path_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"evidence_count": 0, "successes": 0, "weighted_gain": 0.0,
                     "weight": 0.0, "token_total": 0.0}
        )
        for task in similar:
            similarity = task["similarity"]["score"]
            for evidence in task["evidence"]:
                path = evidence.get("path") or task.get("path")
                if path not in {"c_first", "rtl_direct"}:
                    continue
                if evidence.get("objective") not in {"", objective}:
                    continue
                stat = path_stats[path]
                stat["evidence_count"] += 1
                correct = evidence.get("correctness_status") not in {
                    "failed", "error", "mismatch",
                }
                synthesized = evidence.get("synthesis_status") not in {
                    "failed", "error",
                }
                if correct and synthesized:
                    stat["successes"] += 1
                gain = (
                    evidence.get("area_gain_pct")
                    if objective == "AREA"
                    else evidence.get("timing_gain_ps")
                )
                if correct and synthesized and gain is not None:
                    stat["weighted_gain"] += similarity * float(gain)
                    stat["weight"] += similarity

        normalized: Dict[str, Dict[str, Any]] = {}
        for path, stat in path_stats.items():
            count = int(stat["evidence_count"])
            normalized[path] = {
                "evidence_count": count,
                "success_rate": stat["successes"] / count if count else 0.0,
                "mean_weighted_gain": (
                    stat["weighted_gain"] / stat["weight"] if stat["weight"] else None
                ),
            }

        history_decision = ""
        if normalized:
            history_decision = max(
                normalized,
                key=lambda path: (
                    normalized[path]["success_rate"],
                    normalized[path]["mean_weighted_gain"]
                    if normalized[path]["mean_weighted_gain"] is not None
                    else -math.inf,
                    normalized[path]["evidence_count"],
                ),
            )
        matched_policies = []
        policy_totals = {"c_first": 0.0, "rtl_direct": 0.0}
        policy_weight = 0.0
        for policy in self.store.path_policy_rows(objective):
            if not policy_matches(policy["policy_id"], features):
                continue
            confidence = float(policy.get("confidence", 0.0))
            scores = dict(policy.get("path_scores") or {})
            if confidence <= 0.0 or not all(
                path in scores for path in ("c_first", "rtl_direct")
            ):
                continue
            policy_weight += confidence
            for path in policy_totals:
                policy_totals[path] += confidence * float(scores[path])
            matched_policies.append({
                "policy_id": policy["policy_id"],
                "confidence": confidence,
                "evidence_count": int(policy.get("evidence_count", 0)),
                "path_scores": scores,
                "reason": policy.get("reason", ""),
                "version": policy.get("version", 1),
                "source": policy.get("source", ""),
            })

        policy_scores = {
            path: policy_totals[path] / policy_weight
            for path in policy_totals
        } if policy_weight else {}

        history_scores: Dict[str, float] = {}
        for path, stat in normalized.items():
            gain = stat.get("mean_weighted_gain")
            gain_score = (
                1.0 / (1.0 + math.exp(-float(gain) / 10.0))
                if gain is not None else 0.5
            )
            history_scores[path] = (
                0.70 * float(stat["success_rate"]) + 0.30 * gain_score
            )

        combined_scores: Dict[str, float] = {}
        if policy_scores and history_scores:
            history_count = sum(
                int(stat["evidence_count"]) for stat in normalized.values()
            )
            history_weight = min(0.75, history_count / 10.0)
            for path in ("c_first", "rtl_direct"):
                combined_scores[path] = (
                    history_weight * history_scores.get(path, 0.0)
                    + (1.0 - history_weight) * policy_scores[path]
                )
        elif policy_scores:
            combined_scores = dict(policy_scores)
        elif history_scores:
            combined_scores = dict(history_scores)

        decision = (
            max(combined_scores, key=combined_scores.get)
            if combined_scores else history_decision
        )
        policy_margin = (
            abs(policy_scores["c_first"] - policy_scores["rtl_direct"])
            if policy_scores else 0.0
        )
        policy_is_supported = any(
            item["confidence"] >= 0.60
            and item["evidence_count"] >= 4
            for item in matched_policies
        ) and policy_margin >= 0.10
        history_is_supported = any(
            stat["evidence_count"] >= 3 for stat in normalized.values()
        )
        if policy_scores and normalized:
            decision_source = "policy_and_ppa_history"
        elif policy_scores:
            decision_source = "llm_soft_policy"
        elif decision:
            decision_source = "ppa_history"
        else:
            decision_source = "insufficient_evidence"
        return {
            "recommended_path": decision,
            "objective": objective,
            "path_statistics": normalized,
            "history_scores": history_scores,
            "policy_scores": policy_scores,
            "combined_scores": combined_scores,
            "matched_policies": matched_policies,
            "similar_tasks": similar,
            "has_sufficient_evidence": history_is_supported or policy_is_supported,
            "decision_source": decision_source,
        }

    def rank_actions(
        self,
        region_features: Dict[str, Any],
        objective: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        objective = normalize_objective(objective)
        query_type = str(
            region_features.get("region_type")
            or region_features.get("region_type_guess")
            or ""
        )
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in self.store.action_rows(objective):
            candidate_features = dict(row["context"].get("region_features", {}))
            candidate_type = row.get("region_type", "")
            outcome = row["outcome"]
            gain = (
                outcome.get("area_gain_pct")
                if objective == "AREA"
                else outcome.get("timing_gain_ps")
            )
            if row.get("applied_successfully") is None and gain is None:
                continue
            type_score = 1.0 if query_type and query_type == candidate_type else 0.25
            numeric_fields = ("node_count", "edge_count", "graph_depth", "branch_count")
            structural = [
                _numeric_similarity(region_features.get(key), candidate_features.get(key))
                for key in numeric_fields
                if region_features.get(key) is not None
                and candidate_features.get(key) is not None
            ]
            structural_score = sum(structural) / len(structural) if structural else 0.5
            similarity = 0.65 * type_score + 0.35 * structural_score

            success = row.get("applied_successfully")
            success_prior = 1.0 if success == 1 else 0.0 if success == 0 else 0.5
            gain_score = 0.5
            if gain is not None:
                gain_score = 1.0 / (1.0 + math.exp(-float(gain) / 10.0))
            score = 0.55 * similarity + 0.30 * success_prior + 0.15 * gain_score

            name = row.get("transform_name", "")
            aggregate = grouped.setdefault(name, {
                "transform_name": name,
                "score_total": 0.0,
                "weight": 0.0,
                "gain_total": 0.0,
                "gain_weight": 0.0,
                "evidence_count": 0,
                "success_count": 0,
                "examples": [],
            })
            aggregate["score_total"] += score * similarity
            aggregate["weight"] += similarity
            if gain is not None:
                aggregate["gain_total"] += float(gain) * similarity
                aggregate["gain_weight"] += similarity
            aggregate["evidence_count"] += 1
            aggregate["success_count"] += int(success == 1)
            if len(aggregate["examples"]) < 3:
                aggregate["examples"].append({
                    "benchmark": row.get("benchmark"),
                    "region_id": row.get("region_id"),
                    "region_type": candidate_type,
                    "similarity": round(similarity, 6),
                    "outcome": outcome,
                    "source_path": row.get("source_path"),
                })

        result = []
        for aggregate in grouped.values():
            count = aggregate["evidence_count"]
            result.append({
                "transform_name": aggregate["transform_name"],
                "score": round(
                    aggregate["score_total"] / aggregate["weight"]
                    if aggregate["weight"] else 0.0,
                    6,
                ),
                "evidence_count": count,
                "success_rate": aggregate["success_count"] / count if count else 0.0,
                "mean_gain": (
                    aggregate["gain_total"] / aggregate["gain_weight"]
                    if aggregate["gain_weight"] else None
                ),
                "examples": aggregate["examples"],
            })
        return sorted(
            result,
            key=lambda item: (item["score"], item["evidence_count"]),
            reverse=True,
        )[:max(0, top_k)]

    def retrieve_failure_guidance(
        self,
        context: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        query_type = str(context.get("failure_type", ""))
        query_stage = str(context.get("failure_stage", ""))
        query_tokens = _tokens(context)
        ranked = []
        for row in self.store.failure_rows():
            type_score = float(bool(query_type) and row["failure_type"] == query_type)
            stage_score = float(bool(query_stage) and row["failure_stage"] == query_stage)
            text_score = _jaccard(query_tokens, _tokens(row["context"]))
            score = 0.50 * type_score + 0.25 * stage_score + 0.25 * text_score
            ranked.append({
                "score": round(score, 6),
                "benchmark": row.get("benchmark"),
                "failure_type": row.get("failure_type"),
                "failure_stage": row.get("failure_stage"),
                "context": row.get("context"),
                "fix_applied": row.get("fix_applied"),
                "fix_succeeded": (
                    None if row.get("fix_succeeded") is None
                    else bool(row["fix_succeeded"])
                ),
                "retries_needed": row.get("retries_needed"),
                "source_path": row.get("source_path"),
            })
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:max(0, top_k)]


def _safe_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        import json
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}
