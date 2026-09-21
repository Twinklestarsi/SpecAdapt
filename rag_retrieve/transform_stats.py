"""Robust transform statistics shared by index construction and reranking."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / total if total else 0.0


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    halfway = sum(weight for _, weight in ordered) / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= halfway:
            return value
    return ordered[-1][0]


def summarize_transform_rows(rows: Iterable[Dict[str, Any]], objective: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("objective") != objective:
            continue
        grouped[str(row.get("transform_name") or "")].append(row)

    summaries: List[Dict[str, Any]] = []
    for name, items in grouped.items():
        if not name:
            continue
        requested_count = len(items)
        no_op_count = sum(as_bool(item.get("no_op")) is True for item in items)
        applied_count = sum(as_bool(item.get("transform_applied")) is True for item in items)
        unknown_applied_count = sum(as_bool(item.get("transform_applied")) is None for item in items)

        valid: List[tuple[float, float, Dict[str, Any]]] = []
        for item in items:
            if item.get("pairing_status") != "exact":
                continue
            if as_bool(item.get("no_op")) is True:
                continue
            gain = as_float(item.get("objective_gain"))
            if gain is None:
                continue
            weight = as_float(item.get("attribution_weight")) or 1.0
            if weight <= 0.0:
                continue
            valid.append((gain, weight, item))

        values = [value for value, _, _ in valid]
        weights = [weight for _, weight, _ in valid]
        positive_weight = sum(weight for value, weight, _ in valid if value > 0.0)
        negative_weight = sum(weight for value, weight, _ in valid if value < 0.0)
        zero_weight = sum(weight for value, weight, _ in valid if value == 0.0)
        effective = sum(weights)
        mean_gain = _weighted_mean(values, weights) if values else 0.0
        median_gain = weighted_median(values, weights) if values else 0.0
        if len(values) >= 2:
            variance = _weighted_mean([(value - mean_gain) ** 2 for value in values], weights)
            gain_stddev = math.sqrt(max(variance, 0.0))
        else:
            gain_stddev = 0.0

        evidence_levels = sorted({str(item.get("evidence_level") or "benchmark") for _, _, item in valid})
        backends = sorted({str(item.get("backend") or "") for item in items if item.get("backend")})
        constraints = sorted({str(item.get("constraint_id") or "") for item in items if item.get("constraint_id")})
        summary = {
            "transform_name": name,
            "objective": objective,
            "requested_count": requested_count,
            "effective_sample_count": round(effective, 6),
            "raw_effective_sample_count": len(valid),
            "applied_count": applied_count,
            "unknown_applied_count": unknown_applied_count,
            "no_op_count": no_op_count,
            "applicability_rate": round(applied_count / requested_count, 6) if requested_count else 0.0,
            "positive_count": sum(value > 0.0 for value in values),
            "negative_count": sum(value < 0.0 for value in values),
            "zero_count": sum(value == 0.0 for value in values),
            "positive_rate": round(positive_weight / effective, 6) if effective else 0.0,
            "regression_rate": round(negative_weight / effective, 6) if effective else 0.0,
            "zero_rate": round(zero_weight / effective, 6) if effective else 0.0,
            "median_gain": round(median_gain, 6),
            "mean_gain": round(mean_gain, 6),
            "min_gain": round(min(values), 6) if values else 0.0,
            "max_gain": round(max(values), 6) if values else 0.0,
            "gain_stddev": round(gain_stddev, 6),
            "evidence_levels": evidence_levels,
            "backends": backends,
            "constraint_ids": constraints,
            "sample_ids": sorted({str(item.get("sample_id") or "") for _, _, item in valid if item.get("sample_id")}),
        }

        mapping_rows = [
            (weight, item) for _gain, weight, item in valid if item.get("mapping_method")
        ]
        if mapping_rows:
            mapping_scores = [
                (score, weight)
                for weight, item in mapping_rows
                if (score := as_float(item.get("mapping_score"))) is not None
            ]
            mapping_confidences = [
                (confidence, weight)
                for weight, item in mapping_rows
                if (confidence := as_float(item.get("mapping_confidence"))) is not None
            ]
            column_matched = [
                (weight, item)
                for weight, item in mapping_rows
                if (as_float(item.get("column_match_bonus")) or 0.0) > 0.0
            ]
            summary.update(
                {
                    "mapping_methods": sorted({
                        str(item.get("mapping_method")) for _weight, item in mapping_rows
                    }),
                    "column_matched_sample_count": len(column_matched),
                    "column_matched_effective_sample_count": round(
                        sum(weight for weight, _item in column_matched), 6
                    ),
                    "mean_mapping_score": round(
                        _weighted_mean(
                            [score for score, _weight in mapping_scores],
                            [weight for _score, weight in mapping_scores],
                        ),
                        6,
                    ) if mapping_scores else 0.0,
                    "mean_mapping_confidence": round(
                        _weighted_mean(
                            [confidence for confidence, _weight in mapping_confidences],
                            [weight for _confidence, weight in mapping_confidences],
                        ),
                        6,
                    ) if mapping_confidences else 0.0,
                }
            )
        summaries.append(summary)

    summaries.sort(
        key=lambda item: (
            item["median_gain"],
            item["positive_rate"] - item["regression_rate"],
            item["effective_sample_count"],
        ),
        reverse=True,
    )
    return summaries
