"""Path-policy derivation from structured PPA experience."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Tuple

from memory_agent.sqlite_store import SQLiteMemoryStore


GROUP_FIELDS = (
    "architecture_pattern",
    "suggested_subcategory",
    "complexity",
    "is_sequential",
    "num_clock_domains",
)


def _group_key(features: Dict[str, Any], objective: str) -> Tuple[Any, ...]:
    return (objective,) + tuple(features.get(field) for field in GROUP_FIELDS)


def _conditions(features: Dict[str, Any], objective: str) -> Dict[str, Any]:
    result = {
        field: features[field]
        for field in GROUP_FIELDS
        if features.get(field) is not None
    }
    result["_optimization_target"] = objective
    return result


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (center - margin) / denominator


def derive_ppa_path_rules(
    store: SQLiteMemoryStore,
    min_evidence_per_path: int = 2,
    min_gain_margin: float = 0.0,
) -> List[Dict[str, Any]]:
    rows = store.rows(
        """
        SELECT e.objective, e.path, e.area_gain_pct, e.timing_gain_ps,
               f.features_json
        FROM evaluations e
        JOIN features f ON f.record_id = (
            SELECT latest.record_id
            FROM features latest
            WHERE latest.task_id = e.task_id
            ORDER BY latest.timestamp DESC, latest.record_id DESC
            LIMIT 1
        )
        WHERE lower(e.correctness_status) NOT IN ('failed', 'error', 'mismatch')
          AND lower(e.synthesis_status) IN ('success', 'pass', 'passed', 'met')
          AND e.path IN ('c_first', 'rtl_direct')
          AND e.objective IN ('AREA', 'TIMING')
        """
    )
    import json

    groups: Dict[Tuple[Any, ...], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    profiles: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        features = json.loads(row["features_json"])
        key = _group_key(features, row["objective"])
        gain = (
            row["area_gain_pct"]
            if row["objective"] == "AREA"
            else row["timing_gain_ps"]
        )
        if gain is None:
            continue
        groups[key][row["path"]].append(float(gain))
        profiles.setdefault(key, features)

    rules: List[Dict[str, Any]] = []
    for key, by_path in groups.items():
        c_values = by_path.get("c_first", [])
        r_values = by_path.get("rtl_direct", [])
        if len(c_values) < min_evidence_per_path or len(r_values) < min_evidence_per_path:
            continue
        c_mean, r_mean = mean(c_values), mean(r_values)
        if abs(c_mean - r_mean) <= min_gain_margin:
            continue
        decision = "c_first" if c_mean > r_mean else "rtl_direct"
        winner_values = c_values if decision == "c_first" else r_values
        successes = sum(value > 0 for value in winner_values)
        confidence_lower_bound = _wilson_lower_bound(successes, len(winner_values))
        objective = str(key[0])
        profile = profiles[key]
        rules.append({
            "name": (
                f"verified_{objective.lower()}_"
                f"{profile.get('suggested_subcategory', 'unknown')}_{decision}"
            ),
            "priority": 100,
            "conditions": _conditions(profile, objective),
            "decision": decision,
            "confidence": "high" if confidence_lower_bound >= 0.60 else "medium",
            "evidence_count": len(c_values) + len(r_values),
            "evidence_by_path": {
                "c_first": len(c_values),
                "rtl_direct": len(r_values),
            },
            "mean_gain_by_path": {
                "c_first": round(c_mean, 6),
                "rtl_direct": round(r_mean, 6),
            },
            "confidence_lower_bound": round(confidence_lower_bound, 6),
            "win_rate": round(successes / len(winner_values), 6),
            "objective": objective,
            "source": "ppa_sqlite_memory",
            "rule_version": 1,
        })

    rules.sort(
        key=lambda rule: (
            -rule["confidence_lower_bound"],
            -rule["evidence_count"],
        )
    )
    for index, rule in enumerate(rules):
        rule["priority"] = 100 + index
    return rules
