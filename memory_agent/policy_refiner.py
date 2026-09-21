"""LLM-assisted refinement for soft path-selection policies."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv

from memory_agent.sqlite_store import SQLiteMemoryStore
from memory_agent.validators import normalize_objective


DATA_PROCESSING_OPS = {
    "arithmetic", "bitwise", "shift", "comparison", "mux",
}
MEMORY_PROTOCOL_OPS = {"memory_access", "serial_protocol"}
MEMORY_PROTOCOL_PATTERNS = {
    "protocol_handler", "handshake_interface",
}

SOFT_PATH_POLICIES: Dict[str, Dict[str, Any]] = {
    "data_processing": {
        "description": "Single-clock data-processing behavior.",
        "conditions": {
            "num_clock_domains": {"lte": 1},
            "suggested_subcategory": {
                "in": ["arithmetic", "logical", "selection"],
            },
            "key_operations": {
                "contains_any": sorted(DATA_PROCESSING_OPS),
            },
            "match": "subcategory_or_key_operations",
        },
        "seed_scores": {"c_first": 0.70, "rtl_direct": 0.30},
    },
    "fsm_behavior": {
        "description": "Single-clock finite-state-machine behavior.",
        "conditions": {
            "num_clock_domains": {"lte": 1},
            "has_fsm": True,
        },
        "seed_scores": {"c_first": 0.65, "rtl_direct": 0.35},
    },
    "memory_protocol_behavior": {
        "description": "Single-clock memory or protocol behavior.",
        "conditions": {
            "num_clock_domains": {"lte": 1},
            "architecture_pattern": "memory",
            "key_operations": {
                "contains_any": sorted(MEMORY_PROTOCOL_OPS),
            },
            "sub_patterns": {
                "contains_any": sorted(MEMORY_PROTOCOL_PATTERNS),
            },
            "match": "architecture_or_operations_or_patterns",
        },
        "seed_scores": {"c_first": 0.60, "rtl_direct": 0.40},
    },
    "medium_complex": {
        "description": "Moderate or complex single-clock non-wrapper design.",
        "conditions": {
            "num_clock_domains": {"lte": 1},
            "complexity": {"in": ["moderate", "complex"]},
            "hierarchy": {"not_in": ["wrapper_only"]},
        },
        "seed_scores": {"c_first": 0.60, "rtl_direct": 0.40},
    },
    "single_clock_sequential": {
        "description": "Broad single-clock sequential behavior.",
        "conditions": {
            "num_clock_domains": {"lte": 1},
            "is_sequential": True,
        },
        "seed_scores": {"c_first": 0.55, "rtl_direct": 0.45},
    },
}


def _as_lower_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return {str(item).lower() for item in value}


def policy_matches(policy_id: str, features: Dict[str, Any]) -> bool:
    """Match one of the five supported soft policies against Module 1 features."""
    clocks = int(features.get("num_clock_domains") or 1)
    if clocks > 1:
        return False

    if policy_id == "data_processing":
        subcategory = str(features.get("suggested_subcategory") or "").lower()
        operations = _as_lower_set(features.get("key_operations"))
        return (
            subcategory in {"arithmetic", "logical", "selection"}
            or bool(operations & DATA_PROCESSING_OPS)
        )
    if policy_id == "fsm_behavior":
        return bool(features.get("has_fsm"))
    if policy_id == "memory_protocol_behavior":
        architecture = str(features.get("architecture_pattern") or "").lower()
        operations = _as_lower_set(features.get("key_operations"))
        patterns = _as_lower_set(features.get("sub_patterns"))
        return (
            architecture == "memory"
            or bool(operations & MEMORY_PROTOCOL_OPS)
            or bool(patterns & MEMORY_PROTOCOL_PATTERNS)
        )
    if policy_id == "medium_complex":
        complexity = str(features.get("complexity") or "").lower()
        hierarchy = str(features.get("hierarchy") or "").lower()
        return complexity in {"moderate", "complex"} and hierarchy != "wrapper_only"
    if policy_id == "single_clock_sequential":
        return bool(features.get("is_sequential"))
    return False


def _path_statistics(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(samples)
    success_rows = [row for row in rows if row["successful"]]
    gains = [
        float(row["gain"])
        for row in success_rows
        if row.get("gain") is not None and math.isfinite(float(row["gain"]))
    ]
    return {
        "count": len(rows),
        "success_count": len(success_rows),
        "success_rate": len(success_rows) / len(rows) if rows else 0.0,
        "metric_count": len(gains),
        "mean_gain": mean(gains) if gains else None,
        "gain_stddev": pstdev(gains) if len(gains) > 1 else 0.0 if gains else None,
        "regression_rate": (
            sum(gain <= 0.0 for gain in gains) / len(gains)
            if gains else None
        ),
    }


def build_policy_statistics(
    store: SQLiteMemoryStore,
    objectives: Iterable[str] = ("AREA", "TIMING"),
) -> List[Dict[str, Any]]:
    """Build deterministic evidence summaries before any LLM call."""
    requested = {normalize_objective(item) for item in objectives}
    rows = store.rows(
        """
        SELECT e.objective, e.path, e.correctness_status, e.synthesis_status,
               e.area_gain_pct, e.timing_gain_ps, f.features_json
        FROM evaluations e
        JOIN features f ON f.record_id = (
            SELECT latest.record_id
            FROM features latest
            WHERE latest.task_id = e.task_id
            ORDER BY latest.timestamp DESC, latest.record_id DESC
            LIMIT 1
        )
        WHERE e.path IN ('c_first', 'rtl_direct')
          AND e.objective IN ('AREA', 'TIMING')
        """
    )

    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        objective = normalize_objective(row.get("objective"))
        if objective not in requested:
            continue
        features = json.loads(row.get("features_json") or "{}")
        correctness = str(row.get("correctness_status") or "").lower()
        synthesis = str(row.get("synthesis_status") or "").lower()
        successful = (
            correctness not in {"failed", "error", "mismatch"}
            and synthesis in {"success", "pass", "passed", "met"}
        )
        gain = (
            row.get("area_gain_pct")
            if objective == "AREA"
            else row.get("timing_gain_ps")
        )
        for policy_id in SOFT_PATH_POLICIES:
            if policy_matches(policy_id, features):
                grouped.setdefault(
                    (policy_id, objective, str(row["path"])), []
                ).append({"successful": successful, "gain": gain})

    result = []
    for objective in sorted(requested):
        for policy_id, definition in SOFT_PATH_POLICIES.items():
            by_path = {
                path: _path_statistics(
                    grouped.get((policy_id, objective, path), [])
                )
                for path in ("c_first", "rtl_direct")
            }
            result.append(
                {
                    "policy_id": policy_id,
                    "objective": objective,
                    "description": definition["description"],
                    "conditions": definition["conditions"],
                    "seed_scores": definition["seed_scores"],
                    "path_results": by_path,
                    "evidence_count": sum(
                        item["count"] for item in by_path.values()
                    ),
                }
            )
    return result


def _extract_json(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        lines = lines[1:-1] if len(lines) >= 2 else lines
        if lines and lines[0].strip().lower() == "json":
            lines = lines[1:]
        content = "\n".join(lines).strip()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Policy refiner response must be a JSON object")
    return parsed


def validate_policy_proposals(
    payload: Dict[str, Any],
    statistics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate LLM output against known policy IDs and deterministic evidence."""
    evidence = {
        (item["policy_id"], item["objective"]): item
        for item in statistics
    }
    proposals = payload.get("policies")
    if not isinstance(proposals, list):
        raise ValueError("Policy refiner response is missing a policies list")

    validated: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in proposals:
        if not isinstance(item, dict):
            raise ValueError("Each policy proposal must be an object")
        policy_id = str(item.get("policy_id") or "")
        objective = normalize_objective(item.get("objective"))
        key = (policy_id, objective)
        if key not in evidence:
            raise ValueError(f"Unknown or unsupported policy proposal: {key}")
        if key in seen:
            raise ValueError(f"Duplicate policy proposal: {key}")
        seen.add(key)

        scores = item.get("path_scores")
        if not isinstance(scores, dict):
            raise ValueError(f"{policy_id} is missing path_scores")
        c_score = float(scores.get("c_first"))
        r_score = float(scores.get("rtl_direct"))
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (c_score, r_score)):
            raise ValueError(f"{policy_id} path scores must be within [0, 1]")
        total = c_score + r_score
        if total <= 0:
            raise ValueError(f"{policy_id} path scores cannot both be zero")
        confidence = float(item.get("confidence"))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{policy_id} confidence must be within [0, 1]")
        reason = str(item.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"{policy_id} is missing a reason")

        source = evidence[key]
        path_results = source["path_results"]
        low_or_one_sided = (
            source["evidence_count"] < 4
            or path_results["c_first"]["count"] < 2
            or path_results["rtl_direct"]["count"] < 2
        )
        if low_or_one_sided and confidence > 0.40:
            raise ValueError(
                f"{policy_id} confidence exceeds 0.40 for low or one-sided evidence"
            )
        validated.append(
            {
                "policy_id": policy_id,
                "objective": objective,
                "conditions": source["conditions"],
                "path_scores": {
                    "c_first": c_score / total,
                    "rtl_direct": r_score / total,
                },
                "confidence": confidence,
                "evidence_count": source["evidence_count"],
                "evidence": path_results,
                "reason": reason,
                "source": "llm_policy_refiner",
                "enabled": True,
            }
        )
    return validated


class PolicyRefiner:
    """Generate validated soft-policy proposals from deterministic statistics."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        *,
        env_path: str | Path = ".env",
        client: Any = None,
        model: str = "",
    ) -> None:
        self.store = store
        self.env_path = Path(env_path)
        self.client = client
        self.model = model

    def _ensure_client(self) -> None:
        if self.client is not None:
            return
        load_dotenv(self.env_path, override=True)
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        self.model = (
            self.model
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("LLM_MODEL")
            or "gpt-4o-mini"
        ).strip()
        if not api_key or not base_url:
            raise RuntimeError(
                "Policy refinement requires OPENAI_API_KEY and OPENAI_BASE_URL"
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def refine(
        self,
        *,
        objectives: Iterable[str] = ("AREA", "TIMING"),
        max_tokens: int = 4096,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        statistics = build_policy_statistics(self.store, objectives)
        self._ensure_client()
        system = (
            "You refine five soft path-selection policies for an RTL optimization "
            "pipeline. Hard safety rules are outside your authority. Use only the "
            "provided statistics. Do not invent evidence or change policy IDs or "
            "conditions. Return JSON only with key 'policies'. Each policy must have "
            "policy_id, objective, path_scores containing c_first and rtl_direct "
            "values in [0,1], confidence in [0,1], and a concise reason. Low or "
            "one-sided evidence means either path has fewer than 2 samples. Such "
            "policies must produce confidence <= 0.4 and scores close to the "
            "supplied seed_scores."
        )
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "retries": 0,
        }
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {"policy_statistics": statistics},
                    ensure_ascii=False,
                ),
            },
        ]
        last_error = ""
        raw_text = ""
        for attempt in range(max_retries + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            usage = getattr(response, "usage", None)
            token_usage["prompt_tokens"] += int(
                getattr(usage, "prompt_tokens", 0) or 0
            )
            token_usage["completion_tokens"] += int(
                getattr(usage, "completion_tokens", 0) or 0
            )
            token_usage["total_tokens"] += int(
                getattr(usage, "total_tokens", 0) or 0
            )
            raw_text = response.choices[0].message.content or ""
            try:
                proposals = validate_policy_proposals(
                    _extract_json(raw_text),
                    statistics,
                )
                return {
                    "model": self.model,
                    "statistics": statistics,
                    "policies": proposals,
                    "token_usage": token_usage,
                    "raw_response": raw_text,
                }
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                if attempt >= max_retries:
                    break
                token_usage["retries"] += 1
                messages.extend([
                    {"role": "assistant", "content": raw_text},
                    {
                        "role": "user",
                        "content": (
                            "The response failed deterministic validation: "
                            f"{last_error}. Return a complete corrected JSON object. "
                            "Do not omit previously requested policies."
                        ),
                    },
                ])
        raise ValueError(
            f"Policy refiner failed validation after {max_retries + 1} attempts: "
            f"{last_error}"
        )
