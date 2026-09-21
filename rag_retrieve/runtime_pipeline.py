"""In-process Module 4 pipeline with live Memory Agent guidance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from rag_retrieve.query_cdfg import extract_query_cdfg
from rag_retrieve.region_extract import extract_regions_from_query_record
from rag_retrieve.reranker import _transform_applicable, rerank_query_regions
from rag_retrieve.transform_planner import build_transform_plan
from rag_retrieve.defaults import DEFAULT_HISTORICAL_REGION_INDEX, DEFAULT_TRANSFORM_BLACKLIST


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _merge_memory_guidance(
    rerank_payload: Dict[str, Any],
    memory_session: Any,
) -> None:
    for graph in rerank_payload.get("graphs", []):
        for region in graph.get("regions", []):
            features = {
                "region_type": region.get("query_region_type", ""),
                **dict(region.get("query_region_features") or {}),
            }
            ranked = memory_session.rank_actions(features)
            existing = {
                item["transform_name"]: item
                for item in region.get("recommended_transforms", [])
            }
            for memory_item in ranked:
                mean_gain = memory_item.get("mean_gain")
                name = str(memory_item.get("transform_name") or "")
                if not name:
                    continue
                applicable, _reason = _transform_applicable(name, features)
                if not applicable:
                    continue
                if name not in existing and (
                    mean_gain is None or float(mean_gain) <= 0.0
                ):
                    continue
                rec = existing.setdefault(
                    name,
                    {
                        "transform_name": name,
                        "aggregated_weight": 0.0,
                        "priority_score": 0.0,
                        "expected_metric_gain": float(mean_gain or 0.0),
                        "samples": 0,
                        "raw_samples": 0,
                        "evidence_levels": ["memory"],
                    },
                )
                memory_count = max(1, int(memory_item.get("evidence_count", 0)))
                memory_priority = float(memory_item.get("score", 0.0)) * memory_count
                rec["priority_score"] = float(rec.get("priority_score", rec.get("aggregated_weight", 0.0))) + memory_priority
                rec["aggregated_weight"] = rec["priority_score"]
                if mean_gain is not None:
                    prior_count = max(float(rec.get("raw_samples", rec.get("samples", 0.0))), 0.0)
                    prior_gain = float(rec.get("expected_metric_gain", 0.0))
                    rec["expected_metric_gain"] = (
                        prior_gain * prior_count + float(mean_gain) * memory_count
                    ) / (prior_count + memory_count)
                rec["samples"] = max(
                    int(rec.get("samples", 0)),
                    int(memory_item.get("evidence_count", 0)),
                )
                rec["raw_samples"] = max(int(rec.get("raw_samples", 0)), memory_count)
                rec["memory_score"] = float(memory_item.get("score", 0.0))
                rec["memory_success_rate"] = float(
                    memory_item.get("success_rate", 0.0)
                )
                rec["memory_evidence_count"] = int(
                    memory_item.get("evidence_count", 0)
                )
                rec["memory_evidence"] = list(memory_item.get("examples", []))
            region["recommended_transforms"] = sorted(
                existing.values(),
                key=lambda item: (
                    float(item.get("memory_score", 0.0)),
                    float(item.get("aggregated_weight", 0.0)),
                    float(item.get("expected_metric_gain", 0.0)),
                ),
                reverse=True,
            )


def build_runtime_transform_plan(
    *,
    c_path: str | Path,
    benchmark: str,
    objective: str,
    historical_region_index_path: str | Path = DEFAULT_HISTORICAL_REGION_INDEX,
    blacklist_path: str | Path = DEFAULT_TRANSFORM_BLACKLIST,
    query_root: str | Path = "/tmp/module4_queries",
    top_k_regions: int = 5,
    subcategory: Optional[str] = None,
    memory_session: Optional[Any] = None,
) -> Dict[str, Any]:
    objective = objective.lower()
    query = extract_query_cdfg(
        c_path=c_path,
        query_root=query_root,
        benchmark=benchmark,
    )
    regions = extract_regions_from_query_record(query.to_dict())
    reranked = rerank_query_regions(
        query_regions_payload=regions,
        historical_region_index=_load_json(historical_region_index_path),
        objective=objective,
        top_k_regions=top_k_regions,
        subcategory=subcategory,
    )
    if memory_session is not None:
        _merge_memory_guidance(reranked, memory_session)
    plan = build_transform_plan(reranked, _load_json(blacklist_path))
    if memory_session is not None:
        memory_session.record_module4_plan(plan)
        memory_session.record_artifact(
            "query_cdfg",
            query.copied_c_path,
            producer="rag_retrieve",
            metadata={"cdfg_dir": query.cdfg_dir},
        )
    return plan


def refresh_transform_plan_memory(
    plan: Dict[str, Any],
    memory_session: Any,
) -> Dict[str, Any]:
    for region in plan.get("llm_ready_payload", {}).get("regions", []):
        ranked = memory_session.rank_actions(
            {
                "region_type": region.get("region_type", ""),
                **dict(region.get("region_features") or {}),
            }
        )
        by_name = {
            str(item.get("transform_name") or ""): item for item in ranked
        }
        for rec in region.get("recommended_transforms", []):
            memory_item = by_name.get(str(rec.get("transform_name") or ""))
            if memory_item is None:
                continue
            rec["memory_score"] = float(memory_item.get("score", 0.0))
            rec["memory_success_rate"] = float(
                memory_item.get("success_rate", 0.0)
            )
            rec["memory_evidence_count"] = int(
                memory_item.get("evidence_count", 0)
            )
            rec["memory_evidence"] = list(memory_item.get("examples", []))
            mean_gain = memory_item.get("mean_gain")
            if mean_gain is not None:
                prior_count = max(float(rec.get("raw_samples", rec.get("samples", 0.0))), 0.0)
                memory_count = max(int(memory_item.get("evidence_count", 0)), 1)
                rec["expected_metric_gain"] = (
                    float(rec.get("expected_metric_gain", 0.0)) * prior_count
                    + float(mean_gain) * memory_count
                ) / (prior_count + memory_count)
    memory_session.emit(
        "module4_5",
        "module5_feedback_read",
        {"region_count": len(plan.get("llm_ready_payload", {}).get("regions", []))},
    )
    return plan
