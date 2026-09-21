"""Adapters from repository artifacts into canonical memory records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from memory_agent.ids import file_sha256, stable_id
from memory_agent.ingest.common import ensure_run, ensure_task, iter_json_files, load_json
from memory_agent.schemas import (
    ActionRecord,
    ArtifactRecord,
    EvaluationRecord,
    FailureRecord,
    FeatureRecord,
    SelectionRecord,
    TrajectoryRecord,
    now_iso,
)
from memory_agent.sqlite_store import SQLiteMemoryStore
from memory_agent.validators import (
    normalize_area_gain,
    normalize_objective,
    normalize_path,
    normalize_timing_gain,
    validate_feature_profile,
)
from project_paths import relocate_project_path


def _list_payload(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Expected a JSON object or list in {path}")


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


RAG_INDEX_MIN_SCHEMA_VERSION = 4
"""Historical region indices below this schema version attributed benchmark-level
PPA numbers to every region of the benchmark, which inflated evidence counts by up
to 100x. Refuse to import them."""


def _rag_index_evidence_key(
    benchmark: str,
    objective: str,
    transform_name: str,
    region_type: str,
    area_gain: float | None,
    timing_gain: float | None,
) -> tuple:
    """One measurement is one vote.

    A single benchmark-level PPA measurement is legitimately attributed to several
    regions, but it must not be counted several times when aggregating gains.
    Regions of *different* types keep their own evidence row because region type
    drives similarity matching; identical (type, gain) repeats collapse into one.
    """

    def _round(value: float | None) -> float | None:
        return None if value is None else round(float(value), 6)

    return (
        benchmark,
        objective,
        transform_name,
        region_type,
        _round(area_gain),
        _round(timing_gain),
    )


def _rag_transform_evidence(
    transform: Dict[str, Any],
    entry_objective: str,
) -> Dict[str, Any]:
    """Normalize the two historical `top_transforms` schemas into one shape.

    Pre-schema-4 indices stored one raw CSV row per transform::

        {"transform_name", "area_improvement_pct", "delay_improvement_ps", "slack_status"}

    Schema 4 stores an aggregated summary from transform_stats.summarize_transform_rows::

        {"transform_name", "objective", "median_gain", "mean_gain", "positive_rate",
         "raw_effective_sample_count", ...}

    The summary form is preferred and its **median** is used as the representative
    gain: these aggregates routinely contain a few -300%..-700% outliers that drag
    the mean far below anything the transform typically does.
    """

    name = str(transform.get("transform_name") or "")
    if "median_gain" in transform:
        objective = normalize_objective(transform.get("objective") or entry_objective)
        gain = _float_or_none(transform.get("median_gain"))
        sample_count = int(transform.get("raw_effective_sample_count") or 0)
        positive_rate = _float_or_none(transform.get("positive_rate"))
        if objective == "AREA":
            area_gain, timing_gain = normalize_area_gain(gain), None
        else:
            area_gain, timing_gain = None, normalize_timing_gain(gain)
        return {
            "transform_name": name,
            "objective": objective,
            "area_gain": area_gain,
            "timing_gain": timing_gain,
            "sample_count": sample_count,
            "positive_rate": positive_rate,
            "mean_gain": _float_or_none(transform.get("mean_gain")),
            "min_gain": _float_or_none(transform.get("min_gain")),
            "max_gain": _float_or_none(transform.get("max_gain")),
            "slack_status": str(transform.get("slack_status") or ""),
            "evidence_schema": "summary",
        }

    return {
        "transform_name": name,
        "objective": normalize_objective(entry_objective),
        "area_gain": normalize_area_gain(transform.get("area_improvement_pct")),
        "timing_gain": normalize_timing_gain(transform.get("delay_improvement_ps")),
        "sample_count": 1,
        "positive_rate": None,
        "mean_gain": None,
        "min_gain": None,
        "max_gain": None,
        "slack_status": str(transform.get("slack_status") or ""),
        "evidence_schema": "raw_row",
    }


def _check_rag_index_schema(payload: Any, source: Path) -> int:
    if not isinstance(payload, dict):
        return 0
    version = int(payload.get("summary", {}).get("schema_version") or 0)
    if version < RAG_INDEX_MIN_SCHEMA_VERSION:
        raise ValueError(
            f"Refusing to import historical region index {source}: schema_version="
            f"{version or 'missing'} < {RAG_INDEX_MIN_SCHEMA_VERSION}. Indices built "
            "before schema 4 copied each benchmark's PPA numbers onto every one of "
            "its regions, which inflates memory evidence counts. Rebuild the index "
            "with rag_retrieve/region_index.py first."
        )
    return version


def import_spec_analysis(
    store: SQLiteMemoryStore,
    path: str | Path,
) -> Dict[str, int]:
    source = Path(path).resolve()
    counts = {"tasks": 0, "runs": 0, "features": 0, "artifacts": 0}
    for item in _list_payload(source):
        benchmark = str(item.get("benchmark") or item.get("module_name") or "unknown")
        objective = normalize_objective(item.get("optimization_target", ""))
        task_id = ensure_task(
            store,
            benchmark,
            spec_text=str(item.get("spec_text") or ""),
            source_path=str(source),
            metadata={"input_type": item.get("input_type", "")},
        )
        run_id = ensure_run(
            store, task_id, benchmark, f"spec:{source}",
            objective=objective, producer="spec_agent",
        )
        record_id = stable_id("feature", task_id, str(source))
        store.upsert_feature(FeatureRecord(
            record_id=record_id,
            task_id=task_id,
            run_id=run_id,
            producer="spec_agent",
            features=validate_feature_profile(item.get("llm_features", {})),
            confidence=dict(item.get("confidence") or {}),
            overall_confidence=str(item.get("overall_confidence") or ""),
            optimization_target=objective,
            source_path=str(source),
        ))
        store.upsert_artifact(ArtifactRecord(
            record_id=stable_id("artifact", record_id, str(source)),
            task_id=task_id,
            run_id=run_id,
            parent_id=record_id,
            producer="spec_agent",
            artifact_type="spec_analysis",
            path=str(source),
            sha256=file_sha256(source),
        ))
        for key in counts:
            counts[key] += 1
    return counts


def import_path_decisions(
    store: SQLiteMemoryStore,
    path: str | Path,
) -> Dict[str, int]:
    source = Path(path).resolve()
    payload = load_json(source)
    decisions = payload.get("decisions", []) if isinstance(payload, dict) else payload
    counts = {"tasks": 0, "runs": 0, "features": 0, "selections": 0, "evaluations": 0}
    occurrence: Dict[tuple[str, str], int] = {}

    for item in decisions:
        benchmark = str(item.get("benchmark") or "unknown")
        path_value = normalize_path(item.get("path", ""))
        objective = normalize_objective(item.get("optimization_target", ""))
        timestamp = str(item.get("timestamp") or "")
        key = (benchmark, timestamp)
        occurrence[key] = occurrence.get(key, 0) + 1
        source_key = f"path:{source}:{timestamp}:{path_value}:{occurrence[key]}"
        task_id = ensure_task(store, benchmark, source_path=str(source))
        run_id = ensure_run(
            store, task_id, benchmark, source_key,
            objective=objective, path=path_value, producer="path_selector",
        )
        feature_id = stable_id("feature", run_id, "path_decision")
        store.upsert_feature(FeatureRecord(
            record_id=feature_id,
            task_id=task_id,
            run_id=run_id,
            producer="path_selector",
            timestamp=timestamp or now_iso(),
            features=validate_feature_profile(item.get("feature_profile", {})),
            optimization_target=objective,
            source_path=str(source),
        ))
        selection_id = stable_id("selection", run_id)
        store.upsert_selection(SelectionRecord(
            record_id=selection_id,
            task_id=task_id,
            run_id=run_id,
            parent_id=feature_id,
            producer="path_selector",
            timestamp=timestamp or now_iso(),
            path=path_value,
            rule_fired=str(item.get("rule_fired") or ""),
            confidence=str(item.get("confidence") or ""),
            tier=int(item.get("tier") or 0),
            reason=str(item.get("reason") or item.get("llm_reasoning") or ""),
            token_usage=dict(item.get("token_usage") or {}),
        ))
        counts["tasks"] += 1
        counts["runs"] += 1
        counts["features"] += 1
        counts["selections"] += 1

        ppa = item.get("ppa_outcome")
        if isinstance(ppa, dict):
            legacy_area = ppa.get("area_improvement_pct")
            legacy_timing = ppa.get("timing_improvement_pct")
            delay_timing = ppa.get("timing_improvement_ps", ppa.get("delay_improvement_ps"))
            store.upsert_evaluation(EvaluationRecord(
                record_id=stable_id("evaluation", run_id, "legacy_ppa"),
                task_id=task_id,
                run_id=run_id,
                parent_id=selection_id,
                producer="legacy_memory",
                benchmark=benchmark,
                objective=objective,
                path=path_value,
                correctness_status="passed",
                synthesis_status="failed" if ppa.get("synthesis_failed") else "success",
                area_gain_pct=(-float(legacy_area) if legacy_area is not None else None),
                timing_gain_ps=(
                    float(delay_timing)
                    if delay_timing is not None
                    else (-float(legacy_timing) if legacy_timing is not None else None)
                ),
                slack_status=str(ppa.get("slack_status") or ""),
                failure_reason=str(ppa.get("notes") or ""),
                metadata={"legacy_cell_count_delta": ppa.get("cell_count_delta")},
                source_path=str(source),
            ))
            counts["evaluations"] += 1
    return counts


def import_rag_index(
    store: SQLiteMemoryStore,
    path: str | Path,
) -> Dict[str, int]:
    source = Path(path).resolve()
    payload = load_json(source)
    schema_version = _check_rag_index_schema(payload, source)
    entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    counts = {
        "tasks": 0,
        "runs": 0,
        "actions": 0,
        "evaluations": 0,
        "duplicate_evidence_skipped": 0,
    }
    seen_evidence: Dict[tuple, str] = {}
    duplicate_counts: Dict[tuple, int] = {}
    with store.transaction():
        for entry_index, entry in enumerate(entries):
            benchmark = str(entry.get("benchmark") or "unknown")
            objective = normalize_objective(entry.get("objective", ""))
            task_id = ensure_task(
                store, benchmark,
                source_path=str(entry.get("benchmark_dir") or source),
                metadata={"subcategory": entry.get("subcategory", "")},
            )
            run_id = ensure_run(
                store, task_id, benchmark,
                f"rag:{source}:{entry_index}:{entry.get('graph_function', '')}",
                objective=objective, path="c_first", producer="rag_history",
            )
            region = dict(entry.get("region") or {})
            region_id = str(region.get("region_id") or "")
            region_type = str(region.get("region_type") or "")
            for transform_index, transform in enumerate(entry.get("top_transforms", [])):
                evidence = _rag_transform_evidence(transform, objective)
                name = evidence["transform_name"]
                transform_objective = evidence["objective"] or objective
                area_gain = evidence["area_gain"]
                timing_gain = evidence["timing_gain"]
                evidence_key = _rag_index_evidence_key(
                    benchmark,
                    transform_objective,
                    name,
                    region_type,
                    area_gain,
                    timing_gain,
                )
                if evidence_key in seen_evidence:
                    duplicate_counts[evidence_key] = duplicate_counts.get(evidence_key, 0) + 1
                    counts["duplicate_evidence_skipped"] += 1
                    continue
                action_id = stable_id("action", run_id, region_id, name, transform_index)
                seen_evidence[evidence_key] = action_id
                ppa_improved = (
                    area_gain is not None and area_gain > 0
                    if transform_objective == "AREA"
                    else timing_gain is not None and timing_gain > 0
                )
                store.upsert_action(ActionRecord(
                    record_id=action_id,
                    task_id=task_id,
                    run_id=run_id,
                    producer="rag_history",
                    benchmark=benchmark,
                    objective=transform_objective,
                    path="c_first",
                    region_id=region_id,
                    region_type=region_type,
                    transform_name=name,
                    applied_successfully=ppa_improved,
                    context={
                        "region_features": region.get("features", {}),
                        "graph_function": entry.get("graph_function", ""),
                        "subcategory": entry.get("subcategory", ""),
                        "warnings": entry.get("warnings", []),
                        "rag_index_schema_version": schema_version,
                        "evidence_schema": evidence["evidence_schema"],
                        "evidence_sample_count": evidence["sample_count"],
                        "evidence_positive_rate": evidence["positive_rate"],
                        "evidence_key": list(evidence_key),
                    },
                    outcome={
                        "area_gain_pct": area_gain,
                        "timing_gain_ps": timing_gain,
                        "slack_status": evidence["slack_status"],
                        "mean_gain": evidence["mean_gain"],
                        "min_gain": evidence["min_gain"],
                        "max_gain": evidence["max_gain"],
                        "sample_count": evidence["sample_count"],
                    },
                    source_path=str(source),
                ))
                store.upsert_evaluation(EvaluationRecord(
                    record_id=stable_id("evaluation", action_id),
                    task_id=task_id,
                    run_id=run_id,
                    parent_id=action_id,
                    producer="rag_history",
                    benchmark=benchmark,
                    objective=transform_objective,
                    path="c_first",
                    correctness_status="historical_unverified",
                    synthesis_status="success",
                    area_gain_pct=area_gain,
                    timing_gain_ps=timing_gain,
                    slack_status=evidence["slack_status"],
                    metadata={
                        "region_id": region_id,
                        "region_type": region_type,
                        "transform_name": name,
                        "evidence_schema": evidence["evidence_schema"],
                        "evidence_sample_count": evidence["sample_count"],
                    },
                    source_path=str(source),
                ))
                counts["actions"] += 1
                counts["evaluations"] += 1
            counts["tasks"] += 1
            counts["runs"] += 1
        for evidence_key, extra in duplicate_counts.items():
            store.annotate_action_context(
                seen_evidence[evidence_key],
                {"collapsed_duplicate_regions": extra},
            )
    return counts


def import_module5(
    store: SQLiteMemoryStore,
    path: str | Path,
) -> Dict[str, int]:
    root = Path(path).resolve()
    counts = {
        "tasks": 0, "runs": 0, "actions": 0,
        "evaluations": 0, "failures": 0, "artifacts": 0,
    }
    for result_path in iter_json_files(root, "result.json"):
        result = load_json(result_path)
        if not isinstance(result, dict) or not result.get("benchmark"):
            continue
        actions = _load_module5_actions(result_path, result)
        benchmark = str(result["benchmark"])
        objective = normalize_objective(result.get("objective", ""))
        # ``direct_rtl`` is the C-first Module 5 AI C-to-RTL backend, not the
        # specification-direct route.  The latter is explicitly identified by
        # route/backend fields in its own result schema.
        path_value = (
            "rtl_direct"
            if result.get("route") == "rtl_direct"
            or result.get("backend") == "spec_to_rtl"
            else "c_first"
        )
        task_id = ensure_task(
            store, benchmark,
            source_path=str(result.get("source_c_path") or result_path),
        )
        run_id = ensure_run(
            store, task_id, benchmark, f"module5:{result_path}",
            objective=objective, path=path_value,
            status=str(result.get("status") or "imported"),
            producer="module5",
            metadata={
                "backend": result.get("backend", ""),
                "execution_mode": result.get("execution_mode", "single_action"),
            },
        )

        action_ids = []
        for index, action in enumerate(actions):
            record_id = stable_id("action", run_id, action.get("action_id", ""), index)
            action_ids.append(record_id)
            store.upsert_action(ActionRecord(
                record_id=record_id,
                task_id=task_id,
                run_id=run_id,
                producer="module5",
                benchmark=benchmark,
                objective=objective,
                path=path_value,
                region_id=str(action.get("region_id") or ""),
                region_type=str(action.get("region_type") or ""),
                transform_name=str(action.get("transform_name") or ""),
                selected=True,
                applied_successfully=result.get("status") == "success",
                planning_score=_float_or_none(action.get("planning_score")),
                expected_metric_gain=_float_or_none(action.get("expected_metric_gain")),
                confidence=_float_or_none(action.get("confidence")),
                context={
                    "region_features": action.get("source_anchor") or {},
                    "problem_hypothesis": action.get("problem_hypothesis", ""),
                    "execution_hint": action.get("execution_hint", ""),
                    "constraints": action.get("constraints", {}),
                    "supporting_matches": action.get("supporting_matches", []),
                },
                outcome={
                    "status": result.get("status", ""),
                    "objective_feedback": result.get("objective_feedback", {}),
                    "edit_summary": result.get("edit_summary", ""),
                },
                source_path=str(result_path),
            ))
            counts["actions"] += 1

        metrics = dict(result.get("metrics") or {})
        area = _float_or_none(metrics.get("total_cell_area"))
        baseline_area = _float_or_none(metrics.get("baseline_total_cell_area"))
        area_gain = (
            (baseline_area - area) / baseline_area * 100.0
            if area is not None and baseline_area not in (None, 0) else None
        )
        timing = _float_or_none(metrics.get("delay_ps", metrics.get("data_arrival_time_ps")))
        baseline_timing = _float_or_none(
            metrics.get("baseline_delay_ps", metrics.get("baseline_data_arrival_time_ps"))
        )
        timing_gain = (
            baseline_timing - timing
            if timing is not None and baseline_timing is not None else None
        )
        status = str(result.get("status") or "unknown")
        correctness = (
            "failed" if status != "success"
            or result.get("behavior_check_status") in {"failed", "verification_failed"}
            else "passed"
        )
        evaluation_id = stable_id("evaluation", run_id, "module5")
        store.upsert_evaluation(EvaluationRecord(
            record_id=evaluation_id,
            task_id=task_id,
            run_id=run_id,
            parent_id=action_ids[0] if action_ids else "",
            producer="module5",
            benchmark=benchmark,
            objective=objective,
            path=path_value,
            correctness_status=correctness,
            synthesis_status="success" if status == "success" else "failed",
            area=area,
            baseline_area=baseline_area,
            area_gain_pct=area_gain,
            data_arrival_time_ps=timing,
            baseline_data_arrival_time_ps=baseline_timing,
            timing_gain_ps=timing_gain,
            slack_ps=_float_or_none(metrics.get("slack_ps")),
            baseline_slack_ps=_float_or_none(metrics.get("baseline_slack_ps")),
            slack_status=str(metrics.get("slack_status") or ""),
            token_usage=dict(result.get("llm_token_totals") or {}),
            failure_reason="; ".join(str(note) for note in result.get("notes", [])),
            metadata={
                "dc_status": result.get("dc_status", ""),
                "hls_status": result.get("hls_status", ""),
                "rtl_generation_status": result.get("rtl_generation_status", ""),
                "behavior_check_status": result.get("behavior_check_status", ""),
                "failed_preconditions": result.get("failed_preconditions", []),
            },
            source_path=str(result_path),
        ))
        counts["evaluations"] += 1

        failure_type = _module5_failure_type(result)
        if failure_type:
            store.upsert_failure(FailureRecord(
                record_id=stable_id("failure", run_id, failure_type),
                task_id=task_id,
                run_id=run_id,
                parent_id=evaluation_id,
                producer="module5",
                benchmark=benchmark,
                module="module5",
                failure_type=failure_type,
                failure_stage=_failure_stage(result),
                context={
                    "notes": result.get("notes", []),
                    "failed_preconditions": result.get("failed_preconditions", []),
                    "validator_summary": result.get("validator_summary", {}),
                    "dc_attempt_errors": result.get("dc_attempt_errors", []),
                },
                fix_applied=str(result.get("followup_action_recommendation") or ""),
                fix_succeeded=False,
                retries_needed=int(result.get("dc_retry_count") or 0),
                source_path=str(result_path),
            ))
            counts["failures"] += 1

        counts["artifacts"] += _import_module5_artifacts(
            store, task_id, run_id, evaluation_id, result_path, result
        )
        counts["tasks"] += 1
        counts["runs"] += 1
    return counts


def import_c_generation(
    store: SQLiteMemoryStore,
    path: str | Path,
) -> Dict[str, int]:
    source = Path(path).resolve()
    counts = {
        "tasks": 0, "runs": 0, "artifacts": 0,
        "trajectories": 0, "failures": 0,
    }
    with store.transaction():
        for index, item in enumerate(_list_payload(source)):
            benchmark = str(item.get("benchmark") or "unknown")
            task_id = ensure_task(store, benchmark, source_path=str(source))
            success = bool(item.get("success", False))
            run_id = ensure_run(
                store, task_id, benchmark, f"c_generation:{source}:{index}",
                path="c_first",
                status="success" if success else "failed",
                producer="c_agent",
                metadata={"method": item.get("method", "")},
            )
            token_usage = dict(item.get("token_usage") or {})
            trajectory_id = stable_id("trajectory", run_id, "c_generation")
            store.upsert_trajectory(TrajectoryRecord(
                record_id=trajectory_id,
                task_id=task_id,
                run_id=run_id,
                producer="c_agent",
                path="c_first",
                events=[{
                    "stage": "c_generation",
                    "method": item.get("method", ""),
                    "success": success,
                    "error": item.get("error", ""),
                    "token_usage": token_usage,
                }],
                total_tokens=int(token_usage.get("total_tokens") or 0),
                succeeded=success,
                metadata={"source_manifest": str(source)},
            ))
            counts["trajectories"] += 1

            c_path = item.get("c_path")
            if c_path and relocate_project_path(c_path).is_file():
                artifact_path = relocate_project_path(c_path)
                store.upsert_artifact(ArtifactRecord(
                    record_id=stable_id("artifact", run_id, "generated_c", str(artifact_path)),
                    task_id=task_id,
                    run_id=run_id,
                    parent_id=trajectory_id,
                    producer="c_agent",
                    artifact_type="generated_c",
                    path=str(artifact_path.resolve()),
                    sha256=file_sha256(artifact_path),
                ))
                counts["artifacts"] += 1

            if not success:
                store.upsert_failure(FailureRecord(
                    record_id=stable_id("failure", run_id, "c_generation"),
                    task_id=task_id,
                    run_id=run_id,
                    parent_id=trajectory_id,
                    producer="c_agent",
                    benchmark=benchmark,
                    module="c_agent",
                    failure_type="c_generation_failed",
                    failure_stage="generation",
                    context={"error": item.get("error", "")},
                    source_path=str(source),
                ))
                counts["failures"] += 1
            counts["tasks"] += 1
            counts["runs"] += 1
    return counts


def import_mcts_plan(
    store: SQLiteMemoryStore,
    path: str | Path,
) -> Dict[str, int]:
    source = Path(path).resolve()
    payload = load_json(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    benchmark = str(payload.get("benchmark") or "unknown")
    objective = normalize_objective(payload.get("objective", ""))
    task_id = ensure_task(
        store, benchmark,
        source_path=str(payload.get("source_c_path") or source),
    )
    run_id = ensure_run(
        store, task_id, benchmark, f"mcts:{source}",
        objective=objective, path="c_first",
        producer="mcts_planner",
        metadata={
            "planner_name": payload.get("planner_name", ""),
            "iterations": payload.get("iterations", 0),
            "max_depth": payload.get("max_depth", 0),
            "search_summary": payload.get("search_summary", {}),
        },
    )
    best_ids = {
        str(action.get("action_id") or "")
        for action in payload.get("best_actions", [])
    }
    counts = {"tasks": 1, "runs": 1, "actions": 0, "trajectories": 1, "artifacts": 1}
    with store.transaction():
        for index, action in enumerate(payload.get("action_space", [])):
            action_id = str(action.get("action_id") or "")
            store.upsert_action(ActionRecord(
                record_id=stable_id("action", run_id, action_id, index),
                task_id=task_id,
                run_id=run_id,
                producer="mcts_planner",
                benchmark=benchmark,
                objective=objective,
                path="c_first",
                region_id=str(action.get("region_id") or ""),
                region_type=str(action.get("region_type") or ""),
                transform_name=str(action.get("transform_name") or ""),
                selected=action_id in best_ids,
                planning_score=_float_or_none(action.get("planning_score")),
                expected_metric_gain=_float_or_none(action.get("expected_metric_gain")),
                confidence=_float_or_none(action.get("confidence")),
                context={
                    "region_features": action.get("source_anchor") or {},
                    "problem_hypothesis": action.get("problem_hypothesis", ""),
                    "execution_hint": action.get("execution_hint", ""),
                    "supporting_matches": action.get("supporting_matches", []),
                    "constraints": action.get("constraints", {}),
                },
                source_path=str(source),
            ))
            counts["actions"] += 1
        trajectory_id = stable_id("trajectory", run_id, "mcts")
        store.upsert_trajectory(TrajectoryRecord(
            record_id=trajectory_id,
            task_id=task_id,
            run_id=run_id,
            producer="mcts_planner",
            path="c_first",
            events=list(payload.get("search_trace") or []),
            succeeded=bool(best_ids),
            metadata={
                "best_action_ids": sorted(best_ids),
                "harmful_blacklist": payload.get("harmful_blacklist", []),
            },
        ))
        store.upsert_artifact(ArtifactRecord(
            record_id=stable_id("artifact", run_id, "mcts_plan", str(source)),
            task_id=task_id,
            run_id=run_id,
            parent_id=trajectory_id,
            producer="mcts_planner",
            artifact_type="mcts_plan",
            path=str(source),
            sha256=file_sha256(source),
        ))
    return counts


def _load_module5_actions(
    result_path: Path,
    result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    action_path = result_path.with_name("action.json")
    sequence_path = result_path.with_name("sequence.json")
    if action_path.exists():
        payload = load_json(action_path)
        if isinstance(payload, dict):
            return [payload]
    if sequence_path.exists():
        payload = load_json(sequence_path)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    return [{
        "action_id": result.get("action_id", ""),
        "region_id": (result.get("changed_regions") or [""])[0],
        "region_type": "",
        "transform_name": result.get("applied_transform_name", ""),
    }]


def _import_module5_artifacts(
    store: SQLiteMemoryStore,
    task_id: str,
    run_id: str,
    evaluation_id: str,
    result_path: Path,
    result: Dict[str, Any],
) -> int:
    count = 0
    for artifact_type, value in (
        ("module5_result", result_path),
        ("source_c", result.get("source_c_path")),
        ("edited_c", result.get("edited_c_path")),
        ("generated_rtl", result.get("generated_verilog_path")),
        ("baseline_rtl", result.get("baseline_verilog_path")),
        ("rtl_prompt", result.get("rtl_prompt_path")),
        ("rtl_raw_response", result.get("rtl_raw_response_path")),
    ):
        if not value:
            continue
        artifact_path = relocate_project_path(value)
        if not artifact_path.exists() or not artifact_path.is_file():
            continue
        store.upsert_artifact(ArtifactRecord(
            record_id=stable_id("artifact", run_id, artifact_type, str(artifact_path)),
            task_id=task_id,
            run_id=run_id,
            parent_id=evaluation_id,
            producer="module5",
            artifact_type=artifact_type,
            path=str(artifact_path.resolve()),
            sha256=file_sha256(artifact_path),
        ))
        count += 1
    return count


def _module5_failure_type(result: Dict[str, Any]) -> str:
    status = str(result.get("status") or "unknown")
    if status != "success":
        return status
    if result.get("behavior_check_status") in {"failed", "verification_failed"}:
        return "behavior_verification_failed"
    return ""


def _failure_stage(result: Dict[str, Any]) -> str:
    status = str(result.get("status") or "")
    if status.startswith("edit"):
        return "editing"
    if status.startswith("rtl"):
        return "rtl_generation"
    if status.startswith("hls"):
        return "hls"
    if status.startswith("dc"):
        return "backend"
    if result.get("behavior_check_status") in {"failed", "verification_failed"}:
        return "verification"
    return "execution"
