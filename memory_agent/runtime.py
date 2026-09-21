"""Runtime synchronization for the optimization pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from memory_agent.agent import MemoryAgent
from memory_agent.ids import file_sha256, stable_id
from memory_agent.schemas import (
    ActionRecord,
    ArtifactRecord,
    EvaluationRecord,
    FailureRecord,
    FeatureRecord,
    RunRecord,
    SelectionRecord,
    TrajectoryRecord,
    now_iso,
)
from memory_agent.validators import normalize_objective, normalize_path


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    return dict(to_dict()) if callable(to_dict) else {}


def _token_usage(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    return dict(to_dict()) if callable(to_dict) else {}


@dataclass
class PipelineContext:
    task_id: str
    run_id: str
    benchmark: str
    objective: str
    path: str = ""
    source_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemorySession:
    """One durable memory session shared by all pipeline modules."""

    def __init__(self, agent: MemoryAgent, context: PipelineContext) -> None:
        self.agent = agent
        self.context = context
        self._events: list[Dict[str, Any]] = []

    @classmethod
    def start(
        cls,
        agent: MemoryAgent,
        *,
        benchmark: str,
        objective: str,
        spec_text: str = "",
        source_path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MemorySession":
        task_id = agent.create_task(
            benchmark=benchmark,
            spec_text=spec_text,
            source_path=source_path,
            metadata=metadata,
        )
        run_id = agent.start_run(
            task_id=task_id,
            objective=objective,
            producer="pipeline_orchestrator",
            metadata=metadata,
        )
        session = cls(
            agent,
            PipelineContext(
                task_id=task_id,
                run_id=run_id,
                benchmark=benchmark,
                objective=normalize_objective(objective),
                source_path=source_path,
                metadata=metadata or {},
            ),
        )
        session.emit("pipeline", "run_started", {"source_path": source_path})
        return session

    def emit(
        self,
        stage: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        status: str = "completed",
    ) -> str:
        timestamp = now_iso()
        event_id = stable_id(
            "event",
            self.context.run_id,
            stage,
            event_type,
            len(self._events),
        )
        self._events.append(
            {
                "event_id": event_id,
                "stage": stage,
                "event_type": event_type,
                "status": status,
                "timestamp": timestamp,
                "payload": payload or {},
            }
        )
        self.agent.db.upsert_trajectory(
            TrajectoryRecord(
                record_id=stable_id("trajectory", self.context.run_id),
                task_id=self.context.task_id,
                run_id=self.context.run_id,
                producer="pipeline_runtime",
                path=self.context.path,
                events=list(self._events),
                succeeded=status != "failed",
                metadata={"latest_stage": stage, "latest_event": event_type},
            )
        )
        return event_id

    def set_path(self, path: str) -> None:
        self.context.path = normalize_path(path)
        self.agent.db.upsert_run(
            RunRecord(
                run_id=self.context.run_id,
                task_id=self.context.task_id,
                objective=self.context.objective,
                path=self.context.path,
                status="active",
                producer="pipeline_orchestrator",
                metadata=self.context.metadata,
            )
        )

    def complete(self, status: str = "completed") -> None:
        self.emit("pipeline", "run_completed", {"status": status}, status=status)
        self.agent.db.upsert_run(
            RunRecord(
                run_id=self.context.run_id,
                task_id=self.context.task_id,
                objective=self.context.objective,
                path=self.context.path,
                status=status,
                producer="pipeline_orchestrator",
                metadata=self.context.metadata,
                completed_at=now_iso(),
            )
        )

    def record_artifact(
        self,
        artifact_type: str,
        path: str | Path,
        *,
        parent_id: str = "",
        producer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        artifact_path = Path(path)
        record_id = stable_id(
            "artifact", self.context.run_id, artifact_type, str(artifact_path)
        )
        self.agent.db.upsert_artifact(
            ArtifactRecord(
                record_id=record_id,
                task_id=self.context.task_id,
                run_id=self.context.run_id,
                parent_id=parent_id,
                producer=producer,
                artifact_type=artifact_type,
                path=str(artifact_path),
                sha256=file_sha256(artifact_path) if artifact_path.is_file() else "",
                metadata=metadata or {},
            )
        )
        return record_id

    def record_feature_result(self, result: Any) -> str:
        data = _as_dict(result)
        record_id = stable_id("feature", self.context.run_id, "module1")
        self.agent.record_feature(
            FeatureRecord(
                record_id=record_id,
                task_id=self.context.task_id,
                run_id=self.context.run_id,
                producer="spec_analyze",
                features=dict(data.get("llm_features") or {}),
                confidence=dict(data.get("confidence") or {}),
                overall_confidence=str(data.get("overall_confidence") or ""),
                optimization_target=self.context.objective,
                source_path=str(data.get("verilog_path") or self.context.source_path),
            )
        )
        self.emit("module1", "features_ready", {"record_id": record_id})
        return record_id

    def recommend_path(self, features: Dict[str, Any]) -> Dict[str, Any]:
        guidance = self.agent.recommend_path(features, self.context.objective)
        self.emit(
            "module2",
            "path_guidance_read",
            {
                "recommended_path": guidance.get("recommended_path", ""),
                "decision_source": guidance.get("decision_source", ""),
                "has_sufficient_evidence": guidance.get(
                    "has_sufficient_evidence", False
                ),
                "evidence_run_ids": [
                    item.get("run_id", "")
                    for item in guidance.get("similar_tasks", [])
                    if item.get("run_id")
                ],
            },
        )
        return guidance

    def record_path_decision(
        self,
        decision: Any,
        features: Dict[str, Any],
        *,
        guidance: Optional[Dict[str, Any]] = None,
    ) -> str:
        data = _as_dict(decision)
        self.set_path(str(data.get("path") or ""))
        parent_id = stable_id("feature", self.context.run_id, "module1")
        record_id = stable_id("selection", self.context.run_id)
        reason = str(data.get("reason") or "")
        if guidance:
            reason = f"{reason} Memory evidence: {guidance.get('decision_source', '')}."
        self.agent.record_selection(
            SelectionRecord(
                record_id=record_id,
                task_id=self.context.task_id,
                run_id=self.context.run_id,
                parent_id=parent_id,
                producer="path_select",
                path=self.context.path,
                rule_fired=str(data.get("rule_fired") or ""),
                confidence=str(data.get("confidence") or ""),
                tier=int(data.get("tier") or 0),
                reason=reason,
                token_usage=_token_usage(data.get("token_usage")),
            )
        )
        self.emit(
            "module2",
            "path_selected",
            {"record_id": record_id, "path": self.context.path},
        )
        return record_id

    def failure_guidance(
        self, module: str, failure_stage: str, context: Dict[str, Any]
    ) -> list[Dict[str, Any]]:
        query = {
            **context,
            "module": module,
            "failure_stage": failure_stage,
            "failure_type": context.get("failure_type", ""),
        }
        guidance = self.agent.retrieve_failure_guidance(query)
        self.emit(
            module,
            "failure_guidance_read",
            {"failure_stage": failure_stage, "evidence_count": len(guidance)},
        )
        return guidance

    def record_c_generation(self, result: Any, input_payload: Dict[str, Any]) -> str:
        data = _as_dict(result)
        record_id = stable_id("action", self.context.run_id, "c_generation")
        success = bool(data.get("success"))
        self.agent.record_action(
            ActionRecord(
                record_id=record_id,
                task_id=self.context.task_id,
                run_id=self.context.run_id,
                parent_id=stable_id("selection", self.context.run_id),
                producer="c_gen",
                benchmark=self.context.benchmark,
                objective=self.context.objective,
                path=self.context.path,
                transform_name=str(data.get("method") or "c_generation"),
                selected=True,
                applied_successfully=success,
                context={
                    "input_type": input_payload.get("input_type"),
                    "verilog_path": input_payload.get("verilog_path"),
                },
                outcome={
                    "success": success,
                    "error": str(data.get("error") or ""),
                    "token_usage": _token_usage(data.get("token_usage")),
                },
                source_path=str(data.get("c_path") or ""),
            )
        )
        c_path = str(data.get("c_path") or "")
        if success and c_path:
            self.record_artifact(
                "generated_c",
                c_path,
                parent_id=record_id,
                producer="c_gen",
                metadata={"method": data.get("method", "")},
            )
        if not success:
            self.record_failure(
                module="module3",
                failure_type="c_generation_failed",
                failure_stage="c_generation",
                context={"error": data.get("error", ""), **input_payload},
                parent_id=record_id,
            )
        self.emit(
            "module3",
            "c_generation_completed" if success else "c_generation_failed",
            {"record_id": record_id, "success": success, "c_path": c_path},
            status="completed" if success else "failed",
        )
        return record_id

    def rank_actions(
        self, region_features: Dict[str, Any]
    ) -> list[Dict[str, Any]]:
        return self.agent.rank_actions(region_features, self.context.objective)

    def record_planner_actions(
        self,
        actions: Iterable[Any],
        *,
        producer: str,
        selected_ids: Optional[set[str]] = None,
    ) -> None:
        selected_ids = selected_ids or set()
        for action in actions:
            data = _as_dict(action)
            action_id = str(data.get("action_id") or "")
            self.agent.record_action(
                ActionRecord(
                    record_id=stable_id("action", self.context.run_id, action_id),
                    task_id=self.context.task_id,
                    run_id=self.context.run_id,
                    parent_id=stable_id("selection", self.context.run_id),
                    producer=producer,
                    benchmark=self.context.benchmark,
                    objective=self.context.objective,
                    path=self.context.path,
                    region_id=str(data.get("region_id") or ""),
                    region_type=str(data.get("region_type") or ""),
                    transform_name=str(data.get("transform_name") or ""),
                    selected=action_id in selected_ids,
                    planning_score=data.get("planning_score"),
                    expected_metric_gain=data.get("expected_metric_gain"),
                    confidence=data.get("confidence"),
                    context={
                        "region_features": data.get("region_features", {}),
                        "source_anchor": data.get("source_anchor", {}),
                        "supporting_matches": data.get("supporting_matches", []),
                    },
                    source_path=str(data.get("source_c_path") or ""),
                )
            )

    def record_module4_plan(self, plan: Dict[str, Any]) -> None:
        candidate_count = sum(
            len(region.get("recommended_transforms", []))
            for region in plan.get("llm_ready_payload", {}).get("regions", [])
        )
        self.emit(
            "module4",
            "retrieval_completed",
            {
                "candidate_count": candidate_count,
                "region_count": len(
                    plan.get("llm_ready_payload", {}).get("regions", [])
                ),
            },
        )

    def record_module45_plan(self, plan: Dict[str, Any]) -> None:
        selected_ids = {
            str(item.get("action_id") or "")
            for item in plan.get("best_actions", [])
        }
        self.record_planner_actions(
            plan.get("action_space", []),
            producer="mcts_planner",
            selected_ids=selected_ids,
        )
        self.emit(
            "module4_5",
            "plan_selected",
            {
                "action_space_size": len(plan.get("action_space", [])),
                "selected_action_ids": sorted(selected_ids),
            },
        )

    def record_failure(
        self,
        *,
        module: str,
        failure_type: str,
        failure_stage: str,
        context: Dict[str, Any],
        parent_id: str = "",
        fix_applied: str = "",
        fix_succeeded: Optional[bool] = None,
        retries_needed: int = 0,
        source_path: str = "",
    ) -> str:
        record_id = stable_id(
            "failure",
            self.context.run_id,
            module,
            failure_stage,
            failure_type,
            len(self._events),
        )
        self.agent.record_failure(
            FailureRecord(
                record_id=record_id,
                task_id=self.context.task_id,
                run_id=self.context.run_id,
                parent_id=parent_id,
                producer=module,
                benchmark=self.context.benchmark,
                module=module,
                failure_type=failure_type,
                failure_stage=failure_stage,
                context=context,
                fix_applied=fix_applied,
                fix_succeeded=fix_succeeded,
                retries_needed=retries_needed,
                source_path=source_path,
            )
        )
        return record_id

    def record_module5_result(
        self,
        result: Any,
        actions: Iterable[Any],
    ) -> str:
        action_list = list(actions)
        data = _as_dict(result)
        metrics = dict(data.get("metrics") or {})
        status = str(data.get("status") or "unknown")
        success = status == "success"
        feedback = dict(data.get("objective_feedback") or {})
        reward = feedback.get("reward")
        per_action_summaries = [
            dict(item)
            for item in data.get("per_action_summaries", [])
            if isinstance(item, dict)
        ]

        single_action = len(action_list) == 1
        for index, action in enumerate(action_list):
            action_data = _as_dict(action)
            action_id = str(action_data.get("action_id") or "")
            action_summary = (
                per_action_summaries[index]
                if index < len(per_action_summaries)
                else {}
            )
            applied = action_summary.get("applied")
            action_succeeded = (
                success and applied
                if isinstance(applied, bool)
                else success
            )
            self.agent.record_action(
                ActionRecord(
                    record_id=stable_id("action", self.context.run_id, action_id),
                    task_id=self.context.task_id,
                    run_id=self.context.run_id,
                    parent_id=stable_id("selection", self.context.run_id),
                    producer="module5",
                    benchmark=self.context.benchmark,
                    objective=self.context.objective,
                    path=self.context.path,
                    region_id=str(action_data.get("region_id") or ""),
                    region_type=str(action_data.get("region_type") or ""),
                    transform_name=str(action_data.get("transform_name") or ""),
                    selected=True,
                    applied_successfully=action_succeeded,
                    planning_score=action_data.get("planning_score"),
                    expected_metric_gain=action_data.get("expected_metric_gain"),
                    confidence=action_data.get("confidence"),
                    context={
                        "region_features": action_data.get("region_features", {}),
                        "source_anchor": action_data.get("source_anchor", {}),
                        "application": {
                            "reported": isinstance(applied, bool),
                            "applied": applied,
                            "region": action_summary.get("region", ""),
                            "summary": action_summary.get("summary", ""),
                        },
                    },
                    outcome={
                        "status": status,
                        "sequence_status": status,
                        "action_applied": applied,
                        "action_execution_succeeded": action_succeeded,
                        # A sequence-level PPA reward cannot be causally assigned
                        # to each member of a multi-action chain.
                        "reward": reward if single_action else None,
                        "sequence_reward": reward,
                        "improved": feedback.get("improved"),
                        "area_gain_pct": (
                            self._area_gain_pct(metrics) if single_action else None
                        ),
                        "timing_gain_ps": (
                            self._timing_gain_ps(metrics) if single_action else None
                        ),
                        "metrics": metrics,
                    },
                    source_path=str(data.get("work_dir") or ""),
                )
            )

        evaluation_id = stable_id("evaluation", self.context.run_id, data.get("action_id"))
        self.agent.record_evaluation(
            EvaluationRecord(
                record_id=evaluation_id,
                task_id=self.context.task_id,
                run_id=self.context.run_id,
                parent_id=stable_id(
                    "action", self.context.run_id, str(data.get("action_id") or "")
                ),
                producer="module5",
                benchmark=self.context.benchmark,
                objective=self.context.objective,
                path=self.context.path,
                correctness_status=self._correctness_status(data),
                synthesis_status="success" if success else "failed",
                area=metrics.get("total_cell_area"),
                baseline_area=metrics.get("baseline_total_cell_area"),
                area_gain_pct=self._area_gain_pct(metrics),
                data_arrival_time_ps=metrics.get("delay_ps", metrics.get("data_arrival_time_ps")),
                baseline_data_arrival_time_ps=metrics.get(
                    "baseline_delay_ps", metrics.get("baseline_data_arrival_time_ps")
                ),
                timing_gain_ps=self._timing_gain_ps(metrics),
                slack_ps=metrics.get("slack_ps"),
                baseline_slack_ps=metrics.get("baseline_slack_ps"),
                slack_status=str(metrics.get("slack_status") or ""),
                token_usage=dict(data.get("llm_token_totals") or {}),
                failure_reason="" if success else status,
                metadata={
                    "reward": reward,
                    "backend": data.get("backend"),
                    "execution_mode": data.get("execution_mode"),
                    **self._verification_metadata(data),
                },
                source_path=str(data.get("work_dir") or ""),
            )
        )

        work_dir = str(data.get("work_dir") or "")
        result_path = Path(work_dir) / "result.json" if work_dir else None
        if result_path and result_path.is_file():
            self.record_artifact(
                "module5_result",
                result_path,
                parent_id=evaluation_id,
                producer="module5",
            )
        if not success:
            self.record_failure(
                module="module5",
                failure_type=status,
                failure_stage=self._failure_stage(data),
                context={
                    "failed_preconditions": data.get("failed_preconditions", []),
                    "notes": data.get("notes", []),
                    "dc_attempt_errors": data.get("dc_attempt_errors", []),
                },
                parent_id=evaluation_id,
                retries_needed=int(data.get("dc_retry_count") or 0),
                source_path=work_dir,
            )
        self.emit(
            "module5",
            "evaluation_completed" if success else "evaluation_failed",
            {"evaluation_id": evaluation_id, "status": status, "reward": reward},
            status="completed" if success else "failed",
        )
        return evaluation_id

    def record_direct_rtl_result(self, result: Any) -> str:
        """Record a specification-direct RTL result without inventing an action."""
        data = _as_dict(result)
        metrics = dict(data.get("metrics") or {})
        status = str(data.get("status") or "unknown")
        success = status == "success"
        evaluation_id = stable_id("evaluation", self.context.run_id, "spec_direct_rtl")
        self.agent.record_evaluation(
            EvaluationRecord(
                record_id=evaluation_id,
                task_id=self.context.task_id,
                run_id=self.context.run_id,
                parent_id=stable_id("selection", self.context.run_id),
                producer="spec_direct_rtl",
                benchmark=self.context.benchmark,
                objective=self.context.objective,
                path=self.context.path,
                correctness_status=self._correctness_status(data),
                synthesis_status=(
                    "success"
                    if str(data.get("dc_status") or "") == "ok"
                    else "failed" if status == "dc_failed" else "not_run"
                ),
                area=metrics.get("total_cell_area"),
                data_arrival_time_ps=metrics.get("delay_ps", metrics.get("data_arrival_time_ps")),
                slack_ps=metrics.get("slack_ps"),
                slack_status=str(metrics.get("slack_status") or ""),
                token_usage=dict(data.get("llm_token_totals") or {}),
                failure_reason="" if success else str(data.get("error") or status),
                metadata={
                    "route": data.get("route", "rtl_direct"),
                    "backend": data.get("backend", "spec_to_rtl"),
                    "model": data.get("model", ""),
                    "dc_retry_count": data.get("dc_retry_count", 0),
                    **self._verification_metadata(data),
                },
                source_path=str(data.get("work_dir") or ""),
            )
        )

        for artifact_type, raw_path in (
            ("spec_direct_input", data.get("input_manifest_path")),
            ("generated_rtl", data.get("generated_verilog_path")),
            ("spec_direct_result", data.get("result_path")),
        ):
            if raw_path and Path(str(raw_path)).is_file():
                self.record_artifact(
                    artifact_type,
                    str(raw_path),
                    parent_id=evaluation_id,
                    producer="spec_direct_rtl",
                )

        if not success:
            self.record_failure(
                module="spec_direct_rtl",
                failure_type=status,
                failure_stage=str(data.get("failure_stage") or "execution"),
                context={
                    "error": data.get("error", ""),
                    "dc_attempt_errors": data.get("dc_attempt_errors", []),
                },
                parent_id=evaluation_id,
                retries_needed=int(data.get("dc_retry_count") or 0),
                source_path=str(data.get("work_dir") or ""),
            )
        self.emit(
            "spec_direct_rtl",
            "evaluation_completed" if success else "evaluation_failed",
            {"evaluation_id": evaluation_id, "status": status},
            status="completed" if success else "failed",
        )
        return evaluation_id

    @staticmethod
    def _area_gain_pct(metrics: Dict[str, Any]) -> Optional[float]:
        area = metrics.get("total_cell_area")
        baseline = metrics.get("baseline_total_cell_area")
        if area is None or baseline in (None, 0):
            return None
        return (float(baseline) - float(area)) / float(baseline) * 100.0

    @staticmethod
    def _timing_gain_ps(metrics: Dict[str, Any]) -> Optional[float]:
        arrival = metrics.get("delay_ps", metrics.get("data_arrival_time_ps"))
        baseline = metrics.get(
            "baseline_delay_ps", metrics.get("baseline_data_arrival_time_ps")
        )
        if arrival is None or baseline is None:
            return None
        return float(baseline) - float(arrival)

    @staticmethod
    def _verification_metadata(data: Dict[str, Any]) -> Dict[str, str]:
        """Preserve verification evidence before Memory normalises correctness.

        ``_correctness_status`` intentionally retains the historical policy that
        a successful but unverified run is stored as ``passed``.  That value is
        therefore not suitable evidence for C5.  Keep the original verification
        mode and equivalence verdict in metadata so retrieval can apply a strict
        JasperGold-only gate without a database migration.

        The nested pre-DC result is authoritative when present.  If it is absent,
        the result-level ``behavior_check_status`` is the only raw equivalence
        signal available.  Missing values stay empty and are rejected by C5.
        """
        pre_dc = data.get("pre_dc_verification")
        if not isinstance(pre_dc, dict):
            pre_dc = {}

        verification_mode = str(
            data.get("verification_mode")
            or pre_dc.get("verification_mode")
            or ""
        ).strip().lower()

        pre_dc_status = ""
        for key in ("equivalence_status", "status", "behavior_check_status"):
            value = pre_dc.get(key)
            if value is not None and str(value).strip():
                pre_dc_status = str(value).strip().lower()
                break

        behavior_status = str(data.get("behavior_check_status") or "").strip().lower()
        # A present pre-DC record, including an explicit not_run/error, must not
        # be overwritten by the post-hoc correctness field or by a fallback.
        equivalence_status = pre_dc_status or behavior_status

        return {
            "verification_mode": verification_mode,
            "equivalence_status": equivalence_status,
            "pre_dc_verification_status": pre_dc_status,
            "behavior_check_status": behavior_status,
        }

    @staticmethod
    def _correctness_status(data: Dict[str, Any]) -> str:
        explicit = str(data.get("correctness_status") or "").lower()
        if explicit in {"passed", "success", "ok"}:
            return "passed"
        if explicit in {"failed", "error", "mismatch"}:
            return "failed"
        behavior = str(data.get("behavior_check_status") or "not_run").lower()
        if behavior in {"passed", "success", "ok"}:
            return "passed"
        if behavior in {"failed", "error", "mismatch"}:
            return "failed"
        # Project policy requested on 2026-08-10: unknown/not-run
        # correctness is assumed to be passed for Memory use.
        return "passed"

    @staticmethod
    def _failure_stage(data: Dict[str, Any]) -> str:
        for field in (
            "dc_status",
            "rtl_generation_status",
            "hls_status",
            "compile_status",
            "syntax_status",
        ):
            value = str(data.get(field) or "")
            if value and value not in {"ok", "success"}:
                return field.removesuffix("_status")
        return "execution"
