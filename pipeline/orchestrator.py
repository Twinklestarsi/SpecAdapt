"""Orchestrate Modules 1-5 with automatic Memory Agent synchronization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from c_gen.generator import CGenerator
from memory_agent import MemoryAgent, MemorySession
from module5.executor import execute_action, execute_action_sequence
from path_select.selector import PathDecision, PathSelector
from pipeline.rtl_direct_executor import execute_spec_direct_rtl
from project_paths import PROJECT_ROOT
from rag_retrieve.mcts_planner import build_module45_plan
from rag_retrieve.defaults import DEFAULT_HISTORICAL_REGION_INDEX, DEFAULT_TRANSFORM_BLACKLIST
from rag_retrieve.runtime_pipeline import (
    build_runtime_transform_plan,
)
from spec_analyze.analyzer import Analyzer
from spec_analyze.schema import FeatureResult


@dataclass
class PipelineResult:
    task_id: str
    run_id: str
    benchmark: str
    objective: str
    experiment_group: str = "optimized"
    path: str = ""
    status: str = "pending"
    feature_result: Dict[str, Any] = field(default_factory=dict)
    path_decision: Dict[str, Any] = field(default_factory=dict)
    direct_rtl_result: Dict[str, Any] = field(default_factory=dict)
    c_generation: Dict[str, Any] = field(default_factory=dict)
    module4_plan_path: str = ""
    module45_plan_path: str = ""
    module45_feedback_plan_path: str = ""
    module5_result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PipelineOrchestrator:
    def __init__(
        self,
        *,
        memory_store_path: str | Path = PROJECT_ROOT / "path_decisions_log.json",
        memory_db_path: str | Path | None = PROJECT_ROOT / "memory_agent.db",
        output_root: str | Path = PROJECT_ROOT / "pipeline_runs",
        c_output_dir: str | Path = PROJECT_ROOT / "c_gen_output",
        historical_region_index_path: str | Path = DEFAULT_HISTORICAL_REGION_INDEX,
        blacklist_path: str | Path = DEFAULT_TRANSFORM_BLACKLIST,
        env_path: str | Path = PROJECT_ROOT / ".env",
    ) -> None:
        self.memory = MemoryAgent(memory_store_path, db_path=memory_db_path)
        self.output_root = Path(output_root)
        self.c_output_dir = Path(c_output_dir)
        self.historical_region_index_path = Path(historical_region_index_path)
        self.blacklist_path = Path(blacklist_path)
        self.env_path = Path(env_path)

    def run(
        self,
        *,
        benchmark: str,
        objective: str,
        verilog_path: str | Path | None = None,
        spec_text: str = "",
        use_llm_path_selection: bool = True,
        module4_top_k: int = 5,
        mcts_iterations: int = 240,
        mcts_max_depth: int = 5,
        mcts_seed: int = 7,
        mcts_candidate_limit: int = 48,
        module5_backend: str = "direct_rtl",
        execute_best_path: bool = True,
        max_actions: Optional[int] = None,
        run_baseline: bool = True,
        module5_output_root: str | Path | None = None,
        rtl_max_retries: int = 2,
        dc_max_retries: int = 0,
        experiment_group: str = "optimized",
        forced_path: str = "auto",
        feature_override: FeatureResult | Dict[str, Any] | None = None,
        pre_dc_golden_rtl_path: str | Path = "",
        pre_dc_golden_top: str = "",
        pre_dc_design_type: str = "combinational",
        pre_dc_verification_timeout: int = 180,
        verification_mode: str = "none",
        jg_max_retries: int = 1,
    ) -> PipelineResult:
        if experiment_group != "optimized":
            raise ValueError(f"Unsupported experiment group: {experiment_group}")
        if module5_backend not in {"direct_rtl", "hls"}:
            raise ValueError(
                "Unsupported Module 5 backend: "
                f"{module5_backend}. Choose direct_rtl or hls."
            )
        if mcts_iterations < 1 or mcts_max_depth < 1 or mcts_candidate_limit < 1:
            raise ValueError("MCTS iterations, depth, and candidate limit must be positive")
        forced_path = str(forced_path or "auto").strip().lower()
        if forced_path not in {"auto", "c_first", "rtl_direct"}:
            raise ValueError(f"Unsupported forced_path: {forced_path}")
        verification_mode = str(verification_mode or "none").strip().lower()
        if verification_mode not in {"jaspergold", "none"}:
            raise ValueError(
                f"Unsupported verification_mode: {verification_mode}"
            )
        if jg_max_retries < 0:
            raise ValueError("jg_max_retries must be non-negative")
        if verification_mode == "jaspergold" and not pre_dc_golden_rtl_path:
            raise ValueError(
                "jaspergold verification requires pre_dc_golden_rtl_path"
            )
        objective = objective.upper()
        source_path = str(Path(verilog_path).resolve()) if verilog_path else ""
        session = MemorySession.start(
            self.memory,
            benchmark=benchmark,
            objective=objective,
            spec_text=spec_text,
            source_path=source_path,
        )
        result = PipelineResult(
            task_id=session.context.task_id,
            run_id=session.context.run_id,
            benchmark=benchmark,
            objective=objective,
            experiment_group=experiment_group,
        )
        run_dir = self.output_root / session.context.run_id

        try:
            if feature_override is None:
                feature = self._run_module1(
                    session=session,
                    benchmark=benchmark,
                    objective=objective,
                    verilog_path=verilog_path,
                    spec_text=spec_text,
                )
            else:
                feature = self._coerce_feature_override(feature_override)
                if feature.benchmark != benchmark:
                    raise ValueError(
                        "Frozen feature benchmark does not match the run benchmark"
                    )
                if feature.optimization_target.upper() != objective:
                    raise ValueError(
                        "Frozen feature objective does not match the run objective"
                    )
                if spec_text and feature.spec_text and feature.spec_text != spec_text:
                    raise ValueError("Frozen feature spec text does not match run input")
                session.record_feature_result(feature)
                session.emit(
                    "module1",
                    "frozen_feature_reused",
                    {"benchmark": benchmark, "objective": objective},
                )
            result.feature_result = feature.to_dict()

            if forced_path != "auto":
                decision = PathDecision(
                    benchmark=benchmark,
                    optimization_target=objective,
                    path=forced_path,
                    rule_fired=f"forced_{forced_path}",
                    confidence="high",
                    reason=(
                        "Path forced by the paired-path experiment controller; "
                        "adaptive routing was bypassed."
                    ),
                    tier=0,
                )
                session.record_path_decision(decision, feature.llm_features)
            else:
                selector = PathSelector(
                    env_path=self.env_path,
                    memory_session=session,
                )
                decision = selector.select(feature, use_llm=use_llm_path_selection)

            result.path = decision.path
            result.path_decision = decision.to_dict()

            if decision.path == "rtl_direct":
                direct_execution = execute_spec_direct_rtl(
                    benchmark=benchmark,
                    objective=objective,
                    spec_text=feature.spec_text or spec_text,
                    llm_features=dict(feature.llm_features or {}),
                    output_root=run_dir / "rtl_direct",
                    env_path=self.env_path,
                    rtl_max_retries=rtl_max_retries,
                    dc_max_retries=dc_max_retries,
                    pre_dc_golden_rtl_path=pre_dc_golden_rtl_path,
                    pre_dc_golden_top=pre_dc_golden_top,
                    pre_dc_design_type=pre_dc_design_type,
                    pre_dc_verification_timeout=pre_dc_verification_timeout,
                    verification_mode=verification_mode,
                    jg_max_retries=jg_max_retries,
                )
                result.direct_rtl_result = direct_execution.to_dict()
                result.status = direct_execution.status
                session.record_direct_rtl_result(direct_execution)
                session.complete(
                    "completed" if result.status == "success" else "failed"
                )
                return result

            generator = CGenerator(
                output_dir=self.c_output_dir,
                env_path=self.env_path,
                memory_session=session,
                inject_interface_pragmas=False,
            )
            c_result = generator.generate(
                benchmark=feature.benchmark,
                input_type=feature.input_type,
                verilog_path=feature.verilog_path,
                spec_text=feature.spec_text,
                features=feature.llm_features,
            )
            result.c_generation = c_result.to_dict()
            if not c_result.success:
                result.status = "c_generation_failed"
                session.complete("failed")
                return result

            module4_plan = build_runtime_transform_plan(
                c_path=c_result.c_path,
                benchmark=benchmark,
                objective=objective,
                historical_region_index_path=self.historical_region_index_path,
                blacklist_path=self.blacklist_path,
                top_k_regions=module4_top_k,
                memory_session=session,
            )
            module4_path = self._write_json(
                run_dir / "module4_plan.json", module4_plan
            )
            result.module4_plan_path = str(module4_path)
            session.record_artifact(
                "module4_plan", module4_path, producer="rag_retrieve"
            )

            # The cycle-exact contract is a hard constraint for the planner:
            # transforms that trade combinational logic for extra cycles cannot
            # pass equivalence checking once output latency is pinned.
            behavioral_contract = dict(feature.llm_features or {}).get(
                "behavioral_contract"
            )
            module45_plan = build_module45_plan(
                module4_plan,
                iterations=mcts_iterations,
                max_depth=mcts_max_depth,
                seed=mcts_seed,
                candidate_limit=mcts_candidate_limit,
                memory_session=session,
                behavioral_contract=behavioral_contract
                if isinstance(behavioral_contract, dict)
                else None,
            )
            module45_path = self._write_json(
                run_dir / "module45_plan.json", module45_plan
            )
            result.module45_plan_path = str(module45_path)
            session.record_artifact(
                "module45_plan", module45_path, producer="mcts_planner"
            )

            actions = list(module45_plan.get("best_actions", []))
            if max_actions is not None and max_actions > 0:
                actions = actions[:max_actions]
            if not actions:
                session.record_failure(
                    module="module4_5",
                    failure_type="empty_action_plan",
                    failure_stage="planning",
                    context={
                        "action_space_size": len(
                            module45_plan.get("action_space", [])
                        )
                    },
                    source_path=str(module45_path),
                )
                result.status = "planning_failed"
                session.complete("failed")
                return result

            typed_actions = [self._action_from_dict(action) for action in actions]
            module5_root = (
                Path(module5_output_root)
                if module5_output_root is not None
                else run_dir / "module5"
            )
            common = {
                "output_root": module5_root,
                "run_baseline": run_baseline,
                "env_path": self.env_path,
                "backend": module5_backend,
                "rtl_max_retries": rtl_max_retries,
                "dc_max_retries": dc_max_retries,
                "golden_rtl_path": pre_dc_golden_rtl_path,
                # Legacy one-way alias: module5._effective_verification_mode only
                # reads it to force "jaspergold" ON, never to turn it off. Kept in
                # sync with verification_mode so it cannot read as a kill switch.
                "enable_pre_dc_equivalence": verification_mode == "jaspergold",
                "pre_dc_golden_top": pre_dc_golden_top,
                "pre_dc_design_type": pre_dc_design_type,
                "pre_dc_verification_timeout": pre_dc_verification_timeout,
                "verification_mode": verification_mode,
                "jg_max_retries": jg_max_retries,
                "memory_session": session,
                "spec_context_override": {
                    "benchmark": feature.benchmark,
                    "spec_text": feature.spec_text or "",
                    "optimization_target": feature.optimization_target,
                    "llm_features": dict(feature.llm_features or {}),
                    "input_type": feature.input_type,
                    "confidence": dict(feature.confidence or {}),
                    "overall_confidence": feature.overall_confidence,
                    "verilog_path": feature.verilog_path or "",
                },
            }
            if execute_best_path and len(typed_actions) > 1:
                execution = execute_action_sequence(typed_actions, **common)
            else:
                execution = execute_action(typed_actions[0], **common)
            result.module5_result = execution.to_dict()
            result.status = (
                "success" if execution.status == "success" else execution.status
            )
            # Deliberately stop after executing the originally selected action
            # chain.  There is no post-execution replan and no fallback action.
            session.complete(
                "completed" if result.status == "success" else "failed"
            )
            return result
        except Exception as exc:
            session.record_failure(
                module="pipeline",
                failure_type=type(exc).__name__,
                failure_stage="orchestration",
                context={"message": str(exc)},
                source_path=source_path,
            )
            session.emit(
                "pipeline",
                "run_exception",
                {"type": type(exc).__name__, "message": str(exc)},
                status="failed",
            )
            session.complete("failed")
            raise

    def _run_module1(
        self,
        *,
        session: MemorySession,
        benchmark: str,
        objective: str,
        verilog_path: str | Path | None,
        spec_text: str,
    ) -> Any:
        analyzer = Analyzer(
            env_path=str(self.env_path),
            memory_session=session,
        )
        if verilog_path is not None:
            return analyzer.analyze_file(
                str(verilog_path),
                spec_text=spec_text or None,
                benchmark_name=benchmark,
                optimization_target=objective,
            )
        if not spec_text:
            raise ValueError("Either verilog_path or spec_text is required")
        return analyzer.analyze_spec(
            spec_text,
            benchmark_name=benchmark,
            optimization_target=objective,
        )

    @staticmethod
    def _coerce_feature_override(
        value: FeatureResult | Dict[str, Any],
    ) -> FeatureResult:
        if isinstance(value, FeatureResult):
            return value
        allowed = {
            "benchmark",
            "input_type",
            "optimization_target",
            "verilog_path",
            "spec_text",
            "llm_features",
            "confidence",
            "overall_confidence",
            "regex_features",
            "token_usage",
            "llm_raw",
        }
        return FeatureResult(**{key: value[key] for key in allowed if key in value})

    @staticmethod
    def _action_from_dict(data: Dict[str, Any]) -> Any:
        from module5.io_utils import action_from_dict

        return action_from_dict(data)

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path
