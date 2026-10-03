from __future__ import annotations

import json
import shutil
import re
from pathlib import Path
from typing import Any, Dict, List

from project_paths import PROJECT_ROOT
from rag_retrieve.schema import Module5Action, Module5ActionConstraints

from module5.dc_runner import run_dc_for_verilog
from module5.editor import apply_action_edit, apply_multi_action_edit
from module5.hls_runner import run_hls
from module5.io_utils import action_to_work_key, ensure_clean_dir, read_json, write_json
from module5.jg_verifier import verify_rtl_with_jg_retry
from module5.result_schema import Module5ExecutionResult, Module5Metrics
from module5.rtl_direct_runner import generate_direct_rtl
from module5.reward import build_objective_feedback, populate_deltas
from module5.spec_loader import build_spec_context
from module5.token_usage import merge_token_usage, token_totals
from module5.validators import summarize_failed_checks, validate_c_file


_HLS_BACKENDS = {"hls", "hls_control"}


def _is_hls_backend(backend: str) -> bool:
    return backend in _HLS_BACKENDS


def _jg_retry_budget(backend: str, requested: int) -> int:
    """Return the formal retry budget for a backend.

    HLS produces a canonical RTL artifact that must be checked as emitted.
    Keeping its budget at zero prevents JasperGold feedback from sending that
    artifact through an LLM rewrite.  The direct RTL route retains its caller's
    retry budget.
    """
    return 0 if _is_hls_backend(backend) else requested


def _effective_verification_mode(
    verification_mode: str,
    *,
    enable_jg_verification: bool = False,
    enable_pre_dc_equivalence: bool = False,
) -> str:
    """Normalize the new mode while preserving old formal-mode aliases."""
    mode = str(verification_mode or "none").strip().lower()
    if enable_jg_verification or enable_pre_dc_equivalence:
        mode = "jaspergold"
    if mode not in {"jaspergold", "none"}:
        raise ValueError(f"Unsupported verification_mode: {mode}")
    return mode


def _constraint_summary(action: Module5Action) -> list[str]:
    constraints = [
        "must_preserve_behavior" if action.constraints.must_preserve_behavior else "",
        f"apply_scope:{action.constraints.apply_scope}",
        f"expected_risk:{action.constraints.expected_risk}",
    ]
    constraints.extend(f"forbidden:{name}" for name in action.constraints.forbidden_transforms)
    return [item for item in constraints if item]


def _copy_text_file(src: str | Path, dst: str | Path) -> Path:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _baseline_cache_dir(output_root: str | Path, benchmark: str, backend: str) -> Path:
    return Path(output_root) / benchmark / f"__baseline_{backend}__"


def _load_cached_baseline(cache_dir: Path) -> Dict[str, Any] | None:
    cache_json = cache_dir / "baseline_result.json"
    if not cache_json.is_file():
        return None
    try:
        payload = read_json(cache_json)
    except Exception:
        return None
    if payload.get("status") != "success":
        return None
    return payload


def _store_cached_baseline(cache_dir: Path, baseline_eval: Dict[str, Any]) -> None:
    payload = {
        "status": baseline_eval.get("status", ""),
        "artifact": baseline_eval.get("artifact", {}),
        "metrics": baseline_eval.get("metrics", {}),
        "validator_summary": baseline_eval.get("validator_summary", {}),
        "failed_preconditions": baseline_eval.get("failed_preconditions", []),
        "dc": baseline_eval.get("dc", {}),
    }
    write_json(cache_dir / "baseline_result.json", payload)


def _update_result_token_usage(
    result: Module5ExecutionResult,
    *groups: List[Dict[str, Any]] | None,
) -> None:
    result.llm_token_usage = merge_token_usage(result.llm_token_usage, *groups)
    result.llm_token_totals = token_totals(result.llm_token_usage)


def _finalize_result_status(result: Module5ExecutionResult) -> None:
    """Normalize cross-backend status fields before persistence."""

    if result.status == "success":
        result.execution_status = "completed"
    elif result.status == "pending":
        result.execution_status = "pending"
    else:
        result.execution_status = "failed"

    if result.dc_status == "ok":
        result.synthesis_status = "success"
    elif result.dc_status and result.dc_status not in {"not_run", "pending"}:
        result.synthesis_status = "failed"

    if result.status == "success" and result.correctness_status in {
        "", "unknown", "not_run", "unknow"
    }:
        # Project policy: successful unverified outcomes are treated as correct.
        result.correctness_status = "passed"


def _apply_artifact_correctness(
    result: Module5ExecutionResult, artifact: Dict[str, Any]
) -> None:
    if artifact.get("verified") is True:
        result.behavior_check_status = "passed"
        result.correctness_status = "passed"
    elif artifact.get("status") == "verification_failed":
        result.behavior_check_status = "failed"
        result.correctness_status = "failed"


def _artifact_model(artifact: Dict[str, Any]) -> str:
    """Recover the generating model from artifact metadata or token records."""
    direct = str(artifact.get("model") or "").strip()
    if direct:
        return direct
    for usage in artifact.get("llm_token_usage", []) or []:
        model = str((usage or {}).get("model") or "").strip()
        if model:
            return model
    return ""


def _infer_c_top_name(c_path: Path, preferred: str = "") -> str:
    source = c_path.read_text(encoding="utf-8", errors="replace")
    if preferred and re.search(rf"\bvoid\s+{re.escape(preferred)}\s*\(", source):
        return preferred
    match = re.search(r"\bvoid\s+(?!main\b)([A-Za-z_]\w*)\s*\(", source)
    return match.group(1) if match else preferred


def _evaluate_candidate(
    action: Module5Action,
    c_path: Path,
    *,
    candidate_name: str,
    work_dir: Path,
    backend: str,
    env_path: str | Path,
    spec_context: Dict[str, Any] | None = None,
    enable_jg_verification: bool = False,
    golden_rtl_path: str | Path = "",
    jg_max_retries: int = 1,
    rtl_max_retries: int = 2,
    dc_max_retries: int = 0,
    enable_pre_dc_equivalence: bool = False,
    pre_dc_golden_top: str = "",
    pre_dc_design_type: str = "combinational",
    pre_dc_verification_timeout: int = 180,
    verification_mode: str = "none",
) -> Dict[str, Any]:
    # Backward-compatible aliases now select the one supported formal mode;
    # Yosys is no longer an active pre-DC backend in this executor.
    verification_mode = _effective_verification_mode(
        verification_mode,
        enable_jg_verification=enable_jg_verification,
        enable_pre_dc_equivalence=enable_pre_dc_equivalence,
    )
    if jg_max_retries < 0:
        raise ValueError("jg_max_retries must be non-negative")
    if verification_mode == "jaspergold" and not golden_rtl_path:
        raise ValueError("jaspergold verification requires a golden RTL path")
    top_name = str((action.source_anchor or {}).get("function", ""))
    validation = validate_c_file(c_path, expected_function=top_name)
    failed = summarize_failed_checks(validation)
    if failed:
        return {
            "status": "validate_failed",
            "validator_summary": validation,
            "failed_preconditions": failed,
        }

    if _is_hls_backend(backend):
        hls_dir = work_dir / f"{candidate_name}_hls"
        hls_features = dict((spec_context or {}).get("llm_features", {}) or {})
        artifact = run_hls(
            c_path,
            hls_dir,
            top_name=top_name,
            interface_spec=dict(hls_features.get("interface", {}) or {}),
            public_top_name=str(hls_features.get("module_name", "") or top_name),
            rtl_cleanup="interface_only" if backend == "hls_control" else "canonical",
        )
        if not artifact["success"]:
            return {
                "status": "hls_failed",
                "validator_summary": validation,
                "failed_preconditions": [],
                "artifact": artifact,
            }
        if backend == "hls":
            generated_path = Path(str(artifact.get("generated_verilog_path", "")))
            generated_files = [
                Path(str(path))
                for path in (artifact.get("generated_verilog_files") or [])
                if str(path)
            ]
            if (
                not generated_path.is_file()
                or len(generated_files) != 1
                or generated_files[0].resolve() != generated_path.resolve()
            ):
                artifact["verified"] = False
                return {
                    "status": "hls_failed",
                    "validator_summary": validation,
                    "failed_preconditions": ["hls_noncanonical_output"],
                    "artifact": artifact,
                }
    elif backend == "direct_rtl":
        spec_context = spec_context or {}
        rtl_dir = work_dir / f"{candidate_name}_rtl"
        artifact = generate_direct_rtl(
            action,
            c_path=c_path,
            out_dir=rtl_dir,
            spec_text=str(spec_context.get("spec_text", "")),
            llm_features=dict(spec_context.get("llm_features", {}) or {}),
            env_path=env_path,
            top_name=top_name,
            max_retries=rtl_max_retries,
            enable_jg_verification=False,
            golden_rtl_path=golden_rtl_path,
            jg_max_retries=jg_max_retries,
        )
        if not artifact["success"]:
            return {
                "status": "rtl_failed",
                "validator_summary": validation,
                "failed_preconditions": [],
                "artifact": artifact,
                "llm_token_usage": artifact.get("llm_token_usage", []),
            }
    else:
        return {
            "status": "backend_failed",
            "validator_summary": validation,
            "failed_preconditions": [f"unsupported_backend:{backend}"],
        }

    dc_stem = f"{action.benchmark}_{candidate_name}"
    dc_attempt_errors: List[Dict[str, Any]] = []
    dc_retry_count = 0
    pre_dc_verification: Dict[str, Any] = {
        "status": "not_run",
        "equivalent": None,
        "verified": False,
        "method": "none",
        "verification_mode": "none",
        "reason": "syntax_then_dc_without_equivalence",
    }
    jg_retry_count = 0
    jg_attempts: List[Dict[str, Any]] = []
    eval_token_usage = list(artifact.get("llm_token_usage", [])) if isinstance(artifact, dict) else []
    for dc_attempt in range(dc_max_retries + 1):
        if verification_mode == "jaspergold":
            source_context = (
                "Specification:\n"
                + str((spec_context or {}).get("spec_text", ""))
                + "\n\nFrozen structured feature contract:\n"
                + json.dumps(
                    dict((spec_context or {}).get("llm_features", {}) or {}),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n\nOptimized C source:\n"
                + c_path.read_text(encoding="utf-8", errors="replace")
            )
            pre_dc_verification = verify_rtl_with_jg_retry(
                candidate_path=artifact["generated_verilog_path"],
                golden_path=golden_rtl_path,
                output_dir=(
                    work_dir
                    / f"{candidate_name}_pre_dc_verification"
                    / f"attempt_{dc_attempt}"
                ),
                env_path=env_path,
                source_label="specification, frozen feature contract, and optimized C",
                source_context=source_context,
                module_name=str(
                    artifact.get("public_top")
                    or artifact.get("module_name")
                    or ""
                ),
                golden_top=pre_dc_golden_top,
                design_type=pre_dc_design_type,
                verification_timeout=pre_dc_verification_timeout,
                max_retries=_jg_retry_budget(backend, jg_max_retries),
            )
            pre_dc_verification["verified"] = (
                str(pre_dc_verification.get("status") or "") == "passed"
            )
            jg_retry_count += int(pre_dc_verification.get("retry_count", 0) or 0)
            jg_attempts.extend(list(pre_dc_verification.get("attempts") or []))
            eval_token_usage = merge_token_usage(
                eval_token_usage,
                pre_dc_verification.get("llm_token_usage", []),
            )
            final_candidate = str(
                pre_dc_verification.get("final_candidate_path") or ""
            )
            if final_candidate:
                artifact["generated_verilog_path"] = final_candidate
            if str(pre_dc_verification.get("status") or "") != "passed":
                artifact["verified"] = False
                artifact["jg_verification"] = pre_dc_verification
                return {
                    "status": "equivalence_failed",
                    "validator_summary": validation,
                    "failed_preconditions": [],
                    "artifact": artifact,
                    "pre_dc_verification": pre_dc_verification,
                    "dc_attempt_errors": dc_attempt_errors,
                    "dc_retry_count": dc_retry_count,
                    "jg_retry_count": jg_retry_count,
                    "jg_attempts": jg_attempts,
                    "llm_token_usage": eval_token_usage,
                }
            artifact["verified"] = True
            artifact["jg_verification"] = pre_dc_verification

        dc = run_dc_for_verilog(
            artifact["generated_verilog_path"],
            benchmark=action.benchmark,
            goal=action.objective,
            stem=dc_stem,
            top_module=str(artifact.get("public_top", "")),
        )
        metrics = dc.get("metrics", {})
        if dc.get("success", False):
            break

        dc_error = str(dc.get("error", "") or "Design Compiler failed")
        dc_attempt_errors.append(
            {
                "attempt": dc_attempt,
                "error": dc_error,
                "report_dir": dc.get("report_dir", ""),
            }
        )
        if backend != "direct_rtl" or dc_attempt >= dc_max_retries:
            return {
                "status": "dc_failed",
                "validator_summary": validation,
                "failed_preconditions": [],
                "artifact": artifact,
                "dc": dc,
                "dc_attempt_errors": dc_attempt_errors,
                "dc_retry_count": dc_retry_count,
                "jg_retry_count": jg_retry_count,
                "jg_attempts": jg_attempts,
                "llm_token_usage": eval_token_usage,
                "pre_dc_verification": pre_dc_verification,
            }

        previous_rtl = ""
        generated_path = Path(str(artifact.get("generated_verilog_path", "")))
        if generated_path.is_file():
            previous_rtl = generated_path.read_text(encoding="utf-8", errors="replace")

        dc_retry_count += 1
        print(
            f"  [DC] {action.benchmark}/{candidate_name} retry {dc_retry_count}/{dc_max_retries} "
            "with Design Compiler feedback"
        )
        retry_dir = work_dir / f"{candidate_name}_rtl"
        artifact = generate_direct_rtl(
            action,
            c_path=c_path,
            out_dir=retry_dir,
            spec_text=str((spec_context or {}).get("spec_text", "")),
            llm_features=dict((spec_context or {}).get("llm_features", {}) or {}),
            env_path=env_path,
            top_name=top_name,
            max_retries=rtl_max_retries,
            initial_feedback=dc_error,
            previous_rtl_text=previous_rtl,
            attempt_label=f"dc_retry_{dc_attempt + 1}",
            enable_jg_verification=False,
            golden_rtl_path=golden_rtl_path,
            jg_max_retries=jg_max_retries,
        )
        if not artifact["success"]:
            eval_token_usage = merge_token_usage(eval_token_usage, artifact.get("llm_token_usage", []))
            return {
                "status": "rtl_failed",
                "validator_summary": validation,
                "failed_preconditions": [],
                "artifact": artifact,
                "dc": dc,
                "dc_attempt_errors": dc_attempt_errors,
                "dc_retry_count": dc_retry_count,
                "jg_retry_count": jg_retry_count,
                "jg_attempts": jg_attempts,
                "llm_token_usage": eval_token_usage,
            }
        eval_token_usage = merge_token_usage(eval_token_usage, artifact.get("llm_token_usage", []))

    return {
        "status": "success",
        "validator_summary": validation,
        "failed_preconditions": [],
        "artifact": artifact,
        "dc": dc,
        "metrics": metrics,
        "dc_attempt_errors": dc_attempt_errors,
        "dc_retry_count": dc_retry_count,
        "jg_retry_count": jg_retry_count,
        "jg_attempts": jg_attempts,
        "llm_token_usage": eval_token_usage,
        "pre_dc_verification": pre_dc_verification,
    }


def execute_unmodified_c(
    c_path: str | Path,
    *,
    benchmark: str,
    objective: str,
    output_root: str | Path = PROJECT_ROOT / "module5_runs",
    backend: str = "hls_control",
    spec_context_override: Dict[str, Any] | None = None,
    env_path: str | Path = ".env",
) -> Module5ExecutionResult:
    """Evaluate generated C without applying any Module 4/5 optimization action."""
    if backend not in _HLS_BACKENDS:
        raise ValueError("execute_unmodified_c supports only HLS backends")

    source_path = Path(c_path).resolve()
    preferred_top = benchmark.replace(" ", "_").replace("-", "_")
    top_name = _infer_c_top_name(source_path, preferred_top) if source_path.is_file() else preferred_top
    action = Module5Action(
        action_id=f"{benchmark}::ir_hls_control",
        benchmark=benchmark,
        objective=objective.upper(),
        source_c_path=str(source_path),
        region_id="whole_function",
        region_type="control_group",
        transform_name="NO_C_OPTIMIZATION",
        priority=0.0,
        planning_score=0.0,
        expected_metric_gain=0.0,
        confidence=1.0,
        problem_hypothesis="Measure the effect of C/C++ IR plus HLS only.",
        execution_hint="Do not modify the generated C source.",
        source_anchor={"function": top_name},
        constraints=Module5ActionConstraints(
            must_preserve_behavior=True,
            forbidden_transforms=["all_c_optimizations", "rtl_canonicalization"],
            apply_scope="none",
            expected_risk="low",
        ),
    )
    work_dir = ensure_clean_dir(Path(output_root) / benchmark / "ir_hls_control")
    result = Module5ExecutionResult(
        action_id=action.action_id,
        benchmark=benchmark,
        objective=objective.upper(),
        source_c_path=str(source_path),
        backend=backend,
        work_dir=str(work_dir),
        applied_transform_name=action.transform_name,
        changed_functions=[],
        changed_regions=[],
        preserved_constraints=_constraint_summary(action),
        execution_mode="unmodified_c_control",
        applied_action_ids=[],
        sequence_length=0,
    )

    if not source_path.is_file():
        result.status = "validate_failed"
        result.failed_preconditions.append("source_c_missing")
        write_json(work_dir / "result.json", result.to_dict())
        return result

    spec_context = dict(spec_context_override or {})
    interface = dict((spec_context.get("llm_features", {}) or {}).get("interface", {}) or {})
    if not interface.get("ports"):
        result.status = "hls_failed"
        result.failed_preconditions.append("interface_contract_missing")
        result.notes.append("The IR plus HLS control group requires a frozen exact interface manifest.")
        write_json(work_dir / "result.json", result.to_dict())
        return result

    source_snapshot = _copy_text_file(source_path, work_dir / "source.c")
    evaluation = _evaluate_candidate(
        action,
        source_snapshot,
        candidate_name="control",
        work_dir=work_dir,
        backend=backend,
        env_path=env_path,
        spec_context=spec_context,
        dc_max_retries=0,
    )
    result.validator_summary = dict(evaluation.get("validator_summary", {}) or {})
    result.failed_preconditions = list(evaluation.get("failed_preconditions", []))
    result.syntax_status = "ok" if result.validator_summary.get("syntax_ok") else "failed"
    result.compile_status = result.syntax_status
    artifact = dict(evaluation.get("artifact", {}) or {})
    dc = dict(evaluation.get("dc", {}) or {})
    result.hls_status = str(artifact.get("status", ""))
    result.dc_status = "ok" if dc.get("success") else (
        evaluation.get("status", "") if "dc" in evaluation else ""
    )
    result.generated_verilog_path = str(artifact.get("generated_verilog_path", ""))
    result.generated_verilog_files = list(artifact.get("generated_verilog_files", []))
    result.raw_generated_verilog_path = str(artifact.get("raw_generated_verilog_path", ""))
    result.hls_interface_policy = dict(artifact.get("interface_policy", {}) or {})
    result.hls_interface_normalization = dict(artifact.get("interface_normalization", {}) or {})
    result.hls_syntax_validation = dict(artifact.get("syntax_validation", {}) or {})

    if evaluation.get("status") == "success":
        metrics = dict(evaluation.get("metrics", {}) or {})
        result.metrics = Module5Metrics(
            total_cell_area=metrics.get("total_cell_area"),
            data_arrival_time_ps=metrics.get("delay_ps", metrics.get("data_arrival_time_ps")),
            delay_ps=metrics.get("delay_ps", metrics.get("data_arrival_time_ps")),
            slack_ps=metrics.get("slack_ps"),
            slack_status=str(metrics.get("slack_status", "")),
        )
        result.status = "success"
        result.hls_status = "ok"
        result.dc_status = "ok"
        result.followup_action_recommendation = "compare_with_direct_rtl_and_optimized_hls_groups"
    else:
        result.status = str(evaluation.get("status", "hls_failed"))
        if artifact.get("stderr"):
            result.notes.append(str(artifact["stderr"])[-1000:])
        if dc.get("error"):
            result.notes.append(str(dc["error"]))

    write_json(work_dir / "control_group.json", action.to_dict())
    write_json(work_dir / "result.json", result.to_dict())
    return result


def execute_action(
    action: Module5Action,
    *,
    output_root: str | Path = PROJECT_ROOT / "module5_runs",
    run_baseline: bool = True,
    env_path: str | Path = ".env",
    backend: str = "direct_rtl",
    spec_db_path: str | Path = PROJECT_ROOT / "spec_analysis.json",
    enable_jg_verification: bool = False,
    golden_rtl_path: str | Path = "",
    jg_max_retries: int = 1,
    rtl_max_retries: int = 2,
    dc_max_retries: int = 0,
    enable_pre_dc_equivalence: bool = False,
    pre_dc_golden_top: str = "",
    pre_dc_design_type: str = "combinational",
    pre_dc_verification_timeout: int = 180,
    verification_mode: str = "none",
    memory_session: Any = None,
    spec_context_override: Dict[str, Any] | None = None,
) -> Module5ExecutionResult:
    effective_verification_mode = _effective_verification_mode(
        verification_mode,
        enable_jg_verification=enable_jg_verification,
        enable_pre_dc_equivalence=enable_pre_dc_equivalence,
    )
    if jg_max_retries < 0:
        raise ValueError("jg_max_retries must be non-negative")
    if effective_verification_mode == "jaspergold" and not golden_rtl_path:
        raise ValueError("jaspergold verification requires a golden RTL path")
    if memory_session is not None:
        guidance = memory_session.failure_guidance(
            "module5",
            "execution",
            {
                "benchmark": action.benchmark,
                "transform_name": action.transform_name,
                "region_type": action.region_type,
                "backend": backend,
            },
        )
        fixes = [
            str(item.get("fix_applied") or "")
            for item in guidance
            if item.get("fix_succeeded") is True
            and item.get("fix_applied")
            and float(item.get("score", 0.0)) > 0.0
        ]
        if fixes:
            action.execution_hint = (
                f"{action.execution_hint}\nHistorical successful fixes: "
                + "; ".join(fixes[:3])
            ).strip()

    work_key = action_to_work_key(action.action_id)
    work_dir = ensure_clean_dir(Path(output_root) / action.benchmark / work_key)
    source_path = Path(action.source_c_path)
    result = Module5ExecutionResult(
        action_id=action.action_id,
        benchmark=action.benchmark,
        objective=action.objective,
        source_c_path=str(source_path),
        backend=backend,
        work_dir=str(work_dir),
        applied_transform_name=action.transform_name,
        changed_regions=[action.region_id],
        changed_functions=[str((action.source_anchor or {}).get("function", ""))] if action.source_anchor else [],
        preserved_constraints=_constraint_summary(action),
    )

    def finish() -> Module5ExecutionResult:
        _finalize_result_status(result)
        write_json(work_dir / "result.json", result.to_dict())
        if memory_session is not None:
            memory_session.record_module5_result(result, [action])
        return result

    if not source_path.exists():
        result.status = "edit_failed"
        result.failed_preconditions.append("source_c_missing")
        result.notes.append(f"Source C path does not exist: {source_path}")
        return finish()

    spec_context = dict(spec_context_override or {})
    if not spec_context and backend in {"direct_rtl", *_HLS_BACKENDS}:
        try:
            spec_context = build_spec_context(action.benchmark, spec_db_path=spec_db_path)
        except Exception as exc:
            result.status = "edit_failed"
            result.failed_preconditions.append("spec_context_missing")
            result.notes.append(str(exc))
            return finish()
    if _is_hls_backend(backend):
        interface = dict((spec_context.get("llm_features", {}) or {}).get("interface", {}) or {})
        if not interface.get("ports"):
            result.status = "hls_failed"
            result.failed_preconditions.append("interface_contract_missing")
            result.notes.append("HLS requires an exact port, clock, and reset contract from Module 1.")
            return finish()

    source_snapshot = _copy_text_file(source_path, work_dir / "source.c")
    write_json(work_dir / "action.json", action.to_dict())
    source_code = source_snapshot.read_text(encoding="utf-8", errors="replace")

    edited_code, meta, differs, error, edit_token_usage = apply_action_edit(
        action, source_code, env_path=env_path
    )
    if edited_code is None:
        result.status = "edit_failed"
        result.edit_summary = str(meta.get("summary", "edit failed"))
        result.failed_preconditions.append(error or "llm_edit_failed")
        _update_result_token_usage(result, edit_token_usage)
        return finish()

    edited_path = work_dir / "edited.c"
    edited_path.write_text(edited_code, encoding="utf-8")
    result.edited_c_path = str(edited_path)
    result.edit_summary = str(meta.get("summary", ""))
    if not differs:
        result.notes.append("LLM edit did not materially change the source code.")

    candidate_eval = _evaluate_candidate(
        action,
        edited_path,
        candidate_name="candidate",
        work_dir=work_dir,
        backend=backend,
        env_path=env_path,
        spec_context=spec_context,
        enable_jg_verification=enable_jg_verification,
        golden_rtl_path=golden_rtl_path,
        jg_max_retries=jg_max_retries,
        rtl_max_retries=rtl_max_retries,
        dc_max_retries=dc_max_retries,
        enable_pre_dc_equivalence=enable_pre_dc_equivalence,
        pre_dc_golden_top=pre_dc_golden_top,
        pre_dc_design_type=pre_dc_design_type,
        pre_dc_verification_timeout=pre_dc_verification_timeout,
        verification_mode=effective_verification_mode,
    )
    result.validator_summary = candidate_eval.get("validator_summary", {})
    result.failed_preconditions = list(candidate_eval.get("failed_preconditions", []))
    result.dc_retry_count = int(candidate_eval.get("dc_retry_count", 0) or 0)
    result.dc_attempt_errors = list(candidate_eval.get("dc_attempt_errors", []))
    result.pre_dc_verification = dict(candidate_eval.get("pre_dc_verification", {}) or {})
    result.verification_mode = effective_verification_mode
    result.jg_retry_count = int(candidate_eval.get("jg_retry_count", 0) or 0)
    result.jg_attempts = list(candidate_eval.get("jg_attempts", []) or [])
    if result.pre_dc_verification:
        verification_status = str(result.pre_dc_verification.get("status") or "error")
        result.behavior_check_status = (
            "passed" if verification_status == "passed"
            else "failed" if verification_status == "failed"
            else verification_status
        )
        result.correctness_status = (
            "passed" if verification_status == "passed"
            else "failed" if verification_status == "failed"
            else "unknown"
        )
    _update_result_token_usage(result, edit_token_usage, candidate_eval.get("llm_token_usage", []))
    result.syntax_status = "ok" if result.validator_summary.get("syntax_ok") else "failed"
    result.compile_status = result.syntax_status

    artifact = candidate_eval.get("artifact", {})
    result.model = _artifact_model(artifact)
    _apply_artifact_correctness(result, artifact)
    dc = candidate_eval.get("dc", {})
    if _is_hls_backend(backend):
        result.hls_status = artifact.get("status", "")
    else:
        result.rtl_generation_status = artifact.get("status", "")
        result.rtl_prompt_path = artifact.get("prompt_path", "")
        result.rtl_raw_response_path = artifact.get("raw_response_path", "")
    result.dc_status = "ok" if dc.get("success") else (candidate_eval["status"] if "dc" in candidate_eval else "")
    result.generated_verilog_path = artifact.get("generated_verilog_path", "")
    result.generated_verilog_files = list(artifact.get("generated_verilog_files", []))
    result.raw_generated_verilog_path = artifact.get("raw_generated_verilog_path", "")
    result.hls_interface_policy = dict(artifact.get("interface_policy", {}) or {})
    result.hls_interface_normalization = dict(artifact.get("interface_normalization", {}) or {})
    result.hls_syntax_validation = dict(artifact.get("syntax_validation", {}) or {})

    if candidate_eval["status"] != "success":
        result.status = candidate_eval["status"]
        result.followup_action_recommendation = "retry_with_different_action"
        if artifact.get("stderr"):
            result.notes.append(str(artifact["stderr"])[-1000:])
        if dc.get("error"):
            result.notes.append(str(dc["error"]))
        return finish()

    if result.dc_retry_count:
        result.notes.append(
            f"DC failed {len(result.dc_attempt_errors)} time(s); regenerated RTL "
            f"{result.dc_retry_count} time(s) using Design Compiler feedback."
        )

    metrics = Module5Metrics(
        total_cell_area=candidate_eval["metrics"].get("total_cell_area"),
        data_arrival_time_ps=candidate_eval["metrics"].get("delay_ps", candidate_eval["metrics"].get("data_arrival_time_ps")),
        delay_ps=candidate_eval["metrics"].get("delay_ps", candidate_eval["metrics"].get("data_arrival_time_ps")),
        slack_ps=candidate_eval["metrics"].get("slack_ps"),
        slack_status=str(candidate_eval["metrics"].get("slack_status", "")),
    )

    if run_baseline:
        baseline_cache = _baseline_cache_dir(output_root, action.benchmark, backend)
        cached_baseline = _load_cached_baseline(baseline_cache) if backend == "direct_rtl" else None

        if cached_baseline is not None:
            metrics.baseline_total_cell_area = cached_baseline.get("metrics", {}).get("total_cell_area")
            metrics.baseline_data_arrival_time_ps = cached_baseline.get("metrics", {}).get("delay_ps", cached_baseline.get("metrics", {}).get("data_arrival_time_ps"))
            metrics.baseline_delay_ps = cached_baseline.get("metrics", {}).get("delay_ps", metrics.baseline_data_arrival_time_ps)
            metrics.baseline_slack_ps = cached_baseline.get("metrics", {}).get("slack_ps")
            result.baseline_verilog_path = cached_baseline.get("artifact", {}).get("generated_verilog_path", "")
            result.notes.append(f"Reused cached baseline from {baseline_cache}")
        else:
            baseline_work_dir = baseline_cache if backend == "direct_rtl" else work_dir
            if backend == "direct_rtl":
                baseline_work_dir.mkdir(parents=True, exist_ok=True)
                _copy_text_file(source_snapshot, baseline_work_dir / "source.c")

            baseline_eval = _evaluate_candidate(
                action,
                source_snapshot,
                candidate_name="baseline",
                work_dir=baseline_work_dir,
                backend=backend,
                env_path=env_path,
                spec_context=spec_context,
                enable_jg_verification=False,  # Don't verify baseline
                golden_rtl_path=golden_rtl_path,
                jg_max_retries=jg_max_retries,
                rtl_max_retries=rtl_max_retries,
                dc_max_retries=dc_max_retries,
                enable_pre_dc_equivalence=False,
                pre_dc_golden_top=pre_dc_golden_top,
                pre_dc_design_type=pre_dc_design_type,
                pre_dc_verification_timeout=pre_dc_verification_timeout,
                verification_mode="none",
            )
            if baseline_eval["status"] == "success":
                metrics.baseline_total_cell_area = baseline_eval["metrics"].get("total_cell_area")
                metrics.baseline_data_arrival_time_ps = baseline_eval["metrics"].get("delay_ps", baseline_eval["metrics"].get("data_arrival_time_ps"))
                metrics.baseline_delay_ps = baseline_eval["metrics"].get("delay_ps", metrics.baseline_data_arrival_time_ps)
                metrics.baseline_slack_ps = baseline_eval["metrics"].get("slack_ps")
                _update_result_token_usage(result, baseline_eval.get("llm_token_usage", []))
                baseline_artifact = baseline_eval.get("artifact", {})
                result.baseline_verilog_path = baseline_artifact.get("generated_verilog_path", "")
                if backend == "direct_rtl":
                    _store_cached_baseline(baseline_cache, baseline_eval)
            else:
                result.notes.append(f"Baseline evaluation failed: {baseline_eval['status']}")

    metrics = populate_deltas(metrics)
    result.metrics = metrics
    result.objective_feedback = build_objective_feedback(action.objective, metrics)
    result.status = "success"
    if _is_hls_backend(backend):
        result.hls_status = "ok"
    else:
        result.rtl_generation_status = "ok"
    result.dc_status = "ok"
    result.followup_action_recommendation = "feed_reward_back_to_module45"

    return finish()


# ---------------------------------------------------------------------------
# Sequence (multi-action single-pass) executor
# ---------------------------------------------------------------------------

def _sequence_work_key(actions: List[Module5Action]) -> str:
    ids = "_".join(action_to_work_key(a.action_id) for a in actions)
    return f"seq_{ids}"[:200]


def execute_action_sequence(
    actions: List[Module5Action],
    *,
    output_root: str | Path = PROJECT_ROOT / "module5_runs",
    run_baseline: bool = True,
    env_path: str | Path = ".env",
    backend: str = "direct_rtl",
    spec_db_path: str | Path = PROJECT_ROOT / "spec_analysis.json",
    enable_jg_verification: bool = False,
    golden_rtl_path: str | Path = "",
    jg_max_retries: int = 1,
    rtl_max_retries: int = 2,
    dc_max_retries: int = 0,
    enable_pre_dc_equivalence: bool = False,
    pre_dc_golden_top: str = "",
    pre_dc_design_type: str = "combinational",
    pre_dc_verification_timeout: int = 180,
    verification_mode: str = "none",
    memory_session: Any = None,
    spec_context_override: Dict[str, Any] | None = None,
) -> Module5ExecutionResult:
    """Execute multiple actions in a single-pass LLM edit, then evaluate once."""
    if not actions:
        raise ValueError("actions list is empty")
    effective_verification_mode = _effective_verification_mode(
        verification_mode,
        enable_jg_verification=enable_jg_verification,
        enable_pre_dc_equivalence=enable_pre_dc_equivalence,
    )
    if jg_max_retries < 0:
        raise ValueError("jg_max_retries must be non-negative")
    if effective_verification_mode == "jaspergold" and not golden_rtl_path:
        raise ValueError("jaspergold verification requires a golden RTL path")

    if memory_session is not None:
        for action in actions:
            guidance = memory_session.failure_guidance(
                "module5",
                "execution",
                {
                    "benchmark": action.benchmark,
                    "transform_name": action.transform_name,
                    "region_type": action.region_type,
                    "backend": backend,
                },
            )
            fixes = [
                str(item.get("fix_applied") or "")
                for item in guidance
                if item.get("fix_succeeded") is True
                and item.get("fix_applied")
                and float(item.get("score", 0.0)) > 0.0
            ]
            if fixes:
                action.execution_hint = (
                    f"{action.execution_hint}\nHistorical successful fixes: "
                    + "; ".join(fixes[:3])
                ).strip()

    # Validate all actions share the same benchmark / objective / source
    benchmarks = {a.benchmark for a in actions}
    objectives = {a.objective for a in actions}
    sources = {a.source_c_path for a in actions}
    if len(benchmarks) > 1:
        raise ValueError(f"Actions span multiple benchmarks: {benchmarks}")
    if len(objectives) > 1:
        raise ValueError(f"Actions span multiple objectives: {objectives}")
    if len(sources) > 1:
        raise ValueError(f"Actions span multiple source files: {sources}")

    # Use first action as representative for evaluation metadata
    lead = actions[0]
    top_name = str((lead.source_anchor or {}).get("function", ""))
    work_key = _sequence_work_key(actions)
    work_dir = ensure_clean_dir(Path(output_root) / lead.benchmark / work_key)
    source_path = Path(lead.source_c_path)

    result = Module5ExecutionResult(
        action_id=lead.action_id,
        benchmark=lead.benchmark,
        objective=lead.objective,
        source_c_path=str(source_path),
        backend=backend,
        work_dir=str(work_dir),
        applied_transform_name=", ".join(a.transform_name for a in actions),
        changed_regions=[a.region_id for a in actions],
        changed_functions=list({
            str((a.source_anchor or {}).get("function", ""))
            for a in actions if a.source_anchor
        }),
        preserved_constraints=_constraint_summary(lead),
        execution_mode="action_sequence",
        applied_action_ids=[a.action_id for a in actions],
        sequence_length=len(actions),
    )

    def finish() -> Module5ExecutionResult:
        _finalize_result_status(result)
        write_json(work_dir / "result.json", result.to_dict())
        if memory_session is not None:
            memory_session.record_module5_result(result, actions)
        return result

    if not source_path.exists():
        result.status = "edit_failed"
        result.failed_preconditions.append("source_c_missing")
        result.notes.append(f"Source C path does not exist: {source_path}")
        return finish()

    spec_context = dict(spec_context_override or {})
    if not spec_context and backend in {"direct_rtl", *_HLS_BACKENDS}:
        try:
            spec_context = build_spec_context(lead.benchmark, spec_db_path=spec_db_path)
        except Exception as exc:
            result.status = "edit_failed"
            result.failed_preconditions.append("spec_context_missing")
            result.notes.append(str(exc))
            return finish()
    if _is_hls_backend(backend):
        interface = dict((spec_context.get("llm_features", {}) or {}).get("interface", {}) or {})
        if not interface.get("ports"):
            result.status = "hls_failed"
            result.failed_preconditions.append("interface_contract_missing")
            result.notes.append("HLS requires an exact port, clock, and reset contract from Module 1.")
            return finish()

    source_snapshot = _copy_text_file(source_path, work_dir / "source.c")
    write_json(work_dir / "sequence.json", [a.to_dict() for a in actions])
    source_code = source_snapshot.read_text(encoding="utf-8", errors="replace")

    # Single-pass multi-action edit
    edited_code, per_action_metas, differs, error, edit_token_usage = apply_multi_action_edit(
        actions, source_code, env_path=env_path,
    )
    result.per_action_summaries = per_action_metas

    if error:
        result.status = "edit_failed"
        result.edit_summary = error
        result.failed_preconditions.append(error)
        _update_result_token_usage(result, edit_token_usage)
        return finish()

    applied_count = sum(1 for m in per_action_metas if m.get("applied"))
    summaries = "; ".join(
        f"[{i}] {m.get('summary', '?')}" for i, m in enumerate(per_action_metas)
    )
    result.edit_summary = f"{applied_count}/{len(actions)} applied: {summaries}"

    edited_path = work_dir / "edited.c"
    edited_path.write_text(edited_code, encoding="utf-8")
    result.edited_c_path = str(edited_path)
    if not differs:
        result.notes.append("LLM edit did not materially change the source code.")

    # Evaluate the final edited C once
    candidate_eval = _evaluate_candidate(
        lead,
        edited_path,
        candidate_name="candidate",
        work_dir=work_dir,
        backend=backend,
        env_path=env_path,
        spec_context=spec_context,
        enable_jg_verification=enable_jg_verification,
        golden_rtl_path=golden_rtl_path,
        jg_max_retries=jg_max_retries,
        rtl_max_retries=rtl_max_retries,
        dc_max_retries=dc_max_retries,
        enable_pre_dc_equivalence=enable_pre_dc_equivalence,
        pre_dc_golden_top=pre_dc_golden_top,
        pre_dc_design_type=pre_dc_design_type,
        pre_dc_verification_timeout=pre_dc_verification_timeout,
        verification_mode=effective_verification_mode,
    )
    result.validator_summary = candidate_eval.get("validator_summary", {})
    result.failed_preconditions = list(candidate_eval.get("failed_preconditions", []))
    result.dc_retry_count = int(candidate_eval.get("dc_retry_count", 0) or 0)
    result.dc_attempt_errors = list(candidate_eval.get("dc_attempt_errors", []))
    result.pre_dc_verification = dict(candidate_eval.get("pre_dc_verification", {}) or {})
    result.verification_mode = effective_verification_mode
    result.jg_retry_count = int(candidate_eval.get("jg_retry_count", 0) or 0)
    result.jg_attempts = list(candidate_eval.get("jg_attempts", []) or [])
    if result.pre_dc_verification:
        verification_status = str(result.pre_dc_verification.get("status") or "error")
        result.behavior_check_status = (
            "passed" if verification_status == "passed"
            else "failed" if verification_status == "failed"
            else verification_status
        )
        result.correctness_status = (
            "passed" if verification_status == "passed"
            else "failed" if verification_status == "failed"
            else "unknown"
        )
    _update_result_token_usage(result, edit_token_usage, candidate_eval.get("llm_token_usage", []))
    result.syntax_status = "ok" if result.validator_summary.get("syntax_ok") else "failed"
    result.compile_status = result.syntax_status

    artifact = candidate_eval.get("artifact", {})
    result.model = _artifact_model(artifact)
    _apply_artifact_correctness(result, artifact)
    dc = candidate_eval.get("dc", {})
    if _is_hls_backend(backend):
        result.hls_status = artifact.get("status", "")
    else:
        result.rtl_generation_status = artifact.get("status", "")
        result.rtl_prompt_path = artifact.get("prompt_path", "")
        result.rtl_raw_response_path = artifact.get("raw_response_path", "")
    result.dc_status = "ok" if dc.get("success") else (candidate_eval["status"] if "dc" in candidate_eval else "")
    result.generated_verilog_path = artifact.get("generated_verilog_path", "")
    result.generated_verilog_files = list(artifact.get("generated_verilog_files", []))
    result.raw_generated_verilog_path = artifact.get("raw_generated_verilog_path", "")
    result.hls_interface_policy = dict(artifact.get("interface_policy", {}) or {})
    result.hls_interface_normalization = dict(artifact.get("interface_normalization", {}) or {})
    result.hls_syntax_validation = dict(artifact.get("syntax_validation", {}) or {})

    if candidate_eval["status"] != "success":
        result.status = candidate_eval["status"]
        result.followup_action_recommendation = "retry_with_different_actions"
        if artifact.get("stderr"):
            result.notes.append(str(artifact["stderr"])[-1000:])
        if dc.get("error"):
            result.notes.append(str(dc["error"]))
        return finish()

    if result.dc_retry_count:
        result.notes.append(
            f"DC failed {len(result.dc_attempt_errors)} time(s); regenerated RTL "
            f"{result.dc_retry_count} time(s) using Design Compiler feedback."
        )

    metrics = Module5Metrics(
        total_cell_area=candidate_eval["metrics"].get("total_cell_area"),
        data_arrival_time_ps=candidate_eval["metrics"].get("delay_ps", candidate_eval["metrics"].get("data_arrival_time_ps")),
        delay_ps=candidate_eval["metrics"].get("delay_ps", candidate_eval["metrics"].get("data_arrival_time_ps")),
        slack_ps=candidate_eval["metrics"].get("slack_ps"),
        slack_status=str(candidate_eval["metrics"].get("slack_status", "")),
    )

    if run_baseline:
        baseline_cache = _baseline_cache_dir(output_root, lead.benchmark, backend)
        cached_baseline = _load_cached_baseline(baseline_cache) if backend == "direct_rtl" else None

        if cached_baseline is not None:
            metrics.baseline_total_cell_area = cached_baseline.get("metrics", {}).get("total_cell_area")
            metrics.baseline_data_arrival_time_ps = cached_baseline.get("metrics", {}).get("delay_ps", cached_baseline.get("metrics", {}).get("data_arrival_time_ps"))
            metrics.baseline_delay_ps = cached_baseline.get("metrics", {}).get("delay_ps", metrics.baseline_data_arrival_time_ps)
            metrics.baseline_slack_ps = cached_baseline.get("metrics", {}).get("slack_ps")
            result.baseline_verilog_path = cached_baseline.get("artifact", {}).get("generated_verilog_path", "")
            result.notes.append(f"Reused cached baseline from {baseline_cache}")
        else:
            baseline_work_dir = baseline_cache if backend == "direct_rtl" else work_dir
            if backend == "direct_rtl":
                baseline_work_dir.mkdir(parents=True, exist_ok=True)
                _copy_text_file(source_snapshot, baseline_work_dir / "source.c")

            baseline_eval = _evaluate_candidate(
                lead,
                source_snapshot,
                candidate_name="baseline",
                work_dir=baseline_work_dir,
                backend=backend,
                env_path=env_path,
                spec_context=spec_context,
                enable_jg_verification=False,
                golden_rtl_path=golden_rtl_path,
                jg_max_retries=jg_max_retries,
                rtl_max_retries=rtl_max_retries,
                dc_max_retries=dc_max_retries,
                enable_pre_dc_equivalence=False,
                pre_dc_golden_top=pre_dc_golden_top,
                pre_dc_design_type=pre_dc_design_type,
                pre_dc_verification_timeout=pre_dc_verification_timeout,
                verification_mode="none",
            )
            if baseline_eval["status"] == "success":
                metrics.baseline_total_cell_area = baseline_eval["metrics"].get("total_cell_area")
                metrics.baseline_data_arrival_time_ps = baseline_eval["metrics"].get("delay_ps", baseline_eval["metrics"].get("data_arrival_time_ps"))
                metrics.baseline_delay_ps = baseline_eval["metrics"].get("delay_ps", metrics.baseline_data_arrival_time_ps)
                metrics.baseline_slack_ps = baseline_eval["metrics"].get("slack_ps")
                _update_result_token_usage(result, baseline_eval.get("llm_token_usage", []))
                baseline_artifact = baseline_eval.get("artifact", {})
                result.baseline_verilog_path = baseline_artifact.get("generated_verilog_path", "")
                if backend == "direct_rtl":
                    _store_cached_baseline(baseline_cache, baseline_eval)
            else:
                result.notes.append(f"Baseline evaluation failed: {baseline_eval['status']}")

    metrics = populate_deltas(metrics)
    result.metrics = metrics
    result.objective_feedback = build_objective_feedback(lead.objective, metrics)
    result.status = "success"
    if _is_hls_backend(backend):
        result.hls_status = "ok"
    else:
        result.rtl_generation_status = "ok"
    result.dc_status = "ok"
    result.followup_action_recommendation = "feed_reward_back_to_module45"

    return finish()
