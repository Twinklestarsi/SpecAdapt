"""Execute the specification-direct RTL path with the shared DC backend."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from RTL_DIRECT_compare.generator import RTLDirectGenerator
from module5.dc_runner import run_dc_for_verilog
from module5.jg_verifier import golden_interface_contract, verify_rtl_with_jg_retry
from token_counter import TokenUsage


@dataclass
class SpecDirectExecutionResult:
    """Normalized result for ``specification -> RTL -> DC`` execution."""

    benchmark: str
    objective: str
    work_dir: str
    route: str = "rtl_direct"
    backend: str = "spec_to_rtl"
    status: str = "pending"
    execution_status: str = "pending"
    model: str = ""
    module_name: str = ""
    generated_verilog_path: str = ""
    syntax_status: str = "not_run"
    rtl_generation_status: str = "not_run"
    rtl_attempt_count: int = 0
    rtl_attempt_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    dc_status: str = "not_run"
    behavior_check_status: str = "not_run"
    correctness_status: str = "unknown"
    verification_mode: str = "none"
    pre_dc_verification: Dict[str, Any] = field(default_factory=dict)
    generation_interface_contract: str = ""
    jg_retry_count: int = 0
    jg_attempts: List[Dict[str, Any]] = field(default_factory=list)
    synthesis_status: str = "not_run"
    metrics: Dict[str, Any] = field(default_factory=dict)
    llm_token_totals: Dict[str, Any] = field(default_factory=dict)
    dc_retry_count: int = 0
    dc_attempt_errors: List[Dict[str, Any]] = field(default_factory=list)
    failure_stage: str = ""
    error: str = ""
    input_manifest_path: str = ""
    result_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def execute_spec_direct_rtl(
    *,
    benchmark: str,
    objective: str,
    spec_text: str,
    llm_features: Dict[str, Any],
    output_root: str | Path,
    env_path: str | Path,
    rtl_max_retries: int = 2,
    dc_max_retries: int = 0,
    pre_dc_golden_rtl_path: str | Path = "",
    pre_dc_golden_top: str = "",
    pre_dc_design_type: str = "combinational",
    pre_dc_verification_timeout: int = 180,
    verification_mode: str = "none",
    jg_max_retries: int = 1,
) -> SpecDirectExecutionResult:
    """Generate RTL directly from the spec, validate it, and run shared DC.

    ``verification_mode=jaspergold`` requires syntax and JG equivalence before
    DC.  ``verification_mode=none`` sends a syntax-valid candidate directly to
    DC and retains the project's explicit unverified-correctness policy.
    """

    objective = objective.upper()
    verification_mode = str(verification_mode or "none").strip().lower()
    if verification_mode not in {"jaspergold", "none"}:
        raise ValueError(f"Unsupported verification_mode: {verification_mode}")
    if jg_max_retries < 0:
        raise ValueError("jg_max_retries must be non-negative")
    if verification_mode == "jaspergold" and not pre_dc_golden_rtl_path:
        raise ValueError("jaspergold verification requires a golden RTL path")
    work_dir = Path(output_root).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    result = SpecDirectExecutionResult(
        benchmark=benchmark,
        objective=objective,
        work_dir=str(work_dir),
        verification_mode=verification_mode,
    )
    result_path = work_dir / "result.json"
    result.result_path = str(result_path)
    input_manifest = _write_json(
        work_dir / "input.json",
        {
            "benchmark": benchmark,
            "objective": objective,
            "route": "rtl_direct",
            "backend": "spec_to_rtl",
            "spec_text": spec_text,
            "llm_features": llm_features,
            "rtl_max_retries": rtl_max_retries,
            "dc_max_retries": dc_max_retries,
            "pre_dc_golden_rtl_path": str(pre_dc_golden_rtl_path or ""),
            "pre_dc_golden_top": pre_dc_golden_top,
            "pre_dc_design_type": pre_dc_design_type,
            "pre_dc_verification_timeout": pre_dc_verification_timeout,
            "verification_mode": verification_mode,
            "jg_max_retries": jg_max_retries,
        },
    )
    result.input_manifest_path = str(input_manifest)

    if not spec_text.strip():
        result.status = "rtl_failed"
        result.execution_status = "failed"
        result.rtl_generation_status = "missing_spec"
        result.failure_stage = "rtl_generation"
        result.error = "Specification text is empty"
        _write_json(result_path, result.to_dict())
        return result

    generator = RTLDirectGenerator(
        output_dir=work_dir / "generation_0",
        env_path=env_path,
        max_llm_retries=max(1, rtl_max_retries + 1),
    )
    # The interface is the part of the reference the candidate is *required* to
    # reproduce, so stating it up front prevents the rejection that used to
    # happen before any proof ran.  It is derived with the same module-selection
    # rule the equivalence checker uses, so the contract can never describe a
    # different module than the one that will be compared, and only port names,
    # directions, widths and parameter defaults are exposed — never reference
    # logic.  An unparsable reference yields "" and generation proceeds as before.
    interface_contract = (
        golden_interface_contract(
            pre_dc_golden_rtl_path,
            pre_dc_golden_top,
            verilog_2001=True,
        )
        if pre_dc_golden_rtl_path
        else ""
    )
    result.generation_interface_contract = interface_contract
    aggregate_usage = TokenUsage()
    current_spec = spec_text

    for dc_attempt in range(dc_max_retries + 1):
        generation_dir = work_dir / f"generation_{dc_attempt}"
        generated = generator.generate(
            benchmark=benchmark,
            spec_text=current_spec,
            features=llm_features,
            output_dir=generation_dir,
            interface_contract=interface_contract,
        )
        aggregate_usage += generated.token_usage
        result.model = generated.model
        result.module_name = generated.module_name
        result.generated_verilog_path = generated.rtl_path
        result.syntax_status = "ok" if generated.syntax_ok else "failed"
        result.rtl_generation_status = "ok" if generated.success else "failed"
        result.llm_token_totals = aggregate_usage.to_dict()
        result.rtl_attempt_count += int(generated.attempt_count or 0)
        result.rtl_attempt_artifacts.extend(generated.attempt_artifacts or [])

        if not generated.success:
            result.status = "rtl_failed"
            result.execution_status = "failed"
            result.failure_stage = "rtl_generation"
            result.error = generated.error or "Specification-direct RTL generation failed"
            _write_json(result_path, result.to_dict())
            return result

        if verification_mode == "jaspergold":
            source_context = (
                f"Specification:\n{spec_text}\n\n"
                "Frozen structured feature contract:\n"
                + json.dumps(llm_features or {}, indent=2, ensure_ascii=False)
            )
            verification = verify_rtl_with_jg_retry(
                candidate_path=generated.rtl_path,
                golden_path=pre_dc_golden_rtl_path,
                output_dir=generation_dir / "pre_dc_verification",
                env_path=env_path,
                source_label="specification and frozen feature contract",
                source_context=source_context,
                module_name=generated.module_name,
                golden_top=pre_dc_golden_top,
                design_type=pre_dc_design_type,
                verification_timeout=pre_dc_verification_timeout,
                max_retries=jg_max_retries,
            )
            result.pre_dc_verification = dict(verification)
            result.jg_retry_count += int(verification.get("retry_count", 0) or 0)
            result.jg_attempts.extend(list(verification.get("attempts") or []))
            for usage in verification.get("llm_token_usage") or []:
                aggregate_usage += TokenUsage(
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    retries=1,
                    total_tokens=int(usage.get("total_tokens", 0) or 0),
                )
            result.llm_token_totals = aggregate_usage.to_dict()
            final_candidate = str(verification.get("final_candidate_path") or "")
            if final_candidate:
                result.generated_verilog_path = final_candidate
            verification_status = str(verification.get("status") or "error")
            if verification_status != "passed":
                result.status = "equivalence_failed"
                result.execution_status = "failed"
                result.behavior_check_status = (
                    "failed" if verification_status == "failed" else verification_status
                )
                result.correctness_status = (
                    "failed" if verification_status == "failed" else "unknown"
                )
                result.failure_stage = "equivalence"
                result.error = str(
                    verification.get("error")
                    or f"Pre-DC equivalence ended with {verification_status}"
                )
                _write_json(result_path, result.to_dict())
                return result
            result.behavior_check_status = "passed"
            result.correctness_status = "passed"
        else:
            result.pre_dc_verification = {
                "status": "not_run",
                "equivalent": None,
                "method": "none",
                "verification_mode": "none",
                "reason": "syntax_then_dc_without_equivalence",
            }

        dc_stem = f"{benchmark}_spec_direct_{dc_attempt}"
        dc_result = run_dc_for_verilog(
            result.generated_verilog_path,
            benchmark=benchmark,
            goal=objective.lower(),
            stem=dc_stem,
            output_root=work_dir / "dc_runs",
            top_module=generated.module_name,
        )
        result.metrics = dict(dc_result.get("metrics", {}) or {})
        if dc_result.get("success", False):
            result.status = "success"
            result.execution_status = "completed"
            result.dc_status = "ok"
            result.synthesis_status = "success"
            result.correctness_status = "passed"
            result.failure_stage = ""
            result.error = ""
            _write_json(result_path, result.to_dict())
            return result

        dc_error = str(dc_result.get("error") or "Design Compiler failed")
        result.dc_attempt_errors.append(
            {
                "attempt": dc_attempt,
                "error": dc_error,
                "report_dir": dc_result.get("report_dir", ""),
            }
        )
        if dc_attempt >= dc_max_retries:
            result.status = "dc_failed"
            result.execution_status = "failed"
            result.dc_status = "failed"
            result.synthesis_status = "failed"
            result.failure_stage = "dc"
            result.error = dc_error
            _write_json(result_path, result.to_dict())
            return result

        result.dc_retry_count += 1
        current_spec = (
            f"{spec_text}\n\n"
            "The previous RTL passed syntax validation but failed Synopsys "
            "Design Compiler. Regenerate the complete module while preserving "
            "the exact specification and interface.\n"
            f"Design Compiler feedback:\n{dc_error}"
        )

    _write_json(result_path, result.to_dict())
    return result
