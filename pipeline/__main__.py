"""Command-line entry point for the synchronized pipeline."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from pipeline.orchestrator import PipelineOrchestrator
from spec_analyze.spec_loader import load_spec_entries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Modules 1-5 with automatic Memory Agent synchronization."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--verilog", help="Input Verilog file")
    source.add_argument(
        "--spec-file",
        help="Input text spec, JSON spec object, or JSON spec list",
    )
    source.add_argument("--spec-text", help="Inline text specification")
    parser.add_argument("--benchmark", default="", help="Benchmark name override")
    parser.add_argument("--objective", required=True, choices=["area", "timing"])
    parser.add_argument(
        "--db",
        default=None,
        help="Memory Agent SQLite override (default: project memory_agent.db)",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--c-output-dir", default=None)
    parser.add_argument("--module5-output-root", default=None)
    parser.add_argument(
        "--backend",
        choices=["direct_rtl", "hls"],
        default="direct_rtl",
        help=(
            "C-to-RTL backend used after C-first optimization: direct_rtl or hls"
        ),
    )
    parser.add_argument(
        "--experiment-group",
        choices=["optimized"],
        default="optimized",
        help="AI RTL pipeline group",
    )
    parser.add_argument("--module4-top-k", type=int, default=5)
    parser.add_argument("--mcts-iterations", type=int, default=240)
    parser.add_argument("--mcts-max-depth", type=int, default=5)
    parser.add_argument("--mcts-seed", type=int, default=7)
    parser.add_argument("--mcts-candidate-limit", type=int, default=48)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--single-action", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-path-llm", action="store_true")
    parser.add_argument(
        "--forced-path",
        choices=["auto", "c_first", "rtl_direct"],
        default="auto",
        help="Bypass adaptive routing for controlled path experiments",
    )
    parser.add_argument("--rtl-max-retries", type=int, default=2)
    parser.add_argument("--dc-max-retries", type=int, default=0)
    parser.add_argument(
        "--verification-mode",
        choices=("jaspergold", "none"),
        default="jaspergold",
        help=(
            "jaspergold (default): syntax -> JG equivalence -> DC; "
            "direct_rtl may retry after JG feedback, while hls uses zero JG retries; "
            "none: syntax -> DC WITHOUT equivalence verification. "
            "'none' leaves correctness_status unverified, so its runs must not be "
            "used as adaptive-router training labels."
        ),
    )
    parser.add_argument("--golden-rtl", default="")
    parser.add_argument("--golden-top", default="")
    parser.add_argument(
        "--design-type",
        choices=("combinational", "sequential"),
        default=None,
        help=(
            "Miter style for JasperGold equivalence. When omitted it is inferred "
            "from the golden RTL (clocked process -> sequential)."
        ),
    )
    parser.add_argument("--jg-max-retries", type=int, default=1)
    parser.add_argument(
        "--refine-policies-after-batch",
        action="store_true",
        help="Run Memory Agent LLM policy refinement after all spec entries finish",
    )
    parser.add_argument(
        "--policy-refine-model",
        default="",
        help="Optional model override for post-batch policy refinement",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between batch spec entries in seconds",
    )
    return parser.parse_args()


def _build_orchestrator(args: argparse.Namespace) -> PipelineOrchestrator:
    kwargs = {}
    if args.db:
        kwargs["memory_db_path"] = args.db
    if args.output_root:
        kwargs["output_root"] = args.output_root
    if args.c_output_dir:
        kwargs["c_output_dir"] = args.c_output_dir
    return PipelineOrchestrator(**kwargs)


def _summary_path(orchestrator: PipelineOrchestrator) -> Path:
    output_root = getattr(
        orchestrator,
        "output_root",
        Path(tempfile.gettempdir()) / "agent_pipeline",
    )
    return Path(output_root) / "summary.json"


def _write_summary(orchestrator: PipelineOrchestrator, summary: Dict[str, Any]) -> Path:
    path = _summary_path(orchestrator)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _extract_module_name(payload: Dict[str, Any]) -> str:
    features = payload.get("feature_result", {})
    llm_features = features.get("llm_features", {}) if isinstance(features, dict) else {}
    return str(llm_features.get("module_name") or "")


def _extract_module5_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    module5 = payload.get("module5_result", {})
    if not isinstance(module5, dict):
        return {}
    metrics = module5.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _compact_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    module5 = payload.get("module5_result", {})
    if not isinstance(module5, dict):
        module5 = {}
    direct_rtl = payload.get("direct_rtl_result", {})
    if not isinstance(direct_rtl, dict):
        direct_rtl = {}
    execution = direct_rtl if direct_rtl else module5
    c_generation = payload.get("c_generation", {})
    if not isinstance(c_generation, dict):
        c_generation = {}
    path_decision = payload.get("path_decision", {})
    if not isinstance(path_decision, dict):
        path_decision = {}

    # Raw, un-laundered equivalence verdict. Keep this separate from
    # ``correctness_status``: Memory Agent deliberately maps an unverified run to
    # "passed" (memory_agent/runtime.py:_correctness_status, project policy of
    # 2026-08-10), which is fine for retrieval heuristics but must never be used
    # to mint an adaptive-router training label.
    equivalence_status = str(
        execution.get("behavior_check_status", "not_run") or "not_run"
    ).lower()

    compact = {
        "benchmark": payload.get("benchmark", ""),
        "module_name": _extract_module_name(payload),
        "objective": str(payload.get("objective", "")).lower(),
        "experiment_group": payload.get("experiment_group", "optimized"),
        "status": payload.get("status", ""),
        "task_id": payload.get("task_id", ""),
        "run_id": payload.get("run_id", ""),
        "path": payload.get("path", ""),
        "rule_fired": path_decision.get("rule_fired", ""),
        "confidence": path_decision.get("confidence", ""),
        # The latent router's P(c_first | S). None whenever the predictor stage
        # produced no probability (not configured, no torch, hard rule fired
        # first); ``predictor_status`` records which. Kept in the compact summary
        # because path-selection accuracy is measured against this number.
        "p_c_first": path_decision.get("p_c_first"),
        "p_c_first_raw": path_decision.get("p_c_first_raw"),
        "predictor_status": path_decision.get("predictor_status", ""),
        "tier": path_decision.get("tier", ""),
        "generated_c_path": c_generation.get("c_path", ""),
        "generated_verilog_path": execution.get("generated_verilog_path", ""),
        "work_dir": execution.get("work_dir", ""),
        "module4_plan_path": payload.get("module4_plan_path", ""),
        "module45_plan_path": payload.get("module45_plan_path", ""),
        "module45_feedback_plan_path": payload.get("module45_feedback_plan_path", ""),
        "metrics": dict(execution.get("metrics", {}) or {}),
        "execution_status": execution.get("status", ""),
        "execution_backend": execution.get("backend", ""),
        "module5_status": module5.get("status", ""),
        "dc_status": execution.get("dc_status", ""),
        "syntax_status": execution.get("syntax_status", ""),
        "correctness_status": execution.get(
            "correctness_status",
            execution.get("behavior_check_status", "unknown"),
        ),
        "equivalence_status": equivalence_status,
        "label_eligible": equivalence_status in {"passed", "failed"},
        "compile_status": module5.get("compile_status", ""),
        "hls_status": module5.get("hls_status", ""),
        "rtl_generation_status": module5.get("rtl_generation_status", ""),
        "llm_token_totals": execution.get("llm_token_totals", {}),
        "full_result": payload,
    }
    if "error" in payload:
        compact["error"] = payload["error"]
    if "error_type" in payload:
        compact["error_type"] = payload["error_type"]
    return compact


def _build_summary(
    *,
    args: argparse.Namespace,
    source: str,
    total: int,
    failed: int,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    experiment_group = getattr(args, "experiment_group", "optimized")
    verification_mode = getattr(args, "verification_mode", "none")
    compact_results = [_compact_result(item) for item in results]
    return {
        "method": "pipeline",
        "pipeline": "modules_1_to_5_synchronized",
        "model": os.environ.get("OPENAI_MODEL", ""),
        "source": source,
        "default_objective": args.objective,
        "backend": args.backend,
        "experiment_group": experiment_group,
        "forced_path": getattr(args, "forced_path", "auto"),
        "verification_mode": verification_mode,
        "golden_rtl": getattr(args, "golden_rtl", ""),
        "golden_top": getattr(args, "golden_top", ""),
        "design_type": getattr(args, "design_type", None) or "combinational",
        # Label provenance: only runs with a real equivalence verdict may become
        # adaptive-router training labels (see spec revise §5).
        "equivalence_verified": verification_mode == "jaspergold",
        "label_eligible_count": sum(
            1 for item in compact_results if item.get("label_eligible")
        ),
        "total": total,
        "succeeded": total - failed,
        "failed": failed,
        "results": compact_results,
    }


def _run_one(
    orchestrator: PipelineOrchestrator,
    args: argparse.Namespace,
    *,
    benchmark: str,
    spec_text: str = "",
    objective: str,
    verilog_path: str | None = None,
):
    return orchestrator.run(
        benchmark=benchmark,
        objective=objective,
        verilog_path=verilog_path,
        spec_text=spec_text,
        use_llm_path_selection=not args.no_path_llm,
        module4_top_k=args.module4_top_k,
        mcts_iterations=args.mcts_iterations,
        mcts_max_depth=args.mcts_max_depth,
        mcts_seed=getattr(args, "mcts_seed", 7),
        mcts_candidate_limit=getattr(args, "mcts_candidate_limit", 48),
        module5_backend=args.backend,
        execute_best_path=not args.single_action,
        max_actions=args.max_actions,
        run_baseline=not args.no_baseline,
        module5_output_root=args.module5_output_root,
        rtl_max_retries=args.rtl_max_retries,
        dc_max_retries=args.dc_max_retries,
        experiment_group=getattr(args, "experiment_group", "optimized"),
        forced_path=getattr(args, "forced_path", "auto"),
        pre_dc_golden_rtl_path=getattr(args, "golden_rtl", ""),
        pre_dc_golden_top=getattr(args, "golden_top", ""),
        pre_dc_design_type=getattr(args, "design_type", None) or "combinational",
        verification_mode=getattr(args, "verification_mode", "none"),
        jg_max_retries=getattr(args, "jg_max_retries", 1),
    )


def _run_spec_file(
    orchestrator: PipelineOrchestrator,
    args: argparse.Namespace,
) -> int:
    entries = load_spec_entries(args.spec_file)
    if args.benchmark and len(entries) > 1:
        raise ValueError(
            "--benchmark can only override a single spec-file entry"
        )

    results = []
    failed = 0
    accepted_statuses = {"success"}
    for index, entry in enumerate(entries, start=1):
        benchmark = args.benchmark or entry["id"]
        entry_target = str(entry.get("optimization_target") or "").lower()
        objective = (
            entry_target if entry_target in {"area", "timing"} else args.objective
        )
        print(f"[Pipeline {index}/{len(entries)}] {benchmark}")
        try:
            result = _run_one(
                orchestrator,
                args,
                benchmark=benchmark,
                spec_text=entry["spec"],
                objective=objective,
            )
            payload = result.to_dict()
            if result.status not in accepted_statuses:
                failed += 1
        except Exception as exc:
            failed += 1
            payload = {
                "benchmark": benchmark,
                "objective": objective.upper(),
                "status": "pipeline_exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(payload)
        if index < len(entries) and args.delay > 0:
            time.sleep(args.delay)

    summary = _build_summary(
        args=args,
        source=str(Path(args.spec_file).resolve()),
        total=len(entries),
        failed=failed,
        results=results,
    )
    summary["mode"] = "batch" if len(entries) > 1 else "single"
    refinement_failed = False
    if getattr(args, "refine_policies_after_batch", False):
        objectives = tuple(sorted({
            str(item.get("objective") or args.objective).upper()
            for item in results
            if str(item.get("objective") or args.objective).upper()
            in {"AREA", "TIMING"}
        }))
        try:
            refinement = orchestrator.memory.refine_path_policies(
                objectives=objectives or (args.objective.upper(),),
                env_path=orchestrator.env_path,
                model=getattr(args, "policy_refine_model", ""),
            )
            summary["policy_refinement"] = {
                "status": "success",
                "model": refinement["model"],
                "policy_count": len(refinement["policies"]),
                "token_usage": refinement["token_usage"],
            }
        except Exception as exc:
            refinement_failed = True
            summary["policy_refinement"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    summary_path = _write_summary(orchestrator, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[Pipeline] Wrote summary to {summary_path}")
    return 0 if failed == 0 and not refinement_failed else 1


_SEQUENTIAL_RTL_MARKERS = ("posedge", "negedge")


def _looks_sequential(rtl_path: str | Path) -> bool:
    """Cheap textual check: does this RTL have a clocked process?"""
    try:
        text = Path(rtl_path).read_text(errors="replace")
    except OSError:
        return False
    return any(marker in text for marker in _SEQUENTIAL_RTL_MARKERS)


def _normalize_verification_args(args: argparse.Namespace) -> None:
    """Resolve JasperGold inputs before the orchestrator's hard validation.

    ``--verification-mode`` defaults to ``jaspergold`` because equivalence pass /
    fail is part of the reported correctness signal and of the adaptive-router
    label definition.  The orchestrator refuses ``jaspergold`` without a golden
    RTL, so fail here instead with an actionable message, and fill in the two
    inputs that can be derived unambiguously:

    * ``--golden-rtl``: for a ``--verilog`` run the input RTL *is* the reference.
    * ``--design-type``: inferred from the golden RTL when not given explicitly,
      since a clocked design checked with a combinational miter is meaningless.
    """
    if getattr(args, "verification_mode", "none") != "jaspergold":
        return

    golden = str(getattr(args, "golden_rtl", "") or "").strip()
    if not golden and getattr(args, "verilog", ""):
        golden = str(Path(args.verilog).resolve())
        args.golden_rtl = golden
        print(
            f"[Pipeline] --golden-rtl not given; using the input RTL as reference: {golden}"
        )
    if not golden:
        raise SystemExit(
            "[Pipeline] --verification-mode jaspergold requires a reference design.\n"
            "  Pass --golden-rtl <path> (and --golden-top <module> when the top "
            "module name differs), or run with --verification-mode none.\n"
            "  Note: --verification-mode none leaves correctness_status unverified, "
            "so those runs are not valid adaptive-router training labels."
        )
    if not Path(golden).is_file():
        raise SystemExit(f"[Pipeline] --golden-rtl not found: {golden}")

    if getattr(args, "design_type", None) is None:
        detected = "sequential" if _looks_sequential(golden) else "combinational"
        args.design_type = detected
        print(f"[Pipeline] --design-type inferred from golden RTL: {detected}")


def main() -> int:
    args = _parse_args()
    _normalize_verification_args(args)
    orchestrator = _build_orchestrator(args)

    if args.spec_file:
        return _run_spec_file(orchestrator, args)

    if args.verilog:
        benchmark = args.benchmark or Path(args.verilog).stem
        result = _run_one(
            orchestrator,
            args,
            benchmark=benchmark,
            objective=args.objective,
            verilog_path=args.verilog,
        )
    else:
        result = _run_one(
            orchestrator,
            args,
            benchmark=args.benchmark or "spec_input",
            objective=args.objective,
            spec_text=args.spec_text or "",
        )

    payload = result.to_dict()
    source = str(Path(args.verilog).resolve()) if args.verilog else "inline_spec_text"
    summary = _build_summary(
        args=args,
        source=source,
        total=1,
        failed=0 if result.status == "success" else 1,
        results=[payload],
    )
    summary["mode"] = "single"
    summary_path = _write_summary(orchestrator, summary)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[Pipeline] Wrote summary to {summary_path}")
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
