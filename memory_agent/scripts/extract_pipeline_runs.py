#!/usr/bin/env python3
"""Extract existing pipeline_runs artifacts into summary JSON and CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_ROOT = PROJECT_ROOT / "pipeline_runs"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def maybe_load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    data = load_json(path)
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def first_json(root: Path, pattern: str) -> Optional[Path]:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percent_reduction(baseline: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def delta(candidate: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if baseline is None or candidate is None:
        return None
    return candidate - baseline


def token_total(records: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for record in records:
        value = record.get("total_tokens")
        if isinstance(value, int):
            total += value
    return total


def normalize_objective(value: Any) -> str:
    return str(value or "").lower()


def extract_action_fields(action: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    return {
        f"{prefix}action_id": action.get("action_id", ""),
        f"{prefix}region_id": action.get("region_id", ""),
        f"{prefix}region_type": action.get("region_type", ""),
        f"{prefix}transform_name": action.get("transform_name", ""),
        f"{prefix}planning_score": action.get("planning_score", ""),
        f"{prefix}expected_metric_gain": action.get("expected_metric_gain", ""),
        f"{prefix}confidence": action.get("confidence", ""),
        f"{prefix}priority": action.get("priority", ""),
        f"{prefix}supporting_benchmarks": action.get("supporting_benchmarks", []),
    }


def find_module5_result(run_dir: Path) -> tuple[Dict[str, Any], str]:
    result_paths = [
        path for path in sorted(run_dir.glob("module5/**/result.json"))
        if "__baseline" not in str(path)
    ]
    if not result_paths:
        return {}, ""
    # Current pipeline executes one selected action sequence per run. If future
    # runs contain more, prefer the newest/deepest result by path order.
    result_path = result_paths[-1]
    return maybe_load_json(result_path), str(result_path)


def find_baseline_result(run_dir: Path, benchmark: str = "") -> tuple[Dict[str, Any], str]:
    pattern = "module5/**/__baseline_direct_rtl__/baseline_result.json"
    candidates = sorted(run_dir.glob(pattern))
    if benchmark:
        filtered = [path for path in candidates if f"/{benchmark}/" in str(path)]
        if filtered:
            candidates = filtered
    if not candidates:
        return {}, ""
    path = candidates[-1]
    return maybe_load_json(path), str(path)


def build_run_record(run_dir: Path) -> Dict[str, Any]:
    run_id = run_dir.name
    module4_path = run_dir / "module4_plan.json"
    module45_path = run_dir / "module45_plan.json"
    feedback_path = run_dir / "module45_feedback_plan.json"

    module4 = maybe_load_json(module4_path)
    module45 = maybe_load_json(module45_path)
    feedback = maybe_load_json(feedback_path)
    module5, module5_result_path = find_module5_result(run_dir)

    benchmark = (
        module5.get("benchmark")
        or module45.get("benchmark")
        or module4.get("benchmark")
        or ""
    )
    objective = normalize_objective(
        module5.get("objective")
        or module45.get("objective")
        or module4.get("objective")
    )
    baseline, baseline_result_path = find_baseline_result(run_dir, str(benchmark))

    candidate_metrics = module5.get("metrics", {})
    if not isinstance(candidate_metrics, dict):
        candidate_metrics = {}
    baseline_metrics = baseline.get("metrics", {})
    if not isinstance(baseline_metrics, dict):
        baseline_metrics = {}

    candidate_area = metric(candidate_metrics, "total_cell_area")
    baseline_area = metric(baseline_metrics, "total_cell_area")
    # Timing is the critical-path delay.  Prefer the explicit public alias and
    # keep the historical DC spellings only as compatibility fallbacks.
    candidate_delay = metric(candidate_metrics, "delay_ps")
    if candidate_delay is None:
        candidate_delay = metric(candidate_metrics, "data_arrival_time_ps")
    if candidate_delay is None:
        candidate_delay = metric(candidate_metrics, "timing_ps")
    baseline_delay = metric(baseline_metrics, "delay_ps")
    if baseline_delay is None:
        baseline_delay = metric(baseline_metrics, "data_arrival_time_ps")
    if baseline_delay is None:
        baseline_delay = metric(baseline_metrics, "timing_ps")
    candidate_slack = metric(candidate_metrics, "slack_ps")
    baseline_slack = metric(baseline_metrics, "slack_ps")

    best_actions = module45.get("best_actions", [])
    if not isinstance(best_actions, list):
        best_actions = []
    first_action = best_actions[0] if best_actions and isinstance(best_actions[0], dict) else {}
    sequence = []
    sequence_path = first_json(run_dir, "module5/**/sequence.json")
    if sequence_path:
        loaded_sequence = load_json(sequence_path)
        if isinstance(loaded_sequence, list):
            sequence = loaded_sequence

    status = str(module5.get("status") or "no_module5_result")
    success = status == "success"
    llm_token_totals = module5.get("llm_token_totals", {})
    if not isinstance(llm_token_totals, dict):
        llm_token_totals = {}
    baseline_artifact = baseline.get("artifact", {})
    if not isinstance(baseline_artifact, dict):
        baseline_artifact = {}

    record: Dict[str, Any] = {
        "run_id": run_id,
        "benchmark": benchmark,
        "objective": objective,
        "status": status,
        "success": success,
        "backend": module5.get("backend", ""),
        "source_c_path": module45.get("source_c_path") or module4.get("source_c_path", ""),
        "generated_c_path": module45.get("source_c_path") or module4.get("source_c_path", ""),
        "generated_verilog_path": module5.get("generated_verilog_path", ""),
        "work_dir": module5.get("work_dir", ""),
        "module4_plan_path": str(module4_path) if module4_path.is_file() else "",
        "module45_plan_path": str(module45_path) if module45_path.is_file() else "",
        "module45_feedback_plan_path": str(feedback_path) if feedback_path.is_file() else "",
        "module5_result_path": module5_result_path,
        "baseline_result_path": baseline_result_path,
        "candidate_area": candidate_area,
        "baseline_area": baseline_area,
        "area_delta": delta(candidate_area, baseline_area),
        "area_reduction_pct": percent_reduction(baseline_area, candidate_area),
        "candidate_delay_ps": candidate_delay,
        "baseline_delay_ps": baseline_delay,
        # Positive means the candidate reduced critical-path delay.
        "delay_improvement_ps": (
            baseline_delay - candidate_delay
            if candidate_delay is not None and baseline_delay is not None
            else None
        ),
        "delay_delta_ps": delta(candidate_delay, baseline_delay),
        "delay_reduction_pct": percent_reduction(baseline_delay, candidate_delay),
        "candidate_slack_ps": candidate_slack,
        "baseline_slack_ps": baseline_slack,
        "slack_improvement_ps": delta(candidate_slack, baseline_slack),
        "dc_status": module5.get("dc_status", ""),
        "syntax_status": module5.get("syntax_status", ""),
        "compile_status": module5.get("compile_status", ""),
        "hls_status": module5.get("hls_status", ""),
        "rtl_generation_status": module5.get("rtl_generation_status", ""),
        "behavior_check_status": module5.get("behavior_check_status", ""),
        "applied_transform_name": module5.get("applied_transform_name", ""),
        "sequence_length": module5.get("sequence_length") or len(sequence) or len(best_actions),
        "best_action_ids": module45.get("search_summary", {}).get("best_action_ids", [])
        if isinstance(module45.get("search_summary"), dict) else [],
        "action_space_size": module45.get("search_summary", {}).get("action_space_size", "")
        if isinstance(module45.get("search_summary"), dict) else "",
        "best_reward": module45.get("search_summary", {}).get("best_reward", "")
        if isinstance(module45.get("search_summary"), dict) else "",
        "feedback_best_action_ids": feedback.get("search_summary", {}).get("best_action_ids", [])
        if isinstance(feedback.get("search_summary"), dict) else [],
        "prompt_tokens": llm_token_totals.get("prompt_tokens", ""),
        "completion_tokens": llm_token_totals.get("completion_tokens", ""),
        "total_tokens": llm_token_totals.get("total_tokens", ""),
        "baseline_tokens": token_total(baseline_artifact.get("llm_token_usage", []))
        if isinstance(baseline_artifact.get("llm_token_usage"), list) else "",
        "notes": module5.get("notes", []),
        "full_module5_result": module5,
        "baseline_result": baseline,
    }
    record.update(extract_action_fields(first_action, "first_"))
    return record


def build_action_rows(run_record: Dict[str, Any], plan: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    actions = plan.get(kind, [])
    if not isinstance(actions, list):
        return rows
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        row = {
            "run_id": run_record["run_id"],
            "benchmark": run_record["benchmark"],
            "objective": run_record["objective"],
            "kind": kind,
            "rank": index + 1,
        }
        row.update(extract_action_fields(action))
        constraints = action.get("constraints", {})
        if isinstance(constraints, dict):
            row["expected_risk"] = constraints.get("expected_risk", "")
            row["apply_scope"] = constraints.get("apply_scope", "")
        rows.append(row)
    return rows


def build_trace_rows(run_record: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    trace = plan.get("search_trace", [])
    if not isinstance(trace, list):
        return rows
    for item in trace:
        if not isinstance(item, dict):
            continue
        rows.append({
            "run_id": run_record["run_id"],
            "benchmark": run_record["benchmark"],
            "objective": run_record["objective"],
            "iteration": item.get("iteration", ""),
            "reward": item.get("reward", ""),
            "chosen_action_ids": item.get("chosen_action_ids", []),
        })
    return rows


def extract(root: Path, output_dir: Path) -> Dict[str, Any]:
    run_dirs = sorted(path for path in root.glob("run_*") if path.is_dir())
    results: List[Dict[str, Any]] = []
    best_action_rows: List[Dict[str, Any]] = []
    action_space_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        record = build_run_record(run_dir)
        results.append(record)
        module45 = maybe_load_json(run_dir / "module45_plan.json")
        best_action_rows.extend(build_action_rows(record, module45, "best_actions"))
        action_space_rows.extend(build_action_rows(record, module45, "action_space"))
        trace_rows.extend(build_trace_rows(record, module45))

    succeeded = sum(1 for item in results if item.get("success"))
    summary = {
        "method": "pipeline",
        "source": str(root.resolve()),
        "total": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "outputs": {
            "results_csv": str(output_dir / "results.csv"),
            "best_actions_csv": str(output_dir / "best_actions.csv"),
            "action_space_csv": str(output_dir / "action_space.csv"),
            "search_trace_csv": str(output_dir / "search_trace.csv"),
        },
        "results": results,
    }

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "results.csv", results, RESULT_FIELDS)
    write_csv(output_dir / "best_actions.csv", best_action_rows, ACTION_FIELDS)
    write_csv(output_dir / "action_space.csv", action_space_rows, ACTION_FIELDS)
    write_csv(output_dir / "search_trace.csv", trace_rows, TRACE_FIELDS)
    return summary


RESULT_FIELDS = [
    "run_id", "benchmark", "objective", "status", "success", "backend",
    "source_c_path", "generated_c_path", "generated_verilog_path", "work_dir",
    "module4_plan_path", "module45_plan_path", "module45_feedback_plan_path",
    "module5_result_path", "baseline_result_path",
    "candidate_area", "baseline_area", "area_delta", "area_reduction_pct",
    "candidate_delay_ps", "baseline_delay_ps", "delay_improvement_ps",
    "delay_delta_ps",
    "delay_reduction_pct", "candidate_slack_ps", "baseline_slack_ps",
    "slack_improvement_ps", "dc_status", "syntax_status", "compile_status",
    "hls_status", "rtl_generation_status", "behavior_check_status",
    "applied_transform_name", "sequence_length", "best_action_ids",
    "action_space_size", "best_reward", "feedback_best_action_ids",
    "prompt_tokens", "completion_tokens", "total_tokens", "baseline_tokens",
    "first_action_id", "first_region_id", "first_region_type",
    "first_transform_name", "first_planning_score", "first_expected_metric_gain",
    "first_confidence", "first_priority", "notes",
]

ACTION_FIELDS = [
    "run_id", "benchmark", "objective", "kind", "rank", "action_id",
    "region_id", "region_type", "transform_name", "planning_score",
    "expected_metric_gain", "confidence", "priority", "expected_risk",
    "apply_scope", "supporting_benchmarks",
]

TRACE_FIELDS = [
    "run_id", "benchmark", "objective", "iteration", "reward",
    "chosen_action_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract existing pipeline_runs into summary JSON and CSV tables."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_PIPELINE_ROOT),
        help="Pipeline runs root containing run_* directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to --root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root
    if not root.is_dir():
        raise SystemExit(f"Pipeline root does not exist: {root}")
    summary = extract(root, output_dir)
    print(
        f"Extracted {summary['total']} runs: "
        f"{summary['succeeded']} succeeded, {summary['failed']} failed"
    )
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'results.csv'}")
    print(f"Wrote {output_dir / 'best_actions.csv'}")
    print(f"Wrote {output_dir / 'action_space.csv'}")
    print(f"Wrote {output_dir / 'search_trace.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
