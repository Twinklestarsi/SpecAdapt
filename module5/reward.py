from __future__ import annotations

from typing import Any, Dict

from module5.result_schema import Module5Metrics


def populate_deltas(metrics: Module5Metrics) -> Module5Metrics:
    if metrics.delay_ps is None:
        metrics.delay_ps = metrics.data_arrival_time_ps
    if metrics.baseline_delay_ps is None:
        metrics.baseline_delay_ps = metrics.baseline_data_arrival_time_ps
    # `delay_ps` is authoritative.  Keep the raw DC spelling synchronized so
    # legacy consumers cannot accidentally score a different timing value.
    if metrics.delay_ps is not None:
        metrics.data_arrival_time_ps = metrics.delay_ps
    if metrics.baseline_delay_ps is not None:
        metrics.baseline_data_arrival_time_ps = metrics.baseline_delay_ps
    if metrics.total_cell_area is not None and metrics.baseline_total_cell_area is not None:
        metrics.delta_area = metrics.total_cell_area - metrics.baseline_total_cell_area
    if metrics.delay_ps is not None and metrics.baseline_delay_ps is not None:
        metrics.delta_data_arrival_time = (
            metrics.delay_ps - metrics.baseline_delay_ps
        )
        metrics.delay_improvement_ps = (
            metrics.baseline_delay_ps - metrics.delay_ps
        )
    if metrics.slack_ps is not None and metrics.baseline_slack_ps is not None:
        metrics.delta_slack = metrics.slack_ps - metrics.baseline_slack_ps
    return metrics


def build_objective_feedback(objective: str, metrics: Module5Metrics) -> Dict[str, Any]:
    objective = objective.lower()
    feedback: Dict[str, Any] = {
        "objective": objective,
        "primary_metric": "total_cell_area" if objective == "area" else "delay_ps",
        "reward": None,
        "improved": None,
    }

    if objective == "area":
        if metrics.delta_area is not None:
            feedback["reward"] = -metrics.delta_area
            feedback["improved"] = metrics.delta_area < 0
        elif metrics.total_cell_area is not None:
            feedback["reward"] = -metrics.total_cell_area
    else:
        if metrics.delta_data_arrival_time is not None:
            feedback["reward"] = -metrics.delta_data_arrival_time
            feedback["improved"] = metrics.delta_data_arrival_time < 0
        elif metrics.delay_ps is not None:
            feedback["reward"] = -metrics.delay_ps

        feedback["delay_ps"] = metrics.delay_ps
        feedback["baseline_delay_ps"] = metrics.baseline_delay_ps
        feedback["delay_improvement_ps"] = metrics.delay_improvement_ps

    feedback["slack_status"] = metrics.slack_status
    return feedback
