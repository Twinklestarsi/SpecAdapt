"""
memory_agent — Module 8: Central intelligence for the 8-module pipeline.

Public API:

    from memory_agent import MemoryAgent

    agent = MemoryAgent("path_decisions_log.json")

    # Derive path-selection rules from accumulated decisions + PPA outcomes
    rules = agent.derive_path_rules()
    # → writes rules to "path_selection_rules" in the staging file
    # → Module 2 loads them automatically on next startup

    # Record a PPA outcome from Module 7
    agent.record_ppa_outcome(
        benchmark="uart_tx",
        path="rtl_direct",
        area_improvement_pct=-8.3,
        slack_status="MET",
    )

    # Status summary
    agent.status()   # → dict of record counts per category

    # Inspect current rules
    agent.get_path_selection_rules()   # → list of condition-dicts
"""

from memory_agent.agent import MemoryAgent
from memory_agent.schemas import (
    ActionRecord,
    ArtifactRecord,
    EvaluationRecord,
    FailureRecord,
    FeatureRecord,
    RunRecord,
    SelectionRecord,
    TaskRecord,
    TrajectoryRecord,
)
from memory_agent.runtime import MemorySession, PipelineContext

__all__ = [
    "MemoryAgent",
    "ActionRecord",
    "ArtifactRecord",
    "EvaluationRecord",
    "FailureRecord",
    "FeatureRecord",
    "RunRecord",
    "SelectionRecord",
    "TaskRecord",
    "TrajectoryRecord",
    "MemorySession",
    "PipelineContext",
]
