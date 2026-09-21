"""Canonical records for cross-agent memory exchange."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RecordBase:
    record_id: str
    task_id: str
    run_id: str = ""
    parent_id: str = ""
    producer: str = ""
    schema_version: int = SCHEMA_VERSION
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRecord:
    task_id: str
    benchmark: str
    spec_text: str = ""
    source_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    objective: str = ""
    path: str = ""
    status: str = "active"
    producer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    started_at: str = field(default_factory=now_iso)
    completed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureRecord(RecordBase):
    features: Dict[str, Any] = field(default_factory=dict)
    confidence: Dict[str, Any] = field(default_factory=dict)
    overall_confidence: str = ""
    optimization_target: str = ""
    source_path: str = ""


@dataclass
class SelectionRecord(RecordBase):
    path: str = ""
    rule_fired: str = ""
    confidence: str = ""
    tier: int = 0
    reason: str = ""
    token_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionRecord(RecordBase):
    benchmark: str = ""
    objective: str = ""
    path: str = ""
    region_id: str = ""
    region_type: str = ""
    transform_name: str = ""
    selected: bool = False
    applied_successfully: Optional[bool] = None
    planning_score: Optional[float] = None
    # `expected_metric_gain` is unit-ambiguous by construction: it carries
    # percent on AREA runs and picoseconds on TIMING runs.  Any statistic taken
    # over the column without grouping by objective therefore mixes two
    # incompatible scales -- in the current database that is percent values in
    # 0.08..100 averaged together with picosecond values in 2532..3808.  The
    # two typed fields below are what new code should read; the old field stays
    # populated so every existing reader and exported CSV keeps working.
    expected_metric_gain: Optional[float] = None
    expected_area_gain_pct: Optional[float] = None
    expected_timing_gain_ps: Optional[float] = None
    confidence: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    def resolved_expected_gains(self) -> tuple[Optional[float], Optional[float]]:
        """Return `(expected_area_gain_pct, expected_timing_gain_ps)`.

        Callers that genuinely know both scales may set the typed fields
        themselves; every current producer only has the single ambiguous
        number, so route it by `objective`.  An empty or unrecognised
        objective leaves both typed fields NULL rather than guessing a unit.
        """
        area = self.expected_area_gain_pct
        timing = self.expected_timing_gain_ps
        if area is None and timing is None and self.expected_metric_gain is not None:
            objective = str(self.objective or "").strip().upper()
            if objective == "AREA":
                area = float(self.expected_metric_gain)
            elif objective == "TIMING":
                timing = float(self.expected_metric_gain)
        return area, timing


@dataclass
class EvaluationRecord(RecordBase):
    benchmark: str = ""
    objective: str = ""
    path: str = ""
    correctness_status: str = "unknown"
    synthesis_status: str = "unknown"
    area: Optional[float] = None
    baseline_area: Optional[float] = None
    area_gain_pct: Optional[float] = None
    data_arrival_time_ps: Optional[float] = None
    baseline_data_arrival_time_ps: Optional[float] = None
    # Positive means the candidate's critical-path delay is shorter.  The
    # historical field name is retained for storage/API compatibility.
    timing_gain_ps: Optional[float] = None
    slack_ps: Optional[float] = None
    baseline_slack_ps: Optional[float] = None
    slack_status: str = ""
    token_usage: Dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""


@dataclass
class FailureRecord(RecordBase):
    benchmark: str = ""
    module: str = ""
    failure_type: str = ""
    failure_stage: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    fix_applied: str = ""
    fix_succeeded: Optional[bool] = None
    retries_needed: int = 0
    source_path: str = ""


@dataclass
class ArtifactRecord(RecordBase):
    artifact_type: str = ""
    path: str = ""
    sha256: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryRecord(RecordBase):
    path: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    succeeded: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
