from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Module5Metrics:
    total_cell_area: Optional[float] = None
    data_arrival_time_ps: Optional[float] = None
    # Public timing name.  ``data_arrival_time_ps`` is retained as the raw DC
    # spelling and remains an alias of this critical-path delay value.
    delay_ps: Optional[float] = None
    slack_ps: Optional[float] = None
    slack_status: str = ""
    baseline_total_cell_area: Optional[float] = None
    baseline_data_arrival_time_ps: Optional[float] = None
    baseline_delay_ps: Optional[float] = None
    baseline_slack_ps: Optional[float] = None
    delta_area: Optional[float] = None
    delta_data_arrival_time: Optional[float] = None
    delay_improvement_ps: Optional[float] = None
    delta_slack: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Module5ExecutionResult:
    action_id: str
    benchmark: str
    objective: str
    source_c_path: str
    backend: str = "direct_rtl"
    model: str = ""
    edited_c_path: str = ""
    generated_verilog_path: str = ""
    generated_verilog_files: List[str] = field(default_factory=list)
    raw_generated_verilog_path: str = ""
    baseline_verilog_path: str = ""
    work_dir: str = ""
    status: str = "pending"
    execution_status: str = "pending"
    applied_transform_name: str = ""
    edit_summary: str = ""
    changed_functions: List[str] = field(default_factory=list)
    changed_regions: List[str] = field(default_factory=list)
    syntax_status: str = ""
    compile_status: str = ""
    hls_status: str = ""
    hls_interface_policy: Dict[str, Any] = field(default_factory=dict)
    hls_interface_normalization: Dict[str, Any] = field(default_factory=dict)
    hls_syntax_validation: Dict[str, Any] = field(default_factory=dict)
    rtl_generation_status: str = ""
    rtl_prompt_path: str = ""
    rtl_raw_response_path: str = ""
    dc_status: str = ""
    behavior_check_status: str = "not_run"
    correctness_status: str = "unknown"
    verification_mode: str = "none"
    pre_dc_verification: Dict[str, Any] = field(default_factory=dict)
    jg_retry_count: int = 0
    jg_attempts: List[Dict[str, Any]] = field(default_factory=list)
    synthesis_status: str = "not_run"
    objective_feedback: Dict[str, Any] = field(default_factory=dict)
    metrics: Module5Metrics = field(default_factory=Module5Metrics)
    preserved_constraints: List[str] = field(default_factory=list)
    validator_summary: Dict[str, Any] = field(default_factory=dict)
    failed_preconditions: List[str] = field(default_factory=list)
    followup_action_recommendation: str = ""
    notes: List[str] = field(default_factory=list)
    llm_token_usage: List[Dict[str, Any]] = field(default_factory=list)
    llm_token_totals: Dict[str, int] = field(default_factory=dict)
    dc_retry_count: int = 0
    dc_attempt_errors: List[Dict[str, Any]] = field(default_factory=list)
    # Sequence mode fields
    execution_mode: str = "single_action"
    applied_action_ids: List[str] = field(default_factory=list)
    sequence_length: int = 0
    per_action_summaries: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data
