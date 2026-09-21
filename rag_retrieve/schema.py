"""
Schema objects for Module 4 CDFG/RAG indexing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class CDFGGraphStats:
    function_name: str
    dot_path: str
    png_path: str
    node_count: int
    edge_count: int
    data_edge_count: int
    control_edge_count: int
    opcode_histogram: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CDFGBenchmarkRecord:
    objective: str
    subcategory: str
    benchmark: str
    benchmark_dir: str
    c_path: str = ""
    cpp_path: str = ""
    verilog_path: str = ""
    cdfg_dir: str = ""
    graphs: List[CDFGGraphStats] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["graphs"] = [graph.to_dict() for graph in self.graphs]
        return data


@dataclass
class RAGCSVRow:
    sample_id: str = ""
    objective: str = ""
    subcategory: str = ""
    benchmark: str = ""
    transform_name: str = ""
    transform_parameters: str = ""
    dc_collection: str = ""
    backend: str = ""
    library: str = ""
    clock_period_ps: float = 0.0
    constraint_id: str = ""
    sdc_path: str = ""
    original_c_path: str = ""
    optimized_c_path: str = ""
    original_c_hash: str = ""
    optimized_c_hash: str = ""
    transform_requested: str = "unknown"
    transform_applied: str = "unknown"
    source_changed: str = "unknown"
    no_op: str = "false"
    apply_failure_reason: str = ""
    transform_summary: str = ""
    pairing_status: str = ""
    verification_status: str = "unknown"
    evidence_level: str = "benchmark"
    attribution_weight: float = 1.0
    total_cell_area: float = 0.0
    baseline_area: float = 0.0
    area_gain: float = 0.0
    area_improvement_pct: float = 0.0
    delay_ps: float = 0.0
    baseline_delay_ps: float = 0.0
    delay_improvement_ps: float = 0.0
    slack_ps: float = 0.0
    baseline_slack: float = 0.0
    slack_improvement_ps: float = 0.0
    slack_status: str = ""
    objective_gain: float = 0.0
    number_of_ports: float = 0.0
    number_of_cells: float = 0.0
    number_of_combinational_cells: float = 0.0
    number_of_sequential_cells: float = 0.0
    area_report_path: str = ""
    timing_report_path: str = ""
    raw_duplicate_count: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CDFGRAGLinkedRecord:
    cdfg_record: CDFGBenchmarkRecord
    rag_rows: List[RAGCSVRow] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cdfg_record": self.cdfg_record.to_dict(),
            "rag_rows": [row.to_dict() for row in self.rag_rows],
            "warnings": list(self.warnings),
        }


@dataclass
class QueryCDFGRecord:
    benchmark: str
    source_c_path: str
    query_root: str
    copied_c_path: str
    cdfg_dir: str
    graphs: List[CDFGGraphStats] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["graphs"] = [graph.to_dict() for graph in self.graphs]
        return data


@dataclass
class CDFGRegion:
    region_id: str
    graph_function: str
    region_type: str
    anchor_node_ids: List[str] = field(default_factory=list)
    anchor_labels: List[str] = field(default_factory=list)
    node_ids: List[str] = field(default_factory=list)
    edge_count: int = 0
    features: Dict[str, Any] = field(default_factory=dict)
    source_anchor: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Module5ActionConstraints:
    must_preserve_behavior: bool = True
    forbidden_transforms: List[str] = field(default_factory=list)
    conflict_group: str = ""
    apply_scope: str = "local"
    expected_risk: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Module5Action:
    action_id: str
    benchmark: str
    objective: str
    source_c_path: str
    region_id: str
    region_type: str
    transform_name: str
    priority: float
    planning_score: float
    expected_metric_gain: float
    confidence: float
    problem_hypothesis: str
    execution_hint: str
    source_anchor: Dict[str, Any] = field(default_factory=dict)
    supporting_matches: List[Dict[str, Any]] = field(default_factory=list)
    supporting_benchmarks: List[str] = field(default_factory=list)
    constraints: Module5ActionConstraints = field(default_factory=Module5ActionConstraints)
    memory_score: float = 0.0
    memory_success_rate: float = 0.0
    memory_evidence_count: int = 0
    memory_evidence: List[Dict[str, Any]] = field(default_factory=list)
    region_features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["constraints"] = self.constraints.to_dict()
        return data


@dataclass
class Module45Plan:
    benchmark: str
    objective: str
    source_c_path: str
    planner_name: str
    iterations: int
    max_depth: int
    harmful_blacklist: List[str] = field(default_factory=list)
    action_space: List[Module5Action] = field(default_factory=list)
    best_actions: List[Module5Action] = field(default_factory=list)
    search_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "objective": self.objective,
            "source_c_path": self.source_c_path,
            "planner_name": self.planner_name,
            "iterations": self.iterations,
            "max_depth": self.max_depth,
            "harmful_blacklist": list(self.harmful_blacklist),
            "action_space": [action.to_dict() for action in self.action_space],
            "best_actions": [action.to_dict() for action in self.best_actions],
            "search_trace": list(self.search_trace),
        }
