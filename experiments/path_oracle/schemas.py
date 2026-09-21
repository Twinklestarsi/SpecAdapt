"""Schemas for paired C-first and specification-direct RTL runs.

These records intentionally contain no winner or label field.  A later,
separately reviewed stage may derive labels only after functional correctness
and the comparison policy have been defined.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, List


SCHEMA_VERSION = "path_oracle_pair_v3"

_TOKEN_TOTAL_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "retries",
)


def _token_int(value: Any) -> int | None:
    """Return a non-negative integer token field, or ``None`` if absent."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _token_totals_from_mapping(raw: Any) -> Dict[str, int]:
    """Read an already aggregated usage mapping without inventing fields.

    ``TokenUsage.to_dict()`` is used by CGen and the direct-RTL executor, while
    older Module5 results may omit ``retries``.  A provider response marked
    unavailable is deliberately ignored; in particular, a timeout must never
    turn into estimated token counts.
    """

    if not isinstance(raw, dict) or raw.get("available") is False:
        return {}
    totals: Dict[str, int] = {}
    for key in _TOKEN_TOTAL_KEYS:
        value = _token_int(raw.get(key))
        if value is not None:
            totals[key] = value
    return totals


def _token_totals_from_records(raw: Any) -> Dict[str, int]:
    """Aggregate Module5 usage records when no aggregate was persisted."""

    if not isinstance(raw, (list, tuple)):
        return {}
    totals: Dict[str, int] = {}
    for record in raw:
        if not isinstance(record, dict) or record.get("available") is False:
            continue
        for key in _TOKEN_TOTAL_KEYS:
            value = _token_int(record.get(key))
            if value is not None:
                totals[key] = totals.get(key, 0) + value
    return totals


def _execution_token_totals(execution: Dict[str, Any]) -> Dict[str, int]:
    """Read Module5/direct-RTL usage without double-counting its records."""

    aggregate = execution.get("llm_token_totals")
    if isinstance(aggregate, dict) and any(
        key in aggregate for key in _TOKEN_TOTAL_KEYS
    ):
        # An explicit aggregate is authoritative.  Falling back to records in
        # this case would count the same calls twice in newer result JSON.
        return _token_totals_from_mapping(aggregate)
    return _token_totals_from_records(execution.get("llm_token_usage"))


def _combine_token_totals(*sources: Dict[str, Any]) -> Dict[str, int]:
    """Sum independent usage sources and fill missing standard fields with 0."""

    combined = {key: 0 for key in _TOKEN_TOTAL_KEYS}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in _TOKEN_TOTAL_KEYS:
            value = _token_int(source.get(key))
            if value is not None:
                combined[key] += value
    return combined


def _file_sha256(raw_path: Any) -> str:
    path = Path(str(raw_path or ""))
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PathRunManifest:
    """One forced route in a paired experiment."""

    forced_path: str
    status: str = "planned"
    task_id: str = ""
    run_id: str = ""
    backend: str = ""
    model: str = ""
    syntax_status: str = "not_run"
    correctness_status: str = "unknown"
    # Raw functional-equivalence evidence.  Keep this separate from
    # correctness_status: the latter may be filled by runtime/Memory policy
    # after DC and is not proof that JasperGold ran.
    equivalence_status: str = ""
    verification_mode: str = "none"
    pre_dc_verification: Dict[str, Any] = field(default_factory=dict)
    jg_retry_count: int = 0
    jg_attempt_count: int = 0
    synthesis_status: str = "not_run"
    area: float | None = None
    # Public timing metric: critical-path delay in ps.  The DC spelling below
    # remains as a compatibility alias for historical manifests.
    delay_ps: float | None = None
    data_arrival_time_ps: float | None = None
    slack_ps: float | None = None
    llm_token_totals: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float | None = None
    generated_verilog_path: str = ""
    generated_verilog_sha256: str = ""
    work_dir: str = ""
    memory_store_path: str = ""
    memory_db_path: str = ""
    failure_stage: str = ""
    failure_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PairedRunManifest:
    """Two forced-path candidates for one spec/objective/repetition."""

    pair_id: str
    benchmark: str
    objective: str
    repeat_index: int
    route_order: List[str]
    family_id: str = ""
    state: str = "planned"
    schema_version: str = SCHEMA_VERSION
    spec_source: str = ""
    spec_sha256: str = ""
    feature_snapshot_path: str = ""
    feature_sha256: str = ""
    mcts_seed: int = 7
    llm_seed: int = 7
    mcts_config: Dict[str, Any] = field(default_factory=dict)
    run_config_sha256: str = ""
    code_tree_sha256: str = ""
    prompt_tree_sha256: str = ""
    toolchain_status: Dict[str, Any] = field(default_factory=dict)
    memory_baseline: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    runs: Dict[str, PathRunManifest] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Make the absence of labels an enforced schema property, rather than
        # merely leaving their values empty.
        forbidden = {"label", "winner", "preferred_path"}
        if forbidden.intersection(payload):
            raise ValueError("Paired run manifests must not contain labels")
        return payload


def path_run_from_pipeline(
    payload: Dict[str, Any],
    *,
    forced_path: str,
    elapsed_seconds: float,
    memory_store_path: str = "",
    memory_db_path: str = "",
) -> PathRunManifest:
    """Normalize either pipeline backend without claiming correctness."""

    if forced_path == "rtl_direct":
        execution = payload.get("direct_rtl_result", {})
    else:
        execution = payload.get("module5_result", {})
    if not isinstance(execution, dict):
        execution = {}
    c_generation = payload.get("c_generation", {})
    if not isinstance(c_generation, dict):
        c_generation = {}
    metrics = execution.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    behavior = str(
        execution.get(
            "correctness_status",
            execution.get("behavior_check_status", "unknown"),
        )
        or "unknown"
    ).lower()
    correctness = (
        "passed"
        if behavior in {"passed", "success", "ok"}
        else "failed"
        if behavior in {"failed", "error", "mismatch"}
        else "unknown"
    )
    explicit_synthesis = str(execution.get("synthesis_status") or "").lower()
    dc_status = str(execution.get("dc_status") or "not_run").lower()
    synthesis_status = explicit_synthesis if explicit_synthesis in {
        "success", "failed", "not_run"
    } else (
        "success"
        if dc_status in {"ok", "passed", "success"}
        else "failed"
        if dc_status in {"failed", "error"}
        else "not_run"
    )
    generated_verilog_path = str(execution.get("generated_verilog_path") or "")

    # Preserve the raw equivalence verdict without consulting
    # correctness_status.  Different pipeline generations put the JasperGold
    # result in one of these three locations; the first explicitly supplied
    # value is authoritative.
    equivalence_status = execution.get("equivalence_status")
    if equivalence_status is None or (
        isinstance(equivalence_status, str) and not equivalence_status.strip()
    ):
        equivalence_status = execution.get("behavior_check_status")
    if equivalence_status is None or (
        isinstance(equivalence_status, str) and not equivalence_status.strip()
    ):
        pre_dc = execution.get("pre_dc_verification")
        if isinstance(pre_dc, dict):
            equivalence_status = pre_dc.get("status")
    equivalence_status = str(equivalence_status or "")

    # Module 1 feature extraction is shared by both forced routes and is kept
    # only in ``frozen_feature_result.json``.  For c_first, the route-specific
    # total is CGen (generation + semantic review) plus Module5 (C editing,
    # C-to-RTL generation, and any JG repair).  This also works when CGen or
    # Module5 stops early: whatever was already returned is retained.
    cgen_usage = _token_totals_from_mapping(c_generation.get("token_usage"))
    if not cgen_usage:
        # A small compatibility bridge for early result writers that used the
        # aggregate name on the nested C-generation record.
        cgen_usage = _token_totals_from_mapping(
            c_generation.get("llm_token_totals")
        )
    execution_usage = _execution_token_totals(execution)
    route_usage = (
        _combine_token_totals(cgen_usage, execution_usage)
        if forced_path == "c_first"
        else _combine_token_totals(execution_usage)
    )

    return PathRunManifest(
        forced_path=forced_path,
        status=str(payload.get("status") or execution.get("status") or "unknown"),
        task_id=str(payload.get("task_id") or ""),
        run_id=str(payload.get("run_id") or ""),
        backend=str(execution.get("backend") or ""),
        model=str(execution.get("model") or ""),
        syntax_status=str(execution.get("syntax_status") or "not_run"),
        correctness_status=correctness,
        equivalence_status=equivalence_status,
        verification_mode=str(execution.get("verification_mode") or "none"),
        pre_dc_verification=dict(execution.get("pre_dc_verification") or {}),
        jg_retry_count=int(execution.get("jg_retry_count", 0) or 0),
        jg_attempt_count=len(execution.get("jg_attempts") or []),
        synthesis_status=synthesis_status,
        area=metrics.get("total_cell_area"),
        delay_ps=metrics.get("delay_ps", metrics.get("data_arrival_time_ps")),
        data_arrival_time_ps=metrics.get("delay_ps", metrics.get("data_arrival_time_ps")),
        slack_ps=metrics.get("slack_ps"),
        llm_token_totals=route_usage,
        elapsed_seconds=elapsed_seconds,
        generated_verilog_path=generated_verilog_path,
        generated_verilog_sha256=_file_sha256(generated_verilog_path),
        work_dir=str(execution.get("work_dir") or ""),
        memory_store_path=memory_store_path,
        memory_db_path=memory_db_path,
        failure_stage=str(execution.get("failure_stage") or ""),
        failure_reason=str(execution.get("error") or ""),
    )
