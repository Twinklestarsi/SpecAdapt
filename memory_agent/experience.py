"""
experience.py — Dataclasses for the four Module 8 experience categories.

These are the structured records that all pipeline modules write into the
Memory Agent's store, and that the Memory Agent reads to derive rules.

Category 1: GenerationPathRecord
    Written by Module 2 (decision) and Module 7 (PPA outcome).
    Used by: Module 2 (future path decisions).

Category 2: OptimizationDecisionRecord
    Written by Modules 3, 5, 6 after each transform.
    Used by: Module 3 (transform ranking).

Category 3: CorrectionPatternRecord
    Written by Modules 4, 5, 6, 7 on failure + fix.
    Used by: all modules (avoid known failures).

Category 4: CollaborationTrajectoryRecord
    Written by the pipeline orchestrator after each benchmark.
    Used by: Module 2 (path planning), token tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ── Helpers ────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Category 1: Effective generation paths ─────────────────────────────

@dataclass
class PPAOutcome:
    """PPA delta recorded by Module 7 after DC synthesis."""
    area_improvement_pct: Optional[float] = None    # negative = area reduction (good)
    # Deprecated legacy field.  Older staging files used a negative-valued
    # percentage for timing; new records must use timing_improvement_ps.
    timing_improvement_pct: Optional[float] = None
    slack_status: Optional[str] = None              # "MET" | "VIOLATED" | "UNKNOWN"
    cell_count_delta: Optional[int] = None
    synthesis_failed: bool = False
    notes: str = ""
    # Appended after the legacy fields so positional construction remains
    # compatible with older staging clients.
    timing_improvement_ps: Optional[float] = None   # positive = critical-path delay reduction (good)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationPathRecord:
    """
    Category 1: one benchmark's path decision + eventual PPA result.

    Module 2 writes the initial record (ppa_outcome=None).
    Module 7 fills in ppa_outcome after synthesis completes.
    The Memory Agent uses completed records (ppa_outcome not None)
    to derive path-selection rules.
    """
    benchmark: str
    path: str                          # "rtl_direct" | "c_first"
    rule_fired: str
    confidence: str                    # "high" | "medium" | "low"
    tier: int                          # 1 or 2
    feature_profile: Dict[str, Any]    # compact routing-relevant features
    ppa_outcome: Optional[PPAOutcome] = None
    timestamp: str = field(default_factory=_now_iso)
    module_token_total: int = 0        # sum of all module token costs for this benchmark

    def is_complete(self) -> bool:
        """True when PPA outcome is available (Module 7 has run)."""
        return self.ppa_outcome is not None and not self.ppa_outcome.synthesis_failed

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ── Category 2: Optimization decisions ────────────────────────────────

@dataclass
class OptimizationDecisionRecord:
    """
    Category 2: one transform applied during C/RTL optimization.

    Written by Modules 3, 5, 6.
    Read by Module 3 for transform ranking.
    """
    benchmark: str
    module: int                        # 3, 5, or 6
    transform_name: str                # e.g. "loop_unroll", "pipeline", "dead_code_elim"
    transform_context: Dict[str, Any]  # feature profile or design context at time of apply
    path: str                          # which path this benchmark is on
    # ``timing`` is the positive critical-path delay reduction in ps.
    ppa_delta: Optional[Dict[str, float]] = None  # {"area": -5.2, "timing": +1.1}
    applied_successfully: bool = True
    notes: str = ""
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Category 3: Correction patterns ────────────────────────────────────

@dataclass
class CorrectionPatternRecord:
    """
    Category 3: a failure and how it was fixed.

    Written by Modules 4, 5, 6, 7 on any error that required intervention.
    Read by all modules to avoid repeating known failure modes.
    """
    benchmark: str
    module: int                        # which module failed
    failure_type: str                  # e.g. "jasper_equiv_fail", "hls_synth_error",
                                       #       "dc_timing_violation", "llm_parse_error"
    failure_context: Dict[str, Any]    # feature profile / code snippet / error message
    fix_applied: str                   # description of the fix (prompt change, retry, etc.)
    fix_succeeded: bool = True
    retries_needed: int = 0
    notes: str = ""
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Category 4: Collaboration trajectories ─────────────────────────────

@dataclass
class ModuleCall:
    """One module invocation within a benchmark run."""
    module: int
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    succeeded: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CollaborationTrajectoryRecord:
    """
    Category 4: full execution trajectory for one benchmark.

    Written by the pipeline orchestrator after a benchmark completes.
    Read by Module 2 for path planning and token efficiency analysis.
    """
    benchmark: str
    path: str                          # "rtl_direct" | "c_first"
    modules_invoked: List[ModuleCall] = field(default_factory=list)
    backtracking_occurred: bool = False
    backtrack_reason: str = ""
    total_tokens: int = 0
    succeeded: bool = True
    notes: str = ""
    timestamp: str = field(default_factory=_now_iso)

    def add_module_call(self, call: ModuleCall) -> None:
        self.modules_invoked.append(call)
        self.total_tokens += call.prompt_tokens + call.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Store key names (used by store.py) ────────────────────────────────

STORE_KEY_PATH_RULES     = "path_selection_rules"
STORE_KEY_DECISIONS      = "decisions"
STORE_KEY_OPT_DECISIONS  = "optimization_decisions"
STORE_KEY_CORRECTIONS    = "correction_patterns"
STORE_KEY_TRAJECTORIES   = "collaboration_trajectories"
STORE_KEY_PPA_OUTCOMES   = "ppa_outcomes"
