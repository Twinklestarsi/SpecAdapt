"""
rules.py — Deterministic (rule-based) path selection heuristics.

Tier 1 of the two-tier path selector.  Pure functions: no I/O, no state,
no LLM calls.  Takes a feature dict (from FeatureResult.llm_features or
merged with regex_features) and returns a RuleVerdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


# ── Verdict ───────────────────────────────────────────────────────────

@dataclass
class RuleVerdict:
    """Outcome of Tier-1 rule evaluation."""
    path: str              # "rtl_direct" | "c_first" | "uncertain"
    rule_fired: str        # name of the decisive rule
    reason: str            # human-readable explanation
    confidence: str        # "high" | "medium" | "low"


# ── Behavioral patterns that are good C-first candidates ──────────────
C_FIRST_SUBCATEGORIES = {"arithmetic", "logical", "selection"}
DATA_PROCESSING_OPS = {
    "arithmetic", "bitwise", "shift", "comparison", "mux",
}
MEMORY_PROTOCOL_OPS = {"memory_access", "serial_protocol"}
MEMORY_PROTOCOL_PATTERNS = {
    "protocol_handler", "handshake_interface",
}

# ── Structural patterns that should remain RTL-direct ─────────────────
SKIP_C_SUBCATEGORIES = {"constant"}


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _has_any(values: Any, candidates: set[str]) -> bool:
    return any(str(v).lower() in candidates for v in _as_list(values))


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clock_domains(f: Dict[str, Any]) -> int:
    num_clocks = _to_int(f.get("num_clock_domains", 1), default=1)
    regex_clocks = _to_int(f.get("_regex_num_clock_domains", 0), default=0)
    return max(num_clocks, regex_clocks)


def _has_effective_fsm(f: Dict[str, Any]) -> bool:
    return bool(f.get("has_fsm", False) or f.get("_regex_has_fsm", False))


def _has_data_processing_behavior(f: Dict[str, Any]) -> bool:
    subcat = (f.get("suggested_subcategory") or "").lower()
    return (
        subcat in C_FIRST_SUBCATEGORIES
        or _has_any(f.get("key_operations", []), DATA_PROCESSING_OPS)
    )


def _has_memory_protocol_behavior(f: Dict[str, Any]) -> bool:
    pattern = (f.get("architecture_pattern") or "").lower()
    return (
        pattern == "memory"
        or _has_any(f.get("key_operations", []), MEMORY_PROTOCOL_OPS)
        or _has_any(f.get("sub_patterns", []), MEMORY_PROTOCOL_PATTERNS)
    )


# ── Individual rule functions ─────────────────────────────────────────
# Each returns (matched: bool, rule_name: str, reason: str, confidence: str)
# Rules are evaluated in priority order by apply_rules().

def _rule_multi_clock_protection(f: Dict[str, Any]) -> Optional[RuleVerdict]:
    """Multi-clock/CDC designs stay RTL-direct to preserve clock semantics."""
    effective = _clock_domains(f)
    if effective >= 2:
        return RuleVerdict(
            path="rtl_direct",
            rule_fired="multi_clock_protection",
            reason=f"Design has {effective} clock domains; C behavioral modeling can "
                   "weaken clock-domain and CDC semantics.",
            confidence="high",
        )
    return None


def _rule_structural_simple_protection(f: Dict[str, Any]) -> Optional[RuleVerdict]:
    """Strongly structural or extremely simple designs remain RTL-direct."""
    hierarchy = (f.get("hierarchy") or "").lower()
    subcat = (f.get("suggested_subcategory") or "").lower()
    complexity = (f.get("complexity") or "").lower()

    if hierarchy == "wrapper_only":
        return RuleVerdict(
            path="rtl_direct",
            rule_fired="structural_simple_protection",
            reason="Design is wrapper-only; C behavioral modeling may flatten "
                   "RTL structural connectivity.",
            confidence="high",
        )
    if subcat in SKIP_C_SUBCATEGORIES:
        return RuleVerdict(
            path="rtl_direct",
            rule_fired="structural_simple_protection",
            reason=f"Subcategory '{subcat}' is constant-like; direct RTL is "
                   "shorter and more deterministic.",
            confidence="high",
        )
    if (
        complexity == "trivial"
        and not _has_effective_fsm(f)
        and not _has_data_processing_behavior(f)
        and not _has_memory_protocol_behavior(f)
    ):
        return RuleVerdict(
            path="rtl_direct",
            rule_fired="structural_simple_protection",
            reason="Design is extremely simple and stateless; direct RTL is already clear.",
            confidence="high",
        )
    return None


# ── Rule priority list ────────────────────────────────────────────────
# Rules are evaluated top-to-bottom; first match wins.

_HARD_RULES = [
    _rule_multi_clock_protection,
    _rule_structural_simple_protection,
]


# ── RuleSet ───────────────────────────────────────────────────────────

class RuleSet:
    """
    Holds the two hard RTL-direct safety rules for Module 2.

    Soft path guidance lives in Memory Agent path policies. The legacy
    soft-rule functions remain only for compatibility with old artifacts.
    """

    def __init__(self) -> None:
        self._rules = list(_HARD_RULES)

    def load_from_memory(self, memory_rules: list) -> None:
        """Legacy JSON rules cannot override Module 2 safety constraints."""
        self._rules = list(_HARD_RULES)

    def apply(
        self,
        llm_features: Dict[str, Any],
        overall_confidence: str = "medium",
        regex_features: Optional[Dict[str, Any]] = None,
        optimization_target: str = "",
    ) -> RuleVerdict:
        """Evaluate rules in priority order; return first match."""
        merged: Dict[str, Any] = dict(llm_features)
        merged["_overall_confidence"] = overall_confidence
        merged["_optimization_target"] = optimization_target

        if regex_features:
            merged["_regex_num_clock_domains"] = regex_features.get("num_clock_domains")
            merged["_regex_is_sequential"]     = regex_features.get("is_sequential")
            merged["_regex_has_fsm"]           = regex_features.get("has_fsm")

        for rule_fn in self._rules:
            verdict = rule_fn(merged)
            if verdict is not None:
                return verdict

        return RuleVerdict(
            path="uncertain",
            rule_fired="no_rule_matched",
            reason="No Tier-1 rule fired decisively; routing to Tier-2 LLM judgment.",
            confidence="low",
        )

    @staticmethod
    def _merged_features(
        llm_features: Dict[str, Any],
        overall_confidence: str,
        regex_features: Optional[Dict[str, Any]],
        optimization_target: str,
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(llm_features)
        merged["_overall_confidence"] = overall_confidence
        merged["_optimization_target"] = optimization_target
        if regex_features:
            merged["_regex_num_clock_domains"] = regex_features.get("num_clock_domains")
            merged["_regex_is_sequential"] = regex_features.get("is_sequential")
            merged["_regex_has_fsm"] = regex_features.get("has_fsm")
        return merged

    def apply_hard(
        self,
        llm_features: Dict[str, Any],
        overall_confidence: str = "medium",
        regex_features: Optional[Dict[str, Any]] = None,
        optimization_target: str = "",
    ) -> RuleVerdict:
        merged = self._merged_features(
            llm_features, overall_confidence, regex_features, optimization_target
        )
        for rule_fn in _HARD_RULES:
            verdict = rule_fn(merged)
            if verdict is not None:
                return verdict
        return RuleVerdict("uncertain", "no_hard_rule", "No hard rule matched.", "low")

    def apply_non_hard(
        self,
        llm_features: Dict[str, Any],
        overall_confidence: str = "medium",
        regex_features: Optional[Dict[str, Any]] = None,
        optimization_target: str = "",
    ) -> RuleVerdict:
        return RuleVerdict(
            "uncertain",
            "soft_rules_migrated_to_memory",
            "Soft path guidance is provided by Memory Agent policies.",
            "low",
        )


# Module-level default instance (used by apply_rules() convenience function)
_default_ruleset = RuleSet()


# ── Public API ────────────────────────────────────────────────────────

def apply_rules(
    llm_features: Dict[str, Any],
    overall_confidence: str = "medium",
    regex_features: Optional[Dict[str, Any]] = None,
    optimization_target: str = "",
) -> RuleVerdict:
    """
    Convenience wrapper: applies the default RuleSet to the given features.
    PathSelector uses its own RuleSet instance (self._ruleset) so that
    Module 8 can inject learned rules without affecting other callers.
    """
    return _default_ruleset.apply(
        llm_features,
        overall_confidence,
        regex_features,
        optimization_target=optimization_target,
    )
