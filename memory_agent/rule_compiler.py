"""
rule_compiler.py — Compile condition-dicts (from the staging file) into
Python callables that path_select.rules.RuleSet.load_from_memory() accepts.

This is the bridge that closes the Module 2 ↔ Module 8 feedback loop.

Condition-dict format (written by rule_deriver.py):
{
  "name":           str,           # unique rule id
  "priority":       int,           # lower = higher priority (prepended in order)
  "conditions": {
    "field_name":   value,         # exact equality (eq)
    "field_name":   {"lte": n},    # <=
    "field_name":   {"gte": n},    # >=
    "field_name":   {"in":  [...]} # membership
    "field_name":   {"not_in": [...]}
  },
  "decision":       "rtl_direct" | "c_first",
  "confidence":     "high" | "medium" | "low",
  "evidence_count": int,           # how many decisions this is based on
  "win_rate":       float,         # fraction of benchmarks where this path won
  "source":         "module8_learned"
}

A rule compiled from this dict has the same signature as the seed rules
in path_select/rules.py:
    (merged_features: Dict[str, Any]) -> Optional[RuleVerdict]

Usage (from path_select/selector.py after Module 8 is ready):
    from memory_agent.rule_compiler import compile_rules
    compiled = compile_rules(rule_dicts)
    self._ruleset.load_from_memory(compiled)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# Import RuleVerdict from path_select.  This is a clean import because
# memory_agent is a sibling package and path_select is already on sys.path
# when running from the project root.
from path_select.rules import RuleVerdict


# ── Supported condition operators ──────────────────────────────────────

_OPS = {"lte", "gte", "in", "not_in"}


def _check_condition(value: Any, condition: Any) -> bool:
    """
    Evaluate one field condition against a feature value.

    condition is either:
      - a scalar (exact equality), or
      - a single-key dict {"op": operand}
    """
    if isinstance(condition, dict):
        if len(condition) != 1:
            return False  # malformed — skip rule
        op, operand = next(iter(condition.items()))
        if op == "lte":
            return value is not None and value <= operand
        if op == "gte":
            return value is not None and value >= operand
        if op == "in":
            return value in operand
        if op == "not_in":
            return value not in operand
        return False  # unknown op — skip
    # Scalar: exact equality
    return value == condition


def compile_rule(rule_dict: Dict[str, Any]) -> Callable:
    """
    Convert one condition-dict into a rule callable compatible with
    path_select.rules._RULES list signature.

    The compiled function is a closure over the rule fields.
    It returns a RuleVerdict if ALL conditions match, else None.
    """
    name       = rule_dict.get("name", "unnamed_learned_rule")
    conditions = rule_dict.get("conditions", {})
    decision   = rule_dict.get("decision", "rtl_direct")
    confidence = rule_dict.get("confidence", "medium")
    win_rate   = rule_dict.get("win_rate", 0.0)
    evidence   = rule_dict.get("evidence_count", 0)

    reason_template = (
        f"Module 8 learned rule '{name}' matched "
        f"(win_rate={win_rate:.0%}, evidence={evidence})."
    )

    def rule_fn(merged: Dict[str, Any]) -> Optional[RuleVerdict]:
        for field, condition in conditions.items():
            value = merged.get(field)
            if not _check_condition(value, condition):
                return None
        return RuleVerdict(
            path=decision,
            rule_fired=name,
            reason=reason_template,
            confidence=confidence,
        )

    rule_fn.__name__ = name
    return rule_fn


def compile_rules(rule_dicts: List[Dict[str, Any]]) -> List[Callable]:
    """
    Compile a list of condition-dicts into callables.

    Rules are sorted by "priority" (ascending) before compilation so
    that lower priority numbers fire first when prepended into RuleSet.
    Rules with compilation errors are skipped with a warning.
    """
    import sys
    sorted_dicts = sorted(rule_dicts, key=lambda r: r.get("priority", 99))
    compiled = []
    for rd in sorted_dicts:
        try:
            compiled.append(compile_rule(rd))
        except Exception as exc:
            print(
                f"[rule_compiler] Warning: could not compile rule "
                f"'{rd.get('name', '?')}': {exc}",
                file=sys.stderr,
            )
    return compiled
