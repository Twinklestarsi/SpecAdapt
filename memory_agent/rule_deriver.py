"""
rule_deriver.py — Analyze accumulated path decisions + PPA outcomes
to derive new condition-dict rules for Module 2.

Algorithm
---------
1. Collect completed decisions: records that have ppa_outcome filled in
   by Module 7 (area_improvement_pct available, synthesis_failed=False).

2. At bootstrap (no completed PPA data), return empty list — seed rules
   in path_select/rules.py handle all cases.

3. When enough completed decisions exist, group by key feature dimensions
   and compute per-path win rates within each group.

4. Emit a condition-dict rule for groups where:
   - evidence_count >= MIN_EVIDENCE
   - win_rate >= WIN_THRESHOLD  (one path clearly dominates)
   - the group is not already covered by a seed rule (avoid conflicts)

Win metric
----------
  For timing objectives, ``c_first wins`` if delay_improvement_ps > 0
  (positive means the critical-path delay is shorter).

  For area objectives, ``c_first wins`` if area_improvement_pct < -AREA_WIN_THRESHOLD.
  Slack is retained only as a legacy feasibility diagnostic.

  "rtl_direct wins" otherwise when the corresponding objective metric is
  available and does not improve.

When both paths have been tried on similar designs, the path with the
higher win rate becomes the derived rule's decision.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ── Thresholds ────────────────────────────────────────────────────────

MIN_EVIDENCE    = 3     # minimum decisions needed to emit a rule
WIN_THRESHOLD   = 0.70  # fraction of wins required (e.g. 70%)
AREA_WIN_THRESHOLD = 5.0  # % area reduction counts as a "c_first win"

# ── Feature dimensions used for grouping ─────────────────────────────
# Subset of _PROFILE_FIELDS that are most predictive and low-cardinality.

_GROUP_FIELDS = [
    "suggested_subcategory",
    "architecture_pattern",
    "complexity",
    "is_sequential",
    "num_clock_domains",
]


# ── Seed rule coverage (to avoid emitting conflicting rules) ──────────
# These patterns are already deterministically handled in rules.py.
# If a derived group matches any of these, skip it.

def _is_covered_by_seed(profile: Dict[str, Any]) -> bool:
    """Return True if this feature profile is deterministically handled."""
    # Multi-clock
    if (profile.get("num_clock_domains") or 1) >= 2:
        return True
    # Combinational
    if (profile.get("architecture_pattern") or "").lower() == "combinational":
        return True
    # Constant subcategory
    if (profile.get("suggested_subcategory") or "").lower() == "constant":
        return True
    return False


# ── Win evaluation ────────────────────────────────────────────────────

def _is_c_first_win(ppa: Dict[str, Any], objective: str = "") -> Optional[bool]:
    """
    Return True if c_first clearly won, False if rtl_direct won, None if unknown.
    """
    if ppa is None:
        return None
    objective = str(objective or ppa.get("objective") or "").strip().upper()
    delay_gain = ppa.get("timing_improvement_ps")
    if delay_gain is None:
        delay_gain = ppa.get("delay_improvement_ps")
    if objective == "TIMING" or delay_gain is not None:
        if delay_gain is None:
            return None
        try:
            return float(delay_gain) > 0.0
        except (TypeError, ValueError):
            return None

    area = ppa.get("area_improvement_pct")
    if area is None:
        return None
    # Legacy area path rule: the area field uses the historical negative-good
    # convention.  Slack is deliberately not consulted for timing decisions.
    try:
        return float(area) < -AREA_WIN_THRESHOLD
    except (TypeError, ValueError):
        return None


# ── GroupKey ──────────────────────────────────────────────────────────

def _group_key(profile: Dict[str, Any]) -> Tuple:
    """Extract grouping tuple from a feature profile."""
    return tuple(
        profile.get(f) for f in _GROUP_FIELDS
    )


def _profile_to_conditions(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a representative feature profile into a conditions dict
    for the derived rule.
    Only include fields that were used for grouping (non-None).
    """
    conditions: Dict[str, Any] = {}
    for f in _GROUP_FIELDS:
        v = profile.get(f)
        if v is None:
            continue
        if f == "num_clock_domains":
            # Use lte to handle any single-clock design
            conditions[f] = {"lte": int(v)}
        else:
            conditions[f] = v
    return conditions


# ── Main deriver ──────────────────────────────────────────────────────

def derive_path_rules(
    decisions: List[Dict[str, Any]],
    existing_rules: Optional[List[Dict[str, Any]]] = None,
    min_evidence: int = MIN_EVIDENCE,
    win_threshold: float = WIN_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Analyze path decisions and PPA outcomes to produce condition-dict rules.

    Args:
        decisions:       List of decision records from the staging file.
                         Records with ppa_outcome filled are "completed".
        existing_rules:  Currently active rules (to avoid duplicates).
        min_evidence:    Minimum completed decisions per group to emit a rule.
        win_threshold:   Minimum win rate (0-1) to emit a rule.

    Returns:
        List of condition-dicts ready to write to "path_selection_rules".
    """
    # Group completed decisions by feature profile key
    # Each group: {path → [win_bool, ...]}
    groups: Dict[Tuple, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
    profiles: Dict[Tuple, Dict[str, Any]] = {}  # representative profile per group

    for dec in decisions:
        ppa = dec.get("ppa_outcome")
        if ppa is None:
            continue  # not completed yet
        profile = dec.get("feature_profile", {})
        if _is_covered_by_seed(profile):
            continue  # skip seed-covered cases

        key = _group_key(profile)
        path = dec.get("path", "rtl_direct")
        win = _is_c_first_win(
            ppa,
            str(dec.get("optimization_target") or ppa.get("objective") or ""),
        )
        if win is None:
            continue

        groups[key][path].append(win)
        if key not in profiles:
            profiles[key] = profile  # save first-seen as representative

    if not groups:
        return []  # bootstrap: no completed data yet

    rules: List[Dict[str, Any]] = []
    priority_counter = 10  # start at 10, below seed rules' implicit priority

    for key, path_wins in groups.items():
        profile = profiles[key]
        conditions = _profile_to_conditions(profile)
        if not conditions:
            continue

        # Compute win rate for c_first
        c_first_wins = path_wins.get("c_first", [])
        rtl_wins     = path_wins.get("rtl_direct", [])
        total_c  = len(c_first_wins)
        total_r  = len(rtl_wins)
        evidence = total_c + total_r

        if evidence < min_evidence:
            continue

        c_first_win_rate = (
            sum(c_first_wins) / total_c if total_c else 0.0
        )
        rtl_win_rate = (
            sum(1 - w for w in rtl_wins) / total_r if total_r else 0.0
        )

        if c_first_win_rate >= win_threshold:
            decision   = "c_first"
            win_rate   = c_first_win_rate
            confidence = "high" if win_rate >= 0.90 else "medium"
        elif rtl_win_rate >= win_threshold:
            decision   = "rtl_direct"
            win_rate   = rtl_win_rate
            confidence = "high" if win_rate >= 0.90 else "medium"
        else:
            continue  # no clear winner — don't emit a rule

        # Build rule name from key fields
        subcat  = profile.get("suggested_subcategory", "unknown")
        arch    = profile.get("architecture_pattern", "")
        compl   = profile.get("complexity", "")
        name    = f"learned_{subcat}_{compl}_{decision}".replace(" ", "_").lower()

        rule: Dict[str, Any] = {
            "name":           name,
            "priority":       priority_counter,
            "conditions":     conditions,
            "decision":       decision,
            "confidence":     confidence,
            "evidence_count": evidence,
            "win_rate":       round(win_rate, 4),
            "source":         "module8_learned",
        }
        rules.append(rule)
        priority_counter += 10

    # Sort by evidence descending so most-supported rules get lower priority numbers
    rules.sort(key=lambda r: -r["evidence_count"])
    for i, r in enumerate(rules):
        r["priority"] = 10 + i * 10

    return rules
