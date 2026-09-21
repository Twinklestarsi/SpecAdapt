"""
confidence_rules.py — Post-LLM confidence scoring rules.

After the LLM returns a feature JSON from a spec-only input, this module
cross-checks the raw spec text against the LLM output to determine which
fields are grounded in the spec vs fabricated by the LLM.

Rules are derived empirically from 20 test specs across 5 detail levels
(high / medium / low / partial / vague).  See test_specs_results.json.

Key findings from the empirical test:
  - When the spec explicitly states a feature, the LLM respects it (6/6 pass).
  - When the spec is silent, the LLM fabricates: default widths (32 or 8),
    sequential=True, FSM states, clock domains.
  - Word count alone is insufficient: 4-word precise specs score better than
    61-word rambling ones.
  - The reliable signal is whether the spec text *mentions or negates* a feature.
  - Contradictions cause the LLM to pick the "more complex" interpretation.

This module does NOT call the LLM.  All rules are deterministic regex checks.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, Tuple


# ── Confidence levels ─────────────────────────────────────────────────

class Confidence(str, Enum):
    HIGH = "high"       # explicitly stated in spec or observed in Verilog
    MEDIUM = "medium"   # reasonably inferred from context
    LOW = "low"         # fabricated / defaulted by LLM
    NONE = "none"       # not available or contradicted


# ── Keyword detectors ─────────────────────────────────────────────────
# Each returns True if the spec text contains evidence relevant to a field.

def _mentions_width(spec: str) -> bool:
    """Spec mentions explicit bit widths."""
    return bool(re.search(
        r"\d+[\s-]*bits?"
        r"|\[\s*\d+\s*:\s*\d+\s*\]"
        r"|\d+[\s-]*bit\s+(?:data|width|input|output|port|bus)"
        r"|width\s*(?:=|:|\s+of\s+)\s*\d+",
        spec, re.I,
    ))


def _mentions_clock(spec: str) -> bool:
    """Spec mentions clocking."""
    return bool(re.search(
        r"\bclock\b|\bclk\b|\bclocked\b|\bsynchronous\b"
        r"|\bposedge\b|\bnegedge\b|\brising\s+edge\b"
        r"|\bclock\s+domain\b|\bclock\s+cycle\b"
        r"|\bMHz\b|\bfrequency\b",
        spec, re.I,
    ))


def _mentions_multi_clock(spec: str) -> bool:
    """Spec explicitly describes multiple clock domains."""
    return bool(re.search(
        r"dual[\s-]*clock|two\s+clock|multi[\s-]*clock|cross[\s-]*domain"
        r"|CDC\b|clock\s+domain\s+crossing"
        r"|async(?:hronous)?\s+FIFO"
        r"|\bclk_\w+\b.*\bclk_\w+\b"
        r"|\bwclk\b.*\brclk\b|\brclk\b.*\bwclk\b",
        spec, re.I,
    ))


def _mentions_reset(spec: str) -> bool:
    """Spec mentions reset."""
    return bool(re.search(
        r"\breset\b|\brst\b|\brst_[ni]\b"
        r"|\basync(?:hronous)?\s+reset\b|\bsync(?:hronous)?\s+reset\b"
        r"|\bactive[\s-]*(?:high|low)\s+reset\b",
        spec, re.I,
    ))


def _mentions_fsm(spec: str) -> bool:
    """Spec mentions FSM or state machine concepts."""
    return bool(re.search(
        r"\bFSM\b|\bstate\s+machine\b|\bstates?\b"
        r"|\bIDLE\b|\bstate\s*:\s*\w+"
        r"|\b\d+[\s-]*state\b",
        spec, re.I,
    ))


def _mentions_sequential(spec: str) -> bool:
    """Spec mentions sequential / registered behavior."""
    return bool(re.search(
        r"\bsequential\b|\bregister(?:ed)?\b|\bflip[\s-]*flop\b"
        r"|\blatche?[sd]?\b|\bpipeline\b"
        r"|\bclock\s+cycle\b|\bclock\s+edge\b",
        spec, re.I,
    ))


def _mentions_combinational(spec: str) -> bool:
    """Spec explicitly says combinational / no clock."""
    return bool(re.search(
        r"\bcombinational\b|\bpurely\s+combinational\b"
        r"|\bno\s+clock\b|\bno\s+register\b|\bno\s+clk\b"
        r"|\bunclocked\b|\basynchronous\s+logic\b",
        spec, re.I,
    ))


def _mentions_hierarchy(spec: str) -> bool:
    """Spec mentions sub-modules or instantiation."""
    return bool(re.search(
        r"\binstantiat\w*\b|\bsub[\s-]*module\b|\bwrapper\b"
        r"|\bhierarch\w*\b|\bblack[\s-]*box\b",
        spec, re.I,
    ))


def _mentions_domain_term(spec: str) -> bool:
    """Spec names a recognized digital-design block type.

    These terms give the LLM strong architectural context even without
    hardware keywords like clock/reset/width.  Empirical motivation:
    "round-robin arbiter for 4 requesters" is a clear medium-quality spec
    but triggers zero hardware-keyword detectors.
    """
    return bool(re.search(
        r"\b(?:arbiter|priority\s+encoder|decoder|crossbar|multiplexer"
        r"|mux|demux|FIFO|UART|SPI|I2C|ALU|adder|subtractor"
        r"|comparator|shifter|barrel\s+shifter|CRC|cache"
        r"|DMA|PWM|timer|watchdog|interrupt\s+controller"
        r"|bus\s+bridge|NoC|router|scheduler|scoreboard"
        r"|filter|FIR|IIR|DSP|accumulator|divider|multiplier"
        r"|register\s+file|memory|RAM|ROM|SRAM|DRAM"
        r"|counter|encoder|LFSR|PRBS|ECC"
        r"|CSRNG|DRBG|RNG|AES|SHA|HMAC|entropy|cryptograph\w*"
        r"|FIPS|cipher|hash)\b",
        spec, re.I,
    ))


def _is_hedging(spec: str) -> bool:
    """Spec is dominated by uncertain / hedging language.

    Empirical motivation: "I need something for my project, maybe it needs
    to remember things between clock cycles, I think..." triggers 4 keyword
    categories despite being completely vague.  Hedging markers indicate the
    keywords are aspirational, not definitive.
    """
    hedges = re.findall(
        r"\bmaybe\b|\bprobably\b|\bnot\s+sure\b|\bI\s+think\b"
        r"|\bkind\s+of\b|\bsort\s+of\b|\bperhaps\b|\bmight\b"
        r"|\bnot\s+certain\b|\bI\s+guess\b|\bsomething\s+like\b"
        r"|\bapproximately\b|\baround\b(?=\s+\d)",
        spec, re.I,
    )
    word_count = len(spec.split())
    # If hedging markers appear frequently relative to spec length,
    # the spec is unreliable
    return len(hedges) >= 3 or (len(hedges) >= 2 and word_count < 30)


def _has_contradiction(spec: str) -> bool:
    """Spec contains contradictory statements."""
    comb = _mentions_combinational(spec)
    seq_words = bool(re.search(
        r"\bregistered\b|\bFSM\b|\bstate\s+machine\b|\bflip[\s-]*flop\b",
        spec, re.I,
    ))
    no_clock = bool(re.search(r"\bno\s+clock\b|\bno\s+clk\b", spec, re.I))
    has_clock = _mentions_clock(spec) and not no_clock

    return (comb and seq_words) or (no_clock and has_clock)


# ── Spec quality classification ───────────────────────────────────────

def classify_spec_quality(spec: str) -> Tuple[str, Dict[str, bool]]:
    """
    Classify the overall quality of a spec and return which feature
    categories have textual evidence.

    Returns:
        (quality_level, evidence_map)
        quality_level: "high", "medium", "low", "vague"
        evidence_map: dict of category -> bool
    """
    evidence = {
        "width":         _mentions_width(spec),
        "clock":         _mentions_clock(spec),
        "multi_clock":   _mentions_multi_clock(spec),
        "reset":         _mentions_reset(spec),
        "fsm":           _mentions_fsm(spec),
        "sequential":    _mentions_sequential(spec) or _mentions_clock(spec),
        "combinational": _mentions_combinational(spec),
        "hierarchy":     _mentions_hierarchy(spec),
        "domain_term":   _mentions_domain_term(spec),
        "hedging":       _is_hedging(spec),
        "contradiction": _has_contradiction(spec),
    }

    # Count how many categories have evidence (exclude meta-flags)
    positive = sum(1 for k, v in evidence.items()
                   if v and k not in ("contradiction", "hedging"))

    word_count = len(spec.split())

    # Classification logic
    if evidence["contradiction"]:
        quality = "contradictory"
    elif positive >= 5 and word_count >= 20:
        quality = "high"
    elif positive >= 3 and word_count >= 10:
        quality = "medium"
    elif positive >= 1 and word_count >= 4:
        quality = "low"
    else:
        quality = "vague"

    # Hedging downgrade: specs dominated by uncertain language are unreliable
    # regardless of how many keywords they happen to contain.
    # "maybe it needs clock cycles" ≠ "clocked on system_clk".
    if evidence["hedging"] and quality not in ("contradictory", "vague"):
        quality = "vague"

    return quality, evidence


# ── Per-field confidence scoring ──────────────────────────────────────

def score_confidence(
    spec: str,
    llm_features: Dict[str, Any],
) -> Tuple[Dict[str, str], str, Dict[str, bool]]:
    """
    Score the confidence of each LLM-extracted field by cross-checking
    against the raw spec text.

    Args:
        spec: raw natural-language spec text
        llm_features: dict returned by the LLM

    Returns:
        (field_confidence, overall_confidence, evidence_map)
    """
    quality, evidence = classify_spec_quality(spec)
    conf: Dict[str, str] = {}

    # ── module_name / purpose ──
    # Always derivable if the LLM returned them
    conf["module_name"] = Confidence.HIGH.value if llm_features.get("module_name") else Confidence.LOW.value
    conf["purpose"] = Confidence.HIGH.value if llm_features.get("purpose") else Confidence.LOW.value

    # ── architecture_pattern ──
    # Reliable if spec gives enough context (medium+ quality)
    if quality in ("high", "medium"):
        conf["architecture_pattern"] = Confidence.HIGH.value
    elif quality == "low":
        conf["architecture_pattern"] = Confidence.MEDIUM.value
    else:
        conf["architecture_pattern"] = Confidence.LOW.value

    # ── sub_patterns ──
    conf["sub_patterns"] = Confidence.MEDIUM.value if quality != "vague" else Confidence.LOW.value

    # ── is_sequential ──
    if evidence["combinational"]:
        # Spec explicitly says combinational
        expected = False
        actual = llm_features.get("is_sequential", True)
        if actual == expected:
            conf["is_sequential"] = Confidence.HIGH.value
        else:
            conf["is_sequential"] = Confidence.LOW.value  # LLM contradicted spec
    elif evidence["sequential"] or evidence["clock"]:
        conf["is_sequential"] = Confidence.HIGH.value
    elif quality in ("high", "medium"):
        conf["is_sequential"] = Confidence.MEDIUM.value
    else:
        conf["is_sequential"] = Confidence.LOW.value  # fabricated

    # ── num_clock_domains ──
    if evidence["multi_clock"]:
        conf["num_clock_domains"] = Confidence.HIGH.value
    elif evidence["clock"] and not evidence["multi_clock"]:
        # Mentioned clock but not multi-clock → 1 domain is reasonable
        conf["num_clock_domains"] = Confidence.MEDIUM.value
    else:
        conf["num_clock_domains"] = Confidence.LOW.value

    # ── reset_type ──
    if evidence["reset"]:
        conf["reset_type"] = Confidence.HIGH.value
    elif evidence["clock"]:
        conf["reset_type"] = Confidence.MEDIUM.value  # clocked designs usually have reset
    else:
        conf["reset_type"] = Confidence.LOW.value

    # ── has_fsm / estimated_fsm_states ──
    if evidence["fsm"]:
        conf["has_fsm"] = Confidence.HIGH.value
        # If spec mentions specific state count
        state_nums = re.findall(r"(\d+)[\s-]*states?", spec, re.I)
        if state_nums:
            conf["estimated_fsm_states"] = Confidence.HIGH.value
        else:
            conf["estimated_fsm_states"] = Confidence.MEDIUM.value
    elif evidence["combinational"]:
        # Combinational → no FSM is reliable
        if not llm_features.get("has_fsm", False):
            conf["has_fsm"] = Confidence.HIGH.value
        else:
            conf["has_fsm"] = Confidence.LOW.value  # LLM fabricated FSM
        conf["estimated_fsm_states"] = Confidence.HIGH.value
    else:
        # Spec doesn't mention FSM → LLM may have fabricated
        if llm_features.get("has_fsm", False):
            conf["has_fsm"] = Confidence.LOW.value
            conf["estimated_fsm_states"] = Confidence.LOW.value
        else:
            conf["has_fsm"] = Confidence.MEDIUM.value
            conf["estimated_fsm_states"] = Confidence.MEDIUM.value

    # ── key_operations ──
    if quality in ("high", "medium"):
        conf["key_operations"] = Confidence.HIGH.value
    elif quality == "low":
        conf["key_operations"] = Confidence.MEDIUM.value
    else:
        conf["key_operations"] = Confidence.LOW.value

    # ── data_widths ──
    if evidence["width"]:
        conf["data_widths"] = Confidence.HIGH.value
    else:
        conf["data_widths"] = Confidence.LOW.value  # always fabricated

    # ── complexity ──
    if quality in ("high", "medium"):
        conf["complexity"] = Confidence.MEDIUM.value  # always somewhat subjective
    else:
        conf["complexity"] = Confidence.LOW.value

    # ── hierarchy ──
    if evidence["hierarchy"]:
        conf["hierarchy"] = Confidence.HIGH.value
    elif quality in ("high", "medium"):
        conf["hierarchy"] = Confidence.MEDIUM.value
    else:
        conf["hierarchy"] = Confidence.LOW.value

    # ── suggested_subcategory ──
    if quality in ("high", "medium"):
        conf["suggested_subcategory"] = Confidence.HIGH.value
    elif quality == "low":
        conf["suggested_subcategory"] = Confidence.MEDIUM.value
    else:
        conf["suggested_subcategory"] = Confidence.LOW.value

    # ── optimization_notes ──
    conf["optimization_notes"] = Confidence.MEDIUM.value if quality != "vague" else Confidence.LOW.value
    conf["interface"] = Confidence.HIGH.value if llm_features.get("interface", {}).get("ports") else Confidence.LOW.value
    contract = llm_features.get("behavioral_contract", {}) or {}
    conf["behavioral_contract"] = (
        Confidence.HIGH.value
        if contract.get("cycle_semantics")
        or contract.get("registered_outputs")
        or contract.get("combinational_outputs")
        else Confidence.LOW.value
    )

    # ── Overall confidence ──
    if evidence["contradiction"]:
        overall = Confidence.LOW.value
    elif quality == "high":
        overall = Confidence.HIGH.value
    elif quality in ("medium", "low"):
        overall = Confidence.MEDIUM.value
    else:
        overall = Confidence.LOW.value

    return conf, overall, evidence
