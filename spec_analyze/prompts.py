"""
prompts.py — Prompt templates for LLM-based feature extraction.

Prompt JSON templates are auto-generated from schema.py so they stay
in sync with the feature definitions.
"""

from __future__ import annotations

from spec_analyze.schema import llm_json_template


# ── System prompt (shared) ────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert RTL/FPGA design analyst. You analyze Verilog modules "
    "or hardware specifications and extract structured features for a "
    "downstream optimization pipeline.\n\n"
    "Your analysis must be precise and based only on what is present in the "
    "source code or specification. Output valid JSON only — no markdown "
    "fences, no extra text before or after the JSON."
)


# ── Build prompt strings at module load time ──────────────────────────
# We use plain string concatenation to embed the JSON template, and
# leave {verilog_src} / {spec_text} as literal placeholders for
# .replace() at call time.

_JSON_TEMPLATE = llm_json_template()

VERILOG_PROMPT = (
    "Analyze the following Verilog module and extract structured features.\n\n"
    "The interface field must reproduce every top-level port exactly once, "
    "including exact name, direction, width, clock edge, and reset semantics.\n\n"
    "The behavioral_contract field must describe whether each output is registered "
    "or combinational and preserve any stated cycle-after, same-cycle, sampling-edge, "
    "or state-update semantics.\n\n"
    "Return a JSON object with exactly these fields:\n\n"
    + _JSON_TEMPLATE + "\n\n"
    "Verilog source:\n"
    "```verilog\n"
    "{verilog_src}\n"
    "```\n\n"
    "Respond with the JSON object only."
)

SPEC_PROMPT = (
    "Analyze the following natural-language hardware specification and extract "
    "structured features as if the design were already implemented.\n\n"
    "The natural-language specification is the primary authority. Distinguish "
    "facts explicitly stated by the specification from engineering assumptions. "
    "For fields you cannot determine from the spec alone, use the schema default "
    "when it supports unknown/none; otherwise choose the smallest conservative "
    "assumption and state that assumption explicitly in optimization_notes or "
    "behavioral_contract.cycle_semantics. Never present an inferred reset style, "
    "latency, or output-register classification as if the specification stated it.\n\n"
    "The interface field is a hard contract: include every top-level port exactly once, "
    "preserve exact names and widths, and identify clock/reset ports and semantics. "
    "Do not invent AXI or HLS control ports.\n\n"
    "The behavioral_contract field is also a hard contract. Phrases such as "
    "'on the cycle after', 'updated on the rising edge', and 'registered output' "
    "must be represented explicitly; do not collapse them into combinational behavior. "
    "Audit valid/data alignment, enable bubbles, back-to-back inputs, FSM priority, "
    "and counter terminal conditions before responding.\n\n"
    "Return a JSON object with exactly these fields:\n\n"
    + _JSON_TEMPLATE + "\n\n"
    "Specification:\n"
    "{spec_text}\n\n"
    "Respond with the JSON object only."
)

MIXED_PROMPT = (
    "Analyze the following Verilog module together with its specification. "
    "Use both sources to produce the most accurate feature extraction.\n\n"
    "The interface field must reproduce every Verilog top-level port exactly once, "
    "including exact name, direction, width, clock edge, and reset semantics.\n\n"
    "Use the RTL and specification to fill behavioral_contract with exact registered "
    "outputs, combinational outputs, output latencies, and state-update semantics.\n\n"
    "Return a JSON object with exactly these fields:\n\n"
    + _JSON_TEMPLATE + "\n\n"
    "Specification:\n"
    "{spec_text}\n\n"
    "Verilog source:\n"
    "```verilog\n"
    "{verilog_src}\n"
    "```\n\n"
    "Respond with the JSON object only."
)


def format_verilog_prompt(verilog_src: str) -> str:
    return VERILOG_PROMPT.replace("{verilog_src}", verilog_src)


def format_spec_prompt(spec_text: str) -> str:
    return SPEC_PROMPT.replace("{spec_text}", spec_text)


def format_mixed_prompt(verilog_src: str, spec_text: str) -> str:
    return MIXED_PROMPT.replace("{verilog_src}", verilog_src).replace("{spec_text}", spec_text)
