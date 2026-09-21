"""
schema.py — Feature schema definition, validation, and confidence scoring.

Single source of truth for the feature fields. The prompt templates in
prompts.py are generated from this schema so they never drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ── Confidence levels ─────────────────────────────────────────────────

class Confidence(str, Enum):
    HIGH = "high"       # directly observed or explicitly stated
    MEDIUM = "medium"   # reasonably inferred
    LOW = "low"         # guessed / default
    NONE = "none"       # not available


_INTERFACE_DEFAULT: Dict[str, Any] = {
    "ports": [],
    "clock": {"name": "", "edge": "none"},
    "reset": {"name": "", "active_level": "none", "type": "none"},
}

_BEHAVIORAL_CONTRACT_DEFAULT: Dict[str, Any] = {
    "registered_outputs": [],
    "combinational_outputs": [],
    "output_latency_cycles": {},
    "state_update_edge": "none",
    "cycle_semantics": "",
}


# ── Feature field definitions ─────────────────────────────────────────
# Each tuple: (field_name, type_hint, description, default_value)
# These define what the LLM should return.

LLM_FEATURE_FIELDS: List[Tuple[str, str, str, Any]] = [
    ("module_name",           "str",       "Top module name",                                  ""),
    ("purpose",               "str",       "One-sentence functional description",               ""),
    ("architecture_pattern",  "str",       "Primary pattern (prefer a specific label; use "
                                           "mixed only when two or more patterns contribute "
                                           "equally): fsm, pipeline, datapath, control_logic, "
                                           "memory, combinational, mixed",                      "mixed"),
    ("sub_patterns",          "list[str]", "Zero or more from: counter, shift_register, "
                                           "mux_tree, encoder, decoder, arbiter, "
                                           "protocol_handler, crypto_block, "
                                           "handshake_interface",                               []),
    ("is_sequential",         "bool",      "True if clocked / has registers",                   False),
    ("num_clock_domains",     "int",       "Number of independent clock domains",               0),
    ("reset_type",            "str",       "One of: async, sync, both, none, unknown",          "unknown"),
    ("has_fsm",               "bool",      "True if contains a finite state machine",           False),
    ("estimated_fsm_states",  "int",       "Number of FSM states (0 if no FSM)",                0),
    ("key_operations",        "list[str]", "Dominant operations: arithmetic, bitwise, shift, "
                                           "comparison, mux, memory_access, serial_protocol",   []),
    ("data_widths",           "list[int]", "Distinct data widths used (bits), e.g. "
                                           "[8, 32], empty if unknown",                         []),
    ("complexity",            "str",       "One of: trivial, simple, moderate, complex",        "moderate"),
    ("hierarchy",             "str",       "One of: flat, instantiates_submodules, "
                                           "wrapper_only",                                      "flat"),
    ("suggested_subcategory", "str",       "Best fit: arithmetic, bit_reorganization, "
                                           "constant, logical, selection",                      "arithmetic"),
    ("optimization_notes",    "str",       "1-2 sentences on what transforms might help",       ""),
    ("interface",             "interface", "Exact complete top-level port contract, including "
                                           "each port name, direction, bit width, and role; plus "
                                           "clock edge and reset polarity/type",                  _INTERFACE_DEFAULT),
    ("behavioral_contract",   "behavioral_contract", "Cycle-level behavior contract: identify "
                                           "registered versus combinational outputs, per-output "
                                           "latency, the state-update edge, and concise state/output "
                                           "update semantics",                                  _BEHAVIORAL_CONTRACT_DEFAULT),
]


def llm_json_template() -> str:
    """Generate the JSON template string to embed in prompts."""
    lines = ["{"]
    for i, (name, type_hint, desc, default) in enumerate(LLM_FEATURE_FIELDS):
        # Show the type as a placeholder value
        if type_hint == "str":
            val = f'"<{desc}>"'
        elif type_hint == "bool":
            val = "<true/false>"
        elif type_hint == "int":
            val = f"<int>"
        elif type_hint.startswith("list"):
            val = f'["<{desc}>"]'
        elif type_hint == "interface":
            val = (
                '{"ports": [{"name": "<exact port name>", '
                '"direction": "input|output|inout", "width": 1, '
                '"role": "data|clock|reset"}], '
                '"clock": {"name": "<clock port or empty>", '
                '"edge": "posedge|negedge|none"}, '
                '"reset": {"name": "<reset port or empty>", '
                '"active_level": "low|high|none", '
                '"type": "async|sync|none"}}'
            )
        elif type_hint == "behavioral_contract":
            val = (
                '{"registered_outputs": ["<output name>"], '
                '"combinational_outputs": ["<output name>"], '
                '"output_latency_cycles": {"<output name>": 0}, '
                '"state_update_edge": "posedge|negedge|none", '
                '"cycle_semantics": "<concise sample/update relationship>"}'
            )
        else:
            val = f'"<{type_hint}>"'
        comma = "," if i < len(LLM_FEATURE_FIELDS) - 1 else ""
        lines.append(f'  "{name}": {val}{comma}')
    lines.append("}")
    return "\n".join(lines)


# ── Regex feature field names ─────────────────────────────────────────

REGEX_FEATURE_FIELDS: List[str] = [
    "loc", "loc_nonblank", "num_modules_defined", "module_name",
    "num_always_blocks", "num_instantiations",
    "num_inputs", "num_outputs", "num_ports",
    "total_input_bits", "total_output_bits",
    "max_input_width", "max_output_width",
    "is_sequential", "has_async_reset", "has_sync_reset",
    "has_signed", "num_regs",
    "has_fsm", "num_fsm_states",
    "num_clock_domains", "clock_signals",
    "has_generate",
    "num_add_sub", "num_multiply", "num_divide_mod",
    "num_shift", "num_bitwise", "num_reduction",
    "num_comparison", "num_ternary", "num_logical_and_or",
    "num_if", "num_else", "num_case", "num_assign_stmt",
    "max_nesting_depth",
    "num_numeric_constants", "has_concatenation", "has_bit_select",
    "num_params",
    "mux_density", "arithmetic_density", "control_ratio",
]


# ── Unified result ────────────────────────────────────────────────────

@dataclass
class FeatureResult:
    """Unified output of Stage 1 analysis."""
    benchmark: str
    input_type: str                                # "verilog", "spec", "mixed"
    optimization_target: str = ""                  # "AREA" | "TIMING"
    verilog_path: Optional[str] = None
    spec_text: Optional[str] = None

    # LLM-extracted semantic features
    llm_features: Dict[str, Any] = field(default_factory=dict)

    # Per-feature confidence (field_name -> Confidence)
    confidence: Dict[str, str] = field(default_factory=dict)

    # Overall confidence for the analysis
    overall_confidence: str = Confidence.MEDIUM.value

    # Regex-extracted numeric features (None if input is spec-only)
    regex_features: Optional[Dict[str, Any]] = None

    # Token usage returned by the LLM API, if the endpoint provides it.
    token_usage: Dict[str, Any] = field(default_factory=dict)

    # Raw LLM response (for debugging)
    llm_raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_spec_only(self) -> bool:
        return self.input_type == "spec"


# ── Enum constraints ─────────────────────────────────────────────────
# Allowed values for string-enum fields. Used by validate_features()
# to reject out-of-vocabulary LLM outputs.

_ENUM_ALLOWED: Dict[str, set] = {
    "architecture_pattern": {"fsm", "pipeline", "datapath", "control_logic",
                             "memory", "combinational", "mixed"},
    "reset_type":           {"async", "sync", "both", "none", "unknown"},
    "complexity":           {"trivial", "simple", "moderate", "complex"},
    "hierarchy":            {"flat", "instantiates_submodules", "wrapper_only"},
    "suggested_subcategory": {"arithmetic", "bit_reorganization", "constant",
                              "logical", "selection"},
}


def normalize_interface(value: Any) -> Dict[str, Any]:
    """Validate the exact interface contract returned by the feature model."""
    if not isinstance(value, dict):
        value = {}

    ports: List[Dict[str, Any]] = []
    seen = set()
    for raw in value.get("ports", []) if isinstance(value.get("ports", []), list) else []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        direction = str(raw.get("direction", raw.get("dir", ""))).strip().lower()
        role = str(raw.get("role", "data")).strip().lower()
        try:
            width = int(raw.get("width", 1))
        except (TypeError, ValueError):
            width = 0
        if not name or name in seen or direction not in {"input", "output", "inout"} or width <= 0:
            continue
        if role not in {"data", "clock", "reset"}:
            role = "data"
        ports.append({"name": name, "direction": direction, "width": width, "role": role})
        seen.add(name)

    raw_clock = value.get("clock", {}) if isinstance(value.get("clock", {}), dict) else {}
    clock_name = str(raw_clock.get("name", "")).strip()
    clock_edge = str(raw_clock.get("edge", "none")).strip().lower()
    if clock_edge not in {"posedge", "negedge", "none"}:
        clock_edge = "none"

    raw_reset = value.get("reset", {}) if isinstance(value.get("reset", {}), dict) else {}
    reset_name = str(raw_reset.get("name", "")).strip()
    reset_level = str(raw_reset.get("active_level", "none")).strip().lower()
    reset_type = str(raw_reset.get("type", "none")).strip().lower()
    if reset_level not in {"low", "high", "none"}:
        reset_level = "none"
    if reset_type not in {"async", "sync", "none"}:
        reset_type = "none"

    for port in ports:
        if port["role"] == "clock" and not clock_name:
            clock_name = port["name"]
        elif port["role"] == "reset" and not reset_name:
            reset_name = port["name"]
        if port["name"] == clock_name:
            port["role"] = "clock"
        elif port["name"] == reset_name:
            port["role"] = "reset"

    return {
        "ports": ports,
        "clock": {"name": clock_name, "edge": clock_edge},
        "reset": {"name": reset_name, "active_level": reset_level, "type": reset_type},
    }


def normalize_behavioral_contract(value: Any) -> Dict[str, Any]:
    """Normalize cycle-level output and state semantics from the feature model."""

    if not isinstance(value, dict):
        value = {}

    def names(key: str) -> List[str]:
        raw = value.get(key, [])
        if not isinstance(raw, list):
            raw = [raw] if raw else []
        cleaned: List[str] = []
        for item in raw:
            name = str(item).strip()
            if name and name not in cleaned:
                cleaned.append(name)
        return cleaned

    latencies: Dict[str, int] = {}
    raw_latencies = value.get("output_latency_cycles", {})
    if isinstance(raw_latencies, dict):
        for raw_name, raw_cycles in raw_latencies.items():
            name = str(raw_name).strip()
            try:
                cycles = int(raw_cycles)
            except (TypeError, ValueError):
                continue
            if name and cycles >= 0:
                latencies[name] = cycles

    edge = str(value.get("state_update_edge", "none")).strip().lower()
    if edge not in {"posedge", "negedge", "none"}:
        edge = "none"

    return {
        "registered_outputs": names("registered_outputs"),
        "combinational_outputs": names("combinational_outputs"),
        "output_latency_cycles": latencies,
        "state_update_edge": edge,
        "cycle_semantics": str(value.get("cycle_semantics", "")).strip(),
    }


# ── Validation ────────────────────────────────────────────────────────

def validate_features(llm_output: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Validate and normalize LLM output against the schema.

    Returns:
        (cleaned_features, confidence_map)
        - cleaned_features: dict with all expected fields, defaults filled in
        - confidence_map: field_name -> confidence level
    """
    cleaned = {}
    confidence = {}

    for name, type_hint, desc, default in LLM_FEATURE_FIELDS:
        if name in llm_output:
            val = llm_output[name]
            # Type coercion
            try:
                if type_hint == "bool" and not isinstance(val, bool):
                    val = str(val).lower() in ("true", "1", "yes")
                elif type_hint == "int" and not isinstance(val, int):
                    val = int(val)
                elif type_hint == "str" and not isinstance(val, str):
                    val = str(val)
                elif type_hint.startswith("list") and not isinstance(val, list):
                    val = [val] if val else []
                elif type_hint == "interface":
                    val = normalize_interface(val)
                elif type_hint == "behavioral_contract":
                    val = normalize_behavioral_contract(val)
            except (ValueError, TypeError):
                val = default
                confidence[name] = Confidence.NONE.value
                cleaned[name] = val
                continue

            cleaned[name] = val
            # Enum validation: reject out-of-vocabulary values
            if name in _ENUM_ALLOWED and val not in _ENUM_ALLOWED[name]:
                cleaned[name] = default
                confidence[name] = Confidence.LOW.value
            else:
                confidence[name] = Confidence.HIGH.value
        else:
            if type_hint == "interface":
                cleaned[name] = normalize_interface(default)
            elif type_hint == "behavioral_contract":
                cleaned[name] = normalize_behavioral_contract(default)
            else:
                cleaned[name] = default
            confidence[name] = Confidence.NONE.value

    return cleaned, confidence


# ── Confidence scoring for spec-only input ────────────────────────────

#  Fields that can usually be inferred well from a spec
_SPEC_RELIABLE = {
    "module_name", "purpose", "architecture_pattern", "is_sequential",
    "has_fsm", "key_operations", "suggested_subcategory",
}

# Fields that are hard to infer without code
_SPEC_UNRELIABLE = {
    "num_clock_domains", "estimated_fsm_states", "data_widths",
    "hierarchy",
}


def adjust_confidence_for_spec(confidence: Dict[str, str]) -> Dict[str, str]:
    """
    Downgrade confidence for fields that are unreliable when
    extracted from a spec (no Verilog source).
    """
    adjusted = dict(confidence)
    for name in _SPEC_UNRELIABLE:
        if adjusted.get(name) == Confidence.HIGH.value:
            adjusted[name] = Confidence.LOW.value
    for name in _SPEC_RELIABLE:
        # Keep as-is (HIGH if LLM returned it)
        pass
    # Everything else: cap at MEDIUM
    for name in adjusted:
        if name not in _SPEC_RELIABLE and name not in _SPEC_UNRELIABLE:
            if adjusted[name] == Confidence.HIGH.value:
                adjusted[name] = Confidence.MEDIUM.value
    return adjusted
