"""Fit-free vectorization of the typed Spec Agent feature schema."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


FEATURE_SCHEMA_VERSION = "adaptive_router_explicit_v1"

ARCHITECTURES = (
    "fsm",
    "pipeline",
    "datapath",
    "control_logic",
    "memory",
    "combinational",
    "mixed",
)
RESET_TYPES = ("async", "sync", "both", "none", "unknown")
COMPLEXITIES = ("trivial", "simple", "moderate", "complex")
HIERARCHIES = ("flat", "instantiates_submodules", "wrapper_only")
SUBCATEGORIES = (
    "arithmetic",
    "bit_reorganization",
    "constant",
    "logical",
    "selection",
)
SUB_PATTERNS = (
    "counter",
    "shift_register",
    "mux_tree",
    "encoder",
    "decoder",
    "arbiter",
    "protocol_handler",
    "crypto_block",
    "handshake_interface",
)
KEY_OPERATIONS = (
    "arithmetic",
    "bitwise",
    "shift",
    "comparison",
    "mux",
    "memory_access",
    "serial_protocol",
)
COMMON_WIDTHS = (1, 8, 16, 32, 64, 128, 256, 512, 1024)
MISSING_FIELDS = (
    "architecture_pattern",
    "sub_patterns",
    "is_sequential",
    "num_clock_domains",
    "reset_type",
    "has_fsm",
    "estimated_fsm_states",
    "key_operations",
    "data_widths",
    "complexity",
    "hierarchy",
    "suggested_subcategory",
    "interface",
    "behavioral_contract",
)


def _clean_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _clean_token(value) in {"true", "1", "yes"}


def _bounded_log(value: Any, cap: float) -> float:
    numeric = min(max(_safe_float(value), 0.0), cap)
    return math.log1p(numeric) / math.log1p(cap)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def _sha256_array(vector: np.ndarray) -> str:
    stable = np.asarray(vector, dtype="<f4")
    return hashlib.sha256(stable.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class ExplicitVector:
    vector: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


class FeatureVectorizer:
    """Convert FeatureResult/llm_features JSON into a stable numeric vector."""

    schema_version = FEATURE_SCHEMA_VERSION

    def _vectorize(
        self, payload: dict[str, Any], objective: str | None
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        llm = payload.get("llm_features", payload)
        if not isinstance(llm, dict):
            raise TypeError("Feature payload must contain a llm_features object")
        values: list[float] = []
        names: list[str] = []

        def add(name: str, value: Any) -> None:
            names.append(name)
            values.append(_safe_float(value))

        def one_hot(prefix: str, value: Any, categories: Iterable[str]) -> None:
            token = _clean_token(value)
            known = tuple(categories)
            for category in known:
                add(f"{prefix}={category}", token == category)
            add(f"{prefix}=__other", bool(token) and token not in known)

        def multi_hot(prefix: str, raw: Any, categories: Iterable[str]) -> None:
            tokens = {_clean_token(item) for item in _as_list(raw)}
            known = tuple(categories)
            for category in known:
                add(f"{prefix}:{category}", category in tokens)
            add(f"{prefix}:__other", any(token and token not in known for token in tokens))

        add("is_sequential", _safe_bool(llm.get("is_sequential", False)))
        add("num_clock_domains_log", _bounded_log(llm.get("num_clock_domains", 0), 8))
        add("has_fsm", _safe_bool(llm.get("has_fsm", False)))
        add(
            "estimated_fsm_states_log",
            _bounded_log(llm.get("estimated_fsm_states", 0), 256),
        )
        one_hot("architecture_pattern", llm.get("architecture_pattern"), ARCHITECTURES)
        one_hot("reset_type", llm.get("reset_type"), RESET_TYPES)
        one_hot("complexity", llm.get("complexity"), COMPLEXITIES)
        one_hot("hierarchy", llm.get("hierarchy"), HIERARCHIES)
        one_hot(
            "suggested_subcategory",
            llm.get("suggested_subcategory"),
            SUBCATEGORIES,
        )
        multi_hot("sub_pattern", llm.get("sub_patterns"), SUB_PATTERNS)
        multi_hot("key_operation", llm.get("key_operations"), KEY_OPERATIONS)

        widths = sorted(
            {
                int(_safe_float(item))
                for item in _as_list(llm.get("data_widths"))
                if int(_safe_float(item)) > 0
            }
        )
        add("data_width_count_log", _bounded_log(len(widths), 32))
        add("data_width_min_log", _bounded_log(min(widths) if widths else 0, 65536))
        add("data_width_max_log", _bounded_log(max(widths) if widths else 0, 65536))
        add(
            "data_width_mean_log",
            _bounded_log(sum(widths) / len(widths) if widths else 0, 65536),
        )
        for width in COMMON_WIDTHS:
            add(f"data_width_has_{width}", width in widths)

        interface = llm.get("interface") if isinstance(llm.get("interface"), dict) else {}
        ports = [item for item in _as_list(interface.get("ports")) if isinstance(item, dict)]
        input_ports = [p for p in ports if _clean_token(p.get("direction")) == "input"]
        output_ports = [p for p in ports if _clean_token(p.get("direction")) == "output"]
        inout_ports = [p for p in ports if _clean_token(p.get("direction")) == "inout"]

        def port_width(port: dict[str, Any]) -> int:
            return max(0, int(_safe_float(port.get("width", 1), 1.0)))

        input_widths = [port_width(port) for port in input_ports]
        output_widths = [port_width(port) for port in output_ports]
        add("interface_num_inputs_log", _bounded_log(len(input_ports), 64))
        add("interface_num_outputs_log", _bounded_log(len(output_ports), 64))
        add("interface_num_inouts_log", _bounded_log(len(inout_ports), 16))
        add("interface_input_bits_log", _bounded_log(sum(input_widths), 1_000_000))
        add("interface_output_bits_log", _bounded_log(sum(output_widths), 1_000_000))
        add("interface_max_input_width_log", _bounded_log(max(input_widths, default=0), 65536))
        add("interface_max_output_width_log", _bounded_log(max(output_widths, default=0), 65536))
        clock = interface.get("clock") if isinstance(interface.get("clock"), dict) else {}
        reset = interface.get("reset") if isinstance(interface.get("reset"), dict) else {}
        add("interface_has_clock", bool(str(clock.get("name", "")).strip()))
        add("interface_has_reset", bool(str(reset.get("name", "")).strip()))
        one_hot("interface_clock_edge", clock.get("edge"), ("posedge", "negedge", "none"))
        one_hot("interface_reset_level", reset.get("active_level"), ("low", "high", "none"))
        one_hot("interface_reset_type", reset.get("type"), ("async", "sync", "none"))

        contract = (
            llm.get("behavioral_contract")
            if isinstance(llm.get("behavioral_contract"), dict)
            else {}
        )
        registered = _as_list(contract.get("registered_outputs"))
        combinational = _as_list(contract.get("combinational_outputs"))
        latency_map = (
            contract.get("output_latency_cycles")
            if isinstance(contract.get("output_latency_cycles"), dict)
            else {}
        )
        latencies = [max(0.0, _safe_float(value)) for value in latency_map.values()]
        total_classified = len(registered) + len(combinational)
        add("contract_registered_outputs_log", _bounded_log(len(registered), 64))
        add("contract_combinational_outputs_log", _bounded_log(len(combinational), 64))
        add(
            "contract_registered_ratio",
            len(registered) / total_classified if total_classified else 0.0,
        )
        add("contract_latency_count_log", _bounded_log(len(latencies), 64))
        add("contract_latency_max_log", _bounded_log(max(latencies, default=0), 64))
        add(
            "contract_latency_mean_log",
            _bounded_log(sum(latencies) / len(latencies) if latencies else 0, 64),
        )
        add("contract_has_cycle_semantics", bool(str(contract.get("cycle_semantics", "")).strip()))
        one_hot(
            "contract_state_update_edge",
            contract.get("state_update_edge"),
            ("posedge", "negedge", "none"),
        )

        resolved_objective = _clean_token(objective or payload.get("optimization_target"))
        one_hot("objective", resolved_objective, ("area", "timing"))

        for field_name in MISSING_FIELDS:
            add(f"missing:{field_name}", field_name not in llm or llm.get(field_name) is None)

        vector = np.asarray(values, dtype=np.float32)
        if not np.isfinite(vector).all():
            raise RuntimeError("Explicit feature vector contains NaN or infinity")
        if len(names) != len(set(names)):
            raise RuntimeError("Explicit feature names are not unique")
        return vector, tuple(names)

    def transform_one(
        self, payload: dict[str, Any], *, objective: str | None = None
    ) -> ExplicitVector:
        vector, feature_names = self._vectorize(payload, objective)
        names_payload = json.dumps(feature_names, separators=(",", ":"))
        metadata = {
            "schema_version": self.schema_version,
            "dimension": int(vector.shape[0]),
            "feature_names_sha256": hashlib.sha256(
                names_payload.encode("utf-8")
            ).hexdigest(),
            "vector_sha256": _sha256_array(vector),
            "excluded_text_fields": [
                "module_name",
                "purpose",
                "optimization_notes",
            ],
        }
        return ExplicitVector(vector, feature_names, metadata)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.transform_one({"llm_features": {}}, objective="AREA").feature_names

    @property
    def dimension(self) -> int:
        return len(self.feature_names)
