"""Validation and metric normalization for imported experience."""

from __future__ import annotations

from typing import Any, Dict


VALID_PATHS = {"", "c_first", "rtl_direct"}
VALID_OBJECTIVES = {"", "AREA", "TIMING"}


def normalize_objective(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    if normalized not in VALID_OBJECTIVES:
        raise ValueError(f"Unsupported optimization objective: {value!r}")
    return normalized


def normalize_path(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in VALID_PATHS:
        raise ValueError(f"Unsupported generation path: {value!r}")
    return normalized


def normalize_area_gain(value: Any) -> float | None:
    """Canonical convention: positive means area reduction/improvement."""
    if value in (None, ""):
        return None
    return float(value)


def normalize_timing_gain(value: Any) -> float | None:
    """Canonical convention: positive means critical-path delay reduction."""
    if value in (None, ""):
        return None
    return float(value)


def validate_feature_profile(features: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(features, dict):
        raise TypeError("features must be a dictionary")
    result = dict(features)
    if "num_clock_domains" in result and result["num_clock_domains"] is not None:
        result["num_clock_domains"] = int(result["num_clock_domains"])
    for key in ("is_sequential", "has_fsm"):
        if key in result and not isinstance(result[key], bool):
            result[key] = str(result[key]).lower() in {"1", "true", "yes"}
    return result
