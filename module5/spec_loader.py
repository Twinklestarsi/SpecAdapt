from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from project_paths import PROJECT_ROOT


DEFAULT_SPEC_DB = PROJECT_ROOT / "spec_analysis.json"


def load_spec_record(
    benchmark: str,
    *,
    spec_db_path: str | Path = DEFAULT_SPEC_DB,
) -> Dict[str, Any]:
    spec_db_path = Path(spec_db_path)
    if not spec_db_path.is_file():
        raise FileNotFoundError(f"spec_analysis.json not found: {spec_db_path}")

    payload = json.loads(spec_db_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]

    for item in payload:
        if isinstance(item, dict) and str(item.get("benchmark", "")) == benchmark:
            return item

    raise KeyError(f"Benchmark not found in spec db: {benchmark}")


def build_spec_context(
    benchmark: str,
    *,
    spec_db_path: str | Path = DEFAULT_SPEC_DB,
) -> Dict[str, Any]:
    record = load_spec_record(benchmark, spec_db_path=spec_db_path)
    return {
        "benchmark": benchmark,
        "spec_text": str(record.get("spec_text", "")).strip(),
        "optimization_target": str(record.get("optimization_target", "")).strip(),
        "llm_features": dict(record.get("llm_features", {}) or {}),
        "input_type": str(record.get("input_type", "")).strip(),
        "confidence": record.get("confidence"),
        "overall_confidence": record.get("overall_confidence"),
        "verilog_path": str(record.get("verilog_path", "")).strip(),
    }
