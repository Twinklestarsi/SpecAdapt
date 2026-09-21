from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from c_gen import validator as c_validator


def validate_c_file(c_path: str | Path, expected_function: str = "") -> Dict[str, Any]:
    c_path = Path(c_path)
    summary: Dict[str, Any] = {
        "path": str(c_path),
        "exists": c_path.exists(),
        "syntax_ok": False,
        "syntax_error": "",
        "expected_function": expected_function,
        "function_found": False,
        "checks": [],
    }

    if not c_path.exists():
        summary["syntax_error"] = f"File not found: {c_path}"
        return summary

    source = c_path.read_text(encoding="utf-8", errors="replace")
    summary["checks"].append("file_exists")

    ok, stderr = c_validator.validate(c_path)
    summary["syntax_ok"] = ok
    summary["syntax_error"] = stderr
    summary["checks"].append("clang_syntax")

    if expected_function:
        signature = f"void {expected_function}("
        summary["function_found"] = signature in source
        summary["checks"].append("expected_top_function")
    else:
        summary["function_found"] = True

    summary["state_struct_found"] = "state_elements_" in source
    summary["static_state_found"] = bool(re.search(r"\bstatic\b", source))
    if summary["state_struct_found"]:
        summary["state_model"] = "state_struct"
    elif summary["static_state_found"]:
        summary["state_model"] = "static_variables"
    else:
        summary["state_model"] = "combinational_or_external_state"
    summary["checks"].append("state_model_classification")
    return summary


def summarize_failed_checks(summary: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    if not summary.get("exists", False):
        failures.append("source_missing")
    if not summary.get("syntax_ok", False):
        failures.append("clang_syntax")
    if not summary.get("function_found", False):
        failures.append("expected_top_function")
    return failures
