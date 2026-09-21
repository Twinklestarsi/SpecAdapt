"""
validator.py — Clang-16 syntax validation for generated C code.

Runs `clang-16 -fsyntax-only` to check if C code is parseable.
Used in the retry loop: if validation fails, the error is fed back to the LLM.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple

from toolchain import ToolNotFoundError, require_tool


def validate(c_path: str | Path) -> Tuple[bool, str]:
    """
    Run clang-16 syntax check on a C file.

    Args:
        c_path: Path to the .c file to validate.

    Returns:
        (ok, stderr): ok=True if valid, stderr contains error messages if not.
    """
    c_path = Path(c_path)
    if not c_path.exists():
        return False, f"File not found: {c_path}"

    try:
        clang = require_tool("CLANG_PATH", "clang-16", "clang")
        result = subprocess.run(
            [clang, "-fsyntax-only", "-std=c99", str(c_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "clang-16 validation timed out (30s)"
    except (FileNotFoundError, ToolNotFoundError) as exc:
        return False, str(exc)
    except Exception as e:
        return False, f"Validation error: {type(e).__name__}: {e}"
