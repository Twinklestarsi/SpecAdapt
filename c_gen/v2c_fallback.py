"""
v2c_fallback.py — Fallback to v2c binary for Verilog-to-C conversion.

Only used when LLM generation fails for Verilog-only input.
Requires the `v2c` binary to be installed and on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional


def v2c_convert(
    verilog_path: str | Path,
    output_path: str | Path,
    timeout: int = 60,
) -> tuple[bool, str]:
    """
    Convert Verilog to C using the v2c binary.

    Args:
        verilog_path: Path to input .v file
        output_path: Path to write output .c file
        timeout: Timeout in seconds

    Returns:
        (success, error_msg): success=True if conversion succeeded
    """
    if not shutil.which("v2c"):
        return False, "v2c binary not found on PATH"

    verilog_path = Path(verilog_path)
    output_path = Path(output_path)

    if not verilog_path.exists():
        return False, f"Verilog file not found: {verilog_path}"

    try:
        result = subprocess.run(
            ["v2c", str(verilog_path), "-o", str(output_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0 and output_path.exists():
            return True, ""

        return False, result.stderr or "v2c conversion failed (no error output)"

    except subprocess.TimeoutExpired:
        return False, f"v2c conversion timed out ({timeout}s)"
    except Exception as e:
        return False, f"v2c error: {type(e).__name__}: {e}"
