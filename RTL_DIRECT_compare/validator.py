"""
validator.py — Lightweight Verilog syntax validation for direct RTL outputs.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Tuple

from toolchain import find_tool


#: Yosys reports an index or part-select that reaches past the declared range as
#: a *warning*, keeps going, and still exits 0, so the return code alone cannot
#: be the gate.  Only this one diagnostic class is promoted to an error below;
#: every other Yosys message is deliberately ignored, because the Icarus front
#: end is the authority on syntax and Yosys is stricter about constructs the
#: Design Compiler flow accepts.
_OUT_OF_BOUNDS_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):\s*Warning:\s*(?P<detail>.*out of bounds.*)$",
    flags=re.MULTILINE,
)


#: Yosys writes an escaped identifier with a leading backslash (``\cmd_packed``)
#: and closes it with a quote; neither character belongs in the message we hand
#: the model.
_SIGNAL_RE = re.compile(r"on signal `\\?(?P<name>[^']*)'")


def _run(cmd: List[str]) -> Tuple[int, str]:
    """Run a validator, returning its exit code and merged output."""

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, ((proc.stderr or "") + (proc.stdout or "")).strip()


def _yosys_out_of_bounds(path: Path, yosys: str) -> str:
    """Return de-duplicated Yosys out-of-bounds diagnostics, or ``""``.

    The file name is quoted inside the Yosys command string because ``-p``
    takes one script argument that Yosys tokenizes itself; an unquoted path
    containing a space would be read as a second argument.

    Yosys prints each warning twice — once where it is detected and once in its
    summary block — so identical messages are collapsed to keep the retry
    prompt free of noise.  The escaped-identifier backslash is dropped because
    it is Yosys' internal spelling, not Verilog the model should reproduce.
    """

    _, output = _run([yosys, "-p", f'read_verilog "{path}"'])
    messages: List[str] = []
    for match in _OUT_OF_BOUNDS_RE.finditer(output):
        signal = _SIGNAL_RE.search(match.group("detail"))
        core = (
            f"range select is out of bounds on '{signal.group('name')}'"
            if signal
            else "range select is out of bounds"
        )
        message = (
            f"{match.group('path')}:{match.group('line')}: error: {core}. "
            f"The select reaches past the declared range, so JasperGold rejects "
            f"the module (VERI-1216) even though the local simulator accepts "
            f"it. Widen the signal or narrow the select so it stays in range."
        )
        if message not in messages:
            messages.append(message)
    return "\n".join(messages)


def validate(verilog_path: str | Path) -> Tuple[bool, str]:
    """Validate Verilog against both front ends used by this flow.

    Two gates run in sequence:

    * ``iverilog -g2005`` is the fast, authoritative syntax check.  It reports
      what the Design Compiler flow rejects, and its line numbers are the ones
      the retry prompt quotes back to the model, so it runs first and its output
      is what a syntax failure returns.
    * ``yosys`` runs only once Icarus has passed, and only its out-of-bounds
      diagnostics are promoted to errors.  Icarus accepts a part-select that
      reaches past the declared range — ``reg [63:0] p; p[66:54] = ...;`` —
      while the Verific front end behind JasperGold rejects it outright with
      ``[ERROR (VERI-1216)] index 66 is out of range [63:0]``.  Catching it here
      costs a local subprocess instead of a full remote JasperGold round trip.

    A missing Icarus falls back to Verilator lint mode.  A missing Yosys only
    means the second gate is unavailable, which is not fatal: the first gate
    remains authoritative, so the file is not rejected for that reason.
    """

    path = Path(verilog_path)

    iverilog = find_tool("IVERILOG_PATH", ("iverilog",))
    if iverilog:
        returncode, output = _run([iverilog, "-g2005", "-t", "null", str(path)])
        if returncode != 0:
            return False, output
        primary_output = output
    else:
        verilator = find_tool("VERILATOR_PATH", ("verilator",))
        if not verilator:
            return False, (
                "No Verilog validator found; set IVERILOG_PATH or VERILATOR_PATH "
                "to an executable path."
            )
        returncode, output = _run([verilator, "--lint-only", str(path)])
        if returncode != 0:
            return False, output
        primary_output = output

    yosys = find_tool("YOSYS_PATH", ("yosys",))
    if yosys:
        range_errors = _yosys_out_of_bounds(path, yosys)
        if range_errors:
            return False, range_errors

    return True, primary_output
