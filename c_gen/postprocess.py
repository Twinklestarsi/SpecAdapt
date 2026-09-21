"""
postprocess.py — Post-processing for generated C code.

Wraps the core logic from patch_v2c_outputs.py and inject_hls_pragmas.py
for single-file operation (not batch directory processing).

Two steps:
  1. patch_state_writebacks() — insert pointer writebacks for state_elements
  2. inject_hls_pragmas()      — insert HLS interface pragmas

Both are idempotent (safe to re-run).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# ─── Sentinel markers ─────────────────────────────────────────────────

_PATCH_MARKER = "// V2C_PTR_WRITEBACK_PATCHED"
_PRAGMA_MARKER = "// HLS_INTERFACE_PRAGMAS_INJECTED"

# ─── Regex for primary function (first non-main void function) ────────

_FUNC_SIG_RE = re.compile(
    r"^(void\s+(?!main\b)(\w+)\s*\(([^)]*)\))\s*\{",
    re.MULTILINE,
)


# ─── Patch state writebacks ───────────────────────────────────────────

def _parse_ptr_params(param_str: str) -> list[str]:
    """Extract output pointer parameter names from a function signature."""
    names = []
    for raw in param_str.split(","):
        raw = raw.strip()
        if "*" not in raw:
            continue
        after_star = raw.split("*")[-1].strip()
        name = re.match(r"([A-Za-z_]\w*)", after_star)
        if name:
            names.append(name.group(1))
    return names


def _find_struct_instance(code: str, func_name: str) -> Optional[str]:
    """Find the struct instance variable name, e.g. 'sexample' for func 'example'."""
    m = re.search(
        r"struct\s+state_elements_" + re.escape(func_name) + r"\s+(\w+)\s*;",
        code,
    )
    return m.group(1) if m else None


def _find_struct_fields_written(code: str, inst_name: str) -> set[str]:
    """Return all struct field names that appear in writes: inst.field = ..."""
    return set(re.findall(re.escape(inst_name) + r"\.(\w+)\s*=", code))


def _already_deref(code: str, param: str) -> bool:
    """Check if *param = ... already exists in the code body."""
    return bool(re.search(r"\*\s*" + re.escape(param) + r"\s*=", code))


def _find_primary_func_end(code: str, sig_match: re.Match) -> Optional[int]:
    """Find the closing brace index of the primary function."""
    start = code.index("{", sig_match.end() - 1)
    depth = 0
    for i in range(start, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def patch_state_writebacks(c_path: str | Path) -> bool:
    """
    Insert pointer writebacks for state_elements struct fields.

    Returns True if the file was modified, False if already patched or no-op.
    """
    c_path = Path(c_path)
    code = c_path.read_text(errors="ignore")

    if _PATCH_MARKER in code:
        return False

    if "state_elements_" not in code:
        return False

    sig = _FUNC_SIG_RE.search(code)
    if not sig:
        return False

    func_name = sig.group(2)
    param_str = sig.group(3)

    ptr_params = _parse_ptr_params(param_str)
    if not ptr_params:
        return False

    inst = _find_struct_instance(code, func_name)
    if not inst:
        return False

    needs_fix = [p for p in ptr_params if not _already_deref(code, p)]
    if not needs_fix:
        return False

    fields = _find_struct_fields_written(code, inst)

    patches = [(p, inst, p) for p in needs_fix if p in fields]
    if not patches:
        return False

    close_idx = _find_primary_func_end(code, sig)
    if close_idx is None:
        return False

    lines = [f"  *{param} = {inst}.{field};" for param, inst, field in patches]
    insert = "\n" + _PATCH_MARKER + "\n" + "\n".join(lines) + "\n"

    new_code = code[:close_idx] + insert + code[close_idx:]
    c_path.write_text(new_code)
    return True


# ─── Inject HLS pragmas ───────────────────────────────────────────────

def _parse_params(param_str: str) -> list[tuple[bool, str]]:
    """Parse function parameters. Returns list of (is_pointer, param_name)."""
    params = []
    for raw in param_str.split(","):
        raw = raw.strip()
        if not raw:
            continue
        is_ptr = "*" in raw
        if is_ptr:
            after_star = raw.split("*")[-1].strip()
            m = re.match(r"([A-Za-z_]\w*)", after_star)
        else:
            m = re.search(r"([A-Za-z_]\w*)\s*$", raw)
        if m:
            params.append((is_ptr, m.group(1)))
    return params


def _build_pragma_block(params: list[tuple[bool, str]]) -> str:
    """Build the pragma block string to insert after the opening brace."""
    lines = [_PRAGMA_MARKER]
    lines.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
    for _is_ptr, name in params:
        # Pointer parameters are scalar outputs in the generated C contract.
        # ap_none exposes only the data signal.  ap_ovld adds an unnecessary
        # <port>_ap_vld signal and makes the RTL interface HLS-specific.
        lines.append(f"#pragma HLS INTERFACE ap_none port={name}")
    return "\n".join(lines) + "\n"


def inject_hls_pragmas(c_path: str | Path) -> bool:
    """
    Inject HLS interface pragmas after the function's opening brace.

    Returns True if the file was modified, False if already injected or no-op.
    """
    c_path = Path(c_path)
    code = c_path.read_text(errors="ignore")

    if _PRAGMA_MARKER in code:
        # Migrate files produced by the previous policy.  Keeping this path
        # idempotent lets old generated C be normalized before a new HLS run.
        new_code = re.sub(
            r"(#pragma\s+HLS\s+INTERFACE\s+)ap_ovld(\s+port\s*=)",
            r"\1ap_none\2",
            code,
        )
        if new_code == code:
            return False
        c_path.write_text(new_code)
        return True

    sig = _FUNC_SIG_RE.search(code)
    if not sig:
        return False

    param_str = sig.group(3)
    params = _parse_params(param_str)

    if not params:
        return False

    brace_pos = code.index("{", sig.end() - 1)
    insert_pos = brace_pos + 1

    pragma_block = _build_pragma_block(params)

    new_code = code[:insert_pos] + "\n" + pragma_block + code[insert_pos:]
    c_path.write_text(new_code)
    return True
