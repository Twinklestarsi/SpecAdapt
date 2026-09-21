#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HLS helper utilities: generate RTL from C and collect syn/verilog/*.v files.

Supports two backends:
  - vitis:  Xilinx Vitis HLS (v++) — full HLS with optimization
  - bambu:  PandA Bambu — configurable optimization (-O0 for structural fidelity)

Usage:
    # Bambu backend (default, -O0 preserves C-level structure):
    python3 verilogc2x.py /path/to/c_files --backend bambu -j 16

    # Vitis backend:
    source /path/to/Vitis/settings64.sh
    python3 verilogc2x.py /path/to/c_files --backend vitis -j 16
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


import re as _re


def infer_hls_top_from_cpp_path(cpp_path: str) -> str:
    """
    Infer the HLS top function name.

    Strategy (in order):
    1. Scan the source file for 'void <name>(' function definitions that are
       NOT 'main'.  If exactly one is found, use it.  This handles RTL-derived
       C files where the filename may carry a transform suffix that does not
       appear in the actual function name.
    2. Fall back to filename-based heuristics (_hls_opt / _PROMPT stripping).
    """
    # --- Strategy 1: read the source and find function names ---------------
    try:
        with open(cpp_path, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
        # Match top-level void function definitions (not declarations)
        func_names = _re.findall(
            r'^void\s+(\w+)\s*\([^)]*\)\s*$', src, _re.MULTILINE
        )
        non_main = [n for n in func_names if n != "main"]
        if len(non_main) == 1:
            return non_main[0]
        # If multiple non-main functions, prefer the one matching the struct name
        for n in non_main:
            if f"state_elements_{n}" in src:
                return n
    except OSError:
        pass

    # --- Strategy 2: filename heuristics (legacy) --------------------------
    stem = Path(cpp_path).stem
    if stem.endswith("_hls_opt"):
        base = stem[: -len("_hls_opt")]
        for tag in ("POWER", "TIMING", "AREA"):
            if base.endswith(f"_{tag}") or base.endswith(f"_{tag.lower()}"):
                return base[: -(len(tag) + 1)]
        return base
    return stem


# Patterns that are valid CBMC/formal-verification C but unsynthesizable by HLS
_CBMC_PATTERNS = [
    (_re.compile(r'\birep\s*\('), "irep() CBMC expression literal"),
    (_re.compile(r'__CPROVER_bitvector'), "__CPROVER_bitvector type"),
    (_re.compile(r'\w+\[\s*\d+\s*,\s*\d+\s*\]'), "CBMC bit-slice [hi,lo] syntax"),
]


def check_synthesizable(cpp_path: str) -> tuple[bool, str]:
    """Quick pre-check: reject C files with known unsynthesizable CBMC constructs.

    Returns (ok, reason).  ok=True means the file looks synthesizable.
    """
    try:
        with open(cpp_path, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
    except OSError as e:
        return False, f"cannot read file: {e}"

    for pat, desc in _CBMC_PATTERNS:
        if pat.search(src):
            return False, f"contains {desc}"

    return True, ""


def write_hls_config(
    cfg_path: str,
    cpp_path: str,
    top: str,
    *,
    part: str,
    clock: str,
    flow_target: str = "vivado",
    minimal_interface: bool = True,
    auto_pipeline: bool = False,
    reset_mode: str | None = None,
    reset_async: bool | None = None,
    reset_level: str | None = None,
) -> None:
    lines = [
        "# hls_config.cfg",
        f"part={part}",
        "",
        "[hls]",
        f"flow_target={flow_target}",
        f"clock={clock}",
        f"syn.file={os.path.abspath(cpp_path)}",
        f"syn.top={top}",
        "",
        "# Generate plain RTL without an implicit AXI control wrapper.",
        "syn.output.format=rtl",
    ]
    if minimal_interface:
        lines.extend(
            [
                "syn.interface.default_slave_interface=none",
                "syn.interface.clock_enable=0",
            ]
        )
    if not auto_pipeline:
        # Vitis otherwise automatically pipelines small loops.  The experiment
        # applies optimizations to C explicitly, so compiler-added pipelining
        # would be an uncontrolled optimization variable.
        lines.append("syn.compile.pipeline_loops=0")
    if reset_mode is not None:
        if reset_mode not in {"none", "control", "state", "all"}:
            raise ValueError(f"Unsupported HLS reset mode: {reset_mode}")
        lines.append(f"syn.rtl.reset={reset_mode}")
    if reset_async is not None:
        lines.append(f"syn.rtl.reset_async={1 if reset_async else 0}")
    if reset_level is not None:
        if reset_level not in {"low", "high"}:
            raise ValueError(f"Unsupported HLS reset level: {reset_level}")
        lines.append(f"syn.rtl.reset_level={reset_level}")
    lines.append("")
    cfg_text = "\n".join(lines)
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg_text)


# Thread-safe print lock (used when running concurrent HLS jobs)
_print_lock = threading.Lock()


def _tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


def run_vpp_hls(cpp_path: str, *, part: str, clock: str, flow_target: str = "vivado") -> bool:
    """
    Run v++ HLS for a single .c file and generate RTL.

    To avoid CIFS/SMB filesystem issues (Vitis HLS clang fails to create
    intermediate files on network mounts), the actual HLS build runs in a
    local /tmp directory and results are copied back.

    Final output directory (next to the source file):
    - hls_<name>/hls_config.cfg
    - hls_<name>/hls_work/...
    - hls_<name>/vpp_stdout.txt
    - hls_<name>/vpp_stderr.txt
    """
    cpp_path = os.path.abspath(cpp_path)
    if not os.path.isfile(cpp_path):
        _tprint(f"[ERROR] cpp does not exist: {cpp_path}")
        return False

    # Pre-check: skip files with known unsynthesizable CBMC constructs
    synth_ok, synth_reason = check_synthesizable(cpp_path)
    if not synth_ok:
        _tprint(f"[SKIP] Not synthesizable ({synth_reason}): {cpp_path}")
        return False

    top = infer_hls_top_from_cpp_path(cpp_path)
    src_stem = Path(cpp_path).stem

    # Final destination on the (possibly network) filesystem
    dest_hls_dir = os.path.join(os.path.dirname(cpp_path), f"hls_{src_stem}")

    # Clean up any existing HLS output folder for a fresh run
    if os.path.isdir(dest_hls_dir):
        _tprint(f"[CLEAN] Removing existing HLS directory: {dest_hls_dir}")
        shutil.rmtree(dest_hls_dir)

    # Run HLS in a local temp directory to avoid CIFS intermediate-file issues
    with tempfile.TemporaryDirectory(prefix=f"hls_{src_stem}_") as tmp_hls_dir:
        # Copy source file into the temp directory so relative paths work
        tmp_src = os.path.join(tmp_hls_dir, os.path.basename(cpp_path))
        shutil.copy2(cpp_path, tmp_src)

        cfg_path = os.path.join(tmp_hls_dir, "hls_config.cfg")
        write_hls_config(cfg_path, tmp_src, top, part=part, clock=clock, flow_target=flow_target)

        cmd = ["v++", "-c", "--mode", "hls", "--config", "./hls_config.cfg", "--work_dir", "./hls_work"]
        _tprint(f"[HLS] Running: {' '.join(cmd)}  (cwd={tmp_hls_dir}, top={top})")

        stdout_path = os.path.join(tmp_hls_dir, "vpp_stdout.txt")
        stderr_path = os.path.join(tmp_hls_dir, "vpp_stderr.txt")
        with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
            p = subprocess.run(cmd, cwd=tmp_hls_dir, stdout=out, stderr=err, text=True)

        # Copy the entire HLS directory back to the destination
        shutil.copytree(tmp_hls_dir, dest_hls_dir)

    if p.returncode != 0:
        # Print stderr details from the copied-back location
        dest_stderr = os.path.join(dest_hls_dir, "vpp_stderr.txt")
        _tprint(f"[ERROR] v++ HLS failed: {cpp_path} (rc={p.returncode})")
        _tprint(f"        Details: {dest_stderr}")
        return False

    summary_path = os.path.join(dest_hls_dir, "hls_work", "hls_work.hlscompile_summary")
    if os.path.isfile(summary_path):
        _tprint(f"[DONE] HLS summary: {summary_path}")
    else:
        _tprint(f"[DONE] v++ returned success, but summary was not found.")
    return True


def flatten_bambu_verilog(verilog_src: str, top_name: str) -> str:
    """Flatten Bambu's hierarchical Verilog into a single module with assigns.

    Bambu emits: library cells + datapath_<top> + controller_<top> + _<top> + <top>.
    This extracts just the datapath, inlines all library cell instantiations into
    direct assign statements, and emits a single flat module named <top_name>.

    If the datapath contains sequential elements (regs, always blocks), falls back
    to keeping the datapath + used library cells without full inlining.
    """

    # ── Split file into (module_name, full_text) pairs ────────────────────
    mod_re = _re.compile(
        r'^(`timescale[^\n]*\n)?module\s+(\w+)\s*\(',
        _re.MULTILINE,
    )
    mod_blocks: dict[str, str] = {}
    starts = [(m.start(), m.group(2)) for m in mod_re.finditer(verilog_src)]
    for i, (start, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(verilog_src)
        mod_blocks[name] = verilog_src[start:end]

    # ── Find the datapath module ──────────────────────────────────────────
    dp_name = f"datapath_{top_name}"
    dp_text = mod_blocks.get(dp_name)
    if not dp_text:
        return verilog_src  # can't flatten

    # ── Check for sequential elements (regs, always blocks) ───────────────
    if _re.search(r'\breg\b|\balways\b', dp_text):
        # Sequential datapath — don't try full inlining, just strip wrappers
        return _strip_bambu_wrappers(mod_blocks, dp_name, top_name)

    # ── Build expression templates for library cells ──────────────────────
    # Operator map: module_name -> lambda(connections, params) -> expr string
    _BINOP = {
        'ui_plus_expr_FU':     '+',
        'ui_minus_expr_FU':    '-',
        'ui_bit_and_expr_FU':  '&',
        'ui_bit_xor_expr_FU':  '^',
        'ui_eq_expr_FU':       '==',
        'ui_ne_expr_FU':       '!=',
        'ui_lt_expr_FU':       '<',
        'ui_le_expr_FU':       '<=',
        'ui_gt_expr_FU':       '>',
        'ui_ge_expr_FU':       '>=',
    }

    def _resolve_inst(mod_name, conns, params):
        """Return (out_signal, assign_expr) or None if unknown."""
        out = conns.get('out1', '')

        if mod_name == 'constant_value':
            return out, params.get('value', "1'b0")

        if mod_name in _BINOP:
            return out, f"({conns.get('in1', '?')} {_BINOP[mod_name]} {conns.get('in2', '?')})"

        if mod_name == 'ui_mult_expr_FU':
            return out, f"({conns.get('in1', '?')} * {conns.get('in2', '?')})"

        if mod_name == 'ui_cond_expr_FU':
            return out, f"({conns.get('in1', '?')} != 0 ? {conns.get('in2', '?')} : {conns.get('in3', '?')})"

        if mod_name == 'ui_ternary_plus_expr_FU':
            return out, f"({conns.get('in1', '?')} + {conns.get('in2', '?')} + {conns.get('in3', '?')})"

        if mod_name == 'ui_ternary_mm_expr_FU':
            return out, f"({conns.get('in1', '?')} - {conns.get('in2', '?')} - {conns.get('in3', '?')})"

        if mod_name == 'ui_extract_bit_expr_FU':
            return out, f"(({conns.get('in1', '?')} >> {conns.get('in2', '?')}) & 1)"

        if mod_name in ('ui_view_convert_expr_FU', 'ASSIGN_UNSIGNED_FU'):
            return out, conns.get('in1', '?')

        if mod_name == 'ui_lshift_expr_FU':
            return out, f"({conns.get('in1', '?')} << {conns.get('in2', '?')})"

        if mod_name == 'ui_rshift_expr_FU':
            return out, f"({conns.get('in1', '?')} >> {conns.get('in2', '?')})"

        if mod_name == 'ui_negate_expr_FU':
            return out, f"(-{conns.get('in1', '?')})"

        if mod_name == 'ui_bit_not_expr_FU':
            return out, f"(~{conns.get('in1', '?')})"

        if mod_name == 'ui_bit_ior_concat_expr_FU':
            # {in1[N-1:OFFSET], in2[OFFSET-1:0]}  — keep as bitwise or for DC
            return out, f"({conns.get('in1', '?')} | {conns.get('in2', '?')})"

        if mod_name == 'UUdata_converter_FU':
            bw_in = int(params.get('BITSIZE_in1', '1'))
            bw_out = int(params.get('BITSIZE_out1', '1'))
            sig = conns.get('in1', '?')
            if bw_out <= bw_in:
                return out, f"{sig}[{bw_out - 1}:0]"
            else:
                return out, f"{{{{{bw_out - bw_in}{{1'b0}}}}, {sig}}}"

        if mod_name == 'MUX_GATE':
            return out, f"({conns.get('sel', '?')} ? {conns.get('in1', '?')} : {conns.get('in2', '?')})"

        if mod_name == 'read_cond_FU':
            return out, f"({conns.get('in1', '?')} != 0)"

        return None  # unknown module

    # ── Parse datapath ports and wires ────────────────────────────────────
    port_lines: list[str] = []
    wire_lines: list[str] = []
    assign_lines: list[str] = []
    unknown_insts: list[str] = []
    unknown_mods: set[str] = set()

    skip_ports = {'clock', 'reset'}

    # Ports
    for m in _re.finditer(r'^\s*(input|output)\s+(\[[\d:]+\]\s+)?(\w+)\s*;',
                          dp_text, _re.MULTILINE):
        direction, width, name = m.group(1), (m.group(2) or '').strip(), m.group(3)
        if name in skip_ports:
            continue
        # Rename in_port_X -> X for cleaner interface
        clean_name = _re.sub(r'^in_port_', '', name)
        if clean_name != name:
            # We'll need a wire alias if the datapath references in_port_X internally
            wire_lines.append(f"  wire {width + ' ' if width else ''}{name};")
            assign_lines.append(f"  assign {name} = {clean_name};")
        port_lines.append(f"  {direction} {width + ' ' if width else ''}{clean_name};")

    # Wires
    for m in _re.finditer(r'^\s*wire\s+(\[[\d:]+\]\s+)?(\w+)\s*;',
                          dp_text, _re.MULTILINE):
        width, name = (m.group(1) or '').strip(), m.group(2)
        wire_lines.append(f"  wire {width + ' ' if width else ''}{name};")

    # ── Parse and inline instantiations ───────────────────────────────────
    # Match parameterized: module_name #(.P1(V1), ...) inst_name (.port1(sig1), ...);
    inst_param_re = _re.compile(
        r'(\w+)\s+#\(([^)]*(?:\([^)]*\)[^)]*)*)\)\s+(\w+)\s+\(([^;]+)\);',
        _re.DOTALL,
    )
    # Match non-parameterized: module_name inst_name (.port1(sig1), ...);
    # (must not match keywords: wire, reg, input, output, assign, always, etc.)
    _verilog_kw = {'wire', 'reg', 'input', 'output', 'assign', 'always',
                   'initial', 'module', 'endmodule', 'parameter', 'localparam',
                   'begin', 'end', 'if', 'else', 'case', 'default', 'generate',
                   'endgenerate', 'for', 'integer'}

    already_matched = set()

    for m in inst_param_re.finditer(dp_text):
        already_matched.add(m.span())
        mod_name = m.group(1)
        params_str = m.group(2)
        ports_str = m.group(4)

        params = dict(_re.findall(r'\.(\w+)\(([^)]*)\)', params_str))
        conns = dict(_re.findall(r'\.(\w+)\(([^)]*)\)', ports_str))

        result = _resolve_inst(mod_name, conns, params)
        if result:
            out_sig, expr = result
            assign_lines.append(f"  assign {out_sig} = {expr};")
        else:
            unknown_insts.append(m.group(0))
            unknown_mods.add(mod_name)

    # Non-parameterized instantiations (e.g. sub-module calls)
    inst_noparam_re = _re.compile(
        r'(\w+)\s+(\w+)\s+\((\.[^;]+)\);',
        _re.DOTALL,
    )
    for m in inst_noparam_re.finditer(dp_text):
        if any(m.start() >= s and m.end() <= e for s, e in already_matched):
            continue
        mod_name = m.group(1)
        if mod_name in _verilog_kw:
            continue
        inst_name = m.group(2)
        ports_str = m.group(3)

        conns = dict(_re.findall(r'\.(\w+)\(([^)]*)\)', ports_str))
        result = _resolve_inst(mod_name, conns, {})
        if result:
            out_sig, expr = result
            assign_lines.append(f"  assign {out_sig} = {expr};")
        else:
            unknown_insts.append(m.group(0))
            unknown_mods.add(mod_name)

    # Direct assigns in the datapath
    for m in _re.finditer(r'^\s*assign\s+(\w+)\s*=\s*(.+?);',
                          dp_text, _re.MULTILINE):
        assign_lines.append(f"  assign {m.group(1)} = {m.group(2).strip()};")

    # ── Build the port list for the module header ─────────────────────────
    port_names = []
    for pl in port_lines:
        nm = pl.rstrip(';').split()[-1]
        port_names.append(nm)

    # ── Emit flat module ──────────────────────────────────────────────────
    out: list[str] = []
    out.append("`timescale 1ns / 1ps")

    # Include definitions for any unknown modules we couldn't inline
    for umod in sorted(unknown_mods):
        if umod in mod_blocks:
            out.append(mod_blocks[umod])

    out.append(f"module {top_name}(")
    out.append(",\n".join(f"  {n}" for n in port_names))
    out.append(");")
    for pl in port_lines:
        out.append(pl)
    out.append("")
    for wl in wire_lines:
        out.append(wl)
    if wire_lines:
        out.append("")
    for al in assign_lines:
        out.append(al)
    if unknown_insts:
        out.append("")
        out.append("  // Instances that could not be inlined:")
        for ui in unknown_insts:
            out.append(f"  {ui.strip()}")
    out.append("")
    out.append(f"endmodule // {top_name}")
    out.append("")

    return "\n".join(out)


def _strip_bambu_wrappers(mod_blocks: dict[str, str], dp_name: str,
                          top_name: str) -> str:
    """Fallback for sequential datapaths: keep datapath + used library cells,
    strip controller and wrapper modules, rename datapath to top_name."""
    dp_text = mod_blocks[dp_name]

    # Find which library modules are instantiated in the datapath
    used_mods = set(_re.findall(r'(\w+)\s+#\(', dp_text))
    used_mods.discard(dp_name)

    parts: list[str] = ["`timescale 1ns / 1ps\n"]
    for name, text in mod_blocks.items():
        if name in used_mods:
            parts.append(text)

    # Rename datapath module
    renamed = _re.sub(
        rf'\bmodule\s+{_re.escape(dp_name)}\b',
        f'module {top_name}',
        dp_text,
    )
    parts.append(renamed)
    return "\n".join(parts)


def prep_for_bambu(src: str, top_name: str | None = None) -> str:
    """Transform v2c-generated C for Bambu: remove HLS boilerplate, convert
    pointer outputs to return values so Bambu produces clean datapath logic
    instead of a memory controller.

    Handles both single-output (return value) and multi-output (return struct).
    """
    # ── Step 1: strip includes, pragmas, macros ──────────────────────────
    _skip_includes = {"<stdio.h>", "<assert.h>", "<stdlib.h>", "<string.h>"}
    lines = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#pragma HLS"):
            continue
        if "HLS_INTERFACE_PRAGMAS_INJECTED" in s:
            continue
        if s.startswith("#include") and any(i in s for i in _skip_includes):
            continue
        if s.startswith("assert("):
            continue
        lines.append(line)
    src = "\n".join(lines)

    # ── Step 2: remove main() ────────────────────────────────────────────
    m = _re.search(r'^void\s+main\s*\(\s*\)\s*\{', src, _re.MULTILINE)
    if m:
        src = src[:m.start()].rstrip() + "\n"

    # ── Step 3: remove TRANSFORM_META comment ────────────────────────────
    src = _re.sub(r'^//\s*TRANSFORM_META:.*$', '', src, flags=_re.MULTILINE)

    # ── Step 3b: add __attribute__((noinline)) to non-top helper functions
    # so Clang doesn't inline them and Bambu synthesizes them as separate HW
    _top = top_name or 'example'
    def _add_noinline(m):
        pre, fname = m.group(1), m.group(2)
        if fname == _top or fname == 'main':
            return m.group(0)
        # Already has noinline?
        if '__attribute__' in pre:
            return m.group(0)
        return f'__attribute__((noinline)) {m.group(0)}'

    src = _re.sub(
        r'^((?:static\s+)?(?:unsigned\s+)?(?:long\s+)?(?:int|char|short|void|_Bool|float|double|struct\s+\w+)'
        r'(?:\s+(?:unsigned|long|int|char|short))*\s*\*?\s+)'
        r'(\w+)\s*\([^)]*\)\s*\{',
        _add_noinline,
        src,
        flags=_re.MULTILINE,
    )

    # ── Step 4: find the top function and parse its signature ────────────
    # Match: void funcname(params) {
    func_re = _re.compile(
        r'^(void)\s+(' + (top_name or r'(?!main\b)\w+') + r')\s*\(([^)]*)\)\s*\{',
        _re.MULTILINE,
    )
    func_m = func_re.search(src)
    if not func_m:
        return src  # can't transform, return cleaned source as-is

    func_name = func_m.group(2)
    param_str = func_m.group(3)

    # Parse params into (type_str, name, is_pointer) tuples
    params = []
    for raw in param_str.split(","):
        raw = raw.strip()
        if not raw:
            continue
        is_ptr = "*" in raw
        if is_ptr:
            # "unsigned long int *Z" → type="unsigned long int", name="Z"
            parts = raw.split("*")
            type_str = parts[0].strip()
            name = parts[-1].strip()
            name = _re.match(r'([A-Za-z_]\w*)', name).group(1)
        else:
            # "unsigned int A" → type="unsigned int", name="A"
            tokens = raw.split()
            name = tokens[-1]
            type_str = " ".join(tokens[:-1])
        params.append((type_str, name, is_ptr))

    input_params = [(t, n) for t, n, p in params if not p]
    output_params = [(t, n) for t, n, p in params if p]

    if not output_params:
        return src  # no pointer outputs, nothing to transform

    # ── Multi-output: Bambu can't handle struct returns, so keep pointers ─
    # The memory controller overhead is a constant additive factor across
    # all transform variants, so relative DC comparisons remain valid.
    if len(output_params) > 1:
        return src

    # ── Single output: convert pointer to return value ────────────────────
    wb_marker = "V2C_PTR_WRITEBACK_PATCHED"
    wb_idx = src.find(wb_marker)

    wb_assignments = {}  # name -> expr
    if wb_idx >= 0:
        wb_section = src[wb_idx:]
        for wm in _re.finditer(r'\*(\w+)\s*=\s*([^;]+);', wb_section):
            wb_assignments[wm.group(1)] = wm.group(2).strip()

    pre_func = src[:func_m.start()]
    ret_type, ret_name = output_params[0]

    brace_pos = src.index("{", func_m.start())
    if wb_idx >= 0:
        wb_line_start = src.rfind("\n", 0, wb_idx)
        body = src[brace_pos + 1:wb_line_start]
    else:
        body_end = src.rfind("}")
        body = src[brace_pos + 1:body_end]
        # Rewrite inline *name = expr;  →  return expr;
        body = _re.sub(
            r'\*' + _re.escape(ret_name) + r'\s*=\s*([^;]+);',
            r'return \1;',
            body,
        )

    input_param_str = ", ".join(f"{t} {n}" for t, n in input_params)

    out_parts = []
    out_parts.append(pre_func.rstrip())
    out_parts.append("")
    out_parts.append(f"{ret_type} {func_name}({input_param_str})")
    out_parts.append("{")
    if wb_idx >= 0:
        out_parts.append(body.rstrip())
        ret_expr = wb_assignments.get(ret_name, f"s{func_name}.{ret_name}")
        out_parts.append(f"  return {ret_expr};")
    else:
        out_parts.append(body.rstrip())
    out_parts.append("}")

    out_parts.append("")
    result = "\n".join(out_parts)

    # Clean up excessive blank lines
    result = _re.sub(r'\n{3,}', '\n\n', result)
    return result


def run_bambu(cpp_path: str, *, clock_period: float, opt_level: str = "0",
              device: str = "") -> bool:
    """
    Run Bambu HLS for a single .c file and generate Verilog.

    Bambu is run in a local /tmp directory (same CIFS workaround as v++),
    and results are copied back next to the source file.

    Final output directory (next to the source file):
    - bambu_<name>/  (contains .v files, logs, etc.)
    """
    cpp_path = os.path.abspath(cpp_path)
    if not os.path.isfile(cpp_path):
        _tprint(f"[ERROR] source does not exist: {cpp_path}")
        return False

    synth_ok, synth_reason = check_synthesizable(cpp_path)
    if not synth_ok:
        _tprint(f"[SKIP] Not synthesizable ({synth_reason}): {cpp_path}")
        return False

    top = infer_hls_top_from_cpp_path(cpp_path)
    src_stem = Path(cpp_path).stem

    dest_dir = os.path.join(os.path.dirname(cpp_path), f"bambu_{src_stem}")

    if os.path.isdir(dest_dir):
        _tprint(f"[CLEAN] Removing existing Bambu directory: {dest_dir}")
        shutil.rmtree(dest_dir)

    with tempfile.TemporaryDirectory(prefix=f"bambu_{src_stem}_") as tmp_dir:
        # Prepare source: strip HLS pragmas and main()
        with open(cpp_path, "r", encoding="utf-8", errors="replace") as f:
            original_src = f.read()

        cleaned_src = prep_for_bambu(original_src, top)

        tmp_src = os.path.join(tmp_dir, os.path.basename(cpp_path))
        with open(tmp_src, "w", encoding="utf-8") as f:
            f.write(cleaned_src)

        cmd = [
            "bambu", tmp_src,
            f"--top-fname={top}",
            f"-O{opt_level}",
            f"--clock-period={clock_period}",
            "--generate-interface=MINIMAL",
            "--compiler=I386_CLANG16",
            "--panda-parameter=inline-max-cost=0",
            "-v2",
        ]
        if device:
            cmd.append(f"--device-name={device}")

        _tprint(f"[BAMBU] Running: {' '.join(cmd)}  (top={top})")

        stdout_path = os.path.join(tmp_dir, "bambu_stdout.txt")
        stderr_path = os.path.join(tmp_dir, "bambu_stderr.txt")
        with open(stdout_path, "w", encoding="utf-8") as out, \
             open(stderr_path, "w", encoding="utf-8") as err:
            p = subprocess.run(cmd, cwd=tmp_dir, stdout=out, stderr=err, text=True)

        shutil.copytree(tmp_dir, dest_dir)

    if p.returncode != 0:
        dest_stderr = os.path.join(dest_dir, "bambu_stderr.txt")
        _tprint(f"[ERROR] Bambu failed: {cpp_path} (rc={p.returncode})")
        _tprint(f"        Details: {dest_stderr}")
        return False

    # Check that Verilog was produced
    v_files = list(Path(dest_dir).glob("*.v"))
    if not v_files:
        _tprint(f"[WARN] Bambu returned success but no .v files found: {dest_dir}")
        return False

    # Flatten the Bambu output into a single module
    for vf in v_files:
        try:
            raw_v = vf.read_text(encoding="utf-8")
            flat_v = flatten_bambu_verilog(raw_v, top)
            vf.write_text(flat_v, encoding="utf-8")
        except Exception as e:
            _tprint(f"[WARN] Flatten failed for {vf}: {e}")

    _tprint(f"[DONE] Bambu -> {', '.join(f.name for f in v_files[:3])}  ({cpp_path})")
    return True


def collect_syn_verilog(root_dir: str, output_dir: str) -> int:
    """
    Extract HLS syn/verilog/*.v files from each project under root_dir
    and copy them to output_dir/<project_name>/.

    Matching rules:
    - Find files under paths like */hls_<name>/hls_work/hls/syn/verilog/*.v
      (also compatible with legacy */hls/hls_work/hls/syn/verilog/*.v)
    - Project name is the first-level subdirectory under root_dir
      (for example: stream_pipe_task_061_Y)
    """
    root_dir = os.path.abspath(root_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    copied = 0
    manifest_lines: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in (".git", ".idea", "__pycache__", "ai_opt")]

        normalized = dirpath.replace("\\", "/")
        if not normalized.endswith("/hls_work/hls/syn/verilog"):
            continue

        rel = os.path.relpath(dirpath, root_dir)
        rel_parts = rel.split(os.sep)
        project = rel_parts[0] if rel_parts else "unknown_project"
        proj_out = os.path.join(output_dir, project)
        os.makedirs(proj_out, exist_ok=True)

        for fn in filenames:
            if not fn.lower().endswith(".v"):
                continue
            src = os.path.join(dirpath, fn)
            dst = os.path.join(proj_out, fn)
            if os.path.exists(dst):
                base, ext = os.path.splitext(fn)
                k = 1
                while True:
                    cand = os.path.join(proj_out, f"{base}__{k}{ext}")
                    if not os.path.exists(cand):
                        dst = cand
                        break
                    k += 1

            shutil.copy2(src, dst)
            copied += 1
            manifest_lines.append(f"{src}\t{dst}")

    manifest_path = os.path.join(output_dir, "manifest.tsv")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines))
        if manifest_lines:
            f.write("\n")

    print(f"[DONE] Copied {copied} .v file(s) to: {output_dir}")
    print(f"[DONE] Manifest: {manifest_path}")
    return copied


def iter_c_files(input_dir: str, recursive: bool) -> list[str]:
    input_dir = os.path.abspath(input_dir)
    matches: list[str] = []
    exts = {".c"}

    if recursive:
        for dirpath, dirnames, filenames in os.walk(input_dir):
            dirnames[:] = [d for d in dirnames
                           if d not in (".git", ".idea", "__pycache__")
                           and not d.startswith("hls_")
                           and not d.startswith("bambu_")]
            for fn in filenames:
                if Path(fn).suffix.lower() in exts:
                    matches.append(os.path.join(dirpath, fn))
    else:
        for p in Path(input_dir).iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                matches.append(str(p))

    return sorted(set(matches))


def _dispatch_one(fp: str, args: argparse.Namespace) -> bool:
    """Dispatch a single file to the selected backend."""
    if args.backend == "bambu":
        return run_bambu(
            fp, clock_period=args.clock_period,
            opt_level=args.bambu_opt, device=args.device,
        )
    else:
        return run_vpp_hls(
            fp, part=args.part, clock=args.clock,
            flow_target=args.flow_target,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run C-to-Verilog synthesis for all .c files in a folder."
    )
    default_workers = min(os.cpu_count() or 1, 32)
    parser.add_argument("input_dir", help="Input folder containing .c sources")

    # Backend selection
    parser.add_argument(
        "--backend", default="bambu", choices=["bambu", "vitis"],
        help="Synthesis backend: 'bambu' (default, faithful C-to-V) or "
             "'vitis' (Vitis HLS v++)",
    )

    # Bambu options
    bambu_group = parser.add_argument_group("Bambu options")
    bambu_group.add_argument(
        "--bambu-opt", default="0", choices=["0", "1", "2", "3"],
        help="Bambu optimization level (default: 0 = structural fidelity)",
    )
    bambu_group.add_argument(
        "--clock-period", type=float, default=5.0,
        help="Clock period in ns (default: 5.0)",
    )
    bambu_group.add_argument(
        "--device", default="",
        help="Bambu device name (default: Bambu's built-in default)",
    )

    # Vitis options
    vitis_group = parser.add_argument_group("Vitis options")
    vitis_group.add_argument(
        "--part", default="xck24-ubva530-2LV-c",
        help="Target FPGA part (default: xck24-ubva530-2LV-c)",
    )
    vitis_group.add_argument(
        "--clock", default="5ns",
        help="HLS clock constraint (default: 5ns)",
    )
    vitis_group.add_argument(
        "--flow-target", default="vivado", choices=["vivado", "vitis"],
        help="HLS flow target: 'vivado' (default) or 'vitis'",
    )

    # Common options
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Only process .c files directly under input_dir",
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=default_workers,
        help=f"Max concurrent jobs (default: {default_workers})",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"[ERROR] input_dir is not a directory: {input_dir}")
        return 2

    recursive = not args.no_recursive
    files = iter_c_files(input_dir, recursive=recursive)
    if not files:
        print(f"[INFO] No .c files found in: {input_dir}")
        return 0

    workers = max(1, args.workers)
    backend_label = f"Bambu -O{args.bambu_opt}" if args.backend == "bambu" else "Vitis HLS"

    if workers == 1:
        ok = 0
        fail = 0
        print(f"[INFO] Found {len(files)} file(s). Starting {backend_label} (serial)...")
        for fp in files:
            if _dispatch_one(fp, args):
                ok += 1
            else:
                fail += 1
    else:
        print(f"[INFO] Found {len(files)} file(s). Starting {backend_label} "
              f"with {workers} workers...")
        ok = 0
        fail = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_dispatch_one, fp, args): fp
                for fp in files
            }
            for future in as_completed(futures):
                fp = futures[future]
                try:
                    if future.result():
                        ok += 1
                    else:
                        fail += 1
                except Exception as exc:
                    _tprint(f"[ERROR] Exception for {fp}: {exc}")
                    fail += 1
                done = ok + fail
                if done % 10 == 0 or done == len(files):
                    _tprint(f"[PROGRESS] {done}/{len(files)} "
                            f"(success={ok}, fail={fail})")

    print(f"[SUMMARY] backend={args.backend} total={len(files)} success={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
