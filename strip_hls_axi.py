#!/usr/bin/env python3
"""
Strip AXI-Lite wrappers from Vitis HLS-generated Verilog, producing
clean datapath modules suitable for DC synthesis comparison against
the original RTL baselines.

For each top-level .v file under CDFG_HLS_*/…/syn/verilog/, creates
a *_stripped.v alongside it with the AXI infrastructure removed and
functional signals promoted to module ports.

The stripping is idempotent: existing _stripped.v files are overwritten.

Usage:
    python strip_hls_axi.py                         # process all
    python strip_hls_axi.py --dirs CDFG_HLS_AREA    # specific dir
    python strip_hls_axi.py --dry-run                # preview only
    python strip_hls_axi.py -j 16                    # 16 workers
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

_BASE = Path(__file__).resolve().parent

DEFAULT_DIRS = [
    "CDFG_HLS_AREA",
    "CDFG_HLS_TIMING",
]

# ─── Thread-safe print ───────────────────────────────────────────────────────

_print_lock = threading.Lock()


def _tprint(*args, **kwargs):
    kwargs.setdefault("flush", True)
    with _print_lock:
        print(*args, **kwargs)


# ─── AXI / control port names (to be excluded) ──────────────────────────────

_AXI_PORTS = {
    "AWVALID", "AWREADY", "AWADDR", "WVALID", "WREADY", "WDATA", "WSTRB",
    "ARVALID", "ARREADY", "ARADDR", "RVALID", "RREADY", "RDATA", "RRESP",
    "BVALID", "BREADY", "BRESP", "ACLK", "ARESET", "ACLK_EN",
}

_CTRL_SIGNALS = {
    "ap_start", "ap_done", "ap_idle", "ap_ready", "ap_continue", "interrupt",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _extract_latency(code: str) -> int:
    """Extract HLS_SYN_LAT from CORE_GENERATION_INFO."""
    m = re.search(r"HLS_SYN_LAT=(\d+)", code)
    return int(m.group(1)) if m else -1


def _extract_module_name(code: str) -> str | None:
    m = re.search(r"^\s*module\s+(\w+)\s*\(", code, re.MULTILINE)
    return m.group(1) if m else None


def _extract_functional_signals(code: str) -> list[tuple[str, str]]:
    """Parse control_s_axi_U instantiation for functional signal mappings.

    Returns list of (axi_port_name, internal_wire_name) tuples.
    """
    m = re.search(r"control_s_axi_U\s*\((.*?)\)\s*;", code, re.DOTALL)
    if not m:
        return []

    block = m.group(1)
    signals = []
    for pm in re.finditer(r"\.(\w+)\s*\(([^)]*)\)", block):
        port_name = pm.group(1)
        wire_name = pm.group(2).strip()

        if port_name in _AXI_PORTS or port_name in _CTRL_SIGNALS:
            continue
        if wire_name in ("1'b1", "1'b0", "") or wire_name.startswith("ap_"):
            continue

        signals.append((port_name, wire_name))

    return signals


def _get_signal_info(code: str, wire_name: str) -> tuple[str, str | None]:
    """Determine direction ('input'/'output') and width string for a signal.

    Returns (direction, width_str_or_None).
    """
    esc = re.escape(wire_name)

    # Check reg declaration
    reg_m = re.search(
        rf"(?:^|\s)reg\s+(?:signed\s+)?(?:\[([^\]]+)\]\s+)?{esc}\s*;",
        code, re.MULTILINE,
    )
    if reg_m:
        return "output reg", reg_m.group(1)

    # Check wire declaration
    wire_m = re.search(
        rf"(?:^|\s)wire\s+(?:signed\s+)?(?:\[([^\]]+)\]\s+)?{esc}\s*;",
        code, re.MULTILINE,
    )
    width = wire_m.group(1) if wire_m else None

    # If driven by an assign → output
    if re.search(rf"\bassign\s+{esc}\s*=", code):
        return "output", width

    return "input", width


# ─── Submodule instantiation block removal ───────────────────────────────────


def _find_inst_block(code: str, inst_name: str) -> tuple[int, int] | None:
    """Find the start..end byte range of a submodule instantiation block.

    Matches:  <module_type> #( ... ) <inst_name> ( ... );
    or:       <module_type> <inst_name> ( ... );
    """
    # Look for the instance name with optional # parameter block before it
    pattern = re.compile(
        r"^\w+(?:_\w+)*\s+#\s*\(.*?\)\s*\n\s*"
        + re.escape(inst_name)
        + r"\s*\(.*?\)\s*;",
        re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(code)
    if m:
        return m.start(), m.end()

    # Simpler: no parameter block
    pattern2 = re.compile(
        r"^\w+(?:_\w+)*\s+"
        + re.escape(inst_name)
        + r"\s*\(.*?\)\s*;",
        re.DOTALL | re.MULTILINE,
    )
    m2 = pattern2.search(code)
    if m2:
        return m2.start(), m2.end()

    return None


# ─── Core stripping ─────────────────────────────────────────────────────────


def strip_axi_wrapper(code: str) -> tuple[str | None, dict]:
    """Strip AXI-Lite wrapper from HLS Verilog.

    Returns (stripped_code, info_dict) or (None, info_dict) on failure.
    """
    module_name = _extract_module_name(code)
    if not module_name:
        return None, {"status": "skip_no_module"}

    lat = _extract_latency(code)
    is_comb = lat == 0

    func_signals = _extract_functional_signals(code)
    if not func_signals:
        return None, {"status": "skip_no_axi_wrapper"}

    # Gather signal info
    sig_info: dict[str, tuple[str, str | None]] = {}
    for _, wire_name in func_signals:
        sig_info[wire_name] = _get_signal_info(code, wire_name)

    # ── Build stripped code ───────────────────────────────────────────────

    lines = code.split("\n")
    out: list[str] = []

    # -- Header comment
    out.append(f"// Stripped from Vitis HLS output — AXI-Lite wrapper removed")
    out.append(f"// Original latency: {lat}  ({'combinational' if is_comb else 'sequential'})")
    out.append("")
    out.append("`timescale 1 ns / 1 ps")
    out.append("")

    # -- Module declaration
    port_names: list[str] = []
    if not is_comb:
        port_names += ["ap_clk", "ap_rst_n"]
    for _, wn in func_signals:
        port_names.append(wn)

    out.append(f"module {module_name} (")
    for i, pn in enumerate(port_names):
        comma = "," if i < len(port_names) - 1 else ""
        out.append(f"        {pn}{comma}")
    out.append(");")
    out.append("")

    # -- Port declarations
    if not is_comb:
        out.append("input   ap_clk;")
        out.append("input   ap_rst_n;")

    for _, wire_name in func_signals:
        direction, width = sig_info[wire_name]
        if width:
            out.append(f"{direction}  [{width}] {wire_name};")
        else:
            out.append(f"{direction}   {wire_name};")
    out.append("")

    # -- Reset handling
    if not is_comb:
        out.append("wire    ap_rst_n_inv;")
        out.append("assign ap_rst_n_inv = ~ap_rst_n;")
        out.append("")

    # -- Process remaining code: extract everything after port declarations
    #    up to endmodule, filtering out AXI infrastructure

    # Find the end of the original port declarations section
    # (after the last input/output line before the first wire/reg/assign/always)
    in_body = False
    skip_block = False
    paren_depth = 0
    i = 0

    # Patterns to skip (line-level)
    skip_line_pats = [
        re.compile(r"^\s*module\s"),
        re.compile(r"^\s*`timescale"),
        re.compile(r"^\s*\(\*\s*CORE_GENERATION_INFO"),
        re.compile(r"^\s*(input|output)\s+.*\bs_axi_control_"),
        re.compile(r"^\s*(input|output)\s+.*\bm_axi_gmem_"),
        re.compile(r"^\s*(input|output)\s+.*\bap_clk\b"),
        re.compile(r"^\s*(input|output)\s+.*\bap_rst_n\b"),
        re.compile(r"^\s*parameter\s+C_S_AXI_"),
        re.compile(r"^\s*parameter\s+C_M_AXI_"),
        re.compile(r"^\s*parameter\s+C_S_AXI_\w+_WSTRB_WIDTH"),
        re.compile(r"^\s*parameter\s+C_M_AXI_\w+_WSTRB_WIDTH"),
        re.compile(r"^\s*parameter\s+C_S_AXI_DATA_WIDTH"),
        re.compile(r"^\s*parameter\s+C_M_AXI_DATA_WIDTH"),
        re.compile(r"^\s*parameter\s+C_S_AXI_WSTRB_WIDTH"),
        re.compile(r"^\s*parameter\s+C_M_AXI_WSTRB_WIDTH"),
        # Reset sync registers
        re.compile(r".*\bap_rst_reg_[12]\b"),
        re.compile(r".*shreg_extract.*ap_rst"),
        # AXI infrastructure wires
        re.compile(r"^\s*(wire|reg)\s+.*\bs_axi_control_"),
        re.compile(r"^\s*(wire|reg)\s+.*\bm_axi_gmem_"),
        re.compile(r"^\s*(wire|reg)\s+.*\bgmem_blk_n_"),
        re.compile(r"^\s*(wire|reg)\s+.*\bgmem_\d+_"),
        # HLS control signals
        re.compile(r"^\s*(wire|reg)\s+\b(ap_start|ap_done_reg|ap_idle|ap_ready|ap_continue)\b"),
        re.compile(r"^\s*wire\s+\bap_ce_reg\b"),
        re.compile(r"^\s*wire\s+\binterrupt\b"),
        re.compile(r"^\s*reg\s+\bap_done\b\s*;"),
        # _blk signals
        re.compile(r"^\s*(wire|reg)\s+\bap_ST_fsm_state\d+_blk\b"),
        re.compile(r"^\s*assign\s+ap_ST_fsm_state\d+_blk\s*="),
        # Block state signals
        re.compile(r"^\s*reg\s+\bap_block_state\d+"),
    ]

    # Functional signals we already declared as ports — remove their
    # wire/reg declarations from the body to avoid redeclaration
    func_wire_names = {wn for _, wn in func_signals}
    for wn in func_wire_names:
        skip_line_pats.append(
            re.compile(r"^\s*(wire|reg)\s+(?:signed\s+)?(?:\[[^\]]+\]\s+)?"
                       + re.escape(wn) + r"\s*;")
        )

    # Process body
    # First, remove submodule instantiation blocks
    # Remove control_s_axi_U block
    ctrl_block = _find_inst_block(code, "control_s_axi_U")
    gmem_block = _find_inst_block(code, "gmem_m_axi_U")

    # Build a set of line ranges to skip (convert byte offsets to line numbers)
    skip_line_ranges: list[tuple[int, int]] = []
    for block_range in [ctrl_block, gmem_block]:
        if block_range:
            start_byte, end_byte = block_range
            start_line = code[:start_byte].count("\n")
            end_line = code[:end_byte].count("\n")
            skip_line_ranges.append((start_line, end_line))

    # ── Find always/initial blocks to remove ──
    # Strategy: find all always/initial block start lines, determine their
    # end lines by tracking begin/end depth, then check if the block body
    # contains signals we want to remove.
    # Only match always blocks — initial blocks are handled by line-level
    # filtering which preserves FSM initialization while removing reset regs
    _always_start_re = re.compile(
        r"^\s*(always\s*@\s*\([^)]*\)\s*begin)", re.MULTILINE
    )
    _rst_sync_sigs = {"ap_rst_n_inv", "ap_rst_reg_1", "ap_rst_reg_2"}
    _ctrl_remove_sigs = {
        "ap_done", "ap_idle", "ap_ready",
        "gmem_blk_n_AW", "gmem_blk_n_W", "gmem_blk_n_B",
        "gmem_blk_n_AR", "gmem_blk_n_R",
    }

    for m in _always_start_re.finditer(code):
        block_start_line = code[:m.start()].count("\n")
        # Find matching 'end' by tracking begin/end depth
        depth = 1
        block_end_line = block_start_line
        for scan_idx in range(block_start_line + 1, len(lines)):
            sline = lines[scan_idx].strip()
            # Count nested begin/end
            if sline == "begin" or sline.endswith(" begin"):
                depth += 1
            if sline == "end" or sline.startswith("end "):
                depth -= 1
                if depth == 0:
                    block_end_line = scan_idx
                    break

        if depth != 0:
            continue  # couldn't find matching end, skip

        # Extract block body text
        block_body = "\n".join(lines[block_start_line:block_end_line + 1])

        # Check if this is a reset synchronizer block
        is_rst_sync = any(sig in block_body for sig in _rst_sync_sigs)
        # Check if only contains reset sync (no functional signals)
        if is_rst_sync:
            has_functional = False
            for _, wn in func_signals:
                if wn in block_body:
                    has_functional = True
                    break
            # If block mixes reset sync with functional logic (e.g. FSM reset),
            # don't remove the whole block — line-level filtering will handle it
            if not has_functional:
                skip_line_ranges.append((block_start_line, block_end_line))
                continue

        # Check if this is a control-only block (ap_done, ap_idle, etc.)
        is_ctrl = any(sig in block_body for sig in _ctrl_remove_sigs)
        if is_ctrl:
            has_functional = False
            for _, wn in func_signals:
                if wn in block_body:
                    has_functional = True
                    break
            if not has_functional:
                skip_line_ranges.append((block_start_line, block_end_line))
                continue

        # Check if this is an ap_block_state or ap_ST_fsm_state*_blk block
        if re.search(r"ap_block_state\d+|ap_ST_fsm_state\d+_blk", block_body):
            has_functional = False
            for _, wn in func_signals:
                if wn in block_body:
                    has_functional = True
                    break
            if not has_functional:
                skip_line_ranges.append((block_start_line, block_end_line))
                continue

    def _in_skip_range(lineno: int) -> bool:
        for s, e in skip_line_ranges:
            if s <= lineno <= e:
                return True
        return False

    # Find where the body starts (after the module port closing paren)
    body_start = 0
    for idx, line in enumerate(lines):
        if re.match(r"^\s*\);\s*$", line):
            body_start = idx + 1
            break

    # Also handle initial blocks: keep FSM init, remove reset reg init
    in_initial = False
    initial_depth = 0

    body_lines: list[str] = []
    for idx in range(body_start, len(lines)):
        line = lines[idx]

        # Skip if in a removed block range
        if _in_skip_range(idx):
            continue

        # Skip lines matching patterns
        if any(p.search(line) for p in skip_line_pats):
            continue

        # Skip module/endmodule port list remnants
        if idx < body_start:
            continue

        # Handle initial blocks: filter out reset reg initialization
        stripped = line.strip()
        if stripped.startswith("initial begin"):
            in_initial = True
            initial_depth = 0
            body_lines.append(line)
            continue

        if in_initial:
            if "ap_rst_reg" in line or "ap_rst_n_inv" in line:
                continue
            if "ap_done_reg" in line:
                continue
            if "begin" in stripped:
                initial_depth += 1
            if stripped == "end":
                if initial_depth > 0:
                    initial_depth -= 1
                else:
                    in_initial = False
            body_lines.append(line)
            continue

        # Handle ap_done_reg always block (reset and update)
        # Skip ap_done_reg related always blocks
        if "ap_done_reg" in line and "always" not in line:
            # Single-line references to ap_done_reg — keep only if
            # it's not a declaration or reset
            pass

        body_lines.append(line)

    # Add body lines
    out.extend(body_lines)

    # Add dummy drivers for control signals that FSM may depend on
    # Insert before endmodule
    endmod_idx = None
    for i in range(len(out) - 1, -1, -1):
        if "endmodule" in out[i]:
            endmod_idx = i
            break

    if endmod_idx is not None:
        inserts = []
        # Check if ap_start is referenced but not driven
        stripped_code = "\n".join(out)
        if "ap_start" in stripped_code and "assign ap_start" not in stripped_code:
            inserts.append("assign ap_start = 1'b1;")
        if "ap_continue" in stripped_code and "assign ap_continue" not in stripped_code:
            inserts.append("assign ap_continue = 1'b1;")
        if "ap_done_reg" in stripped_code:
            if "reg" not in "".join(
                l for l in out if "ap_done_reg" in l and "assign" not in l
            ):
                pass  # already declared or not needed

        if inserts:
            out.insert(endmod_idx, "")
            for ins in inserts:
                out.insert(endmod_idx, ins)
            out.insert(endmod_idx, "// Tie off HLS control signals")

    result = "\n".join(out)

    # Clean up: remove empty always/initial blocks left after line filtering
    # Matches: always @ (...) begin\n[whitespace only]\nend
    result = re.sub(
        r"(?:// power-on initialization\n)?"
        r"always\s*@\s*\([^)]*\)\s*begin\s*\n\s*end\n?",
        "", result,
    )
    result = re.sub(
        r"(?:// power-on initialization\n)?"
        r"initial\s+begin\s*\n\s*end\n?",
        "", result,
    )

    # Clean up: remove excessive blank lines (more than 2 consecutive)
    result = re.sub(r"\n{4,}", "\n\n\n", result)

    info = {
        "status": "stripped",
        "module_name": module_name,
        "latency": lat,
        "combinational": is_comb,
        "functional_signals": len(func_signals),
        "signal_details": {wn: sig_info[wn][0] for _, wn in func_signals},
    }

    return result, info


# ─── File discovery ──────────────────────────────────────────────────────────


def _find_top_verilog_files(scan_dirs: list[Path]) -> list[Path]:
    """Find top-level HLS Verilog files (exclude _control_s_axi, _gmem_m_axi)."""
    files = []
    for sd in scan_dirs:
        for dirpath, dirnames, filenames in os.walk(sd):
            dp = Path(dirpath)
            # Only look in syn/verilog directories
            if dp.name != "verilog" or dp.parent.name != "syn":
                continue
            for fn in filenames:
                if not fn.endswith(".v"):
                    continue
                if "_control_s_axi" in fn or "_gmem_m_axi" in fn:
                    continue
                # Check for other known infrastructure modules
                if "_flow_control" in fn or "_mul_" in fn or "_sparsemux_" in fn:
                    continue
                files.append(dp / fn)
    return sorted(files)


# ─── Worker ──────────────────────────────────────────────────────────────────


def _process_one(
    path: Path, dry_run: bool
) -> tuple[Path, dict, bool]:
    """Analyse and optionally strip a single file. Thread-safe."""
    try:
        code = path.read_text(errors="ignore")
    except OSError:
        return path, {"status": "skip_read_error"}, False

    if "control_s_axi_U" not in code:
        return path, {"status": "skip_no_axi_wrapper"}, False

    stripped_code, info = strip_axi_wrapper(code)

    if stripped_code is None:
        return path, info, False

    if not dry_run:
        out_path = path.with_name(path.stem + "_stripped.v")
        out_path.write_text(stripped_code)

    return path, info, True


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Strip AXI-Lite wrappers from HLS Verilog for DC evaluation."
    )
    parser.add_argument(
        "--dirs", nargs="+", default=None,
        help="Directories to scan. "
             f"Default: {' '.join(DEFAULT_DIRS)}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyse only — do not write stripped files",
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=min(os.cpu_count() or 4, 32),
        help="Max parallel workers",
    )
    parser.add_argument(
        "--report", default=None,
        help="Write JSON report to this path",
    )
    args = parser.parse_args()

    dir_names = args.dirs or DEFAULT_DIRS
    scan_dirs = []
    for d in dir_names:
        p = Path(d) if os.path.isabs(d) else _BASE / d
        if p.is_dir():
            scan_dirs.append(p)
        else:
            _tprint(f"[WARN] Directory does not exist, skipping: {p}")

    if not scan_dirs:
        _tprint("[ERROR] No directories to scan.")
        return

    all_files = _find_top_verilog_files(scan_dirs)
    workers = max(1, args.workers)
    _tprint(f"Scanning {len(all_files)} top-level .v files across {len(scan_dirs)} "
            f"directories, {workers} workers"
            f"{' (dry run)' if args.dry_run else ''}\n")

    counters: dict[str, int] = {}
    stripped_files: list[dict] = []
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, fp, args.dry_run): fp
            for fp in all_files
        }
        for future in as_completed(futures):
            done += 1
            path, info, modified = future.result()
            status = info.get("status", "skip_read_error")
            counters[status] = counters.get(status, 0) + 1

            try:
                rel = str(path.relative_to(_BASE))
            except ValueError:
                rel = str(path)

            if modified:
                tag = "WOULD STRIP" if args.dry_run else "STRIPPED"
                n_sig = info.get("functional_signals", 0)
                lat = info.get("latency", "?")
                comb = " comb" if info.get("combinational") else f" lat={lat}"
                _tprint(f"  [{tag}] {rel} "
                        f"({info['module_name']}: {n_sig} signals,{comb})")
                stripped_files.append({
                    "file": rel,
                    "module": info.get("module_name"),
                    "latency": lat,
                    "signals": info.get("signal_details", {}),
                })

            if done % 100 == 0 or done == len(all_files):
                _tprint(f"[PROGRESS] {done}/{len(all_files)}")

    # Summary
    _tprint(f"\n{'DRY RUN ' if args.dry_run else ''}Summary:")
    _tprint(f"  Stripped / would strip: {counters.get('stripped', 0)}")
    _tprint(f"  No AXI wrapper:        {counters.get('skip_no_axi_wrapper', 0)}")
    _tprint(f"  No module found:       {counters.get('skip_no_module', 0)}")
    _tprint(f"  Read errors:           {counters.get('skip_read_error', 0)}")

    # Write report
    report_path = args.report or str(_BASE / "strip_hls_report.json")
    report = {
        "dry_run": args.dry_run,
        "counters": counters,
        "stripped_files": stripped_files,
    }
    Path(report_path).write_text(json.dumps(report, indent=2))
    _tprint(f"\nReport → {report_path}")


if __name__ == "__main__":
    main()
