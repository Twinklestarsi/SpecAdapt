#!/usr/bin/env python3
"""
Inject HLS interface pragmas into v2c-generated C files so that
Vitis HLS produces wire-level Verilog (no AXI wrappers), making
the output structurally comparable to the original RTL for DC evaluation.

Pragmas injected:
  #pragma HLS INTERFACE ap_ctrl_none port=return
  #pragma HLS INTERFACE ap_none      port=<scalar_param>
  #pragma HLS INTERFACE ap_ovld      port=<pointer_param>

The injection is idempotent: re-running on already-injected files is safe.

Usage:
    python inject_hls_pragmas.py                        # inject in default dirs
    python inject_hls_pragmas.py --dirs original_AREA   # specific dirs
    python inject_hls_pragmas.py --dry-run              # preview only
    python inject_hls_pragmas.py -j 16                  # 16 workers
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
    # "original_AREA",
    # "original_TIMING",
    # "optimized_AREA",
    # "optimized_TIMING",
    # "optimized_output_AREA",
    # "optimized_output_TIMING",
    # "optimized_output_AREA_LLM",
    # "optimized_output_TIMING_LLM",
    "CDFG_AREA",
    "CDFG_TIMING",
    "CDFG_HLS_AREA",
    "CDFG_HLS_TIMING",
]

# ─── Thread-safe print ───────────────────────────────────────────────────────

_print_lock = threading.Lock()


def _tprint(*args, **kwargs):
    kwargs.setdefault("flush", True)
    with _print_lock:
        print(*args, **kwargs)


# ─── Sentinel comment to mark already-injected files ─────────────────────────

_PRAGMA_MARKER = "// HLS_INTERFACE_PRAGMAS_INJECTED"

# ─── Regex for primary function (first non-main void function) ───────────────

_FUNC_SIG_RE = re.compile(
    r"^(void\s+(?!main\b)(\w+)\s*\(([^)]*)\))\s*\{",
    re.MULTILINE,
)


# ─── Parameter parsing ───────────────────────────────────────────────────────

def _parse_params(param_str: str) -> list[tuple[bool, str]]:
    """Parse function parameters.

    Returns list of (is_pointer, param_name) tuples.
    """
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
            # Last identifier in the declaration
            m = re.search(r"([A-Za-z_]\w*)\s*$", raw)
        if m:
            params.append((is_ptr, m.group(1)))
    return params


# ─── Pragma generation ───────────────────────────────────────────────────────

def _build_pragma_block(params: list[tuple[bool, str]]) -> str:
    """Build the pragma block string to insert after the opening brace."""
    lines = [_PRAGMA_MARKER]
    lines.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
    for is_ptr, name in params:
        if is_ptr:
            lines.append(f"#pragma HLS INTERFACE ap_ovld port={name}")
        else:
            lines.append(f"#pragma HLS INTERFACE ap_none port={name}")
    return "\n".join(lines) + "\n"


# ─── Core analysis & injection ───────────────────────────────────────────────

def analyse_file(path: Path) -> dict:
    """Analyse a single .c file and return injection info.

    Returns dict with keys:
        status:    "skip_already_injected" | "skip_no_function" |
                   "skip_no_params" | "injectable"
        func_name: str
        params:    list of (is_pointer, name) tuples  (for injectable)
    """
    try:
        code = path.read_text(errors="ignore")
    except OSError:
        return {"status": "skip_read_error"}

    if _PRAGMA_MARKER in code:
        return {"status": "skip_already_injected"}

    sig = _FUNC_SIG_RE.search(code)
    if not sig:
        return {"status": "skip_no_function"}

    func_name = sig.group(2)
    param_str = sig.group(3)
    params = _parse_params(param_str)

    if not params:
        return {"status": "skip_no_params", "func_name": func_name}

    return {
        "status": "injectable",
        "func_name": func_name,
        "params": params,
        "sig_match_start": sig.start(),
        "sig_match_end": sig.end(),
    }


def apply_injection(path: Path, info: dict) -> bool:
    """Insert HLS interface pragmas after the function's opening brace.

    Returns True if the file was modified.
    """
    if info["status"] != "injectable":
        return False

    code = path.read_text(errors="ignore")

    # Re-find the function signature (file unchanged since analyse)
    sig = _FUNC_SIG_RE.search(code)
    if not sig:
        return False

    # Find the opening brace
    brace_pos = code.index("{", sig.end() - 1)
    insert_pos = brace_pos + 1

    pragma_block = _build_pragma_block(info["params"])

    new_code = code[:insert_pos] + "\n" + pragma_block + code[insert_pos:]
    path.write_text(new_code)
    return True


# ─── Worker ──────────────────────────────────────────────────────────────────

def _process_one(path: Path, dry_run: bool) -> tuple[Path, dict, bool]:
    """Analyse and optionally inject pragmas into a single file. Thread-safe."""
    info = analyse_file(path)
    modified = False
    if not dry_run and info["status"] == "injectable":
        modified = apply_injection(path, info)
    return path, info, modified


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inject HLS interface pragmas (ap_none/ap_ovld/ap_ctrl_none) "
                    "into v2c-generated C files for wire-level HLS output."
    )
    parser.add_argument(
        "--dirs", nargs="+", default=None,
        help="Directories to scan (relative to project root, or absolute). "
             f"Default: {' '.join(DEFAULT_DIRS)}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyse only — do not modify any files",
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=min(os.cpu_count() or 4, 32),
        help="Max parallel workers (default: min(cpu_count, 32))",
    )
    parser.add_argument(
        "--report", default=None,
        help="Write JSON report to this path (default: inject_hls_report.json)",
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

    # Collect all .c files
    all_files: list[Path] = []
    for sd in scan_dirs:
        for dirpath, dirnames, filenames in os.walk(sd):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith("cdfg") and not d.startswith("hls_")
                and d not in (".git", "__pycache__", ".Xil")
            ]
            for fn in filenames:
                if fn.endswith(".c"):
                    all_files.append(Path(dirpath) / fn)

    all_files.sort()
    workers = max(1, args.workers)
    _tprint(f"Scanning {len(all_files)} .c files across {len(scan_dirs)} "
            f"directories, {workers} workers"
            f"{' (dry run)' if args.dry_run else ''}\n")

    # Process
    counters = {
        "skip_no_function": 0,
        "skip_no_params": 0,
        "skip_already_injected": 0,
        "skip_read_error": 0,
        "injectable": 0,
    }
    injected_files: list[str] = []
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
                n_ptr = sum(1 for is_ptr, _ in info.get("params", []) if is_ptr)
                n_scl = sum(1 for is_ptr, _ in info.get("params", []) if not is_ptr)
                _tprint(f"  [INJECTED] {rel} "
                        f"({info['func_name']}: {n_scl} ap_none, {n_ptr} ap_ovld)")
                injected_files.append(rel)
            elif status == "injectable" and args.dry_run:
                n_ptr = sum(1 for is_ptr, _ in info.get("params", []) if is_ptr)
                n_scl = sum(1 for is_ptr, _ in info.get("params", []) if not is_ptr)
                _tprint(f"  [WOULD INJECT] {rel} "
                        f"({info['func_name']}: {n_scl} ap_none, {n_ptr} ap_ovld)")
                injected_files.append(rel)

            if done % 200 == 0 or done == len(all_files):
                _tprint(f"[PROGRESS] {done}/{len(all_files)}")

    # Summary
    _tprint(f"\n{'DRY RUN ' if args.dry_run else ''}Summary:")
    _tprint(f"  Injected / would inject:  {counters['injectable']}")
    _tprint(f"  Already injected:         {counters['skip_already_injected']}")
    _tprint(f"  No function found:        {counters['skip_no_function']}")
    _tprint(f"  No params:                {counters['skip_no_params']}")

    # Write report
    report_path = args.report or str(_BASE / "inject_hls_report.json")
    report = {
        "dry_run": args.dry_run,
        "counters": counters,
        "injected_files": injected_files,
    }
    Path(report_path).write_text(json.dumps(report, indent=2))
    _tprint(f"\nReport → {report_path}")


if __name__ == "__main__":
    main()
