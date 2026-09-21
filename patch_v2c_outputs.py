#!/usr/bin/env python3
"""
Patch v2c-generated C files so that struct state writes are also
written through the corresponding output pointer parameters.

Without this, Vitis HLS sees no externally visible side effects and
optimizes the entire datapath away.

Targets:
  benchmark_output/     — original v2c sources
  optimized_AREA/       — LLM area-optimised outputs
  optimized_TIMING/     — LLM timing-optimised outputs
  optimized_output_AREA/      — rule-based area outputs
  optimized_output_TIMING/    — rule-based timing outputs
  optimized_output_AREA_LLM/  — LLM area outputs (alternate dir)

The patch is idempotent: re-running on already-patched files is safe.

Usage:
    python patch_v2c_outputs.py                    # patch all dirs
    python patch_v2c_outputs.py --dirs optimized_AREA optimized_TIMING
    python patch_v2c_outputs.py --dry-run           # preview only
    python patch_v2c_outputs.py -j 16               # 16 workers
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
    # "benchmark_output",
    "optimized_AREA",
    # "original_AREA",
    # "original_TIMING",
    "optimized_TIMING",
    "optimized_output_AREA",
    "optimized_output_TIMING",
    "optimized_output_AREA_LLM",
    "optimized_output_TIMING_LLM",
]

# ─── Thread-safe print ───────────────────────────────────────────────────────

_print_lock = threading.Lock()


def _tprint(*args, **kwargs):
    kwargs.setdefault("flush", True)
    with _print_lock:
        print(*args, **kwargs)


# ─── Sentinel comment to mark already-patched files ──────────────────────────

_PATCH_MARKER = "// V2C_PTR_WRITEBACK_PATCHED"


# ─── Core analysis & patching ────────────────────────────────────────────────

# Regex to find the first non-main function definition (v2c convention:
# primary function comes before void main).
_FUNC_SIG_RE = re.compile(
    r"^(void\s+(?!main\b)(\w+)\s*\(([^)]*)\))\s*\{",
    re.MULTILINE,
)


def _parse_ptr_params(param_str: str) -> list[str]:
    """Extract output pointer parameter names from a function signature."""
    names = []
    for raw in param_str.split(","):
        raw = raw.strip()
        if "*" not in raw:
            continue
        # Handle both `type *name` and `type * name`
        after_star = raw.split("*")[-1].strip()
        # Remove any trailing paren artifacts
        name = re.match(r"([A-Za-z_]\w*)", after_star)
        if name:
            names.append(name.group(1))
    return names


def _find_struct_instance(code: str, func_name: str) -> str | None:
    """Find the struct instance variable name, e.g. 'sexample' for func 'example'."""
    m = re.search(
        r"struct\s+state_elements_" + re.escape(func_name)
        + r"\s+(\w+)\s*;",
        code,
    )
    return m.group(1) if m else None


def _find_struct_fields_written(code: str, inst_name: str) -> set[str]:
    """Return all struct field names that appear in writes: inst.field = ..."""
    return set(re.findall(re.escape(inst_name) + r"\.(\w+)\s*=", code))


def _already_deref(code: str, param: str) -> bool:
    """Check if *param = ... already exists in the code body."""
    return bool(re.search(r"\*\s*" + re.escape(param) + r"\s*=", code))


def _find_primary_func_end(code: str, sig_match: re.Match) -> int | None:
    """Find the closing brace index of the primary function.

    Counts brace nesting from the opening '{' of the function.
    """
    # Find the opening brace
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


def analyse_file(path: Path) -> dict:
    """Analyse a single .c file and return patch info.

    Returns dict with keys:
        status:  "skip_no_struct" | "skip_no_ptr_params" | "skip_already_patched"
                 | "skip_already_ok" | "patchable" | "mismatch"
        patches: list of (param, inst_name, field) tuples  (for patchable)
        unmatched: list of param names (for mismatch)
        func_name: str
    """
    try:
        code = path.read_text(errors="ignore")
    except OSError:
        return {"status": "skip_read_error"}

    # Already patched?
    if _PATCH_MARKER in code:
        return {"status": "skip_already_patched"}

    # Has state_elements struct?
    if "state_elements_" not in code:
        return {"status": "skip_no_struct"}

    # Find primary function
    sig = _FUNC_SIG_RE.search(code)
    if not sig:
        return {"status": "skip_no_struct"}

    func_name = sig.group(2)
    param_str = sig.group(3)

    ptr_params = _parse_ptr_params(param_str)
    if not ptr_params:
        return {"status": "skip_no_ptr_params", "func_name": func_name}

    # Find struct instance
    inst = _find_struct_instance(code, func_name)
    if not inst:
        return {"status": "skip_no_struct", "func_name": func_name}

    # Which pointer params already have writeback?
    needs_fix = [p for p in ptr_params if not _already_deref(code, p)]
    if not needs_fix:
        return {"status": "skip_already_ok", "func_name": func_name}

    # Which struct fields exist in writes?
    fields = _find_struct_fields_written(code, inst)

    patches = []
    unmatched = []
    for p in needs_fix:
        if p in fields:
            patches.append((p, inst, p))
        else:
            unmatched.append(p)

    if not patches and unmatched:
        return {
            "status": "mismatch",
            "func_name": func_name,
            "unmatched": unmatched,
        }

    return {
        "status": "patchable" if patches else "skip_already_ok",
        "func_name": func_name,
        "inst": inst,
        "patches": patches,
        "unmatched": unmatched,
    }


def apply_patch(path: Path, info: dict) -> bool:
    """Insert pointer writebacks just before the primary function's closing brace.

    Returns True if the file was modified.
    """
    if info["status"] != "patchable" or not info.get("patches"):
        return False

    code = path.read_text(errors="ignore")

    sig = _FUNC_SIG_RE.search(code)
    if not sig:
        return False

    close_idx = _find_primary_func_end(code, sig)
    if close_idx is None:
        return False

    inst = info["inst"]
    lines = []
    for param, _, field in info["patches"]:
        lines.append(f"  *{param} = {inst}.{field};")

    # Build the insertion block
    insert = "\n" + _PATCH_MARKER + "\n" + "\n".join(lines) + "\n"

    new_code = code[:close_idx] + insert + code[close_idx:]
    path.write_text(new_code)
    return True


# ─── Worker ──────────────────────────────────────────────────────────────────

def _process_one(path: Path, dry_run: bool) -> tuple[Path, dict, bool]:
    """Analyse and optionally patch a single file. Thread-safe."""
    info = analyse_file(path)
    modified = False
    if not dry_run and info["status"] == "patchable":
        modified = apply_patch(path, info)
    return path, info, modified


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Patch v2c outputs: insert pointer writebacks for state_elements structs."
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
        help="Write JSON report to this path (default: patch_v2c_report.json)",
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
            # Skip cdfg / HLS artifact directories
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
        "skip_no_struct": 0,
        "skip_no_ptr_params": 0,
        "skip_already_patched": 0,
        "skip_already_ok": 0,
        "skip_read_error": 0,
        "patchable": 0,
        "mismatch": 0,
    }
    patched_files: list[str] = []
    mismatch_files: list[dict] = []
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

            if modified:
                n = len(info.get("patches", []))
                _tprint(f"  [PATCHED] {path.relative_to(_BASE)} "
                        f"(+{n} writeback{'s' if n != 1 else ''})")
                patched_files.append(str(path.relative_to(_BASE)))
            elif status == "patchable" and args.dry_run:
                n = len(info.get("patches", []))
                _tprint(f"  [WOULD PATCH] {path.relative_to(_BASE)} "
                        f"(+{n} writeback{'s' if n != 1 else ''})")
                patched_files.append(str(path.relative_to(_BASE)))
            elif status == "mismatch":
                mismatch_files.append({
                    "file": str(path.relative_to(_BASE)),
                    "func": info.get("func_name", "?"),
                    "unmatched_params": info.get("unmatched", []),
                })

            if done % 200 == 0 or done == len(all_files):
                _tprint(f"[PROGRESS] {done}/{len(all_files)}")

    # Summary
    _tprint(f"\n{'DRY RUN ' if args.dry_run else ''}Summary:")
    _tprint(f"  Patched / would patch:  {counters['patchable']}")
    _tprint(f"  Mismatch (skipped):     {counters['mismatch']}")
    _tprint(f"  Already patched:        {counters['skip_already_patched']}")
    _tprint(f"  Already has ptr writes: {counters['skip_already_ok']}")
    _tprint(f"  No state_elements:      {counters['skip_no_struct']}")
    _tprint(f"  No pointer params:      {counters['skip_no_ptr_params']}")

    if mismatch_files:
        _tprint(f"\nMismatch files ({len(mismatch_files)}):")
        for mf in mismatch_files[:20]:
            _tprint(f"  {mf['file']}  params={mf['unmatched_params']}")
        if len(mismatch_files) > 20:
            _tprint(f"  ... and {len(mismatch_files) - 20} more")

    # Write report
    report_path = args.report or str(_BASE / "patch_v2c_report.json")
    report = {
        "dry_run": args.dry_run,
        "counters": counters,
        "patched_files": patched_files,
        "mismatch_files": mismatch_files,
    }
    Path(report_path).write_text(json.dumps(report, indent=2))
    _tprint(f"\nReport → {report_path}")


if __name__ == "__main__":
    main()
