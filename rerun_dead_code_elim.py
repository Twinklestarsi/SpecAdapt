#!/usr/bin/env python3
"""
Re-run DEAD_CODE_ELIM LLM transform only, overwriting existing outputs in place.

Uses the same LLM invocation flow as apply_transforms_area.py --llm but scoped
to the single DEAD_CODE_ELIM transform.  Existing _DEAD_CODE_ELIM.c files are
regenerated (not skipped).

Usage:
    python rerun_dead_code_elim.py                # all files
    python rerun_dead_code_elim.py --category arithmetic
    python rerun_dead_code_elim.py --files path/to/a.c path/to/b.c
    python rerun_dead_code_elim.py -j 4 --rpm 30  # throttle
"""

import json
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Paths ────────────────────────────────────────────────────────────────────

BENCHMARK_DIR  = Path(__file__).parent / "benchmark_output"
LLM_OUTPUT_DIR = Path(__file__).parent / "optimized_output_AREA_LLM"
TRANSFORM      = "DEAD_CODE_ELIM"

# ─── Per-file summary locks ──────────────────────────────────────────────────

_summary_locks: dict[str, threading.Lock] = {}
_summary_locks_lock = threading.Lock()


def _get_summary_lock(path: str) -> threading.Lock:
    with _summary_locks_lock:
        if path not in _summary_locks:
            _summary_locks[path] = threading.Lock()
        return _summary_locks[path]


# ─── Helpers (mirror apply_transforms_area.py) ───────────────────────────────

def _llm_output_path(c_file: Path) -> Path:
    rel = c_file.relative_to(BENCHMARK_DIR)
    out_dir = LLM_OUTPUT_DIR / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{rel.stem}_{TRANSFORM}.c"


def _llm_summary_path(c_file: Path) -> Path:
    rel = c_file.relative_to(BENCHMARK_DIR)
    return LLM_OUTPUT_DIR / rel.parent / f"{rel.stem}_transforms.json"


def _run_one(client, c_file, description, optimization_goal):
    """Re-run DEAD_CODE_ELIM for a single file. Always overwrites."""
    from llm_transform import apply_llm_transform, tprint

    rel = str(c_file.relative_to(BENCHMARK_DIR))
    out_file = _llm_output_path(c_file)
    sp = _llm_summary_path(c_file)
    sp_lock = _get_summary_lock(str(sp))
    short = Path(rel).stem

    source = c_file.read_text()
    tprint(f"  [{short}] {TRANSFORM} ...", flush=True)

    try:
        code, meta, differs = apply_llm_transform(
            client, source, rel, TRANSFORM, description,
            optimization_goal=optimization_goal,
        )
        out_file.write_text(code)
        result = {
            "output_file": str(out_file.relative_to(LLM_OUTPUT_DIR)),
            "applied": meta.get("applied", True),
            "differs": differs,
            "summary": meta.get("summary", ""),
            "mode": "llm",
        }
        status = "CHANGED" if differs else "SAME"
        applied = "applied" if meta.get("applied", True) else "no-op"
        tprint(f"  [{short}] {TRANSFORM} done [{status}, {applied}]")
    except Exception as e:
        tprint(f"  [{short}] {TRANSFORM} ERROR: {e}")
        result = {
            "output_file": None, "applied": False,
            "differs": False, "summary": f"ERROR: {e}",
            "mode": "llm",
        }

    # Atomic summary update
    with sp_lock:
        existing = {}
        if sp.exists():
            try:
                existing = json.loads(sp.read_text())
            except Exception:
                pass
        existing[TRANSFORM] = result
        sp.write_text(json.dumps(existing, indent=2))

    return rel, result


def run_all(client, c_files, description, *, optimization_goal="area",
            workers=8):
    from llm_transform import tprint

    total = len(c_files)
    tprint(f"[RERUN] {total} files x {TRANSFORM}, {workers} workers\n")

    grand: dict[str, dict] = {}
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one, client, cf, description, optimization_goal): cf
            for cf in c_files
        }
        for future in as_completed(futures):
            done += 1
            try:
                rel, result = future.result()
                grand[rel] = {TRANSFORM: result}
            except Exception as exc:
                cf = futures[future]
                rel = str(cf.relative_to(BENCHMARK_DIR))
                tprint(f"  [FATAL] {rel}: {exc}")
                grand[rel] = {TRANSFORM: {
                    "output_file": None, "applied": False,
                    "differs": False, "summary": f"FATAL: {exc}",
                    "mode": "llm",
                }}
            if done % 20 == 0 or done == total:
                tprint(f"[PROGRESS] {done}/{total}")

    return grand


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Re-run DEAD_CODE_ELIM LLM transform (overwrites existing)")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Specific .c files to process")
    parser.add_argument("--category", default=None,
                        help="Limit to a benchmark category (e.g. arithmetic)")
    parser.add_argument("--llm-model", default=None,
                        help="Override LLM model name")
    parser.add_argument("--workers", "-j", type=int, default=8,
                        help="Max concurrent LLM requests (default: 8)")
    parser.add_argument("--rpm", type=int, default=60,
                        help="API rate limit in requests/minute (default: 60)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files that would be processed, then exit")
    args = parser.parse_args()

    # Collect files
    if args.files:
        c_files = [Path(f).resolve() for f in args.files]
    else:
        c_files = sorted(BENCHMARK_DIR.rglob("*.c"))
        if args.category:
            c_files = [f for f in c_files
                       if f.relative_to(BENCHMARK_DIR).parts[0] == args.category]

    # Only process files that already have an existing DEAD_CODE_ELIM output
    # (i.e., were processed before), unless --files is given explicitly
    if not args.files:
        filtered = []
        for cf in c_files:
            if _llm_output_path(cf).exists():
                filtered.append(cf)
        if filtered:
            print(f"Found {len(filtered)}/{len(c_files)} files with existing "
                  f"{TRANSFORM} outputs to regenerate")
            c_files = filtered
        else:
            print(f"No existing {TRANSFORM} outputs found; processing all "
                  f"{len(c_files)} files")

    print(f"Files: {len(c_files)}  Transform: {TRANSFORM}  "
          f"Workers: {args.workers}  RPM: {args.rpm}\n")

    if args.dry_run:
        for f in c_files:
            print(f"  {f.relative_to(BENCHMARK_DIR)}")
        return

    from llm_transform import (create_client, AREA_TRANSFORM_DESCRIPTIONS,
                                init_rate_limiter)
    import llm_transform
    if args.llm_model:
        llm_transform.MODEL = args.llm_model

    client = create_client()
    init_rate_limiter(rpm=args.rpm, max_concurrent=args.workers)

    description = AREA_TRANSFORM_DESCRIPTIONS[TRANSFORM]
    grand = run_all(client, c_files, description,
                    optimization_goal="area", workers=args.workers)

    # Write a dedicated summary
    out = LLM_OUTPUT_DIR / f"rerun_{TRANSFORM}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(grand, indent=2))
    print(f"\nDone. Summary → {out}")


if __name__ == "__main__":
    main()
