#!/usr/bin/env python3
"""
Collect LLM-generated .v files from LLM_V2V_TIMING and LLM_V2V_AREA,
organised by the subcategory/benchmark ordering from LLM_GR_RTL_TIMING
(and LLM_GR_RTL_AREA respectively).

Output structure:
  <collect_dir>/
    <metric>/
      <subcategory>/
        <benchmark>/
          <benchmark>_<TRANSFORM>.v   # renamed to match reference .c convention
          <benchmark>_original.v      # copy of the reference .v

Usage:
  python collect_v2v_results.py                         # default: ./LLM_V2V_COLLECTED
  python collect_v2v_results.py -o /tmp/collected
  python collect_v2v_results.py --metric TIMING         # only TIMING
  python collect_v2v_results.py --dry-run               # show what would be copied
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT


def _find_baseline_stem(bench_dir: Path) -> str | None:
    """Return the baseline .c stem (the one without _ALLCAPS suffix)."""
    suffix_re = re.compile(r"_[A-Z][A-Z_]+$")
    for f in sorted(bench_dir.glob("*.c")):
        if not suffix_re.search(f.stem):
            return f.stem
    return None


def collect(
    metric: str,
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Collect generated .v files for one metric.
    Returns stats dict {copied, missing, total}.
    """
    ref_dir = ROOT / f"LLM_GR_RTL_{metric}"
    v2v_dir = ROOT / f"LLM_V2V_{metric}"

    if not ref_dir.is_dir():
        print(f"  Reference dir not found: {ref_dir}", file=sys.stderr)
        return {"copied": 0, "missing": 0, "total": 0}
    if not v2v_dir.is_dir():
        print(f"  Output dir not found: {v2v_dir}", file=sys.stderr)
        return {"copied": 0, "missing": 0, "total": 0}

    copied = 0
    missing = 0
    total = 0

    # Walk subcategories in sorted order (arithmetic, bit_reorganization, ...)
    for sub in sorted(ref_dir.iterdir()):
        if not sub.is_dir():
            continue
        sub_name = sub.name

        # Walk benchmarks in sorted order
        for bench in sorted(sub.iterdir()):
            if not bench.is_dir():
                continue
            bench_name = bench.name

            # Determine baseline stem for output naming
            base_stem = _find_baseline_stem(bench)
            if base_stem is None:
                continue

            # Find reference .v
            ref_v_files = (
                list(bench.glob("*_original.v"))
                + list(bench.glob("*_combinational.v"))
                + list(bench.glob("sampled_*.v"))
            )
            if not ref_v_files:
                continue

            # Find transforms from reference dir
            suffix_re = re.compile(r"_[A-Z][A-Z_]+$")
            transforms = []
            for f in sorted(bench.glob("*.c")):
                if f.stem.startswith(base_stem + "_") and suffix_re.search(f.stem):
                    tname = f.stem[len(base_stem) + 1:]
                    transforms.append(tname)

            if not transforms:
                continue

            if not dry_run:
                print(f"  [{sub_name}] {bench_name}: {len(transforms)} transforms")

            # Collect generated .v for each transform
            for tname in transforms:
                total += 1
                # Source: LLM_V2V_<metric>/<subcategory>/<bench>/<transform>/<transform>.v
                src = v2v_dir / sub_name / bench_name / tname / f"{tname}.v"

                # Destination: <output>/<metric>/<sub>/<bench>/<base_stem>_<transform>.v
                dst_dir = output_dir / metric / sub_name / bench_name
                dst = dst_dir / f"{base_stem}_{tname}.v"

                if src.is_file():
                    copied += 1
                    if dry_run:
                        print(f"  [copy] {src.relative_to(ROOT)}")
                        print(f"      -> {dst.relative_to(output_dir)}")
                    else:
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                else:
                    missing += 1
                    if dry_run:
                        print(f"  [MISS] {bench_name}/{tname}")

                if not dry_run and total % 100 == 0:
                    print(f"    progress: {copied}/{total} copied, {missing} missing")

            # Also copy the reference .v into the collected dir
            if not dry_run and copied > 0:
                dst_dir = output_dir / metric / sub_name / bench_name
                if dst_dir.is_dir():
                    ref_v = ref_v_files[0]
                    ref_dst = dst_dir / ref_v.name
                    if not ref_dst.exists():
                        shutil.copy2(ref_v, ref_dst)
                    # Also copy the baseline .c
                    baseline_c = bench / f"{base_stem}.c"
                    if baseline_c.is_file():
                        c_dst = dst_dir / baseline_c.name
                        if not c_dst.exists():
                            shutil.copy2(baseline_c, c_dst)

    return {"copied": copied, "missing": missing, "total": total}


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect LLM-generated .v files")
    ap.add_argument("-o", "--output", type=Path,
                    default=ROOT / "LLM_V2V_COLLECTED",
                    help="Output directory (default: %(default)s)")
    ap.add_argument("--metric", choices=["TIMING", "AREA", "ALL"], default="ALL",
                    help="Which metric to collect (default: ALL)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be copied without copying")
    args = ap.parse_args()

    metrics = ["TIMING", "AREA"] if args.metric == "ALL" else [args.metric]

    print(f"Collecting to: {args.output}")
    if args.dry_run:
        print("[dry-run mode]")
    print()

    grand_total = 0
    grand_copied = 0
    grand_missing = 0

    for metric in metrics:
        print(f"--- {metric} ---")
        stats = collect(metric, args.output, args.dry_run)
        print(f"  {stats['copied']}/{stats['total']} collected, "
              f"{stats['missing']} missing")
        grand_total += stats["total"]
        grand_copied += stats["copied"]
        grand_missing += stats["missing"]
        print()

    print(f"Total: {grand_copied}/{grand_total} collected, {grand_missing} missing")
    if not args.dry_run and grand_copied > 0:
        print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
