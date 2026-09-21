"""
CLI entry point for Stage 1: Spec Understanding & Feature Extraction.

Usage:
    python -m spec_analyze                              # all benchmarks
    python -m spec_analyze -f benchmark/uart_tx.v       # single Verilog file
    python -m spec_analyze --spec "A 32-bit UART TX"    # from spec text
    python -m spec_analyze --spec-file desc.txt         # from text spec file
    python -m spec_analyze --spec-file specs.json       # from JSON spec list
    python -m spec_analyze -f uart.v --spec "UART TX"   # mixed input
    python -m spec_analyze -d benchmark_old/            # custom directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spec_analyze.analyzer import Analyzer
from spec_analyze.spec_loader import load_spec_entries

_load_spec_entries = load_spec_entries


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: LLM + regex spec understanding & feature extraction"
    )
    parser.add_argument(
        "-d", "--bench-dir",
        default=str(Path(__file__).resolve().parent.parent / "benchmark"),
        help="Benchmark directory (default: benchmark/)",
    )
    parser.add_argument(
        "-f", "--file", type=str, default=None,
        help="Analyze a single Verilog file",
    )
    parser.add_argument(
        "--spec", type=str, default=None,
        help="Analyze a natural-language spec string",
    )
    parser.add_argument(
        "--spec-file", type=str, default=None,
        help="Analyze a spec from a text file or JSON spec list",
    )
    parser.add_argument(
        "-o", "--output",
        default=str(Path(__file__).resolve().parent.parent / "spec_analysis.json"),
        help="Output JSON path (default: spec_analysis.json)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override LLM model (default: from .env OPENAI_MODEL)",
    )
    parser.add_argument(
        "--env", type=str, default=None,
        help="Path to .env file (default: auto-detect)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay between LLM calls in batch mode (seconds)",
    )
    args = parser.parse_args()

    analyzer = Analyzer(env_path=args.env, model=args.model)
    print(f"Endpoint: {analyzer.endpoint}")
    print(f"Model:    {analyzer.model}")

    results = []

    # ── Single Verilog file (optionally with --spec or --spec-file) ─
    if args.file:
        spec_text = args.spec
        if not spec_text and args.spec_file:
            spec_text = Path(args.spec_file).read_text(errors="replace")
        mode = "mixed" if spec_text else "verilog"
        print(f"\nAnalyzing ({mode}): {args.file}")
        result = analyzer.analyze_file(args.file, spec_text=spec_text)
        results.append(result)
        _print_result(result)

    # ── Spec string only (no Verilog file) ────────────────────────
    elif args.spec:
        print(f"\nAnalyzing spec text...")
        result = analyzer.analyze_spec(args.spec)
        results.append(result)
        _print_result(result)

    # ── Spec file only (no Verilog file) ──────────────────────────
    elif args.spec_file:
        print(f"\nAnalyzing spec file: {args.spec_file}")
        spec_entries = load_spec_entries(args.spec_file)

        for i, entry in enumerate(spec_entries):
            if len(spec_entries) > 1:
                print(f"[{i + 1}/{len(spec_entries)}] {entry['id']}")

            result = analyzer.analyze_spec(
                entry["spec"],
                benchmark_name=entry["id"],
                optimization_target=entry.get("optimization_target"),
            )
            results.append(result)
            _print_result(result)

            if i < len(spec_entries) - 1 and args.delay > 0:
                import time
                time.sleep(args.delay)

    # ── Batch directory ───────────────────────────────────────────
    else:
        print(f"\nBatch analyzing: {args.bench_dir}")
        results_list = analyzer.analyze_dir(args.bench_dir, delay=args.delay)
        results.extend(results_list)

    # ── Write output ──────────────────────────────────────────────
    output_data = [r.to_dict() for r in results]
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(results)} results to {args.output}")

    # ── Summary ───────────────────────────────────────────────────
    if len(results) > 1:
        _print_summary(results)


def _print_result(result):
    """Print a single analysis result."""
    feat = result.llm_features
    if not feat:
        print("  -> LLM ERROR")
        return

    print(f"  Module:      {feat.get('module_name', '?')}")
    print(f"  Purpose:     {feat.get('purpose', '?')}")
    print(f"  Pattern:     {feat.get('architecture_pattern', '?')}")
    print(f"  Complexity:  {feat.get('complexity', '?')}")
    print(f"  Sequential:  {feat.get('is_sequential', '?')}")
    print(f"  FSM:         {feat.get('has_fsm', '?')} "
          f"({feat.get('estimated_fsm_states', 0)} states)")
    print(f"  Subcategory: {feat.get('suggested_subcategory', '?')}")
    print(f"  Opt Target:  {result.optimization_target or '?'}")
    print(f"  Confidence:  {result.overall_confidence}")

    # Show per-feature confidence for spec-only input
    if result.input_type == "spec":
        low_conf = [k for k, v in result.confidence.items() if v == "low"]
        if low_conf:
            print(f"  Low-confidence fields: {', '.join(low_conf)}")


def _print_summary(results):
    """Print batch summary statistics."""
    from collections import Counter

    patterns = Counter(r.llm_features.get("architecture_pattern", "?")
                       for r in results if r.llm_features)
    complexities = Counter(r.llm_features.get("complexity", "?")
                           for r in results if r.llm_features)
    types = Counter(r.input_type for r in results)
    targets = Counter(r.optimization_target or "?" for r in results)

    print(f"\n{'─' * 50}")
    print(f"Summary: {len(results)} benchmarks")
    print(f"  Input types:  {dict(types)}")
    print(f"  Opt targets:  {dict(targets)}")
    print(f"  Patterns:     {dict(patterns)}")
    print(f"  Complexity:   {dict(complexities)}")

    errors = sum(1 for r in results if not r.llm_features)
    if errors:
        print(f"  LLM errors:   {errors}")


if __name__ == "__main__":
    main()
