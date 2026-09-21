"""
__main__.py — CLI entry point for path_select (Module 2).

Usage examples:
    # Batch: read spec_analysis.json, print decisions for all benchmarks
    python -m path_select

    # Batch with custom JSON
    python -m path_select --json results/my_analysis.json

    # Single Verilog file (runs Stage 1 first, then Stage 2)
    python -m path_select -f benchmark/uart_tx.v

    # Tier-1 rules only, no LLM API calls
    python -m path_select --no-llm

    # JSON output (one object per line, stderr-clean for piping)
    python -m path_select --output-json

    # Save decisions to file
    python -m path_select --save path_selection.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Locate the project root (parent of path_select/)."""
    return Path(__file__).resolve().parent.parent


def _print_table(decisions) -> None:
    """Print decisions as a formatted ASCII table."""
    header = f"{'Benchmark':<35} {'Target':<8} {'Path':<12} {'Tier':<6} {'Conf':<8} Rule"
    print(header)
    print("-" * len(header))
    for d in decisions:
        print(
            f"{d.benchmark:<35} {d.optimization_target:<8} {d.path:<12} {d.tier:<6} {d.confidence:<8} "
            f"{d.rule_fired}"
        )
    print()

    # Path summary
    rtl = sum(1 for d in decisions if d.path == "rtl_direct")
    cf  = sum(1 for d in decisions if d.path == "c_first")
    unc = sum(1 for d in decisions if d.path == "uncertain")
    print(f"Summary: {len(decisions)} benchmarks — "
          f"rtl_direct: {rtl}, c_first: {cf}, uncertain: {unc}")

    # Latent predictor summary. Scored counts every spec that got a p, decided
    # only the ones where p fell outside the hand-off band; the difference is the
    # LLM's remaining workload.
    scored = [d for d in decisions if d.p_c_first is not None]
    if scored:
        tier3 = [d for d in scored if d.tier == 3]
        probabilities = ", ".join(
            f"{d.benchmark} p={d.p_c_first:.3f}" for d in scored
        )
        print(
            f"Router:  {len(scored)} scored, {len(tier3)} decided without an "
            f"LLM call — {probabilities}"
        )
    else:
        statuses = {
            d.predictor_status for d in decisions if d.predictor_status
        }
        if statuses:
            print(f"Router:  no probabilities ({', '.join(sorted(statuses))})")

    # Token cost summary
    tier2 = [d for d in decisions if d.tier == 2]
    total_tok = sum(d.token_usage.total_tokens for d in decisions)
    total_ret = sum(d.token_usage.retries for d in decisions)

    if not tier2:
        print(f"Tokens:  0  ({len(decisions)} Tier-1 rule-based, no LLM calls)")
    else:
        prompt_tok = sum(d.token_usage.prompt_tokens for d in decisions)
        compl_tok  = sum(d.token_usage.completion_tokens for d in decisions)
        retry_str  = f"  retries: {total_ret}" if total_ret else ""
        print(f"Tokens:  {total_tok:,} total — "
              f"prompt: {prompt_tok:,}  completion: {compl_tok:,}{retry_str}")
        per = "  Tier-2: " + ", ".join(
            f"{d.benchmark} ({d.token_usage.total_tokens:,} tok)" for d in tier2
        )
        print(per)


def main() -> None:
    root = _find_project_root()

    parser = argparse.ArgumentParser(
        prog="python -m path_select",
        description="Module 2: Path selection for RTL optimization pipeline",
    )
    parser.add_argument(
        "--json", "-j",
        default=str(root / "spec_analysis.json"),
        help="Path to Stage 1 batch results JSON (default: spec_analysis.json)",
    )
    parser.add_argument(
        "-f", "--file",
        help="Single Verilog file — runs Stage 1 then Module 2",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Tier-1 rules only; no LLM API calls",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print JSON output (one object per line) instead of table",
    )
    parser.add_argument(
        "--save", "-s",
        help="Save decisions to this JSON file",
    )
    parser.add_argument(
        "--memory-path", "-m",
        default=str(root / "path_decisions_log.json"),
        help="Module 8 staging file — decisions are appended here "
             "(default: path_decisions_log.json in project root)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable writing to the Module 8 staging file",
    )
    parser.add_argument(
        "--router-checkpoint",
        default="",
        help="Trained router_mlp.pt for the latent predictor stage. Defaults to "
             "$ADAPTIVE_ROUTER_CHECKPOINT; without either the stage is skipped "
             "and only the rule and LLM tiers run",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(root))

    try:
        from path_select import PathSelector
    except ImportError as e:
        print(f"ERROR: Could not import path_select: {e}", file=sys.stderr)
        sys.exit(1)

    memory_path = None if args.no_memory else args.memory_path
    selector = PathSelector(
        memory_path=memory_path,
        router_checkpoint=args.router_checkpoint or None,
    )
    use_llm = not args.no_llm

    # ── Single-file mode ──────────────────────────────────────────────
    if args.file:
        try:
            from spec_analyze import Analyzer
        except ImportError:
            print("ERROR: spec_analyze not found — cannot run Stage 1", file=sys.stderr)
            sys.exit(1)

        verilog_path = Path(args.file)
        if not verilog_path.exists():
            print(f"ERROR: Verilog file not found: {verilog_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Running Stage 1 on {verilog_path.name}...", flush=True, file=sys.stderr)
        analyzer = Analyzer()
        feature_result = analyzer.analyze_file(str(verilog_path))
        print("Stage 1 complete. Running Module 2...", flush=True, file=sys.stderr)
        decisions = [selector.select(feature_result, use_llm=use_llm)]

    # ── Batch-from-JSON mode ──────────────────────────────────────────
    else:
        json_path = Path(args.json)
        if not json_path.exists():
            print(f"ERROR: JSON file not found: {json_path}", file=sys.stderr)
            print("Run Stage 1 first: python -m spec_analyze", file=sys.stderr)
            sys.exit(1)

        print(f"Loading Stage 1 results from {json_path.name}...", flush=True, file=sys.stderr)
        decisions = selector.select_from_json(json_path, use_llm=use_llm)

    # ── Output ────────────────────────────────────────────────────────
    if args.output_json:
        for d in decisions:
            print(json.dumps(d.to_dict()))
    else:
        _print_table(decisions)

    # ── Save to file ──────────────────────────────────────────────────
    if args.save:
        save_path = Path(args.save)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in decisions], f, indent=2)
        print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
