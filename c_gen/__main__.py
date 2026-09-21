"""
CLI entry point for Module 3: C generation.

Usage:
    python -m c_gen                              # batch from spec_analysis.json
    python -m c_gen --decisions-json module2.json
    python -m c_gen --spec "8-bit counter"       # single spec
    python -m c_gen --verilog path/to/file.v     # single verilog
    python -m c_gen --json custom.json           # custom input JSON
    python -m c_gen --output-json results.json   # save results to JSON
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from c_gen import CGenerator


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Module 3: Generate HLS-compatible C code from specs or Verilog"
    )

    # Input modes
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--spec",
        help="Natural-language specification (single-spec mode)",
    )
    input_group.add_argument(
        "--verilog",
        help="Path to Verilog file (single-verilog mode)",
    )
    input_group.add_argument(
        "--json",
        default="spec_analysis.json",
        help="Path to spec_analysis.json (batch mode, default)",
    )

    # Options
    parser.add_argument(
        "--benchmark",
        default="design",
        help="Benchmark name (for single-spec/verilog mode)",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/c_gen_output",
        help="Output directory for generated .c files",
    )
    parser.add_argument(
        "--decisions-json",
        help="Path to Module 2 decisions JSON; when set, only c_first benchmarks are generated",
    )
    parser.add_argument(
        "--output-json",
        help="Save CGenResult list to JSON file",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max LLM retry attempts with clang errors",
    )

    args = parser.parse_args()

    generator = CGenerator(
        output_dir=args.output_dir,
        max_llm_retries=args.max_retries,
    )

    # Single-spec mode
    if args.spec:
        result = generator.generate(
            benchmark=args.benchmark,
            input_type="spec",
            spec_text=args.spec,
            features={},
        )
        results = [result]

    # Single-verilog mode
    elif args.verilog:
        result = generator.generate(
            benchmark=args.benchmark,
            input_type="verilog",
            verilog_path=args.verilog,
        )
        results = [result]

    # Batch mode from JSON
    else:
        json_path = Path(args.json)
        if not json_path.exists():
            print(f"Error: {json_path} not found", file=sys.stderr)
            sys.exit(1)

        if args.decisions_json:
            decisions_path = Path(args.decisions_json)
            if not decisions_path.exists():
                print(f"Error: {decisions_path} not found", file=sys.stderr)
                sys.exit(1)

            print(
                f"Loading Stage 1 features from {json_path} and Module 2 decisions from {decisions_path}..."
            )
            results = generator.generate_from_pipeline_json(
                features_json=json_path,
                decisions_json=decisions_path,
            )
        else:
            print(f"Loading input from {json_path}...")
            results = generator.generate_from_json(json_path)

    # Output
    for r in results:
        status = "✓" if r.success else "✗"
        print(f"{status} {r.benchmark}: {r.method} → {r.c_path}")
        if not r.success:
            print(f"  Error: {r.error}")
        print(f"  Tokens: {r.token_usage.total_tokens}")

    # Save to JSON if requested
    if args.output_json:
        output_data = [r.to_dict() for r in results]
        Path(args.output_json).write_text(json.dumps(output_data, indent=2))
        print(f"\nSaved results to {args.output_json}")

    # Exit code
    failed = sum(1 for r in results if not r.success)
    if failed > 0:
        print(f"\n{failed}/{len(results)} benchmarks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
