"""
CLI entry point for the direct spec-to-RTL comparison experiment.

Usage:
    python -m RTL_DIRECT_compare
    python -m RTL_DIRECT_compare --json spec_analysis.json --decisions-json path_select/module2_result.json
    python -m RTL_DIRECT_compare --spec "8-bit counter" --benchmark counter8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from RTL_DIRECT_compare import RTLDirectGenerator


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    root = _project_root()
    project_dir = root / "RTL_DIRECT_compare"

    parser = argparse.ArgumentParser(
        description="Direct spec-to-RTL comparison run for c_first benchmarks",
    )
    parser.add_argument(
        "--json",
        default=str(root / "spec_analysis.json"),
        help="Path to Module 1 output JSON (default: spec_analysis.json)",
    )
    parser.add_argument(
        "--decisions-json",
        default=str(root / "path_select" / "module2_result.json"),
        help="Path to Module 2 decisions JSON (default: path_select/module2_result.json)",
    )
    parser.add_argument(
        "--spec",
        help="Single-spec mode: generate RTL directly from one spec",
    )
    parser.add_argument(
        "--benchmark",
        default="direct_rtl_design",
        help="Benchmark name for single-spec mode",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "generated_rtl"),
        help="Directory for generated Verilog files",
    )
    parser.add_argument(
        "--output-json",
        default=str(project_dir / "rtl_direct_results.json"),
        help="Where to save the result manifest",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max LLM retry attempts after syntax-validation failures",
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Run generation across all models defined in model config",
    )
    parser.add_argument(
        "--model-config",
        default=str(project_dir / "model_config.json"),
        help="Path to model configuration JSON (default: model_config.json)",
    )
    args = parser.parse_args()

    generator = RTLDirectGenerator(
        output_dir=args.output_dir,
        max_llm_retries=args.max_retries,
    )

    if args.multi_model:
        model_configs = RTLDirectGenerator.load_model_config(args.model_config)
        print(f"Multi-model mode: {len(model_configs)} models configured.")

        if args.spec:
            results = []
            for cfg in model_configs:
                mid = cfg["model_id"]
                model_dir = Path(args.output_dir) / mid
                print(f"\n{'='*60}\nModel: {mid}\n{'='*60}")
                result = generator.generate(
                    benchmark=args.benchmark,
                    spec_text=args.spec,
                    features={},
                    model_id=mid,
                    api_key=cfg.get("api_key"),
                    base_url=cfg.get("base_url"),
                    output_dir=model_dir,
                )
                results.append(result)
                status = "OK" if result.success else "FAIL"
                print(f"  {status} {args.benchmark}")
        else:
            features_path = Path(args.json)
            decisions_path = Path(args.decisions_json)
            if not features_path.exists():
                print(f"Error: {features_path} not found", file=sys.stderr)
                sys.exit(1)
            if not decisions_path.exists():
                print(f"Error: {decisions_path} not found", file=sys.stderr)
                sys.exit(1)

            selected = generator.load_selected_features(features_path, decisions_path)
            print(f"Selected {len(selected)} c_first benchmarks for direct RTL generation.")
            results = generator.generate_multi_model(selected, model_configs)

    elif args.spec:
        results = [
            generator.generate(
                benchmark=args.benchmark,
                spec_text=args.spec,
                features={},
            )
        ]
    else:
        features_path = Path(args.json)
        decisions_path = Path(args.decisions_json)
        if not features_path.exists():
            print(f"Error: {features_path} not found", file=sys.stderr)
            sys.exit(1)
        if not decisions_path.exists():
            print(f"Error: {decisions_path} not found", file=sys.stderr)
            sys.exit(1)

        selected = generator.load_selected_features(features_path, decisions_path)
        print(f"Selected {len(selected)} c_first benchmarks for direct RTL generation.")
        results = generator.generate_batch(selected)

    for result in results:
        status = "OK" if result.success else "FAIL"
        model_tag = f"model={result.model:<24} " if result.model else ""
        print(
            f"{status} {result.benchmark:<35} "
            f"{model_tag}"
            f"module={result.module_name:<24} "
            f"syntax={result.syntax_ok} "
            f"tokens={result.token_usage.total_tokens}"
        )
        if not result.success and result.error:
            print(f"  Error: {result.error}")

    output_path = Path(args.output_json)
    output_path.write_text(
        json.dumps([item.to_dict() for item in results], indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved results to {output_path}")

    failures = sum(1 for item in results if not item.success)
    if failures:
        print(f"{failures}/{len(results)} runs failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
