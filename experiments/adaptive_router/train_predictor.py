"""Validate or explicitly train router baselines; default mode never trains."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from path_select.adaptive_predictor import (
    PredictorConfig,
    dry_run,
    family_disjoint_splits,
    load_router_dataset,
    split_coverage_warnings,
    summarize_split,
    train_logistic_baseline,
    train_mlp,
    validate_training_eligibility,
)
from project_paths import PROJECT_ROOT


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive router trainer. Default is a forward-only dry-run with zero "
            "optimizer steps; --execute-training is required to fit models."
        )
    )
    parser.add_argument("--dataset", required=True, help="Router dataset .npz")
    parser.add_argument("--manifest", help="Dataset JSON; defaults beside .npz")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "adaptive_router_artifacts" / "training_runs"),
    )
    parser.add_argument(
        "--model", choices=("logistic", "mlp", "both"), default="both"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--architecture",
        choices=("tiny", "legacy_large", "overfit_control"),
        default=None,
        help="MLP architecture; tiny is default, legacy widths require explicit opt-in",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Family-disjoint split seeds (default: 7 17 27 37 47)",
    )
    parser.add_argument(
        "--execute-training",
        action="store_true",
        help=(
            "Fit and save models. Refused unless manifest training_allowed=true and "
            "minimum sample/family gates pass."
        ),
    )
    return parser.parse_args()


def _aggregate_metrics(
    per_seed: dict[int, dict[str, dict[str, float | None]]]
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Aggregate only validation/test metrics across split seeds."""

    aggregate: dict[str, dict[str, dict[str, float | None]]] = {}
    for partition in ("validation", "test"):
        aggregate[partition] = {}
        metric_names = sorted(
            {
                metric
                for metrics in per_seed.values()
                for metric in metrics.get(partition, {})
            }
        )
        for metric in metric_names:
            values = [
                float(metrics[partition][metric])
                for metrics in per_seed.values()
                if metrics.get(partition, {}).get(metric) is not None
            ]
            if not values:
                aggregate[partition][metric] = {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "seed_count": 0,
                }
                continue
            aggregate[partition][metric] = {
                "mean": sum(values) / len(values),
                "std": (
                    sum((value - sum(values) / len(values)) ** 2 for value in values)
                    / len(values)
                )
                ** 0.5,
                "min": min(values),
                "max": max(values),
                "seed_count": len(values),
            }
    return aggregate


def _paired_model_comparison(
    logistic: dict[int, dict[str, dict[str, float | None]]],
    mlp: dict[int, dict[str, dict[str, float | None]]],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Count head-to-head wins on the same unseen-family partitions."""

    shared_seeds = sorted(set(logistic) & set(mlp))
    comparison: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for partition in ("validation", "test"):
        comparison[partition] = {}
        metric_names = sorted(
            set.intersection(
                *(
                    set(logistic[seed].get(partition, {}))
                    & set(mlp[seed].get(partition, {}))
                    for seed in shared_seeds
                )
            )
        ) if shared_seeds else []
        for metric in metric_names:
            rows = [
                (
                    float(logistic[seed][partition][metric]),
                    float(mlp[seed][partition][metric]),
                )
                for seed in shared_seeds
                if logistic[seed][partition].get(metric) is not None
                and mlp[seed][partition].get(metric) is not None
            ]
            lower_is_better = metric == "brier" or metric.endswith("_brier")
            mlp_wins = 0
            logistic_wins = 0
            ties = 0
            deltas: list[float] = []
            for logistic_value, mlp_value in rows:
                delta = mlp_value - logistic_value
                deltas.append(delta)
                if abs(delta) <= 1e-12:
                    ties += 1
                elif (mlp_value < logistic_value) if lower_is_better else (
                    mlp_value > logistic_value
                ):
                    mlp_wins += 1
                else:
                    logistic_wins += 1
            comparison[partition][metric] = {
                "higher_is_better": not lower_is_better,
                "paired_seed_count": len(rows),
                "mlp_wins": mlp_wins,
                "logistic_wins": logistic_wins,
                "ties": ties,
                "mean_mlp_minus_logistic": (
                    sum(deltas) / len(deltas) if deltas else None
                ),
            }
    return comparison


def main() -> int:
    args = _parse_args()
    dataset = load_router_dataset(args.dataset, args.manifest)
    config_updates: dict[str, Any] = {
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "seed": args.seed,
    }
    if args.architecture is not None:
        config_updates["architecture"] = args.architecture
    if args.split_seeds is not None:
        config_updates["split_seeds"] = tuple(args.split_seeds)
    config = replace(PredictorConfig(), **config_updates)
    if not args.execute_training:
        result = {
            "schema_version": "adaptive_router_training_run_v1",
            "training_executed": False,
            "dataset": str(dataset.npz_path),
            "dataset_role": dataset.manifest.get("dataset_role"),
            "training_allowed": dataset.manifest.get("training_allowed"),
            "dry_run": dry_run(dataset, config),
            "config": asdict(config),
            "message": (
                "Input/model wiring passed. No optimizer was created, no model was "
                "fitted, and no weights were saved."
            ),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    validate_training_eligibility(dataset, config)
    splits = family_disjoint_splits(dataset, config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir).resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema_version": "adaptive_router_training_run_v1",
        "training_executed": True,
        "dataset": str(dataset.npz_path),
        "dataset_manifest": str(dataset.manifest_path),
        "config": asdict(config),
        "split_seeds": list(splits),
        "splits": {},
        "models": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    per_model_metrics: dict[
        str, dict[int, dict[str, dict[str, float | None]]]
    ] = {name: {} for name in ("logistic", "mlp") if args.model in {name, "both"}}

    # This is deliberately a plain loop.  A split's logistic and MLP results
    # share exactly the same indices, and no training jobs are launched in
    # parallel.
    for split_seed, split in splits.items():
        split_config = replace(config, seed=split_seed)
        split_dir = output_dir / f"split_seed_{split_seed}"
        split_dir.mkdir(parents=True, exist_ok=False)
        result["splits"][str(split_seed)] = {
            "seed": split_seed,
            "indices": {
                name: indices.tolist() for name, indices in split.items()
            },
            "families": {
                name: sorted(set(dataset.family_ids[indices].tolist()))
                for name, indices in split.items()
            },
            "summary": summarize_split(dataset, split),
            "coverage_warnings": split_coverage_warnings(
                dataset, split, split_config
            ),
            "models": {},
        }
        if args.model in {"logistic", "both"}:
            logistic_result = train_logistic_baseline(
                dataset, split, split_dir, split_config
            )
            result["splits"][str(split_seed)]["models"]["logistic"] = logistic_result
            per_model_metrics["logistic"][split_seed] = logistic_result["metrics"]
        if args.model in {"mlp", "both"}:
            mlp_result = train_mlp(
                dataset, split, split_dir, split_config, device=args.device
            )
            result["splits"][str(split_seed)]["models"]["mlp"] = mlp_result
            per_model_metrics["mlp"][split_seed] = mlp_result["metrics"]

    for model_name, metrics_by_seed in per_model_metrics.items():
        result["models"][model_name] = {
            "split_count": len(metrics_by_seed),
            "aggregated_metrics": _aggregate_metrics(metrics_by_seed),
        }
    if "logistic" in per_model_metrics and "mlp" in per_model_metrics:
        result["paired_model_comparison"] = _paired_model_comparison(
            per_model_metrics["logistic"], per_model_metrics["mlp"]
        )
    _write_json(output_dir / "training_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
