"""Build a *labeled* adaptive-router dataset from executed dual-path runs.

This is the missing half of revise plan §5. ``build_pilot_inputs.py`` produces an
unlabeled smoke set (``labels`` all ``-1``); this script reads run trees where
both routes were actually forced and executed, applies
:mod:`path_select.route_score`, and writes an npz whose labels are real.

Input layout (as produced by ``experiments/path_oracle/run_dual_path.py`` and
``run_phase6_pilot.py``)::

    <run-root>/pairs/<pair_id>/pair_manifest.json        <- runs.{c_first,rtl_direct}
    <run-root>/pairs/<pair_id>/frozen_feature_result.json <- Module 1 output, shared

Guarantees, all of which the paper depends on:

* ``unlabelable`` pairs (both routes disqualified) are dropped BEFORE the npz is
  written -- ``path_select.adaptive_predictor`` rejects anything outside
  ``{0, 1}``, and that check is correct, so filtering must happen here.
* ``training_allowed`` is only set when every surviving label came from a
  JasperGold-verified run and the dataset clears the predictor's own eligibility
  thresholds. An unverified batch still produces a readable manifest, marked
  ``training_allowed: false`` with the reason spelled out.
* ``label_basis`` is carried per record and aggregated in the manifest, so the
  paper can report how many labels came from a genuine two-sided PPA comparison
  versus a one-sided gate disqualification.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from path_select.feature_vectorizer import FeatureVectorizer
from path_select.latent_encoder import (
    LatentEncoder,
    canonicalize_spec_text,
    sha256_text,
)
from path_select.route_score import SCORE_SCHEMA_VERSION, label_pair
from project_paths import PROJECT_ROOT

ROUTES = ("c_first", "rtl_direct")
JASPERGOLD_MODE = "jaspergold"
OBJECTIVE_FILTERS = ("ALL", "AREA", "TIMING")


def _normalize_objective_filter(value: Any) -> str:
    """Normalize and validate the dataset-level objective filter."""
    objective = str(value or "ALL").strip().upper()
    if objective not in OBJECTIVE_FILTERS:
        raise ValueError(
            f"Unsupported objective filter: {value!r}; "
            f"expected one of {', '.join(OBJECTIVE_FILTERS)}"
        )
    return objective


def _normalize_run_roots(
    run_root: Path | str | Iterable[Path | str] | None = None,
    run_roots: Iterable[Path | str] | None = None,
) -> tuple[Path, ...]:
    """Return unique, absolute run roots while preserving caller order.

    ``run_root`` is retained for callers of the original single-root API.  A
    repeated path is harmless and is removed before collection so that a CLI
    typo cannot duplicate every pair in the output.
    """
    roots: list[Path] = []
    if run_root is not None:
        if isinstance(run_root, (Path, str)):
            roots.append(Path(run_root))
        else:
            roots.extend(Path(root) for root in run_root)
    if run_roots is not None:
        roots.extend(Path(root) for root in run_roots)
    if not roots:
        raise ValueError("At least one --run-root is required")

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return tuple(unique)


def _predictor_module():
    """Import ``adaptive_predictor`` lazily.

    It pulls in torch/sklearn/joblib. ``--labels-only`` needs none of that, and
    being able to inspect the label distribution on a machine without the ML
    stack is the whole point of that mode.
    """
    from path_select import adaptive_predictor

    return adaptive_predictor



def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spec_text_for(pair_root: Path, feature_payload: dict[str, Any]) -> str:
    """Recover the spec text the pair was run on.

    The frozen FeatureResult carries it; fall back to a sibling spec file so that
    older run trees still work.
    """
    text = str(feature_payload.get("spec_text") or "").strip()
    if text:
        return text
    for candidate in ("spec.txt", "spec.md", "input_spec.txt"):
        path = pair_root / candidate
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"No spec text for {pair_root.name}: frozen_feature_result.json has no "
        "spec_text and no spec.txt sits next to it"
    )


def _normalize_verification_mode(value: Any) -> str:
    """Return a comparable route-level verification mode.

    Missing route metadata is treated as ``none``.  A pair-level mode is not
    used as a substitute: the dataset must prove that *both* route records were
    verified independently.
    """
    return str(value or "none").strip().lower() or "none"


def _route_verification_modes(runs: Mapping[str, Any]) -> dict[str, str]:
    """Read verification metadata separately for the two forced routes."""
    modes: dict[str, str] = {}
    for route in ROUTES:
        record = runs.get(route)
        value = record.get("verification_mode") if isinstance(record, Mapping) else None
        modes[route] = _normalize_verification_mode(value)
    return modes


def _verification_block_reason(modes: Mapping[str, str]) -> str:
    """Explain why route-level metadata cannot support a training label."""
    invalid = [
        f"{route}={modes.get(route, 'none')}"
        for route in ROUTES
        if _normalize_verification_mode(modes.get(route)) != JASPERGOLD_MODE
    ]
    if not invalid:
        return ""
    return (
        "training labels require verification_mode=jaspergold for both routes; "
        "unverified route(s): " + ", ".join(invalid)
    )


def _item_drop_reason(item: Mapping[str, Any]) -> str:
    """Combine label and route-verification reasons for manifests/reports."""
    reasons: list[str] = []
    skip_reason = str(item.get("skip_reason") or "")
    if skip_reason:
        reasons.append(skip_reason)
    verdict = item.get("verdict")
    if verdict is not None and not verdict.trainable:
        reasons.append(str(verdict.label_reason))
    verification_reason = str(item.get("verification_block_reason") or "")
    if verification_reason:
        reasons.append(verification_reason)
    return "; ".join(reasons) or "not eligible for training"


def _source_metadata(
    *,
    pair: Mapping[str, Any],
    runs: Mapping[str, Any],
    run_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Copy provenance fields from one executed pair manifest.

    Pair manifests from the current phase-6 flow keep model information on
    each route, while seeds and tree/config hashes live at pair level.  The
    flattened fields make each training row auditable without requiring a
    consumer to reopen the original run tree.
    """
    route_models = {
        route: str(
            (runs.get(route) or {}).get("model")
            or pair.get("model")
            or ""
        )
        for route in ROUTES
    }
    model_values = sorted({model for model in route_models.values() if model})
    route_mcts_seeds = {
        route: (runs.get(route) or {}).get("mcts_seed") for route in ROUTES
    }
    route_llm_seeds = {
        route: (runs.get(route) or {}).get("llm_seed") for route in ROUTES
    }
    memory_baseline = pair.get("memory_baseline")
    if not isinstance(memory_baseline, Mapping):
        memory_baseline = {}
    return {
        "source_run_root": str(run_root),
        "source_pair_manifest": str(manifest_path),
        "spec_sha256": str(pair.get("spec_sha256") or ""),
        "code_tree_sha256": str(pair.get("code_tree_sha256") or ""),
        "run_config_sha256": str(pair.get("run_config_sha256") or ""),
        "prompt_tree_sha256": str(pair.get("prompt_tree_sha256") or ""),
        "model": model_values[0] if len(model_values) == 1 else "mixed",
        "models_by_route": route_models,
        "mcts_seed": pair.get("mcts_seed"),
        "llm_seed": pair.get("llm_seed"),
        "seeds_by_route": {
            "mcts": route_mcts_seeds,
            "llm": route_llm_seeds,
        },
        "memory_baseline": {
            key: memory_baseline.get(key)
            for key in (
                "database_sha256",
                "json_sha256",
                "source_database",
                "source_json",
            )
            if memory_baseline.get(key) is not None
        },
        "pair_schema_version": str(pair.get("schema_version") or ""),
    }


def _spec_hash_for_pair(
    *,
    pair_root: Path,
    pair: Mapping[str, Any],
) -> str:
    """Read the recorded spec hash, with a canonical-text fallback.

    Current phase-6 manifests always carry ``spec_sha256``.  The fallback is
    for older run trees and test fixtures that only froze the feature payload;
    it still lets the multi-root deduplication catch identical specification
    text instead of treating those rows as independent.
    """
    recorded = str(pair.get("spec_sha256") or "")
    if recorded:
        return recorded
    feature_path = pair_root / "frozen_feature_result.json"
    if not feature_path.is_file():
        return ""
    try:
        payload = json.loads(feature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    text = str(payload.get("spec_text") or "")
    if not text:
        return ""
    return sha256_text(canonicalize_spec_text(text))


def _duplicate_skip_item(
    *,
    pair_id: str,
    pair_root: Path,
    source: Mapping[str, Any],
    reasons: Sequence[str],
) -> dict[str, Any]:
    """Make a manifest-visible record for a deduplicated pair."""
    return {
        "pair_id": pair_id,
        "pair_root": pair_root,
        "skipped": True,
        "skip_reason": "; ".join(reasons),
        **dict(source),
    }


def collect_labeled_pairs(
    run_root: Path | str | Iterable[Path | str] | None = None,
    *,
    run_roots: Iterable[Path | str] | None = None,
    objective: str = "ALL",
) -> list[dict[str, Any]]:
    """Label pairs from one or more run roots without touching any model.

    Pair IDs and recorded spec hashes are both treated as dataset identities.
    If either identity has already appeared, the later row is retained only as
    a skipped manifest item.  This protects combined datasets from duplicate
    repeats or the same design being evaluated under multiple run roots.
    """
    roots = _normalize_run_roots(run_root, run_roots)
    objective_filter = _normalize_objective_filter(objective)
    collected: list[dict[str, Any]] = []
    seen_pair_ids: dict[str, str] = {}
    seen_spec_hashes: dict[str, str] = {}

    for source_root in roots:
        pair_dirs = sorted(p for p in (source_root / "pairs").glob("*") if p.is_dir())
        if not pair_dirs:
            raise FileNotFoundError(f"No pairs/ directory under {source_root}")
        for pair_root in pair_dirs:
            manifest_path = pair_root / "pair_manifest.json"
            if not manifest_path.is_file():
                continue
            pair = json.loads(manifest_path.read_text(encoding="utf-8"))
            runs = pair.get("runs") or {}
            pair_id = str(pair.get("pair_id") or pair_root.name)
            pair_objective = str(pair.get("objective") or "AREA").strip().upper()
            if pair_objective not in {"AREA", "TIMING"}:
                raise ValueError(
                    f"{pair_id}: unsupported pair objective {pair_objective!r}"
                )
            if objective_filter != "ALL" and pair_objective != objective_filter:
                continue

            source = _source_metadata(
                pair=pair,
                runs=runs,
                run_root=source_root,
                manifest_path=manifest_path,
            )
            source["spec_sha256"] = _spec_hash_for_pair(
                pair_root=pair_root,
                pair=pair,
            )
            missing = [route for route in ROUTES if not runs.get(route)]
            if missing:
                collected.append(
                    {
                        "pair_id": pair_id,
                        "skipped": True,
                        "skip_reason": (
                            "pair has no run record for: "
                            + ", ".join(missing)
                        ),
                        **source,
                    }
                )
                continue

            spec_sha256 = str(source.get("spec_sha256") or "")
            duplicate_reasons: list[str] = []
            if pair_id in seen_pair_ids:
                duplicate_reasons.append(
                    f"duplicate pair_id already in {seen_pair_ids[pair_id]}"
                )
            if spec_sha256 and spec_sha256 in seen_spec_hashes:
                duplicate_reasons.append(
                    "duplicate spec_sha256 already in "
                    + seen_spec_hashes[spec_sha256]
                )
            if duplicate_reasons:
                collected.append(
                    _duplicate_skip_item(
                        pair_id=pair_id,
                        pair_root=pair_root,
                        source=source,
                        reasons=duplicate_reasons,
                    )
                )
                continue
            seen_pair_ids[pair_id] = str(source_root)
            if spec_sha256:
                seen_spec_hashes[spec_sha256] = str(source_root)

            verification_modes = _route_verification_modes(runs)
            verdict = label_pair(
                runs["c_first"], runs["rtl_direct"], objective=pair_objective
            )
            verification_block_reason = _verification_block_reason(verification_modes)
            collected.append(
                {
                    "pair_id": pair_id,
                    "pair_root": pair_root,
                    "benchmark": str(pair.get("benchmark") or ""),
                    "objective": pair_objective,
                    "family_id": str(
                        pair.get("family_id") or pair.get("benchmark") or pair_root.name
                    ),
                    # Keep the old scalar field for consumers that only display a
                    # single mode, but make the route-specific map authoritative.
                    "verification_mode": (
                        verification_modes["c_first"]
                        if verification_modes["c_first"]
                        == verification_modes["rtl_direct"]
                        else "mixed"
                    ),
                    "verification_modes": verification_modes,
                    "verification_eligible": not verification_block_reason,
                    "verification_block_reason": verification_block_reason,
                    "verdict": verdict,
                    "skipped": False,
                    **source,
                }
            )
    return collected


def summarize_labels(collected: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the label distribution the paper has to report."""
    usable = [item for item in collected if not item["skipped"]]
    deduplicated = [
        item
        for item in collected
        if item.get("skipped")
        and str(item.get("skip_reason") or "").startswith("duplicate ")
    ]
    verdicts = [item["verdict"] for item in usable]
    trainable_items = [
        item
        for item in usable
        if item["verdict"].trainable and item.get("verification_eligible", False)
    ]
    trainable = [item["verdict"] for item in trainable_items]
    basis = collections.Counter(v.label_basis for v in verdicts)
    two_sided = [v for v in trainable if v.label_basis == "both_routes_scored"]
    verification_ineligible = [
        item
        for item in usable
        if item["verdict"].trainable and not item.get("verification_eligible", False)
    ]
    return {
        "pair_count": len(collected),
        "skipped_pairs": len(collected) - len(usable),
        "deduplicated_pairs": len(deduplicated),
        "labeled_pairs": len(usable),
        "trainable_pairs": len(trainable),
        "dropped_unlabelable": sum(1 for verdict in verdicts if not verdict.trainable),
        "dropped_verification_ineligible": len(verification_ineligible),
        "label_counts": {
            "c_first_y1": sum(1 for v in trainable if v.label == 1),
            "rtl_direct_y0": sum(1 for v in trainable if v.label == 0),
        },
        "label_basis_counts": dict(basis),
        # The two numbers a reviewer will ask for.
        "two_sided_ppa_comparisons": len(two_sided),
        "exact_metric_ties_labeled_y0": sum(1 for v in two_sided if v.exact_equal),
        "one_sided_gate_disqualifications": sum(
            1 for v in trainable if v.label_basis.endswith("_only")
        ),
    }


def _source_version_sets(collected: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Collect the distinct provenance versions observed in input pairs."""
    fields = (
        "code_tree_sha256",
        "run_config_sha256",
        "prompt_tree_sha256",
        "model",
        "mcts_seed",
        "llm_seed",
        "pair_schema_version",
    )
    versions: dict[str, list[str]] = {}
    for field in fields:
        values = {
            str(item.get(field))
            for item in collected
            if item.get(field) not in (None, "")
        }
        versions[field] = sorted(values)

    memory_fields = (
        "memory_database_sha256",
        "memory_json_sha256",
    )
    for field in memory_fields:
        memory_key = field.removeprefix("memory_")
        values = {
            str((item.get("memory_baseline") or {}).get(memory_key))
            for item in collected
            if (item.get("memory_baseline") or {}).get(memory_key) not in (None, "")
        }
        versions[field] = sorted(values)
    return versions


def _record_source_fields(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return provenance fields shared by labels-only and npz manifests."""
    return {
        "source_run_root": item.get("source_run_root", ""),
        "source_pair_manifest": item.get("source_pair_manifest", ""),
        "spec_sha256": item.get("spec_sha256", ""),
        "code_tree_sha256": item.get("code_tree_sha256", ""),
        "run_config_sha256": item.get("run_config_sha256", ""),
        "prompt_tree_sha256": item.get("prompt_tree_sha256", ""),
        "model": item.get("model", ""),
        "models_by_route": dict(item.get("models_by_route") or {}),
        "mcts_seed": item.get("mcts_seed"),
        "llm_seed": item.get("llm_seed"),
        "seeds_by_route": item.get("seeds_by_route") or {},
        "memory_baseline": item.get("memory_baseline") or {},
    }


def _eligibility_block(
    *,
    labels: np.ndarray,
    family_ids: np.ndarray,
    verification_modes: Mapping[str, set[str]] | set[str],
    config,
) -> dict[str, Any]:
    """Evaluate the predictor's own training gate and explain every failure.

    Mirrors ``adaptive_predictor.validate_training_eligibility`` instead of
    calling it, because that function raises on the first problem and the useful
    artifact here is the full list of what is still missing.
    """
    blockers: list[str] = []
    if isinstance(verification_modes, Mapping):
        modes_by_route = {
            route: {
                _normalize_verification_mode(mode)
                for mode in (verification_modes.get(route) or set())
            }
            for route in ROUTES
        }
    else:
        # Backward-compatible handling for callers that only have a union of
        # modes.  New dataset manifests always pass the route-specific map.
        union = {_normalize_verification_mode(mode) for mode in verification_modes}
        modes_by_route = {route: set(union) for route in ROUTES}

    for route in ROUTES:
        route_modes = modes_by_route[route]
        unverified = sorted(route_modes - {JASPERGOLD_MODE})
        if unverified or not route_modes:
            observed = unverified or ["missing"]
            blockers.append(
                f"{route} labels came from runs without JasperGold equivalence "
                f"(verification_mode={observed}); both routes must use "
                "verification_mode=jaspergold"
            )
    if labels.size < config.min_training_samples:
        blockers.append(
            f"{labels.size} labeled samples < required {config.min_training_samples}"
        )
    families = np.unique(family_ids)
    if families.size < config.min_training_families:
        blockers.append(
            f"{families.size} design families < required "
            f"{config.min_training_families}"
        )
    present = set(labels.tolist())
    if present != {0, 1}:
        blockers.append(
            f"labels present = {sorted(present)}; both 0=rtl_direct and "
            "1=c_first are required"
        )
    per_label_families = {}
    for label, name in ((0, "rtl_direct"), (1, "c_first")):
        count = int(np.unique(family_ids[labels == label]).size)
        per_label_families[name] = count
        if count < 3:
            blockers.append(
                f"label {name} occurs in only {count} families; "
                "family-disjoint splitting needs at least 3"
            )
    return {
        "training_allowed": not blockers,
        "training_block_reasons": blockers,
        "thresholds": {
            "min_training_samples": config.min_training_samples,
            "min_training_families": config.min_training_families,
            "min_families_per_label": 3,
        },
        "observed": {
            "labeled_samples": int(labels.size),
            "families": int(families.size),
            "families_per_label": per_label_families,
            "verification_modes_by_route": {
                route: sorted(modes_by_route[route]) for route in ROUTES
            },
        },
    }


def build_dataset(
    *,
    run_root: Path | str | Iterable[Path | str] | None = None,
    run_roots: Iterable[Path | str] | None = None,
    output_root: Path,
    model_alias: str,
    device: str,
    latent_dim: int = 256,
    labels_only: bool = False,
    objective: str = "ALL",
) -> tuple[Path | None, Path]:
    """Label pairs under one or more run roots and write dataset artifacts.

    With ``labels_only=True`` no model is loaded and no npz is written: only the
    label report. Use it to inspect the distribution before spending GPU time.

    ``run_root`` remains a compatibility alias for the original single-root
    API.  New callers should pass ``run_roots`` so that several independent
    batches can be combined without hand-editing or concatenating npz files.
    """
    roots = _normalize_run_roots(run_root, run_roots)
    output_root = output_root.resolve()
    objective_filter = _normalize_objective_filter(objective)
    collected = collect_labeled_pairs(
        run_roots=roots,
        objective=objective_filter,
    )
    label_summary = summarize_labels(collected)
    if len(roots) == 1 and objective_filter == "ALL":
        source_stem = roots[0].name
    elif len(roots) == 1:
        source_stem = f"{roots[0].name}_{objective_filter.lower()}"
    elif objective_filter == "ALL":
        source_stem = "combined"
    else:
        source_stem = f"combined_{objective_filter.lower()}"
    stem = f"router_dataset_{source_stem}_qwen3_{model_alias}"

    trainable = [
        item
        for item in collected
        if (
            not item["skipped"]
            and item["verdict"].trainable
            and item.get("verification_eligible", False)
        )
    ]
    dropped = [
        item
        for item in collected
        if (
            item["skipped"]
            or (
                not item["verdict"].trainable
                or not item.get("verification_eligible", False)
            )
        )
    ]

    # Observe every route in the input batch, including pairs that will be
    # dropped.  This keeps the final training gate honest: an unverified pair
    # cannot disappear from the manifest merely because it was filtered out of
    # the npz rows.
    verification_modes_by_route: dict[str, set[str]] = {
        route: set() for route in ROUTES
    }
    for item in collected:
        if item.get("skipped"):
            continue
        for route in ROUTES:
            verification_modes_by_route[route].add(
                _normalize_verification_mode(
                    (item.get("verification_modes") or {}).get(route)
                )
            )
    verification_modes = set().union(*verification_modes_by_route.values())
    source_run_roots = [str(root) for root in roots]
    source_version_sets = _source_version_sets(collected)

    def dropped_record(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "pair_id": item["pair_id"],
            "reason": _item_drop_reason(item),
            **_record_source_fields(item),
        }

    if labels_only:
        report = {
            "schema_version": SCORE_SCHEMA_VERSION,
            "mode": "labels_only",
            "run_root": source_run_roots[0] if len(roots) == 1 else None,
            "run_roots": source_run_roots,
            "source_run_roots": source_run_roots,
            "source_version_sets": source_version_sets,
            "source_versions": source_version_sets,
            "objective_filter": objective_filter,
            "label_summary": label_summary,
            "dropped_pairs": [dropped_record(item) for item in dropped],
            "verification_modes_present": sorted(verification_modes),
            "verification_modes_by_route": {
                route: sorted(modes)
                for route, modes in verification_modes_by_route.items()
            },
            "records": [
                {
                    "pair_id": item["pair_id"],
                    "benchmark": item.get("benchmark", ""),
                    "family_id": item["family_id"],
                    "objective": item.get("objective", ""),
                    "verification_modes": dict(item["verification_modes"]),
                    **_record_source_fields(item),
                    **item["verdict"].to_dict(),
                }
                for item in trainable
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        report_path = output_root / f"{stem}.labels.json"
        _write_json(report_path, report)
        return None, report_path

    if not trainable:
        blocked_path = output_root / f"{stem}.blocked.json"
        _write_json(
            blocked_path,
            {
                "schema_version": SCORE_SCHEMA_VERSION,
                "mode": "blocked",
                "dataset_role": "training_candidate",
                "training_allowed": False,
                "training_block_reason": (
                    "No route pair passed both the label gates and the "
                    "per-route JasperGold verification-mode gate."
                ),
                "label_summary": label_summary,
                "verification_modes_present": sorted(verification_modes),
                "verification_modes_by_route": {
                    route: sorted(modes)
                    for route, modes in verification_modes_by_route.items()
                },
                "dropped_pairs": [dropped_record(item) for item in dropped],
                "run_root": source_run_roots[0] if len(roots) == 1 else None,
                "run_roots": source_run_roots,
                "source_run_roots": source_run_roots,
                "source_version_sets": source_version_sets,
                "source_versions": source_version_sets,
                "objective_filter": objective_filter,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise ValueError(
            f"No trainable pairs under {', '.join(source_run_roots)}: "
            f"{label_summary['labeled_pairs']} labeled pairs were rejected by "
            "the label or per-route verification gates. "
            f"See {blocked_path} for each reason."
        )

    vectorizer = FeatureVectorizer()
    encoder = LatentEncoder(
        model_alias,
        output_dim=latent_dim,
        max_seq_length=8192,
        device=device,
        embedding_cache=output_root / "embedding_cache",
    )

    explicit_rows: list[np.ndarray] = []
    latent_rows: list[np.ndarray] = []
    label_values: list[int] = []
    spec_ids: list[str] = []
    family_ids: list[str] = []
    objectives: list[str] = []
    records: list[dict[str, Any]] = []
    feature_names: tuple[str, ...] | None = None

    for index, item in enumerate(trainable):
        pair_root: Path = item["pair_root"]
        feature_path = pair_root / "frozen_feature_result.json"
        if not feature_path.is_file():
            raise FileNotFoundError(
                f"{item['pair_id']}: frozen_feature_result.json is missing, so the "
                "explicit features of the two routes cannot be proven identical"
            )
        feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
        spec_text = _spec_text_for(pair_root, feature_payload)
        explicit = vectorizer.transform_one(
            feature_payload, objective=item["objective"]
        )
        if feature_names is None:
            feature_names = explicit.feature_names
        elif explicit.feature_names != feature_names:
            raise RuntimeError(
                f"{item['pair_id']}: explicit feature schema changed mid-dataset"
            )
        latent = encoder.encode_one(item["pair_id"], spec_text)

        explicit_rows.append(explicit.vector)
        latent_rows.append(latent.vector)
        label_values.append(int(item["verdict"].label))
        spec_ids.append(item["pair_id"])
        family_ids.append(item["family_id"])
        objectives.append(item["objective"])
        records.append(
            {
                "index": index,
                "spec_id": item["pair_id"],
                "benchmark": item["benchmark"],
                "family_id": item["family_id"],
                "objective": item["objective"],
                "verification_mode": item["verification_mode"],
                "verification_modes": dict(item["verification_modes"]),
                **_record_source_fields(item),
                "spec_sha256_canonical": sha256_text(
                    canonicalize_spec_text(spec_text)
                ),
                "frozen_feature_path": str(feature_path),
                "frozen_feature_sha256": _file_sha256(feature_path),
                "explicit_vector_sha256": explicit.metadata["vector_sha256"],
                "latent_vector_sha256": latent.metadata["vector_sha256"],
                "latent_cache_hit": latent.cache_hit,
                **item["verdict"].to_dict(),
            }
        )
        print(
            f"[{index + 1}/{len(trainable)}] {item['pair_id']}: "
            f"y={item['verdict'].label} basis={item['verdict'].label_basis} "
            f"cache_hit={latent.cache_hit}",
            flush=True,
        )

    explicit_matrix = np.stack(explicit_rows).astype(np.float32)
    latent_matrix = np.stack(latent_rows).astype(np.float32)
    labels = np.asarray(label_values, dtype=np.int64)
    family_array = np.asarray(family_ids, dtype=str)

    eligibility = _eligibility_block(
        labels=labels,
        family_ids=family_array,
        verification_modes=verification_modes_by_route,
        config=_predictor_module().PredictorConfig(),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    npz_path = output_root / f"{stem}.npz"
    manifest_path = output_root / f"{stem}.json"
    np.savez_compressed(
        npz_path,
        explicit=explicit_matrix,
        latent=latent_matrix,
        labels=labels,
        spec_ids=np.asarray(spec_ids, dtype=str),
        family_ids=family_array,
        objectives=np.asarray(objectives, dtype=str),
    )
    manifest = {
        "schema_version": _predictor_module().DATASET_SCHEMA_VERSION,
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "dataset_role": "training_candidate",
        "training_allowed": eligibility["training_allowed"],
        "training_block_reason": "; ".join(eligibility["training_block_reasons"]),
        "training_eligibility": eligibility,
        "sample_count": int(labels.size),
        "explicit_dim": int(explicit_matrix.shape[1]),
        "latent_dim": int(latent_matrix.shape[1]),
        "combined_dim": int(explicit_matrix.shape[1] + latent_matrix.shape[1]),
        "label_encoding": {"0": "rtl_direct", "1": "c_first"},
        "labels_present": True,
        "label_definition": (
            "revise plan §5: Score = -inf when any of "
            "syntax/equivalence/synthesis/metric fails, else -primary_metric; "
            "y = 1 iff Score_c > Score_d. No tie state: equal finite scores give "
            "y = 0. Pairs with both scores at -inf are dropped as unlabelable."
        ),
        "label_summary": label_summary,
        "run_root": source_run_roots[0] if len(roots) == 1 else None,
        "run_roots": source_run_roots,
        "source_run_roots": source_run_roots,
        "source_version_sets": source_version_sets,
        "source_versions": source_version_sets,
        "objective_filter": objective_filter,
        "objectives_present": sorted(set(objectives)),
        "verification_modes_present": sorted(verification_modes),
        "verification_modes_by_route": {
            route: sorted(modes)
            for route, modes in verification_modes_by_route.items()
        },
        "verification_mode_policy": (
            "Both c_first and rtl_direct route records must declare "
            "verification_mode=jaspergold before their labels are eligible "
            "for training."
        ),
        "explicit_feature_schema": vectorizer.schema_version,
        "explicit_feature_names": list(feature_names or ()),
        "latent_protocol": encoder.protocol,
        "latent_protocol_sha256": encoder.protocol_sha256,
        "npz_path": str(npz_path),
        "npz_sha256": _file_sha256(npz_path),
        "records": records,
        "dropped_pairs": [dropped_record(item) for item in dropped],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)
    return npz_path, manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a labeled adaptive-router dataset from executed dual-path runs "
            "(revise plan §5 Score comparison)."
        )
    )
    parser.add_argument(
        "--run-root",
        action="append",
        required=True,
        help=(
            "Directory containing pairs/<pair_id>/pair_manifest.json; repeat "
            "the option to combine multiple run roots"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "experiments" / "adaptive_router" / "datasets"),
    )
    parser.add_argument("--model", choices=("0.6b", "8b"), default="8b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument(
        "--objective",
        choices=OBJECTIVE_FILTERS,
        default="ALL",
        type=str.upper,
        help="Keep all pairs, or only pairs for one PPA objective",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Only report the label distribution; load no model and write no npz",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    npz_path, manifest_path = build_dataset(
        run_roots=[Path(root) for root in args.run_root],
        output_root=Path(args.output_root),
        model_alias=args.model,
        device=args.device,
        latent_dim=args.latent_dim,
        labels_only=args.labels_only,
        objective=args.objective,
    )
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_summary = summary.get("label_summary", {})
    print()
    print(f"pairs found            : {label_summary.get('pair_count')}")
    print(f"trainable labels       : {label_summary.get('trainable_pairs')}")
    print(f"  two-sided PPA        : {label_summary.get('two_sided_ppa_comparisons')}")
    print(
        f"  of which exact ties  : "
        f"{label_summary.get('exact_metric_ties_labeled_y0')} (labeled y=0)"
    )
    print(
        f"  one-sided gate DQ    : "
        f"{label_summary.get('one_sided_gate_disqualifications')}"
    )
    print(f"dropped unlabelable    : {label_summary.get('dropped_unlabelable')}")
    print(f"label counts           : {label_summary.get('label_counts')}")
    if npz_path is not None:
        print(f"training_allowed       : {summary.get('training_allowed')}")
        for reason in summary.get("training_eligibility", {}).get(
            "training_block_reasons", []
        ):
            print(f"  blocked by           : {reason}")
        print(f"wrote {npz_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
