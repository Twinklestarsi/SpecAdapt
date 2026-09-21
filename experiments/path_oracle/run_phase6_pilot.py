"""Run the phase-6 paired pilot strictly serially and verify every candidate."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import statistics
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List

from dotenv import dotenv_values

from experiments.path_oracle.run_dual_path import (
    CODE_DIRS,
    EXECUTION_POLICY,
    PROMPT_FILES,
    _canonical_sha256,
    _execute_pair,
    _file_sha256,
    _safe_name,
    _tree_sha256,
    _write_json,
    build_pair_plans,
)
from pipeline.preflight import build_preflight
from project_paths import PROJECT_ROOT


SERIAL_POLICY = dict(EXECUTION_POLICY)
TERMINAL_VERIFICATION = {"passed", "failed", "timeout", "error", "not_run"}
CACHE_SCHEMA_VERSION = "phase6_processed_pairs_v1"
ROUTES = ("c_first", "rtl_direct")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _exclusive_serial_lock(lock_path: Path) -> Iterator[None]:
    """Prevent two phase-6 processes from issuing overlapping AI/DC work."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another phase-6 pilot process holds {lock_path}; refusing concurrency"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={_utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _shared_memory_paths(output_root: Path) -> tuple[Path, Path]:
    root = output_root / "shared_memory_baseline"
    return root / "memory.db", root / "memory.json"


def _ensure_shared_memory_baseline(
    output_root: Path,
    *,
    source_db: Path,
    source_json: Path,
) -> tuple[Path, Path]:
    """Freeze one Memory baseline for every current and future resumed pair."""

    shared_db, shared_json = _shared_memory_paths(output_root)
    if shared_db.is_file() and shared_json.is_file():
        return shared_db, shared_json
    if shared_db.exists() or shared_json.exists():
        raise FileExistsError(
            f"Incomplete shared Memory baseline under {shared_db.parent}"
        )
    if not source_db.is_file() or not source_json.is_file():
        raise FileNotFoundError("Project Memory baseline files are required")
    shared_db.parent.mkdir(parents=True, exist_ok=True)
    temporary_db = shared_db.with_name(f".{shared_db.name}.tmp.{os.getpid()}")
    temporary_json = shared_json.with_name(
        f".{shared_json.name}.tmp.{os.getpid()}"
    )
    shutil.copy2(source_db, temporary_db)
    shutil.copy2(source_json, temporary_json)
    os.replace(temporary_db, shared_db)
    os.replace(temporary_json, shared_json)
    _write_json(
        shared_db.parent / "baseline_manifest.json",
        {
            "schema_version": "phase6_shared_memory_v1",
            "source_database": str(source_db),
            "source_json": str(source_json),
            "database_path": str(shared_db),
            "database_sha256": _file_sha256(shared_db),
            "json_path": str(shared_json),
            "json_sha256": _file_sha256(shared_json),
            "created_at": _utc_now(),
        },
    )
    return shared_db, shared_json


def _load_pilot_cases(pilot_root: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest_path = pilot_root / "pilot_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = dict(payload.get("execution_policy") or {})
    if policy.get("mode") != "serial" or policy.get("max_concurrency") != 1:
        raise ValueError("pilot_manifest.json must enforce serial max_concurrency=1")
    cases = list(payload.get("cases") or [])
    if len(cases) != int(payload.get("case_count", -1)):
        raise ValueError("pilot case_count does not match cases")
    entries: List[Dict[str, Any]] = []
    for case in cases:
        spec_path = pilot_root / str(case["spec_path"])
        golden_path = pilot_root / str(case["golden_rtl_path"])
        if not spec_path.is_file() or not golden_path.is_file():
            raise FileNotFoundError(f"Missing pilot pair for {case.get('id')}")
        entries.append(
            {
                "id": str(case["id"]),
                "family_id": str(case.get("family_id") or case["id"]),
                "spec": spec_path.read_text(encoding="utf-8"),
            }
        )
    return payload, entries


def _verify_pair_payload(
    pair_payload: Dict[str, Any],
    *,
    case: Dict[str, Any],
    pilot_root: Path,
    output_root: Path,
    timeout: int,
    verification_mode: str = "jaspergold",
    env_path: str | Path = PROJECT_ROOT / ".env",
) -> Dict[str, Any]:
    del timeout, env_path  # Verification already ran once inside each route.
    verification_mode = str(verification_mode or "jaspergold").strip().lower()
    if verification_mode not in {"jaspergold", "none"}:
        raise ValueError(f"Unsupported verification_mode: {verification_mode}")
    pair_id = str(pair_payload["pair_id"])
    golden_path = pilot_root / str(case["golden_rtl_path"])
    testbench_path = pilot_root / str(case.get("testbench_path") or "")
    verifications: Dict[str, Any] = {}
    for route in pair_payload.get("route_order") or ("c_first", "rtl_direct"):
        route_run = dict((pair_payload.get("runs") or {}).get(route) or {})
        verification_dir = output_root / "pairs" / pair_id / "verification" / route
        if verification_mode == "jaspergold":
            verification = dict(route_run.get("pre_dc_verification") or {})
            if not verification:
                verification = {
                    "status": "error",
                    "equivalent": False,
                    "error": "missing_pre_dc_jaspergold_result",
                }
            verification["verification_policy"] = (
                "syntax_then_jaspergold_with_llm_retry_then_dc"
            )
            verification["backend_results"] = {
                "jaspergold": dict(route_run.get("pre_dc_verification") or {})
            }
        else:
            verification = {
                "status": "not_run",
                "equivalent": None,
                "method": "none",
                "verification_policy": "syntax_then_dc_without_equivalence",
                "backend_results": {},
                "reason": "verification_mode_none",
            }
        verification["verification_mode"] = verification_mode
        verification["route"] = route
        verification["pipeline_status"] = str(route_run.get("status") or "")
        verification["dc_synthesis_status"] = str(
            route_run.get("synthesis_status") or ""
        )
        _write_json(verification_dir / "combined_verification.json", verification)
        verifications[route] = verification
        print(
            f"[verify] {pair_id} {route}: {verification['status']}",
            flush=True,
        )
    result = {
        "schema_version": "phase6_pair_result_v2",
        "verification_mode": verification_mode,
        "pair": pair_payload,
        "case": {
            "id": case["id"],
            "category": case.get("category", ""),
            "design_type": case.get("design_type", ""),
            "golden_rtl_path": str(golden_path.resolve()),
            "golden_rtl_sha256": _file_sha256(golden_path),
            "golden_top_module": case.get("golden_top_module", ""),
            "testbench_path": (
                str(testbench_path.resolve()) if testbench_path.is_file() else ""
            ),
            "testbench_sha256": _file_sha256(testbench_path),
        },
        "execution_policy": dict(SERIAL_POLICY),
        "verifications": verifications,
        "updated_at": _utc_now(),
    }
    _write_json(
        output_root / "pairs" / pair_id / "phase6_pair_result.json", result
    )
    return result


def _resume_validation_errors(
    existing: Dict[str, Any],
    *,
    planned_pair: Any,
    case: Dict[str, Any],
    pilot_root: Path,
    verification_mode: str,
) -> List[str]:
    """Return reasons an existing result is unsafe to reuse.

    Cache reuse is deliberately content-addressed.  A terminal status alone is
    insufficient because the spec, golden RTL, prompts, implementation code or
    backend configuration may have changed since the result was produced.
    """

    errors: List[str] = []
    existing_pair = dict(existing.get("pair") or {})
    existing_case = dict(existing.get("case") or {})
    expected = {
        "verification_mode": verification_mode,
        "pair_id": planned_pair.pair_id,
        "benchmark": planned_pair.benchmark,
        "objective": planned_pair.objective,
        "repeat_index": planned_pair.repeat_index,
        "spec_sha256": planned_pair.spec_sha256,
        "run_config_sha256": planned_pair.run_config_sha256,
        "code_tree_sha256": planned_pair.code_tree_sha256,
        "prompt_tree_sha256": planned_pair.prompt_tree_sha256,
        "mcts_seed": planned_pair.mcts_seed,
        "llm_seed": planned_pair.llm_seed,
    }
    actual = {
        "verification_mode": str(existing.get("verification_mode") or "")
        .strip()
        .lower(),
        **{key: existing_pair.get(key) for key in expected if key != "verification_mode"},
    }
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            errors.append(
                f"{key}: cached={actual.get(key)!r}, expected={expected_value!r}"
            )

    golden_path = pilot_root / str(case["golden_rtl_path"])
    golden_sha256 = _file_sha256(golden_path)
    if not golden_sha256:
        errors.append(f"current golden RTL is missing: {golden_path}")
    elif existing_case.get("golden_rtl_sha256") != golden_sha256:
        errors.append(
            "golden_rtl_sha256: "
            f"cached={existing_case.get('golden_rtl_sha256')!r}, "
            f"expected={golden_sha256!r}"
        )

    testbench_raw_path = str(case.get("testbench_path") or "")
    if testbench_raw_path:
        testbench_path = pilot_root / testbench_raw_path
        testbench_sha256 = _file_sha256(testbench_path)
        if not testbench_sha256:
            errors.append(f"current testbench is missing: {testbench_path}")
        elif existing_case.get("testbench_sha256") != testbench_sha256:
            errors.append(
                "testbench_sha256: "
                f"cached={existing_case.get('testbench_sha256')!r}, "
                f"expected={testbench_sha256!r}"
            )

    verifications = dict(existing.get("verifications") or {})
    runs = dict(existing_pair.get("runs") or {})
    for route in ROUTES:
        verification = dict(verifications.get(route) or {})
        status = str(verification.get("status") or "")
        if status not in TERMINAL_VERIFICATION:
            errors.append(f"{route} verification is not terminal: {status or 'missing'}")
        route_run = dict(runs.get(route) or {})
        if not route_run or str(route_run.get("status") or "") == "planned":
            errors.append(f"{route} route result is missing or still planned")

    feature_path = Path(str(existing_pair.get("feature_snapshot_path") or ""))
    feature_sha256 = str(existing_pair.get("feature_sha256") or "")
    if not feature_path.is_file():
        errors.append(f"frozen feature snapshot is missing: {feature_path}")
    else:
        try:
            feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
            actual_feature_sha256 = _canonical_sha256(feature_payload)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"frozen feature snapshot is unreadable: {exc}")
        else:
            if actual_feature_sha256 != feature_sha256:
                errors.append(
                    "feature_sha256: "
                    f"cached={feature_sha256!r}, actual={actual_feature_sha256!r}"
                )
    return errors


def _median(values: List[float]) -> float | None:
    return statistics.median(values) if values else None


#: Which route statistic decides the label, per PPA objective.  Both metrics are
#: "lower is better", so the comparison direction below is the same for either one.
#: ``area`` uses the reported cell area; ``timing`` uses the real critical-path
#: arrival time rather than slack, because slack saturates as soon as the clock
#: period is met and would collapse a genuine delay spread into a tie.
PPA_OBJECTIVES: Dict[str, Dict[str, str]] = {
    "area": {
        "metric_word": "area",
        "median_key": "median_area",
        "eligible_key": "eligible_area_samples",
        "samples_key": "area_samples",
        "run_field": "area",
        "units": "synthesis area units",
    },
    "timing": {
        "metric_word": "delay",
        "median_key": "median_delay_ps",
        "eligible_key": "eligible_delay_samples",
        "samples_key": "delay_samples_ps",
        "run_field": "delay_ps",
        "units": "picoseconds of critical-path arrival time",
    },
}


def aggregate_pilot_results(
    pair_results: List[Dict[str, Any]],
    *,
    repeats: int,
    ppa_tie_tolerance_pct: float = 1.0,
    verification_mode: str = "jaspergold",
    ppa_objective: str = "area",
) -> Dict[str, Any]:
    """Build mode-aware provisional labels from route outcomes and the PPA metric.

    ``ppa_objective`` selects which metric decides the label.  It must match the
    Design Compiler goal the routes were actually run under: the ``area`` preset
    uses a 5000ps clock and ``set_max_area 0`` while ``timing`` uses 1000ps and a
    plain ``compile``, so scoring a timing-goal run on area (or the reverse) reads
    the wrong column of a five-times-different experiment.
    """
    verification_mode = str(verification_mode or "jaspergold").strip().lower()
    if verification_mode not in {"jaspergold", "none"}:
        raise ValueError(f"Unsupported verification_mode: {verification_mode}")
    ppa_objective = str(ppa_objective or "area").strip().lower()
    if ppa_objective not in PPA_OBJECTIVES:
        raise ValueError(f"Unsupported ppa_objective: {ppa_objective}")
    metric = PPA_OBJECTIVES[ppa_objective]
    metric_word = metric["metric_word"]

    by_benchmark: Dict[str, List[Dict[str, Any]]] = {}
    for item in pair_results:
        benchmark = str(item["pair"]["benchmark"])
        by_benchmark.setdefault(benchmark, []).append(item)

    majority = math.floor(repeats / 2) + 1
    cases: List[Dict[str, Any]] = []
    for benchmark, items in sorted(by_benchmark.items()):
        route_stats: Dict[str, Any] = {}
        for route in ("c_first", "rtl_direct"):
            declared_verification_modes = sorted(
                {
                    str(
                        item["pair"]["runs"][route].get("verification_mode")
                        or "none"
                    )
                    .strip()
                    .lower()
                    for item in items
                }
            )
            verified_items = [
                item
                for item in items
                if (item.get("verifications") or {}).get(route, {}).get("status")
                == "passed"
            ]
            synthesis_successes = [
                item
                for item in items
                if str(
                    item["pair"]["runs"][route].get("synthesis_status") or ""
                ).lower()
                == "success"
            ]
            syntax_passes = [
                item
                for item in items
                if str(
                    item["pair"]["runs"][route].get("syntax_status") or ""
                ).lower()
                in {"ok", "passed", "success"}
            ]
            eligible_source = (
                verified_items if verification_mode == "jaspergold" else syntax_passes
            )
            ppa_items = [
                item
                for item in eligible_source
                if str(
                    item["pair"]["runs"][route].get("synthesis_status") or ""
                ).lower()
                == "success"
            ]
            areas = [
                float(item["pair"]["runs"][route]["area"])
                for item in ppa_items
                if item["pair"]["runs"][route].get("area") is not None
            ]
            delays = []
            for item in ppa_items:
                route_run = item["pair"]["runs"][route]
                raw_delay = route_run.get("delay_ps")
                if raw_delay is None:
                    raw_delay = route_run.get("data_arrival_time_ps")
                if raw_delay is not None:
                    delays.append(float(raw_delay))
            route_stats[route] = {
                "verification_modes": declared_verification_modes,
                "verification_passes": len(verified_items),
                "verification_pass_rate": len(verified_items) / max(1, len(items)),
                "verification_not_run": (
                    len(items) if verification_mode == "none" else 0
                ),
                "syntax_passes": len(syntax_passes),
                "synthesis_successes": len(synthesis_successes),
                "verified_synthesis_successes": (
                    len(ppa_items) if verification_mode == "jaspergold" else 0
                ),
                "eligible_synthesis_successes": len(ppa_items),
                "eligible_area_samples": len(areas),
                "eligible_delay_samples": len(delays),
                "area_samples": areas,
                "median_area": _median(areas),
                "area_sample_stdev": (
                    statistics.stdev(areas)
                    if len(areas) > 1
                    else 0.0
                    if areas
                    else None
                ),
                "delay_samples_ps": delays,
                "median_delay_ps": _median(delays),
            }

        evidence_field = (
            "verification_passes"
            if verification_mode == "jaspergold"
            else "syntax_passes"
        )
        c_verified = route_stats["c_first"][evidence_field] >= majority
        r_verified = route_stats["rtl_direct"][evidence_field] >= majority
        c_ok = route_stats["c_first"][metric["eligible_key"]] >= majority
        r_ok = route_stats["rtl_direct"][metric["eligible_key"]] >= majority
        label = "unsolved"
        qualifier = "verified" if verification_mode == "jaspergold" else "unverified"
        reason = f"neither_route_reached_{qualifier}_{metric_word}_majority"
        label_basis = (
            "verified_ppa"
            if verification_mode == "jaspergold"
            else "syntax_and_unverified_ppa"
        )
        relative_metric_gap_pct = None
        if c_ok and not r_ok:
            label = "c_first"
            reason = f"only_c_first_reached_{qualifier}_{metric_word}_majority"
        elif r_ok and not c_ok:
            label = "rtl_direct"
            reason = f"only_rtl_direct_reached_{qualifier}_{metric_word}_majority"
        elif c_ok and r_ok:
            # Both metrics are lower-is-better, so one comparison serves both goals.
            c_value = route_stats["c_first"][metric["median_key"]]
            r_value = route_stats["rtl_direct"][metric["median_key"]]
            relative_gap_pct = abs(c_value - r_value) / max(
                min(c_value, r_value), 1e-12
            ) * 100.0
            if relative_gap_pct <= ppa_tie_tolerance_pct:
                label = "tie"
                reason = f"{qualifier}_{metric_word}_difference_within_tolerance"
            elif c_value < r_value:
                label = "c_first"
                reason = f"both_{qualifier}_c_first_lower_median_{metric_word}"
            else:
                label = "rtl_direct"
                reason = f"both_{qualifier}_rtl_direct_lower_median_{metric_word}"
            relative_metric_gap_pct = relative_gap_pct
        elif c_verified or r_verified:
            label = "flow_error"
            reason = (
                f"{qualifier}_evidence_majority_without_valid_{metric_word}_majority"
            )
            label_basis = (
                "flow_health"
                if verification_mode == "jaspergold"
                else "unverified_flow_health"
            )

        both_routes_declared_jaspergold = all(
            route_stats[route]["verification_modes"] == ["jaspergold"]
            for route in ("c_first", "rtl_direct")
        )
        case_training_eligible = (
            verification_mode == "jaspergold"
            and both_routes_declared_jaspergold
            and label in {"c_first", "rtl_direct"}
        )

        cases.append(
            {
                "benchmark": benchmark,
                "repeat_count": len(items),
                "verification_majority_required": majority,
                "verification_mode": verification_mode,
                "training_eligible": case_training_eligible,
                "training_rejection_reason": (
                    ""
                    if case_training_eligible
                    else "both route records must declare verification_mode=jaspergold"
                    if verification_mode == "jaspergold"
                    and not both_routes_declared_jaspergold
                    else "label is not a trainable c_first/rtl_direct decision"
                ),
                "routes": route_stats,
                "provisional_label": label,
                "label_reason": reason,
                "label_basis": label_basis,
                "ppa_objective": ppa_objective,
                "ppa_metric": metric["median_key"],
                "relative_metric_gap_pct": relative_metric_gap_pct,
                # Kept under its historical name so existing readers of the
                # AREA runs keep working; it is null on a timing run.
                "relative_area_gap_pct": (
                    relative_metric_gap_pct if ppa_objective == "area" else None
                ),
            }
        )

    counts: Dict[str, int] = {}
    for item in cases:
        label = str(item["provisional_label"])
        counts[label] = counts.get(label, 0) + 1
    training_eligible_case_count = sum(
        1 for item in cases if item.get("training_eligible") is True
    )
    return {
        "schema_version": "phase6_pilot_summary_v3",
        "objective": ppa_objective.upper(),
        "ppa_objective": ppa_objective,
        "ppa_metric": metric["median_key"],
        "ppa_metric_units": metric["units"],
        "verification_mode": verification_mode,
        "training_eligible": training_eligible_case_count > 0,
        "training_eligible_case_count": training_eligible_case_count,
        "training_rejected_case_count": len(cases) - training_eligible_case_count,
        "execution_policy": dict(SERIAL_POLICY),
        "repeats": repeats,
        "ppa_tie_tolerance_pct": ppa_tie_tolerance_pct,
        "provisional_labels_only": True,
        "label_eligibility": (
            "both route records must declare verification_mode=jaspergold; "
            "a scored route must also reach the repeat majority with JasperGold "
            f"equivalence passed, synthesis success, and a non-null {metric_word} "
            "sample"
            if verification_mode == "jaspergold"
            else "route needs syntax success, synthesis success, and a non-null "
            f"{metric_word} sample; labels are unverified and not training-eligible"
        ),
        "label_counts": counts,
        "cases": cases,
        "updated_at": _utc_now(),
    }


def _write_progress_artifacts(
    *,
    output_root: Path,
    collection_path: Path,
    collection: Dict[str, Any],
    pair_results: List[Dict[str, Any]],
    planned_pair_count: int,
    repeats: int,
    ppa_tie_tolerance_pct: float,
    verification_mode: str,
    ppa_objective: str = "area",
) -> None:
    """Persist a compact cache index and partial aggregate after every pair."""

    entries: List[Dict[str, Any]] = []
    result_paths: List[str] = []
    for item in pair_results:
        pair = dict(item.get("pair") or {})
        case = dict(item.get("case") or {})
        pair_id = str(pair.get("pair_id") or "")
        result_path = output_root / "pairs" / pair_id / "phase6_pair_result.json"
        result_paths.append(str(result_path))
        entries.append(
            {
                "pair_id": pair_id,
                "benchmark": pair.get("benchmark"),
                "objective": pair.get("objective"),
                "repeat_index": pair.get("repeat_index"),
                "verification_mode": item.get("verification_mode"),
                "route_status": {
                    route: dict((pair.get("runs") or {}).get(route) or {}).get(
                        "status"
                    )
                    for route in ROUTES
                },
                "verification_status": {
                    route: dict((item.get("verifications") or {}).get(route) or {}).get(
                        "status"
                    )
                    for route in ROUTES
                },
                "spec_sha256": pair.get("spec_sha256"),
                "golden_rtl_sha256": case.get("golden_rtl_sha256"),
                "testbench_sha256": case.get("testbench_sha256"),
                "run_config_sha256": pair.get("run_config_sha256"),
                "code_tree_sha256": pair.get("code_tree_sha256"),
                "prompt_tree_sha256": pair.get("prompt_tree_sha256"),
                "feature_sha256": pair.get("feature_sha256"),
                "result_path": str(result_path),
                "result_sha256": _file_sha256(result_path),
            }
        )

    cache_index = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "reuse_policy": (
            "Reuse only when pair/spec/golden/run/code/prompt/feature fingerprints "
            "match and both routes have terminal verification records."
        ),
        "output_root": str(output_root),
        "planned_pair_count": planned_pair_count,
        "processed_pair_count": len(entries),
        "entries": entries,
        "updated_at": _utc_now(),
    }
    cache_path = _write_json(output_root / "processed_pairs.json", cache_index)

    partial_summary = aggregate_pilot_results(
        pair_results,
        repeats=repeats,
        ppa_tie_tolerance_pct=ppa_tie_tolerance_pct,
        verification_mode=verification_mode,
        ppa_objective=ppa_objective,
    )
    partial_summary["partial"] = len(pair_results) < planned_pair_count
    partial_summary["planned_pair_count"] = planned_pair_count
    partial_summary["processed_pair_count"] = len(pair_results)
    partial_summary_path = _write_json(
        output_root / "phase6_partial_summary.json", partial_summary
    )

    collection["completed_pair_count"] = len(pair_results)
    collection["pair_results"] = result_paths
    collection["processed_pairs_path"] = str(cache_path)
    collection["partial_summary_path"] = str(partial_summary_path)
    collection["updated_at"] = _utc_now()
    _write_json(collection_path, collection)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phase-6 pilot serially; no concurrency option exists."
    )
    parser.add_argument(
        "--pilot-root", default=str(PROJECT_ROOT / "pilot_test")
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "pilot_test" / "phase6_area_serial"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mcts-iterations", type=int, default=240)
    parser.add_argument("--mcts-max-depth", type=int, default=5)
    parser.add_argument("--mcts-candidate-limit", type=int, default=48)
    parser.add_argument(
        "--max-actions",
        type=int,
        default=5,
        help="Maximum number of C-first planning actions; default 5.",
    )
    parser.add_argument("--rtl-max-retries", type=int, default=2)
    parser.add_argument("--dc-max-retries", type=int, default=0)
    parser.add_argument(
        "--verification-timeout",
        type=int,
        default=1000,
        help=(
            "Maximum JasperGold verification time per attempt in seconds; "
            "default 1000. This explicit Phase-6 value overrides JG_TIMEOUT."
        ),
    )
    parser.add_argument(
        "--verification-mode",
        choices=("jaspergold", "none"),
        default="jaspergold",
        help=(
            "jaspergold: syntax -> JG (with optional AI retry) -> DC; "
            "none: syntax -> DC without equivalence verification."
        ),
    )
    parser.add_argument(
        "--jg-max-retries",
        type=int,
        default=1,
        help="Maximum JG-driven AI RTL regenerations; default 1.",
    )
    parser.add_argument(
        "--require-jaspergold",
        action="store_true",
        help=(
            "Backward-compatible alias for --verification-mode jaspergold."
        ),
    )
    parser.add_argument("--ppa-tie-tolerance-pct", type=float, default=1.0)
    parser.add_argument(
        "--ppa-objective",
        choices=sorted(PPA_OBJECTIVES),
        default="area",
        help=(
            "Which PPA metric decides the label, and which Design Compiler goal "
            "preset the routes are run under.  `area` scores median cell area "
            "under the 5000ps/set_max_area-0 preset; `timing` scores median "
            "delay_ps (DC data_arrival_time_ps) under the 1000ps/plain-compile preset.  A "
            "delay-headroom dataset run on the area preset meets timing "
            "everywhere and measures nothing."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually issue serial AI/DC calls. Without this flag, plan only.",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild the summary from completed pair results; no AI/DC calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.execute and args.summarize_existing:
        raise ValueError("--execute and --summarize-existing are mutually exclusive")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.max_actions < 1:
        raise ValueError("--max-actions must be at least 1")
    if args.verification_timeout < 1:
        raise ValueError("--verification-timeout must be at least 1")
    if args.jg_max_retries < 0:
        raise ValueError("--jg-max-retries must be non-negative")
    if args.require_jaspergold and args.verification_mode == "none":
        raise ValueError(
            "--require-jaspergold conflicts with --verification-mode none"
        )
    pilot_root = Path(args.pilot_root).resolve()
    output_root = Path(args.output_root).resolve()
    pilot_manifest, entries = _load_pilot_cases(pilot_root)
    cases_by_id = {str(item["id"]): item for item in pilot_manifest["cases"]}

    if args.summarize_existing:
        expected = len(entries) * args.repeats
        with _exclusive_serial_lock(PROJECT_ROOT / ".phase6_serial.lock"):
            pair_results = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(
                    (output_root / "pairs").glob("*/phase6_pair_result.json")
                )
            ]
            if len(pair_results) != expected:
                raise ValueError(
                    f"Expected {expected} completed pair results, found "
                    f"{len(pair_results)}"
                )
            summary = aggregate_pilot_results(
                pair_results,
                repeats=args.repeats,
                ppa_tie_tolerance_pct=args.ppa_tie_tolerance_pct,
                verification_mode=str(
                    (
                        pair_results[0].get("verification_mode")
                        if pair_results
                        else ""
                    )
                    or args.verification_mode
                ),
                ppa_objective=args.ppa_objective,
            )
            summary_path = _write_json(
                output_root / "phase6_summary.json", summary
            )
            collection_path = output_root / "phase6_collection.json"
            collection = json.loads(collection_path.read_text(encoding="utf-8"))
            collection["completed_pair_count"] = len(pair_results)
            collection["summary_path"] = str(summary_path)
            collection["updated_at"] = _utc_now()
            _write_json(collection_path, collection)
        print(f"Rebuilt phase-6 summary without AI/DC: {summary_path}")
        return 0

    env = dotenv_values(PROJECT_ROOT / ".env")
    project_memory_db = PROJECT_ROOT / "memory_agent.db"
    project_memory_json = PROJECT_ROOT / "path_decisions_log.json"
    shared_memory_db, shared_memory_json = _shared_memory_paths(output_root)
    fingerprint_memory_db = (
        shared_memory_db if shared_memory_db.is_file() else project_memory_db
    )
    fingerprint_memory_json = (
        shared_memory_json if shared_memory_json.is_file() else project_memory_json
    )
    if args.execute and (
        not fingerprint_memory_db.is_file()
        or not fingerprint_memory_json.is_file()
    ):
        raise FileNotFoundError("Project Memory baseline files are required")
    preflight = build_preflight()
    if args.execute and args.verification_mode == "jaspergold" and not bool(
        (preflight.get("jaspergold") or {}).get("configured")
    ):
        raise RuntimeError(
            "Phase-6 requires JasperGold, but its remote configuration is not ready"
        )
    if args.execute and not (
        preflight["ready_for_c_first_execution"]
        and preflight["ready_for_rtl_direct_execution"]
    ):
        raise RuntimeError(
            "Phase-6 preflight failed; refusing to create partial pair outputs"
        )
    mcts_config = {
        "iterations": args.mcts_iterations,
        "max_depth": args.mcts_max_depth,
        "candidate_limit": args.mcts_candidate_limit,
        "max_actions": args.max_actions,
    }
    backend_environment_keys = (
        "OPENAI_API_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "CLOUD_MAX_TOKENS",
        "CLOUD_TIMEOUT",
        "CLOUD_REASONING_EFFORT",
        "DC_REMOTE_USER",
        "DC_REMOTE_HOST",
        "DC_REMOTE_BASE",
        "DC_ENV_SCRIPT",
        "DC_SHELL_PATH",
        "JG_REMOTE_USER",
        "JG_REMOTE_HOST",
        "JG_REMOTE_BASE",
        "JG_ENV_SCRIPT",
        "JG_BIN",
        "JG_TIMEOUT",
        "ASAP7_DB_PATH",
    )
    run_config = {
        "model": str(env.get("OPENAI_MODEL") or ""),
        "objective": args.ppa_objective.upper(),
        "repeats": args.repeats,
        "execution_policy": dict(SERIAL_POLICY),
        "mcts": mcts_config,
        "rtl_max_retries": args.rtl_max_retries,
        "dc_max_retries": args.dc_max_retries,
        "verification_timeout": args.verification_timeout,
        "verification_mode": args.verification_mode,
        "jg_max_retries": args.jg_max_retries,
        "ppa_tie_tolerance_pct": args.ppa_tie_tolerance_pct,
        "memory_baseline": {
            "database_sha256": _file_sha256(fingerprint_memory_db),
            "json_sha256": _file_sha256(fingerprint_memory_json),
        },
        "backend_environment": {
            key: str(env.get(key) or "") for key in backend_environment_keys
        },
    }
    plans = build_pair_plans(
        entries,
        default_objective=args.ppa_objective,
        repeats=args.repeats,
        spec_source=str(pilot_root / "pilot_manifest.json"),
        mcts_seed_base=7,
        llm_seed_base=7000,
        mcts_config=mcts_config,
        run_config_sha256=_canonical_sha256(run_config),
        code_tree_sha256=_tree_sha256(PROJECT_ROOT / item for item in CODE_DIRS),
        prompt_tree_sha256=_tree_sha256(PROJECT_ROOT / item for item in PROMPT_FILES),
        toolchain_status=preflight,
    )
    collection_path = output_root / "phase6_collection.json"
    collection: Dict[str, Any] = {
        "schema_version": "phase6_pilot_collection_v1",
        "mode": "execute" if args.execute else "plan_only",
        "objective": args.ppa_objective.upper(),
        "pilot_manifest": str((pilot_root / "pilot_manifest.json").resolve()),
        "execution_policy": dict(SERIAL_POLICY),
        "run_config": run_config,
        "run_config_sha256": _canonical_sha256(run_config),
        "preflight": preflight,
        "planned_pair_count": len(plans),
        "completed_pair_count": 0,
        "pair_results": [],
        "updated_at": _utc_now(),
    }
    _write_json(collection_path, collection)
    if not args.execute:
        print(f"Wrote serial phase-6 plan: {collection_path}")
        return 0

    # These limit numerical/tokenizer helper libraries too. Active AI calls are
    # synchronous already; the exclusive lock prevents a second pilot process.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    specs_by_id = {str(item["id"]): str(item["spec"]) for item in entries}
    pair_results: List[Dict[str, Any]] = []
    frozen_features: Dict[str, Dict[str, Any]] = {}
    with _exclusive_serial_lock(PROJECT_ROOT / ".phase6_serial.lock"):
        memory_db, memory_json = _ensure_shared_memory_baseline(
            output_root,
            source_db=project_memory_db,
            source_json=project_memory_json,
        )
        for index, pair in enumerate(plans, start=1):
            shared_feature_path = (
                output_root / "frozen_features" / f"{_safe_name(pair.benchmark)}.json"
            )
            if pair.benchmark not in frozen_features and shared_feature_path.is_file():
                frozen_features[pair.benchmark] = json.loads(
                    shared_feature_path.read_text(encoding="utf-8")
                )
            pair_result_path = (
                output_root / "pairs" / pair.pair_id / "phase6_pair_result.json"
            )
            if pair_result_path.is_file():
                existing = json.loads(pair_result_path.read_text(encoding="utf-8"))
                resume_errors = _resume_validation_errors(
                    existing,
                    planned_pair=pair,
                    case=cases_by_id[pair.benchmark],
                    pilot_root=pilot_root,
                    verification_mode=args.verification_mode,
                )
                if not resume_errors:
                    pair_results.append(existing)
                    feature_snapshot = Path(
                        str((existing.get("pair") or {}).get("feature_snapshot_path") or "")
                    )
                    if pair.benchmark not in frozen_features and feature_snapshot.is_file():
                        frozen_features[pair.benchmark] = json.loads(
                            feature_snapshot.read_text(encoding="utf-8")
                        )
                        _write_json(
                            shared_feature_path,
                            frozen_features[pair.benchmark],
                        )
                    print(
                        f"[resume {index}/{len(plans)}] {pair.pair_id}: complete",
                        flush=True,
                    )
                    _write_progress_artifacts(
                        output_root=output_root,
                        collection_path=collection_path,
                        collection=collection,
                        pair_results=pair_results,
                        planned_pair_count=len(plans),
                        repeats=args.repeats,
                        ppa_tie_tolerance_pct=args.ppa_tie_tolerance_pct,
                        verification_mode=args.verification_mode,
                        ppa_objective=args.ppa_objective,
                    )
                    continue
                raise FileExistsError(
                    f"Existing pair {pair.pair_id} is unsafe to reuse:\n- "
                    + "\n- ".join(resume_errors)
                    + "\nUse a different output root or archive the stale pair."
                )

            pair_manifest_path = (
                output_root / "pairs" / pair.pair_id / "pair_manifest.json"
            )
            if pair_manifest_path.exists():
                raise FileExistsError(
                    f"Partial pair requires manual review before resume: {pair_manifest_path}"
                )

            print(
                f"[execute {index}/{len(plans)}] {pair.pair_id} "
                f"routes={pair.route_order}",
                flush=True,
            )
            _execute_pair(
                pair,
                spec_text=specs_by_id[pair.benchmark],
                output_root=output_root,
                max_actions=args.max_actions,
                rtl_max_retries=args.rtl_max_retries,
                dc_max_retries=args.dc_max_retries,
                mcts_iterations=args.mcts_iterations,
                mcts_max_depth=args.mcts_max_depth,
                mcts_candidate_limit=args.mcts_candidate_limit,
                memory_baseline_db=memory_db,
                memory_baseline_json=memory_json,
                feature_payload_override=frozen_features.get(pair.benchmark),
                pre_dc_golden_rtl_path=(
                    pilot_root
                    / str(cases_by_id[pair.benchmark]["golden_rtl_path"])
                ),
                pre_dc_golden_top=str(
                    cases_by_id[pair.benchmark].get("golden_top_module") or ""
                ),
                pre_dc_design_type=str(
                    cases_by_id[pair.benchmark].get("design_type") or "combinational"
                ),
                pre_dc_verification_timeout=args.verification_timeout,
                verification_mode=args.verification_mode,
                jg_max_retries=args.jg_max_retries,
            )
            feature_snapshot = Path(pair.feature_snapshot_path)
            if pair.benchmark not in frozen_features and feature_snapshot.is_file():
                frozen_features[pair.benchmark] = json.loads(
                    feature_snapshot.read_text(encoding="utf-8")
                )
                _write_json(shared_feature_path, frozen_features[pair.benchmark])
            result = _verify_pair_payload(
                pair.to_dict(),
                case=cases_by_id[pair.benchmark],
                pilot_root=pilot_root,
                output_root=output_root,
                timeout=args.verification_timeout,
                verification_mode=args.verification_mode,
                env_path=PROJECT_ROOT / ".env",
            )
            pair_results.append(result)
            _write_progress_artifacts(
                output_root=output_root,
                collection_path=collection_path,
                collection=collection,
                pair_results=pair_results,
                planned_pair_count=len(plans),
                repeats=args.repeats,
                ppa_tie_tolerance_pct=args.ppa_tie_tolerance_pct,
                verification_mode=args.verification_mode,
                ppa_objective=args.ppa_objective,
            )

    summary = aggregate_pilot_results(
        pair_results,
        repeats=args.repeats,
        ppa_tie_tolerance_pct=args.ppa_tie_tolerance_pct,
        verification_mode=args.verification_mode,
        ppa_objective=args.ppa_objective,
    )
    summary_path = _write_json(output_root / "phase6_summary.json", summary)
    collection["completed_pair_count"] = len(pair_results)
    collection["pair_results"] = [
        str(
            output_root
            / "pairs"
            / item["pair"]["pair_id"]
            / "phase6_pair_result.json"
        )
        for item in pair_results
    ]
    collection["summary_path"] = str(summary_path)
    collection["updated_at"] = _utc_now()
    _write_json(collection_path, collection)
    print(f"Phase-6 serial pilot complete: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
