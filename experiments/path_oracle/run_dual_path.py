"""Plan or execute paired forced-path experiments without producing labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from experiments.path_oracle.schemas import (
    PairedRunManifest,
    PathRunManifest,
    path_run_from_pipeline,
)
from memory_agent import MemoryAgent
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.preflight import build_preflight
from project_paths import PROJECT_ROOT
from spec_analyze.analyzer import Analyzer
from spec_analyze.spec_loader import load_spec_entries


ROUTES = ("c_first", "rtl_direct")
EXECUTION_POLICY = {
    "mode": "serial",
    "max_concurrency": 1,
    "parallel_pairs": False,
    "parallel_routes": False,
    "parallel_llm_calls": False,
    "parallel_dc_jobs": False,
}
CODE_DIRS = (
    "c_gen",
    "experiments/path_oracle",
    "memory_agent",
    "module5",
    "path_select",
    "pipeline",
    "rag_retrieve",
    "spec_analyze",
    "RTL_DIRECT_compare",
    "llm_request.py",
    "llm_transform.py",
    "project_paths.py",
    "toolchain.py",
    "vcpp_cdfg.py",
)
PROMPT_FILES = (
    "spec_analyze/prompts.py",
    "spec_analyze/llm_features.py",
    "c_gen/llm_gen.py",
    "module5/editor.py",
    "module5/jg_verifier.py",
    "module5/rtl_direct_runner.py",
    "RTL_DIRECT_compare/llm_gen.py",
)

_GOLDEN_PATH_KEYS = (
    "golden_rtl_path",
    "golden_rtl",
    "golden_path",
    "path",
)
_GOLDEN_TOP_KEYS = ("golden_top_module", "golden_top", "top_module", "top")
_DESIGN_TYPE_KEYS = ("design_type", "type")
_ENTRY_LIST_KEYS = ("specs", "cases", "entries")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    return cleaned.strip("_") or "benchmark"


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(paths: Iterable[Path]) -> str:
    files: List[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*.py") if item.is_file())
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_input_path(value: Any, *, base_dir: Path) -> Path:
    """Resolve a path stored in a spec/manifest relative to that file."""

    raw = str(value or "").strip()
    if not raw:
        return Path("")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _first_nonempty(mapping: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _copy_entry_metadata(entry: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Copy fields that affect planning or per-entry verification."""

    for key in (
        "optimization_target",
        "family_id",
        *_GOLDEN_TOP_KEYS,
        *_DESIGN_TYPE_KEYS,
    ):
        if key in source and source[key] is not None:
            entry[key] = source[key]
    if source.get("family_id") or source.get("family"):
        entry["family_id"] = str(source.get("family_id") or source.get("family"))
    for key in _GOLDEN_PATH_KEYS:
        if key in source and source[key] is not None:
            entry[key] = source[key]


def _load_dual_path_entries(spec_path: Path) -> List[Dict[str, Any]]:
    """Load specs while retaining optional per-entry golden metadata.

    ``spec_analyze.spec_loader`` intentionally returns only the fields needed
    by the analyzer.  This entry point additionally accepts those fields from
    JSON specs and from pilot-style manifests.  A manifest case may point to a
    text file through ``spec_path``; that reference is resolved relative to
    the manifest, never guessed from the benchmark name.
    """

    raw_text = spec_path.read_text(errors="replace")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return load_spec_entries(spec_path)

    if isinstance(parsed, dict):
        raw_entries: Any = next(
            (parsed[key] for key in _ENTRY_LIST_KEYS if isinstance(parsed.get(key), list)),
            None,
        )
        if raw_entries is None and isinstance(parsed.get("spec"), str):
            raw_entries = [parsed]
    elif isinstance(parsed, list):
        raw_entries = parsed
    else:
        raw_entries = None

    if raw_entries is None:
        return load_spec_entries(spec_path)

    entries: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_entries, start=1):
        if isinstance(item, str):
            entries.append({
                "id": f"{spec_path.stem}_{index:02d}",
                "spec": item,
            })
            continue
        if not isinstance(item, dict):
            continue

        benchmark = str(item.get("id") or f"{spec_path.stem}_{index:02d}")
        if benchmark in seen_ids:
            raise ValueError(f"Duplicate spec entry id: {benchmark}")
        seen_ids.add(benchmark)
        spec_text = item.get("spec")
        if not isinstance(spec_text, str):
            spec_ref = item.get("spec_path")
            if not isinstance(spec_ref, str) or not spec_ref.strip():
                continue
            resolved_spec = _resolve_input_path(spec_ref, base_dir=spec_path.parent)
            if not resolved_spec.is_file():
                raise FileNotFoundError(
                    f"Spec for entry '{benchmark}' not found: {resolved_spec}"
                )
            spec_text = resolved_spec.read_text(errors="replace")

        entry = {"id": benchmark, "spec": spec_text}
        _copy_entry_metadata(entry, item)
        entries.append(entry)

    return entries or load_spec_entries(spec_path)


def _golden_descriptor(raw: Any, *, base_dir: Path) -> Dict[str, str]:
    """Normalize a path or descriptor from a spec/manifest mapping."""

    if isinstance(raw, str):
        if not raw.strip():
            return {"path": "", "top": "", "design_type": ""}
        return {
            "path": str(_resolve_input_path(raw, base_dir=base_dir)),
            "top": "",
            "design_type": "",
        }
    if not isinstance(raw, dict):
        return {"path": "", "top": "", "design_type": ""}

    nested = raw.get("golden")
    if isinstance(nested, (str, dict)):
        descriptor = _golden_descriptor(nested, base_dir=base_dir)
    else:
        descriptor = {"path": "", "top": "", "design_type": ""}
    path_value = _first_nonempty(raw, _GOLDEN_PATH_KEYS)
    top_value = _first_nonempty(raw, _GOLDEN_TOP_KEYS)
    design_value = _first_nonempty(raw, _DESIGN_TYPE_KEYS)
    if path_value:
        descriptor["path"] = str(
            _resolve_input_path(path_value, base_dir=base_dir)
        )
    if top_value:
        descriptor["top"] = top_value
    if design_value:
        descriptor["design_type"] = design_value
    return descriptor


def _entry_golden_descriptor(entry: Dict[str, Any], *, base_dir: Path) -> Dict[str, str]:
    return _golden_descriptor(entry, base_dir=base_dir)


def _golden_mapping_records(raw: Any, *, base_dir: Path) -> Dict[str, Dict[str, str]]:
    """Read an explicit per-benchmark golden mapping.

    Accepted forms are ``{"benchmark": "golden.v"}``,
    ``{"benchmark": {"golden_rtl_path": "..."}}``, and pilot/spec-style
    lists under ``cases``, ``specs`` or ``entries``.  A single descriptor is
    kept under ``__single__`` so it can be used for a one-spec input only.
    """

    if isinstance(raw, list):
        records: Dict[str, Dict[str, str]] = {}
        for index, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Golden mapping entry {index} must be an object with an id"
                )
            benchmark = str(item.get("id") or "").strip()
            if not benchmark:
                raise ValueError(
                    f"Golden mapping entry {index} is missing its id"
                )
            records[benchmark] = _golden_descriptor(item, base_dir=base_dir)
        return records

    if not isinstance(raw, dict):
        raise ValueError("Golden mapping JSON must be an object or a list")

    for key in _ENTRY_LIST_KEYS:
        items = raw.get(key)
        if isinstance(items, list):
            return _golden_mapping_records(items, base_dir=base_dir)

    if any(key in raw for key in _GOLDEN_PATH_KEYS):
        return {"__single__": _golden_descriptor(raw, base_dir=base_dir)}

    return {
        str(benchmark): _golden_descriptor(value, base_dir=base_dir)
        for benchmark, value in raw.items()
    }


def _resolve_golden_configs(
    entries: List[Dict[str, Any]],
    *,
    golden_rtl_arg: str,
    golden_top: str,
    design_type: str,
    spec_base_dir: Path,
) -> Dict[str, Dict[str, str]]:
    """Resolve one golden configuration for every spec entry.

    A direct ``--golden-rtl FILE`` is deliberately accepted only for one
    entry.  Batch inputs must carry per-entry metadata or use a JSON mapping,
    which prevents silently comparing every benchmark against one unrelated
    reference design.
    """

    external_records: Dict[str, Dict[str, str]] = {}
    if golden_rtl_arg:
        source = Path(golden_rtl_arg).expanduser()
        if not source.is_absolute():
            source = Path.cwd() / source
        source = source.resolve()
        if source.suffix.lower() == ".json":
            if not source.is_file():
                raise ValueError(
                    f"Golden mapping JSON not found: {source}"
                )
            try:
                raw = json.loads(source.read_text(errors="replace"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Golden mapping is not valid JSON: {source}"
                ) from exc
            external_records = _golden_mapping_records(
                raw, base_dir=source.parent
            )
        else:
            if len(entries) != 1:
                raise ValueError(
                    "A direct --golden-rtl path is valid only for one spec; "
                    "for batch specs provide per-entry golden_rtl_path fields "
                    "or a JSON mapping keyed by benchmark id"
                )
            external_records = {
                "__single__": {
                    "path": str(source),
                    "top": "",
                    "design_type": "",
                }
            }

    configs: Dict[str, Dict[str, str]] = {}
    for entry in entries:
        benchmark = str(entry["id"])
        descriptor = _entry_golden_descriptor(entry, base_dir=spec_base_dir)
        external = external_records.get(benchmark)
        if external is None and len(entries) == 1:
            external = external_records.get("__single__")
        if external is not None:
            descriptor = {
                "path": external.get("path", "") or descriptor.get("path", ""),
                "top": external.get("top", "") or descriptor.get("top", ""),
                "design_type": external.get("design_type", "")
                or descriptor.get("design_type", ""),
            }
        descriptor["top"] = golden_top or descriptor.get("top", "")
        descriptor["design_type"] = (
            design_type or descriptor.get("design_type", "") or "combinational"
        )
        configs[benchmark] = descriptor
    return configs


def _validate_golden_configs(
    entries: List[Dict[str, Any]],
    configs: Dict[str, Dict[str, str]],
    *,
    verification_mode: str,
    execute: bool,
) -> None:
    """Fail before any output is written when JG execution lacks references."""

    if not execute or verification_mode != "jaspergold":
        return
    errors: List[str] = []
    for entry in entries:
        benchmark = str(entry["id"])
        raw_path = str(configs.get(benchmark, {}).get("path", "")).strip()
        if not raw_path:
            errors.append(
                f"{benchmark}: no golden RTL path (provide --golden-rtl, "
                "a per-entry golden_rtl_path, or a JSON mapping)"
            )
            continue
        path = Path(raw_path)
        if not path.is_file():
            errors.append(f"{benchmark}: golden RTL not found: {path}")
    if errors:
        raise ValueError(
            "JasperGold execution requires one existing golden RTL per spec; "
            "refusing to create partial results. "
            + " | ".join(errors)
        )


@contextmanager
def _llm_seed(seed: int) -> Iterator[None]:
    previous = os.environ.get("OPENAI_SEED")
    os.environ["OPENAI_SEED"] = str(seed)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPENAI_SEED", None)
        else:
            os.environ["OPENAI_SEED"] = previous


def build_pair_plans(
    entries: Iterable[Dict[str, Any]],
    *,
    default_objective: str,
    repeats: int,
    spec_source: str,
    mcts_seed_base: int = 7,
    llm_seed_base: int = 7000,
    mcts_config: Dict[str, Any] | None = None,
    run_config_sha256: str = "",
    code_tree_sha256: str = "",
    prompt_tree_sha256: str = "",
    toolchain_status: Dict[str, Any] | None = None,
) -> List[PairedRunManifest]:
    """Build deterministic pair plans; no pipeline code is invoked here."""

    plans: List[PairedRunManifest] = []
    source = str(Path(spec_source).resolve())
    for entry in entries:
        benchmark = str(entry["id"])
        family_id = str(entry.get("family_id") or benchmark)
        spec_text = str(entry["spec"])
        requested = str(entry.get("optimization_target") or "").upper()
        objective = (
            requested
            if requested in {"AREA", "TIMING"}
            else default_objective.upper()
        )
        digest = hashlib.sha256(spec_text.encode("utf-8")).hexdigest()
        for repeat_index in range(repeats):
            order = list(ROUTES if repeat_index % 2 == 0 else reversed(ROUTES))
            pair_id = (
                f"{_safe_name(benchmark)}__{objective.lower()}__r{repeat_index:03d}"
            )
            created = _utc_now()
            plans.append(
                PairedRunManifest(
                    pair_id=pair_id,
                    benchmark=benchmark,
                    family_id=family_id,
                    objective=objective,
                    repeat_index=repeat_index,
                    route_order=order,
                    spec_source=source,
                    spec_sha256=digest,
                    mcts_seed=mcts_seed_base + repeat_index,
                    llm_seed=llm_seed_base + repeat_index,
                    mcts_config=dict(mcts_config or {}),
                    run_config_sha256=run_config_sha256,
                    code_tree_sha256=code_tree_sha256,
                    prompt_tree_sha256=prompt_tree_sha256,
                    toolchain_status=dict(toolchain_status or {}),
                    created_at=created,
                    updated_at=created,
                    runs={route: PathRunManifest(forced_path=route) for route in ROUTES},
                )
            )
    return plans


def _pair_state(pair: PairedRunManifest) -> str:
    statuses = [pair.runs[route].status for route in ROUTES]
    if all(status == "success" for status in statuses):
        return "complete"
    if any(status != "planned" for status in statuses):
        return "incomplete"
    return "planned"


def _prepare_memory_baseline(
    pair_root: Path,
    *,
    source_db: Path | None,
    source_json: Path | None,
) -> tuple[Path, Path, Dict[str, Any]]:
    baseline_root = pair_root / "memory_baseline"
    baseline_root.mkdir(parents=True, exist_ok=True)
    baseline_db = baseline_root / "memory.db"
    baseline_json = baseline_root / "memory.json"
    if baseline_db.exists() or baseline_json.exists():
        raise FileExistsError(
            f"Pair output already contains a Memory baseline: {baseline_root}"
        )

    if source_db is not None:
        shutil.copy2(source_db, baseline_db)
    else:
        MemoryAgent(baseline_json, db_path=baseline_db)

    if source_json is not None:
        shutil.copy2(source_json, baseline_json)
    elif not baseline_json.exists():
        _write_json(baseline_json, {})

    provenance = {
        "database_path": str(baseline_db),
        "database_sha256": _file_sha256(baseline_db),
        "json_path": str(baseline_json),
        "json_sha256": _file_sha256(baseline_json),
        "source_database": str(source_db or ""),
        "source_json": str(source_json or ""),
    }
    return baseline_json, baseline_db, provenance


def _copy_route_memory(
    baseline_json: Path,
    baseline_db: Path,
    route_root: Path,
) -> tuple[Path, Path]:
    memory_root = route_root / "memory"
    memory_root.mkdir(parents=True, exist_ok=True)
    route_json = memory_root / "memory.json"
    route_db = memory_root / "memory.db"
    if route_json.exists() or route_db.exists():
        raise FileExistsError(f"Route Memory already exists: {memory_root}")
    shutil.copy2(baseline_json, route_json)
    shutil.copy2(baseline_db, route_db)
    return route_json, route_db


def _execute_pair(
    pair: PairedRunManifest,
    *,
    spec_text: str,
    output_root: Path,
    max_actions: int | None,
    rtl_max_retries: int,
    dc_max_retries: int,
    mcts_iterations: int,
    mcts_max_depth: int,
    mcts_candidate_limit: int,
    memory_baseline_db: Path | None,
    memory_baseline_json: Path | None,
    feature_payload_override: Dict[str, Any] | None = None,
    pre_dc_golden_rtl_path: str | Path = "",
    pre_dc_golden_top: str = "",
    pre_dc_design_type: str = "combinational",
    pre_dc_verification_timeout: int = 180,
    verification_mode: str = "none",
    jg_max_retries: int = 1,
) -> None:
    if verification_mode == "jaspergold":
        golden_raw = str(pre_dc_golden_rtl_path or "").strip()
        if not golden_raw:
            raise ValueError(
                f"JasperGold execution for '{pair.benchmark}' requires a golden RTL path"
            )
        golden_path = Path(golden_raw).expanduser()
        if not golden_path.is_file():
            raise ValueError(
                f"JasperGold execution for '{pair.benchmark}' cannot find golden RTL: "
                f"{golden_path.resolve()}"
            )
    pair_root = output_root / "pairs" / pair.pair_id
    baseline_json, baseline_db, baseline_info = _prepare_memory_baseline(
        pair_root,
        source_db=memory_baseline_db,
        source_json=memory_baseline_json,
    )
    pair.memory_baseline = baseline_info

    # Module 1 is deliberately evaluated once.  Both forced routes receive the
    # exact same serialized FeatureResult object below.
    if feature_payload_override is None:
        with _llm_seed(pair.llm_seed):
            feature = Analyzer(env_path=str(PROJECT_ROOT / ".env")).analyze_spec(
                spec_text,
                benchmark_name=pair.benchmark,
                optimization_target=pair.objective,
            )
        feature_payload = feature.to_dict()
    else:
        feature_payload = dict(feature_payload_override)
    feature_path = _write_json(pair_root / "frozen_feature_result.json", feature_payload)
    pair.feature_snapshot_path = str(feature_path)
    pair.feature_sha256 = _canonical_sha256(feature_payload)
    pair.updated_at = _utc_now()
    _write_json(pair_root / "pair_manifest.json", pair.to_dict())

    for route in pair.route_order:
        route_root = pair_root / route
        route_json, route_db = _copy_route_memory(
            baseline_json, baseline_db, route_root
        )
        orchestrator = PipelineOrchestrator(
            memory_store_path=route_json,
            memory_db_path=route_db,
            output_root=route_root / "pipeline_runs",
            c_output_dir=route_root / "c_gen_output",
        )
        started = time.perf_counter()
        try:
            with _llm_seed(pair.llm_seed):
                result = orchestrator.run(
                    benchmark=pair.benchmark,
                    objective=pair.objective,
                    spec_text=spec_text,
                    use_llm_path_selection=False,
                    module5_backend="direct_rtl",
                    max_actions=max_actions,
                    run_baseline=False,
                    module5_output_root=route_root / "module5",
                    rtl_max_retries=rtl_max_retries,
                    dc_max_retries=dc_max_retries,
                    mcts_iterations=mcts_iterations,
                    mcts_max_depth=mcts_max_depth,
                    mcts_seed=pair.mcts_seed,
                    mcts_candidate_limit=mcts_candidate_limit,
                    forced_path=route,
                    feature_override=feature_payload,
                    pre_dc_golden_rtl_path=pre_dc_golden_rtl_path,
                    pre_dc_golden_top=pre_dc_golden_top,
                    pre_dc_design_type=pre_dc_design_type,
                    pre_dc_verification_timeout=pre_dc_verification_timeout,
                    verification_mode=verification_mode,
                    jg_max_retries=jg_max_retries,
                )
            pair.runs[route] = path_run_from_pipeline(
                result.to_dict(),
                forced_path=route,
                elapsed_seconds=time.perf_counter() - started,
                memory_store_path=str(route_json),
                memory_db_path=str(route_db),
            )
        except Exception as exc:
            pair.runs[route] = PathRunManifest(
                forced_path=route,
                status="pipeline_exception",
                elapsed_seconds=time.perf_counter() - started,
                memory_store_path=str(route_json),
                memory_db_path=str(route_db),
                failure_stage="orchestration",
                failure_reason=f"{type(exc).__name__}: {exc}",
            )
        pair.state = _pair_state(pair)
        pair.updated_at = _utc_now()
        _write_json(pair_root / "pair_manifest.json", pair.to_dict())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paired c_first/rtl_direct run manifests. By default this "
            "only writes a plan; --execute is required to call LLM/DC tools."
        )
    )
    parser.add_argument("--spec-file", required=True)
    parser.add_argument("--objective", choices=["area", "timing"], required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--mcts-iterations", type=int, default=240)
    parser.add_argument("--mcts-max-depth", type=int, default=5)
    parser.add_argument("--mcts-candidate-limit", type=int, default=48)
    parser.add_argument("--mcts-seed-base", type=int, default=7)
    parser.add_argument("--llm-seed-base", type=int, default=7000)
    parser.add_argument("--rtl-max-retries", type=int, default=2)
    parser.add_argument("--dc-max-retries", type=int, default=0)
    parser.add_argument("--memory-baseline-db", default=None)
    parser.add_argument("--memory-baseline-json", default=None)
    parser.add_argument(
        "--verification-mode",
        choices=("jaspergold", "none"),
        default="jaspergold",
        help=(
            "jaspergold: syntax -> JasperGold -> DC (default); "
            "none: syntax -> DC without equivalence verification."
        ),
    )
    parser.add_argument(
        "--golden-rtl",
        default="",
        help=(
            "Golden RTL path for one spec, or a JSON mapping keyed by "
            "benchmark id for batch specs. JSON spec/manifest entries may "
            "also provide golden_rtl_path."
        ),
    )
    parser.add_argument(
        "--golden-top",
        default="",
        help="Optional global golden top-module override.",
    )
    parser.add_argument(
        "--design-type",
        choices=("combinational", "sequential"),
        default="",
        help=(
            "Optional global design-type override; otherwise use the per-entry "
            "value, falling back to combinational."
        ),
    )
    parser.add_argument(
        "--verification-timeout",
        type=int,
        default=1000,
        help="Maximum JasperGold verification time per attempt; default 1000 seconds.",
    )
    parser.add_argument(
        "--jg-max-retries",
        type=int,
        default=1,
        help="Maximum JG-driven AI RTL regenerations; default 1.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run both routes. Omit this flag to produce a plan only.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.max_actions is not None and args.max_actions < 1:
        raise ValueError("--max-actions must be at least 1")
    if args.verification_timeout < 1:
        raise ValueError("--verification-timeout must be at least 1")
    if args.jg_max_retries < 0:
        raise ValueError("--jg-max-retries must be non-negative")
    for name in ("mcts_iterations", "mcts_max_depth", "mcts_candidate_limit"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")

    spec_path = Path(args.spec_file).resolve()
    output_root = Path(args.output_root).resolve()
    baseline_db = (
        Path(args.memory_baseline_db).resolve() if args.memory_baseline_db else None
    )
    baseline_json = (
        Path(args.memory_baseline_json).resolve()
        if args.memory_baseline_json
        else None
    )
    for label, path in (
        ("Memory baseline DB", baseline_db),
        ("Memory baseline JSON", baseline_json),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    mcts_config = {
        "iterations": args.mcts_iterations,
        "max_depth": args.mcts_max_depth,
        "candidate_limit": args.mcts_candidate_limit,
        "max_actions": args.max_actions,
    }
    preflight = build_preflight()
    run_config = {
        "model": os.environ.get("OPENAI_MODEL", ""),
        "objective": args.objective.upper(),
        "repeats": args.repeats,
        "execution_policy": dict(EXECUTION_POLICY),
        "mcts": mcts_config,
        "mcts_seed_base": args.mcts_seed_base,
        "llm_seed_base": args.llm_seed_base,
        "rtl_max_retries": args.rtl_max_retries,
        "dc_max_retries": args.dc_max_retries,
        "route_backend": {
            "c_first": "c_optimization_to_ai_rtl",
            "rtl_direct": "spec_to_rtl_ai",
        },
    }
    code_hash = _tree_sha256(PROJECT_ROOT / item for item in CODE_DIRS)
    prompt_hash = _tree_sha256(PROJECT_ROOT / item for item in PROMPT_FILES)
    entries = _load_dual_path_entries(spec_path)
    golden_configs = _resolve_golden_configs(
        entries,
        golden_rtl_arg=args.golden_rtl,
        golden_top=args.golden_top,
        design_type=args.design_type,
        spec_base_dir=spec_path.parent,
    )
    # Validate every reference before collection_manifest.json is written.
    # A batch with one missing reference must not leave a partially runnable
    # collection behind.
    _validate_golden_configs(
        entries,
        golden_configs,
        verification_mode=args.verification_mode,
        execute=args.execute,
    )
    golden_manifest = {
        benchmark: {
            "path": config.get("path", ""),
            "sha256": _file_sha256(Path(config["path"]))
            if config.get("path")
            else "",
            "top": config.get("top", ""),
            "design_type": config.get("design_type", ""),
        }
        for benchmark, config in golden_configs.items()
    }
    run_config.update(
        {
            "verification_mode": args.verification_mode,
            "verification_timeout": args.verification_timeout,
            "jg_max_retries": args.jg_max_retries,
            "golden_rtl_by_benchmark": golden_manifest,
        }
    )
    plans = build_pair_plans(
        entries,
        default_objective=args.objective,
        repeats=args.repeats,
        spec_source=str(spec_path),
        mcts_seed_base=args.mcts_seed_base,
        llm_seed_base=args.llm_seed_base,
        mcts_config=mcts_config,
        run_config_sha256=_canonical_sha256(run_config),
        code_tree_sha256=code_hash,
        prompt_tree_sha256=prompt_hash,
        toolchain_status=preflight,
    )
    specs_by_benchmark = {str(entry["id"]): str(entry["spec"]) for entry in entries}

    collection = {
        "schema_version": "path_oracle_collection_v2",
        "mode": "execute" if args.execute else "plan_only",
        "labels_generated": False,
        "spec_source": str(spec_path),
        "default_objective": args.objective.upper(),
        "repeats": args.repeats,
        "execution_policy": dict(EXECUTION_POLICY),
        "verification_mode": args.verification_mode,
        "verification_timeout": args.verification_timeout,
        "jg_max_retries": args.jg_max_retries,
        "golden_rtl_by_benchmark": golden_manifest,
        "run_config": run_config,
        "run_config_sha256": _canonical_sha256(run_config),
        "code_tree_sha256": code_hash,
        "prompt_tree_sha256": prompt_hash,
        "toolchain_status": preflight,
        "pairs": [pair.to_dict() for pair in plans],
    }
    plan_path = _write_json(output_root / "collection_manifest.json", collection)

    if not args.execute:
        print(f"Wrote plan only (no LLM/DC calls, no labels): {plan_path}")
        return 0

    # Intentional serial policy: finish one pair, including both routes and all
    # remote DC work, before beginning the next pair. There is no worker or
    # concurrency CLI option for path-oracle experiments.
    for pair in plans:
        _execute_pair(
            pair,
            spec_text=specs_by_benchmark[pair.benchmark],
            output_root=output_root,
            max_actions=args.max_actions,
            rtl_max_retries=args.rtl_max_retries,
            dc_max_retries=args.dc_max_retries,
            mcts_iterations=args.mcts_iterations,
            mcts_max_depth=args.mcts_max_depth,
            mcts_candidate_limit=args.mcts_candidate_limit,
            memory_baseline_db=baseline_db,
            memory_baseline_json=baseline_json,
            pre_dc_golden_rtl_path=golden_configs[pair.benchmark]["path"],
            pre_dc_golden_top=golden_configs[pair.benchmark]["top"],
            pre_dc_design_type=golden_configs[pair.benchmark]["design_type"],
            pre_dc_verification_timeout=args.verification_timeout,
            verification_mode=args.verification_mode,
            jg_max_retries=args.jg_max_retries,
        )
        collection["pairs"] = [item.to_dict() for item in plans]
        _write_json(plan_path, collection)

    print(f"Executed paired routes; no labels were generated: {plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
