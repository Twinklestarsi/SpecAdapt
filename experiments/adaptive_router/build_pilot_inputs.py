"""Build test-only adaptive-router inputs from the eight completed pilot cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from path_select.adaptive_predictor import DATASET_SCHEMA_VERSION
from path_select.feature_vectorizer import FeatureVectorizer
from path_select.latent_encoder import LatentEncoder, canonicalize_spec_text, sha256_text
from project_paths import PROJECT_ROOT


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


def _load_cases(pilot_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = pilot_root / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = list(manifest.get("cases") or [])
    if len(cases) != 8 or manifest.get("case_count") != 8:
        raise ValueError("This smoke builder requires exactly the eight pinned pilot cases")
    return manifest, cases


def _feature_path(pilot_root: Path, case_id: str) -> Path:
    return (
        pilot_root
        / "phase6_area_serial"
        / "pairs"
        / f"{case_id}__area__r000"
        / "frozen_feature_result.json"
    )


def build_dataset(
    *,
    pilot_root: Path,
    output_root: Path,
    model_alias: str,
    device: str,
) -> tuple[Path, Path]:
    pilot_manifest, cases = _load_cases(pilot_root)
    vectorizer = FeatureVectorizer()
    encoder = LatentEncoder(
        model_alias,
        output_dim=256,
        max_seq_length=8192,
        device=device,
        embedding_cache=output_root / "embedding_cache",
    )

    explicit_rows: list[np.ndarray] = []
    latent_rows: list[np.ndarray] = []
    spec_ids: list[str] = []
    family_ids: list[str] = []
    objectives: list[str] = []
    records: list[dict[str, Any]] = []
    feature_names: tuple[str, ...] | None = None

    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        spec_path = pilot_root / str(case["spec_path"])
        spec_text = spec_path.read_text(encoding="utf-8")
        canonical_sha = sha256_text(canonicalize_spec_text(spec_text))
        source_feature_path = _feature_path(pilot_root, case_id)
        if not source_feature_path.is_file():
            raise FileNotFoundError(
                f"Frozen phase-6 FeatureResult is missing for {case_id}: "
                f"{source_feature_path}"
            )
        feature_payload = json.loads(source_feature_path.read_text(encoding="utf-8"))
        if str(feature_payload.get("benchmark")) != case_id:
            raise ValueError(f"FeatureResult benchmark mismatch for {case_id}")
        explicit = vectorizer.transform_one(feature_payload, objective="AREA")
        if feature_names is None:
            feature_names = explicit.feature_names
        elif explicit.feature_names != feature_names:
            raise RuntimeError("Explicit feature schema changed between pilot cases")
        latent = encoder.encode_one(case_id, spec_text)

        explicit_rows.append(explicit.vector)
        latent_rows.append(latent.vector)
        spec_ids.append(case_id)
        family_ids.append(str(case.get("family_id") or case_id))
        objectives.append("AREA")
        records.append(
            {
                "index": index - 1,
                "spec_id": case_id,
                "family_id": family_ids[-1],
                "spec_path": str(spec_path.resolve()),
                "spec_sha256_canonical": canonical_sha,
                "manifest_spec_sha256_raw": case.get("spec_sha256", ""),
                "frozen_feature_path": str(source_feature_path.resolve()),
                "frozen_feature_sha256": _file_sha256(source_feature_path),
                "explicit_vector_sha256": explicit.metadata["vector_sha256"],
                "latent_vector_sha256": latent.metadata["vector_sha256"],
                "latent_cache_hit": latent.cache_hit,
                "provisional_phase6_status_excluded_from_training": True,
            }
        )
        print(
            f"[{index}/8] {case_id}: explicit={explicit.vector.shape[0]} "
            f"latent={latent.vector.shape[0]} cache_hit={latent.cache_hit}",
            flush=True,
        )

    explicit_matrix = np.stack(explicit_rows).astype(np.float32)
    latent_matrix = np.stack(latent_rows).astype(np.float32)
    labels = np.full(len(spec_ids), -1, dtype=np.int64)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"pilot_router_inputs_qwen3_{model_alias}"
    npz_path = output_root / f"{stem}.npz"
    manifest_path = output_root / f"{stem}.json"
    np.savez_compressed(
        npz_path,
        explicit=explicit_matrix,
        latent=latent_matrix,
        labels=labels,
        spec_ids=np.asarray(spec_ids, dtype=str),
        family_ids=np.asarray(family_ids, dtype=str),
        objectives=np.asarray(objectives, dtype=str),
    )
    output_manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_role": "integration_test_only",
        "training_allowed": False,
        "training_block_reason": (
            "Only eight pilot cases; provisional tie/unsolved/flow_error statuses are "
            "not binary training labels"
        ),
        "sample_count": len(spec_ids),
        "explicit_dim": int(explicit_matrix.shape[1]),
        "latent_dim": int(latent_matrix.shape[1]),
        "combined_dim": int(explicit_matrix.shape[1] + latent_matrix.shape[1]),
        "label_encoding": {"-1": "unlabeled", "0": "rtl_direct", "1": "c_first"},
        "labels_present": False,
        "objective": "AREA",
        "explicit_feature_schema": vectorizer.schema_version,
        "explicit_feature_names": list(feature_names or ()),
        "latent_protocol": encoder.protocol,
        "latent_protocol_sha256": encoder.protocol_sha256,
        "execution_policy": {
            "mode": "serial",
            "batch_size": 1,
            "max_concurrency": 1,
            "ai_calls": 0,
            "dc_calls": 0,
        },
        "source_pilot_manifest": str((pilot_root / "pilot_manifest.json").resolve()),
        "source_pilot_manifest_sha256": _file_sha256(
            pilot_root / "pilot_manifest.json"
        ),
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": _file_sha256(npz_path),
        "records": records,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, output_manifest)
    return npz_path, manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unlabeled, test-only inputs from the eight pilot specs."
    )
    parser.add_argument(
        "--pilot-root", default=str(PROJECT_ROOT / "pilot_test")
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "pilot_test" / "adaptive_router_smoke"),
    )
    parser.add_argument(
        "--model", choices=("0.6b", "8b", "both"), default="both"
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    aliases = ("0.6b", "8b") if args.model == "both" else (args.model,)
    for alias in aliases:
        npz_path, manifest_path = build_dataset(
            pilot_root=Path(args.pilot_root).resolve(),
            output_root=Path(args.output_root).resolve(),
            model_alias=alias,
            device=args.device,
        )
        print(f"wrote {npz_path}")
        print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
