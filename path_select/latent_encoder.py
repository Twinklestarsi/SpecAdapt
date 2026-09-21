"""Pinned, offline latent encoder with deterministic per-spec caching."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from project_paths import PROJECT_ROOT


LATENT_SCHEMA_VERSION = "adaptive_router_latent_v1"
DEFAULT_TASK_PROMPT = (
    "Represent this hardware specification for predicting whether C-first or "
    "direct RTL generation gives better correctness and PPA: "
)
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "0.6b": {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "native_dim": 1024,
    },
    "8b": {
        "model_id": "Qwen/Qwen3-Embedding-8B",
        "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "native_dim": 4096,
    },
}


def canonicalize_spec_text(text: str) -> str:
    """Normalize line endings and trailing whitespace without changing wording."""

    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip() + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_vector(vector: np.ndarray) -> str:
    stable = np.asarray(vector, dtype="<f4")
    return hashlib.sha256(stable.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class LatentRecord:
    spec_id: str
    vector: np.ndarray
    metadata: dict[str, Any]
    cache_hit: bool


class LatentEncoder:
    """Encode specs with one pinned local model; network access is never used."""

    def __init__(
        self,
        model_alias: str = "8b",
        *,
        output_dim: int = 256,
        max_seq_length: int = 8192,
        device: str = "cuda",
        task_prompt: str = DEFAULT_TASK_PROMPT,
        hf_hub_cache: str | Path | None = None,
        embedding_cache: str | Path | None = None,
    ) -> None:
        if model_alias not in MODEL_SPECS:
            raise ValueError(
                f"Unknown model alias {model_alias!r}; choose from {sorted(MODEL_SPECS)}"
            )
        self.model_alias = model_alias
        self.model_spec = dict(MODEL_SPECS[model_alias])
        self.output_dim = int(output_dim)
        self.max_seq_length = int(max_seq_length)
        self.device = str(device)
        self.task_prompt = str(task_prompt)
        if not 1 <= self.output_dim <= int(self.model_spec["native_dim"]):
            raise ValueError(
                f"output_dim must be in [1, {self.model_spec['native_dim']}]"
            )
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")

        default_hub = PROJECT_ROOT / "model_cache" / "huggingface" / "hub"
        self.hf_hub_cache = Path(hf_hub_cache or default_hub).resolve()
        default_vectors = PROJECT_ROOT / "adaptive_router_artifacts" / "embeddings"
        self.embedding_cache = Path(embedding_cache or default_vectors).resolve()
        self._model: Any = None

    @property
    def snapshot_path(self) -> Path:
        repo_dir = "models--" + str(self.model_spec["model_id"]).replace("/", "--")
        return (
            self.hf_hub_cache
            / repo_dir
            / "snapshots"
            / str(self.model_spec["revision"])
        )

    @property
    def protocol(self) -> dict[str, Any]:
        return {
            "schema_version": LATENT_SCHEMA_VERSION,
            "model_alias": self.model_alias,
            "model_id": self.model_spec["model_id"],
            "revision": self.model_spec["revision"],
            "native_dim": self.model_spec["native_dim"],
            "output_dim": self.output_dim,
            "max_seq_length": self.max_seq_length,
            "task_prompt": self.task_prompt,
            "task_prompt_sha256": sha256_text(self.task_prompt),
            "pooling": "last_non_padding_token",
            "dimensionality_reduction": "matryoshka_prefix",
            "normalization": "l2_after_truncation",
            "dtype": "bfloat16_on_cuda_float32_output",
        }

    @property
    def protocol_sha256(self) -> str:
        payload = json.dumps(
            self.protocol, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return sha256_text(payload)

    def _cache_paths(self, spec_sha256: str) -> tuple[Path, Path]:
        stem = (
            f"{spec_sha256}__qwen3_{self.model_alias}__d{self.output_dim}__"
            f"{self.protocol_sha256[:12]}"
        )
        model_dir = self.embedding_cache / self.model_alias
        return model_dir / f"{stem}.npz", model_dir / f"{stem}.json"

    def _load_cached(self, spec_id: str, spec_sha256: str) -> LatentRecord | None:
        vector_path, metadata_path = self._cache_paths(spec_sha256)
        if not vector_path.is_file() or not metadata_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("protocol_sha256") != self.protocol_sha256:
            return None
        with np.load(vector_path, allow_pickle=False) as payload:
            vector = np.asarray(payload["embedding"], dtype=np.float32)
        if vector.shape != (self.output_dim,):
            raise ValueError(f"Invalid cached embedding shape in {vector_path}")
        if sha256_vector(vector) != metadata.get("vector_sha256"):
            raise ValueError(f"Cached embedding checksum mismatch: {vector_path}")
        return LatentRecord(spec_id, vector, metadata, True)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.snapshot_path.is_dir():
            raise FileNotFoundError(
                f"Pinned model snapshot is missing: {self.snapshot_path}"
            )

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "LatentEncoder requires the project .conda-env with torch and "
                "sentence-transformers installed"
            ) from exc

        use_cuda = self.device.startswith("cuda")
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        model_kwargs = {"torch_dtype": torch.bfloat16} if use_cuda else {}
        self._model = SentenceTransformer(
            str(self.snapshot_path),
            device=self.device,
            model_kwargs=model_kwargs,
        )
        self._model.max_seq_length = self.max_seq_length
        self._model.eval()
        return self._model

    def _write_cache(
        self,
        spec_id: str,
        canonical_text: str,
        spec_sha256: str,
        vector: np.ndarray,
    ) -> LatentRecord:
        vector_path, metadata_path = self._cache_paths(spec_sha256)
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        stable_vector = np.asarray(vector, dtype=np.float32)
        metadata = {
            **self.protocol,
            "protocol_sha256": self.protocol_sha256,
            "spec_id": spec_id,
            "spec_sha256": spec_sha256,
            "spec_char_count": len(canonical_text),
            "vector_sha256": sha256_vector(stable_vector),
            "vector_path": str(vector_path),
            "snapshot_path": str(self.snapshot_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with tempfile.NamedTemporaryFile(
            suffix=".npz", dir=vector_path.parent, delete=False
        ) as handle:
            temp_vector = Path(handle.name)
        try:
            np.savez_compressed(temp_vector, embedding=stable_vector)
            os.replace(temp_vector, vector_path)
        finally:
            temp_vector.unlink(missing_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            dir=metadata_path.parent,
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            temp_metadata = Path(handle.name)
        try:
            os.replace(temp_metadata, metadata_path)
        finally:
            temp_metadata.unlink(missing_ok=True)
        return LatentRecord(spec_id, stable_vector, metadata, False)

    def encode_one(self, spec_id: str, spec_text: str) -> LatentRecord:
        canonical_text = canonicalize_spec_text(spec_text)
        spec_sha256 = sha256_text(canonical_text)
        cached = self._load_cached(spec_id, spec_sha256)
        if cached is not None:
            return cached

        model = self._load_model()
        native = model.encode(
            [canonical_text],
            prompt=self.task_prompt,
            batch_size=1,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        expected = (1, int(self.model_spec["native_dim"]))
        if native.shape != expected:
            raise RuntimeError(f"Unexpected embedding shape {native.shape}, expected {expected}")
        vector = np.asarray(native[0, : self.output_dim], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            raise RuntimeError("Embedding norm is not finite and positive")
        vector = vector / norm
        if not np.isfinite(vector).all():
            raise RuntimeError("Embedding contains NaN or infinity")
        return self._write_cache(spec_id, canonical_text, spec_sha256, vector)

    def encode_many(self, specs: Iterable[tuple[str, str]]) -> list[LatentRecord]:
        """Encode strictly serially; each item is cached before the next starts."""

        return [self.encode_one(spec_id, text) for spec_id, text in specs]
