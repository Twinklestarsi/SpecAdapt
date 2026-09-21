"""Central discovery of the active, versioned RAG artifacts.

The rebuild entry point publishes ``latest_rag.json`` only after every
versioned artifact has been written successfully.  Runtime consumers use this
module instead of embedding a particular artifact version in their own code.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from rag_retrieve.path_utils import project_root, resolve_stored_path

FALLBACK_RAG_VERSION = "v3"
ACTIVE_RAG_MANIFEST = Path("rag_retrieve/indices/latest_rag.json")
RAG_VERSION_ENV = "VIVADO_RAG_VERSION"

_VERSION_RE = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RAGArtifacts:
    """Paths belonging to one complete RAG release."""

    version: str
    knowledge_base: Path
    knowledge_summary: Path
    joined_index: Path
    region_index: Path
    transform_stats: Path
    build_report: Path
    active_manifest: Path

    def versioned_paths(self) -> Dict[str, Path]:
        return {
            "knowledge_base": self.knowledge_base,
            "knowledge_summary": self.knowledge_summary,
            "joined_index": self.joined_index,
            "region_index": self.region_index,
            "transform_stats": self.transform_stats,
            "build_report": self.build_report,
        }


def _validate_version(version: str) -> str:
    normalized = str(version).strip()
    if not _VERSION_RE.fullmatch(normalized):
        raise ValueError(f"unsafe RAG version name: {version!r}")
    return normalized


def versioned_rag_artifacts(
    root: str | Path | None = None,
    version: str = FALLBACK_RAG_VERSION,
) -> RAGArtifacts:
    """Construct paths for a specific version without reading the manifest."""

    base = project_root(root)
    selected = _validate_version(version)
    return RAGArtifacts(
        version=selected,
        knowledge_base=base / "LLM_DC_LOG" / f"rag_knowledge_base_{selected}.csv",
        knowledge_summary=base / "LLM_DC_LOG" / f"rag_knowledge_base_{selected}.summary.json",
        joined_index=base / "rag_retrieve" / "indices" / f"module4_cdfg_rag_index_{selected}.json",
        region_index=base / "rag_retrieve" / "indices" / f"module4_historical_region_index_{selected}.json",
        transform_stats=base / "rag_retrieve" / "indices" / f"module4_transform_stats_{selected}.json",
        build_report=base / "rag_retrieve" / "indices" / f"rag_rebuild_report_{selected}.json",
        active_manifest=base / ACTIVE_RAG_MANIFEST,
    )


def _artifacts_from_manifest(base: Path, payload: Dict[str, Any]) -> RAGArtifacts:
    version = _validate_version(str(payload["version"]))
    paths = payload["artifacts"]
    required = {
        "knowledge_base",
        "knowledge_summary",
        "joined_index",
        "region_index",
        "transform_stats",
        "build_report",
    }
    missing = required - set(paths)
    if missing:
        raise ValueError(f"active RAG manifest is missing: {sorted(missing)}")
    return RAGArtifacts(
        version=version,
        knowledge_base=resolve_stored_path(paths["knowledge_base"], base),
        knowledge_summary=resolve_stored_path(paths["knowledge_summary"], base),
        joined_index=resolve_stored_path(paths["joined_index"], base),
        region_index=resolve_stored_path(paths["region_index"], base),
        transform_stats=resolve_stored_path(paths["transform_stats"], base),
        build_report=resolve_stored_path(paths["build_report"], base),
        active_manifest=base / ACTIVE_RAG_MANIFEST,
    )


def get_active_rag_artifacts(root: str | Path | None = None) -> RAGArtifacts:
    """Return the explicitly selected or atomically activated RAG release.

    ``VIVADO_RAG_VERSION`` is an explicit per-process override.  Otherwise the
    active manifest is used, with v3 as a compatibility fallback for checkouts
    created before the manifest existed.
    """

    base = project_root(root)
    forced_version = os.environ.get(RAG_VERSION_ENV)
    if forced_version:
        return versioned_rag_artifacts(base, forced_version)

    manifest = base / ACTIVE_RAG_MANIFEST
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            return _artifacts_from_manifest(base, payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            # Do not make every downstream module unimportable because a
            # manually edited/stale pointer is malformed.
            pass
    return versioned_rag_artifacts(base, FALLBACK_RAG_VERSION)


ACTIVE_RAG = get_active_rag_artifacts()
DEFAULT_KNOWLEDGE_BASE = ACTIVE_RAG.knowledge_base
DEFAULT_JOINED_INDEX = ACTIVE_RAG.joined_index
DEFAULT_HISTORICAL_REGION_INDEX = ACTIVE_RAG.region_index
DEFAULT_TRANSFORM_STATS = ACTIVE_RAG.transform_stats

# Keep the blacklist on the same release boundary as the active RAG.  Older
# checkouts only have the unversioned artifact, so retain that as a fallback.
_LEGACY_TRANSFORM_BLACKLIST = (
    project_root() / "rag_retrieve" / "indices" / "module4_transform_blacklist.json"
)
_VERSIONED_TRANSFORM_BLACKLIST = (
    project_root() / "rag_retrieve" / "indices"
    / f"module4_transform_blacklist_{ACTIVE_RAG.version}.json"
)
DEFAULT_TRANSFORM_BLACKLIST = (
    _VERSIONED_TRANSFORM_BLACKLIST
    if _VERSIONED_TRANSFORM_BLACKLIST.is_file()
    else _LEGACY_TRANSFORM_BLACKLIST
)
