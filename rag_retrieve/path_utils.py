"""Portable project-path helpers for persisted RAG artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def project_root(start: str | Path | None = None) -> Path:
    configured = os.environ.get("VIVADO_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if start is not None:
        path = Path(start).expanduser().resolve()
        return path if path.is_dir() else path.parent
    return Path(__file__).resolve().parent.parent


def rebase_legacy_path(value: str | Path, root: str | Path | None = None) -> Path:
    """Resolve current relative paths and old ``.../Vivado/...`` paths."""
    base = project_root(root)
    raw = str(value or "")
    path = Path(raw).expanduser()
    if not raw:
        return path
    if not path.is_absolute():
        return (base / path).resolve()
    if path.exists():
        return path.resolve()
    marker = "/Vivado/"
    if marker in raw:
        return (base / raw.split(marker, 1)[1]).resolve()
    return path


def stored_path(value: str | Path, root: str | Path | None = None) -> str:
    """Serialize paths under the project as POSIX relative paths."""
    if not value:
        return ""
    base = project_root(root)
    path = rebase_legacy_path(value, base)
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def resolve_stored_path(value: str | Path, root: str | Path | None = None) -> Path:
    return rebase_legacy_path(value, project_root(root))
