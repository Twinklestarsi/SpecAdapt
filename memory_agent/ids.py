"""Stable identifiers used by the Memory Agent."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any


_NAMESPACE = uuid.UUID("c58dcf5d-b9d1-49f9-9347-b1f9271cce8f")


def stable_id(kind: str, *parts: Any) -> str:
    """Return a deterministic, readable UUID5 identifier."""
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    value = uuid.uuid5(_NAMESPACE, f"{kind}:{payload}")
    return f"{kind}_{value.hex}"


def new_id(kind: str) -> str:
    return f"{kind}_{uuid.uuid4().hex}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

