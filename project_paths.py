"""Project-wide portable path helpers.

The repository used to be tied to one machine-specific absolute directory.
All local paths should now be derived from this file's location or overridden
with ``VIVADO_PROJECT_ROOT`` when the source tree is mounted elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT_ENV = "VIVADO_PROJECT_ROOT"


def get_project_root() -> Path:
    configured = os.environ.get(PROJECT_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent


PROJECT_ROOT = get_project_root()


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def relocate_project_path(value: str | Path) -> Path:
    """Resolve an artifact path written under an older ``.../Vivado`` root."""

    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    positions = [index for index, part in enumerate(path.parts) if part == "Vivado"]
    if positions:
        candidate = PROJECT_ROOT.joinpath(*path.parts[positions[-1] + 1 :])
        if candidate.exists():
            return candidate.resolve()
    return path
