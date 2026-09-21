"""Resolve local tool executables without relying on shell activation alone."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional


class ToolNotFoundError(RuntimeError):
    """Raised when a required local executable cannot be resolved."""


def find_tool(env_var: str, candidates: Iterable[str]) -> Optional[str]:
    """Return an executable path from an override, the active prefix, or PATH."""

    configured = os.environ.get(env_var, "").strip()
    if configured:
        expanded = Path(configured).expanduser()
        if expanded.is_file() and os.access(expanded, os.X_OK):
            return str(expanded.resolve())
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        return None

    prefix_bin = Path(sys.prefix) / "bin"
    for name in candidates:
        prefixed = prefix_bin / name
        if prefixed.is_file() and os.access(prefixed, os.X_OK):
            return str(prefixed.resolve())

    for name in candidates:
        resolved = shutil.which(name)
        if resolved:
            return resolved
    return None


def require_tool(env_var: str, *candidates: str) -> str:
    """Resolve a required executable and give a configuration-oriented error."""

    resolved = find_tool(env_var, candidates)
    if resolved:
        return resolved
    names = ", ".join(candidates)
    raise ToolNotFoundError(
        f"Required tool not found ({names}); set {env_var} to an executable path"
    )


TOOL_SPECS = {
    "clang": ("CLANG_PATH", ("clang-16", "clang")),
    "llvm_opt": ("LLVM_OPT_PATH", ("opt-16", "opt")),
    "iverilog": ("IVERILOG_PATH", ("iverilog",)),
    "yosys": ("YOSYS_PATH", ("yosys",)),
    "dot": ("DOT_PATH", ("dot",)),
}


def local_tool_status() -> dict[str, dict[str, object]]:
    """Return non-sensitive local tool availability for preflight/manifests."""

    status: dict[str, dict[str, object]] = {}
    for label, (env_var, candidates) in TOOL_SPECS.items():
        path = find_tool(env_var, candidates)
        status[label] = {
            "available": path is not None,
            "path": path or "",
            "override": env_var,
        }
    return status
