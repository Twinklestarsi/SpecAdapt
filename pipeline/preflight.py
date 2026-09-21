"""Report dependencies for the two AI RTL generation routes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv

from project_paths import PROJECT_ROOT
from toolchain import local_tool_status


def build_preflight(env_path: str | Path = PROJECT_ROOT / ".env") -> dict[str, Any]:
    load_dotenv(env_path, override=False)
    local = local_tool_status()
    dc_required = ("DC_REMOTE_USER", "DC_REMOTE_HOST", "DC_REMOTE_BASE")
    jg_required = (
        "JG_REMOTE_USER",
        "JG_REMOTE_HOST",
        "JG_REMOTE_BASE",
        "JG_ENV_SCRIPT",
        "JG_BIN",
    )
    dc_configured = all(os.environ.get(name, "").strip() for name in dc_required)
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    parsed_base_url = urlsplit(base_url)
    base_url_valid = bool(
        parsed_base_url.scheme in {"http", "https"} and parsed_base_url.netloc
    )
    llm_configured = bool(
        os.environ.get("OPENAI_API_KEY", "").strip()
        and base_url_valid
        and os.environ.get("OPENAI_MODEL", "").strip()
    )
    jg_enabled = os.environ.get("JG_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    jg_variable_status = {
        name: bool(os.environ.get(name, "").strip()) for name in jg_required
    }
    jg_missing = [name for name, present in jg_variable_status.items() if not present]
    return {
        "schema_version": "pipeline_preflight_v1",
        "project_root": str(PROJECT_ROOT),
        "route_backends": {
            "c_first": "ai_c_to_rtl",
            "rtl_direct": "ai_spec_to_rtl",
        },
        "vitis_required": False,
        "llm": {
            "configured": llm_configured,
            "base_url_valid": base_url_valid,
        },
        "local_tools": local,
        "dc": {
            "configured": dc_configured,
            "required_variables": list(dc_required),
        },
        "jaspergold": {
            "enabled": jg_enabled,
            # This is a local configuration report only.  In particular,
            # preflight never opens SSH or probes the remote JG executable.
            "configured": jg_enabled and not jg_missing,
            "required_variables": list(jg_required),
            "variable_status": jg_variable_status,
            "missing_variables": jg_missing,
        },
        "ready_for_c_first_planning": bool(
            local["clang"]["available"]
            and local["llvm_opt"]["available"]
            and local["dot"]["available"]
        ),
        "ready_for_c_first_execution": bool(
            local["clang"]["available"]
            and local["llvm_opt"]["available"]
            and local["dot"]["available"]
            and local["iverilog"]["available"]
            and llm_configured
            and dc_configured
        ),
        "ready_for_rtl_direct_execution": bool(
            local["iverilog"]["available"]
            and llm_configured
            and dc_configured
        ),
    }


def main() -> int:
    payload = build_preflight()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ready_for_c_first_execution"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
