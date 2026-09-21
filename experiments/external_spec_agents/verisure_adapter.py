#!/usr/bin/env python3
"""Boundary adapter for the upstream Veri-Sure TopAgent.

It runs the project's own Architect/Generator/Verifier/Debugger loop in a
separate process.  The case JSON supplies the natural-language specification
and (for evaluation only) the public C-first testbench; the golden RTL is not
passed to Veri-Sure or its model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "external_agents" / "verisure"
PYTHON = SRC / ".venv" / "bin" / "python"
TOOLS = SRC / ".tools" / "usr" / "bin"


class AdapterError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"invalid case/config JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"JSON root must be an object: {path.name}")
    return value


def resolve(path_value: Any, case_file: Path, label: str, required: bool = True) -> Optional[Path]:
    if path_value is None or str(path_value).strip() == "":
        if required:
            raise AdapterError(f"case is missing {label}")
        return None
    p = Path(str(path_value)).expanduser()
    options = [p] if p.is_absolute() else [case_file.parent / p, ROOT / p, Path.cwd() / p]
    for candidate in options:
        if candidate.is_file():
            return candidate.resolve()
    if required:
        raise AdapterError(f"{label} does not resolve to a file")
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_top(source: Path, expected_top: str, output: Path) -> Path:
    text = source.read_text(encoding="utf-8", errors="replace")
    modules = re.findall(r"^\s*module\s+([A-Za-z_]\w*)\b", text, re.MULTILINE)
    if not modules:
        raise AdapterError("generated RTL contains no module declaration")
    if expected_top in modules:
        output.write_text(text, encoding="utf-8")
        return output
    # Veri-Sure normally follows the module name from the spec.  If a model
    # falls back to TopModule, rename the sole generated top while preserving
    # internal helper modules and named endmodule labels.
    source_top = modules[0]
    if len(modules) > 1 and source_top != "TopModule" and expected_top not in modules:
        raise AdapterError(f"generated RTL has no expected top {expected_top}")
    decl = re.compile(rf"(?m)^(\s*module\s+){re.escape(source_top)}\b")
    normalized = decl.sub(rf"\1{expected_top}", text, count=1)
    normalized = re.sub(
        rf"(\bendmodule\s*:\s*){re.escape(source_top)}\b",
        rf"\1{expected_top}", normalized,
    )
    output.write_text(normalized, encoding="utf-8")
    return output


def load_env() -> dict[str, str]:
    # Avoid importing python-dotenv in the caller environment.  The isolated
    # Veri-Sure venv has it, but this adapter itself also works with stdlib
    # parsing for the simple KEY=value entries used by this project.
    try:
        from dotenv import dotenv_values

        values = dotenv_values(ROOT / ".env")
        return {str(k): str(v) for k, v in values.items() if k and v is not None}
    except Exception:  # noqa: BLE001
        values: dict[str, str] = {}
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values


def find_rtl(run_root: Path) -> Optional[Path]:
    candidates = [p for p in run_root.rglob("rtl.sv") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def find_token_usage(run_root: Path, log_path: Path) -> dict[str, Any]:
    """Read the provider-reported usage written by the patched upstream CLI."""
    usage_files = sorted(run_root.rglob("token_usage.json")) if run_root.exists() else []
    if usage_files:
        try:
            value = json.loads(usage_files[-1].read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return {
                    "prompt_tokens": int(value.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(value.get("completion_tokens", 0) or 0),
                    "total_tokens": int(value.get("total_tokens", 0) or 0),
                    "request_count": value.get("request_count"),
                    "source": value.get("source", "verisure_token_usage.json"),
                    "path": str(usage_files[-1]),
                }
        except (OSError, ValueError, TypeError):
            pass
    # Keep an explicit unknown state; do not estimate tokens from characters.
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "request_count": None,
        "source": "unavailable",
        "path": str(log_path),
    }


def run_case(case_file: Path, output_dir: Path, config_file: Optional[Path]) -> dict[str, Any]:
    case = read_json(case_file)
    config = read_json(config_file) if config_file else {}
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = resolve(case.get("spec_path"), case_file, "spec_path")
    golden = resolve(case.get("golden_path"), case_file, "golden_path")
    tb = resolve(case.get("testbench_path"), case_file, "testbench_path", required=False)
    assert spec is not None and golden is not None
    expected_top = str(case.get("top_module") or "TopModule")

    env = os.environ.copy()
    env.update(load_env())
    env["PATH"] = os.pathsep.join([str(TOOLS), str(ROOT / ".conda-env" / "bin"), env.get("PATH", "")])
    env["PYTHONNOUSERSITE"] = "1"
    env["VERISURE_SIMULATOR"] = "iverilog"
    env["VERISURE_ADAPTER_OUTPUT"] = str(output_dir)

    model = str(config.get("model") or env.get("OPENAI_MODEL") or "deepseek-v4-flash")
    base_url = str(config.get("base_url") or env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE_URL") or "")
    if not env.get("OPENAI_API_KEY"):
        raise AdapterError("OPENAI_API_KEY is not configured in the project environment")
    if not base_url:
        raise AdapterError("OPENAI_BASE_URL is not configured in the project environment")
    env["OPENAI_MODEL"] = model
    env["OPENAI_BASE_URL"] = base_url

    prompt = spec.read_text(encoding="utf-8")
    runs_root = output_dir / "upstream_runs"
    command = [
        str(PYTHON), "-m", "eda_agent", "run",
        "--prompt", prompt,
        "--model", model,
        "--base-url", base_url,
        "--runs-root", str(runs_root),
        "--temperature", str(config.get("temperature", 0.0)),
        "--top-p", str(config.get("top_p", 1.0)),
        "--sim-max-retry", str(config.get("sim_max_retry", 2)),
        "--debug-max-trials", str(config.get("debug_max_trials", 4)),
    ]
    # ``None``/missing deliberately means no client-side completion cap.  The
    # upstream CLI also defaults to None, so omitting this option leaves the
    # provider's own context/output policy in control.
    max_completion_tokens = config.get("max_completion_tokens")
    if max_completion_tokens not in (None, "", "unset", "none", "null"):
        command.extend(["--max-completion-tokens", str(max_completion_tokens)])
    # The nvdla22 no-JG runner requests generation-only mode explicitly.  This
    # keeps Veri-Sure's own simulation/debug loop from being mistaken for the
    # requested external PPA flow; the adapter-level Icarus gate and DC runs
    # remain in force in the caller.
    if bool(case.get("ablation", False)) or bool(config.get("ablation", False)):
        command.append("--ablation")
    if tb is not None:
        command.extend(["--golden-tb", str(tb)])

    log_path = output_dir / "verisure_run.log"
    timeout = float(config.get("process_timeout", 1800))
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=str(SRC), env=env, stdout=log, stderr=subprocess.STDOUT,
                text=True, timeout=timeout, check=False,
            )
    except subprocess.TimeoutExpired:
        result = {"status": "timeout", "rtl_path": None, "error": "Veri-Sure process timeout", "usage": {}, "agent": "xyjoey/Veri-Sure"}
        (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    raw_rtl = find_rtl(runs_root)
    token_usage = find_token_usage(runs_root, log_path)
    normalized = None
    error = None
    if raw_rtl is not None:
        try:
            normalized = normalize_top(raw_rtl, expected_top, output_dir / "rtl_normalized.sv")
        except AdapterError as exc:
            error = str(exc)
    if normalized is not None:
        status = "candidate"
    else:
        status = "error"
        error = error or f"Veri-Sure exited with code {completed.returncode} without rtl.sv"
    result = {
        "status": status,
        "rtl_path": str(normalized) if normalized else None,
        "error": error,
        "usage": token_usage,
        "agent": "xyjoey/Veri-Sure",
        "exit_code": completed.returncode,
        "log_path": str(log_path),
        "case_id": case.get("id", case_file.stem),
        "source_sha256": {"spec": sha256(spec), "golden": sha256(golden), **({"testbench": sha256(tb)} if tb else {})},
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config-json", type=Path)
    args = ap.parse_args(argv)
    try:
        result = run_case(args.case_json.resolve(), args.output_dir, args.config_json.resolve() if args.config_json else None)
    except AdapterError as exc:
        result = {"status": "error", "rtl_path": None, "error": str(exc), "usage": {}, "agent": "xyjoey/Veri-Sure"}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "candidate" else 1


if __name__ == "__main__":
    raise SystemExit(main())
