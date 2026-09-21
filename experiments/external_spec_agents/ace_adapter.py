#!/usr/bin/env python3
"""Run one project case through the upstream ACE-RTL VerilogEval entry point.

The project has a C-first case format, while ACE-RTL's native non-CVDP path
expects a VerilogEval directory containing ``*_prompt.txt``, ``*_test.sv`` and
``*_ref.sv``.  This adapter creates that private directory and then invokes
the upstream ``ace_agent_runner.py``.  In particular, it deliberately does
not implement another generation/debugging loop: ACE's ``AceRTLAgent`` keeps
ownership of its native FocusedDebugger/FreshStartCoordinator loop.

The golden RTL is used only to make the private ``RefModule`` compilation
file.  It is never appended to the prompt or passed as an ACE command-line
argument.  The adapter itself does not call an LLM, JasperGold, Design
Compiler, or any other external evaluator until the upstream process is
explicitly launched by the caller.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
ACE_ROOT = REPO_ROOT / "external_agents" / "ace_rtl"
ACE_ENTRY = ACE_ROOT / "skills" / "ace-rtl" / "scripts" / "ace_agent_runner.py"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_BACKOFF = 5.0
DEFAULT_CONTEXT_LIMIT = 300000
DEFAULT_TEMPERATURE = 0.2


class AdapterError(RuntimeError):
    """An input or local setup error that should be reported without a trace."""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AdapterError(f"cannot read JSON file: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise AdapterError(f"invalid JSON file: {path.name} ({exc.msg})") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"JSON root must be an object: {path.name}")
    return value


def _resolve_input(raw: Any, case_file: Path, *, label: str, required: bool = True) -> Optional[Path]:
    """Resolve a case path without assuming the caller's current directory."""

    if raw is None or str(raw).strip() == "":
        if required:
            raise AdapterError(f"case is missing {label}")
        return None

    candidate = Path(str(raw)).expanduser()
    options = [candidate] if candidate.is_absolute() else [
        case_file.parent / candidate,
        REPO_ROOT / candidate,
        Path.cwd() / candidate,
    ]
    for option in options:
        if option.is_file():
            return option.resolve()
    if required:
        raise AdapterError(f"{label} does not resolve to a file")
    return None


def _first_case_value(case: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = case.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _safe_problem_id(case: Dict[str, Any], case_file: Path) -> str:
    raw = _first_case_value(case, ("id", "case_id", "benchmark", "name"))
    if raw is None:
        raw = case_file.stem
    problem_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("._-")
    if not problem_id:
        raise AdapterError("case id is empty after sanitization")
    return problem_id


def _find_testbench(case: Dict[str, Any], case_file: Path, prompt_path: Path, golden_path: Path) -> Path:
    raw = _first_case_value(case, ("testbench_path", "tb_path", "test_path"))
    if raw is not None:
        resolved = _resolve_input(raw, case_file, label="testbench_path")
        assert resolved is not None
        return resolved

    # C-first manifests keep the testbench alongside spec.txt/golden.v.  The
    # fallback makes the adapter usable with a compact case JSON containing
    # only the required prompt/golden fields.
    for sibling in (
        prompt_path.parent / "tb.sv",
        prompt_path.parent / "tb.v",
        prompt_path.parent / "testbench.sv",
        prompt_path.parent / "testbench.v",
        golden_path.parent / "tb.sv",
        golden_path.parent / "tb.v",
        golden_path.parent / "testbench.sv",
        golden_path.parent / "testbench.v",
    ):
        if sibling.is_file():
            return sibling.resolve()
    raise AdapterError("case is missing testbench_path and no sibling testbench was found")


def _rename_top_module(golden_text: str, source_top: str) -> str:
    """Rename only the golden top declaration for ACE's private RefModule."""

    if source_top == "RefModule":
        raise AdapterError("golden_top must not already be RefModule")
    declaration = re.compile(
        rf"(?m)^(\s*module\s+){re.escape(source_top)}\b"
    )
    matches = list(declaration.finditer(golden_text))
    if len(matches) != 1:
        raise AdapterError(
            f"expected exactly one top declaration for {source_top}, found {len(matches)}"
        )
    renamed = declaration.sub(r"\1RefModule", golden_text, count=1)
    # Preserve named endmodule syntax when a golden file uses it.  This is
    # intentionally limited to the matching label, not arbitrary references.
    renamed = re.sub(
        rf"(\bendmodule\s*:\s*){re.escape(source_top)}\b",
        r"\1RefModule",
        renamed,
    )
    return renamed


def _normalize_candidate_top(source: Path, expected_top: str, output: Path) -> Path:
    """Normalize ACE's generated top to the benchmark's original top name."""
    text = source.read_text(encoding="utf-8", errors="replace")
    modules = re.findall(r"^\s*module\s+([A-Za-z_]\w*)\b", text, re.MULTILINE)
    if not modules:
        raise AdapterError("generated RTL contains no module declaration")
    if expected_top in modules:
        output.write_text(text, encoding="utf-8")
        return output.resolve()
    source_top = modules[0]
    decl = re.compile(rf"(?m)^(\s*module\s+){re.escape(source_top)}\b")
    normalized = decl.sub(rf"\1{expected_top}", text, count=1)
    normalized = re.sub(
        rf"(\bendmodule\s*:\s*){re.escape(source_top)}\b",
        rf"\1{expected_top}", normalized,
    )
    output.write_text(normalized, encoding="utf-8")
    return output.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(config_file: Optional[Path]) -> Dict[str, Any]:
    if config_file is None:
        return {}
    return _read_json(config_file)


def _as_int(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"config field {key} must be an integer") from exc


def _as_float(config: Dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"config field {key} must be numeric") from exc


def _build_ace_command(
    problem_id: str,
    verilogeval_root: Path,
    ace_output: Path,
    config: Dict[str, Any],
) -> List[str]:
    """Build only upstream CLI flags; no custom repair loop is introduced."""

    model = str(config.get("model", DEFAULT_MODEL))
    max_iterations = _as_int(config, "max_iterations", DEFAULT_MAX_ITERATIONS)
    timeout = _as_int(config, "timeout", DEFAULT_TIMEOUT)
    max_retries = _as_int(config, "max_retries", DEFAULT_MAX_RETRIES)
    initial_backoff = _as_float(config, "initial_backoff", DEFAULT_INITIAL_BACKOFF)
    context_limit = _as_int(config, "llm_context_limit", DEFAULT_CONTEXT_LIMIT)
    temperature = _as_float(config, "ace_rtl_temperature", DEFAULT_TEMPERATURE)
    parallel = _as_int(config, "parallel_experiments", 1)

    command = [
        sys.executable,
        str(ACE_ENTRY),
        "--verilogeval-problem",
        problem_id,
        "--verilogeval-path",
        str(verilogeval_root),
        "--output-dir",
        str(ace_output),
        "--model",
        model,
        "--max-iterations",
        str(max_iterations),
        "--timeout",
        str(timeout),
        "--max-retries",
        str(max_retries),
        "--initial-backoff",
        str(initial_backoff),
        "--llm-context-limit",
        str(context_limit),
        "--ace_rtl-temperature",
        str(temperature),
        "--parallel-experiments",
        str(parallel),
    ]

    stages = config.get("use_ace_rtl_model", config.get("use_ace_rtl-model"))
    if stages is not None:
        if not isinstance(stages, (list, tuple)) or not all(str(item) in {"0", "1", "2"} for item in stages):
            raise AdapterError("config field use_ace_rtl_model must be a list of 0, 1, and 2")
        command.append("--use-ace_rtl-model")
        command.extend(str(item) for item in stages)

    if bool(config.get("debug", False)):
        command.append("--debug")
    if config.get("systematic_debugging", True) is False:
        command.append("--no-systematic-debugging")
    if bool(config.get("with_testbench", False)):
        command.append("--with-testbench")
    if config.get("save_intermediate", True) is False:
        command.append("--no-save-intermediate")
    return command


def _candidate_sort_key(path: Path) -> Tuple[int, float, str]:
    matches = re.findall(r"iteration_(\d+)", str(path))
    iteration = int(matches[-1]) if matches else -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (iteration, mtime, str(path))


def _find_generated_rtl(ace_output: Path) -> Optional[Path]:
    if not ace_output.exists():
        return None
    candidates: List[Path] = []
    preferred_names = {"TopModule.sv", "TopModule.v", "generated.sv", "generated.v"}
    for path in ace_output.rglob("*"):
        if not path.is_file() or "iteration_" not in str(path):
            continue
        if path.name in preferred_names:
            candidates.append(path)
    if not candidates:
        # Keep a conservative fallback for an upstream filename change, but
        # never select the private reference/testbench files from the adapter
        # workspace because the search is confined to ace_output.
        candidates = [
            path for path in ace_output.rglob("*")
            if path.is_file()
            and "iteration_" in str(path)
            and path.suffix.lower() in {".sv", ".v"}
            and "ref" not in path.name.lower()
            and "test" not in path.name.lower()
        ]
    if not candidates:
        return None
    return max(candidates, key=_candidate_sort_key).resolve()


def _write_result(output_dir: Path, result: Dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path


def _read_token_usage(path: Path) -> Dict[str, Any]:
    """Aggregate provider usage emitted by ACE's patched LLM backend."""
    prompt = completion = total = requests = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            try:
                prompt += max(0, int(item.get("prompt_tokens", 0) or 0))
                completion += max(0, int(item.get("completion_tokens", 0) or 0))
                total += max(0, int(item.get("total_tokens", 0) or 0))
            except (TypeError, ValueError):
                continue
            requests += 1
    if requests == 0:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "request_count": 0,
            "source": "unavailable",
            "path": str(path),
        }
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total or prompt + completion,
        "request_count": requests,
        "source": "ace_llm_backend_response_usage",
        "path": str(path),
    }


def _prepare_case(case_file: Path, output_dir: Path) -> Tuple[str, Path, Path, Dict[str, Any]]:
    case = _read_json(case_file)
    problem_id = _safe_problem_id(case, case_file)
    prompt_raw = _first_case_value(case, ("prompt_path", "spec_path", "specification_path"))
    golden_raw = _first_case_value(case, ("golden_path", "golden_rtl_path", "reference_rtl_path"))
    prompt_path = _resolve_input(prompt_raw, case_file, label="prompt_path")
    golden_path = _resolve_input(golden_raw, case_file, label="golden_path")
    assert prompt_path is not None and golden_path is not None
    testbench_path = _find_testbench(case, case_file, prompt_path, golden_path)

    top = str(_first_case_value(case, ("golden_top", "top_module", "requested_top_module")) or "TopModule")

    prompt_text = prompt_path.read_text(encoding="utf-8")
    golden_text = golden_path.read_text(encoding="utf-8")
    testbench_text = testbench_path.read_text(encoding="utf-8")
    # ACE's VerilogEval integration fixes the generated DUT name to
    # TopModule.  Rewrite only the evaluator copy; the original public TB is
    # never modified.  CVDP rows use their documented module name here.
    if top != "TopModule":
        testbench_text = re.sub(rf"\b{re.escape(top)}\b", "TopModule", testbench_text)
    if not re.search(r"\bTopModule\b", testbench_text):
        raise AdapterError("testbench does not reference the normalized TopModule")
    ref_text = _rename_top_module(golden_text, top)

    workspace = output_dir / "ace_workspace"
    dataset = workspace / "verilogeval" / "dataset_spec-to-rtl"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / f"{problem_id}_prompt.txt").write_text(prompt_text, encoding="utf-8")
    (dataset / f"{problem_id}_test.sv").write_text(testbench_text, encoding="utf-8")
    (dataset / f"{problem_id}_ref.sv").write_text(ref_text, encoding="utf-8")

    # This metadata deliberately contains hashes and labels only.  It gives a
    # later shared runner provenance without copying golden RTL into result.json.
    metadata = {
        "case_id": problem_id,
        "design_type": case.get("design_type"),
        "top_module": top,
        "prompt_sha256": _sha256(prompt_path),
        "golden_sha256": _sha256(golden_path),
        "testbench_sha256": _sha256(testbench_path),
        "golden_is_private": True,
    }
    (workspace / "adapter_case.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return problem_id, workspace / "verilogeval", testbench_path, case


def run_case(case_file: Path, output_dir: Path, config_file: Optional[Path]) -> Dict[str, Any]:
    """Materialize one case and run the upstream ACE executable."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_file)
    problem_id, verilogeval_root, _, case = _prepare_case(case_file, output_dir)

    if not ACE_ENTRY.is_file():
        raise AdapterError(f"vendored ACE entry point is missing: {ACE_ENTRY}")

    ace_output = output_dir / "ace_output"
    ace_output.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "ace_run.log"
    command = _build_ace_command(problem_id, verilogeval_root, ace_output, config)
    environment = os.environ.copy()
    try:
        from dotenv import dotenv_values

        for key, value in dotenv_values(REPO_ROOT / ".env").items():
            if key and value is not None:
                environment.setdefault(str(key), str(value))
    except Exception:  # noqa: BLE001
        pass
    environment["PATH"] = os.pathsep.join([
        str(REPO_ROOT / ".conda-env" / "bin"),
        environment.get("PATH", ""),
    ])
    # Route the upstream NVIDIA-only transport through the project's active
    # OpenAI-compatible endpoint.  No API key is placed in a command line or
    # result artifact.
    environment["ACE_RTL_OPENAI_COMPAT"] = "1"
    environment["ACE_RTL_OPENAI_BASE_URL"] = (
        environment.get("OPENAI_BASE_URL") or environment.get("OPENAI_API_BASE_URL") or ""
    )
    environment["ACE_RTL_OPENAI_API_KEY"] = environment.get("OPENAI_API_KEY", "")
    environment["ACE_RTL_STREAM"] = str(config.get("stream", 0))
    environment["ACE_RTL_MAX_TOKENS"] = str(config.get("max_tokens", 8192))
    scripts_dir = str(ACE_ROOT / "skills" / "ace-rtl" / "scripts")
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = scripts_dir + (os.pathsep + old_pythonpath if old_pythonpath else "")
    environment["ACE_RTL_WORKSPACE"] = str(output_dir)
    usage_path = output_dir / "llm_usage.jsonl"
    if usage_path.exists():
        usage_path.unlink()
    environment["ACE_RTL_USAGE_PATH"] = str(usage_path)
    environment.setdefault("PYTHONNOUSERSITE", "1")

    process_timeout = config.get("process_timeout")
    if process_timeout is None:
        max_iterations = _as_int(config, "max_iterations", DEFAULT_MAX_ITERATIONS)
        per_iteration = _as_int(config, "timeout", DEFAULT_TIMEOUT)
        process_timeout = max(1800, (max_iterations + 1) * per_iteration + 300)
    try:
        process_timeout = float(process_timeout)
    except (TypeError, ValueError) as exc:
        raise AdapterError("config field process_timeout must be numeric") from exc

    return_code: Optional[int] = None
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=str(ACE_ROOT),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=process_timeout,
                check=False,
            )
            return_code = completed.returncode
    except subprocess.TimeoutExpired:
        result = {
            "status": "error",
            "rtl_path": None,
            "error": "ACE process timed out before producing a final result",
            "usage": {},
            "case_id": problem_id,
            "agent": "NVlabs/ACE-RTL",
            "log_path": str(log_path),
        }
        _write_result(output_dir, result)
        return result
    except OSError as exc:
        result = {
            "status": "error",
            "rtl_path": None,
            "error": f"could not start ACE process: {exc.strerror or 'OS error'}",
            "usage": {},
            "case_id": problem_id,
            "agent": "NVlabs/ACE-RTL",
            "log_path": str(log_path),
        }
        _write_result(output_dir, result)
        return result

    raw_rtl_path = _find_generated_rtl(ace_output)
    token_usage = _read_token_usage(usage_path)
    error: Optional[str] = None
    rtl_path = None
    if raw_rtl_path is not None:
        try:
            top = str(case.get("top_module") or "TopModule")
            rtl_path = _normalize_candidate_top(raw_rtl_path, "TopModule", output_dir / "rtl_normalized.sv")
        except AdapterError as exc:
            error = str(exc)
    if rtl_path is not None:
        # Candidate availability is deliberately separate from functional
        # correctness; the shared JasperGold/DC runner decides that.
        status = "candidate"
    else:
        status = "error"
        error = error or f"ACE exited with code {return_code} without a generated RTL artifact"

    result = {
        "status": status,
        "rtl_path": str(rtl_path) if rtl_path else None,
        "error": error,
        "usage": token_usage,
        "case_id": problem_id,
        "design_type": case.get("design_type"),
        "agent": "NVlabs/ACE-RTL",
        "exit_code": return_code,
        "raw_rtl_path": str(raw_rtl_path) if raw_rtl_path else None,
        "log_path": str(log_path),
    }
    _write_result(output_dir, result)
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt one C-first case to the upstream ACE-RTL agent")
    parser.add_argument("--case-json", required=True, type=Path, help="JSON object describing one case")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for ACE workspace and result")
    parser.add_argument("--config-json", type=Path, default=None, help="Optional ACE CLI configuration JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    case_id = args.case_json.stem
    try:
        result = run_case(args.case_json.resolve(), args.output_dir, args.config_json.resolve() if args.config_json else None)
    except AdapterError as exc:
        result = {
            "status": "error",
            "rtl_path": None,
            "error": str(exc),
            "usage": {},
            "case_id": case_id,
            "agent": "NVlabs/ACE-RTL",
        }
        _write_result(args.output_dir.expanduser().resolve(), result)
    except Exception as exc:  # keep result.json available for the shared runner
        result = {
            "status": "error",
            "rtl_path": None,
            "error": f"adapter failure: {type(exc).__name__}",
            "usage": {},
            "case_id": case_id,
            "agent": "NVlabs/ACE-RTL",
        }
        _write_result(args.output_dir.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") in {"candidate", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
