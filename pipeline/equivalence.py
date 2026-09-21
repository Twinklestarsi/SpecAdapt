"""Shared serial RTL equivalence checking used before expensive synthesis."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from toolchain import require_tool


_MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_]\w*)\b", re.MULTILINE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_top(path: Path, preferred: str = "") -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    modules = _MODULE_RE.findall(text)
    if preferred and preferred in modules:
        return preferred
    if len(modules) == 1:
        return modules[0]
    if not modules:
        raise ValueError(f"No Verilog module found in {path}")
    raise ValueError(f"Multiple modules found in {path}; provide a top module: {modules}")


def _yosys_quote(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


def _run_syntax_check(path: Path, top: str, timeout: int) -> Dict[str, Any]:
    iverilog = require_tool("IVERILOG_PATH", "iverilog")
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [iverilog, "-g2012", "-s", top, "-t", "null", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "elapsed_seconds": time.perf_counter() - started,
            "error": str(exc),
        }
    return {
        "status": "passed" if proc.returncode == 0 else "failed",
        "elapsed_seconds": time.perf_counter() - started,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def verify_equivalence(
    candidate_path: str | Path,
    golden_path: str | Path,
    *,
    output_dir: str | Path,
    candidate_top: str = "",
    golden_top: str = "",
    design_type: str = "combinational",
    timeout: int = 180,
    induction_depth: int = 20,
) -> Dict[str, Any]:
    """Prove candidate/reference equivalence in one local process at a time."""

    candidate = Path(candidate_path).resolve()
    golden = Path(golden_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "verification.json"
    started = time.perf_counter()
    result: Dict[str, Any] = {
        "schema_version": "path_oracle_equivalence_v1",
        "method": "yosys_equiv",
        "execution_policy": {"mode": "serial", "max_concurrency": 1},
        "candidate_path": str(candidate),
        "golden_path": str(golden),
        "candidate_sha256": "",
        "golden_sha256": "",
        "candidate_top": "",
        "golden_top": "",
        "design_type": str(design_type),
        "status": "error",
        "equivalent": False,
    }

    def finish() -> Dict[str, Any]:
        result["elapsed_seconds"] = time.perf_counter() - started
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result

    if not candidate.is_file():
        result["error"] = "candidate_missing"
        return finish()
    if not golden.is_file():
        result["error"] = "golden_missing"
        return finish()

    try:
        resolved_candidate_top = _detect_top(candidate, candidate_top)
        resolved_golden_top = _detect_top(golden, golden_top)
    except Exception as exc:
        result["error"] = f"top_detection_failed: {exc}"
        return finish()

    result.update(
        {
            "candidate_sha256": _sha256(candidate),
            "golden_sha256": _sha256(golden),
            "candidate_top": resolved_candidate_top,
            "golden_top": resolved_golden_top,
        }
    )
    result["candidate_syntax"] = _run_syntax_check(candidate, resolved_candidate_top, timeout)
    result["golden_syntax"] = _run_syntax_check(golden, resolved_golden_top, timeout)
    if result["candidate_syntax"]["status"] != "passed":
        result["status"] = "failed"
        result["error"] = "candidate_syntax_failed"
        return finish()
    if result["golden_syntax"]["status"] != "passed":
        result["status"] = "error"
        result["error"] = "golden_syntax_failed"
        return finish()

    yosys = require_tool("YOSYS_PATH", "yosys")
    script_path = out_dir / "equivalence.ys"
    log_path = out_dir / "yosys.log"
    commands = [
        f"read_verilog -sv {_yosys_quote(golden)}",
        f"prep -top {resolved_golden_top}",
        f"rename {resolved_golden_top} gold",
        "design -stash gold_design",
        f"read_verilog -sv {_yosys_quote(candidate)}",
        f"prep -top {resolved_candidate_top}",
        f"rename {resolved_candidate_top} gate",
        "design -stash gate_design",
        "design -reset",
        "design -copy-from gold_design -as gold gold",
        "design -copy-from gate_design -as gate gate",
        "equiv_make gold gate equiv",
        "hierarchy -top equiv",
        "proc",
    ]
    is_sequential = str(design_type).lower() == "sequential"
    if is_sequential:
        # Yosys SAT/equiv passes have no direct model for $adff cells.  Apply
        # the same standard async-reset lowering to both sides of the miter
        # before attempting the proof.
        commands.append("async2sync")
    commands.extend(["memory", "opt", "equiv_simple"])
    if is_sequential:
        commands.append(f"equiv_induct -undef -seq {max(1, induction_depth)}")
    commands.append("equiv_status -assert")
    script_path.write_text("\n".join(commands) + "\n", encoding="utf-8")

    try:
        proc = subprocess.run(
            [yosys, "-s", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\n" + (exc.stderr or "")
        log_path.write_text(output, encoding="utf-8", errors="replace")
        result["status"] = "timeout"
        result["error"] = f"yosys_timeout_after_{timeout}s"
        return finish()

    output = proc.stdout + "\n" + proc.stderr
    log_path.write_text(output, encoding="utf-8", errors="replace")
    result["yosys_returncode"] = proc.returncode
    result["yosys_log_path"] = str(log_path)
    if proc.returncode == 0:
        result["status"] = "passed"
        result["equivalent"] = True
        result["error"] = ""
    else:
        result["status"] = "failed"
        result["equivalent"] = False
        tail = "\n".join(line for line in output.splitlines() if line.strip())[-4000:]
        result["error"] = tail or "yosys_equivalence_failed"
    return finish()
