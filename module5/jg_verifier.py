"""
JasperGold formal verification module for Module 5 RTL generation.

This module integrates JasperGold equivalence checking into Module 5's direct RTL
generation flow. It verifies generated Verilog against a golden reference and
provides feedback for iterative LLM-based correction.

Workflow:
  1. Generate initial RTL via LLM
  2. Run JasperGold equivalence check against golden reference
  3. If verification fails, extract error feedback
  4. Pass feedback to LLM for correction
  5. Repeat until verification passes or max retries exhausted

All intermediate files are stored in module5_mid/ directory.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from llm_request import (
    completion_max_tokens,
    completion_token_kwargs,
    completion_was_truncated,
    deepseek_thinking_kwargs,
    openai_client_kwargs,
    qwen_thinking_kwargs,
    seed_kwargs,
    truncated_completion_message,
)
from module5.token_usage import token_totals, usage_from_response
from project_paths import PROJECT_ROOT


# JasperGold configuration
#
# The closure gate compiles every reference with ``-DVERILINT -DSYNTHESIS``.
# JasperGold used to analyze the same file with no defines, so the two tools
# elaborated different ``ifdef`` branches: the NVDLA ``sync`` cell then emitted
# ``defparam sync_0.first_stage_of_sync.mode``, which does not exist once the
# ``NV_GENERIC_CELL`` level is present, and JasperGold reported VERI-1187.
# Keeping one macro set for both tools is what makes the gate meaningful.
DEFAULT_ANALYZE_DEFINES: Tuple[str, ...] = ("VERILINT", "SYNTHESIS")


def _parse_analyze_defines(raw_value: str) -> Tuple[str, ...]:
    """Parse ``JG_ANALYZE_DEFINES`` into an ordered, de-duplicated macro tuple."""

    names: list[str] = []
    for token in re.split(r"[,\s]+", str(raw_value or "").strip()):
        if not token:
            continue
        if not re.fullmatch(r"[A-Za-z_]\w*(?:=[^\s+]*)?", token):
            raise ValueError(
                f"JG_ANALYZE_DEFINES contains an unusable macro: {token!r}"
            )
        if token not in names:
            names.append(token)
    return tuple(names)


def _analyze_define_prefix(analyze_defines: Tuple[str, ...] | list[str]) -> str:
    """Render macro defines for a JasperGold ``analyze`` command.

    ``+define+A+B`` is the only form the installed JasperGold 2023 accepts;
    ``-define {A B}`` is rejected.  Verified live in
    ``runs/phase6/nvdla13_fix_20260908/gate/jg_analyze_macro_probe.json``.
    """

    names = [str(name).strip() for name in analyze_defines if str(name).strip()]
    if not names:
        return ""
    return "+define+" + "+".join(names) + " "


def _load_jg_config(env_path: str | Path) -> Dict[str, Any]:
    """Load JasperGold remote server configuration from .env file."""
    load_dotenv(env_path, override=True)
    enabled = os.environ.get("JG_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        raise RuntimeError(
            "JasperGold is disabled; set JG_ENABLED=true after installation"
        )
    raw_defines = os.environ.get("JG_ANALYZE_DEFINES")
    config = {
        "remote_user": os.environ.get("JG_REMOTE_USER", "").strip(),
        "remote_host": os.environ.get("JG_REMOTE_HOST", "").strip(),
        "remote_base": os.environ.get("JG_REMOTE_BASE", "").rstrip("/"),
        "env_script": os.environ.get("JG_ENV_SCRIPT", "").strip(),
        "jg_bin": os.environ.get("JG_BIN", "").strip(),
        "timeout": int(os.environ.get("JG_TIMEOUT", "600")),
        "analyze_defines": (
            DEFAULT_ANALYZE_DEFINES
            if raw_defines is None
            else _parse_analyze_defines(raw_defines)
        ),
    }
    missing = [
        name
        for name, value in (
            ("JG_REMOTE_USER", config["remote_user"]),
            ("JG_REMOTE_HOST", config["remote_host"]),
            ("JG_REMOTE_BASE", config["remote_base"]),
            ("JG_ENV_SCRIPT", config["env_script"]),
            ("JG_BIN", config["jg_bin"]),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "JasperGold is not configured; set " + ", ".join(missing)
        )
    if not os.path.isabs(config["jg_bin"]):
        raise ValueError("JG_BIN must be an absolute remote path")
    return config


def _with_verification_timeout(
    jg_config: Dict[str, Any], verification_timeout: int | None
) -> Dict[str, Any]:
    """Apply an optional per-verification timeout without changing .env."""

    config = dict(jg_config)
    if verification_timeout is None:
        return config
    try:
        timeout = int(verification_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("verification_timeout must be an integer") from exc
    if timeout < 1:
        raise ValueError("verification_timeout must be at least 1")
    config["timeout"] = timeout
    return config


def _local_to_remote(local_path: Path, local_base: Path, remote_base: str) -> str:
    """Convert local path to remote path."""
    if not remote_base:
        raise ValueError("JG_REMOTE_BASE is not configured")
    rel = local_path.resolve().relative_to(local_base.resolve())
    return f"{remote_base}/{rel}"


def _remote_jg_shell_prefix(jg_config: Dict[str, Any]) -> str:
    """Build the common shell prefix for every remote JasperGold command.

    JasperGold starts a local session server as part of a batch invocation.  A
    proxy inherited from the login environment can make that loopback
    connection fail, so the prefix deliberately sources the configured Cadence
    environment first, removes proxy variables in both common casings, and
    then restores only loopback entries in ``NO_PROXY``/``no_proxy``.

    Values are shell-quoted because this string is passed to the remote login
    shell as one SSH command argument.  ``JG_BIN`` is checked here as well as
    in :func:`_load_jg_config` so callers constructing a config dictionary
    directly cannot accidentally run a relative or missing executable.
    """
    env_script = str(jg_config.get("env_script", "")).strip()
    jg_bin = str(jg_config.get("jg_bin", "")).strip()
    if not env_script:
        raise ValueError("JG_ENV_SCRIPT is not configured")
    if not jg_bin:
        raise ValueError("JG_BIN is not configured")
    if not os.path.isabs(jg_bin):
        raise ValueError("JG_BIN must be an absolute remote path")

    # Keep the final ``&&`` so callers can append their command while ensuring
    # that an unavailable environment script prevents JasperGold from running.
    return (
        "set -e; "
        f"source {shlex.quote(env_script)} && "
        "unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY "
        "no_proxy NO_PROXY && "
        "export NO_PROXY=127.0.0.1,localhost,::1 "
        "no_proxy=127.0.0.1,localhost,::1 && "
    )


def _build_remote_jg_command(
    remote_dir: str,
    remote_tcl: str,
    project_name: str,
    jg_config: Dict[str, Any],
) -> str:
    """Construct a quoted remote shell command for one JG batch invocation."""
    prefix = _remote_jg_shell_prefix(jg_config)
    return (
        f"{prefix}"
        f"cd {shlex.quote(remote_dir)} && "
        f"{shlex.quote(str(jg_config['jg_bin']))} -proj "
        f"{shlex.quote(project_name)} -batch {shlex.quote(remote_tcl)}"
    )


def _cleanup_remote_jgproject(
    remote_dir: str,
    remote_tcl: str,
    project_name: str,
    jg_config: Dict[str, Any],
) -> None:
    """Clean up stale JasperGold project directories after timeout/errors."""
    remote_user = jg_config["remote_user"]
    remote_host = jg_config["remote_host"]

    kill_cmd = (
        f"pkill -9 -f {shlex.quote(remote_tcl)} 2>/dev/null || true; "
        f"pkill -9 -f {shlex.quote(f'jaspergold.*{project_name}')} 2>/dev/null || true; "
        f"sleep 1"
    )
    project_glob = shlex.quote(project_name)
    clean_cmd = (
        f"cd {shlex.quote(remote_dir)} && "
        f"rm -rf {project_glob} {project_glob}_* "
        "jgproject jgproject_* sessionLogs* 2>/dev/null || true"
    )
    try:
        remote_cmd = (
            f"{_remote_jg_shell_prefix(jg_config)}"
            f"{kill_cmd}; {clean_cmd}"
        )
        subprocess.run(
            ["ssh", f"{remote_user}@{remote_host}", remote_cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        pass


def _cleanup_local_jgproject(project_dir: Path) -> None:
    """Remove stale local JasperGold project artifacts in the shared workspace."""
    try:
        if project_dir.exists():
            shutil.rmtree(project_dir, ignore_errors=True)
    except Exception:
        pass

    try:
        for lock_path in project_dir.parent.glob(f"{project_dir.name}*.lock"):
            if lock_path.is_file():
                lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def _find_local_jg_session_log(project_dir: Path) -> Optional[Path]:
    """Find the most relevant local JasperGold session log for a project."""
    candidates = sorted(project_dir.glob("sessionLogs*/session_0/jg_session_0.log"))
    if candidates:
        return candidates[-1]
    return None


def _recover_jg_result_from_session_log(project_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Recover JasperGold status from local session logs.

    This is mainly used when the outer SSH command times out even though JG has
    already produced useful progress, such as counterexamples.
    """
    log_path = _find_local_jg_session_log(project_dir)
    if not log_path or not log_path.is_file():
        return None

    text = log_path.read_text(encoding="utf-8", errors="replace")
    raw_tail = text[-3000:] if len(text) > 3000 else text
    result = {
        "success": False,
        "proven": 0,
        "cex": 0,
        "undetermined": 0,
        "error": "",
        "raw_output": raw_tail,
    }

    summary_m = re.search(
        r"(=+\s*\n\s*SUMMARY\s*\n\s*=+.*?)(?=\n\s*=+\s*\n|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if summary_m:
        summary = summary_m.group(1)
        for key in ("proven", "cex", "undetermined", "unknown"):
            m = re.search(rf"-\s*{key}\s*:\s*(\d+)", summary, re.IGNORECASE)
            if m:
                result[key] = int(m.group(1))

        m = re.search(r"assertions\s*:\s*(\d+)", summary, re.IGNORECASE)
        assertions = int(m.group(1)) if m else 0
        if assertions > 0 and assertions == result["proven"]:
            result["success"] = True
            return result
        if result["cex"] > 0:
            result["error"] = "counterexample found"
            return result
        if result.get("unknown", 0) > 0:
            result["error"] = "unknown (JasperGold engine error)"
            return result
        if result["undetermined"] > 0:
            result["error"] = "undetermined"
            return result

    cex_count = len(re.findall(r"counterexample\s*\(cex\)", text, re.IGNORECASE))
    if cex_count > 0:
        result["cex"] = cex_count
        result["error"] = "counterexample found"
        return result

    proven_count = len(re.findall(r"was proven", text, re.IGNORECASE))
    if proven_count > 0 and "Starting proof on task" in text:
        result["proven"] = proven_count
        result["error"] = "partial proof progress before timeout"
        return result

    if "cannot obtain ownership of project directory" in text.lower():
        result["error"] = "project directory lock"
        return result

    if "ERROR" in text:
        err_lines = [ln.strip() for ln in text.splitlines() if "ERROR" in ln]
        if err_lines:
            result["error"] = err_lines[0]
            return result

    return None


def _property_output_name(property_name: str) -> str:
    """Recover the compared output name from a labelled FPV assertion."""
    match = re.search(r"a_eq_([A-Za-z_][A-Za-z0-9_]*)$", property_name or "")
    return match.group(1) if match else ""


def _properties_from_session_log(log_text: str) -> list[Dict[str, Any]]:
    """Extract property status/trace length when the TSV export is unavailable."""
    records: Dict[str, Dict[str, Any]] = {}
    cex_pattern = re.compile(
        r"counterexample\s*\(cex\)\s+with\s+(\d+)\s+cycles\s+was\s+found\s+"
        r"for\s+the\s+property\s+\"([^\"]+)\"",
        re.IGNORECASE,
    )
    proven_pattern = re.compile(
        r"The\s+property\s+\"([^\"]+)\"\s+was\s+proven",
        re.IGNORECASE,
    )
    codex_pattern = re.compile(
        r"CODEX_PROPERTY_RESULT\s+([^\s]+)\s+([^\s]+)\s+(\d+)",
        re.IGNORECASE,
    )
    for match in codex_pattern.finditer(log_text or ""):
        name, status, trace_length = match.groups()
        records[name] = {
            "name": name,
            "output": _property_output_name(name),
            "status": status,
            "trace_length": int(trace_length),
        }
    for match in cex_pattern.finditer(log_text or ""):
        trace_length, name = match.groups()
        records[name] = {
            "name": name,
            "output": _property_output_name(name),
            "status": "cex",
            "trace_length": int(trace_length),
        }
    for match in proven_pattern.finditer(log_text or ""):
        name = match.group(1)
        records.setdefault(
            name,
            {
                "name": name,
                "output": _property_output_name(name),
                "status": "proven",
                "trace_length": 0,
            },
        )
    return list(records.values())


def _parse_vcd_signal_sequences(vcd_path: Path) -> Dict[str, str]:
    """Return compact ``time=value`` sequences keyed by hierarchical name.

    JasperGold's ``visualize -get_value`` output differs across releases, while
    the VCD writer is stable.  Reading the exported VCD here keeps the retry
    prompt independent of that console formatting and avoids sending the full
    waveform to the LLM.
    """
    if not vcd_path.is_file():
        return {}

    identifier_names: Dict[str, list[str]] = {}
    scopes: list[str] = []
    in_header = True
    current_time = 0
    events: Dict[str, list[str]] = {}

    for raw_line in vcd_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if in_header:
            if line.startswith("$scope "):
                parts = line.split()
                if len(parts) >= 3:
                    scopes.append(parts[2])
            elif line.startswith("$upscope"):
                if scopes:
                    scopes.pop()
            elif line.startswith("$var "):
                match = re.match(
                    r"\$var\s+\S+\s+\d+\s+(\S+)\s+(.+?)\s+\$end$",
                    line,
                )
                if match:
                    identifier, reference = match.groups()
                    # Drop an optional range suffix (``state [2:0]``) from the
                    # lookup key; the value itself retains its full bit vector.
                    signal_name = reference.split()[0]
                    full_name = ".".join([*scopes, signal_name])
                    identifier_names.setdefault(identifier, []).append(full_name)
            elif line.startswith("$enddefinitions"):
                in_header = False
            continue

        if line.startswith("#"):
            try:
                current_time = int(line[1:])
            except ValueError:
                pass
            continue

        identifier = ""
        value = ""
        if line[0] in "01xXzZ":
            value, identifier = line[0].lower(), line[1:].strip()
        elif line[0] in "bBrR":
            parts = line.split(None, 1)
            if len(parts) == 2:
                value = parts[0][1:].lower()
                identifier = parts[1].strip()
        if not identifier or identifier not in identifier_names:
            continue
        for full_name in identifier_names[identifier]:
            sequence = events.setdefault(full_name, [])
            item = f"t{current_time}={value}"
            if not sequence or sequence[-1] != item:
                sequence.append(item)

    return {
        name: ", ".join(sequence[:64])
        for name, sequence in events.items()
        if sequence
    }


def _load_jg_diagnostic_artifacts(
    work_dir: Path,
    session_log_path: Optional[Path] = None,
    fallback_log_text: str = "",
) -> Dict[str, Any]:
    """Load property statuses and compact CEX traces emitted by FPV Tcl."""
    properties: list[Dict[str, Any]] = []
    property_path = work_dir / "fpv_property_results.tsv"
    if property_path.is_file():
        for line in property_path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            try:
                trace_length = int((parts[2] if len(parts) > 2 else "0") or 0)
            except ValueError:
                trace_length = 0
            properties.append(
                {
                    "name": name,
                    "output": _property_output_name(name),
                    "status": parts[1].strip(),
                    "trace_length": trace_length,
                }
            )

    log_text = fallback_log_text or ""
    if session_log_path and session_log_path.is_file():
        log_text += "\n" + session_log_path.read_text(
            encoding="utf-8", errors="replace"
        )
    if not properties and log_text:
        properties = _properties_from_session_log(log_text)

    counterexamples: list[Dict[str, Any]] = []
    for trace_path in sorted(work_dir.glob("cex_*.trace.tsv")):
        trace_text = trace_path.read_text(encoding="utf-8", errors="replace")
        property_name = ""
        status = "cex"
        trace_length = 0
        signals: list[Dict[str, str]] = []
        for line in trace_text.splitlines():
            parts = line.split("\t", 2)
            if not parts:
                continue
            if parts[0] == "property" and len(parts) >= 2:
                property_name = parts[1]
            elif parts[0] == "status" and len(parts) >= 2:
                status = parts[1]
            elif parts[0] == "trace_length" and len(parts) >= 2:
                try:
                    trace_length = int(parts[1] or 0)
                except ValueError:
                    trace_length = 0
            elif parts[0] == "signal" and len(parts) >= 3:
                signals.append({"name": parts[1], "values": parts[2]})
        vcd_path = trace_path.with_suffix("").with_suffix(".vcd")
        vcd_sequences = _parse_vcd_signal_sequences(vcd_path)
        for signal in signals:
            if not signal.get("values"):
                signal["values"] = vcd_sequences.get(signal.get("name", ""), "")
        if vcd_sequences:
            # Keep only wrapper inputs and compared outputs requested above.
            # Internal ``u_ref`` signals would disclose golden implementation
            # structure, so they must never enter the repair prompt.
            signals = [signal for signal in signals if signal.get("values")]
        counterexamples.append(
            {
                "property": property_name,
                "output": _property_output_name(property_name),
                "status": status,
                "trace_length": trace_length,
                "signals": signals,
                "trace_path": str(trace_path),
                "vcd_path": str(vcd_path) if vcd_path.is_file() else "",
            }
        )

    summary_lines: list[str] = []
    if properties:
        summary_lines.append("Property-level JasperGold results:")
        for item in properties:
            label = item.get("output") or item.get("name")
            summary_lines.append(
                f"- {label}: {item.get('status', 'unknown')}; "
                f"trace_length={item.get('trace_length', 0)}; "
                f"property={item.get('name', '')}"
            )
    for cex in counterexamples:
        summary_lines.append(
            f"Counterexample signal sequences for "
            f"{cex.get('output') or cex.get('property')}:"
        )
        for signal in cex.get("signals", []):
            summary_lines.append(
                f"- {signal.get('name')}: {signal.get('values')}"
            )

    return {
        "properties": properties,
        "counterexamples": counterexamples,
        "diagnostic_summary": "\n".join(summary_lines),
        "property_report_path": str(property_path) if property_path.is_file() else "",
        "session_log_path": str(session_log_path) if session_log_path else "",
    }


def _attach_jg_diagnostics(
    result: Dict[str, Any],
    work_dir: Path,
    project_dir: Path,
) -> Dict[str, Any]:
    session_log_path = _find_local_jg_session_log(project_dir)
    result.update(
        _load_jg_diagnostic_artifacts(
            work_dir,
            session_log_path,
            fallback_log_text=str(result.get("raw_output", "") or ""),
        )
    )
    return result


def _mask_comments_and_strings(
    verilog_src: str, *, mask_strings: bool = True
) -> str:
    """Blank comments and string literals while preserving character offsets.

    Port scanning is regex based, so a declaration keyword inside a comment or
    a string literal used to be collected as a real port.  NVDLA references
    contain assertion messages such as ``"input int8 ..."`` and ``"Receive
    input data when not busy"``, which invented the ports ``int8``, ``int16``,
    ``fp16``, ``data`` and ``when``.  Those phantom ports then looked like
    missing candidate ports and produced false interface mismatches.

    Every masked character is replaced by a space (newlines are kept) so the
    result has exactly the same length as the input.  Callers may therefore use
    offsets found in the masked text to slice the original source.
    ``mask_strings=False`` blanks comments only and keeps literal text, which is
    what parameter-default extraction needs.
    """

    chars = list(verilog_src)
    total = len(chars)
    index = 0
    state = ""  # "", "line", "block", "string"
    while index < total:
        char = chars[index]
        if state == "":
            if char == "/" and index + 1 < total and chars[index + 1] == "/":
                chars[index] = " "
                chars[index + 1] = " "
                state = "line"
                index += 2
                continue
            if char == "/" and index + 1 < total and chars[index + 1] == "*":
                chars[index] = " "
                chars[index + 1] = " "
                state = "block"
                index += 2
                continue
            # Verilog string literals use double quotes.  A single quote
            # belongs to a based literal such as 8'hff and never opens a string.
            if char == '"':
                state = "string"
            index += 1
            continue
        if state == "line":
            if char == "\n":
                state = ""
            else:
                chars[index] = " "
            index += 1
            continue
        if state == "block":
            if char == "*" and index + 1 < total and chars[index + 1] == "/":
                chars[index] = " "
                chars[index + 1] = " "
                state = ""
                index += 2
                continue
            if char != "\n":
                chars[index] = " "
            index += 1
            continue
        # Inside a string literal.
        if char == "\n":
            # A Verilog string cannot contain a raw newline.  Ending the state
            # here bounds the damage of an unbalanced quote to a single line.
            state = ""
            index += 1
            continue
        if char == "\\" and index + 1 < total and chars[index + 1] != "\n":
            if mask_strings:
                chars[index] = " "
                chars[index + 1] = " "
            index += 2
            continue
        if char == '"':
            state = ""
            index += 1
            continue
        if mask_strings:
            chars[index] = " "
        index += 1
    return "".join(chars)


def parse_verilog_ports(verilog_src: str) -> Tuple[str, list[Dict[str, str]], bool]:
    """
    Parse module name, ports, and detect sequential logic from Verilog source.
    Returns (module_name, ports, is_sequential).
    """
    # Comments and string literals are blanked once, before any pattern runs, so
    # no declaration keyword inside them can be mistaken for a port.  The mask
    # keeps the original length, so ``port_section_match`` offsets stay valid.
    masked_src = _mask_comments_and_strings(verilog_src)

    # Extract module name
    m = re.search(r"\bmodule\s+(\w+)", masked_src)
    if not m:
        raise ValueError("Cannot find module declaration")
    mod_name = m.group(1)

    # Extract the port list section (between module declaration and first semicolon/begin)
    port_section_match = re.search(
        r"\bmodule\s+\w+[^;]*?\((.*?)\);",
        masked_src,
        re.DOTALL
    )

    if not port_section_match:
        raise ValueError("Cannot find module port list")

    port_section = port_section_match.group(1)

    ports = []
    seen_names = set()
    reserved_names = {"input", "output", "logic", "wire", "reg", "signed"}

    # Parse each complete ANSI declaration, including comma-separated names
    # that inherit the declaration's direction and type:
    #   input wire [7:0] a, b, c,
    #   output reg valid, output reg [15:0] result
    # The old parser matched only the first name (``a``), which silently left
    # ``b`` and ``c`` unconnected in the generated equivalence wrapper.
    clean_port_section = port_section
    declaration_pattern = re.compile(
        r"\b(input|output)\b\s*(.*?)"
        r"(?=,\s*(?:input|output)\b|$)",
        re.DOTALL,
    )
    declaration_body_pattern = re.compile(
        r"^(?:(logic|wire|reg)\s+)?"
        r"(signed\s+)?"
        r"((?:\[[^\]]+\]\s*)*)"
        r"(.+?)\s*$",
        re.DOTALL,
    )
    name_pattern = re.compile(
        r"^([A-Za-z_]\w*)\s*((?:\[[^\]]+\]\s*)*)"
        r"(?:=\s*.*)?$",
        re.DOTALL,
    )

    for declaration in declaration_pattern.finditer(clean_port_section):
        direction = declaration.group(1)
        body_match = declaration_body_pattern.match(declaration.group(2).strip())
        if not body_match:
            continue
        type_keyword = (body_match.group(1) or "").strip()
        signed = (body_match.group(2) or "").strip()
        packed_dims = (body_match.group(3) or "").strip()
        names_text = body_match.group(4) or ""

        for raw_name in _split_top_level_commas(names_text):
            name_match = name_pattern.match(raw_name.strip())
            if not name_match:
                continue
            name = name_match.group(1)
            unpacked_dims = (name_match.group(2) or "").strip()
            if name in seen_names or name in reserved_names:
                continue

            type_parts = []
            if type_keyword:
                type_parts.append(type_keyword)
            if signed:
                type_parts.append(signed)
            if packed_dims:
                type_parts.append(packed_dims)
            if unpacked_dims:
                type_parts.append(unpacked_dims)
            type_str = " ".join(type_parts) if type_parts else ""
            ports.append({"dir": direction, "type": type_str, "name": name})
            seen_names.add(name)

    # Also support old-style Verilog modules:
    #   module foo(a, b);
    #   input [3:0] a;
    #   output reg b;
    # This style is common in the NVDLA golden RTL.
    body_text = masked_src[port_section_match.end():]
    for dm in re.finditer(
        r"\b(input|output)\s+(?:(logic|wire|reg)\s+)?(signed\s+)?(\[[^\]]+\]\s*)?([^;]+);",
        body_text,
        re.DOTALL,
    ):
        direction = dm.group(1)
        type_keyword = (dm.group(2) or "").strip()
        signed = (dm.group(3) or "").strip()
        packed_dims = (dm.group(4) or "").strip()
        names_text = dm.group(5) or ""

        for raw_name in names_text.split(","):
            raw_name = raw_name.strip()
            nm = re.match(r"(\w+)(\s*\[[^\]]+\])?", raw_name)
            if not nm:
                continue
            name = nm.group(1)
            unpacked_dims = (nm.group(2) or "").strip()
            if name in seen_names or name in ("input", "output", "logic", "wire", "reg", "signed"):
                continue

            type_parts = []
            if type_keyword:
                type_parts.append(type_keyword)
            if signed:
                type_parts.append(signed)
            if packed_dims:
                type_parts.append(packed_dims)
            if unpacked_dims:
                type_parts.append(unpacked_dims)
            type_str = " ".join(type_parts) if type_parts else ""
            ports.append({"dir": direction, "type": type_str, "name": name})
            seen_names.add(name)

    # Detect sequential logic
    is_seq = bool(
        re.search(r"always(?:_ff)?\s*@\s*\([^)]*(?:posedge|negedge)", masked_src, re.IGNORECASE)
    )

    return mod_name, ports, is_seq


def _port_shape(port: Dict[str, str]) -> tuple[str, bool, tuple[str, ...]]:
    """Return interface-relevant attributes, ignoring wire/reg/logic spelling."""

    type_text = str(port.get("type", "") or "")
    dimensions = tuple(
        re.sub(r"\s+", "", item)
        for item in re.findall(r"\[[^\]]+\]", type_text)
    )
    is_signed = bool(re.search(r"\bsigned\b", type_text))
    return str(port.get("dir", "")), is_signed, dimensions


def _dimension_shape_key(dimension: str) -> Optional[tuple[int, bool]]:
    """Return ``(bit_width, is_descending)`` for one concrete range, else ``None``.

    The orientation is kept because ``[7:0]`` and ``[0:7]`` hold the same number
    of bits in the opposite order; only equal-width, equal-orientation ranges
    such as ``[7:0]`` and ``[8:1]`` describe the same interface slice.
    """

    range_match = re.fullmatch(r"\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]", dimension)
    if range_match:
        high = int(range_match.group(1))
        low = int(range_match.group(2))
        return abs(high - low) + 1, high >= low
    size_match = re.fullmatch(r"\[\s*(\d+)\s*\]", dimension)
    if size_match:
        return int(size_match.group(1)), True
    return None


def _dimension_bit_width(dimension: str) -> Optional[int]:
    """Return the bit width of one concrete packed range, else ``None``."""

    key = _dimension_shape_key(dimension)
    return None if key is None else key[0]


_EXPR_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|\*\*|[-+*/()]|\s+")


def _split_dimension_bounds(dimension: str) -> Optional[tuple[str, str]]:
    """Return the ``(msb, lsb)`` expression texts of one packed dimension."""

    inner_match = re.fullmatch(r"\[\s*(.+?)\s*\]", dimension, re.DOTALL)
    if not inner_match:
        return None
    inner = inner_match.group(1)
    depth = 0
    for index, char in enumerate(inner):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == ":" and depth == 0:
            return inner[:index].strip(), inner[index + 1:].strip()
    # ``[N]`` is the SystemVerilog shorthand for ``[N-1:0]``.
    return f"({inner})-1", "0"


def _expression_symbols(expression: str) -> Optional[set[str]]:
    """Return the identifiers in an arithmetic expression, or ``None``.

    ``None`` means the expression uses syntax this checker deliberately does not
    interpret (``$clog2``, macros, concatenations, shifts).  Callers must then
    fall back to the conservative rules instead of guessing a width.
    """

    text = str(expression or "").strip()
    if not text:
        return None
    symbols: set[str] = set()
    position = 0
    while position < len(text):
        match = _EXPR_TOKEN_RE.match(text, position)
        if not match:
            return None
        token = match.group(0)
        position = match.end()
        if not token.strip():
            continue
        if token[0].isalpha() or token[0] == "_":
            symbols.add(token)
    return symbols


_SymbolicPolynomial = dict[tuple[str, ...], int]


def _polynomial_add(
    left: _SymbolicPolynomial, right: _SymbolicPolynomial, sign: int = 1
) -> _SymbolicPolynomial:
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = result.get(monomial, 0) + sign * coefficient
        if result[monomial] == 0:
            del result[monomial]
    return result


def _polynomial_multiply(
    left: _SymbolicPolynomial, right: _SymbolicPolynomial
) -> _SymbolicPolynomial:
    result: _SymbolicPolynomial = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            monomial = tuple(sorted((*left_monomial, *right_monomial)))
            result[monomial] = result.get(monomial, 0) + (
                left_coefficient * right_coefficient
            )
    return {
        monomial: coefficient
        for monomial, coefficient in result.items()
        if coefficient
    }


def _symbolic_polynomial(expression: str) -> Optional[_SymbolicPolynomial]:
    """Parse a small arithmetic expression into an exact integer polynomial.

    A shape comparison is allowed to claim flattening only when the complete
    width expression can be proven equal symbolically.  Unsupported syntax is
    rejected, which is safer than evaluating a handful of parameter values:
    two different expressions can coincide at every sampled point.
    """

    text = str(expression or "").strip()
    if _expression_symbols(text) is None:
        return None
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def visit(node: ast.AST) -> Optional[_SymbolicPolynomial]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return {(): int(node.value)}
        if isinstance(node, ast.Name):
            return {(node.id,): 1}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            if value is None:
                return None
            if isinstance(node.op, ast.UAdd):
                return value
            return {
                monomial: -coefficient
                for monomial, coefficient in value.items()
            }
        if not isinstance(node, ast.BinOp):
            return None
        left = visit(node.left)
        right = visit(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return _polynomial_add(left, right)
        if isinstance(node.op, ast.Sub):
            return _polynomial_add(left, right, sign=-1)
        if isinstance(node.op, ast.Mult):
            return _polynomial_multiply(left, right)
        if isinstance(node.op, ast.Pow):
            if set(right) != {()}:
                return None
            exponent = right[()]
            if exponent < 0 or exponent > 8:
                return None
            result: _SymbolicPolynomial = {(): 1}
            for _ in range(exponent):
                result = _polynomial_multiply(result, left)
            return result
        # Division and all other operators are rejected because proving exact
        # equality over arbitrary parameter values would need a richer domain
        # analysis than this interface gate should attempt.
        return None

    return visit(tree.body)


def _dimension_width_polynomial(dimension: str) -> Optional[_SymbolicPolynomial]:
    """Return an exact, conservative width proof for one packed dimension.

    Flattening accepts ranges with a literal zero as the low bound.  This
    covers the NVDLA form ``[N-1:0]`` and rejects an unresolved or ascending
    range instead of assuming a parameter value.
    """

    bounds = _split_dimension_bounds(dimension)
    if bounds is None or bounds[1].strip() != "0":
        return None
    high = _symbolic_polynomial(bounds[0])
    if high is None:
        return None
    width = _polynomial_add(high, {(): 1})
    if not width or any(coefficient < 0 for coefficient in width.values()):
        return None
    if not width.get((), 0) and not any(monomial for monomial in width):
        return None
    return width


def _packed_flatten_equivalent(
    golden_dims: tuple[str, ...],
    candidate_dims: tuple[str, ...],
    parameter_defaults: Optional[Dict[str, str]] = None,
) -> bool:
    """Accept a multi-dimensional packed port against its flattened form.

    The csrng reference declares ``input logic [NUM_HW_APPS-1:0][31:0]`` while
    the candidate declares ``input wire [(NUM_HW_APPS*32)-1:0]`` and slices it
    with ``[i*32 +: 32]``.  Those describe the same pins in the same bit order,
    so rejecting the pair as a width mismatch hid the real comparison.

    The width identity is checked as an exact symbolic polynomial, rather than
    by substituting a finite set of parameter values.  Anything outside the
    conservative ``[expression:0]`` form is rejected.  Symbolic dimensions
    also require a concrete, positive default in the wrapper parameter set:
    ``[N-1:0]`` is not a width proof for the illegal/degenerate ``N=0`` case.
    """

    if not golden_dims or not candidate_dims:
        return False
    if len(golden_dims) == len(candidate_dims):
        return False
    if len(golden_dims) == 1:
        flat_dims, packed_dims = golden_dims, candidate_dims
    elif len(candidate_dims) == 1:
        flat_dims, packed_dims = candidate_dims, golden_dims
    else:
        # Only a genuine multi-dimensional-to-one-dimensional flattening is
        # covered. Treating 3-D versus 2-D as equivalent could hide a wiring or
        # bit-order mistake.
        return False
    flat_width = _dimension_width_polynomial(flat_dims[0])
    if flat_width is None:
        return False
    packed_width: _SymbolicPolynomial = {(): 1}
    for dimension in packed_dims:
        width = _dimension_width_polynomial(dimension)
        if width is None:
            return False
        packed_width = _polynomial_multiply(packed_width, width)
    symbols = {
        symbol
        for monomial in (*flat_width.keys(), *packed_width.keys())
        for symbol in monomial
    }
    defaults = parameter_defaults or {}
    for symbol in symbols:
        default = defaults.get(symbol)
        if default is None:
            return False
        default_poly = _symbolic_polynomial(default)
        if default_poly is None or set(default_poly) != {()} or default_poly[()] < 1:
            return False
    return flat_width == packed_width


def _normalise_single_bit_dims(dimensions: tuple[str, ...]) -> tuple[str, ...]:
    """Drop a single concrete one-bit range so it matches a scalar port.

    ``input a`` and ``input [0:0] a`` are the same one-bit interface, but the
    strict shape check compared the packed dimension count and rejected the
    pair.  That produced false ``width mismatch`` reports on NVDLA references
    such as ``reg2dp_op_en``.  Only the one-bit case is collapsed: a wider or
    multi-dimensional declaration keeps its dimensions so genuine width and
    packing differences are still rejected.
    """

    if len(dimensions) == 1 and _dimension_bit_width(dimensions[0]) == 1:
        return ()
    return dimensions


def _port_shapes_compatible(
    golden_port: Dict[str, str],
    candidate_port: Dict[str, str],
    parameter_defaults: Optional[Dict[str, str]] = None,
) -> bool:
    """Compare port shapes without rejecting unresolved parameter widths."""

    golden_direction, golden_signed, golden_dims = _port_shape(golden_port)
    candidate_direction, candidate_signed, candidate_dims = _port_shape(
        candidate_port
    )
    # Direction and signedness stay strict: they change the interface contract
    # even when the total bit count matches.
    if golden_direction != candidate_direction or golden_signed != candidate_signed:
        return False
    golden_dims = _normalise_single_bit_dims(golden_dims)
    candidate_dims = _normalise_single_bit_dims(candidate_dims)
    if golden_dims == candidate_dims:
        return True
    if len(golden_dims) != len(candidate_dims):
        # A 2-D packed reference port and its flattened candidate form describe
        # the same pins; every other dimension-count difference stays rejected.
        return _packed_flatten_equivalent(
            golden_dims, candidate_dims, parameter_defaults
        )

    # Two concrete ranges of equal width and orientation, such as [7:0] and
    # [8:1], describe the same bit slice and are connected positionally.
    golden_keys = [_dimension_shape_key(item) for item in golden_dims]
    candidate_keys = [_dimension_shape_key(item) for item in candidate_dims]
    if all(key is not None for key in golden_keys) and all(
        key is not None for key in candidate_keys
    ):
        return golden_keys == candidate_keys

    # A parameterised dimension such as [size-1:0] may be equivalent to a
    # concrete candidate dimension such as [7:0].  JasperGold resolves those
    # parameter defaults during elaboration; reject here only when both sides
    # are concrete numeric expressions and visibly differ.
    all_dimensions = golden_dims + candidate_dims
    has_unresolved_symbol = any(
        re.search(r"[A-Za-z_$`]", dimension)
        for dimension in all_dimensions
    )
    return has_unresolved_symbol


class InterfaceMismatchError(ValueError):
    """Raised when the candidate interface differs from the golden interface.

    The attached ``diff`` describes the port-level difference only.  It is safe
    to forward to a repair prompt: the interface is a contract both sides must
    honour, whereas the golden body is the answer and must never be forwarded.
    """

    def __init__(self, message: str, diff: Dict[str, Any]) -> None:
        super().__init__(message)
        self.diff = diff


def _port_declaration_text(port: Dict[str, str]) -> str:
    """Render one port as the declaration a candidate is required to match."""

    direction = str(port.get("dir", "") or "")
    type_text = re.sub(r"\s+", " ", str(port.get("type", "") or "")).strip()
    name = str(port.get("name", "") or "")
    return " ".join(part for part in (direction, type_text, name) if part)


def _dimension_width_text(dimension: str) -> Optional[str]:
    """Return a readable width expression for one packed dimension."""

    key = _dimension_shape_key(dimension)
    if key is not None:
        return str(key[0])
    bounds = _split_dimension_bounds(dimension)
    if bounds is None:
        return None
    high, low = bounds
    # ``[X-1:0]`` is by far the most common parameterised form; keeping it as
    # ``X`` makes the flattened width readable instead of ``((X-1)-(0)+1)``.
    simple = re.fullmatch(r"(.+?)\s*-\s*1", high)
    if simple and low.strip() == "0":
        return simple.group(1).strip()
    return f"(({high})-({low})+1)"


def _verilog_2001_port_declaration(port: Dict[str, str]) -> str:
    """Render one port in the flattened Verilog-2001 form the pipeline requires.

    The generation prompt forbids multi-dimensional packed ports, so a reference
    port declared ``input logic [NUM_HW_APPS-1:0][31:0]`` has to appear in the
    contract as ``input [(NUM_HW_APPS*32)-1:0]``: the same pins in the same bit
    order, written the way the candidate must declare them.
    """

    direction = str(port.get("dir", "") or "")
    type_text = str(port.get("type", "") or "")
    name = str(port.get("name", "") or "")
    dimensions = tuple(
        re.sub(r"\s+", "", item) for item in re.findall(r"\[[^\]]+\]", type_text)
    )
    signed = " signed" if re.search(r"\bsigned\b", type_text) else ""
    width_texts = [_dimension_width_text(item) for item in dimensions]
    if len(dimensions) >= 2 and all(text is not None for text in width_texts):
        product = "*".join(str(text) for text in width_texts)
        dim_text = f" [({product})-1:0]"
    elif dimensions:
        dim_text = " " + "".join(dimensions)
    else:
        dim_text = ""
    return f"{direction}{signed}{dim_text} {name}".strip()


def build_interface_contract(
    module_name: str,
    ports: list[Dict[str, str]],
    param_defaults: Dict[str, str] | None = None,
    *,
    verilog_2001: bool = False,
) -> str:
    """Render the frozen interface: port names, directions, widths, parameters.

    Generation and repair prompts previously described the interface in prose,
    so candidates invented widths: ``wt0``/``wt1`` came back 1-bit instead of
    5-bit, and the IMGpack 256/1024/128-bit buses were generalised to 8 bits.
    Every one of those runs failed the interface check before JasperGold ran.

    This block is derived from the golden port list and parameter defaults only.
    It contains no logic, so it fixes the contract without revealing the answer.

    With ``verilog_2001`` the widths are rendered in the flattened, type-free
    form the generation prompt demands, so the contract cannot contradict the
    coding rules the same prompt imposes.
    """

    lines = [f"Required module name: {module_name}"]
    defaults = dict(param_defaults or {})
    if defaults:
        lines.append("Required parameters (name = default):")
        lines.extend(
            f"  parameter {name} = {value}" for name, value in defaults.items()
        )
    lines.append(
        "Required ports, in this order, with exactly these directions and widths:"
    )
    for port in ports:
        lines.append(
            "  "
            + (
                _verilog_2001_port_declaration(port)
                if verilog_2001
                else _port_declaration_text(port)
            )
        )
    lines.append(
        "Declare every port exactly as listed. Do not rename, reorder, add, "
        "remove, resize, or re-sign any port, and do not replace a vector with "
        "a scalar."
    )
    return "\n".join(lines)


def golden_interface_contract(
    golden_path: str | Path,
    golden_top: str = "",
    *,
    verilog_2001: bool = True,
) -> str:
    """Derive the interface contract from a reference file, or return "".

    Used by the generation stage: the candidate must hit the reference interface
    on the first attempt, because a width or direction difference is rejected
    before JasperGold and therefore burns the whole route.  Only the interface is
    read; the reference body is never returned.

    ``golden_top`` is passed through to the same ``_select_verilog_module_source``
    rule the equivalence checkers use — an explicit name selects that module, an
    empty name keeps the historical first-module behaviour.  Sharing the rule is
    what makes the contract safe: it can never describe a different module than
    the one the proof will compare against.  The contract text names the module
    it was derived from, so a wrong ``golden_top`` is visible in the prompt
    artifact instead of silently mis-stating the interface.
    """

    try:
        path = Path(golden_path).expanduser().resolve()
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        selected = _select_verilog_module_source(text, golden_top)
        module_name, ports, _is_seq = parse_verilog_ports(selected)
        defaults = parse_verilog_parameter_defaults(selected)
        return build_interface_contract(
            module_name, ports, defaults, verilog_2001=verilog_2001
        )
    except Exception:
        # A contract is an aid, not a gate: generation must still run if the
        # reference cannot be parsed here.
        return ""


def _interface_diff(
    golden_ports: list[Dict[str, str]],
    candidate_ports: list[Dict[str, str]],
    parameter_defaults: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Return the port-level difference between golden and candidate."""

    golden_by_name = {str(port["name"]): port for port in golden_ports}
    candidate_by_name = {str(port["name"]): port for port in candidate_ports}
    missing = sorted(set(golden_by_name) - set(candidate_by_name))
    extra = sorted(set(candidate_by_name) - set(golden_by_name))
    mismatched = [
        {
            "name": name,
            "required": _port_declaration_text(golden_by_name[name]),
            "candidate": _port_declaration_text(candidate_by_name[name]),
        }
        for name in sorted(set(golden_by_name) & set(candidate_by_name))
        if not _port_shapes_compatible(
            golden_by_name[name], candidate_by_name[name], parameter_defaults
        )
    ]
    return {
        "missing_ports": [
            {"name": name, "required": _port_declaration_text(golden_by_name[name])}
            for name in missing
        ],
        "extra_ports": [
            {"name": name, "candidate": _port_declaration_text(candidate_by_name[name])}
            for name in extra
        ],
        "mismatched_ports": mismatched,
        "compatible": not (missing or extra or mismatched),
    }


def _validate_compatible_ports(
    golden_ports: list[Dict[str, str]],
    candidate_ports: list[Dict[str, str]],
    parameter_defaults: Optional[Dict[str, str]] = None,
) -> None:
    """Fail before JG when the candidate interface differs from the golden."""

    diff = _interface_diff(golden_ports, candidate_ports, parameter_defaults)
    if diff["compatible"]:
        return

    problems = []
    if diff["missing_ports"]:
        problems.append(
            "missing candidate ports: "
            + ", ".join(item["name"] for item in diff["missing_ports"])
        )
    if diff["extra_ports"]:
        problems.append(
            "extra candidate ports: "
            + ", ".join(item["name"] for item in diff["extra_ports"])
        )
    if diff["mismatched_ports"]:
        problems.append(
            "direction/width/signedness mismatch: "
            + ", ".join(
                f"{item['name']} (required: {item['required']}; "
                f"candidate: {item['candidate']})"
                for item in diff["mismatched_ports"]
            )
        )
    raise InterfaceMismatchError(
        "RTL interface mismatch; " + "; ".join(problems), diff
    )


def _select_verilog_module_source(verilog_src: str, module_name: str) -> str:
    """Return one module body, preserving the original source text.

    Golden files often contain helper modules before the reference top.  The
    historical parser intentionally handled the first module only; selecting
    a named top must therefore narrow the source before port/parameter parsing
    rather than merely renaming the first declaration.
    """

    requested = str(module_name or "").strip()
    if not requested:
        return verilog_src
    # Search the masked text so a module name or ``endmodule`` inside a comment
    # or a string literal cannot select or truncate the wrong region.  Masking
    # preserves offsets, so the slice below still returns the original source.
    masked = _mask_comments_and_strings(verilog_src)
    match = re.search(
        rf"\bmodule\s+{re.escape(requested)}\b", masked
    )
    if not match:
        raise ValueError(f"Golden top module not found: {requested}")
    end_match = re.search(r"\bendmodule\b", masked[match.start():], re.IGNORECASE)
    if not end_match:
        raise ValueError(f"Golden top module has no endmodule: {requested}")
    end = match.start() + end_match.end()
    return verilog_src[match.start():end]


def _effective_design_type(
    requested_design_type: str,
    inferred_sequential: bool,
) -> tuple[bool, str]:
    """Resolve explicit design type while retaining inference for old callers."""

    requested = str(requested_design_type or "").strip().lower()
    if not requested:
        return inferred_sequential, (
            "sequential" if inferred_sequential else "combinational"
        )
    if requested not in {"combinational", "sequential"}:
        raise ValueError(
            "design_type must be 'combinational' or 'sequential'"
        )
    return requested == "sequential", requested


def _module_parameter_section(verilog_src: str) -> str:
    """Return the balanced ``#(...)`` section of the first module."""
    # Masking blanks comments and string literals in one pass, so a parenthesis
    # inside either cannot move the balance counter.  Both masks preserve
    # offsets, so the returned slice comes from the comment-only text and keeps
    # a legitimate string parameter default intact.
    scan = _mask_comments_and_strings(verilog_src)
    source = _mask_comments_and_strings(verilog_src, mask_strings=False)
    start_match = re.search(r"\bmodule\s+\w+\s*#\s*\(", scan)
    if not start_match:
        return ""

    start = start_match.end()
    depth = 1
    for index in range(start, len(scan)):
        char = scan[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return source[start:index]
    return ""


def _split_top_level_commas(text: str) -> list[str]:
    """Split a parameter list without splitting expressions/concatenations."""
    chunks: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    quote = ""
    escaped = False
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char == '"':
            quote = char
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
        elif char == "," and not stack:
            chunks.append(text[start:index])
            start = index + 1
    chunks.append(text[start:])
    return chunks


def _parse_verilog_parameter_declarations(
    verilog_src: str,
) -> tuple[set[str], Dict[str, str]]:
    """Parse names and default expressions from an ANSI module parameter list."""
    section = _module_parameter_section(verilog_src)
    names: set[str] = set()
    defaults: Dict[str, str] = {}
    for raw_chunk in _split_top_level_commas(section):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        chunk = re.sub(r"^\s*(?:parameter|localparam)\b", "", chunk).strip()
        lhs, separator, rhs = chunk.partition("=")
        identifiers = re.findall(r"[A-Za-z_]\w*", lhs)
        if not identifiers:
            continue
        name = identifiers[-1]
        names.add(name)
        if separator and rhs.strip():
            defaults[name] = rhs.strip()
    return names, defaults


def parse_verilog_parameters(verilog_src: str) -> set[str]:
    """Extract module parameter names from ANSI-style parameter lists."""
    names, _defaults = _parse_verilog_parameter_declarations(verilog_src)
    return names


def parse_verilog_parameter_defaults(verilog_src: str) -> Dict[str, str]:
    """Extract module parameter defaults while preserving declaration order."""
    _names, defaults = _parse_verilog_parameter_declarations(verilog_src)
    return defaults


def _infer_clock_name(ports: list[Dict[str, str]]) -> str:
    """Best-effort clock port detection."""
    exact_names = {"clk", "clk_i", "clock", "pclk", "nvdla_core_clk"}
    for p in ports:
        if p["dir"] == "input" and p["name"].lower() in exact_names:
            return p["name"]
    for p in ports:
        lower = p["name"].lower()
        if p["dir"] == "input" and (
            lower.endswith(("_clk", "_clock")) or lower.startswith("clk_")
        ):
            return p["name"]
    return "clk_i"


def _infer_reset_name(ports: list[Dict[str, str]]) -> Optional[str]:
    """Best-effort reset port detection."""
    resets = [
        p["name"]
        for p in ports
        if p["dir"] == "input" and ("rst" in p["name"].lower() or "reset" in p["name"].lower())
    ]
    return resets[0] if resets else None


def _is_active_low_reset(name: str) -> bool:
    """Heuristic for rst_n / rst_ni naming."""
    tail = name.lower().replace("reset", "").replace("rst", "")
    return "n" in tail or "_ni" in name.lower()


def generate_fpv_wrapper(
    ref_module: str,
    opt_module: str,
    ports: list[Dict[str, str]],
    is_sequential: bool,
    ref_params: set[str] | None = None,
    opt_params: set[str] | None = None,
    ref_param_defaults: Dict[str, str] | None = None,
) -> str:
    """Generate FPV wrapper that asserts equivalence between ref and opt modules."""
    lines = []
    fpv_name = f"{ref_module}_FPV"
    ref_params = ref_params or set()
    opt_params = opt_params or set()
    wrapper_param_defaults = dict(ref_param_defaults or {})

    # Keep compatibility with the historical NVDLA wrapper even if a caller
    # only supplied parameter names.  Other parameters must retain the golden
    # RTL's actual default rather than receiving a guessed value.
    if (
        any("NUM_HW_APPS" in p.get("type", "") for p in ports)
        or "NUM_HW_APPS" in ref_params
    ):
        wrapper_param_defaults.setdefault("NUM_HW_APPS", "2")

    if wrapper_param_defaults:
        lines.append(f"module {fpv_name} #(")
        declarations = [
            f"    parameter {name} = {default}"
            for name, default in wrapper_param_defaults.items()
        ]
        lines.append(",\n".join(declarations))
        lines.append(f")(")
    else:
        lines.append(f"module {fpv_name}(")

    # Port declarations
    port_decls = []
    for p in ports:
        if p["dir"] == "input":
            port_decls.append(f"    input {p['type']} {p['name']}")
        else:
            port_decls.append(f"    output {p['type']} {p['name']}_ref")
            port_decls.append(f"    output {p['type']} {p['name']}_opt")

    lines.append(",\n".join(port_decls))
    lines.append(");")
    lines.append("")

    def emit_instance(module_name: str, inst_name: str, module_params: set[str], suffix: str) -> None:
        forwarded_params = [
            name for name in wrapper_param_defaults if name in module_params
        ]
        if forwarded_params:
            lines.append(f"    {module_name} #(")
            parameter_connections = [
                f"        .{name}({name})" for name in forwarded_params
            ]
            lines.append(",\n".join(parameter_connections))
            lines.append(f"    ) {inst_name}(")
        else:
            lines.append(f"    {module_name} {inst_name}(")

        conns = []
        for port in ports:
            if port["dir"] == "input":
                conns.append(f"        .{port['name']}({port['name']})")
            else:
                conns.append(f"        .{port['name']}({port['name']}_{suffix})")
        lines.append(",\n".join(conns))
        lines.append("    );")
        lines.append("")

    # Instantiate reference module
    emit_instance(ref_module, "u_ref", ref_params, "ref")

    # Instantiate optimized module
    emit_instance(opt_module, "u_opt", opt_params, "opt")

    # Assertions
    outputs = [p for p in ports if p["dir"] == "output"]
    if is_sequential:
        clk_name = _infer_clock_name(ports)
        rst = _infer_reset_name(ports)
        for o in outputs:
            safe = re.sub(r"\W", "_", o["name"])
            lines.append(f"    property p_eq_{safe};")
            if rst:
                if _is_active_low_reset(rst):
                    lines.append(f"        @(posedge {clk_name}) disable iff (!{rst})")
                else:
                    lines.append(f"        @(posedge {clk_name}) disable iff ({rst})")
            else:
                lines.append(f"        @(posedge {clk_name})")
            lines.append(f"            {o['name']}_ref == {o['name']}_opt;")
            lines.append(f"    endproperty")
            lines.append(f"    a_eq_{safe}: assert property (p_eq_{safe});")
            lines.append("")
    else:
        # Combinational
        for o in outputs:
            safe = re.sub(r"\W", "_", o["name"])
            lines.append(
                f"    a_eq_{safe}: assert property "
                f"({o['name']}_ref == {o['name']}_opt);"
            )

    lines.append("endmodule")
    return "\n".join(lines)


def _both_edge_clock_reasons(*sources: str) -> list[str]:
    """Report constructs that make the declared clock active on both edges.

    ``clock <clk>`` tells JasperGold the design only reacts to the rising edge.
    The NVDLA references break that assumption in two ways:

    * ``CKLNQD12`` style clock gating latches its enable in
      ``always @(negedge CP)`` and drives ``Q = CP & qd``;
    * behavioural RAM models derive a read clock through a delayed assignment,
      ``assign #0.1 r0_clk_d0p1 = r0_clk; assign r0_clk_read = r0_clk_d0p1 & r0_clk``.

    Both are derived from the declared clock, so JasperGold reported ``ECK112``
    (activity on the negedge of the fast clock) and the proof failed with
    ``EPF059``.  The honest fix is to declare the clock completely; masking the
    ``negedge`` blocks or black-boxing the RAM would hide real behaviour.
    """

    reasons: list[str] = []
    for source in sources:
        masked = _mask_comments_and_strings(str(source or ""))
        for event_match in re.finditer(
            r"\balways(?:_ff)?\s*@\s*\(([^)]*)\)",
            masked,
            re.IGNORECASE,
        ):
            event_text = event_match.group(1)
            negedge_signals = re.findall(
                r"\bnegedge\s+([A-Za-z_]\w*)", event_text, re.IGNORECASE
            )
            if not negedge_signals:
                continue
            # ``always @(posedge clk or negedge rst_n)`` is ordinary
            # asynchronous reset logic.  Only a falling-edge event that is
            # itself clock-like (or otherwise not reset-like) is evidence for
            # a both-edge clock model.  This keeps the BDMA CKLNQD12
            # ``always @(negedge CP)`` evidence while excluding reset blocks.
            if all(
                re.search(
                    r"(?:^|_)(?:rst|rstn|reset|resetn|clr|clrn|clear|por)(?:_|$)",
                    signal,
                    re.IGNORECASE,
                )
                for signal in negedge_signals
            ):
                continue
            reasons.append("negedge-sensitive always block")
        if re.search(
            r"assign\s*#\s*[0-9._]+\s+\w*clk\w*\s*=", masked, re.IGNORECASE
        ):
            reasons.append("delay-derived clock signal")
    # Preserve first-seen order while removing duplicates across both sources.
    return list(dict.fromkeys(reasons))


def generate_fpv_tcl(
    remote_dir: str,
    ref_filename: str,
    opt_filename: str,
    fpv_filename: str,
    fpv_module: str,
    is_sequential: bool,
    ports: list[Dict[str, str]],
    analyze_defines: Tuple[str, ...] | list[str] = (),
    clock_both_edges: bool = False,
) -> str:
    """Generate JasperGold TCL script for equivalence checking."""
    lines = []
    define_prefix = _analyze_define_prefix(analyze_defines)
    lines.append(f"cd {remote_dir}")
    lines.append("")
    # The same macro set as the Icarus closure gate, so both tools elaborate
    # the same ``ifdef`` branch of the reference.
    lines.append(f"analyze -sv {define_prefix}{ref_filename}")
    lines.append(f"analyze -sv {define_prefix}{opt_filename}")
    lines.append(f"analyze -sv {define_prefix}{fpv_filename}")
    lines.append("")
    # Keep arithmetic operators concrete during equivalence checking.  Without
    # this switch JasperGold may automatically black-box multipliers, leaving
    # corresponding operations in the reference and candidate unconstrained
    # independently and producing spurious counterexamples.
    lines.append(f"elaborate -disable_auto_bbox -top {fpv_module}")
    lines.append("")

    if is_sequential:
        clk_name = _infer_clock_name(ports)
        # A design whose gating cells or behavioural RAM models react to the
        # falling edge of this same clock must be declared on both edges;
        # otherwise JasperGold reports ECK112 and the proof cannot converge.
        if clock_both_edges:
            lines.append(f"clock {clk_name} -both_edges")
        else:
            lines.append(f"clock {clk_name}")
        rst = _infer_reset_name(ports)
        if rst:
            if _is_active_low_reset(rst):
                lines.append(f"reset -expression {{!{rst}}}")
            else:
                lines.append(f"reset -expression {{{rst}}}")
        lines.append("")
    else:
        lines.append("clock -none")
        lines.append("reset -none")
        lines.append("")

    lines.append("prove -all")
    lines.append("")

    # Persist machine-readable per-property status and compact counterexample
    # traces in the shared attempt directory.  The repair prompt consumes these
    # files without exposing the golden RTL source.
    lines.append('set codex_property_fh [open "fpv_property_results.tsv" "w"]')
    lines.append('puts $codex_property_fh "property\\tstatus\\ttrace_length"')
    lines.append('set codex_assertions [get_property_list -include {type assert}]')
    lines.append('foreach codex_prop $codex_assertions {')
    lines.append('    set codex_status [get_property_info -list {validity_status} $codex_prop]')
    lines.append('    set codex_trace_length 0')
    lines.append(
        '    catch {set codex_trace_length '
        '[get_property_info -list {trace_length} $codex_prop]}'
    )
    lines.append(
        '    puts $codex_property_fh '
        '"$codex_prop\\t$codex_status\\t$codex_trace_length"'
    )
    lines.append(
        '    puts "CODEX_PROPERTY_RESULT\\t$codex_prop\\t$codex_status\\t$codex_trace_length"'
    )
    lines.append(
        '    if {$codex_status eq "cex" || $codex_status eq "ar_cex"} {'
    )
    lines.append(
        '        set codex_safe_prop [string map {":" "_" "/" "_" "." "_" '
        '"[" "_" "]" "_"} $codex_prop]'
    )
    lines.append('        set codex_window "cex_$codex_safe_prop"')
    lines.append(
        '        if {![catch {visualize -violation -property $codex_prop '
        '-batch -silent -new_window $codex_window} codex_viz_error]} {'
    )
    signal_names: list[str] = []
    for port in ports:
        if port["dir"] == "input":
            signal_names.append(f"{fpv_module}.{port['name']}")
        else:
            signal_names.append(f"{fpv_module}.{port['name']}_ref")
            signal_names.append(f"{fpv_module}.{port['name']}_opt")
    tcl_signal_list = " ".join("{" + name + "}" for name in signal_names)
    lines.append(f"            set codex_signals [list {tcl_signal_list}]")
    lines.append('            foreach codex_sig $codex_signals {')
    lines.append(
        '                catch {visualize -add_sig $codex_sig -window $codex_window}'
    )
    lines.append('            }')
    lines.append(
        '            set codex_trace_fh '
        '[open "cex_${codex_safe_prop}.trace.tsv" "w"]'
    )
    lines.append('            puts $codex_trace_fh "property\\t$codex_prop"')
    lines.append('            puts $codex_trace_fh "status\\t$codex_status"')
    lines.append('            set codex_viz_length $codex_trace_length')
    lines.append(
        '            catch {set codex_viz_length '
        '[visualize -get_length -window $codex_window]}'
    )
    lines.append(
        '            puts $codex_trace_fh "trace_length\\t$codex_viz_length"'
    )
    lines.append('            foreach codex_sig $codex_signals {')
    lines.append('                set codex_values ""')
    lines.append(
        '                catch {set codex_values [visualize -get_value '
        '$codex_sig 0:$codex_viz_length -radix hex -window $codex_window]}'
    )
    lines.append(
        '                puts $codex_trace_fh '
        '"signal\\t$codex_sig\\t$codex_values"'
    )
    lines.append('            }')
    lines.append('            close $codex_trace_fh')
    lines.append(
        '            catch {visualize -save -vcd '
        '"cex_${codex_safe_prop}.vcd" -force -window $codex_window}'
    )
    lines.append('        } else {')
    lines.append(
        '            puts "CODEX_CEX_EXPORT_ERROR\\t$codex_prop\\t$codex_viz_error"'
    )
    lines.append('        }')
    lines.append('    }')
    lines.append('}')
    lines.append('close $codex_property_fh')
    lines.append("")
    return "\n".join(lines)


def run_jaspergold(
    ref_v_path: Path,
    opt_v_path: Path,
    fpv_sv_path: Path,
    tcl_path: Path,
    jg_config: Dict[str, Any],
    local_base: Path,
) -> Dict[str, Any]:
    """
    Run JasperGold equivalence check on remote server via SSH.
    Returns dict with keys: success, proven, cex, undetermined, error, raw_output.
    """
    remote_base = jg_config["remote_base"]
    remote_user = jg_config["remote_user"]
    remote_host = jg_config["remote_host"]
    timeout = jg_config["timeout"]

    remote_tcl = _local_to_remote(tcl_path, local_base, remote_base)
    remote_dir = str(Path(remote_tcl).parent)
    project_name = f"jgproject_{opt_v_path.stem}"
    local_project_dir = opt_v_path.parent / project_name

    # Validate the executable and construct the prefix before any SSH cleanup
    # is attempted.  This keeps direct callers fail-closed when they provide a
    # hand-built/incomplete config dictionary.
    _remote_jg_shell_prefix(jg_config)

    # Clean up stale local shared-directory artifacts before invoking JasperGold.
    _cleanup_local_jgproject(local_project_dir)

    # Clean up stale project directories
    _cleanup_remote_jgproject(remote_dir, remote_tcl, project_name, jg_config)

    cmd = [
        "ssh",
        f"{remote_user}@{remote_host}",
        _build_remote_jg_command(
            remote_dir=remote_dir,
            remote_tcl=remote_tcl,
            project_name=project_name,
            jg_config=jg_config,
        ),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
        )
        output = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        recovered = _recover_jg_result_from_session_log(local_project_dir)
        _cleanup_remote_jgproject(remote_dir, remote_tcl, project_name, jg_config)
        if recovered is not None:
            return _attach_jg_diagnostics(
                recovered, opt_v_path.parent, local_project_dir
            )
        return _attach_jg_diagnostics(
            {"success": False, "error": "timeout", "raw_output": ""},
            opt_v_path.parent,
            local_project_dir,
        )

    # Parse results
    result = {
        "success": False,
        "proven": 0,
        "cex": 0,
        "undetermined": 0,
        "error": "",
        "raw_output": output[-3000:] if len(output) > 3000 else output,
    }

    # Extract SUMMARY block
    summary_m = re.search(
        r"(=+\s*\n\s*SUMMARY\s*\n\s*=+.*?)(?=\n\s*=+\s*\n|\Z)",
        output,
        re.DOTALL | re.IGNORECASE,
    )
    if summary_m:
        summary = summary_m.group(1)
        for key in ("proven", "cex", "undetermined", "unknown"):
            m = re.search(rf"-\s*{key}\s*:\s*(\d+)", summary, re.IGNORECASE)
            if m:
                result[key] = int(m.group(1))

        m = re.search(r"assertions\s*:\s*(\d+)", summary, re.IGNORECASE)
        assertions = int(m.group(1)) if m else 0

        if assertions > 0 and assertions == result["proven"]:
            result["success"] = True
        elif result["cex"] > 0:
            result["error"] = "counterexample found"
        elif result.get("unknown", 0) > 0:
            result["error"] = "unknown (JasperGold engine error)"
        elif result["undetermined"] > 0:
            result["error"] = "undetermined"
    else:
        # Check for errors
        lock_lines = [
            ln.strip()
            for ln in output.splitlines()
            if "cannot obtain ownership of project directory" in ln.lower()
        ]
        err_lines = [ln.strip() for ln in output.splitlines() if "ERROR" in ln and "VERI-" in ln]

        if lock_lines:
            result["error"] = "project directory lock: " + lock_lines[0]
            _cleanup_remote_jgproject(remote_dir, remote_tcl, project_name, jg_config)
        elif err_lines:
            result["error"] = "compile error: " + err_lines[0]
        else:
            recovered = _recover_jg_result_from_session_log(local_project_dir)
            if recovered is not None:
                return _attach_jg_diagnostics(
                    recovered, opt_v_path.parent, local_project_dir
                )
            result["error"] = "no SUMMARY in output"

    return _attach_jg_diagnostics(result, opt_v_path.parent, local_project_dir)


def _normalise_jg_verify_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the lower-level JG result into the verify-only API schema."""
    equivalent = bool(raw_result.get("success", False))
    error = str(raw_result.get("error", "") or "")
    if equivalent:
        status = "passed"
    elif "timeout" in error.lower():
        status = "timeout"
    elif (
        raw_result.get("cex", 0)
        or raw_result.get("undetermined", 0)
        or any(
            marker in error.lower()
            for marker in ("counterexample", "undetermined", "not equivalent")
        )
    ):
        status = "failed"
    else:
        status = "error"

    def _count(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "status": status,
        "equivalent": equivalent,
        # Keep ``success`` for callers that consume the historical
        # ``run_jaspergold`` shape while exposing the explicit status above.
        "success": equivalent,
        "proven": _count(raw_result.get("proven")),
        "cex": _count(raw_result.get("cex")),
        "undetermined": _count(raw_result.get("undetermined")),
        "error": error,
        "raw_output": str(raw_result.get("raw_output", "") or ""),
        "properties": list(raw_result.get("properties") or []),
        "counterexamples": list(raw_result.get("counterexamples") or []),
        "diagnostic_summary": str(
            raw_result.get("diagnostic_summary", "") or ""
        ),
        "property_report_path": str(
            raw_result.get("property_report_path", "") or ""
        ),
        "session_log_path": str(raw_result.get("session_log_path", "") or ""),
    }


# Stage classification patterns.  Order matters: a session that never finished
# elaboration cannot have produced a meaningful proof result, so compile-stage
# evidence outranks the property counts.
_COMPILE_STAGE_PATTERNS: Tuple[str, ...] = (
    r"ERROR \(VERI-",
    r"ERROR \(ENL0",
    r"Unable to elaborate",
    r"\bsyntax error\b",
    r"cannot find (?:module|design unit)",
)
_CLOCK_STAGE_PATTERNS: Tuple[str, ...] = (
    r"\(ECK1\d+\)",
    r"activity on the (?:negedge|posedge)",
)
_INFRA_STAGE_PATTERNS: Tuple[str, ...] = (
    r"no SUMMARY in output",
    r"ssh: connect",
    r"Permission denied",
    r"license",
)


def classify_failure_stage(result: Dict[str, Any]) -> str:
    """Name the stage a JasperGold attempt failed in.

    The original run reported 39 failures as one undifferentiated pile, so an
    interface rejection, a macro-driven elaboration failure, an incomplete clock
    declaration and a genuine counterexample all looked alike.  They need
    different fixes, so they must be counted separately.

    Returned stages: ``none`` (passed), ``interface``, ``compile``, ``clock``,
    ``timeout``, ``counterexample``, ``undetermined``, ``infrastructure``,
    ``unknown``.
    """

    status = str(result.get("status", "") or "")
    if status == "passed" or result.get("equivalent"):
        return "none"
    diff = result.get("interface_diff") or {}
    if isinstance(diff, dict) and diff and diff.get("compatible") is False:
        return "interface"
    text = "\n".join(
        str(result.get(key, "") or "")
        for key in ("error", "diagnostic_summary", "raw_output")
    )
    if status == "timeout" or "timeout" in str(result.get("error", "")).lower():
        return "timeout"
    for pattern in _COMPILE_STAGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "compile"
    for pattern in _CLOCK_STAGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "clock"
    try:
        cex = int(result.get("cex") or 0)
        undetermined = int(result.get("undetermined") or 0)
    except (TypeError, ValueError):
        cex, undetermined = 0, 0
    if cex > 0:
        return "counterexample"
    if undetermined > 0:
        return "undetermined"
    for pattern in _INFRA_STAGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "infrastructure"
    return "unknown"


_INSTANCE_RE = re.compile(
    r"^[ \t]*([A-Za-z_]\w*)[ \t]+(?:#[ \t]*\([^;]*?\)[ \t]*)?([A-Za-z_]\w*)[ \t]*\(",
    re.MULTILINE,
)
# Words that can legally start a line and be followed by an identifier and an
# open parenthesis without being a module instantiation.
_NOT_AN_INSTANCE = frozenset(
    """
    module endmodule function endfunction task endtask primitive endprimitive
    generate endgenerate begin end case casex casez endcase if else for while
    repeat forever wait fork join always always_ff always_comb always_latch
    initial assign deassign force release disable input output inout reg wire
    logic integer real time realtime genvar parameter localparam defparam
    specify endspecify specparam table endtable posedge negedge edge signed
    unsigned automatic static return break continue typedef struct union enum
    packed interface endinterface package endpackage import export default
    and or nand nor xor xnor not buf bufif0 bufif1 notif0 notif1 pmos nmos
    cmos rpmos rnmos rcmos tran tranif0 tranif1 rtran rtranif0 rtranif1
    pullup pulldown supply0 supply1 tri triand trior tri0 tri1 trireg
    """.split()
)


def prescreen_rtl_artifact(
    verilog_text: str, module_name: str = ""
) -> Dict[str, Any]:
    """Reject structurally broken repair artifacts before JasperGold runs.

    Several repair retries in the failed run reached JasperGold carrying
    markdown prose, a module emitted twice, an unpacked array in the port list,
    or an instantiation of a module that was never defined.  Those are local,
    cheap-to-detect defects; sending them to a remote formal tool spent a full
    proof slot and returned a tool error that the next repair prompt then had to
    interpret as if it were a functional bug.

    Returns ``{"ok", "problems", "declared_modules"}``.  ``problems`` entries
    carry ``kind``, ``line`` and ``detail`` so the repair prompt can point at the
    exact defect.
    """

    text = str(verilog_text or "")
    masked = _mask_comments_and_strings(text)
    problems: list[Dict[str, Any]] = []

    def add(kind: str, line: int, detail: str) -> None:
        problems.append({"kind": kind, "line": line, "detail": detail[:200]})

    for lineno, line in enumerate(masked.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            add("markdown_fence", lineno, stripped)
        elif re.match(r"^#{1,6}\s+[A-Za-z]", stripped):
            add("markdown_heading", lineno, stripped)
        elif re.match(r"^\*\*\S", stripped):
            add("markdown_emphasis", lineno, stripped)

    declared = [
        (match.group(1), masked[: match.start()].count("\n") + 1)
        for match in re.finditer(r"\bmodule\s+([A-Za-z_]\w*)", masked)
    ]
    declared_names = [name for name, _line in declared]
    end_count = len(re.findall(r"\bendmodule\b", masked))
    if len(declared_names) != end_count:
        add(
            "module_endmodule_imbalance",
            0,
            f"{len(declared_names)} module declarations vs {end_count} endmodule",
        )
    seen: set[str] = set()
    for name, lineno in declared:
        if name in seen:
            add("duplicate_module", lineno, f"module {name} declared more than once")
        seen.add(name)
    target = str(module_name or "").strip()
    if target and target not in declared_names:
        add(
            "missing_target_module",
            0,
            f"required module {target} is not declared in this artifact",
        )

    for match in re.finditer(
        r"\b(input|output|inout)\b[^;,)]*?\b([A-Za-z_]\w*)\s*(\[[^\]]*\])\s*(?=[,;)])",
        masked,
    ):
        lineno = masked[: match.start()].count("\n") + 1
        add(
            "unpacked_array_port",
            lineno,
            f"port {match.group(2)} carries an unpacked dimension "
            f"{match.group(3)}; ports must be packed vectors",
        )

    defined = set(declared_names)
    for match in _INSTANCE_RE.finditer(masked):
        type_name, instance_name = match.group(1), match.group(2)
        if type_name in _NOT_AN_INSTANCE or instance_name in _NOT_AN_INSTANCE:
            continue
        if type_name in defined:
            continue
        lineno = masked[: match.start()].count("\n") + 1
        add(
            "undefined_module_instance",
            lineno,
            f"instance {instance_name} uses module {type_name}, which this "
            "artifact does not define",
        )

    return {
        "ok": not problems,
        "problems": problems,
        "declared_modules": declared_names,
    }


def build_structural_repair_prompt(
    *,
    module_name: str,
    interface_contract: str,
    previous_verilog: str,
    prescreen: Dict[str, Any],
) -> str:
    """Ask for a structurally valid artifact without mentioning golden logic."""

    problem_lines = "\n".join(
        f"  line {item['line']}: {item['kind']} — {item['detail']}"
        for item in prescreen.get("problems", [])
    )
    return textwrap.dedent(
        f"""\
The previous artifact was rejected before verification because it is not a
single valid Verilog-2001 module. No formal verification was attempted.

Structural problems found:
{problem_lines}

{interface_contract}

Previous artifact:
```verilog
{previous_verilog}
```

Requirements:
1. Return exactly one complete module named {module_name}, ending with
   endmodule, and define every module it instantiates in the same file.
2. Remove all markdown, headings, prose and code fences. Output Verilog only.
3. Declare each port once, as a packed vector; no unpacked array ports.
4. Keep the interface above unchanged.
5. Do not change the intended function while fixing the structure.
"""
    )


def summarise_error_progression(attempts: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Record how the failure changed from the first attempt to the retries.

    Without this, a retry that swapped one failure for another looked identical
    in the results to a retry that made progress, and repeated identical
    failures were not visible at all.
    """

    steps = [
        {
            "attempt": item.get("attempt"),
            "stage": item.get("failure_stage", ""),
            "error": str(item.get("error", "") or "")[:300],
        }
        for item in attempts
    ]
    stages = [step["stage"] for step in steps]
    return {
        "steps": steps,
        "first_stage": stages[0] if stages else "",
        "last_stage": stages[-1] if stages else "",
        "stage_changed": bool(stages) and len(set(stages)) > 1,
        "repeated_identical_error": len(
            {(step["stage"], step["error"]) for step in steps}
        )
        < len(steps),
    }


def verify_rtl_equivalence_with_jg(
    candidate_path: str | Path,
    golden_path: str | Path,
    output_dir: str | Path,
    env_path: str | Path,
    *,
    golden_top: str = "",
    design_type: str = "",
    verification_timeout: int | None = None,
) -> Dict[str, Any]:
    """Verify one candidate RTL against a golden RTL using JasperGold once.

    This is intentionally a verify-only entry point: it creates the remote
    input files, FPV wrapper, and TCL script, invokes :func:`run_jaspergold`
    exactly once, and never calls an LLM or performs a correction retry.  The
    returned ``status`` is one of ``passed``, ``failed``, ``timeout``, or
    ``error`` and is suitable for symmetric use by independent RTL routes.

    The generated files live in ``output_dir``.  As with the existing Module 5
    flow, that directory must be under ``PROJECT_ROOT`` so its local path can be
    mapped to the configured shared remote workspace.
    """
    # Populated while the setup progresses so the failure path can report the
    # interface contract and the stage that failed instead of a bare message.
    setup_info: Dict[str, Any] = {
        "stage": "setup",
        "interface_contract": "",
        "interface_diff": {},
    }
    try:
        jg_config = _with_verification_timeout(
            _load_jg_config(env_path), verification_timeout
        )
        candidate = Path(candidate_path).expanduser().resolve()
        golden = Path(golden_path).expanduser().resolve()
        out_dir = Path(output_dir).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"candidate RTL does not exist: {candidate}")
        if not golden.is_file():
            raise FileNotFoundError(f"golden RTL does not exist: {golden}")

        candidate_text = candidate.read_text(encoding="utf-8", errors="replace")
        golden_text = golden.read_text(encoding="utf-8", errors="replace")
        selected_golden = _select_verilog_module_source(golden_text, golden_top)
        golden_module, golden_ports, inferred_seq = parse_verilog_ports(
            selected_golden
        )
        is_seq, resolved_design_type = _effective_design_type(
            design_type, inferred_seq
        )
        golden_params = parse_verilog_parameters(selected_golden)
        golden_param_defaults = parse_verilog_parameter_defaults(selected_golden)
        setup_info["interface_contract"] = build_interface_contract(
            golden_module, golden_ports, golden_param_defaults
        )
        # Validate the complete candidate interface before generating any JG
        # files.  This prevents missing or mismatched ports from becoming
        # silently floating signals in the equivalence wrapper.
        setup_info["stage"] = "interface"
        _candidate_module, candidate_ports, _candidate_is_seq = (
            parse_verilog_ports(candidate_text)
        )
        setup_info["interface_diff"] = _interface_diff(
            golden_ports, candidate_ports, golden_param_defaults
        )
        _validate_compatible_ports(
            golden_ports, candidate_ports, golden_param_defaults
        )
        setup_info["stage"] = "elaborate"

        out_dir.mkdir(parents=True, exist_ok=True)
        ref_v_path = out_dir / f"{golden_module}_ref.v"
        opt_v_path = out_dir / "candidate_opt.v"
        fpv_sv_path = out_dir / "equivalence_FPV.sv"
        tcl_path = out_dir / "equivalence_FPV.tcl"
        opt_module_name = f"{golden_module}_opt"
        candidate_renamed = re.sub(
            r"\bmodule\s+\w+",
            f"module {opt_module_name}",
            candidate_text,
            count=1,
        )
        opt_params = parse_verilog_parameters(candidate_renamed)

        ref_v_path.write_text(golden_text, encoding="utf-8")
        opt_v_path.write_text(candidate_renamed, encoding="utf-8")
        fpv_sv_path.write_text(
            generate_fpv_wrapper(
                golden_module,
                opt_module_name,
                golden_ports,
                is_seq,
                ref_params=golden_params,
                opt_params=opt_params,
                ref_param_defaults=golden_param_defaults,
            ),
            encoding="utf-8",
        )

        remote_dir = _local_to_remote(
            out_dir,
            PROJECT_ROOT,
            jg_config["remote_base"],
        )
        both_edge_reasons = (
            _both_edge_clock_reasons(golden_text, candidate_text) if is_seq else []
        )
        tcl_path.write_text(
            generate_fpv_tcl(
                remote_dir=remote_dir,
                ref_filename=ref_v_path.name,
                opt_filename=opt_v_path.name,
                fpv_filename=fpv_sv_path.name,
                fpv_module=f"{golden_module}_FPV",
                is_sequential=is_seq,
                ports=golden_ports,
                analyze_defines=tuple(jg_config.get("analyze_defines") or ()),
                clock_both_edges=bool(both_edge_reasons),
            ),
            encoding="utf-8",
        )

        raw_result = run_jaspergold(
            ref_v_path,
            opt_v_path,
            fpv_sv_path,
            tcl_path,
            jg_config,
            PROJECT_ROOT,
        )
        result = _normalise_jg_verify_result(raw_result)
        result.update(
            {
                "golden_top": golden_module,
                "design_type": resolved_design_type,
                "verification_timeout": int(jg_config["timeout"]),
                "analyze_defines": list(jg_config.get("analyze_defines") or ()),
                "clock_both_edges": bool(both_edge_reasons),
                "clock_both_edges_reasons": both_edge_reasons,
                "interface_contract": setup_info["interface_contract"],
                "interface_diff": setup_info["interface_diff"],
                "failure_stage": classify_failure_stage(result),
            }
        )
        return result
    except Exception as exc:
        # A verify-only caller receives a deterministic result and, crucially,
        # no SSH/JG invocation is attempted when setup or configuration fails.
        return {
            "status": "error",
            "equivalent": False,
            "success": False,
            "proven": 0,
            "cex": 0,
            "undetermined": 0,
            "error": str(exc),
            "raw_output": "",
            "properties": [],
            "counterexamples": [],
            "diagnostic_summary": "",
            "property_report_path": "",
            "session_log_path": "",
            "golden_top": str(golden_top or ""),
            "design_type": str(design_type or ""),
            "verification_timeout": verification_timeout,
            "interface_contract": setup_info["interface_contract"],
            "interface_diff": setup_info["interface_diff"],
            "failure_stage": (
                "interface"
                if isinstance(exc, InterfaceMismatchError)
                else setup_info["stage"]
            ),
        }


def _general_rtl_rules(start: int) -> str:
    """The shared Verilog constraints, numbered to continue a prompt's list.

    Imported lazily because ``rtl_direct_runner`` imports this module, so a
    top-level import would be circular.
    """
    from module5.rtl_direct_runner import _numbered_rules

    return _numbered_rules(start)


def build_jg_retry_prompt(
    *,
    source_label: str,
    source_context: str,
    previous_verilog: str,
    jg_result: Dict[str, Any],
    module_name: str,
    interface_contract: str = "",
) -> str:
    """Build a JG repair prompt without exposing the golden implementation."""
    feedback = {
        key: jg_result.get(key)
        for key in (
            "status",
            "equivalent",
            "proven",
            "cex",
            "undetermined",
            "error",
            "failure_stage",
        )
    }
    raw_tail = str(jg_result.get("raw_output", "") or "")[-4000:]
    properties = list(jg_result.get("properties") or [])
    counterexamples = list(jg_result.get("counterexamples") or [])
    diagnostic_summary = str(jg_result.get("diagnostic_summary", "") or "")
    general_rules = _general_rtl_rules(6)
    # The interface is a contract, not an answer: the candidate must match these
    # port names, directions and widths exactly, and repeated runs failed the
    # pre-JG interface check because the prompt never stated them.
    contract_block = (
        f"\nFrozen interface contract (must match exactly):\n{interface_contract}\n"
        if str(interface_contract or "").strip()
        else ""
    )
    interface_diff = jg_result.get("interface_diff") or {}
    diff_block = ""
    if isinstance(interface_diff, dict) and interface_diff.get("compatible") is False:
        diff_block = (
            "\nInterface differences detected in the previous candidate:\n"
            + json.dumps(interface_diff, indent=2, ensure_ascii=False)
            + "\n"
        )
    return textwrap.dedent(
        f"""\
The previous RTL candidate did not pass JasperGold equivalence verification.
Regenerate one complete corrected RTL module from the original route source and
the formal-verification feedback below.

Required top module name: {module_name}
{contract_block}{diff_block}
Original route source ({source_label}):
{source_context}

Previous RTL candidate:
```verilog
{previous_verilog}
```

JasperGold result:
{json.dumps(feedback, indent=2, ensure_ascii=False)}

Property-level results:
{json.dumps(properties, indent=2, ensure_ascii=False)}

Compact counterexample traces:
{json.dumps(counterexamples, indent=2, ensure_ascii=False)}

Diagnostic summary:
{diagnostic_summary}

JasperGold output tail:
{raw_tail}

Hard requirements:
1. Preserve the exact top-level interface and cycle-level behavior required by
   the original route source.
2. Correct the functional cause identified by the failed property and trace.
   Preserve reset polarity/synchrony, output latency, valid-signal alignment,
   and FSM priority from the authoritative specification/feature contract; if
   the previous RTL violates that contract, repair it rather than preserving
   the previous candidate's behavior.
3. Return exactly one complete synthesizable Verilog-2001 module named
   {module_name}.
4. Use reg/wire and Verilog-2001 constructs accepted by Icarus and Design
   Compiler. Do not use logic, always_ff, always_comb, typedef, struct, or
   interface.
5. Output Verilog only, with no markdown fences or explanation.
{general_rules}

The golden RTL implementation is intentionally not provided. Repair the design
from the original source contract and the formal feedback.
"""
    )


def _load_jg_retry_client(
    env_path: str | Path,
    llm_client: Any = None,
    llm_model: str = "",
) -> tuple[Any, str]:
    if llm_client is not None:
        return llm_client, llm_model or os.environ.get("OPENAI_MODEL", "")
    load_dotenv(env_path, override=True)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or ""
    ).strip()
    model = (
        llm_model
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("LLM_MODEL")
        or ""
    ).strip()
    if not api_key or not base_url or not model:
        raise RuntimeError(
            "JG repair requires OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL"
        )
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        **openai_client_kwargs(),
    ), model


def verify_rtl_with_jg_retry(
    candidate_path: str | Path,
    golden_path: str | Path,
    output_dir: str | Path,
    env_path: str | Path,
    *,
    source_label: str,
    source_context: str,
    module_name: str = "",
    golden_top: str = "",
    design_type: str = "",
    verification_timeout: int | None = None,
    max_retries: int = 1,
    llm_client: Any = None,
    llm_model: str = "",
    max_tokens: int | None = None,
) -> Dict[str, Any]:
    """Run syntax -> JG and optionally regenerate RTL from JG feedback.

    ``max_retries`` counts JG-driven LLM regenerations, not the initial JG
    attempt.  With the default value, at most two JG invocations and one repair
    LLM conversation occur.  The golden implementation is used only by JG and
    is never included in the repair prompt.
    """
    from RTL_DIRECT_compare.validator import validate
    from llm_code_sanitize import sanitize_verilog_response

    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    max_tokens = completion_max_tokens(max_tokens)
    if verification_timeout is not None:
        # Validate before creating the output directory so an invalid CLI/API
        # value cannot leave a misleading retry artifact behind.
        _with_verification_timeout({}, verification_timeout)

    candidate = Path(candidate_path).expanduser().resolve()
    golden = Path(golden_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "jg_retry_result.json"
    token_usage: list[Dict[str, Any]] = []
    attempts: list[Dict[str, Any]] = []
    retry_count = 0
    resolved_golden_top = str(golden_top or "")
    resolved_design_type = str(design_type or "")
    resolved_timeout = verification_timeout
    # The last artifact that compiled locally.  A failed repair must never
    # replace it: the previous run reported the final, uncompilable retry as the
    # route's artifact, which lost the only usable RTL the route had produced.
    last_compilable: Optional[Path] = None
    interface_contract = ""

    def finish(
        *,
        status: str,
        equivalent: bool,
        error: str,
        final_candidate: Path,
    ) -> Dict[str, Any]:
        payload = {
            "schema_version": "jg_retry_gate_v2",
            "method": "structure_syntax_then_jaspergold_with_llm_retry",
            "verification_mode": "jaspergold",
            "status": status,
            "equivalent": equivalent,
            "success": equivalent,
            "retry_budget": max_retries,
            "retry_count": retry_count,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "final_candidate_path": str(final_candidate),
            "last_attempt_candidate_path": (
                str(attempts[-1].get("candidate_path", "")) if attempts else ""
            ),
            "last_compilable_candidate_path": (
                str(last_compilable) if last_compilable else ""
            ),
            "error_progression": summarise_error_progression(attempts),
            "interface_contract": interface_contract,
            "golden_path": str(golden),
            "golden_top": resolved_golden_top,
            "design_type": resolved_design_type,
            "verification_timeout": resolved_timeout,
            "error": error,
            "llm_token_usage": token_usage,
            "llm_token_totals": token_totals(token_usage),
            "boundary": (
                "status only reports which gate stage the artifact reached; a "
                "cleared error is not evidence of functional correctness."
            ),
        }
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return payload

    if not candidate.is_file():
        return finish(
            status="error",
            equivalent=False,
            error=f"candidate RTL does not exist: {candidate}",
            final_candidate=candidate,
        )
    if not golden.is_file():
        return finish(
            status="error",
            equivalent=False,
            error=f"golden RTL does not exist: {golden}",
            final_candidate=candidate,
        )

    current_candidate = candidate
    current_verilog = candidate.read_text(encoding="utf-8", errors="replace")
    if not module_name:
        try:
            module_name = parse_verilog_ports(current_verilog)[0]
        except Exception:
            module_name = "TopModule"
    last_status = "error"
    last_error = "verification did not run"
    last_feedback: Dict[str, Any] = {
        "status": "error",
        "error": last_error,
        "raw_output": "",
    }

    for attempt_index in range(max_retries + 1):
        attempt_dir = out_dir / f"attempt_{attempt_index}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        archived_candidate = attempt_dir / "candidate.v"
        if current_candidate != archived_candidate:
            archived_candidate.write_text(current_verilog, encoding="utf-8")
        # Stage 1: local structure. Markdown, a duplicated module, an unpacked
        # array port or an undefined submodule can be found without any tool, so
        # they never justify spending a remote formal-proof slot.
        prescreen = prescreen_rtl_artifact(current_verilog, module_name)
        structure_ok = bool(prescreen["ok"])
        # Stage 2: local compilation, which is also what keeps the artifact
        # eligible to be reported as the route's last usable RTL.
        syntax_ok, syntax_error = (
            validate(archived_candidate) if structure_ok else (False, "")
        )
        attempt_record: Dict[str, Any] = {
            "attempt": attempt_index,
            "candidate_path": str(archived_candidate),
            "structure": {
                "status": "passed" if structure_ok else "failed",
                "problems": prescreen["problems"],
            },
            "syntax": {
                "status": "passed"
                if syntax_ok
                else ("skipped" if not structure_ok else "failed"),
                "error": syntax_error,
            },
        }
        attempts.append(attempt_record)
        current_candidate = archived_candidate
        if syntax_ok:
            last_compilable = archived_candidate

        if not structure_ok:
            kinds = ", ".join(
                sorted({str(item["kind"]) for item in prescreen["problems"]})
            )
            last_status = "error"
            last_error = f"artifact rejected before JasperGold: {kinds}"
            attempt_record["failure_stage"] = "repair_artifact"
            attempt_record["error"] = last_error
            last_feedback = {
                "status": "error",
                "equivalent": False,
                "error": last_error,
                "raw_output": json.dumps(
                    prescreen["problems"], indent=2, ensure_ascii=False
                ),
            }
        elif syntax_ok:
            jg_result = verify_rtl_equivalence_with_jg(
                candidate_path=archived_candidate,
                golden_path=golden,
                output_dir=attempt_dir / "jaspergold",
                env_path=env_path,
                golden_top=golden_top,
                design_type=design_type,
                verification_timeout=verification_timeout,
            )
            attempt_record["jaspergold"] = jg_result
            if jg_result.get("golden_top"):
                resolved_golden_top = str(jg_result["golden_top"])
            if jg_result.get("design_type"):
                resolved_design_type = str(jg_result["design_type"])
            if jg_result.get("verification_timeout") is not None:
                resolved_timeout = jg_result["verification_timeout"]
            if jg_result.get("interface_contract"):
                interface_contract = str(jg_result["interface_contract"])
            last_feedback = dict(jg_result)
            last_status = str(jg_result.get("status") or "error")
            last_error = str(
                jg_result.get("error")
                or ("" if last_status == "passed" else f"jaspergold_{last_status}")
            )
            attempt_record["failure_stage"] = str(
                jg_result.get("failure_stage") or classify_failure_stage(jg_result)
            )
            attempt_record["error"] = last_error
            if last_status == "passed":
                return finish(
                    status="passed",
                    equivalent=True,
                    error="",
                    final_candidate=archived_candidate,
                )
        else:
            last_status = "error"
            last_error = f"JG repair candidate failed syntax validation: {syntax_error}"
            attempt_record["failure_stage"] = "compile"
            attempt_record["error"] = last_error
            last_feedback = {
                "status": "error",
                "equivalent": False,
                "error": last_error,
                "raw_output": syntax_error,
            }

        if attempt_index >= max_retries:
            break

        retry_count += 1
        try:
            client, model = _load_jg_retry_client(
                env_path,
                llm_client=llm_client,
                llm_model=llm_model,
            )
            # A structural rejection, a syntax rejection and an equivalence
            # failure need different repair prompts. Routing a syntax failure
            # through the JG prompt framed it as a functional bug and dropped the
            # compiler's line numbers, so the model kept re-emitting
            # uncompilable code.
            if not structure_ok:
                prompt = build_structural_repair_prompt(
                    module_name=module_name,
                    interface_contract=interface_contract,
                    previous_verilog=current_verilog,
                    prescreen=prescreen,
                )
                system_role = (
                    "You are an expert RTL designer repairing a Verilog-2001 "
                    "artifact that is not a single valid module."
                )
            elif not syntax_ok:
                # Imported here, not at module scope: rtl_direct_runner imports
                # this module, so a top-level import would be circular.
                from module5.rtl_direct_runner import _build_syntax_retry_message

                prompt = _build_syntax_retry_message(
                    syntax_error,
                    current_verilog,
                    attempt=attempt_index,
                    module_name=module_name,
                )
                system_role = (
                    "You are an expert RTL designer repairing a Verilog-2001 "
                    "module that failed local syntax validation."
                )
            else:
                prompt = build_jg_retry_prompt(
                    source_label=source_label,
                    source_context=source_context,
                    previous_verilog=current_verilog,
                    jg_result=last_feedback,
                    module_name=module_name,
                    interface_contract=interface_contract,
                )
                system_role = (
                    "You are an expert RTL designer repairing a design "
                    "from formal-verification feedback."
                )
            prompt_path = out_dir / f"attempt_{attempt_index + 1}_repair_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                **completion_token_kwargs(max_tokens),
                **seed_kwargs(),
                **qwen_thinking_kwargs(model),
                **deepseek_thinking_kwargs(model),
            )
            token_usage.append(
                usage_from_response(
                    stage="jg_repair",
                    model=model,
                    response=response,
                    attempt=retry_count,
                )
            )
            raw_response = response.choices[0].message.content or ""
            raw_path = out_dir / f"attempt_{attempt_index + 1}_repair_raw.txt"
            raw_path.write_text(raw_response, encoding="utf-8")
            if completion_was_truncated(response):
                # The repair answer is a prefix, not a module. Sanitizing it
                # would extract an unterminated module and hand the syntax gate
                # a line number that does not exist, so the attempt is recorded
                # as a transport failure and the loop stops: re-issuing an
                # unchanged repair prompt would only truncate again.
                last_status = "error"
                last_error = truncated_completion_message(response, stage="jg_repair")
                (
                    out_dir / f"attempt_{attempt_index + 1}_repair_truncated.txt"
                ).write_text(last_error, encoding="utf-8")
                break
            current_verilog = sanitize_verilog_response(raw_response).strip()
            if not current_verilog:
                last_status = "error"
                last_error = "JG repair LLM returned empty Verilog"
                break
            current_verilog += "\n"
            current_candidate = out_dir / f"attempt_{attempt_index + 1}" / "candidate.v"
            current_candidate.parent.mkdir(parents=True, exist_ok=True)
            current_candidate.write_text(current_verilog, encoding="utf-8")
        except Exception as exc:
            last_status = "error"
            last_error = f"JG repair LLM call failed: {exc}"
            break

    # A failed repair must not become the route's artifact when an earlier
    # attempt still compiles; the uncompilable retry stays on disk under its own
    # attempt directory as evidence.
    return finish(
        status=last_status,
        equivalent=False,
        error=last_error,
        final_candidate=last_compilable or current_candidate,
    )


def build_correction_prompt(
    interface_contract: str,
    c_code: str,
    spec_text: str,
    previous_verilog: str,
    jg_error: str,
    attempt: int,
    module_name: str,
) -> str:
    """Build LLM prompt for correcting failed Verilog based on JG feedback.

    This prompt used to embed the golden reference body, which handed the model
    the answer and made a later "verified" result meaningless as evidence.  Only
    the interface contract — port names, directions, widths and parameters — is
    shared now; the specification and the tool feedback carry the rest.
    """
    general_rules = _general_rtl_rules(8)
    return textwrap.dedent(f"""\
Your previous Verilog output (attempt {attempt}) failed formal equivalence checking against the reference design.

Frozen interface contract (must match exactly):

{interface_contract}

Your previous (incorrect) Verilog output:

{previous_verilog}

JasperGold equivalence checking error:

{jg_error}

Original C specification:

{c_code}

Specification description:

{spec_text}

CRITICAL REQUIREMENTS:
1. The output MUST implement the behavior required by the specification above.
2. Keep the same module name: {module_name}
3. Keep the interface exactly as listed in the contract above.
4. Fix the functional bug that caused the equivalence check to fail.
5. Do NOT simplify or change the interface.
6. Use Verilog-2001 syntax for Design Compiler compatibility:
   - Use 'reg' and 'wire' types (NOT 'logic')
   - Use explicit bit patterns like 32'h0 or 8'd0 (NOT '0 or '1 shorthand)
   - Avoid SystemVerilog features: typedef, enum, etc.
   - Use localparam for constants
   - For hardware replication, use 'generate' blocks with 'genvar', NOT runtime 'for' loops
   - Runtime 'for' loops cannot be used for array indexing with variable indices
7. Output ONLY the corrected Verilog module. No markdown fences, no explanations.
{general_rules}

The reference implementation is intentionally not provided.

Generate the corrected Verilog module now:
""")


def verify_rtl_with_jg(
    generated_verilog: str,
    golden_rtl_path: Path,
    c_code: str,
    spec_text: str,
    module_name: str,
    work_dir: Path,
    env_path: Path,
    llm_client,
    llm_model: str,
    max_retries: int = 1,
    golden_top: str = "",
    design_type: str = "",
    verification_timeout: int | None = None,
    max_tokens: int | None = None,
) -> Dict[str, Any]:
    """
    Verify generated RTL against golden reference using JasperGold.
    Iteratively corrects the RTL using LLM feedback until verification passes.

    Args:
        generated_verilog: Initial LLM-generated Verilog code
        golden_rtl_path: Path to golden reference RTL (csrng.sv)
        c_code: Original C source code
        spec_text: Specification description
        module_name: Target module name
        work_dir: Working directory for intermediate files (module5_mid/)
        env_path: Path to .env file with JG configuration
        llm_client: OpenAI-compatible client for LLM calls
        llm_model: Model name for LLM
        max_retries: Maximum correction attempts

    Returns:
        Dict with keys:
            - success: bool, whether verification passed
            - verified: bool, same as success
            - attempts: int, number of attempts made
            - final_verilog: str, final Verilog code
            - jg_results: list of JG results per attempt
            - error: str, error message if failed
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    max_tokens = completion_max_tokens(max_tokens)
    jg_config = _with_verification_timeout(
        _load_jg_config(env_path), verification_timeout
    )
    local_base = PROJECT_ROOT

    # Read golden reference
    golden_rtl = golden_rtl_path.read_text(encoding="utf-8", errors="replace")
    selected_golden = _select_verilog_module_source(golden_rtl, golden_top)
    golden_module, golden_ports, inferred_seq = parse_verilog_ports(selected_golden)
    is_seq, resolved_design_type = _effective_design_type(
        design_type, inferred_seq
    )
    golden_params = parse_verilog_parameters(selected_golden)
    golden_param_defaults = parse_verilog_parameter_defaults(selected_golden)
    interface_contract = build_interface_contract(
        golden_module, golden_ports, golden_param_defaults
    )

    result = {
        "success": False,
        "verified": False,
        "attempts": 0,
        "final_verilog": generated_verilog,
        "jg_results": [],
        "error": "",
        "llm_token_usage": [],
        "golden_top": golden_module,
        "design_type": resolved_design_type,
        "verification_timeout": int(jg_config["timeout"]),
        "interface_contract": interface_contract,
        "attempt_records": [],
        "last_compilable_verilog": "",
        "last_compilable_path": "",
    }

    current_verilog = generated_verilog

    for attempt in range(1, max_retries + 2):  # +1 for initial attempt
        result["attempts"] = attempt
        attempt_record: Dict[str, Any] = {"attempt": attempt}
        result["attempt_records"].append(attempt_record)

        # Structure and local compilation come before any remote proof: a
        # markdown-contaminated or uncompilable artifact used to consume a full
        # JasperGold session and return a tool error that the next repair prompt
        # then had to interpret as a functional bug.
        prescreen = prescreen_rtl_artifact(current_verilog, module_name)
        attempt_record["structure"] = {
            "status": "passed" if prescreen["ok"] else "failed",
            "problems": prescreen["problems"],
        }
        if not prescreen["ok"]:
            kinds = ", ".join(
                sorted({str(item["kind"]) for item in prescreen["problems"]})
            )
            attempt_record["failure_stage"] = "repair_artifact"
            result["error"] = f"artifact rejected before JasperGold: {kinds}"
            break

        try:
            _candidate_module, candidate_ports, _candidate_is_seq = (
                parse_verilog_ports(current_verilog)
            )
            _validate_compatible_ports(
                golden_ports, candidate_ports, golden_param_defaults
            )
        except Exception as exc:
            attempt_record["failure_stage"] = (
                "interface" if isinstance(exc, InterfaceMismatchError) else "setup"
            )
            if isinstance(exc, InterfaceMismatchError):
                attempt_record["interface_diff"] = exc.diff
            result["error"] = str(exc)
            break

        # Write current Verilog
        opt_module_name = f"{golden_module}_opt"
        opt_v_path = work_dir / f"attempt_{attempt}.v"

        # Rename module in generated Verilog to avoid conflicts
        current_verilog_renamed = re.sub(
            r"\bmodule\s+\w+",
            f"module {opt_module_name}",
            current_verilog,
            count=1,
        )
        opt_params = parse_verilog_parameters(current_verilog_renamed)
        opt_v_path.write_text(current_verilog_renamed, encoding="utf-8")

        from RTL_DIRECT_compare.validator import validate as _validate_syntax

        syntax_ok, syntax_error = _validate_syntax(opt_v_path)
        attempt_record["syntax"] = {
            "status": "passed" if syntax_ok else "failed",
            "error": syntax_error,
        }
        if syntax_ok:
            # Keep the last artifact that compiles, so a later failed repair
            # cannot leave the route without usable RTL.
            result["last_compilable_verilog"] = current_verilog
            result["last_compilable_path"] = str(opt_v_path)
        else:
            attempt_record["failure_stage"] = "compile"
            result["error"] = (
                f"attempt {attempt} failed local syntax validation: {syntax_error}"
            )
            break

        # Copy golden reference to work_dir
        ref_v_path = work_dir / golden_rtl_path.name
        ref_v_path.write_text(golden_rtl, encoding="utf-8")

        # Generate FPV wrapper
        fpv_sv_path = work_dir / f"attempt_{attempt}_FPV.sv"
        fpv_wrapper = generate_fpv_wrapper(
            golden_module,
            opt_module_name,
            golden_ports,
            is_seq,
            ref_params=golden_params,
            opt_params=opt_params,
            ref_param_defaults=golden_param_defaults,
        )
        fpv_sv_path.write_text(fpv_wrapper, encoding="utf-8")

        # Generate TCL script
        tcl_path = work_dir / f"attempt_{attempt}_FPV.tcl"
        both_edge_reasons = (
            _both_edge_clock_reasons(golden_rtl, current_verilog) if is_seq else []
        )
        attempt_record["clock_both_edges_reasons"] = both_edge_reasons
        fpv_tcl = generate_fpv_tcl(
            remote_dir=_local_to_remote(work_dir, local_base, jg_config["remote_base"]),
            ref_filename=golden_rtl_path.name,
            opt_filename=opt_v_path.name,
            fpv_filename=fpv_sv_path.name,
            fpv_module=f"{golden_module}_FPV",
            is_sequential=is_seq,
            ports=golden_ports,
            analyze_defines=tuple(jg_config.get("analyze_defines") or ()),
            clock_both_edges=bool(both_edge_reasons),
        )
        tcl_path.write_text(fpv_tcl, encoding="utf-8")

        # Run JasperGold
        print(f"  [JG] Attempt {attempt}: Running JasperGold verification...")
        jg_result = run_jaspergold(
            ref_v_path,
            opt_v_path,
            fpv_sv_path,
            tcl_path,
            jg_config,
            local_base,
        )
        result["jg_results"].append(jg_result)
        attempt_record["failure_stage"] = classify_failure_stage(
            _normalise_jg_verify_result(jg_result)
        )
        attempt_record["error"] = str(jg_result.get("error", "") or "")

        # Save JG output
        jg_output_path = work_dir / f"attempt_{attempt}_jg_output.txt"
        jg_output_path.write_text(jg_result.get("raw_output", ""), encoding="utf-8")

        if jg_result["success"]:
            print(f"  [JG] Attempt {attempt}: VERIFIED ✓")
            result["success"] = True
            result["verified"] = True
            result["final_verilog"] = current_verilog
            break

        print(f"  [JG] Attempt {attempt}: FAILED - {jg_result.get('error', 'unknown')}")

        # If max retries exhausted, stop
        if attempt > max_retries:
            result["error"] = f"Verification failed after {attempt} attempts: {jg_result.get('error', 'unknown')}"
            break

        # Generate correction prompt
        jg_error_detail = jg_result.get("error", "unknown error")
        raw_snippet = jg_result.get("raw_output", "")[-1500:]
        full_error = f"{jg_error_detail}\n\nJasperGold output (last 1500 chars):\n{raw_snippet}"

        correction_prompt = build_correction_prompt(
            interface_contract=interface_contract,
            c_code=c_code,
            spec_text=spec_text,
            previous_verilog=current_verilog,
            jg_error=full_error,
            attempt=attempt,
            module_name=module_name,
        )

        # Save correction prompt
        prompt_path = work_dir / f"attempt_{attempt + 1}_correction_prompt.txt"
        prompt_path.write_text(correction_prompt, encoding="utf-8")

        # Call LLM for correction
        print(f"  [LLM] Attempt {attempt + 1}: Requesting correction from LLM...")
        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert RTL designer. Fix the Verilog to pass formal verification."},
                    {"role": "user", "content": correction_prompt},
                ],
                temperature=0.0,
                **completion_token_kwargs(max_tokens),
                **qwen_thinking_kwargs(llm_model),
                **deepseek_thinking_kwargs(llm_model),
            )
            result["llm_token_usage"].append(
                usage_from_response(
                    stage="jg_repair",
                    model=llm_model,
                    response=response,
                    attempt=attempt + 1,
                )
            )
            raw_response = response.choices[0].message.content or ""

            # Save raw response
            response_path = work_dir / f"attempt_{attempt + 1}_llm_response.txt"
            response_path.write_text(raw_response, encoding="utf-8")

            # Extract Verilog
            current_verilog = _sanitize_verilog(raw_response)
            if not current_verilog.strip():
                result["error"] = f"LLM returned empty Verilog on attempt {attempt + 1}"
                break

        except Exception as e:
            result["error"] = f"LLM call failed on attempt {attempt + 1}: {str(e)}"
            break

    # A failed retry must not erase the last artifact that compiled.
    if not result["success"] and result["last_compilable_verilog"]:
        result["final_verilog"] = result["last_compilable_verilog"]
    result["error_progression"] = summarise_error_progression(
        result["attempt_records"]
    )
    return result


def _sanitize_verilog(text: str) -> str:
    """Extract Verilog code from LLM response, removing markdown fences."""
    from llm_code_sanitize import sanitize_verilog_response

    return sanitize_verilog_response(text)
