#!/usr/bin/env python3
"""
LLM-based C-to-Verilog pipeline with JasperGold equivalence checking.

Workflow per (benchmark_dir, transform_variant):
  1. Read original.v  (reference Verilog)
  2. Read baseline .c  (original C)
  3. Read transform .c  (transformed C)
  4. Prompt LLM: given original.v + transform.c, generate optimised Verilog
  5. Write LLM output as <variant>.v
  6. Run JasperGold FPV on remote server to check equivalence vs original.v
  7. If JG fails, feed error back to LLM for correction (up to --max-retries)
  8. Record as generation failure if all retries exhausted

Parallelism:
  - LLM calls are rate-limited by --llm-workers (default: 1).
  - JasperGold runs are parallelized across --workers SSH sessions.
  - In full-pipeline mode, JasperGold work can overlap across tasks while LLM
    requests still respect the global --llm-workers limit.

Usage:
  python llm_v2v.py <root_dir> [--metric TIMING|AREA] [--dry-run] [--jg-only] [--llm-only]
  python llm_v2v.py . --metric TIMING --workers 4 --llm-workers 2
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from project_paths import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Config — everything loaded from LOCAL_BASE/.env
# ---------------------------------------------------------------------------

LOCAL_BASE = PROJECT_ROOT
_ENV_PATH = LOCAL_BASE / ".env"


def _load_env(path: Path) -> dict[str, str]:
    env = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


_env = _load_env(_ENV_PATH)

# JasperGold remote server
REMOTE_USER = _env.get("JG_REMOTE_USER", "2660001")
REMOTE_HOST = _env.get("JG_REMOTE_HOST", "")
REMOTE_BASE = _env.get("JG_REMOTE_BASE", "").rstrip("/")
JG_TIMEOUT  = int(_env.get("JG_TIMEOUT", "600"))

# LLM provider configurations (local / cloud), both from the same .env
LLM_PROVIDERS = {
    "local": {
        "base_url":   _env.get("LOCAL_API_BASE_URL", "http://127.0.0.1:11434/v1"),
        "api_key":    _env.get("LOCAL_API_KEY", "sk-123456"),
        "model":      _env.get("LOCAL_MODEL", "qwen3.5:35b"),
        "max_tokens": int(_env.get("LOCAL_MAX_TOKENS", "16384")),
        "timeout":    int(_env.get("LOCAL_TIMEOUT", "600")),
        "no_proxy":   True,   # bypass http_proxy for local network
    },
    "cloud": {
        "base_url":   _env.get("CLOUD_API_BASE_URL", "https://api.openai.com/v1"),
        "api_key":    _env.get("CLOUD_API_KEY", ""),
        "model":      _env.get("CLOUD_MODEL", "gpt-4o"),
        "max_tokens": int(_env.get("CLOUD_MAX_TOKENS", "4096")),
        "timeout":    int(_env.get("CLOUD_TIMEOUT", "600")),
        "no_proxy":   False,
    },
}

# Active provider — set by main() based on --local / --no-local flag
_active_provider: dict = LLM_PROVIDERS["local"]
_llm_semaphore = threading.Semaphore(1)


def set_provider(local: bool) -> None:
    """Select the active LLM provider."""
    global _active_provider
    key = "local" if local else "cloud"
    _active_provider = LLM_PROVIDERS[key]
    cfg = _active_provider
    if not cfg["api_key"]:
        print(f"WARNING: no API key configured for '{key}' provider.", file=sys.stderr)
        print(f"  Set {key.upper()}_API_KEY in {_ENV_PATH}", file=sys.stderr)


# ---------------------------------------------------------------------------
# LLM client (minimal, no SDK dependency)
# ---------------------------------------------------------------------------

def llm_generate(system_prompt: str, user_prompt: str) -> str:
    """Call the active LLM provider via OpenAI-compatible chat completions API."""
    import urllib.request

    cfg = _active_provider
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": cfg["max_tokens"],
        "temperature": 0.2,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
    )

    if cfg["no_proxy"]:
        # Bypass http_proxy for local network
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()

    resp = opener.open(req, timeout=cfg["timeout"])
    body = json.loads(resp.read().decode())
    return body["choices"][0]["message"]["content"]


def _run_llm_request(system_prompt: str, user_prompt: str) -> str:
    """Run an LLM request under the global concurrency limit."""
    with _llm_semaphore:
        return llm_generate(system_prompt, user_prompt)


def extract_verilog(llm_output: str) -> str:
    """Extract Verilog code from LLM response (strip markdown fences, thinking tags, etc.)."""
    text = llm_output

    # Strip <think>...</think> blocks (Qwen reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Try to find ```verilog ... ``` block
    m = re.search(r"```(?:verilog|v|sv)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        # Try to find module ... endmodule
        m = re.search(r"((?:`timescale[^\n]*\n)?module\b.*?endmodule)", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

    # Remove any lines before the first `timescale or module declaration
    lines = text.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("`timescale") or stripped.startswith("module ") or stripped.startswith("//"):
            start_idx = i
            break
    text = "\n".join(lines[start_idx:])

    return text.strip()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert hardware designer. You translate optimised C code into
synthesisable Verilog, using a reference Verilog module as a structural template.

Rules:
- Output ONLY the Verilog module. No explanation, no markdown fences.
- Keep the same module name, port names, port widths, and port directions as
  the reference Verilog.
- Implement the logic described in the transformed C code.
- The transformed C expresses an optimisation intent (e.g. split carry chain,
  resource sharing, algebraic simplification). Reflect that intent faithfully
  in the Verilog structure.
- Do NOT simplify or restructure the computation. Preserve the computation
  graph from the transformed C code exactly: every arithmetic, logic, or
  comparison operation in the C must appear as a corresponding Verilog
  operator. Do NOT collapse, strength-reduce, or replace operations with
  equivalent but structurally different forms — even if the result is
  functionally identical.
- Use the same coding style as the reference (always blocks for sequential,
  assign for combinational, same reset polarity, etc.).
- Do NOT add or remove ports. Do NOT change bit widths.
- Do NOT wrap in markdown code fences.
""")

# ---------------------------------------------------------------------------
# Per-transform lightweight RTL hints
# ---------------------------------------------------------------------------

TRANSFORM_RTL_HINT: dict[str, str] = {
    "BOOLEAN_TO_ARITHMETIC":
        "RTL hint: replace ternary muxes (cond ? val+1 : val) with "
        "arithmetic (val + cond_bit). The boolean condition becomes a 1-bit addend.",
    "REASSOCIATE_ARITHMETIC":
        "RTL hint: regroup or reorder arithmetic operands to change the "
        "dependency chain. E.g., convert conditional increment to direct "
        "bit addition using concatenation.",
    "SPLIT_OP":
        "RTL hint: split wide operations into narrower independent "
        "sub-operations (e.g., separate high/low halves of an adder).",
    "BALANCE_TREE":
        "RTL hint: restructure linear chains (e.g., a+b+c+d) into "
        "balanced binary trees ((a+b)+(c+d)) to reduce combinational depth.",
    "CARRY_SAVE_REWRITE":
        "RTL hint: rewrite additions using carry-save form. Separate "
        "the increment/carry term from the base value.",
    "COMMON_SUBEXPR_EXTRACT":
        "RTL hint: extract repeated subexpressions (e.g., bit slices) "
        "into named intermediate wires, then reference the wire.",
    "GUARD_RELAXATION":
        "RTL hint: pre-compute guarded values in combinational assigns "
        "outside the enable condition; register the result in the always block.",
    "SPECULATIVE_COMPUTE":
        "RTL hint: compute all branch outcomes speculatively, then select "
        "the correct one. May collapse if/else into nested ternaries.",
    "PREDICATE_TO_DATAFLOW":
        "RTL hint: pre-compute both branch results into intermediate wires "
        "(e.g., _base, _plus), then mux-select in the always block.",
    "IF_CONVERSION":
        "RTL hint: convert if/else to parallel unconditional computations "
        "followed by a select mux.",
    "PIPELINE_STAGE_INSERT":
        "RTL hint: insert pipeline register stages (additional reg "
        "declarations) to break long combinational paths.",
    "MUX_TREE_BALANCE":
        "RTL hint: rebalance nested mux/ternary trees to minimize "
        "selection depth.",
    "ALGEBRAIC_SIMPLIFY":
        "RTL hint: apply algebraic identities (x+0→x, x&x→x, DeMorgan). "
        "Do NOT change the fundamental operation type.",
    "BREAK_CHAIN":
        "RTL hint: break serial dependency chains by introducing parallel "
        "computation paths that combine at the end.",
    "LOOP_PIPELINING":
        "RTL hint: pipeline repeated computation blocks by adding "
        "intermediate register stages.",
    "LOOP_FISSION":
        "RTL hint: split a monolithic always block into multiple smaller "
        "independent always blocks or stages.",
    "LOOP_INTERCHANGE":
        "RTL hint: reorder evaluation of independent operations.",
    "PARTIAL_UNROLL":
        "RTL hint: replicate logic for multiple iterations to expose "
        "parallelism.",
    "INLINE_CRITICAL_FUNCTION":
        "RTL hint: inline sub-module or function calls on the critical "
        "path into flattened logic.",
    "OUTLINE_LONG_COMPUTE":
        "RTL hint: extract long combinational expressions into separate "
        "named wire assignments.",
    "CONTROL_FLATTEN":
        "RTL hint: flatten nested if/else into parallel mux logic.",
    "COPY_PROPAGATION":
        "RTL hint: replace intermediate wire/reg names with their source "
        "expressions.",
    "DEPENDENCE_BREAK":
        "RTL hint: restructure computations into independent parallel "
        "paths.",
    "ENCODE_ONEHOT_TO_BINARY":
        "RTL hint: convert one-hot selection logic to binary-encoded "
        "selection.",
}


def _compute_c_diff(baseline_c: str, transform_c: str) -> str:
    """Compute a unified diff between baseline and transform C."""
    import difflib
    baseline_lines = baseline_c.splitlines(keepends=True)
    transform_lines = transform_c.splitlines(keepends=True)
    diff = difflib.unified_diff(
        baseline_lines, transform_lines,
        fromfile="baseline.c", tofile="transform.c",
        lineterm="",
    )
    result = "".join(diff)
    return result if result.strip() else "(no textual difference)"


def build_user_prompt(
    original_v: str,
    baseline_c: str,
    transform_c: str,
    transform_name: str,
) -> str:
    c_diff = _compute_c_diff(baseline_c, transform_c)
    rtl_hint = TRANSFORM_RTL_HINT.get(transform_name, "")
    hint_section = f"\n{rtl_hint}\n" if rtl_hint else ""

    return textwrap.dedent(f"""\
Here is the reference Verilog module:

{original_v}

Here is the baseline C that corresponds to the reference Verilog:

{baseline_c}

Here is the transformed C ({transform_name}):

{transform_c}

Here is what changed between the baseline and transformed C:

{c_diff}
{hint_section}
Generate the Verilog module that implements the transformed C code.
Use the reference Verilog as a template for module name, ports, and coding style.

IMPORTANT: Do NOT simplify the transformed C's computation structure. Every
arithmetic, logic, and comparison operation in the transformed C must appear as
a corresponding Verilog operator. If the C uses `val + ((Z >> 13) & 1)`, your
Verilog must use `val + ((Z >> 13) & 1)` — do NOT rewrite it as a ternary mux
or any other equivalent form.

Output ONLY the Verilog code, nothing else.
""")


def build_correction_prompt(
    original_v: str,
    transform_c: str,
    transform_name: str,
    previous_v: str,
    jg_error: str,
    attempt: int,
) -> str:
    """Build a prompt that asks the LLM to fix a failed Verilog generation."""
    rtl_hint = TRANSFORM_RTL_HINT.get(transform_name, "")
    hint_section = f"\n{rtl_hint}\n" if rtl_hint else ""

    return textwrap.dedent(f"""\
Your previous Verilog output (attempt {attempt}) failed formal equivalence checking.

Reference Verilog module:

{original_v}

Transformed C ({transform_name}):

{transform_c}
{hint_section}
Your previous (incorrect) Verilog output:

{previous_v}

JasperGold error:
{jg_error}

Please fix the Verilog so that it is functionally equivalent to the reference module
while reflecting the optimisation intent of the transformed C code.
IMPORTANT: Do NOT simplify the computation structure. Preserve every operation
from the transformed C as a distinct Verilog operator.
Keep the same module name, port names, port widths, and port directions as the reference.
Output ONLY the corrected Verilog module, nothing else.
""")


# ---------------------------------------------------------------------------
# JasperGold equivalence checking
# ---------------------------------------------------------------------------

def _local_to_remote(local_path: Path) -> str:
    """Convert a local path under LOCAL_BASE to the corresponding remote path."""
    if not REMOTE_BASE:
        raise RuntimeError("JG_REMOTE_BASE is not configured in .env")
    rel = local_path.resolve().relative_to(LOCAL_BASE.resolve())
    return f"{REMOTE_BASE}/{rel}"


def _cleanup_remote_jgproject(remote_dir: str, remote_tcl: str, project_name: str) -> None:
    """
    Best-effort cleanup for JasperGold artifacts after timeout/lock errors.
    This avoids stale `jgproject` ownership blocking a retry in the same folder.

    Strategy:
    1. Kill the JG process tree (parent + children) via SIGKILL.
    2. Wait briefly for processes to die before removing directories.
    3. Use unquoted globs so the shell can expand jgproject* patterns.
    """
    # Step 1: kill JG process tree — use pkill -9 to force-kill, and also
    # kill by working directory to catch child proof-engine processes.
    kill_cmd = (
        f"pkill -9 -f {shlex.quote(remote_tcl)} 2>/dev/null || true; "
        f"pkill -9 -f 'jaspergold.*{project_name}' 2>/dev/null || true; "
        f"sleep 1"
    )
    # Step 2: remove project dirs — unquoted glob so shell expands jgproject*
    clean_cmd = (
        f"cd {shlex.quote(remote_dir)} && "
        f"rm -rf {project_name} {project_name}_* jgproject jgproject_* sessionLogs* 2>/dev/null || true"
    )
    remote_cmd = f"{kill_cmd}; {clean_cmd}"
    try:
        subprocess.run(
            ["ssh", f"{REMOTE_USER}@{REMOTE_HOST}", remote_cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        pass


def _infer_clock_name(ports: list[dict]) -> str:
    """Best-effort clock port detection for sequential modules."""
    return next(
        (p["name"] for p in ports if p["dir"] == "input" and p["name"].lower() in ("clk", "clock")),
        "clk",
    )


def _infer_reset_name(ports: list[dict]) -> Optional[str]:
    """Best-effort reset port detection; only consider input ports."""
    resets = [
        p["name"]
        for p in ports
        if p["dir"] == "input" and ("rst" in p["name"].lower() or "reset" in p["name"].lower())
    ]
    return resets[0] if resets else None


def _is_active_low_reset(name: str) -> bool:
    """Heuristic for rst_n / resetn naming."""
    tail = name.lower().replace("reset", "").replace("rst", "")
    return "n" in tail


def _detect_sequential_logic(verilog_src: str) -> bool:
    """
    Detect common sequential RTL styles:
    - always @(posedge/negedge ...)
    - always_ff @(posedge/negedge ...)
    """
    return bool(
        re.search(r"always(?:_ff)?\s*@\s*\([^)]*(?:posedge|negedge)", verilog_src, re.IGNORECASE)
    )


def generate_fpv_wrapper(
    ref_module: str,
    opt_module: str,
    ports: list[dict],
    is_sequential: bool,
) -> str:
    """Generate an FPV wrapper that asserts equivalence between ref and opt modules."""
    lines = []

    fpv_name = f"{ref_module}_FPV"
    lines.append(f"module {fpv_name}(")

    # Port declarations for wrapper
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

    # Instantiate ref
    lines.append(f"    {ref_module} u_ref(")
    conns = []
    for p in ports:
        if p["dir"] == "input":
            conns.append(f"        .{p['name']}({p['name']})")
        else:
            conns.append(f"        .{p['name']}({p['name']}_ref)")
    lines.append(",\n".join(conns))
    lines.append("    );")
    lines.append("")

    # Instantiate opt
    lines.append(f"    {opt_module} u_opt(")
    conns = []
    for p in ports:
        if p["dir"] == "input":
            conns.append(f"        .{p['name']}({p['name']})")
        else:
            conns.append(f"        .{p['name']}({p['name']}_opt)")
    lines.append(",\n".join(conns))
    lines.append("    );")
    lines.append("")

    # Assertions
    outputs = [p for p in ports if p["dir"] == "output"]
    if is_sequential:
        clk_name = _infer_clock_name(ports)
        rst_name = _infer_reset_name(ports)
        for o in outputs:
            safe = re.sub(r"\W", "_", o["name"])
            lines.append(f"    property p_eq_{safe};")
            if rst_name:
                if _is_active_low_reset(rst_name):
                    lines.append(f"        @(posedge {clk_name}) disable iff (!{rst_name})")
                else:
                    lines.append(f"        @(posedge {clk_name}) disable iff ({rst_name})")
            else:
                lines.append(f"        @(posedge {clk_name})")
            lines.append(f"        ({o['name']}_ref == {o['name']}_opt);")
            lines.append(f"    endproperty")
            lines.append(f"    assert property (p_eq_{safe});")
            lines.append("")
    else:
        # Combinational: immediate assertions
        for o in outputs:
            safe = re.sub(r"\W", "_", o["name"])
            lines.append(f"    always_comb begin")
            lines.append(f"        assert_eq_{safe}: assert ({o['name']}_ref == {o['name']}_opt);")
            lines.append(f"    end")
            lines.append("")

    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def generate_fpv_tcl(
    remote_dir: str,
    ref_filename: str,
    opt_filename: str,
    fpv_filename: str,
    fpv_module: str,
    is_sequential: bool,
    ports: list[dict],
) -> str:
    """Generate JasperGold FPV tcl script."""
    lines = [
        f"set WORK_DIR {remote_dir}",
        "",
        "analyze -sv12 \\",
        f"  ${{WORK_DIR}}/{ref_filename} \\",
        f"  ${{WORK_DIR}}/{opt_filename} \\",
        f"  ${{WORK_DIR}}/{fpv_filename}",
        "",
        f"elaborate -bbox_mul 128 -top {fpv_module}",
        "",
    ]

    if is_sequential:
        clk_name = _infer_clock_name(ports)
        lines.append(f"clock {clk_name}")

        rst_name = _infer_reset_name(ports)
        if rst_name:
            if _is_active_low_reset(rst_name):
                lines.append(f"reset -expression {{!{rst_name}}}")
            else:
                lines.append(f"reset -expression {{{rst_name}}}")
        lines.append("")
    else:
        # Combinational: tell JasperGold there is no clock/reset
        lines.append("clock -none")
        lines.append("reset -none")
        lines.append("")

    lines.append("prove -all")
    lines.append("")
    return "\n".join(lines)


def parse_verilog_ports(verilog_src: str) -> tuple[str, list[dict], bool]:
    """
    Parse module name, ports, and detect if sequential from a Verilog source.
    Returns (module_name, ports, is_sequential).
    """
    # Module name
    m = re.search(r"\bmodule\s+(\w+)\s*[\(#]", verilog_src)
    if not m:
        raise ValueError("Cannot find module declaration")
    mod_name = m.group(1)

    ports = []
    # Match port declarations: input/output [wire|reg] [signed] [width] name
    for pm in re.finditer(
        r"\b(input|output)\s+(wire\s+|reg\s+)?(signed\s+)?(\[[^\]]+\]\s*)?(\w+)\s*[;,)]",
        verilog_src,
    ):
        direction = pm.group(1)
        wire_reg = (pm.group(2) or "").strip()
        signed = (pm.group(3) or "").strip()
        width = (pm.group(4) or "").strip()
        name = pm.group(5)
        # Build type string
        type_parts = []
        if wire_reg:
            type_parts.append(wire_reg)
        if signed:
            type_parts.append(signed)
        if width:
            type_parts.append(width)
        type_str = " ".join(type_parts) if type_parts else ""
        ports.append({"dir": direction, "type": type_str, "name": name})

    is_seq = _detect_sequential_logic(verilog_src)

    return mod_name, ports, is_seq


def run_jaspergold(
    ref_v_path: Path,
    opt_v_path: Path,
    fpv_sv_path: Path,
    tcl_path: Path,
    timeout: int = JG_TIMEOUT,
) -> dict:
    """
    Run JasperGold on the remote server via SSH.
    Returns dict with keys: success, proven, cex, undetermined, raw_output.
    """
    remote_tcl = _local_to_remote(tcl_path)
    remote_dir = str(Path(remote_tcl).parent)
    project_name = f"jgproject_{opt_v_path.stem}"

    # Clean up stale project directories left by a previous timeout/retry.
    _cleanup_remote_jgproject(remote_dir, remote_tcl, project_name)

    cmd = [
        "ssh", f"{REMOTE_USER}@{REMOTE_HOST}",
        f"cd {shlex.quote(remote_dir)} && jaspergold -proj {shlex.quote(project_name)} -batch {shlex.quote(remote_tcl)}",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,  # extra margin for SSH
        )
        output = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        _cleanup_remote_jgproject(remote_dir, remote_tcl, project_name)
        return {"success": False, "error": "timeout", "raw_output": ""}

    # Parse SUMMARY
    result = {
        "success": False,
        "proven": 0,
        "cex": 0,
        "undetermined": 0,
        "error": "",
        "raw_output": output[-3000:] if len(output) > 3000 else output,
    }

    # Extract summary block
    summary_m = re.search(
        r"(=+\s*\n\s*SUMMARY\s*\n\s*=+.*?)(?=\n\s*=+\s*\n|\Z)",
        output, re.DOTALL | re.IGNORECASE,
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
        lock_lines = []
        for ln in output.splitlines():
            stripped = ln.strip()
            lower = stripped.lower()
            if (
                "cannot obtain ownership of project directory" in lower
                or ("sessionlogs.bak" in lower and "project directory" in lower)
            ):
                lock_lines.append(stripped)

        # Check for compile/runtime errors
        err_lines = [ln.strip() for ln in output.splitlines() if "ERROR" in ln and "VERI-" in ln]
        generic_error_lines = [
            ln.strip()
            for ln in output.splitlines()
            if re.match(r"error:", ln.strip(), re.IGNORECASE)
        ]
        if lock_lines:
            result["error"] = "project directory lock: " + lock_lines[0]
            _cleanup_remote_jgproject(remote_dir, remote_tcl, project_name)
        elif err_lines:
            result["error"] = "compile error: " + err_lines[0]
        elif generic_error_lines:
            result["error"] = "runtime error: " + generic_error_lines[0]
        else:
            result["error"] = "no SUMMARY in output"

    return result


# ---------------------------------------------------------------------------
# Benchmark discovery
# ---------------------------------------------------------------------------

def discover_benchmarks(root: Path, metric: str) -> list[dict]:
    """
    Discover all (benchmark_dir, original.v, baseline.c, transform_variants).
    Returns list of dicts with keys:
      dir, original_v, baseline_c, module_name, transforms: [{name, c_path}]
    """
    top_dir = root / f"LLM_GR_RTL_{metric.upper()}"
    if not top_dir.is_dir():
        print(f"ERROR: {top_dir} not found", file=sys.stderr)
        return []

    # Collect all benchmark dirs across subcategories (arithmetic, logical, …)
    bench_dirs = []  # list of (subcategory_name, bench_path)
    for sub in sorted(top_dir.iterdir()):
        if not sub.is_dir():
            continue
        for d in sorted(sub.iterdir()):
            if d.is_dir():
                bench_dirs.append((sub.name, d))

    benchmarks = []
    for subcategory, d in bench_dirs:

        # Find original .v (the only .v, or prefer *_original.v)
        v_files = list(d.glob("*_original.v")) + list(d.glob("sampled_*.v"))
        if not v_files:
            # Fallback: any .v in the directory
            v_files = list(d.glob("*.v"))
        if not v_files:
            continue
        original_v = v_files[0]

        # Split .c files into baseline vs transforms.
        # Transform files have an _ALLCAPS suffix (e.g. example_SPLIT_OP.c).
        # The baseline is the .c without such a suffix.
        _transform_suffix_re = re.compile(r"_[A-Z][A-Z_]+$")
        all_c = sorted(d.glob("*.c"))
        baseline_candidates = [f for f in all_c if not _transform_suffix_re.search(f.stem)]
        if not baseline_candidates:
            continue
        baseline_c = baseline_candidates[0]
        base_stem = baseline_c.stem

        # Find transform .c files
        transforms = []
        for f in all_c:
            if f == baseline_c:
                continue
            # Extract transform name: <base_stem>_<TRANSFORM>.c
            if f.stem.startswith(base_stem + "_"):
                tname = f.stem[len(base_stem) + 1:]
                transforms.append({"name": tname, "c_path": f})

        if not transforms:
            continue

        benchmarks.append({
            "dir": d,
            "subcategory": subcategory,
            "original_v": original_v,
            "baseline_c": baseline_c,
            "transforms": transforms,
        })

    return benchmarks




# ---------------------------------------------------------------------------
# Thread-safe logging
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def _log(msg: str, end: str = "\n", flush: bool = True) -> None:
    with _print_lock:
        print(msg, end=end, flush=flush)


# ---------------------------------------------------------------------------
# Main pipeline — split into prepare / llm / jg phases
# ---------------------------------------------------------------------------

def _prepare_task(
    benchmark: dict,
    transform: dict,
    output_dir: Path,
) -> dict:
    """
    Prepare paths and parse the reference Verilog.
    Returns a task dict with all info needed for subsequent phases.
    """
    bench_dir = benchmark["dir"]
    subcategory = benchmark.get("subcategory", "")
    original_v_path = benchmark["original_v"]
    baseline_c_path = benchmark["baseline_c"]
    transform_c_path = transform["c_path"]
    transform_name = transform["name"]
    bench_name = bench_dir.name

    out_dir = output_dir / subcategory / bench_name / transform_name if subcategory else output_dir / bench_name / transform_name
    out_dir.mkdir(parents=True, exist_ok=True)

    original_v = original_v_path.read_text(encoding="utf-8", errors="replace")
    baseline_c = baseline_c_path.read_text(encoding="utf-8", errors="replace")
    transform_c = transform_c_path.read_text(encoding="utf-8", errors="replace")

    ref_module, ports, is_seq = parse_verilog_ports(original_v)

    return {
        "bench_name": bench_name,
        "transform_name": transform_name,
        "out_dir": out_dir,
        "original_v_path": original_v_path,
        "original_v": original_v,
        "baseline_c": baseline_c,
        "transform_c": transform_c,
        "ref_module": ref_module,
        "ports": ports,
        "is_seq": is_seq,
        "opt_v_path": out_dir / f"{transform_name}.v",
        "fpv_sv_path": out_dir / f"{transform_name}_FPV.sv",
        "tcl_path": out_dir / f"{transform_name}_FPV.tcl",
    }


def _generate_and_write(task: dict, user_prompt: str, label: str) -> tuple[bool, str, str]:
    """
    Call LLM, extract Verilog, write to opt_v_path.
    Returns (ok, verilog_text, error_msg).
    """
    ref_module = task["ref_module"]
    opt_v_path = task["opt_v_path"]

    _log(f"  [LLM] {task['bench_name']}/{task['transform_name']} {label} ...", end=" ")
    try:
        raw = _run_llm_request(SYSTEM_PROMPT, user_prompt)
        verilog = extract_verilog(raw)

        if not verilog.strip():
            _log("EMPTY")
            return False, "", "LLM returned empty Verilog"

        # Ensure module name matches reference
        vm = re.search(r"\bmodule\s+(\w+)", verilog)
        if vm and vm.group(1) != ref_module:
            verilog = verilog.replace(vm.group(1), ref_module, 1)

        opt_v_path.write_text(verilog + "\n", encoding="utf-8")
        _log("OK")
        return True, verilog, ""
    except Exception as e:
        _log(f"FAIL ({e})")
        return False, "", f"LLM: {e}"


def _run_llm(task: dict) -> dict:
    """Run initial LLM generation for a single task. Returns result dict."""
    result = {
        "benchmark": task["bench_name"],
        "transform": task["transform_name"],
        "llm_ok": False,
        "jg_result": None,
        "attempts": 0,
        "attempt_history": [],
    }

    user_prompt = build_user_prompt(
        task["original_v"], task["baseline_c"],
        task["transform_c"], task["transform_name"],
    )
    ok, verilog, err = _generate_and_write(task, user_prompt, "(attempt 1)")
    result["attempts"] = 1
    if ok:
        result["llm_ok"] = True
        result["_last_verilog"] = verilog
    else:
        result["error"] = err
        result["attempt_history"].append({"attempt": 1, "error": err})

    return result


def _run_llm_correction(task: dict, result: dict, jg_error: str, attempt: int) -> bool:
    """
    Run a correction LLM call using the previous error.
    Updates result in-place. Returns True if LLM generation succeeded.
    """
    previous_v = task["opt_v_path"].read_text(encoding="utf-8", errors="replace")
    user_prompt = build_correction_prompt(
        original_v=task["original_v"],
        transform_c=task["transform_c"],
        transform_name=task["transform_name"],
        previous_v=previous_v,
        jg_error=jg_error,
        attempt=attempt,
    )
    ok, verilog, err = _generate_and_write(task, user_prompt, f"(retry {attempt})")
    result["attempts"] = attempt
    if ok:
        result["llm_ok"] = True
        result["_last_verilog"] = verilog
        return True
    else:
        result["attempt_history"].append({"attempt": attempt, "error": err})
        return False


def _prepare_jg_files(task: dict) -> None:
    """Rename opt module and generate FPV wrapper + TCL files."""
    ref_module = task["ref_module"]
    opt_v_path = task["opt_v_path"]
    opt_module = f"{ref_module}_opt"

    # Rename module in generated Verilog
    opt_v_src = opt_v_path.read_text(encoding="utf-8", errors="replace")
    vm = re.search(r"\bmodule\s+(\w+)", opt_v_src)
    if vm:
        llm_module_name = vm.group(1)
        if llm_module_name != opt_module:
            opt_v_src = re.sub(
                r"\b" + re.escape(llm_module_name) + r"\b",
                opt_module,
                opt_v_src,
            )
    opt_v_path.write_text(opt_v_src, encoding="utf-8")

    # FPV wrapper
    fpv_sv = generate_fpv_wrapper(ref_module, opt_module, task["ports"], task["is_seq"])
    task["fpv_sv_path"].write_text(fpv_sv, encoding="utf-8")

    # Copy reference Verilog
    out_dir = task["out_dir"]
    ref_in_out = out_dir / task["original_v_path"].name
    if not ref_in_out.exists() or ref_in_out.read_text() != task["original_v"]:
        ref_in_out.write_text(task["original_v"], encoding="utf-8")
    task["ref_in_out"] = ref_in_out

    # TCL
    remote_dir = _local_to_remote(out_dir)
    fpv_tcl = generate_fpv_tcl(
        remote_dir=remote_dir,
        ref_filename=task["original_v_path"].name,
        opt_filename=task["opt_v_path"].name,
        fpv_filename=task["fpv_sv_path"].name,
        fpv_module=f"{ref_module}_FPV",
        is_sequential=task["is_seq"],
        ports=task["ports"],
    )
    task["tcl_path"].write_text(fpv_tcl, encoding="utf-8")


def _run_jg(task: dict, result: dict) -> dict:
    """Run JasperGold for a single task. Updates and returns result dict."""
    bench_name = task["bench_name"]
    transform_name = task["transform_name"]

    _log(f"  [JG]  {bench_name}/{transform_name} ...", end=" ")
    jg = run_jaspergold(
        task["ref_in_out"], task["opt_v_path"],
        task["fpv_sv_path"], task["tcl_path"],
    )
    result["jg_result"] = jg
    if jg["success"]:
        _log("PROVEN")
    else:
        _log(f"FAIL ({jg.get('error', 'unknown')})")
    return result


def _run_full_task(task: dict, max_retries: int) -> dict:
    """
    Run full pipeline for one task: LLM generation → JG check → retry loop.
    Self-contained — safe to call from a thread pool.
    """
    # Initial LLM generation
    r = _run_llm(task)
    if not r["llm_ok"]:
        r["generation_failure"] = True
        return r

    # Retry loop: generate → JG check → correction → JG check → ...
    proven = False
    for attempt in range(1, max_retries + 1):
        _prepare_jg_files(task)
        _run_jg(task, r)

        jg = r["jg_result"]
        r["attempt_history"].append({
            "attempt": r["attempts"],
            "jg_success": jg["success"],
            "jg_error": jg.get("error", ""),
        })

        if jg["success"]:
            proven = True
            break

        # Not proven — if retries remain, ask LLM to fix
        if attempt < max_retries:
            jg_error = jg.get("error", "unknown error")
            raw_snippet = jg.get("raw_output", "")[-1500:]
            error_detail = f"{jg_error}\n\nJasperGold output (last 1500 chars):\n{raw_snippet}"

            ok = _run_llm_correction(task, r, error_detail, attempt + 1)
            if not ok:
                break

    if not proven:
        r["generation_failure"] = True
        _log(f"  [FAIL] {task['bench_name']}/{task['transform_name']}: "
             f"exhausted {r['attempts']} attempts")

    r.pop("_last_verilog", None)
    return r


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict], llm_only: bool) -> None:
    print(f"\n{'='*60}")
    print(f"Processed {len(results)} variants")
    llm_ok = sum(1 for r in results if r["llm_ok"])
    print(f"  LLM generated: {llm_ok}/{len(results)}")
    if not llm_only:
        jg_proven = sum(1 for r in results
                        if r.get("jg_result") and r["jg_result"].get("success"))
        jg_cex = sum(1 for r in results
                     if r.get("jg_result") and r["jg_result"].get("cex", 0) > 0)
        jg_undet = sum(1 for r in results
                       if r.get("jg_result") and r["jg_result"].get("undetermined", 0) > 0)
        jg_compile = sum(1 for r in results
                         if r.get("jg_result") and "compile error" in str(r["jg_result"].get("error", "")))
        gen_fail = sum(1 for r in results if r.get("generation_failure"))
        retried = [r for r in results if r.get("attempts", 1) > 1]
        retry_success = sum(1 for r in retried
                            if r.get("jg_result") and r["jg_result"].get("success"))
        print(f"  JG proven:       {jg_proven}")
        print(f"  JG cex:          {jg_cex}")
        print(f"  JG compile err:  {jg_compile}")
        print(f"  JG undetermined: {jg_undet}")
        print(f"  Generation fail: {gen_fail}  (exhausted all retries)")
        if retried:
            print(f"  Retried:         {len(retried)}  (fixed after retry: {retry_success})")


def _write_results(results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="LLM C→Verilog + JasperGold equivalence pipeline")
    ap.add_argument("root", type=Path, nargs="?", default=LOCAL_BASE,
                    help="Root directory (default: %(default)s)")
    ap.add_argument("--metric", choices=["TIMING", "AREA"], default="TIMING")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output directory for generated files (default: <root>/LLM_V2V_<metric>)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be done")
    ap.add_argument("--llm-only", action="store_true", help="Only generate Verilog, skip JasperGold")
    ap.add_argument("--jg-only", action="store_true", help="Only run JasperGold on existing .v files")
    ap.add_argument("--bench", type=str, default=None,
                    help="Run only benchmarks matching this substring")
    ap.add_argument("--transform", type=str, default=None,
                    help="Run only transforms matching this substring")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max number of (benchmark, transform) pairs to process")
    ap.add_argument("-j", "--workers", type=int, default=4,
                    help="Number of parallel task/JasperGold workers (default: 4)")
    ap.add_argument("--llm-workers", type=int, default=1,
                    help="Max parallel LLM requests (default: 1)")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="Max LLM correction attempts on JG failure (default: 3)")
    ap.add_argument("--local", action="store_true", default=True, dest="local",
                    help="Use locally deployed LLM (default)")
    ap.add_argument("--no-local", action="store_false", dest="local",
                    help="Use cloud/API provider (CLOUD_* keys in .env)")
    args = ap.parse_args()

    # Select LLM provider
    set_provider(args.local)
    provider = "local" if args.local else "cloud"
    cfg = _active_provider
    print(f"LLM provider: {provider}  model: {cfg['model']}  endpoint: {cfg['base_url']}")

    llm_workers = max(1, args.llm_workers)
    global _llm_semaphore
    _llm_semaphore = threading.Semaphore(llm_workers)

    root = args.root.resolve()
    output_dir = args.output or root / f"LLM_V2V_{args.metric}"
    # Output dir must be under LOCAL_BASE for remote path mapping
    try:
        output_dir.resolve().relative_to(LOCAL_BASE.resolve())
    except ValueError:
        print(f"WARNING: output dir {output_dir} is not under {LOCAL_BASE}.")
        print(f"  JasperGold remote runs will fail. Use --llm-only or set --output under {LOCAL_BASE}.")
        if not (args.llm_only or args.dry_run):
            print("  Aborting. Use --llm-only to generate files only.")
            return 1

    benchmarks = discover_benchmarks(root, args.metric)
    if not benchmarks:
        print("No benchmarks found.")
        return 1

    # Build work list
    work: list[tuple[dict, dict]] = []  # (benchmark, transform)
    for bench in benchmarks:
        if args.bench and args.bench not in bench["dir"].name:
            continue
        for tx in bench["transforms"]:
            if args.transform and args.transform not in tx["name"]:
                continue
            work.append((bench, tx))
            if args.limit and len(work) >= args.limit:
                break
        if args.limit and len(work) >= args.limit:
            break

    total = len(work)
    print(f"Found {len(benchmarks)} benchmarks, {total} tasks selected, {args.workers} task/JG workers, {llm_workers} LLM workers")

    if args.dry_run:
        for bench, tx in work:
            print(f"  {bench['dir'].name}/{tx['name']}")
        print(f"\n[dry-run] Would process {total} tasks.")
        return 0

    # --- Prepare all tasks (fast, local I/O only) ---
    tasks: list[dict] = []
    for bench, tx in work:
        try:
            t = _prepare_task(bench, tx, output_dir)
            tasks.append(t)
        except ValueError as e:
            _log(f"  SKIP {bench['dir'].name}/{tx['name']}: {e}")

    results: list[dict] = []

    # --- Skip tasks whose output .v already exists (resume support) ---
    if not args.jg_only:
        existing_results: dict[tuple[str, str], dict] = {}
        results_path = output_dir / "results.json"
        if results_path.is_file():
            try:
                with open(results_path) as f:
                    for entry in json.load(f):
                        key = (entry.get("benchmark", ""), entry.get("transform", ""))
                        existing_results[key] = entry
            except (json.JSONDecodeError, KeyError):
                pass

        remaining: list[dict] = []
        for t in tasks:
            if t["opt_v_path"].is_file():
                _log(f"  SKIP {t['bench_name']}/{t['transform_name']}: output already exists")
                key = (t["bench_name"], t["transform_name"])
                if key in existing_results:
                    results.append(existing_results[key])
            else:
                remaining.append(t)
        skipped = len(tasks) - len(remaining)
        if skipped:
            print(f"  Skipped {skipped} tasks with existing output files")
        tasks = remaining

    # ===================================================================
    # MODE 1: --jg-only — parallel JasperGold on existing .v files
    # ===================================================================
    if args.jg_only:
        print(f"\n--- JasperGold only (parallel, {args.workers} workers) ---")
        jg_tasks = []
        for t in tasks:
            if not t["opt_v_path"].is_file():
                _log(f"  SKIP {t['bench_name']}/{t['transform_name']}: .v not found")
                results.append({
                    "benchmark": t["bench_name"],
                    "transform": t["transform_name"],
                    "llm_ok": False,
                    "jg_result": None,
                    "attempts": 0,
                    "attempt_history": [],
                    "error": "missing .v file",
                })
                continue
            _prepare_jg_files(t)
            r = {
                "benchmark": t["bench_name"],
                "transform": t["transform_name"],
                "llm_ok": True,
                "jg_result": None,
                "attempts": 1,
                "attempt_history": [],
            }
            jg_tasks.append((t, r))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_jg, t, r): (t, r) for t, r in jg_tasks}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)

    # ===================================================================
    # MODE 2: --llm-only
    #   local  → serial (single GPU)
    #   cloud  → parallel (-j workers)
    # ===================================================================
    elif args.llm_only:
        if args.local:
            print(f"\n--- LLM generation only (serial, local GPU) ---")
            for t in tasks:
                r = _run_llm(t)
                if r["llm_ok"]:
                    _prepare_jg_files(t)
                r.pop("_last_verilog", None)
                results.append(r)
        else:
            print(f"\n--- LLM generation only (parallel, {args.workers} task workers, {llm_workers} LLM workers, cloud) ---")

            def _llm_only_worker(t: dict) -> dict:
                r = _run_llm(t)
                if r["llm_ok"]:
                    _prepare_jg_files(t)
                r.pop("_last_verilog", None)
                return r

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_llm_only_worker, t): t for t in tasks}
                for fut in as_completed(futures):
                    results.append(fut.result())

    # ===================================================================
    # MODE 3: full pipeline — LLM → JG → retry
    #   local  → serial per task (single GPU)
    #   cloud  → parallel tasks (-j workers), each task's
    #            LLM→JG→retry cycle runs in its own thread
    # ===================================================================
    else:
        max_retries = args.max_retries
        if args.local:
            print(f"\n--- Full pipeline (serial, local GPU, max retries {max_retries}) ---")
            for t in tasks:
                results.append(_run_full_task(t, max_retries))
        else:
            print(f"\n--- Full pipeline (parallel, {args.workers} task workers, {llm_workers} LLM workers, cloud, max retries {max_retries}) ---")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_run_full_task, t, max_retries): t for t in tasks}
                for fut in as_completed(futures):
                    results.append(fut.result())

    # --- Summary ---
    _print_summary(results, args.llm_only)
    results_path = _write_results(results, output_dir)
    print(f"\nResults written to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
