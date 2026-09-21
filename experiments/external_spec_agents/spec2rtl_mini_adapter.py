#!/usr/bin/env python3
"""Boundary adapter for the public community Mini-Spec2RTL agent.

The upstream mini project is a lightweight Spec2RTL-style flow.  Its normal
CLI asks the model to emit a very large JSON design plan; long C-first specs
can exceed the provider's visible completion budget.  This adapter keeps the
agent's own RTL/TB prompts and verifier boundary, but builds the small plan
shell deterministically from the *public specification only*.  The golden RTL
and benchmark testbench never enter the generation prompts.

The adapter is intentionally named ``spec2rtl_mini`` so its results cannot be
confused with NVIDIA's unpublished Spec2RTL-Agent implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external_agents" / "AgAgAggA_mini-spec2rtl-agent"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_root_env() -> dict[str, str]:
    try:
        from dotenv import dotenv_values

        values = dotenv_values(ROOT / ".env")
        return {str(k): str(v) for k, v in values.items() if k and v is not None}
    except Exception:  # pragma: no cover - stdlib fallback
        out: dict[str, str] = {}
        env_path = ROOT / ".env"
        if not env_path.is_file():
            return out
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip('"').strip("'")
        return out


def _resolve(path_value: Any, case_file: Path, label: str) -> Path:
    p = Path(str(path_value)).expanduser()
    candidates = [p] if p.is_absolute() else [case_file.parent / p, ROOT / p]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} does not resolve to a file: {path_value}")


def _parse_ports(spec: str) -> list[dict[str, Any]]:
    ports: list[dict[str, Any]] = []
    # C-first specs intentionally use one port per bullet.  Accept optional
    # wire/reg/logic keywords while preserving names and widths exactly.
    rx = re.compile(
        r"^\s*-\s*(input|output)\s+(?:(wire|reg|logic)\s+)?"
        r"(?:\[\s*([^]]+)\s*:\s*([^]]+)\s*\]\s+)?([A-Za-z_]\w*)\b",
        re.IGNORECASE,
    )
    # CVDP specifications put the authoritative interface in a fenced
    # SystemVerilog declaration instead of bullet points.  Restrict fallback
    # parsing to the module header so prose cannot invent ports.
    if not any(re.match(r"^\s*-\s*(input|output)\b", line, re.IGNORECASE) for line in spec.splitlines()):
        block = re.search(r"module\s+[A-Za-z_]\w*\s*(?:#\s*\(.*?\))?\s*\((.*?)\);", spec, re.IGNORECASE | re.DOTALL)
        if block:
            header = block.group(1)
            spec_lines = [x.strip().rstrip(",") for x in header.splitlines()]
            # Feed the same normalizer below; declarations have no bullet.
            spec = "\n".join("- " + x for x in spec_lines)
    for line in spec.splitlines():
        inline = re.match(
            r"^\s*-\s*(input|output)\s+(?:\[\s*([^]]+)\s*:\s*([^]]+)\s*\]\s+)?(.+)$",
            line,
            re.IGNORECASE,
        )
        if inline and "," in inline.group(4):
            direction, hi, lo, names = inline.groups()
            # Strip prose after the comma-separated interface list.  These
            # specs use identifiers only on such lines.
            names = [n.strip().split()[0] for n in names.split(",") if n.strip()]
            for name in names:
                if not re.fullmatch(r"[A-Za-z_]\w*", name):
                    continue
                if hi is None:
                    width = 1
                    rng = "[0:0]"
                elif re.fullmatch(r"\d+", hi or "") and re.fullmatch(r"\d+", lo or ""):
                    width = abs(int(hi) - int(lo)) + 1
                    rng = f"[{hi}:{lo}]"
                elif re.fullmatch(r"N\s*-\s*1", hi or "", re.IGNORECASE) and lo.strip() == "0":
                    width, rng = 8, "[7:0]"
                else:
                    width, rng = 1, "[0:0]"
                ports.append({
                    "name": name,
                    "direction": direction.lower(),
                    "width": width,
                    "range": rng,
                    "type": "wire" if direction.lower() == "input" else "reg",
                })
            continue
        m = rx.match(line)
        if not m:
            continue
        direction, typ, hi, lo, name = m.groups()
        if hi is None:
            width = 1
        elif re.fullmatch(r"\d+", hi or "") and re.fullmatch(r"\d+", lo or ""):
            width = abs(int(hi) - int(lo)) + 1
        elif re.fullmatch(r"N\s*-\s*1", hi or "", re.IGNORECASE) and lo.strip() == "0":
            width = 8  # documented default parameter in the CVDP cases
        else:
            width = 1
        ports.append(
            {
                "name": name,
                "direction": direction.lower(),
                "width": width,
                "range": f"[{hi}:{lo}]" if hi is not None and not re.search(r"[A-Za-z]", hi + lo) else f"[{width - 1}:0]",
                "type": (typ or ("wire" if direction.lower() == "input" else "reg")).lower(),
            }
        )
    # NVDLA case specifications use Markdown port tables instead of the
    # bullet-style interface used by the original C-first cases.  Parse only
    # rows whose second column is an explicit direction; internal state tables
    # therefore cannot accidentally become module ports.  Widths are written
    # as either a scalar count (``22``/``22 bits``), a Verilog range, or the
    # Chinese ``22位`` form used by the generated NVDLA documents.
    if not ports:
        table_rx = re.compile(
            r"^\s*\|\s*([A-Za-z_]\w*)\s*\|\s*"
            r"(input|output|inout|in|out|输入|输出|双向)(?:\s+(?:wire|reg|logic))?\s*\|\s*"
            r"([^|]+?)\s*\|",
            re.IGNORECASE,
        )
        seen: set[tuple[str, str]] = set()
        for line in spec.splitlines():
            match = table_rx.match(line)
            if not match:
                continue
            name, direction, width_text = match.groups()
            direction = {"输入": "input", "输出": "output", "双向": "inout", "in": "input", "out": "output"}.get(direction, direction.lower())
            key = (name, direction)
            if key in seen:
                continue
            seen.add(key)
            range_match = re.search(r"\[\s*([^]]+)\s*:\s*([^]]+)\s*\]", width_text)
            if range_match:
                hi, lo = range_match.groups()
                if re.fullmatch(r"\d+", hi) and re.fullmatch(r"\d+", lo):
                    width = abs(int(hi) - int(lo)) + 1
                    rng = f"[{hi}:{lo}]"
                else:
                    width = 1
                    rng = "[0:0]"
            else:
                scalar_match = re.search(r"(?<![A-Za-z_])(\d+)\s*(?:bits?|位)?\b", width_text, re.IGNORECASE)
                width = int(scalar_match.group(1)) if scalar_match else 1
                rng = f"[{width - 1}:0]" if width > 1 else "[0:0]"
            ports.append(
                {
                    "name": name,
                    "direction": direction,
                    "width": width,
                    "range": rng,
                    "type": "wire" if direction == "input" else "reg",
                }
            )
    # A few conservative NVDLA specs use prose lists such as
    # ``- input: foo, bar`` or ``Inputs: - foo, bar``.  These lists are still
    # public specification text, so recover the declared names without
    # consulting the golden RTL.  Unknown widths intentionally remain scalar
    # rather than inventing a vector size.
    if not ports:
        def add_decl(name: str, direction: str, range_text: str | None = None) -> None:
            name = name.strip()
            if not re.fullmatch(r"[A-Za-z_]\w*", name):
                return
            if range_text:
                match = re.fullmatch(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", range_text)
                if match:
                    hi, lo = match.groups()
                    width = abs(int(hi) - int(lo)) + 1
                    rng = f"[{hi}:{lo}]"
                else:
                    width, rng = 1, "[0:0]"
            else:
                width, rng = 1, "[0:0]"
            if not any(p["name"] == name and p["direction"] == direction for p in ports):
                ports.append({"name": name, "direction": direction, "width": width, "range": rng, "type": "wire" if direction == "input" else "reg"})

        section_direction: str | None = None
        for line in spec.splitlines():
            heading = re.match(r"^\s*(?:APB-side|CSB-side)?\s*(inputs?|outputs?)\s*:\s*$", line, re.IGNORECASE)
            if heading:
                section_direction = "input" if heading.group(1).lower().startswith("input") else "output"
                continue
            colon = re.match(r"^\s*-\s*(input|output|inout|in|out|输入|输出|双向)\s*:\s*(.+)$", line, re.IGNORECASE)
            if colon:
                direction = {"输入": "input", "输出": "output", "双向": "inout", "in": "input", "out": "output"}.get(colon.group(1).lower(), colon.group(1).lower())
                body = colon.group(2)
                # A line may describe two interfaces separated by ``;``;
                # stop each segment before explanatory prose after a colon.
                for segment in re.split(r";", body):
                    segment = segment.strip()
                    for token in re.split(r"\s*,\s*|\s+\/\s+", segment):
                        m = re.match(r"([A-Za-z_]\w*)\s*(\[[^]]+\])?", token.strip())
                        if m:
                            add_decl(m.group(1), direction, m.group(2))
                continue
            if section_direction:
                bullet = re.match(r"^\s*-\s*(.+)$", line)
                if bullet:
                    body = bullet.group(1).split(":", 1)[0].strip()
                    for token in re.split(r"\s*,\s*|\s+\/\s+", body):
                        m = re.match(r"([A-Za-z_]\w*)\s*(\[[^]]+\])?", token.strip())
                        if m:
                            add_decl(m.group(1), section_direction, m.group(2))
    if not ports:
        raise ValueError("no interface ports found in specification")
    return ports


def _module_header(plan: dict[str, Any]) -> str:
    decls: list[str] = []
    for port in plan["ports"]:
        d = port["direction"]
        typ = port.get("type", "wire")
        width = "" if int(port.get("width", 1)) == 1 else str(port["range"])
        decls.append(f"{d} {typ} {width} {port['name']}".replace("  ", " ").strip())
    return f"module {plan['module_name']} (" + ", ".join(decls) + ");"


def _build_plan(case: dict[str, Any], spec: str) -> dict[str, Any]:
    from src.schema import validate_design_plan

    # nvdla22 includes several conservative/path-level specs that deliberately
    # omit an exhaustive port table.  The harness may provide an explicit
    # interface contract extracted from the public module declaration.  It is
    # used only to keep the generated top compilable; the RTL body and all
    # behavior still come from the specification text, and the reference RTL
    # is never sent to the LLM prompt.
    ports = case.get("interface_ports") or _parse_ports(spec)
    design_type = "sequential" if re.search(
        r"rising edge|posedge|clock|registered|state changes|cycle-by-cycle",
        spec,
        re.IGNORECASE,
    ) else "combinational"
    clock_name = next((p["name"] for p in ports if p["name"] in {"clk", "clock"}), None)
    reset_name = next(
        (p["name"] for p in ports if p["name"].lower() in {"rst", "rst_n", "reset", "reset_n", "rstn"}),
        None,
    )
    state_registers = [
        {"name": p["name"], "width": p["width"], "reset_value": "0", "type": p.get("type", "reg")}
        for p in ports
        if p["direction"] == "output" and design_type == "sequential"
    ]
    raw = {
        "design_name": str(case.get("id") or case.get("benchmark") or "spec2rtl_case"),
        "module_name": str(case.get("top_module") or "TopModule"),
        "design_type": design_type,
        "clock": {"name": clock_name or "clk", "edge": "posedge", "period_ns": 10}
        if design_type == "sequential"
        else None,
        "reset": {
            "name": reset_name or "rst_n",
            "active_low": bool(reset_name and reset_name.lower().endswith("_n")),
            "synchronous": bool(re.search(r"synchronous", spec, re.IGNORECASE)),
            "type": "sync" if re.search(r"synchronous", spec, re.IGNORECASE) else "async",
        }
        if design_type == "sequential"
        else None,
        "ports": ports,
        "state_registers": state_registers,
        "behavior": [spec],
        "test_scenarios": [],
        "candidate_assertions": [],
        "ambiguities": [],
    }
    return validate_design_plan(raw)


def run_case(case_file: Path, output_dir: Path, config_file: Optional[Path]) -> dict[str, Any]:
    from src.config import load_settings
    from src.llm_client import build_llm_client
    from src.rtl_generator import _try_llm_rtl, _strip_code_fence as strip_rtl
    from src.tb_generator import _try_llm_tb

    case = _read_json(case_file)
    spec = _resolve(case["spec_path"], case_file, "spec_path")
    # Resolve these only for provenance.  Their contents are deliberately not
    # passed to either generation prompt.
    golden = _resolve(case["golden_path"], case_file, "golden_path")
    tb = _resolve(case["testbench_path"], case_file, "testbench_path") if case.get("testbench_path") else None
    spec_text = spec.read_text(encoding="utf-8")
    plan = _build_plan(case, spec_text)
    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "design_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(_load_root_env())
    cfg = _read_json(config_file) if config_file else {}
    model = str(cfg.get("model") or env.get("OPENAI_MODEL") or "deepseek-v4-flash")
    try:
        mini_rtl_max_tokens = int(cfg.get("mini_rtl_max_tokens", 8192))
    except (TypeError, ValueError) as exc:
        raise ValueError("mini_rtl_max_tokens must be an integer") from exc
    base_url = str(cfg.get("base_url") or env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE_URL") or "")
    if not env.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is not configured")
    env.update({
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": model,
        "LLM_PROVIDER_SPEC": "openai",
        "LLM_MODEL_SPEC": model,
        "LLM_PROVIDER_PLAN": "openai",
        "LLM_MODEL_PLAN": model,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_BASE_URL": base_url,
        "PYTHONNOUSERSITE": "1",
    })
    os.environ.update(env)
    client = build_llm_client(load_settings(), role="default")
    header = _module_header(plan)
    generation_log: list[str] = [f"model={model}", f"base_url={base_url}", f"header={header}"]
    rtl = None
    syntax_feedback = "No external context. Emit concise synthesizable SystemVerilog only."
    syntax_log = ""
    # Keep a small local repair loop, matching the upstream project's
    # reflection intent.  It catches formatting/compile errors before the
    # shared JG/DC gate, without ever exposing the golden RTL.
    for attempt in range(3):
        rtl_try = _try_llm_rtl(
            client,
            plan,
            syntax_feedback,
            lang="systemverilog",
            authoritative_module=header,
            problem_description=spec_text,
            max_tokens=mini_rtl_max_tokens,
        )
        if not rtl_try:
            syntax_feedback = "Previous generation returned no RTL. Retry with a compact complete module."
            continue
        rtl_try = strip_rtl(rtl_try)
        candidate = out / f"rtl_attempt_{attempt}.sv"
        candidate.write_text(rtl_try, encoding="utf-8")
        cp = subprocess.run(
            [str(ROOT / ".conda-env/bin/iverilog"), "-g2012", "-t", "null", "-s", str(plan["module_name"]), str(candidate)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        syntax_log = (cp.stdout or "") + (cp.stderr or "")
        if cp.returncode == 0:
            rtl = rtl_try
            break
        syntax_feedback = (
            "The previous RTL failed Icarus syntax checking. Repair it while "
            "preserving the exact interface and behavior. Compiler diagnostics:\n"
            + syntax_log[-4000:]
            + "\nOutput only the repaired module."
        )
    if rtl:
        (out / "rtl_generated.sv").write_text(rtl, encoding="utf-8")
    (out / "syntax_generation.log").write_text(syntax_log, encoding="utf-8")
    # The shared benchmark runner uses JasperGold with the public golden RTL
    # for the correctness gate.  Keep the upstream TB stage available for a
    # deliberate smoke/debug run, but skip it by default in the batch so one
    # case consumes one generation call rather than an unrelated second call.
    tb_code = None
    if os.environ.get("SPEC2RTL_MINI_GENERATE_TB", "0").lower() in {"1", "true", "yes"}:
        tb_code = _try_llm_tb(
            client,
            plan,
            "No external context. Build a self-checking testbench from the specification below.",
            lang="systemverilog",
            authoritative_module=header,
            problem_description=spec_text,
        )
    if tb_code:
        (out / "tb_generated.sv").write_text(tb_code, encoding="utf-8")
    (out / "generation.log").write_text("\n".join(generation_log) + "\n", encoding="utf-8")
    result = {
        "status": "candidate" if rtl else "error",
        "rtl_path": str((out / "rtl_generated.sv").resolve()) if rtl else None,
        "tb_path": str((out / "tb_generated.sv").resolve()) if tb_code else None,
        "error": None if rtl else "LLM did not return RTL",
        "agent": "AgAgAggA/mini-spec2rtl-agent",
        "model": model,
        "case_id": case.get("id", case_file.stem),
        "source_sha256": {
            "spec": __import__("hashlib").sha256(spec.read_bytes()).hexdigest(),
            "golden": __import__("hashlib").sha256(golden.read_bytes()).hexdigest(),
            **({"testbench": __import__("hashlib").sha256(tb.read_bytes()).hexdigest()} if tb else {}),
        },
        "golden_used_in_prompt": False,
        "benchmark_tb_used_in_prompt": False,
        "token_usage": {
            "prompt_tokens": int(getattr(client, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(client, "output_tokens", 0) or 0),
            "total_tokens": int(getattr(client, "total_tokens", 0) or 0),
            "request_count": int(getattr(client, "request_count", 0) or 0),
            "source": "mini_spec2rtl_llm_client_response_usage",
        },
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config-json", type=Path)
    args = ap.parse_args(argv)
    try:
        result = run_case(args.case_json.resolve(), args.output_dir, args.config_json.resolve() if args.config_json else None)
    except Exception as exc:  # noqa: BLE001
        result = {"status": "error", "rtl_path": None, "error": f"{type(exc).__name__}: {exc}", "agent": "AgAgAggA/mini-spec2rtl-agent"}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "candidate" else 1


if __name__ == "__main__":
    raise SystemExit(main())
