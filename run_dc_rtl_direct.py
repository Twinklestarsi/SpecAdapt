#!/usr/bin/env python3
"""
Run Synopsys Design Compiler synthesis on direct RTL outputs under
RTL_DIRECT_compare/generated_rtl, using per-benchmark goals from spec_analysis.json.

For each .v file:
  1. Match file stem to a benchmark in spec_analysis.json
  2. Read optimization_target (AREA or TIMING) and map it to a DC goal
  3. Generate deterministic SDC constraints
  4. Generate a DC TCL script from synthesis_template.tcl
  5. Run dc_shell on the remote server via SSH
  6. Collect timing/area/power reports

Output structure:
  generated_rtl_DC_BY_TARGET/
    <metric>/<subcategory>/<benchmark>/<stem>/
      <name>.tcl
      <name>.sdc
      syn_output/
        timing.rpt  area.rpt  power.rpt  resources.rpt
        <name>_syn.sv  <name>_syn.ddc  <name>_syn.sdc

Usage:
  python run_dc_rtl_direct.py
  python run_dc_rtl_direct.py --dry-run
  python run_dc_rtl_direct.py --metric TIMING
  python run_dc_rtl_direct.py --bench uart
  python run_dc_rtl_direct.py -j 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from project_paths import PROJECT_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

ROOT = PROJECT_ROOT
INPUT_DIR = ROOT / "RTL_DIRECT_compare" / "generated_rtl"
GOAL_MAP_JSON = ROOT / "spec_analysis.json"
TOOLS_DIR = ROOT / "tools"
TEMPLATE_TCL = TOOLS_DIR / "synthesis_template.tcl"

_ENV_PATH = ROOT / ".env"


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
for _k, _v in _env.items():
    os.environ.setdefault(_k, _v)

DC_REMOTE_USER = os.environ.get("DC_REMOTE_USER", "").strip()
DC_REMOTE_HOST = os.environ.get("DC_REMOTE_HOST", "").strip()
DC_REMOTE_BASE = os.environ.get("DC_REMOTE_BASE", "").rstrip("/")
DC_ENV_SCRIPT = os.environ.get("DC_ENV_SCRIPT", "").strip()
DC_SHELL_PATH = os.environ.get("DC_SHELL_PATH", "dc_shell").strip() or "dc_shell"
LOCAL_BASE = ROOT

ASAP7_DB_PATH = os.environ.get("ASAP7_DB_PATH", "").rstrip("/")
ASAP7_LIBS = (
    "asap7sc7p5t_AO_RVT_TT_ccs_211120.db "
    "asap7sc7p5t_INVBUF_RVT_TT_ccs_220122.db "
    "asap7sc7p5t_OA_RVT_TT_ccs_211120.db "
    "asap7sc7p5t_SEQ_RVT_TT_ccs_220123.db "
    "asap7sc7p5t_SIMPLE_RVT_TT_ccs_211120.db"
)

DC_TIMEOUT = 600

_MODULE_RE = re.compile(r"^\s*module\s+([a-zA-Z_]\w*)\b", re.MULTILINE)


def detect_top_module(verilog_text: str) -> str | None:
    m = _MODULE_RE.search(verilog_text)
    return m.group(1) if m else None


def is_sequential(verilog_text: str) -> bool:
    return bool(re.search(r"always\s*@\s*\(\s*posedge", verilog_text))


def find_clock_port(verilog_text: str) -> str | None:
    m = re.search(r"always\s*@\s*\(\s*posedge\s+(\w+)", verilog_text)
    return m.group(1) if m else None


_GOAL_PRESETS = {
    "timing": {
        "clock_ps": 1000,
        "uncertainty_ps": 50,
        "src_latency_ps": 30,
        "net_latency_ps": 30,
        "io_delay_ps": 80,
        "max_transition_ps": 20,
        "max_cap_ff": 3.0,
        "compile_section": "compile",
    },
    "area": {
        "clock_ps": 5000,
        "uncertainty_ps": 200,
        "src_latency_ps": 100,
        "net_latency_ps": 100,
        "io_delay_ps": 400,
        "max_transition_ps": 50,
        "max_cap_ff": 10.0,
        "compile_section": "set_max_area 0\ncompile",
    },
}


def generate_sdc(verilog_text: str, top_module: str, goal: str) -> str:
    p = _GOAL_PRESETS[goal]
    seq = is_sequential(verilog_text)
    clk_port = find_clock_port(verilog_text) if seq else None

    clk_ps = p["clock_ps"]
    lines = [
        "###############################################",
        f"# SDC for {top_module}  (goal: {goal})",
        f"# Clock period: {clk_ps} ps ({clk_ps/1000:.1f} ns)",
        "###############################################",
        "",
    ]

    if seq and clk_port:
        lines += [
            f"create_clock -name clk -period {clk_ps} [get_ports {clk_port}]",
            f"set_clock_uncertainty {p['uncertainty_ps']} [get_clocks clk]",
            f"set_clock_latency -source {p['src_latency_ps']} [get_clocks clk]",
            f"set_clock_latency {p['net_latency_ps']} [get_clocks clk]",
            "",
            f"set_input_delay -clock clk {p['io_delay_ps']} [all_inputs]",
            f"set_output_delay -clock clk {p['io_delay_ps']} [all_outputs]",
        ]
    else:
        lines += [
            f"create_clock -name vclk -period {clk_ps}",
            f"set_clock_uncertainty {p['uncertainty_ps']} [get_clocks vclk]",
            "",
            f"set_input_delay -clock vclk {p['io_delay_ps']} [all_inputs]",
            f"set_output_delay -clock vclk {p['io_delay_ps']} [all_outputs]",
        ]

    lines += [
        "",
        f"# Electrical constraints (ASAP7, goal={goal})",
        f"set_max_transition {p['max_transition_ps']} [current_design]",
        f"set_max_capacitance {p['max_cap_ff']} [current_design]",
        "",
    ]
    return "\n".join(lines) + "\n"


def _local_to_remote(local_path: Path) -> str:
    if not DC_REMOTE_BASE:
        raise RuntimeError("DC_REMOTE_BASE is not configured in .env")
    rel = local_path.resolve().relative_to(LOCAL_BASE.resolve())
    return f"{DC_REMOTE_BASE}/{rel}"


def _build_remote_dc_command(remote_out: str, remote_tcl: str) -> str:
    steps = []
    if DC_ENV_SCRIPT:
        steps.append(f"source {shlex.quote(DC_ENV_SCRIPT)}")
    steps.extend(
        [
            f"cd {shlex.quote(remote_out)}",
            f"{shlex.quote(DC_SHELL_PATH)} -f {shlex.quote(remote_tcl)}",
        ]
    )
    return " && ".join(steps)


def load_goal_map(spec_json: Path) -> dict[str, str]:
    if not spec_json.is_file():
        print(f"ERROR: goal map JSON not found: {spec_json}", file=sys.stderr)
        return {}

    data = json.loads(spec_json.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]

    mapping: dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        benchmark = str(item.get("benchmark", "")).strip()
        target = str(item.get("optimization_target", "")).strip().lower()
        if benchmark and target in {"area", "timing"}:
            mapping[benchmark] = target
    return mapping


def discover_tasks(
    input_dir: Path,
    goal_map: dict[str, str],
    bench_filter: str | None = None,
    metric_filter: str | None = None,
) -> list[dict]:
    tasks = []

    if not input_dir.is_dir():
        print(f"ERROR: {input_dir} not found", file=sys.stderr)
        return []

    for vf in sorted(input_dir.rglob("*.v")):
        stem = vf.stem
        benchmark = stem
        if bench_filter and bench_filter not in benchmark:
            continue

        goal = goal_map.get(benchmark)
        if goal is None:
            continue
        metric = goal.upper()
        if metric_filter and metric != metric_filter:
            continue

        rel_parent = vf.parent.relative_to(input_dir)
        subcategory = rel_parent.as_posix() if str(rel_parent) != "." else "flat"

        tasks.append({
            "v_path": vf,
            "metric": metric,
            "subcategory": subcategory,
            "benchmark": benchmark,
            "stem": stem,
            "goal": goal,
        })

    return tasks


_print_lock = threading.Lock()


def _log(msg: str, end: str = "\n") -> None:
    with _print_lock:
        print(msg, end=end, flush=True)


def run_dc_task(task: dict, output_root: Path, timeout: int = DC_TIMEOUT) -> dict:
    v_path = task["v_path"]
    stem = task["stem"]
    goal = task["goal"]
    label = f"{task['benchmark']}/{stem}"

    out_dir = output_root / task["metric"] / task["subcategory"] / task["benchmark"] / stem
    result = {
        "benchmark": task["benchmark"],
        "transform": stem,
        "metric": task["metric"],
        "subcategory": task["subcategory"],
        "goal": goal,
        "success": False,
        "error": "",
    }
    missing_remote = [
        name for name, value in (
            ("DC_REMOTE_USER", DC_REMOTE_USER),
            ("DC_REMOTE_HOST", DC_REMOTE_HOST),
            ("DC_REMOTE_BASE", DC_REMOTE_BASE),
            ("ASAP7_DB_PATH", ASAP7_DB_PATH),
        ) if not value
    ]
    if missing_remote:
        result["error"] = "missing DC configuration: " + ", ".join(missing_remote)
        return result

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verilog_text = v_path.read_text(encoding="utf-8", errors="replace")
    top = detect_top_module(verilog_text)
    if not top:
        result["error"] = "cannot detect top module"
        _log(f"  [SKIP] {label}: no module found")
        return result

    sdc_path = out_dir / f"{stem}.sdc"
    sdc_path.write_text(generate_sdc(verilog_text, top, goal), encoding="utf-8")

    local_v = out_dir / v_path.name
    shutil.copy2(v_path, local_v)

    remote_out = _local_to_remote(out_dir)
    remote_v = _local_to_remote(local_v)
    remote_sdc = _local_to_remote(sdc_path)

    tcl_path = out_dir / f"{stem}.tcl"
    from tcl_agent import generate_tcl_from_template

    generate_tcl_from_template(
        TEMPLATE_TCL,
        tcl_path,
        top_module=top,
        design_name=stem,
        rtl_path=str(Path(remote_v).parent),
        rtl_files=f"[list {remote_v}]",
        sdc_path=remote_sdc,
        search_path=ASAP7_DB_PATH,
        target_library=ASAP7_LIBS,
        compile_section=_GOAL_PRESETS[goal]["compile_section"],
    )

    remote_tcl = _local_to_remote(tcl_path)
    cmd = [
        "ssh", f"{DC_REMOTE_USER}@{DC_REMOTE_HOST}",
        _build_remote_dc_command(remote_out, remote_tcl),
    ]

    _log(f"  [DC] {label} ...", end=" ")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
        )
        output = proc.stdout + "\n" + proc.stderr

        log_path = out_dir / "dc_shell.log"
        log_path.write_text(output, encoding="utf-8")

        if "Synthesis completed successfully!" in output:
            result["success"] = True
            _log("OK")
        else:
            err_lines = [ln.strip() for ln in output.splitlines() if "Error:" in ln or "ERROR" in ln]
            result["error"] = err_lines[0] if err_lines else "DC did not complete"
            result["raw_tail"] = output[-2000:]
            _log(f"FAIL ({result['error'][:80]})")
    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
        _log("TIMEOUT")

    return result


def parse_reports(out_dir: Path) -> dict:
    reports = {}
    syn_out = out_dir / "syn_output"

    timing_rpt = syn_out / "timing.rpt"
    if timing_rpt.is_file():
        text = timing_rpt.read_text(errors="replace")
        m = re.search(r"data arrival time\s+([-\d.]+)", text)
        if m:
            reports["delay_ps"] = abs(float(m.group(1)))

    area_rpt = syn_out / "area.rpt"
    if area_rpt.is_file():
        text = area_rpt.read_text(errors="replace")
        m = re.search(r"Total cell area:\s*([\d.]+)", text)
        if m:
            reports["total_cell_area"] = float(m.group(1))

    power_rpt = syn_out / "power.rpt"
    if power_rpt.is_file():
        text = power_rpt.read_text(errors="replace")
        m = re.search(r"Total Dynamic Power\s*=\s*([\d.eE+-]+)\s*(\w+)", text)
        if m:
            reports["dynamic_power"] = f"{m.group(1)} {m.group(2)}"
        m = re.search(r"Cell Leakage Power\s*=\s*([\d.eE+-]+)\s*(\w+)", text)
        if m:
            reports["leakage_power"] = f"{m.group(1)} {m.group(2)}"

    return reports


def _print_summary(results: list[dict]) -> None:
    total = len(results)
    ok = sum(1 for r in results if r["success"])
    fail = total - ok
    print(f"\n{'='*60}")
    print(f"DC synthesis: {ok}/{total} succeeded, {fail} failed")
    if fail > 0:
        errors = {}
        for r in results:
            if not r["success"]:
                e = r.get("error", "unknown")[:60]
                errors[e] = errors.get(e, 0) + 1
        for e, cnt in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {cnt:4d}x  {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run DC synthesis on RTL_DIRECT_compare/generated_rtl")
    ap.add_argument("--input", type=Path, default=INPUT_DIR,
                    help=f"Input directory (default: {INPUT_DIR})")
    ap.add_argument("--goal-map-json", type=Path, default=GOAL_MAP_JSON,
                    help=f"Stage 1 JSON with optimization_target mapping (default: {GOAL_MAP_JSON})")
    ap.add_argument("--output", type=Path, default=ROOT / "generated_rtl_DC_BY_TARGET",
                    help="Output directory")
    ap.add_argument("--metric", choices=["TIMING", "AREA"], default=None,
                    help="Only process one metric")
    ap.add_argument("--bench", type=str, default=None,
                    help="Filter benchmarks by substring")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max tasks to process")
    ap.add_argument("-j", "--workers", type=int, default=4,
                    help="Parallel DC workers (default: 4)")
    ap.add_argument("--timeout", type=int, default=DC_TIMEOUT,
                    help=f"DC timeout in seconds (default: {DC_TIMEOUT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show tasks without running")
    args = ap.parse_args()

    goal_map = load_goal_map(args.goal_map_json)
    tasks = discover_tasks(args.input, goal_map, args.bench, args.metric)
    if not tasks:
        print("No .v files found to synthesize.")
        return 1

    if args.limit:
        tasks = tasks[:args.limit]

    print(f"Found {len(tasks)} .v files to synthesize")
    print(f"Goals: per-file from {args.goal_map_json} -> {sorted({t['goal'] for t in tasks})}")
    print(f"Output: {args.output}")
    print(f"Workers: {args.workers}")

    if args.dry_run:
        for t in tasks:
            print(f"  {t['metric']}/{t['subcategory']}/{t['benchmark']}/{t['stem']}  goal={t['goal']}")
        print(f"\n[dry-run] Would process {len(tasks)} files.")
        return 0

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_dc_task, t, args.output, args.timeout): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            if r["success"]:
                out_dir = args.output / r["metric"] / r["subcategory"] / r["benchmark"] / r["transform"]
                r["reports"] = parse_reports(out_dir)
            results.append(r)

    _print_summary(results)

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
