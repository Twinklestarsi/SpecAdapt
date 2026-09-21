#!/usr/bin/env python3
"""
Run Synopsys Design Compiler synthesis on all .v files under LLM_V2V_COLLECTED.

For each .v file (excluding *_original.v and *sampled_*.v reference copies):
  1. Generate a deterministic SDC (2000ps clock, ASAP7 constraints)
  2. Generate a DC TCL script from synthesis_template.tcl
  3. Run dc_shell on the remote server via SSH
  4. Collect timing/area/power reports

Output structure (mirrors input):
  LLM_V2V_COLLECTED_DC/
    <metric>/<subcategory>/<benchmark>/<vfile_stem>/
      <name>.tcl
      <name>.sdc
      syn_output/
        timing.rpt  area.rpt  power.rpt  resources.rpt
        <name>_syn.sv  <name>_syn.ddc  <name>_syn.sdc

Usage:
  python run_dc.py --goal timing                      # aggressive timing
  python run_dc.py --goal area                        # aggressive area
  python run_dc.py --goal timing --llm-sdc            # LLM-generated SDC
  python run_dc.py --metric TIMING                    # only TIMING metric
  python run_dc.py --dry-run                          # preview
  python run_dc.py -j 1                               # serial SSH sessions (default)
  python run_dc.py --bench add1 --transform SPLIT_OP  # filter
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import dotenv_values

from project_paths import PROJECT_ROOT

# Make tools/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = PROJECT_ROOT
INPUT_DIR = ROOT / "LLM_V2V_COLLECTED"
TOOLS_DIR = ROOT / "tools"
TEMPLATE_TCL = TOOLS_DIR / "synthesis_template.tcl"

# Remote Design Compiler configuration
_ENV_PATH = ROOT / ".env"


def _load_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {
        str(key): str(value)
        for key, value in dotenv_values(path).items()
        if value is not None
    }


_env = _load_env(_ENV_PATH)
# Expose .env values as os.environ so tools/llm_api.py can read them
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

DC_TIMEOUT = 600  # seconds


def _resolve_dc_max_cores(value: object = None) -> int:
    """Resolve the per-job DC core cap; active flows default to one core."""

    raw = value if value is not None else os.environ.get("DC_MAX_CORES", "1")
    try:
        cores = int(raw)
    except (TypeError, ValueError):
        cores = 1
    return max(1, cores)

# ---------------------------------------------------------------------------
# Verilog analysis
# ---------------------------------------------------------------------------

_MODULE_RE = re.compile(r"^\s*module\s+([a-zA-Z_]\w*)\b", re.MULTILINE)


def detect_top_module(verilog_text: str) -> str | None:
    m = _MODULE_RE.search(verilog_text)
    return m.group(1) if m else None


def is_sequential(verilog_text: str) -> bool:
    return bool(re.search(r"always\s*@\s*\(\s*posedge", verilog_text))


def find_clock_port(verilog_text: str) -> str | None:
    """Find clock port name from sequential Verilog."""
    m = re.search(r"always\s*@\s*\(\s*posedge\s+(\w+)", verilog_text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# SDC generation (deterministic, no LLM)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Goal-specific SDC / compile presets
# ---------------------------------------------------------------------------

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
        "output_suffix": "DC_TIMING",
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
        "output_suffix": "DC_AREA",
    },
}


def generate_sdc(verilog_text: str, top_module: str, goal: str = "timing") -> str:
    """
    Generate ASAP7-targeted SDC with goal-dependent constraints.
    goal='timing' → tight clock (1000ps), aggressive electrical limits.
    goal='area'   → relaxed clock (5000ps), loose limits so DC can minimise area.
    """
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
            f"set data_inputs [remove_from_collection [all_inputs] [get_ports {clk_port}]]",
            "if {[sizeof_collection $data_inputs] > 0} {",
            f"  set_input_delay -clock clk {p['io_delay_ps']} $data_inputs",
            "}",
            "set data_outputs [all_outputs]",
            "if {[sizeof_collection $data_outputs] > 0} {",
            f"  set_output_delay -clock clk {p['io_delay_ps']} $data_outputs",
            "}",
        ]
    else:
        lines += [
            f"create_clock -name vclk -period {clk_ps}",
            f"set_clock_uncertainty {p['uncertainty_ps']} [get_clocks vclk]",
            "",
            "set data_inputs [all_inputs]",
            "if {[sizeof_collection $data_inputs] > 0} {",
            f"  set_input_delay -clock vclk {p['io_delay_ps']} $data_inputs",
            "}",
            "set data_outputs [all_outputs]",
            "if {[sizeof_collection $data_outputs] > 0} {",
            f"  set_output_delay -clock vclk {p['io_delay_ps']} $data_outputs",
            "}",
        ]

    lines += [
        "",
        f"# Electrical constraints (ASAP7, goal={goal})",
        f"set_max_transition {p['max_transition_ps']} [current_design]",
        f"set_max_capacitance {p['max_cap_ff']} [current_design]",
        "",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Path mapping
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

def discover_tasks(
    input_dir: Path,
    metric_filter: str | None = None,
    bench_filter: str | None = None,
    transform_filter: str | None = None,
) -> list[dict]:
    """
    Find all LLM-generated .v files in the collected directory.
    Structure: <input_dir>/<METRIC>/<subcategory>/<benchmark>/*.v
    Skips reference copies (*_original.v, sampled_*.v).
    """
    tasks = []
    suffix_re = re.compile(r"_[A-Z][A-Z_]+$")

    if not input_dir.is_dir():
        print(f"ERROR: {input_dir} not found", file=sys.stderr)
        return []

    for metric_dir in sorted(input_dir.iterdir()):
        if not metric_dir.is_dir():
            continue
        metric = metric_dir.name
        if metric_filter and metric != metric_filter:
            continue

        for sub_dir in sorted(metric_dir.iterdir()):
            if not sub_dir.is_dir():
                continue

            for bench_dir in sorted(sub_dir.iterdir()):
                if not bench_dir.is_dir():
                    continue
                bench_name = bench_dir.name
                if bench_filter and bench_filter not in bench_name:
                    continue

                for vf in sorted(bench_dir.glob("*.v")):
                    # Skip reference .v files
                    if "_original.v" in vf.name:
                        continue
                    if vf.name.startswith("sampled_"):
                        continue
                    if "_combinational.v" in vf.name:
                        continue

                    # This should be a <baseline>_<TRANSFORM>.v file
                    stem = vf.stem
                    if not suffix_re.search(stem):
                        continue

                    if transform_filter and transform_filter not in stem:
                        continue

                    tasks.append({
                        "v_path": vf,
                        "metric": metric,
                        "subcategory": sub_dir.name,
                        "benchmark": bench_name,
                        "stem": stem,
                    })

    return tasks


def discover_baseline_tasks(
    input_dir: Path,
    bench_filter: str | None = None,
) -> list[dict]:
    """
    Find baseline .v files in original_AREA / original_TIMING directories.
    Structure: <input_dir>/<subcategory>/<benchmark>/<baseline>.v
    The metric is inferred from the input_dir name (original_AREA → AREA).
    """
    tasks = []

    if not input_dir.is_dir():
        print(f"ERROR: {input_dir} not found", file=sys.stderr)
        return []

    # Infer metric from dir name: original_AREA → AREA, original_TIMING → TIMING
    dir_name = input_dir.name
    if "_" in dir_name:
        metric = dir_name.split("_", 1)[1]
    else:
        metric = dir_name

    for sub_dir in sorted(input_dir.iterdir()):
        if not sub_dir.is_dir():
            continue

        for bench_dir in sorted(sub_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            bench_name = bench_dir.name
            if bench_filter and bench_filter not in bench_name:
                continue

            # Find the single .v file (either *_original.v or sampled_*.v)
            v_files = list(bench_dir.glob("*_original.v")) + list(bench_dir.glob("sampled_*.v"))
            if not v_files:
                v_files = [f for f in bench_dir.glob("*.v")]
            if not v_files:
                continue

            vf = v_files[0]
            tasks.append({
                "v_path": vf,
                "metric": metric,
                "subcategory": sub_dir.name,
                "benchmark": bench_name,
                "stem": vf.stem,
            })

    return tasks




# ---------------------------------------------------------------------------
# Run DC
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def _log(msg: str, end: str = "\n") -> None:
    with _print_lock:
        print(msg, end=end, flush=True)


def _prepare_clean_dir(path: Path) -> None:
    """Create an empty directory, tolerating delayed deletes on SMB/CIFS mounts."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return

    for attempt in range(3):
        try:
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            return
        except OSError:
            if attempt == 2:
                break
            time.sleep(0.25 * (attempt + 1))

    stale = path.with_name(f"{path.name}.stale_{int(time.time() * 1000)}")
    try:
        path.rename(stale)
        path.mkdir(parents=True, exist_ok=True)
        _log(f"  [DC] warning: moved stale output dir to {stale}")
        return
    except OSError:
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)


def run_dc_task(task: dict, output_root: Path, timeout: int = DC_TIMEOUT,
                llm_sdc: bool = False, goal: str = "timing") -> dict:
    """Run DC synthesis for one .v file. Returns result dict."""
    v_path = task["v_path"]
    stem = task["stem"]
    label = f"{task['benchmark']}/{stem}"

    # Output directory: one folder per .v file
    out_dir = (output_root / task["metric"] / task["subcategory"]
               / task["benchmark"] / stem)

    result = {
        "benchmark": task["benchmark"],
        "transform": stem,
        "metric": task["metric"],
        "subcategory": task["subcategory"],
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

    # Clean previous artifacts only after configuration has been validated.
    _prepare_clean_dir(out_dir)

    # Read Verilog
    verilog_text = v_path.read_text(encoding="utf-8", errors="replace")
    top = str(task.get("top_module") or "").strip() or detect_top_module(verilog_text)
    if not top:
        result["error"] = "cannot detect top module"
        _log(f"  [SKIP] {label}: no module found")
        return result
    if not re.search(rf"^\s*module\s+{re.escape(top)}\b", verilog_text, re.MULTILINE):
        result["error"] = f"requested top module not found: {top}"
        _log(f"  [SKIP] {label}: requested top {top} not found")
        return result

    # Generate SDC
    sdc_path = out_dir / f"{stem}.sdc"
    if llm_sdc:
        from tcl_agent import generate_sdc_for_verilog
        _log(f"  [SDC-LLM] {label} ...", end=" ")
        try:
            sdc_text = generate_sdc_for_verilog(
                verilog_text, stem, top,
                str(TOOLS_DIR / "agent_prompts.json"),
            )
            _log("OK")
        except Exception as e:
            _log(f"FAIL ({e}), falling back to deterministic SDC")
            sdc_text = generate_sdc(verilog_text, top, goal)
    else:
        sdc_text = generate_sdc(verilog_text, top, goal)
    sdc_path.write_text(sdc_text, encoding="utf-8")

    # Copy .v into output dir
    local_v = out_dir / v_path.name
    shutil.copy2(v_path, local_v)
    rtl_format = "sverilog" if local_v.suffix.lower() == ".sv" else "verilog"

    # Generate TCL via tools/synthesis_template.tcl
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
        rtl_format=rtl_format,
        compile_section=_GOAL_PRESETS[goal]["compile_section"],
        max_cores=_resolve_dc_max_cores(task.get("max_cores")),
    )

    # Run DC via SSH
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

        # Save raw log
        log_path = out_dir / "dc_shell.log"
        log_path.write_text(output, encoding="utf-8")

        # Check for success. The TCL always prints a success banner at the end,
        # so that banner alone is not trustworthy. We additionally require:
        #   1. no Error/ERROR lines in the DC log
        #   2. at least one parsed metric from syn_output reports
        err_lines = [ln.strip() for ln in output.splitlines()
                     if "Error:" in ln or "ERROR" in ln]
        reports = parse_reports(out_dir)
        has_metrics = bool(reports)

        if "Synthesis completed successfully!" in output and not err_lines and has_metrics:
            result["success"] = True
            result["reports"] = reports
            _log("OK")
        else:
            if err_lines:
                result["error"] = err_lines[0]
            elif not has_metrics:
                result["error"] = "DC finished without parsable reports"
            else:
                result["error"] = "DC did not complete"
            result["raw_tail"] = output[-2000:]
            result["reports"] = reports
            _log(f"FAIL ({result['error'][:80]})")

    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
        _log("TIMEOUT")

    return result


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def parse_reports(out_dir: Path) -> dict:
    """Parse timing/area/power from DC report files."""
    reports = {}
    syn_out = out_dir / "syn_output"

    # Timing: data arrival time is the actual path delay reported by DC.
    # Keep slack as a secondary feasibility metric, but do not use it as the
    # primary timing value.
    timing_rpt = syn_out / "timing.rpt"
    if timing_rpt.is_file():
        text = timing_rpt.read_text(errors="replace")
        arrival_vals = [
            abs(float(x))
            for x in re.findall(r"data arrival time\s+([-\d.]+)", text)
        ]
        if arrival_vals:
            reports["data_arrival_time_ps"] = max(arrival_vals)
            reports["delay_ps"] = reports["data_arrival_time_ps"]
            reports["timing_ps"] = reports["data_arrival_time_ps"]
        m = re.search(r"slack\s*\((?:MET|VIOLATED)\)\s*([-\d.]+)", text)
        if m:
            reports["slack_ps"] = float(m.group(1))

    # Area: total cell area
    area_rpt = syn_out / "area.rpt"
    if area_rpt.is_file():
        text = area_rpt.read_text(errors="replace")
        m = re.search(r"Total cell area:\s*([\d.]+)", text)
        if m:
            reports["total_cell_area"] = float(m.group(1))

    # Power: total dynamic + leakage
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


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Run DC synthesis on collected .v files")
    ap.add_argument("--input", type=Path, default=INPUT_DIR,
                    help=f"Input directory (default: {INPUT_DIR})")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output directory (default: LLM_V2V_COLLECTED_DC_<GOAL>)")
    ap.add_argument("--metric", choices=["TIMING", "AREA"], default=None,
                    help="Only process one metric")
    ap.add_argument("--bench", type=str, default=None,
                    help="Filter benchmarks by substring")
    ap.add_argument("--transform", type=str, default=None,
                    help="Filter transforms by substring")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max tasks to process")
    ap.add_argument("-j", "--workers", type=int, default=1,
                    help="Parallel DC workers (default: 1, serial)")
    ap.add_argument("--timeout", type=int, default=DC_TIMEOUT,
                    help=f"DC timeout in seconds (default: {DC_TIMEOUT})")
    ap.add_argument("--goal", choices=["timing", "area"], default="timing",
                    help="Optimization goal: timing (tight clock) or area (relaxed clock) (default: timing)")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="Run on baseline .v files from this directory (e.g. original_AREA). "
                         "Output goes to LLM_V2V_COLLECTED_DC_<GOAL>_BASELINE")
    ap.add_argument("--llm-sdc", action="store_true",
                    help="Use LLM to generate SDC constraints (via tools/tcl_agent.py)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show tasks without running")
    args = ap.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    # Default output dir depends on goal and baseline mode
    if args.output is None:
        suffix = _GOAL_PRESETS[args.goal]["output_suffix"]
        if args.baseline:
            suffix += "_BASELINE"
        args.output = ROOT / f"LLM_V2V_COLLECTED_{suffix}"

    if args.baseline:
        tasks = discover_baseline_tasks(args.baseline, args.bench)
    else:
        tasks = discover_tasks(args.input, args.metric, args.bench, args.transform)
    if not tasks:
        print("No .v files found to synthesize.")
        return 1

    if args.limit:
        tasks = tasks[:args.limit]

    print(f"Found {len(tasks)} .v files to synthesize")
    preset = _GOAL_PRESETS[args.goal]
    print(f"Goal: {args.goal}  (clock={preset['clock_ps']}ps, compile={preset['compile_section'].splitlines()[0]})")
    print(f"Output: {args.output}")
    print(f"Workers: {args.workers}")
    if args.llm_sdc:
        print(f"SDC mode: LLM-generated (via tools/tcl_agent.py)")

    if args.dry_run:
        for t in tasks:
            print(f"  {t['metric']}/{t['subcategory']}/{t['benchmark']}/{t['stem']}")
        print(f"\n[dry-run] Would process {len(tasks)} files.")
        return 0

    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_dc_task, t, args.output, args.timeout, args.llm_sdc, args.goal): t
            for t in tasks
        }
        for fut in as_completed(futures):
            r = fut.result()
            # Parse reports if successful
            if r["success"]:
                out_dir = (args.output / r["metric"] / r["subcategory"]
                           / r["benchmark"] / r["transform"])
                r["reports"] = parse_reports(out_dir)
            results.append(r)

    _print_summary(results)

    # Write results JSON
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
