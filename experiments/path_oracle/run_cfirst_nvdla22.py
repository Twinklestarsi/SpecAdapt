#!/usr/bin/env python3
"""Run only the forced C-first route for all nvdla22 cases.

This deliberately uses a fresh per-case Memory snapshot and output tree.  The
existing paired pilot entry runs both routes; this entry is for a clean,
C-first-only rerun under the nvdla22 no-JasperGold PPA convention.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from experiments.path_oracle.run_dual_path import _llm_seed
from pipeline.orchestrator import PipelineOrchestrator


ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = ROOT / "runs/phase6/nvdla22_njg_20260908/pilot"
DEFAULT_OUTPUT = ROOT / "runs/external_spec_agents/20260915_cfirst_nvdla22_no_jg"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_") or "case"


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def copy_memory(case_root: Path) -> tuple[Path, Path]:
    memory_root = case_root / "memory"
    memory_root.mkdir(parents=True, exist_ok=True)
    source_db = ROOT / "memory_agent.db"
    source_json = ROOT / "path_decisions_log.json"
    db_path = memory_root / "memory.db"
    json_path = memory_root / "memory.json"
    if source_db.is_file():
        shutil.copy2(source_db, db_path)
    if source_json.is_file():
        shutil.copy2(source_json, json_path)
    return json_path, db_path


def run_case(
    case: dict[str, Any],
    *,
    objective: str,
    output_root: Path,
    force: bool,
    index: int,
) -> dict[str, Any]:
    case_id = str(case["id"])
    case_root = output_root / objective / safe(case_id)
    result_path = case_root / "result.json"
    if result_path.is_file() and not force:
        return read_json(result_path)
    case_root.mkdir(parents=True, exist_ok=True)

    spec_path = (PILOT_ROOT / str(case["spec_path"])).resolve()
    golden_path = (PILOT_ROOT / str(case["golden_rtl_path"])).resolve()
    spec_text = spec_path.read_text(encoding="utf-8")
    memory_json, memory_db = copy_memory(case_root)
    pipeline_root = case_root / "pipeline_runs"
    c_output = case_root / "c_gen_output"
    module5_root = case_root / "module5"
    started = time.time()
    result: dict[str, Any] = {
        "schema_version": "nvdla22_cfirst_only_result_v1",
        "case_id": case_id,
        "objective": objective,
        "route": "c_first",
        "verification_mode": "none",
        "model": os.environ.get("OPENAI_MODEL", ""),
        "spec_path": str(spec_path),
        "golden_rtl_path": str(golden_path),
    }
    try:
        orchestrator = PipelineOrchestrator(
            memory_store_path=memory_json,
            memory_db_path=memory_db,
            output_root=pipeline_root,
            c_output_dir=c_output,
        )
        with _llm_seed(7000 + index):
            payload = orchestrator.run(
                benchmark=case_id,
                objective=objective,
                spec_text=spec_text,
                use_llm_path_selection=False,
                module4_top_k=5,
                mcts_iterations=240,
                mcts_max_depth=5,
                mcts_seed=7 + index,
                mcts_candidate_limit=48,
                module5_backend="direct_rtl",
                execute_best_path=True,
                max_actions=None,
                run_baseline=False,
                module5_output_root=module5_root,
                rtl_max_retries=2,
                dc_max_retries=0,
                experiment_group="optimized",
                forced_path="c_first",
                pre_dc_golden_rtl_path=golden_path,
                pre_dc_golden_top=str(case.get("golden_top_module") or case.get("top_module") or ""),
                pre_dc_design_type=str(case.get("design_type") or "combinational"),
                verification_mode="none",
                jg_max_retries=0,
            )
        payload_dict = payload.to_dict()
        execution = payload_dict.get("module5_result") or {}
        metrics = execution.get("metrics") or {}
        result.update(
            {
                "status": payload_dict.get("status", ""),
                "task_id": payload_dict.get("task_id", ""),
                "run_id": payload_dict.get("run_id", ""),
                "generated_c_path": (payload_dict.get("c_generation") or {}).get("c_path", ""),
                "generated_rtl_path": execution.get("generated_verilog_path", ""),
                "syntax_status": execution.get("syntax_status", ""),
                "dc_status": execution.get("dc_status", ""),
                "area": metrics.get("total_cell_area"),
                "timing_ps": metrics.get("data_arrival_time_ps", metrics.get("delay_ps")),
                "slack_ps": metrics.get("slack_ps"),
                "dynamic_power": metrics.get("dynamic_power"),
                "leakage_power": metrics.get("leakage_power"),
                "pipeline_result_path": str(pipeline_root / "summary.json"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "status": "pipeline_exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    result["elapsed_seconds"] = round(time.time() - started, 3)
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def write_summary(output_root: Path, objectives: list[str], cases: list[dict[str, Any]]) -> None:
    lines = [
        "# nvdla22 C-first-only rerun",
        "",
        "本目录只包含强制 `c_first` 路线；验证模式为 `none`，即语法检查后直接运行 DC，未调用 JasperGold。",
        "",
        "| Objective | Case | Status | Area | Timing (ps) | DC |",
        "|---|---|---|---:|---:|---|",
    ]
    all_results: list[dict[str, Any]] = []
    for objective in objectives:
        for case in cases:
            path = output_root / objective / safe(str(case["id"])) / "result.json"
            if not path.is_file():
                continue
            row = read_json(path)
            all_results.append(row)
            lines.append(
                f"| {objective} | {row.get('case_id')} | {row.get('status', '')} | "
                f"{fmt(row.get('area'))} | {fmt(row.get('timing_ps'))} | {row.get('dc_status', '')} |"
            )
    summary = {
        "schema_version": "nvdla22_cfirst_only_summary_v1",
        "case_count": len(cases),
        "objectives": objectives,
        "route": "c_first",
        "verification_mode": "none",
        "model": os.environ.get("OPENAI_MODEL", ""),
        "results": all_results,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--objectives", nargs="+", choices=("area", "timing"), default=["area", "timing"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = read_json(PILOT_ROOT / "pilot_manifest.json")
    cases = list(manifest.get("cases") or [])
    if len(cases) != int(manifest.get("case_count", len(cases))):
        raise RuntimeError("pilot manifest case_count mismatch")
    args.output_root.mkdir(parents=True, exist_ok=True)
    total = len(cases) * len(args.objectives)
    current = 0
    for objective in args.objectives:
        for index, case in enumerate(cases):
            current += 1
            print(f"[c_first {current}/{total}] {objective} {case['id']}", flush=True)
            row = run_case(
                case,
                objective=objective,
                output_root=args.output_root,
                force=args.force,
                index=index,
            )
            print(
                json.dumps(
                    {
                        "objective": objective,
                        "case": case["id"],
                        "status": row.get("status"),
                        "area": row.get("area"),
                        "timing_ps": row.get("timing_ps"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        write_summary(args.output_root, list(args.objectives), cases)
    write_summary(args.output_root, list(args.objectives), cases)
    print(f"summary: {args.output_root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
