#!/usr/bin/env python3
"""Run the external JG/DC stages on completed Veri-Sure CSV candidates.

This is intentionally separate from generation so a Python environment import
failure in the post-generation tools cannot invalidate a completed full-flow
Veri-Sure run or its token ledger.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_one(path: Path, *, force: bool, verification_timeout: Optional[int]) -> dict[str, Any]:
    result = read_json(path)
    candidate_raw = result.get("candidate_path")
    candidate = Path(str(candidate_raw)).resolve() if candidate_raw else None
    if candidate is None or not candidate.is_file():
        return result
    eq = result.get("equivalence") or {}
    syn = result.get("syntax") or {}
    if not force and str(eq.get("status", "")).lower() in {"passed", "pass", "success"} and (result.get("synthesis") or {}).get("metrics"):
        return result
    case_dir = path.parent
    top = str(result.get("top_module") or "TopModule")
    golden = Path(str(result["golden_path"])).resolve()
    if str(syn.get("status")) == "passed":
        try:
            from module5.jg_verifier import verify_rtl_equivalence_with_jg

            result["equivalence"] = verify_rtl_equivalence_with_jg(
                candidate,
                golden,
                case_dir / "verification",
                ROOT / ".env",
                golden_top=top,
                design_type=str(result.get("design_type") or ""),
                verification_timeout=verification_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            result["equivalence"] = {"status": "error", "equivalent": False, "error": f"JG invocation: {type(exc).__name__}: {exc}"}
        try:
            from module5.dc_runner import run_dc_for_verilog

            result["synthesis"] = run_dc_for_verilog(
                candidate,
                benchmark=str(result.get("source_case_id") or result.get("case_id")),
                goal=str(result.get("objective") or "AREA"),
                stem=str(result.get("case_id") or case_dir.name),
                output_root=case_dir / "dc",
                top_module=top,
                max_cores=1,
            )
        except Exception as exc:  # noqa: BLE001
            result["synthesis"] = {"status": "error", "error": f"DC invocation: {type(exc).__name__}: {exc}"}
        metrics = result["synthesis"].get("metrics") if isinstance(result.get("synthesis"), dict) else {}
        metrics = metrics if isinstance(metrics, dict) else {}
        eq_now = result.get("equivalence") or {}
        eq_pass = str(eq_now.get("status", "")).lower() in {"passed", "pass", "success"} and bool(eq_now.get("equivalent", False))
        result["ppa"] = {
            "area": metrics.get("total_cell_area"),
            "timing_ps": metrics.get("delay_ps", metrics.get("data_arrival_time_ps")),
            "delay_ps": metrics.get("delay_ps", metrics.get("data_arrival_time_ps")),
            "slack_ps": metrics.get("slack_ps"),
            "verification_status": eq_now.get("status"),
            "verified_equivalent": eq_pass,
        }
        if result["ppa"]["area"] is not None or result["ppa"]["timing_ps"] is not None:
            result["terminal_status"] = "ppa_complete_verified" if eq_pass else "ppa_complete_unverified"
        else:
            result["terminal_status"] = "candidate_syntax_passed_verified" if eq_pass else "candidate_syntax_passed"
        result["postprocessed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(path, result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--verification-timeout", type=int, default=600)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_root = args.run_root.expanduser().resolve()
    paths = sorted(run_root.glob("rows/*/case_result.json"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.max_workers, len(paths) or 1))) as pool:
        futures = {pool.submit(run_one, path, force=args.force, verification_timeout=args.verification_timeout): path for path in paths}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                print(f"{result.get('csv_row')} {result.get('source_case_id')} -> {result.get('terminal_status')} JG={(result.get('equivalence') or {}).get('status')} tokens={result.get('total_tokens')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"{path}: {type(exc).__name__}: {exc}", flush=True)
    # Reuse the canonical report writer after updating case_result.json files.
    manifest_path = run_root / "input_manifest.json"
    config_path = run_root / "verisure_config.json"
    if manifest_path.is_file():
        from run_verisure_csv_full import write_reports

        manifest = read_json(manifest_path)
        cases = manifest.get("cases", [])
        config = read_json(config_path) if config_path.is_file() else {}
        input_csv = Path(str(manifest.get("input_csv")))
        write_reports(run_root, cases, input_csv, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
