#!/usr/bin/env python3
"""Materialize the verified C-first winner set for external agent runs.

The source manifests and RTL files remain read-only.  This script writes one
small JSON description per case; adapters consume those descriptions and keep
the golden RTL out of the generation prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / "runs" / "external_spec_agents" / "20260912" / "cases"

AREA_IDS = {
    "ca03_square_bias_engine", "ca07_paired_scale_engine",
    "ca08_offset_product_engine", "ca09_triple_term_engine",
    "ca11_scaled_bin_table", "ca15_masked_scale_table",
    "ca20_capped_bin_table", "ca21_weighted_reduce_bank",
    "ca25_neighbor_energy_bank", "ca26_square_extrema_bank",
    "ca27_weight_span_bank", "ca29_gain_reduce_bank",
    "ca30_self_energy_bank", "ca31_session_mac_accountant",
    "ca34_session_nibble_accountant", "ca35_session_square_accountant",
    "ca37_session_blend_accountant",
}
TIMING_IDS = {
    "ct05_peak_and_scan", "ct07_peak_addw_scan", "ct08_valley_addw_scan",
    "ct25_nibble_ring_checksum15", "ct30_rate_split_prime_mid",
}
CLEAN_IDS = {
    "cf05_threshold_clip_engine", "cf06_rounded_average_modes",
    "cf15_adjacent_distance", "cf16_saturating_vector_sum",
    "cf17_even_odd_partition", "cf29_recent_duplicate_filter4",
}
CVDP_IDS = {
    "cvdp_agentic_universal_shift_reg_0003": "area",
    "cvdp_copilot_fibonacci_series_0001": "timing",
    "cvdp_copilot_kogge_stone_adder_0007": "timing",
    "cvdp_copilot_set_bit_calculator_0001": "timing",
    "cvdp_copilot_single_number_0001": "timing",
}

SUMMARY_FILES = {
    "area": ROOT / "runs/phase6/cfirst_area40_deepseek_official_ca01_ca40_jg1000s_2026-09-04/results/phase6_summary.json",
    "timing": ROOT / "runs/phase6/cfirst_timing40_tjeda_ds_ct01_ct40_2026-09-09-03/results/phase6_summary.json",
    "clean": ROOT / "runs/phase6/cfirst40_clean_deepseek_official_2026-09-02/results/phase6_summary.json",
}
# The timing directory was renamed with the date in the actual checkout; keep
# the correct path in a separate constant to make a typo fail loudly.
SUMMARY_FILES["timing"] = ROOT / "runs/phase6/cfirst_timing40_tjeda_ds_ct01_ct40_2026-09-03/results/phase6_summary.json"
CVDP_STUB_TB = ROOT / "experiments/external_spec_agents/cvdp_stub_tb.sv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_summary(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(c["benchmark"]): c for c in data["cases"]}


def self_authored_case(case_id: str, objective: str, dataset: str, summary: dict[str, Any], comparison: str) -> dict[str, Any]:
    case_dir = ROOT / "datasets" / dataset / "cases" / case_id
    spec = case_dir / "spec.txt"
    golden = case_dir / "golden.v"
    tb = case_dir / "tb.sv"
    for p in (spec, golden, tb):
        if not p.is_file():
            raise FileNotFoundError(p)
    croute = summary["routes"]["c_first"]
    return {
        "id": case_id,
        "benchmark": case_id,
        "objective": objective,
        "design_type": "sequential",
        "top_module": "TopModule",
        "spec_path": str(spec),
        "golden_path": str(golden),
        "testbench_path": str(tb),
        "source_manifest": str(ROOT / "datasets" / dataset / "pilot_manifest.json"),
        "source_metrics": {
            "area": croute.get("median_area"),
            "delay_ps": croute.get("median_delay_ps"),
            "timing_ps": croute.get("median_delay_ps"),
            "slack_ps": None,
        },
        "comparison_class": comparison,
        "source_sha256": {"spec": sha256(spec), "golden": sha256(golden), "testbench": sha256(tb)},
    }


def cvdp_case(case_id: str, objective: str) -> dict[str, Any]:
    pilot = ROOT / "datasets/candidates/cvdp_expand_20260906/pilots" / f"{case_id}__{objective}"
    manifest = pilot / "pilot_manifest.json"
    row = json.loads(manifest.read_text(encoding="utf-8"))["cases"][0]
    spec = pilot / row["spec_path"]
    golden = pilot / row["golden_rtl_path"]
    # The official context testbench is used only as a public evaluator input;
    # private CVDP scorer files are never copied to an adapter prompt.
    materialized = ROOT / "datasets/candidates/cvdp_expand_20260906/materialized/cases" / case_id
    candidates = sorted(materialized.glob("upstream/context/verif/tb_*.sv"))
    tb = candidates[0] if candidates else CVDP_STUB_TB
    result: dict[str, Any] = {
        "id": case_id,
        "benchmark": case_id,
        "objective": objective,
        "design_type": row.get("design_type", "sequential"),
        "top_module": row.get("top_module") or row.get("golden_top_module"),
        "spec_path": str(spec),
        "golden_path": str(golden),
        "source_manifest": str(manifest),
        "source_metrics": {},
        "comparison_class": "paired_ppa_win",
        "source_sha256": {"spec": sha256(spec), "golden": sha256(golden)},
    }
    result["testbench_path"] = str(tb)
    result["source_sha256"]["testbench"] = sha256(tb)
    if not candidates:
        result["testbench_origin"] = "generated_compile_smoke"
    return result


def iter_cases() -> Iterable[dict[str, Any]]:
    summaries = {name: load_summary(path) for name, path in SUMMARY_FILES.items()}
    for cid in sorted(AREA_IDS):
        comparison = "paired_ppa_win" if cid in {"ca03_square_bias_engine", "ca07_paired_scale_engine", "ca08_offset_product_engine", "ca09_triple_term_engine", "ca30_self_energy_bank", "ca31_session_mac_accountant", "ca34_session_nibble_accountant", "ca35_session_square_accountant", "ca37_session_blend_accountant"} else "cfirst_only_valid"
        yield self_authored_case(cid, "area", "cfirst_area40_2026-08-25", summaries["area"][cid], comparison)
    for cid in sorted(TIMING_IDS):
        comparison = "paired_ppa_win" if cid != "ct30_rate_split_prime_mid" else "cfirst_only_valid"
        yield self_authored_case(cid, "timing", "cfirst_timing40_2026-09-02", summaries["timing"][cid], comparison)
    for cid in sorted(CLEAN_IDS):
        yield self_authored_case(cid, "area", "cfirst40_clean_2026-09-02", summaries["clean"][cid], "paired_ppa_win")
    for cid, objective in sorted(CVDP_IDS.items()):
        yield cvdp_case(cid, objective)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = list(iter_cases())
    for case in cases:
        path = args.output / f"{case['id']}__{case['objective']}.json"
        path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index = {
        "schema_version": "external_spec_agent_case_set_v1",
        "created_at": "2026-09-12",
        "case_count": len(cases),
        "paired_count": sum(c["comparison_class"] == "paired_ppa_win" for c in cases),
        "cfirst_only_count": sum(c["comparison_class"] == "cfirst_only_valid" for c in cases),
        "cases": [f"{c['id']}__{c['objective']}.json" for c in cases],
    }
    (args.output / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(index, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
