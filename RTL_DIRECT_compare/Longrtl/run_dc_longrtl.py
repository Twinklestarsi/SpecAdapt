#!/usr/bin/env python3
"""Run Design Compiler on a single LongRTL integration.v file.

Usage (standalone):
    python3 run_dc_longrtl.py \
        --verilog /path/to/integration.v \
        --benchmark NV_NVDLA_apb2csb \
        --goal area \
        --output-root /path/to/dc_runs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from module5.dc_runner import run_dc_for_verilog

parser = argparse.ArgumentParser(description="DC evaluation for a LongRTL integration.v")
parser.add_argument("--verilog",      required=True, help="Path to integration.v")
parser.add_argument("--benchmark",    required=True, help="Benchmark name (e.g. NV_NVDLA_apb2csb)")
parser.add_argument("--goal",         default="area", choices=["area", "timing"],
                    help="Synthesis goal")
parser.add_argument("--output-root",  required=True, help="Root directory for DC run artifacts")
args = parser.parse_args()

result = run_dc_for_verilog(
    args.verilog,
    benchmark=args.benchmark,
    goal=args.goal,
    stem="longrtl_candidate",
    output_root=args.output_root,
)

print(json.dumps(result, indent=2, default=str))
sys.exit(0 if result.get("success") else 1)
