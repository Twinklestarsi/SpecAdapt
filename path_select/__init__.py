"""
path_select — Stage 2: Path Selection

Takes Stage 1 feature vectors (FeatureResult) and decides the optimization
path for each benchmark:
  - "rtl_direct": optimize Verilog directly via LLM (llm_v2v.py)
  - "c_first":    convert to C, apply C-level transforms, re-synthesize via HLS

Two-tier decision:
  Tier 1: Deterministic rule-based (fast, no LLM needed)
  Tier 2: LLM-assisted for ambiguous cases (optional)

Usage:
    from path_select import PathSelector

    selector = PathSelector()
    decision = selector.select(feature_result)          # single benchmark
    decisions = selector.select_batch(results_list)     # batch
    decisions = selector.select_from_json("spec_analysis.json")
"""

from path_select.selector import PathSelector, PathDecision

__all__ = ["PathSelector", "PathDecision"]
