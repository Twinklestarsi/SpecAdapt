"""
schema.py — CGenResult dataclass for Module 3 output.

Tracks the result of C generation for one benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

from token_counter import TokenUsage


@dataclass
class CGenResult:
    """Output of Module 3 C generation for one benchmark."""
    benchmark: str
    c_path: str              # absolute path to generated .c file
    method: str              # "llm_verilog" | "llm_spec" | "llm_mixed" | "v2c_fallback"
    success: bool
    error: str = ""          # empty if success=True
    token_usage: TokenUsage = None
    semantic_review_status: str = "not_run"
    pre_review_c_path: str = ""

    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = TokenUsage()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["token_usage"] = self.token_usage.to_dict()
        return d
