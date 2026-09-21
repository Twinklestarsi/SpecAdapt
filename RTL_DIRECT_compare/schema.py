"""
schema.py — Result dataclass for the direct RTL comparison experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from token_counter import TokenUsage


@dataclass
class RTLDirectResult:
    """Result of generating RTL directly from a spec."""
    benchmark: str
    module_name: str
    rtl_path: str
    method: str
    success: bool
    model: str = ""
    syntax_ok: bool = False
    error: str = ""
    attempt_count: int = 0
    attempt_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: TokenUsage = None

    def __post_init__(self) -> None:
        if self.token_usage is None:
            self.token_usage = TokenUsage()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["token_usage"] = self.token_usage.to_dict()
        return data
