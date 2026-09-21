"""
usage.py — TokenUsage dataclass.

Tracks LLM token consumption for one module invocation on one benchmark.
All modules in the pipeline import from here; token counting logic
(reading response.usage) is added per-module in later steps.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class TokenUsage:
    """LLM token usage for one module call on one benchmark."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Combine two TokenUsage records (e.g. sum across retries)."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            retries=self.retries + other.retries,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.retries += other.retries
        self.total_tokens += other.total_tokens
        return self
