"""
token_counter — Shared token usage tracking for all pipeline modules.

Each module imports TokenUsage and attaches a filled instance to its
output object. Actual counting (reading response.usage) is done
per-module; this package only defines the shared dataclass.

Usage:
    from token_counter import TokenUsage

    usage = TokenUsage()
    # ... after LLM call:
    # usage.prompt_tokens     = response.usage.prompt_tokens
    # usage.completion_tokens = response.usage.completion_tokens
    # usage.total_tokens      = response.usage.total_tokens
"""

from token_counter.usage import TokenUsage

__all__ = ["TokenUsage"]
