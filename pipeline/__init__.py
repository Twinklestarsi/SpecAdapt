"""Unified in-process optimization pipeline with lazy public imports."""

from typing import Any

__all__ = ["PipelineOrchestrator", "PipelineResult"]


def __getattr__(name: str) -> Any:
    # Keeping this import lazy lets low-level helpers such as
    # ``pipeline.equivalence`` be used by Module 5 without importing the full
    # orchestrator back into a partially initialized Module 5 package.
    if name in __all__:
        from pipeline.orchestrator import PipelineOrchestrator, PipelineResult

        return {
            "PipelineOrchestrator": PipelineOrchestrator,
            "PipelineResult": PipelineResult,
        }[name]
    raise AttributeError(name)
