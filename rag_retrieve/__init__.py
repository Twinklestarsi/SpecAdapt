"""
rag_retrieve - Module 4 data access helpers.

This package starts with the CDFG corpus/index layer so that existing
CDFG datasets can be joined with the current RAG CSV knowledge base.
"""

from rag_retrieve.cdfg_index import (
    build_default_joined_index,
    build_joined_index,
    write_joined_index,
)
from rag_retrieve.matcher import match_query_to_index
from rag_retrieve.defaults import (
    ACTIVE_RAG,
    RAGArtifacts,
    get_active_rag_artifacts,
    versioned_rag_artifacts,
)

__all__ = [
    "build_default_joined_index",
    "build_joined_index",
    "ACTIVE_RAG",
    "RAGArtifacts",
    "get_active_rag_artifacts",
    "match_query_to_index",
    "versioned_rag_artifacts",
    "write_joined_index",
]
