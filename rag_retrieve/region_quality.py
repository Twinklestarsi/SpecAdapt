"""
Quality scoring for region proposals in region_extract_v2.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Set

from rag_retrieve.region_features import normalize_opcode

_TYPE_OPS = {
    "arith_region": {"add", "sub", "mul", "shl", "lshr", "ashr", "trunc", "zext", "sext"},
    "select_region": {"phi", "br", "icmp", "select"},
    "state_region": {"load", "store", "phi", "add", "sub", "and", "or", "xor"},
    "bit_region": {"and", "or", "xor", "not", "shl", "lshr", "ashr"},
    "memory_region": {"load", "store", "getelementptr"},
    "reduction_region": {"add", "and", "or", "xor", "icmp", "mul"},
}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _motif_purity(node_ids: Set[str], proposal_type: str, labels: Dict[str, str]) -> float:
    if not node_ids:
        return 0.0
    allowed = _TYPE_OPS.get(proposal_type, set())
    opcodes = [normalize_opcode(labels.get(node_id, "")) for node_id in node_ids]
    counts = Counter(opcodes)
    if not allowed:
        return 0.4
    motif_hits = sum(count for opcode, count in counts.items() if opcode in allowed)
    return _safe_div(motif_hits, len(opcodes))


def _boundary_clarity(node_ids: Set[str], succs: Dict[str, Set[str]], preds: Dict[str, Set[str]]) -> float:
    internal_edges = 0
    crossing_edges = 0
    for node_id in node_ids:
        for dst in succs.get(node_id, set()):
            if dst in node_ids:
                internal_edges += 1
            else:
                crossing_edges += 1
        for src in preds.get(node_id, set()):
            if src not in node_ids:
                crossing_edges += 1
    return _safe_div(internal_edges, internal_edges + crossing_edges + 1)


def _internal_density(node_ids: Set[str], succs: Dict[str, Set[str]]) -> float:
    size = len(node_ids)
    if size <= 1:
        return 0.0
    internal_edges = 0
    for node_id in node_ids:
        internal_edges += sum(1 for dst in succs.get(node_id, set()) if dst in node_ids)
    max_edges = size * max(size - 1, 1)
    return min(1.0, _safe_div(internal_edges, max_edges) * 2.5)


def _size_reasonableness(node_count: int) -> float:
    if node_count < 4:
        return 0.15
    if node_count <= 20:
        return 1.0
    if node_count <= 40:
        return max(0.3, 1.0 - (node_count - 20) / 30.0)
    if node_count <= 60:
        return max(0.15, 0.35 - (node_count - 40) / 40.0)
    return 0.05


def score_region_proposal(proposal: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, float]:
    node_ids = set(proposal.get("node_ids", []))
    labels = graph["labels"]
    succs = graph["succs"]
    preds = graph["preds"]
    proposal_type = proposal.get("proposal_type", "hybrid_region")
    anchor_confidence = float(proposal.get("confidence", 0.0))

    motif_purity = _motif_purity(node_ids, proposal_type, labels)
    boundary_clarity = _boundary_clarity(node_ids, succs, preds)
    internal_density = _internal_density(node_ids, succs)
    size_reasonableness = _size_reasonableness(len(node_ids))

    quality = (
        0.30 * motif_purity
        + 0.20 * boundary_clarity
        + 0.20 * internal_density
        + 0.15 * anchor_confidence
        + 0.15 * size_reasonableness
    )
    return {
        "quality_score": round(quality, 6),
        "motif_purity": round(motif_purity, 6),
        "boundary_clarity": round(boundary_clarity, 6),
        "internal_density": round(internal_density, 6),
        "size_reasonableness": round(size_reasonableness, 6),
        "anchor_confidence": round(anchor_confidence, 6),
    }
