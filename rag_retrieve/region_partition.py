"""
Proposal de-duplication and trimming for region_extract_v2.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Set

from rag_retrieve.region_patterns import proposal_type_priority


def proposal_iou(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _sort_key(proposal: Dict[str, Any]) -> tuple[float, int, int, str, tuple[str, ...]]:
    return (
        float(proposal.get("quality_score", 0.0)),
        int(proposal.get("semantic_priority", proposal_type_priority(proposal.get("proposal_type", "")))),
        len(proposal.get("node_ids", [])),
        str(proposal.get("proposal_type", "")),
        tuple(sorted(proposal.get("node_ids", []))),
    )


def _trim_nodes(candidate: Dict[str, Any], occupied: Set[str], min_size: int) -> Dict[str, Any] | None:
    trimmed = deepcopy(candidate)
    trimmed_nodes = set(trimmed.get("node_ids", set())) - occupied
    trimmed["node_ids"] = trimmed_nodes
    trimmed["anchor_node_ids"] = [node_id for node_id in trimmed.get("anchor_node_ids", []) if node_id in trimmed_nodes]
    trimmed["anchor_labels"] = trimmed.get("anchor_labels", [])[: len(trimmed["anchor_node_ids"])]
    trimmed["overlap_suppressed_count"] = int(trimmed.get("overlap_suppressed_count", 0)) + 1
    if len(trimmed_nodes) < min_size:
        return None
    return trimmed


def select_final_proposals(
    proposals: List[Dict[str, Any]],
    min_size: int = 4,
    same_type_iou: float = 0.70,
    diff_type_iou: float = 0.45,
) -> List[Dict[str, Any]]:
    accepted: List[Dict[str, Any]] = []

    ranked = sorted(proposals, key=_sort_key, reverse=True)
    for proposal in ranked:
        current = deepcopy(proposal)
        current.setdefault("overlap_suppressed_count", 0)
        if len(current.get("node_ids", [])) < min_size:
            continue

        suppressed = False
        for accepted_item in accepted:
            iou = proposal_iou(set(current["node_ids"]), set(accepted_item["node_ids"]))
            if iou <= 0.0:
                continue
            if current["proposal_type"] == accepted_item["proposal_type"] and iou >= same_type_iou:
                suppressed = True
                break
            # Different semantic region types may legitimately overlap.  Keep
            # both intact instead of deleting nodes from whichever was ranked
            # later.  ``diff_type_iou`` remains in the signature for caller
            # compatibility and future diagnostics.

        if suppressed:
            continue
        accepted.append(current)

    accepted.sort(key=lambda item: (-float(item.get("quality_score", 0.0)), item["proposal_type"]))
    return accepted
