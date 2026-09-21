"""
Pattern-driven region proposal helpers for region_extract_v2.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Set, Tuple

from rag_retrieve.region_features import normalize_opcode

_ARITH_OPS = {"add", "sub", "mul", "shl", "lshr", "ashr", "trunc", "zext", "sext"}
_SELECT_OPS = {"phi", "br", "icmp", "select"}
_BIT_OPS = {"and", "or", "xor", "not", "shl", "lshr", "ashr"}
_MEMORY_OPS = {"load", "store", "getelementptr"}
_CAST_OPS = {"trunc", "zext", "sext", "bitcast"}
_CONTROL_STOP_OPS = {"br", "ret"}
_PROPOSAL_TYPE_PRIORITY = {
    "select_region": 6,
    "state_region": 6,
    "reduction_region": 5,
    "arith_region": 4,
    "bit_region": 4,
    "memory_region": 3,
    "hybrid_region": 1,
}


def build_graph_view(
    labels: Dict[str, str],
    succs: Dict[str, Set[str]],
    preds: Dict[str, Set[str]],
    edge_types: Dict[Tuple[str, str], str],
) -> Dict[str, Any]:
    opcodes = {node_id: normalize_opcode(label) for node_id, label in labels.items()}
    reverse_depth = _measure_reverse_depth(labels.keys(), preds)
    forward_depth = _measure_reverse_depth(labels.keys(), succs)

    return {
        "labels": labels,
        "succs": succs,
        "preds": preds,
        "edge_types": edge_types,
        "opcodes": opcodes,
        "reverse_depth": reverse_depth,
        "forward_depth": forward_depth,
        "fanin": {node_id: len(preds.get(node_id, set())) for node_id in labels},
        "fanout": {node_id: len(succs.get(node_id, set())) for node_id in labels},
    }


def _measure_reverse_depth(nodes: Iterable[str], adjacency: Dict[str, Set[str]]) -> Dict[str, int]:
    cache: Dict[str, int] = {}

    def _depth(node_id: str, seen: Set[str]) -> int:
        if node_id in cache:
            return cache[node_id]
        if node_id in seen:
            return 1
        next_nodes = adjacency.get(node_id, set())
        if not next_nodes:
            cache[node_id] = 1
            return 1
        value = 1 + max(_depth(child, seen | {node_id}) for child in sorted(next_nodes))
        cache[node_id] = value
        return value

    return {node_id: _depth(node_id, set()) for node_id in sorted(nodes)}


def proposal_type_priority(proposal_type: str) -> int:
    return _PROPOSAL_TYPE_PRIORITY.get(proposal_type, 0)


def _internal_neighbors(node_id: str, graph: Dict[str, Any], allowed_edge_types: Set[str] | None = None) -> List[str]:
    neighbors = set()
    succs = graph["succs"]
    preds = graph["preds"]
    edge_types = graph["edge_types"]

    for dst in succs.get(node_id, set()):
        edge_type = edge_types.get((node_id, dst), "data")
        if allowed_edge_types and edge_type not in allowed_edge_types:
            continue
        neighbors.add(dst)
    for src in preds.get(node_id, set()):
        edge_type = edge_types.get((src, node_id), "data")
        if allowed_edge_types and edge_type not in allowed_edge_types:
            continue
        neighbors.add(src)
    return sorted(neighbors)


def _bounded_expand(
    seed_nodes: Iterable[str],
    graph: Dict[str, Any],
    max_depth: int,
    allowed_ops: Set[str] | None = None,
    allowed_edge_types: Set[str] | None = None,
    stop_ops: Set[str] | None = None,
) -> Set[str]:
    labels = graph["labels"]
    opcodes = graph["opcodes"]
    ordered_seeds = sorted(set(seed_nodes))
    seed_set = set(ordered_seeds)
    visited: Set[str] = set(ordered_seeds)
    queue = deque((node_id, 0) for node_id in ordered_seeds)
    while queue:
        node_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in _internal_neighbors(node_id, graph, allowed_edge_types=allowed_edge_types):
            if neighbor in visited:
                continue
            opcode = opcodes.get(neighbor, "unknown")
            if stop_ops and opcode in stop_ops and neighbor not in seed_set:
                continue
            if allowed_ops and opcode not in allowed_ops:
                continue
            visited.add(neighbor)
            queue.append((neighbor, depth + 1))
    return {node_id for node_id in visited if node_id in labels}


def _proposal(
    proposal_type: str,
    seed_nodes: Iterable[str],
    node_ids: Iterable[str],
    graph: Dict[str, Any],
    confidence: float,
    boundary_reason: str,
) -> Dict[str, Any]:
    labels = graph["labels"]
    seed_nodes = sorted(node_id for node_id in seed_nodes if node_id in labels)
    node_ids = {node_id for node_id in node_ids if node_id in labels}
    anchor_labels = [labels[node_id] for node_id in seed_nodes]
    return {
        "proposal_type": proposal_type,
        "anchor_node_ids": seed_nodes,
        "anchor_labels": anchor_labels,
        "node_ids": set(node_ids),
        "confidence": confidence,
        "boundary_reason": boundary_reason,
        "semantic_priority": proposal_type_priority(proposal_type),
    }


def detect_arith_chains(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    opcodes = graph["opcodes"]
    candidates = [node_id for node_id, opcode in opcodes.items() if opcode in _ARITH_OPS]
    proposals = []
    seen: Set[frozenset[str]] = set()

    for node_id in candidates:
        seed_nodes = {node_id}
        local = _bounded_expand(
            seed_nodes=seed_nodes,
            graph=graph,
            max_depth=4,
            allowed_ops=_ARITH_OPS | _CAST_OPS | {"arg", "const", "load"},
            allowed_edge_types={"data", "memory"},
            stop_ops=_SELECT_OPS | {"store"},
        )
        arith_nodes = {item for item in local if opcodes.get(item) in _ARITH_OPS}
        if len(arith_nodes) < 2:
            continue
        key = frozenset(local)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(
            _proposal(
                proposal_type="arith_region",
                seed_nodes=sorted(arith_nodes, key=lambda item: (-graph["fanout"].get(item, 0), item))[:2],
                node_ids=local,
                graph=graph,
                confidence=min(1.0, 0.45 + 0.1 * len(arith_nodes)),
                boundary_reason="dataflow_arith_chain",
            )
        )
    return proposals


def detect_select_structures(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    opcodes = graph["opcodes"]
    select_nodes = [node_id for node_id, opcode in opcodes.items() if opcode in _SELECT_OPS]
    proposals = []
    seen: Set[frozenset[str]] = set()

    for node_id in select_nodes:
        local = _bounded_expand(
            seed_nodes={node_id},
            graph=graph,
            max_depth=3,
            allowed_ops=_SELECT_OPS | _ARITH_OPS | _CAST_OPS | {"arg", "const", "load"},
            allowed_edge_types={"data", "control"},
            stop_ops={"store", "call", "ret"},
        )
        control_nodes = {item for item in local if opcodes.get(item) in _SELECT_OPS}
        if len(control_nodes) < 2 and opcodes.get(node_id) not in {"phi", "select"}:
            continue
        key = frozenset(local)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(
            _proposal(
                proposal_type="select_region",
                seed_nodes=sorted(control_nodes)[:3] or [node_id],
                node_ids=local,
                graph=graph,
                confidence=min(1.0, 0.5 + 0.08 * len(control_nodes)),
                boundary_reason="control_select_structure",
            )
        )
    return proposals


def detect_state_update_regions(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    opcodes = graph["opcodes"]
    load_nodes = [node_id for node_id, opcode in opcodes.items() if opcode == "load"]
    store_nodes = [node_id for node_id, opcode in opcodes.items() if opcode == "store"]
    proposals = []

    for store_id in store_nodes:
        neighborhood = _bounded_expand(
            seed_nodes={store_id},
            graph=graph,
            max_depth=3,
            allowed_ops=_ARITH_OPS | _BIT_OPS | _CAST_OPS | _MEMORY_OPS | _SELECT_OPS | {"arg", "const"},
            allowed_edge_types={"data", "control", "memory"},
            stop_ops={"call", "ret"},
        )
        if store_id not in neighborhood:
            neighborhood.add(store_id)
        load_overlap = [node_id for node_id in neighborhood if node_id in load_nodes]
        if not load_overlap:
            continue
        proposals.append(
            _proposal(
                proposal_type="state_region",
                seed_nodes=load_overlap[:2] + [store_id],
                node_ids=neighborhood,
                graph=graph,
                confidence=min(1.0, 0.55 + 0.08 * len(load_overlap)),
                boundary_reason="load_compute_store_path",
            )
        )
    return proposals


def detect_bit_clusters(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    opcodes = graph["opcodes"]
    bit_nodes = [node_id for node_id, opcode in opcodes.items() if opcode in _BIT_OPS]
    proposals = []
    seen: Set[frozenset[str]] = set()

    for node_id in bit_nodes:
        local = _bounded_expand(
            seed_nodes={node_id},
            graph=graph,
            max_depth=3,
            allowed_ops=_BIT_OPS | _CAST_OPS | {"arg", "const", "load"},
            allowed_edge_types={"data", "memory"},
            stop_ops=_SELECT_OPS | {"store", "call"},
        )
        bit_only = {item for item in local if opcodes.get(item) in _BIT_OPS}
        if len(bit_only) < 2:
            continue
        key = frozenset(local)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(
            _proposal(
                proposal_type="bit_region",
                seed_nodes=sorted(bit_only)[:2],
                node_ids=local,
                graph=graph,
                confidence=min(1.0, 0.42 + 0.1 * len(bit_only)),
                boundary_reason="bit_cluster",
            )
        )
    return proposals


def detect_memory_regions(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    opcodes = graph["opcodes"]
    memory_nodes = [node_id for node_id, opcode in opcodes.items() if opcode in _MEMORY_OPS]
    proposals = []

    for node_id in memory_nodes:
        local = _bounded_expand(
            seed_nodes={node_id},
            graph=graph,
            max_depth=2,
            allowed_ops=_MEMORY_OPS | _CAST_OPS | {"arg", "const", "load", "store"},
            allowed_edge_types={"data", "memory"},
            stop_ops=_SELECT_OPS | _ARITH_OPS | _BIT_OPS | {"call", "ret"},
        )
        if len(local) < 3:
            continue
        proposals.append(
            _proposal(
                proposal_type="memory_region",
                seed_nodes=[node_id],
                node_ids=local,
                graph=graph,
                confidence=0.5,
                boundary_reason="memory_access_cluster",
            )
        )
    return proposals


def detect_reduction_trees(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    opcodes = graph["opcodes"]
    fanin = graph["fanin"]
    candidates = [
        node_id
        for node_id, opcode in opcodes.items()
        if fanin.get(node_id, 0) >= 2 and opcode in (_ARITH_OPS | {"and", "or", "xor", "icmp"})
    ]
    proposals = []
    seen: Set[frozenset[str]] = set()

    for node_id in candidates:
        local = _bounded_expand(
            seed_nodes={node_id},
            graph=graph,
            max_depth=3,
            allowed_ops=_ARITH_OPS | _BIT_OPS | _CAST_OPS | {"arg", "const", "icmp"},
            allowed_edge_types={"data"},
            stop_ops=_SELECT_OPS | {"store", "call", "ret"},
        )
        if len(local) < 4:
            continue
        internal_roots = [item for item in local if fanin.get(item, 0) >= 2]
        if len(internal_roots) < 2:
            continue
        key = frozenset(local)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(
            _proposal(
                proposal_type="reduction_region",
                seed_nodes=sorted(internal_roots)[:3],
                node_ids=local,
                graph=graph,
                confidence=min(1.0, 0.5 + 0.08 * len(internal_roots)),
                boundary_reason="fanin_reduction_tree",
            )
        )
    return proposals


def fallback_anchor_proposals(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    opcodes = graph["opcodes"]
    priorities = defaultdict(int)
    for node_id, opcode in opcodes.items():
        if opcode in _SELECT_OPS:
            priorities[node_id] = 5
        elif opcode in {"store", "load"}:
            priorities[node_id] = 4
        elif opcode in _ARITH_OPS | _BIT_OPS:
            priorities[node_id] = 3

    ranked = sorted(
        priorities,
        key=lambda node_id: (
            -priorities[node_id],
            -(graph["fanin"].get(node_id, 0) + graph["fanout"].get(node_id, 0)),
            node_id,
        ),
    )
    proposals = []
    for node_id in ranked[:6]:
        opcode = opcodes.get(node_id, "unknown")
        proposal_type = "hybrid_region"
        if opcode in _SELECT_OPS:
            proposal_type = "select_region"
        elif opcode in _ARITH_OPS:
            proposal_type = "arith_region"
        elif opcode in _BIT_OPS:
            proposal_type = "bit_region"
        elif opcode in _MEMORY_OPS:
            proposal_type = "memory_region"

        local = _bounded_expand(
            seed_nodes={node_id},
            graph=graph,
            max_depth=2,
            allowed_ops=None,
            allowed_edge_types={"data", "control", "memory"},
            stop_ops={"call", "ret"},
        )
        if len(local) < 3:
            continue
        proposals.append(
            _proposal(
                proposal_type=proposal_type,
                seed_nodes=[node_id],
                node_ids=local,
                graph=graph,
                confidence=0.3,
                boundary_reason="fallback_anchor_seed",
            )
        )
    return proposals


def generate_region_proposals(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    proposals: List[Dict[str, Any]] = []
    for detector in (
        detect_select_structures,
        detect_state_update_regions,
        detect_reduction_trees,
        detect_arith_chains,
        detect_bit_clusters,
        detect_memory_regions,
    ):
        proposals.extend(detector(graph))

    if not proposals:
        proposals.extend(fallback_anchor_proposals(graph))
    return proposals
