"""
Feature extraction utilities for CDFG regions.
"""

from __future__ import annotations

from collections import Counter, deque
import re
from typing import Dict, Iterable, List, Set


def normalize_opcode(label: str) -> str:
    text = (label or "").strip()
    if not text:
        return "unknown"
    if text.startswith("ARG "):
        return "arg"
    if " = " in text:
        text = text.split(" = ", 1)[1].strip()
    return text.split(None, 1)[0].lower()


def classify_region_type(opcodes: Iterable[str]) -> str:
    opcode_set = set(opcodes)
    arith = opcode_set & {"add", "mul", "sub", "shl", "lshr", "ashr", "trunc", "zext", "sext"}
    control = opcode_set & {"phi", "br", "icmp", "select"}
    bit = opcode_set & {"and", "or", "xor", "not", "shl", "lshr", "ashr"}
    memory = opcode_set & {"store", "load", "getelementptr"}

    if memory and (arith or bit or control):
        return "state_region"
    if control and not (arith or bit):
        return "select_region"
    if bit and not control and not memory and not (arith - {"shl", "lshr", "ashr"}):
        return "bit_region"
    if arith and not control and not memory:
        return "arith_region"
    if memory and not (arith or bit or control):
        return "memory_region"
    if sum(opcode_set.__contains__(item) for item in ("add", "and", "or", "xor", "mul")) >= 2 and not memory:
        return "reduction_region"
    if control or arith or bit:
        return "hybrid_region"
    return "generic_region"


def _longest_path_length(nodes: Set[str], succs: Dict[str, Set[str]]) -> int:
    indegree = {node: 0 for node in nodes}
    for src in nodes:
        for dst in succs.get(src, set()):
            if dst in nodes:
                indegree[dst] += 1

    queue = deque(sorted(node for node, deg in indegree.items() if deg == 0))
    depth = {node: 1 for node in queue}
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1
        for nxt in sorted(succs.get(node, set())):
            if nxt not in nodes:
                continue
            if depth.get(nxt, 1) < depth[node] + 1:
                depth[nxt] = depth[node] + 1
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if visited != len(nodes):
        return max(len(nodes) // 2, 1)
    return max(depth.values(), default=1)


def extract_region_features(
    node_ids: Iterable[str],
    labels: Dict[str, str],
    succs: Dict[str, Set[str]],
    preds: Dict[str, Set[str]],
    edge_types: Dict[tuple[str, str], str],
) -> Dict[str, object]:
    node_set = set(node_ids)
    ordered_nodes = sorted(node_set)
    opcodes = [normalize_opcode(labels[node]) for node in ordered_nodes]
    opcode_hist = dict(sorted(Counter(opcodes).items()))
    region_labels = [labels[node] for node in ordered_nodes]
    bitwidths = [int(value) for label in region_labels for value in re.findall(r"\bi(\d+)\b", label)]
    constants = [
        int(value, 16) if value.lower().startswith("0x") else int(value)
        for label in region_labels
        for value in re.findall(r"(?<![%A-Za-z_])(?:-?\d+|0x[0-9A-Fa-f]+)\b", label)
        if not value.startswith("-") or value[1:].isdigit()
    ]
    signed_ops = sum(opcode_hist.get(name, 0) for name in ("sext", "sdiv", "srem", "ashr"))
    unsigned_ops = sum(opcode_hist.get(name, 0) for name in ("zext", "udiv", "urem", "lshr"))

    internal_edges = []
    control_edges = 0
    data_edges = 0
    memory_edges = 0
    fanins: List[int] = []
    fanouts: List[int] = []

    for src in ordered_nodes:
        local_succs = sorted(dst for dst in succs.get(src, set()) if dst in node_set)
        local_preds = sorted(p for p in preds.get(src, set()) if p in node_set)
        fanouts.append(len(local_succs))
        fanins.append(len(local_preds))
        for dst in local_succs:
            internal_edges.append((src, dst))
            edge_type = edge_types.get((src, dst), "data")
            if edge_type == "control":
                control_edges += 1
            elif edge_type == "memory":
                memory_edges += 1
            else:
                data_edges += 1

    branch_count = opcode_hist.get("br", 0)
    phi_count = opcode_hist.get("phi", 0)
    load_count = opcode_hist.get("load", 0)
    store_count = opcode_hist.get("store", 0)
    call_count = opcode_hist.get("call", 0)

    features: Dict[str, object] = {
        "node_count": len(node_set),
        "edge_count": len(internal_edges),
        "data_edge_count": data_edges,
        "control_edge_count": control_edges,
        "memory_edge_count": memory_edges,
        "opcode_histogram": opcode_hist,
        "top_opcodes": sorted(opcode_hist.items(), key=lambda item: (-item[1], item[0]))[:8],
        "graph_depth": _longest_path_length(node_set, succs),
        "branch_count": branch_count,
        "phi_count": phi_count,
        "load_count": load_count,
        "store_count": store_count,
        "call_count": call_count,
        "max_fanin": max(fanins, default=0),
        "max_fanout": max(fanouts, default=0),
        "avg_fanin": round(sum(fanins) / len(fanins), 4) if fanins else 0.0,
        "avg_fanout": round(sum(fanouts) / len(fanouts), 4) if fanouts else 0.0,
        "bitwidth_histogram": dict(sorted(Counter(bitwidths).items())),
        "min_bitwidth": min(bitwidths, default=0),
        "max_bitwidth": max(bitwidths, default=0),
        "avg_bitwidth": round(sum(bitwidths) / len(bitwidths), 4) if bitwidths else 0.0,
        "constant_count": len(constants),
        "unique_constants": sorted(set(constants))[:16],
        "signed_operation_count": signed_ops,
        "unsigned_operation_count": unsigned_ops,
        "signedness_hint": "signed" if signed_ops > unsigned_ops else "unsigned" if unsigned_ops > signed_ops else "mixed_or_unknown",
        "array_gep_count": opcode_hist.get("getelementptr", 0),
        "expensive_operator_count": sum(opcode_hist.get(name, 0) for name in ("mul", "sdiv", "udiv", "srem", "urem")),
        "loop_phi_count": phi_count,
        "loop_backedge_count": sum(
            1
            for src in ordered_nodes
            for dst in sorted(succs.get(src, set()))
            if dst in node_set and src.startswith("n") and dst.startswith("n")
            and src[1:].isdigit() and dst[1:].isdigit() and int(dst[1:]) <= int(src[1:])
        ),
    }
    features["dependency_density"] = round(
        len(internal_edges) / max(len(node_set) * max(len(node_set) - 1, 1), 1), 6
    )
    features["depth_per_node"] = round(float(features["graph_depth"]) / max(len(node_set), 1), 6)
    features["has_loop_like"] = bool(features["loop_backedge_count"] or phi_count >= 2)
    features["has_array_access"] = bool(features["array_gep_count"] or (load_count + store_count >= 3))

    features["has_add_chain"] = opcode_hist.get("add", 0) >= 2 or (
        opcode_hist.get("add", 0) >= 1 and features["graph_depth"] >= 4
    )
    features["has_const_mult_like"] = opcode_hist.get("mul", 0) > 0 or (
        opcode_hist.get("shl", 0) > 0 and opcode_hist.get("add", 0) > 0
    )
    features["has_priority_select"] = branch_count >= 2 and phi_count >= 1
    features["has_reduction_like"] = opcode_hist.get("and", 0) + opcode_hist.get("or", 0) + opcode_hist.get("xor", 0) >= 3
    features["has_state_update"] = load_count >= 1 and store_count >= 1
    features["has_bit_reorg"] = opcode_hist.get("and", 0) + opcode_hist.get("or", 0) + opcode_hist.get("xor", 0) + opcode_hist.get("shl", 0) + opcode_hist.get("lshr", 0) + opcode_hist.get("ashr", 0) >= 2
    features["region_type_guess"] = classify_region_type(opcodes)
    return features
