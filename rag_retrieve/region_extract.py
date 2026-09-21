"""
Pattern-driven region extraction for CDFG precise reranking.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from rag_retrieve.region_features import extract_region_features
from rag_retrieve.region_partition import select_final_proposals
from rag_retrieve.region_patterns import build_graph_view, generate_region_proposals
from rag_retrieve.region_quality import score_region_proposal
from rag_retrieve.schema import CDFGRegion
from rag_retrieve.path_utils import resolve_stored_path

_NODE_RE = re.compile(r'^\s*(n\d+)\s+\[(.*)\];\s*$')
_DOT_ATTR_RE = re.compile(
    r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"((?:\\.|[^"\\])*)"|([^,\]\s]+))'
)
_EDGE_RE = re.compile(r'^\s*(n\d+)\s*->\s*(n\d+)\s+\[color="([^"]+)"\];\s*$')

_EDGE_TYPE_MAP = {
    "gray50": "control",
    "black": "data",
    "red3": "memory",
}

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_NUMBER_RE = re.compile(r"\b\d+\b")


def _parse_dot_attributes(text: str) -> Dict[str, str]:
    attributes: Dict[str, str] = {}
    for match in _DOT_ATTR_RE.finditer(text):
        attributes[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
    return attributes


def _extract_function_span(source_lines: List[str], function_name: str) -> Dict[str, Any]:
    if not source_lines or not function_name:
        return {}

    start_idx = -1
    signature_start = -1
    pattern = re.compile(rf"\b{re.escape(function_name)}\s*\(")
    for idx, line in enumerate(source_lines):
        if pattern.search(line):
            signature_start = idx
            break
    if signature_start < 0:
        return {}

    brace_depth = 0
    seen_open = False
    end_idx = len(source_lines) - 1
    for idx in range(signature_start, len(source_lines)):
        line = source_lines[idx]
        if "{" in line and not seen_open:
            start_idx = signature_start
            seen_open = True
        if seen_open:
            brace_depth += line.count("{")
            brace_depth -= line.count("}")
            if brace_depth <= 0:
                end_idx = idx
                break

    if start_idx < 0:
        return {}
    return {
        "function_line_start": start_idx + 1,
        "function_line_end": end_idx + 1,
        "body_start_index": start_idx,
        "body_end_index": end_idx,
    }


def _terms_from_labels(labels: List[str]) -> Dict[str, List[str]]:
    opcodes: Set[str] = set()
    identifiers: Set[str] = set()
    constants: Set[str] = set()
    expressions: List[str] = []

    skip = {
        "ARG", "ptr", "label", "align", "inbounds", "getelementptr", "llvm",
        "loop", "true", "false",
    }
    for label in labels:
        text = str(label).strip()
        if not text:
            continue
        expressions.append(text)
        rhs = text.split(" = ", 1)[1] if " = " in text else text
        parts = rhs.split()
        if parts:
            opcode = parts[0].lower()
            if opcode not in {"i1", "i8", "i16", "i32", "i64"}:
                opcodes.add(opcode)
        for ident in _IDENT_RE.findall(text):
            if ident in skip or ident.startswith("i") and ident[1:].isdigit():
                continue
            if ident.startswith("arg") and ident[3:].isdigit():
                continue
            identifiers.add(ident)
        constants.update(_NUMBER_RE.findall(text))

    return {
        "ir_opcodes": sorted(opcodes),
        "key_variables": sorted(identifiers),
        "key_constants": sorted(constants, key=lambda item: (len(item), item))[:12],
        "key_expressions": expressions[:8],
    }


def _line_matches_region_type(line: str, region_type: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("//"):
        return False
    if "select" in region_type:
        return any(token in stripped for token in ("if", "switch", "?", "&&", "||", "==", "!=", "<", ">"))
    if "state" in region_type:
        return "se." in stripped or "state" in stripped or "=" in stripped
    if "arith" in region_type or "reduction" in region_type:
        return any(token in stripped for token in ("+", "-", "*", "/", "%", "+=", "-=", "*="))
    if "bit" in region_type:
        return any(token in stripped for token in ("<<", ">>", "&", "|", "^", "~"))
    if "memory" in region_type:
        return "[" in stripped or "*" in stripped or "->" in stripped
    return "=" in stripped or "if" in stripped or "switch" in stripped


def _line_anchor_score(line: str, region_type: str, terms: Dict[str, List[str]]) -> float:
    score = 2.0 if _line_matches_region_type(line, region_type) else 0.0
    identifiers = set(_IDENT_RE.findall(line))
    constants = set(_NUMBER_RE.findall(line))
    score += min(len(identifiers & set(terms.get("key_variables", []))), 3) * 0.75
    score += min(len(constants & set(terms.get("key_constants", []))), 2) * 0.5
    return score


def _build_source_anchor(
    *,
    source_lines: List[str],
    graph_function: str,
    region_type: str,
    anchor_node_ids: List[str],
    anchor_labels: List[str],
    region_node_ids: List[str],
    node_locations: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    terms = _terms_from_labels(anchor_labels)
    anchor: Dict[str, Any] = {
        "function": graph_function,
        "anchor_node_ids": anchor_node_ids,
        "anchor_labels": anchor_labels,
        **terms,
        "mapping_confidence": "ir_only",
        "mapping_note": "Anchor labels come from CDFG/LLVM IR and may not appear verbatim in C source.",
    }

    span = _extract_function_span(source_lines, graph_function)
    if span:
        anchor.update(
            {
                "function_line_start": int(span["function_line_start"]),
                "function_line_end": int(span["function_line_end"]),
            }
        )

    location_scope = "anchor_nodes"
    location_node_ids = list(anchor_node_ids)
    positions = [
        {"node_id": node_id, **node_locations[node_id]}
        for node_id in location_node_ids
        if node_id in node_locations and int(node_locations[node_id].get("line") or 0) > 0
    ]
    if not positions:
        location_scope = "region_nodes_fallback"
        location_node_ids = list(region_node_ids)
        positions = [
            {"node_id": node_id, **node_locations[node_id]}
            for node_id in location_node_ids
            if node_id in node_locations and int(node_locations[node_id].get("line") or 0) > 0
        ]

    if positions:
        exact_lines = sorted({int(position["line"]) for position in positions})
        source_files = sorted({
            str(position.get("file") or "") for position in positions if position.get("file")
        })
        anchor.update(
            {
                "line_start": exact_lines[0],
                "line_end": exact_lines[-1],
                "source_lines": exact_lines,
                "source_files": source_files,
                "source_positions": positions,
                "source_location_scope": location_scope,
                "mapping_confidence": "llvm_debug_location",
                "mapping_note": (
                    "Source lines come from LLVM DILocation metadata attached to CDFG nodes."
                ),
            }
        )
        valid_lines = [line for line in exact_lines if 1 <= line <= len(source_lines)]
        if valid_lines:
            snippet_start = max(0, min(valid_lines) - 3)
            snippet_end = min(len(source_lines) - 1, max(valid_lines) + 1)
            if snippet_end - snippet_start > 24:
                snippet_start = max(0, valid_lines[0] - 3)
                snippet_end = min(len(source_lines) - 1, valid_lines[0] + 1)
            snippet_lines = [
                f"{idx + 1}: {source_lines[idx]}"
                for idx in range(snippet_start, snippet_end + 1)
            ]
            anchor.update(
                {
                    "snippet_line_start": snippet_start + 1,
                    "snippet_line_end": snippet_end + 1,
                    "c_snippet": "\n".join(snippet_lines),
                }
            )
        return anchor

    if not span:
        return anchor

    body_start = int(span["body_start_index"])
    body_end = int(span["body_end_index"])

    candidate_indices = [
        idx for idx in range(body_start, body_end + 1)
        if _line_matches_region_type(source_lines[idx], region_type)
    ]
    if not candidate_indices:
        candidate_indices = list(range(body_start, min(body_start + 12, body_end + 1)))

    center = max(
        candidate_indices or [body_start],
        key=lambda idx: (_line_anchor_score(source_lines[idx], region_type, terms), -idx),
    )
    snippet_start = max(body_start, center - 4)
    snippet_end = min(body_end, center + 8)
    snippet_lines = [
        f"{idx + 1}: {source_lines[idx]}"
        for idx in range(snippet_start, snippet_end + 1)
    ]
    anchor.update(
        {
            "line_start": center + 1,
            "line_end": center + 1,
            "snippet_line_start": snippet_start + 1,
            "snippet_line_end": snippet_end + 1,
            "c_snippet": "\n".join(snippet_lines),
            "mapping_confidence": "heuristic_function_scope",
            "mapping_note": (
                "Snippet is selected from the target C function using region-type heuristics; "
                "use it as a localization hint, not an exact source mapping."
            ),
        }
    )
    return anchor


def parse_cdfg_dot(dot_path: str | Path) -> Dict[str, Any]:
    labels: Dict[str, str] = {}
    node_locations: Dict[str, Dict[str, Any]] = {}
    succs: Dict[str, Set[str]] = defaultdict(set)
    preds: Dict[str, Set[str]] = defaultdict(set)
    edge_types: Dict[Tuple[str, str], str] = {}

    path = resolve_stored_path(dot_path)
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        node_match = _NODE_RE.match(raw_line)
        if node_match:
            node_id = node_match.group(1)
            attributes = _parse_dot_attributes(node_match.group(2))
            if "label" not in attributes:
                continue
            labels[node_id] = attributes["label"]
            try:
                source_line = int(attributes.get("source_line") or 0)
                if source_line > 0:
                    node_locations[node_id] = {
                        "file": attributes.get("source_file", ""),
                        "line": source_line,
                        "column": int(attributes.get("source_column") or 0),
                        "debug_location_id": int(attributes.get("debug_location_id") or 0),
                    }
            except ValueError:
                pass
            continue
        edge_match = _EDGE_RE.match(raw_line)
        if edge_match:
            src, dst, color = edge_match.groups()
            succs[src].add(dst)
            preds[dst].add(src)
            edge_types[(src, dst)] = _EDGE_TYPE_MAP.get(color, "data")

    return {
        "dot_path": str(path),
        "labels": labels,
        "node_locations": node_locations,
        "succs": {key: set(value) for key, value in succs.items()},
        "preds": {key: set(value) for key, value in preds.items()},
        "edge_types": edge_types,
    }


def _edge_count(node_ids: Set[str], succs: Dict[str, Set[str]]) -> int:
    return sum(1 for src in node_ids for dst in succs.get(src, set()) if dst in node_ids)


def _proposal_to_region(
    proposal: Dict[str, Any],
    graph_name: str,
    graph: Dict[str, Any],
    index: int,
    source_lines: List[str] | None = None,
) -> CDFGRegion:
    node_ids = set(proposal["node_ids"])
    features = extract_region_features(
        node_ids=node_ids,
        labels=graph["labels"],
        succs=graph["succs"],
        preds=graph["preds"],
        edge_types=graph["edge_types"],
    )
    features.update(
        {
            "proposal_type": proposal["proposal_type"],
            "quality_score": proposal.get("quality_score", 0.0),
            "motif_purity": proposal.get("motif_purity", 0.0),
            "boundary_clarity": proposal.get("boundary_clarity", 0.0),
            "internal_density": proposal.get("internal_density", 0.0),
            "size_reasonableness": proposal.get("size_reasonableness", 0.0),
            "anchor_confidence": proposal.get("anchor_confidence", 0.0),
            "boundary_reason": proposal.get("boundary_reason", ""),
            "overlap_suppressed_count": proposal.get("overlap_suppressed_count", 0),
            "semantic_priority": proposal.get("semantic_priority", 0),
        }
    )
    if proposal["proposal_type"] != "hybrid_region":
        features["region_type_guess"] = proposal["proposal_type"]

    anchor_node_ids = sorted(proposal.get("anchor_node_ids", []))
    anchor_labels = list(proposal.get("anchor_labels", []))
    source_anchor = _build_source_anchor(
        source_lines=source_lines or [],
        graph_function=graph_name,
        region_type=str(features.get("region_type_guess") or proposal["proposal_type"]),
        anchor_node_ids=anchor_node_ids,
        anchor_labels=anchor_labels,
        region_node_ids=sorted(node_ids),
        node_locations=graph.get("node_locations", {}),
    )

    return CDFGRegion(
        region_id=f"{graph_name}_region_{index}",
        graph_function=graph_name,
        region_type=str(features.get("region_type_guess") or proposal["proposal_type"]),
        anchor_node_ids=anchor_node_ids,
        anchor_labels=anchor_labels,
        node_ids=sorted(node_ids),
        edge_count=_edge_count(node_ids, graph["succs"]),
        features=features,
        source_anchor=source_anchor,
    )


def extract_regions_from_graph(
    dot_path: str | Path,
    graph_function: str | None = None,
    source_lines: List[str] | None = None,
) -> List[CDFGRegion]:
    parsed = parse_cdfg_dot(dot_path)
    graph = build_graph_view(
        labels=parsed["labels"],
        succs=parsed["succs"],
        preds=parsed["preds"],
        edge_types=parsed["edge_types"],
    )
    graph["node_locations"] = parsed["node_locations"]
    graph_name = graph_function or Path(dot_path).name.replace(".cdfg.dot", "")

    proposals = generate_region_proposals(graph)
    enriched = []
    for proposal in proposals:
        proposal = dict(proposal)
        proposal.update(score_region_proposal(proposal, graph))
        enriched.append(proposal)

    selected = select_final_proposals(enriched)
    regions = [
        _proposal_to_region(
            proposal=proposal,
            graph_name=graph_name,
            graph=graph,
            index=index,
            source_lines=source_lines,
        )
        for index, proposal in enumerate(selected, start=1)
    ]
    return regions


def extract_regions_from_query_record(query_record: Dict[str, Any]) -> Dict[str, Any]:
    outputs = []
    source_lines: List[str] = []
    source_path = resolve_stored_path(str(query_record.get("source_c_path", "")))
    if source_path.is_file():
        source_lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()

    for graph in query_record.get("graphs", []):
        dot_path = str(resolve_stored_path(graph["dot_path"]))
        regions = extract_regions_from_graph(
            dot_path=dot_path,
            graph_function=graph.get("function_name"),
            source_lines=source_lines,
        )
        outputs.append(
            {
                "graph_function": graph.get("function_name"),
                "dot_path": dot_path,
                "node_count": graph.get("node_count", 0),
                "edge_count": graph.get("edge_count", 0),
                "regions": [region.to_dict() for region in regions],
            }
        )
    return {
        "benchmark": query_record.get("benchmark"),
        "source_c_path": query_record.get("source_c_path"),
        "query_root": query_record.get("query_root"),
        "graphs": outputs,
    }


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract reranking regions from a query CDFG JSON.")
    parser.add_argument("query_json", help="Path to query CDFG JSON")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = extract_regions_from_query_record(_load_json(args.query_json))
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
