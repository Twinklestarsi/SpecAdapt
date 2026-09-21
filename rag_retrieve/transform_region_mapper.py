"""Conservative source-diff to CDFG-region attribution.

Design-level PPA remains design-level evidence.  A row is promoted to region
evidence only when changed source lines can be localized to a small number of
region anchors; otherwise it stays a benchmark prior.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

from rag_retrieve.path_utils import project_root, resolve_stored_path
from rag_retrieve.transform_stats import as_bool

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_NUMBER_RE = re.compile(r"\b(?:0x[0-9A-Fa-f]+|\d+)\b")
_COLUMN_MATCH_BONUS = 3.0
_NEAR_TOP_RATIO = 0.80
_MAX_DIRECT_REGIONS = 3
_LOW_CONFIDENCE_WEIGHT = 0.25


def changed_source_locations(
    original: Path,
    optimized: Path,
) -> tuple[Set[int], str, Dict[int, List[Tuple[int, int]]]]:
    """Return changed original lines plus reliable one-to-one column spans."""

    before = original.read_text(encoding="utf-8", errors="replace").splitlines()
    after = optimized.read_text(encoding="utf-8", errors="replace").splitlines()
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    changed: Set[int] = set()
    fragments: List[str] = []
    column_spans: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        fragments.extend(before[i1:i2])
        fragments.extend(after[j1:j2])
        if i1 < i2:
            changed.update(range(i1 + 1, i2 + 1))
        elif before:
            changed.add(min(max(i1 + 1, 1), len(before)))

        # Column positions are trustworthy only when changed source lines can
        # be paired one-to-one.  Inserted/deleted/reflowed blocks remain
        # line-only evidence rather than receiving a guessed column.
        if tag != "replace" or i2 - i1 != j2 - j1:
            continue
        for offset, (old_line, new_line) in enumerate(zip(before[i1:i2], after[j1:j2])):
            line_number = i1 + offset + 1
            char_matcher = difflib.SequenceMatcher(a=old_line, b=new_line, autojunk=False)
            for char_tag, c1, c2, _d1, _d2 in char_matcher.get_opcodes():
                if char_tag == "equal":
                    continue
                if c1 < c2:
                    column_spans[line_number].append((c1 + 1, c2))
                else:
                    insertion_column = min(c1 + 1, len(old_line) + 1)
                    column_spans[line_number].append((insertion_column, insertion_column))
    return changed, "\n".join(fragments), dict(column_spans)


def changed_source_lines(original: Path, optimized: Path) -> tuple[Set[int], str]:
    """Compatibility wrapper for callers that only need changed lines/text."""

    changed, text, _column_spans = changed_source_locations(original, optimized)
    return changed, text


def _type_syntax_score(region_type: str, text: str) -> float:
    if "arith" in region_type or "reduction" in region_type:
        return 2.0 if any(token in text for token in ("+", "-", "*", "/", "%")) else 0.0
    if "select" in region_type or "control" in region_type:
        return 2.0 if any(token in text for token in ("if", "switch", "?", "&&", "||", "==", "!=")) else 0.0
    if "bit" in region_type:
        return 2.0 if any(token in text for token in ("<<", ">>", "&", "|", "^", "~")) else 0.0
    if "memory" in region_type or "state" in region_type:
        return 1.5 if any(token in text for token in ("[", "]", "->", "*", "=")) else 0.0
    return 0.5 if "=" in text else 0.0


def _column_match_bonus(
    region: Dict[str, Any],
    changed_columns: Dict[int, List[Tuple[int, int]]] | None,
) -> float:
    if not changed_columns:
        return 0.0
    anchor = region.get("source_anchor", {}) or {}
    for position in anchor.get("source_positions", []):
        line = int(position.get("line") or 0)
        column = int(position.get("column") or 0)
        if column <= 0:
            continue
        if any(start <= column <= end for start, end in changed_columns.get(line, [])):
            return _COLUMN_MATCH_BONUS
    return 0.0


def _score_region(
    region: Dict[str, Any],
    changed_lines: Set[int],
    changed_text: str,
    changed_columns: Dict[int, List[Tuple[int, int]]] | None = None,
) -> float:
    anchor = region.get("source_anchor", {}) or {}
    exact_source_lines = {
        int(line) for line in anchor.get("source_lines", [])
        if str(line).isdigit() and int(line) > 0
    }
    line_start = int(anchor.get("line_start") or 0)
    line_end = int(anchor.get("line_end") or line_start)
    function_start = int(anchor.get("function_line_start") or 0)
    function_end = int(anchor.get("function_line_end") or 0)
    score = 0.0
    if exact_source_lines and changed_lines & exact_source_lines:
        score += 5.0
    elif not exact_source_lines and line_start and any(line_start <= line <= line_end for line in changed_lines):
        score += 5.0
    elif function_start and any(function_start <= line <= function_end for line in changed_lines):
        score += 1.0

    region_type = str(region.get("region_type") or "")
    score += _type_syntax_score(region_type, changed_text)
    identifiers = set(_IDENT_RE.findall(changed_text))
    constants = set(_NUMBER_RE.findall(changed_text))
    key_variables = set(anchor.get("key_variables") or [])
    key_constants = set(anchor.get("key_constants") or [])
    score += min(len(identifiers & key_variables), 3) * 0.75
    score += min(len(constants & key_constants), 2) * 0.5
    score += _column_match_bonus(region, changed_columns)
    return score


def _region_identity(region: Dict[str, Any]) -> str:
    return f"{region.get('graph_function') or ''}:{region.get('region_id') or ''}"


def _composite_group_definition(
    regions: Sequence[Dict[str, Any]],
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a stable, auditable group for regions tied at the top score."""

    ordered = sorted(regions, key=_region_identity)
    fingerprint_parts = [
        str(row.get("objective") or ""),
        str(row.get("subcategory") or ""),
        str(row.get("benchmark") or ""),
        *(_region_identity(region) for region in ordered),
    ]
    digest = hashlib.sha256("\n".join(fingerprint_parts).encode("utf-8")).hexdigest()[:16]
    member_regions = [
        {
            "region_id": str(region.get("region_id") or ""),
            "graph_function": str(region.get("graph_function") or ""),
            "region_type": str(region.get("region_type") or ""),
        }
        for region in ordered
    ]
    source_lines = sorted({
        int(line)
        for region in ordered
        for line in (region.get("source_anchor", {}) or {}).get("source_lines", [])
        if str(line).isdigit() and int(line) > 0
    })
    return {
        "group_id": f"composite_region_{digest}",
        "group_type": "composite_region",
        "grouping_reason": "exact_top_score_tie",
        "member_semantics": "alternative_candidates_not_individually_verified",
        "member_region_ids": [member["region_id"] for member in member_regions],
        "member_region_keys": [_region_identity(region) for region in ordered],
        "member_region_count": len(member_regions),
        "member_region_types": sorted({member["region_type"] for member in member_regions}),
        "graph_functions": sorted({member["graph_function"] for member in member_regions}),
        "source_lines": source_lines,
        "member_regions": member_regions,
    }


def _mapping_evidence(
    row: Dict[str, Any],
    *,
    evidence_level: str,
    weight: float,
    mapping_method: str,
    mapping_score: float,
    mapping_confidence: float,
    column_bonus: float,
    changed_lines: Set[int],
    changed_columns: Dict[int, List[Tuple[int, int]]],
    candidate_region_count: int,
    tied_top_region_count: int,
    score_margin: float,
) -> Dict[str, Any]:
    evidence = dict(row)
    evidence["evidence_level"] = evidence_level
    evidence["attribution_weight"] = weight
    evidence["mapping_method"] = mapping_method
    evidence["mapping_confidence"] = round(mapping_confidence, 6)
    evidence["mapping_score"] = round(mapping_score, 6)
    evidence["mapping_score_margin"] = round(score_margin, 6)
    evidence["column_match_bonus"] = column_bonus
    evidence["candidate_region_count"] = candidate_region_count
    evidence["tied_top_region_count"] = tied_top_region_count
    evidence["changed_original_lines"] = sorted(changed_lines)
    evidence["changed_original_column_spans"] = [
        {
            "line": line,
            "column_start": start,
            "column_end": end,
        }
        for line, spans in sorted(changed_columns.items())
        for start, end in spans
    ]
    return evidence


def map_transform_rows_to_regions(
    rows: Iterable[Dict[str, Any]],
    regions: Sequence[Dict[str, Any]],
    root: str | Path | None = None,
) -> tuple[
    Dict[str, List[Dict[str, Any]]],
    Dict[str, Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    base = project_root(root)
    by_region: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_composite_group: Dict[str, Dict[str, Any]] = {}
    benchmark_rows: List[Dict[str, Any]] = []
    mapped = 0
    low_confidence_mapped = 0
    composite_mapped = 0
    mapping_failures: DefaultDict[str, int] = defaultdict(int)
    rows_with_reliable_column_spans = 0
    mapped_rows_with_column_match = 0
    attributed_rows_with_column_match = 0

    for source in rows:
        row = dict(source)
        if row.get("pairing_status") != "exact":
            benchmark_rows.append(row)
            mapping_failures[str(row.get("pairing_status") or "unpaired")] += 1
            continue
        if as_bool(row.get("no_op")) is True:
            benchmark_rows.append(row)
            mapping_failures["no_op"] += 1
            continue
        original = resolve_stored_path(str(row.get("original_c_path") or ""), base)
        optimized = resolve_stored_path(str(row.get("optimized_c_path") or ""), base)
        if not original.is_file() or not optimized.is_file():
            benchmark_rows.append(row)
            mapping_failures["missing_source_pair"] += 1
            continue
        try:
            changed_lines, changed_text, changed_columns = changed_source_locations(original, optimized)
        except OSError:
            benchmark_rows.append(row)
            mapping_failures["source_read_error"] += 1
            continue
        if not changed_lines:
            benchmark_rows.append(row)
            mapping_failures["empty_diff"] += 1
            continue
        rows_with_reliable_column_spans += int(bool(changed_columns))

        scored = sorted(
            (
                (region, _score_region(region, changed_lines, changed_text, changed_columns))
                for region in regions
            ),
            key=lambda item: (-item[1], str(item[0].get("region_id") or "")),
        )
        if not scored or scored[0][1] < 3.0:
            benchmark_rows.append(row)
            mapping_failures["low_mapping_score"] += 1
            continue
        top_score = scored[0][1]
        threshold = max(3.0, top_score * _NEAR_TOP_RATIO)
        near_top = [(region, score) for region, score in scored if score >= threshold]
        selected = near_top[:_MAX_DIRECT_REGIONS]
        tied_top = [
            (region, score) for region, score in scored
            if abs(score - top_score) < 1e-9
        ]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        score_margin = max(0.0, top_score - second_score)

        if len(near_top) > _MAX_DIRECT_REGIONS:
            if len(tied_top) > 1:
                group = _composite_group_definition(
                    [region for region, _score in tied_top], row
                )
                group_id = str(group["group_id"])
                payload = by_composite_group.setdefault(
                    group_id, {"group": group, "rows": []}
                )
                column_bonus = max(
                    (_column_match_bonus(region, changed_columns) for region, _score in tied_top),
                    default=0.0,
                )
                evidence = _mapping_evidence(
                    row,
                    evidence_level="region_group",
                    weight=1.0,
                    mapping_method="source_diff_composite_region_v1",
                    mapping_score=top_score,
                    mapping_confidence=min(top_score / 8.0, 1.0),
                    column_bonus=column_bonus,
                    changed_lines=changed_lines,
                    changed_columns=changed_columns,
                    candidate_region_count=len(near_top),
                    tied_top_region_count=len(tied_top),
                    score_margin=score_margin,
                )
                evidence["composite_group_id"] = group_id
                evidence["member_region_ids"] = list(group["member_region_ids"])
                payload["rows"].append(evidence)
                composite_mapped += 1
                attributed_rows_with_column_match += int(column_bonus > 0.0)
                continue

            # A unique first place with several close followers is useful for
            # retrieval, but it is not strong enough to become full-weight
            # ground truth.  Keep it on the best atomic region at low weight.
            region, score = scored[0]
            column_bonus = _column_match_bonus(region, changed_columns)
            evidence = _mapping_evidence(
                row,
                evidence_level="region_low_confidence",
                weight=_LOW_CONFIDENCE_WEIGHT,
                mapping_method="source_diff_unique_top_low_confidence_v1",
                mapping_score=score,
                mapping_confidence=(score_margin / top_score if top_score else 0.0),
                column_bonus=column_bonus,
                changed_lines=changed_lines,
                changed_columns=changed_columns,
                candidate_region_count=len(near_top),
                tied_top_region_count=1,
                score_margin=score_margin,
            )
            by_region[str(region.get("region_id"))].append(evidence)
            low_confidence_mapped += 1
            attributed_rows_with_column_match += int(column_bonus > 0.0)
            continue

        weight = 1.0 / len(selected)
        selected_column_bonuses = [
            _column_match_bonus(region, changed_columns) for region, _score in selected
        ]
        mapped_rows_with_column_match += int(any(selected_column_bonuses))
        attributed_rows_with_column_match += int(any(selected_column_bonuses))
        for region, score in selected:
            column_bonus = _column_match_bonus(region, changed_columns)
            evidence = _mapping_evidence(
                row,
                evidence_level="region",
                weight=weight,
                mapping_method="source_diff_debug_line_column_v2",
                mapping_score=score,
                mapping_confidence=min(score / 8.0, 1.0),
                column_bonus=column_bonus,
                changed_lines=changed_lines,
                changed_columns=changed_columns,
                candidate_region_count=len(near_top),
                tied_top_region_count=len(tied_top),
                score_margin=score_margin,
            )
            by_region[str(region.get("region_id"))].append(evidence)
        mapped += 1

    region_attributed = mapped + low_confidence_mapped + composite_mapped
    summary = {
        "row_count": region_attributed + len(benchmark_rows),
        "mapped_row_count": mapped,
        "atomic_low_confidence_mapped_row_count": low_confidence_mapped,
        "composite_mapped_row_count": composite_mapped,
        "region_attributed_row_count": region_attributed,
        "benchmark_only_row_count": len(benchmark_rows),
        "mapping_failures": dict(sorted(mapping_failures.items())),
        "ambiguity_resolution": {
            "broad_candidate_row_count": low_confidence_mapped + composite_mapped,
            "unique_top_low_confidence_row_count": low_confidence_mapped,
            "exact_top_tie_composite_row_count": composite_mapped,
            "unresolved_ambiguous_row_count": 0,
            "low_confidence_attribution_weight": _LOW_CONFIDENCE_WEIGHT,
        },
        "source_location_attribution": {
            "rows_with_reliable_column_spans": rows_with_reliable_column_spans,
            "mapped_rows_with_column_match": mapped_rows_with_column_match,
            "attributed_rows_with_column_match": attributed_rows_with_column_match,
        },
    }
    return dict(by_region), by_composite_group, benchmark_rows, summary
