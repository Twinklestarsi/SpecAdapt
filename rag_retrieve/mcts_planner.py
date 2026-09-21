"""
Module 4.5 MCTS-style planner bridging Module 4 retrieval and Module 5 execution.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rag_retrieve.schema import Module45Plan, Module5Action, Module5ActionConstraints
from rag_retrieve.reranker import _transform_applicable


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return [1.0 if maximum > 0.0 else 0.0 for _ in values]
    scale = maximum - minimum
    return [(value - minimum) / scale for value in values]


def _risk_label(
    confidence: float,
    match_count: int,
    region_type: str,
    risk_flag: str = "",
) -> str:
    # A watchlisted transform (bad mean, acceptable median) is high variance by
    # construction: strong evidence for it does not make it safe.
    if risk_flag:
        return "high"
    if confidence >= 0.8 and match_count >= 2 and "mixed" not in region_type:
        return "low"
    if confidence < 0.45 or match_count == 0:
        return "high"
    return "medium"


def _apply_scope(region_type: str) -> str:
    if "mixed" in region_type:
        return "regional"
    return "local"


_AREA_GAIN_SCALE_PCT = 10.0
"""A 10% area saving is treated as a "full" gain signal (tanh(1) = 0.76)."""

_TIMING_GAIN_SCALE_PS = 500.0
"""Timing gains are picoseconds, so they need their own scale."""


def _gain_norm(expected_metric_gain: float, objective: str) -> float:
    """Squash a raw expected gain into [0, 1] on an objective-aware scale.

    ``expected_metric_gain`` carries percent for AREA and picoseconds for
    TIMING, so a single linear scale cannot serve both.  A negative expected
    gain contributes nothing rather than a negative prior: the risk term of the
    search reward already penalises bad actions, and letting the prior go
    negative would make an untested action look better than a measured-bad one.
    Min-max normalisation is deliberately avoided here -- it would hand the
    best-of-a-bad-batch action a full 1.0 even when every candidate regresses.
    """

    scale = _AREA_GAIN_SCALE_PCT if str(objective).upper() == "AREA" else _TIMING_GAIN_SCALE_PS
    return max(0.0, math.tanh(_safe_float(expected_metric_gain) / scale))


_CYCLE_CHANGING_TRANSFORMS = {
    # These rewrite the schedule, not just the logic: they trade combinational
    # work for extra cycles.  Under a fixed-latency behavioral contract that is
    # exactly what JasperGold refuses, so they cannot pay off.
    "SERIALIZE_PARALLELISM",
    "OPERATOR_TIME_MULTIPLEX",
    "PIPELINE_STAGE_INSERT",
    "LOOP_PIPELINING",
}


def _contract_fixes_latency(behavioral_contract: Optional[Dict[str, Any]]) -> bool:
    """True when the spec pins every output to a fixed cycle latency."""

    if not isinstance(behavioral_contract, dict):
        return False
    latencies = behavioral_contract.get("output_latency_cycles")
    return isinstance(latencies, dict) and bool(latencies)


def _contract_rejects(
    transform_name: str, behavioral_contract: Optional[Dict[str, Any]]
) -> tuple[bool, str]:
    """Reject cycle-changing transforms when the contract fixes latency."""

    if not _contract_fixes_latency(behavioral_contract):
        return False, ""
    upper = str(transform_name).upper()
    for banned in _CYCLE_CHANGING_TRANSFORMS:
        if banned in upper:
            return True, "contract_fixed_output_latency"
    return False, ""


def _stratified_candidates(
    actions: List[Module5Action],
    limit: int,
    max_per_transform: int = 2,
    min_distinct_transforms: int = 5,
) -> List[Module5Action]:
    """Pick ``limit`` actions without letting one transform own the search tree.

    A flat top-``limit`` cut by ``planning_score`` collapses into a single
    transform whenever the prior favours one: in the cfirst40 run 76% of every
    selected action was ``SERIALIZE_PARALLELISM``.  Round-robin over transform
    kinds instead, so the first slots go to distinct kinds.

    ``max_per_transform`` is a floor on the per-kind cap, not a hard ceiling:
    when the candidate pool holds fewer than ``min_distinct_transforms`` kinds
    the cap widens evenly (every kind gets the same extra slot) so the caller
    still gets the search budget it asked for, instead of the widening going to
    whichever kind happens to score highest.
    """

    limit = max(1, limit)
    if len(actions) <= limit:
        return list(actions)

    by_transform: Dict[str, List[Module5Action]] = {}
    for action in actions:  # `actions` arrives sorted by planning_score desc
        by_transform.setdefault(action.transform_name, []).append(action)

    # Order kinds by their single best action so the strongest kind still leads.
    kinds = sorted(
        by_transform,
        key=lambda name: (
            by_transform[name][0].planning_score,
            by_transform[name][0].expected_metric_gain,
        ),
        reverse=True,
    )

    target_kinds = max(1, min(min_distinct_transforms, len(kinds)))
    per_transform_cap = max(
        max_per_transform, -(-limit // target_kinds)  # ceil division
    )

    selected: List[Module5Action] = []
    chosen_ids: Set[str] = set()
    for slot in range(per_transform_cap):
        if len(selected) >= limit:
            break
        for name in kinds:
            if len(selected) >= limit:
                break
            bucket = by_transform[name]
            if slot >= len(bucket):
                continue
            action = bucket[slot]
            selected.append(action)
            chosen_ids.add(action.action_id)

    if len(selected) < limit:
        # Buckets ran dry before the budget did.  Fill by score; the cap has
        # nothing left to protect at this point.
        for action in actions:
            if len(selected) >= limit:
                break
            if action.action_id in chosen_ids:
                continue
            selected.append(action)
            chosen_ids.add(action.action_id)

    selected.sort(
        key=lambda item: (
            item.planning_score,
            item.expected_metric_gain,
            item.confidence,
            item.priority,
        ),
        reverse=True,
    )
    return selected


def _rank_bonus(global_plan: Dict[str, Dict[str, Any]], transform_name: str) -> float:
    entry = global_plan.get(transform_name)
    if not entry:
        return 0.0
    rank = int(entry.get("rank", 999))
    return max(0.0, 1.0 - (rank - 1) * 0.08)


def build_action_space(
    plan_payload: Dict[str, Any],
    behavioral_contract: Optional[Dict[str, Any]] = None,
) -> List[Module5Action]:
    benchmark = plan_payload["benchmark"]
    objective = plan_payload["objective"]
    source_c_path = plan_payload["source_c_path"]
    harmful_blacklist = list(plan_payload.get("harmful_blacklist", []))
    if behavioral_contract is None:
        behavioral_contract = plan_payload.get("behavioral_contract")

    global_plan_map: Dict[str, Dict[str, Any]] = {}
    for index, entry in enumerate(plan_payload.get("global_plan", []), start=1):
        global_plan_map[entry["transform_name"]] = {**entry, "rank": index}

    candidate_rows = []
    for region in plan_payload.get("llm_ready_payload", {}).get("regions", []):
        region_id = region["region_id"]
        region_type = region["region_type"]
        priority = _safe_float(region.get("priority", 0.0))
        for rec in region.get("recommended_transforms", []):
            transform_name = str(rec.get("transform_name") or "")
            applicable, _reason = _transform_applicable(
                transform_name,
                {
                    "region_type": region_type,
                    **dict(region.get("region_features") or {}),
                },
            )
            if not applicable:
                continue
            rejected_by_contract, _contract_reason = _contract_rejects(
                transform_name, behavioral_contract
            )
            if rejected_by_contract:
                continue
            candidate_rows.append(
                {
                    "region": region,
                    "transform_name": rec["transform_name"],
                    "priority": priority,
                    "expected_metric_gain": _safe_float(rec.get("expected_metric_gain", 0.0)),
                    "confidence": _safe_float(rec.get("confidence", 0.0)),
                    "samples": int(rec.get("samples", 0)),
                    "match_count": len(rec.get("supporting_matches", [])),
                    "rank_bonus": _rank_bonus(global_plan_map, rec["transform_name"]),
                    "memory_score": _safe_float(rec.get("memory_score", 0.0)),
                    "memory_success_rate": _safe_float(
                        rec.get("memory_success_rate", 0.0)
                    ),
                    "memory_evidence_count": int(
                        rec.get("memory_evidence_count", 0)
                    ),
                    "memory_evidence": list(rec.get("memory_evidence", [])),
                    "risk_flag": str(rec.get("risk_flag") or ""),
                    "region_id": region_id,
                    "region_type": region_type,
                }
            )

    priority_norm = _normalize([row["priority"] for row in candidate_rows])
    memory_norm = _normalize([row["memory_score"] for row in candidate_rows])

    actions: List[Module5Action] = []
    for row, priority_score, memory_score in zip(
        candidate_rows, priority_norm, memory_norm
    ):
        # Keep the proposal prior s_i separate from confidence c_i.  The
        # expected gain g_i also enters here on purpose: `candidate_limit`
        # truncates the action space by `planning_score`, so a transform with a
        # large measured gain but a modest retrieval priority never reached the
        # search tree while this term was missing.  The search reward still adds
        # its own (smaller, capped) gain term on top -- that one ranks whole
        # action sets, this one decides which actions get to compete at all.
        gain_norm = _gain_norm(row["expected_metric_gain"], objective)
        planning_score = (
            0.35 * priority_score
            + 0.15 * row["rank_bonus"]
            + 0.25 * memory_score
            + 0.25 * gain_norm
        )
        if row["memory_evidence_count"]:
            planning_score -= 0.15 * (1.0 - row["memory_success_rate"])

        region = row["region"]
        transform_name = row["transform_name"]
        risk = _risk_label(
            row["confidence"],
            row["match_count"],
            row["region_type"],
            row["risk_flag"],
        )
        action = Module5Action(
            action_id=f"{row['region_id']}::{transform_name}",
            benchmark=benchmark,
            objective=objective,
            source_c_path=source_c_path,
            region_id=row["region_id"],
            region_type=row["region_type"],
            transform_name=transform_name,
            priority=round(row["priority"], 6),
            planning_score=round(planning_score, 6),
            expected_metric_gain=round(row["expected_metric_gain"], 6),
            confidence=round(row["confidence"], 6),
            problem_hypothesis=region.get("problem_hypothesis", ""),
            execution_hint=region.get("execution_hint", ""),
            source_anchor=region.get("source_anchor", {}),
            supporting_matches=region.get("supporting_matches", []),
            supporting_benchmarks=sorted(
                {
                    match["benchmark"]
                    for match in region.get("supporting_matches", [])
                    if "benchmark" in match
                }
            ),
            memory_score=round(row["memory_score"], 6),
            memory_success_rate=round(row["memory_success_rate"], 6),
            memory_evidence_count=row["memory_evidence_count"],
            memory_evidence=row["memory_evidence"],
            region_features=dict(region.get("region_features", {})),
            constraints=Module5ActionConstraints(
                must_preserve_behavior=True,
                forbidden_transforms=harmful_blacklist,
                conflict_group=row["region_id"],
                apply_scope=_apply_scope(row["region_type"]),
                expected_risk=risk,
            ),
        )
        actions.append(action)

    actions.sort(
        key=lambda item: (
            item.planning_score,
            item.expected_metric_gain,
            item.confidence,
            item.priority,
        ),
        reverse=True,
    )
    return actions


def _structural_filter_audit(
    plan_payload: Dict[str, Any],
    behavioral_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Explain which recommendations are removed before MCTS search."""

    if behavioral_contract is None:
        behavioral_contract = plan_payload.get("behavioral_contract")
    contract_active = _contract_fixes_latency(behavioral_contract)

    raw_count = 0
    rejected: List[Dict[str, str]] = []
    contract_rejected: List[Dict[str, str]] = []
    for region in plan_payload.get("llm_ready_payload", {}).get("regions", []):
        region_id = str(region.get("region_id") or "")
        region_type = str(region.get("region_type") or "")
        features = {
            "region_type": region_type,
            **dict(region.get("region_features") or {}),
        }
        for rec in region.get("recommended_transforms", []):
            raw_count += 1
            name = str(rec.get("transform_name") or "")
            applicable, reason = _transform_applicable(name, features)
            if not applicable:
                rejected.append(
                    {
                        "action_id": f"{region_id}::{name}",
                        "transform_name": name,
                        "region_id": region_id,
                        "reason": reason,
                    }
                )
                continue
            by_contract, contract_reason = _contract_rejects(name, behavioral_contract)
            if by_contract:
                contract_rejected.append(
                    {
                        "action_id": f"{region_id}::{name}",
                        "transform_name": name,
                        "region_id": region_id,
                        "reason": contract_reason,
                    }
                )
    return {
        "raw_recommendation_count": raw_count,
        "rejected_count": len(rejected) + len(contract_rejected),
        "rejections": rejected + contract_rejected,
        "applicability_rejected_count": len(rejected),
        "contract_rejected_count": len(contract_rejected),
        "contract_filter_active": contract_active,
        "contract_fixed_latency_outputs": sorted(
            (behavioral_contract or {}).get("output_latency_cycles", {})
        )
        if contract_active
        else [],
        "contract_blocked_transforms": sorted(
            {item["transform_name"] for item in contract_rejected}
        ),
        "fallback_added": False,
        "post_execution_replan": False,
    }


_INCOMPATIBLE_TRANSFORM_PAIRS = {
    frozenset({"PIPELINE_STAGE_INSERT", "PIPELINE_RELAX"}),
    frozenset({"INLINE_CRITICAL_FUNCTION", "OUTLINE_LONG_COMPUTE"}),
    frozenset({"INLINE_CRITICAL_FUNCTION", "FUNCTION_OUTLINE_REUSE"}),
    frozenset({"ENCODE_ONEHOT_TO_BINARY", "FSM_REENCODE"}),
}


def _anchor_nodes(action: Module5Action) -> Set[str]:
    return {
        str(item)
        for item in action.source_anchor.get("anchor_node_ids", [])
        if str(item)
    }


def _line_interval(action: Module5Action) -> Optional[Tuple[str, int, int]]:
    anchor = action.source_anchor
    try:
        start = int(anchor.get("line_start"))
        end = int(anchor.get("line_end", start))
    except (TypeError, ValueError):
        return None
    function = str(anchor.get("function") or "")
    return function, min(start, end), max(start, end)


def _anchors_overlap(left: Module5Action, right: Module5Action) -> bool:
    left_nodes = _anchor_nodes(left)
    right_nodes = _anchor_nodes(right)
    if left_nodes and right_nodes and left_nodes.intersection(right_nodes):
        return True

    left_interval = _line_interval(left)
    right_interval = _line_interval(right)
    if not left_interval or not right_interval:
        return False
    left_function, left_start, left_end = left_interval
    right_function, right_start, right_end = right_interval
    if left_function and right_function and left_function != right_function:
        return False
    return max(left_start, right_start) <= min(left_end, right_end)


def _conflicts(candidate: Module5Action, selected: List[Module5Action]) -> bool:
    for action in selected:
        if action.region_id == candidate.region_id:
            return True
        if _anchors_overlap(action, candidate):
            return True
        if frozenset({action.transform_name, candidate.transform_name}) in _INCOMPATIBLE_TRANSFORM_PAIRS:
            return True
    return False


def _state_reward_breakdown(selected: List[Module5Action]) -> Dict[str, float]:
    planning_prior = 0.0
    confidence = 0.0
    expected_gain = 0.0
    risk = 0.0
    region_types = set()
    for action in selected:
        planning_prior += action.planning_score
        confidence += 0.20 * action.confidence
        expected_gain += 0.25 * min(max(action.expected_metric_gain, 0.0), 100.0) / 100.0
        risk += {"low": 0.01, "medium": 0.05, "high": 0.12}[
            action.constraints.expected_risk
        ]
        region_types.add(action.region_type)
    redundancy = 0.04 * max(0, len(selected) - len(region_types))
    total = planning_prior + confidence + expected_gain - risk - redundancy
    return {
        "planning_prior": round(planning_prior, 6),
        "confidence": round(confidence, 6),
        "expected_gain": round(expected_gain, 6),
        "risk_penalty": round(risk, 6),
        "redundancy_penalty": round(redundancy, 6),
        "total": round(total, 6),
    }


def _state_reward(selected: List[Module5Action]) -> float:
    return _state_reward_breakdown(selected)["total"]


@dataclass
class SearchNode:
    chosen_indices: List[int] = field(default_factory=list)
    remaining_indices: List[int] = field(default_factory=list)
    untried_indices: List[int] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    children: List["SearchNode"] = field(default_factory=list)
    parent: Optional["SearchNode"] = None
    terminal: bool = False

    def uct_score(self, exploration: float) -> float:
        if self.parent is None:
            return float("inf")
        if self.visits == 0:
            return float("inf")
        return (self.value / self.visits) + exploration * math.sqrt(math.log(self.parent.visits + 1) / self.visits)


def _expand_one(
    node: SearchNode,
    actions: List[Module5Action],
    max_depth: int,
    rng: random.Random,
) -> Optional[SearchNode]:
    if node.terminal:
        return None

    selected_actions = [actions[index] for index in node.chosen_indices]
    if len(selected_actions) >= max_depth:
        node.terminal = True
        return None

    while node.untried_indices:
        position = rng.randrange(len(node.untried_indices))
        index = node.untried_indices.pop(position)
        candidate = actions[index]
        if _conflicts(candidate, selected_actions):
            continue
        # Only extend with higher indices.  A set of actions therefore has one
        # canonical tree state rather than one state per action permutation.
        remaining = [item for item in node.remaining_indices if item > index]
        child = SearchNode(
            chosen_indices=node.chosen_indices + [index],
            remaining_indices=remaining,
            untried_indices=list(remaining),
            parent=node,
        )
        node.children.append(child)
        return child

    if not node.children:
        node.terminal = True
    return None


def _select_leaf(
    root: SearchNode,
    actions: List[Module5Action],
    max_depth: int,
    exploration: float,
    rng: random.Random,
) -> SearchNode:
    node = root
    while True:
        if node.terminal or len(node.chosen_indices) >= max_depth:
            node.terminal = True
            return node
        widening_limit = max(1, int(2.0 * math.sqrt(node.visits + 1)))
        if node.untried_indices and len(node.children) < widening_limit:
            expanded = _expand_one(node, actions, max_depth, rng)
            if expanded is not None:
                return expanded
        if node.children:
            node = max(node.children, key=lambda child: child.uct_score(exploration))
            continue
        node.terminal = True
        return node


def _rollout(
    node: SearchNode,
    actions: List[Module5Action],
    max_depth: int,
    rng: random.Random,
) -> tuple[float, List[int]]:
    selected_indices = list(node.chosen_indices)
    selected = [actions[index] for index in selected_indices]
    last_index = selected_indices[-1] if selected_indices else -1
    remaining_indices = [
        index for index in node.remaining_indices if index > last_index
    ]
    rng.shuffle(remaining_indices)

    for index in remaining_indices:
        candidate = actions[index]
        if len(selected) >= max_depth:
            break
        if _conflicts(candidate, selected):
            continue
        if candidate.planning_score < 0.15:
            continue
        selected.append(candidate)
        selected_indices.append(index)

    return _state_reward(selected), selected_indices


def _backpropagate(node: SearchNode, reward: float) -> None:
    current: Optional[SearchNode] = node
    while current is not None:
        current.visits += 1
        current.value += reward
        current = current.parent


def _robust_path(root: SearchNode) -> List[int]:
    chosen: List[int] = []
    node = root
    while node.children:
        node = max(
            node.children,
            key=lambda child: (
                child.visits,
                child.value / child.visits if child.visits else float("-inf"),
                -child.chosen_indices[-1],
            ),
        )
        chosen = list(node.chosen_indices)
    return chosen


def _tree_diagnostics(root: SearchNode) -> Dict[str, Any]:
    stack = [root]
    states = set()
    node_count = 0
    max_depth = 0
    while stack:
        node = stack.pop()
        node_count += 1
        states.add(tuple(node.chosen_indices))
        max_depth = max(max_depth, len(node.chosen_indices))
        stack.extend(node.children)
    return {
        "node_count": node_count,
        "unique_state_count": len(states),
        "max_searched_depth": max_depth,
        "root_revisited_child_count": sum(
            1 for child in root.children if child.visits > 1
        ),
    }


def run_mcts(
    actions: List[Module5Action],
    iterations: int = 240,
    max_depth: int = 3,
    exploration: float = 1.1,
    seed: int = 7,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    root = SearchNode(
        chosen_indices=[],
        remaining_indices=list(range(len(actions))),
        untried_indices=list(range(len(actions))),
    )

    best_reward = float("-inf")
    best_indices: List[int] = []
    trace: List[Dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        rollout_source = _select_leaf(
            root, actions, max_depth, exploration, rng
        )
        reward, rollout_indices = _rollout(
            rollout_source, actions, max_depth, rng
        )
        _backpropagate(rollout_source, reward)

        if reward > best_reward:
            best_reward = reward
            best_indices = list(rollout_indices)

        if iteration <= 20 or iteration == iterations:
            trace.append(
                {
                    "iteration": iteration,
                    "reward": round(reward, 6),
                    "chosen_action_ids": [
                        actions[index].action_id for index in rollout_indices
                    ],
                }
            )

    robust_indices = _robust_path(root)
    if not robust_indices:
        robust_indices = best_indices
    best_actions = [actions[index] for index in robust_indices]
    robust_reward = _state_reward(best_actions)
    tree_diagnostics = _tree_diagnostics(root)
    root_stats = []
    for child in sorted(root.children, key=lambda item: item.visits, reverse=True):
        root_stats.append(
            {
                "action_id": actions[child.chosen_indices[-1]].action_id,
                "visits": child.visits,
                "mean_value": round(
                    child.value / child.visits if child.visits else 0.0, 6
                ),
            }
        )
    return {
        "best_actions": best_actions,
        "best_reward": round(robust_reward, 6),
        "search_trace": trace,
        "diagnostics": {
            "seed": seed,
            "iterations": iterations,
            **tree_diagnostics,
            "root_child_stats": root_stats,
            "robust_action_ids": [action.action_id for action in best_actions],
            "robust_reward_breakdown": _state_reward_breakdown(best_actions),
            "max_rollout_reward": round(
                best_reward if best_reward != float("-inf") else 0.0, 6
            ),
            "max_rollout_action_ids": [
                actions[index].action_id for index in best_indices
            ],
        },
    }


def build_module45_plan(
    module4_plan_payload: Dict[str, Any],
    iterations: int = 240,
    max_depth: int = 3,
    exploration: float = 1.1,
    seed: int = 7,
    candidate_limit: int = 48,
    memory_session: Any = None,
    behavioral_contract: Optional[Dict[str, Any]] = None,
    max_actions_per_transform: int = 2,
    min_distinct_transforms: int = 5,
) -> Dict[str, Any]:
    actions = build_action_space(module4_plan_payload, behavioral_contract)
    if not actions:
        # The contract filter must never starve the planner.  Fall back to the
        # unfiltered space and say so in the audit rather than return no plan.
        actions = build_action_space(module4_plan_payload, behavioral_contract=None)
        contract_filter_bypassed = bool(actions)
    else:
        contract_filter_bypassed = False
    searched_actions = _stratified_candidates(
        actions,
        max(1, candidate_limit),
        max_per_transform=max_actions_per_transform,
        min_distinct_transforms=min_distinct_transforms,
    )
    search = run_mcts(
        actions=searched_actions,
        iterations=iterations,
        max_depth=max_depth,
        exploration=exploration,
        seed=seed,
    )

    plan = Module45Plan(
        benchmark=module4_plan_payload["benchmark"],
        objective=module4_plan_payload["objective"],
        source_c_path=module4_plan_payload["source_c_path"],
        planner_name="module4_5_mcts_v2_progressive",
        iterations=iterations,
        max_depth=max_depth,
        harmful_blacklist=list(module4_plan_payload.get("harmful_blacklist", [])),
        action_space=actions,
        best_actions=search["best_actions"],
        search_trace=search["search_trace"],
    )
    payload = plan.to_dict()
    structural_filter = _structural_filter_audit(
        module4_plan_payload, behavioral_contract
    )
    structural_filter["contract_filter_bypassed"] = contract_filter_bypassed
    searched_transform_counts: Dict[str, int] = defaultdict(int)
    for action in searched_actions:
        searched_transform_counts[action.transform_name] += 1
    payload["search_summary"] = {
        "action_space_size": len(actions),
        "searched_action_space_size": len(searched_actions),
        "candidate_limit": candidate_limit,
        "candidate_selection": "stratified_by_transform",
        "max_actions_per_transform": max_actions_per_transform,
        "min_distinct_transforms": min_distinct_transforms,
        "searched_transform_counts": dict(
            sorted(searched_transform_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "searched_distinct_transform_count": len(searched_transform_counts),
        "best_reward": search["best_reward"],
        "best_action_ids": [action.action_id for action in search["best_actions"]],
        "diagnostics": search["diagnostics"],
        "structural_filter": structural_filter,
    }
    if memory_session is not None:
        memory_session.record_module45_plan(payload)
    return payload


def write_plan(output_path: str | Path, payload: Dict[str, Any]) -> None:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first Module 4.5 MCTS planner over a Module 4 plan JSON.")
    parser.add_argument("module4_plan_json", help="Path to Module 4 plan JSON")
    parser.add_argument("--iterations", type=int, default=240, help="Number of MCTS iterations")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum number of selected actions")
    parser.add_argument("--exploration", type=float, default=1.1, help="UCT exploration constant")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic random seed")
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=48,
        help="Maximum number of ranked actions admitted to the search tree",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_module45_plan(
        module4_plan_payload=_load_json(args.module4_plan_json),
        iterations=args.iterations,
        max_depth=args.max_depth,
        exploration=args.exploration,
        seed=args.seed,
        candidate_limit=args.candidate_limit,
    )
    if args.output:
        write_plan(args.output, payload)

    print(
        json.dumps(
            {
                "benchmark": payload["benchmark"],
                "objective": payload["objective"],
                "action_space_size": payload["search_summary"]["action_space_size"],
                "best_reward": payload["search_summary"]["best_reward"],
                "best_action_ids": payload["search_summary"]["best_action_ids"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
