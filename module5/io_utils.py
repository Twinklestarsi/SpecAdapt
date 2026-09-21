from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict

from rag_retrieve.schema import Module45Plan, Module5Action, Module5ActionConstraints


def _sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "item"


def action_to_work_key(action_id: str) -> str:
    return _sanitize_token(action_id.replace("::", "__"))


def ensure_clean_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _constraints_from_dict(data: Dict[str, Any]) -> Module5ActionConstraints:
    return Module5ActionConstraints(
        must_preserve_behavior=bool(data.get("must_preserve_behavior", True)),
        forbidden_transforms=list(data.get("forbidden_transforms", [])),
        conflict_group=str(data.get("conflict_group", "")),
        apply_scope=str(data.get("apply_scope", "local")),
        expected_risk=str(data.get("expected_risk", "medium")),
    )


def action_from_dict(data: Dict[str, Any]) -> Module5Action:
    return Module5Action(
        action_id=str(data["action_id"]),
        benchmark=str(data["benchmark"]),
        objective=str(data["objective"]),
        source_c_path=str(data["source_c_path"]),
        region_id=str(data["region_id"]),
        region_type=str(data["region_type"]),
        transform_name=str(data["transform_name"]),
        priority=float(data.get("priority", 0.0)),
        planning_score=float(data.get("planning_score", 0.0)),
        expected_metric_gain=float(data.get("expected_metric_gain", 0.0)),
        confidence=float(data.get("confidence", 0.0)),
        problem_hypothesis=str(data.get("problem_hypothesis", "")),
        execution_hint=str(data.get("execution_hint", "")),
        source_anchor=dict(data.get("source_anchor", {})),
        supporting_matches=list(data.get("supporting_matches", [])),
        supporting_benchmarks=list(data.get("supporting_benchmarks", [])),
        constraints=_constraints_from_dict(dict(data.get("constraints", {}))),
        memory_score=float(data.get("memory_score", 0.0)),
        memory_success_rate=float(data.get("memory_success_rate", 0.0)),
        memory_evidence_count=int(data.get("memory_evidence_count", 0)),
        memory_evidence=list(data.get("memory_evidence", [])),
        region_features=dict(data.get("region_features", {})),
    )


def load_plan(path: str | Path) -> Module45Plan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Module45Plan(
        benchmark=str(payload["benchmark"]),
        objective=str(payload["objective"]),
        source_c_path=str(payload["source_c_path"]),
        planner_name=str(payload["planner_name"]),
        iterations=int(payload.get("iterations", 0)),
        max_depth=int(payload.get("max_depth", 0)),
        harmful_blacklist=list(payload.get("harmful_blacklist", [])),
        action_space=[action_from_dict(item) for item in payload.get("action_space", [])],
        best_actions=[action_from_dict(item) for item in payload.get("best_actions", [])],
        search_trace=list(payload.get("search_trace", [])),
    )


def load_best_action(path: str | Path, action_index: int = 0) -> Module5Action:
    plan = load_plan(path)
    if not plan.best_actions:
        raise ValueError(f"No best_actions found in plan: {path}")
    if action_index < 0 or action_index >= len(plan.best_actions):
        raise IndexError(
            f"action_index {action_index} out of range for {len(plan.best_actions)} best_actions"
        )
    return plan.best_actions[action_index]


def load_best_actions(
    path: str | Path, *, max_actions: int | None = None
) -> list[Module5Action]:
    """Load all best_actions from a plan, optionally truncated."""
    plan = load_plan(path)
    if not plan.best_actions:
        raise ValueError(f"No best_actions found in plan: {path}")
    actions = list(plan.best_actions)
    if max_actions is not None and max_actions > 0:
        actions = actions[:max_actions]
    return actions
