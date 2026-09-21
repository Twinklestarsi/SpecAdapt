"""
agent.py — MemoryAgent class: the main interface to Module 8.

Wraps MemoryStore (I/O) and the rule pipeline (derive → compile → inject).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory_agent.store import MemoryStore
from memory_agent.rule_deriver import derive_path_rules
from memory_agent.experience import (
    OptimizationDecisionRecord,
    CorrectionPatternRecord,
    CollaborationTrajectoryRecord,
)
from memory_agent.ids import new_id, stable_id
from memory_agent.ingest import (
    import_c_generation,
    import_mcts_plan,
    import_module5,
    import_path_decisions,
    import_rag_index,
    import_spec_analysis,
)
from memory_agent.retrieval import MemoryRetriever
from memory_agent.policy import derive_ppa_path_rules
from memory_agent.policy_refiner import PolicyRefiner, SOFT_PATH_POLICIES
from memory_agent.schemas import (
    ActionRecord,
    EvaluationRecord,
    FailureRecord,
    FeatureRecord,
    RunRecord,
    SelectionRecord,
    TaskRecord,
    now_iso,
)
from memory_agent.sqlite_store import SQLiteMemoryStore
from memory_agent.validators import normalize_objective, normalize_path


class MemoryAgent:
    """
    Module 8: Memory Agent.

    Provides three services:
      1. Rule derivation  — analyze decisions + PPA → write new rules
      2. Outcome recording — accept PPA, transform, correction records
      3. Diagnostics      — status, dump rules
    """

    def __init__(
        self,
        store_path: str | Path = "path_decisions_log.json",
        db_path: str | Path | None = None,
    ) -> None:
        self.store = MemoryStore(store_path)
        legacy_path = Path(store_path)
        resolved_db_path = (
            Path(db_path)
            if db_path is not None
            else self._default_db_path(legacy_path)
        )
        self.db = SQLiteMemoryStore(resolved_db_path)
        self._ensure_seed_path_policies()
        self.retriever = MemoryRetriever(self.db)

    @staticmethod
    def _default_db_path(store_path: Path) -> Path:
        configured = os.environ.get("MEMORY_AGENT_DB_PATH")
        if configured:
            return Path(configured).expanduser()
        state_root = Path(
            os.environ.get(
                "XDG_STATE_HOME",
                str(Path.home() / ".local" / "state"),
            )
        )
        project_key = stable_id("project", str(store_path.resolve()))[-16:]
        return state_root / "memory_agent" / f"{project_key}.db"

    def _ensure_seed_path_policies(self) -> None:
        existing = {
            (row["policy_id"], row["objective"])
            for row in self.db.path_policy_rows(enabled_only=False)
        }
        for objective in ("AREA", "TIMING"):
            for policy_id, definition in SOFT_PATH_POLICIES.items():
                if (policy_id, objective) in existing:
                    continue
                self.db.upsert_path_policy(
                    {
                        "policy_id": policy_id,
                        "objective": objective,
                        "conditions": definition["conditions"],
                        "path_scores": definition["seed_scores"],
                        "confidence": 0.25,
                        "evidence_count": 0,
                        "evidence": {},
                        "reason": definition["description"],
                        "source": "seed_policy",
                        "enabled": True,
                    },
                    model="",
                    token_usage={},
                    updated_at=now_iso(),
                )

    # ── Rule pipeline ─────────────────────────────────────────────────

    def derive_path_rules(
        self,
        min_evidence: int = 3,
        win_threshold: float = 0.70,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Analyze accumulated decisions + PPA outcomes and write derived rules
        back to the staging file.

        Args:
            min_evidence:   Minimum completed records per feature group.
            win_threshold:  Minimum win rate to emit a rule (0–1).
            dry_run:        If True, return rules without writing to store.

        Returns:
            List of derived condition-dicts (may be empty at bootstrap).
        """
        decisions = self.store.get_decisions()
        existing  = self.store.get_path_rules()
        rules = derive_path_rules(
            decisions,
            existing_rules=existing,
            min_evidence=min_evidence,
            win_threshold=win_threshold,
        )
        if not dry_run:
            self.store.set_path_rules(rules)
        return rules

    def get_path_selection_rules(self) -> List[Dict[str, Any]]:
        """Return current learned rules from the staging file."""
        return self.store.get_path_rules()

    # ── PPA outcome recording (Module 7 → Module 8) ──────────────────

    def record_ppa_outcome(
        self,
        benchmark: str,
        path: str,
        area_improvement_pct: Optional[float] = None,
        timing_improvement_pct: Optional[float] = None,
        slack_status: Optional[str] = None,
        cell_count_delta: Optional[int] = None,
        synthesis_failed: bool = False,
        notes: str = "",
        timing_improvement_ps: Optional[float] = None,
    ) -> None:
        """
        Record a PPA outcome from Module 7.

        Also back-fills the ppa_outcome field in the matching decisions record
        so the Memory Agent can correlate path → PPA in one query.
        """
        self.store.record_ppa_outcome(
            benchmark=benchmark,
            path=path,
            area_improvement_pct=area_improvement_pct,
            timing_improvement_ps=timing_improvement_ps,
            timing_improvement_pct=timing_improvement_pct,
            slack_status=slack_status,
            cell_count_delta=cell_count_delta,
            synthesis_failed=synthesis_failed,
            notes=notes,
        )

    # ── Optimization decisions (Modules 3/5/6 → Module 8) ───────────

    def record_optimization_decision(
        self, record: OptimizationDecisionRecord
    ) -> None:
        self.store.append_optimization_decision(record.to_dict())

    # ── Correction patterns (Modules 4/5/6/7 → Module 8) ────────────

    def record_correction(self, record: CorrectionPatternRecord) -> None:
        self.store.append_correction(record.to_dict())

    # ── Collaboration trajectories (orchestrator → Module 8) ─────────

    def record_trajectory(self, record: CollaborationTrajectoryRecord) -> None:
        self.store.append_trajectory(record.to_dict())

    # ── Canonical cross-agent records ──────────────────────────────────

    def create_task(
        self,
        benchmark: str,
        spec_text: str = "",
        source_path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        task_id = stable_id("task", benchmark)
        self.db.upsert_task(TaskRecord(
            task_id=task_id,
            benchmark=benchmark,
            spec_text=spec_text,
            source_path=source_path,
            metadata=metadata or {},
        ))
        return task_id

    def start_run(
        self,
        task_id: str,
        objective: str,
        path: str = "",
        producer: str = "orchestrator",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        run_id = new_id("run")
        self.db.upsert_run(RunRecord(
            run_id=run_id,
            task_id=task_id,
            objective=normalize_objective(objective),
            path=normalize_path(path),
            producer=producer,
            metadata=metadata or {},
        ))
        return run_id

    def record_feature(self, record: FeatureRecord) -> None:
        self.db.upsert_feature(record)

    def record_selection(self, record: SelectionRecord) -> None:
        self.db.upsert_selection(record)

    def record_action(self, record: ActionRecord) -> None:
        self.db.upsert_action(record)

    def record_evaluation(self, record: EvaluationRecord) -> None:
        self.db.upsert_evaluation(record)

    def record_failure(self, record: FailureRecord) -> None:
        self.db.upsert_failure(record)

    # ── Historical import ──────────────────────────────────────────────

    def import_spec_analysis(self, path: str | Path) -> Dict[str, int]:
        return import_spec_analysis(self.db, path)

    def import_path_decisions(self, path: str | Path) -> Dict[str, int]:
        return import_path_decisions(self.db, path)

    def import_rag_index(self, path: str | Path) -> Dict[str, int]:
        return import_rag_index(self.db, path)

    def import_module5(self, path: str | Path) -> Dict[str, int]:
        return import_module5(self.db, path)

    def import_c_generation(self, path: str | Path) -> Dict[str, int]:
        return import_c_generation(self.db, path)

    def import_mcts_plan(self, path: str | Path) -> Dict[str, int]:
        return import_mcts_plan(self.db, path)

    def record_path_decision_payload(self, item: Dict[str, Any]) -> str:
        benchmark = str(item.get("benchmark") or "unknown")
        timestamp = str(item.get("timestamp") or "")
        objective = normalize_objective(item.get("optimization_target", ""))
        path = normalize_path(item.get("path", ""))
        task_id = self.create_task(benchmark)
        run_id = stable_id("run", benchmark, timestamp, path, objective)
        self.db.upsert_run(RunRecord(
            run_id=run_id,
            task_id=task_id,
            objective=objective,
            path=path,
            status="selected",
            producer="path_selector",
        ))
        feature_id = stable_id("feature", run_id, "path_decision")
        self.db.upsert_feature(FeatureRecord(
            record_id=feature_id,
            task_id=task_id,
            run_id=run_id,
            producer="path_selector",
            timestamp=timestamp or now_iso(),
            features=dict(item.get("feature_profile") or {}),
            optimization_target=objective,
            source_path=str(self.store.path),
        ))
        self.db.upsert_selection(SelectionRecord(
            record_id=stable_id("selection", run_id),
            task_id=task_id,
            run_id=run_id,
            parent_id=feature_id,
            producer="path_selector",
            timestamp=timestamp or now_iso(),
            path=path,
            rule_fired=str(item.get("rule_fired") or ""),
            confidence=str(item.get("confidence") or ""),
            tier=int(item.get("tier") or 0),
            reason=str(item.get("reason") or ""),
            token_usage=dict(item.get("token_usage") or {}),
        ))
        return run_id

    def derive_ppa_path_rules(
        self,
        min_evidence_per_path: int = 2,
        min_gain_margin: float = 0.0,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        rules = derive_ppa_path_rules(
            self.db,
            min_evidence_per_path=min_evidence_per_path,
            min_gain_margin=min_gain_margin,
        )
        if not dry_run:
            self.store.set_path_rules(rules)
        return rules

    def refine_path_policies(
        self,
        *,
        objectives: List[str] | tuple[str, ...] = ("AREA", "TIMING"),
        env_path: str | Path = ".env",
        dry_run: bool = False,
        client: Any = None,
        model: str = "",
    ) -> Dict[str, Any]:
        """Use an LLM to refine the five soft path policies from PPA statistics."""
        refiner = PolicyRefiner(
            self.db,
            env_path=env_path,
            client=client,
            model=model,
        )
        result = refiner.refine(objectives=objectives)
        if not dry_run:
            for policy in result["policies"]:
                self.db.upsert_path_policy(
                    policy,
                    model=result["model"],
                    token_usage=result["token_usage"],
                    updated_at=now_iso(),
                )
        return result

    def get_path_policies(
        self,
        objective: str = "",
        *,
        enabled_only: bool = True,
    ) -> List[Dict[str, Any]]:
        return self.db.path_policy_rows(
            normalize_objective(objective) if objective else "",
            enabled_only=enabled_only,
        )

    # ── Retrieval and guidance ─────────────────────────────────────────

    def retrieve_similar_tasks(
        self,
        features: Dict[str, Any],
        objective: str = "",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        return self.retriever.retrieve_similar_tasks(
            features, objective=objective, top_k=top_k
        )

    def recommend_path(
        self,
        features: Dict[str, Any],
        objective: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        return self.retriever.recommend_path(features, objective, top_k=top_k)

    def rank_actions(
        self,
        region_features: Dict[str, Any],
        objective: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        return self.retriever.rank_actions(
            region_features, objective, top_k=top_k
        )

    def retrieve_failure_guidance(
        self,
        context: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        return self.retriever.retrieve_failure_guidance(context, top_k=top_k)

    def get_run_experience(self, run_id: str) -> Dict[str, Any]:
        return self.db.get_run_experience(run_id)

    # ── Diagnostics ───────────────────────────────────────────────────

    def status(self) -> Dict[str, int]:
        """Return record counts per category."""
        status = self.store.status()
        status.update({f"db_{key}": value for key, value in self.db.status().items()})
        return status

    def dump_rules(self) -> str:
        """Return current rules as formatted JSON string."""
        rules = self.get_path_selection_rules()
        return json.dumps(rules, indent=2)

    def summary(self) -> str:
        """Human-readable status summary."""
        s = self.status()
        lines = [
            "Module 8 Memory Agent — Store Summary",
            f"  path_selection_rules : {s['path_rules']}",
            f"  decisions (M2)       : {s['decisions']}",
            f"  ppa_outcomes (M7)    : {s['ppa_outcomes']}",
            f"  opt_decisions (M3/5/6): {s['opt_decisions']}",
            f"  corrections (M4-7)   : {s['corrections']}",
            f"  trajectories (orch.) : {s['trajectories']}",
            "  -- structured SQLite memory --",
            f"  tasks                 : {s['db_tasks']}",
            f"  runs                  : {s['db_runs']}",
            f"  features              : {s['db_features']}",
            f"  selections            : {s['db_selections']}",
            f"  actions               : {s['db_actions']}",
            f"  evaluations           : {s['db_evaluations']}",
            f"  failures              : {s['db_failures']}",
            f"  artifacts             : {s['db_artifacts']}",
            f"  path policies         : {s['db_path_policies']}",
        ]
        # Completed (decisions with ppa_outcome)
        decisions = self.store.get_decisions()
        completed = sum(1 for d in decisions if d.get("ppa_outcome") is not None)
        lines.append(f"  decisions completed  : {completed} / {s['decisions']}")
        return "\n".join(lines)
