from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from memory_agent.ingest import (
    import_module5,
    import_path_decisions,
    import_rag_index,
    import_spec_analysis,
)
from memory_agent.retrieval import MemoryRetriever, feature_similarity
from memory_agent.policy import derive_ppa_path_rules
from memory_agent.sqlite_store import SQLiteMemoryStore
from memory_agent import EvaluationRecord, FeatureRecord, MemoryAgent, MemorySession
from memory_agent.ids import stable_id
from memory_agent.policy_refiner import build_policy_statistics


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_feature_similarity_is_objective_aware() -> None:
    features = {
        "architecture_pattern": "fsm",
        "is_sequential": True,
        "num_clock_domains": 1,
        "has_fsm": True,
        "complexity": "moderate",
    }
    same = feature_similarity(features, features, "AREA", "AREA")
    other_objective = feature_similarity(features, features, "AREA", "TIMING")
    assert same["score"] == 1.0
    assert same["score"] > other_objective["score"]
    sparse = feature_similarity(features, {}, "AREA", "AREA")
    assert sparse["score"] < 0.2


def test_imports_are_idempotent_and_retrievable(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    spec_path = _write(tmp_path / "spec.json", [{
        "benchmark": "demo",
        "optimization_target": "AREA",
        "spec_text": "A single clock arithmetic FSM.",
        "llm_features": {
            "architecture_pattern": "fsm",
            "suggested_subcategory": "arithmetic",
            "is_sequential": True,
            "num_clock_domains": 1,
            "has_fsm": True,
            "complexity": "moderate",
            "data_widths": [32],
        },
    }])
    path_path = _write(tmp_path / "paths.json", {
        "decisions": [{
            "benchmark": "demo",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "optimization_target": "AREA",
            "feature_profile": {
                "architecture_pattern": "fsm",
                "suggested_subcategory": "arithmetic",
                "is_sequential": True,
                "num_clock_domains": 1,
                "has_fsm": True,
                "complexity": "moderate",
                "data_widths": [32],
            },
            "path": "c_first",
            "rule_fired": "fsm_behavior",
            "confidence": "medium",
            "tier": 1,
            "ppa_outcome": {
                "area_improvement_pct": -12.0,
                "slack_status": "MET",
                "synthesis_failed": False,
            },
        }]
    })

    import_spec_analysis(store, spec_path)
    import_path_decisions(store, path_path)
    first = store.status()
    import_spec_analysis(store, spec_path)
    import_path_decisions(store, path_path)
    assert store.status() == first

    retriever = MemoryRetriever(store)
    result = retriever.retrieve_similar_tasks({
        "architecture_pattern": "fsm",
        "suggested_subcategory": "arithmetic",
        "is_sequential": True,
        "num_clock_domains": 1,
        "has_fsm": True,
        "complexity": "moderate",
        "data_widths": [32],
    }, "AREA")
    assert result[0]["benchmark"] == "demo"
    assert result[0]["similarity"]["score"] == 1.0
    assert any(item["area_gain_pct"] == 12.0 for item in result[0]["evidence"])


def test_rag_actions_can_be_ranked(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    rag_path = _write(tmp_path / "rag.json", {
        # import_rag_index refuses indices older than schema 4: those copied a
        # benchmark's PPA numbers onto every one of its regions.
        "summary": {"schema_version": 4},
        "entries": [{
            "objective": "area",
            "subcategory": "arithmetic",
            "benchmark": "history",
            "graph_function": "main",
            "region": {
                "region_id": "main_region_1",
                "region_type": "arith_region",
                "features": {
                    "node_count": 5,
                    "edge_count": 4,
                    "graph_depth": 3,
                },
            },
            "top_transforms": [{
                "transform_name": "RESOURCE_SHARE",
                "area_improvement_pct": 20.0,
                "slack_improvement_ps": 5.0,
                "slack_status": "MET",
            }],
        }]
    })
    import_rag_index(store, rag_path)
    ranked = MemoryRetriever(store).rank_actions({
        "region_type": "arith_region",
        "node_count": 5,
        "edge_count": 4,
        "graph_depth": 3,
    }, "AREA")
    assert ranked[0]["transform_name"] == "RESOURCE_SHARE"
    assert ranked[0]["success_rate"] == 1.0
    assert ranked[0]["score"] > 0.5


def test_module5_failure_is_recorded(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    run_dir = tmp_path / "runs" / "demo" / "action"
    run_dir.mkdir(parents=True)
    _write(run_dir / "action.json", {
        "action_id": "demo_region_1::COMMON_SUBEXPR_EXTRACT",
        "benchmark": "demo",
        "objective": "area",
        "region_id": "demo_region_1",
        "region_type": "select_region",
        "transform_name": "COMMON_SUBEXPR_EXTRACT",
    })
    _write(run_dir / "result.json", {
        "action_id": "demo_region_1::COMMON_SUBEXPR_EXTRACT",
        "benchmark": "demo",
        "objective": "area",
        "backend": "direct_rtl",
        "status": "rtl_failed",
        "behavior_check_status": "not_run",
        "metrics": {},
        "notes": ["Verification failed"],
        "changed_regions": ["demo_region_1"],
    })
    import_module5(store, tmp_path / "runs")
    failures = store.failure_rows()
    assert failures[0]["failure_type"] == "rtl_failed"
    evaluation = store.rows("SELECT * FROM evaluations")[0]
    assert evaluation["correctness_status"] == "failed"
    assert evaluation["synthesis_status"] == "failed"
    run = store.rows("SELECT path FROM runs")[0]
    assert run["path"] == "c_first"


def test_single_path_history_cannot_publish_comparative_rules(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    rag_path = _write(tmp_path / "rag.json", {
        # import_rag_index refuses indices older than schema 4: those copied a
        # benchmark's PPA numbers onto every one of its regions.
        "summary": {"schema_version": 4},
        "entries": [{
            "objective": "area",
            "benchmark": "history",
            "region": {
                "region_id": "region_1",
                "region_type": "arith_region",
                "features": {"node_count": 3},
            },
            "top_transforms": [{
                "transform_name": "RESOURCE_SHARE",
                "area_improvement_pct": 30.0,
            }],
        }]
    })
    import_rag_index(store, rag_path)
    assert derive_ppa_path_rules(store) == []


def test_hard_path_rules_precede_learned_rules() -> None:
    from path_select.rules import RuleSet, RuleVerdict

    def learned_rule(_features):
        return RuleVerdict(
            path="c_first",
            rule_fired="learned",
            reason="learned",
            confidence="high",
        )

    rules = RuleSet()
    rules.load_from_memory([learned_rule])
    verdict = rules.apply({"num_clock_domains": 2})
    assert verdict.path == "rtl_direct"
    assert verdict.rule_fired == "multi_clock_protection"


def test_path_selector_writes_structured_memory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from path_select.selector import PathSelector

    db_path = tmp_path / "memory.db"
    log_path = tmp_path / "path_decisions.json"
    monkeypatch.setenv("MEMORY_AGENT_DB_PATH", str(db_path))
    selector = PathSelector(memory_path=log_path)
    decision = selector.select({
        "benchmark": "multi_clock_demo",
        "optimization_target": "AREA",
        "llm_features": {
            "architecture_pattern": "mixed",
            "is_sequential": True,
            "num_clock_domains": 2,
            "complexity": "complex",
        },
        "overall_confidence": "high",
    }, use_llm=False)
    assert decision.path == "rtl_direct"
    status = SQLiteMemoryStore(db_path).status()
    assert status["features"] == 1
    assert status["selections"] == 1


def test_runtime_session_links_module_records(tmp_path: Path) -> None:
    agent = MemoryAgent(tmp_path / "paths.json", db_path=tmp_path / "memory.db")
    session = MemorySession.start(
        agent,
        benchmark="runtime_demo",
        objective="AREA",
        spec_text="A single-clock arithmetic datapath.",
    )
    session.record_feature_result({
        "benchmark": "runtime_demo",
        "llm_features": {
            "architecture_pattern": "datapath",
            "is_sequential": True,
            "num_clock_domains": 1,
        },
        "confidence": {"architecture_pattern": "high"},
        "overall_confidence": "high",
    })
    session.record_path_decision({
        "path": "c_first",
        "reason": "test",
        "rule_fired": "test_rule",
        "confidence": "high",
        "tier": 1,
        "token_usage": {},
    }, {
        "architecture_pattern": "datapath",
        "is_sequential": True,
        "num_clock_domains": 1,
    })
    session.record_c_generation({
        "benchmark": "runtime_demo",
        "c_path": str(tmp_path / "missing.c"),
        "method": "llm_spec",
        "success": False,
        "error": "generation failed",
        "token_usage": {},
    }, {"input_type": "spec"})
    session.complete("failed")

    experience = agent.get_run_experience(session.context.run_id)
    assert len(experience["features"]) == 1
    assert len(experience["selections"]) == 1
    assert len(experience["actions"]) == 1
    assert len(experience["failures"]) == 1
    assert len(experience["trajectories"]) == 1


def test_path_selector_uses_live_ppa_memory(tmp_path: Path) -> None:
    agent = MemoryAgent(tmp_path / "paths.json", db_path=tmp_path / "memory.db")
    features = {
        "architecture_pattern": "mixed",
        "suggested_subcategory": "bit_reorganization",
        "is_sequential": False,
        "num_clock_domains": 1,
        "has_fsm": False,
        "complexity": "simple",
        "data_widths": [32],
    }
    for index in range(3):
        task_id = agent.create_task(f"history_{index}")
        run_id = agent.start_run(task_id, "AREA", path="c_first")
        agent.record_feature(FeatureRecord(
            record_id=stable_id("feature", run_id),
            task_id=task_id,
            run_id=run_id,
            producer="test",
            features=features,
            optimization_target="AREA",
        ))
        agent.record_evaluation(EvaluationRecord(
            record_id=stable_id("evaluation", run_id),
            task_id=task_id,
            run_id=run_id,
            producer="test",
            objective="AREA",
            path="c_first",
            correctness_status="passed",
            synthesis_status="success",
            area_gain_pct=10.0 + index,
        ))

    session = MemorySession.start(agent, benchmark="query", objective="AREA")
    from path_select.selector import PathSelector

    decision = PathSelector(memory_session=session).select({
        "benchmark": "query",
        "optimization_target": "AREA",
        "llm_features": features,
        "overall_confidence": "high",
    }, use_llm=False)
    assert decision.path == "c_first"
    assert decision.rule_fired == "memory_guidance_no_llm"
    assert decision.memory_evidence["has_sufficient_evidence"] is True


def test_module5_feedback_updates_action_and_evaluation(tmp_path: Path) -> None:
    agent = MemoryAgent(tmp_path / "paths.json", db_path=tmp_path / "memory.db")
    session = MemorySession.start(
        agent,
        benchmark="feedback_demo",
        objective="AREA",
    )
    action = {
        "action_id": "r1::RESOURCE_SHARE",
        "region_id": "r1",
        "region_type": "arith_region",
        "transform_name": "RESOURCE_SHARE",
        "planning_score": 0.8,
        "expected_metric_gain": 10.0,
        "confidence": 0.9,
        "region_features": {"node_count": 5, "edge_count": 4},
    }
    session.record_module45_plan({
        "action_space": [action],
        "best_actions": [action],
    })
    session.record_module5_result({
        "action_id": action["action_id"],
        "status": "success",
        "backend": "hls",
        "work_dir": str(tmp_path / "run"),
        "behavior_check_status": "not_run",
        "metrics": {
            "total_cell_area": 90.0,
            "baseline_total_cell_area": 100.0,
            "data_arrival_time_ps": 80.0,
            "baseline_data_arrival_time_ps": 100.0,
            "slack_status": "MET",
        },
        "objective_feedback": {"reward": 10.0, "improved": True},
    }, [action])

    action_row = agent.db.action_rows("AREA")[0]
    assert action_row["applied_successfully"] == 1
    assert action_row["outcome"]["area_gain_pct"] == 10.0
    evaluation = agent.db.rows("SELECT * FROM evaluations")[0]
    assert evaluation["area_gain_pct"] == 10.0
    ranked = agent.rank_actions({
        "region_type": "arith_region",
        "node_count": 5,
        "edge_count": 4,
    }, "AREA")
    assert ranked[0]["transform_name"] == "RESOURCE_SHARE"
    assert ranked[0]["success_rate"] == 1.0


def test_module5_feedback_records_each_action_application_result(
    tmp_path: Path,
) -> None:
    agent = MemoryAgent(tmp_path / "paths.json", db_path=tmp_path / "memory.db")
    session = MemorySession.start(
        agent,
        benchmark="partial_feedback_demo",
        objective="AREA",
    )
    actions = [
        {
            "action_id": "r1::SERIALIZE_PARALLELISM",
            "region_id": "r1",
            "region_type": "control_region",
            "transform_name": "SERIALIZE_PARALLELISM",
        },
        {
            "action_id": "r2::COMMON_SUBEXPR_EXTRACT",
            "region_id": "r2",
            "region_type": "arith_region",
            "transform_name": "COMMON_SUBEXPR_EXTRACT",
        },
        {
            "action_id": "r3::LOGIC_MINIMIZATION",
            "region_id": "r3",
            "region_type": "logic_region",
            "transform_name": "LOGIC_MINIMIZATION",
        },
    ]
    session.record_module5_result({
        "status": "success",
        "backend": "direct_rtl",
        "work_dir": str(tmp_path / "run"),
        "per_action_summaries": [
            {
                "applied": False,
                "region": "r1",
                "summary": "No parallel operations found.",
            },
            {
                "applied": False,
                "region": "r2",
                "summary": "No repeated subexpressions found.",
            },
            {
                "applied": True,
                "region": "r3",
                "summary": "Simplified boolean logic.",
            },
        ],
        "metrics": {
            "total_cell_area": 95.0,
            "baseline_total_cell_area": 100.0,
        },
        "objective_feedback": {"reward": 5.0, "improved": True},
    }, actions)

    rows = {
        row["transform_name"]: row
        for row in agent.db.action_rows("AREA")
    }
    assert rows["SERIALIZE_PARALLELISM"]["applied_successfully"] == 0
    assert rows["COMMON_SUBEXPR_EXTRACT"]["applied_successfully"] == 0
    assert rows["LOGIC_MINIMIZATION"]["applied_successfully"] == 1
    assert rows["LOGIC_MINIMIZATION"]["outcome"]["action_applied"] is True
    assert rows["LOGIC_MINIMIZATION"]["outcome"]["reward"] is None
    assert rows["LOGIC_MINIMIZATION"]["outcome"]["sequence_reward"] == 5.0
    assert rows["LOGIC_MINIMIZATION"]["outcome"]["area_gain_pct"] is None
    assert (
        rows["SERIALIZE_PARALLELISM"]["context"]["application"]["summary"]
        == "No parallel operations found."
    )


def test_module5_feedback_marks_applied_action_failed_when_sequence_fails(
    tmp_path: Path,
) -> None:
    agent = MemoryAgent(tmp_path / "paths.json", db_path=tmp_path / "memory.db")
    session = MemorySession.start(
        agent,
        benchmark="failed_feedback_demo",
        objective="AREA",
    )
    action = {
        "action_id": "r1::LOGIC_MINIMIZATION",
        "region_id": "r1",
        "region_type": "logic_region",
        "transform_name": "LOGIC_MINIMIZATION",
    }
    session.record_module5_result({
        "status": "dc_failed",
        "backend": "direct_rtl",
        "work_dir": str(tmp_path / "run"),
        "per_action_summaries": [{
            "applied": True,
            "region": "r1",
            "summary": "Simplified boolean logic.",
        }],
    }, [action])

    row = agent.db.action_rows("AREA")[0]
    assert row["applied_successfully"] == 0
    assert row["outcome"]["action_applied"] is True
    assert row["outcome"]["action_execution_succeeded"] is False
    assert row["outcome"]["sequence_status"] == "dc_failed"


def test_llm_policy_refiner_persists_and_influences_path_guidance(
    tmp_path: Path,
) -> None:
    agent = MemoryAgent(tmp_path / "paths.json", db_path=tmp_path / "memory.db")
    features = {
        "architecture_pattern": "datapath",
        "suggested_subcategory": "arithmetic",
        "is_sequential": True,
        "num_clock_domains": 1,
        "has_fsm": False,
        "complexity": "moderate",
        "key_operations": ["arithmetic", "mux"],
    }
    samples = [
        ("c_first", 1.0),
        ("c_first", 2.0),
        ("rtl_direct", 6.0),
        ("rtl_direct", 7.0),
    ]
    for index, (path, gain) in enumerate(samples):
        task_id = agent.create_task(f"policy_sample_{index}")
        run_id = agent.start_run(task_id, "AREA", path=path)
        agent.record_feature(FeatureRecord(
            record_id=stable_id("feature", run_id),
            task_id=task_id,
            run_id=run_id,
            producer="test",
            features=features,
            optimization_target="AREA",
        ))
        agent.record_evaluation(EvaluationRecord(
            record_id=stable_id("evaluation", run_id),
            task_id=task_id,
            run_id=run_id,
            producer="test",
            objective="AREA",
            path=path,
            correctness_status="passed",
            synthesis_status="success",
            area_gain_pct=gain,
        ))

    statistics = build_policy_statistics(agent.db, ("AREA",))
    data_processing = next(
        item for item in statistics
        if item["policy_id"] == "data_processing"
    )
    assert data_processing["evidence_count"] == 4

    class FakeUsage:
        prompt_tokens = 100
        completion_tokens = 30
        total_tokens = 130

    class FakeMessage:
        content = json.dumps({
            "policies": [{
                "policy_id": "data_processing",
                "objective": "AREA",
                "path_scores": {
                    "c_first": 0.25,
                    "rtl_direct": 0.75,
                },
                "confidence": 0.8,
                "reason": "RTL-direct has higher observed area gain.",
            }]
        })

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

    class FakeCompletions:
        def create(self, **_kwargs):
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    result = agent.refine_path_policies(
        objectives=("AREA",),
        client=FakeClient(),
        model="test-policy-model",
    )
    assert result["token_usage"]["total_tokens"] == 130
    policies = agent.get_path_policies("AREA")
    assert len(policies) == 5
    data_policy = next(
        item for item in policies
        if item["policy_id"] == "data_processing"
    )
    assert data_policy["path_scores"]["rtl_direct"] == 0.75

    guidance = agent.recommend_path(features, "AREA")
    assert guidance["recommended_path"] == "rtl_direct"
    assert guidance["has_sufficient_evidence"] is True
    assert guidance["matched_policies"][0]["policy_id"] == "data_processing"
    assert guidance["decision_source"] == "policy_and_ppa_history"


def test_module2_llm_receives_memory_policy_guidance(tmp_path: Path) -> None:
    agent = MemoryAgent(tmp_path / "paths.json", db_path=tmp_path / "memory.db")
    assert len(agent.get_path_policies("AREA")) == 5
    assert len(agent.get_path_policies("TIMING")) == 5
    session = MemorySession.start(agent, benchmark="guided", objective="TIMING")

    from path_select.selector import PathSelector

    selector = PathSelector(memory_session=session)
    captured = {}

    def fake_tier2(objective, features, guidance, max_retries=2):
        captured["objective"] = objective
        captured["features"] = features
        captured["guidance"] = guidance
        from token_counter import TokenUsage
        return "rtl_direct", "Memory timing guidance favors RTL.", TokenUsage()

    selector._tier2_llm = fake_tier2
    decision = selector.select({
        "benchmark": "guided",
        "optimization_target": "TIMING",
        "llm_features": {
            "architecture_pattern": "datapath",
            "suggested_subcategory": "arithmetic",
            "is_sequential": True,
            "num_clock_domains": 1,
            "complexity": "moderate",
            "key_operations": ["arithmetic"],
        },
        "overall_confidence": "high",
    }, use_llm=True)

    assert decision.path == "rtl_direct"
    assert decision.rule_fired == "memory_guided_llm"
    assert captured["objective"] == "TIMING"
    assert captured["guidance"]["matched_policies"]
    assert captured["guidance"]["policy_scores"]


def test_shared_spec_loader_reads_json_list(tmp_path: Path) -> None:
    from spec_analyze.spec_loader import load_spec_entries

    spec_path = _write(tmp_path / "specs.json", [
        {"id": "first", "spec": "First specification."},
        {
            "id": "second",
            "spec": "Second specification.",
            "optimization_target": "timing",
        },
    ])
    entries = load_spec_entries(spec_path)
    assert [item["id"] for item in entries] == ["first", "second"]
    assert entries[0]["spec"] == "First specification."
    assert entries[1]["optimization_target"] == "TIMING"


def test_pipeline_spec_file_runs_each_entry(tmp_path: Path, capsys) -> None:
    from pipeline.__main__ import _run_spec_file

    spec_path = _write(tmp_path / "specs.json", [
        {"id": "first", "spec": "First specification."},
        {
            "id": "second",
            "spec": "Second specification.",
            "optimization_target": "timing",
        },
    ])

    @dataclass
    class FakeResult:
        benchmark: str
        objective: str
        status: str = "success"

        def to_dict(self):
            return {
                "benchmark": self.benchmark,
                "objective": self.objective,
                "status": self.status,
            }

    class FakeOrchestrator:
        def __init__(self):
            self.calls = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            return FakeResult(
                benchmark=kwargs["benchmark"],
                objective=str(kwargs["objective"]).upper(),
            )

    args = Namespace(
        spec_file=str(spec_path),
        benchmark="",
        objective="area",
        delay=0.0,
        no_path_llm=True,
        module4_top_k=5,
        mcts_iterations=5,
        mcts_max_depth=2,
        backend="direct_rtl",
        single_action=True,
        max_actions=1,
        no_baseline=True,
        module5_output_root=None,
        rtl_max_retries=0,
        dc_max_retries=0,
    )
    orchestrator = FakeOrchestrator()
    assert _run_spec_file(orchestrator, args) == 0
    assert [call["benchmark"] for call in orchestrator.calls] == [
        "first",
        "second",
    ]
    assert [call["objective"] for call in orchestrator.calls] == [
        "area",
        "timing",
    ]
    summary = capsys.readouterr().out
    assert '"total": 2' in summary
    assert '"failed": 0' in summary
