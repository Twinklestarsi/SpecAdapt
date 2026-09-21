"""
CLI entry point for Module 8 Memory Agent.

Usage:
    python -m memory_agent status
    python -m memory_agent derive-rules
    python -m memory_agent derive-rules --dry-run
    python -m memory_agent derive-rules --min-evidence 2 --win-threshold 0.65
    python -m memory_agent dump-rules
    python -m memory_agent record-ppa --benchmark uart_tx --path rtl_direct
                                      --area -8.3 --slack MET
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_store() -> Path:
    """Auto-detect path_decisions_log.json in project root."""
    here = Path(__file__).resolve().parent
    for candidate in [here.parent / "path_decisions_log.json"]:
        if candidate.exists():
            return candidate
    # Default: project root
    return here.parent / "path_decisions_log.json"


def cmd_status(agent, args) -> None:
    print(agent.summary())


def cmd_derive_rules(agent, args) -> None:
    rules = agent.derive_path_rules(
        min_evidence=args.min_evidence,
        win_threshold=args.win_threshold,
        dry_run=args.dry_run,
    )
    if not rules:
        print("No rules derived (bootstrap state or insufficient evidence).")
        print("Seed rules in path_select/rules.py cover all current cases.")
        return

    action = "Would write" if args.dry_run else "Wrote"
    print(f"{action} {len(rules)} rule(s) to path_selection_rules:")
    for r in rules:
        print(
            f"  [{r['priority']:3d}] {r['name']:<45} "
            f"→ {r['decision']:<12} "
            f"win={r['win_rate']:.0%}  n={r['evidence_count']}"
        )


def cmd_dump_rules(agent, args) -> None:
    rules = agent.get_path_selection_rules()
    if not rules:
        print("No learned rules in store (empty path_selection_rules).")
    else:
        print(json.dumps(rules, indent=2))


def cmd_record_ppa(agent, args) -> None:
    agent.record_ppa_outcome(
        benchmark=args.benchmark,
        path=args.path,
        area_improvement_pct=args.area,
        slack_status=args.slack,
        synthesis_failed=args.failed,
        notes=args.notes or "",
    )
    print(f"Recorded PPA outcome for '{args.benchmark}' (path={args.path}).")


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def cmd_init_db(agent, args) -> None:
    _print_json({"database": str(agent.db.path), "status": agent.db.status()})


def cmd_import_spec(agent, args) -> None:
    _print_json(agent.import_spec_analysis(args.path))


def cmd_import_paths(agent, args) -> None:
    _print_json(agent.import_path_decisions(args.path))


def cmd_import_rag(agent, args) -> None:
    _print_json(agent.import_rag_index(args.path))


def cmd_import_module5(agent, args) -> None:
    _print_json(agent.import_module5(args.path))


def cmd_import_c_generation(agent, args) -> None:
    _print_json(agent.import_c_generation(args.path))


def cmd_import_mcts(agent, args) -> None:
    _print_json(agent.import_mcts_plan(args.path))


def _load_object_arg(path: str):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def cmd_similar(agent, args) -> None:
    _print_json(agent.retrieve_similar_tasks(
        _load_object_arg(args.features),
        objective=args.objective,
        top_k=args.top_k,
    ))


def cmd_recommend_path(agent, args) -> None:
    _print_json(agent.recommend_path(
        _load_object_arg(args.features),
        objective=args.objective,
        top_k=args.top_k,
    ))


def cmd_rank_actions(agent, args) -> None:
    _print_json(agent.rank_actions(
        _load_object_arg(args.region_features),
        objective=args.objective,
        top_k=args.top_k,
    ))


def cmd_failure_guidance(agent, args) -> None:
    _print_json(agent.retrieve_failure_guidance(
        _load_object_arg(args.context),
        top_k=args.top_k,
    ))


def cmd_run_experience(agent, args) -> None:
    _print_json(agent.get_run_experience(args.run_id))


def cmd_derive_ppa_rules(agent, args) -> None:
    rules = agent.derive_ppa_path_rules(
        min_evidence_per_path=args.min_evidence_per_path,
        min_gain_margin=args.min_gain_margin,
        dry_run=args.dry_run,
    )
    _print_json(rules)


def cmd_refine_policies(agent, args) -> None:
    objectives = (
        ("AREA", "TIMING")
        if args.objective == "ALL"
        else (args.objective,)
    )
    result = agent.refine_path_policies(
        objectives=objectives,
        env_path=args.env,
        dry_run=args.dry_run,
        model=args.model,
    )
    _print_json({
        "model": result["model"],
        "stored": not args.dry_run,
        "token_usage": result["token_usage"],
        "policies": result["policies"],
    })


def cmd_dump_policies(agent, args) -> None:
    objective = "" if args.objective == "ALL" else args.objective
    _print_json(agent.get_path_policies(
        objective,
        enabled_only=not args.include_disabled,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m memory_agent",
        description="Module 8 Memory Agent CLI",
    )
    parser.add_argument(
        "--store", default=None,
        help="Path to staging JSON file (default: auto-detect path_decisions_log.json)",
    )
    parser.add_argument(
        "--db", default=None,
        help=(
            "Path to structured SQLite memory "
            "(default: local XDG state directory; override with MEMORY_AGENT_DB_PATH)"
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Show record counts per category")

    # derive-rules
    p_derive = sub.add_parser("derive-rules", help="Derive path-selection rules from decisions")
    p_derive.add_argument("--min-evidence", type=int, default=3,
                          help="Min completed records per group (default: 3)")
    p_derive.add_argument("--win-threshold", type=float, default=0.70,
                          help="Min win rate to emit a rule (default: 0.70)")
    p_derive.add_argument("--dry-run", action="store_true",
                          help="Print rules without writing to store")

    # dump-rules
    sub.add_parser("dump-rules", help="Print current learned rules as JSON")

    # record-ppa
    p_ppa = sub.add_parser("record-ppa", help="Record a PPA outcome from Module 7")
    p_ppa.add_argument("--benchmark", required=True)
    p_ppa.add_argument("--path", required=True, choices=["rtl_direct", "c_first"])
    p_ppa.add_argument("--area", type=float, default=None,
                       help="Area improvement %% (negative = reduction)")
    p_ppa.add_argument("--slack", default=None,
                       choices=["MET", "VIOLATED", "UNKNOWN"])
    p_ppa.add_argument("--failed", action="store_true",
                       help="Mark synthesis as failed")
    p_ppa.add_argument("--notes", default="")

    sub.add_parser("init-db", help="Create the structured SQLite memory")

    for name, help_text in (
        ("import-spec", "Import spec analysis JSON"),
        ("import-paths", "Import path decision JSON"),
        ("import-rag", "Import historical region/RAG index"),
        ("import-module5", "Import Module 5 result.json files"),
        ("import-c-generation", "Import C generation result JSON"),
        ("import-mcts", "Import a Module 4.5 MCTS plan"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("path")

    p_similar = sub.add_parser("similar", help="Retrieve similar historical tasks")
    p_similar.add_argument("features", help="JSON file containing a feature object")
    p_similar.add_argument("--objective", default="", choices=["", "AREA", "TIMING"])
    p_similar.add_argument("--top-k", type=int, default=5)

    p_path = sub.add_parser("recommend-path", help="Recommend a generation path")
    p_path.add_argument("features", help="JSON file containing a feature object")
    p_path.add_argument("--objective", required=True, choices=["AREA", "TIMING"])
    p_path.add_argument("--top-k", type=int, default=10)

    p_actions = sub.add_parser("rank-actions", help="Rank transforms for a region")
    p_actions.add_argument("region_features", help="JSON file containing region features")
    p_actions.add_argument("--objective", required=True, choices=["AREA", "TIMING"])
    p_actions.add_argument("--top-k", type=int, default=10)

    p_failures = sub.add_parser(
        "failure-guidance", help="Retrieve similar failures and fixes"
    )
    p_failures.add_argument("context", help="JSON file containing failure context")
    p_failures.add_argument("--top-k", type=int, default=5)

    p_run = sub.add_parser("run-experience", help="Dump a complete run experience chain")
    p_run.add_argument("run_id")

    p_ppa_rules = sub.add_parser(
        "derive-ppa-rules",
        help="Derive path rules from structured PPA experience",
    )
    p_ppa_rules.add_argument("--min-evidence-per-path", type=int, default=2)
    p_ppa_rules.add_argument("--min-gain-margin", type=float, default=0.0)
    p_ppa_rules.add_argument("--dry-run", action="store_true")

    p_refine = sub.add_parser(
        "refine-policies",
        help="Refine soft path policies with an LLM from structured PPA statistics",
    )
    p_refine.add_argument(
        "--objective",
        default="ALL",
        choices=["AREA", "TIMING", "ALL"],
    )
    p_refine.add_argument("--env", default=".env")
    p_refine.add_argument("--model", default="")
    p_refine.add_argument("--dry-run", action="store_true")

    p_dump_policies = sub.add_parser(
        "dump-policies",
        help="Print persisted soft path policies",
    )
    p_dump_policies.add_argument(
        "--objective",
        default="ALL",
        choices=["AREA", "TIMING", "ALL"],
    )
    p_dump_policies.add_argument("--include-disabled", action="store_true")

    args = parser.parse_args()

    store_path = args.store if args.store else _find_store()

    # Import here to keep startup fast
    from memory_agent import MemoryAgent
    agent = MemoryAgent(store_path, db_path=args.db)

    dispatch = {
        "status":       cmd_status,
        "derive-rules": cmd_derive_rules,
        "dump-rules":   cmd_dump_rules,
        "record-ppa":   cmd_record_ppa,
        "init-db":      cmd_init_db,
        "import-spec":  cmd_import_spec,
        "import-paths": cmd_import_paths,
        "import-rag":   cmd_import_rag,
        "import-module5": cmd_import_module5,
        "import-c-generation": cmd_import_c_generation,
        "import-mcts": cmd_import_mcts,
        "similar":      cmd_similar,
        "recommend-path": cmd_recommend_path,
        "rank-actions": cmd_rank_actions,
        "failure-guidance": cmd_failure_guidance,
        "run-experience": cmd_run_experience,
        "derive-ppa-rules": cmd_derive_ppa_rules,
        "refine-policies": cmd_refine_policies,
        "dump-policies": cmd_dump_policies,
    }
    dispatch[args.command](agent, args)


if __name__ == "__main__":
    main()
