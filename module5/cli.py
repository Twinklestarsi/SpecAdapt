from __future__ import annotations

import argparse
import json

from module5.executor import execute_action, execute_action_sequence
from module5.io_utils import action_from_dict, load_best_action, load_best_actions
from project_paths import PROJECT_ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description="Execute Module 4.5 actions through Module 5.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--plan", help="Path to Module45Plan JSON")
    src.add_argument("--action-json", help="Path to a single Module5Action JSON")
    ap.add_argument("--action-index", type=int, default=0, help="Index into best_actions when --plan is used (single-action mode)")
    ap.add_argument("--execute-best-path", action="store_true", help="Execute all best_actions in a single pass (sequence mode)")
    ap.add_argument("--max-actions", type=int, default=None, help="Limit number of actions in sequence mode")
    ap.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "module5_runs"),
        help="Work root for Module 5 artifacts",
    )
    ap.add_argument("--env-path", default=".env", help="Path to .env for LLM configuration")
    ap.add_argument(
        "--backend",
        choices=["direct_rtl"],
        default="direct_rtl",
        help="AI C-to-RTL backend; Vitis HLS is inactive",
    )
    ap.add_argument("--spec-db-path", default=str(PROJECT_ROOT / "spec_analysis.json"), help="Path to spec_analysis.json")
    ap.add_argument("--no-baseline", action="store_true", help="Skip baseline RTL/DC evaluation")
    ap.add_argument(
        "--verification-mode",
        choices=("jaspergold", "none"),
        default="none",
        help="jaspergold: syntax -> JG with one feedback retry -> DC; none: syntax -> DC",
    )
    ap.add_argument(
        "--enable-jg-verification",
        action="store_true",
        help="Deprecated alias for --verification-mode jaspergold",
    )
    ap.add_argument("--golden-rtl-path", default="", help="Golden RTL required by JasperGold mode")
    ap.add_argument("--jg-max-retries", type=int, default=1, help="Max JG-driven AI RTL regenerations")
    ap.add_argument("--rtl-max-retries", type=int, default=2, help="Max direct RTL syntax correction retry attempts")
    ap.add_argument("--dc-max-retries", type=int, default=0, help="Max direct RTL regeneration attempts after DC failure")
    args = ap.parse_args()

    common_kwargs = dict(
        output_root=args.output_root,
        run_baseline=not args.no_baseline,
        env_path=args.env_path,
        backend=args.backend,
        spec_db_path=args.spec_db_path,
        verification_mode=(
            "jaspergold"
            if args.enable_jg_verification
            else args.verification_mode
        ),
        golden_rtl_path=args.golden_rtl_path,
        jg_max_retries=args.jg_max_retries,
        rtl_max_retries=args.rtl_max_retries,
        dc_max_retries=args.dc_max_retries,
    )

    if args.execute_best_path:
        if not args.plan:
            ap.error("--execute-best-path requires --plan")
        actions = load_best_actions(args.plan, max_actions=args.max_actions)
        print(f"[Module5] Sequence mode: {len(actions)} actions")
        for i, a in enumerate(actions):
            print(f"  [{i}] {a.action_id}  ({a.transform_name})")
        result = execute_action_sequence(actions, **common_kwargs)
    else:
        if args.plan:
            action = load_best_action(args.plan, action_index=args.action_index)
        else:
            with open(args.action_json, "r", encoding="utf-8") as fh:
                action = action_from_dict(json.load(fh))
        result = execute_action(action, **common_kwargs)

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
