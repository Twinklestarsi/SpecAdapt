"""
selector.py — Core path selection orchestrator (Module 2).

Decision design:
  1. Enforce two deterministic RTL-direct safety rules.
  2. Score the spec with the trained latent router, if a checkpoint is
     configured: decide outright outside the 0.45-0.55 hand-off band.
  3. Read AREA/TIMING soft-policy and PPA guidance from Memory Agent.
  4. Let the Module 2 LLM make the final non-hard path decision.

Stage 2 is opt-in and fails soft. The interpreter that runs this pipeline has
``openai`` but not ``torch``, so an unconfigured or unloadable predictor records
why it was skipped and the flow continues exactly as it did before.

Input:  FeatureResult (from spec_analyze) or plain dict (from JSON).
Output: PathDecision — path + reason + token_usage.
        No RAG, no transforms — those are Module 3's responsibility.

Memory staging file layout (JSON):
  {
    "path_selection_rules": [],   ← written by Module 8, read at startup (Step 8)
    "decisions": [ ... ]          ← written here after each select() (Step 6)
  }
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from path_select.rules import RuleSet, RuleVerdict
from token_counter import TokenUsage


# ── Fields kept in the feature profile written to the staging file ────
# Compact subset — the routing-relevant signals Module 8 will match on.
_PROFILE_FIELDS = frozenset({
    "architecture_pattern",
    "is_sequential",
    "num_clock_domains",
    "has_fsm",
    "estimated_fsm_states",
    "complexity",
    "hierarchy",
    "suggested_subcategory",
    "data_widths",
})


# ── Result type ───────────────────────────────────────────────────────

@dataclass
class PathDecision:
    """
    Output of Module 2 for one benchmark.

    path:
        "rtl_direct" — generate/optimize RTL directly from the specification
        "c_first"    — generate a C behavioral model, then translate C to RTL

    token_usage:
        Zero for Tier-1 (rule-based) decisions.
        Populated from response.usage for Tier-2 LLM calls.
    """
    benchmark: str
    path: str           # "rtl_direct" | "c_first"
    reason: str
    rule_fired: str
    confidence: str     # "high" | "medium" | "low"
    optimization_target: str = ""
    llm_reasoning: str = ""
    tier: int = 1       # 1 = rule-based, 2 = LLM-assisted, 3 = latent predictor
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    memory_evidence: Dict[str, Any] = field(default_factory=dict)

    # ── Latent predictor stage (revise plan C3/§7) ────────────────────
    # p_c_first is P(c_first | S) as produced by the trained router. It is
    # recorded on EVERY decision the predictor scored, including the ones it
    # handed to the LLM, because "path selection accuracy" in the paper is
    # measured against this number and cannot be computed if we only keep it
    # when the predictor happened to decide. None means no p exists, and
    # predictor_status says why -- never a stand-in value like 0.5.
    p_c_first: Optional[float] = None
    p_c_first_raw: Optional[float] = None
    predictor_status: str = ""
    predictor_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── LLM prompt for Tier-2 ─────────────────────────────────────────────

_TIER2_SYSTEM = """\
You are an FPGA/RTL optimization expert. Given a design's feature vector,
decide whether to generate/optimize via:
  (A) rtl_direct: apply LLM-based Verilog transformations directly.
  (B) c_first: generate a C behavioral model first, then translate C to RTL with an LLM.

Module 2 has already enforced the hard RTL-direct safety constraints before
calling you. Do not reinterpret or recreate those rules.

Optimization target guidance:
- TIMING: prefer the path more likely to reduce critical-path delay.
- AREA: prefer the path more likely to reduce logic / resource usage.

Use the Memory Agent policy scores and historical evidence as guidance. Apply
your engineering judgment when evidence is sparse, imbalanced, or conflicting.

Respond with a JSON object only, no commentary outside the JSON:
{
  "path": "rtl_direct" or "c_first",
  "reasoning": "<1-3 sentences explaining the choice>"
}
"""

_TIER2_USER_TEMPLATE = """\
Optimization target:
{optimization_target}

Design features:
{features_json}

The two hard RTL-direct safety rules did not match.

Memory Agent guidance:
{memory_guidance_json}

Memory guidance is advisory, not a hard constraint. Evaluate the design
features, optimization target, policy scores, confidence, evidence balance,
historical PPA, regression risk, and uncertainty. Make the final path
selection decision.
"""


# ── PathSelector ──────────────────────────────────────────────────────

class PathSelector:
    """
    Module 2: path selector.

    Usage:
        selector = PathSelector()
        decision = selector.select(feature_result)
        decision = selector.select(result_dict)
        decisions = selector.select_batch(results_list)
        decisions = selector.select_from_json("spec_analysis.json")

    With memory staging (writes decisions for Module 8):
        selector = PathSelector(memory_path="path_decisions_log.json")
    """

    def __init__(
        self,
        env_path: Optional[str | Path] = None,
        model: Optional[str] = None,
        memory_path: Optional[str | Path] = None,
        rules_source: Optional[list] = None,
        memory_session: Optional[Any] = None,
        router_checkpoint: Optional[str | Path] = None,
        router_config: Optional[Any] = None,
        latent_store: Optional[Any] = None,
    ):
        """
        Args:
            env_path:      Path to .env for LLM credentials (Tier-2).
                           Auto-detected from project root if None.
            model:         LLM model override for Tier-2.
            memory_path:   Path to the Module 8 staging JSON file.
                           If set: decisions are written after each select(),
                           and any Module 8 rules are loaded at startup.
                           If None: no file I/O (default).
            rules_source:  Explicit rule list to load into RuleSet instead of
                           reading from memory_path. Useful for testing.
            router_checkpoint:
                           Trained router_mlp.pt. If None, falls back to
                           $ADAPTIVE_ROUTER_CHECKPOINT; if that is unset too,
                           the latent predictor stage is skipped entirely.
            router_config: Pre-built RouterInferenceConfig, overriding
                           router_checkpoint. For tests and sweeps that need a
                           non-default hand-off band or delta.
            latent_store:  SQLiteMemoryStore for latent retrieval. Defaults to
                           the one behind memory_session when it has one, so
                           the pipeline gets C5 feedback without extra wiring.
        """
        self._llm_client = None
        self._model = model
        self._env_path = Path(env_path) if env_path else self._find_env()
        self._memory_path = Path(memory_path) if memory_path else None
        self._memory_session = memory_session
        self._ruleset = RuleSet()

        self._router_config = router_config
        self._router_checkpoint = router_checkpoint
        # getattr chain rather than an isinstance check: memory_session is a
        # duck-typed collaborator and tests pass in stubs that have no agent.
        self._latent_store = latent_store or getattr(
            getattr(memory_session, "agent", None), "db", None
        )
        self._router: Optional[Any] = None

        # Step 8: load Module 8 rules if available, else use explicit list
        if rules_source is not None:
            self._ruleset.load_from_memory(rules_source)
        else:
            self._load_rules_from_memory()

    # ── Latent predictor stage (lazy) ──────────────────────────────────

    def _ensure_router(self) -> Optional[Any]:
        """Build the RouterInference wrapper once, or return None.

        Construction itself does no heavy work -- the checkpoint, torch and the
        embedding model are only touched on the first ``predict`` call -- so this
        costs nothing on runs that never reach the stage.
        """
        if self._router is not None:
            return self._router
        try:
            from path_select.router_inference import (
                RouterInference,
                RouterInferenceConfig,
            )
        except Exception:  # noqa: BLE001 - stage is optional by design
            return None
        config = self._router_config
        if config is None:
            checkpoint = (
                Path(self._router_checkpoint) if self._router_checkpoint else None
            )
            try:
                config = RouterInferenceConfig.from_env(checkpoint_path=checkpoint)
            except Exception:  # noqa: BLE001
                return None
        # Keep the latent row on the same task as the evaluations recorded by
        # this MemorySession.  Without this, C5 can retrieve the vector but
        # cannot find the task's c_first/rtl_direct measurements.
        task_id = str(
            getattr(getattr(self._memory_session, "context", None), "task_id", "")
            or ""
        )
        self._router = RouterInference(
            config,
            memory_store=self._latent_store,
            task_id=task_id,
        )
        return self._router

    # ── Env detection ─────────────────────────────────────────────────

    @staticmethod
    def _find_env() -> Path:
        here = Path(__file__).resolve().parent
        for p in [here.parent, here.parent.parent]:
            candidate = p / ".env"
            if candidate.exists():
                return candidate
        return here.parent / ".env"

    # ── Step 8: load rules from Module 8 staging file ─────────────────

    def _load_rules_from_memory(self) -> None:
        """
        Load path-selection rules from the Module 8 staging file.

        Reads "path_selection_rules" (condition-dicts written by Module 8).
        Compiles them into callables via memory_agent.rule_compiler and
        prepends them ahead of the seed heuristics in RuleSet.

        Silently keeps seed rules on any read/compile/import error so that
        a missing or broken Module 8 store never blocks Module 2.
        """
        if not self._memory_path or not self._memory_path.exists():
            return
        try:
            with open(self._memory_path, encoding="utf-8") as f:
                data = json.load(f)
            rule_dicts = data.get("path_selection_rules", [])
            if rule_dicts:
                from memory_agent.rule_compiler import compile_rules
                compiled = compile_rules(rule_dicts)
                if compiled:
                    self._ruleset.load_from_memory(compiled)
        except Exception:
            pass  # silently keep seed rules on any read/parse/import error

    # ── LLM initialization (lazy) ─────────────────────────────────────

    def _ensure_llm_client(self) -> bool:
        if self._llm_client is not None:
            return True
        try:
            from openai import OpenAI
            load_dotenv(self._env_path, override=True)
            self._llm_client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
            if not self._model:
                self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            return True
        except Exception:
            return False

    # ── Feature extraction helpers ────────────────────────────────────

    @staticmethod
    def _resolve_optimization_target(
        benchmark: str,
        spec_text: str = "",
        explicit_target: str = "",
    ) -> str:
        if explicit_target:
            normalized = str(explicit_target).strip().upper()
            if normalized in {"AREA", "TIMING"}:
                return normalized

        seed_text = f"{benchmark}::{spec_text or ''}"
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        return "AREA" if int(digest[:8], 16) % 2 == 0 else "TIMING"

    @staticmethod
    def _extract_fields(result: Union[Dict[str, Any], Any]) -> tuple:
        if isinstance(result, dict):
            benchmark    = result.get("benchmark", "unknown")
            spec_text    = result.get("spec_text", "") or ""
            opt_target   = result.get("optimization_target", "")
            llm_feat     = result.get("llm_features", {})
            overall_conf = result.get("overall_confidence", "medium")
            regex_feat   = result.get("regex_features", None)
        else:
            benchmark    = getattr(result, "benchmark", "unknown")
            spec_text    = getattr(result, "spec_text", "") or ""
            opt_target   = getattr(result, "optimization_target", "")
            llm_feat     = getattr(result, "llm_features", {})
            overall_conf = getattr(result, "overall_confidence", "medium")
            regex_feat   = getattr(result, "regex_features", None)
        opt_target = PathSelector._resolve_optimization_target(
            benchmark, spec_text=spec_text, explicit_target=opt_target
        )
        # spec_text is returned as well as consumed: the latent predictor stage
        # embeds it, and it used to be dropped here.
        return benchmark, opt_target, llm_feat, overall_conf, regex_feat, spec_text

    # ── Step 5: Tier-2 LLM call with token capture ────────────────────

    def _tier2_llm(
        self,
        optimization_target: str,
        llm_features: Dict[str, Any],
        memory_guidance: Dict[str, Any],
        max_retries: int = 2,
    ) -> Optional[tuple]:
        """
        Call LLM for uncertain cases using Stage 1 features only.

        Returns (path, reasoning, TokenUsage) on success, None on total failure.
        Token usage is read from response.usage on each attempt (successful
        or partially successful); retries are counted for failed attempts.
        """
        if not self._ensure_llm_client():
            return None

        usage = TokenUsage()

        features_json = json.dumps(
            {k: v for k, v in llm_features.items()
             if k not in ("optimization_notes", "purpose")},
            indent=2,
        )
        user_msg = _TIER2_USER_TEMPLATE.replace(
            "{optimization_target}", optimization_target or "UNSPECIFIED"
        ).replace(
            "{features_json}", features_json
        ).replace(
            "{memory_guidance_json}",
            json.dumps(
                {
                    "recommended_path": memory_guidance.get(
                        "recommended_path", ""
                    ),
                    "decision_source": memory_guidance.get(
                        "decision_source", ""
                    ),
                    "has_sufficient_evidence": memory_guidance.get(
                        "has_sufficient_evidence", False
                    ),
                    "policy_scores": memory_guidance.get("policy_scores", {}),
                    "history_scores": memory_guidance.get("history_scores", {}),
                    "combined_scores": memory_guidance.get(
                        "combined_scores", {}
                    ),
                    "path_statistics": memory_guidance.get(
                        "path_statistics", {}
                    ),
                    "matched_policies": memory_guidance.get(
                        "matched_policies", []
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
        )

        for attempt in range(max_retries + 1):
            try:
                response = self._llm_client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _TIER2_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.1,
                    max_tokens=256,
                )

                # Capture token counts from this attempt
                if response.usage:
                    usage.prompt_tokens     += response.usage.prompt_tokens
                    usage.completion_tokens += response.usage.completion_tokens
                    usage.total_tokens      += response.usage.total_tokens

                text = response.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                text = text.strip()

                parsed = json.loads(text)
                path = parsed.get("path", "").lower().strip()
                reasoning = parsed.get("reasoning", "")
                if path in ("rtl_direct", "c_first"):
                    return path, reasoning, usage

            except Exception:
                if attempt < max_retries:
                    usage.retries += 1
                    time.sleep(1)
                continue

        return None

    # ── Step 6: write decision to Module 8 staging file ───────────────

    def _write_decision_to_memory(
        self,
        decision: PathDecision,
        llm_features: Dict[str, Any],
    ) -> None:
        """
        Append one decision record to the Module 8 staging JSON file.

        Only runs if memory_path is set. Silently skips on any I/O error
        to avoid blocking the pipeline (CIFS filesystem may be unreliable).

        Record format:
          benchmark, timestamp, feature_profile (compact), path,
          rule_fired, confidence, tier, token_usage
        """
        if not self._memory_path:
            return

        record = {
            "benchmark":       decision.benchmark,
            "optimization_target": decision.optimization_target,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "feature_profile": {k: v for k, v in llm_features.items()
                                if k in _PROFILE_FIELDS},
            "path":            decision.path,
            "reason":          decision.reason,
            "rule_fired":      decision.rule_fired,
            "confidence":      decision.confidence,
            "tier":            decision.tier,
            "token_usage":     decision.token_usage.to_dict(),
            "p_c_first":       decision.p_c_first,
            "predictor_status": decision.predictor_status,
        }

        try:
            from memory_agent.store import MemoryStore
            from memory_agent import MemoryAgent
            MemoryStore(self._memory_path).append_decision(record)
            MemoryAgent(self._memory_path).record_path_decision_payload(record)

        except Exception as exc:
            print(f"Warning: could not write to memory staging file: {exc}",
                  file=sys.stderr)

    # ── Internal implementation ───────────────────────────────────────

    def _select_impl(
        self,
        benchmark: str,
        optimization_target: str,
        llm_feat: Dict[str, Any],
        overall_conf: str,
        regex_feat: Optional[Dict[str, Any]],
        use_llm: bool,
        spec_text: str = "",
    ) -> PathDecision:
        """Core logic — called by select(), which then writes to memory."""

        hard_verdict: RuleVerdict = self._ruleset.apply_hard(
            llm_feat,
            overall_conf,
            regex_feat,
            optimization_target=optimization_target,
        )

        if hard_verdict.path != "uncertain":
            return PathDecision(
                benchmark=benchmark,
                optimization_target=optimization_target,
                path=hard_verdict.path,
                reason=hard_verdict.reason,
                rule_fired=hard_verdict.rule_fired,
                confidence=hard_verdict.confidence,
                tier=1,
                predictor_status="skipped:hard_rule",
            )

        # ── Stage 2: the trained latent router (revise plan C3/§7) ────
        # Runs only after the hard rules declined, and adds no rule of its own
        # (C4: no new c_first hard rule). Outside the hand-off band it answers
        # without an LLM call; inside it, it explicitly abstains and the flow
        # falls through to the existing Memory + LLM tiers below.
        prediction = self._run_predictor(
            benchmark=benchmark,
            llm_feat=llm_feat,
            spec_text=spec_text,
            objective=optimization_target,
        )
        if prediction is not None and prediction.decided:
            return PathDecision(
                benchmark=benchmark,
                optimization_target=optimization_target,
                path=prediction.path,
                reason=prediction.reason,
                rule_fired="latent_predictor",
                confidence=self._predictor_confidence(prediction),
                tier=3,
                p_c_first=prediction.p_c_first,
                p_c_first_raw=prediction.p_c_first_raw,
                predictor_status=prediction.status,
                predictor_evidence=prediction.to_dict(),
            )

        guidance = (
            self._memory_session.recommend_path(llm_feat)
            if self._memory_session is not None
            else {}
        )
        memory_path = str(guidance.get("recommended_path") or "")

        # Whatever the later tiers decide, carry the predictor's p forward so the
        # paper's accuracy metric can be computed on hand-off cases too.
        predictor_fields: Dict[str, Any] = (
            {
                "p_c_first": prediction.p_c_first,
                "p_c_first_raw": prediction.p_c_first_raw,
                "predictor_status": prediction.status,
                "predictor_evidence": prediction.to_dict(),
            }
            if prediction is not None
            else {}
        )

        # Module 2 agent makes the final decision with Memory guidance.
        if use_llm:
            llm_result = self._tier2_llm(
                optimization_target,
                llm_feat,
                guidance,
            )
            if llm_result is not None:
                llm_path, llm_reasoning, usage = llm_result
                return PathDecision(
                    benchmark=benchmark,
                    optimization_target=optimization_target,
                    path=llm_path,
                    reason="Module 2 agent decision guided by Memory Agent policies.",
                    rule_fired="memory_guided_llm",
                    confidence="medium",
                    llm_reasoning=llm_reasoning,
                    tier=2,
                    token_usage=usage,
                    memory_evidence=guidance,
                    **predictor_fields,
                )

        if memory_path in {"rtl_direct", "c_first"}:
            return PathDecision(
                benchmark=benchmark,
                optimization_target=optimization_target,
                path=memory_path,
                reason=(
                    "Module 2 LLM was disabled or unavailable; selected from "
                    "Memory Agent policy and PPA scores."
                ),
                rule_fired="memory_guidance_no_llm",
                confidence=(
                    "medium"
                    if guidance.get("has_sufficient_evidence")
                    else "low"
                ),
                tier=1,
                memory_evidence=guidance,
                **predictor_fields,
            )

        return PathDecision(
            benchmark=benchmark,
            optimization_target=optimization_target,
            path="rtl_direct",
            reason=(
                "Defaulting to RTL-direct because no hard rule, Memory "
                "recommendation, or Module 2 LLM decision was available."
            ),
            rule_fired="fallback_rtl_direct",
            confidence="low",
            tier=1,
            **predictor_fields,
        )

    # ── Stage 2 helpers ────────────────────────────────────────────────

    def _run_predictor(
        self,
        *,
        benchmark: str,
        llm_feat: Dict[str, Any],
        spec_text: str,
        objective: str,
    ) -> Optional[Any]:
        """Score one spec with the latent router, or return None.

        None means the stage produced nothing at all (no wrapper could be
        built). A RouterPrediction with a ``disabled``/``unavailable:*`` status is
        still returned, because "there is no p and here is why" belongs in the
        decision record.
        """
        router = self._ensure_router()
        if router is None:
            return None
        try:
            return router.predict(
                benchmark=benchmark,
                llm_features=llm_feat,
                spec_text=spec_text,
                objective=objective,
            )
        except Exception as exc:  # noqa: BLE001 - the stage may never break Module 2
            print(
                f"Warning: latent predictor stage failed, continuing without it: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return None

    @staticmethod
    def _predictor_confidence(prediction: Any) -> str:
        """Confidence label from how far p sits outside the hand-off band."""
        p = prediction.p_c_first
        if p is None:
            return "low"
        margin = min(abs(p - 0.5), 0.5)
        if margin >= 0.35:
            return "high"
        if margin >= 0.15:
            return "medium"
        return "low"

    # ── Public API ────────────────────────────────────────────────────

    def select(
        self,
        result: Union[Dict[str, Any], Any],
        use_llm: bool = True,
    ) -> PathDecision:
        """
        Select optimization path for a single benchmark.

        Args:
            result:   FeatureResult object or plain dict from spec_analysis.json.
            use_llm:  If True, call LLM for uncertain cases (Tier-2).

        Returns:
            PathDecision. If memory_path is set, the decision is also
            appended to the staging file for Module 8.
        """
        (
            benchmark,
            optimization_target,
            llm_feat,
            overall_conf,
            regex_feat,
            spec_text,
        ) = self._extract_fields(result)
        decision = self._select_impl(
            benchmark,
            optimization_target,
            llm_feat,
            overall_conf,
            regex_feat,
            use_llm,
            spec_text=spec_text,
        )
        self._write_decision_to_memory(decision, llm_feat)
        if self._memory_session is not None:
            self._memory_session.record_path_decision(
                decision,
                llm_feat,
                guidance=decision.memory_evidence or None,
            )
        return decision

    def select_batch(
        self,
        results: List[Union[Dict[str, Any], Any]],
        use_llm: bool = True,
    ) -> List[PathDecision]:
        return [self.select(r, use_llm=use_llm) for r in results]

    def select_from_json(
        self,
        json_path: str | Path,
        use_llm: bool = True,
    ) -> List[PathDecision]:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return self.select_batch(data, use_llm=use_llm)
