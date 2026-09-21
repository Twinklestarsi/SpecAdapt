"""The adaptive router's inference stage (revise plan C3/§7).

``adaptive_predictor.train_mlp`` wrote ``router_mlp.pt`` and nothing ever read
it: the trained model existed but could not influence a single decision. This
module is the missing middle of the plan's three-stage controller ::

    hard rules  ->  latent predictor (here)  ->  LLM fallback

For one spec it produces ``p = P(c_first | S)`` and turns it into either a
decision or a hand-off:

* ``p >= handoff_high``  -> c_first, no LLM call
* ``p <= handoff_low``   -> rtl_direct, no LLM call
* in between             -> hand off to the LLM, because the model itself says
                            it cannot tell

Two properties matter more than the feature itself:

**It is optional and fails soft.** The real pipeline runs in an environment with
``openai`` but without ``torch``; the ML environment has ``torch`` but no
``openai``. So every heavy import here is lazy and every failure degrades to
"predictor unavailable, carry on with the existing tiers" with the reason
recorded. Turning the predictor on must never be able to break path selection.

**It never invents a p.** No checkpoint, no spec text, no usable encoder, or a
feature-dimension mismatch all produce ``p = None`` and a status string, not a
default of 0.5. A 0.5 would silently land in the hand-off band and look like a
real "the model is unsure" verdict.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "ROUTER_INFERENCE_VERSION",
    "RouterInferenceConfig",
    "RouterPrediction",
    "RouterInference",
]

ROUTER_INFERENCE_VERSION = "adaptive_router_inference_v1"

#: Environment variable holding the trained checkpoint path. The predictor stays
#: off unless a checkpoint is named, so an unconfigured run behaves exactly as
#: it did before this stage existed.
CHECKPOINT_ENV_VAR = "ADAPTIVE_ROUTER_CHECKPOINT"


@dataclass(frozen=True)
class RouterInferenceConfig:
    """How the predictor stage is wired. Defaults keep it switched off."""

    checkpoint_path: Optional[Path] = None
    #: The plan's hand-off band: inside it the LLM decides.
    handoff_low: float = 0.45
    handoff_high: float = 0.55
    #: Memory feedback strength; 0.0 disables blending entirely.
    delta: float = 0.10
    top_k: int = 5
    min_similarity: float = 0.80
    min_evidence: int = 3
    device: str = "cpu"
    model_alias: str = "8b"
    latent_dim: int = 256
    max_seq_length: int = 8192
    embedding_cache: Optional[Path] = None
    #: Write each encoded spec into the Memory Agent so later runs can retrieve
    #: it. Off when no store is attached.
    store_latents: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.handoff_low <= self.handoff_high <= 1.0:
            raise ValueError(
                "handoff band must satisfy 0 <= low <= high <= 1, got "
                f"({self.handoff_low}, {self.handoff_high})"
            )
        if self.delta < 0.0:
            raise ValueError("delta must be non-negative")

    @classmethod
    def from_env(cls, **overrides: Any) -> "RouterInferenceConfig":
        checkpoint = overrides.pop("checkpoint_path", None)
        if checkpoint is None:
            raw = os.environ.get(CHECKPOINT_ENV_VAR, "").strip()
            checkpoint = Path(raw) if raw else None
        return cls(checkpoint_path=checkpoint, **overrides)


@dataclass
class RouterPrediction:
    """What the predictor stage concluded for one spec."""

    status: str
    path: str = ""
    p_c_first: Optional[float] = None
    p_c_first_raw: Optional[float] = None
    reason: str = ""
    trained_on_allowed_dataset: bool = False
    memory_feedback: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str = ""

    @property
    def decided(self) -> bool:
        """True only when the predictor is confident enough to choose a path."""
        return self.path in {"c_first", "rtl_direct"}

    @property
    def handed_off(self) -> bool:
        return self.status == "handoff"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": ROUTER_INFERENCE_VERSION,
            "status": self.status,
            "path": self.path,
            "p_c_first": (
                None if self.p_c_first is None else round(self.p_c_first, 6)
            ),
            "p_c_first_raw": (
                None if self.p_c_first_raw is None else round(self.p_c_first_raw, 6)
            ),
            "reason": self.reason,
            "trained_on_allowed_dataset": self.trained_on_allowed_dataset,
            "memory_feedback": self.memory_feedback,
            "checkpoint_path": self.checkpoint_path,
        }


def _unavailable(reason: str, detail: str, checkpoint: Optional[Path]) -> RouterPrediction:
    return RouterPrediction(
        status=f"unavailable:{reason}",
        reason=detail,
        checkpoint_path=str(checkpoint or ""),
    )


class RouterInference:
    """Loads the trained router once and scores specs with it.

    Construct it even when no checkpoint is configured: it reports
    ``available == False`` and every ``predict`` call returns a
    ``disabled`` prediction. That keeps the call site in
    :mod:`path_select.selector` unconditional and free of import guards.
    """

    def __init__(
        self,
        config: Optional[RouterInferenceConfig] = None,
        *,
        memory_store: Any = None,
        task_id: str = "",
    ) -> None:
        self.config = config or RouterInferenceConfig()
        self.memory_store = memory_store
        # The latent row must point at the same task whose evaluations will be
        # consulted by C5.  Keep this as a plain string so RouterInference also
        # remains usable without a MemorySession (for example in a smoke test).
        self.task_id = str(task_id or "")
        self._predictor: Any = None
        self._vectorizer: Any = None
        self._encoder: Any = None
        self._load_error: str = ""
        self._load_reason: str = ""

    # ── Lazy component loading ────────────────────────────────────────

    @property
    def configured(self) -> bool:
        return self.config.checkpoint_path is not None

    def _ensure_loaded(self) -> bool:
        """Load checkpoint, vectorizer and encoder on first use.

        Every failure is captured as a reason string instead of propagating:
        a missing torch install is a configuration fact about the environment,
        not an error in path selection.
        """
        if self._predictor is not None:
            return True
        if self._load_error:
            return False
        checkpoint = self.config.checkpoint_path
        if checkpoint is None:
            self._load_error = "not_configured"
            self._load_reason = (
                f"no checkpoint configured (set {CHECKPOINT_ENV_VAR} or pass "
                "router_checkpoint) so the latent predictor stage is skipped"
            )
            return False
        if not Path(checkpoint).is_file():
            self._load_error = "checkpoint_missing"
            self._load_reason = f"no router checkpoint at {checkpoint}"
            return False
        try:
            from path_select.adaptive_predictor import load_predictor
            from path_select.feature_vectorizer import FeatureVectorizer
            from path_select.latent_encoder import LatentEncoder
        except ImportError as exc:
            self._load_error = "ml_stack_missing"
            self._load_reason = (
                f"the latent predictor needs torch/sklearn in this interpreter: {exc}"
            )
            return False
        try:
            predictor = load_predictor(checkpoint, device=self.config.device)
            vectorizer = FeatureVectorizer()
            encoder = LatentEncoder(
                self.config.model_alias,
                output_dim=self.config.latent_dim,
                max_seq_length=self.config.max_seq_length,
                device=self.config.device,
                embedding_cache=self.config.embedding_cache,
            )
        except Exception as exc:  # noqa: BLE001 - never break path selection
            self._load_error = "load_failed"
            self._load_reason = f"{type(exc).__name__}: {exc}"
            return False
        # A checkpoint trained on pilot/unverified data is useful for wiring
        # tests, but must never be allowed to make a real path decision.  The
        # training-side gate is deliberately repeated here because inference
        # can be invoked independently of the training command.
        dataset_manifest = getattr(predictor, "dataset_manifest", {}) or {}
        if dataset_manifest.get("training_allowed") is not True:
            self._load_error = "dataset_not_allowed"
            self._load_reason = (
                "router checkpoint dataset_manifest.training_allowed is not true; "
                "refusing to use an unapproved dataset for path selection"
            )
            return False
        if predictor.explicit_dim != vectorizer.dimension:
            self._load_error = "explicit_dim_mismatch"
            self._load_reason = (
                f"checkpoint expects {predictor.explicit_dim} explicit features "
                f"but this FeatureVectorizer emits {vectorizer.dimension}; the "
                "feature schema changed since training"
            )
            return False
        if predictor.latent_dim != self.config.latent_dim:
            self._load_error = "latent_dim_mismatch"
            self._load_reason = (
                f"checkpoint expects a {predictor.latent_dim}-dim latent but the "
                f"encoder is configured for {self.config.latent_dim}"
            )
            return False
        self._predictor = predictor
        self._vectorizer = vectorizer
        self._encoder = encoder
        return True

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    # ── Memory feedback ───────────────────────────────────────────────

    def _apply_memory_feedback(
        self,
        p_raw: float,
        latent_vector: Any,
        *,
        spec_sha256: str,
        objective: str,
    ) -> tuple[float, Dict[str, Any]]:
        """Nudge p toward what similar past specs actually did (revise plan C5).

        ``p_final = clip(p + delta * (win_rate - 0.5), 0, 1)``. Returns p
        unchanged, with the reason recorded, whenever there is no store, delta is
        zero, or the neighbourhood carries too little verified evidence.
        """
        if self.memory_store is None or self.config.delta <= 0.0:
            return p_raw, {
                "applied": False,
                "reason": (
                    "no memory store attached"
                    if self.memory_store is None
                    else "delta=0 disables memory feedback"
                ),
            }
        try:
            from memory_agent import latent_retrieval

            neighbors = latent_retrieval.find_similar_specs(
                self.memory_store,
                latent_vector,
                protocol_sha256=self._encoder.protocol_sha256,
                model_alias=self.config.model_alias,
                output_dim=self.config.latent_dim,
                top_k=self.config.top_k,
                min_similarity=self.config.min_similarity,
                # The spec being decided must not vote on itself.
                exclude_spec_sha256=spec_sha256,
            )
            evidence = latent_retrieval.c_first_win_rate(
                self.memory_store, neighbors, objective=objective
            )
            blended = latent_retrieval.blend_probability(
                p_raw,
                evidence,
                delta=self.config.delta,
                min_evidence=self.config.min_evidence,
            )
        except Exception as exc:  # noqa: BLE001 - feedback is advisory only
            return p_raw, {
                "applied": False,
                "reason": f"memory feedback failed: {type(exc).__name__}: {exc}",
            }
        return blended.p_final, blended.to_dict()

    def _record_latent(
        self,
        latent_vector: Any,
        *,
        spec_sha256: str,
        benchmark: str,
    ) -> None:
        """Persist this spec's latent so later runs can retrieve it."""
        if self.memory_store is None or not self.config.store_latents:
            return
        try:
            from memory_agent import latent_retrieval

            latent_retrieval.store_spec_latent(
                self.memory_store,
                spec_sha256=spec_sha256,
                vector=latent_vector,
                protocol=self._encoder.protocol,
                protocol_sha256=self._encoder.protocol_sha256,
                task_id=self.task_id,
                benchmark=benchmark,
                spec_id=benchmark,
            )
        except Exception:  # noqa: BLE001 - caching is never worth a crash
            return

    # ── Public API ────────────────────────────────────────────────────

    def predict(
        self,
        *,
        benchmark: str,
        llm_features: Dict[str, Any],
        spec_text: str,
        objective: str,
    ) -> RouterPrediction:
        """Score one spec and either choose a path or hand off to the LLM."""
        checkpoint = self.config.checkpoint_path
        if not self._ensure_loaded():
            status = (
                "disabled"
                if self._load_error == "not_configured"
                else f"unavailable:{self._load_error}"
            )
            return RouterPrediction(
                status=status,
                reason=self._load_reason,
                checkpoint_path=str(checkpoint or ""),
            )

        text = str(spec_text or "").strip()
        if not text:
            return _unavailable(
                "no_spec_text",
                "the latent half of the feature vector needs the spec text, and "
                "this record carries none; refusing to score on explicit "
                "features alone",
                checkpoint,
            )

        try:
            explicit = self._vectorizer.transform_one(
                llm_features or {}, objective=objective
            )
        except Exception as exc:  # noqa: BLE001
            return _unavailable(
                "explicit_features_failed", f"{type(exc).__name__}: {exc}", checkpoint
            )

        try:
            latent = self._encoder.encode_one(benchmark or "spec", text)
        except Exception as exc:  # noqa: BLE001
            return _unavailable(
                "latent_encoding_failed",
                f"{type(exc).__name__}: {exc}",
                checkpoint,
            )

        try:
            probability = self._predictor.predict_proba(
                explicit.vector, latent.vector
            )
        except Exception as exc:  # noqa: BLE001
            return _unavailable(
                "forward_pass_failed", f"{type(exc).__name__}: {exc}", checkpoint
            )

        spec_sha256 = str(latent.metadata.get("spec_sha256") or "")
        p_raw = float(probability.p_c_first)
        p_final, feedback = self._apply_memory_feedback(
            p_raw, latent.vector, spec_sha256=spec_sha256, objective=objective
        )
        self._record_latent(
            latent.vector, spec_sha256=spec_sha256, benchmark=benchmark
        )

        cfg = self.config
        if p_final >= cfg.handoff_high:
            path, status = "c_first", "decided"
            reason = (
                f"latent predictor p={p_final:.4f} >= {cfg.handoff_high} so "
                "c_first was selected without an LLM call"
            )
        elif p_final <= cfg.handoff_low:
            path, status = "rtl_direct", "decided"
            reason = (
                f"latent predictor p={p_final:.4f} <= {cfg.handoff_low} so "
                "rtl_direct was selected without an LLM call"
            )
        else:
            path, status = "", "handoff"
            reason = (
                f"latent predictor p={p_final:.4f} falls inside the hand-off "
                f"band ({cfg.handoff_low}, {cfg.handoff_high}); the model "
                "cannot tell, so the decision goes to the LLM"
            )

        return RouterPrediction(
            status=status,
            path=path,
            p_c_first=p_final,
            p_c_first_raw=p_raw,
            reason=reason,
            trained_on_allowed_dataset=bool(
                self._predictor.trained_on_allowed_dataset
            ),
            memory_feedback=feedback,
            checkpoint_path=str(probability.checkpoint_path),
        )
