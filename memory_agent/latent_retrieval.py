"""Latent (embedding) storage and cosine retrieval for the adaptive router.

This is revise-plan C5. The Memory Agent's existing retrieval
(:mod:`memory_agent.retrieval`) compares specs by their *structured* features:
a hand-written similarity over architecture pattern, widths, FSM state count and
so on. That works, but it can only see what Module 1 chose to name.

Here a spec is compared by its Qwen3-Embedding latent vector instead, so two
specs that describe the same behaviour in different words rank as neighbours
even when their structured features differ.

Three things this module is careful about:

1.  **Vectors from different protocols are never mixed.** ``protocol_sha256``
    pins model id, revision, output_dim, pooling, truncation and normalisation.
    A cosine between a 256-dim matryoshka prefix and some other encoding is a
    number with no meaning, so every query filters on the protocol.

2.  **Vectors are stored L2-normalised**, so cosine similarity is a plain dot
    product. The norm is re-checked on read: a vector that drifted off the unit
    sphere means the writer bypassed the encoder, and is rejected rather than
    quietly producing similarities above 1.

3.  **The win rate is only computed from evidence that passed correctness.**
    A neighbour whose c_first run failed equivalence is not evidence that
    c_first wins; it is evidence about that run. Runs without a real verdict are
    excluded from the denominator, and the count that survived is reported so a
    blend built on two samples cannot be mistaken for one built on fifty.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from memory_agent.sqlite_store import SQLiteMemoryStore

__all__ = [
    "LATENT_RETRIEVAL_VERSION",
    "DEFAULT_DELTA",
    "DEFAULT_TOP_K",
    "DEFAULT_MIN_SIMILARITY",
    "DEFAULT_MIN_EVIDENCE",
    "LatentNeighbor",
    "WinRateEvidence",
    "BlendedProbability",
    "encode_vector",
    "decode_vector",
    "vector_sha256",
    "store_spec_latent",
    "find_similar_specs",
    "c_first_win_rate",
    "blend_probability",
]

LATENT_RETRIEVAL_VERSION = "memory_latent_retrieval_v1"

#: Feedback strength in ``p_final = clip(p + delta * (win_rate - 0.5), 0, 1)``.
#: 0.10 means a neighbourhood where c_first won every time can move p by at most
#: +0.05 -- enough to break a tie inside the 0.45-0.55 handoff band, not enough
#: to overturn a confident predictor. The paper must report this value and show
#: a sensitivity sweep, because it is a free parameter chosen by us.
DEFAULT_DELTA = 0.10
DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.80
#: Below this many correctness-passing neighbour runs the win rate is noise and
#: no blending happens at all.
DEFAULT_MIN_EVIDENCE = 3

_PASS_TOKENS = {"passed", "pass", "success", "ok"}
_FAIL_TOKENS = {"failed", "fail", "error", "mismatch"}
_NORM_TOLERANCE = 1e-3


# ── Vector encoding ───────────────────────────────────────────────────────


def encode_vector(vector: Sequence[float] | np.ndarray) -> bytes:
    """Pack an L2-normalised vector as little-endian float32 bytes."""
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("Refusing to store an empty latent vector")
    if not np.all(np.isfinite(array)):
        raise ValueError("Latent vector contains NaN or infinity")
    norm = float(np.linalg.norm(array))
    if abs(norm - 1.0) > _NORM_TOLERANCE:
        raise ValueError(
            f"Latent vector is not L2-normalised (norm={norm:.6f}); cosine "
            "similarity is computed as a dot product and would be wrong"
        )
    return np.ascontiguousarray(array, dtype="<f4").tobytes()


def decode_vector(blob: bytes, output_dim: int) -> np.ndarray:
    """Unpack stored bytes back into a float32 vector."""
    expected = int(output_dim) * 4
    if len(blob) != expected:
        raise ValueError(
            f"Stored latent is {len(blob)} bytes, expected {expected} for "
            f"{output_dim} float32 values"
        )
    return np.frombuffer(blob, dtype="<f4").astype(np.float32)


def vector_sha256(vector: Sequence[float] | np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(vector, dtype=np.float32).reshape(-1))
    return hashlib.sha256(array.tobytes()).hexdigest()


# ── Records ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LatentNeighbor:
    """One retrieved spec and how close it is."""

    spec_sha256: str
    similarity: float
    task_id: str = ""
    benchmark: str = ""
    spec_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_sha256": self.spec_sha256,
            "similarity": round(float(self.similarity), 6),
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "spec_id": self.spec_id,
        }


@dataclass(frozen=True)
class WinRateEvidence:
    """How often c_first beat rtl_direct among retrieved neighbours."""

    win_rate: float | None
    c_first_wins: int
    comparisons: int
    excluded_no_verdict: int
    excluded_incomparable: int
    objective: str
    neighbors: List[LatentNeighbor] = field(default_factory=list)

    @property
    def has_sufficient_evidence(self) -> bool:
        return self.win_rate is not None and self.comparisons >= DEFAULT_MIN_EVIDENCE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "win_rate": None if self.win_rate is None else round(self.win_rate, 6),
            "c_first_wins": self.c_first_wins,
            "comparisons": self.comparisons,
            "excluded_no_verdict": self.excluded_no_verdict,
            "excluded_incomparable": self.excluded_incomparable,
            "objective": self.objective,
            "neighbors": [item.to_dict() for item in self.neighbors],
        }


@dataclass(frozen=True)
class BlendedProbability:
    """The predictor's p after the memory feedback term."""

    p_raw: float
    p_final: float
    delta: float
    applied: bool
    reason: str
    evidence: WinRateEvidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": LATENT_RETRIEVAL_VERSION,
            "p_raw": round(self.p_raw, 6),
            "p_final": round(self.p_final, 6),
            "delta": self.delta,
            "applied": self.applied,
            "reason": self.reason,
            "evidence": self.evidence.to_dict(),
        }


# ── Write path ────────────────────────────────────────────────────────────


def store_spec_latent(
    store: SQLiteMemoryStore,
    *,
    spec_sha256: str,
    vector: Sequence[float] | np.ndarray,
    protocol: Dict[str, Any],
    task_id: str = "",
    benchmark: str = "",
    spec_id: str = "",
    protocol_sha256: str = "",
) -> str:
    """Persist one spec's latent vector. Returns the row key.

    ``protocol`` is :attr:`path_select.latent_encoder.LatentEncoder.protocol`;
    ``protocol_sha256`` its digest. They are stored verbatim so that a later
    query can prove two vectors came from the same recipe.
    """
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    output_dim = int(protocol.get("output_dim") or array.size)
    if array.size != output_dim:
        raise ValueError(
            f"Vector has {array.size} dims but protocol says output_dim="
            f"{output_dim}"
        )
    digest = protocol_sha256 or str(protocol.get("protocol_sha256") or "")
    if not digest:
        raise ValueError(
            "protocol_sha256 is required: without it a later cosine query "
            "cannot tell comparable vectors from incomparable ones"
        )
    return store.upsert_spec_latent(
        spec_sha256=spec_sha256,
        model_alias=str(protocol.get("model_alias") or ""),
        output_dim=output_dim,
        protocol_sha256=digest,
        latent_vector=encode_vector(array),
        task_id=task_id,
        benchmark=benchmark,
        spec_id=spec_id,
        model_id=str(protocol.get("model_id") or ""),
        vector_sha256=vector_sha256(array),
        metadata={"protocol": protocol},
    )


# ── Read path: cosine ranking ─────────────────────────────────────────────


def find_similar_specs(
    store: SQLiteMemoryStore,
    query_vector: Sequence[float] | np.ndarray,
    *,
    protocol_sha256: str,
    model_alias: str = "",
    output_dim: int = 0,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    exclude_spec_sha256: str = "",
) -> List[LatentNeighbor]:
    """Rank stored specs by cosine similarity to ``query_vector``.

    Both sides are unit vectors, so the cosine is a dot product. Rows whose
    stored vector is not on the unit sphere are skipped rather than trusted --
    that can only happen if something wrote to the table without going through
    :func:`store_spec_latent`.
    """
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if query.size == 0:
        return []
    query_norm = float(np.linalg.norm(query))
    if not math.isfinite(query_norm) or query_norm <= 0.0:
        raise ValueError("Query latent vector has no finite positive norm")
    query = query / query_norm

    rows = store.spec_latent_rows(
        model_alias=model_alias,
        output_dim=output_dim or int(query.size),
        protocol_sha256=protocol_sha256,
        exclude_spec_sha256=exclude_spec_sha256,
    )
    neighbors: List[LatentNeighbor] = []
    for row in rows:
        stored_dim = int(row["output_dim"])
        if stored_dim != query.size:
            continue
        candidate = decode_vector(row["latent_vector"], stored_dim)
        norm = float(np.linalg.norm(candidate))
        if not math.isfinite(norm) or abs(norm - 1.0) > _NORM_TOLERANCE:
            continue
        similarity = float(np.dot(query, candidate))
        if similarity < min_similarity:
            continue
        neighbors.append(
            LatentNeighbor(
                spec_sha256=str(row["spec_sha256"]),
                similarity=similarity,
                task_id=str(row.get("task_id") or ""),
                benchmark=str(row.get("benchmark") or ""),
                spec_id=str(row.get("spec_id") or ""),
            )
        )
    neighbors.sort(key=lambda item: item.similarity, reverse=True)
    return neighbors[: max(0, int(top_k))]


def _verdict(status: Any) -> bool | None:
    token = str(status or "").strip().lower()
    if token in _PASS_TOKENS:
        return True
    if token in _FAIL_TOKENS:
        return False
    return None


def _verification_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    """Decode the raw verification evidence stored with an evaluation.

    ``correctness_status`` is not used here: Memory Agent may deliberately map
    an unverified successful run to ``passed`` for its legacy consumers.  C5
    needs the original evidence, which runtime stores in ``metadata_json``.
    Malformed or absent metadata is treated as missing evidence.
    """
    raw = row.get("metadata_json")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        decoded = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _verified_equivalence(row: Dict[str, Any]) -> bool:
    """Return true only for explicit JasperGold-passed evidence."""
    metadata = _verification_metadata(row)
    mode = str(metadata.get("verification_mode") or "").strip().lower()
    equivalence = metadata.get("equivalence_status")
    return mode == "jaspergold" and _verdict(equivalence) is True


def _better(objective: str, c_value: float, d_value: float) -> bool:
    """Lower area and lower delay both win, so the test is the same either way."""
    return c_value < d_value


def c_first_win_rate(
    store: SQLiteMemoryStore,
    neighbors: Iterable[LatentNeighbor],
    *,
    objective: str,
) -> WinRateEvidence:
    """How often c_first beat rtl_direct on the retrieved neighbours.

    One comparison needs BOTH routes measured on the same task under the same
    objective, and both must have explicit JasperGold-passed evidence in
    ``metadata_json``. The legacy ``correctness_status`` column is deliberately
    ignored because Memory may map an unverified success to ``passed``. Anything
    else is counted in an ``excluded_*`` bucket instead of being guessed at, so
    the caller can see how thin the evidence is.
    """
    objective_token = str(objective or "AREA").strip().upper()
    # The SQLite schema predates the public ``delay_ps`` name.  Its
    # ``data_arrival_time_ps`` column is the persisted critical-path delay;
    # keep using that storage column while treating it as delay (never slack).
    metric_column = "area" if objective_token == "AREA" else "data_arrival_time_ps"
    neighbor_list = list(neighbors)
    task_ids = [item.task_id for item in neighbor_list if item.task_id]

    wins = 0
    comparisons = 0
    excluded_no_verdict = 0
    excluded_incomparable = 0

    for task_id in task_ids:
        rows = store.rows(
            f"""
            SELECT path, correctness_status, metadata_json,
                   {metric_column} AS metric, timestamp
            FROM evaluations
            WHERE task_id = ? AND UPPER(objective) = ?
            ORDER BY timestamp DESC
            """,
            (task_id, objective_token),
        )
        best: Dict[str, float] = {}
        saw_unverified = False
        for row in rows:
            path = str(row.get("path") or "").strip().lower()
            if path not in {"c_first", "rtl_direct"}:
                continue
            # Do not trust correctness_status: runtime can contain the legacy
            # unverified->passed value.  Missing or malformed raw evidence is
            # intentionally excluded as well.
            if (
                _verdict(row.get("correctness_status")) is not True
                or not _verified_equivalence(row)
            ):
                saw_unverified = True
                continue
            value = row.get("metric")
            if value is None:
                saw_unverified = True
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                continue
            # Keep each route's best measurement for this objective.
            if path not in best or number < best[path]:
                best[path] = number

        if "c_first" in best and "rtl_direct" in best:
            comparisons += 1
            if _better(objective_token, best["c_first"], best["rtl_direct"]):
                wins += 1
        elif saw_unverified or not best:
            excluded_no_verdict += 1
        else:
            excluded_incomparable += 1

    win_rate = (wins / comparisons) if comparisons else None
    return WinRateEvidence(
        win_rate=win_rate,
        c_first_wins=wins,
        comparisons=comparisons,
        excluded_no_verdict=excluded_no_verdict,
        excluded_incomparable=excluded_incomparable,
        objective=objective_token,
        neighbors=neighbor_list,
    )


# ── Feedback blend ────────────────────────────────────────────────────────


def blend_probability(
    p_raw: float,
    evidence: WinRateEvidence,
    *,
    delta: float = DEFAULT_DELTA,
    min_evidence: int = DEFAULT_MIN_EVIDENCE,
) -> BlendedProbability:
    """``p_final = clip(p + delta * (win_rate - 0.5), 0, 1)``.

    The blend is skipped -- not silently applied with a made-up win rate --
    whenever the neighbourhood is too thin to mean anything. ``applied`` says
    which happened, and the reason is carried into the decision record so a
    reviewer can tell a blended p from an unblended one.
    """
    if not math.isfinite(p_raw) or not 0.0 <= p_raw <= 1.0:
        raise ValueError(f"p_raw must be a probability in [0, 1], got {p_raw!r}")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")

    if evidence.win_rate is None:
        reason = (
            "no neighbour had both routes measured and correctness-passed under "
            "this objective; predictor p used unchanged"
        )
        return BlendedProbability(p_raw, p_raw, delta, False, reason, evidence)
    if evidence.comparisons < int(min_evidence):
        reason = (
            f"only {evidence.comparisons} usable neighbour comparisons "
            f"(< {int(min_evidence)}); win rate is noise, predictor p used "
            "unchanged"
        )
        return BlendedProbability(p_raw, p_raw, delta, False, reason, evidence)

    adjusted = p_raw + delta * (evidence.win_rate - 0.5)
    p_final = min(1.0, max(0.0, adjusted))
    reason = (
        f"blended with {evidence.comparisons} neighbour comparisons "
        f"(c_first won {evidence.c_first_wins}, win_rate="
        f"{evidence.win_rate:.3f}, delta={delta})"
    )
    return BlendedProbability(p_raw, p_final, delta, True, reason, evidence)
