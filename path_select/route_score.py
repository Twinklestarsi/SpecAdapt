"""Route scoring and the specification-reasoning-necessity label (revise plan §5).

The revision plan defines the adaptive-router label by *running both routes* and
comparing their scores::

    Score = { -inf                      if any correctness gate fails
            { f(area, delay)            if the route completed

    y = 1  (c_first)     if Score_c > Score_d
    y = 0  (rtl_direct)  otherwise

Two design decisions, both deliberate and both reportable:

1.  **No tie state.** The comparison is a plain magnitude comparison, as written
    in the plan. When two finite scores are exactly equal, ``Score_c > Score_d``
    is false, so the label is ``0`` -- which is the correct reading: c_first did
    not beat rtl_direct, therefore direct RTL is sufficient. The exact-equality
    case is still recorded in ``label_basis`` so the paper can report how often
    it happens.

2.  **``unlabelable`` when both routes fail.** ``Score_c == Score_d == -inf``
    would otherwise fall through to ``y = 0`` and assert "direct RTL is
    sufficient" about a design where direct RTL did not work either. Those pairs
    are dropped before the dataset is written; ``path_select.adaptive_predictor``
    only accepts labels in ``{0, 1}``, so the filtering must happen upstream.

``label_basis`` distinguishes a real two-sided PPA comparison from a one-sided
decision where the loser was disqualified by a gate. The paper needs that ratio:
a reviewer will ask how many labels come from an actual PPA measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

__all__ = [
    "GATE_NAMES",
    "LABEL_UNLABELABLE",
    "SCORE_SCHEMA_VERSION",
    "RouteScore",
    "PairLabel",
    "score_route",
    "label_pair",
]

SCORE_SCHEMA_VERSION = "adaptive_router_score_v1"

#: Correctness gates, in evaluation order. A route must clear all of them before
#: its PPA numbers mean anything. ``equivalence`` is the gate that requires
#: ``--verification-mode jaspergold``; with ``none`` it can never be cleared, so
#: an unverified run cannot mint a label.
GATE_NAMES = ("syntax", "equivalence", "synthesis", "metric")

LABEL_UNLABELABLE = "unlabelable"

_PASS_TOKENS = {"passed", "pass", "success", "ok", "true"}
_SYNTHESIS_PASS_TOKENS = {"success", "passed", "ok"}

#: Which measured metric decides the comparison, per optimization objective. The
#: other metric is still recorded on the RouteScore for reporting, but it does
#: not enter the comparison -- the plan asks for a single magnitude comparison.
_PRIMARY_METRIC = {
    "AREA": "area",
    "TIMING": "delay_ps",
}


def _is_pass(value: Any, tokens: frozenset[str] | set[str] = _PASS_TOKENS) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in tokens


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _raw_equivalence_status(outcome: Mapping[str, Any]) -> Any:
    """Return only explicit functional-equivalence evidence.

    The pipeline has several result formats.  Newer results expose
    ``equivalence_status``; older ones expose the same JasperGold verdict as
    ``behavior_check_status`` or under ``pre_dc_verification.status``.  These
    are ordered intentionally.  In particular, ``correctness_status`` is not
    a fallback: it is allowed to describe a runtime/Memory conclusion after DC
    and does not prove equivalence.
    """
    for key in ("equivalence_status", "behavior_check_status"):
        value = outcome.get(key)
        if value is not None and (not isinstance(value, str) or value.strip()):
            return value
    pre_dc = outcome.get("pre_dc_verification")
    if isinstance(pre_dc, Mapping):
        value = pre_dc.get("status")
        if value is not None and (not isinstance(value, str) or value.strip()):
            return value
    return None


@dataclass
class RouteScore:
    """One route's gate outcomes and its resulting Score."""

    route: str
    objective: str
    score: float
    gates: Dict[str, bool] = field(default_factory=dict)
    failed_gate: str = ""
    area: float | None = None
    delay_ps: float | None = None
    primary_metric: str = ""
    primary_value: float | None = None
    verification_mode: str = ""
    equivalence_status: str = ""

    @property
    def scored(self) -> bool:
        """True when every gate passed and Score is a finite number."""
        return math.isfinite(self.score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "objective": self.objective,
            # JSON has no -inf; keep the numeric field JSON-safe and carry the
            # real verdict in `scored`.
            "score": self.score if self.scored else None,
            "score_is_neg_inf": not self.scored,
            "scored": self.scored,
            "gates": dict(self.gates),
            "failed_gate": self.failed_gate,
            "area": self.area,
            "delay_ps": self.delay_ps,
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "verification_mode": self.verification_mode,
            "equivalence_status": self.equivalence_status,
        }


def score_route(
    outcome: Mapping[str, Any],
    *,
    route: str,
    objective: str,
    require_equivalence: bool = True,
) -> RouteScore:
    """Turn one route's raw run record into a Score.

    ``outcome`` is read loosely so that a pipeline summary entry, a phase-6
    ``PathRunManifest`` dict, and a hand-written test fixture all work:

    ==================  ==========================================================
    gate                accepted keys
    ==================  ==========================================================
    ``syntax``          ``syntax_status``
    ``equivalence``     ``equivalence_status``, ``behavior_check_status``,
                        ``pre_dc_verification.status``; the route must also
                        declare ``verification_mode=jaspergold``
    ``synthesis``       ``synthesis_status``, ``dc_status``
    ``metric``          ``area``; ``delay_ps`` (with the historical
                        ``data_arrival_time_ps`` alias accepted)
    ==================  ==========================================================

    ``require_equivalence=False`` exists only for flows that deliberately run
    without JasperGold. It produces scores that must NOT be used as training
    labels, and the caller is responsible for saying so in its manifest.
    """
    objective = str(objective or "AREA").strip().upper()
    primary_metric = _PRIMARY_METRIC.get(objective)
    if primary_metric is None:
        raise ValueError(
            f"Unsupported objective for scoring: {objective!r} "
            f"(expected one of {sorted(_PRIMARY_METRIC)})"
        )

    area = _as_float(outcome.get("area"))
    # ``delay_ps`` is the public timing metric.  Older manifests only carry
    # DC's ``data_arrival_time_ps`` spelling, so use it strictly as a
    # compatibility fallback; never fall back to slack.
    delay_ps = _as_float(
        outcome.get("delay_ps")
        if outcome.get("delay_ps") is not None
        else outcome.get("data_arrival_time_ps")
    )
    primary_value = area if primary_metric == "area" else delay_ps

    # Keep the evidence chain deliberately narrow.  ``correctness_status`` is
    # a runtime/Memory result and may be set to ``passed`` after DC even when
    # no functional equivalence check ran.  It is therefore never an input to
    # the training-label equivalence gate.  The explicit field wins when both
    # fields are present; this prevents a contradictory secondary field from
    # laundering an explicit JasperGold result.
    equivalence_raw = _raw_equivalence_status(outcome)
    verification_mode = str(outcome.get("verification_mode") or "").strip().lower()
    gates = {
        "syntax": _is_pass(outcome.get("syntax_status")),
        # A positive status is insufficient on its own: only the JasperGold
        # route is allowed to establish functional equivalence for a label.
        "equivalence": (
            verification_mode == "jaspergold" and _is_pass(equivalence_raw)
            if require_equivalence
            else True
        ),
        "synthesis": _is_pass(outcome.get("synthesis_status"), _SYNTHESIS_PASS_TOKENS)
        or _is_pass(outcome.get("dc_status"), _SYNTHESIS_PASS_TOKENS),
        "metric": primary_value is not None,
    }

    failed_gate = next((name for name in GATE_NAMES if not gates[name]), "")
    if failed_gate:
        score = -math.inf
    else:
        # Lower area / lower delay is better, so negate to make "greater Score
        # wins" hold. This is the whole of f(area, delay): a single magnitude
        # comparison on the objective's own metric, per the plan.
        score = -float(primary_value)  # type: ignore[arg-type]

    return RouteScore(
        route=route,
        objective=objective,
        score=score,
        gates=gates,
        failed_gate=failed_gate,
        area=area,
        delay_ps=delay_ps,
        primary_metric=primary_metric,
        primary_value=primary_value,
        verification_mode=verification_mode,
        equivalence_status=str(equivalence_raw or ""),
    )


@dataclass
class PairLabel:
    """The label derived from one (c_first, rtl_direct) pair."""

    label: int | str
    label_basis: str
    label_reason: str
    trainable: bool
    c_first: RouteScore
    rtl_direct: RouteScore
    relative_gap_pct: float | None = None
    exact_equal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCORE_SCHEMA_VERSION,
            "label": self.label,
            "label_name": (
                self.label
                if isinstance(self.label, str)
                else {0: "rtl_direct", 1: "c_first"}[self.label]
            ),
            "label_basis": self.label_basis,
            "label_reason": self.label_reason,
            "trainable": self.trainable,
            "relative_gap_pct": self.relative_gap_pct,
            "exact_equal": self.exact_equal,
            "routes": {
                "c_first": self.c_first.to_dict(),
                "rtl_direct": self.rtl_direct.to_dict(),
            },
        }


def label_pair(
    c_first: Mapping[str, Any] | RouteScore,
    rtl_direct: Mapping[str, Any] | RouteScore,
    *,
    objective: str = "AREA",
    require_equivalence: bool = True,
) -> PairLabel:
    """Apply the §5 rule to a route pair.

    Returns a :class:`PairLabel` whose ``label`` is ``1``, ``0``, or the string
    ``"unlabelable"``. Only ``trainable=True`` rows may be written into a
    dataset npz.
    """
    c = (
        c_first
        if isinstance(c_first, RouteScore)
        else score_route(
            c_first,
            route="c_first",
            objective=objective,
            require_equivalence=require_equivalence,
        )
    )
    d = (
        rtl_direct
        if isinstance(rtl_direct, RouteScore)
        else score_route(
            rtl_direct,
            route="rtl_direct",
            objective=objective,
            require_equivalence=require_equivalence,
        )
    )

    if not c.scored and not d.scored:
        return PairLabel(
            label=LABEL_UNLABELABLE,
            label_basis="neither_route_scored",
            label_reason=(
                f"both routes hit a gate failure "
                f"(c_first:{c.failed_gate}, rtl_direct:{d.failed_gate}); "
                "Score_c == Score_d == -inf carries no information about which "
                "path was necessary"
            ),
            trainable=False,
            c_first=c,
            rtl_direct=d,
        )

    if c.scored and not d.scored:
        return PairLabel(
            label=1,
            label_basis="c_first_only",
            label_reason=(
                f"rtl_direct disqualified at the {d.failed_gate} gate, so "
                "Score_d = -inf < Score_c"
            ),
            trainable=True,
            c_first=c,
            rtl_direct=d,
        )

    if d.scored and not c.scored:
        return PairLabel(
            label=0,
            label_basis="rtl_direct_only",
            label_reason=(
                f"c_first disqualified at the {c.failed_gate} gate, so "
                "Score_c = -inf < Score_d"
            ),
            trainable=True,
            c_first=c,
            rtl_direct=d,
        )

    # Both routes produced a real measurement: this is the two-sided PPA
    # comparison the paper's label definition is really about.
    c_value = float(c.primary_value)  # type: ignore[arg-type]
    d_value = float(d.primary_value)  # type: ignore[arg-type]
    exact_equal = c_value == d_value
    denominator = max(min(abs(c_value), abs(d_value)), 1e-12)
    relative_gap_pct = abs(c_value - d_value) / denominator * 100.0

    if c.score > d.score:
        label, reason = 1, (
            f"both routes scored; c_first {c.primary_metric}={c_value:.6g} beats "
            f"rtl_direct {d_value:.6g}"
        )
    else:
        label, reason = 0, (
            "both routes scored; c_first did not beat rtl_direct "
            f"({c.primary_metric}: {c_value:.6g} vs {d_value:.6g}"
            f"{', exactly equal' if exact_equal else ''})"
        )

    return PairLabel(
        label=label,
        label_basis="both_routes_scored",
        label_reason=reason,
        trainable=True,
        c_first=c,
        rtl_direct=d,
        relative_gap_pct=relative_gap_pct,
        exact_equal=exact_equal,
    )
