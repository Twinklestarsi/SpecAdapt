"""Shared optional controls for OpenAI-compatible chat requests."""

from __future__ import annotations

import os
import math
import re
from typing import Any, Dict, Union


Number = Union[int, float]


def _positive_timeout(raw: str, variable: str) -> Number:
    """Parse a positive timeout value from an environment variable."""

    value_text = raw.strip()
    try:
        value = float(value_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{variable} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{variable} must be a positive number")
    # Keep integral values as ints for predictable OpenAI client kwargs.
    return int(value) if value.is_integer() else value


def _non_negative_int(raw: str, variable: str) -> int:
    """Parse a non-negative integer from an environment variable."""

    value_text = raw.strip()
    if not re.fullmatch(r"\+?\d+", value_text):
        raise ValueError(f"{variable} must be a non-negative integer")
    value = int(value_text)
    if value < 0:  # Kept for clarity if the accepted syntax changes later.
        raise ValueError(f"{variable} must be a non-negative integer")
    return value


def openai_client_kwargs() -> Dict[str, Number]:
    """Return shared kwargs for every OpenAI-compatible pipeline client.

    ``OPENAI_TIMEOUT`` takes precedence over the legacy ``CLOUD_TIMEOUT``;
    both default to 120 seconds.  SDK retries are deliberately disabled by
    default so that each pipeline's explicit retry budget remains authoritative.
    """

    timeout_name = "OPENAI_TIMEOUT"
    timeout_raw = os.environ.get(timeout_name, "").strip()
    if not timeout_raw:
        timeout_name = "CLOUD_TIMEOUT"
        timeout_raw = os.environ.get(timeout_name, "").strip()
    timeout = _positive_timeout(timeout_raw or "120", timeout_name)

    retries_name = "OPENAI_SDK_MAX_RETRIES"
    retries_raw = os.environ.get(retries_name, "").strip()
    max_retries = _non_negative_int(retries_raw or "0", retries_name)

    return {"timeout": timeout, "max_retries": max_retries}


def seed_kwargs() -> Dict[str, int]:
    """Return a request seed only when the experiment controller set one."""

    raw = os.environ.get("OPENAI_SEED", "").strip()
    if not raw:
        return {}
    try:
        return {"seed": int(raw)}
    except ValueError as exc:
        raise ValueError("OPENAI_SEED must be an integer") from exc


_UNLIMITED_TOKEN_SPELLINGS = frozenset(
    {"", "0", "none", "null", "unlimited", "unbounded", "off", "nolimit", "no_limit"}
)


def _unlimited_tokens_requested(raw_value: object) -> bool:
    """Return True when a token-budget spelling asks for no explicit cap."""

    if isinstance(raw_value, bool):
        # ``True``/``False`` are invalid budgets, never "unlimited".
        return False
    if isinstance(raw_value, (int, float)):
        return raw_value == 0
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in _UNLIMITED_TOKEN_SPELLINGS
    return False


def completion_max_tokens(max_tokens: int | str | None = None) -> int | None:
    """Resolve the completion token budget for an LLM request.

    An explicit ``max_tokens`` argument takes precedence over environment
    configuration.  Otherwise ``OPENAI_MAX_TOKENS`` takes precedence over the
    legacy ``CLOUD_MAX_TOKENS`` variable, with a default of 4096 when neither
    variable is set.  Every configured value must be a positive integer.

    A value of ``0``, ``none``, ``null``, ``unlimited``, ``unbounded``, ``off``
    or ``nolimit`` (case-insensitive) returns ``None``, meaning "send no
    ``max_tokens`` at all" so the server applies the model's own ceiling.
    Callers must route that through :func:`completion_token_kwargs` rather than
    passing the result straight to ``max_tokens=``, because an explicit JSON
    ``null`` is rejected by parts of the OpenAI-compatible ecosystem.
    """

    if max_tokens is not None:
        raw_value: object = max_tokens
        variable = "max_tokens"
    elif "OPENAI_MAX_TOKENS" in os.environ:
        raw_value = os.environ["OPENAI_MAX_TOKENS"]
        variable = "OPENAI_MAX_TOKENS"
    elif "CLOUD_MAX_TOKENS" in os.environ:
        raw_value = os.environ["CLOUD_MAX_TOKENS"]
        variable = "CLOUD_MAX_TOKENS"
    else:
        return 4096

    if _unlimited_tokens_requested(raw_value):
        return None

    if isinstance(raw_value, bool):
        raise ValueError(f"{variable} must be a positive integer")
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, str):
        value_text = raw_value.strip()
        if not re.fullmatch(r"\+?\d+", value_text):
            raise ValueError(f"{variable} must be a positive integer")
        value = int(value_text)
    else:
        raise ValueError(f"{variable} must be a positive integer")

    if value < 1:
        raise ValueError(f"{variable} must be a positive integer")
    return value


def completion_token_kwargs(max_tokens: int | str | None = None) -> Dict[str, Any]:
    """Return ``{"max_tokens": n}``, or ``{}`` when no explicit cap is wanted.

    Use this as ``**completion_token_kwargs(max_tokens)`` at every
    ``chat.completions.create`` call site so that an "unlimited" configuration
    omits the parameter entirely instead of sending ``max_tokens=null``.
    """

    value = completion_max_tokens(max_tokens)
    return {} if value is None else {"max_tokens": value}


class TruncatedCompletionError(RuntimeError):
    """Raised when a completion stopped at the token budget, not at its end.

    A truncated completion is a *transport* failure, not a design failure: the
    model never finished writing, so the payload is a prefix of an answer.
    Feeding that prefix to a syntax gate produces a misleading error and an
    identical retry, which is exactly the loop this type exists to break.
    """


def completion_finish_reason(response: Any) -> str:
    """Return the first choice's ``finish_reason``, or ``""`` when unavailable."""

    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    try:
        reason = getattr(choices[0], "finish_reason", "")
    except (IndexError, TypeError, AttributeError):
        return ""
    return str(reason or "")


def completion_was_truncated(response: Any) -> bool:
    """Return True when the provider cut the answer off at the token budget.

    ``finish_reason == "length"`` is the OpenAI-compatible spelling.  Any other
    value (``stop``, ``tool_calls``, ``content_filter``, ...) or a missing field
    is treated as *not* truncated, so an unusual provider never turns a normal
    answer into a spurious failure.
    """

    return completion_finish_reason(response) == "length"


def truncated_completion_message(response: Any, stage: str = "") -> str:
    """Build the operator-facing and model-facing text for a cut-off answer."""

    reason = completion_finish_reason(response) or "unknown"
    prefix = f"{stage}: " if stage else ""
    return (
        f"{prefix}LLM output was truncated at the token budget "
        f"(finish_reason={reason}); the returned RTL is an incomplete prefix, "
        f"not a finished module."
    )


def qwen_thinking_kwargs(model: str) -> Dict[str, Any]:
    """Return optional Qwen chat-template kwargs for a model request.

    Thinking control is intentionally opt-in: non-Qwen models and an unset
    ``QWEN_ENABLE_THINKING`` variable produce no extra request parameters.
    When the variable is set for a Qwen model, only common boolean spellings
    are accepted and invalid values raise ``ValueError``.
    """

    if "qwen" not in model.lower() or "QWEN_ENABLE_THINKING" not in os.environ:
        return {}

    raw = os.environ["QWEN_ENABLE_THINKING"].strip().lower()
    if raw in {"true", "1", "yes", "on"}:
        enabled = True
    elif raw in {"false", "0", "no", "off"}:
        enabled = False
    else:
        raise ValueError(
            "QWEN_ENABLE_THINKING must be one of true/false/1/0/yes/no/on/off"
        )

    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": enabled}}}


def deepseek_thinking_kwargs(model: str) -> Dict[str, Any]:
    """Return optional DeepSeek thinking controls for a model request.

    The setting is opt-in: non-DeepSeek models and an unset
    ``DEEPSEEK_ENABLE_THINKING`` variable produce no extra request parameters.
    When enabled for a DeepSeek model, only common boolean spellings are
    accepted and invalid values raise ``ValueError``.  An optional
    ``DEEPSEEK_REASONING_EFFORT`` is included as a top-level request option;
    it accepts only ``low``, ``high``, or ``max``.  A configured effort is
    ignored while thinking is disabled, so ``reasoning_effort`` is never sent
    in that mode.
    """

    if "deepseek" not in model.lower() or "DEEPSEEK_ENABLE_THINKING" not in os.environ:
        return {}

    raw = os.environ["DEEPSEEK_ENABLE_THINKING"].strip().lower()
    if raw in {"true", "1", "yes", "on"}:
        thinking_type = "enabled"
    elif raw in {"false", "0", "no", "off"}:
        thinking_type = "disabled"
    else:
        raise ValueError(
            "DEEPSEEK_ENABLE_THINKING must be one of true/false/1/0/yes/no/on/off"
        )

    request_kwargs: Dict[str, Any] = {
        "extra_body": {"thinking": {"type": thinking_type}}
    }
    if thinking_type == "disabled":
        return request_kwargs

    effort_raw = os.environ.get("DEEPSEEK_REASONING_EFFORT", "").strip().lower()
    if not effort_raw:
        return request_kwargs
    if effort_raw not in {"low", "high", "max"}:
        raise ValueError(
            "DEEPSEEK_REASONING_EFFORT must be one of low/high/max"
        )
    request_kwargs["reasoning_effort"] = effort_raw
    return request_kwargs
