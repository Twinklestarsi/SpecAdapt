from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _usage_value(usage: Any, key: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def usage_from_response(
    *,
    stage: str,
    model: str,
    response: Any,
    attempt: int | None = None,
) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    record: Dict[str, Any] = {
        "stage": stage,
        "model": model,
    }
    if attempt is not None:
        record["attempt"] = attempt

    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")

    if prompt_tokens is not None:
        record["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        record["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        record["total_tokens"] = total_tokens

    if len(record) == (3 if attempt is not None else 2):
        record["available"] = False
    else:
        record["available"] = True
    return record


def token_totals(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for record in records:
        if not record.get("available", True):
            continue
        for key in totals:
            value = record.get(key)
            if isinstance(value, int):
                totals[key] += value
    return totals


def merge_token_usage(*groups: Iterable[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for group in groups:
        if not group:
            continue
        merged.extend(dict(item) for item in group)
    return merged
