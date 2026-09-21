"""
llm_features.py — LLM client, response parsing, and feature extraction.

Handles communication with the OpenAI-compatible API endpoint,
response parsing with fallbacks, and confidence assignment.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from llm_request import (
    completion_max_tokens,
    completion_token_kwargs,
    deepseek_thinking_kwargs,
    openai_client_kwargs,
    qwen_thinking_kwargs,
    seed_kwargs,
)
from spec_analyze.schema import validate_features
from spec_analyze.confidence_rules import score_confidence
from spec_analyze.prompts import (
    SYSTEM_PROMPT,
    format_verilog_prompt,
    format_spec_prompt,
    format_mixed_prompt,
)

# ── Client management ─────────────────────────────────────────────────

_client: Optional[OpenAI] = None


def init_client(env_path: Optional[Path] = None) -> OpenAI:
    """Initialize the OpenAI client from .env."""
    global _client
    if env_path:
        load_dotenv(env_path, override=True)
    _client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        **openai_client_kwargs(),
    )
    return _client


def get_client() -> OpenAI:
    global _client
    if _client is None:
        raise RuntimeError("LLM client not initialized. Call init_client() first.")
    return _client


def get_model() -> str:
    return os.environ.get("OPENAI_MODEL", "Kimi-K2.5")


# ── LLM call ──────────────────────────────────────────────────────────

def _call_llm(
    prompt: str,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    timeout: int | float | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.3,
) -> Tuple[str, Dict[str, Any]]:
    """Call the LLM with retries. Returns raw response text and token usage.

    By default the request inherits the timeout configured on the shared
    OpenAI client.  A caller may still pass ``timeout`` for a deliberately
    shorter one-off request, but the normal pipeline no longer hard-codes a
    120-second override here.
    """
    client = client or get_client()
    model = model or get_model()
    max_tokens = completion_max_tokens(max_tokens)

    for attempt in range(max_retries):
        try:
            request_options: Dict[str, Any] = dict(completion_token_kwargs(max_tokens))
            if timeout is not None:
                request_options["timeout"] = timeout
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                **seed_kwargs(),
                **qwen_thinking_kwargs(model),
                **deepseek_thinking_kwargs(model),
                **request_options,
            )
            usage = _normalize_usage(getattr(response, "usage", None))
            return response.choices[0].message.content.strip(), usage
        except Exception as e:
            print(f"    LLM attempt {attempt + 1}/{max_retries} failed: {e}",
                  file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return "", {}


def _normalize_usage(usage: Any) -> Dict[str, Any]:
    """Convert OpenAI/OpenAI-compatible usage objects into JSON-safe dicts."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        raw_usage = usage
    elif hasattr(usage, "model_dump"):
        raw_usage = usage.model_dump(exclude_none=True)
    elif hasattr(usage, "to_dict"):
        raw_usage = usage.to_dict()
    else:
        raw_usage = {
            name: getattr(usage, name)
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            if hasattr(usage, name)
        }

    return {
        str(key): value
        for key, value in raw_usage.items()
        if value is not None
    }


# ── Response parsing ──────────────────────────────────────────────────

_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)


def _strip_think_sections(raw: str) -> str:
    """Remove model reasoning markers without changing JSON payload text.

    A complete ``<think>...</think>`` section is discarded.  If the model
    emitted only a closing marker, everything before the last closing marker
    is reasoning and only the suffix is retained.  For an unmatched opening
    marker, the suffix after the last opening marker is retained; this lets a
    response that omitted ``</think>`` still expose a final JSON answer.
    """
    text = raw.strip()
    if not text:
        return ""

    # Remove all complete reasoning sections first.  Keeping the text on
    # either side is important when the answer follows the closing marker.
    text = _THINK_BLOCK_RE.sub("", text)

    # A closing marker without a matching opening marker is the boundary
    # between reasoning and the answer.  Use the final one in case the model
    # repeated the marker.
    close_matches = list(_THINK_CLOSE_RE.finditer(text))
    if close_matches:
        text = text[close_matches[-1].end() :]

    # If only an opening marker remains, treat its suffix as the possible
    # answer.  This is deliberately extraction-only: no content is invented
    # or rewritten.
    open_matches = list(_THINK_OPEN_RE.finditer(text))
    if open_matches:
        text = text[open_matches[-1].end() :]

    return text.strip()


def _decoded_json_values(text: str) -> list[tuple[int, int, Any]]:
    """Decode JSON values beginning at every possible object/array marker."""
    decoder = json.JSONDecoder()
    values: list[tuple[int, int, Any]] = []
    for marker in re.finditer(r"[\{\[]", text):
        try:
            value, end = decoder.raw_decode(text, marker.start())
        except json.JSONDecodeError:
            continue
        values.append((marker.start(), end, value))
    return values


def _top_level_object_candidates(
    text: str,
) -> list[tuple[int, int, Dict[str, Any]]]:
    """Return non-nested object candidates in response order.

    ``raw_decode`` also succeeds at the opening brace of a nested object.  A
    nested candidate must not hide its enclosing object (for example, the
    ``interface`` object inside a feature response), so candidates contained
    in another decoded object are removed.  Objects contained in a decoded
    array are likewise ignored when the array is the response-level value;
    this preserves the rule that a non-dictionary top-level JSON answer is
    not accepted as a feature dictionary.
    """
    values = _decoded_json_values(text)
    objects = [
        (start, end, value)
        for start, end, value in values
        if isinstance(value, dict)
    ]
    arrays = [
        (start, end)
        for start, end, value in values
        if isinstance(value, list)
    ]

    outer_objects: list[tuple[int, int, Dict[str, Any]]] = []
    for candidate in objects:
        start, end, _value = candidate
        contained_in_object = any(
            parent_start <= start
            and end <= parent_end
            and (parent_start, parent_end) != (start, end)
            for parent_start, parent_end, _parent_value in objects
        )
        if contained_in_object:
            continue

        contained_in_array = any(
            array_start <= start and end <= array_end
            for array_start, array_end in arrays
        )
        if contained_in_array:
            continue

        outer_objects.append(candidate)

    return outer_objects


def _parse_json(raw: str) -> Optional[Dict]:
    """Extract the last decodable dictionary from an LLM response.

    The parser accepts plain JSON, fenced JSON, and JSON embedded in answer
    text.  It intentionally only extracts an object: it never repairs or
    guesses at malformed JSON content.
    """
    text = _strip_think_sections(raw or "")
    if not text:
        return None

    # Preserve the ordinary pure-JSON path and, importantly, reject a pure
    # non-dictionary answer instead of returning a dictionary nested in it.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    else:
        return parsed if isinstance(parsed, dict) else None

    candidates = _top_level_object_candidates(text)
    if not candidates:
        return None

    # Candidates are discovered in source order; the final one is the final
    # decodable dictionary answer, including when earlier examples exist.
    return candidates[-1][2]


# ── Public extraction functions ───────────────────────────────────────

def extract_from_verilog(
    verilog_src: str,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Tuple[Dict, Dict, str, Dict[str, Any]]:
    """
    Extract LLM features from Verilog source code.

    Returns:
        (features, confidence, raw_response, token_usage)
    """
    prompt = format_verilog_prompt(verilog_src)
    raw, usage = _call_llm(prompt, client, model)
    parsed = _parse_json(raw)

    if parsed is None:
        return {}, {}, raw, usage

    features, confidence = validate_features(parsed)
    return features, confidence, raw, usage


def extract_from_spec(
    spec_text: str,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Tuple[Dict, Dict, str, Dict, Dict[str, Any]]:
    """
    Extract LLM features from a natural-language specification.
    Confidence is scored by cross-checking spec text against LLM output.

    Returns:
        (features, confidence, raw_response, evidence, token_usage)
    """
    prompt = format_spec_prompt(spec_text)
    raw, usage = _call_llm(prompt, client, model)
    parsed = _parse_json(raw)

    if parsed is None:
        return {}, {}, raw, {}, usage

    features, _ = validate_features(parsed)
    confidence, _, evidence = score_confidence(spec_text, features)
    return features, confidence, raw, evidence, usage


def extract_from_mixed(
    verilog_src: str,
    spec_text: str,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Tuple[Dict, Dict, str, Dict, Dict[str, Any]]:
    """
    Extract LLM features from both Verilog source and spec text.

    Returns:
        (features, confidence, raw_response, evidence, token_usage)
    """
    prompt = format_mixed_prompt(verilog_src, spec_text)
    raw, usage = _call_llm(prompt, client, model)
    parsed = _parse_json(raw)

    if parsed is None:
        return {}, {}, raw, {}, usage

    features, confidence = validate_features(parsed)
    # Mixed mode: Verilog grounds the features, so confidence stays high
    return features, confidence, raw, {}, usage
