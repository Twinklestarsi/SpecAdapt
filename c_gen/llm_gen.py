"""
llm_gen.py — LLM-based C code generation from Verilog or natural-language spec.

Three generation modes:
  1. spec-only:    Generate hardware-oriented C from natural-language specification
  2. verilog-only: Convert Verilog RTL to hardware-oriented C
  3. mixed:        Use Verilog as ground truth, spec as context

All modes preserve cycle-level state using either explicit state storage or
static variables suitable for CDFG analysis and downstream AI RTL generation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
from llm_request import (
    completion_max_tokens,
    completion_token_kwargs,
    deepseek_thinking_kwargs,
    openai_client_kwargs,
    qwen_thinking_kwargs,
    seed_kwargs,
)
from llm_code_sanitize import sanitize_c_response
from openai import OpenAI

from token_counter import TokenUsage

logger = logging.getLogger(__name__)


_sanitize_c_response = sanitize_c_response


# ─── Prompt templates ─────────────────────────────────────────────────

_SPEC_ONLY_SYSTEM = """\
You are an expert in hardware-oriented C generation for CDFG optimization and downstream RTL generation.

Given a natural-language specification, generate a complete cycle-step C implementation.

Requirements:
1. Use `#include <stdbool.h>` and `bool` for every 1-bit data port and 1-bit state element. Use `#include <stdint.h>` for standard 8/16/32/64-bit values.
2. Top function signature: `void <funcname>(scalar inputs, pointer outputs)`
3. All state (registers) must be in a `static struct state_elements_<funcname>` instance
4. No malloc, no recursion, no floating-point (unless explicitly required)
5. Pointer parameters are outputs — write final values via `*param = value;`
6. Use simple control flow and constant loop bounds when possible so LLVM/CDFG analysis stays explicit
7. Preserve every explicitly specified port name and width. Never widen a 1-bit signal to uint8_t.
8. Do not add AXI, stream, valid, ready, or block-control arguments unless they are functional ports explicitly required by the specification.
9. Treat the `interface` object in the feature JSON as authoritative. Do not put ports whose role is `clock` or `reset` in the C function signature; downstream AI RTL generation restores them from the interface and behavioral contracts.
10. Initialize static state to its specified reset value. Do not implement physical reset as a normal C data input.
11. For non-standard widths such as 5, 13, 27, or 63 bits, use the smallest standard unsigned container and mask every assignment/state update to the declared hardware width so upper bits synthesize away.
12. The natural-language specification is the primary behavioral authority. Use
    feature JSON to make explicit interface/cycle details precise, but never let
    an inferred feature silently override behavior explicitly stated by the spec.
13. For every sequential design, model one active clock edge with strict
    old-state/next-state semantics:
    - Copy the complete persistent state into an immutable local `old_state`.
    - Initialize a separate local `next_state` from `old_state`.
    - Compute every register update using only inputs and `old_state`; never read
      a struct field after modifying that same persistent struct in the call.
    - Commit the persistent struct exactly once, after all next-state values have
      been computed.
    This is required to preserve Verilog nonblocking-assignment semantics.
14. Treat each function call as one active clock edge followed by combinational
    settling. Registered outputs must be persistent state fields. Valid/data
    pipeline stages, output latency, FSM priority, and counter boundary behavior
    must match the specification and behavioral_contract exactly.
15. Before returning, perform a cycle-semantics audit covering reset values,
    current-versus-next state, simultaneous register updates, enable bubbles,
    back-to-back valid transactions, FSM transition priority, and off-by-one
    counter thresholds.

Output only the C code, no explanations.
"""

_SPEC_ONLY_USER = """\
Generate hardware-oriented intermediate C code for the following specification:

{spec_text}

Additional context from feature extraction:
{features_json}

Generate complete C code with:
- Function name: {funcname}
- state_elements_{funcname} struct for all state
- Pointer outputs for all results
- Data ports only in the function signature; omit interface clock/reset ports
"""


_SPEC_SEMANTIC_REVIEW_SYSTEM = """\
You are the semantic review gate for a hardware-oriented cycle-step C model.

Audit the candidate C against the natural-language specification and frozen
feature contract, then return one complete corrected C translation unit.

Authority and safety rules:
1. The explicit natural-language specification is the primary behavioral
   authority. The frozen feature contract refines explicit interface and timing
   facts, but an inferred field must not override an explicit statement.
2. Do not use or assume a golden RTL implementation.
3. Preserve the exact C function name, data-port names, widths, pointer outputs,
   state struct naming convention, and reset initial values.
4. For sequential designs, enforce strict old-state/next-state behavior. Take an
   `old_state` snapshot, compute a separate `next_state` only from old state and
   current inputs, and commit persistent state exactly once at the end.
5. Registered outputs must live in persistent state. Preserve the exact valid
   pipeline, output latency, enable-bubble behavior, back-to-back transactions,
   FSM priority, direction-sensitive conditions, and counter boundary semantics.
6. A function call represents one active clock edge followed by combinational
   settling. Pointer outputs must expose the post-edge behavior required by the
   specification without collapsing a register stage.
7. Do not optimize for PPA in this review. Only repair semantic discrepancies.
8. Use C99, fixed-width unsigned types, bounded loops, no dynamic allocation,
   no recursion, and no HLS pragmas.
9. Return C source only, with no markdown fences or explanation.
"""


_SPEC_SEMANTIC_REVIEW_USER = """\
Perform the mandatory cycle-semantics review for this Spec-to-C result.

Function name: {funcname}

Natural-language specification:
{spec_text}

Frozen feature contract:
{features_json}

Candidate C:
```c
{c_code}
```

Audit checklist:
- reset initialization and reset semantics recoverable downstream
- old-state versus next-state separation
- simultaneous register updates and nonblocking-assignment equivalence
- registered/combinational output classification
- exact output latency and valid/data alignment
- enable bubbles and consecutive valid inputs
- FSM transition priority and direction-sensitive inputs
- counter start point, saturation, comparison threshold, and terminal transition

Return the complete corrected C translation unit only.
"""

_VERILOG_ONLY_SYSTEM = """\
You are an expert in converting Verilog RTL to hardware-oriented cycle-step C.

Given Verilog source, generate functionally equivalent C that preserves the design's behavior.

Requirements:
1. Use `#include <stdbool.h>` and `bool` for every 1-bit data port and 1-bit state element. Use `#include <stdint.h>` for standard 8/16/32/64-bit values matching Verilog widths.
2. Top function signature: `void <funcname>(scalar inputs, pointer outputs)`
3. All `reg` declarations → fields in `static struct state_elements_<funcname>`
4. Combinational logic → direct assignments
5. Sequential logic (always @posedge) → update struct fields, then write to pointers
6. No malloc, no recursion, no floating-point
7. Preserve port names from Verilog module
8. Never add AXI or HLS block-control arguments. Never widen a 1-bit signal to uint8_t.
9. Omit Verilog clock and reset ports from the C function signature. Initialize static state to the Verilog reset values; downstream AI RTL generation restores the physical clock/reset from the interface contract.
10. For non-standard widths, use the smallest standard unsigned container and mask every assignment/state update to the exact Verilog width.

Output only the C code, no explanations.
"""

_VERILOG_ONLY_USER = """\
Convert the following Verilog module to hardware-oriented intermediate C:

```verilog
{verilog_source}
```

Generate C code with:
- Function name matching the module name
- state_elements_<funcname> struct for all registers
- Pointer outputs for all output ports
"""

_MIXED_SYSTEM = """\
You are an expert in hardware-oriented C generation for CDFG optimization and downstream RTL generation.

Given both Verilog RTL (ground truth) and a natural-language specification (context),
generate cycle-step C that implements the Verilog's functionality.

Use the Verilog for structural accuracy (port names, bit widths, state elements).
Use the spec for understanding intent and edge cases.

Requirements:
1. Use `#include <stdbool.h>` and `bool` for every 1-bit data port and 1-bit state element. Use `#include <stdint.h>` for standard 8/16/32/64-bit values.
2. Top function signature: `void <funcname>(scalar inputs, pointer outputs)`
3. All `reg` → fields in `static struct state_elements_<funcname>`
4. Preserve Verilog port names and behavior
5. No malloc, no recursion, no floating-point
6. Never add AXI or HLS block-control arguments. Never widen a 1-bit signal to uint8_t.
7. Omit Verilog clock and reset ports from the C function signature. Initialize static state to the Verilog reset values; downstream AI RTL generation restores physical clock/reset behavior.
8. For non-standard widths, use the smallest standard unsigned container and mask every assignment/state update to the exact Verilog width.

Output only the C code, no explanations.
"""

_MIXED_USER = """\
Generate hardware-oriented intermediate C code using:

Verilog (ground truth):
```verilog
{verilog_source}
```

Specification (context):
{spec_text}

Generate C code with:
- Function name matching the Verilog module
- state_elements_<funcname> struct for all registers
- Pointer outputs matching Verilog output ports
"""


# ─── LLM client ───────────────────────────────────────────────────────

def _get_llm_client(env_path: str | Path = ".env") -> Tuple[Optional[OpenAI], Optional[str]]:
    """Load LLM client from .env. Returns (client, model) or (None, None) on failure."""
    try:
        load_dotenv(env_path, override=True)
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=(
                os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE_URL")
                or os.environ.get("OPENAI_API_BASE")
            ),
            **openai_client_kwargs(),
        )
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return client, model
    except ValueError:
        raise
    except Exception as e:
        logger.error("LLM client init failed: %s", e)
        return None, None


# ─── Generation functions ─────────────────────────────────────────────

def generate_from_spec(
    spec_text: str,
    funcname: str,
    features: dict,
    max_retries: int = 2,
    env_path: str | Path = ".env",
    max_tokens: int | None = None,
) -> Tuple[Optional[str], TokenUsage]:
    """
    Generate C code from natural-language specification.

    Args:
        spec_text: Natural-language description of the design
        funcname: Desired function name
        features: Dict of extracted features (from Module 1)
        max_retries: Number of retry attempts on failure
        env_path: Path to .env file
        max_tokens: Optional explicit completion token budget

    Returns:
        (c_code, token_usage): C code string or None on failure
    """
    client, model = _get_llm_client(env_path)
    if client is None:
        logger.error("Cannot proceed: LLM client is None (check .env)")
        return None, TokenUsage()

    max_tokens = completion_max_tokens(max_tokens)
    usage = TokenUsage()
    features_json = json.dumps(features, indent=2)
    user_msg = _SPEC_ONLY_USER.format(
        spec_text=spec_text,
        features_json=features_json,
        funcname=funcname,
    )

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SPEC_ONLY_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                **completion_token_kwargs(max_tokens),
                **seed_kwargs(),
                **qwen_thinking_kwargs(model),
                **deepseek_thinking_kwargs(model),
            )

            if response.usage:
                usage.prompt_tokens += response.usage.prompt_tokens
                usage.completion_tokens += response.usage.completion_tokens
                usage.total_tokens += response.usage.total_tokens

            text = response.choices[0].message.content or ""
            return _sanitize_c_response(text), usage

        except Exception as e:
            logger.warning("LLM call attempt %d failed: %s", attempt + 1, e)
            usage.retries += 1
            if attempt == max_retries:
                return None, usage

    return None, usage


def review_from_spec(
    spec_text: str,
    funcname: str,
    features: dict,
    c_code: str,
    max_retries: int = 1,
    env_path: str | Path = ".env",
    max_tokens: int | None = None,
) -> Tuple[Optional[str], TokenUsage]:
    """Run a serial semantic audit over one syntax-valid Spec-to-C candidate."""
    client, model = _get_llm_client(env_path)
    if client is None:
        logger.error("Cannot review Spec-to-C output: LLM client is None")
        return None, TokenUsage()

    max_tokens = completion_max_tokens(max_tokens)
    usage = TokenUsage()
    user_msg = _SPEC_SEMANTIC_REVIEW_USER.format(
        funcname=funcname,
        spec_text=spec_text,
        features_json=json.dumps(features or {}, indent=2),
        c_code=c_code,
    )

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SPEC_SEMANTIC_REVIEW_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                **completion_token_kwargs(max_tokens),
                **seed_kwargs(),
                **qwen_thinking_kwargs(model),
                **deepseek_thinking_kwargs(model),
            )
            if response.usage:
                usage.prompt_tokens += response.usage.prompt_tokens
                usage.completion_tokens += response.usage.completion_tokens
                usage.total_tokens += response.usage.total_tokens
            text = response.choices[0].message.content or ""
            return _sanitize_c_response(text), usage
        except Exception as exc:
            logger.warning("Spec-to-C semantic review attempt %d failed: %s", attempt + 1, exc)
            usage.retries += 1
            if attempt == max_retries:
                return None, usage

    return None, usage


def generate_from_verilog(
    verilog_source: str,
    max_retries: int = 2,
    env_path: str | Path = ".env",
    max_tokens: int | None = None,
) -> Tuple[Optional[str], TokenUsage]:
    """
    Generate C code from Verilog RTL source.

    Args:
        verilog_source: Verilog module source code
        max_retries: Number of retry attempts on failure
        env_path: Path to .env file
        max_tokens: Optional explicit completion token budget

    Returns:
        (c_code, token_usage): C code string or None on failure
    """
    client, model = _get_llm_client(env_path)
    if client is None:
        logger.error("Cannot proceed: LLM client is None (check .env)")
        return None, TokenUsage()

    max_tokens = completion_max_tokens(max_tokens)
    usage = TokenUsage()
    user_msg = _VERILOG_ONLY_USER.format(verilog_source=verilog_source)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _VERILOG_ONLY_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                **completion_token_kwargs(max_tokens),
                **seed_kwargs(),
                **qwen_thinking_kwargs(model),
                **deepseek_thinking_kwargs(model),
            )

            if response.usage:
                usage.prompt_tokens += response.usage.prompt_tokens
                usage.completion_tokens += response.usage.completion_tokens
                usage.total_tokens += response.usage.total_tokens

            text = response.choices[0].message.content or ""
            return _sanitize_c_response(text), usage

        except Exception as e:
            logger.warning("LLM call attempt %d failed: %s", attempt + 1, e)
            usage.retries += 1
            if attempt == max_retries:
                return None, usage

    return None, usage


def generate_mixed(
    verilog_source: str,
    spec_text: str,
    max_retries: int = 2,
    env_path: str | Path = ".env",
    max_tokens: int | None = None,
) -> Tuple[Optional[str], TokenUsage]:
    """
    Generate C code from both Verilog (ground truth) and spec (context).

    Args:
        verilog_source: Verilog module source code
        spec_text: Natural-language specification
        max_retries: Number of retry attempts on failure
        env_path: Path to .env file
        max_tokens: Optional explicit completion token budget

    Returns:
        (c_code, token_usage): C code string or None on failure
    """
    client, model = _get_llm_client(env_path)
    if client is None:
        logger.error("Cannot proceed: LLM client is None (check .env)")
        return None, TokenUsage()

    max_tokens = completion_max_tokens(max_tokens)
    usage = TokenUsage()
    user_msg = _MIXED_USER.format(
        verilog_source=verilog_source,
        spec_text=spec_text,
    )

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _MIXED_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                **completion_token_kwargs(max_tokens),
                **seed_kwargs(),
                **qwen_thinking_kwargs(model),
                **deepseek_thinking_kwargs(model),
            )

            if response.usage:
                usage.prompt_tokens += response.usage.prompt_tokens
                usage.completion_tokens += response.usage.completion_tokens
                usage.total_tokens += response.usage.total_tokens

            text = response.choices[0].message.content or ""
            return _sanitize_c_response(text), usage

        except Exception as e:
            logger.warning("LLM call attempt %d failed: %s", attempt + 1, e)
            usage.retries += 1
            if attempt == max_retries:
                return None, usage

    return None, usage
