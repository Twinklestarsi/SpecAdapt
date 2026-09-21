"""
llm_gen.py — LLM-based direct RTL generation from natural-language specs.
"""

from __future__ import annotations

import json
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
from llm_code_sanitize import sanitize_verilog_response
from module5.rtl_direct_runner import _numbered_rules
from openai import OpenAI

from token_counter import TokenUsage


_SYSTEM = """\
You are an expert RTL designer for FPGA and ASIC synthesis.

Given a natural-language hardware specification, generate a single complete,
synthesizable Verilog module.

Requirements:
1. Output only Verilog code, with no markdown fences or explanations.
2. Generate exactly one top-level module named `{module_name}`.
3. Use strict synthesizable Verilog-2001 only. Do NOT use SystemVerilog.
4. Infer ports, widths, reset style, sequential/combinational structure, and FSMs
   from the specification first, using feature hints only to refine details that do
   not conflict with explicit specification text.
5. Preserve explicitly stated behavior; do not invent extra interfaces unless needed
   to make the design implementable.
6. Do not include a testbench.
7. Prefer clear RTL: separate combinational and sequential logic when appropriate.
8. If the spec is underspecified, make the smallest reasonable assumption and keep
   the design simple.
9. The output must be accepted by older Verilog compilers used by Synopsys DC in
   `analyze -format verilog` mode.
10. Preserve every explicitly specified interface fact exactly. If feature hints
    contain an `interface` object, use it as the normalized port contract unless
    an inferred field conflicts with explicit specification text; explicit text
    wins. Do not rename, add, remove, or resize explicitly specified ports.
11. Preserve every explicitly stated cycle-level behavior. Use
    `behavioral_contract` to make those facts concrete, but treat inferred details
    as assumptions rather than permission to override the specification. Every
    explicitly registered output must remain clocked; preserve stated latency and
    sample/update edges exactly.
12. Phrases such as "on the cycle after", "registered output", or "updated at
    the rising edge" require registered output behavior unless the specification
    explicitly says otherwise.
13. In ANSI-style module declarations, declare a procedurally assigned output as
    `output reg` exactly once. Do not redeclare that output as a separate `reg`.
14. Before responding, perform a cycle audit covering old versus next state,
    simultaneous nonblocking updates, reset release, enable bubbles, back-to-back
    transactions, valid/data alignment, FSM transition priority, and counter
    off-by-one boundaries.
{shared_rtl_rules}

Forbidden SystemVerilog constructs:
- `logic`
- `bit`
- `byte`
- `int`
- `shortint`
- `longint`
- `always_ff`
- `always_comb`
- `always_latch`
- `typedef`
- `enum`
- packed arrays after identifiers
- interfaces, structs, unions, packages, classes
- `unique case`, `priority case`
- `inside`, `foreach`, `automatic`

Use these instead:
- `wire` / `reg`
- `parameter` / `localparam` without explicit type keywords
- `integer` only for loop indices if needed
- `always @(*)` and `always @(posedge clk)`
"""


_USER = """\
Benchmark: {benchmark}
Requested module name: {module_name}

Specification:
{spec_text}

Feature hints extracted upstream:
{features_json}
{contract_block}
Return a complete Verilog module now.

Reminder: return strict Verilog-2001 syntax only. If you use any SystemVerilog
keyword such as `logic`, `int`, `always_ff`, or typed parameters, the result is invalid.
"""


def _get_llm_client(
    env_path: str | Path = ".env",
    api_key: str | None = None,
    base_url: str | None = None,
    model_id: str | None = None,
) -> Tuple[Optional[OpenAI], Optional[str]]:
    try:
        load_dotenv(env_path, override=True)
        client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=(
                base_url
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE_URL")
                or os.environ.get("OPENAI_API_BASE")
            ),
            **openai_client_kwargs(),
        )
        model = model_id or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return client, model
    except Exception:
        return None, None


def _interface_contract_block(interface_contract: str) -> str:
    """Render the frozen interface contract, or "" when there is none.

    The interface is a contract, not an answer: equivalence checking rejects a
    candidate whose port names, directions or widths differ before any proof
    runs, so a guessed width burns the whole route.  Only names, directions,
    widths and parameter defaults are stated here; no reference logic is
    included.
    """

    text = str(interface_contract or "").strip()
    if not text:
        return ""
    return (
        "\nFrozen interface contract (must match exactly — do not rename, add,\n"
        "remove or resize these ports):\n"
        f"{text}\n"
    )


def generate_from_spec(
    benchmark: str,
    spec_text: str,
    module_name: str,
    features: dict,
    max_retries: int = 2,
    env_path: str | Path = ".env",
    model_id: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    interface_contract: str = "",
) -> Tuple[Optional[str], TokenUsage]:
    """
    Generate Verilog RTL from a natural-language spec.
    """
    client, model = _get_llm_client(
        env_path, api_key=api_key, base_url=base_url, model_id=model_id,
    )
    if client is None:
        return None, TokenUsage()

    max_tokens = completion_max_tokens()
    usage = TokenUsage()
    features_json = json.dumps(features or {}, indent=2)
    system_msg = _SYSTEM.format(
        module_name=module_name,
        shared_rtl_rules=_numbered_rules(15),
    )
    user_msg = _USER.format(
        benchmark=benchmark,
        module_name=module_name,
        spec_text=spec_text,
        features_json=features_json,
        contract_block=_interface_contract_block(interface_contract),
    )

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
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
            return sanitize_verilog_response(text), usage

        except Exception:
            usage.retries += 1
            if attempt == max_retries:
                return None, usage

    return None, usage
