from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from llm_transform import (
    AREA_TRANSFORM_DESCRIPTIONS,
    TIMING_TRANSFORM_DESCRIPTIONS,
    code_actually_differs,
    extract_meta_comment,
    strip_markdown_fences,
)
from rag_retrieve.schema import Module5Action
from module5.token_usage import usage_from_response


def _get_client(env_path: str | Path) -> Tuple[OpenAI | None, str]:
    load_dotenv(env_path, override=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
    )
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    if not api_key or not base_url:
        return None, model
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        **openai_client_kwargs(),
    ), model


def _transform_description(action: Module5Action) -> str:
    combined = dict(TIMING_TRANSFORM_DESCRIPTIONS)
    combined.update(AREA_TRANSFORM_DESCRIPTIONS)
    return combined.get(
        action.transform_name,
        f"Apply the {action.transform_name} transform conservatively and locally.",
    )


def _build_messages(action: Module5Action, source_code: str) -> list[Dict[str, str]]:
    source_anchor = action.source_anchor or {}
    function_name = str(source_anchor.get("function", ""))
    anchor_labels = source_anchor.get("anchor_labels", [])
    forbidden = ", ".join(action.constraints.forbidden_transforms) or "none"
    anchor_text = "\n".join(f"- {label}" for label in anchor_labels) if anchor_labels else "- none"
    c_snippet = str(source_anchor.get("c_snippet", "")).strip()
    line_start = source_anchor.get("line_start", "")
    line_end = source_anchor.get("line_end", "")
    function_line_start = source_anchor.get("function_line_start", "")
    function_line_end = source_anchor.get("function_line_end", "")
    key_variables = source_anchor.get("key_variables", [])
    key_constants = source_anchor.get("key_constants", [])
    key_expressions = source_anchor.get("key_expressions", [])
    mapping_note = str(source_anchor.get("mapping_note", "")).strip()
    c_context = "\n".join(
        [
            f"Function line range: {function_line_start}-{function_line_end}" if function_line_start or function_line_end else "Function line range: unknown",
            f"Suggested local C line range: {line_start}-{line_end}" if line_start or line_end else "Suggested local C line range: unknown",
            f"Mapping note: {mapping_note or 'none'}",
            "Key variables: " + (", ".join(map(str, key_variables)) if key_variables else "none"),
            "Key constants: " + (", ".join(map(str, key_constants)) if key_constants else "none"),
            "Key IR expressions:\n" + ("\n".join(f"- {expr}" for expr in key_expressions) if key_expressions else "- none"),
            "Suggested C snippet:\n" + (c_snippet if c_snippet else "none"),
        ]
    )
    description = _transform_description(action)

    system = f"""You are Module 5, a constrained C optimization executor for downstream AI RTL generation.

You must apply exactly one transform-oriented edit to the provided C file.

Hard requirements:
1. Preserve behavior.
2. Keep edits local to the target region and target function.
3. Do not redesign the whole file.
4. Do not add HLS pragmas.
5. Do not use any forbidden transform.
6. Preserve state_elements structs, function signature shape, and pointer writebacks.
7. Output only the full updated C source code.
8. End the file with one line exactly in this form:
// TRANSFORM_META: {{"applied": true_or_false, "summary": "short summary"}}
9. Return a complete, syntactically valid C file. Do not truncate the file.
10. Do not leave partial identifiers, incomplete statements, missing semicolons, or unbalanced braces.
11. If no safe edit is possible, return the original source code unchanged and set applied=false.
12. Before final output, audit that all braces, parentheses, brackets, and comments are balanced.
13. Preserve cycle-step semantics: old-state/next-state separation, simultaneous register updates, registered-output latency, valid/data alignment, FSM priority, and counter boundaries must not change.

Region localization guidance:
- Anchor labels may come from LLVM IR / CDFG nodes and may not appear verbatim in the C source.
- Do not reject a transform only because an anchor label string is not found literally.
- Use anchor labels as semantic hints, together with the target function, region type, problem hypothesis, and execution hint, to locate the corresponding C logic.
- If the exact region is ambiguous, apply only a conservative local edit near the most likely matching logic; if no safe matching logic exists, then leave the code unchanged.
"""

    user = f"""Action ID: {action.action_id}
Benchmark: {action.benchmark}
Objective: {action.objective}
Transform: {action.transform_name}
Transform description: {description}
Problem hypothesis: {action.problem_hypothesis}
Execution hint: {action.execution_hint}
Target function: {function_name or "unknown"}
Target region type: {action.region_type}
Apply scope: {action.constraints.apply_scope}
Expected risk: {action.constraints.expected_risk}
Forbidden transforms: {forbidden}
Anchor labels:
{anchor_text}

Source localization hints:
{c_context}

Apply a conservative local edit that follows the target transform intent.
If the transform should not be applied safely, return the original code and set "applied" to false in TRANSFORM_META.

Source code:
```c
{source_code}
```"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def apply_action_edit(
    action: Module5Action,
    source_code: str,
    env_path: str | Path = ".env",
    max_tokens: int | None = None,
) -> Tuple[str | None, Dict[str, Any], bool, str, List[Dict[str, Any]]]:
    client, model = _get_client(env_path)
    if client is None:
        unchanged = source_code.rstrip() + "\n// TRANSFORM_META: {\"applied\": false, \"summary\": \"LLM unavailable; reused source for downstream evaluation\"}\n"
        return unchanged, {"applied": False, "summary": "LLM unavailable; reused source for downstream evaluation"}, False, "", []

    max_tokens = completion_max_tokens(max_tokens)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=_build_messages(action, source_code),
            temperature=0.1,
            **completion_token_kwargs(max_tokens),
            **seed_kwargs(),
            **qwen_thinking_kwargs(model),
            **deepseek_thinking_kwargs(model),
        )
    except Exception as exc:
        unchanged = source_code.rstrip() + f"\n// TRANSFORM_META: {{\"applied\": false, \"summary\": \"LLM error; reused source for downstream evaluation: {str(exc).replace(chr(34), chr(39))}\"}}\n"
        return unchanged, {"applied": False, "summary": f"LLM error; reused source for downstream evaluation: {exc}"}, False, "", []

    content = (response.choices[0].message.content or "").strip()
    content = strip_markdown_fences(content)
    meta = extract_meta_comment(content)
    differs = code_actually_differs(source_code, content)
    token_usage = [usage_from_response(stage="c_edit_single", model=model, response=response)]
    return content, meta, differs, "", token_usage


# ---------------------------------------------------------------------------
# Multi-action single-pass editor
# ---------------------------------------------------------------------------

def _build_multi_action_messages(
    actions: List[Module5Action], source_code: str
) -> list[Dict[str, str]]:
    combined = dict(TIMING_TRANSFORM_DESCRIPTIONS)
    combined.update(AREA_TRANSFORM_DESCRIPTIONS)

    system = (
        "You are Module 5, a constrained C optimization executor for downstream AI RTL generation.\n\n"
        "You must apply ALL of the listed transforms to the provided C file in a single pass.\n\n"
        "Hard requirements:\n"
        "1. Preserve behavior.\n"
        "2. Keep each edit local to its target region.\n"
        "3. Do not redesign the whole file.\n"
        "4. Do not add HLS pragmas.\n"
        "5. Do not use any forbidden transform.\n"
        "6. Preserve state_elements structs, function signature shape, and pointer writebacks.\n"
        "7. Output only the full updated C source code with ALL transforms applied.\n"
        "8. End the file with per-transform annotations, one per line, in this exact form:\n"
        '// TRANSFORM_META_<index>: {"applied": true_or_false, "region": "region_id", "summary": "short summary"}\n'
        "9. Return a complete, syntactically valid C file. Do not truncate the file.\n"
        "10. Do not leave partial identifiers, incomplete statements, missing semicolons, or unbalanced braces.\n"
        "11. If no transform can be applied safely, return the original source code unchanged and set all applied fields to false.\n"
        "12. If only some transforms are safe, apply only those safe transforms and leave the others unchanged with applied=false.\n"
        "13. Before final output, audit that all braces, parentheses, brackets, and comments are balanced.\n"
        "14. Do not output prose, markdown, explanations, or partial snippets outside the C file.\n"
        "15. Preserve cycle-step semantics: old-state/next-state separation, simultaneous register updates, registered-output latency, valid/data alignment, FSM priority, and counter boundaries must not change.\n"
        "\n"
        "Region localization guidance:\n"
        "- Anchor labels may come from LLVM IR / CDFG nodes and may not appear verbatim in the C source.\n"
        "- Do not mark a transform as not applicable only because an anchor label string is not found literally.\n"
        "- Use anchor labels as semantic hints, together with the target function, region type, problem hypothesis, and execution hint, to locate the corresponding C logic.\n"
        "- If the exact region is ambiguous, apply only a conservative local edit near the most likely matching logic.\n"
        "- If no safe matching logic exists after semantic inspection, leave that region unchanged and set applied to false.\n"
    )

    action_blocks: list[str] = []
    for idx, act in enumerate(actions):
        anchor = act.source_anchor or {}
        func = str(anchor.get("function", ""))
        labels = anchor.get("anchor_labels", [])
        anchor_text = "\n".join(f"  - {l}" for l in labels) if labels else "  - none"
        c_snippet = str(anchor.get("c_snippet", "")).strip()
        line_start = anchor.get("line_start", "")
        line_end = anchor.get("line_end", "")
        function_line_start = anchor.get("function_line_start", "")
        function_line_end = anchor.get("function_line_end", "")
        key_variables = anchor.get("key_variables", [])
        key_constants = anchor.get("key_constants", [])
        key_expressions = anchor.get("key_expressions", [])
        mapping_note = str(anchor.get("mapping_note", "")).strip()
        key_expr_text = "\n".join(f"    - {expr}" for expr in key_expressions) if key_expressions else "    - none"
        forbidden = ", ".join(act.constraints.forbidden_transforms) or "none"
        desc = combined.get(
            act.transform_name,
            f"Apply the {act.transform_name} transform conservatively and locally.",
        )
        block = (
            f"Transform {idx}:\n"
            f"  Region ID: {act.region_id}\n"
            f"  Transform: {act.transform_name}\n"
            f"  Description: {desc}\n"
            f"  Problem hypothesis: {act.problem_hypothesis}\n"
            f"  Execution hint: {act.execution_hint}\n"
            f"  Target function: {func or 'unknown'}\n"
            f"  Target region type: {act.region_type}\n"
            f"  Apply scope: {act.constraints.apply_scope}\n"
            f"  Expected risk: {act.constraints.expected_risk}\n"
            f"  Forbidden transforms: {forbidden}\n"
            f"  Anchor labels:\n{anchor_text}"
            f"\n  Function line range: {function_line_start or '?'}-{function_line_end or '?'}"
            f"\n  Suggested local C line range: {line_start or '?'}-{line_end or '?'}"
            f"\n  Mapping note: {mapping_note or 'none'}"
            f"\n  Key variables: {', '.join(map(str, key_variables)) if key_variables else 'none'}"
            f"\n  Key constants: {', '.join(map(str, key_constants)) if key_constants else 'none'}"
            f"\n  Key IR expressions:\n{key_expr_text}"
            f"\n  Suggested C snippet:\n{c_snippet or 'none'}"
        )
        action_blocks.append(block)

    user = (
        f"Benchmark: {actions[0].benchmark}\n"
        f"Objective: {actions[0].objective}\n"
        f"Number of transforms to apply: {len(actions)}\n\n"
        + "\n\n".join(action_blocks)
        + "\n\nApply all transforms above. For any transform that cannot be applied safely, "
        "leave that region unchanged and set \"applied\" to false in its TRANSFORM_META line. "
        "The returned file must remain complete and clang-syntax-valid; if there is any risk of producing "
        "an incomplete C file, return the original source unchanged with applied=false metadata instead.\n\n"
        f"Source code:\n```c\n{source_code}\n```"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_multi_meta(content: str, n_actions: int) -> List[Dict[str, Any]]:
    """Extract per-action TRANSFORM_META_<i> annotations from LLM output."""
    results: List[Dict[str, Any]] = [
        {"applied": False, "region": "", "summary": "not found in output"}
    ] * n_actions
    for m in re.finditer(
        r'//\s*TRANSFORM_META_(\d+)\s*:\s*(\{.*\})', content
    ):
        idx = int(m.group(1))
        if 0 <= idx < n_actions:
            try:
                results[idx] = json.loads(m.group(2))
            except json.JSONDecodeError:
                results[idx] = {"applied": False, "region": "", "summary": m.group(2)}
    return results


def apply_multi_action_edit(
    actions: List[Module5Action],
    source_code: str,
    env_path: str | Path = ".env",
    max_tokens: int | None = None,
) -> Tuple[str | None, List[Dict[str, Any]], bool, str, List[Dict[str, Any]]]:
    """Apply multiple actions in a single LLM pass.

    Returns (edited_code, per_action_metas, differs, error, token_usage).
    """
    n = len(actions)
    fallback_metas = [{"applied": False, "summary": "not attempted"}] * n

    client, model = _get_client(env_path)
    if client is None:
        return source_code, fallback_metas, False, "llm_unavailable", []

    max_tokens = completion_max_tokens(max_tokens)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=_build_multi_action_messages(actions, source_code),
            temperature=0.1,
            **completion_token_kwargs(max_tokens),
            **seed_kwargs(),
            **qwen_thinking_kwargs(model),
            **deepseek_thinking_kwargs(model),
        )
    except Exception as exc:
        return source_code, fallback_metas, False, f"llm_error: {exc}", []

    content = (response.choices[0].message.content or "").strip()
    content = strip_markdown_fences(content)
    per_action_metas = _parse_multi_meta(content, n)
    differs = code_actually_differs(source_code, content)
    token_usage = [usage_from_response(stage="c_edit_sequence", model=model, response=response)]
    return content, per_action_metas, differs, "", token_usage
