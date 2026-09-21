"""C -> RTL generation with a syntax-feedback retry loop.

The retry contract matters as much as the generation prompt: when the local
syntax gate rejects an attempt, the next request carries *three* things back to
the model -- the complete previous Verilog (line-numbered so the compiler's line
numbers can be located), the unabridged Icarus Verilog output, and an explicit
list of what to fix.  Sending only the error text made the model repair code it
could no longer see, which is how the same `syntax error` survived every retry.
"""

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
    completion_was_truncated,
    deepseek_thinking_kwargs,
    openai_client_kwargs,
    qwen_thinking_kwargs,
    seed_kwargs,
    truncated_completion_message,
)
from llm_code_sanitize import sanitize_verilog_response
from rag_retrieve.schema import Module5Action
from module5.jg_verifier import golden_interface_contract, verify_rtl_with_jg
from module5.token_usage import merge_token_usage, usage_from_response
from project_paths import PROJECT_ROOT


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


def _sanitize_verilog(text: str) -> str:
    return sanitize_verilog_response(text)


def _check_dc_compatibility(verilog_text: str) -> Tuple[bool, str]:
    """
    Lightweight DC-oriented compatibility checks beyond iverilog -g2005.

    The current DC flow rejects declarations such as:
      input wire [N-1:0][31:0] bus;
      reg   [N-1:0][2:0]  foo;
    which use multiple packed dimensions before the identifier.
    """
    for lineno, line in enumerate((verilog_text or "").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        if re.search(
            r"\b(?:input|output|wire|reg)\b[^;,\n]*\[[^\]]+\]\s*\[[^\]]+\]\s*\w+",
            stripped,
        ):
            return (
                False,
                "DC compatibility error: multiple packed dimensions are not supported "
                f"(line {lineno}): {stripped}",
            )

    return True, ""


# ── Constraints shared by the initial prompt and every repair prompt ──────
#
# These are generic Verilog-2001 traps that keep reappearing across benchmarks,
# so they belong in both the first request and every retry: a rule that is only
# stated after the failure cannot prevent the failure.

GENERAL_RTL_RULES: Tuple[str, ...] = (
    "Never use a Verilog-2001 reserved keyword as an identifier. This covers "
    "module names, port names, signal names, parameter names, instance names and "
    "block labels. `table` is a reserved keyword and is a very common cause of a "
    "bare `syntax error`; so are time, event, edge, wait, force, release, "
    "disable, cell, config, design, instance, library, use, small, medium, "
    "large, tran, trireg, specify, deassign, defparam, primitive, initial, "
    "always, assign, task, function, generate, repeat, forever, while, fork, "
    "join, default, signed and unsigned. When a natural name collides, add a "
    "suffix instead of using the keyword, for example `table` -> `lut_mem` or "
    "`entry_table`, and `time` -> `time_cnt`.",
    "Every signal assigned inside an `always` block must be declared `reg` "
    "(or `output reg` / `integer`), never `wire`. A `wire` may only be driven by "
    "a continuous `assign` or by an instance output port. Never drive the same "
    "signal from both an `assign` statement and an `always` block.",
    "When compiler output lists several errors, fix the FIRST reported error "
    "first: the later messages are usually cascade effects of it, because one "
    "missing `end`, one undeclared identifier or one reserved-word identifier "
    "makes the parser mis-read everything after it. Do not start unrelated "
    "rewrites in response to cascade errors.",
    "Before returning the code, review the whole module yourself the way "
    "`iverilog -g2005` would: every `begin` has a matching `end`, every `module` "
    "has its `endmodule`, every declaration ends with `;`, every identifier is "
    "declared before use, port directions and widths agree, no "
    "SystemVerilog-only construct is present, and no reserved keyword is used as "
    "an identifier.",
)


def _numbered_rules(start: int) -> str:
    """Render the shared rules as a continuation of a numbered prompt list."""
    return "\n".join(
        f"{start + offset}. {rule}" for offset, rule in enumerate(GENERAL_RTL_RULES)
    )


def _bulleted_rules() -> str:
    """Render the shared rules for repair prompts, which use bullet lists."""
    return "\n".join(f"- {rule}" for rule in GENERAL_RTL_RULES)


# ── Reading the compiler output back to the model ─────────────────────────

#: Reserved words scanned for in the failing code. Restricted to keywords that a
#: hardware generator has no legitimate reason to write on a normal RTL line, so
#: that the hint below points at a real collision rather than at `always` in
#: `always @(posedge clk)`.
_RESERVED_IDENTIFIER_TRAPS: Tuple[str, ...] = (
    "table", "endtable", "time", "realtime", "event", "edge", "wait", "force",
    "release", "disable", "cell", "config", "endconfig", "design", "instance",
    "library", "liblist", "incdir", "include", "use", "small", "medium",
    "large", "specify", "endspecify", "specparam", "deassign", "defparam",
    "macromodule", "primitive", "endprimitive", "scalared", "vectored",
    "trireg", "tri0", "tri1", "triand", "trior", "wand", "wor", "highz0",
    "highz1", "pull0", "pull1", "strong0", "strong1", "weak0", "weak1",
    "showcancelled", "noshowcancelled", "pulsestyle_onevent",
    "pulsestyle_ondetect", "ifnone", "automatic", "real",
)

#: Lines in Icarus Verilog output look like ``path/to/file.sv:42: syntax error``.
_ERROR_LINE_RE = re.compile(r"^\s*\S*?:(\d+):\s*(.*)$")

#: A generated module is a few hundred lines; the cap only guards against a
#: runaway response, and line numbers stay valid because they are assigned
#: before any trimming.
_MAX_CODE_CHARS = 120_000


def _number_lines(code: str, *, max_chars: int = _MAX_CODE_CHARS) -> str:
    """Prefix each line with its line number so compiler messages can be located."""
    lines = (code or "").splitlines()
    if not lines:
        return "(the previous attempt produced no code at all)"
    width = max(len(str(len(lines))), 3)
    numbered = [
        f"{index:>{width}} | {line}" for index, line in enumerate(lines, start=1)
    ]
    body = "\n".join(numbered)
    if len(body) <= max_chars:
        return body

    head_budget = max_chars * 2 // 3
    head: List[str] = []
    used = 0
    for row in numbered:
        if used + len(row) + 1 > head_budget:
            break
        head.append(row)
        used += len(row) + 1
    tail: List[str] = []
    used = 0
    for row in reversed(numbered[len(head):]):
        if used + len(row) + 1 > max_chars - head_budget:
            break
        tail.insert(0, row)
        used += len(row) + 1
    omitted = len(numbered) - len(head) - len(tail)
    marker = f"... [{omitted} lines omitted to fit the request; line numbers above and below are still correct] ..."
    return "\n".join(head + [marker] + tail)


def _error_line_numbers(compiler_output: str) -> List[int]:
    """Line numbers the compiler complained about, in the order reported."""
    numbers: List[int] = []
    for line in (compiler_output or "").splitlines():
        match = _ERROR_LINE_RE.match(line)
        if not match:
            continue
        number = int(match.group(1))
        if number not in numbers:
            numbers.append(number)
    return numbers


def _first_error(compiler_output: str) -> str:
    """The first real error line, which is the one worth fixing first."""
    for line in (compiler_output or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _ERROR_LINE_RE.match(stripped) or "error" in stripped.lower():
            # The trailing "N error(s) during elaboration." summary is not a
            # locatable error and must not be presented as the first one.
            if re.match(r"^\d+\s+error", stripped, re.IGNORECASE):
                continue
            return stripped
    return (compiler_output or "").strip().splitlines()[0] if (compiler_output or "").strip() else ""


def _reserved_identifier_hints(previous_rtl: str, compiler_output: str) -> List[str]:
    """Name the reserved keyword behind a bare `syntax error`, when there is one.

    Icarus Verilog reports a reserved word used as a signal name only as
    ``syntax error``, with no clue about which token is at fault, so the model
    usually rewrites something unrelated and fails again. Locating the keyword
    here turns that dead end into a concrete rename instruction.
    """
    lines = (previous_rtl or "").splitlines()
    if not lines:
        return []
    pattern = re.compile(
        r"\b(" + "|".join(_RESERVED_IDENTIFIER_TRAPS) + r")\b"
    )

    def _scan(numbers: List[int]) -> List[str]:
        found: List[str] = []
        for number in numbers:
            if not 1 <= number <= len(lines):
                continue
            text = lines[number - 1]
            code = text.split("//", 1)[0]
            for match in pattern.finditer(code):
                keyword = match.group(1)
                found.append(
                    f"Line {number} uses the Verilog-2001 reserved keyword "
                    f"`{keyword}` as an identifier: {text.strip()}\n"
                    f"  Rename it (for example `{keyword}` -> `{keyword}_r` or "
                    f"`{keyword}_mem`) at every occurrence in the module. This "
                    "alone can be the whole reason the parser reports a syntax "
                    "error here."
                )
                break
            if len(found) >= 3:
                break
        return found

    hints = _scan(_error_line_numbers(compiler_output))
    if hints:
        return hints
    # The parser can blame a line after the real collision, so fall back to the
    # whole file rather than reporting nothing.
    return _scan(list(range(1, len(lines) + 1)))


def _validator_label(compiler_output: str) -> str:
    if (compiler_output or "").strip().lower().startswith("dc compatibility error"):
        return "local Design Compiler compatibility check"
    return "Icarus Verilog (iverilog -g2005)"


#: ``'tmp' is not a valid l-value for a procedural assignment.`` and
#: ``'q' is declared here as a wire.`` — both name the signal in quotes, which is
#: the one thing the model needs in order to change the right declaration.
_LVALUE_RE = re.compile(
    r"'([A-Za-z_]\w*)'\s+(?:is not a valid l-value|is declared here as a (?:wire|net)|has already been declared)"
)


def _wire_in_always_hints(compiler_output: str) -> List[str]:
    """Name the signals that are declared as nets but assigned procedurally."""
    names: List[str] = []
    for match in _LVALUE_RE.finditer(compiler_output or ""):
        name = match.group(1)
        if name not in names:
            names.append(name)
    if not names:
        return []
    listed = ", ".join(f"`{name}`" for name in names[:6])
    return [
        f"The compiler names the offending signal(s): {listed}.",
        "Each of those is declared as a wire/net but assigned inside an `always` "
        "block, which Verilog forbids. Change the declaration to `reg` "
        "(use `output reg` for a port, and declare it exactly once — never both "
        "`output wire x` and `reg x`).",
        "If instead the signal is meant to be combinational glue, keep it a "
        "`wire` and drive it with a continuous `assign` rather than from an "
        "`always` block. Never do both.",
    ]


def _build_messages(
    action: Module5Action,
    *,
    c_code: str,
    module_name: str,
    spec_text: str,
    llm_features: Dict[str, Any],
    interface_contract: str = "",
) -> list[Dict[str, str]]:
    forbidden = ", ".join(action.constraints.forbidden_transforms) or "none"
    feature_text = json.dumps(llm_features or {}, indent=2, ensure_ascii=False)
    # The interface is a contract, not an answer. Equivalence checking rejects a
    # candidate whose ports differ before any proof runs, so guessed widths
    # (5-bit weights returned as 1 bit, 256/1024/128-bit buses returned as 8
    # bits) wasted whole routes. Only names, directions, widths and parameter
    # defaults are stated here; no reference logic is included.
    has_contract = bool(str(interface_contract or "").strip())
    contract_block = (
        "\nFrozen interface contract (must match exactly):\n"
        + str(interface_contract).strip()
        + "\n"
        if has_contract
        else ""
    )
    # Rules 5 and 9 tell the model to invent an interface from the specification.
    # That is right when the specification is all there is, and wrong the moment
    # a frozen contract is supplied: a width or direction difference is rejected
    # before JasperGold runs, so a self-invented `clk`/`rst_n` pair (the
    # reference designs use `nvdla_core_clk`/`nvdla_core_rstn`) burned the whole
    # route.  The contract is authoritative, so both rules are rewritten to point
    # at it rather than contradict it.
    if has_contract:
        rule_5 = (
            "5. Copy the interface from the frozen interface contract verbatim: "
            "the same port names, directions, widths and order, and the same "
            "parameter names and defaults. Do not rename a port, do not add a "
            "clock or reset the contract does not list, and do not narrow or "
            "widen any port."
        )
        rule_9 = (
            "9. Use exactly the clock and reset named in the frozen interface "
            "contract. The contract overrides any clock or reset the "
            "specification implies."
        )
    else:
        rule_5 = "5. Keep the interface minimal and hardware-native."
        rule_9 = (
            "9. If the design is sequential, include a normal clock/reset "
            "interface inferred from the spec."
        )

    system = f"""You are an expert RTL designer.

Generate exactly one synthesizable Verilog-2001 module named `{module_name}`.

Hard requirements:
1. Output only Verilog code. No markdown fences and no explanations.
2. Generate exactly one top-level module.
3. Use Verilog-2001 syntax ONLY - compatible with Design Compiler:
   - Use 'reg' and 'wire' types (NOT 'logic')
   - Use explicit bit patterns like 32'h0 or 8'd0 (NOT '0 or '1 shorthand)
   - Avoid SystemVerilog-only features: typedef, enum, struct, union, interface, logic, always_ff, always_comb
   - Internal Verilog-2001 memory arrays are allowed, for example: reg [31:0] mem [0:NUM_HW_APPS-1];
   - Do NOT use multi-dimensional packed declarations such as [N-1:0][W-1:0] signal
   - Do NOT use multi-dimensional packed ports such as input [N-1:0][W-1:0] bus
   - For module ports, flatten repeated lanes into one packed bus, for example: input [(NUM_HW_APPS*32)-1:0] csrng_cmd_req_bus_i
   - Use localparam for constants
   - For hardware replication (multiple instances), use 'generate' blocks with 'genvar', NOT runtime 'for' loops
   - Runtime 'for' loops in 'always' blocks cannot be used for array indexing with variable indices
4. Use plain, readable RTL. Avoid AXI, AP_CTRL, or HLS wrapper interfaces.
{rule_5}
6. The explicit specification and its explicit cycle/interface facts are the
   primary behavioral authority. The optimized C is an implementation aid and
   must be followed only where it is behavior-preserving. If inferred features or
   optimized C conflict with explicit specification text, follow the specification.
7. Prefer simple always blocks and explicit reg/wire structure over tool-style autogenerated logic.
8. Do not include a testbench.
{rule_9}
10. Do not invent unrelated features.
11. Never use variable part-selects in the form [i*W+MSB : i*W+LSB].
12. If variable extraction is required, use one of:
   - indexed part-select: [base +: width] or [base -: width]
   - shift-and-mask logic
   - explicit arrays instead of packed-bus slicing
13. Do not use runtime for-loops to create variable slices from packed buses.
14. The output must compile with: iverilog -g2005
15. If repeated state entries exist, prefer arrays of registers over one large packed vector.
16. Do not use multiple packed dimensions in declarations such as [N-1:0][W-1:0] signal.
17. For module ports, use a single packed bus like [(N*W)-1:0] bus instead of multi-dimensional packed ports.
18. Prevent Design Compiler multiple-driver errors:
   - Every reg must be assigned in one and only one procedural always block.
   - Every writable memory array element must be written from one and only one procedural always block.
   - Do not assign the same storage variable from separate FSM blocks, helper blocks, reset blocks, or output blocks.
   - For buffer state such as valid bits, data arrays, read pointers, and write pointers, keep all writes in one centralized sequential state-update block.
   - Combinational always blocks may compute next-state signals, but must not assign storage registers that are assigned in clocked blocks.
   - If multiple control paths update the same register, merge those updates into one clocked block using priority if/else or case logic.
19. Before returning RTL, audit the whole module for single-owner storage assignment and DC synthesizability.
20. Treat explicitly stated facts represented in the upstream
    `behavioral_contract` as immutable. Inferred fields are assumptions and must
    not override explicit specification text. Outputs explicitly required to be
    registered must remain clocked and retain their stated cycle latency.
21. In ANSI-style module declarations, declare every procedurally assigned output
    as `output reg` exactly once; never redeclare the same output as another `reg`.
22. Phrases such as "on the cycle after", "registered output", or "updated at the
    rising edge" describe clocked output semantics and must remain clocked.
23. Before responding, audit old-state/next-state behavior, simultaneous register
    updates, reset release, valid/data alignment, enable bubbles, back-to-back
    transactions, FSM priority, and counter terminal thresholds.
{_numbered_rules(24)}
"""

    user = f"""Benchmark: {action.benchmark}
Objective: {action.objective}
Action ID: {action.action_id}
Transform: {action.transform_name}
Problem hypothesis: {action.problem_hypothesis}
Execution hint: {action.execution_hint}
Forbidden transforms at the C-edit stage: {forbidden}
Requested top module name: {module_name}

Specification:
{spec_text}

Upstream feature hints:
{feature_text}

Optimized C source to preserve behavior against:
```c
{c_code}
```

Important forbidden coding pattern:
- Do NOT write expressions like bus[i*32+23 : i*32+12].
- These are illegal in Verilog-2001 elaboration.
- Use indexed part-select ([base +: width] / [base -: width]) or shift-and-mask instead.
- Do NOT drive one reg or writable memory from multiple always blocks.
- Do NOT split SBUF/read-pointer/write-pointer/valid-bit updates across multiple clocked blocks.
{contract_block}
Return one complete Verilog module now."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_syntax_retry_message(
    syntax_err: str,
    previous_rtl: str = "",
    *,
    attempt: int = 0,
    module_name: str = "",
    failure_history: Tuple[str, ...] = (),
) -> str:
    """Hand the failed attempt back to the model in full.

    Three blocks are non-negotiable here: the complete previous Verilog, the
    unabridged validator output, and an explicit fix list.  Only the error text
    was sent before, which asked the model to repair code it could not see.
    """
    syntax_err = (syntax_err or "").strip()
    previous_rtl = (previous_rtl or "").strip()
    lower_err = syntax_err.lower()
    label = _validator_label(syntax_err)

    sections: List[str] = [
        f"Your previous Verilog (attempt {attempt}) was rejected by the local "
        f"syntax gate, so it was never synthesized. Repair it below.",
    ]

    if failure_history:
        sections.append(
            "Attempts already rejected in this conversation:\n"
            + "\n".join(f"- {item}" for item in failure_history)
        )

    first_error = _first_error(syntax_err)
    if first_error:
        sections.append(
            "=== FIRST REPORTED ERROR — FIX THIS ONE FIRST ===\n"
            f"{first_error}\n"
            "Every message after this one may simply be a cascade effect of it."
        )

    sections.append(
        f"=== COMPLETE {label.upper()} OUTPUT (unabridged) ===\n{syntax_err}"
    )

    if previous_rtl:
        sections.append(
            f"=== YOUR COMPLETE PREVIOUS VERILOG (attempt {attempt}) ===\n"
            "The `NNN | ` prefix is the line number and matches the line numbers "
            "in the compiler output above. It is for reference only: do NOT copy "
            "the prefixes into your answer.\n"
            "```verilog\n"
            f"{_number_lines(previous_rtl)}\n"
            "```"
        )
    else:
        sections.append(
            "=== YOUR COMPLETE PREVIOUS VERILOG ===\n"
            "(unavailable — regenerate the module from the specification above)"
        )

    requirements: List[str] = [
        "=== WHAT TO DO ===",
        "1. Locate the first reported error in the numbered code above and "
        "identify its actual root cause on that line or on the line before it.",
        "2. Fix that root cause, then re-check whether the remaining messages "
        "were only cascade effects of it.",
        "3. Search the whole module for the same mistake instead of patching "
        "only the reported line.",
        "4. Keep the module name, ports, port widths, port directions and "
        "intended behavior of the previous attempt"
        + (f" (top module `{module_name}`)." if module_name else "."),
        "5. Change nothing that the compiler did not complain about.",
    ]

    hints = _reserved_identifier_hints(previous_rtl, syntax_err)
    if hints:
        requirements.append(
            "Reserved-keyword collisions found in your previous code:\n"
            + "\n".join(hints)
        )

    if (
        "part select expressions must be constant" in lower_err
        or "constant expression" in lower_err
        or "reference to a wire or reg" in lower_err
    ):
        requirements.extend(
            [
                "Your previous Verilog used variable part-selects with [msb:lsb] syntax.",
                "This is illegal in Verilog-2001 when msb/lsb depend on runtime variables.",
                "Rewrite that logic using one of the following legal forms:",
                "- indexed part-select: [base +: width] or [base -: width]",
                "- shift-and-mask logic",
                "- register arrays instead of packed-bus slicing",
                "Do not use the previous slicing style again.",
            ]
        )

    if (
        "multiple packed dimensions" in lower_err
        or "dc compatibility error" in lower_err
    ):
        requirements.extend(
            [
                "Your previous Verilog used multiple packed dimensions in a declaration, such as [N-1:0][W-1:0] signal.",
                "This is not accepted by the current Design Compiler flow.",
                "Rewrite those declarations using one of the following legal styles:",
                "- a single packed bus on module ports, for example [(N*W)-1:0] bus",
                "- unpacked arrays for internal storage only, where appropriate",
                "Do not use multi-dimensional packed ports or packed declarations again.",
            ]
        )

    if (
        "l-value" in lower_err
        or "procedural assignment" in lower_err
        or "declared here as a wire" in lower_err
        or "declared here as a net" in lower_err
        or "already been declared" in lower_err
        or "continuous assignment" in lower_err
        or "cannot be driven" in lower_err
    ):
        requirements.append(
            "The error is about an illegal assignment target or a duplicated "
            "declaration."
        )
        requirements.extend(
            _wire_in_always_hints(syntax_err)
            or [
                "A signal assigned inside an always block must be declared "
                "`reg`, not `wire`.",
                "Change that declaration to `reg` (or `output reg` for a port), "
                "and make sure the same signal is not also driven by an `assign` "
                "statement or by an instance output.",
            ]
        )

    if "syntax error" in lower_err and not hints:
        requirements.extend(
            [
                "A bare `syntax error` in Icarus Verilog almost always means one of:",
                "- a reserved keyword such as `table` used as a signal, port or parameter name",
                "- a missing `;`, `end`, `endmodule`, `)` or `,`",
                "- a SystemVerilog-only construct (logic, always_ff, typedef, enum, struct, '0/'1 literals)",
                "Check the reported line and the line immediately before it for these three cases before changing anything else.",
            ]
        )

    sections.append("\n".join(requirements))
    sections.append("=== GENERAL CONSTRAINTS (still apply) ===\n" + _bulleted_rules())
    sections.append(
        "Return one complete corrected Verilog module and nothing else: no "
        "markdown fences, no line-number prefixes, no explanations. It must "
        "compile with `iverilog -g2005`."
    )
    return "\n\n".join(sections)


def _build_dc_retry_message(dc_error: str, previous_rtl: str) -> str:
    dc_error = (dc_error or "").strip()
    previous_rtl = (previous_rtl or "").strip()
    guidance = [
        "The previous RTL passed local syntax validation but failed Design Compiler analysis or synthesis.",
    ]
    first_error = _first_error(dc_error)
    if first_error and first_error != dc_error:
        guidance.append(
            "First reported error — fix this one first, later messages may be "
            f"cascade effects of it:\n{first_error}"
        )
    guidance.extend(
        [
            f"Complete Design Compiler output (unabridged):\n{dc_error}",
            "Regenerate a complete corrected RTL module.",
            "Fix the root cause reported by Design Compiler while preserving the same module behavior and interface intent.",
            "Before returning the corrected RTL, perform a global synthesizability audit over the whole module.",
            "Every reg and every writable memory element must have exactly one procedural owner.",
            "Do not fix only the reported line if the same structural bug exists elsewhere.",
        ]
    )

    lower_err = dc_error.lower()
    if "not defined" in lower_err or "is not defined" in lower_err or "ver-956" in lower_err:
        guidance.extend(
            [
                "The error indicates an undefined symbol.",
                "Declare the missing signal with the correct width, or replace it with the intended existing signal.",
                "Do not reference any wire/reg/localparam before declaring it.",
            ]
        )
    if "multiple packed dimensions" in lower_err or "ver-720" in lower_err:
        guidance.extend(
            [
                "The error indicates unsupported multiple packed dimensions.",
                "Use a single packed bus for module ports, for example [(N*W)-1:0] bus.",
                "Do not use declarations like [N-1:0][W-1:0] signal.",
            ]
        )
    if "cannot find the design" in lower_err or "current design is not defined" in lower_err:
        guidance.extend(
            [
                "The top-level module may not have been elaborated correctly.",
                "Ensure the generated file contains exactly one top-level module with the requested module name.",
            ]
        )
    if (
        "driven by more than one source" in lower_err
        or "multiple drivers" in lower_err
        or "elab-366" in lower_err
    ):
        guidance.extend(
            [
                "The error indicates a multiple-driver problem.",
                "Find every assignment to the reported reg/net/memory and all related state variables.",
                "A reg must be assigned in one and only one always block.",
                "A memory element must not be written from more than one always block.",
                "Merge scattered assignments into a single owner sequential always block.",
                "Do not create a new always block that writes the same reg or memory.",
                "For buffer state such as valid bits, data arrays, read pointers, and write pointers, keep all writes in one centralized state-update block.",
                "Combinational always blocks may compute next-state wires/regs, but they must not assign storage registers that are also assigned in the sequential block.",
                "Use next-state signals if necessary: compute next_* in combinational logic, then assign each storage reg once in a single clocked block.",
                "Check the entire module for this rule before returning, not only the exact net mentioned by Design Compiler.",
            ]
        )

    if previous_rtl:
        guidance.extend(
            [
                "Your complete previous RTL that failed Design Compiler. The "
                "`NNN | ` prefix is the line number and matches the line numbers "
                "in the tool output above; it is for reference only, do NOT copy "
                "the prefixes into your answer.",
                "```verilog",
                _number_lines(previous_rtl),
                "```",
            ]
        )

    guidance.append("General constraints that still apply:\n" + _bulleted_rules())
    guidance.append("Return only one complete corrected synthesizable RTL module. No markdown fences, no line-number prefixes and no explanations.")
    return "\n".join(guidance)


def generate_direct_rtl(
    action: Module5Action,
    *,
    c_path: str | Path,
    out_dir: str | Path,
    spec_text: str,
    llm_features: Dict[str, Any],
    env_path: str | Path = ".env",
    top_name: str = "",
    max_retries: int = 2,
    initial_feedback: str = "",
    previous_rtl_text: str = "",
    attempt_label: str = "attempt",
    enable_jg_verification: bool = False,
    golden_rtl_path: str | Path = "",
    jg_max_retries: int = 1,
    max_tokens: int | None = None,
) -> Dict[str, Any]:
    # Import lazily to avoid an import cycle during pipeline startup:
    # module5 -> RTL_DIRECT_compare.__init__ -> generator/llm_gen -> module5.
    # At execution time both packages are already fully initialized.
    from RTL_DIRECT_compare.validator import validate

    c_path = Path(c_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not c_path.is_file():
        return {
            "success": False,
            "status": "missing_source",
            "generated_verilog_path": "",
            "syntax_ok": False,
            "stderr": f"Edited C file not found: {c_path}",
            "module_name": top_name or action.benchmark,
        }

    c_code = c_path.read_text(encoding="utf-8", errors="replace")
    module_name = (
        str((llm_features or {}).get("module_name", "")).strip()
        or top_name
        or action.benchmark
    )
    module_name = module_name.replace("-", "_")

    client, model = _get_client(env_path)
    prompt_path = out_dir / "rtl_prompt.json"
    raw_response_path = out_dir / "rtl_raw_response.txt"
    rtl_path = out_dir / f"{action.benchmark}.sv"
    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    token_usage_records: list[Dict[str, Any]] = []

    messages = _build_messages(
        action,
        c_code=c_code,
        module_name=module_name,
        spec_text=spec_text,
        llm_features=llm_features,
        interface_contract=(
            golden_interface_contract(golden_rtl_path, verilog_2001=True)
            if golden_rtl_path
            else ""
        ),
    )
    prompt_path.write_text(
        json.dumps({"model": model, "messages": messages}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if initial_feedback:
        messages.append(
            {
                "role": "user",
                "content": _build_dc_retry_message(initial_feedback, previous_rtl_text),
            }
        )
        prompt_path.write_text(
            json.dumps({"model": model, "messages": messages}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if client is None:
        return {
            "success": False,
            "status": "llm_unavailable",
            "model": model,
            "generated_verilog_path": "",
            "syntax_ok": False,
            "stderr": "LLM client unavailable",
            "module_name": module_name,
            "prompt_path": str(prompt_path),
            "raw_response_path": str(raw_response_path),
        }

    max_tokens = completion_max_tokens(max_tokens)
    last_error = ""
    # One syntax-feedback message is kept in the conversation and rewritten on
    # each failure, so the model always sees the *latest* full RTL rather than
    # several stale copies competing for attention; the headline of every earlier
    # failure is carried along inside it.
    failure_history: List[str] = []
    feedback_index: int | None = None
    # Attempts the provider cut off at the token budget. Tracked separately
    # from syntax failures so a run that never produced a complete answer is
    # reported as a truncation problem rather than a design problem.
    truncated_attempts: List[int] = []
    for attempt in range(max_retries + 1):
        try:
            (attempts_dir / f"{attempt_label}_{attempt}_prompt.json").write_text(
                json.dumps({"model": model, "messages": messages}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                **completion_token_kwargs(max_tokens),
                **seed_kwargs(),
                **qwen_thinking_kwargs(model),
                **deepseek_thinking_kwargs(model),
            )
            token_usage_records.append(
                usage_from_response(
                    stage="direct_rtl",
                    model=model,
                    response=response,
                    attempt=attempt,
                )
            )
            raw_text = response.choices[0].message.content or ""
            raw_response_path.write_text(raw_text, encoding="utf-8")
            (attempts_dir / f"{attempt_label}_{attempt}_raw_response.txt").write_text(raw_text, encoding="utf-8")
            if completion_was_truncated(response):
                # The payload is a prefix of an answer, not a design: the model
                # never reached `endmodule`. Handing it to the syntax gate would
                # report a line number the model never wrote, so the attempt is
                # recorded as a transport failure and the loop stops. Re-issuing
                # the same prompt would reproduce the same cut-off: the budget
                # was spent, and re-asking costs another full-length answer for
                # the same prefix.
                truncated_attempts.append(attempt)
                last_error = truncated_completion_message(response, stage="direct_rtl")
                (
                    attempts_dir / f"{attempt_label}_{attempt}_truncated.txt"
                ).write_text(last_error, encoding="utf-8")
                break
            rtl_text = _sanitize_verilog(raw_text)
            rtl_path.write_text(rtl_text + "\n", encoding="utf-8")
            attempt_rtl_path = attempts_dir / f"{attempt_label}_{attempt}.sv"
            attempt_rtl_path.write_text(rtl_text + "\n", encoding="utf-8")
            syntax_ok, syntax_err = validate(rtl_path)
            if syntax_ok:
                dc_ok, dc_err = _check_dc_compatibility(rtl_text)
                if not dc_ok:
                    syntax_ok = False
                    syntax_err = dc_err

            if syntax_ok:
                # If JG verification is enabled, run it now
                if enable_jg_verification and golden_rtl_path:
                    golden_path = Path(golden_rtl_path).resolve()
                    if not golden_path.is_file():
                        return {
                            "success": False,
                            "status": "golden_rtl_missing",
                            "model": model,
                            "generated_verilog_path": str(rtl_path),
                            "syntax_ok": True,
                            "verified": False,
                            "stderr": f"Golden RTL not found: {golden_path}",
                            "module_name": module_name,
                            "prompt_path": str(prompt_path),
                            "raw_response_path": str(raw_response_path),
                            "llm_token_usage": token_usage_records,
                        }

                    # Run JasperGold verification with iterative correction
                    print(f"[Module5] Running JasperGold verification against {golden_path.name}...")
                    jg_work_dir = PROJECT_ROOT / "module5_mid" / action.benchmark

                    jg_result = verify_rtl_with_jg(
                        generated_verilog=rtl_text,
                        golden_rtl_path=golden_path,
                        c_code=c_code,
                        spec_text=spec_text,
                        module_name=module_name,
                        work_dir=jg_work_dir,
                        env_path=Path(env_path),
                        llm_client=client,
                        llm_model=model,
                        max_retries=jg_max_retries,
                    )

                    if jg_result["success"]:
                        # Update the final verified Verilog
                        rtl_path.write_text(jg_result["final_verilog"] + "\n", encoding="utf-8")
                        return {
                            "success": True,
                            "status": "verified",
                            "model": model,
                            "generated_verilog_path": str(rtl_path),
                            "syntax_ok": True,
                            "verified": True,
                            "jg_attempts": jg_result["attempts"],
                            "stderr": "",
                            "module_name": module_name,
                            "prompt_path": str(prompt_path),
                            "raw_response_path": str(raw_response_path),
                            "llm_token_usage": merge_token_usage(
                                token_usage_records,
                                jg_result.get("llm_token_usage", []),
                            ),
                        }
                    else:
                        return {
                            "success": False,
                            "status": "verification_failed",
                            "model": model,
                            "generated_verilog_path": str(rtl_path),
                            "syntax_ok": True,
                            "verified": False,
                            "jg_attempts": jg_result["attempts"],
                            "stderr": jg_result.get("error", "JasperGold verification failed"),
                            "module_name": module_name,
                            "prompt_path": str(prompt_path),
                            "raw_response_path": str(raw_response_path),
                            "llm_token_usage": merge_token_usage(
                                token_usage_records,
                                jg_result.get("llm_token_usage", []),
                            ),
                        }

                # No JG verification - return success
                return {
                    "success": True,
                    "status": "ok",
                    "model": model,
                    "generated_verilog_path": str(rtl_path),
                    "syntax_ok": True,
                    "stderr": "",
                    "module_name": module_name,
                    "prompt_path": str(prompt_path),
                    "raw_response_path": str(raw_response_path),
                    "llm_token_usage": token_usage_records,
                }

            last_error = syntax_err
            (attempts_dir / f"{attempt_label}_{attempt}_error.txt").write_text(syntax_err, encoding="utf-8")
            if attempt < max_retries:
                feedback = {
                    "role": "user",
                    "content": _build_syntax_retry_message(
                        syntax_err,
                        # The sanitized text is exactly what the validator
                        # compiled, so its line numbers match the error report.
                        rtl_text,
                        attempt=attempt,
                        module_name=module_name,
                        failure_history=tuple(failure_history),
                    ),
                }
                headline = (
                    _first_error(syntax_err)
                    or (syntax_err.strip().splitlines() or ["(no validator output)"])[0]
                )
                failure_history.append(f"attempt {attempt}: {headline}")
                if feedback_index is None:
                    messages.append(feedback)
                    feedback_index = len(messages) - 1
                else:
                    messages[feedback_index] = feedback
        except Exception as exc:
            last_error = str(exc)
            (attempts_dir / f"{attempt_label}_{attempt}_exception.txt").write_text(last_error, encoding="utf-8")
            if attempt >= max_retries:
                break

    return {
        "success": False,
        "status": (
            "llm_truncated" if truncated_attempts else "rtl_failed"
        ),
        "model": model,
        "generated_verilog_path": str(rtl_path) if rtl_path.exists() else "",
        "syntax_ok": False,
        "stderr": last_error,
        "module_name": module_name,
        "prompt_path": str(prompt_path),
        "raw_response_path": str(raw_response_path),
        "attempts_dir": str(attempts_dir),
        "llm_token_usage": token_usage_records,
        "truncated_attempts": truncated_attempts,
    }
