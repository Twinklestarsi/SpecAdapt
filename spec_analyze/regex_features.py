"""
regex_features.py — Rule-based numeric feature extraction from Verilog.

Deterministic, no API calls. Extracts structural, operator, control-flow,
and data features via regex and token counting.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple


# ── Text cleaning ─────────────────────────────────────────────────────

def _strip_comments(src: str) -> str:
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    return src


def _strip_strings(src: str) -> str:
    return re.sub(r'"[^"]*"', '""', src)


def _count(pattern: str, src: str, flags: int = 0) -> int:
    return len(re.findall(pattern, src, flags))


# ── Range parser ──────────────────────────────────────────────────────

_RANGE_RE = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]")


def _port_width(text: str) -> int:
    m = _RANGE_RE.search(text)
    if m:
        return abs(int(m.group(1)) - int(m.group(2))) + 1
    return 1


# ── Port parser ───────────────────────────────────────────────────────

def _parse_ports(src: str) -> Tuple[List[int], List[int]]:
    input_widths: List[int] = []
    output_widths: List[int] = []

    for line in src.splitlines():
        ls = line.strip().rstrip(",")
        if re.match(r"^\s*input\b", ls):
            w = _port_width(ls)
            names = re.sub(
                r"^\s*input\s+(?:wire\s+|reg\s+|logic\s+)?"
                r"(?:signed\s+)?(?:\[.*?\]\s*)?",
                "", ls,
            )
            names = names.rstrip(";,").strip()
            n = max(len([p for p in names.split(",") if p.strip()]), 1)
            input_widths.extend([w] * n)
        elif re.match(r"^\s*output\b", ls):
            w = _port_width(ls)
            names = re.sub(
                r"^\s*output\s+(?:wire\s+|reg\s+|logic\s+)?"
                r"(?:signed\s+)?(?:\[.*?\]\s*)?",
                "", ls,
            )
            names = names.rstrip(";,").strip()
            n = max(len([p for p in names.split(",") if p.strip()]), 1)
            output_widths.extend([w] * n)

    return input_widths, output_widths


# ── Nesting depth ─────────────────────────────────────────────────────

def _max_nesting(src: str) -> int:
    depth = 0
    max_depth = 0
    for tok in re.findall(r"\bbegin\b|\bend\b|\bif\b|\bcase[xz]?\b", src):
        if tok in ("begin", "if") or tok.startswith("case"):
            depth += 1
            max_depth = max(max_depth, depth)
        elif tok == "end":
            depth = max(0, depth - 1)
    return max_depth


# ── FSM detector ──────────────────────────────────────────────────────

def _detect_fsm(raw_src: str, clean_src: str) -> Tuple[int, int]:
    state_params = re.findall(
        r"\b(?:localparam|parameter)\s+"
        r"(\w*(?:[Ss][Tt][Aa][Tt][Ee]|[Ss][Tt][Aa][Tt][Uu][Ss]|IDLE|ST_)\w*)\s*=",
        clean_src,
    )
    define_states = re.findall(
        r"`define\s+(\w*(?:[Ss][Tt][Aa][Tt][Ee]|[Ss][Tt][Aa][Tt][Uu][Ss]|IDLE|ST_)\w*)",
        raw_src,
    )
    all_states = state_params + define_states

    if not all_states:
        blocks = re.findall(
            r"(?:localparam\s+\w+\s*=\s*\d+\s*;\s*){3,}", clean_src
        )
        if blocks:
            all_states = re.findall(r"localparam\s+(\w+)\s*=\s*\d+", blocks[0])

    has_fsm = int(bool(all_states) and bool(re.search(r"\bcase\b", clean_src)))
    return has_fsm, len(all_states)


# ── Instantiation counter ────────────────────────────────────────────

_VERILOG_KEYWORDS = {
    "module", "endmodule", "input", "output", "inout", "wire", "reg",
    "logic", "assign", "always", "initial", "if", "else", "case",
    "casex", "casez", "for", "while", "begin", "end", "function",
    "task", "generate", "endgenerate", "parameter", "localparam",
    "integer", "real", "genvar", "default", "posedge", "negedge",
}


def _count_instantiations(src: str) -> int:
    count = 0
    for m in re.finditer(
        r"(?<![.\w])"
        r"([A-Za-z_]\w*)"
        r"\s+"
        r"(?:#\s*\(.*?\)\s+)?"
        r"([A-Za-z_]\w*)"
        r"\s*\(",
        src, re.DOTALL,
    ):
        if m.group(1).lower() not in _VERILOG_KEYWORDS:
            count += 1
    return count


# ── Clock domain detector ────────────────────────────────────────────

def _detect_clocks(src: str) -> Tuple[int, List[str]]:
    edge_sigs = set(re.findall(r"(?:posedge|negedge)\s+(\w+)", src))
    reset_pat = re.compile(r"rst|reset|arst", re.I)
    clocks = [s for s in edge_sigs if not reset_pat.search(s)]
    return len(clocks), sorted(clocks)


# ── Main extraction function ─────────────────────────────────────────

def extract(vpath: Path, subcategory: str = "", benchmark_name: str = "") -> Dict:
    """
    Extract all regex-based features from a Verilog file.

    Returns a flat dict of feature_name -> value.
    """
    raw = vpath.read_text(errors="replace")
    src = _strip_strings(_strip_comments(raw))

    lines = raw.splitlines()
    loc = len(lines)
    loc_nonblank = sum(1 for l in lines if l.strip())

    # Modules defined
    module_names = re.findall(r"\bmodule\s+(\w+)", src)
    num_modules_defined = len(module_names)
    module_name = module_names[0] if module_names else ""

    # Always blocks
    num_always = _count(r"\balways\b", src)

    # Instantiations
    num_inst = _count_instantiations(src)

    # Ports
    input_widths, output_widths = _parse_ports(src)
    num_inputs = len(input_widths)
    num_outputs = len(output_widths)

    # Sequential / reset
    is_sequential = int(bool(
        re.search(r"\balways\s*@\s*\(\s*(?:posedge|negedge)\b", src)
    ))
    has_async_reset = int(bool(re.search(
        r"\balways\s*@?\s*\(.*(?:posedge|negedge)\s+\w*"
        r"(?:[Rr]eset|[Rr]st|RST|do_reset)\w*",
        src, re.DOTALL,
    )))
    has_sync_reset = int(bool(
        is_sequential
        and re.search(r"\bif\s*\(\s*!?\s*\w*(?:[Rr]eset|[Rr]st|RST|do_reset)\w*", src)
        and not has_async_reset
    ))

    has_signed = int(bool(re.search(r"\bsigned\b", src)))
    num_regs = _count(r"^\s*reg\b", src, re.MULTILINE)

    # FSM
    has_fsm, num_fsm_states = _detect_fsm(raw, src)

    # Clock domains
    num_clk_dom, clk_sigs = _detect_clocks(src)

    # Generate
    has_generate = int(bool(re.search(r"\bgenerate\b", src)))

    # Operators (on body, outside declarations)
    body = re.sub(r"\bmodule\b.*?;", "", src, count=1, flags=re.DOTALL)
    body_ops = re.sub(
        r"^\s*(?:reg|wire|input|output|inout|parameter|localparam|genvar|integer)\b[^;]*;",
        "", body, flags=re.MULTILINE,
    )

    num_add_sub = _count(r"(?<![<>=!&|*/])[+\-](?![>])", body_ops)
    num_multiply = _count(r"\*(?!\*)", body_ops)
    num_divide_mod = _count(r"[/%]", body_ops)
    num_shift = _count(r"<<|>>|<<<|>>>", body_ops)
    num_bitwise = _count(r"(?<![&|])[&|^~](?![&|])", body_ops)
    num_reduction = _count(r"[&|^~]\s*\w+\s*(?:;|\))", body_ops)
    num_comparison = _count(r"[=!]=|[<>]=?", body_ops)
    num_ternary = _count(r"\?", body_ops)
    num_logical = _count(r"&&|\|\|", body_ops)

    # Control flow
    num_if = _count(r"\bif\b", src)
    num_else = _count(r"\belse\b", src)
    num_case = _count(r"\bcase[xz]?\b", src)
    num_assign_stmt = _count(r"<=|(?<![<>=!])=(?!=)", body_ops)
    max_nest = _max_nesting(src)

    # Data features
    num_numeric_constants = _count(
        r"\b\d+\'[bBoOdDhH][\da-fA-F_xXzZ]+\b|\b\d+\b", body_ops
    )
    has_concat = int(bool(re.search(r"\{.*\}", body_ops, re.DOTALL)))
    has_bit_sel = int(bool(re.search(r"\w+\s*\[\s*\d+\s*:\s*\d+\s*\]", body_ops)))
    num_params = _count(r"\b(?:parameter|localparam)\b", src)

    # Derived ratios
    denom = max(loc_nonblank, 1)
    mux_density = round((num_ternary + num_if + num_case) / denom, 4)
    arithmetic_density = round(
        (num_add_sub + num_multiply + num_divide_mod) / denom, 4
    )
    total_stmts = max(num_assign_stmt + num_if + num_case, 1)
    control_ratio = round((num_if + num_else + num_case) / total_stmts, 4)

    return {
        "benchmark": benchmark_name or vpath.stem,
        "subcategory": subcategory,
        "verilog_path": str(vpath),
        "loc": loc,
        "loc_nonblank": loc_nonblank,
        "num_modules_defined": num_modules_defined,
        "module_name": module_name,
        "num_always_blocks": num_always,
        "num_instantiations": num_inst,
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "num_ports": num_inputs + num_outputs,
        "total_input_bits": sum(input_widths),
        "total_output_bits": sum(output_widths),
        "max_input_width": max(input_widths) if input_widths else 0,
        "max_output_width": max(output_widths) if output_widths else 0,
        "is_sequential": is_sequential,
        "has_async_reset": has_async_reset,
        "has_sync_reset": has_sync_reset,
        "has_signed": has_signed,
        "num_regs": num_regs,
        "has_fsm": has_fsm,
        "num_fsm_states": num_fsm_states,
        "num_clock_domains": num_clk_dom,
        "clock_signals": ";".join(clk_sigs),
        "has_generate": has_generate,
        "num_add_sub": num_add_sub,
        "num_multiply": num_multiply,
        "num_divide_mod": num_divide_mod,
        "num_shift": num_shift,
        "num_bitwise": num_bitwise,
        "num_reduction": num_reduction,
        "num_comparison": num_comparison,
        "num_ternary": num_ternary,
        "num_logical_and_or": num_logical,
        "num_if": num_if,
        "num_else": num_else,
        "num_case": num_case,
        "num_assign_stmt": num_assign_stmt,
        "max_nesting_depth": max_nest,
        "num_numeric_constants": num_numeric_constants,
        "has_concatenation": has_concat,
        "has_bit_select": has_bit_sel,
        "num_params": num_params,
        "mux_density": mux_density,
        "arithmetic_density": arithmetic_density,
        "control_ratio": control_ratio,
    }
