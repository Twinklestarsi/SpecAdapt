#!/usr/bin/env python3
"""
RTL-derived C Optimizer — Rule-Based (No API Required)

Applies 14 transformation strategies to each .c file under benchmark_output/
using deterministic, pattern-based code transformation.

Output layout:
  optimized_output/<category>/<block>/<stem>_<TRANSFORM>.c
  optimized_output/<category>/<block>/<stem>_transforms.json
  optimized_output/grand_summary.json
"""

import re, os, json, textwrap
from pathlib import Path
from collections import Counter, defaultdict

BENCHMARK_DIR      = Path(__file__).parent / "benchmark_output"
OUTPUT_DIR         = Path(__file__).parent / "optimized_output"
LLM_OUTPUT_DIR     = Path(__file__).parent / "optimized_output_LLM"
AREA_OUTPUT_DIR    = Path(__file__).parent / "optimized_output_AREA"

# Timing-only transforms (not in the area script)
TRANSFORMS_TIMING_ONLY = [
    "BALANCE_TREE", "REASSOCIATE_ARITHMETIC", "BREAK_CHAIN", "SPLIT_OP",
    "INLINE_CRITICAL_FUNCTION", "OUTLINE_LONG_COMPUTE", "LOOP_FISSION",
    "LOOP_INTERCHANGE", "CONTROL_FLATTEN", "IF_CONVERSION", "SPECULATIVE_COMPUTE",
    "DEPENDENCE_BREAK", "PREDICATE_TO_DATAFLOW",
    "LOOP_PIPELINING", "PARTIAL_UNROLL", "PIPELINE_STAGE_INSERT",
    "MUX_TREE_BALANCE", "CARRY_SAVE_REWRITE", "GUARD_RELAXATION",
]

# Transforms shared with the area script (identical rule-based logic).
# In rule-based mode these produce byte-identical output, so the timing
# runner can symlink/copy the area result instead of re-running.
# In LLM mode the optimisation_goal differs, so they must run separately.
TRANSFORMS_SHARED = [
    "COMMON_SUBEXPR_EXTRACT",
    "COPY_PROPAGATION", "ALGEBRAIC_SIMPLIFY", "BOOLEAN_TO_ARITHMETIC",
    "ENCODE_ONEHOT_TO_BINARY",
]

TRANSFORMS = TRANSFORMS_TIMING_ONLY + TRANSFORMS_SHARED

# ─── Utility helpers ──────────────────────────────────────────────────────────

def header_comment(transform: str, changed: str, benefit: str) -> str:
    return (f"/* TRANSFORM: {transform}\n"
            f"   Changed: {changed}\n"
            f"   Benefit: {benefit}\n */\n")

def meta_line(applied: bool, summary: str) -> str:
    d = json.dumps({"applied": applied, "summary": summary})
    return f"\n// TRANSFORM_META: {d}\n"

def split_file(code: str):
    """Split file into: (preamble, body_functions, main_block).
    preamble  = everything up to and including the struct definition.
    body_functions = list of non-main void functions (as strings).
    main_block = the void main() function as a string.
    """
    lines = code.split('\n')
    # find the split points by tracking void declarations
    func_starts = []
    for i, ln in enumerate(lines):
        if re.match(r'^void\s+', ln.strip()):
            func_starts.append(i)
    if not func_starts:
        return code, [], ""
    preamble = '\n'.join(lines[:func_starts[0]])
    funcs = []
    for idx, start in enumerate(func_starts):
        end = func_starts[idx + 1] if idx + 1 < len(func_starts) else len(lines)
        funcs.append('\n'.join(lines[start:end]))
    # last func with 'void main' is the test harness
    main_block = ""
    body_funcs = []
    for f in funcs:
        if re.match(r'void\s+main\s*\(', f.strip()):
            main_block = f
        else:
            body_funcs.append(f)
    return preamble, body_funcs, main_block

def join_file(preamble, body_funcs, main_block):
    parts = [preamble.rstrip()]
    for f in body_funcs:
        parts.append(f.strip())
    if main_block:
        parts.append(main_block.strip())
    return '\n'.join(parts) + '\n'

def get_old_var_map(func_body: str) -> dict:
    """Return {field_name: old_var_name} from lines like:
       count_of_old = su_block.count_of;
    """
    mapping = {}
    for m in re.finditer(r'(\w+_old)\s*=\s*\w+\.(\w+)\s*;', func_body):
        old_var, field = m.group(1), m.group(2)
        mapping[field] = old_var
    return mapping

def get_struct_name(code: str) -> str:
    m = re.search(r'struct\s+state_elements_(\w+)', code)
    return m.group(1) if m else ""

def get_state_var(code: str) -> str:
    """Return the state variable name, e.g. su_block_13."""
    name = get_struct_name(code)
    if not name:
        return ""
    m = re.search(r'struct\s+state_elements_\w+\s+(\w+)\s*;', code)
    return m.group(1) if m else f"s{name}"

def eval_const_expr(expr: str) -> str:
    """Evaluate simple constant arithmetic expressions."""
    try:
        # Only allow safe literals and arithmetic ops
        cleaned = re.sub(r'\s+', '', expr)
        if re.fullmatch(r'[\d\+\-\*\/\(\)]+', cleaned):
            result = eval(cleaned)  # safe: only digits and operators
            if isinstance(result, int) and result >= 0:
                return str(result)
    except Exception:
        pass
    return expr

def fold_const_exprs(expr: str) -> str:
    """Replace constant sub-expressions like '4294967295 - 2047' with value."""
    def replacer(m):
        return eval_const_expr(m.group(0))
    return re.sub(r'\d+\s*[-+*]\s*\d+', replacer, expr)

# ─── if-else block parser ─────────────────────────────────────────────────────

def find_matching_brace(text: str, open_pos: int) -> int:
    """Find the closing brace matching the { at open_pos."""
    depth = 0
    i = open_pos
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1

def extract_block_content(text: str, start: int) -> tuple:
    """Given text[start]=='(', return (inner, end_pos)."""
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return text[start+1:i], i
        i += 1
    return "", len(text)-1

# ─── Transform base ───────────────────────────────────────────────────────────

class Transform:
    name = "BASE"
    def apply(self, code: str) -> tuple:
        """Returns (new_code, applied: bool, summary: str)."""
        raise NotImplementedError

    def run(self, code: str) -> str:
        new_code, applied, summary = self.apply(code)
        tag = meta_line(applied, summary)
        return new_code.rstrip() + tag

# ─── 1. BALANCE_TREE ─────────────────────────────────────────────────────────

class BalanceTree(Transform):
    name = "BALANCE_TREE"

    def balance(self, operands: list, op: str) -> str:
        if len(operands) == 1:
            return operands[0].strip()
        mid = len(operands) // 2
        left  = self.balance(operands[:mid], op)
        right = self.balance(operands[mid:], op)
        return f"({left} {op} {right})"

    def try_balance_expr(self, expr: str) -> tuple:
        """Try to balance a flat addition chain of 4+ operands."""
        # Only handle simple + chains with cast-prefixed terms
        # e.g.: (unsigned int)a + (unsigned int)b + (unsigned int)c + (unsigned int)d
        parts = re.split(r'\s*\+\s*', expr)
        if len(parts) >= 4:
            balanced = self.balance(parts, '+')
            if balanced != f"({expr})":
                return balanced, True
        return expr, False

    def apply(self, code: str) -> tuple:
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed = False
        for func in body_funcs:
            lines = func.split('\n')
            new_lines = []
            for ln in lines:
                # find assignment with 4+ additions
                m = re.match(r'^(\s*\S.*?=\s*)(.+)(\s*;.*)$', ln)
                if m:
                    lhs, rhs, tail = m.group(1), m.group(2), m.group(3)
                    # count top-level + operators (not inside parens/casts)
                    top_plus = []
                    depth = 0
                    for idx, ch in enumerate(rhs):
                        if ch in '([': depth += 1
                        elif ch in ')]': depth -= 1
                        elif ch == '+' and depth == 0:
                            top_plus.append(idx)
                    if len(top_plus) >= 3:
                        # split at top-level +
                        splits = [-1] + top_plus + [len(rhs)]
                        operands = [rhs[splits[i]+1:splits[i+1]].strip()
                                    for i in range(len(splits)-1)]
                        balanced = self.balance(operands, '+')
                        new_lines.append(lhs + balanced + tail)
                        changed = True
                        continue
                new_lines.append(ln)
            new_funcs.append('\n'.join(new_lines))

        if not changed:
            hdr = header_comment(self.name,
                "No 4+ operand addition chains found",
                "N/A — transformation not applicable")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No balanceable chains found"

        hdr = header_comment(self.name,
            "Balanced flat addition chains into binary tree form",
            "Reduces critical-path depth from O(N) to O(log N) addition levels")
        result = preamble + "\n" + hdr + '\n'.join(new_funcs + [main_block])
        return result, True, "Balanced addition chains into binary tree"

# ─── 2. REASSOCIATE_ARITHMETIC ───────────────────────────────────────────────

class ReassociateArithmetic(Transform):
    name = "REASSOCIATE_ARITHMETIC"

    def apply(self, code: str) -> tuple:
        # 1) Fold constant sub-expressions: e.g. 4294967295 - 2047 → 4294965248
        #    Apply repeatedly until no more changes (handles nested: 4294967295 - 4294965248)
        new_code = code
        for _ in range(5):
            folded = re.sub(r'\b(\d{4,})\s*([-+])\s*(\d+)\b', self._fold, new_code)
            if folded == new_code:
                break
            new_code = folded
        # 2) Remove no-op masks: & 4294967295 (32-bit all-ones mask)
        new_code = re.sub(r'\s*&\s*4294967295\b', '', new_code)
        # 3) Simplify << 0 (shift by zero)
        new_code = re.sub(r'\s*<<\s*0\b', '', new_code)
        # 4) Simplify & 0 expressions → 0 (only standalone)
        new_code = re.sub(r'\(\w+\s*&\s*0\)', '0', new_code)
        # 5) Fold i = (int)i + 1 & 4294967295 patterns in for-loops
        new_code = re.sub(r'\(int\)(\w+)\s*\+\s*1\s*&\s*4294967295', r'\1 + 1', new_code)
        # 6) Fold (i = 0 & 4294967295) → i = 0
        new_code = re.sub(r'(\w+)\s*=\s*(\w+)\s*&\s*4294967295', r'\1 = \2', new_code)

        changed = new_code != code
        if not changed:
            hdr = header_comment(self.name,
                "No constant arithmetic folding opportunities found",
                "N/A — transformation not applicable")
            return hdr + code, False, "No reassociatable patterns found"

        hdr = header_comment(self.name,
            "Folded constant sub-expressions and removed no-op masks/shifts",
            "Reduces expression depth and clarifies mask constants for HLS synthesis")
        return hdr + new_code, True, "Folded constants and simplified masks/shifts"

    def _fold(self, m):
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        try:
            if op == '-': return str(a - b)
            if op == '+': return str(a + b)
            if op == '*': return str(a * b)
        except Exception:
            pass
        return m.group(0)

# ─── 3. BREAK_CHAIN ──────────────────────────────────────────────────────────

class BreakChain(Transform):
    name = "BREAK_CHAIN"

    def apply(self, code: str) -> tuple:
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            lines = func.split('\n')
            new_lines = []
            temp_idx = [0]
            func_changed = False

            for ln in lines:
                m = re.match(r'^(\s*)(\w[\w. >*\[\]]+)\s*=\s*(.+?)\s*;(\s*(?://.*)?)\s*$', ln)
                if m:
                    indent, lvalue, rhs, comment = m.groups()
                    # Split long && or || chains (4+ operands) into two temps
                    # Count top-level && or ||
                    ops, op_sym = self._find_top_ops(rhs)
                    if len(ops) >= 3 and op_sym:
                        splits = [-1] + ops + [len(rhs)]
                        operands = [rhs[splits[i]+1:splits[i+1]].strip()
                                    for i in range(len(splits)-1)]
                        mid = len(operands) // 2
                        t0 = f"_chain_t{temp_idx[0]}"
                        t1 = f"_chain_t{temp_idx[0]+1}"
                        temp_idx[0] += 2
                        left_expr  = f" {op_sym} ".join(operands[:mid])
                        right_expr = f" {op_sym} ".join(operands[mid:])
                        # Insert temps before assignment
                        # Find insertion point — before the enclosing block
                        # (insert right before this line)
                        ty = "_Bool" if op_sym in ("&&", "||") else "unsigned int"
                        new_lines.append(f"{indent}{ty} {t0} = {left_expr};")
                        new_lines.append(f"{indent}{ty} {t1} = {right_expr};")
                        new_lines.append(f"{indent}{lvalue} = {t0} {op_sym} {t1};{comment}")
                        func_changed = True
                        continue
                new_lines.append(ln)

            new_funcs.append('\n'.join(new_lines))
            if func_changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No long dependency chains (4+ ops) detected",
                "N/A — transformation not applicable")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No breakable chains found"

        hdr = header_comment(self.name,
            "Split long &&/|| chains into two parallel sub-expressions",
            "Enables parallel evaluation of sub-chains, reducing critical-path depth")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Broke long &&/|| chains into parallel sub-chains"

    def _find_top_ops(self, rhs: str):
        """Find top-level && or || positions. Returns (positions, symbol)."""
        depth = 0
        and_pos, or_pos = [], []
        i = 0
        while i < len(rhs):
            ch = rhs[i]
            if ch in '([': depth += 1
            elif ch in ')]': depth -= 1
            elif depth == 0:
                if rhs[i:i+2] == '&&':
                    and_pos.append(i)
                    i += 2
                    continue
                elif rhs[i:i+2] == '||':
                    or_pos.append(i)
                    i += 2
                    continue
            i += 1
        if len(and_pos) >= 3:
            return and_pos, '&&'
        if len(or_pos) >= 3:
            return or_pos, '||'
        return [], None

# ─── 4. SPLIT_OP ─────────────────────────────────────────────────────────────

class SplitOp(Transform):
    name = "SPLIT_OP"

    def apply(self, code: str) -> tuple:
        """Split compound boolean+comparison expressions.
        E.g.:  result = !lrg_ok && sizu_c > max_pl_sz;
          →    _Bool _cmp0 = sizu_c > max_pl_sz;
               _Bool _neg0 = !lrg_ok;
               result = _neg0 && _cmp0;
        """
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            lines = func.split('\n')
            new_lines = []
            tidx = [0]
            func_changed = False

            for ln in lines:
                m = re.match(r'^(\s*)(\w[\w. >*\[\]]+)\s*=\s*(.+?)\s*;(\s*(?://.*)?)\s*$', ln)
                if m:
                    indent, lvalue, rhs, comment = m.groups()
                    # Look for: expr1 && expr2 where one is a comparison
                    # Pattern: X && Y > Z  or !X && Y > Z
                    mp = re.match(
                        r'^(!?\w[\w.()\[\] ]*?)\s*&&\s*(\w[\w.()\[\] ]*?\s*[<>!=]=?\s*\w[\w.()\[\] ]*?)$',
                        rhs.strip())
                    if mp:
                        left_part = mp.group(1).strip()
                        right_part = mp.group(2).strip()
                        t_cmp = f"_cmp{tidx[0]}"
                        t_lft = f"_pred{tidx[0]}"
                        tidx[0] += 1
                        new_lines.append(f"{indent}_Bool {t_cmp} = {right_part};")
                        new_lines.append(f"{indent}_Bool {t_lft} = {left_part};")
                        new_lines.append(f"{indent}{lvalue} = {t_lft} && {t_cmp};{comment}")
                        func_changed = True
                        continue
                new_lines.append(ln)

            new_funcs.append('\n'.join(new_lines))
            if func_changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No splittable compound conditions found",
                "N/A — transformation not applicable")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No splittable operations found"

        hdr = header_comment(self.name,
            "Split compound (bool && comparison) into separate named sub-operations",
            "Exposes the comparison as an independent datapath, enabling parallel compute")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Split compound bool+comparison into separate operations"

# ─── 5. INLINE_CRITICAL_FUNCTION ─────────────────────────────────────────────

class InlineCriticalFunction(Transform):
    name = "INLINE_CRITICAL_FUNCTION"

    def apply(self, code: str) -> tuple:
        preamble, body_funcs, main_block = split_file(code)
        if len(body_funcs) < 2:
            # No helpers to inline
            hdr = header_comment(self.name,
                "No helper functions to inline",
                "N/A — file has only one function")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No helper functions present"

        # Identify small helper functions (< 8 body lines) that are CALLED
        # by at least one other function in this file
        all_names = set()
        parsed = {}
        for func in body_funcs:
            info = self._parse_func(func)
            if info:
                all_names.add(info['name'])
                parsed[info['name']] = (info, func)

        # Find which names are actually called somewhere
        called_names = set()
        for name, (info, raw) in parsed.items():
            for other_name, (_, other_raw) in parsed.items():
                if other_name != name and re.search(r'\b' + re.escape(name) + r'\s*\(', other_raw):
                    called_names.add(name)

        helpers = {}
        keep_funcs = []
        for func in body_funcs:
            info = self._parse_func(func)
            if info and info['name'] in called_names and len(info['body_lines']) <= 8:
                helpers[info['name']] = info
            else:
                keep_funcs.append(func)

        if not helpers:
            hdr = header_comment(self.name,
                "All helper functions too large to inline",
                "N/A — inlining only safe for small functions")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No small helper functions to inline"

        # Inline in the remaining non-main functions
        changed_any = False
        new_keep = []
        for func in keep_funcs:
            new_func, ch = self._inline_calls(func, helpers)
            new_keep.append(new_func)
            if ch:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "Helper functions exist but no inlinable call sites found",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No inlinable call sites found"

        hdr = header_comment(self.name,
            f"Inlined helper function(s): {', '.join(helpers.keys())}",
            "Eliminates function-call overhead and exposes cross-boundary optimizations")
        return preamble + "\n" + hdr + '\n'.join(new_keep+[main_block]), True, \
               f"Inlined: {', '.join(helpers.keys())}"

    def _parse_func(self, func: str):
        lines = [l for l in func.split('\n')]
        m = re.match(r'void\s+(\w+)\s*\((.+?)\)\s*$', lines[0].strip(), re.DOTALL)
        if not m:
            # try multi-line signature
            sig = ''
            for i, ln in enumerate(lines):
                sig += ln
                if ')' in ln:
                    m = re.match(r'void\s+(\w+)\s*\((.+)\)\s*$', sig.replace('\n', ' ').strip())
                    if m:
                        break
        if not m:
            return None
        name = m.group(1)
        params_str = m.group(2)
        # Parse params: type name, type *name, ...
        params = []
        for p in params_str.split(','):
            p = p.strip()
            pm = re.match(r'(.+?)\s+\*?(\w+)\s*$', p)
            if pm:
                params.append({'type': pm.group(1).strip(),
                               'name': pm.group(2).strip(),
                               'ptr': '*' in p})
        body_lines = [l for l in lines[1:] if l.strip() not in ('{', '}', '')]
        return {'name': name, 'params': params,
                'body_lines': body_lines, 'raw': func}

    def _inline_calls(self, func: str, helpers: dict) -> tuple:
        """Replace calls like helper(a,b,c,&out) with inlined body."""
        changed = False
        for hname, hinfo in helpers.items():
            # Match: hname(arg1, arg2, ..., &out_arg);
            pat = re.compile(r'(\s*)' + re.escape(hname) + r'\s*\(([^)]+)\)\s*;')
            def replacer(m, hinfo=hinfo, hname=hname):
                indent = m.group(1)
                args_str = m.group(2)
                args = [a.strip() for a in args_str.split(',')]
                params = hinfo['params']
                if len(args) != len(params):
                    return m.group(0)
                # Build substitution map
                subst = {}
                for p, a in zip(params, args):
                    pname = p['name']
                    if p['ptr']:
                        # pointer param: *pname = val → actual = val
                        actual = a.lstrip('&')
                        subst[f'*{pname}'] = actual    # *y → sel1_out
                        subst[f'&{pname}'] = a         # &y → &sel1_out
                        subst[pname] = actual           # y  → sel1_out
                    else:
                        subst[pname] = a
                # Substitute body
                result_lines = [f"{indent}/* inlined: {hname}({args_str}) */"]
                for bl in hinfo['body_lines']:
                    new_bl = bl
                    # Replace longest matches first
                    for old, new in sorted(subst.items(), key=lambda x: -len(x[0])):
                        new_bl = re.sub(r'\b' + re.escape(old) + r'\b', new, new_bl)
                    result_lines.append(indent + new_bl.strip())
                return '\n'.join(result_lines)

            new_func = pat.sub(replacer, func)
            if new_func != func:
                func = new_func
                changed = True
        return func, changed

# ─── 6. OUTLINE_LONG_COMPUTE ─────────────────────────────────────────────────

class OutlineLongCompute(Transform):
    name = "OUTLINE_LONG_COMPUTE"

    def apply(self, code: str) -> tuple:
        """Extract the enable-block of large functions into a helper."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        outlines = []
        changed_any = False

        for func in body_funcs:
            # Find the main function name and its 'enable' block
            fname_m = re.match(r'void\s+(\w+)\s*\(', func.strip())
            if not fname_m:
                new_funcs.append(func)
                continue
            fname = fname_m.group(1)

            # Find large if(enable) { ... } blocks (> 10 assignments)
            # Pattern: else\n    if(enable)\n    {\n      ... many assignments ...\n    }
            enable_m = re.search(
                r'(else\s*\n\s*if\s*\(\s*\w+\s*\)\s*\n\s*\{)([^}]{500,}?)(\n\s*\})',
                func, re.DOTALL)
            if not enable_m:
                new_funcs.append(func)
                continue

            enable_intro = enable_m.group(1)
            enable_body  = enable_m.group(2)
            enable_close = enable_m.group(3)

            # Count assignments in block
            n_assigns = enable_body.count('= ')
            if n_assigns < 8:
                new_funcs.append(func)
                continue

            # Extract condition variable from if(enable_var)
            cond_m = re.search(r'if\s*\(\s*(\w+)\s*\)', enable_intro)
            cond_var = cond_m.group(1) if cond_m else "enable"

            # Extract all variables written in enable block
            written = re.findall(r'\w+\.(\w+)\s*=', enable_body)
            written_unique = list(dict.fromkeys(written))

            # Build helper function name
            helper_name = f"{fname}_compute_enable"

            # Build helper signature: takes all state pointer + old vars
            # Simpler: take the state struct pointer
            svar = get_state_var(func)
            struct_name = get_struct_name(code)

            # Collect ALL local variable declarations in the function
            # (these are variables declared before the enable block that
            #  may be referenced inside it, e.g. _old vars, orc_N, etc.)
            local_decls = {}  # var_name -> type_string
            for dm in re.finditer(
                    r'((?:unsigned\s+)?(?:char|int|short|long|_Bool))\s+'
                    r'(\w+)\s*[;=]', func):
                vtype = dm.group(1).strip()
                vname = dm.group(2).strip()
                # Skip the state struct variable itself
                if vname == svar:
                    continue
                local_decls[vname] = vtype

            # Filter to only those actually referenced in the enable body
            used_locals = {}
            for vname, vtype in local_decls.items():
                if re.search(r'\b' + re.escape(vname) + r'\b', enable_body):
                    used_locals[vname] = vtype

            # Build helper
            params = [f"struct state_elements_{struct_name} *s"]
            for vname, vtype in used_locals.items():
                params.append(f"{vtype} {vname}")
            helper_sig = (f"void {helper_name}("
                          + ", ".join(params) + ")")
            # Replace svar. → s-> in body
            helper_body = re.sub(re.escape(svar) + r'\.', 's->', enable_body)
            helper_func = (helper_sig + "\n{\n" + helper_body + "\n}\n")
            outlines.append(helper_func)

            # Replace enable block in original with helper call
            call_args = ["&" + svar] + list(used_locals.keys())
            call = (f"{enable_intro}\n"
                    f"      {helper_name}({', '.join(call_args)});\n"
                    f"{enable_close}")
            new_func = func[:enable_m.start()] + call + func[enable_m.end():]
            new_funcs.append(new_func)
            changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No sufficiently large compute blocks found for outlining",
                "N/A — functions are too small to benefit")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No large compute blocks to outline"

        hdr = header_comment(self.name,
            "Extracted large enable-block into dedicated helper function",
            "Allows HLS to schedule helper independently and aids readability")
        all_funcs = outlines + new_funcs
        return preamble + "\n" + hdr + '\n'.join(all_funcs+[main_block]), True, \
               "Outlined large compute block into helper function"

# ─── 7. LOOP_FISSION ─────────────────────────────────────────────────────────

class LoopFission(Transform):
    name = "LOOP_FISSION"

    def apply(self, code: str) -> tuple:
        """Split a for-loop with multiple independent body statements."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            # Find for-loop with braced multi-statement body
            m = re.search(
                r'([ \t]*for\s*\([^)]+\))\s*\n\s*\{([^}]+)\}',
                func, re.DOTALL)
            if m:
                loop_header = m.group(1).strip()
                loop_body   = m.group(2)
                indent = '  '
                stmts = [s.strip() for s in loop_body.split(';') if s.strip()]
                if len(stmts) >= 2:
                    loops = [f"{indent}{loop_header}\n{indent}  {stmt};"
                             for stmt in stmts]
                    replacement = '\n'.join(loops)
                    new_func = func[:m.start()] + replacement + func[m.end():]
                    new_funcs.append(new_func)
                    changed_any = True
                    continue

            # Also look for popcount-style: for(...) \n    single_stmt; (accumulator)
            # Split into two: half-range each
            m2 = re.search(
                r'([ \t]*)(for\s*\(\s*(\w+)\s*=\s*(\d+)\s*[&\d\s]*;\s*'
                r'\(int\)(\w+)\s*<\s*(\d+)\s*;\s*[^)]+\))\s*\n'
                r'([ \t]+)(\w[\w. >\[\]]*\s*=\s*\w[\w. >\[\]]*\s*\+[^;]+;)',
                func, re.DOTALL)
            if m2:
                base_ind = m2.group(1)
                loop_hdr = m2.group(2)
                var      = m2.group(3)
                start    = int(m2.group(4))
                end      = int(m2.group(6))
                stmt_ind = m2.group(7)
                body_stmt= m2.group(8)
                mid = (start + end) // 2
                l1 = (f"{base_ind}for({var} = {start}; (int){var} < {mid}; {var} = (int){var} + 1)\n"
                      f"{stmt_ind}{body_stmt}")
                l2 = (f"{base_ind}for({var} = {mid}; (int){var} < {end}; {var} = (int){var} + 1)\n"
                      f"{stmt_ind}{body_stmt}")
                replacement = l1 + '\n' + l2
                new_func = func[:m2.start()] + replacement + func[m2.end():]
                new_funcs.append(new_func)
                changed_any = True
                continue

            new_funcs.append(func)

        if not changed_any:
            hdr = header_comment(self.name,
                "No multi-statement loops found for fission",
                "N/A — loops have single-statement bodies")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No multi-statement loops to split"

        hdr = header_comment(self.name,
            "Split multi-statement loop body into independent per-statement loops",
            "Each loop can be pipelined independently by HLS tool")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Applied loop fission to split multi-statement loop"

# ─── 8. LOOP_INTERCHANGE ─────────────────────────────────────────────────────

class LoopInterchange(Transform):
    name = "LOOP_INTERCHANGE"

    def apply(self, code: str) -> tuple:
        """Swap nested for-loop order for better memory locality."""
        # Find nested for-loops
        outer_m = re.search(
            r'([ \t]*)(for\s*\(([^)]+)\))\s*\n\s*\{\s*\n'
            r'\s*(for\s*\(([^)]+)\))\s*\n\s*\{([^}]+)\}\s*\n\s*\}',
            code, re.DOTALL)
        if not outer_m:
            hdr = header_comment(self.name,
                "No nested for-loops found for interchange",
                "N/A — no nested loops present")
            return hdr + code, False, "No nested loops to interchange"

        indent     = outer_m.group(1)
        outer_hdr  = outer_m.group(2)
        inner_hdr  = outer_m.group(4)
        body       = outer_m.group(6)

        # Extract loop variables
        outer_init = outer_m.group(3)
        inner_init = outer_m.group(5)

        swapped = (f"{indent}{inner_hdr}\n{indent}{{\n"
                   f"  {indent}{outer_hdr}\n  {indent}{{\n"
                   f"{body}\n  {indent}}}\n{indent}}}")
        new_code = code[:outer_m.start()] + swapped + code[outer_m.end():]
        hdr = header_comment(self.name,
            "Swapped inner and outer loop order",
            "Improves data locality (inner loop now iterates over contiguous dimension)")
        return hdr + new_code, True, "Interchanged nested loop order"

# ─── 9. CONTROL_FLATTEN ──────────────────────────────────────────────────────

class ControlFlatten(Transform):
    name = "CONTROL_FLATTEN"

    def apply(self, code: str) -> tuple:
        """Convert deeply nested if(v==N) chains to switch statement."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._flatten_func(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No deeply nested if(v==N) chains found",
                "N/A — no convertible if-else ladder")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No if-equality chains to flatten"

        hdr = header_comment(self.name,
            "Converted nested if(var==N) chains to switch statement",
            "Reduces control-flow depth; HLS tool can implement as efficient mux tree")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Converted if-equality chain to switch statement"

    def _flatten_func(self, func: str) -> tuple:
        """Parse line-by-line to detect nested if(cast var == N) chains."""
        lines = func.split('\n')

        # Find a run of lines that form: if(... var == N)\n  stmt;\n\nelse\n  if(... var==N)\n stmt;
        # collect all (start_idx, end_idx, [(var,val,stmt)], default_stmt)
        chain_start = None
        cases = []
        cond_var = None
        i = 0
        while i < len(lines):
            ln = lines[i].rstrip()
            # Look for: if(... VAR == DIGIT)
            m = re.match(r'^\s*if\s*\(\s*(?:\([^)]+\)\s*)?(\w+)\s*==\s*(\d+)\s*\)\s*$', ln)
            if m:
                var, val = m.group(1), m.group(2)
                if cond_var is None:
                    cond_var = var
                    chain_start = i
                if var == cond_var:
                    # Next non-blank line should be the assignment
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines):
                        stmt_ln = lines[j].strip()
                        if stmt_ln.endswith(';') and '=' in stmt_ln:
                            cases.append((val, stmt_ln))
                            i = j + 1
                            # skip blank lines and 'else' lines until next if
                            while i < len(lines) and lines[i].strip() in ('', 'else'):
                                i += 1
                            continue
            i += 1

        if len(cases) < 3 or cond_var is None:
            return func, False

        # Find default: else without a following if(var==)
        default_stmt = None
        dm = re.search(
            r'\belse\s*\n\s+(?!if\s*\(\s*(?:\([^)]+\)\s*)?' + re.escape(cond_var) + r'\s*==)'
            r'(\w[\w. >*\[\]]+\s*=[^;]+;)', func)
        if dm:
            default_stmt = dm.group(1).strip()

        # Find the enclosing { ... } block that contains the chain
        # Look for '  {' before chain_start and matching '  }' after
        block_start = None
        for k in range(chain_start - 1, -1, -1):
            if lines[k].strip() == '{':
                block_start = k
                break

        if block_start is None:
            # No explicit enclosing block — build from scratch around the range
            block_end = len(lines)
        else:
            # Find matching close brace
            block_end = None
            depth = 0
            for k in range(block_start, len(lines)):
                depth += lines[k].count('{') - lines[k].count('}')
                if depth == 0:
                    block_end = k
                    break
            if block_end is None:
                block_end = len(lines)

        # Build switch statement
        ind = '    '
        sw_lines = [f"{ind}switch((unsigned int){cond_var}) {{"]
        for val, stmt in cases:
            sw_lines.append(f"{ind}  case {val}: {stmt} break;")
        if default_stmt:
            sw_lines.append(f"{ind}  default: {default_stmt} break;")
        sw_lines.append(f"{ind}}}")

        new_lines = (lines[:block_start] +
                     ['  {'] + sw_lines + ['  }'] +
                     lines[block_end+1:])
        return '\n'.join(new_lines), True

# ─── 10. IF_CONVERSION ────────────────────────────────────────────────────────

class IfConversion(Transform):
    name = "IF_CONVERSION"

    def apply(self, code: str) -> tuple:
        """Convert if(C) X=A; else X=B; → X = C ? A : B; """
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._convert(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No simple if/else single-assignment pairs found",
                "N/A — no convertible branches")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No convertible if-else pairs"

        hdr = header_comment(self.name,
            "Converted if(C) X=A; else X=B; to ternary X = C ? A : B;",
            "Eliminates branches; HLS synthesizes as parallel datapaths with final mux")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Converted if-else single assignments to ternary expressions"

    def _convert(self, func: str) -> tuple:
        changed = False
        # Pattern: if(COND)\n INDENT LVAL = EXPR_T;\n\n INDENT else\n INDENT LVAL = EXPR_F;
        # Handle both with and without braces
        pat = re.compile(
            r'([ \t]*)if\s*\(([^)]+)\)\s*\n'
            r'(?:\s*\{\s*\n)?'
            r'([ \t]+)([\w][\w. >*\[\]]+)\s*=\s*([^;{}]+);\s*\n'
            r'(?:\s*\}\s*\n)?'
            r'\s*\n?\s*else\s*\n'
            r'(?:\s*\{\s*\n)?'
            r'\s+([\w][\w. >*\[\]]+)\s*=\s*([^;{}]+);\s*\n'
            r'(?:\s*\}\s*\n)?',
            re.MULTILINE)

        def repl(m):
            nonlocal changed
            base_indent = m.group(1)
            cond   = m.group(2).strip()
            lval_t = m.group(4).strip()
            expr_t = m.group(5).strip()
            lval_f = m.group(6).strip()
            expr_f = m.group(7).strip()
            if lval_t == lval_f:
                changed = True
                return (f"{base_indent}{lval_t} = ({cond}) ? ({expr_t}) : ({expr_f});\n")
            return m.group(0)

        new_func = pat.sub(repl, func)
        return new_func, changed

# ─── 11. SPECULATIVE_COMPUTE ─────────────────────────────────────────────────

class SpeculativeCompute(Transform):
    name = "SPECULATIVE_COMPUTE"

    def apply(self, code: str) -> tuple:
        """Pre-compute both branches, then select:
           if(C) X=A; else X=B;
           → _spec_t=A; _spec_f=B; X = C ? _spec_t : _spec_f;
        """
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._speculate(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No if-else single-assignment pairs suitable for speculation",
                "N/A — no speculative compute opportunities")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No speculative compute opportunities"

        hdr = header_comment(self.name,
            "Pre-computed both branch values before condition select",
            "Removes condition from critical path; both computations proceed in parallel")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Added speculative pre-computation for if-else branches"

    def _speculate(self, func: str) -> tuple:
        changed = False
        tidx = [0]

        pat = re.compile(
            r'([ \t]*)if\s*\(([^)]+)\)\s*\n'
            r'(?:\s*\{\s*\n)?'
            r'([ \t]+)([\w][\w. >*\[\]]+)\s*=\s*([^;{}]+);\s*\n'
            r'(?:\s*\}\s*\n)?'
            r'\s*\n?\s*else\s*\n'
            r'(?:\s*\{\s*\n)?'
            r'\s+([\w][\w. >*\[\]]+)\s*=\s*([^;{}]+);\s*\n'
            r'(?:\s*\}\s*\n)?',
            re.MULTILINE)

        # Determine the type of lvalue by looking at struct field type
        def repl(m):
            nonlocal changed
            base_indent = m.group(1)
            cond   = m.group(2).strip()
            lval_t = m.group(4).strip()
            expr_t = m.group(5).strip()
            lval_f = m.group(6).strip()
            expr_f = m.group(7).strip()
            if lval_t == lval_f:
                changed = True
                t_name = f"_spec_{tidx[0]}_t"
                f_name = f"_spec_{tidx[0]}_f"
                tidx[0] += 1
                # Use auto-inferred type via __typeof__ for portability
                return (f"{base_indent}__typeof__({lval_t}) {t_name} = {expr_t};\n"
                        f"{base_indent}__typeof__({lval_t}) {f_name} = {expr_f};\n"
                        f"{base_indent}{lval_t} = ({cond}) ? {t_name} : {f_name};\n")
            return m.group(0)

        new_func = pat.sub(repl, func)
        return new_func, changed

# ─── 12. COMMON_SUBEXPR_EXTRACT ──────────────────────────────────────────────

class CommonSubexprExtract(Transform):
    name = "COMMON_SUBEXPR_EXTRACT"

    def apply(self, code: str) -> tuple:
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        defines_block = ""
        changed_any = False

        # Phase 1: extract repeated constant sub-expressions as #defines
        # Find all decimal integer constant arithmetic expressions
        const_exprs = re.findall(r'\b\d+\s*[-+]\s*\d+\b', code)
        counted = Counter(const_exprs)
        # Only extract if occurs 3+ times
        replacements = {}  # expr_str → macro_name
        define_lines = []
        idx = 0
        for expr, cnt in counted.items():
            if cnt >= 3:
                val = eval_const_expr(expr)
                macro = f"CSE_CONST_{idx}"
                define_lines.append(f"#define {macro} ({val})")
                replacements[expr] = macro
                idx += 1

        if replacements:
            defines_block = '\n'.join(define_lines) + '\n'
            new_code = preamble + '\n' + defines_block
            rest = '\n'.join(body_funcs + [main_block])
            for expr, macro in replacements.items():
                rest = rest.replace(expr, macro)
            changed_any = True
            hdr = header_comment(self.name,
                f"Extracted {len(replacements)} repeated constant expression(s) as #define macros",
                "Reduces logic duplication; HLS shares the computed value across uses")
            return new_code + hdr + rest, True, \
                   f"Extracted {len(replacements)} common constant sub-expression(s)"

        # Phase 2: extract repeated non-trivial sub-expressions within a function
        new_funcs_out = []
        for func in body_funcs:
            new_func, ch, ndefs = self._extract_in_func(func)
            new_funcs_out.append(ndefs + new_func if ndefs else new_func)
            if ch:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No repeated sub-expressions found (threshold: 3 occurrences)",
                "N/A — no common sub-expressions to extract")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No common sub-expressions found"

        hdr = header_comment(self.name,
            "Extracted repeated sub-expressions into named temporaries",
            "Reduces area by sharing logic; clarifies expression structure")
        return preamble + "\n" + hdr + '\n'.join(new_funcs_out+[main_block]), True, \
               "Extracted repeated sub-expressions into named variables"

    def _extract_in_func(self, func: str) -> tuple:
        """Extract repeated non-trivial sub-expressions within a function."""
        # Find all parenthesized sub-expressions
        subexprs = re.findall(r'\((?:[^()]+)\)', func)
        counted = Counter(subexprs)
        replacements = {}
        new_decls = []
        idx = 0
        for expr, cnt in counted.items():
            # Skip trivial: single token, cast expressions
            inner = expr[1:-1].strip()
            if cnt >= 2 and len(inner) > 5 and not re.match(r'^[\w ]+$', inner):
                if not re.match(r'^(?:unsigned|signed|int|char|short|long|_Bool)', inner):
                    name = f"_cse{idx}"
                    # Determine a suitable type (use __typeof__ or just unsigned int)
                    new_decls.append(f"  unsigned int {name} = {expr};")
                    replacements[expr] = name
                    idx += 1

        if not replacements:
            return func, False, ""

        # Only apply if we found meaningful ones (non-cast)
        new_func = func
        for expr, name in replacements.items():
            new_func = new_func.replace(expr, name)

        # Insert declarations at start of function body
        insert_after = new_func.find('{')
        if insert_after == -1:
            return func, False, ""
        new_func = (new_func[:insert_after+1] + '\n' +
                    '\n'.join(new_decls) +
                    new_func[insert_after+1:])
        return new_func, True, ""

# ─── 13. DEPENDENCE_BREAK ────────────────────────────────────────────────────

class DependenceBreak(Transform):
    name = "DEPENDENCE_BREAK"

    def apply(self, code: str) -> tuple:
        """Remove dead first-write in double-assignment patterns (WAW hazard).
        Pattern: s.x = expr1;  // immediately overwritten
                 s.x = expr2;  // second write uses *_old, not s.x
        """
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, n_removed = self._remove_dead_writes(func)
            new_funcs.append(new_func)
            if n_removed > 0:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No dead write-after-write patterns found",
                "N/A — no redundant assignments detected")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No dead WAW patterns found"

        hdr = header_comment(self.name,
            "Removed dead first-writes in write-after-write sequences",
            "Eliminates redundant assignments; HLS no longer needs to schedule dead logic")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Removed dead write-after-write assignments"

    def _remove_dead_writes(self, func: str) -> tuple:
        """Find consecutive assignments to the same lvalue where the first is dead."""
        lines = func.split('\n')
        # Build assignment map: lvalue → list of line indices
        assign_lval = {}
        for i, ln in enumerate(lines):
            m = re.match(r'\s*([\w.]+)\s*=\s*([^=][^;]*);', ln)
            if m and not ln.strip().startswith('//'):
                lv = m.group(1)
                assign_lval.setdefault(lv, []).append(i)

        # Find lvalues assigned more than once
        dead_lines = set()
        for lv, idxs in assign_lval.items():
            if len(idxs) < 2:
                continue
            for first, second in zip(idxs, idxs[1:]):
                # Check if second assignment's RHS does NOT use the lvalue
                rhs_m = re.match(r'\s*[\w.]+\s*=\s*(.+);', lines[second])
                if rhs_m:
                    rhs = rhs_m.group(1)
                    lv_simple = lv.split('.')[-1]
                    # If RHS references the _old version or doesn't use lv at all,
                    # the first write is dead
                    uses_lv = lv in rhs
                    uses_old = (lv_simple + '_old') in rhs or (lv + '_old') in rhs
                    if not uses_lv or uses_old:
                        dead_lines.add(first)

        if not dead_lines:
            return func, 0

        # Replace dead lines with a no-op instead of deleting, so that
        # if-bodies are never left empty (which would cause HLS syntax errors).
        new_lines = list(lines)
        for i in dead_lines:
            indent = re.match(r'^(\s*)', lines[i]).group(1)
            new_lines[i] = f"{indent}(void)0; /* DEPENDENCE_BREAK: dead write removed */"

        return '\n'.join(new_lines), len(dead_lines)

# ─── 14. PREDICATE_TO_DATAFLOW ───────────────────────────────────────────────

class PredicateToDataflow(Transform):
    name = "PREDICATE_TO_DATAFLOW"

    def apply(self, code: str) -> tuple:
        """Convert nested if(rst)/if(enable) pattern to per-variable ternary chains.

        Input:
          if(rst) { s.x = 0; s.y = 0; }
          else if(enable) { s.x = new_x; s.y = new_y; }

        Output:
          s.x = rst ? 0 : (enable ? new_x : x_old);
          s.y = rst ? 0 : (enable ? new_y : y_old);
        """
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._convert_func(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No rst/enable predicated pattern found for dataflow conversion",
                "N/A — no applicable predicated assignment structure")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No predicated patterns to convert"

        hdr = header_comment(self.name,
            "Converted predicated if(rst)/if(enable) to per-variable ternary dataflow",
            "Exposes explicit mux structure to HLS; reduces control depth to 2 levels")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Converted predicated control to ternary dataflow"

    def _convert_func(self, func: str) -> tuple:
        # Get old-var mapping
        old_map = get_old_var_map(func)

        # Match: if(RST_VAR) { assignments... } \n else \n if(ENA_VAR) { assignments... }
        # Flexible: handles 0, 1, or 2 levels
        pat = re.compile(
            r'([ \t]*)if\s*\((\w+)\)\s*\n\s*\{([^}]+)\}\s*\n\s*\n?\s*else\s*\n'
            r'\s*if\s*\((\w+)\)\s*\n\s*\{([^}]+)\}',
            re.DOTALL)

        m = pat.search(func)
        if not m:
            # Try simpler: if(RST) { ... } else { ... }
            return self._convert_simple(func, old_map)

        indent   = m.group(1)
        rst_var  = m.group(2)
        rst_body = m.group(3)
        ena_var  = m.group(4)
        ena_body = m.group(5)

        # Parse assignments in each branch: lvalue → expr
        rst_assigns = self._parse_assigns(rst_body)
        ena_assigns = self._parse_assigns(ena_body)

        all_lvalues = list(dict.fromkeys(
            list(rst_assigns.keys()) + list(ena_assigns.keys())))

        if not all_lvalues:
            return func, False

        lines = []
        for lv in all_lvalues:
            rst_val = rst_assigns.get(lv, None)
            ena_val = ena_assigns.get(lv, None)
            # Find old var
            field = lv.split('.')[-1] if '.' in lv else lv
            old_var = old_map.get(field, field + "_old")

            if rst_val is not None and ena_val is not None:
                expr = (f"({rst_var}) ? ({rst_val}) : "
                        f"(({ena_var}) ? ({ena_val}) : {old_var})")
            elif rst_val is not None:
                expr = (f"({rst_var}) ? ({rst_val}) : {old_var}")
            else:
                expr = (f"({ena_var}) ? ({ena_val}) : {old_var}")

            lines.append(f"{indent}{lv} = {expr};")

        replacement = '\n'.join(lines)
        new_func = func[:m.start()] + replacement + func[m.end():]
        return new_func, True

    def _convert_simple(self, func: str, old_map: dict) -> tuple:
        """Handle simple if(C) { assigns } else { assigns }."""
        pat = re.compile(
            r'([ \t]*)if\s*\((\w+)\)\s*\n\s*\{([^}]+)\}\s*\n\s*\n?\s*else\s*\n\s*\{([^}]+)\}',
            re.DOTALL)
        m = pat.search(func)
        if not m:
            return func, False

        indent = m.group(1)
        cond   = m.group(2)
        t_body = m.group(3)
        f_body = m.group(4)

        t_assigns = self._parse_assigns(t_body)
        f_assigns = self._parse_assigns(f_body)

        all_lvalues = list(dict.fromkeys(
            list(t_assigns.keys()) + list(f_assigns.keys())))
        if not all_lvalues:
            return func, False

        lines = []
        for lv in all_lvalues:
            tv = t_assigns.get(lv)
            fv = f_assigns.get(lv)
            field = lv.split('.')[-1] if '.' in lv else lv
            old_var = old_map.get(field, field + "_old")
            if tv is not None and fv is not None:
                lines.append(f"{indent}{lv} = ({cond}) ? ({tv}) : ({fv});")
            elif tv is not None:
                lines.append(f"{indent}{lv} = ({cond}) ? ({tv}) : {old_var};")
            else:
                lines.append(f"{indent}{lv} = ({cond}) ? {old_var} : ({fv});")

        replacement = '\n'.join(lines)
        new_func = func[:m.start()] + replacement + func[m.end():]
        return new_func, True

    def _parse_assigns(self, body: str) -> dict:
        """Extract {lvalue: rhs_expr} from block body."""
        assigns = {}
        for m in re.finditer(r'(\w[\w. >*\[\]]*)\s*=\s*([^;]+);', body):
            lv  = m.group(1).strip()
            rhs = m.group(2).strip()
            assigns[lv] = rhs
        return assigns

# ─── 15. LOOP_PIPELINING ──────────────────────────────────────────────────────

class LoopPipelining(Transform):
    name = "LOOP_PIPELINING"

    def apply(self, code: str) -> tuple:
        """Add #pragma HLS PIPELINE II=1 annotations inside for/while loops."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._annotate_loops(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No for/while loops found for pipeline annotation",
                "N/A — transformation not applicable")
            return preamble + "\n" + hdr + '\n'.join(body_funcs + [main_block]), False, \
                   "No loops found for pipelining"

        hdr = header_comment(self.name,
            "Added /* #pragma HLS PIPELINE II=1 */ annotations to loops",
            "Hints HLS to pipeline loop iterations with initiation interval of 1")
        return preamble + "\n" + hdr + '\n'.join(new_funcs + [main_block]), True, \
               "Added pipeline annotations to loops"

    def _annotate_loops(self, func: str) -> tuple:
        changed = False
        lines = func.split('\n')
        new_lines = []
        pragma = "/* #pragma HLS PIPELINE II=1 */"
        i = 0
        while i < len(lines):
            ln = lines[i]
            # Match for(...) or while(...) followed by {
            if re.match(r'^\s*(for|while)\s*\(', ln.strip()):
                new_lines.append(ln)
                # Look for the opening brace
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    new_lines.append(lines[j])
                    j += 1
                if j < len(lines) and '{' in lines[j]:
                    new_lines.append(lines[j])
                    # Determine indent inside the brace
                    brace_indent = re.match(r'^(\s*)', lines[j]).group(1)
                    inner_indent = brace_indent + "  "
                    new_lines.append(f"{inner_indent}{pragma}")
                    changed = True
                    i = j + 1
                    continue
                elif '{' in ln:
                    # Brace on same line as for/while
                    loop_indent = re.match(r'^(\s*)', ln).group(1)
                    inner_indent = loop_indent + "  "
                    new_lines.append(f"{inner_indent}{pragma}")
                    changed = True
                    i += 1
                    continue
                else:
                    i = j
                    continue
            else:
                new_lines.append(ln)
            i += 1
        return '\n'.join(new_lines), changed

# ─── 16. PARTIAL_UNROLL ──────────────────────────────────────────────────────

class PartialUnroll(Transform):
    name = "PARTIAL_UNROLL"

    def apply(self, code: str) -> tuple:
        """Unroll for-loops by factor 2: duplicate body with index adjustments, halve iteration count."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._unroll_loops(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No for-loops found suitable for partial unrolling",
                "N/A — transformation not applicable")
            return preamble + "\n" + hdr + '\n'.join(body_funcs + [main_block]), False, \
                   "No for-loops found for unrolling"

        hdr = header_comment(self.name,
            "Partially unrolled for-loops by factor 2",
            "Reduces loop overhead and exposes instruction-level parallelism")
        return preamble + "\n" + hdr + '\n'.join(new_funcs + [main_block]), True, \
               "Partially unrolled for-loops by factor 2"

    def _unroll_loops(self, func: str) -> tuple:
        """Find for(var = START; (int)var < END; var = (int)var + 1) and unroll by 2."""
        changed = False
        # Pattern: for(VAR = START; (int)VAR < END; VAR = (int)VAR + 1)\n  BODY;
        pat = re.compile(
            r'([ \t]*)(for\s*\(\s*(\w+)\s*=\s*(\d+)\s*(?:&\s*\d+\s*)?;\s*'
            r'\(int\)(\w+)\s*<\s*(\d+)\s*;\s*'
            r'(\w+)\s*=\s*\(int\)\7\s*\+\s*1\s*\))\s*\n'
            r'([ \t]+)(.+;)',
            re.MULTILINE)

        def replacer(m):
            nonlocal changed
            indent = m.group(1)
            var = m.group(3)
            start = int(m.group(4))
            loop_var2 = m.group(5)
            end = int(m.group(6))
            body_indent = m.group(8)
            body_stmt = m.group(9)

            if var != loop_var2:
                return m.group(0)

            # Halve iteration count (round down to even)
            new_end = start + ((end - start) // 2) * 2
            half_end = start + (end - start) // 2

            # Build unrolled body: original + copy with var+1
            body_copy = body_stmt
            # Replace loop variable references with var+1 in the copy
            body_copy = re.sub(r'\b' + re.escape(var) + r'\b', f'({var} + 1)', body_copy)

            # New loop iterates by 2
            new_loop = (
                f"{indent}for({var} = {start}; (int){var} < {new_end}; {var} = (int){var} + 2)\n"
                f"{indent}{{\n"
                f"{body_indent}{body_stmt}\n"
                f"{body_indent}{body_copy}\n"
                f"{indent}}}"
            )

            # Handle remainder if odd count
            remainder = end - start
            if remainder % 2 != 0:
                new_loop += (
                    f"\n{indent}/* remainder iteration */\n"
                    f"{indent}{var} = {new_end};\n"
                    f"{body_indent}{body_stmt}"
                )

            changed = True
            return new_loop

        new_func = pat.sub(replacer, func)
        return new_func, changed

# ─── 17. PIPELINE_STAGE_INSERT ────────────────────────────────────────────────

class PipelineStageInsert(Transform):
    name = "PIPELINE_STAGE_INSERT"

    def apply(self, code: str) -> tuple:
        """Insert pipeline stage boundaries for long chains of dependent assignments."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._insert_stages(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No long dependent assignment chains (4+) found",
                "N/A — transformation not applicable")
            return preamble + "\n" + hdr + '\n'.join(body_funcs + [main_block]), False, \
                   "No long dependent chains for pipeline staging"

        hdr = header_comment(self.name,
            "Inserted _pipe_N intermediate variables at every 3rd stage",
            "Suggests pipeline register boundaries to HLS for better timing")
        return preamble + "\n" + hdr + '\n'.join(new_funcs + [main_block]), True, \
               "Inserted pipeline stage boundaries in dependent chains"

    def _insert_stages(self, func: str) -> tuple:
        lines = func.split('\n')
        # Find sequences of state struct assignments where one feeds the next
        # Pattern: s.field1 = expr1;  then s.field2 = ... field1 ...;
        assign_lines = []
        for i, ln in enumerate(lines):
            m = re.match(r'^\s+(\w+\.\w+)\s*=\s*(.+?)\s*;', ln)
            if m:
                assign_lines.append((i, m.group(1), m.group(2)))

        if len(assign_lines) < 4:
            return func, False

        # Build dependency chains
        chains = []
        current_chain = [assign_lines[0]]
        for j in range(1, len(assign_lines)):
            prev_lv = current_chain[-1][1]
            cur_rhs = assign_lines[j][2]
            prev_field = prev_lv.split('.')[-1] if '.' in prev_lv else prev_lv
            if prev_field in cur_rhs or prev_lv in cur_rhs:
                current_chain.append(assign_lines[j])
            else:
                if len(current_chain) >= 4:
                    chains.append(current_chain)
                current_chain = [assign_lines[j]]
        if len(current_chain) >= 4:
            chains.append(current_chain)

        if not chains:
            return func, False

        # Insert pipe variables at every 3rd stage in each chain
        pipe_idx = 0
        insertions = {}  # line_idx -> (pipe_var_decl, new_rhs_replacement)
        replacements = {}  # line_idx -> new_line

        for chain in chains:
            for k, (line_idx, lv, rhs) in enumerate(chain):
                if k > 0 and k % 3 == 0:
                    # Insert a pipeline stage: capture previous result in _pipe_N
                    prev_idx, prev_lv, prev_rhs = chain[k - 1]
                    pipe_var = f"_pipe_{pipe_idx}"
                    pipe_idx += 1
                    indent = re.match(r'^(\s*)', lines[line_idx]).group(1)
                    prev_field = prev_lv.split('.')[-1] if '.' in prev_lv else prev_lv

                    # Add pipe var assignment before current line
                    pipe_decl = f"{indent}unsigned int {pipe_var} = {prev_lv}; /* PIPELINE_STAGE */"
                    insertions[line_idx] = pipe_decl

                    # Replace reference to prev_lv in current line's RHS
                    # Use word-boundary regex to avoid partial replacements
                    # (e.g. replacing 'count' inside 'count_old' → '_pipe_0_old')
                    new_rhs = re.sub(r'\b' + re.escape(prev_lv) + r'\b', pipe_var, rhs)
                    new_rhs = re.sub(r'\b' + re.escape(prev_field) + r'\b', pipe_var, new_rhs)
                    replacements[line_idx] = f"{indent}{lv} = {new_rhs};"

        if not insertions:
            return func, False

        new_lines = []
        for i, ln in enumerate(lines):
            if i in insertions:
                new_lines.append(insertions[i])
            if i in replacements:
                new_lines.append(replacements[i])
            else:
                new_lines.append(ln)

        return '\n'.join(new_lines), True

# ─── 18. MUX_TREE_BALANCE ────────────────────────────────────────────────────

class MuxTreeBalance(Transform):
    name = "MUX_TREE_BALANCE"

    def apply(self, code: str) -> tuple:
        """Restructure deeply nested if-else-if chains testing DIFFERENT variables
        into balanced binary selection trees."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._balance_mux(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No if-else chains testing different variables found (4+ branches)",
                "N/A — transformation not applicable")
            return preamble + "\n" + hdr + '\n'.join(body_funcs + [main_block]), False, \
                   "No multi-variable if-else chains to balance"

        hdr = header_comment(self.name,
            "Balanced if-else-if chain (different conditions) into binary selection tree",
            "Reduces mux depth from O(N) to O(log N) for heterogeneous conditions")
        return preamble + "\n" + hdr + '\n'.join(new_funcs + [main_block]), True, \
               "Balanced heterogeneous if-else chain into binary tree"

    def _balance_mux(self, func: str) -> tuple:
        lines = func.split('\n')
        # Collect cascaded if/else-if chains testing different variables
        # Pattern: if(condA) { body } else if(condB) { body } else if(condC) ...
        # where condA, condB, condC test different variables

        # Find if-else-if chains
        chain_pat = re.compile(
            r'if\s*\(\s*(.+?)\s*\)\s*$')
        eq_pat = re.compile(
            r'^\s*(?:\(\s*(?:unsigned\s+int\s*)\)\s*)?(\w+)\s*==\s*(\d+)\s*$')

        i = 0
        chains = []
        while i < len(lines):
            ln = lines[i].rstrip()
            # Start of a chain: plain 'if' (not 'else if')
            stripped = ln.strip()
            if stripped.startswith('if(') or stripped.startswith('if ('):
                m = chain_pat.match(stripped)
                if m:
                    cond = m.group(1)
                    eq_m = eq_pat.match(cond)
                    chain_start = i
                    conditions = [(cond, i)]
                    tested_vars = set()
                    if eq_m:
                        tested_vars.add(eq_m.group(1))
                    same_var = True

                    # Scan forward for else-if continuations
                    j = i + 1
                    # Skip body lines until we find else
                    depth = 0
                    while j < len(lines):
                        for ch in lines[j]:
                            if ch == '{': depth += 1
                            elif ch == '}': depth -= 1
                        if depth <= 0:
                            # Check if next non-blank is 'else'
                            k = j + 1
                            while k < len(lines) and not lines[k].strip():
                                k += 1
                            if k < len(lines):
                                else_ln = lines[k].strip()
                                if else_ln == 'else':
                                    # Next should be if(...)
                                    k2 = k + 1
                                    while k2 < len(lines) and not lines[k2].strip():
                                        k2 += 1
                                    if k2 < len(lines):
                                        em = chain_pat.match(lines[k2].strip())
                                        if em:
                                            econd = em.group(1)
                                            eq_m2 = eq_pat.match(econd)
                                            if eq_m2:
                                                tested_vars.add(eq_m2.group(1))
                                            conditions.append((econd, k2))
                                            j = k2 + 1
                                            depth = 0
                                            continue
                            break
                        j += 1

                    # Check: 4+ branches, testing different variables (not all same)
                    if len(conditions) >= 4 and len(tested_vars) != 1:
                        chains.append((chain_start, conditions, lines))
            i += 1

        if not chains:
            return func, False

        # For the first qualifying chain, restructure
        chain_start, conditions, _ = chains[0]

        # Extract condition-body pairs from the original text
        func_text = func
        pairs = []
        for cond, line_idx in conditions:
            # Find the body (the assignment statement) after this if
            body_start = line_idx + 1
            while body_start < len(lines) and not lines[body_start].strip():
                body_start += 1
            if body_start < len(lines):
                body_ln = lines[body_start].strip()
                if body_ln == '{':
                    # Find matching close
                    body_start += 1
                    body_lines_inner = []
                    bd = 1
                    while body_start < len(lines) and bd > 0:
                        for ch in lines[body_start]:
                            if ch == '{': bd += 1
                            elif ch == '}': bd -= 1
                        if bd > 0:
                            body_lines_inner.append(lines[body_start].strip())
                        body_start += 1
                    body_text = ' '.join(body_lines_inner)
                else:
                    body_text = body_ln
                pairs.append((cond, body_text))

        if len(pairs) < 4:
            return func, False

        # Check that all bodies assign to the same lvalue
        lvalues = set()
        bodies = []
        for cond, body in pairs:
            m = re.match(r'(\w[\w. >*\[\]]*)\s*=\s*(.+?)\s*;', body)
            if m:
                lvalues.add(m.group(1))
                bodies.append((cond, m.group(1), m.group(2)))

        if len(lvalues) != 1 or len(bodies) < 4:
            return func, False

        target_lv = list(lvalues)[0]
        indent = re.match(r'^(\s*)', lines[chain_start]).group(1)

        # Build balanced ternary tree
        # First, compute all conditions into named variables
        cond_lines = []
        cond_vars = []
        for idx_c, (cond, _, _) in enumerate(bodies):
            cvar = f"_mux_cond_{idx_c}"
            cond_vars.append(cvar)
            cond_lines.append(f"{indent}_Bool {cvar} = {cond};")

        # Build balanced ternary: recursively pair up
        def build_balanced_ternary(items, start_idx=0):
            if len(items) == 1:
                return items[0][2]  # just the rhs value
            if len(items) == 2:
                return f"({cond_vars[start_idx]} ? ({items[0][2]}) : ({items[1][2]}))"
            mid = len(items) // 2
            left = build_balanced_ternary(items[:mid], start_idx)
            right = build_balanced_ternary(items[mid:], start_idx + mid)
            # Use first condition of left half vs right half
            left_conds = " || ".join(cond_vars[start_idx:start_idx + mid])
            return f"(({left_conds}) ? ({left}) : ({right}))"

        ternary_expr = build_balanced_ternary(bodies)

        # Build replacement block
        result_lines = cond_lines
        result_lines.append(f"{indent}{target_lv} = {ternary_expr};")

        # Find the end of the original chain in the source
        # We need to find from chain_start to the end of the last else-if body
        last_cond_line = conditions[-1][1]
        chain_end = last_cond_line + 1
        depth = 0
        while chain_end < len(lines):
            for ch in lines[chain_end]:
                if ch == '{': depth += 1
                elif ch == '}': depth -= 1
            if depth <= 0:
                chain_end += 1
                break
            chain_end += 1

        new_lines = lines[:chain_start] + result_lines + lines[chain_end:]
        return '\n'.join(new_lines), True

# ─── 19. CARRY_SAVE_REWRITE ──────────────────────────────────────────────────

class CarrySaveRewrite(Transform):
    name = "CARRY_SAVE_REWRITE"

    def apply(self, code: str) -> tuple:
        """Find expressions with 3+ addends and rewrite as named partial sums."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._rewrite_sums(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No 3+ addend expressions found for carry-save rewrite",
                "N/A — transformation not applicable")
            return preamble + "\n" + hdr + '\n'.join(body_funcs + [main_block]), False, \
                   "No multi-addend expressions for carry-save rewrite"

        hdr = header_comment(self.name,
            "Rewrote 3+ addend sums as named partial sums",
            "Reduces adder chain depth; partial sums can be computed in parallel")
        return preamble + "\n" + hdr + '\n'.join(new_funcs + [main_block]), True, \
               "Rewrote multi-addend sums as carry-save partial sums"

    def _rewrite_sums(self, func: str) -> tuple:
        lines = func.split('\n')
        new_lines = []
        changed = False
        ps_idx = 0

        for ln in lines:
            m = re.match(r'^(\s*)(\w[\w. >*\[\]]+)\s*=\s*(.+?)\s*;(\s*(?://.*)?)\s*$', ln)
            if m:
                indent, lvalue, rhs, comment = m.groups()
                # Count top-level + operators
                top_plus = []
                depth = 0
                for idx, ch in enumerate(rhs):
                    if ch in '([': depth += 1
                    elif ch in ')]': depth -= 1
                    elif ch == '+' and depth == 0:
                        top_plus.append(idx)
                if len(top_plus) >= 2:  # 3+ addends
                    splits = [-1] + top_plus + [len(rhs)]
                    operands = [rhs[splits[i]+1:splits[i+1]].strip()
                                for i in range(len(splits)-1)]
                    # Build partial sum pairs
                    ps_lines = []
                    ps_vars = []
                    i = 0
                    while i < len(operands) - 1:
                        ps_var = f"_ps{ps_idx}"
                        ps_idx += 1
                        ps_lines.append(
                            f"{indent}unsigned int {ps_var} = {operands[i]} + {operands[i+1]};")
                        ps_vars.append(ps_var)
                        i += 2
                    if i < len(operands):
                        ps_vars.append(operands[i])

                    # Combine partial sums
                    while len(ps_vars) > 1:
                        new_ps_vars = []
                        j = 0
                        while j < len(ps_vars) - 1:
                            ps_var = f"_ps{ps_idx}"
                            ps_idx += 1
                            ps_lines.append(
                                f"{indent}unsigned int {ps_var} = {ps_vars[j]} + {ps_vars[j+1]};")
                            new_ps_vars.append(ps_var)
                            j += 2
                        if j < len(ps_vars):
                            new_ps_vars.append(ps_vars[j])
                        ps_vars = new_ps_vars

                    for pl in ps_lines:
                        new_lines.append(pl)
                    new_lines.append(f"{indent}{lvalue} = {ps_vars[0]};{comment}")
                    changed = True
                    continue
            new_lines.append(ln)

        return '\n'.join(new_lines), changed

# ─── 20. GUARD_RELAXATION ────────────────────────────────────────────────────

class GuardRelaxation(Transform):
    name = "GUARD_RELAXATION"

    def apply(self, code: str) -> tuple:
        """Hoist computation out of single-condition guards on state writes."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._relax_guards(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No single-condition guarded state writes found",
                "N/A — transformation not applicable")
            return preamble + "\n" + hdr + '\n'.join(body_funcs + [main_block]), False, \
                   "No guarded state writes to relax"

        hdr = header_comment(self.name,
            "Hoisted computation out of enable guards",
            "Computation happens unconditionally, reducing critical path through enable")
        return preamble + "\n" + hdr + '\n'.join(new_funcs + [main_block]), True, \
               "Hoisted computation out of single-condition guards"

    def _relax_guards(self, func: str) -> tuple:
        """Find if(enable) { s.field = expr; } and hoist the computation."""
        changed = False
        guard_idx = 0

        # Pattern: if(COND) {\n  s.field = EXPR;\n }
        # But NOT if(rst) or if-else patterns
        pat = re.compile(
            r'([ \t]*)if\s*\((\w+)\)\s*\n'
            r'\s*\{\s*\n'
            r'(\s*(\w+\.\w+)\s*=\s*(.+?)\s*;\s*\n)'
            r'\s*\}',
            re.MULTILINE)

        def replacer(m):
            nonlocal changed, guard_idx
            indent = m.group(1)
            cond = m.group(2)
            lvalue = m.group(4)
            expr = m.group(5).strip()

            # Skip rst patterns
            if 'rst' in cond.lower() or 'reset' in cond.lower():
                return m.group(0)

            # Skip if expression is trivial (just a number or single variable)
            if re.match(r'^[\w.]+$', expr) or re.match(r'^\d+$', expr):
                return m.group(0)

            guard_var = f"_guard_val_{guard_idx}"
            guard_idx += 1
            changed = True

            return (f"{indent}unsigned int {guard_var} = {expr};\n"
                    f"{indent}if({cond})\n"
                    f"{indent}{{\n"
                    f"{indent}  {lvalue} = {guard_var};\n"
                    f"{indent}}}")

        new_func = pat.sub(replacer, func)
        return new_func, changed

# ─── 21. COPY_PROPAGATION ────────────────────────────────────────────────────

class CopyPropagation(Transform):
    name = "COPY_PROPAGATION"

    def apply(self, code: str) -> tuple:
        """Propagate simple copy assignments, particularly field_old = s.field patterns."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._propagate_copies(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No safe copy propagation opportunities found",
                "N/A — transformation not applicable")
            return preamble + "\n" + hdr + '\n'.join(body_funcs + [main_block]), False, \
                   "No copy propagation opportunities"

        hdr = header_comment(self.name,
            "Propagated copy assignments inline and removed dead copies",
            "Reduces register usage and shortens dependency chains")
        return preamble + "\n" + hdr + '\n'.join(new_funcs + [main_block]), True, \
               "Propagated copy variables inline"

    def _propagate_copies(self, func: str) -> tuple:
        """Find X_old = s.X; patterns and replace uses of X_old with s.X where safe."""
        lines = func.split('\n')
        # Collect copy assignments: var_old = struct.field;
        copies = {}  # old_var -> source_expr
        copy_lines = set()  # line indices of copy statements

        for i, ln in enumerate(lines):
            m = re.match(r'^\s+(\w+_old)\s*=\s*(\w+\.\w+)\s*;', ln)
            if m:
                old_var = m.group(1)
                source = m.group(2)
                copies[old_var] = (source, i)
                copy_lines.add(i)

        if not copies:
            return func, False

        # For each copy, check if the source (s.field) is modified between
        # the copy line and any use of old_var. If not, we can propagate.
        changed = False
        propagatable = {}
        for old_var, (source, copy_line) in copies.items():
            # Find all lines that use old_var (after the copy)
            uses = []
            for j in range(copy_line + 1, len(lines)):
                if re.search(r'\b' + re.escape(old_var) + r'\b', lines[j]):
                    uses.append(j)

            if not uses:
                continue

            # Check if source is written between copy_line and last use
            source_written = False
            field = source.split('.')[-1] if '.' in source else source
            for j in range(copy_line + 1, max(uses) + 1):
                # Check if source is on the LHS of an assignment
                if re.match(r'^\s+' + re.escape(source) + r'\s*=', lines[j]):
                    source_written = True
                    break

            if not source_written:
                propagatable[old_var] = source

        if not propagatable:
            return func, False

        # Apply propagation
        new_lines = []
        removed_copies = set()
        for old_var, source in propagatable.items():
            removed_copies.add(copies[old_var][1])

        for i, ln in enumerate(lines):
            if i in removed_copies:
                # Remove the copy line
                changed = True
                continue
            new_ln = ln
            for old_var, source in propagatable.items():
                new_ln = re.sub(r'\b' + re.escape(old_var) + r'\b', source, new_ln)
                if new_ln != ln:
                    changed = True
            new_lines.append(new_ln)

        if not changed:
            return func, False

        return '\n'.join(new_lines), True

# ─── 22. ALGEBRAIC_SIMPLIFY ──────────────────────────────────────────────────

class AlgebraicSimplify(Transform):
    name = "ALGEBRAIC_SIMPLIFY"

    def apply(self, code: str) -> tuple:
        """Apply algebraic identities to simplify expressions."""
        new_code = code

        # x + 0 -> x,  0 + x -> x
        new_code = re.sub(r'\b(\w[\w.>\[\] ]*)\s*\+\s*0\b', r'\1', new_code)
        new_code = re.sub(r'\b0\s*\+\s*(\w[\w.>\[\] ]*)\b', r'\1', new_code)

        # x - 0 -> x
        new_code = re.sub(r'\b(\w[\w.>\[\] ]*)\s*-\s*0\b', r'\1', new_code)

        # x * 1 -> x,  1 * x -> x
        new_code = re.sub(r'\b(\w[\w.>\[\] ]*)\s*\*\s*1\b', r'\1', new_code)
        new_code = re.sub(r'\b1\s*\*\s*(\w[\w.>\[\] ]*)\b', r'\1', new_code)

        # x * 0 -> 0,  0 * x -> 0
        new_code = re.sub(r'\b\w[\w.>\[\] ]*\s*\*\s*0\b', '0', new_code)
        new_code = re.sub(r'\b0\s*\*\s*\w[\w.>\[\] ]*\b', '0', new_code)

        # x & 0xFFFFFFFF -> x (32-bit all-ones for unsigned int)
        new_code = re.sub(r'\b(\w[\w.>\[\] ]*)\s*&\s*0xFFFFFFFF\b', r'\1', new_code)
        new_code = re.sub(r'\b0xFFFFFFFF\s*&\s*(\w[\w.>\[\] ]*)\b', r'\1', new_code)
        # Also decimal form
        new_code = re.sub(r'\b(\w[\w.>\[\] ]*)\s*&\s*4294967295\b', r'\1', new_code)
        new_code = re.sub(r'\b4294967295\s*&\s*(\w[\w.>\[\] ]*)\b', r'\1', new_code)

        # x | 0 -> x,  0 | x -> x
        new_code = re.sub(r'\b(\w[\w.>\[\] ]*)\s*\|\s*0\b', r'\1', new_code)
        new_code = re.sub(r'\b0\s*\|\s*(\w[\w.>\[\] ]*)\b', r'\1', new_code)

        # x ^ 0 -> x,  0 ^ x -> x
        new_code = re.sub(r'\b(\w[\w.>\[\] ]*)\s*\^\s*0\b', r'\1', new_code)
        new_code = re.sub(r'\b0\s*\^\s*(\w[\w.>\[\] ]*)\b', r'\1', new_code)

        # x << 0 -> x
        new_code = re.sub(r'\b(\w[\w.>\[\] ]*)\s*<<\s*0\b', r'\1', new_code)

        # x >> 0 -> x
        new_code = re.sub(r'\b(\w[\w.>\[\] ]*)\s*>>\s*0\b', r'\1', new_code)

        # !!x -> x (double negation)
        new_code = re.sub(r'!!(\w[\w.>\[\] ]*)', r'\1', new_code)

        # ((x)) -> (x) — remove one layer of redundant parens
        new_code = re.sub(r'\(\(([^()]+)\)\)', r'(\1)', new_code)

        changed = new_code != code
        if not changed:
            hdr = header_comment(self.name,
                "No algebraic simplification opportunities found",
                "N/A — transformation not applicable")
            return hdr + code, False, "No algebraic simplification opportunities"

        hdr = header_comment(self.name,
            "Applied algebraic identities to simplify expressions",
            "Reduces logic and area; eliminates no-op operations from synthesis")
        return hdr + new_code, True, "Applied algebraic simplifications"

# ─── 23. BOOLEAN_TO_ARITHMETIC ────────────────────────────────────────────────

class BooleanToArithmetic(Transform):
    name = "BOOLEAN_TO_ARITHMETIC"

    def apply(self, code: str) -> tuple:
        """Convert short-circuit && to bitwise & and || to | in conditions
        when operands are boolean (comparisons, _Bool, single-bit)."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._convert_booleans(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No boolean short-circuit chains found with known-boolean operands",
                "N/A — transformation not applicable")
            return preamble + "\n" + hdr + '\n'.join(body_funcs + [main_block]), False, \
                   "No boolean-to-arithmetic conversion opportunities"

        hdr = header_comment(self.name,
            "Converted && to & and || to | for known-boolean operands",
            "Removes short-circuit branches; enables parallel evaluation of all conditions")
        return preamble + "\n" + hdr + '\n'.join(new_funcs + [main_block]), True, \
               "Converted boolean short-circuit to bitwise operations"

    def _convert_booleans(self, func: str) -> tuple:
        changed = False
        lines = func.split('\n')
        new_lines = []

        # Boolean operand: comparison (x == y, x != y, x < y, etc.), !x, or _Bool var
        bool_operand = r'(?:!?\w[\w. >*\[\]]*\s*(?:[<>=!]=?)\s*\w[\w. >*\[\]]*|!\w[\w. >*\[\]]*|\w[\w. >*\[\]]*)'

        for ln in lines:
            # Find if(...) conditions with && or ||
            m = re.match(r'^(\s*if\s*\()(.+)(\)\s*(?:\{)?\s*)$', ln)
            if m:
                prefix, cond, suffix = m.group(1), m.group(2), m.group(3)

                # Check if condition has && with comparison operands on both sides
                # Pattern: comp1 && comp2 && comp3
                # A comparison operand contains a comparison operator
                comp_pat = re.compile(
                    r'(\w[\w. >*\[\]]*\s*(?:==|!=|<=|>=|<|>)\s*\w[\w. >*\[\]]*)')

                if '&&' in cond:
                    parts = re.split(r'\s*&&\s*', cond)
                    if len(parts) >= 2:
                        all_bool = all(
                            comp_pat.match(p.strip()) or
                            re.match(r'^!\w', p.strip()) or
                            re.match(r'^\w+$', p.strip())
                            for p in parts
                        )
                        if all_bool:
                            new_cond = ' & '.join(
                                f'({p.strip()})' if comp_pat.match(p.strip()) else p.strip()
                                for p in parts
                            )
                            new_lines.append(f"{prefix}{new_cond}{suffix}")
                            changed = True
                            continue

                if '||' in cond and '&&' not in cond:
                    parts = re.split(r'\s*\|\|\s*', cond)
                    if len(parts) >= 2:
                        all_bool = all(
                            comp_pat.match(p.strip()) or
                            re.match(r'^!\w', p.strip()) or
                            re.match(r'^\w+$', p.strip())
                            for p in parts
                        )
                        if all_bool:
                            new_cond = ' | '.join(
                                f'({p.strip()})' if comp_pat.match(p.strip()) else p.strip()
                                for p in parts
                            )
                            new_lines.append(f"{prefix}{new_cond}{suffix}")
                            changed = True
                            continue

            new_lines.append(ln)

        return '\n'.join(new_lines), changed

# ─── 24. ENCODE_ONEHOT_TO_BINARY ─────────────────────────────────────────────

class EncodeOnehotToBinary(Transform):
    name = "ENCODE_ONEHOT_TO_BINARY"

    def apply(self, code: str) -> tuple:
        """Detect one-hot bit testing patterns and convert to binary-encoded lookup."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, changed = self._convert_onehot(func)
            new_funcs.append(new_func)
            if changed:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No one-hot bit-testing if-else patterns found",
                "N/A — transformation not applicable")
            return preamble + "\n" + hdr + '\n'.join(body_funcs + [main_block]), False, \
                   "No one-hot patterns found for binary encoding"

        hdr = header_comment(self.name,
            "Converted one-hot bit-testing if-else chain to binary index + array lookup",
            "Reduces mux depth; array lookup is O(1) vs O(N) for if-else chain")
        return preamble + "\n" + hdr + '\n'.join(new_funcs + [main_block]), True, \
               "Converted one-hot selection to binary-encoded array lookup"

    def _convert_onehot(self, func: str) -> tuple:
        lines = func.split('\n')

        # Find patterns: if(x & 1) out = a; else if(x & 2) out = b; else if(x & 4) out = c;
        # The bit masks should be powers of 2
        onehot_pat = re.compile(
            r'if\s*\(\s*(\w+)\s*&\s*(\d+)\s*\)')

        i = 0
        chains = []
        while i < len(lines):
            ln = lines[i].strip()
            m = onehot_pat.match(ln)
            if m:
                var = m.group(1)
                val = int(m.group(2))
                # Check if power of 2
                if val > 0 and (val & (val - 1)) == 0:
                    chain_start = i
                    entries = []
                    # Collect this entry
                    j = i
                    while j < len(lines):
                        ln_j = lines[j].strip()
                        # Match: [else] if(var & POWER_OF_2)
                        mj = re.match(
                            r'(?:else\s+)?if\s*\(\s*' + re.escape(var) + r'\s*&\s*(\d+)\s*\)',
                            ln_j)
                        if mj:
                            bit_val = int(mj.group(1))
                            if bit_val > 0 and (bit_val & (bit_val - 1)) == 0:
                                # Get the body (next non-empty line)
                                k = j + 1
                                while k < len(lines) and not lines[k].strip():
                                    k += 1
                                if k < len(lines):
                                    body = lines[k].strip()
                                    if body == '{':
                                        k += 1
                                        body_lines_inner = []
                                        bd = 1
                                        while k < len(lines) and bd > 0:
                                            for ch in lines[k]:
                                                if ch == '{': bd += 1
                                                elif ch == '}': bd -= 1
                                            if bd > 0:
                                                body_lines_inner.append(lines[k].strip())
                                            k += 1
                                        body = ' '.join(body_lines_inner)
                                    entries.append((bit_val, body, j, k))
                                    j = k
                                    # Skip blank/else lines
                                    while j < len(lines) and lines[j].strip() in ('', 'else'):
                                        j += 1
                                    continue
                        break

                    if len(entries) >= 3:
                        chains.append((var, chain_start, entries))
            i += 1

        if not chains:
            return func, False

        # Process the first chain
        var, chain_start, entries = chains[0]
        indent = re.match(r'^(\s*)', lines[chain_start]).group(1)

        # Check all bodies assign to same lvalue
        lvalues = set()
        rhs_values = {}
        for bit_val, body, _, _ in entries:
            m = re.match(r'(\w[\w. >*\[\]]*)\s*=\s*(.+?)\s*;', body)
            if m:
                lvalues.add(m.group(1))
                # bit_val -> bit position
                bit_pos = 0
                v = bit_val
                while v > 1:
                    v >>= 1
                    bit_pos += 1
                rhs_values[bit_pos] = m.group(2)

        if len(lvalues) != 1 or not rhs_values:
            return func, False

        target_lv = list(lvalues)[0]
        max_bit = max(rhs_values.keys())

        # Build array lookup
        result_lines = []
        # Determine the type of values (use unsigned int as default)
        arr_name = f"_onehot_lut_{var}"
        arr_entries = []
        for bp in range(max_bit + 1):
            arr_entries.append(rhs_values.get(bp, "0"))

        arr_size = max_bit + 1
        result_lines.append(
            f"{indent}unsigned int {arr_name}[{arr_size}] = {{{', '.join(arr_entries)}}};")

        # Compute binary index from one-hot
        idx_name = f"_onehot_idx_{var}"
        # Build index computation: log2 of the one-hot value
        # Use a simple priority encoder
        result_lines.append(f"{indent}unsigned int {idx_name} = 0;")
        for bp in range(1, max_bit + 1):
            result_lines.append(
                f"{indent}if({var} & {1 << bp}) {idx_name} = {bp};")

        result_lines.append(f"{indent}{target_lv} = {arr_name}[{idx_name}];")

        # Find end of chain
        chain_end = entries[-1][3]

        new_lines = lines[:chain_start] + result_lines + lines[chain_end:]
        return '\n'.join(new_lines), True

# ─── Registry ─────────────────────────────────────────────────────────────────

TRANSFORM_REGISTRY = {
    "BALANCE_TREE":             BalanceTree(),
    "REASSOCIATE_ARITHMETIC":   ReassociateArithmetic(),
    "BREAK_CHAIN":              BreakChain(),
    "SPLIT_OP":                 SplitOp(),
    "INLINE_CRITICAL_FUNCTION": InlineCriticalFunction(),
    "OUTLINE_LONG_COMPUTE":     OutlineLongCompute(),
    "LOOP_FISSION":             LoopFission(),
    "LOOP_INTERCHANGE":         LoopInterchange(),
    "CONTROL_FLATTEN":          ControlFlatten(),
    "IF_CONVERSION":            IfConversion(),
    "SPECULATIVE_COMPUTE":      SpeculativeCompute(),
    "COMMON_SUBEXPR_EXTRACT":   CommonSubexprExtract(),
    "DEPENDENCE_BREAK":         DependenceBreak(),
    "PREDICATE_TO_DATAFLOW":    PredicateToDataflow(),
    "LOOP_PIPELINING":          LoopPipelining(),
    "PARTIAL_UNROLL":           PartialUnroll(),
    "PIPELINE_STAGE_INSERT":    PipelineStageInsert(),
    "MUX_TREE_BALANCE":         MuxTreeBalance(),
    "CARRY_SAVE_REWRITE":       CarrySaveRewrite(),
    "GUARD_RELAXATION":         GuardRelaxation(),
    "COPY_PROPAGATION":         CopyPropagation(),
    "ALGEBRAIC_SIMPLIFY":       AlgebraicSimplify(),
    "BOOLEAN_TO_ARITHMETIC":    BooleanToArithmetic(),
    "ENCODE_ONEHOT_TO_BINARY":  EncodeOnehotToBinary(),
}

# ─── Runner ───────────────────────────────────────────────────────────────────

def output_path(c_file: Path, transform: str) -> Path:
    rel = c_file.relative_to(BENCHMARK_DIR)
    out_dir = OUTPUT_DIR / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{rel.stem}_{transform}.c"

def summary_path(c_file: Path) -> Path:
    rel = c_file.relative_to(BENCHMARK_DIR)
    return OUTPUT_DIR / rel.parent / f"{rel.stem}_transforms.json"

def _try_reuse_area_output(c_file: Path, tname: str, out_file: Path) -> dict | None:
    """For shared transforms in rule-based mode, try to copy the already-
    generated area output instead of re-running the identical transform.

    Returns a result dict on success, or None if no reusable output exists.
    """
    if tname not in TRANSFORMS_SHARED:
        return None
    rel = c_file.relative_to(BENCHMARK_DIR)
    area_file = AREA_OUTPUT_DIR / rel.parent / f"{rel.stem}_{tname}.c"
    if not area_file.exists():
        return None
    # Also check the area summary for metadata
    area_summary_file = AREA_OUTPUT_DIR / rel.parent / f"{rel.stem}_transforms.json"
    area_meta = {}
    if area_summary_file.exists():
        try:
            area_summary = json.loads(area_summary_file.read_text())
            area_meta = area_summary.get(tname, {})
        except Exception:
            pass
    import shutil
    shutil.copy2(area_file, out_file)
    return {
        "output_file": str(out_file.relative_to(OUTPUT_DIR)),
        "applied": area_meta.get("applied", True),
        "summary": area_meta.get("summary", "reused from area output"),
        "reused_from": str(area_file),
    }


def process_file(c_file: Path, transforms: list, skip_existing: bool = True) -> dict:
    source = c_file.read_text()
    rel = str(c_file.relative_to(BENCHMARK_DIR))
    results = {}

    # Load existing summary
    sp = summary_path(c_file)
    if sp.exists():
        try:
            results = json.loads(sp.read_text())
        except Exception:
            pass

    for tname in transforms:
        out_file = output_path(c_file, tname)
        if skip_existing and out_file.exists() and tname in results:
            print(f"  [skip] {tname}")
            continue

        # For shared transforms, reuse area output if available
        reused = _try_reuse_area_output(c_file, tname, out_file)
        if reused is not None:
            results[tname] = reused
            print(f"  [reuse] {tname} (from area output)")
            sp.write_text(json.dumps(results, indent=2))
            continue

        print(f"  [run ] {tname} ...", end=" ", flush=True)
        try:
            transformer = TRANSFORM_REGISTRY[tname]
            new_code, applied, summary = transformer.apply(source)
            # Append meta comment
            meta = meta_line(applied, summary)
            final = new_code.rstrip() + meta
            out_file.write_text(final)
            results[tname] = {
                "output_file": str(out_file.relative_to(OUTPUT_DIR)),
                "applied": applied,
                "summary": summary,
            }
            status = "APPLIED" if applied else "NO-OP"
            print(f"done [{status}]")
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            results[tname] = {"output_file": None, "applied": False,
                               "summary": f"ERROR: {e}"}

        sp.write_text(json.dumps(results, indent=2))

    return results

# ─── LLM-based runner (parallel) ─────────────────────────────────────────────

import threading as _threading
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from concurrent.futures import as_completed as _as_completed

def _llm_output_path(c_file: Path, transform: str) -> Path:
    rel = c_file.relative_to(BENCHMARK_DIR)
    out_dir = LLM_OUTPUT_DIR / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{rel.stem}_{transform}.c"

def _llm_summary_path(c_file: Path) -> Path:
    rel = c_file.relative_to(BENCHMARK_DIR)
    return LLM_OUTPUT_DIR / rel.parent / f"{rel.stem}_transforms.json"

# Per-file locks for thread-safe JSON summary updates
_summary_locks: dict[str, _threading.Lock] = {}
_summary_locks_lock = _threading.Lock()

def _get_summary_lock(path: str) -> _threading.Lock:
    with _summary_locks_lock:
        if path not in _summary_locks:
            _summary_locks[path] = _threading.Lock()
        return _summary_locks[path]

def _run_one_llm_task(client, c_file, tname, descriptions, optimization_goal,
                      skip_existing):
    """Run a single (file, transform) LLM task.  Thread-safe."""
    from llm_transform import apply_llm_transform, tprint

    rel = str(c_file.relative_to(BENCHMARK_DIR))
    out_file = _llm_output_path(c_file, tname)
    sp = _llm_summary_path(c_file)
    sp_lock = _get_summary_lock(str(sp))

    # Check skip (under lock for consistent read)
    if skip_existing and out_file.exists():
        with sp_lock:
            if sp.exists():
                try:
                    existing = json.loads(sp.read_text())
                    if tname in existing:
                        return rel, tname, existing[tname]
                except Exception:
                    pass

    desc = descriptions.get(tname, tname)
    source = c_file.read_text()
    short_rel = Path(rel).stem
    tprint(f"  [{short_rel}] {tname} ...", flush=True)

    try:
        code, meta, differs = apply_llm_transform(
            client, source, rel, tname, desc,
            optimization_goal=optimization_goal,
        )
        out_file.write_text(code)
        result = {
            "output_file": str(out_file.relative_to(LLM_OUTPUT_DIR)),
            "applied": meta.get("applied", True),
            "differs": differs,
            "summary": meta.get("summary", ""),
            "mode": "llm",
        }
        status = "CHANGED" if differs else "SAME"
        applied = "applied" if meta.get("applied", True) else "no-op"
        tprint(f"  [{short_rel}] {tname} done [{status}, {applied}]")
    except Exception as e:
        tprint(f"  [{short_rel}] {tname} ERROR: {e}")
        result = {"output_file": None, "applied": False,
                  "differs": False, "summary": f"ERROR: {e}",
                  "mode": "llm"}

    # Atomic summary update (per-file lock)
    with sp_lock:
        existing = {}
        if sp.exists():
            try:
                existing = json.loads(sp.read_text())
            except Exception:
                pass
        existing[tname] = result
        sp.write_text(json.dumps(existing, indent=2))

    return rel, tname, result


def process_files_llm_parallel(client, c_files, transforms, descriptions,
                               *, optimization_goal="timing",
                               skip_existing=True, workers=8):
    """Process all (file x transform) pairs in parallel. Returns grand dict."""
    from llm_transform import tprint

    work = [(cf, t) for cf in c_files for t in transforms]
    total = len(work)
    tprint(f"[LLM] {total} tasks ({len(c_files)} files x "
           f"{len(transforms)} transforms), {workers} workers\n")

    grand: dict[str, dict] = {}
    done_count = 0

    with _ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_one_llm_task, client, cf, tname, descriptions,
                optimization_goal, skip_existing,
            ): (cf, tname)
            for cf, tname in work
        }

        for future in _as_completed(futures):
            done_count += 1
            try:
                rel, tname, result = future.result()
                grand.setdefault(rel, {})[tname] = result
            except Exception as exc:
                cf, tname = futures[future]
                rel = str(cf.relative_to(BENCHMARK_DIR))
                tprint(f"  [FATAL] {rel} / {tname}: {exc}")
                grand.setdefault(rel, {})[tname] = {
                    "output_file": None, "applied": False,
                    "differs": False, "summary": f"FATAL: {exc}",
                    "mode": "llm",
                }

            if done_count % 20 == 0 or done_count == total:
                tprint(f"[PROGRESS] {done_count}/{total}")

    return grand

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms", nargs="+", default=TRANSFORMS)
    parser.add_argument("--files",      nargs="+", default=None)
    parser.add_argument("--category",   default=None)
    parser.add_argument("--no-skip",    action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--llm",        action="store_true",
                        help="Use LLM (OpenAI-compatible API) instead of rule-based transforms")
    parser.add_argument("--llm-model",  default=None,
                        help="Override LLM model (e.g. claude-opus-4-6)")
    parser.add_argument("--workers", "-j", type=int, default=8,
                        help="Max concurrent LLM requests (default: 8)")
    parser.add_argument("--rpm", type=int, default=60,
                        help="API rate limit in requests per minute (default: 60)")
    args = parser.parse_args()

    if args.files:
        c_files = [Path(f).resolve() for f in args.files]
    else:
        c_files = sorted(BENCHMARK_DIR.rglob("*.c"))
        if args.category:
            c_files = [f for f in c_files
                       if f.relative_to(BENCHMARK_DIR).parts[0] == args.category]

    transforms = args.transforms
    active_dir = LLM_OUTPUT_DIR if args.llm else OUTPUT_DIR
    print(f"Files: {len(c_files)}  Transforms: {len(transforms)}  "
          f"Mode: {'LLM' if args.llm else 'rule-based'}  "
          f"→ up to {len(c_files)*len(transforms)} outputs in {active_dir}/\n")

    if args.dry_run:
        for f in c_files:
            print(f"  {f.relative_to(BENCHMARK_DIR)}")
        return

    if args.llm:
        from llm_transform import (create_client, TIMING_TRANSFORM_DESCRIPTIONS,
                                    init_rate_limiter)
        import llm_transform
        if args.llm_model:
            llm_transform.MODEL = args.llm_model
        client = create_client()
        init_rate_limiter(rpm=args.rpm, max_concurrent=args.workers)
        grand = process_files_llm_parallel(
            client, c_files, transforms, TIMING_TRANSFORM_DESCRIPTIONS,
            optimization_goal="timing",
            skip_existing=not args.no_skip,
            workers=args.workers,
        )
        out = LLM_OUTPUT_DIR / "grand_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(grand, indent=2))
        print(f"\nDone. Summary → {out}")
    else:
        grand = {}
        for i, cf in enumerate(c_files, 1):
            rel = str(cf.relative_to(BENCHMARK_DIR))
            print(f"[{i:3d}/{len(c_files)}] {rel}")
            grand[rel] = process_file(cf, transforms, skip_existing=not args.no_skip)
        out = OUTPUT_DIR / "grand_summary.json"
        out.write_text(json.dumps(grand, indent=2))
        print(f"\nDone. Summary → {out}")

if __name__ == "__main__":
    main()
