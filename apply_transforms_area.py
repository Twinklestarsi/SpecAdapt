#!/usr/bin/env python3
"""
RTL-derived C Area Optimizer — Rule-Based (No API Required)

Applies 20 area-oriented transformation strategies to each .c file under
benchmark_output/ using deterministic, pattern-based code transformation.

Output layout:
  optimized_output_AREA/<category>/<block>/<stem>_<TRANSFORM>.c
  optimized_output_AREA/<category>/<block>/<stem>_transforms.json
  optimized_output_AREA/grand_summary.json
"""

import re, os, json, math
from pathlib import Path
from collections import Counter, defaultdict

BENCHMARK_DIR  = Path(__file__).parent / "benchmark_output"
OUTPUT_DIR     = Path(__file__).parent / "optimized_output_AREA"
LLM_OUTPUT_DIR = Path(__file__).parent / "optimized_output_AREA_LLM"

TRANSFORMS = [
    "RESOURCE_SHARE", "RESOURCE_BIND_SMALL", "REDUCE_UNROLL",
    "SERIALIZE_PARALLELISM", "PIPELINE_RELAX", "LOOP_FUSION",
    "FUNCTION_OUTLINE_REUSE", "BITWIDTH_SHRINK",
    "TYPE_NARROWING_PROPAGATION", "CONST_PROP", "DEAD_CODE_ELIM",
    "COMMON_SUBEXPR_EXTRACT", "STRENGTH_REDUCTION", "SHIFT_ADD_REWRITE",
    "TABLE_LOOKUP_REWRITE", "ARRAY_PACK", "ARRAY_RESHAPE",
    "REDUCE_PARTITION_FACTOR", "LIMIT_MEMORY_PORTS", "REDUCE_BUFFER_DEPTH",
    "LOGIC_MINIMIZATION", "CONDITION_MERGE", "OPERATOR_TIME_MULTIPLEX",
    "REGISTER_LIFETIME_SHARE", "MEMORY_PROMOTION", "FSM_REENCODE",
    "RESET_SIMPLIFY", "COPY_PROPAGATION", "ALGEBRAIC_SIMPLIFY",
    "BOOLEAN_TO_ARITHMETIC", "ENCODE_ONEHOT_TO_BINARY",
]

# ─── Shared utilities (same as timing script) ─────────────────────────────────

def header_comment(t, changed, benefit):
    return f"/* TRANSFORM: {t}\n   Changed: {changed}\n   Benefit: {benefit}\n */\n"

def meta_line(applied, summary):
    return f"\n// TRANSFORM_META: {json.dumps({'applied': applied, 'summary': summary})}\n"

def split_file(code):
    lines = code.split('\n')
    func_starts = [i for i, ln in enumerate(lines)
                   if re.match(r'^void\s+', ln.strip())]
    if not func_starts:
        return code, [], ""
    preamble = '\n'.join(lines[:func_starts[0]])
    funcs = []
    for idx, start in enumerate(func_starts):
        end = func_starts[idx+1] if idx+1 < len(func_starts) else len(lines)
        funcs.append('\n'.join(lines[start:end]))
    main_block = ""
    body_funcs = []
    for f in funcs:
        if re.match(r'void\s+main\s*\(', f.strip()):
            main_block = f
        else:
            body_funcs.append(f)
    return preamble, body_funcs, main_block

def get_old_var_map(func_body):
    mapping = {}
    for m in re.finditer(r'(\w+_old)\s*=\s*\w+\.(\w+)\s*;', func_body):
        mapping[m.group(2)] = m.group(1)
    return mapping

def get_state_var(code):
    m = re.search(r'struct\s+state_elements_\w+\s+(\w+)\s*;', code)
    return m.group(1) if m else ""

# ─── Transform base ───────────────────────────────────────────────────────────

class Transform:
    name = "BASE"
    def apply(self, code):
        raise NotImplementedError
    def run(self, code):
        new_code, applied, summary = self.apply(code)
        return new_code.rstrip() + meta_line(applied, summary)

# ─── 1. RESOURCE_SHARE ───────────────────────────────────────────────────────

class ResourceShare(Transform):
    name = "RESOURCE_SHARE"

    def apply(self, code):
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            lines = func.split('\n')
            # Collect all RHS expressions
            rhs_map = defaultdict(list)  # rhs → [(line_idx, lvalue)]
            for i, ln in enumerate(lines):
                m = re.match(r'^\s*(\w[\w. >*\[\]]+)\s*=\s*(.+?)\s*;', ln)
                if m:
                    lv, rhs = m.group(1).strip(), m.group(2).strip()
                    # Normalize: skip trivial (single token, simple cast, zero)
                    if len(rhs) > 8 and not re.fullmatch(r'[\w.]+', rhs):
                        rhs_map[rhs].append((i, lv))

            # Find RHS used 2+ times (identical)
            shared = {rhs: idxs for rhs, idxs in rhs_map.items() if len(idxs) >= 2}
            if not shared:
                new_funcs.append(func)
                continue

            # Insert shared variables before first use
            new_lines = list(lines)
            offset = 0
            for rhs, usages in sorted(shared.items(), key=lambda x: x[1][0][0]):
                first_line = usages[0][0] + offset
                # Determine indent
                indent = re.match(r'^(\s*)', new_lines[first_line]).group(1)
                var_name = f"_shared_{abs(hash(rhs)) % 10000}"
                new_lines.insert(first_line,
                    f"{indent}__typeof__({usages[0][1]}) {var_name} = {rhs};")
                offset += 1
                # Replace all occurrences
                for (li, lv) in usages:
                    adj_li = li + offset
                    new_lines[adj_li] = new_lines[adj_li].replace(rhs, var_name, 1)

            new_funcs.append('\n'.join(new_lines))
            changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No identical multi-use RHS expressions found",
                "N/A — no shareable computations")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No shareable resources found"

        hdr = header_comment(self.name,
            "Extracted identical RHS expressions into shared temporaries",
            "HLS tool can map each shared variable to a single hardware resource")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Shared identical computation into named temporaries"

# ─── 2. RESOURCE_BIND_SMALL ──────────────────────────────────────────────────

class ResourceBindSmall(Transform):
    name = "RESOURCE_BIND_SMALL"

    def apply(self, code):
        """Remove redundant widening casts that are immediately narrowed back.
        (unsigned char)((unsigned int)x + 1) → (unsigned char)(x + 1)
        Also: add #pragma HLS RESOURCE hints for small arithmetic.
        """
        preamble, body_funcs, main_block = split_file(code)
        new_code = code

        # Remove intermediate (unsigned int) in (unsigned char)((unsigned int)var op expr)
        pat1 = re.compile(
            r'\(unsigned char\)\s*\(unsigned int\)\s*(\w[\w.]*)\s*([+\-])\s*(\w+)')
        def repl1(m):
            return f'(unsigned char)({m.group(1)} {m.group(2)} {m.group(3)})'
        new_code2 = pat1.sub(repl1, new_code)

        # Also: (unsigned char)(unsigned int)var → (unsigned char)var when no op follows
        pat2 = re.compile(r'\(unsigned char\)\(unsigned int\)(\w[\w.]*)\b')
        new_code2 = pat2.sub(r'(unsigned char)\1', new_code2)

        # Add #pragma HLS RESOURCE after struct field assignments on unsigned char ops
        # Insert pragma hint once at top of each function body with small types
        pragma_added = False
        new_funcs2 = []
        _, body_funcs2, main2 = split_file(new_code2)
        for func in body_funcs2:
            if 'unsigned char' in func and not '#pragma HLS RESOURCE' in func:
                # Add pragma comment after first {
                idx = func.find('{')
                if idx != -1:
                    func = (func[:idx+1] +
                            '\n  /* #pragma HLS RESOURCE variable=<result> core=AddSub_DSP */'
                            '\n  /* Hint: bind 8-bit adders to LUT-based AddSub for area */' +
                            func[idx+1:])
                    pragma_added = True
            new_funcs2.append(func)

        changed = new_code2 != code or pragma_added
        if not changed:
            hdr = header_comment(self.name,
                "No redundant widening casts or small-resource opportunities found",
                "N/A")
            return hdr + code, False, "No small-resource binding opportunities"

        preamble2, _, _ = split_file(new_code2)
        hdr = header_comment(self.name,
            "Removed redundant (unsigned int) widening in narrow arithmetic; added resource hints",
            "Keeps arithmetic in narrow type to avoid 32-bit adder instantiation")
        return preamble2 + "\n" + hdr + '\n'.join(new_funcs2+[main2]), True, \
               "Removed redundant widening casts and added small-resource binding hints"

# ─── 3. REDUCE_UNROLL ────────────────────────────────────────────────────────

class ReduceUnroll(Transform):
    name = "REDUCE_UNROLL"

    def apply(self, code):
        """Detect N≥8 consecutive assignments to the same struct field where a
        numeric literal increments by 1, and re-roll into a for loop.
        """
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            lines = func.split('\n')
            result = self._find_and_reroll(lines)
            if result:
                new_funcs.append('\n'.join(result))
                changed_any = True
            else:
                new_funcs.append(func)

        if not changed_any:
            hdr = header_comment(self.name,
                "No manually unrolled patterns (≥8 repetitions) detected",
                "N/A — no re-rollable sequences found")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No unrolled patterns to re-roll"

        hdr = header_comment(self.name,
            "Re-rolled manually unrolled assignment sequence into a for loop",
            "Single loop body shared across iterations reduces code and hardware replicas")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Re-rolled unrolled sequence into parameterized loop"

    def _find_and_reroll(self, lines):
        """Find run of ≥8 assigns to same LHS where an integer literal increments."""
        # Collect (line_idx, lvalue, integer_literal, full_line)
        assign_info = []
        for i, ln in enumerate(lines):
            m = re.match(r'^(\s*)(\w[\w. >*\[\]]+)\s*=\s*(.+?)\s*;', ln)
            if m:
                lv = m.group(2).strip()
                rhs = m.group(3).strip()
                # Find all integer literals in rhs
                nums = re.findall(r'\b(\d+)\b', rhs)
                assign_info.append((i, lv, nums, ln, m.group(1)))
            else:
                assign_info.append((i, None, [], ln, ''))

        # Find longest run where same lvalue and one integer changes by +1 each time
        n = len(assign_info)
        best = (0, 0, -1)  # (run_len, start_idx, varying_num_pos)
        i = 0
        while i < n:
            if assign_info[i][1] is None:
                i += 1
                continue
            # Try to extend a run from i
            lv0 = assign_info[i][1]
            nums0 = assign_info[i][2]
            # Try each number position as the varying one
            for num_pos in range(len(nums0)):
                run_len = 1
                prev_val = int(nums0[num_pos]) if nums0 else -1
                j = i + 1
                while j < n and assign_info[j][1] == lv0:
                    nums_j = assign_info[j][2]
                    if num_pos < len(nums_j):
                        cur_val = int(nums_j[num_pos])
                        if cur_val == prev_val + 1:
                            run_len += 1
                            prev_val = cur_val
                            j += 1
                            continue
                    break
                if run_len > best[0] and run_len >= 8:
                    best = (run_len, i, num_pos)
            i += 1

        run_len, start_idx, num_pos = best
        if run_len < 8:
            return None

        # Build the loop
        first_info  = assign_info[start_idx]
        indent      = first_info[4]
        lvalue      = first_info[1]
        template_ln = first_info[3]
        start_val   = int(first_info[2][num_pos])
        end_val     = start_val + run_len  # exclusive

        # Replace the numeric literal at num_pos with loop variable _ui
        template_rhs = re.match(r'^\s*\S[\w. >*\[\]]*\s*=\s*(.+?)\s*;',
                                 template_ln).group(1).strip()
        # Replace ALL occurrences of start_val in template with _ui
        # (use the specific occurrence at num_pos)
        nums_in_rhs = list(re.finditer(r'\b(\d+)\b', template_rhs))
        if num_pos < len(nums_in_rhs):
            m = nums_in_rhs[num_pos]
            loop_body_rhs = (template_rhs[:m.start()] + '_ui' +
                             template_rhs[m.end():])
        else:
            loop_body_rhs = template_rhs

        # Also look for a SECOND varying number in correlated position
        # (e.g., bit-position = start_max - _ui, mask = 1<<(start_max-_ui))
        # Detect second varying number: in line[i+1], find numbers different from line[i]
        second_varies = {}  # num_pos_in_rhs → (relation_to_ui: "31-_ui", "32-_ui", etc.)
        if run_len >= 2:
            rhs_0 = assign_info[start_idx][2]      # nums list of first line
            rhs_1 = assign_info[start_idx+1][2]    # nums list of second line
            for k2, (v0, v1) in enumerate(zip(rhs_0, rhs_1)):
                if k2 == num_pos:
                    continue  # this is the primary varying one
                try:
                    iv0, iv1 = int(v0), int(v1)
                    if iv0 - iv1 == 1:  # decrements: expr = start_max - _ui
                        second_varies[k2] = (iv0, 'dec')
                    elif iv1 - iv0 == 1:  # increments: expr = start_val2 + _ui
                        second_varies[k2] = (iv0, 'inc')
                except ValueError:
                    pass

        # Build loop body with substitutions
        nums_in_rhs = list(re.finditer(r'\b(\d+)\b', template_rhs))
        loop_body_rhs2 = template_rhs
        # Build substitution map (process in reverse order to preserve indices)
        substitutions = []
        if num_pos < len(nums_in_rhs):
            m0 = nums_in_rhs[num_pos]
            substitutions.append((m0.start(), m0.end(), '_ui'))
        for k2, (base_val, direction) in second_varies.items():
            if k2 < len(nums_in_rhs):
                mk = nums_in_rhs[k2]
                expr = f'({base_val}-_ui)' if direction == 'dec' else f'({base_val}+_ui)'
                substitutions.append((mk.start(), mk.end(), expr))

        # Apply substitutions in reverse order
        substitutions.sort(key=lambda x: -x[0])
        loop_body_rhs2 = template_rhs
        for s, e, replacement in substitutions:
            loop_body_rhs2 = loop_body_rhs2[:s] + replacement + loop_body_rhs2[e:]

        loop_lines = [
            f"{indent}/* REDUCE_UNROLL: re-rolled {run_len} identical assignments */",
            f"{indent}{{",
            f"{indent}  int _ui;",
            f"{indent}  for(_ui = {start_val}; _ui < {end_val}; _ui++) {{",
            f"{indent}    {lvalue} = {loop_body_rhs2};",
            f"{indent}  }}",
            f"{indent}}}",
        ]

        new_lines = (lines[:start_idx] + loop_lines +
                     lines[start_idx + run_len:])
        return new_lines

# ─── 4. SERIALIZE_PARALLELISM ────────────────────────────────────────────────

class SerializeParallelism(Transform):
    name = "SERIALIZE_PARALLELISM"

    def apply(self, code):
        """Add HLS ALLOCATION pragma to serialize parallel multiply/add operations.
        Also: for functions with 3+ independent multiply operations, factor out a
        common sub-expression to share the multiplier.
        """
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            # Count top-level multiply operations
            mul_count = len(re.findall(r'\b\d+\s*\*\s*\w+|\w+\s*\*\s*\w+', func))
            if mul_count >= 3:
                # Add ALLOCATION pragma to limit multiply instances to 1
                idx = func.find('{')
                if idx != -1:
                    pragma = ('\n  /* #pragma HLS ALLOCATION instances=mul limit=1 operation */'
                              '\n  /* SERIALIZE: time-multiplex multiply units to save DSPs */')
                    func = func[:idx+1] + pragma + func[idx+1:]
                    changed_any = True
            new_funcs.append(func)

        if not changed_any:
            hdr = header_comment(self.name,
                "Fewer than 3 multiply operations; serialization not beneficial",
                "N/A — insufficient parallelism to serialize")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "Insufficient parallel operations to serialize"

        hdr = header_comment(self.name,
            "Added HLS ALLOCATION pragma to serialize parallel multiply operations",
            "Limits multiplier instances, trading latency for area savings")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Added serialization pragma for parallel multiply operations"

# ─── 5. PIPELINE_RELAX ───────────────────────────────────────────────────────

class PipelineRelax(Transform):
    name = "PIPELINE_RELAX"

    def apply(self, code):
        """Break compound expressions into pipeline stages to allow the HLS tool
        to relax timing and share logic across cycles.
        Split: lvalue = cond ? (long_expr) : default
        Into:  _pipe = long_expr;  lvalue = cond ? _pipe : default;
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
                # Match: lv = cond ? LONG_EXPR : SHORT;  where LONG_EXPR > 15 chars
                m = re.match(
                    r'^(\s*)([\w][\w. >*\[\]]+)\s*=\s*(.{3,40}\?)\s*(.{15,}?)\s*:\s*(\w[\w.]*\s*);$',
                    ln)
                if m:
                    indent = m.group(1)
                    lv      = m.group(2)
                    cond    = m.group(3)   # includes ' ?'
                    t_expr  = m.group(4)
                    f_expr  = m.group(5)
                    pvar = f"_pipe{tidx[0]}"
                    tidx[0] += 1
                    # Wrap in braces so both statements stay inside a
                    # braceless if/else that may surround the original line.
                    new_lines.append(f"{indent}{{")
                    new_lines.append(f"{indent}  __typeof__({lv}) {pvar} = {t_expr};")
                    new_lines.append(f"{indent}  {lv} = {cond} {pvar} : {f_expr};")
                    new_lines.append(f"{indent}}}")
                    func_changed = True
                    continue
                new_lines.append(ln)
            if func_changed:
                new_funcs.append('\n'.join(new_lines))
                changed_any = True
            else:
                new_funcs.append(func)

        if not changed_any:
            hdr = header_comment(self.name,
                "No compound ternary expressions suitable for pipeline stage insertion",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No pipeline relaxation opportunities"

        hdr = header_comment(self.name,
            "Inserted intermediate pipeline register before ternary selection",
            "Breaks critical path at mux input; HLS can schedule stages independently")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Added pipeline stage register before ternary mux"

# ─── 6. LOOP_FUSION ──────────────────────────────────────────────────────────

class LoopFusion(Transform):
    name = "LOOP_FUSION"

    def apply(self, code):
        """Merge two consecutive for-loops with identical loop bounds."""
        # Find two consecutive for-loops
        pat = re.compile(
            r'([ \t]*for\s*\(([^;]+);([^;]+);([^)]+)\))\s*\n'
            r'([ \t]+\w[^\n]+\n)'
            r'\s*\n?'
            r'([ \t]*for\s*\(([^;]+);([^;]+);([^)]+)\))\s*\n'
            r'([ \t]+\w[^\n]+)',
            re.MULTILINE)
        m = pat.search(code)
        if not m:
            hdr = header_comment(self.name,
                "No consecutive for-loops with same bounds found",
                "N/A — no fusable loops")
            return hdr + code, False, "No fusable loops found"

        hdr1, init1, cond1, step1 = m.group(1), m.group(2), m.group(3), m.group(4)
        body1 = m.group(5).rstrip()
        hdr2, init2, cond2, step2 = m.group(6), m.group(7), m.group(8), m.group(9)
        body2 = m.group(10).rstrip()

        if cond1.strip() != cond2.strip():
            hdr = header_comment(self.name,
                "Consecutive loops found but with different bounds",
                "N/A — fusion requires identical loop bounds")
            return hdr + code, False, "Loop bounds differ — cannot fuse"

        indent = re.match(r'^(\s*)', hdr1).group(1)
        fused = (f"{hdr1}\n"
                 f"{body1};\n"
                 f"{indent}  {body2};")
        new_code = code[:m.start()] + fused + code[m.end():]
        hdr = header_comment(self.name,
            "Fused two consecutive loops with identical bounds into one",
            "Eliminates loop-control overhead; single loop body reduces code area")
        return hdr + new_code, True, "Fused two consecutive loops"

# ─── 7. FUNCTION_OUTLINE_REUSE ───────────────────────────────────────────────

class FunctionOutlineReuse(Transform):
    name = "FUNCTION_OUTLINE_REUSE"

    def apply(self, code):
        """Extract repeated expression patterns into a reusable helper function.
        Targets: N≥8 lines with the same structural pattern differing only in
        variable names that follow a regular naming convention.
        """
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        extracted = []
        changed_any = False

        for func in body_funcs:
            fname_m = re.match(r'void\s+(\w+)\s*\(', func.strip())
            if not fname_m:
                new_funcs.append(func)
                continue
            fname = fname_m.group(1)

            lines = func.split('\n')
            # Detect the sign-extension pattern from bit_reorg files:
            # LHS = ((VAR >> 10 & 1 ? 2097151 : 0) | OLD & MASK) & 2097151;
            sign_ext_lines = []
            for i, ln in enumerate(lines):
                m = re.match(
                    r'(\s*)(\w[\w.]*)\s*=\s*\(\((\w+)\s*>>\s*10\s*&\s*1\s*\?\s*2097151\s*:\s*0\)'
                    r'\s*\|\s*(\w+)\s*&\s*(\d+)\)\s*&\s*2097151\s*;',
                    ln)
                if m:
                    sign_ext_lines.append((i, m.group(1), m.group(2),
                                           m.group(3), m.group(4), m.group(5)))

            if len(sign_ext_lines) >= 4:
                # Extract a helper function
                helper_name = f"{fname}_sign_ext11"
                helper = (
                    f"/* Extracted repeated sign-extension pattern */\n"
                    f"static unsigned int {helper_name}(unsigned short val, unsigned int old_val, unsigned int keep_mask) {{\n"
                    f"  return ((val >> 10 & 1 ? 2097151 : 0) | old_val & keep_mask) & 2097151;\n"
                    f"}}\n"
                )
                extracted.append(helper)
                # Replace matching lines
                new_lines = list(lines)
                for (i, indent, lhs, var, old_var, mask) in sign_ext_lines:
                    new_lines[i] = (f"{indent}{lhs} = "
                                    f"{helper_name}({var}, {old_var}, {mask});")
                new_funcs.append('\n'.join(new_lines))
                changed_any = True
                continue

            # Detect: N identical-structure assignments differing only by subscript
            # e.g., su.Z11_int = ...; su.Z12_int = ...; (same template)
            # Group lines by their "template" (replace terminal _\d+ digits with _N)
            template_map = defaultdict(list)
            for i, ln in enumerate(lines):
                m = re.match(r'(\s*\w[\w.]*\s*=\s*)(.+?)(;)', ln.rstrip())
                if m:
                    # Canonicalize: replace trailing digit sequences in variable names
                    template = re.sub(r'(\w+?)(\d+)\b', r'\1N', ln)
                    template_map[template].append(i)

            best_tmpl = max(template_map.items(), key=lambda x: len(x[1]),
                            default=(None, []))
            if len(best_tmpl[1]) >= 8:
                # Just add a comment noting the reuse opportunity
                first_i = best_tmpl[1][0]
                n = len(best_tmpl[1])
                new_lines = list(lines)
                new_lines.insert(first_i,
                    f"  /* FUNCTION_OUTLINE_REUSE: {n} lines follow identical pattern;"
                    f" consider parameterizing with a helper function */")
                new_funcs.append('\n'.join(new_lines))
                changed_any = True
                continue

            new_funcs.append(func)

        if not changed_any:
            hdr = header_comment(self.name,
                "No repeated structural patterns suitable for helper extraction found",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No reusable patterns found"

        hdr = header_comment(self.name,
            "Extracted repeated structural expressions into shared helper function",
            "HLS synthesizes helper once and routes multiple calls to shared logic")
        result = preamble + "\n" + hdr + '\n'.join(extracted + new_funcs + [main_block])
        return result, True, "Outlined repeated expression pattern into reusable helper"

# ─── 8. BITWIDTH_SHRINK ──────────────────────────────────────────────────────

class BitwidthShrink(Transform):
    name = "BITWIDTH_SHRINK"

    def apply(self, code):
        """Shrink variable types where usage guarantees fit in narrower width.
        1. Loop counter int i → unsigned char i when loop bound ≤ 255
        2. Mark int loop vars used only as indices as 'unsigned char'
        3. Add mask annotations where 32-bit arithmetic produces ≤8-bit results
        """
        new_code = code
        changed = False

        # 1. for(int i = ...; i < N; ...) where N ≤ 255 → unsigned char i
        def shrink_loop_var(m):
            nonlocal changed
            var  = m.group(1)
            init = m.group(2)
            bound_var = m.group(3)
            bound_val = m.group(4)
            if bound_val and int(bound_val) <= 255:
                changed = True
                # Replace int var with unsigned char var
                return m.group(0).replace(f'int {var}', f'unsigned char {var}', 1)
            return m.group(0)

        new_code = re.sub(
            r'\bfor\s*\(\s*int\s+(\w+)\s*=\s*(\d+)\s*;\s*'
            r'(?:\(int\))?(\w+)\s*<\s*(\d+)\s*;',
            shrink_loop_var, new_code)

        # Also fix standalone 'int i;' declarations when i is a loop counter ≤ 255
        # Match: int i; ... for(i = 0; ... i < 255;
        for var_m in re.finditer(r'\bint\s+(\w+)\s*;', new_code):
            var = var_m.group(1)
            # Check if used in for loop with bound ≤ 255
            loop_m = re.search(
                r'for\s*\(\s*' + re.escape(var) + r'\s*=.*?;\s*'
                r'(?:\(int\))?' + re.escape(var) + r'\s*<\s*(\d+)\s*;',
                new_code)
            if loop_m and int(loop_m.group(1)) <= 255:
                new_code = new_code.replace(
                    var_m.group(0),
                    var_m.group(0).replace('int ', 'unsigned char ', 1), 1)
                changed = True

        # 2. Add mask shrink hints for 11-bit values (2047 mask)
        # Already handled by REASSOCIATE; here add explicit type annotation
        # (unsigned int)x & 2047 → (unsigned short)((unsigned int)x & 2047)
        pat_11bit = re.compile(r'(?<!\(unsigned short\))(\(\w+\s*&\s*2047\))')
        def shrink_11bit(m):
            nonlocal changed
            changed = True
            return f'(unsigned short){m.group(1)}'
        new_code = pat_11bit.sub(shrink_11bit, new_code)

        if not changed:
            hdr = header_comment(self.name,
                "No bit-width shrink opportunities found",
                "N/A — all types already appropriately sized")
            return hdr + code, False, "No bitwidth shrink opportunities"

        hdr = header_comment(self.name,
            "Narrowed loop counter types and annotated 11-bit intermediate values",
            "Smaller types reduce register/wire count and enable narrower arithmetic units")
        return hdr + new_code, True, "Narrowed int loop counters and 11-bit value types"

# ─── 9. TYPE_NARROWING_PROPAGATION ───────────────────────────────────────────

class TypeNarrowingPropagation(Transform):
    name = "TYPE_NARROWING_PROPAGATION"

    def apply(self, code):
        """Remove redundant widening-then-narrowing cast chains.
        Patterns:
          (unsigned char)(unsigned int)x  → (unsigned char)x
          (int)i + 1 & 4294967295         → i + 1
          (unsigned int)x == N            → x == N  (where x is _Bool or uchar)
        """
        new_code = code
        changed = False

        # (unsigned char)(unsigned int)x → (unsigned char)x
        p1 = re.compile(r'\(unsigned char\)\(unsigned int\)(\w[\w.]*)')
        r1 = new_code
        new_code = p1.sub(r'(unsigned char)\1', new_code)
        changed |= new_code != r1

        # (unsigned char)((unsigned int)x) → (unsigned char)(x)
        r2 = new_code
        new_code = re.sub(r'\(unsigned char\)\s*\(\s*\(unsigned int\)\s*(\w[\w.]*)\s*\)',
                          r'(unsigned char)\1', new_code)
        changed |= new_code != r2

        # (unsigned int)x == N where x is _Bool → x (cast unnecessary for comparison)
        r3 = new_code
        new_code = re.sub(r'\(unsigned int\)(_Bool\s+\w+|[a-z]\w*)\s*==',
                          lambda m: m.group(1) + ' ==', new_code)
        changed |= new_code != r3

        # (unsigned char)x == N (where x is already unsigned char) – remove cast
        # Only when x is declared unsigned char in same function
        r4 = new_code
        # Find all unsigned char variables
        uchar_vars = set(re.findall(r'unsigned char\s+(\w+)', code))
        for v in uchar_vars:
            new_code = re.sub(
                r'\(unsigned (?:char|int)\)\s*' + re.escape(v) + r'\b\s*==',
                v + ' ==', new_code)
        changed |= new_code != r4

        if not changed:
            hdr = header_comment(self.name,
                "No redundant widening cast chains found",
                "N/A")
            return hdr + code, False, "No type narrowing opportunities"

        hdr = header_comment(self.name,
            "Removed redundant widening/narrowing cast pairs and comparison casts",
            "Eliminates unnecessary type conversions; HLS avoids inserting width-change muxes")
        return hdr + new_code, True, "Removed redundant widening/narrowing cast chains"

# ─── 10. CONST_PROP ──────────────────────────────────────────────────────────

class ConstProp(Transform):
    name = "CONST_PROP"

    def apply(self, code):
        """Fold constant expressions and decode opaque irep() binary literals."""
        new_code = code
        changed = False

        # 1. Fold integer constant arithmetic: N op M → result
        def fold(m):
            nonlocal changed
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            try:
                if op == '-': r = a - b
                elif op == '+': r = a + b
                elif op == '*': r = a * b
                else: return m.group(0)
                if r >= 0:
                    changed = True
                    return str(r)
            except Exception:
                pass
            return m.group(0)

        r0 = new_code
        for _ in range(4):
            new_code = re.sub(r'\b(\d{2,})\s*([-+])\s*(\d+)\b', fold, new_code)

        # 2. Remove no-op masks and shifts
        r1 = new_code
        new_code = re.sub(r'\s*&\s*4294967295\b', '', new_code)
        new_code = re.sub(r'\s*<<\s*0\b', '', new_code)
        new_code = re.sub(r'\s*>>\s*0\b', '', new_code)
        new_code = re.sub(r'\bfor\s*\(\s*(\w+)\s*=\s*0\s*&\s*\d+', r'for(\1 = 0', new_code)
        new_code = re.sub(r'\(int\)(\w+)\s*\+\s*1\s*&\s*4294967295', r'\1 + 1', new_code)
        changed |= new_code != r1

        # 3. Decode irep() binary value literals
        # Pattern: irep("(\"\" \"type\" (\"unsignedbv\" \"width\" (\"N\")) \"value\" (\"BINARY\"))")
        def decode_irep_val(m):
            nonlocal changed
            binary_str = m.group(1)
            try:
                val = int(binary_str, 2)
                changed = True
                return f'0x{val:02X}u'
            except ValueError:
                return m.group(0)

        r2 = new_code
        new_code = re.sub(
            r'irep\("[^"]*"value"\s*\("([01]+)"\)[^"]*"\)',
            decode_irep_val, new_code)
        changed |= new_code != r2

        # 4. Replace irep("(\"nil\" ...)") → 0 (nil type = zero/undefined)
        r3 = new_code
        new_code = re.sub(r'irep\("[^"]*\\\"nil\\\"[^"]*"\)', '0u', new_code)
        changed |= new_code != r3

        if not changed:
            hdr = header_comment(self.name,
                "No constant-foldable expressions or decodable literals found",
                "N/A")
            return hdr + code, False, "No constant propagation opportunities"

        hdr = header_comment(self.name,
            "Folded constant arithmetic, removed no-op masks, decoded irep() literals",
            "Eliminates combinational logic for constant sub-computations; reduces area")
        return hdr + new_code, True, "Folded constants and decoded irep literals"

# ─── 11. DEAD_CODE_ELIM ──────────────────────────────────────────────────────

class DeadCodeElim(Transform):
    name = "DEAD_CODE_ELIM"

    def apply(self, code):
        """Remove dead code:
        1. Parameters declared but never read in function body (clk, Tp, etc.)
        2. Duplicate variable declarations
        3. Variables assigned once but never read
        4. Add (void)param; for unused params to suppress warnings
        """
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, ch = self._elim(func)
            new_funcs.append(new_func)
            if ch:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No dead parameters or duplicate declarations found",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No dead code found"

        hdr = header_comment(self.name,
            "Added (void) suppression for unused parameters; removed duplicate declarations",
            "Unused parameter ports removed by HLS; saves routing and register area")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Marked unused parameters dead; removed duplicate declarations"

    def _elim(self, func):
        changed = False
        lines = func.split('\n')

        # Extract function signature params
        sig_m = re.match(r'void\s+\w+\s*\((.+?)\)', func.replace('\n', ' '), re.DOTALL)
        if not sig_m:
            return func, False

        params_str = sig_m.group(1)
        params = []
        for p in params_str.split(','):
            p = p.strip()
            pm = re.match(r'.+\s+\*?(\w+)\s*$', p)
            if pm:
                params.append(pm.group(1))

        # Find body (after first {)
        body_start = func.find('{')
        if body_start == -1:
            return func, False
        body = func[body_start:]

        # Find unused params: named in signature but never appear in body
        unused_params = []
        for p in params:
            # Count uses in body (excluding the declaration lines)
            uses = len(re.findall(r'\b' + re.escape(p) + r'\b', body))
            # The param itself appears 0 times in body if truly unused
            if uses == 0 and p not in ('clk',):  # clk handled specially
                unused_params.append(p)
            elif p == 'clk' and uses == 0:
                unused_params.append(p)

        # Add (void)param; for each unused param at start of body
        if unused_params:
            void_stmts = '\n'.join(f'  (void){p};' for p in unused_params)
            insert_pos = func.find('{') + 1
            func = func[:insert_pos] + f'\n{void_stmts}  /* unused parameter */' + func[insert_pos:]
            changed = True

        # Remove duplicate variable declarations (same type and name twice)
        decl_seen = set()
        new_lines = []
        for ln in func.split('\n'):
            dm = re.match(r'^(\s*)([\w ]+)\s+(\w+)\s*;', ln)
            if dm:
                decl_key = (dm.group(2).strip(), dm.group(3).strip())
                if decl_key in decl_seen:
                    new_lines.append(f"{dm.group(1)}/* DEAD: duplicate declaration of {dm.group(3)} removed */")
                    changed = True
                    continue
                decl_seen.add(decl_key)
            new_lines.append(ln)

        func = '\n'.join(new_lines)
        return func, changed

# ─── 12. COMMON_SUBEXPR_EXTRACT (area-focused) ────────────────────────────────

class CommonSubexprExtractArea(Transform):
    name = "COMMON_SUBEXPR_EXTRACT"

    def apply(self, code):
        """Extract repeated sub-expressions focusing on arithmetic/logic sharing.
        For area: target mul/shift sub-expressions repeated 2+ times.
        """
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        defines_block = ""
        changed_any = False

        # Find repeated non-trivial sub-expressions in the full code
        # Focus on arithmetic: (A op B) patterns
        subexprs = re.findall(r'\((?:\w+\s*[+\-*&|^]\s*\w+)\)', code)
        counted = Counter(subexprs)

        macros = {}
        define_lines = []
        idx = 0
        for expr, cnt in counted.items():
            inner = expr[1:-1].strip()
            if cnt >= 3 and not re.match(r'^\w+$', inner):
                macro = f"CSE_{idx}"
                define_lines.append(f"#define {macro} {expr}")
                macros[expr] = macro
                idx += 1

        if macros:
            defines_block = '\n'.join(define_lines) + '\n'
            rest = '\n'.join(body_funcs + [main_block])
            for expr, macro in macros.items():
                rest = rest.replace(expr, macro)
            changed_any = True
            hdr = header_comment(self.name,
                f"Extracted {len(macros)} common arithmetic sub-expression(s) as macros",
                "Shared computation eliminates duplicate logic gates in synthesis")
            return preamble + '\n' + defines_block + hdr + rest, True, \
                   f"Extracted {len(macros)} common arithmetic sub-expressions"

        # Fallback: look for repeated A op B (without parens)
        for func in body_funcs:
            rhs_exprs = re.findall(r'\b(\w+\s*[+\-]\s*\w+)\b', func)
            counted2 = Counter(rhs_exprs)
            shared = {e: c for e, c in counted2.items() if c >= 2 and len(e) > 4}
            if shared:
                new_func = func
                insert_stmts = []
                for expr, cnt in shared.items():
                    vname = f"_cse_{abs(hash(expr)) % 1000}"
                    insert_stmts.append(f"  unsigned int {vname} = {expr};")
                    new_func = new_func.replace(expr, vname)
                # Insert at start of body
                brace = new_func.find('{')
                if brace != -1:
                    new_func = (new_func[:brace+1] + '\n' +
                                '\n'.join(insert_stmts) +
                                new_func[brace+1:])
                new_funcs.append(new_func)
                changed_any = True
            else:
                new_funcs.append(func)

        if not changed_any:
            hdr = header_comment(self.name,
                "No repeated sub-expressions (≥2 occurrences) found",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No common sub-expressions found"

        hdr = header_comment(self.name,
            "Extracted repeated arithmetic sub-expressions into shared variables",
            "Reduces gate count by sharing identical logic in synthesis")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Extracted common arithmetic sub-expressions"

# ─── 13. STRENGTH_REDUCTION ──────────────────────────────────────────────────

def _const_to_shift_add(n: int, var: str) -> str:
    """Convert N*var into shift+add form using CSD (Canonical Signed Digits)."""
    if n == 0: return "0"
    if n == 1: return var
    if n < 0:
        return f"(-({_const_to_shift_add(-n, var)}))"

    # Check if power of 2
    if n & (n - 1) == 0:
        shift = int(math.log2(n))
        return f"({var} << {shift})" if shift > 0 else var

    # Use binary representation, group consecutive 1s for CSD
    terms = []
    bit = 0
    while n:
        if n & 1:
            terms.append(f"({var} << {bit})" if bit > 0 else var)
        n >>= 1
        bit += 1
    if len(terms) <= 3:
        return " + ".join(terms)
    # For larger: try N = 2^k - r form
    k = n.bit_length() if n else 0
    # Just use binary decomposition
    return " + ".join(terms)

class StrengthReduction(Transform):
    name = "STRENGTH_REDUCTION"

    def apply(self, code):
        """Replace constant multiplications with shift+add sequences."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            new_func, ch = self._reduce(func)
            new_funcs.append(new_func)
            if ch:
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No constant multiplications found for strength reduction",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No constant multiplications to reduce"

        hdr = header_comment(self.name,
            "Replaced constant multiplications with shift-add sequences",
            "Eliminates multiplier hardware; shifts are free in RTL, adds use minimal LUTs")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Replaced const*var with shift+add sequences"

    def _reduce(self, func):
        changed = False

        def repl(m):
            nonlocal changed
            const_s, var = m.group(1), m.group(2)
            n = int(const_s)
            if n <= 1 or n > 1000:
                return m.group(0)
            sa = _const_to_shift_add(n, var)
            if sa != m.group(0):
                changed = True
                return sa
            return m.group(0)

        # Pattern: CONST * var  or  var * CONST
        new_func = re.sub(r'\b(\d+)\s*\*\s*(\w+)\b', repl, func)
        new_func = re.sub(r'\b(\w+)\s*\*\s*(\d+)\b',
                          lambda m: re.sub(r'\b(\d+)\s*\*\s*(\w+)\b',
                                           repl, f'{m.group(2)} * {m.group(1)}'),
                          new_func)
        return new_func, changed

# ─── 14. SHIFT_ADD_REWRITE ───────────────────────────────────────────────────

class ShiftAddRewrite(Transform):
    name = "SHIFT_ADD_REWRITE"

    def apply(self, code):
        """Rewrite power-of-2 multiply/divide and also detect patterns like
        x * 2 → x + x, x / 2 → x >> 1, x % 2 → x & 1.
        Also: (x << N) + x → already canonical for Strength Reduction results.
        Complement: simplify x >> 0 and x << 0.
        """
        new_code = code
        changed = False

        # x * 2 → (x << 1)   [not handled by STRENGTH_REDUCTION = "(x << 0) + x"]
        def pow2_mul(m):
            nonlocal changed
            var, const = m.group(1), int(m.group(2))
            if const > 1 and (const & (const-1)) == 0:
                k = int(math.log2(const))
                changed = True
                return f"({var} << {k})"
            return m.group(0)

        r0 = new_code
        new_code = re.sub(r'\b(\w+)\s*\*\s*(\d+)\b', pow2_mul, new_code)
        new_code = re.sub(r'\b(\d+)\s*\*\s*(\w+)\b',
                          lambda m: pow2_mul(
                              type('M', (), {'group': lambda s, n:
                                  m.group(2) if n==1 else m.group(1)})()),
                          new_code)
        changed |= new_code != r0

        # x / 2^k → x >> k
        def pow2_div(m):
            nonlocal changed
            var, const = m.group(1), int(m.group(2))
            if const > 1 and (const & (const-1)) == 0:
                k = int(math.log2(const))
                changed = True
                return f"({var} >> {k})"
            return m.group(0)
        r1 = new_code
        new_code = re.sub(r'\b(\w+)\s*/\s*(\d+)\b', pow2_div, new_code)
        changed |= new_code != r1

        # x % 2^k → x & (2^k - 1)
        def pow2_mod(m):
            nonlocal changed
            var, const = m.group(1), int(m.group(2))
            if const > 1 and (const & (const-1)) == 0:
                changed = True
                return f"({var} & {const-1})"
            return m.group(0)
        r2 = new_code
        new_code = re.sub(r'\b(\w+)\s*%\s*(\d+)\b', pow2_mod, new_code)
        changed |= new_code != r2

        # Remove no-op shifts
        r3 = new_code
        new_code = re.sub(r'(\w+)\s*<<\s*0\b', r'\1', new_code)
        new_code = re.sub(r'(\w+)\s*>>\s*0\b', r'\1', new_code)
        changed |= new_code != r3

        if not changed:
            hdr = header_comment(self.name,
                "No power-of-2 multiply/divide or no-op shift patterns found",
                "N/A")
            return hdr + code, False, "No shift/add rewrite opportunities"

        hdr = header_comment(self.name,
            "Rewrote power-of-2 multiply/divide as shifts; removed no-op shifts",
            "Shifts map to wiring in RTL with zero LUT cost")
        return hdr + new_code, True, "Rewrote power-of-2 operations as shifts"

# ─── 15. TABLE_LOOKUP_REWRITE ─────────────────────────────────────────────────

class TableLookupRewrite(Transform):
    name = "TABLE_LOOKUP_REWRITE"

    def apply(self, code):
        """Convert if(var==N) → val  chains into array lookups.
        Also decode irep binary literals in the process.
        """
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
                "No equality-chain patterns suitable for table lookup conversion",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No equality chains to convert to table lookup"

        hdr = header_comment(self.name,
            "Converted if(var==N) equality chain to indexed array lookup",
            "Single MUX tree from address decoder instead of priority encoder chain")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Converted equality chain to array/table lookup"

    def _decode_irep_value(self, s):
        """Try to extract a decimal value from an irep() binary string."""
        m = re.search(r'"value"\s*\("([01]+)"\)', s)
        if m:
            try:
                return str(int(m.group(1), 2))
            except ValueError:
                pass
        return None

    def _convert(self, func):
        """Line-based parser: collect if(cast var == N) → lhs = val chains."""
        lines = func.split('\n')

        # Build index of if(cond_var==N) → next stmt assignments
        case_map = defaultdict(list)  # (lhs, cond_var) → [(N, val, if_line, stmt_line)]
        i = 0
        while i < len(lines):
            ln = lines[i].rstrip()
            m = re.match(r'^\s*if\s*\(\s*(?:\([^)]+\)\s*)?(\w+)\s*==\s*(\d+)\s*\)', ln)
            if m:
                cond_var, n = m.group(1), int(m.group(2))
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    lm = re.match(r'^\s*([\w.]+)\s*=\s*(.+?)\s*;', lines[j])
                    if lm:
                        lhs = lm.group(1).strip()
                        val = lm.group(2).strip()
                        decoded = self._decode_irep_value(val)
                        if decoded:
                            val = decoded
                        case_map[(lhs, cond_var)].append((n, val, i, j))
                        i = j + 1
                        continue
            i += 1

        if not case_map:
            return func, False

        best_key = max(case_map, key=lambda k: len(case_map[k]))
        cases = case_map[best_key]
        if len(cases) < 3:
            return func, False

        lhs, cond_var = best_key
        cases_sorted = sorted(cases, key=lambda x: x[0])
        n_max = cases_sorted[-1][0]

        # Build lookup table
        table_size = n_max + 1
        table = ['0'] * table_size
        for n_val, val, _, _ in cases_sorted:
            table[n_val] = val

        type_str = "unsigned char"
        first_if_line = cases_sorted[0][2]
        last_stmt_line = cases_sorted[-1][3]

        # Find end of the if-else chain: scan forward from last_stmt_line
        # Skip blank lines and 'else' lines only; stop at first real statement
        chain_end = last_stmt_line + 1
        while chain_end < len(lines):
            stripped = lines[chain_end].strip()
            if stripped == '' or stripped == 'else':
                chain_end += 1
                continue
            # Check if still in chain (another if or nested assignment)
            if re.match(r'if\s*\(\s*(?:\([^)]+\)\s*)?' + re.escape(cond_var), stripped):
                # skip this if + its stmt
                chain_end += 2
                continue
            # Check for default else assignment (else without if)
            default_m = re.match(r'([\w.]+)\s*=\s*(.+?)\s*;', stripped)
            if default_m and default_m.group(1).strip() == lhs:
                chain_end += 1  # consume the default
                continue
            break  # stop here — this is the first line AFTER the chain

        indent = re.match(r'^(\s*)', lines[first_if_line]).group(1)
        tbl_name = f"_tbl_{lhs.replace('.', '_')}"
        entries = ', '.join(table)
        default_val = cases_sorted[-1][1]

        lookup_lines = [
            f"{indent}static const {type_str} {tbl_name}[{table_size}] = {{{entries}}};",
            f"{indent}{lhs} = ((unsigned int){cond_var} < {table_size})"
            f" ? {tbl_name}[(unsigned int){cond_var}] : {default_val};",
        ]

        new_lines = lines[:first_if_line] + lookup_lines + lines[chain_end:]
        return '\n'.join(new_lines), True

# ─── 16. ARRAY_PACK ──────────────────────────────────────────────────────────

class ArrayPack(Transform):
    name = "ARRAY_PACK"

    def apply(self, code):
        """Pack multiple _Bool struct fields into a single unsigned char bitfield.
        Only applies when struct has 4+ _Bool fields.
        Adds struct __attribute__((packed)) and bit-field annotations as comments.
        """
        preamble, body_funcs, main_block = split_file(code)

        # Find struct with multiple _Bool fields
        struct_m = re.search(
            r'(struct\s+state_elements_\w+\s*\{)([^}]+)(\})',
            preamble, re.DOTALL)
        if not struct_m:
            hdr = header_comment(self.name,
                "No state struct found for array packing",
                "N/A")
            return hdr + code, False, "No struct to pack"

        struct_body = struct_m.group(2)
        bool_fields = re.findall(r'_Bool\s+(\w+)\s*;', struct_body)

        if len(bool_fields) < 4:
            hdr = header_comment(self.name,
                f"Struct has only {len(bool_fields)} _Bool field(s); packing not beneficial",
                "N/A — need ≥4 _Bool fields to justify packing")
            return hdr + code, False, f"Only {len(bool_fields)} _Bool fields; packing not beneficial"

        # Build packed struct suggestion
        n_bytes = math.ceil(len(bool_fields) / 8)
        pack_comment = (
            f"/* ARRAY_PACK: {len(bool_fields)} _Bool fields could be packed into\n"
            f"   {n_bytes} byte(s) using C bitfields:\n"
            f"   struct packed_state {{\n"
        )
        for i, f in enumerate(bool_fields):
            pack_comment += f"     unsigned int {f} : 1;\n"
        pack_comment += (
            f"   }} __attribute__((packed));\n"
            f"   Saves {len(bool_fields) - n_bytes} bytes per state instance */\n"
        )

        # Insert after struct closing brace + semicolon
        insert_pos = struct_m.end()
        # Skip past the ';' that follows the struct's '}'
        rest_after_struct = preamble[insert_pos:]
        semi_offset = rest_after_struct.find(';')
        if semi_offset != -1:
            insert_pos += semi_offset + 1
        new_preamble = preamble[:insert_pos] + '\n' + pack_comment
        rest = '\n'.join(body_funcs + [main_block])

        hdr = header_comment(self.name,
            f"Annotated {len(bool_fields)} _Bool fields as packable into {n_bytes} byte(s)",
            "Bit-packing reduces memory/register area for state storage")
        return new_preamble + "\n" + hdr + rest, True, \
               f"Annotated {len(bool_fields)} _Bool fields for bit-packing into {n_bytes} byte(s)"

# ─── 17. ARRAY_RESHAPE ───────────────────────────────────────────────────────

class ArrayReshape(Transform):
    name = "ARRAY_RESHAPE"

    def apply(self, code):
        """Reshape constant initialization patterns into arrays where possible.
        For structs like constant/u_block_1_2 that set many typed constants,
        suggest array representation.
        Also: for __CPROVER_bitvector[N] arrays, add reshape pragma.
        """
        changed = False
        new_code = code

        # Detect: N consecutive assignments like su.T1=v1; su.T21=v2; ... (constants)
        const_assigns = re.findall(
            r'su\w+\.\w+\s*=\s*-?\d+\s*;', code)
        if len(const_assigns) >= 8:
            # Add pragma comment suggesting array reshape
            idx = code.rfind('void ')
            if idx != -1:
                brace = code.find('{', idx)
                if brace != -1:
                    pragma = ('\n  /* ARRAY_RESHAPE: consider replacing individual constant fields\n'
                              '     with a static const array for better memory mapping:\n'
                              '     static const short COEFF[] = { ... values ... };\n'
                              '     #pragma HLS ARRAY_RESHAPE variable=COEFF complete dim=1 */\n')
                    new_code = code[:brace+1] + pragma + code[brace+1:]
                    changed = True

        # Handle __CPROVER_bitvector arrays
        if '__CPROVER_bitvector' in code:
            new_code = re.sub(
                r'(void\s+\w+[^{]+\{)',
                r'\1\n  /* #pragma HLS ARRAY_RESHAPE variable=in complete dim=1 */',
                new_code, count=1)
            changed = True

        if not changed:
            hdr = header_comment(self.name,
                "No array reshape opportunities found",
                "N/A — no large constant initializations or bitvector arrays")
            return hdr + code, False, "No array reshape opportunities"

        hdr = header_comment(self.name,
            "Added array reshape annotations for constant fields and bitvector arrays",
            "HLS can map reshaped arrays to more efficient memory primitives")
        return hdr + new_code, True, "Added array reshape annotations"

# ─── 18. REDUCE_PARTITION_FACTOR ─────────────────────────────────────────────

class ReducePartitionFactor(Transform):
    name = "REDUCE_PARTITION_FACTOR"

    def apply(self, code):
        """Add HLS ARRAY_PARTITION pragmas with factor=1 (no partition) for arrays
        to minimize memory port count and save area.
        """
        # Find array variables in function parameters or local declarations
        arrays = re.findall(r'(\w[\w ]*(?:\[\d+\])+)\s+(\w+)', code)
        cprover_arrays = re.findall(r'(unsigned\s+__CPROVER_bitvector\[\d+\])\s+(\w+)', code)
        all_arrays = [(t.strip(), n) for t, n in arrays + cprover_arrays]

        if not all_arrays:
            hdr = header_comment(self.name,
                "No array variables found for partition control",
                "N/A — no arrays present")
            return hdr + code, False, "No arrays to partition"

        pragmas = '\n'.join(
            f'  /* #pragma HLS ARRAY_PARTITION variable={name} factor=1 dim=1 */'
            f'  /* Reduces to single-port memory, saves area at cost of throughput */'
            for _, name in all_arrays[:8])  # limit to 8

        # Insert after first { in each non-main function
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed = False
        for func in body_funcs:
            brace = func.find('{')
            if brace != -1 and any(n in func for _, n in all_arrays):
                func = func[:brace+1] + '\n' + pragmas + func[brace+1:]
                changed = True
            new_funcs.append(func)

        if not changed:
            hdr = header_comment(self.name,
                "Arrays found in signature but not used in function body",
                "N/A")
            return hdr + code, False, "No function-body arrays to annotate"

        hdr = header_comment(self.name,
            f"Added ARRAY_PARTITION factor=1 pragmas for {len(all_arrays)} array(s)",
            "Prevents HLS from splitting arrays; reduces memory port area")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               f"Added partition-reduction pragmas for {len(all_arrays)} array(s)"

# ─── 19. LIMIT_MEMORY_PORTS ──────────────────────────────────────────────────

class LimitMemoryPorts(Transform):
    name = "LIMIT_MEMORY_PORTS"

    def apply(self, code):
        """Add #pragma HLS RESOURCE to bind arrays/struct to single-port RAM."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed = False

        for func in body_funcs:
            # Find any array or struct state variable
            svar = get_state_var(func)
            arrays = re.findall(r'\b(\w+)\s*\[', func)
            targets = list(dict.fromkeys(([svar] if svar else []) + arrays[:4]))
            if not targets:
                new_funcs.append(func)
                continue

            pragmas = '\n'.join(
                f'  /* #pragma HLS RESOURCE variable={t} core=RAM_1P latency=1 */'
                for t in targets[:4])
            brace = func.find('{')
            if brace != -1:
                func = func[:brace+1] + '\n' + pragmas + func[brace+1:]
                changed = True
            new_funcs.append(func)

        if not changed:
            hdr = header_comment(self.name,
                "No memory variables found for port limitation",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No memory variables to annotate"

        hdr = header_comment(self.name,
            "Added RAM_1P binding pragmas to limit memory ports to 1",
            "Single-port RAM halves memory area vs dual-port at cost of one-port throughput")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Added single-port memory binding pragmas"

# ─── 20. REDUCE_BUFFER_DEPTH ─────────────────────────────────────────────────

class ReduceBufferDepth(Transform):
    name = "REDUCE_BUFFER_DEPTH"

    def apply(self, code):
        """Reduce implicit register/buffer depth by:
        1. Removing intermediate _old variables that are not needed
        2. Adding FIFO depth pragmas
        3. Marking _old vars as pipeline registers
        """
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            old_vars = re.findall(r'(\w+)_old\b', func)
            old_vars = list(dict.fromkeys(old_vars))
            if not old_vars:
                new_funcs.append(func)
                continue

            # Find _old vars that are declared but never read in RHS
            func_body_after_decls = func[func.find('{')+1:] if '{' in func else func
            unused_olds = []
            for v in old_vars:
                old_var = v + '_old'
                # Count uses in RHS (after assignment line)
                assign_m = re.search(re.escape(old_var) + r'\s*=\s*[^;]+;', func)
                if assign_m:
                    body_after = func[assign_m.end():]
                    uses_after = len(re.findall(r'\b' + re.escape(old_var) + r'\b', body_after))
                    if uses_after == 0:
                        unused_olds.append(old_var)

            if unused_olds:
                lines = func.split('\n')
                new_lines = []
                for ln in lines:
                    # Comment out assignment to unused _old
                    if any(re.match(r'\s*' + re.escape(ov) + r'\s*=', ln)
                           for ov in unused_olds):
                        new_lines.append(
                            ln + '  /* REDUCE_BUFFER_DEPTH: _old value unused; depth=0 */')
                    else:
                        new_lines.append(ln)
                new_funcs.append('\n'.join(new_lines))
                changed_any = True
            else:
                # Add pipeline register annotation
                lines = func.split('\n')
                new_lines = []
                for ln in lines:
                    new_lines.append(ln)
                    if any(re.match(r'\s*' + re.escape(v + '_old') + r'\s*=', ln)
                           for v in old_vars[:2]):
                        new_lines.append(
                            re.match(r'^(\s*)', ln).group(1) +
                            '/* #pragma HLS PIPELINE II=1 */'
                            '  /* _old = pipeline register; depth reducible to 1 */')
                new_funcs.append('\n'.join(new_lines))
                changed_any = True

        if not changed_any:
            hdr = header_comment(self.name,
                "No reducible buffer/register depth patterns found",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No reducible buffer depths found"

        hdr = header_comment(self.name,
            "Annotated unused _old buffers and pipeline register depth",
            "Removes unnecessary state registers and guides HLS to minimal buffer depth")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Annotated and reduced unnecessary buffer/register depths"

# ─── 21. LOGIC_MINIMIZATION ───────────────────────────────────────────────────

class LogicMinimization(Transform):
    name = "LOGIC_MINIMIZATION"

    def apply(self, code):
        """Apply boolean logic simplifications:
        - De Morgan's law (when it reduces operator count)
        - Double negation removal
        - Absorption laws
        - Idempotent: x & x -> x, x | x -> x
        - Constant negation: !TRUE -> FALSE, !FALSE -> TRUE
        """
        new_code = code
        changed = False

        # Double negation: !!x -> x
        r0 = new_code
        new_code = re.sub(r'!!\s*(\w[\w.]*)', r'\1', new_code)
        new_code = re.sub(r'!!\s*\(([^)]+)\)', r'(\1)', new_code)
        changed |= new_code != r0

        # !TRUE -> FALSE, !FALSE -> TRUE (handle common C macros and literals)
        r1 = new_code
        new_code = re.sub(r'!\s*TRUE\b', 'FALSE', new_code)
        new_code = re.sub(r'!\s*FALSE\b', 'TRUE', new_code)
        new_code = re.sub(r'!\s*1\b', '0', new_code)
        new_code = re.sub(r'!\s*0\b', '1', new_code)
        changed |= new_code != r1

        # Idempotent: x & x -> x, x | x -> x (same variable both sides)
        r2 = new_code
        new_code = re.sub(r'\b(\w[\w.]*)\s*&\s*\1\b', r'\1', new_code)
        new_code = re.sub(r'\b(\w[\w.]*)\s*\|\s*\1\b', r'\1', new_code)
        changed |= new_code != r2

        # Absorption: a && (a || b) -> a
        r3 = new_code
        new_code = re.sub(
            r'\b(\w[\w.]*)\s*&&\s*\(\s*\1\s*\|\|\s*\w[\w.]*\s*\)',
            r'\1', new_code)
        # Absorption: a || (a && b) -> a
        new_code = re.sub(
            r'\b(\w[\w.]*)\s*\|\|\s*\(\s*\1\s*&&\s*\w[\w.]*\s*\)',
            r'\1', new_code)
        changed |= new_code != r3

        # De Morgan's: !(a && b) -> (!a || !b) when it reduces depth
        # Only apply when a and b are simple identifiers (reduces nesting)
        r4 = new_code
        new_code = re.sub(
            r'!\s*\(\s*(\w[\w.]*)\s*&&\s*(\w[\w.]*)\s*\)',
            r'(!\1 || !\2)', new_code)
        # De Morgan's: !(a || b) -> (!a && !b)
        new_code = re.sub(
            r'!\s*\(\s*(\w[\w.]*)\s*\|\|\s*(\w[\w.]*)\s*\)',
            r'(!\1 && !\2)', new_code)
        changed |= new_code != r4

        if not changed:
            hdr = header_comment(self.name,
                "No boolean logic simplification opportunities found",
                "N/A")
            return hdr + code, False, "No logic minimization opportunities"

        hdr = header_comment(self.name,
            "Applied boolean simplifications (double negation, absorption, idempotent, De Morgan)",
            "Reduces gate count and logic depth in synthesized hardware")
        return hdr + new_code, True, "Applied boolean logic simplifications"

# ─── 22. CONDITION_MERGE ─────────────────────────────────────────────────────

class ConditionMerge(Transform):
    name = "CONDITION_MERGE"

    def apply(self, code):
        """In if-else-if chains where multiple cases assign the SAME value
        to the same lvalue, merge those conditions with ||."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            lines = func.split('\n')
            new_lines, func_changed = self._merge_chains(lines)
            if func_changed:
                new_funcs.append('\n'.join(new_lines))
                changed_any = True
            else:
                new_funcs.append(func)

        if not changed_any:
            hdr = header_comment(self.name,
                "No if-else-if chains with duplicate assignments found",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No condition merge opportunities"

        hdr = header_comment(self.name,
            "Merged if-else-if branches that assign the same value to the same target",
            "Reduces mux inputs and comparison logic; fewer branches = smaller selector")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Merged duplicate-value branches in if-else-if chains"

    def _merge_chains(self, lines):
        """Parse if-else-if chains and merge cases with identical assignments."""
        changed = False
        # Collect chains: list of (condition, lvalue, rhs, line_index)
        chain_pat = re.compile(
            r'^(\s*)(?:else\s+)?if\s*\(\s*(.+?)\s*\)\s*'
            r'(\w[\w. >*\[\]]+)\s*=\s*(.+?)\s*;$')
        i = 0
        result = []
        while i < len(lines):
            ln = lines[i]
            m = chain_pat.match(ln)
            if not m:
                result.append(ln)
                i += 1
                continue

            # Start collecting a chain
            indent = m.group(1)
            chain = []  # (condition, lvalue, rhs, original_line)
            while i < len(lines):
                ln = lines[i]
                m2 = chain_pat.match(ln)
                if m2 and m2.group(1) == indent:
                    chain.append((m2.group(2), m2.group(3).strip(),
                                  m2.group(4).strip(), ln))
                    i += 1
                else:
                    break

            if len(chain) < 2:
                for _, _, _, orig in chain:
                    result.append(orig)
                continue

            # Group by (lvalue, rhs)
            groups = defaultdict(list)
            for cond, lv, rhs, orig in chain:
                groups[(lv, rhs)].append(cond)

            # Check if any group has >=2 conditions (worth merging)
            has_merge = any(len(conds) >= 2 for conds in groups.values())
            if not has_merge:
                for _, _, _, orig in chain:
                    result.append(orig)
                continue

            # Rebuild chain with merged conditions
            changed = True
            seen = set()
            first = True
            for cond, lv, rhs, orig in chain:
                key = (lv, rhs)
                if key in seen:
                    continue
                seen.add(key)
                merged_cond = ' || '.join(groups[key])
                if len(groups[key]) > 1:
                    merged_cond = '(' + merged_cond + ')'
                prefix = f"{indent}if" if first else f"{indent}else if"
                result.append(f"{prefix}({merged_cond}) {lv} = {rhs};")
                first = False

        return result, changed

# ─── 23. OPERATOR_TIME_MULTIPLEX ─────────────────────────────────────────────

class OperatorTimeMultiplex(Transform):
    name = "OPERATOR_TIME_MULTIPLEX"

    def apply(self, code):
        """Find identical operations in mutually exclusive if/else branches
        and hoist them to share one operator:
          if(cond) s.x = a + b; else s.x = c + d;
          -> s.x = (cond ? a : c) + (cond ? b : d);
        """
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        op_pat = re.compile(
            r'^(\s*)if\s*\(\s*(.+?)\s*\)\s*'
            r'(\w[\w. >*\[\]]+)\s*=\s*(.+?)\s*([+\-*&|^])\s*(.+?)\s*;\s*\n'
            r'\s*else\s+'
            r'\3\s*=\s*(.+?)\s*\5\s*(.+?)\s*;',
            re.MULTILINE)

        for func in body_funcs:
            new_func = func
            func_changed = False
            for m in reversed(list(op_pat.finditer(func))):
                indent = m.group(1)
                cond   = m.group(2)
                lv     = m.group(3)
                a      = m.group(4).strip()
                op     = m.group(5)
                b      = m.group(6).strip()
                c      = m.group(7).strip()
                d      = m.group(8).strip()
                replacement = (
                    f"{indent}/* OPERATOR_TIME_MULTIPLEX: shared '{op}' operator */\n"
                    f"{indent}{lv} = ({cond} ? {a} : {c}) {op} ({cond} ? {b} : {d});")
                new_func = new_func[:m.start()] + replacement + new_func[m.end():]
                func_changed = True

            if func_changed:
                new_funcs.append(new_func)
                changed_any = True
            else:
                new_funcs.append(func)

        if not changed_any:
            hdr = header_comment(self.name,
                "No identical operators in mutually exclusive branches found",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No operator time-multiplex opportunities"

        hdr = header_comment(self.name,
            "Hoisted identical operators from if/else branches to share hardware",
            "One operator instance with muxed inputs instead of two separate operators")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Shared operators across mutually exclusive branches"

# ─── 24. REGISTER_LIFETIME_SHARE ─────────────────────────────────────────────

class RegisterLifetimeShare(Transform):
    name = "REGISTER_LIFETIME_SHARE"

    def apply(self, code):
        """Look for struct fields written in mutually exclusive branches.
        Annotate fields with non-overlapping lifetimes for register sharing."""
        preamble, body_funcs, main_block = split_file(code)
        sv = get_state_var(code)
        if not sv:
            hdr = header_comment(self.name,
                "No state struct variable found",
                "N/A")
            return hdr + code, False, "No state struct found"

        # Collect struct fields written in if vs else branches
        branch_pat = re.compile(
            r'if\s*\(.+?\)\s*\{([^}]*)\}\s*else\s*\{([^}]*)\}',
            re.DOTALL)

        field_write_pat = re.compile(
            r'\b' + re.escape(sv) + r'\.(\w+)\s*=')

        annotations = []
        for func in body_funcs:
            for bm in branch_pat.finditer(func):
                if_body = bm.group(1)
                else_body = bm.group(2)
                if_fields = set(field_write_pat.findall(if_body))
                else_fields = set(field_write_pat.findall(else_body))
                # Fields written only in if, not in else (and vice versa)
                only_if = if_fields - else_fields
                only_else = else_fields - if_fields
                if only_if and only_else:
                    for f_if in sorted(only_if):
                        for f_else in sorted(only_else):
                            pair = (f_if, f_else)
                            if pair not in annotations:
                                annotations.append(pair)

        if not annotations:
            hdr = header_comment(self.name,
                "All struct fields are written in both branches or no mutual exclusion found",
                "N/A — no register sharing opportunity")
            return hdr + code, False, "No register lifetime sharing opportunities"

        # Add annotation comments after the struct declaration
        comment_lines = []
        for f_a, f_b in annotations[:10]:  # limit to 10 annotations
            comment_lines.append(
                f"/* REGISTER_SHARE: {f_a} and {f_b} have non-overlapping lifetimes */")
        annotation_block = '\n'.join(comment_lines)

        # Insert after struct declaration
        struct_pat = re.compile(
            r'(struct\s+state_elements_\w+\s+' + re.escape(sv) + r'\s*;)')
        m = struct_pat.search(code)
        if m:
            new_code = code[:m.end()] + '\n' + annotation_block + code[m.end():]
        else:
            new_code = annotation_block + '\n' + code

        hdr = header_comment(self.name,
            f"Identified {len(annotations)} field pairs with non-overlapping lifetimes",
            "Fields can share physical registers, reducing register file area")
        return hdr + new_code, True, \
               f"Annotated {len(annotations)} register sharing opportunities"

# ─── 25. MEMORY_PROMOTION ────────────────────────────────────────────────────

class MemoryPromotion(Transform):
    name = "MEMORY_PROMOTION"

    def apply(self, code):
        """Detect array declarations in structs with size >= 16 elements.
        Add pragma comment for BRAM binding."""
        # Find struct definitions with array fields
        struct_pat = re.compile(
            r'(struct\s+state_elements_\w+\s*\{[^}]*\})\s*;',
            re.DOTALL)
        array_pat = re.compile(
            r'(\w[\w\s*]*)\s+(\w+)\s*\[\s*(\d+)\s*\]')

        new_code = code
        changed = False

        for sm in struct_pat.finditer(code):
            struct_body = sm.group(1)
            arrays = array_pat.findall(struct_body)
            pragmas = []
            for typ, arr_name, size_str in arrays:
                size = int(size_str)
                if size >= 16:
                    pragmas.append(
                        f"/* #pragma HLS RESOURCE variable={arr_name} core=RAM_1P_BRAM */")
            if pragmas:
                insert_pos = sm.end() + 1  # after the ';'
                pragma_block = '\n'.join(pragmas)
                # Find the position after the struct variable declaration
                # (struct def + variable decl)
                after_struct = code[sm.end():]
                var_m = re.match(r'\s*;', after_struct)
                if var_m:
                    insert_pos = sm.end() + var_m.end()
                new_code = (new_code[:insert_pos] + '\n' + pragma_block +
                            new_code[insert_pos:])
                changed = True

        if not changed:
            # Also check for standalone array declarations
            standalone_arr = re.compile(
                r'(\w[\w\s*]*)\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;')
            for am in standalone_arr.finditer(code):
                size = int(am.group(3))
                if size >= 16:
                    pragma = f"\n/* #pragma HLS RESOURCE variable={am.group(2)} core=RAM_1P_BRAM */"
                    new_code = (new_code[:am.end()] + pragma +
                                new_code[am.end():])
                    changed = True
                    break  # re-do positions after first insert

        if not changed:
            hdr = header_comment(self.name,
                "No arrays with >= 16 elements found for BRAM promotion",
                "N/A")
            return hdr + code, False, "No memory promotion opportunities"

        hdr = header_comment(self.name,
            "Added BRAM resource pragmas for large arrays in state struct",
            "Maps arrays to block RAM instead of distributed LUT RAM, saving LUT area")
        return hdr + new_code, True, "Added BRAM promotion pragmas for large arrays"

# ─── 26. FSM_REENCODE ────────────────────────────────────────────────────────

class FsmReencode(Transform):
    name = "FSM_REENCODE"

    def apply(self, code):
        """Detect state-machine-like patterns with sparse integer constants
        in if-else chains. Re-encode to contiguous values."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False
        defines = []

        # Find if-else-if chains comparing same variable to integer constants
        chain_pat = re.compile(
            r'if\s*\(\s*\(unsigned int\)\s*(\w[\w.]*)\s*==\s*(\d+)\s*\)')

        for func in body_funcs:
            lines = func.split('\n')
            # Group by compared variable
            var_values = defaultdict(list)  # var -> [(value, line_idx)]
            for i, ln in enumerate(lines):
                m = chain_pat.search(ln)
                if m:
                    var = m.group(1)
                    val = int(m.group(2))
                    var_values[var].append((val, i))

            func_changed = False
            new_lines = list(lines)
            offset = 0
            for var, val_lines in var_values.items():
                if len(val_lines) < 3:
                    continue
                values = [v for v, _ in val_lines]
                sorted_vals = sorted(set(values))

                # Check if already contiguous (0, 1, 2, ..., N)
                if sorted_vals == list(range(len(sorted_vals))):
                    continue

                # Re-encode: create mapping old -> new (contiguous)
                mapping = {old: new for new, old in enumerate(sorted_vals)}
                var_defines = []
                for old_val, new_val in mapping.items():
                    var_defines.append(
                        f"#define STATE_{old_val} {new_val}  "
                        f"/* FSM_REENCODE: was {old_val} */")

                defines.extend(var_defines)

                # Replace constants in the if-else chain lines
                for old_val, line_idx in val_lines:
                    adj_idx = line_idx + offset
                    if adj_idx < len(new_lines):
                        new_lines[adj_idx] = new_lines[adj_idx].replace(
                            f'== {old_val})',
                            f'== {mapping[old_val]})  /* was {old_val} */', 1)
                        func_changed = True

                # Add mapping comment before first use
                first_idx = val_lines[0][1] + offset
                comment = (f"  /* FSM_REENCODE: {var} re-encoded from "
                           f"{sorted_vals} to {list(range(len(sorted_vals)))} */")
                new_lines.insert(first_idx, comment)
                offset += 1

            if func_changed:
                new_funcs.append('\n'.join(new_lines))
                changed_any = True
            else:
                new_funcs.append(func)

        if not changed_any:
            hdr = header_comment(self.name,
                "No sparse FSM state encodings found (values already contiguous or too few states)",
                "N/A")
            return hdr + code, False, "No FSM re-encoding opportunities"

        # Insert #defines at top of preamble
        define_block = '\n'.join(defines) + '\n'
        hdr = header_comment(self.name,
            "Re-encoded sparse FSM state constants to contiguous values",
            "Contiguous encoding uses fewer comparator bits and simpler decoding logic")
        return hdr + define_block + preamble + "\n" + '\n'.join(new_funcs+[main_block]), True, \
               "Re-encoded sparse FSM states to contiguous values"

# ─── 27. RESET_SIMPLIFY ──────────────────────────────────────────────────────

class ResetSimplify(Transform):
    name = "RESET_SIMPLIFY"

    def apply(self, code):
        """In if(rst) { s.x = 0; s.y = 0; ... } blocks, check if reset values
        are default (0 for integers). Replace with memset if all are zero."""
        sv = get_state_var(code)
        if not sv:
            hdr = header_comment(self.name,
                "No state struct variable found",
                "N/A")
            return hdr + code, False, "No state struct found"

        # Find reset blocks: if(rst) { ... } or if(reset) { ... }
        reset_pat = re.compile(
            r'([ \t]*)(if\s*\(\s*(\w*rst\w*)\s*\)\s*\{)([^}]*)\}',
            re.DOTALL | re.IGNORECASE)

        new_code = code
        changed = False

        for m in reversed(list(reset_pat.finditer(code))):
            indent = m.group(1)
            if_header = m.group(2)
            reset_body = m.group(4)

            # Parse assignments in reset block
            assign_pat = re.compile(
                r'\b' + re.escape(sv) + r'\.(\w+)\s*=\s*(.+?)\s*;')
            assignments = assign_pat.findall(reset_body)

            if not assignments:
                continue

            # Check if ALL assignments are to 0
            all_zero = all(
                re.fullmatch(r'0|0u|0x0+|0U|0x0+u', val.strip())
                for _, val in assignments)

            if all_zero:
                # Replace entire body with memset (ensure string.h is included)
                if '#include <string.h>' not in new_code:
                    new_code = new_code.replace(
                        '#include <stdio.h>',
                        '#include <stdio.h>\n#include <string.h>', 1)
                new_body = (
                    f"\n{indent}  /* RESET_SIMPLIFY: memset-equivalent reset */\n"
                    f"{indent}  memset(&{sv}, 0, sizeof({sv}));\n"
                    f"{indent}")
                replacement = f"{indent}{if_header}{new_body}}}"
                new_code = new_code[:m.start()] + replacement + new_code[m.end():]
                changed = True
            else:
                # Remove assignments where value is 0 (default for integer fields)
                new_body_lines = []
                removed = 0
                for line in reset_body.split('\n'):
                    line_stripped = line.strip()
                    zero_assign = re.match(
                        r'\b' + re.escape(sv) + r'\.(\w+)\s*=\s*(?:0|0u|0x0+|0U)\s*;',
                        line_stripped)
                    if zero_assign:
                        # Skip zero assignments (default init handles them)
                        removed += 1
                        continue
                    new_body_lines.append(line)

                if removed > 0:
                    new_body = '\n'.join(new_body_lines)
                    # Add comment about removed assignments
                    comment = (f"\n{indent}  "
                               f"/* RESET_SIMPLIFY: removed {removed} zero-init "
                               f"assignments (default for integer fields) */")
                    replacement = (f"{indent}{if_header}{comment}{new_body}}}")
                    new_code = new_code[:m.start()] + replacement + new_code[m.end():]
                    changed = True

        if not changed:
            hdr = header_comment(self.name,
                "No simplifiable reset blocks found (no if(rst) or non-zero resets)",
                "N/A")
            return hdr + code, False, "No reset simplification opportunities"

        hdr = header_comment(self.name,
            "Simplified reset logic by removing redundant zero-initialization assignments",
            "Fewer reset assignments reduce mux inputs and reset-path area")
        return hdr + new_code, True, "Simplified reset blocks by removing default-value assignments"

# ─── 28. COPY_PROPAGATION ────────────────────────────────────────────────────

class CopyPropagation(Transform):
    name = "COPY_PROPAGATION"

    def apply(self, code):
        """Find simple copy assignments X = Y where X is only used as a copy.
        Key target: field_old = s.field — propagate s.field in place of field_old."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        for func in body_funcs:
            old_map = get_old_var_map(func)
            if not old_map:
                new_funcs.append(func)
                continue

            sv = get_state_var(code)
            new_func = func
            func_changed = False

            for field, old_var in old_map.items():
                # Check that s.field is not written between _old assignment and uses
                # Find the _old assignment
                assign_m = re.search(
                    re.escape(old_var) + r'\s*=\s*\w+\.' + re.escape(field) + r'\s*;',
                    new_func)
                if not assign_m:
                    continue

                body_after = new_func[assign_m.end():]

                # Check if s.field is written before old_var is used
                field_write = re.search(
                    r'\b' + re.escape(sv) + r'\.' + re.escape(field) + r'\s*=',
                    body_after) if sv else None
                old_use = re.search(r'\b' + re.escape(old_var) + r'\b', body_after)

                if field_write and old_use:
                    # If field is written before old_var is used, cannot propagate
                    if field_write.start() < old_use.start():
                        continue

                # Safe to propagate: replace all uses of old_var with s.field
                source_expr = f"{sv}.{field}" if sv else field
                uses_in_body = len(re.findall(
                    r'\b' + re.escape(old_var) + r'\b', body_after))
                if uses_in_body > 0:
                    # Replace uses of old_var (but not the assignment itself)
                    before_assign = new_func[:assign_m.start()]
                    after_assign = new_func[assign_m.end():]
                    after_assign = re.sub(
                        r'\b' + re.escape(old_var) + r'\b',
                        source_expr, after_assign)
                    # Comment out the original assignment
                    assign_line = assign_m.group(0)
                    new_func = (before_assign +
                                f"/* COPY_PROPAGATION: {assign_line} */" +
                                after_assign)
                    func_changed = True

            if func_changed:
                new_funcs.append(new_func)
                changed_any = True
            else:
                new_funcs.append(func)

        if not changed_any:
            hdr = header_comment(self.name,
                "No propagatable copy assignments (_old pattern) found",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No copy propagation opportunities"

        hdr = header_comment(self.name,
            "Propagated _old copy variables back to original struct field references",
            "Eliminates copy registers; HLS reads source directly instead of buffered copy")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Propagated _old copies back to original field references"

# ─── 29. ALGEBRAIC_SIMPLIFY ──────────────────────────────────────────────────

class AlgebraicSimplify(Transform):
    name = "ALGEBRAIC_SIMPLIFY"

    def apply(self, code):
        """Apply algebraic identities:
        x+0->x, x*1->x, x*0->0, x&0xFFFFFFFF->x, x|0->x, x^0->x,
        x<<0->x, x>>0->x, !!x->x, ((x))->(x).
        """
        new_code = code
        changed = False

        # x + 0 -> x, 0 + x -> x
        r0 = new_code
        new_code = re.sub(r'\b(\w[\w.]*)\s*\+\s*0\b', r'\1', new_code)
        new_code = re.sub(r'\b0\s*\+\s*(\w[\w.]*)\b', r'\1', new_code)
        changed |= new_code != r0

        # x - 0 -> x
        r1 = new_code
        new_code = re.sub(r'\b(\w[\w.]*)\s*-\s*0\b', r'\1', new_code)
        changed |= new_code != r1

        # x * 1 -> x, 1 * x -> x
        r2 = new_code
        new_code = re.sub(r'\b(\w[\w.]*)\s*\*\s*1\b', r'\1', new_code)
        new_code = re.sub(r'\b1\s*\*\s*(\w[\w.]*)\b', r'\1', new_code)
        changed |= new_code != r2

        # x * 0 -> 0, 0 * x -> 0
        r3 = new_code
        new_code = re.sub(r'\b\w[\w.]*\s*\*\s*0\b', '0', new_code)
        new_code = re.sub(r'\b0\s*\*\s*\w[\w.]*\b', '0', new_code)
        changed |= new_code != r3

        # x & 0xFFFFFFFF -> x (32-bit all-ones mask is no-op)
        r4 = new_code
        new_code = re.sub(r'\b(\w[\w.]*)\s*&\s*0xFFFFFFFF\b', r'\1', new_code)
        new_code = re.sub(r'\b(\w[\w.]*)\s*&\s*4294967295\b', r'\1', new_code)
        changed |= new_code != r4

        # x | 0 -> x, 0 | x -> x
        r5 = new_code
        new_code = re.sub(r'\b(\w[\w.]*)\s*\|\s*0\b', r'\1', new_code)
        new_code = re.sub(r'\b0\s*\|\s*(\w[\w.]*)\b', r'\1', new_code)
        changed |= new_code != r5

        # x ^ 0 -> x, 0 ^ x -> x
        r6 = new_code
        new_code = re.sub(r'\b(\w[\w.]*)\s*\^\s*0\b', r'\1', new_code)
        new_code = re.sub(r'\b0\s*\^\s*(\w[\w.]*)\b', r'\1', new_code)
        changed |= new_code != r6

        # x << 0 -> x, x >> 0 -> x
        r7 = new_code
        new_code = re.sub(r'\b(\w[\w.]*)\s*<<\s*0\b', r'\1', new_code)
        new_code = re.sub(r'\b(\w[\w.]*)\s*>>\s*0\b', r'\1', new_code)
        changed |= new_code != r7

        # !!x -> x
        r8 = new_code
        new_code = re.sub(r'!!\s*(\w[\w.]*)', r'\1', new_code)
        new_code = re.sub(r'!!\s*\(([^)]+)\)', r'(\1)', new_code)
        changed |= new_code != r8

        # ((x)) -> (x) — remove double parentheses
        r9 = new_code
        # Iteratively remove double parens
        for _ in range(3):
            new_code = re.sub(r'\(\(([^()]*)\)\)', r'(\1)', new_code)
        changed |= new_code != r9

        if not changed:
            hdr = header_comment(self.name,
                "No algebraic identity simplifications found",
                "N/A")
            return hdr + code, False, "No algebraic simplification opportunities"

        hdr = header_comment(self.name,
            "Applied algebraic identities (x+0, x*1, x*0, mask no-ops, double negation, double parens)",
            "Eliminates no-op operations; reduces gate count in synthesized circuit")
        return hdr + new_code, True, "Applied algebraic identity simplifications"

# ─── 30. BOOLEAN_TO_ARITHMETIC ───────────────────────────────────────────────

class BooleanToArithmetic(Transform):
    name = "BOOLEAN_TO_ARITHMETIC"

    def apply(self, code):
        """Convert a && b && c -> a & b & c and a || b || c -> a | b | c
        when operands are boolean (comparisons, _Bool variables)."""
        new_code = code
        changed = False

        # Find conditions with && or || where all operands are comparisons
        # Pattern: (expr1 op1 val1) && (expr2 op2 val2) && ...
        # Replace && with & and || with |

        # Match comparison operands: x == N, x != N, x < N, x > N, etc.
        comp_operand = r'\w[\w.]*\s*(?:==|!=|<=|>=|<|>)\s*\w[\w.]*'

        # a && b where both are comparisons -> a & b
        r0 = new_code
        pat_and = re.compile(
            r'(\(' + comp_operand + r'\))\s*&&\s*(\(' + comp_operand + r'\))')
        new_code = pat_and.sub(r'\1 & \2', new_code)
        changed |= new_code != r0

        # Also handle without parens around individual comparisons
        r1 = new_code
        pat_and2 = re.compile(
            r'(' + comp_operand + r')\s*&&\s*(' + comp_operand + r')')
        new_code = pat_and2.sub(r'(\1) & (\2)', new_code)
        changed |= new_code != r1

        # a || b where both are comparisons -> a | b
        r2 = new_code
        pat_or = re.compile(
            r'(\(' + comp_operand + r'\))\s*\|\|\s*(\(' + comp_operand + r'\))')
        new_code = pat_or.sub(r'\1 | \2', new_code)
        changed |= new_code != r2

        r3 = new_code
        pat_or2 = re.compile(
            r'(' + comp_operand + r')\s*\|\|\s*(' + comp_operand + r')')
        new_code = pat_or2.sub(r'(\1) | (\2)', new_code)
        changed |= new_code != r3

        # _Bool variables: var && var2 -> var & var2 when vars are _Bool
        r4 = new_code
        bool_vars = set(re.findall(r'_Bool\s+(\w+)', code))
        for bv in bool_vars:
            # bv && other_bool -> bv & other_bool
            for bv2 in bool_vars:
                if bv != bv2:
                    new_code = re.sub(
                        r'\b' + re.escape(bv) + r'\s*&&\s*' + re.escape(bv2) + r'\b',
                        f'{bv} & {bv2}', new_code)
                    new_code = re.sub(
                        r'\b' + re.escape(bv) + r'\s*\|\|\s*' + re.escape(bv2) + r'\b',
                        f'{bv} | {bv2}', new_code)
        changed |= new_code != r4

        if not changed:
            hdr = header_comment(self.name,
                "No boolean-to-arithmetic conversion opportunities found",
                "N/A")
            return hdr + code, False, "No boolean-to-arithmetic opportunities"

        hdr = header_comment(self.name,
            "Converted boolean && / || to bitwise & / | for comparison operands",
            "Bitwise ops avoid short-circuit branching; enables parallel evaluation in hardware")
        return hdr + new_code, True, "Converted boolean operators to bitwise for comparisons"

# ─── 31. ENCODE_ONEHOT_TO_BINARY ─────────────────────────────────────────────

class EncodeOnehotToBinary(Transform):
    name = "ENCODE_ONEHOT_TO_BINARY"

    def apply(self, code):
        """Detect one-hot bit-test selection patterns:
          if(x & 1) ... else if(x & 2) ... else if(x & 4) ...
        Convert to binary index + array lookup."""
        preamble, body_funcs, main_block = split_file(code)
        new_funcs = []
        changed_any = False

        # Pattern: if(var & POWER_OF_2) assignment
        onehot_pat = re.compile(
            r'^(\s*)(?:else\s+)?if\s*\(\s*(\w[\w.]*)\s*&\s*(\d+)\s*\)')

        for func in body_funcs:
            lines = func.split('\n')
            i = 0
            new_lines = []
            func_changed = False

            while i < len(lines):
                ln = lines[i]
                m = onehot_pat.match(ln)
                if not m:
                    new_lines.append(ln)
                    i += 1
                    continue

                # Try to collect a chain of one-hot tests on same variable
                indent = m.group(1)
                var = m.group(2)
                chain = []  # (power_of_2, line_text, assignment_rhs)

                j = i
                while j < len(lines):
                    m2 = onehot_pat.match(lines[j])
                    if m2 and m2.group(2) == var:
                        bit_val = int(m2.group(3))
                        # Check if power of 2
                        if bit_val > 0 and (bit_val & (bit_val - 1)) == 0:
                            # Extract the assignment after the condition
                            assign_m = re.search(
                                r'\)\s*(\w[\w. >*\[\]]+)\s*=\s*(.+?)\s*;',
                                lines[j])
                            if assign_m:
                                chain.append((bit_val, lines[j],
                                              assign_m.group(1).strip(),
                                              assign_m.group(2).strip()))
                                j += 1
                                continue
                    break

                if len(chain) >= 3:
                    # All must assign to same lvalue
                    lvalues = set(c[2] for c in chain)
                    if len(lvalues) == 1:
                        lv = chain[0][2]
                        # Build lookup array
                        # Map bit position to value
                        max_bit = max(int(math.log2(c[0])) for c in chain)
                        lookup = ['0'] * (max_bit + 1)
                        for bit_val, _, _, rhs in chain:
                            bit_pos = int(math.log2(bit_val))
                            lookup[bit_pos] = rhs

                        # Generate binary encoder + lookup
                        arr_name = f"_onehot_lut_{abs(hash(var)) % 10000}"
                        new_lines.append(
                            f"{indent}/* ENCODE_ONEHOT_TO_BINARY: "
                            f"converted {len(chain)} one-hot tests to lookup */")
                        new_lines.append(
                            f"{indent}{{ static const __typeof__({lv}) "
                            f"{arr_name}[] = {{{', '.join(lookup)}}};")
                        # Binary encode: find bit position
                        new_lines.append(
                            f"{indent}  unsigned _bit_idx = 0;")
                        new_lines.append(
                            f"{indent}  {{ unsigned _tmp = (unsigned){var};")
                        new_lines.append(
                            f"{indent}    while(_tmp >>= 1) _bit_idx++;")
                        new_lines.append(
                            f"{indent}  }}")
                        new_lines.append(
                            f"{indent}  if(_bit_idx <= {max_bit}) "
                            f"{lv} = {arr_name}[_bit_idx];")
                        new_lines.append(f"{indent}}}")
                        func_changed = True
                        i = j
                        continue

                # Not a valid chain, keep original lines
                new_lines.append(ln)
                i += 1

            if func_changed:
                new_funcs.append('\n'.join(new_lines))
                changed_any = True
            else:
                new_funcs.append(func)

        if not changed_any:
            hdr = header_comment(self.name,
                "No one-hot bit-test selection patterns found",
                "N/A")
            return preamble + "\n" + hdr + '\n'.join(body_funcs+[main_block]), False, \
                   "No one-hot to binary encoding opportunities"

        hdr = header_comment(self.name,
            "Converted one-hot bit-test if-else chains to binary index + array lookup",
            "Replaces N comparators with a priority encoder + ROM; area scales O(log N) vs O(N)")
        return preamble + "\n" + hdr + '\n'.join(new_funcs+[main_block]), True, \
               "Converted one-hot selection to binary-indexed lookup"

# ─── Registry ─────────────────────────────────────────────────────────────────

TRANSFORM_REGISTRY = {
    "RESOURCE_SHARE":              ResourceShare(),
    "RESOURCE_BIND_SMALL":         ResourceBindSmall(),
    "REDUCE_UNROLL":               ReduceUnroll(),
    "SERIALIZE_PARALLELISM":       SerializeParallelism(),
    "PIPELINE_RELAX":              PipelineRelax(),
    "LOOP_FUSION":                 LoopFusion(),
    "FUNCTION_OUTLINE_REUSE":      FunctionOutlineReuse(),
    "BITWIDTH_SHRINK":             BitwidthShrink(),
    "TYPE_NARROWING_PROPAGATION":  TypeNarrowingPropagation(),
    "CONST_PROP":                  ConstProp(),
    "DEAD_CODE_ELIM":              DeadCodeElim(),
    "COMMON_SUBEXPR_EXTRACT":      CommonSubexprExtractArea(),
    "STRENGTH_REDUCTION":          StrengthReduction(),
    "SHIFT_ADD_REWRITE":           ShiftAddRewrite(),
    "TABLE_LOOKUP_REWRITE":        TableLookupRewrite(),
    "ARRAY_PACK":                  ArrayPack(),
    "ARRAY_RESHAPE":               ArrayReshape(),
    "REDUCE_PARTITION_FACTOR":     ReducePartitionFactor(),
    "LIMIT_MEMORY_PORTS":          LimitMemoryPorts(),
    "REDUCE_BUFFER_DEPTH":         ReduceBufferDepth(),
    "LOGIC_MINIMIZATION":          LogicMinimization(),
    "CONDITION_MERGE":             ConditionMerge(),
    "OPERATOR_TIME_MULTIPLEX":     OperatorTimeMultiplex(),
    "REGISTER_LIFETIME_SHARE":     RegisterLifetimeShare(),
    "MEMORY_PROMOTION":            MemoryPromotion(),
    "FSM_REENCODE":                FsmReencode(),
    "RESET_SIMPLIFY":              ResetSimplify(),
    "COPY_PROPAGATION":            CopyPropagation(),
    "ALGEBRAIC_SIMPLIFY":          AlgebraicSimplify(),
    "BOOLEAN_TO_ARITHMETIC":       BooleanToArithmetic(),
    "ENCODE_ONEHOT_TO_BINARY":     EncodeOnehotToBinary(),
}

# ─── Runner ───────────────────────────────────────────────────────────────────

def output_path(c_file, transform):
    rel = c_file.relative_to(BENCHMARK_DIR)
    out_dir = OUTPUT_DIR / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{rel.stem}_{transform}.c"

def summary_path(c_file):
    rel = c_file.relative_to(BENCHMARK_DIR)
    return OUTPUT_DIR / rel.parent / f"{rel.stem}_transforms.json"

def process_file(c_file, transforms, skip_existing=True):
    source = c_file.read_text()
    results = {}
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

        print(f"  [run ] {tname} ...", end=" ", flush=True)
        try:
            transformer = TRANSFORM_REGISTRY[tname]
            new_code, applied, summary = transformer.apply(source)
            final = new_code.rstrip() + meta_line(applied, summary)
            out_file.write_text(final)
            results[tname] = {
                "output_file": str(out_file.relative_to(OUTPUT_DIR)),
                "applied": applied,
                "summary": summary,
            }
            print(f"done [{'APPLIED' if applied else 'NO-OP'}]")
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

def _llm_output_path(c_file, transform):
    rel = c_file.relative_to(BENCHMARK_DIR)
    out_dir = LLM_OUTPUT_DIR / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{rel.stem}_{transform}.c"

def _llm_summary_path(c_file):
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
                               *, optimization_goal="area",
                               skip_existing=True, workers=8):
    """Process all (file x transform) pairs in parallel. Returns grand dict."""
    from llm_transform import tprint

    # Build work list
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
        from llm_transform import (create_client, AREA_TRANSFORM_DESCRIPTIONS,
                                    init_rate_limiter)
        import llm_transform
        if args.llm_model:
            llm_transform.MODEL = args.llm_model
        client = create_client()
        init_rate_limiter(rpm=args.rpm, max_concurrent=args.workers)
        grand = process_files_llm_parallel(
            client, c_files, transforms, AREA_TRANSFORM_DESCRIPTIONS,
            optimization_goal="area",
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
