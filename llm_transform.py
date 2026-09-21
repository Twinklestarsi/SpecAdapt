#!/usr/bin/env python3
"""
Shared LLM-based transform module for apply_transforms.py and apply_transforms_area.py.

Provides:
- Per-transform descriptions for all 24 timing + 31 area transforms
- System/user prompt templates (HLS-compatible, no #pragma)
- OpenAI-compatible API client wrapper with retry/rate-limit handling
- Normalized code diff detection

Configuration is read from .env in the project directory (via python-dotenv),
with fallback to process environment variables.  Expected keys:
  OPENAI_API_KEY   — API key (required)
  OPENAI_BASE_URL  — Base URL for the OpenAI-compatible endpoint (required)
  LLM_MODEL        — Model name override (optional, default: claude-sonnet-4-6)
"""

import json
import os
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# ─── Load .env from the project directory ─────────────────────────────────────

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH, override=False)  # process env vars take precedence

# ─── Model ────────────────────────────────────────────────────────────────────

MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")

# ─── Transform descriptions ──────────────────────────────────────────────────

TIMING_TRANSFORM_DESCRIPTIONS = {
    "BALANCE_TREE": (
        "Balance expression trees to reduce critical-path depth. "
        "Convert linear/skewed addition or logic chains into balanced binary trees "
        "so the longest dependency chain is O(log N) instead of O(N). "
        "Example: a+b+c+d → (a+b)+(c+d)."
    ),
    "REASSOCIATE_ARITHMETIC": (
        "Reorder and regroup arithmetic operations using associativity and commutativity "
        "to expose constant folding, reduce critical-path length, or improve CSE opportunities. "
        "Example: (x + 3) + 5 → x + 8; or group loop-invariant sub-expressions together."
    ),
    "BREAK_CHAIN": (
        "Break long sequential dependency chains by introducing intermediate named variables "
        "that can be computed in parallel. Identify statements where each result feeds the next "
        "and restructure to expose independent sub-computations."
    ),
    "SPLIT_OP": (
        "Split complex or wide operations into smaller, simpler constituent operations. "
        "E.g., split a wide multi-operand expression into per-field or per-bit operations "
        "to reduce individual operator complexity."
    ),
    "INLINE_CRITICAL_FUNCTION": (
        "Inline small helper functions (especially those on the critical path) directly at "
        "call sites to eliminate function-call overhead and expose further optimizations "
        "like constant propagation across the call boundary."
    ),
    "OUTLINE_LONG_COMPUTE": (
        "Extract long sequences of computation that appear inside a single function into "
        "a new dedicated helper function. This aids readability, may enable independent "
        "scheduling by HLS tools, and can expose pipeline opportunities."
    ),
    "LOOP_FISSION": (
        "Split a loop body that performs multiple independent operations into two or more "
        "separate loops over the same range, each doing one operation. This improves "
        "cache locality and enables independent pipelining of each loop."
    ),
    "LOOP_INTERCHANGE": (
        "Swap the nesting order of nested loops to improve data locality or to move the "
        "loop with the highest trip count innermost, enabling better vectorization and "
        "reducing loop-control overhead in HLS."
    ),
    "CONTROL_FLATTEN": (
        "Flatten deeply nested if-else or switch chains into a single-level predicated "
        "structure. Convert cascaded if-else-if ladders to flat parallel conditions where "
        "possible, reducing control-flow depth and mux depth in RTL."
    ),
    "IF_CONVERSION": (
        "Convert if-else statements to conditional (ternary) expressions or select "
        "operations, eliminating branches and enabling the HLS tool to synthesize "
        "parallel datapaths with a final mux instead of sequential branches."
    ),
    "SPECULATIVE_COMPUTE": (
        "Compute both the true-branch and false-branch values speculatively (before the "
        "condition is known), then select the correct result. This removes the condition "
        "from the critical path and exposes parallelism between the two computations."
    ),
    "COMMON_SUBEXPR_EXTRACT": (
        "Identify repeated sub-expressions that are computed more than once and extract "
        "them into a single named variable computed once. This reduces area by sharing "
        "logic and may shorten the critical path."
    ),
    "DEPENDENCE_BREAK": (
        "Identify false or removable data dependencies (e.g., read-after-write on a "
        "temporary that can be renamed, or accumulator recurrence that can be unrolled) "
        "and restructure the code to break those dependencies, enabling more parallelism."
    ),
    "PREDICATE_TO_DATAFLOW": (
        "Convert predicated/conditional assignments (if/mux style) into a dataflow "
        "representation where all inputs flow through combinational logic to a final "
        "select/mux. Replace control-flow-based conditional writes with data-flow "
        "select expressions to produce cleaner RTL with explicit muxes."
    ),
    "LOOP_PIPELINING": (
        "Restructure loop bodies to enable pipelining: ensure loop-carried dependencies "
        "are minimized, split recurrences, and separate independent operations so that "
        "consecutive loop iterations can overlap in execution."
    ),
    "PARTIAL_UNROLL": (
        "Manually unroll loops by a small factor (2x-4x) to expose instruction-level "
        "parallelism within the loop body. Duplicate the loop body and adjust the "
        "induction variable to process multiple elements per iteration."
    ),
    "PIPELINE_STAGE_INSERT": (
        "Insert explicit intermediate pipeline register variables between long chains "
        "of combinational logic. Break a single-cycle computation into multi-cycle "
        "stages by storing intermediate results in named temporaries."
    ),
    "MUX_TREE_BALANCE": (
        "Balance deep if-else or ternary multiplexer chains into a balanced tree form. "
        "Convert linear mux chains into parallel evaluation structures to reduce "
        "the depth of the multiplexer tree in generated RTL."
    ),
    "CARRY_SAVE_REWRITE": (
        "Rewrite multi-operand addition into carry-save form by grouping operands into "
        "triplets and reducing with partial sums. This reduces the critical path of "
        "multi-input additions from O(N) to O(log N) carry-propagation stages."
    ),
    "GUARD_RELAXATION": (
        "Relax over-constraining guard conditions by hoisting computation out of "
        "conditionals. Move assignments that are unconditionally needed outside of "
        "if/else blocks, reducing the control-flow dependency on the critical path."
    ),
    "COPY_PROPAGATION": (
        "Propagate copied values forward: when a variable is assigned a simple copy "
        "of another (x = y), replace subsequent uses of x with y directly. This "
        "eliminates redundant register copies and shortens dependency chains."
    ),
    "ALGEBRAIC_SIMPLIFY": (
        "Apply algebraic identities to simplify expressions: x*1→x, x+0→x, x&~0→x, "
        "x|0→x, double negation removal, de Morgan's law application, and similar "
        "identity-based simplifications that reduce logic depth."
    ),
    "BOOLEAN_TO_ARITHMETIC": (
        "Convert boolean-heavy expressions (chains of &&, ||, !) into arithmetic "
        "equivalents using addition and multiplication. This can produce more "
        "efficient hardware when the boolean chain is long."
    ),
    "ENCODE_ONEHOT_TO_BINARY": (
        "Detect one-hot encoded comparisons (x==1, x==2, x==4, ...) and convert "
        "them to binary-encoded index lookups. This reduces the width of comparison "
        "logic and mux selectors in the generated RTL."
    ),
}

AREA_TRANSFORM_DESCRIPTIONS = {
    "RESOURCE_SHARE": (
        "Identify multiple instances of the same operation type (e.g., multiple "
        "multipliers or adders) and restructure code so they can share a single "
        "hardware unit across different execution paths or clock cycles."
    ),
    "RESOURCE_BIND_SMALL": (
        "Replace wide or complex operators with smaller, area-efficient alternatives. "
        "E.g., replace multiplications with shift-and-add, use narrower data types, "
        "or replace division/modulo with bitwise operations where possible."
    ),
    "REDUCE_UNROLL": (
        "Reduce excessive loop unrolling that creates duplicated hardware. Re-roll "
        "unrolled loops back into compact loop form to save area at the cost of "
        "additional cycles."
    ),
    "SERIALIZE_PARALLELISM": (
        "Convert parallel operations into serialized (sequential) form. Replace "
        "parallel hardware duplication with time-multiplexed reuse of a single "
        "computation unit to reduce area."
    ),
    "PIPELINE_RELAX": (
        "Break compound expressions into pipeline stages to allow the HLS tool to "
        "relax timing and share logic across cycles. Split complex ternary or "
        "compound operations into intermediate variables."
    ),
    "LOOP_FUSION": (
        "Merge multiple loops with the same iteration range into a single loop to "
        "eliminate redundant loop control hardware and enable data reuse within "
        "a single loop body."
    ),
    "FUNCTION_OUTLINE_REUSE": (
        "Extract repeated computation patterns into a shared helper function that "
        "can be called multiple times, allowing HLS to synthesize the logic once "
        "and reuse it, saving area."
    ),
    "BITWIDTH_SHRINK": (
        "Narrow variable bit-widths to the minimum required by the actual value range. "
        "Replace int/unsigned int with smaller types (char, short, or exact-width types) "
        "where the value range permits, reducing register and logic area."
    ),
    "TYPE_NARROWING_PROPAGATION": (
        "Propagate narrowed types through the dataflow graph. When a variable is "
        "assigned from a narrow source, propagate that narrow type to downstream "
        "operations to avoid unnecessary widening."
    ),
    "CONST_PROP": (
        "Propagate constant values through the code. When a variable is assigned a "
        "compile-time constant, substitute the constant at all use sites and eliminate "
        "the variable, reducing register count and enabling further simplification."
    ),
    "DEAD_CODE_ELIM": (
        "Remove dead code: assignments whose results are never read, unreachable "
        "branches, and variables that are written but never used. This directly "
        "reduces the amount of synthesized logic and registers.\n"
        "IMPORTANT: In this RTL-derived C codebase, local structs named "
        "'state_elements_*' (e.g. 'struct state_elements_u_block_13 su_block_13;') "
        "represent PERSISTENT REGISTER STATE — they model flip-flops that retain "
        "values across clock cycles. Writes to these structs are NOT dead code, even "
        "if the values are never read within the same function invocation. Only remove "
        "writes to truly unused local scalar variables or computations whose results "
        "are provably never consumed by any state element or output pointer."
    ),
    "COMMON_SUBEXPR_EXTRACT": (
        "Identify repeated sub-expressions and extract them into a single named "
        "variable computed once. This reduces duplicated logic in the synthesized "
        "hardware, saving area."
    ),
    "STRENGTH_REDUCTION": (
        "Replace expensive operations with cheaper equivalents: multiplication by "
        "constants → shifts and adds; division by powers of 2 → right shifts; "
        "modulo by powers of 2 → bitwise AND. Reduces operator area."
    ),
    "SHIFT_ADD_REWRITE": (
        "Rewrite multiplication by constants as explicit shift-and-add sequences. "
        "E.g., x*5 → (x<<2)+x. This eliminates multiplier hardware and uses "
        "only shifters and adders, which are much smaller."
    ),
    "TABLE_LOOKUP_REWRITE": (
        "Convert complex combinational logic into table lookups (ROM/LUT). When "
        "a function maps a small input space to outputs, replace the logic with "
        "an array lookup, trading logic area for memory."
    ),
    "ARRAY_PACK": (
        "Pack multiple small fields from a struct into a single wider variable "
        "or array element, reducing the number of distinct storage elements and "
        "memory ports needed in the synthesized design."
    ),
    "ARRAY_RESHAPE": (
        "Reshape arrays to reduce the number of memory ports or BRAM usage. "
        "Combine multiple small arrays into one larger array, or reshape "
        "multi-dimensional arrays to improve access patterns."
    ),
    "REDUCE_PARTITION_FACTOR": (
        "Reduce array partitioning to use fewer parallel memory banks. Instead of "
        "fully partitioning arrays into registers, use partial partitioning or "
        "block partitioning to balance area and throughput."
    ),
    "LIMIT_MEMORY_PORTS": (
        "Restructure memory access patterns to require fewer simultaneous reads "
        "and writes, allowing the use of single-port or dual-port memories instead "
        "of multi-ported register files."
    ),
    "REDUCE_BUFFER_DEPTH": (
        "Reduce FIFO buffer or delay line depths to the minimum functionally required. "
        "Analyze data flow to determine the actual required buffering depth and "
        "trim excess capacity."
    ),
    "LOGIC_MINIMIZATION": (
        "Apply boolean logic minimization techniques: simplify complex boolean "
        "expressions using Karnaugh-map-style reductions, consensus theorem, "
        "absorption law, and other logic minimization identities."
    ),
    "CONDITION_MERGE": (
        "Merge multiple if-statements that test related or overlapping conditions "
        "into a single combined conditional block, reducing duplicated comparison "
        "logic and control-flow hardware."
    ),
    "OPERATOR_TIME_MULTIPLEX": (
        "Time-multiplex expensive operators across multiple uses. Instead of "
        "instantiating separate operators for each use, route different operands "
        "to a shared operator in different clock cycles."
    ),
    "REGISTER_LIFETIME_SHARE": (
        "Identify registers with non-overlapping lifetimes and merge them into "
        "a single register, reducing total register count. When one variable is "
        "dead before another is born, they can share storage."
    ),
    "MEMORY_PROMOTION": (
        "Promote frequently accessed array elements to scalar variables (registers). "
        "When a loop repeatedly reads/writes the same array index, cache it in a "
        "local variable to reduce memory port pressure."
    ),
    "FSM_REENCODE": (
        "Re-encode finite state machine states to use fewer bits. Convert one-hot "
        "or sparse encodings to dense binary encoding, reducing register count "
        "and comparison logic width."
    ),
    "RESET_SIMPLIFY": (
        "Simplify reset logic by using block-level initialization (memset or "
        "aggregate initialization) instead of field-by-field reset assignments. "
        "This reduces the amount of reset logic synthesized."
    ),
    "COPY_PROPAGATION": (
        "Propagate copied values forward: when a variable is assigned a simple copy "
        "of another (x = y), replace subsequent uses of x with y directly. This "
        "eliminates redundant registers and reduces area."
    ),
    "ALGEBRAIC_SIMPLIFY": (
        "Apply algebraic identities to simplify expressions: x*1→x, x+0→x, x&~0→x, "
        "double negation removal, and similar identity-based simplifications that "
        "reduce logic area."
    ),
    "BOOLEAN_TO_ARITHMETIC": (
        "Convert boolean-heavy expressions (chains of &&, ||, !) into arithmetic "
        "equivalents using addition and multiplication. This can produce more "
        "area-efficient hardware when the boolean chain is long."
    ),
    "ENCODE_ONEHOT_TO_BINARY": (
        "Detect one-hot encoded comparisons (x==1, x==2, x==4, ...) and convert "
        "them to binary-encoded index lookups. This reduces comparison logic width "
        "and mux selector area."
    ),
}

# ─── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert in RTL-derived C code optimization for High-Level Synthesis (HLS).
You will apply exactly ONE transformation to the given C source file.

Transformation: {transform_name}
Description: {transform_desc}
Optimization goal: {optimization_goal}

Domain context:
This code is auto-generated from RTL (Verilog) via a Verilog-to-C translator.
Local structs named "state_elements_*" (e.g. struct state_elements_u_block_13)
represent PERSISTENT HARDWARE REGISTERS (flip-flops). Writes to fields of these
structs model register updates that persist across clock cycles in the original
hardware. They are NOT dead code, even when the values are never read within the
same function call — the struct state IS the functional output of the module.

Rules:
1. Preserve functional correctness — outputs must be bit-identical for all inputs.
2. Keep the same function signatures and struct definitions.
3. Only apply the requested transformation. Do not do unrelated cleanups.
4. The code MUST remain compatible with Vivado/Vitis HLS synthesis:
   - No dynamic memory allocation (malloc/free/new/delete).
   - No recursion, no exceptions, no RTTI, no complex STL.
   - No function pointers or virtual dispatch.
   - All arrays must have statically determinable sizes.
   - All loops must have bounded trip counts.
5. Do NOT add any #pragma directives. No #pragma HLS, no #pragma anything.
   All optimization must come purely from code restructuring and rewriting.
6. If the transformation does not meaningfully apply to this code, return the
   original code UNCHANGED and set the JSON field "applied": false.
7. Add a brief comment block at the top of the file (after the includes) explaining
   what was changed and why it improves {optimization_goal}, e.g.:
   /* TRANSFORM: {transform_name}
      Changed: <one sentence>
      Benefit: <one sentence>
   */
8. Return ONLY valid C code — no markdown fences, no explanation outside the code.
   The very first line must be a C preprocessor directive or comment.

At the very end of the file, append a special JSON comment on a single line:
// TRANSFORM_META: {{"applied": true_or_false, "summary": "one-line description"}}
"""

USER_PROMPT_TEMPLATE = """\
Apply the {transform_name} transformation to the following C file.
File path (for context): {filepath}

--- SOURCE ---
{source_code}
--- END SOURCE ---
"""

# ─── Thread safety & rate limiting ───────────────────────────────────────────

_print_lock = threading.Lock()


def tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


class RateLimiter:
    """Token-bucket style rate limiter for API requests.

    Allows at most `rpm` requests per 60-second window, with an additional
    global concurrency cap.  When a 429 is hit externally, `backoff()` pauses
    ALL threads for a cooldown period.
    """

    def __init__(self, rpm: int = 60, max_concurrent: int = 8):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._rpm = rpm
        self._interval = 60.0 / rpm          # min seconds between requests
        self._lock = threading.Lock()
        self._last_request = 0.0
        self._backoff_until = 0.0             # global pause timestamp

    def acquire(self):
        """Block until we can safely issue a request."""
        self._semaphore.acquire()
        with self._lock:
            # Honour global backoff (set when any thread hits 429)
            now = time.time()
            if now < self._backoff_until:
                wait = self._backoff_until - now
                self._lock.release()          # release while sleeping
                time.sleep(wait)
                self._lock.acquire()
            # Enforce per-request spacing
            now = time.time()
            elapsed = now - self._last_request
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_request = time.time()

    def release(self):
        self._semaphore.release()

    def backoff(self, seconds: float):
        """Set a global cooldown — all threads will wait."""
        with self._lock:
            target = time.time() + seconds
            if target > self._backoff_until:
                self._backoff_until = target


# Default global rate limiter (overwritten by scripts via init_rate_limiter)
_rate_limiter: RateLimiter | None = None


def init_rate_limiter(rpm: int = 60, max_concurrent: int = 8) -> RateLimiter:
    """Initialise (or reinitialise) the module-level rate limiter."""
    global _rate_limiter
    _rate_limiter = RateLimiter(rpm=rpm, max_concurrent=max_concurrent)
    return _rate_limiter


# ─── Helpers ──────────────────────────────────────────────────────────────────

def create_client() -> OpenAI:
    """Create an OpenAI-compatible client.

    Reads OPENAI_API_KEY and OPENAI_BASE_URL from environment.
    """
    return OpenAI()


def extract_meta_comment(code: str) -> dict:
    """Parse the trailing // TRANSFORM_META: {...} line if present."""
    for line in reversed(code.splitlines()):
        line = line.strip()
        if line.startswith("// TRANSFORM_META:"):
            try:
                return json.loads(line[len("// TRANSFORM_META:"):].strip())
            except json.JSONDecodeError:
                pass
    return {"applied": True, "summary": "unknown"}


def strip_markdown_fences(code: str) -> str:
    """Extract a C translation unit from an over-formatted LLM output."""
    from llm_code_sanitize import sanitize_c_response

    return sanitize_c_response(code)


def code_actually_differs(original: str, transformed: str) -> bool:
    """Compare code after stripping comments, blank lines, and transform metadata.

    Returns True if the functional code content actually changed.
    """
    def normalize(code: str) -> str:
        # Remove TRANSFORM_META line
        code = re.sub(r'//\s*TRANSFORM_META:.*', '', code)
        # Remove TRANSFORM header comment block
        code = re.sub(r'/\*\s*TRANSFORM:.*?\*/', '', code, flags=re.DOTALL)
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Normalize whitespace: collapse to single spaces, strip lines
        lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
        return '\n'.join(lines)

    return normalize(original) != normalize(transformed)


def apply_llm_transform(
    client: OpenAI,
    source_code: str,
    filepath: str,
    transform_name: str,
    transform_desc: str,
    optimization_goal: str,
    *,
    model: str | None = None,
    retries: int = 3,
) -> tuple:
    """Call an OpenAI-compatible LLM to apply a single transform.

    Returns (transformed_code, meta_dict, differs_bool).
    """
    if model is None:
        model = MODEL

    system = SYSTEM_PROMPT_TEMPLATE.format(
        transform_name=transform_name,
        transform_desc=transform_desc,
        optimization_goal=optimization_goal,
    )
    user = USER_PROMPT_TEMPLATE.format(
        transform_name=transform_name,
        filepath=filepath,
        source_code=source_code,
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    for attempt in range(retries):
        # Acquire rate-limiter slot (if one is configured)
        rl = _rate_limiter
        if rl:
            rl.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=8192,
                messages=messages,
            )
            code = resp.choices[0].message.content.strip()
            code = strip_markdown_fences(code)
            meta = extract_meta_comment(code)
            differs = code_actually_differs(source_code, code)
            return code, meta, differs
        except RateLimitError:
            wait = 2 ** attempt * 10
            if rl:
                rl.backoff(wait)       # pause ALL threads
            tprint(f"    [rate-limit] {transform_name}: waiting {wait}s ...",
                   flush=True)
            time.sleep(wait)
        except Exception as e:
            tprint(f"    [error] {transform_name} attempt {attempt + 1}: {e}",
                   flush=True)
            if attempt == retries - 1:
                raise
            time.sleep(5)
        finally:
            if rl:
                rl.release()

    raise RuntimeError("All retries exhausted")
