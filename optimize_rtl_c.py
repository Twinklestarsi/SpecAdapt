#!/usr/bin/env python3
"""
RTL-derived C Optimizer
Applies 14 transformation strategies to each .c file under benchmark_output/
using Claude as the LLM backend.

Output layout:
  optimized_output/<category>/<block>/<stem>_<TRANSFORM>.c
  optimized_output/<category>/<block>/<stem>_transforms.json
"""

import os
import json
import time
import argparse
import traceback
from pathlib import Path
import anthropic

# ─── Configuration ───────────────────────────────────────────────────────────

BENCHMARK_DIR = Path(__file__).parent / "benchmark_output"
OUTPUT_DIR    = Path(__file__).parent / "optimized_output"
MODEL         = "claude-sonnet-4-6"  # change to claude-opus-4-6 for higher quality

TRANSFORMATIONS = [
    "BALANCE_TREE",
    "REASSOCIATE_ARITHMETIC",
    "BREAK_CHAIN",
    "SPLIT_OP",
    "INLINE_CRITICAL_FUNCTION",
    "OUTLINE_LONG_COMPUTE",
    "LOOP_FISSION",
    "LOOP_INTERCHANGE",
    "CONTROL_FLATTEN",
    "IF_CONVERSION",
    "SPECULATIVE_COMPUTE",
    "COMMON_SUBEXPR_EXTRACT",
    "DEPENDENCE_BREAK",
    "PREDICATE_TO_DATAFLOW",
]

# Per-transform instruction injected into the system prompt
TRANSFORM_DESCRIPTIONS = {
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
        "E.g., split a 64-bit multiplication into 32-bit halves, or decompose a multi-bit "
        "conditional assignment into per-bit or per-field operations."
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
}

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert in RTL-derived C code optimization for High-Level Synthesis (HLS).
You will apply exactly ONE transformation to the given C source file.

Transformation: {transform_name}
Description: {transform_desc}

Rules:
1. Preserve functional correctness — outputs must be bit-identical for all inputs.
2. Keep the same function signatures and struct definitions.
3. Only apply the requested transformation. Do not do unrelated cleanups.
4. If the transformation does not meaningfully apply to this code, return the original
   code UNCHANGED and set the JSON field "applied": false.
5. Add a brief comment block at the top of the file (after the includes) explaining
   what was changed and why it improves area/timing, e.g.:
   /* TRANSFORM: {transform_name}
      Changed: <one sentence>
      Benefit: <one sentence>
   */
6. Return ONLY valid C code — no markdown fences, no explanation outside the code.
   The very first line must be a C preprocessor directive or comment.

At the very end of the file, append a special JSON comment block on a single line:
// TRANSFORM_META: {{"applied": true_or_false, "summary": "one-line description"}}
"""

USER_PROMPT_TEMPLATE = """\
Apply the {transform_name} transformation to the following C file.
File path (for context): {filepath}

--- SOURCE ---
{source_code}
--- END SOURCE ---
"""

# ─── Helpers ─────────────────────────────────────────────────────────────────

def discover_c_files(base: Path) -> list[Path]:
    return sorted(base.rglob("*.c"))


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


def output_path_for(c_file: Path, transform: str, base_in: Path, base_out: Path) -> Path:
    rel = c_file.relative_to(base_in)
    stem = c_file.stem
    out_dir = base_out / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{transform}.c"


def json_summary_path(c_file: Path, base_in: Path, base_out: Path) -> Path:
    rel = c_file.relative_to(base_in)
    out_dir = base_out / rel.parent
    return out_dir / f"{rel.stem}_transforms.json"

# ─── Core LLM call ───────────────────────────────────────────────────────────

def apply_transform(client: anthropic.Anthropic,
                    source_code: str,
                    filepath: str,
                    transform: str,
                    retries: int = 3) -> tuple[str, dict]:
    """Call Claude to apply a single transform. Returns (transformed_code, meta)."""
    system = SYSTEM_PROMPT_TEMPLATE.format(
        transform_name=transform,
        transform_desc=TRANSFORM_DESCRIPTIONS[transform],
    )
    user = USER_PROMPT_TEMPLATE.format(
        transform_name=transform,
        filepath=filepath,
        source_code=source_code,
    )
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            code = msg.content[0].text.strip()
            # Strip accidental markdown fences
            if code.startswith("```"):
                lines = code.splitlines()
                code = "\n".join(
                    ln for ln in lines
                    if not ln.startswith("```")
                ).strip()
            meta = extract_meta_comment(code)
            return code, meta
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 10
            print(f"    [rate-limit] waiting {wait}s …")
            time.sleep(wait)
        except Exception as e:
            print(f"    [error] attempt {attempt+1}: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(5)
    raise RuntimeError("All retries exhausted")

# ─── Main pipeline ────────────────────────────────────────────────────────────

def process_file(client: anthropic.Anthropic,
                 c_file: Path,
                 transforms: list[str],
                 base_in: Path,
                 base_out: Path,
                 skip_existing: bool = True) -> dict:
    """Process all transforms for one .c file. Returns summary dict."""
    source = c_file.read_text()
    rel = str(c_file.relative_to(base_in))
    summary_path = json_summary_path(c_file, base_in, base_out)

    # Load existing summary if resuming
    existing_summary = {}
    if summary_path.exists():
        try:
            existing_summary = json.loads(summary_path.read_text())
        except Exception:
            pass

    results = dict(existing_summary)

    for transform in transforms:
        out_file = output_path_for(c_file, transform, base_in, base_out)
        if skip_existing and out_file.exists() and transform in results:
            print(f"  [skip]  {transform}")
            continue

        print(f"  [run ]  {transform} …", end=" ", flush=True)
        try:
            code, meta = apply_transform(client, source, rel, transform)
            out_file.write_text(code)
            results[transform] = {
                "output_file": str(out_file.relative_to(base_out)),
                "applied": meta.get("applied", True),
                "summary": meta.get("summary", ""),
            }
            status = "APPLIED" if meta.get("applied", True) else "NO-OP"
            print(f"done [{status}]")
        except Exception as e:
            print(f"FAILED: {e}")
            results[transform] = {"output_file": None, "applied": False,
                                  "summary": f"ERROR: {e}"}

        # Write summary after each transform (safe checkpointing)
        summary_path.write_text(json.dumps(results, indent=2))
        time.sleep(0.3)  # gentle rate limiting

    return results


def main():
    parser = argparse.ArgumentParser(description="Apply 14 RTL-C transforms via LLM")
    parser.add_argument("--transforms", nargs="+", default=TRANSFORMATIONS,
                        help="Subset of transforms to run (default: all)")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Specific .c files to process (default: all discovered)")
    parser.add_argument("--category", default=None,
                        help="Only process files under this category subdirectory")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run even if output file already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover files and print plan without calling the API")
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    if args.files:
        c_files = [Path(f).resolve() for f in args.files]
    else:
        c_files = discover_c_files(BENCHMARK_DIR)
        if args.category:
            c_files = [f for f in c_files
                       if f.relative_to(BENCHMARK_DIR).parts[0] == args.category]

    transforms = args.transforms
    skip_existing = not args.no_skip

    print(f"Found {len(c_files)} C files, {len(transforms)} transforms "
          f"→ up to {len(c_files)*len(transforms)} outputs under {OUTPUT_DIR}/\n")

    if args.dry_run:
        for f in c_files:
            print(f"  {f.relative_to(BENCHMARK_DIR)}")
        return

    grand_summary = {}
    for i, c_file in enumerate(c_files, 1):
        rel = str(c_file.relative_to(BENCHMARK_DIR))
        print(f"[{i}/{len(c_files)}] {rel}")
        try:
            result = process_file(client, c_file, transforms,
                                  BENCHMARK_DIR, OUTPUT_DIR, skip_existing)
            grand_summary[rel] = result
        except Exception:
            print(f"  [FATAL] skipping file:\n{traceback.format_exc()}")

    # Write top-level summary
    summary_out = OUTPUT_DIR / "grand_summary.json"
    summary_out.write_text(json.dumps(grand_summary, indent=2))
    print(f"\nDone. Grand summary → {summary_out}")


if __name__ == "__main__":
    main()
