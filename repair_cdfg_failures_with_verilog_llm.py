#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI


CLANG = "clang-16"

_ERROR_LINE_RE = re.compile(r"^\s*(.+?:\d+:\d+: error: .+)$", re.M)
_FAILED_BC_RE = re.compile(r"\[ERROR\] clang-16 failed to generate bitcode: (.+)")
_C_SCALAR_SUBSCRIPT_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]")
_C_SCALAR_SLICE_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]")
_C_FUNC_DEF_RE = re.compile(r"^\s*void\s+[A-Za-z_]\w*\s*\(([^)]*)\)\s*\{", flags=re.M)
_C_SIMPLE_SCALAR_DECL_RE = re.compile(
    r"^\s*(?:unsigned\s+|signed\s+)?(?:short\s+int|long\s+int|long|short|int|char|_Bool|unsigned __int128)\s+(.+?)\s*;\s*$"
)

SYSTEM_PROMPT = """You repair broken C generated from Verilog for clang-16 compilation and CDFG extraction.

Use the Verilog as the semantic ground truth. The broken C may be badly damaged.

Requirements:
1. Return a complete C file only. No markdown fences. No explanation.
2. The result must compile with clang-16 as C.
3. Preserve the top function behavior and port names from the Verilog.
4. Prefer fixed-width integer types from stdint.h.
5. Output ports should remain pointer outputs when practical.
6. Do not use these broken constructs: irep(...), CONCATENATION(...), __CPROVER_bitvector, invalid subscripts like a[6,2], nested function definitions.
7. If the original C is too damaged, rewrite the file from the Verilog instead of trying to keep the broken structure.
8. Keep the code HLS-friendly and self-contained in a single C file.
"""


@dataclass
class RepairResult:
    c_path: str
    verilog_path: str | None
    status: str
    initial_error: str
    final_error: str = ""
    strategy: str = ""
    backup_path: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repair CDFG clang failures by using sibling Verilog plus optional LLM rewriting."
    )
    p.add_argument(
        "--log",
        action="append",
        default=[],
        help="cdfg log file(s) to parse. Defaults to cdfg_AREA.log and cdfg_TIMING.log if present.",
    )
    p.add_argument(
        "--file",
        action="append",
        default=[],
        help="Explicit .c file(s) to repair.",
    )
    p.add_argument(
        "--profile",
        choices=("openai", "cloud", "local"),
        default="openai",
        help="API profile loaded from .env.",
    )
    p.add_argument(
        "--env",
        default=".env",
        help="Path to .env with API settings.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override model name from .env.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="LLM repair retries after the initial attempt.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of files to process. 0 means no limit.",
    )
    p.add_argument(
        "--local-only",
        action="store_true",
        help="Only run deterministic text repairs, never call the LLM.",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Write repaired files back in place after creating a backup.",
    )
    p.add_argument(
        "--backup-suffix",
        default=".bak_llm_fix",
        help="Suffix used for backups when --in-place is enabled.",
    )
    p.add_argument(
        "--report-json",
        default="",
        help="Optional JSON report output path.",
    )
    return p.parse_args()


def _is_c_identifier(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z_]\w*$", name))


def collect_scalar_identifiers(c_text: str) -> set[str]:
    scalar_names: set[str] = set()

    for m in _C_FUNC_DEF_RE.finditer(c_text):
        params = (m.group(1) or "").strip()
        if not params or params == "void":
            continue
        for raw in params.split(","):
            p = raw.strip()
            if not p or "*" in p or "[" in p or "]" in p:
                continue
            tokens = [t for t in re.split(r"\s+", p) if t and t not in ("const", "volatile", "register")]
            if tokens:
                name = tokens[-1].strip()
                if _is_c_identifier(name):
                    scalar_names.add(name)

    for line in c_text.splitlines():
        m = _C_SIMPLE_SCALAR_DECL_RE.match(line)
        if not m:
            continue
        decl = m.group(1).strip()
        for part in decl.split(","):
            item = part.strip()
            if not item:
                continue
            if "=" in item:
                item = item.split("=", 1)[0].strip()
            if "*" in item or "[" in item or "]" in item:
                continue
            if _is_c_identifier(item):
                scalar_names.add(item)

    return scalar_names


def apply_local_repairs(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    scalar_names = collect_scalar_identifiers(text)

    if scalar_names:
        def bit_repl(m: re.Match[str]) -> str:
            name = m.group(1)
            bit = int(m.group(2))
            if name not in scalar_names:
                return m.group(0)
            return f"(((unsigned __int128)({name}) >> {bit}) & 1u)"

        new_text = _C_SCALAR_SUBSCRIPT_RE.sub(bit_repl, text)
        if new_text != text:
            notes.append("rewrote scalar bit subscripts")
            text = new_text

        def slice_repl(m: re.Match[str]) -> str:
            name = m.group(1)
            hi = int(m.group(2))
            lo = int(m.group(3))
            if name not in scalar_names or hi < lo:
                return m.group(0)
            width = hi - lo + 1
            if width >= 128:
                return m.group(0)
            mask = f"((((unsigned __int128)1) << {width}) - 1u)"
            return f"((((unsigned __int128)({name}) >> {lo}) & {mask}))"

        new_text = _C_SCALAR_SLICE_RE.sub(slice_repl, text)
        if new_text != text:
            notes.append("rewrote scalar slices")
            text = new_text

    new_text = re.sub(r"\bvoid\s+main\s*\(\s*\)", "int main(void)", text)
    if new_text != text:
        notes.append("normalized main signature")
        text = new_text

    return text, notes


def compile_error_for_text(text: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="cdfg_repair_") as tmpdir:
        c_path = Path(tmpdir) / "candidate.c"
        bc_path = Path(tmpdir) / "candidate.bc"
        c_path.write_text(text, encoding="utf-8")
        cmd = [
            CLANG,
            "-emit-llvm",
            "-c",
            "-O0",
            "-Xclang",
            "-disable-O0-optnone",
            str(c_path),
            "-o",
            str(bc_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = (result.stderr or "").strip()
        return result.returncode == 0, stderr


def first_error_line(stderr: str) -> str:
    m = _ERROR_LINE_RE.search(stderr or "")
    if m:
        return m.group(1)
    return (stderr or "").strip().splitlines()[0] if (stderr or "").strip() else ""


def find_verilog_for_c(c_path: Path) -> Path | None:
    candidates = sorted(
        p for p in c_path.parent.glob("*.v")
        if ".bak" not in p.name and p.is_file()
    )
    return candidates[0] if candidates else None


def iter_failed_files_from_logs(log_paths: Iterable[Path]) -> list[Path]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for log_path in log_paths:
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        for match in _FAILED_BC_RE.finditer(text):
            ordered[match.group(1).strip()] = None
    return [Path(p) for p in ordered.keys()]


def load_client(env_path: str, profile: str, model_override: str | None) -> tuple[OpenAI, str]:
    load_dotenv(env_path, override=True)

    if profile == "local":
        api_key = os.environ.get("LOCAL_API_KEY")
        base_url = os.environ.get("LOCAL_API_BASE_URL")
        model = model_override or os.environ.get("LOCAL_MODEL")
    elif profile == "cloud":
        api_key = os.environ.get("CLOUD_API_KEY")
        base_url = os.environ.get("CLOUD_API_BASE_URL")
        model = model_override or os.environ.get("CLOUD_MODEL")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE_URL")
        model = model_override or os.environ.get("OPENAI_MODEL")

    if not api_key or not base_url or not model:
        raise RuntimeError(f"Missing API configuration for profile '{profile}' in {env_path}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.startswith("c\n"):
            text = text[2:].lstrip()
    return text


def llm_repair_code(
    client: OpenAI,
    model: str,
    c_path: Path,
    verilog_path: Path,
    broken_c: str,
    compile_stderr: str,
    max_retries: int,
) -> tuple[str | None, str]:
    verilog_text = verilog_path.read_text(encoding="utf-8", errors="ignore")
    user_prompt = f"""Repair this C file so that clang-16 can compile it.

Target file: {c_path}

Verilog ground truth:
```verilog
{verilog_text}
```

Broken C:
```c
{broken_c}
```

Current clang-16 errors:
```text
{compile_stderr}
```

Return only the repaired full C file.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_stderr = compile_stderr
    for _ in range(max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=8192,
        )
        candidate = strip_code_fences(response.choices[0].message.content or "")
        if not candidate:
            last_stderr = "LLM returned empty content"
            messages.append({"role": "assistant", "content": ""})
            messages.append({"role": "user", "content": "Your previous answer was empty. Return the full repaired C file only."})
            continue

        ok, stderr = compile_error_for_text(candidate)
        if ok:
            return candidate, ""

        last_stderr = stderr
        messages.append({"role": "assistant", "content": candidate[:12000]})
        messages.append(
            {
                "role": "user",
                "content": (
                    "The previous answer still failed clang-16 compilation.\n\n"
                    f"Compiler errors:\n```text\n{stderr}\n```\n\n"
                    "Revise the file and return the full repaired C only."
                ),
            }
        )

    return None, last_stderr


def write_backup_and_replace(path: Path, new_text: str, backup_suffix: str) -> str:
    backup_path = path.with_name(path.name + backup_suffix)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    path.write_text(new_text if new_text.endswith("\n") else new_text + "\n", encoding="utf-8")
    return str(backup_path)


def process_one(
    c_path: Path,
    client: OpenAI | None,
    model: str | None,
    args: argparse.Namespace,
) -> RepairResult:
    if not c_path.exists():
        return RepairResult(str(c_path), None, "missing", "file not found")

    original_text = c_path.read_text(encoding="utf-8", errors="ignore")
    verilog_path = find_verilog_for_c(c_path)
    ok, stderr = compile_error_for_text(original_text)
    if ok:
        return RepairResult(str(c_path), str(verilog_path) if verilog_path else None, "already_ok", "", strategy="none")

    initial_error = first_error_line(stderr)

    locally_fixed, notes = apply_local_repairs(original_text)
    if locally_fixed != original_text:
        ok, local_stderr = compile_error_for_text(locally_fixed)
        if ok:
            backup = None
            if args.in_place:
                backup = write_backup_and_replace(c_path, locally_fixed, args.backup_suffix)
            return RepairResult(
                str(c_path),
                str(verilog_path) if verilog_path else None,
                "fixed_local",
                initial_error,
                strategy=", ".join(notes) or "local repair",
                backup_path=backup,
            )
        stderr = local_stderr

    if args.local_only:
        return RepairResult(
            str(c_path),
            str(verilog_path) if verilog_path else None,
            "needs_llm",
            initial_error,
            final_error=first_error_line(stderr),
            strategy=", ".join(notes) or "no local repair matched",
        )

    if verilog_path is None:
        return RepairResult(
            str(c_path),
            None,
            "no_verilog",
            initial_error,
            final_error=first_error_line(stderr),
            strategy=", ".join(notes) or "missing .v sibling",
        )

    if client is None or model is None:
        return RepairResult(
            str(c_path),
            str(verilog_path),
            "no_client",
            initial_error,
            final_error=first_error_line(stderr),
            strategy=", ".join(notes) or "llm client unavailable",
        )

    seed_text = locally_fixed if locally_fixed != original_text else original_text
    repaired, llm_stderr = llm_repair_code(
        client=client,
        model=model,
        c_path=c_path,
        verilog_path=verilog_path,
        broken_c=seed_text,
        compile_stderr=stderr,
        max_retries=args.max_retries,
    )
    if repaired is None:
        return RepairResult(
            str(c_path),
            str(verilog_path),
            "llm_failed",
            initial_error,
            final_error=first_error_line(llm_stderr),
            strategy="llm repair",
        )

    backup = None
    if args.in_place:
        backup = write_backup_and_replace(c_path, repaired, args.backup_suffix)

    return RepairResult(
        str(c_path),
        str(verilog_path),
        "fixed_llm",
        initial_error,
        strategy="llm repair" + (f" after {', '.join(notes)}" if notes else ""),
        backup_path=backup,
    )


def main() -> int:
    args = parse_args()

    log_paths = [Path(p) for p in args.log]
    if not log_paths:
        for default_name in ("cdfg_AREA.log", "cdfg_TIMING.log"):
            default_path = Path(default_name)
            if default_path.exists():
                log_paths.append(default_path)

    files = iter_failed_files_from_logs(log_paths)
    files.extend(Path(p) for p in args.file)

    ordered: OrderedDict[str, None] = OrderedDict()
    for p in files:
        ordered[str(p.resolve())] = None
    targets = [Path(p) for p in ordered.keys()]

    if args.max_files > 0:
        targets = targets[: args.max_files]

    if not targets:
        print("No target files found.", file=sys.stderr)
        return 1

    client = None
    model = None
    if not args.local_only:
        client, model = load_client(args.env, args.profile, args.model)

    print(f"Targets: {len(targets)}")
    print(f"Mode: {'local-only' if args.local_only else f'llm:{args.profile}:{model}'}")
    print(f"Write back: {'yes' if args.in_place else 'no'}")

    results: list[RepairResult] = []
    for idx, c_path in enumerate(targets, 1):
        print(f"[{idx}/{len(targets)}] {c_path}")
        result = process_one(c_path, client, model, args)
        results.append(result)
        extra = f" | final={result.final_error}" if result.final_error else ""
        print(f"  -> {result.status} | {result.strategy or 'n/a'} | {result.initial_error}{extra}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.write_text(
            json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    by_status: OrderedDict[str, int] = OrderedDict()
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    print("\nSummary:")
    for status, count in by_status.items():
        print(f"  {status}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
