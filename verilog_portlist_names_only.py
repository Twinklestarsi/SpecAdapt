#!/usr/bin/env python3
from __future__ import annotations

"""
Rewrite Verilog-2001 ANSI-style module ports to name-only port lists.

Example (只扫不换):
  python3 verilog_portlist_names_only.py --root benchmark --verbose

Example (进行替换, with .bak backups):
  python3 verilog_portlist_names_only.py --root benchmark --in-place
"""

import argparse
import dataclasses
import pathlib
import re
import sys
from typing import Iterable, Iterator, Optional


@dataclasses.dataclass(frozen=True)
class ModuleHeader:
    module_start: int
    open_paren: int
    close_paren: int
    semicolon: int


@dataclasses.dataclass(frozen=True)
class PortDecl:
    name: str
    decl: str  # without trailing ';'


_IDENT_RE = re.compile(r"(\\\S+|[A-Za-z_][A-Za-z0-9_$]*)")
_DIR_RE = re.compile(r"^\s*(input|output|inout)\b", re.IGNORECASE)


def _detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch in "_$"


def _skip_ws_and_comments(text: str, i: int) -> int:
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i + 2)
            i = n if nl == -1 else nl + 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        return i
    return i


def _consume_balanced_parens(text: str, i: int) -> int:
    assert text[i] == "("
    n = len(text)
    depth = 1
    i += 1
    in_string = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1

    raise ValueError("Unbalanced parentheses while parsing module header")


def _find_module_headers(text: str) -> list[ModuleHeader]:
    headers: list[ModuleHeader] = []
    n = len(text)
    i = 0
    in_string = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_string:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue

        if ch == "m" and text.startswith("module", i):
            prev = text[i - 1] if i > 0 else ""
            after = text[i + 6] if i + 6 < n else ""
            if (prev and _is_ident_char(prev)) or (after and _is_ident_char(after)):
                i += 1
                continue

            module_start = i
            j = i + 6
            j = _skip_ws_and_comments(text, j)

            m = _IDENT_RE.match(text, j)
            if not m:
                i += 6
                continue
            j = m.end()

            j = _skip_ws_and_comments(text, j)
            if j < n and text[j] == "#":
                j += 1
                j = _skip_ws_and_comments(text, j)
                if j >= n or text[j] != "(":
                    i += 6
                    continue
                j = _consume_balanced_parens(text, j) + 1
                j = _skip_ws_and_comments(text, j)

            if j >= n:
                break
            if text[j] != "(":
                i += 6
                continue
            open_paren = j
            close_paren = _consume_balanced_parens(text, open_paren)
            k = _skip_ws_and_comments(text, close_paren + 1)
            if k >= n or text[k] != ";":
                i = close_paren + 1
                continue
            headers.append(
                ModuleHeader(
                    module_start=module_start,
                    open_paren=open_paren,
                    close_paren=close_paren,
                    semicolon=k,
                )
            )
            i = k + 1
            continue

        i += 1

    return headers


def _split_commas_top_level(text: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    depth_bracket = 0
    i = 0
    n = len(text)
    in_string = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                cur.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            cur.append(ch)
            if ch == "\\" and i + 1 < n:
                cur.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_string = True
            cur.append(ch)
            i += 1
            continue

        if ch == "[":
            depth_bracket += 1
        elif ch == "]" and depth_bracket > 0:
            depth_bracket -= 1

        if ch == "," and depth_bracket == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue

        cur.append(ch)
        i += 1

    parts.append("".join(cur))
    return parts


def _extract_name_suffix(chunk: str) -> Optional[tuple[str, str, str]]:
    """
    Returns (prefix_before_name, name, suffix_after_name) where suffix may include unpacked dimensions.
    """
    s = chunk.strip()
    if not s:
        return None

    m = re.search(
        r"(\\\S+|[A-Za-z_][A-Za-z0-9_$]*)(\s*(?:\[[^\]]*\]\s*)*)$",
        s,
    )
    if not m:
        return None

    name = m.group(1)
    suffix = m.group(2) or ""
    prefix = s[: m.start(1)].rstrip()
    return prefix, name, suffix.strip()


def _normalize_space(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.strip())

def _strip_output_reg_wire(current_prefix: str) -> str:
    # For outputs, strip any explicit "reg"/"wire" token from the declaration prefix.
    tokens = current_prefix.split(" ")
    if not tokens or tokens[0].lower() != "output":
        return current_prefix
    tokens_stripped = [t for t in tokens if t.lower() not in ("reg", "wire")]
    return " ".join(tokens_stripped) if tokens_stripped else current_prefix


def _parse_ports(port_list: str) -> Optional[list[PortDecl]]:
    if not re.search(r"\b(input|output|inout)\b", port_list, re.IGNORECASE):
        return None

    chunks = _split_commas_top_level(port_list)
    ports: list[PortDecl] = []
    current_prefix: Optional[str] = None

    for raw in chunks:
        chunk = raw.strip()
        if not chunk:
            continue

        mdir = _DIR_RE.match(chunk)
        if mdir:
            direction = mdir.group(1)
            rest = chunk[mdir.end() :].strip()
            parsed = _extract_name_suffix(rest)
            if not parsed:
                return None
            prefix, name, suffix = parsed
            prefix_norm = _normalize_space(prefix)
            current_prefix = _normalize_space(f"{direction} {prefix_norm}".strip())
            if direction.lower() == "output":
                # Keep packed width/type tokens but avoid ANSI-style output reg/wire
                # because v2c rejects `output_register`.
                current_prefix = _strip_output_reg_wire(current_prefix)
            name_piece = name if not suffix else f"{name} {suffix}"
            decl = _normalize_space(f"{current_prefix} {name_piece}")
            ports.append(PortDecl(name=name, decl=decl))
            continue

        if current_prefix is None:
            return None

        parsed = _extract_name_suffix(chunk)
        if not parsed:
            return None
        _, name, suffix = parsed
        name_piece = name if not suffix else f"{name} {suffix}"
        decl = _normalize_space(f"{current_prefix} {name_piece}")
        ports.append(PortDecl(name=name, decl=decl))

    if not ports:
        return None
    return ports


def _detect_body_indent(text: str, start: int) -> str:
    n = len(text)
    i = start
    while i < n:
        line_end = text.find("\n", i)
        if line_end == -1:
            line_end = n
        line = text[i:line_end]

        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            return re.match(r"[ \t]*", line).group(0)  # type: ignore[union-attr]

        i = line_end + 1
    return ""


def _already_declared_in_body(module_body: str, port_name: str) -> bool:
    pat = re.compile(
        rf"(?m)^\s*(input|output|inout)\b[^;]*\b{re.escape(port_name)}\b"
    )
    return bool(pat.search(module_body))


def _compute_header_end(text: str, header: ModuleHeader) -> int:
    header_end = header.semicolon + 1
    if text.startswith("\r\n", header_end):
        header_end += 2
    elif text.startswith("\n", header_end):
        header_end += 1
    return header_end


def _find_endmodule(text: str, start: int) -> Optional[tuple[int, int]]:
    n = len(text)
    i = start
    in_string = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue

        if ch == "e" and text.startswith("endmodule", i):
            prev = text[i - 1] if i > 0 else ""
            after = text[i + 9] if i + 9 < n else ""
            if (prev and _is_ident_char(prev)) or (after and _is_ident_char(after)):
                i += 1
                continue
            return i, i + 9

        i += 1

    return None


def rewrite_output_strip_reg_wire(text: str) -> tuple[str, int]:
    """
    Remove token sequence: output [ws/comments] (reg|wire)
    while skipping strings and comments.
    """
    n = len(text)
    i = 0
    changed = 0
    out: list[str] = []

    in_string = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            out.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            out.append(ch)
            if ch == "*" and nxt == "/":
                out.append(nxt)
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            out.append(ch)
            out.append(nxt)
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            out.append(ch)
            out.append(nxt)
            in_block_comment = True
            i += 2
            continue

        if ch == '"':
            out.append(ch)
            in_string = True
            i += 1
            continue

        if text[i : i + 6].lower() == "output":
            prev = text[i - 1] if i > 0 else ""
            after = text[i + 6] if i + 6 < n else ""
            if (prev and _is_ident_char(prev)) or (after and _is_ident_char(after)):
                out.append(ch)
                i += 1
                continue

            out.append(text[i : i + 6])
            j = i + 6
            between: list[str] = []
            k = j
            while k < n:
                if text[k].isspace():
                    between.append(text[k])
                    k += 1
                    continue
                if text.startswith("//", k):
                    break
                if text.startswith("/*", k):
                    end = text.find("*/", k + 2)
                    if end == -1:
                        between.append(text[k:])
                        k = n
                        break
                    between.append(text[k : end + 2])
                    k = end + 2
                    continue
                break

            if text[k : k + 3].lower() == "reg":
                prev2 = text[k - 1] if k > 0 else ""
                after2 = text[k + 3] if k + 3 < n else ""
                if not ((prev2 and _is_ident_char(prev2)) or (after2 and _is_ident_char(after2))):
                    out.append("".join(between))
                    i = k + 3
                    changed += 1
                    continue
            if text[k : k + 4].lower() == "wire":
                prev2 = text[k - 1] if k > 0 else ""
                after2 = text[k + 4] if k + 4 < n else ""
                if not ((prev2 and _is_ident_char(prev2)) or (after2 and _is_ident_char(after2))):
                    out.append("".join(between))
                    i = k + 4
                    changed += 1
                    continue

            out.append("".join(between))
            i = k
            continue

        out.append(ch)
        i += 1

    updated = "".join(out)
    # Clean up spacing left after stripping (e.g., "output  [..]" -> "output [..]").
    updated = re.sub(r"(?i)\boutput[ \t]{2,}", "output ", updated)
    return updated, changed


def ensure_output_regs_for_procedural_assignments(text: str) -> tuple[str, int]:
    """
    Keep 'output' declarations free of 'reg'/'wire', but if an output is assigned in
    procedural code (always/initial), add a matching 'reg' declaration in the module body.
    """
    nl = _detect_newline(text)
    headers = _find_module_headers(text)
    if not headers:
        return text, 0

    out: list[str] = []
    last = 0
    changed = 0

    for header in headers:
        body_start = _compute_header_end(text, header)
        endm = _find_endmodule(text, body_start)
        if endm is None:
            continue
        endmodule_start, _endmodule_end = endm

        module_body = text[body_start:endmodule_start]
        out.append(text[last:body_start])

        reg_decls: list[str] = []
        already_reg = set(re.findall(r"(?m)^\s*reg\b[^;]*\b([A-Za-z_][A-Za-z0-9_$]*)\b", module_body))

        # Collect output declarations (single- or multi-line) in a light-weight way.
        stmts: list[tuple[str, str]] = []  # (indent, stmt_without_semicolon)
        in_block_comment = False
        lines = module_body.splitlines(keepends=True)
        i = 0
        while i < len(lines):
            line = lines[i]

            if in_block_comment:
                if "*/" in line:
                    in_block_comment = False
                i += 1
                continue

            stripped = line.lstrip()
            if stripped.startswith("//"):
                i += 1
                continue

            if "/*" in stripped:
                before, _sep, after = stripped.partition("/*")
                if "*/" not in after:
                    in_block_comment = True
                stripped = before

            if not re.match(r"^\s*output\b", stripped, re.IGNORECASE):
                i += 1
                continue

            stmt_parts = [stripped]
            while ";" not in stmt_parts[-1] and i + 1 < len(lines):
                i += 1
                stmt_parts.append(lines[i])
            stmt_joined = "".join(stmt_parts)
            stmt_no_line_comments = re.sub(r"//.*", "", stmt_joined)
            if ";" not in stmt_no_line_comments:
                i += 1
                continue

            stmt_until_semicolon = stmt_no_line_comments.split(";", 1)[0].strip()
            indent = re.match(r"[ \t]*", line).group(0)  # type: ignore[union-attr]
            stmts.append((indent, stmt_until_semicolon))
            i += 1

        # For each output port, if it's assigned procedurally, ensure it is a reg.
        for indent, stmt in stmts:
            rest = stmt[len("output") :].strip()
            chunks = _split_commas_top_level(rest)
            if not chunks:
                continue

            first = chunks[0].strip()
            parsed = _extract_name_suffix(first)
            if not parsed:
                continue
            prefix, name0, suffix0 = parsed
            prefix_tokens = [t for t in _normalize_space(prefix).split(" ") if t.lower() not in ("reg", "wire")]
            base_prefix = _normalize_space(" ".join(prefix_tokens))
            reg_prefix = _normalize_space(("reg " + base_prefix).strip())

            declared_as_reg = bool(re.search(r"(?i)\breg\b", _normalize_space(prefix)))

            def needs_reg(sig: str, force: bool = False) -> bool:
                if sig in already_reg:
                    return False
                if force:
                    return True
                # Match procedural assignments both at line start and after ';' on the same line.
                pat = re.compile(
                    rf"(?m)(?:^|;)\s*(?:\w+\s*:\s*)?{re.escape(sig)}\s*(<=|=)\s*"
                )
                return bool(pat.search(module_body))

            if needs_reg(name0, force=declared_as_reg):
                suffix_piece = f" {suffix0}" if suffix0 else ""
                reg_decls.append(f"{indent}{reg_prefix} {name0}{suffix_piece};")

            for more in chunks[1:]:
                parsed2 = _extract_name_suffix(more.strip())
                if not parsed2:
                    continue
                _p2, name2, suffix2 = parsed2
                if needs_reg(name2, force=declared_as_reg):
                    suffix_piece = f" {suffix2}" if suffix2 else ""
                    reg_decls.append(f"{indent}{reg_prefix} {name2}{suffix_piece};")

        if not reg_decls:
            out.append(module_body)
            last = endmodule_start
            continue

        # Insert after the initial IO-declaration block (input/output/inout) near the top.
        insert_at = 0
        body_lines2 = module_body.splitlines(keepends=True)
        j = 0
        while j < len(body_lines2):
            ln = body_lines2[j]
            s = ln.strip()
            if not s or s.startswith("//"):
                j += 1
                continue
            if re.match(r"^\s*(input|output|inout)\b", ln):
                j += 1
                continue
            break
        insert_at = sum(len(x) for x in body_lines2[:j])

        insertion = nl.join(dict.fromkeys(reg_decls)) + nl + nl
        out.append(module_body[:insert_at] + insertion + module_body[insert_at:])
        changed += 1
        last = endmodule_start

    out.append(text[last:])
    return "".join(out), changed


def _rewrite_one_module(text: str, header: ModuleHeader, nl: str) -> tuple[str, bool]:
    header_end = _compute_header_end(text, header)

    port_list = text[header.open_paren + 1 : header.close_paren]
    ports = _parse_ports(port_list)
    if ports is None:
        return text[header.module_start:header_end], False

    existing_port_names = [p.name for p in ports]

    # Preserve port indentation if module header is already multi-line.
    port_indent = "    "
    m = re.search(r"\n([ \t]*)\S", port_list)
    if m:
        port_indent = m.group(1)

    new_port_lines: list[str] = []
    for idx, name in enumerate(existing_port_names):
        comma = "," if idx != len(existing_port_names) - 1 else ""
        new_port_lines.append(f"{port_indent}{name}{comma}")
    new_port_list = nl + nl.join(new_port_lines) + nl

    body_indent = _detect_body_indent(text, header_end)
    module_body = text[header_end:]
    decl_lines = [
        f"{body_indent}{p.decl};"
        for p in ports
        if not _already_declared_in_body(module_body, p.name)
    ]
    insertion = ""
    if decl_lines:
        insertion = nl.join(decl_lines) + nl + nl

    replaced = (
        text[header.module_start : header.open_paren + 1]
        + new_port_list
        + text[header.close_paren : header_end]
        + insertion
    )
    return replaced, True


def convert_verilog_ports(text: str) -> tuple[str, int]:
    nl = _detect_newline(text)
    headers = _find_module_headers(text)
    if not headers:
        updated, _ = rewrite_output_strip_reg_wire(text)
        updated2, _ = ensure_output_regs_for_procedural_assignments(updated)
        return updated2, 0

    out: list[str] = []
    last = 0
    changed = 0

    for header in headers:
        out.append(text[last : header.module_start])
        rewritten, did = _rewrite_one_module(text, header, nl)
        out.append(rewritten)
        header_end = header.semicolon + 1
        if text.startswith("\r\n", header_end):
            header_end += 2
        elif text.startswith("\n", header_end):
            header_end += 1
        last = header_end
        if did:
            changed += 1

    out.append(text[last:])
    updated = "".join(out)
    updated2, _ = rewrite_output_strip_reg_wire(updated)
    updated3, _ = ensure_output_regs_for_procedural_assignments(updated2)
    return updated3, changed


def _iter_verilog_files(root: pathlib.Path) -> Iterator[pathlib.Path]:
    for p in sorted(root.rglob("*.v")):
        if p.is_file():
            yield p


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Rewrite ANSI-style module port declarations to name-only port lists, "
            "then re-declare input/output/inout after the header."
        )
    )
    ap.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("benchmark"),
        help='Root directory to scan (default: "benchmark")',
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite files on disk (default: dry-run)",
    )
    ap.add_argument(
        "--backup-ext",
        default=".bak",
        help='When --in-place, write a backup with this extension (default: ".bak"). Use empty string to disable.',
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file changes",
    )

    args = ap.parse_args(argv)
    root: pathlib.Path = args.root
    if not root.exists():
        print(f'error: root not found: "{root}"', file=sys.stderr)
        return 2

    total_files = 0
    changed_files = 0
    changed_modules = 0

    for path in _iter_verilog_files(root):
        total_files += 1
        original = path.read_text(encoding="utf-8", errors="ignore")
        updated, mod_cnt = convert_verilog_ports(original)
        if updated == original:
            continue

        changed_files += 1
        changed_modules += mod_cnt
        if args.verbose or not args.in_place:
            print(f"{path}: modules changed={mod_cnt}")

        if args.in_place:
            if args.backup_ext:
                backup = path.with_name(path.name + args.backup_ext)
                backup.write_text(original, encoding="utf-8")
            path.write_text(updated, encoding="utf-8")

    if args.in_place:
        print(
            f"done: scanned {total_files} files, changed {changed_files} files, changed {changed_modules} modules"
        )
    else:
        print(
            f"dry-run: scanned {total_files} files, would change {changed_files} files, would change {changed_modules} modules"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
