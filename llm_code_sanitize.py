"""Extract code-only payloads from occasionally over-formatted LLM replies.

This module only extracts source. It deliberately does not repair incomplete
C or Verilog; the existing compiler and verifier remain the authority on
syntax and behavior.
"""

from __future__ import annotations

import re

# Regexes used by the extraction implementation.
_FENCED_BLOCK_RE = re.compile(
    r"^[ \t]*\x60\x60\x60(?P<language>[A-Za-z0-9_+.-]*)[ \t]*\r?\n"
    r"(?P<code>.*?)^[ \t]*\x60\x60\x60[ \t]*(?:\r?\n|$)",
    flags=re.MULTILINE | re.DOTALL,
)
_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)
_C_MARKER_RE = re.compile(
    r"(?:^\s*#\s*include\b|\b(?:void|int|unsigned|static|struct|typedef|bool|"
    r"char|long|short|uint\d+_t|int\d+_t)\s+[A-Za-z_]\w*\s*(?:\(|\{|=))",
    flags=re.MULTILINE,
)
_C_INCLUDE_LINE_RE = re.compile(
    r"^[ \t]*#\s*include\b[^\r\n]*",
    flags=re.MULTILINE,
)
_C_START_LINE_RE = re.compile(
    r"^[ \t]*(?:#\s*include\b|typedef\b|struct\b|union\b|enum\b|"
    r"(?:static|extern|inline|const|volatile)\b|"
    r"(?:void|bool|char|short|int|long|float|double|unsigned|signed|"
    r"uint\d+_t|int\d+_t)\b)",
    flags=re.MULTILINE,
)
_C_FUNCTION_DEF_RE = re.compile(
    r"^[ \t]*(?:(?:static|extern|inline|const|volatile)\s+)*"
    r"(?:(?:unsigned|signed|short|long)\s+)*(?:void|bool|char|int|float|double|"
    r"size_t|[A-Za-z_]\w*|(?:struct|union|enum)\s+[A-Za-z_]\w*)"
    r"\s+\**[A-Za-z_]\w*\s*\([^;{}]*\)\s*\{",
    flags=re.MULTILINE,
)
_C_TOP_LEVEL_START_RE = re.compile(
    r"[ \t]*(?:(?:typedef|static|extern|inline|const|volatile|register|"
    r"_Thread_local|_Atomic)\s+)*(?:(?:struct|union|enum)\b|"
    r"(?:(?:unsigned|signed|short|long)\s+)*(?:void|bool|char|short|"
    r"int|long|float|double|size_t|uint\d+_t|int\d+_t|"
    r"[A-Za-z_]\w*)\s+(?:\*+\s*)?[A-Za-z_]\w*\s*(?:\(|=|;|\[|\{|$))"
)
_C_POST_BRACE_DECL_RE = re.compile(
    r"[ \t]*(?:\*+[ \t]*)?[A-Za-z_]\w*(?:[ \t]*\[[^\]\r\n]*\])?"
    r"[ \t]*(?:=|;)",
)
_C_TRANSFORM_META_LINE_RE = re.compile(
    r"^[ \t]*//[ \t]*TRANSFORM_META(?:\b|[_:.]).*(?:\r?\n|$)",
    flags=re.IGNORECASE,
)
_C_TRANSFORM_META_BLOCK_RE = re.compile(
    r"^[ \t]*/\*[ \t]*TRANSFORM_META(?:\b|[_:.]).*?\*/[ \t]*(?:\r?\n|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
_MODULE_START_RE = re.compile(
    r"^[ \t]*module\s+(?:automatic\s+)?[A-Za-z_]\w*\b",
    flags=re.MULTILINE,
)
_ENDMODULE_RE = re.compile(r"\bendmodule\b")
_PROSE_START_RE = re.compile(
    r"^\s*(?:Explanation|Here(?:'s| is)|This (?:C|Verilog|module)|"
    r"The (?:C|Verilog|module)|-\s*\*\*Transform)\b",
    flags=re.MULTILINE | re.IGNORECASE,
)
_C_LANGUAGES = {"", "c", "h", "c99", "c11", "cpp", "c++", "cc"}
_VERILOG_LANGUAGES = {"", "v", "verilog", "sv", "systemverilog"}

# Extraction helpers used by the generation flow.
def _strip_think_blocks(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if _THINK_CLOSE_RE.search(cleaned):
        cleaned = _THINK_CLOSE_RE.split(cleaned)[-1]
    return _THINK_OPEN_RE.sub("", cleaned).strip()


def _fenced_blocks(text: str) -> list[tuple[int, int, str, str]]:
    return [
        (
            match.start(),
            match.end(),
            match.group("language").lower(),
            match.group("code"),
        )
        for match in _FENCED_BLOCK_RE.finditer(text)
    ]


def _prefix_before_fence(text: str) -> str:
    blocks = _fenced_blocks(text)
    return text[: blocks[0][0]].strip() if blocks else ""


def _trim_c_candidate(candidate: str) -> str:
    marker = _C_START_LINE_RE.search(candidate)
    if marker and marker.start() > 0:
        candidate = candidate[marker.start() :]
    prose = _PROSE_START_RE.search(candidate)
    if prose:
        candidate = candidate[: prose.start()].rstrip()
    return candidate.strip()


def _raw_c_candidate(text: str) -> str:
    includes = list(_C_INCLUDE_LINE_RE.finditer(text))
    if includes:
        start = includes[-1].start()
        index = len(includes) - 2
        while index >= 0:
            previous = includes[index]
            if text[previous.end() : start].strip():
                break
            start = previous.start()
            index -= 1
        return _trim_c_candidate(text[start:])

    markers = list(_C_START_LINE_RE.finditer(text))
    if markers:
        return _trim_c_candidate(text[markers[-1].start() :])
    return _trim_c_candidate(text)


def _consume_c_preprocessor_line(text: str, start: int) -> int:
    """Consume one preprocessor directive, including backslash continuations."""

    length = len(text)
    cursor = start
    while cursor < length:
        newline = text.find("\n", cursor)
        if newline < 0:
            return length
        line = text[cursor:newline]
        cursor = newline + 1
        if not line.rstrip("\r").rstrip().endswith("\\"):
            return cursor
    return cursor


def _looks_like_c_top_level_start(text: str) -> bool:
    """Recognize a likely next translation-unit declaration.

    The check is deliberately conservative. Once a complete top-level
    declaration has been found, ordinary English should terminate extraction;
    only a line that resembles another C declaration is allowed to extend it.
    """

    return _C_TOP_LEVEL_START_RE.match(text) is not None


def _looks_like_c_post_brace_declarator(text: str) -> bool:
    """Recognize ``} name;`` tails of typedef/struct declarations."""

    return _C_POST_BRACE_DECL_RE.match(text) is not None


def _preserve_c_transform_meta_tail(text: str, code_end: int) -> str:
    """Keep legal trailing TRANSFORM_META comments while dropping prose."""

    cursor = code_end
    output_end = code_end
    saw_meta = False
    length = len(text)

    while cursor < length:
        whitespace = re.match(r"[ \t\r\n]*", text[cursor:])
        cursor += whitespace.end() if whitespace else 0
        if cursor >= length:
            break

        line_end = text.find("\n", cursor)
        if line_end < 0:
            line_end = length
        else:
            line_end += 1
        line = text[cursor:line_end]
        if _C_TRANSFORM_META_LINE_RE.match(line):
            output_end = line_end
            cursor = line_end
            saw_meta = True
            continue

        block = _C_TRANSFORM_META_BLOCK_RE.match(text[cursor:])
        if block:
            output_end = cursor + block.end()
            cursor = output_end
            saw_meta = True
            continue

        break

    return text[:output_end].rstrip() if saw_meta else text[:code_end].rstrip()


def _trim_c_translation_unit(candidate: str) -> str:
    """Trim a C candidate at the last complete top-level declaration.

    This is a lexical boundary finder rather than a C parser. It tracks
    comments, strings, character literals, and brace nesting, so braces or
    semicolons inside those constructs cannot terminate the translation unit.
    It also stops before an English paragraph once at least one complete
    top-level declaration has been seen.
    """

    text = candidate
    length = len(text)
    depth = 0
    state = "code"
    line_start = True
    last_code_end: int | None = None
    top_level_decl_started = False
    just_closed_top_level_brace = False
    cursor = 0

    while cursor < length:
        char = text[cursor]

        if state == "line_comment":
            if char in "\r\n":
                state = "code"
                line_start = True
            cursor += 1
            continue

        if state == "block_comment":
            if text.startswith("*/", cursor):
                state = "code"
                cursor += 2
                continue
            if char in "\r\n":
                line_start = True
            cursor += 1
            continue

        if state == "string":
            if char == "\\":
                if cursor + 1 < length:
                    if text[cursor + 1] in "\r\n":
                        line_start = True
                    cursor += 2
                else:
                    cursor += 1
                continue
            if char == '"':
                state = "code"
                cursor += 1
                line_start = False
                continue
            if char in "\r\n":
                line_start = True
            else:
                line_start = False
            cursor += 1
            continue

        if state == "char":
            if char == "\\":
                if cursor + 1 < length:
                    if text[cursor + 1] in "\r\n":
                        line_start = True
                    cursor += 2
                else:
                    cursor += 1
                continue
            if char == "'":
                state = "code"
                cursor += 1
                line_start = False
                continue
            if char in "\r\n":
                line_start = True
            else:
                line_start = False
            cursor += 1
            continue

        # Preprocessor directives can contain arbitrary braces and semicolons;
        # skip them as one lexical unit when '#' starts a physical line.
        if char == "#" and line_start:
            cursor = _consume_c_preprocessor_line(text, cursor)
            last_code_end = cursor
            top_level_decl_started = False
            just_closed_top_level_brace = False
            line_start = True
            continue

        if text.startswith("//", cursor):
            state = "line_comment"
            cursor += 2
            continue
        if text.startswith("/*", cursor):
            state = "block_comment"
            cursor += 2
            continue
        if char == '"':
            state = "string"
            line_start = False
            cursor += 1
            continue
        if char == "'":
            state = "char"
            line_start = False
            cursor += 1
            continue

        # After a complete declaration, a non-C token is the beginning of
        # trailing prose. Comments and whitespace are still scanned above.
        if (
            depth == 0
            and last_code_end is not None
            and not top_level_decl_started
            and not char.isspace()
        ):
            preceding = text[:cursor].rstrip()
            token_start = (
                line_start
                or preceding.endswith((";", "}"))
                or preceding.endswith("*/")
            )
            # A closing brace in a struct/enum/initializer is commonly
            # followed by its declaration's semicolon, possibly after spaces.
            valid_next_declaration = _looks_like_c_top_level_start(text[cursor:])
            valid_post_brace_tail = just_closed_top_level_brace and (
                _looks_like_c_post_brace_declarator(text[cursor:])
            )
            if (
                token_start
                and char != ";"
                and not valid_next_declaration
                and not valid_post_brace_tail
            ):
                break

        # Once the first token of a top-level declaration is accepted, do not
        # reinterpret later lines of its signature as trailing prose.
        if depth == 0 and not top_level_decl_started and not char.isspace():
            if char != ";":
                top_level_decl_started = True
                just_closed_top_level_brace = False

        if char == "{":
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    last_code_end = cursor + 1
                    top_level_decl_started = False
                    just_closed_top_level_brace = True
        elif char == ";" and depth == 0:
            last_code_end = cursor + 1
            top_level_decl_started = False
            just_closed_top_level_brace = False

        if char in "\r\n":
            line_start = True
        elif not (line_start and char in " \t\f\v"):
            line_start = False
        cursor += 1

    # Do not hide an unfinished declaration behind an earlier include or
    # global declaration. Returning the original candidate lets the compiler
    # and retry path report the real syntax error; the sanitizer never invents
    # a missing suffix.
    if last_code_end is None or top_level_decl_started or depth != 0:
        return text.strip()
    return _preserve_c_transform_meta_tail(text, last_code_end)


def sanitize_c_response(raw_text: str) -> str:
    """Return one C translation unit and discard fences or trailing prose."""

    text = _strip_think_blocks((raw_text or "").strip())
    if not text:
        return ""

    prefix = _prefix_before_fence(text)
    if prefix and _C_MARKER_RE.search(prefix):
        candidate = _trim_c_translation_unit(_raw_c_candidate(prefix))
        if candidate and _C_MARKER_RE.search(candidate):
            return candidate.strip()

    for _start, _end, language, body in reversed(_fenced_blocks(text)):
        if language not in _C_LANGUAGES:
            continue
        candidate = _trim_c_translation_unit(_trim_c_candidate(body))
        if _C_MARKER_RE.search(candidate):
            return candidate.strip()

    return _trim_c_translation_unit(_raw_c_candidate(text)).strip()


def _last_complete_module(candidate: str) -> str:
    """For each endmodule, use the module start immediately before it."""

    starts = list(_MODULE_START_RE.finditer(candidate))
    ends = list(_ENDMODULE_RE.finditer(candidate))
    for end in reversed(ends):
        preceding = [start for start in starts if start.start() < end.start()]
        if preceding:
            start = preceding[-1]
            return candidate[start.start() : end.end()].strip()
    if starts:
        return candidate[starts[-1].start() :].strip()
    return candidate.strip()


def sanitize_verilog_response(raw_text: str) -> str:
    """Return exactly one Verilog module from a model response."""

    text = _strip_think_blocks((raw_text or "").strip())
    if not text:
        return ""

    for _start, _end, language, body in reversed(_fenced_blocks(text)):
        if language not in _VERILOG_LANGUAGES:
            continue
        if _ENDMODULE_RE.search(body) and _MODULE_START_RE.search(body):
            return _last_complete_module(body)

    return _last_complete_module(text)
