#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import subprocess
import shutil
import tempfile
from pathlib import Path

try:
    from .tools.llm_api import call_llm
except ImportError:
    from tools.llm_api import call_llm

try:
    from .verilog7_mov import collect_syn_verilog
except ImportError:
    from verilog7_mov import collect_syn_verilog


# 基于当前工作目录定位项目文件/目录（避免硬编码绝对路径）
_BASE_DIR = Path.cwd().resolve()


AREA = "hls_opt_area_prompt.txt"
POWER = "hls_opt_power_prompt.txt"
TIMING = "hls_opt_timing_prompt.txt"

ROOT_DIR = str((_BASE_DIR / "benchmark_ai").resolve())
PROMPT_TEMPLATE_PATH = str((_BASE_DIR / "tools").resolve())
_PROMPT_DIR = PROMPT_TEMPLATE_PATH
HLS_PART = "xck24-ubva530-2LV-c"
HLS_CLOCK = "10ns"
OUTPUT_FINAL_DIR = str((_BASE_DIR / "output_final").resolve())


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="POWER",
        help="prompt 选项：POWER/TIMING/AREA，或直接给 prompt 文件名/路径"
    )
    return parser.parse_args()

def _resolve_prompt_path(prompt_arg: str) -> str | None:
    """
    将 --prompt 参数解析为实际 prompt 文件路径。

    支持：
    - POWER/TIMING/AREA
    - 直接给文件名（默认在 tools/ 下查找）
    - 直接给相对/绝对路径
    """
    if not prompt_arg:
        return None

    key = prompt_arg.strip()
    upper = key.upper()
    if upper == "POWER":
        name = POWER
    elif upper == "TIMING":
        name = TIMING
    elif upper == "AREA":
        name = AREA
    else:
        name = key

    # 1) 若用户给的是现成可用路径，直接用
    if os.path.isfile(name):
        return os.path.abspath(name)

    # 2) 否则默认在 tools/ 下查找
    cand = os.path.join(_PROMPT_DIR, name)
    if os.path.isfile(cand):
        return os.path.abspath(cand)

    return None


def _prompt_tag(prompt_arg: str, prompt_path: str) -> str:
    """
    生成用于输出文件命名的 tag：<TAG>_hls_opt。

    规则：
    - POWER/TIMING/AREA -> 对应大写
    - 其它 -> 取 prompt 文件 stem，并做简单清洗（去掉 hls_opt_ / _prompt 等前后缀）
    """
    key = (prompt_arg or "").strip()
    upper = key.upper()
    if upper in ("POWER", "TIMING", "AREA"):
        return upper

    stem = Path(prompt_path).stem
    if stem.startswith("hls_opt_"):
        stem = stem[len("hls_opt_") :]
    if stem.endswith("_prompt"):
        stem = stem[: -len("_prompt")]
    return stem or "PROMPT"


def _safe_dir_suffix(tag: str) -> str:
    """
    将 tag 清洗为适合目录名的后缀（仅保留字母/数字/下划线/中划线）。
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (tag or "").strip())
    cleaned = cleaned.strip("_-")
    return cleaned or "PROMPT"



def iter_cpp_files(root_dir: str):
    """
    递归遍历 root_dir 下所有子目录，找到所有 .cpp 文件（默认跳过已优化产物）。
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in (".git", ".idea", "__pycache__", "ai_opt", "cdfg")
            and not (d == "hls" or d.startswith("hls_"))
        ]
        for filename in filenames:
            low = filename.lower()
            if not low.endswith(".cpp"):
                continue
            # 第一阶段 AI 优化不处理任何已优化产物
            if low.endswith("_opt.cpp") or low.endswith("_hls_opt.cpp"):
                continue
            yield os.path.join(dirpath, filename)

def iter_hls_opt_cpp_files(root_dir: str):
    """
    递归遍历 root_dir 下所有子目录，找到所有 *_hls_opt.cpp 文件。
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in (".git", ".idea", "__pycache__", "ai_opt", "cdfg")
            and not (d == "hls" or d.startswith("hls_"))
        ]
        for filename in filenames:
            if filename.lower().endswith("_hls_opt.cpp"):
                yield os.path.join(dirpath, filename)


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_first_cpp_code_block(text: str) -> str | None:
    """
    从 LLM 输出中提取第一个 ```cpp ... ``` 代码块。
    """
    m = re.search(r"```cpp\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _find_matching_brace(text: str, open_idx: int) -> int | None:
    """
    给定 '{' 的位置，返回匹配的 '}' 位置；失败返回 None。
    处理字符串与注释，避免误计数。
    """
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "{":
        return None

    i = open_idx + 1
    depth = 1
    in_str = False
    in_ch = False
    in_line_comment = False
    in_block_comment = False

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

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
        if in_str:
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if in_ch:
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == "'":
                in_ch = False
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
            in_str = True
            i += 1
            continue
        if ch == "'":
            in_ch = True
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _remove_main_functions(code: str) -> str:
    """
    删除文件中的 main 函数（int/void/main with qualifiers），避免 HLS 将其当作无关 testbench 代码。
    """
    text = code
    pattern = re.compile(r"\bmain\s*\(")
    pos = 0
    while True:
        m = pattern.search(text, pos)
        if not m:
            break

        # 排除函数调用：前一个非空白字符若为 '.' '>' ':' '_' 等通常不是定义
        j = m.start() - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j >= 0 and text[j] in ".>:_$":
            pos = m.end()
            continue

        # 找 ')' 并确认后续是 '{'
        lpar = text.find("(", m.start())
        if lpar == -1:
            pos = m.end()
            continue
        depth = 1
        i = lpar + 1
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        if depth != 0:
            pos = m.end()
            continue

        k = i
        while k < len(text) and text[k].isspace():
            k += 1
        if k >= len(text) or text[k] != "{":
            pos = m.end()
            continue

        end_brace = _find_matching_brace(text, k)
        if end_brace is None:
            pos = m.end()
            continue

        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = end_brace + 1
        if line_end < len(text) and text[line_end] == "\n":
            line_end += 1

        text = text[:line_start] + text[line_end:]
        pos = line_start
    return text


def sanitize_hls_cpp_code(raw_code: str, top_name: str) -> str:
    """
    对 LLM 输出进行轻量清洗，使其更接近 HLS 可接受输入。
    """
    code = (raw_code or "").replace("\r\n", "\n")
    code = code.lstrip("\ufeff")

    # 清除 markdown 围栏
    code = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\s*$", "", code, flags=re.M)

    # 去除常见的非综合/无关头文件，减少 HLS 前端差异
    code = re.sub(r'^[ \t]*#include[ \t]*<stdio\.h>.*$', "", code, flags=re.M)
    code = re.sub(r'^[ \t]*#include[ \t]*<assert\.h>.*$', "", code, flags=re.M)
    code = re.sub(r'^[ \t]*#include[ \t]*<cstdio>.*$', "", code, flags=re.M)
    code = re.sub(r'^[ \t]*#include[ \t]*<cassert>.*$', "", code, flags=re.M)

    # 统一布尔类型与常量
    code = re.sub(r"\b_Bool\b", "bool", code)
    code = re.sub(r"\bTRUE\b", "true", code)
    code = re.sub(r"\bFALSE\b", "false", code)

    # 去掉对 C++ 内建 bool/true/false 的重定义，避免 HLS 前端报错
    code = re.sub(r"^[ \t]*#define[ \t]+true\b.*$", "", code, flags=re.M)
    code = re.sub(r"^[ \t]*#define[ \t]+false\b.*$", "", code, flags=re.M)
    code = re.sub(r"^[ \t]*#define[ \t]+bool\b.*$", "", code, flags=re.M)
    code = re.sub(r"^[ \t]*typedef[ \t]+[^;\n]*\bbool\b\s*;\s*$", "", code, flags=re.M)

    # 删除不适合 HLS 的 clock/reset interface pragma
    code = re.sub(r"^[ \t]*#pragma[ \t]+HLS[ \t]+INTERFACE[ \t]+ap_clk\b.*$", "", code, flags=re.M)
    code = re.sub(r"^[ \t]*#pragma[ \t]+HLS[ \t]+INTERFACE[ \t]+ap_rst\b.*$", "", code, flags=re.M)

    # 删除 testbench main
    code = _remove_main_functions(code)

    # 压缩过多空行
    code = re.sub(r"\n{3,}", "\n\n", code).strip() + "\n"

    # 若没有任何函数定义，直接返回原文（让上层按失败处理）
    if not re.search(r"\b[A-Za-z_]\w*\s*\([^;{}]*\)\s*\{", code):
        return raw_code if raw_code.endswith("\n") else (raw_code + "\n")

    # 顶层函数至少应可见（按文件名推断）
    if top_name and not re.search(rf"\b{re.escape(top_name)}\s*\(", code):
        print(f"[警告] 优化代码中未找到推断 top 函数名: {top_name}")

    return code


def _local_cpp_syntax_ok(code: str) -> bool:
    """
    用本地 clang++ 做快速语法预检，避免把明显错误代码送给 v++。
    """
    clangpp = shutil.which("clang++")
    if not clangpp:
        return True

    fd, tmp_path = tempfile.mkstemp(suffix=".cpp", prefix="hls_precheck_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        p = subprocess.run(
            [clangpp, "-std=c++14", "-fsyntax-only", tmp_path],
            capture_output=True,
            text=True,
        )
        return p.returncode == 0
    except OSError:
        return True
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def optimize_cpp_for_hls(
    cpp_path: str,
    prompt_template_path: str | None = None,
    out_suffix: str = "_hls_opt.cpp",
    model: str = "gemini-2.5-pro",
    temperature: float = 0.3,
) -> str | None:
    """
    对单个 .cpp 进行 HLS 方向 AI 优化，并生成新的 .cpp 文件。

    产物：
    - <stem>_hls_opt.cpp（提取到的优化后代码）
    - ai_opt/<stem>/prompt.txt（实际发送给 LLM 的 prompt）
    - ai_opt/<stem>/response.txt（LLM 原始返回）
    """
    cpp_path = os.path.abspath(cpp_path)
    if not os.path.isfile(cpp_path):
        print(f"[错误] cpp 不存在: {cpp_path}")
        return None

    if not prompt_template_path:
        prompt_template_path = os.path.join(_PROMPT_DIR, POWER)

    template = load_prompt_template(prompt_template_path)
    if "{CPP_CODE}" not in template:
        raise ValueError(f"prompt 模板缺少 {{CPP_CODE}} 占位符: {prompt_template_path}")

    with open(cpp_path, "r", encoding="utf-8", errors="ignore") as f:
        cpp_code = f.read()

    prompt = template.replace("{CPP_CODE}", cpp_code)

    stem = Path(cpp_path).stem
    out_dir = os.path.dirname(cpp_path)
    out_cpp = os.path.join(out_dir, f"{stem}{out_suffix}")

    audit_dir = os.path.join(out_dir, "ai_opt", stem)
    os.makedirs(audit_dir, exist_ok=True)
    with open(os.path.join(audit_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"[AI] 优化: {cpp_path}")
    response = call_llm(prompt, model=model, temperature=temperature, images=None, timeout=120)
    with open(os.path.join(audit_dir, "response.txt"), "w", encoding="utf-8") as f:
        f.write(response or "")

    if (response or "").strip().startswith("OpenAI API error:"):
        print(f"[错误] LLM 调用失败，跳过该文件: {cpp_path}")
        return None

    optimized_code = extract_first_cpp_code_block(response or "")
    if not optimized_code:
        print(f"[警告] 未提取到 cpp 代码块，跳过该文件: {cpp_path}")
        return None

    top_name = _infer_hls_top_from_cpp_path(out_cpp)
    optimized_code = sanitize_hls_cpp_code(optimized_code, top_name=top_name)
    fallback_code = sanitize_hls_cpp_code(cpp_code, top_name=top_name)

    # 优先用 AI 代码；若本地语法预检失败，则回退到原始代码清洗版
    if not _local_cpp_syntax_ok(optimized_code):
        print(f"[警告] AI 优化代码语法预检失败，回退原始代码: {cpp_path}")
        optimized_code = fallback_code

    # 回退后仍失败，说明源代码本身也异常，跳过
    if not _local_cpp_syntax_ok(optimized_code):
        print(f"[错误] 回退代码仍语法失败，跳过该文件: {cpp_path}")
        return None

    with open(out_cpp, "w", encoding="utf-8") as f:
        f.write(optimized_code)
        if not optimized_code.endswith("\n"):
            f.write("\n")

    print(f"[完成] 已生成: {out_cpp}")
    return out_cpp


def _infer_hls_top_from_cpp_path(cpp_path: str) -> str:
    """
    默认规则：
    - <name>.cpp -> top=<name>
    - <name>_hls_opt.cpp -> top=<name>
    """
    stem = Path(cpp_path).stem
    if stem.endswith("_hls_opt"):
        base = stem[: -len("_hls_opt")]
        # 兼容：<name>_<PROMPT>_hls_opt.cpp -> top=<name>
        for tag in ("POWER", "TIMING", "AREA"):
            if base.endswith(f"_{tag}") or base.endswith(f"_{tag.lower()}"):
                return base[: -(len(tag) + 1)]
        return base
    return stem


def write_hls_config(cfg_path: str, cpp_path: str, top: str) -> None:
    cfg_text = "\n".join( # 每一行配置都用“\n连接”
        [
            "# hls_config.cfg",
            f"part={HLS_PART}",
            "",
            "[hls]",
            "flow_target=vitis",
            f"clock={HLS_CLOCK}",
            f"syn.file={os.path.abspath(cpp_path)}",
            f"syn.top={top}",
            "",
            "# 生成 RTL",
            "syn.output.format=rtl",
            "",
        ]
    )
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg_text)


def run_vpp_hls(cpp_path: str, hls_dir_name: str = "hls") -> bool:
    """
    对单个 *_hls_opt.cpp 执行 v++ HLS，生成 RTL。

    目录结构（在 cpp 同级目录下）：
    - <hls_dir_name>/hls_config.cfg
    - <hls_dir_name>/hls_work/...
    - <hls_dir_name>/vpp_stdout.txt
    - <hls_dir_name>/vpp_stderr.txt
    """
    cpp_path = os.path.abspath(cpp_path)
    if not os.path.isfile(cpp_path):
        print(f"[错误] cpp 不存在: {cpp_path}")
        return False

    top = _infer_hls_top_from_cpp_path(cpp_path)

    hls_dir = os.path.join(os.path.dirname(cpp_path), hls_dir_name)
    os.makedirs(hls_dir, exist_ok=True)

    cfg_path = os.path.join(hls_dir, "hls_config.cfg")
    write_hls_config(cfg_path, cpp_path, top)

    work_dir = os.path.join(hls_dir, "hls_work")
    summary_path = os.path.join(work_dir, "hls_work.hlscompile_summary")
    if os.path.isfile(summary_path):
        print(f"[跳过] 已存在 HLS summary: {summary_path}")
        return True

    cmd = ["v++", "-c", "--mode", "hls", "--config", "./hls_config.cfg", "--work_dir", "./hls_work"]
    print(f"[HLS] 运行: {' '.join(cmd)}  (cwd={hls_dir}, top={top})")

    stdout_path = os.path.join(hls_dir, "vpp_stdout.txt")
    stderr_path = os.path.join(hls_dir, "vpp_stderr.txt")
    with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
        p = subprocess.run(cmd, cwd=hls_dir, stdout=out, stderr=err, text=True)

    if p.returncode != 0:
        print(f"[错误] v++ HLS 失败: {cpp_path} (rc={p.returncode})")
        print(f"       详情: {stderr_path}")
        return False

    if os.path.isfile(summary_path):
        print(f"[完成] HLS summary: {summary_path}")
    else:
        print(f"[完成] v++ 返回成功，但未发现 summary: {summary_path}")
    return True


def clear_hls_output_dirs(root_dir: str, hls_dir_name: str) -> int:
    """
    清理 root_dir 下所有名为 hls_dir_name 的旧目录，避免复用历史 HLS 产物。
    """
    root_dir = os.path.abspath(root_dir)
    removed = 0

    for dirpath, dirnames, _ in os.walk(root_dir):
        if hls_dir_name not in dirnames:
            continue
        target = os.path.join(dirpath, hls_dir_name)
        try:
            shutil.rmtree(target)
            removed += 1
            print(f"[清理] 已删除旧目录: {target}")
        except OSError as e:
            print(f"[警告] 删除旧目录失败: {target}: {e}")
        # 避免继续进入已删除目录
        if hls_dir_name in dirnames:
            dirnames.remove(hls_dir_name)

    return removed


def main():
    args = parse_args()
    
    if not os.path.isdir(ROOT_DIR):
        print(f"目录不存在: {ROOT_DIR}")
        return
    prompt_path = _resolve_prompt_path(args.prompt)
    if not prompt_path:
        print(f"prompt 不存在: {args.prompt}（已尝试在 tools/ 下查找）")
        return
    tag = _prompt_tag(args.prompt, prompt_path)
    hls_suffix = _safe_dir_suffix(tag)
    hls_dir_name = f"hls_{hls_suffix}"
    out_suffix = f"_{tag}_hls_opt.cpp"

    # 1) 优先对原始 .cpp 进行 AI 优化，产出 *_hls_opt.cpp（如果已存在则跳过）
    optimized: set[str] = set()
    ai_count = 0
    for cpp_file in iter_cpp_files(ROOT_DIR):
        # 第一阶段 AI 优化不处理任何已优化产物（兜底；iter_cpp_files 也会跳过）
        if cpp_file.lower().endswith("_hls_opt.cpp"):
            continue
        ai_count += 1
        out = optimize_cpp_for_hls(
            cpp_path = cpp_file, 
            prompt_template_path = prompt_path,
            out_suffix=out_suffix,
            )
        if out:
            optimized.add(out)
    print(f"完成 AI 优化，共处理 {ai_count} 个 .cpp 文件")

    # 2) 输出到本次 prompt 对应的 hls_* 目录前，先清理同名旧目录
    removed = clear_hls_output_dirs(ROOT_DIR, hls_dir_name)
    print(f"[清理] 共删除 {removed} 个旧 {hls_dir_name} 目录")

    # 3) 对所有 *_hls_opt.cpp 执行 v++ HLS（包含本次新生成和历史已有）
    hls_count = 0
    hls_ok = 0
    for cpp_file in iter_hls_opt_cpp_files(ROOT_DIR):
        # 用户需求：不处理 /cdfg/ 下的 cpp
        if "/cdfg/" in cpp_file.replace("\\", "/"):
            continue
        # 只对本次 prompt tag 的产物做 HLS
        if not cpp_file.endswith(out_suffix):
            continue
        optimized.add(os.path.abspath(cpp_file))

    for cpp_file in sorted(optimized):
        hls_count += 1
        if run_vpp_hls(cpp_file, hls_dir_name=hls_dir_name):
            hls_ok += 1

    print(f"完成 HLS，共处理 {hls_count} 个优化后 .cpp，成功 {hls_ok} 个（目录: {hls_dir_name}）")

    # 4) 收集 syn 的 Verilog 网表到 output_final/<project>/<PROMPT_TAG>/
    collect_syn_verilog(ROOT_DIR, OUTPUT_FINAL_DIR, hls_dir_name=hls_dir_name, project_subdir=hls_suffix)


if __name__ == "__main__":
    main()
