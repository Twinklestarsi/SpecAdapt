#!/usr/bin/env python3
# coding:utf-8

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple

sys.dont_write_bytecode = True

try:
    from .llm_api import call_llm
    from .prompt_loader import PromptLoader
except ImportError:
    from llm_api import call_llm
    from prompt_loader import PromptLoader

_MODULE_RE = re.compile(r"^\s*module\s+([a-zA-Z_]\w*)\b", re.MULTILINE)

def detect_top_module_from_verilog(verilog_text: str) -> str | None:
    m = _MODULE_RE.search(verilog_text)
    return m.group(1) if m else None


def map_path_for_dc(local_path: str) -> str:
    """
    若设置了 DC_LOCAL_ROOT/DC_REMOTE_ROOT，则将本地路径映射到 DC 运行环境路径。
    否则返回原路径。
    """
    local_root = os.getenv("DC_LOCAL_ROOT")
    remote_root = os.getenv("DC_REMOTE_ROOT")
    if local_root and remote_root and local_path.startswith(local_root):
        return remote_root + local_path[len(local_root) :]
    return local_path

def _read_text(path: Path, limit: int | None = None) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    return data if limit is None else data[:limit]

def _render_template(template_text: str, variables: Dict[str, Any]) -> str:
    rendered = template_text
    for key, value in variables.items():
        pattern = rf"(?<!\$)\{{{re.escape(key)}\}}"
        rendered = re.sub(pattern, str(value), rendered)
    return rendered


def generate_sdc_for_verilog(
    verilog_content: str,
    design_name: str,
    top_module: str,
    prompt_file_path: str,
) -> str:
    """
    按 MARO 的写法：要求 LLM 返回 JSON，并从 timing_sdc 字段取出 SDC 内容。
    """
    prompt_loader = PromptLoader(prompt_file_path)
    prompt = prompt_loader.get_prompt(
        "sdc_generator",
        design_name=design_name,
        top_module=top_module,
        verilog_content=verilog_content,
    )
    model_config = prompt_loader.get_model_config("sdc_generator")
    full_prompt = prompt + "\n\nPlease strictly return a valid JSON format string, for example: \n{\"timing_sdc\": \"...\"}"
    response = call_llm(full_prompt, model=model_config["model"], temperature=model_config["temperature"])

    text = (response or "").strip()
    clean_json = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean_json)
        sdc = (data.get("timing_sdc") or "").strip()
        return sdc if sdc else "# Error generating timing_sdc\n"
    except json.JSONDecodeError:
        # 兼容模型不按 JSON 返回的情况：退化为直接使用原始输出
        fallback = text.replace("```", "").strip()
        return fallback if fallback else "# SDC generation failed or returned empty content\n"


def generate_tcl_from_template(
    template_path: Path,
    out_tcl_path: Path,
    *,
    top_module: str,
    design_name: str,
    rtl_path: str,
    rtl_files: str,
    sdc_path: str | None,
    search_path: str = "",
    target_library: str = "",
    rtl_format: str = "verilog",
    **extra_vars,
) -> None:
    template_text = _read_text(template_path)
    sdc_section = f"source {sdc_path}" if sdc_path else "# No SDC file"
    variables = {
        "top_module": top_module,
        "design_name": design_name,
        "rtl_path": rtl_path,
        "rtl_files": rtl_files,
        "rtl_format": rtl_format,
        "sdc_section": sdc_section,
        "search_path": search_path,
        "target_library": target_library,
        "max_cores": 1,
    }
    variables.update(extra_vars)
    rendered = _render_template(template_text, variables)
    with open(out_tcl_path, "w", encoding="utf-8") as f:
        f.write(rendered)


def generate_tcl_sdc_for_verilog(
    verilog_file: str | Path,
    *,
    out_dir: str | Path | None = None,
    design_name: str | None = None,
    top_module: str | None = None,
    prompt_file_path: str | Path | None = None,
    template_file_path: str | Path | None = None,
    verilog_trunc_limit: int = 4000,
    use_dc_path_mapping: bool = True,
    skip_llm: bool = False,
) -> Tuple[Path, Path]:
    """
    为单个 .v 文件生成：
    - <name>.sdc
    - <name>.tcl
    """
    verilog_path = Path(verilog_file).resolve()
    if not verilog_path.exists():
        raise FileNotFoundError(f"Verilog not found: {verilog_path}")

    out_dir_path = Path(out_dir).resolve() if out_dir else verilog_path.parent
    out_dir_path.mkdir(parents=True, exist_ok=True)

    design = design_name or verilog_path.stem
    primary_text = _read_text(verilog_path, limit=verilog_trunc_limit)
    top = top_module or detect_top_module_from_verilog(primary_text) or design

    # prompt 只读取当前输入的 .v 文件内容
    verilog_text = primary_text

    prompt_file = Path(prompt_file_path) if prompt_file_path else (Path(__file__).parent / "agent_prompts.json")
    template_file = Path(template_file_path) if template_file_path else (Path(__file__).parent / "synthesis_template.tcl")

    sdc_path = out_dir_path / f"{design}.sdc"
    tcl_path = out_dir_path / f"{design}.tcl"

    if skip_llm:
        sdc_text = "# SDC skipped (skip_llm=true)\n"
    else:
        sdc_text = generate_sdc_for_verilog(verilog_text, design, top, str(prompt_file))

    with open(sdc_path, "w", encoding="utf-8") as f:
        f.write(sdc_text)
        if not sdc_text.endswith("\n"):
            f.write("\n")

    rtl_dir_local = str(verilog_path.parent.resolve())
    rtl_dir_dc = map_path_for_dc(rtl_dir_local) if use_dc_path_mapping else rtl_dir_local
    sdc_path_dc = map_path_for_dc(str(sdc_path.resolve())) if use_dc_path_mapping else str(sdc_path.resolve())

    # DC analyze 支持传 list：这里将同目录下所有 .v 作为输入，保证依赖模块可见
    rtl_files: list[str] = []
    for vf in sorted(verilog_path.parent.glob("*.v")):
        rtl_files.append(map_path_for_dc(str(vf.resolve())) if use_dc_path_mapping else str(vf.resolve()))
    rtl_files_expr = "[list " + " ".join(rtl_files) + "]"
    rtl_format = "sverilog" if verilog_path.suffix.lower() == ".sv" else "verilog"
    generate_tcl_from_template(
        Path(template_file),
        tcl_path,
        top_module=top,
        design_name=design,
        rtl_path=rtl_dir_dc,
        rtl_files=rtl_files_expr,
        sdc_path=sdc_path_dc,
        rtl_format=rtl_format,
        max_cores=1,
    )

    return tcl_path, sdc_path
