#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import shlex
import sys

from dotenv import load_dotenv

from project_paths import PROJECT_ROOT

sys.dont_write_bytecode = True

load_dotenv(PROJECT_ROOT / ".env", override=False)
os.environ.setdefault("DC_LOCAL_ROOT", str(PROJECT_ROOT))
if os.environ.get("DC_REMOTE_BASE"):
    os.environ.setdefault("DC_REMOTE_ROOT", os.environ["DC_REMOTE_BASE"])

_BASE_DIR = PROJECT_ROOT
OUTPUT_FINAL_DIR = str((_BASE_DIR / "output_final").resolve())


def _load_tcl_agent():
    tools_dir = Path(__file__).resolve().parent / "tools"
    sys.path.insert(0, str(tools_dir))
    import tcl_agent  # type: ignore
    return tcl_agent


def run_bash_command(cmd: str):
    """
    运行一条 bash 命令，返回 (ok, output)。
    """
    import subprocess

    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode == 0, out


def _check_dc():
    ok, _ = run_bash_command("command -v dc_shell")
    if not ok:
        raise Exception("dc_shell not found in PATH")


def run_tcl_script(syn_design_dir: str, script_path: str):
    """
    在 syn_design_dir 中运行 dc_shell -f <script>，输出 dc.log 与 syn_output/* 到该目录。
    设置 DC_REMOTE_USER/DC_REMOTE_HOST 后通过 SSH 执行；DC_SSH_HOST
    仅作为旧配置兼容。路径按 DC_LOCAL_ROOT/DC_REMOTE_ROOT 映射。
    """
    remote = os.getenv("DC_SSH_HOST", "").strip()
    if not remote:
        remote_user = os.getenv("DC_REMOTE_USER", "").strip()
        remote_host = os.getenv("DC_REMOTE_HOST", "").strip()
        if remote_user and remote_host:
            remote = f"{remote_user}@{remote_host}"
    if remote:
        local_root = os.getenv("DC_LOCAL_ROOT")   # 本机share根
        remote_root = os.getenv("DC_REMOTE_ROOT") # 远程share根
        if not local_root or not remote_root:
            raise Exception("DC_LOCAL_ROOT/DC_REMOTE_ROOT is required when using DC_SSH_HOST")

        rel = os.path.relpath(syn_design_dir, local_root)
        remote_dir = os.path.join(remote_root, rel)
        log_path = os.path.join(syn_design_dir, "dc.log")
        script_name = os.path.basename(script_path)
        dc_env_script = os.getenv("DC_ENV_SCRIPT", "").strip()
        dc_shell_path = os.getenv("DC_SHELL_PATH", "dc_shell").strip() or "dc_shell"
        steps = []
        if dc_env_script:
            steps.append(f"source {shlex.quote(dc_env_script)}")
        steps.extend(
            [
                f"cd {shlex.quote(remote_dir)}",
                f"{shlex.quote(dc_shell_path)} -f {shlex.quote(script_name)} | tee dc.log",
            ]
        )
        remote_cmd = " && ".join(steps)
        cmd = f"ssh {shlex.quote(remote)} {shlex.quote(remote_cmd)}"
        ok, _ = run_bash_command(cmd)
        if not ok:
            raise Exception(f"Failed to run dc_shell script {script_path} on {remote}")
        return log_path

    # 没配置远程时，保留原来的本地调用逻辑
    _check_dc()
    log_path = os.path.join(syn_design_dir, "dc.log")
    script_name = os.path.basename(script_path)
    cmd = f'cd "{syn_design_dir}" && dc_shell -f "{script_name}" | tee "{log_path}"'
    ok, _ = run_bash_command(cmd)
    if not ok:
        raise Exception(f"Failed to run dc_shell script {script_path}")
    return log_path


def main():
    import argparse

    tcl_agent = _load_tcl_agent()

    parser = argparse.ArgumentParser(description="Generate per-Verilog DC TCL/SDC under output_final")
    parser.add_argument("--root", default=str(OUTPUT_FINAL_DIR), help="Root directory containing .v files")
    parser.add_argument("--skip-llm", action="store_true", help="Generate placeholder SDC without calling LLM")
    parser.add_argument("--run-dc", action="store_true", help="Run dc_shell for each generated tcl")
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    if not root_dir.is_dir():
        print(f"目录不存在: {root_dir}")
        return

    count = 0
    for v_path in root_dir.rglob("*.v"):
        if not v_path.is_file():
            continue

        # 在同目录创建与 .v 同名的文件夹，并在该文件夹内输出 <name>.tcl / <name>.sdc / syn_output/*
        out_dir = v_path.parent / v_path.stem
        design_name = v_path.stem
        try:
            tcl_path, sdc_path = tcl_agent.generate_tcl_sdc_for_verilog(
                v_path,
                out_dir=out_dir,
                design_name=design_name,
                top_module=None,
                prompt_file_path=str(Path(__file__).resolve().parent / "tools" / "agent_prompts.json"),
                template_file_path=str(Path(__file__).resolve().parent / "tools" / "synthesis_template.tcl"),
                use_dc_path_mapping=True,
                skip_llm=args.skip_llm,
            )
            count += 1
            print(f"[OK] {v_path} -> {tcl_path.name}, {sdc_path.name}")
            
            log_path = run_tcl_script(str(out_dir), str(tcl_path))
            print(f"[DC] log: {log_path}")
        except Exception as e:
            print(f"[FAIL] {v_path}: {e}")

    print(f"完成：共处理 {count} 个 .v 文件")


if __name__ == "__main__":
    main()
