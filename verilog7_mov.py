#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path

_BASE_DIR = Path.cwd().resolve()
DEFAULT_ROOT_DIR = str((_BASE_DIR / "benchmark_ai").resolve())
DEFAULT_OUTPUT_DIR = str((_BASE_DIR / "output_final").resolve())


def collect_syn_verilog(
    root_dir: str = DEFAULT_ROOT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    hls_dir_name: str = "hls",
    project_subdir: str = "prompt",
) -> int:
    """
    从 root_dir 下每个项目提取 HLS 的 syn/verilog/*.v 文件，并复制到 output_dir/<项目名>/<project_subdir>/。

    识别规则：
    - 在 root_dir 下查找形如 */<hls_dir_name>/hls_work/hls/syn/verilog/*.v 的文件
    - 项目名取 root_dir 的一级子目录名（例如 stream_pipe_task_061_Y）
    """
    root_dir = os.path.abspath(root_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    copied = 0
    manifest_lines: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in (".git", ".idea", "__pycache__", "ai_opt")]

        target_suffix = f"/{hls_dir_name}/hls_work/hls/syn/verilog"
        if not dirpath.replace("\\", "/").endswith(target_suffix):
            continue

        rel = os.path.relpath(dirpath, root_dir)
        rel_parts = rel.split(os.sep)
        project = rel_parts[0] if rel_parts else "unknown_project"
        proj_out = os.path.join(output_dir, project, project_subdir)
        os.makedirs(proj_out, exist_ok=True)

        for fn in filenames:
            if not fn.lower().endswith(".v"):
                continue
            src = os.path.join(dirpath, fn)
            dst = os.path.join(proj_out, fn)
            if os.path.exists(dst):
                base, ext = os.path.splitext(fn)
                k = 1
                while True:
                    cand = os.path.join(proj_out, f"{base}__{k}{ext}")
                    if not os.path.exists(cand):
                        dst = cand
                        break
                    k += 1

            shutil.copy2(src, dst)
            copied += 1
            manifest_lines.append(f"{src}\t{dst}")

    manifest_path = os.path.join(output_dir, "manifest.tsv")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines))
        if manifest_lines:
            f.write("\n")

    print(f"[完成] 已复制 {copied} 个 .v 到: {output_dir}")
    print(f"[完成] 清单: {manifest_path}")
    return copied


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Collect HLS syn/verilog/*.v into output_final/<project>/<subdir>/")
    parser.add_argument("--root", default=DEFAULT_ROOT_DIR, help="Root directory to scan (default: benchmark_ai)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output root directory (default: output_final)")
    parser.add_argument("--hls-dir", default="hls", help="HLS directory name (e.g. hls_AREA)")
    parser.add_argument("--subdir", default="prompt", help="Project subdir under output_final/<project>/ (e.g. AREA)")
    args = parser.parse_args()

    collect_syn_verilog(args.root, args.output, args.hls_dir, args.subdir)


if __name__ == "__main__":
    main()
