  #!/usr/bin/env python3
  # -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import re
from pathlib import Path
try:
    from .tools.llm_api import call_llm
except ImportError:
    from tools.llm_api import call_llm

'''
    首先要运行的脚本 
    
    1. 遍历指定目录下的所有 .v 文件
    2. 对每个 .v 文件，调用 v2c 工具生成对应的 .c 文件
    3. 将生成的 .c 文件重命名为 .cpp（内容不变）
'''

# 基于当前工作目录定位项目文件/目录（避免硬编码绝对路径）
_BASE_DIR = Path.cwd().resolve()
# 要扫描的根目录
ROOT_DIR = str((_BASE_DIR / "benchmark").resolve())
V2C_BIN = str((_BASE_DIR / "v2c").resolve())
TAR_DIR = str((_BASE_DIR / "benchmark_output").resolve())



def iter_verilog_files(root_dir):
    """
      递归遍历 root_dir 下所有子目录，找到所有 .v 文件。

      :param root_dir: 根目录路径
      :return: 逐个返回 .v 文件的绝对路径
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 可选：排除一些无需扫描的目录
        dirnames[:] = [d for d in dirnames if d not in ('.git', '.idea', '__pycache__')]
        for filename in filenames:
            if filename.lower().endswith(".v"):
                yield os.path.join(dirpath, filename)
""" v2c 工具 """
def get_top_module_name(file_path):
    """简单从 .v 文件中找第一个 module 名字。"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        print(f"[错误] 无法读取 {file_path}: {e}")
        return None

    m = re.search(r'\bmodule\s+([a-zA-Z_]\w*)', text)
    if m:
        return m.group(1)
    return None

def process_verilog_file_v2c(file_path):
    """
      处理单个 Verilog 源文件的函数接口。

      你可以在这里实现自己的逻辑，例如：
      - 调用 Pyverilog 解析 AST / 数据流
      - 生成图 / 报告 / 统计信息
      - 拷贝文件、生成脚本等

      :param file_path: .v 文件的绝对路径
    """   
    # 提取文件名
    filename = os.path.basename(file_path)

    # 计算相对于 ROOT_DIR 的相对路径，用于在 TAR_DIR 中复现目录结构
    rel_path = os.path.relpath(file_path, ROOT_DIR)
    rel_dir = os.path.dirname(rel_path)
    dest_dir = os.path.join(TAR_DIR, rel_dir)

    # 创建目标目录
    os.makedirs(dest_dir, exist_ok=True)

    # 将文件保存到目标目录下
    dest_v = os.path.join(dest_dir, filename)
    shutil.copy(file_path, dest_v)

    # 解析顶层模块名
    top_module = get_top_module_name(file_path)
    if not top_module:
        print(f"[警告] {file_path} 未找到 module 名，跳过 v2c 转换。")
        return

    # 构造输出 C 文件名（与 顶层module 同名，后缀改为 .c）
    # name_no_ext, _ = os.path.splitext(filename)
    out_c = f"{top_module}.c"

    # 在目标目录下运行 v2c 程序
    cmd = [V2C_BIN, filename, "--module", top_module, out_c]
    print(f"[运行] {' '.join(cmd)}  (cwd={dest_dir})")

    result = subprocess.run(
        cmd,
        cwd=dest_dir,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print("[STDOUT]")
        print(result.stdout)
    if result.stderr:
        print("[STDERR]")
        print(result.stderr)
    if result.returncode != 0:
        print(f"[错误] v2c 执行失败，返回码: {result.returncode}")
    else:
        print(f"[完成] 已生成 {out_c}")
        # 将 v2c 生成的 .c 仅通过修改后缀名生成同名 .cpp（内容不变）
        out_c_path = os.path.join(dest_dir, out_c)
        out_cpp = os.path.splitext(out_c)[0] + ".cpp"
        out_cpp_path = os.path.join(dest_dir, out_cpp)
        try:
            if os.path.isfile(out_c_path):
                shutil.copy(out_c_path, out_cpp_path)
                print(f"[完成] 已生成 {out_cpp}")
            else:
                print(f"[警告] 未找到 v2c 输出文件，无法生成 .cpp: {out_c_path}")
        except Exception as e:
            print(f"[警告] 生成 .cpp 失败: {out_cpp_path}: {e}")
""" 2.ai 优化c """
def process_verilog_file_ai(file_path):
    call_llm(file_path)



def main():
    root = ROOT_DIR
    output_dir = TAR_DIR

    if not os.path.isdir(root):
        print(f"目录不存在: {root}")
        return

    # 清理旧目录
    if os.path.exists(TAR_DIR):
        shutil.rmtree(TAR_DIR)


    print(f"开始扫描目录: {os.path.abspath(root)}")

    count = 0
    for v_file in iter_verilog_files(root):
        count += 1
        process_verilog_file_v2c(v_file)


    print(f"扫描完成，共处理 {count} 个 .v 文件")
    """ 
        对 v2c出来的文件进行处理 
        1. 先对c文件进行cdfg提取
        2. 对c文件进行ai优化，并生成c++文件
        3. 将c++文件创建到component中，并转换为rtl
    """

    # 暂停等待空行输入
    os.system("read -p '按回车键退出...' var") 

if __name__ == "__main__":
    main()
