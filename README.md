# SpecAdapt

项目链接：[https://github.com/Twinklestarsi/SpecAdapt](https://github.com/Twinklestarsi/SpecAdapt)

本仓库是 SpecAdapt 的代码库。主流程从 specification（规格说明）开始，自动选择 `c_first` 或 `rtl_direct` 路线，生成候选 RTL，再用 JasperGold 做功能等价验证，最后用 Design Compiler 评估面积或时序。

## 1. 主流程中的四个 Agent

四个 Agent 由下面这些模块共同实现。JasperGold 和 Design Compiler 是验证、综合和 PPA 评估后端，不属于四个 Agent。

| Agent | 当前实现位置 | 负责什么 |
| --- | --- | --- |
| **Spec Agent** | [`spec_analyze/`](spec_analyze/)、[`path_select/`](path_select/) | 读取 spec，提取结构化特征；在 `auto` 模式下结合规则、模型和 Memory 记录使用 Adapter 选择 `c_first` 或 `rtl_direct` |
| **C Agent** | [`c_gen/`](c_gen/)、[`rag_retrieve/`](rag_retrieve/)、[`module5/`](module5/) | 在 `c_first` 路线中生成 C；检索历史变换、用 MCTS 形成优化动作链；随后修改并检查 C，使用 HLS 实现 C-to-RTL|
| **RTL Agent** | [`pipeline/`](pipeline/)、[`module5/`](module5/) | 生成、语法检查和重试 RTL。`rtl_direct` 从 spec 直接生成|
| **Memory Agent** | [`memory_agent/`](memory_agent/) | 保存和检索特征、路径决策、历史经验、失败反馈和 PPA 结果，为后续阶段提供上下文 |


验证后端的实现位置是 [`module5/jg_verifier.py`](module5/jg_verifier.py)（JasperGold）和 [`module5/dc_runner.py`](module5/dc_runner.py)（Design Compiler）。使用 `--verification-mode jaspergold` 时，候选必须先通过语法检查和 JasperGold 等价验证，之后才进入 DC。

## 2. 环境准备

先克隆仓库并进入根目录：

```bash
git clone https://github.com/Twinklestarsi/SpecAdapt.git
cd SpecAdapt
```

项目不假定固定的 Python 或 conda 路径，也不提供统一的依赖安装脚本。请准备一个能运行本项目的 Python 3 环境，并确保 `python` 在 PATH 中。若使用专用环境，可以显式指定环境根目录：

```bash
export VIVADO_ENV=/path/to/your/python-env
export PATH="$VIVADO_ENV/bin:$PATH"
```

仅在新克隆目录中确认不存在 `.env` 时，复制配置模板并填入自己的服务配置；如果已有 `.env`，不要覆盖它：

```bash
cp .env.example .env
```

`.env.example` 包含 LLM、JasperGold 和 Design Compiler 的配置项。根据实际环境填写对应的 API、远程服务器和工具路径；不使用某个后端时，可以保留该后端的占位配置。运行 HLS 时需要 Vitis 的 `v++`，并需要 `yosys` 和 `iverilog` 在 PATH 中；如果 `v++` 还没有在 PATH 中，先指定 Vitis 环境脚本：

```bash
export VITIS_ENV_SCRIPT=/path/to/Vitis/settings64.sh
```

如果 `v++` 已经在 PATH 中，则不需要设置 `VITIS_ENV_SCRIPT`。

## 3. 进行实验：`python -m pipeline`

运行 `python -m pipeline` 命令会处理一个设计。默认设置会让路径选择模块自动决定处理路线。每次运行都必须用 `--objective` 指定优化目标。`area` 表示尽量减小芯片面积，`timing` 表示尽量缩短电路延迟。

路径选择模块按以下顺序工作：

1. 路径选择模块先检查两类固定规则。遇到多时钟、纯结构或极简单的电路时，程序直接选择 `rtl_direct`，也就是根据设计需求直接生成 RTL（寄存器传输级硬件描述代码）。
2. 如果这些规则都未命中，程序就调用已训练好的 MLP（多层感知机模型），计算选择 `c_first` 的概率。`c_first` 表示先生成 C 语言代码，再根据 C 代码生成 RTL。

### 3.1 使用 spec 文件

示例代码如下：`golden_closure.v` 是给 JasperGold 的参考设计，`--golden-top your_top` 应与该 case 的顶层模块名一致：

```bash
# 如果 v++ 不在 PATH 中，取消下一行注释并改成自己的 Vitis 安装脚本路径。
# export VITIS_ENV_SCRIPT=/path/to/Vitis/settings64.sh

SPEC=/path/to/your/spec.txt
GOLDEN=/path/to/your/golden_closure.v

python -m pipeline \
  --spec-file "$SPEC" \
  --benchmark your_benchmark \
  --objective area \
  --backend hls \
  --verification-mode jaspergold \
  --golden-rtl "$GOLDEN" \
  --golden-top your_top \
  --output-root "$PWD/pipeline_runs/your_benchmark_area"
```



### 3.2 实验流程结果位置

如果不传路径参数，`pipeline/orchestrator.py` 默认使用根目录的 `memory_agent.db`、`path_decisions_log.json`、`pipeline_runs/` 和 `c_gen_output/`。传入 `--output-root` 后，每次运行的计划和候选结果写到该目录下，例如：

```text
pipeline_runs/csrng_area/
├── summary.json
└── <run_id>/
    ├── rtl_direct/              # rtl_direct 路线
    ├── module4_plan.json        # C-first 的检索结果
    ├── module45_plan.json       # C-first 的 MCTS 动作链
    └── module5/                  # 修改后的 C、RTL、JG/DC 结果
```

## 4. 常用 CLI 速查

```bash
python -m pipeline --help
./run_all.sh --help
```

单设计常用参数：`--spec-file`、`--verilog`、`--objective area|timing`、`--backend direct_rtl|hls`、`--forced-path auto|c_first|rtl_direct`、`--verification-mode jaspergold|none`、`--golden-rtl`、`--golden-top`、`--output-root`。

Phase-6 常用参数：`--pilot-root`、`--output-root`、`--objective area|timing`、`--repeats N`、`--execute`、`--yes`、`--check`、`--summarize-existing`。完整参数和默认值以这两个 `--help` 输出及当前脚本实现为准。
