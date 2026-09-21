# VIVADO 双路线 RTL 优化实验

> [English version / 英文版 →](README.en.md)

用大模型自动优化芯片电路代码，并且**用形式验证保证优化后功能不变**。

> ### ⚠️ 关于数据集
>
> **本仓库只包含代码，不包含任何数据集和参考 RTL。**
>
> 原始实验用的是 NVIDIA NVDLA 和 OpenTitan 的开源硬件代码，出于许可证和数据
> 分发考虑没有一起公开。所以直接克隆下来是**跑不起来**的，你需要自备测试用例
> ——格式见[第五节](#测试用例格式自备)。
>
> 同样不含：`memory_agent.db`(历史经验库)、`rag_retrieve/indices/`(RAG 索引)、
> `LLM_DC_LOG/`(知识库)。这些是长期实验积累的产物，需要自己从零跑起来积累。

---

## 一、这个项目在干什么？（给完全没接触过的人）

芯片设计师写的代码叫 **RTL**（一种叫 Verilog 的硬件描述语言）。同一个功能，代码写法不同，
做出来的芯片可能**面积更小**或者**跑得更快**。人工去调很费劲，这个项目让 AI 来调。

但 AI 改代码有个致命风险：**改着改着功能就错了**。所以这个项目的核心是一条严格的流水线：

```
   规格说明书                                   通不过就打回让 AI 重改
   (spec, 描述电路要做什么)                            ↑
        │                                              │
        ├──► 路线 A：C-first  ────┐                    │
        │    先翻译成 C 语言，                          │
        │    优化 C，再转回 RTL      │                  │
        │                          ├──► 形式验证 ──────┘──► 逻辑综合 ──► 量出面积/速度
        ├──► 路线 B：RTL-direct ──┘   (JasperGold)         (DesignCompiler)
             让 AI 直接写 RTL          数学上证明             真实 EDA 工具
                                       "新代码和原代码
                                        功能完全等价"
```

**两条路线跑同一道题，比谁优化得好。** 这就是 "双路线" 的含义。

三个关键角色：

| 名字 | 是什么 | 作用 |
|------|--------|------|
| **LLM**（大模型） | AI，比如 DeepSeek | 负责改代码 |
| **JasperGold**（简称 JG） | 形式验证工具 | 数学证明"改完还是原来的功能"，**不是测试，是证明** |
| **DesignCompiler**（简称 DC） | 逻辑综合工具 | 把 RTL 变成真实电路，量出面积和速度 |

只有**形式验证通过 + 综合成功**的结果才算数。验证没过的，结果会被明确标记，不会混进数据。

---

## 二、快速开始

> **前置条件**：先按[第四节](#四环境要求)准备好 Python 环境、`.env` 配置和测试用例。
> 缺数据集时脚本会在第 3 步明确告诉你缺什么。

### 第 1 步：先自检（不花钱，强烈建议先跑）

```bash
./run_all.sh --check
```

它会依次检查 Python 环境、EDA 工具、数据文件、API 配置，并**自动补齐缺失的数据文件**。
全绿就说明环境没问题：

```
━━━ 步骤 1/6  检查 Python 运行环境 ━━━
  ✓ Python: Python 3.11.16
  ✓ EDA 工具 clang-16
  ✓ EDA 工具 iverilog
  ...
━━━ 步骤 5/6  运行 preflight 自检 ━━━
  ✓ 大模型接口已配置
  ✓ DesignCompiler 远程服务器已配置
  ✓ JasperGold 已配置 (enabled=True)
  ✓ C-first 路线就绪
  ✓ RTL-direct 路线就绪

环境自检全部通过。
```

### 第 2 步：看看要跑些什么（还是不花钱）

```bash
./run_all.sh
```

不加任何参数 = **安全模式**，只生成实验计划，**不会调用大模型，不会产生任何费用**。

### 第 3 步：真正开跑

```bash
./run_all.sh --execute
```

加了 `--execute` 才是真跑。脚本会先让你输入 `yes` 二次确认，因为这一步会：

- 调用大模型 API（**产生费用**）
- 占用远程 JasperGold 和 DesignCompiler 许可证
- 13 个测试用例，**大概要跑 1 小时以上**

---

## 三、`run_all.sh` 常用参数

```bash
./run_all.sh --help          # 看完整帮助
```

| 参数 | 说明 |
|------|------|
| *(不加参数)* | 安全模式：只生成计划，不花钱 |
| `--check` | 只做环境自检 |
| `--execute` | **真实运行**（会花钱） |
| `--objective area` | 优化**面积**（芯片占地更小）— 默认 |
| `--objective timing` | 优化**时序**（电路跑得更快） |
| `--verification-mode none` | 跳过形式验证，只做综合。快，但结果**不保证功能正确**，会被标记为不可用于训练 |
| `--repeats N` | 每个用例重复跑几次（默认 1，跑多次可看稳定性） |
| `--output-root <目录>` | 结果存哪里（默认 `runs/phase6/oneclick_<目标>`） |
| `--summarize-existing` | 不跑新实验，只把已有结果重新汇总 |
| `--yes` | 跳过二次确认（挂后台无人值守时用） |

常用组合：

```bash
./run_all.sh --objective timing --execute      # 真跑时序优化
./run_all.sh --execute --yes                   # 无人值守
./run_all.sh --summarize-existing              # 只重新出报告
```

> **断点续跑**：同一个 `--output-root` 重复运行，已经跑完的用例会被复用，不会重跑。
> 但**不同的运行配置不能共用一个输出目录** —— 程序会校验配置哈希并拒绝，防止新旧结果混在一起。

---

## 四、环境要求

脚本会自动检查下面这些，缺了会明确告诉你缺什么。

**Python 环境**（默认 `/path/to/Vivado/.conda-env`，Python 3.11）

这个 conda 环境里已经装好了所有 EDA 工具：`clang-16`、`opt`、`iverilog`、`yosys`、`dot`。
代码里的 `toolchain.py` 会**优先从 Python 环境的 bin 目录找工具**，所以只要用对了 Python，
工具链自动就位，**不需要手动改 PATH**。

换环境：

```bash
VIVADO_ENV=/你的/conda环境 ./run_all.sh --check
```

**数据文件**（9 个，约 82MB，公开版不含）

这些是运行时必需的大文件，**不在本仓库里**。如果你有一份原始工作树，可以用
环境变量指向它，脚本第 3 步会自动复制补齐（用 `cp -n`，**绝不覆盖已有文件**）：

```bash
VIVADO_SRC_TREE=/你的/原始工作树 ./run_all.sh --check
```

如果没有，就需要从零开始积累——`memory_agent.db` 和 RAG 索引都是跑实验
攒出来的，`rag_retrieve/build_rag.py` 是重建索引的入口。

清单：

| 文件 | 大小 | 作用 |
|------|------|------|
| `memory_agent.db` | 41M | Memory Agent 的历史经验库 |
| `path_decisions_log.json` | 88K | 历史路径决策记录 |
| `rag_retrieve/indices/module4_historical_region_index_v3.json` | 28M | RAG 历史区域索引 |
| `rag_retrieve/indices/module4_cdfg_rag_index_v3.json` | 9.3M | CDFG 联合索引 |
| `LLM_DC_LOG/rag_knowledge_base_v3.csv` | 3.6M | RAG 知识库 |
| 其余 4 个小文件 | <100K | 索引清单、统计、报告 |

换源树：

```bash
VIVADO_SRC_TREE=/你的/源树 ./run_all.sh --check
```

**`.env` 配置**（当前是软链接，指向源树的 `.env`）

里面配了大模型接口（`OPENAI_*` / `CLOUD_*`）、DesignCompiler 远程服务器（`DC_*`）、
JasperGold 远程服务器（`JG_*`）。**这个文件含 API 密钥，不要提交到 git、不要外发。**

---

## 五、代码结构

### 8 个模块（流水线按编号走）

| 模块 | 目录 | 干什么 |
|------|------|--------|
| Module 1 | `spec_analyze/` | 读规格说明，提取电路特征 |
| Module 2 | `path_select/` | 决定这道题走哪条路线 |
| Module 3 | `c_gen/` | 把电路描述翻译成 C 代码 |
| Module 4 | `rag_retrieve/` | RAG 检索：从历史经验里找相似案例 |
| Module 4.5 | *（在 Module 5 内）* | MCTS 搜索，规划改哪里、怎么改 |
| Module 5 | `module5/` | 执行：改 C → 综合 → 形式验证 → 量 PPA |
| Module 8 | `memory_agent/` | 记忆中枢：记录每次决策和结果，指导后续 |
| 辅助 | `token_counter/` | 统计 token 消耗 |

### 实验编排

| 路径 | 作用 |
|------|------|
| `experiments/path_oracle/run_phase6_pilot.py` | **主入口**，`run_all.sh` 调的就是它 |
| `experiments/path_oracle/run_dual_path.py` | 单个 spec 跑双路线 |
| `pipeline/__main__.py` | Module 1-5 完整流水线（单个设计） |
| `pipeline/preflight.py` | 环境自检 |
| `module5/jg_verifier.py` | JasperGold 形式验证（124K，最核心的文件之一） |
| `module5/rtl_direct_runner.py` | RTL-direct 路线执行器 |

### 测试用例格式（自备）

**本仓库不含测试用例**，你需要自己准备。原始实验用的是 13 个 NVDLA 官方电路
（NVDLA 是英伟达开源的深度学习加速器），你可以自己从 [nvdla/hw](https://github.com/nvdla/hw)
或任何 Verilog 项目取材。

目录结构：

```
你的用例目录/
  pilot_manifest.json          # 用例清单
  cases/
    <用例名>/
      spec.txt                 # 规格说明（喂给 AI 的题目）
      golden_source.v          # 原始代码
      golden_closure.v         # 自包含闭包版（形式验证的标准答案）
```

`pilot_manifest.json` 必须包含这些字段：

```json
{
  "case_count": 1,
  "execution_policy": { "mode": "serial", "max_concurrency": 1 },
  "cases": [
    {
      "id": "my_case",
      "family_id": "my_family",
      "design_type": "sequential",
      "golden_top_module": "my_module",
      "spec_path": "cases/my_case/spec.txt",
      "golden_rtl_path": "cases/my_case/golden_closure.v",
      "golden_source_rtl_path": "cases/my_case/golden_source.v"
    }
  ]
}
```

几个硬性要求（不满足程序会直接拒绝启动）：

- `execution_policy` 必须是 `{"mode": "serial", "max_concurrency": 1}`
- `case_count` 必须等于 `cases` 数组的实际长度
- `spec_path` 和 `golden_rtl_path` 指向的文件必须存在

**什么是 "closure"（闭包）**：形式验证需要一个能独立编译的完整设计。如果你的模块
依赖别的文件（子模块、RAM 模型、库函数），要把依赖全部拼进一个文件，这就是
`golden_closure.v`。`golden_source.v` 保留原样，只作记录。

准备好后指向它：

```bash
./run_all.sh --pilot-root /你的/用例目录 --execute
```

### 其他

- `golden_datas/`、`goldenRTL/` —— 参考 RTL 和规格文档
- `logs/` —— 运行日志（`run_all.sh` 每次运行都会写一份）
- `run_dc.py`、`verilog_dc.py` —— DesignCompiler 调用
- `llm_v2v.py`、`llm_request.py` —— 大模型调用封装

---

## 六、结果怎么看

结果在 `--output-root` 指定的目录（默认 `runs/phase6/oneclick_area/`）：

```
runs/phase6/oneclick_area/
  phase6_collection.json      # ← 总汇总，先看这个
  pairs/
    <用例名>_rep1/
      phase6_pair_result.json # 单个用例的双路线对比结果
```

`phase6_collection.json` 里的关键字段：

| 字段 | 含义 |
|------|------|
| `mode` | `plan_only` = 只是计划；`execute` = 真跑过 |
| `objective` | `AREA` 或 `TIMING` |
| `planned_pair_count` | 计划跑多少对 |
| `completed_pair_count` | 实际完成多少对 |
| `run_config_sha256` | 配置哈希，用于断点续跑时校验 |

每个用例的结果状态（在 `phase6_pair_result.json` 里）：

| 状态 | 意思 |
|------|------|
| `success` | 全程通过，有有效 PPA 数据 ✓ |
| `equivalence_failed` | 形式验证发现功能不等价 —— **AI 改错了** |
| `rtl_failed` | 生成的 RTL 语法或 elaborate 失败 |
| `c_generation_failed` | C 代码生成失败 |
| `validate_failed` | 校验未通过 |
| `pipeline_exception` | 流程异常（通常是环境/文件问题） |

> **看懂 `equivalence_failed`**：这不是程序 bug，而是**形式验证正常工作的证据** ——
> 它抓住了 AI 改错的代码，并给出反例（counterexample）。
> 实际跑下来，一轮 19 个用例的实验里只有少数几个走到 `success`，大部分被验证拦下。
> **这正是这套流水线的价值所在**：没有形式验证，那些错误的优化会被当成成果收下。

---

## 七、常见问题

**Q：跑一次要多少钱 / 多久？**
13 个用例、`--repeats 1`，每个用例几分钟，总计 1 小时以上。费用取决于 `.env` 里配的模型。
先用不带参数的安全模式看计划，确认没问题再 `--execute`。

**Q：提示 "Project Memory baseline files are required"**
说明 `memory_agent.db` 或 `path_decisions_log.json` 缺失。跑 `./run_all.sh --check` 会自动补齐。

**Q：提示 "Required tool not found"**
EDA 工具没找到。确认用的是带工具链的那个 conda 环境（`VIVADO_ENV`）。
注意：**不要用 `conda activate` 后直接 `python`**，`run_all.sh` 用的是绝对路径的 Python，更可靠。

**Q：想跳过形式验证快速看结果？**
`./run_all.sh --verification-mode none --execute`。
但注意结果会被标记 `training_eligible=false`，**不能当作可信数据使用**。

**Q：能并行跑快一点吗？**
主入口**设计上就是串行的**（`max_concurrency=1`），没有 `--workers` 选项。
这是刻意的：并发调 LLM 和同时提交多个 DC 任务会污染实验对比的公平性。

**Q：怎么准备测试用例？**
见[第五节的格式说明](#测试用例格式自备)。`--pilot-root` 指向你自己的目录。
注意 manifest 里必须写 `"execution_policy": {"mode": "serial", "max_concurrency": 1}`，否则程序会拒绝。

**Q：中断了怎么续？**
用同一个 `--output-root` 再跑一次，已完成的用例会自动复用。

---

## 八、安全提醒

- `.env` 含 API 密钥和服务器账号，**不要提交 git，不要外发**
- `--execute` 会真实产生 API 费用并占用 EDA 许可证，脚本的二次确认是故意设计的
- 脚本补齐数据文件时用 `cp -n`，**绝不覆盖**已有文件，可以放心重复跑
- 仓库根的 `.gitignore` 已排除密钥、数据集和实验产物；改动它之前请先想清楚
