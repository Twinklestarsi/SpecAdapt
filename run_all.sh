#!/usr/bin/env bash
#
# run_all.sh —— VIVADO 双路线 RTL 优化实验 · 一键运行脚本
#
# 做的事情（按顺序）：
#   1. 定位 Python 运行环境（自带 clang / iverilog / yosys / dot 等 EDA 工具）
#   2. 检查 Python 第三方依赖
#   3. 检查并补齐运行所需的大文件资产（RAG 索引 / Memory 基线，约 82MB）
#   4. 检查 .env 里的 LLM、DesignCompiler、JasperGold 配置
#   5. 跑 preflight 自检
#   6. 运行 phase-6 实验（默认只规划不真跑；加 --execute 才真的跑）
#
# 用法见： ./run_all.sh --help
#
set -euo pipefail

# ============================================================
# 0. 基本路径
# ============================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ---- Python 环境（conda env 根目录）----
# 优先级：环境变量 VIVADO_ENV > 项目内 .conda-env > 上级目录 Vivado/.conda-env
#        > 当前已激活的 conda 环境 > PATH 上的 python3
# 用别的环境时：export VIVADO_ENV=/你的/conda/env
if [[ -z "${VIVADO_ENV:-}" ]]; then
  for _cand in \
    "$PROJECT_ROOT/.conda-env" \
    "$(dirname "$PROJECT_ROOT")/Vivado/.conda-env" \
    "${CONDA_PREFIX:-}" ; do
    if [[ -n "$_cand" && -x "$_cand/bin/python" ]]; then
      VIVADO_ENV="$_cand"; break
    fi
  done
fi

if [[ -n "${VIVADO_ENV:-}" && -x "$VIVADO_ENV/bin/python" ]]; then
  PY="$VIVADO_ENV/bin/python"
else
  # 兜底：用 PATH 上的 python3。注意 toolchain.py 是按 sys.prefix/bin 找
  # clang / iverilog / yosys / dot 的，此时这些工具必须自己在 PATH 上。
  VIVADO_ENV=""
  PY="$(command -v python3 || true)"
fi

# 数据资产来源树（老工程目录，用于 --skip-assets 之外的资产补齐）。
# 没有老工程就留空，脚本会自动跳过补齐这一步。
VIVADO_SRC_TREE="${VIVADO_SRC_TREE:-}"

# ============================================================
# 1. 默认参数（都可以用命令行改）
# ============================================================
OBJECTIVE="area"                 # area 或 timing
PILOT_ROOT="$PROJECT_ROOT/runs/phase6/nvdla13_jg_bak_20260920/pilot"
OUTPUT_ROOT=""                   # 留空则自动按 objective 生成
EXECUTE=0                        # 0=只规划(安全)  1=真跑(花钱)
ASSUME_YES=0                     # 1=跳过真跑前的二次确认
CHECK_ONLY=0                     # 1=只做环境自检，不跑实验
SKIP_ASSETS=0                    # 1=跳过资产补齐
SUMMARIZE=0                      # 1=只根据已有结果重建汇总

REPEATS=1
MCTS_ITERATIONS=240
MCTS_MAX_DEPTH=5
MCTS_CANDIDATE_LIMIT=8
MAX_ACTIONS=5
RTL_MAX_RETRIES=2
DC_MAX_RETRIES=0
VERIFICATION_MODE="jaspergold"   # jaspergold 或 none
VERIFICATION_TIMEOUT=1000
JG_MAX_RETRIES=1
PPA_TIE_TOLERANCE_PCT=1.0

# ============================================================
# 2. 彩色输出小工具
# ============================================================
if [ -t 1 ]; then
  C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'
  C_BLU=$'\033[36m'; C_BLD=$'\033[1m';  C_OFF=$'\033[0m'
else
  C_RED=''; C_GRN=''; C_YEL=''; C_BLU=''; C_BLD=''; C_OFF=''
fi

step()  { echo; echo "${C_BLU}${C_BLD}━━━ $* ━━━${C_OFF}"; }
ok()    { echo "  ${C_GRN}✓${C_OFF} $*"; }
warn()  { echo "  ${C_YEL}!${C_OFF} $*"; }
fail()  { echo "  ${C_RED}✗${C_OFF} $*"; }
die()   { echo; echo "${C_RED}${C_BLD}出错：$*${C_OFF}" >&2; exit 1; }

usage() {
  cat <<'USAGE'
VIVADO 双路线 RTL 优化实验 · 一键运行脚本

用法:
  ./run_all.sh [选项]

常用:
  (不加任何参数)          只检查环境 + 生成实验计划，不调用 LLM，不花钱   ← 建议先跑这个
  --execute               真实运行：调用大模型 + 远程 JasperGold + DesignCompiler
  --check                 只做环境自检，连计划都不生成
  --summarize-existing    不跑新实验，只把已完成的结果重新汇总成报告

实验设置:
  --objective area|timing 优化目标，默认 area
                          area  = 优化面积（芯片占地）
                          timing= 优化时序（跑多快）
  --pilot-root <目录>     测试用例集目录，默认 13 个 NVDLA 官方用例
  --output-root <目录>    结果输出目录，默认 runs/phase6/oneclick_<objective>
  --repeats N             每个用例重复次数，默认 1
  --verification-mode jaspergold|none
                          jaspergold = 跑形式验证（默认，严格）
                          none       = 跳过验证直接综合（快，但不保证功能正确）
  --max-actions N         C-first 路线最多做几步优化，默认 5
  --mcts-iterations N     搜索迭代次数，默认 240
  --jg-max-retries N      JasperGold 失败后让 AI 重修几次，默认 1
  --verification-timeout N  单次形式验证超时秒数，默认 1000

其他:
  --yes                   真跑前不再二次确认（无人值守时用）
  --skip-assets           跳过大文件资产补齐（确认已经补过时用）
  -h, --help              显示这份帮助

环境变量:
  VIVADO_ENV        conda 环境根目录。不设就自动找：项目内 .conda-env
                    -> 上级 Vivado/.conda-env -> 当前激活的 conda -> python3
  VIVADO_SRC_TREE   运行时大文件（memory_agent.db / RAG 索引等）的来源树。
                    不设就跳过资产补齐这一步。

例子:
  ./run_all.sh                              # 先自检 + 看计划
  ./run_all.sh --execute                    # 真跑面积优化
  ./run_all.sh --objective timing --execute # 真跑时序优化
  ./run_all.sh --execute --yes              # 无人值守真跑
USAGE
}

# ============================================================
# 3. 解析命令行参数
# ============================================================
while [ $# -gt 0 ]; do
  case "$1" in
    --execute)              EXECUTE=1; shift ;;
    --check)                CHECK_ONLY=1; shift ;;
    --summarize-existing)   SUMMARIZE=1; shift ;;
    --yes|-y)               ASSUME_YES=1; shift ;;
    --skip-assets)          SKIP_ASSETS=1; shift ;;
    --objective)            OBJECTIVE="${2:?--objective 需要一个值}"; shift 2 ;;
    --pilot-root)           PILOT_ROOT="${2:?--pilot-root 需要一个值}"; shift 2 ;;
    --output-root)          OUTPUT_ROOT="${2:?--output-root 需要一个值}"; shift 2 ;;
    --repeats)              REPEATS="${2:?}"; shift 2 ;;
    --mcts-iterations)      MCTS_ITERATIONS="${2:?}"; shift 2 ;;
    --mcts-max-depth)       MCTS_MAX_DEPTH="${2:?}"; shift 2 ;;
    --mcts-candidate-limit) MCTS_CANDIDATE_LIMIT="${2:?}"; shift 2 ;;
    --max-actions)          MAX_ACTIONS="${2:?}"; shift 2 ;;
    --rtl-max-retries)      RTL_MAX_RETRIES="${2:?}"; shift 2 ;;
    --dc-max-retries)       DC_MAX_RETRIES="${2:?}"; shift 2 ;;
    --verification-mode)    VERIFICATION_MODE="${2:?}"; shift 2 ;;
    --verification-timeout) VERIFICATION_TIMEOUT="${2:?}"; shift 2 ;;
    --jg-max-retries)       JG_MAX_RETRIES="${2:?}"; shift 2 ;;
    --ppa-tie-tolerance-pct) PPA_TIE_TOLERANCE_PCT="${2:?}"; shift 2 ;;
    -h|--help)              usage; exit 0 ;;
    *) echo "未知参数: $1"; echo "用 ./run_all.sh --help 查看帮助"; exit 2 ;;
  esac
done

case "$OBJECTIVE" in
  area|timing) ;;
  *) die "--objective 只能是 area 或 timing，你给的是: $OBJECTIVE" ;;
esac
case "$VERIFICATION_MODE" in
  jaspergold|none) ;;
  *) die "--verification-mode 只能是 jaspergold 或 none，你给的是: $VERIFICATION_MODE" ;;
esac

if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$PROJECT_ROOT/runs/phase6/oneclick_${OBJECTIVE}"
fi

echo "${C_BLD}VIVADO 双路线 RTL 优化实验 · 一键运行${C_OFF}"
echo "项目目录: $PROJECT_ROOT"

# ============================================================
# 步骤 1 / 6 ：Python 运行环境
# ============================================================
step "步骤 1/6  检查 Python 运行环境"

if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  fail "找不到可用的 Python${PY:+: $PY}"
  echo
  echo "  这个项目需要一个专用的 conda 环境，里面装好了 clang / iverilog / yosys / dot"
  echo "  等 EDA 工具。请用环境变量指定它的位置，例如："
  echo "      VIVADO_ENV=/你的/conda/环境 ./run_all.sh"
  die "Python 环境不可用"
fi
ok "Python: $("$PY" -V 2>&1)"
ok "环境路径: ${VIVADO_ENV:-（PATH 上的 python3，未使用专用 conda 环境）}"

# 工具链：toolchain.py 会优先在 <python环境>/bin 下找这些工具，
# 所以只要用对了 Python，工具链就自动解析，不需要改 PATH。
MISSING_TOOLS=()
for t in clang-16 opt iverilog yosys dot; do
  if [ -x "$VIVADO_ENV/bin/$t" ]; then
    ok "EDA 工具 $t"
  elif command -v "$t" >/dev/null 2>&1; then
    warn "EDA 工具 $t 不在环境内，但系统 PATH 上有: $(command -v "$t")"
  else
    fail "EDA 工具 $t 缺失"
    MISSING_TOOLS+=("$t")
  fi
done
if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
  die "缺少 EDA 工具: ${MISSING_TOOLS[*]}（C-first 路线需要它们）"
fi

# ============================================================
# 步骤 2 / 6 ：Python 第三方依赖
# ============================================================
step "步骤 2/6  检查 Python 依赖"

DEPS_REPORT="$("$PY" - <<'PYEOF'
import importlib
required = ["dotenv", "openai", "numpy", "pandas", "yaml", "requests",
            "tiktoken", "networkx", "sklearn"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
print("MISSING=" + ",".join(missing))
PYEOF
)"
MISSING_DEPS="${DEPS_REPORT#MISSING=}"
if [ -n "$MISSING_DEPS" ]; then
  fail "缺少 Python 包: $MISSING_DEPS"
  echo "      安装命令: $VIVADO_ENV/bin/pip install ${MISSING_DEPS//,/ }"
  die "Python 依赖不全"
fi
ok "核心依赖齐全（openai / numpy / pandas / sklearn / tiktoken ...）"

# ============================================================
# 步骤 3 / 6 ：大文件资产
# ============================================================
step "步骤 3/6  检查运行所需的数据资产"

# 这些是代码仓库里没有、但运行时必需的大文件（RAG 知识库 + Memory 基线）
ASSETS=(
  "memory_agent.db"
  "path_decisions_log.json"
  "rag_retrieve/indices/latest_rag.json"
  "rag_retrieve/indices/module4_cdfg_rag_index_v3.json"
  "rag_retrieve/indices/module4_historical_region_index_v3.json"
  "rag_retrieve/indices/module4_transform_stats_v3.json"
  "rag_retrieve/indices/rag_rebuild_report_v3.json"
  "LLM_DC_LOG/rag_knowledge_base_v3.csv"
  "LLM_DC_LOG/rag_knowledge_base_v3.summary.json"
)

NEED_COPY=()
for rel in "${ASSETS[@]}"; do
  [ -e "$PROJECT_ROOT/$rel" ] || NEED_COPY+=("$rel")
done

if [ ${#NEED_COPY[@]} -eq 0 ]; then
  ok "9 个数据资产齐全"
elif [ "$SKIP_ASSETS" -eq 1 ]; then
  warn "缺 ${#NEED_COPY[@]} 个资产，但你指定了 --skip-assets，跳过补齐"
else
  warn "缺少 ${#NEED_COPY[@]} 个数据资产，将从源树只读复制（不会覆盖已有文件）"
  echo "      来源: ${VIVADO_SRC_TREE:-（未设置 VIVADO_SRC_TREE）}"
  if [ -z "$VIVADO_SRC_TREE" ] || [ ! -d "$VIVADO_SRC_TREE" ]; then
    fail "没有可用的资产来源树${VIVADO_SRC_TREE:+: $VIVADO_SRC_TREE}"
    echo
    echo "  本仓库不含这些运行时大文件。两个办法："
    echo "    1) 有原始工作树： VIVADO_SRC_TREE=/你的/工作树 ./run_all.sh --check"
    echo "    2) 从零积累：     memory_agent.db 和 RAG 索引是跑实验攒出来的，"
    echo "                      重建索引的入口是 rag_retrieve/build_rag.py"
    die "缺少运行时数据资产"
  fi

  COPY_FAILED=()
  for rel in "${NEED_COPY[@]}"; do
    src="$VIVADO_SRC_TREE/$rel"
    dst="$PROJECT_ROOT/$rel"
    if [ ! -e "$src" ]; then
      fail "源树里也没有: $rel"
      COPY_FAILED+=("$rel")
      continue
    fi
    mkdir -p "$(dirname "$dst")"
    # -n 保证绝不覆盖已存在的文件
    if cp -n "$src" "$dst" 2>/dev/null && [ -e "$dst" ]; then
      ok "已补齐 $rel ($(du -h "$dst" 2>/dev/null | cut -f1))"
    else
      fail "复制失败: $rel"
      COPY_FAILED+=("$rel")
    fi
  done
  if [ ${#COPY_FAILED[@]} -gt 0 ]; then
    die "以下资产无法补齐: ${COPY_FAILED[*]}"
  fi
fi

# 测试用例集
if [ ! -f "$PILOT_ROOT/pilot_manifest.json" ]; then
  fail "找不到测试用例清单: $PILOT_ROOT/pilot_manifest.json"
  echo
  echo "  本仓库不含数据集，测试用例需要自备。目录结构："
  echo "      你的用例目录/"
  echo "        pilot_manifest.json"
  echo "        cases/<用例名>/{spec.txt, golden_source.v, golden_closure.v}"
  echo
  echo "  manifest 必须写 \"execution_policy\": {\"mode\": \"serial\", \"max_concurrency\": 1}"
  echo "  完整格式见 README.md 第五节。"
  echo
  echo "  准备好后用： ./run_all.sh --pilot-root /你的/用例目录"
  die "缺少测试用例"
fi
CASE_COUNT="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['case_count'])" \
             "$PILOT_ROOT/pilot_manifest.json" 2>/dev/null || echo '?')"
ok "测试用例集: $CASE_COUNT 个 case  ($PILOT_ROOT)"

# ============================================================
# 步骤 4 / 6 ：.env 配置
# ============================================================
step "步骤 4/6  检查 .env 配置"

if [ ! -r "$PROJECT_ROOT/.env" ]; then
  fail "读不到 .env"
  echo "      .env 里要配置大模型接口、DesignCompiler 和 JasperGold 的远程服务器信息。"
  die ".env 缺失或不可读"
fi
if [ -L "$PROJECT_ROOT/.env" ]; then
  ok ".env（软链接 → $(readlink -f "$PROJECT_ROOT/.env")）"
else
  ok ".env"
fi

# ============================================================
# 步骤 5 / 6 ：preflight 自检
# ============================================================
step "步骤 5/6  运行 preflight 自检"

PREFLIGHT_JSON="$("$PY" -m pipeline.preflight 2>&1)" || true
echo "$PREFLIGHT_JSON" > "$PROJECT_ROOT/.preflight_last.json"

# 把颜色开关传给 Python，保证重定向到文件时不会漏出转义码
"$PY" - "$PROJECT_ROOT/.preflight_last.json" "$C_GRN" "$C_RED" "$C_OFF" <<'PYEOF'
import json, sys
_, report_path, c_grn, c_red, c_off = sys.argv[:5]
try:
    data = json.load(open(report_path, encoding="utf-8"))
except Exception:
    print(f"  {c_red}\u2717{c_off} preflight 输出无法解析，原文见 .preflight_last.json")
    raise SystemExit(1)

def mark(flag, text):
    print(f"  {c_grn}\u2713{c_off} {text}" if flag else f"  {c_red}\u2717{c_off} {text}")

mark(data["llm"]["configured"],  "大模型接口已配置")
mark(data["dc"]["configured"],   "DesignCompiler 远程服务器已配置")
jg = data["jaspergold"]
mark(jg["configured"],           f"JasperGold 已配置 (enabled={jg['enabled']})")
mark(data["ready_for_c_first_execution"],    "C-first 路线就绪")
mark(data["ready_for_rtl_direct_execution"], "RTL-direct 路线就绪")

if not (data["ready_for_c_first_execution"] and data["ready_for_rtl_direct_execution"]):
    raise SystemExit(1)
PYEOF
PREFLIGHT_RC=$?
[ "$PREFLIGHT_RC" -eq 0 ] || die "preflight 未通过，详见 .preflight_last.json"

if [ "$CHECK_ONLY" -eq 1 ]; then
  echo
  echo "${C_GRN}${C_BLD}环境自检全部通过。${C_OFF}"
  echo "下一步可以跑： ./run_all.sh              （生成实验计划，不花钱）"
  echo "          或： ./run_all.sh --execute    （真实运行）"
  exit 0
fi

# ============================================================
# 步骤 6 / 6 ：运行实验
# ============================================================
step "步骤 6/6  运行 phase-6 实验"

RUN_ARGS=(
  -m experiments.path_oracle.run_phase6_pilot
  --pilot-root "$PILOT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --repeats "$REPEATS"
  --ppa-objective "$OBJECTIVE"
)

if [ "$SUMMARIZE" -eq 1 ]; then
  RUN_ARGS+=(--summarize-existing)
  MODE_DESC="只汇总已完成结果（不调用大模型）"
else
  RUN_ARGS+=(
    --mcts-iterations "$MCTS_ITERATIONS"
    --mcts-max-depth "$MCTS_MAX_DEPTH"
    --mcts-candidate-limit "$MCTS_CANDIDATE_LIMIT"
    --max-actions "$MAX_ACTIONS"
    --rtl-max-retries "$RTL_MAX_RETRIES"
    --dc-max-retries "$DC_MAX_RETRIES"
    --verification-mode "$VERIFICATION_MODE"
    --verification-timeout "$VERIFICATION_TIMEOUT"
    --jg-max-retries "$JG_MAX_RETRIES"
    --ppa-tie-tolerance-pct "$PPA_TIE_TOLERANCE_PCT"
  )
  if [ "$EXECUTE" -eq 1 ]; then
    RUN_ARGS+=(--execute)
    MODE_DESC="${C_YEL}真实运行${C_OFF}（会调用大模型 + 远程 JasperGold + DesignCompiler）"
  else
    MODE_DESC="只规划，不执行（安全模式，不花钱）"
  fi
fi

echo "  优化目标  : $OBJECTIVE"
echo "  用例集    : $PILOT_ROOT（$CASE_COUNT 个 case）"
echo "  输出目录  : $OUTPUT_ROOT"
echo "  运行模式  : $MODE_DESC"
[ "$SUMMARIZE" -eq 0 ] && echo "  形式验证  : $VERIFICATION_MODE"

# 真跑前二次确认
if [ "$EXECUTE" -eq 1 ] && [ "$ASSUME_YES" -eq 0 ]; then
  echo
  echo "${C_YEL}${C_BLD}注意：真实运行会${C_OFF}"
  echo "  · 调用大模型 API（产生费用）"
  echo "  · 占用远程 JasperGold 和 DesignCompiler 许可证"
  echo "  · 每个 case 大约几分钟，$CASE_COUNT 个 case 可能要跑 1 小时以上"
  echo
  if [ -t 0 ]; then
    read -r -p "确认继续？输入 yes 回车： " REPLY_TEXT
    [ "$REPLY_TEXT" = "yes" ] || { echo "已取消。"; exit 0; }
  else
    die "非交互环境下真跑请加 --yes 参数"
  fi
fi

mkdir -p "$PROJECT_ROOT/logs"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_LOG="$PROJECT_ROOT/logs/run_${STAMP}_${OBJECTIVE}.log"

echo
echo "  运行日志  : $RUN_LOG"
echo "  开始时间  : $(date '+%F %T')"
echo

set +e
"$PY" "${RUN_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
RC="${PIPESTATUS[0]}"
set -e

echo
echo "  结束时间  : $(date '+%F %T')"

if [ "$RC" -ne 0 ]; then
  echo
  fail "实验退出码 $RC，失败详情见： $RUN_LOG"
  exit "$RC"
fi

echo
echo "${C_GRN}${C_BLD}完成。${C_OFF}"
echo "  结果目录: $OUTPUT_ROOT"
if [ -f "$OUTPUT_ROOT/phase6_collection.json" ]; then
  echo "  汇总文件: $OUTPUT_ROOT/phase6_collection.json"
  "$PY" - "$OUTPUT_ROOT/phase6_collection.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"  运行模式: {d.get('mode')}")
print(f"  优化目标: {d.get('objective')}")
print(f"  计划对数: {d.get('planned_pair_count')}   已完成: {d.get('completed_pair_count')}")
PYEOF
fi
if [ "$EXECUTE" -eq 0 ] && [ "$SUMMARIZE" -eq 0 ]; then
  echo
  echo "  这次只是${C_BLD}生成计划${C_OFF}，没有真的跑。要真跑请加 --execute："
  echo "      ./run_all.sh --objective $OBJECTIVE --execute"
fi
