#!/usr/bin/env python3
"""Build a table-aligned CSV/Markdown report for the external-agent rerun.

The report keeps the two supplied CSV files immutable.  It expands every
source-table row for each locally runnable proxy agent and joins fresh DC
metrics/token usage by the corresponding nvdla22 case.  Official LongRTL,
NVIDIA Spec2RTL-Agent, and VeriAgent values are copied only as source-table
context; they are not relabeled as measurements from the proxy agents.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
AREA_TABLE = ROOT / "paper/C_RTLgen/ICCAD26-C-RTLgen-sub/doc/evaluate_area.csv"
TIMING_TABLE = ROOT / "paper/C_RTLgen/ICCAD26-C-RTLgen-sub/doc/evaluate_timing.csv"
RUN_ROOT = ROOT / "runs/external_spec_agents/20260919_evaluate_agents"
OUT_CSV = RUN_ROOT / "evaluate_agents_20260919.csv"
OUT_MD = RUN_ROOT / "evaluate_agents_20260919.md"

AGENTS = ("verisure", "ace_rtl", "spec2rtl_mini")
AGENT_LABEL = {
    "verisure": "Veri-Sure (proxy; not VeriAgent)",
    "ace_rtl": "ACE-RTL (runnable baseline)",
    "spec2rtl_mini": "Spec2RTL Mini (community proxy; not NVIDIA official)",
}
AGENT_SHORT = {
    "verisure": "Veri-Sure",
    "ace_rtl": "ACE-RTL",
    "spec2rtl_mini": "Spec2RTL Mini",
}

# The C-first table names are mapped to the public nvdla22 cases used by the
# existing adapters.  Two table labels intentionally refer to the same raw
# case; the CSV retains both source rows so no input-table line disappears.
CASE_MAP = {
    "CSRNG": "spec_21_csrng",
    "CACC_CALC": "nvdla_official_NV_NVDLA_CACC_CALC_int8",
    "CDMA_IMG": "spec_5_NV_NVDLA_CDMA_IMG_ctrl",
    "CDMA_WRR": "spec_6_NV_NVDLA_CDMA_WT_wrr_arb",
    "CDMA_WT_WRR": "spec_6_NV_NVDLA_CDMA_WT_wrr_arb",
    "INT_RELU": "nvdla_official_NV_NVDLA_SDP_HLS_X_int_relu",
    "INT_MUL": "nvdla_official_NV_NVDLA_SDP_HLS_X_int_mul",
    "GLB_IC": "spec_4_NV_NVDLA_GLB_ic",
    "CDMA_STATUS": "spec_2_NV_NVDLA_CDMA_status",
    "DMA_MUX": "spec_3_NV_NVDLA_CDMA_dma_mux",
    "INT_SUM": "nvdla_official_int_sum_block",
    # AQB has no matching raw case in the nvdla22 pilot manifest.
    "AQB": None,
}

SOURCE_METHODS = (
    ("Ours", "ours"),
    ("deepseek-v4-flash", "deepseek_v4_flash"),
    ("Kimi-K2.5", "kimi_k2_5"),
    ("LongRTL", "longrtl"),
    ("Spec2RTL", "spec2rtl"),
    ("VeriAgent", "veriagent"),
    ("ACE-RTL", "ace_rtl"),
)


def read_source_table(path: Path) -> list[dict[str, str]]:
    """Read only the first header-width fields, tolerating trailing commas."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows: list[dict[str, str]] = []
        for fields in reader:
            if not fields or not fields[0].strip():
                continue
            fields = (fields + [""] * len(header))[: len(header)]
            rows.append({key: fields[idx].strip() for idx, key in enumerate(header)})
    return rows


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {"/", "-", "—", "NA", "N/A"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def source_pair(row: dict[str, str], method: str) -> tuple[float | None, float | None]:
    return (
        parse_number(row.get(f"{method}_面积(μm²)")),
        parse_number(row.get(f"{method}_延迟(ps)")),
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def result_and_usage(agent: str, case_id: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    result_path = RUN_ROOT / agent / case_id / "result.json"
    result = load_json(result_path)
    usage = result.get("token_usage")
    if not isinstance(usage, dict):
        # The runner intentionally turns a non-zero adapter exit into a
        # generation_failed result.  The adapter's own marker still records
        # provider usage, so recover it without estimating from text length.
        adapter_path = RUN_ROOT / agent / case_id / "generation" / "result.json"
        adapter = load_json(adapter_path)
        usage = adapter.get("token_usage") or adapter.get("usage")
    return result, usage if isinstance(usage, dict) else {}, result_path


def metric(result: dict[str, Any], goal: str, key: str) -> Any:
    block = (result.get("synthesis") or {}).get(goal) or {}
    if block.get("status") != "passed":
        return None
    return ((block.get("metrics") or {}).get(key))


def status(result: dict[str, Any], goal: str) -> str:
    block = (result.get("synthesis") or {}).get(goal) or {}
    return str(block.get("status") or "not_run")


def fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "—"
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def fmt_token(usage: dict[str, Any]) -> str:
    total = parse_number(usage.get("total_tokens"))
    prompt = parse_number(usage.get("prompt_tokens"))
    completion = parse_number(usage.get("completion_tokens"))
    if total is None:
        return "—"
    return f"{int(total)} ({fmt(prompt, 0)}+{fmt(completion, 0)})"


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def build_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    area_rows = read_source_table(AREA_TABLE)
    timing_rows = read_source_table(TIMING_TABLE)
    source_rows: list[tuple[str, dict[str, str]]] = [("area", row) for row in area_rows]
    source_rows.extend(("timing", row) for row in timing_rows)
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}

    for table_name, source in source_rows:
        design = source["设计"]
        case_id = CASE_MAP.get(design)
        for agent in AGENTS:
            result: dict[str, Any] = {}
            usage: dict[str, Any] = {}
            result_path = RUN_ROOT / agent / (case_id or "unmapped") / "result.json"
            if case_id is not None:
                cache_key = f"{agent}\0{case_id}"
                if cache_key not in cache:
                    loaded, loaded_usage, loaded_path = result_and_usage(agent, case_id)
                    cache[cache_key] = {
                        "result": loaded,
                        "usage": loaded_usage,
                        "path": loaded_path,
                    }
                cached = cache[cache_key]
                result = cached["result"]
                usage = cached["usage"]
                result_path = cached["path"]

            row: dict[str, Any] = {
                "source_table": table_name,
                "design": design,
                "case_id": case_id or "",
                "runnable_agent": agent,
                "agent_display": AGENT_LABEL[agent],
                "run_status": (result.get("terminal_status") if result else "unmapped_no_raw_case"),
                "syntax_status": (result.get("syntax") or {}).get("status", "not_run") if result else "not_run",
                "area_status": status(result, "area") if result else "not_run",
                "timing_status": status(result, "timing") if result else "not_run",
                "measured_area_um2": metric(result, "area", "area") if result else None,
                "measured_timing_ps": metric(result, "timing", "timing_ps") if result else None,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "request_count": usage.get("request_count"),
                "token_source": usage.get("source", "unavailable") if usage else "unavailable",
                "official_longrtl_status": "not_run_official_source_missing",
                "official_spec2rtl_status": "not_run_official_source_missing",
                "official_veriagent_status": "not_run_official_source_missing",
                "result_json": rel_path(result_path),
                "notes": (
                    "no matching nvdla22 raw case"
                    if case_id is None
                    else "fresh no-JG proxy run; official target is not this proxy"
                ),
            }
            for method, slug in SOURCE_METHODS:
                area_value, timing_value = source_pair(source, method)
                row[f"input_{slug}_area_um2"] = area_value
                row[f"input_{slug}_timing_ps"] = timing_value
            rows.append(row)
    return rows, cache


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_csv(rows: list[dict[str, Any]]) -> None:
    base_fields = [
        "source_table", "design", "case_id", "runnable_agent", "agent_display",
        "run_status", "syntax_status", "area_status", "timing_status",
        "measured_area_um2", "measured_timing_ps", "prompt_tokens",
        "completion_tokens", "total_tokens", "request_count", "token_source",
        "official_longrtl_status", "official_spec2rtl_status",
        "official_veriagent_status", "result_json", "notes",
    ]
    source_fields = [
        f"input_{slug}_{metric_name}"
        for _, slug in SOURCE_METHODS
        for metric_name in ("area_um2", "timing_ps")
    ]
    fields = base_fields + source_fields
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for agent in AGENTS:
        agent_rows = [r for r in rows if r["runnable_agent"] == agent]
        # A design can occur in both supplied tables (INT_RELU) and two
        # labels can intentionally share one raw case (CDMA_WRR/WRR).  Count
        # generation/token/PPA outcomes once per unique raw case, while still
        # reporting every source-table row in the CSV.
        per_case: dict[str, dict[str, Any]] = {}
        for row in agent_rows:
            case_id = row.get("case_id")
            if case_id:
                per_case.setdefault(str(case_id), row)
        case_rows = list(per_case.values())
        known = [r for r in case_rows if parse_number(r.get("total_tokens")) is not None]
        result[agent] = {
            "table_rows": len(agent_rows),
            "unique_cases": len(case_rows),
            "candidate": sum(r["run_status"] in {"ppa_complete", "candidate_syntax_failed"} for r in case_rows),
            "syntax": sum(r["syntax_status"] == "passed" for r in case_rows),
            "area": sum(r["area_status"] == "passed" for r in case_rows),
            "timing": sum(r["timing_status"] == "passed" for r in case_rows),
            "token_known": len(known),
            "prompt_tokens": sum(parse_number(r.get("prompt_tokens")) or 0 for r in known),
            "completion_tokens": sum(parse_number(r.get("completion_tokens")) or 0 for r in known),
            "total_tokens": sum(parse_number(r.get("total_tokens")) or 0 for r in known),
        }
    return result


def write_md(rows: list[dict[str, Any]]) -> None:
    agg = aggregate(rows)
    created = time.strftime("%Y-%m-%d %H:%M:%S %z")
    lines: list[str] = [
        "# LongRTL / Spec2RTL / VeriAgent 对应实验数据重跑报告",
        "",
        f"生成时间：{created}；模型：`deepseek-v4-flash`；CSV 行数：{len(rows)}。",
        "",
        "## 先说明运行对象",
        "",
        "本机没有 LongRTL、NVIDIA 官方 Spec2RTL-Agent、VeriAgent 的可执行官方源码/模型入口，因此这三者本轮没有冒充运行。原表中这三个名称对应的数值只作为输入表历史值保留。实际重跑的是三个可运行代理：Veri-Sure（不是 VeriAgent）、ACE-RTL，以及社区 mini-spec2rtl-agent（不是 NVIDIA 官方 Spec2RTL-Agent）。",
        "",
        "每一行输入表都展开为三个可运行代理；`AQB` 在现有 nvdla22 原始 case 清单中没有对应 spec/RTL，因此标为 `unmapped_no_raw_case`。`CDMA_WRR` 和 `CDMA_WT_WRR` 映射到同一个原始 `spec_6_NV_NVDLA_CDMA_WT_wrr_arb`，但两个输入表行均保留。",
        "",
        "## 运行口径",
        "",
        "- 代理生成后执行 Icarus 语法检查，并分别执行 DC AREA 与 DC TIMING；`measured_area_um2` 取 AREA 目标的 `total_cell_area`，`measured_timing_ps` 取 TIMING 目标的 `data_arrival_time_ps`。",
        "- 本批次明确跳过 JasperGold/JG，未做功能等价证明；PPA 是 no-JG provisional 结果，失败目标不填估计值。",
        "- token 来自 provider 返回的 usage 元数据；生成失败时从该 case 的 adapter `generation/result.json` 回读，绝不按字符数估算。",
        "- ACE-RTL 本批次使用 `max_tokens=32768`，避免 deepseek-v4-flash 的 reasoning 内容耗尽 8192 输出预算；三类代理统一使用 `deepseek-v4-flash`。",
        "",
        "## 批次汇总",
        "",
        "| 代理 | 输入表行数 | 唯一 case | 候选生成 | 语法通过 | Area DC | Timing DC | token 已知 | prompt | completion | total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for agent in AGENTS:
        a = agg[agent]
        lines.append(
            f"| {AGENT_SHORT[agent]} | {a['table_rows']} | {a['unique_cases']} | {a['candidate']}/{a['unique_cases']} | "
            f"{a['syntax']}/{a['unique_cases']} | {a['area']}/{a['unique_cases']} | {a['timing']}/{a['unique_cases']} | "
            f"{a['token_known']}/{a['unique_cases']} | {int(a['prompt_tokens'])} | {int(a['completion_tokens'])} | {int(a['total_tokens'])} |"
        )

    lines.extend([
        "",
        "## 新鲜重跑的 area / timing / token",
        "",
        "单元格式为 `area_um² / timing_ps / total_tokens`；token 括号内为 `prompt+completion`。数值来自新 run，不是输入 CSV 的历史值。",
        "",
        "| 输入表 | 设计 | 原始 case | Veri-Sure | ACE-RTL | Spec2RTL Mini |",
        "|---|---|---|---|---|---|",
    ])
    for row_idx in range(0, len(rows), len(AGENTS)):
        group = rows[row_idx : row_idx + len(AGENTS)]
        first = group[0]
        cells = [f"{first['source_table']}", first["design"], first["case_id"] or "—"]
        by_agent = {row["runnable_agent"]: row for row in group}
        for agent in AGENTS:
            row = by_agent[agent]
            measured = f"{fmt(row['measured_area_um2'])} / {fmt(row['measured_timing_ps'])} / {fmt_token({'total_tokens': row.get('total_tokens'), 'prompt_tokens': row.get('prompt_tokens'), 'completion_tokens': row.get('completion_tokens')})}"
            cells.append(f"{measured}<br>`{row['run_status']}`")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## 输入表中的历史列（仅作对照）",
        "",
        "下面保留原 CSV 中的 LongRTL、Spec2RTL、VeriAgent、ACE-RTL 数值；这些不是本轮官方 agent 重新运行得到的数值。",
        "",
        "| 输入表 | 设计 | LongRTL area/timing | Spec2RTL area/timing | VeriAgent area/timing | ACE-RTL area/timing |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row_idx in range(0, len(rows), len(AGENTS)):
        row = rows[row_idx]
        def source_cell(slug: str) -> str:
            return f"{fmt(row.get(f'input_{slug}_area_um2'))} / {fmt(row.get(f'input_{slug}_timing_ps'))}"
        lines.append("| " + " | ".join([
            row["source_table"], row["design"], source_cell("longrtl"),
            source_cell("spec2rtl"), source_cell("veriagent"), source_cell("ace_rtl"),
        ]) + " |")

    lines.extend([
        "",
        "## 失败与可追溯性",
        "",
        "- 失败状态（如 `generation_failed`、`candidate_syntax_failed`）保留在 CSV 的 `run_status`、`syntax_status`、`area_status`、`timing_status` 中；失败不填 PPA。",
        f"- 每个 case 的完整结果、adapter 日志、候选 RTL、语法日志和 DC 报告位于 `{rel_path(RUN_ROOT)}/<agent>/<case>/`。",
        f"- 逐行机器可读结果：`{rel_path(OUT_CSV)}`；本报告本身是跨三类代理的最终汇总。",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows, _ = build_rows()
    write_csv(rows)
    write_md(rows)
    print(json.dumps({"csv": str(OUT_CSV), "md": str(OUT_MD), "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
