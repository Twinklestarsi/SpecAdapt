from pathlib import Path
import re

RUN_ROOT = Path("/home/2531930/nas/2531930-1198/llms/Vivado/dc_asap7_20260907")
OUT = Path("/home/2531930/nas/2531930-1198/llms/Vivado/dc_asap7_results_20260907.md")
BATCH_LOG = RUN_ROOT / "batch.log"

SPECS = [
    ("001", "NV_NVDLA_CDMA_status", "001_spec_2_NV_NVDLA_CDMA_status.v", 2553, 1630),
    ("002", "NV_NVDLA_CDMA_dma_mux", "002_spec_3_NV_NVDLA_CDMA_dma_mux.v", 2085, 1337),
    ("003", "NV_NVDLA_GLB_csb", "003_spec_4_NV_NVDLA_GLB_csb.v", 323, 257),
    ("004", "NV_NVDLA_CDMA_WT_wrr_arb", "004_spec_6_NV_NVDLA_CDMA_WT_wrr_arb.v", 671, 415),
    ("005", "NV_NVDLA_CDMA_img", "005_spec_11_NV_NVDLA_CDMA_img.v", 464, 384),
    ("006", "thermostat", "006_thermostat.v", 329, 306),
    ("007", "sbox", "007_sbox.sv", 267, 266),
    ("008", "delete_node_binary_search_tree", "008_search_binary_search_tree.sv", 265, 251),
    ("009", "NV_NVDLA_CACC_CALC_int8", "009_NV_NVDLA_CACC_CALC_int8.v", 289, 208),
    ("010", "halfband_fir", "010_halfband_fir.sv", 248, 203),
]


RUN_DIR_OVERRIDES = {
    "003": "003_NV_NVDLA_GLB_csb_repaired_on17_1_20260907",
    "004": "004_NV_NVDLA_CDMA_WT_wrr_arb_repaired_on17_1_retry_20260907",
    "005": "005_NV_NVDLA_CDMA_img_repaired_compat3_on17_1_20260907",
}


def read_text(path):
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""

def number(pattern, text, flags=0):
    match = re.search(pattern, text, flags)
    return float(match.group(1)) if match else None

def integer(pattern, text):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None

def fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"

def dynamic_power_mw(text):
    match = re.search(r"Total Dynamic Power\s*=\s*([0-9.eE+-]+)\s*(mW|uW|µW)", text)
    if not match:
        return None
    value = float(match.group(1))
    return value if match.group(2) == "mW" else value / 1000.0

def serial_chunk(all_text, ident):
    start = all_text.find(f"=== START {ident} ")
    if start < 0:
        return ""
    end = all_text.find(f"=== END {ident} ", start)
    if end < 0:
        return all_text[start:]
    return all_text[start:end + 200]

def parse_one(ident, top, filename, raw_lines, code_lines):
    module_dir = RUN_ROOT / RUN_DIR_OVERRIDES.get(ident, f"{ident}_{top}")
    status_text = read_text(module_dir / "status.txt")
    status_match = re.search(r"^status=(\S+)", status_text, re.M)
    dc_status = status_match.group(1) if status_match else "MISSING"
    if ident in {"009", "010"} or ident in RUN_DIR_OVERRIDES:
        log_text = read_text(module_dir / "dc.log")
    else:
        log_text = serial_chunk(read_text(BATCH_LOG), ident)
    check_text = read_text(module_dir / "check_design.rpt")
    area_text = read_text(module_dir / "area.rpt")
    timing_text = read_text(module_dir / "timing.rpt")
    power_text = read_text(module_dir / "power.rpt")
    object_text = read_text(module_dir / "object_counts.txt")
    combined = "\n".join((log_text, check_text, area_text, timing_text, power_text, object_text))

    errors = re.findall(r"^Error:.*$", combined, re.M)
    unresolved = []
    for name in re.findall(r"Unable to resolve reference '([^']+)'", combined):
        if name not in unresolved:
            unresolved.append(name)
    has_blackbox = bool(re.search(r"contains black box|contains black boxes|unknown components", combined, re.I))
    area = number(r"Total cell area:\s+([0-9.eE+-]+)", area_text)
    cells = integer(r"Number of cells:\s+([0-9]+)", area_text)
    nets = integer(r"Number of nets:\s+([0-9]+)", area_text)
    if cells is None:
        cells = integer(r"nodes\s+([0-9]+)", object_text)
    if nets is None:
        nets = integer(r"wires\s+([0-9]+)", object_text)
    arrivals = [float(x) for x in re.findall(r"data arrival time\s+([-+0-9.eE]+)", timing_text)]
    delay = max(arrivals) if arrivals else None
    slacks = [float(x) for x in re.findall(r"slack\s+\([^)]*\)\s+([-+0-9.eE]+)", timing_text)]
    wns = min(slacks) if slacks else None
    dynamic_power = dynamic_power_mw(power_text)
    leakage = number(r"Cell Leakage Power\s*=\s*([0-9.eE+-]+)\s*nW", power_text)

    missing_simulate = bool(re.search(
        r"(Unable to open file|No such file|can't open|missing).*simulate_x_tick\.vh",
        combined,
        re.I,
    ))
    notes = []
    repair_notes = {
        "003": "已修复：补充官方 NV_NVDLA_GLB_CSB_reg",
        "004": "已修复：增加 DC 兼容头文件并定义 x_or_0",
        "005": "已修复：补充官方 IMG/FIFO 依赖；RAM/时钟门控使用 DC 兼容模型；保留一个无输出 NV_BLKBOX_SINK 汇点",
    }
    if ident in repair_notes:
        notes.append(repair_notes[ident])
    if unresolved:
        notes.append("未解析引用: " + ", ".join(unresolved))
    if has_blackbox:
        notes.append("报告含 black box，数值只代表部分/外壳")
    if missing_simulate:
        notes.append("缺少 simulate_x_tick.vh，未得到有效设计")
    if "Target library does not contain any 2-1 multiplexor" in combined:
        notes.append("ASAP7 库无 2:1 mux，DC 使用其他逻辑实现")
    if errors:
        first_error = re.sub(r"\s+", " ", errors[0]).strip()
        notes.append(first_error[:180])

    if missing_simulate or not area_text.strip() or (errors and cells in (None, 0)):
        result = "FAIL"
        area = cells = nets = delay = wns = dynamic_power = leakage = None
    elif unresolved or has_blackbox:
        result = "INCOMPLETE"
    else:
        result = "PASS"

    report_rel = f"dc_asap7_20260907/{module_dir.name}"
    return {
        "id": ident, "top": top, "file": filename, "raw": raw_lines, "code": code_lines,
        "nodes": cells, "wires": nets, "delay": delay, "area": area,
        "power": dynamic_power, "leakage": leakage, "wns": wns,
        "dc_status": dc_status, "result": result,
        "notes": "；".join(dict.fromkeys(notes)) if notes else "—",
        "report_rel": report_rel,
    }

rows = [parse_one(*spec) for spec in SPECS]
pass_count = sum(row["result"] == "PASS" for row in rows)
incomplete_count = sum(row["result"] == "INCOMPLETE" for row in rows)
fail_count = sum(row["result"] == "FAIL" for row in rows)

lines = []
lines.append("# DC + ASAP7 单模块综合结果")
lines.append("")
lines.append("> 运行日期：2026-09-07；DC 主机：172.31.17.1；DC 版本：W-2024.09-SP3。")
lines.append("")
lines.append(f"本次共处理 10 个 RTL 文件：完整通过 {pass_count} 个，未解析依赖 {incomplete_count} 个，失败 {fail_count} 个。")
lines.append("")
lines.append("## 结果表")
lines.append("")
lines.append("| Des. | RTL 文件 | Lines | Code lines | Nodes | Wires | Delay (ps) | Area (um²) | Power (mW) | WNS (ps) | Result | Notes |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
for row in rows:
    report_link = f"[{row['file']}]({row['report_rel']}/)"
    lines.append(
        f"| {row['top']} | {report_link} | {row['raw']} | {row['code']} | "
        f"{row['nodes'] if row['nodes'] is not None else '—'} | "
        f"{row['wires'] if row['wires'] is not None else '—'} | "
        f"{fmt(row['delay'])} | {fmt(row['area'], 6)} | {fmt(row['power'], 6)} | "
        f"{fmt(row['wns'])} | {row['result']} | {row['notes']} |"
    )

lines.append("")
lines.append("## 指标口径与统一设置")
lines.append("")
lines.append("- Lines：源文件物理总行数，包含注释和空行；Code lines：去掉注释后剩余的非空代码行。")
lines.append("- Nodes：DC 映射后的叶子标准单元实例数；Wires：DC 报告中的 net 数。")
lines.append("- Delay：DC 最坏 max path 的 data arrival time，ASAP7 报告单位为 ps。")
lines.append("- Area：DC 的 Total cell area；Power：DC 的 Total Dynamic Power。")
lines.append("- 统一约束：有时钟模块使用周期 1000 ps、输入/输出延迟各 100 ps、时钟不确定度 50 ps；无时钟模块使用 1000 ps 的 input-to-output max delay。")
lines.append("- 功耗没有使用 VCD 波形，而是统一使用输入静态概率 0.5、输入翻转率 0.1、输入转换时间 10 ps、输出负载 1 fF，因此功耗是可比较的估算值，不是实测芯片功耗。")
lines.append("- WNS 是最坏负 slack；负值表示在上述约束下没有达到目标，但不等于 DC 失败。")
lines.append("")
lines.append("## ASAP7 库")
lines.append("")
lines.append("- DC 直接读取 /home/2531930/asap7_db 下的 5 个 RVT/TT CCS .db：AO、INVBUF、OA、SEQ、SIMPLE。")
lines.append("- 每个模块目录还保存 area.rpt、timing.rpt、power.rpt、qor.rpt、constraints.rpt、映射后 Verilog 和 SDC（若该模块综合完整）。")
lines.append("- 003, 004, and 005 were repaired with isolated dependency or compatibility inputs; the table uses the repaired runs.")
lines.append("")
lines.append("## 目录")
lines.append("")
lines.append("- 运行目录：dc_asap7_20260907/")
lines.append("- 任务日志：logs/20260907-154821-Vivado-single-module.md")
lines.append("")
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT}")
for row in rows:
    print(row["id"], row["top"], row["result"], row["nodes"], row["wires"], row["delay"], row["area"], row["power"], row["notes"])
