from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from spec_analyze.schema import normalize_interface


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_PORT_DECL_RE = re.compile(
    r"^\s*(input|output|inout)\b"
    r"(?:\s+(?:wire|reg|logic|signed|unsigned))*"
    r"(?:\s*\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\])?\s+([^;]+);",
    re.IGNORECASE | re.MULTILINE,
)


def _module_block(text: str, module_name: str) -> str:
    match = re.search(
        rf"\bmodule\s+{re.escape(module_name)}\b.*?\bendmodule\b",
        text,
        re.DOTALL,
    )
    return match.group(0) if match else ""


def _parse_core_ports(module_text: str) -> Dict[str, Dict[str, Any]]:
    ports: Dict[str, Dict[str, Any]] = {}
    for match in _PORT_DECL_RE.finditer(module_text):
        direction = match.group(1).lower()
        msb = match.group(2)
        lsb = match.group(3)
        width = abs(int(msb) - int(lsb)) + 1 if msb is not None and lsb is not None else 1
        for raw_name in match.group(4).split(","):
            token = raw_name.split("=")[0].strip()
            name_match = re.search(r"([A-Za-z_$][\w$]*)\s*$", token)
            if name_match:
                ports[name_match.group(1)] = {"direction": direction, "width": width}
    return ports


def _width_decl(width: int) -> str:
    return "" if width == 1 else f" [{width - 1}:0]"


def _zero_extend(signal: str, source_width: int, target_width: int) -> str:
    if source_width == target_width:
        return signal
    if source_width > target_width:
        return f"{signal}[{target_width - 1}:0]" if target_width > 1 else f"{signal}[0]"
    return f"{{{{{target_width - source_width}{{1'b0}}}}, {signal}}}"


def _reset_connection(signal: str, source_level: str, target_level: str) -> str:
    if source_level in {"low", "high"} and target_level in {"low", "high"}:
        return signal if source_level == target_level else f"~{signal}"
    return signal


def build_clean_hls_bundle(
    verilog_files: Iterable[str | Path],
    *,
    raw_top: str,
    public_top: str,
    interface: Dict[str, Any],
    output_path: str | Path,
    core_reset_level: str = "high",
    strip_attributes: bool = True,
) -> Dict[str, Any]:
    """Place an exact hardware-native wrapper before a renamed HLS core."""
    files = [Path(path) for path in verilog_files]
    normalized = normalize_interface(interface)
    desired_ports = normalized["ports"]
    output_path = Path(output_path)

    if not desired_ports:
        return {"success": False, "error": "interface manifest has no ports"}
    if not _IDENTIFIER_RE.match(raw_top) or not _IDENTIFIER_RE.match(public_top):
        return {"success": False, "error": "invalid top module name"}

    top_file = None
    top_text = ""
    core_block = ""
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        block = _module_block(text, raw_top)
        if block:
            top_file = path
            top_text = text
            core_block = block
            break
    if top_file is None:
        return {"success": False, "error": f"raw top module not found: {raw_top}"}

    core_ports = _parse_core_ports(core_block)
    manifest_by_name = {port["name"]: port for port in desired_ports}
    clock_name = normalized["clock"]["name"]
    reset_name = normalized["reset"]["name"]
    manifest_reset_level = normalized["reset"]["active_level"]
    data_names = {
        port["name"]
        for port in desired_ports
        if port["role"] not in {"clock", "reset"}
    }

    special_core_ports = {name for name in ("ap_clk", "ap_rst", "ap_rst_n") if name in core_ports}
    unexpected = sorted(set(core_ports) - data_names - special_core_ports)
    missing = sorted(data_names - set(core_ports))
    if unexpected or missing:
        return {
            "success": False,
            "error": "HLS core ports do not match the interface manifest",
            "unexpected_core_ports": unexpected,
            "missing_core_ports": missing,
        }
    if "ap_clk" in core_ports and not clock_name:
        return {"success": False, "error": "HLS core has ap_clk but manifest has no clock"}
    if {"ap_rst", "ap_rst_n"} & set(core_ports) and not reset_name:
        return {"success": False, "error": "HLS core has reset but manifest has no reset"}

    for name in data_names:
        desired = manifest_by_name[name]
        core = core_ports[name]
        if desired["direction"] != core["direction"]:
            return {
                "success": False,
                "error": f"direction mismatch for {name}",
                "manifest_direction": desired["direction"],
                "core_direction": core["direction"],
            }

    core_name = f"{public_top}__impl"
    if core_name == raw_top:
        core_name = f"{raw_top}__core"

    declarations: List[str] = []
    wires: List[str] = []
    assigns: List[str] = []
    connections: List[str] = []

    for port in desired_ports:
        declarations.append(
            f"  {port['direction']}{_width_decl(port['width'])} {port['name']};"
        )

    if "ap_clk" in core_ports:
        connections.append(f"    .core_clk({clock_name})")
    if "ap_rst" in core_ports:
        reset_connection = _reset_connection(reset_name, manifest_reset_level, core_reset_level)
        connections.append(f"    .core_reset({reset_connection})")
    if "ap_rst_n" in core_ports:
        reset_connection = _reset_connection(reset_name, manifest_reset_level, "low")
        connections.append(f"    .core_reset({reset_connection})")

    for port in desired_ports:
        if port["role"] in {"clock", "reset"}:
            continue
        name = port["name"]
        desired_width = int(port["width"])
        core_width = int(core_ports[name]["width"])
        direction = port["direction"]

        if direction == "input":
            connection = _zero_extend(name, desired_width, core_width)
        elif direction == "output" and desired_width != core_width:
            core_signal = f"{name}__impl"
            wires.append(f"  wire{_width_decl(core_width)} {core_signal};")
            assigns.append(
                f"  assign {name} = {_zero_extend(core_signal, core_width, desired_width)};"
            )
            connection = core_signal
        elif direction == "inout" and desired_width != core_width:
            return {"success": False, "error": f"cannot adapt inout width for {name}"}
        else:
            connection = name
        connections.append(f"    .{name}({connection})")

    port_lines = ",\n".join(f"  {port['name']}" for port in desired_ports)
    connection_lines = ",\n".join(connections)
    wrapper_parts = [
        "`timescale 1ns / 1ps",
        "",
        f"module {public_top} (",
        port_lines,
        ");",
        *declarations,
    ]
    if wires:
        wrapper_parts.extend(["", *wires])
    if assigns:
        wrapper_parts.extend(["", *assigns])
    wrapper_parts.extend(
        [
            "",
            f"  {core_name} u_impl (",
            connection_lines,
            "  );",
            "",
            f"endmodule // {public_top}",
            "",
        ]
    )
    wrapper = "\n".join(wrapper_parts)

    bundled_sources: List[str] = [wrapper]
    for path in files:
        source = path.read_text(encoding="utf-8", errors="replace")
        if path == top_file:
            original_block = _module_block(source, raw_top)
            transformed_block, replacements = re.subn(
                rf"(\bmodule\s+){re.escape(raw_top)}\b",
                rf"\g<1>{core_name}",
                original_block,
                count=1,
            )
            if replacements != 1:
                return {"success": False, "error": "failed to rename HLS core"}
            transformed_block = re.sub(r"\bap_clk\b", "core_clk", transformed_block)
            transformed_block = re.sub(r"\b(?:ap_rst|ap_rst_n)\b", "core_reset", transformed_block)
            source = source.replace(original_block, transformed_block, 1)
        if strip_attributes:
            # Attribute blocks contain whitespace after "(*"; an always @ (*)
            # sensitivity list does not.  Requiring whitespace avoids corrupting
            # combinational always blocks while removing generator metadata.
            source = re.sub(r"\(\*\s+.*?\*\)", "", source, flags=re.DOTALL)
        bundled_sources.append(source.strip() + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(bundled_sources), encoding="utf-8")
    return {
        "success": True,
        "output_path": str(output_path),
        "public_top": public_top,
        "core_top": core_name,
        "interface": normalized,
        "raw_top_file": str(top_file),
        "adapted_width_ports": sorted(
            name
            for name in data_names
            if int(manifest_by_name[name]["width"]) != int(core_ports[name]["width"])
        ),
        "core_reset_level": core_reset_level,
        "stripped_attributes": strip_attributes,
    }
