from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from module5.hls_interface_policy import (
    audit_c_interface_contract,
    audit_c_hls_policy,
    audit_exact_rtl_interface,
    audit_rtl_interface,
    audit_rtl_source_style,
    relative_verilog_paths,
)
from module5.hls_rtl_canonicalizer import canonicalize_hls_rtl
from module5.hls_rtl_wrapper import build_clean_hls_bundle
from project_paths import PROJECT_ROOT
from verilogc2x import check_synthesizable, infer_hls_top_from_cpp_path, write_hls_config


VITIS_ENV_SCRIPT = PROJECT_ROOT / "vitis.sh"


def _collect_verilog_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    found: List[Path] = []
    found.extend(sorted(root.rglob("*.v")))
    found.extend(sorted(root.rglob("*.sv")))
    return found


def _pick_top_verilog(verilog_files: List[Path], top_name: str) -> Path | None:
    if not verilog_files:
        return None

    exact = [path for path in verilog_files if path.stem == top_name]
    if exact:
        return exact[0]

    filtered = [
        path for path in verilog_files
        if "axi" not in path.name.lower() and "control" not in path.name.lower()
    ]
    if filtered:
        return filtered[0]
    return verilog_files[0]


def _validate_verilog_files(verilog_files: List[Path], top_name: str) -> Dict[str, Any]:
    compiler = shutil.which("iverilog")
    if compiler is None:
        return {"success": False, "error": "iverilog is not installed", "command": []}
    if not verilog_files:
        return {"success": False, "error": "no Verilog files to validate", "command": []}
    with tempfile.TemporaryDirectory(prefix="module5_iverilog_") as tmp_dir:
        output = Path(tmp_dir) / "design.out"
        command = [compiler, "-g2012", "-s", top_name, "-o", str(output)]
        command.extend(str(path) for path in verilog_files)
        proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
    return {
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": command,
    }


def run_hls(
    c_path: str | Path,
    out_dir: str | Path,
    *,
    top_name: str = "",
    part: str = "xck24-ubva530-2LV-c",
    clock_ns: float = 10.0,
    work_root: str | Path = "/tmp/module5_hls_work",
    interface_spec: Dict[str, Any] | None = None,
    public_top_name: str = "",
    rtl_cleanup: str = "canonical",
) -> Dict[str, Any]:
    c_path = Path(c_path).resolve()
    out_dir = Path(out_dir).resolve()
    work_root = Path(work_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    if rtl_cleanup not in {"interface_only", "canonical"}:
        return {
            "success": False,
            "status": "invalid_cleanup_mode",
            "stdout": "",
            "stderr": f"Unsupported RTL cleanup mode: {rtl_cleanup}",
            "generated_verilog_path": "",
            "manifest_path": "",
            "top_selected": top_name,
        }

    if not c_path.is_file():
        return {
            "success": False,
            "status": "missing_source",
            "stdout": "",
            "stderr": f"Source file not found: {c_path}",
            "generated_verilog_path": "",
            "manifest_path": "",
            "top_selected": top_name,
        }

    c_policy = audit_c_hls_policy(c_path)
    if not c_policy["success"]:
        return {
            "success": False,
            "status": "interface_policy_rejected",
            "stdout": "",
            "stderr": "C source contains a bus protocol or an HLS optimization pragma.",
            "generated_verilog_path": "",
            "generated_verilog_files": [],
            "manifest_path": "",
            "top_selected": top_name,
            "interface_policy": {"c": c_policy},
        }

    synth_ok, synth_reason = check_synthesizable(str(c_path))
    if not synth_ok:
        return {
            "success": False,
            "status": "unsynthesizable",
            "stdout": "",
            "stderr": synth_reason,
            "generated_verilog_path": "",
            "manifest_path": "",
            "top_selected": top_name,
        }

    selected_top = top_name or infer_hls_top_from_cpp_path(str(c_path))
    interface_spec = dict(interface_spec or {})
    interface_ports = interface_spec.get("ports", [])
    has_interface_contract = isinstance(interface_ports, list) and bool(interface_ports)
    reset_spec = interface_spec.get("reset", {}) if isinstance(interface_spec.get("reset", {}), dict) else {}
    reset_name = str(reset_spec.get("name", "")).strip()
    reset_level = str(reset_spec.get("active_level", "")).strip().lower()
    reset_type = str(reset_spec.get("type", "")).strip().lower()
    hls_reset_mode = "state" if has_interface_contract and reset_name else ("none" if has_interface_contract else None)
    hls_reset_async = (reset_type == "async") if hls_reset_mode == "state" else None
    hls_reset_level = reset_level if reset_level in {"low", "high"} and hls_reset_mode == "state" else None
    c_contract = audit_c_interface_contract(c_path, interface_spec, selected_top)
    if not c_contract["success"]:
        return {
            "success": False,
            "status": "interface_contract_rejected",
            "stdout": "",
            "stderr": "C function arguments do not match the exact hardware interface contract.",
            "generated_verilog_path": "",
            "generated_verilog_files": [],
            "manifest_path": "",
            "top_selected": selected_top,
            "public_top": public_top_name or selected_top,
            "interface_policy": {"c": c_policy, "c_contract": c_contract},
        }
    src_stem = c_path.stem
    dest_hls_dir = out_dir / f"hls_{src_stem}"
    if dest_hls_dir.exists():
        shutil.rmtree(dest_hls_dir)

    with tempfile.TemporaryDirectory(prefix=f"module5_hls_{src_stem}_", dir=str(work_root)) as tmp_hls_dir_str:
        tmp_hls_dir = Path(tmp_hls_dir_str)
        tmp_src = tmp_hls_dir / c_path.name
        shutil.copy2(c_path, tmp_src)

        cfg_path = tmp_hls_dir / "hls_config.cfg"
        write_hls_config(
            str(cfg_path),
            str(tmp_src),
            selected_top,
            part=part,
            clock=f"{clock_ns}ns",
            flow_target="vivado",
            reset_mode=hls_reset_mode,
            reset_async=hls_reset_async,
            reset_level=hls_reset_level,
        )

        stdout_path = tmp_hls_dir / "vpp_stdout.txt"
        stderr_path = tmp_hls_dir / "vpp_stderr.txt"
        shell_cmd = (
            f"source {shlex.quote(str(VITIS_ENV_SCRIPT))} && "
            "v++ -c --mode hls --config ./hls_config.cfg --work_dir ./hls_work"
        )
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            proc = subprocess.run(
                ["bash", "-lc", shell_cmd],
                cwd=str(tmp_hls_dir),
                stdout=out,
                stderr=err,
                text=True,
                timeout=3600,
            )

        shutil.copytree(tmp_hls_dir, dest_hls_dir)

    stdout_text = (dest_hls_dir / "vpp_stdout.txt").read_text(encoding="utf-8", errors="replace")
    stderr_text = (dest_hls_dir / "vpp_stderr.txt").read_text(encoding="utf-8", errors="replace")

    verilog_root = dest_hls_dir / "hls_work" / "hls" / "syn" / "verilog"
    verilog_files = _collect_verilog_files(verilog_root)
    raw_generated_verilog = _pick_top_verilog(verilog_files, selected_top)
    raw_rtl_policy = (
        audit_rtl_interface(raw_generated_verilog)
        if raw_generated_verilog is not None
        else {"success": False, "forbidden_ports": [], "ports": []}
    )
    generated_verilog = raw_generated_verilog
    final_verilog_files = list(verilog_files)
    normalization: Dict[str, Any] = {"success": True, "status": "not_requested"}
    if has_interface_contract and raw_generated_verilog is not None:
        clean_top = public_top_name or selected_top
        clean_dir = dest_hls_dir / "clean_rtl"
        bundle_path = clean_dir / f"{clean_top}__hls_bundle.v"
        clean_path = clean_dir / f"{clean_top}.v"
        normalization = build_clean_hls_bundle(
            verilog_files,
            raw_top=selected_top,
            public_top=clean_top,
            interface=interface_spec,
            output_path=bundle_path,
            core_reset_level=hls_reset_level or "high",
            strip_attributes=rtl_cleanup == "canonical",
        )
        normalization["wrapper_bundle_path"] = str(bundle_path)
        if normalization.get("success"):
            if rtl_cleanup == "canonical":
                canonicalization = canonicalize_hls_rtl(
                    bundle_path,
                    clean_path,
                    top_name=clean_top,
                    interface=interface_spec,
                )
                normalization["canonicalization"] = canonicalization
                normalization["success"] = bool(canonicalization.get("success"))
                if canonicalization.get("success"):
                    generated_verilog = clean_path
                    final_verilog_files = [clean_path]
            else:
                normalization["canonicalization"] = {
                    "success": True,
                    "status": "not_requested",
                    "reason": "pure C/C++ plus HLS control group",
                }
                generated_verilog = bundle_path
                final_verilog_files = [bundle_path]
        normalization["status"] = "ok" if normalization.get("success") else "failed"
    rtl_policy = (
        audit_rtl_interface(generated_verilog)
        if generated_verilog is not None
        else {"success": False, "forbidden_ports": [], "ports": []}
    )
    final_top = public_top_name or selected_top
    exact_rtl_contract = (
        audit_exact_rtl_interface(generated_verilog, interface_spec, final_top)
        if has_interface_contract and generated_verilog is not None
        else {"success": not has_interface_contract, "status": "not_requested"}
    )
    rtl_source_style = (
        audit_rtl_source_style(generated_verilog, final_top)
        if has_interface_contract and generated_verilog is not None
        else {"success": not has_interface_contract, "status": "not_requested"}
    )
    rtl_style_gate_passed = (
        bool(rtl_source_style.get("success"))
        if rtl_cleanup == "canonical"
        else True
    )
    syntax_validation = _validate_verilog_files(final_verilog_files, final_top)

    manifest_path = dest_hls_dir / "manifest.json"
    manifest = {
        "src": str(c_path),
        "copied_source": str(dest_hls_dir / c_path.name),
        "hls_dir": str(dest_hls_dir),
        "verilog_root": str(verilog_root),
        "top_selected": selected_top,
        "generated_verilog_path": str(generated_verilog) if generated_verilog else "",
        "generated_verilog_files": relative_verilog_paths(final_verilog_files, dest_hls_dir),
        "raw_generated_verilog_path": str(raw_generated_verilog) if raw_generated_verilog else "",
        "raw_generated_verilog_files": relative_verilog_paths(verilog_files, dest_hls_dir),
        "flow_target": "vivado",
        "part": part,
        "clock_ns": clock_ns,
        "interface_policy": {
            "c": c_policy,
            "c_contract": c_contract,
            "raw_rtl": raw_rtl_policy,
            "rtl": rtl_policy,
            "exact_rtl_contract": exact_rtl_contract,
            "rtl_source_style": rtl_source_style,
        },
        "interface_normalization": normalization,
        "syntax_validation": syntax_validation,
        "compiler_policy": {
            "default_slave_interface": "none",
            "clock_enable": False,
            "automatic_loop_pipelining": False,
            "hls_optimization_pragmas": False,
            "rtl_cleanup": rtl_cleanup,
            "post_hls_rtl_canonicalization": (
                "deterministic_yosys" if rtl_cleanup == "canonical" else "disabled"
            ),
            "reset_mode": hls_reset_mode,
            "reset_async": hls_reset_async,
            "reset_level": hls_reset_level,
        },
        "returncode": proc.returncode,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    compiler_success = proc.returncode == 0 and raw_generated_verilog is not None
    success = (
        compiler_success
        and bool(raw_rtl_policy.get("success"))
        and bool(normalization.get("success"))
        and bool(rtl_policy.get("success"))
        and bool(exact_rtl_contract.get("success"))
        and rtl_style_gate_passed
        and bool(syntax_validation.get("success"))
    )
    if success:
        status = "ok"
    elif compiler_success and not normalization.get("success"):
        status = "interface_normalization_failed"
    elif compiler_success and not syntax_validation.get("success"):
        status = "rtl_syntax_failed"
    elif compiler_success:
        status = "interface_policy_rejected"
    else:
        status = "failed"
    return {
        "success": success,
        "status": status,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "manifest_path": str(manifest_path),
        "generated_verilog_path": str(generated_verilog) if generated_verilog else "",
        "generated_verilog_files": [str(path) for path in final_verilog_files],
        "raw_generated_verilog_path": str(raw_generated_verilog) if raw_generated_verilog else "",
        "raw_generated_verilog_files": [str(path) for path in verilog_files],
        "top_selected": selected_top,
        "public_top": public_top_name or selected_top,
        "interface_policy": {
            "c": c_policy,
            "c_contract": c_contract,
            "raw_rtl": raw_rtl_policy,
            "rtl": rtl_policy,
            "exact_rtl_contract": exact_rtl_contract,
            "rtl_source_style": rtl_source_style,
        },
        "interface_normalization": normalization,
        "syntax_validation": syntax_validation,
    }
