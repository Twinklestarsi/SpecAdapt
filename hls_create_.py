import argparse
import os
import shutil
import sys

import vitis  # Vitis Python CLI package  :contentReference[oaicite:3]{index=3}


def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vitis 2025.2: create/build an HLS component from a single C file (example.c)"
    )
    parser.add_argument("--workspace", required=True, help="Workspace directory (will be created if missing)")
    parser.add_argument("--name", required=True, help="HLS component name")
    parser.add_argument("--src", required=True, help="Path to example.c")
    parser.add_argument("--top", required=True, help="Top function name inside example.c")

    tgt = parser.add_mutually_exclusive_group(required=True)
    tgt.add_argument("--part", help="FPGA part, e.g. xczu7ev-ffvc1156-2-e")
    tgt.add_argument("--platform", help="Path to platform .xpfm")

    # Clocking: for part-flow use --clock-ns; for platform-flow prefer --freqhz (per AMD doc note)
    parser.add_argument("--clock-ns", type=str, default="5", help="Clock period in ns (used when --part is set)")
    parser.add_argument("--freqhz", type=str, default="", help="Clock freq in Hz (used when --platform is set)")

    # Output: default to XO kernel packaging
    parser.add_argument("--output-format", default="xo", choices=["xo", "rtl"], help="Packaging output format")

    args = parser.parse_args()

    ws = abspath(args.workspace)
    src_file = abspath(args.src)
    if not os.path.isfile(src_file):
        print(f"[ERROR] source file not found: {src_file}", file=sys.stderr)
        return 2

    # 1) Connect client + set workspace  :contentReference[oaicite:4]{index=4}
    client = vitis.create_client()
    os.makedirs(ws, exist_ok=True)
    client.set_workspace(path=ws)

    # 2) Create HLS component (empty template + hls_config.cfg)  :contentReference[oaicite:5]{index=5}
    comp = client.create_hls_component(
        name=args.name,
        cfg_file=["hls_config.cfg"],
        template="empty_hls_component",
        # platform/part can also be passed here per API signature; we will set via config to keep it explicit. :contentReference[oaicite:6]{index=6}
    )

    # Component directory is usually <workspace>/<component_name> in Unified IDE workspaces.
    comp_dir = os.path.join(ws, args.name)
    comp_src_dir = os.path.join(comp_dir, "src")
    os.makedirs(comp_src_dir, exist_ok=True)

    # 3) Copy example.c into component folder so syn.file can use a stable relative path
    dst_c = os.path.join(comp_src_dir, os.path.basename(src_file))
    shutil.copy2(src_file, dst_c)

    # 4) Edit hls_config.cfg using config API (get_config_file + set_value)  :contentReference[oaicite:7]{index=7}
    cfg_path = os.path.join(comp_dir, "hls_config.cfg")
    cfg = client.get_config_file(path=cfg_path)

    # General target selection:
    # UG1399 shows part=... outside [hls], and [hls] contains flow_target/syn.file/syn.top/clock, etc. :contentReference[oaicite:8]{index=8}
    if args.part:
        cfg.set_value(key="part", value=args.part)
        cfg.set_value(section="hls", key="clock", value=str(args.clock_ns))
    else:
        cfg.set_value(key="platform", value=abspath(args.platform))
        if args.freqhz:
            cfg.set_value(key="freqhz", value=str(args.freqhz))
        # If platform is used, AMD notes you should prefer freqhz instead of clock. :contentReference[oaicite:9]{index=9}

    # HLS kernel essentials (v++ hls config keys) :contentReference[oaicite:10]{index=10}
    cfg.set_value(section="hls", key="flow_target", value="vivado")
    cfg.set_value(section="hls", key="syn.file", value=os.path.join("src", os.path.basename(dst_c)))
    cfg.set_value(section="hls", key="syn.top", value=args.top)

    # Output format (Unified IDE config often uses package.output.format; example shown in UG1400). :contentReference[oaicite:11]{index=11}
    cfg.set_value(section="hls", key="package.output.format", value=args.output_format)
    # Commonly paired in older examples for XO packaging: package.output.syn=1 :contentReference[oaicite:12]{index=12}
    if args.output_format == "xo":
        cfg.set_value(section="hls", key="package.output.syn", value="1")

    # (Optional) Also set syn.output.format (v++ mode hls example uses this) :contentReference[oaicite:13]{index=13}
    if args.output_format == "xo":
        cfg.set_value(section="hls", key="syn.output.format", value="xo")

    # 5) Run synthesis + package. Operation names shown in AMD docs/examples. :contentReference[oaicite:14]{index=14}
    comp.run(operation="SYNTHESIS")
    comp.run(operation="PACKAGE")

    print("[OK] HLS component build finished.")
    # Output location hint (documented for HLS component output layout). :contentReference[oaicite:15]{index=15}
    print(f"Workspace: {ws}")
    print(f"Component: {comp_dir}")
    print("Check outputs under: <workspace>/<component>/<kernel_name>/ (or component build folders)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



