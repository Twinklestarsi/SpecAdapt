# hls_rtl_batch_enhanced.py
# Vitis 2025.2: Create HLS component -> SYNTHESIS -> PACKAGE -> collect RTL -> copy out
#
# Run:
#   vitis -s hls_rtl_batch_enhanced.py -- --src /abs/path/example.c --part xczu9eg-ffvb1156-2-e --clock-ns 10 --out /abs/path/rtl_out
# or:
#   vitis -source hls_rtl_batch_enhanced.py --src ... (depending on your installation behavior)

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime

import vitis


def _log(message: str) -> None:
    print(f"[hls_rtl_batch] {message}", flush=True)


# ----------------------------
# C parsing helpers (heuristic)
# ----------------------------
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_COMMENT_LINE = re.compile(r"//.*?$", re.M)

# A conservative function-definition regex:
# - tries to match "ret_type name(args) {"
# - excludes prototypes (those end with ';')
_FUNC_DEF = re.compile(
    r"""
    (?P<prefix>^[ \t]*)                                           # indent
    (?P<ret>(?:[A-Za-z_]\w*|struct\s+[A-Za-z_]\w*|enum\s+[A-Za-z_]\w*)
        (?:[\w\s\*\(\)]|\[[^\]]*\]){0,200}?)                      # return type-ish
    [ \t]+
    (?P<name>[A-Za-z_]\w*)                                        # function name
    [ \t]*\(
        (?P<args>(?:[^()"]+|"[^"]*"|\([^()]*\))*)                 # args (best-effort)
    \)
    [ \t\r\n]*\{                                                  # opening brace
    """,
    re.X | re.M,
)

_C_KEYWORDS = {
    "if", "for", "while", "switch", "return", "sizeof", "do", "case",
    "break", "continue", "goto", "else",
}

def _strip_comments(code: str) -> str:
    code = _COMMENT_BLOCK.sub("", code)
    code = _COMMENT_LINE.sub("", code)
    return code

def extract_function_defs(c_path: str):
    raw = open(c_path, "r", encoding="utf-8", errors="ignore").read()
    code = _strip_comments(raw)

    defs = []
    for m in _FUNC_DEF.finditer(code):
        name = m.group("name")
        if name in _C_KEYWORDS:
            continue

        # crude guard: avoid matching "typedef ... name(...) {" patterns accidentally
        ret = " ".join(m.group("ret").split())
        if "typedef" in ret.split():
            continue

        # line number
        start = m.start()
        lineno = code.count("\n", 0, start) + 1

        args = m.group("args").strip()
        # quick param count heuristic
        arg_count = 0
        if args and args != "void":
            arg_count = len([a for a in args.split(",") if a.strip()])

        defs.append({
            "name": name,
            "ret": ret,
            "args": args,
            "arg_count": arg_count,
            "lineno": lineno,
        })

    return raw, defs

def choose_top_function(raw_code: str, defs: list, user_top: str | None):
    if user_top:
        # user override
        for d in defs:
            if d["name"] == user_top:
                return user_top, defs
        raise SystemExit(f"[ERROR] --top {user_top} not found in parsed function definitions.")

    # scoring
    lines = raw_code.splitlines()
    pri_names = {"top", "kernel", "krnl", "hls_top", "dut"}

    def score(d):
        s = 0
        name = d["name"]

        if name == "main":
            s -= 10_000

        if name in pri_names:
            s += 1000

        # Look for HLS pragmas near the definition (previous ~8 lines)
        i = max(0, d["lineno"] - 1)
        window = "\n".join(lines[max(0, i-8):i])
        if "pragma" in window and "HLS" in window:
            s += 200

        # Prefer non-static (slightly), but don't exclude
        if "static" in d["ret"].split():
            s -= 5

        # Prefer void return slightly (common for kernels), but not required
        if re.search(r"\bvoid\b", d["ret"]):
            s += 10

        # More arguments sometimes indicates “top”
        s += min(d["arg_count"], 8)

        # Prefer names that contain 'top' or 'krnl'
        if "top" in name.lower():
            s += 30
        if "krnl" in name.lower() or "kernel" in name.lower():
            s += 30

        return s

    if not defs:
        raise SystemExit("[ERROR] No function definitions were detected in the C file.")

    ranked = sorted(defs, key=lambda d: (score(d), d["lineno"]), reverse=True)
    return ranked[0]["name"], ranked


# ----------------------------
# RTL discovery/copy helpers
# ----------------------------
def find_rtl_folders(search_root: str):
    """
    package.output.format=rtl creates Verilog and VHDL folders in the HLS component working directory.
    We discover them by scanning for directories named 'verilog'/'vhdl' that actually contain RTL files.
    """
    found = {"verilog": [], "vhdl": []}
    for root, dirs, files in os.walk(search_root):
        base = os.path.basename(root).lower()

        if base == "verilog":
            if any(f.endswith((".v", ".sv")) for f in files):
                found["verilog"].append(root)

        if base == "vhdl":
            if any(f.endswith((".vhd", ".vhdl")) for f in files):
                found["vhdl"].append(root)

    return found

def copy_tree(src: str, dst: str):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to example.c (absolute path recommended)")
    ap.add_argument("--out", required=True, help="Output directory to copy RTL into")
    ap.add_argument("--part", default=None, help="FPGA part, e.g. xczu9eg-ffvb1156-2-e (use either --part or --platform)")
    ap.add_argument("--platform", default=None, help="Optional .xpfm platform path (if used, specify --freqhz instead of --clock-ns)")
    ap.add_argument("--clock-ns", type=float, default=10.0, help="Clock period in ns (used with --part)")
    ap.add_argument("--freqhz", default=None, help="Clock frequency in Hz (used with --platform)")
    ap.add_argument("--top", default=None, help="Override top function name (optional)")
    ap.add_argument("--work-root", default=os.getcwd(), help="Root directory under which workspace is created")
    ap.add_argument("--keep-workspace", action="store_true", help="Keep workspace (default: keep)")
    ap.add_argument("--flow-target", default="vivado", choices=["vivado", "vitis"],
                    help="HLS flow target: 'vivado' produces wire-level I/O (default), "
                         "'vitis' wraps all I/O in AXI")
    args, unknown = ap.parse_known_args()
    _log("parsed arguments")

    src = os.path.abspath(args.src)
    out_dir = os.path.abspath(args.out)
    work_root = os.path.abspath(args.work_root)
    safe_mkdir(out_dir)
    safe_mkdir(work_root)
    _log(f"src={src}")
    _log(f"out_dir={out_dir}")
    _log(f"work_root={work_root}")

    if not os.path.isfile(src):
        raise SystemExit(f"[ERROR] --src not found: {src}")

    if bool(args.platform) == bool(args.part) is False:
        # exactly one of them should be provided
        pass
    # Allow both, but prefer platform if given
    if args.platform and not args.freqhz:
        raise SystemExit("[ERROR] When --platform is used, you must specify --freqhz (Hz).")
    if (not args.platform) and (not args.part):
        raise SystemExit("[ERROR] Please specify either --part or --platform.")

    # Parse candidates from C
    raw, defs = extract_function_defs(src)
    top, ranked = choose_top_function(raw, defs, args.top)
    _log(f"selected top function: {top}")

    # Build unique names
    base = os.path.splitext(os.path.basename(src))[0]
    sha = hashlib.sha1(open(src, "rb").read()).hexdigest()[:8]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ws_name = f"vitis_2025_2_ws__{base}__{stamp}__{sha}"
    comp_name = f"hls__{base}__{sha}"

    ws_path = os.path.join(work_root, ws_name)
    _log(f"workspace path: {ws_path}")

    # Create Vitis client + workspace
    _log("creating vitis client")
    client = vitis.create_client()
    _log("setting workspace")
    client.set_workspace(path=ws_path)
    _log("workspace set")

    # Create HLS component
    # API supports creating/building HLS components via Python.
    # We'll keep cfg_file named hls_config.cfg
    create_kwargs = dict(
        name=comp_name,
        cfg_file=["hls_config.cfg"],
        template="empty_hls_component",
    )
    if args.platform:
        create_kwargs["platform"] = os.path.abspath(args.platform)
    else:
        create_kwargs["part"] = args.part

    _log("creating HLS component")
    hls_comp = client.create_hls_component(**create_kwargs)
    _log("HLS component created")

    # Obtain a config handle; try multiple known entry points (tool-version differences)
    cfg = None
    last_err = None
    for getter in (
        lambda: hls_comp.add_cfg_file(cfg_file="hls_config.cfg"),
        lambda: hls_comp.add_cfg_file(cfg_file="/hls_config.cfg"),
        lambda: client.add_cfg_file(cfg_file="hls_config.cfg"),
        lambda: client.add_cfg_file(cfg_file="/hls_config.cfg"),
        lambda: client.add_config_file(cfg_file="/hls_config.cfg"),
        lambda: client.get_config_file(path=f"/{ws_name}/{comp_name}/hls_config.cfg"),
    ):
        try:
            cfg = getter()
            break
        except Exception as e:
            last_err = e

    if cfg is None:
        raise SystemExit(f"[ERROR] Unable to get config-file handle. Last error: {last_err}")
    _log("config handle acquired")

    # Write config values (part is a general option; HLS options go under [hls])
    # Common keys: flow_target, syn.file, syn.top, clock/freqhz, package.output.format
    if args.platform:
        cfg.set_value(key="platform", value=os.path.abspath(args.platform))
        cfg.set_value(key="freqhz", value=str(args.freqhz))
    else:
        cfg.set_value(key="part", value=args.part)
        cfg.set_value(section="hls", key="clock", value=str(args.clock_ns))

    cfg.set_value(section="hls", key="flow_target", value=args.flow_target)
    cfg.set_value(section="hls", key="syn.file", value=src)
    cfg.set_value(section="hls", key="syn.top", value=top)

    # Ensure RTL export behavior (package.output.format=rtl creates Verilog/VHDL folders)
    cfg.set_value(section="hls", key="syn.output.format", value="rtl")
    cfg.set_value(section="hls", key="package.output.format", value="rtl")
    cfg.set_value(section="hls", key="package.output.syn", value="false")
    _log("config values written")

    # Run HLS flow
    _log("starting SYNTHESIS")
    hls_comp.run(operation="SYNTHESIS")
    _log("SYNTHESIS completed")
    _log("starting PACKAGE")
    hls_comp.run(operation="PACKAGE")
    _log("PACKAGE completed")

    # Discover RTL
    rtl_found = find_rtl_folders(ws_path)
    _log("RTL discovery completed")

    # Copy RTL to output
    dest_root = os.path.join(out_dir, f"{comp_name}__{top}")
    safe_mkdir(dest_root)

    copied = {"verilog": [], "vhdl": []}
    for kind in ("verilog", "vhdl"):
        for i, src_dir in enumerate(rtl_found[kind]):
            dst_dir = os.path.join(dest_root, kind if i == 0 else f"{kind}_{i}")
            copy_tree(src_dir, dst_dir)
            copied[kind].append(dst_dir)

    # Write manifest for traceability
    manifest = {
        "src": src,
        "workspace": ws_path,
        "component": comp_name,
        "flow_target": args.flow_target,
        "top_selected": top,
        "top_candidates_ranked": ranked,
        "rtl_found_in_workspace": rtl_found,
        "rtl_copied_to": copied,
        "timestamp": stamp,
    }
    with open(os.path.join(dest_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Save candidate list in a friendly text file
    with open(os.path.join(dest_root, "top_candidates.txt"), "w", encoding="utf-8") as f:
        f.write("Ranked top-function candidates (best first):\n")
        for d in ranked:
            f.write(f"- {d['name']} (line {d['lineno']}), ret='{d['ret']}', args='{d['args']}'\n")

    print("\n=== HLS RTL Batch Done ===")
    print(f"Source C        : {src}")
    print(f"Workspace       : {ws_path}")
    print(f"Component       : {comp_name}")
    print(f"Top selected    : {top}")
    print(f"RTL output dir  : {dest_root}")
    if not copied["verilog"] and not copied["vhdl"]:
        print("[WARN] No verilog/vhdl folders with RTL files were discovered. Check synthesis/package logs and workspace contents.")
    else:
        print(f"Copied Verilog  : {copied['verilog']}")
        print(f"Copied VHDL     : {copied['vhdl']}")

    # Keep workspace by default (you can add auto-delete if you really want)
    try:
        client.dispose()
    except Exception:
        pass


if __name__ == "__main__":
    main()
