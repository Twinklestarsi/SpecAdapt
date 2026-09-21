"""Inventory local spec/reference-RTL pairs without running AI, DC, or labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from project_paths import PROJECT_ROOT


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_reference(spec_path: Path) -> tuple[Path | None, str]:
    name = spec_path.name
    if not name.endswith("_spec.txt"):
        return None, "unsupported_spec_name"
    prefix = name[: -len("_spec.txt")]
    preferred = (
        (spec_path.with_name(f"{prefix}_original.v"), "original"),
        (spec_path.with_name(f"{prefix}_original.sv"), "original"),
        (spec_path.with_name(f"{prefix}_combinational.v"), "combinational_reference"),
        (spec_path.with_name(f"{prefix}_combinational.sv"), "combinational_reference"),
    )
    for path, kind in preferred:
        if path.is_file() and path.stat().st_size > 0:
            return path, kind
    return None, "missing_reference"


def _source_dataset(directory_name: str) -> str:
    if directory_name.startswith("verilog-eval_dataset_spec-to-rtl_"):
        return "verilog_eval_spec_to_rtl"
    if directory_name.startswith("RTLRewriter-Bench_"):
        return "rtlrewriter_bench"
    if directory_name.startswith("sample_partition_"):
        return "sample_partition"
    return "other_local"


def _design_type(spec_text: str, rtl_text: str) -> tuple[str, str]:
    combined = f"{spec_text}\n{rtl_text}".lower()
    sequential_markers = (
        "posedge",
        "negedge",
        "rising edge",
        "falling edge",
        "clock cycle",
        "sequential logic",
        "always_ff",
    )
    if any(marker in combined for marker in sequential_markers):
        return "sequential", "text_marker_heuristic"
    combinational_markers = ("combinational", "always @*", "always_comb")
    if any(marker in combined for marker in combinational_markers):
        return "combinational", "text_marker_heuristic"
    return "unknown", "needs_review"


def _provisional_family(directory_name: str) -> str:
    simplified = re.sub(
        r"^(verilog-eval_dataset_spec-to-rtl_Prob\d+_|RTLRewriter-Bench_|sample_partition_)",
        "",
        directory_name,
    )
    return re.sub(r"[^a-zA-Z0-9]+", "_", simplified).strip("_").lower()


def discover_candidates(
    source_root: Path, pilot_manifest: Path
) -> dict[str, Any]:
    pilot = json.loads(pilot_manifest.read_text(encoding="utf-8"))
    pilot_hashes = {
        str(item.get("spec_sha256"))
        for item in pilot.get("cases") or []
        if item.get("spec_sha256")
    }
    candidates: list[dict[str, Any]] = []
    excluded_duplicates: list[dict[str, str]] = []
    missing_references: list[str] = []
    seen_spec_hashes: dict[str, str] = {}
    duplicate_local_specs: list[dict[str, str]] = []
    ignored_spec_like_reports = [
        str(path)
        for path in sorted(source_root.rglob("*spec*.txt"))
        if not path.name.endswith("_spec.txt")
    ]

    for spec_path in sorted(source_root.rglob("*_spec.txt")):
        reference_path, reference_kind = _find_reference(spec_path)
        if reference_path is None:
            missing_references.append(str(spec_path))
            continue
        spec_sha = _file_sha256(spec_path)
        if spec_sha in pilot_hashes:
            excluded_duplicates.append(
                {"spec_path": str(spec_path), "spec_sha256": spec_sha}
            )
            continue
        if spec_sha in seen_spec_hashes:
            duplicate_local_specs.append(
                {
                    "spec_path": str(spec_path),
                    "duplicate_of": seen_spec_hashes[spec_sha],
                    "spec_sha256": spec_sha,
                }
            )
            continue
        seen_spec_hashes[spec_sha] = str(spec_path)
        spec_text = spec_path.read_text(encoding="utf-8", errors="replace")
        rtl_text = reference_path.read_text(encoding="utf-8", errors="replace")
        design_type, design_type_source = _design_type(spec_text, rtl_text)
        directory_name = spec_path.parent.name
        category_raw = spec_path.relative_to(source_root).parts[0]
        category = category_raw.replace("-", "_")
        candidates.append(
            {
                "id": directory_name,
                "source_dataset": _source_dataset(directory_name),
                "category": category,
                "category_raw": category_raw,
                "design_type": design_type,
                "design_type_source": design_type_source,
                "family_id": _provisional_family(directory_name),
                "family_id_status": "provisional_requires_review",
                "spec_path": str(spec_path.resolve()),
                "reference_rtl_path": str(reference_path.resolve()),
                "reference_kind": reference_kind,
                "spec_sha256": spec_sha,
                "reference_sha256": _file_sha256(reference_path),
                "spec_char_count": len(spec_text),
                "label_status": "not_collected",
                "training_ready": False,
            }
        )

    category_counts = Counter(item["category"] for item in candidates)
    source_counts = Counter(item["source_dataset"] for item in candidates)
    design_type_counts = Counter(item["design_type"] for item in candidates)
    return {
        "schema_version": "adaptive_router_candidate_inventory_v1",
        "created_date": date.today().isoformat(),
        "source_root": str(source_root.resolve()),
        "pilot_manifest": str(pilot_manifest.resolve()),
        "purpose": "Unlabeled candidates for future serial dual-path collection",
        "training_ready": False,
        "training_block_reason": (
            "Candidates have reference RTL but no correctness-first paired route labels"
        ),
        "candidate_count": len(candidates),
        "category_counts": dict(sorted(category_counts.items())),
        "source_dataset_counts": dict(sorted(source_counts.items())),
        "design_type_counts": dict(sorted(design_type_counts.items())),
        "excluded_pilot_duplicate_count": len(excluded_duplicates),
        "excluded_pilot_duplicates": excluded_duplicates,
        "duplicate_local_spec_count": len(duplicate_local_specs),
        "duplicate_local_specs": duplicate_local_specs,
        "missing_reference_count": len(missing_references),
        "missing_reference_specs": missing_references,
        "ignored_spec_like_report_count": len(ignored_spec_like_reports),
        "ignored_spec_like_reports": ignored_spec_like_reports,
        "candidates": candidates,
    }


def _write_outputs(output_dir: Path, inventory: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = date.today().isoformat()
    json_path = output_dir / f"{prefix}_LOCAL_TRAINING_CANDIDATES.json"
    md_path = output_dir / f"{prefix}_LOCAL_TRAINING_CANDIDATES.md"
    json_path.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# 本地自适应路由训练候选盘点",
        "",
        f"- 候选spec/reference RTL对：{inventory['candidate_count']}",
        f"- 与8个pilot重复并排除：{inventory['excluded_pilot_duplicate_count']}",
        f"- 本地重复spec并排除：{inventory['duplicate_local_spec_count']}",
        f"- 缺少可识别reference RTL：{inventory['missing_reference_count']}",
        f"- 名称含spec但实际为报告并忽略：{inventory['ignored_spec_like_report_count']}",
        "- 当前训练可用：否；尚未获得双路correctness-first标签。",
        "",
        "## 来源分布",
        "",
    ]
    for name, count in inventory["source_dataset_counts"].items():
        lines.append(f"- `{name}`：{count}")
    lines.extend(["", "## 类别分布", ""])
    for name, count in inventory["category_counts"].items():
        lines.append(f"- `{name}`：{count}")
    lines.extend(["", "## 设计类型（启发式初筛）", ""])
    for name, count in inventory["design_type_counts"].items():
        lines.append(f"- `{name}`：{count}")
    lines.extend(
        [
            "",
            "## 下一步筛选规则",
            "",
            "1. 人工或语法工具确认top module、端口、时钟、复位和周期语义。",
            "2. 合并同源/同族变体，固定family ID后再划分train/validation/test。",
            "3. 先做小批量串行双路预检；只保留两路均执行到可判定状态的记录。",
            "4. `tie/uncertain/unsolved/flow_error`不转换为二分类标签。",
            "5. 达到至少50个有效二分类标签后才允许试训，正式实验建议200个以上。",
            "",
            "完整逐项路径、哈希和状态见同名JSON。",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", default=str(PROJECT_ROOT.parent / "verilog")
    )
    parser.add_argument(
        "--pilot-manifest",
        default=str(PROJECT_ROOT / "pilot_test" / "pilot_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "datasets" / "candidates"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inventory = discover_candidates(
        Path(args.source_root).resolve(), Path(args.pilot_manifest).resolve()
    )
    json_path, md_path = _write_outputs(Path(args.output_dir).resolve(), inventory)
    print(json.dumps({key: inventory[key] for key in (
        "candidate_count",
        "excluded_pilot_duplicate_count",
        "duplicate_local_spec_count",
        "missing_reference_count",
        "ignored_spec_like_report_count",
        "category_counts",
        "source_dataset_counts",
        "design_type_counts",
    )}, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
