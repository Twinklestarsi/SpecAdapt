"""Load single or batch natural-language specifications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_spec_entries(spec_file: str | Path) -> List[Dict[str, Any]]:
    """Load specs from plain text, a JSON object, or a JSON list."""
    path = Path(spec_file)
    raw_text = path.read_text(errors="replace")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return [{"id": path.stem, "spec": raw_text}]

    if isinstance(parsed, dict):
        if isinstance(parsed.get("spec"), str):
            entry = {
                "id": str(parsed.get("id", path.stem)),
                "spec": parsed["spec"],
                "optimization_target": str(
                    parsed.get("optimization_target", "")
                ).upper(),
            }
            if parsed.get("family_id") or parsed.get("family"):
                entry["family_id"] = str(
                    parsed.get("family_id") or parsed.get("family")
                )
            return [entry]
        if isinstance(parsed.get("specs"), list):
            parsed = parsed["specs"]
        else:
            return [{"id": path.stem, "spec": raw_text}]

    if not isinstance(parsed, list):
        return [{"id": path.stem, "spec": raw_text}]

    entries: List[Dict[str, Any]] = []
    for index, item in enumerate(parsed, start=1):
        if isinstance(item, dict) and isinstance(item.get("spec"), str):
            entry = {
                "id": str(item.get("id", f"{path.stem}_{index:02d}")),
                "spec": item["spec"],
                "optimization_target": str(
                    item.get("optimization_target", "")
                ).upper(),
            }
            if item.get("family_id") or item.get("family"):
                entry["family_id"] = str(
                    item.get("family_id") or item.get("family")
                )
            entries.append(entry)
        elif isinstance(item, str):
            entries.append({
                "id": f"{path.stem}_{index:02d}",
                "spec": item,
            })

    return entries or [{"id": path.stem, "spec": raw_text}]
