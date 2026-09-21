"""Shared helpers for import adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator

from memory_agent.ids import stable_id
from memory_agent.schemas import RunRecord, TaskRecord
from memory_agent.sqlite_store import SQLiteMemoryStore
from memory_agent.validators import normalize_objective, normalize_path


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def iter_json_files(path: str | Path, name: str) -> Iterator[Path]:
    source = Path(path)
    if source.is_file():
        if source.name == name:
            yield source
        return
    yield from sorted(source.rglob(name))


def ensure_task(
    store: SQLiteMemoryStore,
    benchmark: str,
    *,
    spec_text: str = "",
    source_path: str = "",
    metadata: Dict[str, Any] | None = None,
) -> str:
    task_id = stable_id("task", benchmark)
    store.upsert_task(TaskRecord(
        task_id=task_id,
        benchmark=benchmark,
        spec_text=spec_text,
        source_path=source_path,
        metadata=metadata or {},
    ))
    return task_id


def ensure_run(
    store: SQLiteMemoryStore,
    task_id: str,
    benchmark: str,
    source_key: str,
    *,
    objective: str = "",
    path: str = "",
    status: str = "imported",
    producer: str = "",
    metadata: Dict[str, Any] | None = None,
) -> str:
    run_id = stable_id("run", benchmark, source_key, objective, path)
    store.upsert_run(RunRecord(
        run_id=run_id,
        task_id=task_id,
        objective=normalize_objective(objective),
        path=normalize_path(path),
        status=status,
        producer=producer,
        metadata=metadata or {},
    ))
    return run_id

