#!/usr/bin/env python3
"""Rebuild the rag_history evidence of a Memory database from a current region index.

Why this exists
---------------
Historical region indices below schema_version 4 copied each benchmark's
benchmark-level PPA numbers onto *every one of its regions*. Importing such an
index turned one measurement into up to 1310 identical "pieces of evidence"
(u_block_5), so the memory agent's mean_gain was decided by whichever benchmark
happened to have the most regions rather than by which transform actually works.

Measured on memory_agent.db before this repair:
    rag_history AREA actions        3636 rows
    distinct measurements            678 rows   -> 81.4% were duplicate votes
    mean area_gain_pct           +5.238%
    mean after dropping the duplicated >50% block   -3.230%   (sign flip)

This script drops every producer='rag_history' row and re-imports from the index
you point it at. import_rag_index now refuses pre-schema-4 indices and collapses
identical (benchmark, objective, transform, region_type, gains) measurements into
one row, recording how many regions were collapsed under
context.collapsed_duplicate_regions.

Usage
-----
    PYTHONPATH=. python memory_agent/scripts/repair_rag_history_evidence.py \
        --db memory_agent.db \
        --index rag_retrieve/indices/module4_historical_region_index_v6.json

Add --dry-run to report what would change without writing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from memory_agent.ingest.adapters import import_rag_index
from memory_agent.sqlite_store import SQLiteMemoryStore


def _stats(db: Path) -> dict:
    conn = sqlite3.connect(db)
    try:
        def one(sql: str, *params):
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else None

        out = {
            "rag_history_actions": one(
                "SELECT COUNT(*) FROM actions WHERE producer='rag_history'"
            ),
            "rag_history_evaluations": one(
                "SELECT COUNT(*) FROM evaluations WHERE producer='rag_history'"
            ),
            "rag_history_runs": one(
                "SELECT COUNT(*) FROM runs WHERE producer='rag_history'"
            ),
            "area_actions": one(
                "SELECT COUNT(*) FROM actions "
                "WHERE producer='rag_history' AND objective='AREA'"
            ),
            "area_distinct_measurements": one(
                "SELECT COUNT(*) FROM (SELECT DISTINCT benchmark, transform_name, "
                "region_type, json_extract(outcome_json,'$.area_gain_pct'), "
                "json_extract(outcome_json,'$.timing_gain_ps') FROM actions "
                "WHERE producer='rag_history' AND objective='AREA')"
            ),
            "area_gain_mean": one(
                "SELECT AVG(area_gain_pct) FROM evaluations "
                "WHERE producer='rag_history' AND objective='AREA' "
                "AND area_gain_pct IS NOT NULL"
            ),
            "area_gain_over_50": one(
                "SELECT COUNT(*) FROM evaluations WHERE producer='rag_history' "
                "AND objective='AREA' AND area_gain_pct > 50"
            ),
            "other_producer_actions": one(
                "SELECT COUNT(*) FROM actions WHERE producer<>'rag_history'"
            ),
        }
        return out
    finally:
        conn.close()


def _purge_rag_history(db: Path) -> dict:
    conn = sqlite3.connect(db, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        removed = {}
        for table in ("evaluations", "actions", "runs"):
            cur = conn.execute(f"DELETE FROM {table} WHERE producer='rag_history'")
            removed[table] = cur.rowcount
        conn.commit()
        return removed
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Memory database to repair")
    parser.add_argument(
        "--index",
        required=True,
        help="Historical region index (schema_version >= 4) to re-import from",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report current state and validate the index, but do not write",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the timestamped .bak copy (not recommended)",
    )
    args = parser.parse_args()

    db = Path(args.db).resolve()
    index = Path(args.index).resolve()
    if not db.is_file():
        print(f"error: database not found: {db}", file=sys.stderr)
        return 2
    if not index.is_file():
        print(f"error: index not found: {index}", file=sys.stderr)
        return 2

    payload = json.loads(index.read_text(encoding="utf-8"))
    schema_version = int(payload.get("summary", {}).get("schema_version") or 0)
    print(f"index          : {index}")
    print(f"schema_version : {schema_version or 'missing'}")
    if schema_version < 4:
        print(
            "error: this index predates schema 4 and carries benchmark-level gains "
            "on every region. Rebuild it with rag_retrieve/region_index.py.",
            file=sys.stderr,
        )
        return 3

    before = _stats(db)
    print("\nbefore:")
    for key, value in before.items():
        print(f"  {key:28s} {value}")

    if args.dry_run:
        print("\ndry-run: nothing written")
        return 0

    if not args.no_backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = db.with_name(f"{db.name}.{stamp}.bak")
        shutil.copy2(db, backup)
        print(f"\nbackup         : {backup}")

    removed = _purge_rag_history(db)
    print(f"purged         : {removed}")

    store = SQLiteMemoryStore(db)
    store.initialize()
    counts = import_rag_index(store, index)
    print(f"reimported     : {counts}")

    after = _stats(db)
    print("\nafter:")
    for key, value in after.items():
        print(f"  {key:28s} {value}")

    if after["other_producer_actions"] != before["other_producer_actions"]:
        print(
            "error: non-rag_history actions changed count; this should never happen",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
