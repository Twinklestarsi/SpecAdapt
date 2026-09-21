"""
store.py — Read/write the Memory Agent's staging file(s).

Primary file: path_decisions_log.json
  Shared between Module 2 and Module 8.
  Module 2 appends to "decisions".
  Module 8 reads "decisions" and writes "path_selection_rules".

All writes are done via a write-then-rename pattern to avoid partial
reads on the CIFS filesystem.  Falls back to direct write if rename
is not possible (cross-device move on some CIFS mounts).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory_agent.experience import (
    STORE_KEY_PATH_RULES,
    STORE_KEY_DECISIONS,
    STORE_KEY_OPT_DECISIONS,
    STORE_KEY_CORRECTIONS,
    STORE_KEY_TRAJECTORIES,
    STORE_KEY_PPA_OUTCOMES,
    GenerationPathRecord,
    OptimizationDecisionRecord,
    CorrectionPatternRecord,
    CollaborationTrajectoryRecord,
    PPAOutcome,
)


# ── Default store schema ───────────────────────────────────────────────

def _empty_store() -> Dict[str, Any]:
    return {
        STORE_KEY_PATH_RULES:    [],   # learned rules → read by Module 2
        STORE_KEY_DECISIONS:     [],   # path decisions → written by Module 2
        STORE_KEY_PPA_OUTCOMES:  [],   # PPA results → written by Module 7
        STORE_KEY_OPT_DECISIONS: [],   # transform records → written by M3/5/6
        STORE_KEY_CORRECTIONS:   [],   # correction patterns → written by M4/5/6/7
        STORE_KEY_TRAJECTORIES:  [],   # trajectories → written by orchestrator
    }


# ── Safe JSON write ────────────────────────────────────────────────────

def _safe_write(path: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON to path atomically (temp file + rename).
    Falls back to direct write on cross-device rename errors (CIFS).
    """
    text = json.dumps(data, indent=2, ensure_ascii=False)
    tmp_path = None
    try:
        fd, tmp_str = tempfile.mkstemp(
            dir=path.parent, prefix=".tmp_memory_", suffix=".json"
        )
        tmp_path = Path(tmp_str)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        tmp_path.replace(path)
    except OSError:
        # CIFS cross-device: fall back to direct write
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


# ── MemoryStore ────────────────────────────────────────────────────────

class MemoryStore:
    """
    Low-level read/write interface to path_decisions_log.json.

    All public methods are safe to call with a missing store file
    (it will be created on first write).  Read errors return empty
    structures; write errors print a warning and continue.
    """

    def __init__(self, store_path: str | Path) -> None:
        self.path = Path(store_path)

    # ── Internal load/save ─────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return _empty_store()
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            # Ensure all keys exist (forward-compat for old files)
            for k, v in _empty_store().items():
                data.setdefault(k, v)
            return data
        except Exception as exc:
            print(f"[MemoryStore] Warning: could not read {self.path}: {exc}",
                  file=sys.stderr)
            return _empty_store()

    def _save(self, data: Dict[str, Any]) -> None:
        try:
            _safe_write(self.path, data)
        except Exception as exc:
            print(f"[MemoryStore] Warning: could not write {self.path}: {exc}",
                  file=sys.stderr)

    # ── Path-selection rules (written by Module 8, read by Module 2) ───

    def get_path_rules(self) -> List[Dict[str, Any]]:
        return self._load().get(STORE_KEY_PATH_RULES, [])

    def set_path_rules(self, rules: List[Dict[str, Any]]) -> None:
        data = self._load()
        data[STORE_KEY_PATH_RULES] = rules
        self._save(data)

    # ── Path decisions (written by Module 2) ───────────────────────────

    def get_decisions(self) -> List[Dict[str, Any]]:
        return self._load().get(STORE_KEY_DECISIONS, [])

    def append_decision(self, record: Dict[str, Any]) -> None:
        data = self._load()
        data.setdefault(STORE_KEY_DECISIONS, []).append(record)
        self._save(data)

    # ── PPA outcomes (written by Module 7) ────────────────────────────

    def get_ppa_outcomes(self) -> List[Dict[str, Any]]:
        return self._load().get(STORE_KEY_PPA_OUTCOMES, [])

    def record_ppa_outcome(
        self,
        benchmark: str,
        path: str,
        area_improvement_pct: Optional[float] = None,
        timing_improvement_pct: Optional[float] = None,
        slack_status: Optional[str] = None,
        cell_count_delta: Optional[int] = None,
        synthesis_failed: bool = False,
        notes: str = "",
        timing_improvement_ps: Optional[float] = None,
    ) -> None:
        """
        Write a PPA outcome record (called by Module 7).
        Also updates the matching GenerationPathRecord in "decisions"
        so the Memory Agent can correlate path → PPA in one place.
        """
        outcome = PPAOutcome(
            area_improvement_pct=area_improvement_pct,
            timing_improvement_ps=timing_improvement_ps,
            timing_improvement_pct=timing_improvement_pct,
            slack_status=slack_status,
            cell_count_delta=cell_count_delta,
            synthesis_failed=synthesis_failed,
            notes=notes,
        )
        record = {
            "benchmark": benchmark,
            "path": path,
            "ppa_outcome": outcome.to_dict(),
        }

        data = self._load()
        data.setdefault(STORE_KEY_PPA_OUTCOMES, []).append(record)

        # Back-fill ppa_outcome into the matching decisions entry
        for dec in data.get(STORE_KEY_DECISIONS, []):
            if dec.get("benchmark") == benchmark and dec.get("path") == path:
                dec["ppa_outcome"] = outcome.to_dict()

        self._save(data)

    # ── Optimization decisions (written by M3/5/6) ────────────────────

    def append_optimization_decision(self, record: Dict[str, Any]) -> None:
        data = self._load()
        data.setdefault(STORE_KEY_OPT_DECISIONS, []).append(record)
        self._save(data)

    def get_optimization_decisions(self) -> List[Dict[str, Any]]:
        return self._load().get(STORE_KEY_OPT_DECISIONS, [])

    # ── Correction patterns (written by M4/5/6/7) ────────────────────

    def append_correction(self, record: Dict[str, Any]) -> None:
        data = self._load()
        data.setdefault(STORE_KEY_CORRECTIONS, []).append(record)
        self._save(data)

    def get_corrections(self) -> List[Dict[str, Any]]:
        return self._load().get(STORE_KEY_CORRECTIONS, [])

    # ── Collaboration trajectories (written by orchestrator) ──────────

    def append_trajectory(self, record: Dict[str, Any]) -> None:
        data = self._load()
        data.setdefault(STORE_KEY_TRAJECTORIES, []).append(record)
        self._save(data)

    def get_trajectories(self) -> List[Dict[str, Any]]:
        return self._load().get(STORE_KEY_TRAJECTORIES, [])

    # ── Diagnostics ───────────────────────────────────────────────────

    def status(self) -> Dict[str, int]:
        """Return record counts per category."""
        data = self._load()
        return {
            "path_rules":     len(data.get(STORE_KEY_PATH_RULES, [])),
            "decisions":      len(data.get(STORE_KEY_DECISIONS, [])),
            "ppa_outcomes":   len(data.get(STORE_KEY_PPA_OUTCOMES, [])),
            "opt_decisions":  len(data.get(STORE_KEY_OPT_DECISIONS, [])),
            "corrections":    len(data.get(STORE_KEY_CORRECTIONS, [])),
            "trajectories":   len(data.get(STORE_KEY_TRAJECTORIES, [])),
        }
