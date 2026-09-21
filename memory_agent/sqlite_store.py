"""Transactional SQLite persistence for structured cross-agent memory."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from memory_agent.schemas import (
    ActionRecord,
    ArtifactRecord,
    EvaluationRecord,
    FailureRecord,
    FeatureRecord,
    RunRecord,
    SelectionRecord,
    TaskRecord,
    TrajectoryRecord,
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _from_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


class SQLiteMemoryStore:
    """SQLite-backed memory store with idempotent record upserts."""

    def __init__(self, path: str | Path = "memory.db") -> None:
        self.path = Path(path)
        self._active_connection: sqlite3.Connection | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self._active_connection is not None:
            yield self._active_connection
            return
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        self._active_connection = connection
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            self._active_connection = None
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.connect():
            yield

    def initialize(self) -> None:
        with self.connect() as db:
            # The project lives on SMB/CIFS. DELETE journal mode uses ordinary
            # file locks and is reliable there; WAL requires shared-memory
            # locking semantics that SMB does not provide consistently.
            db.execute("PRAGMA journal_mode = DELETE")
            db.execute("PRAGMA synchronous = FULL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO metadata(key, value) VALUES ('schema_version', '1');

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    benchmark TEXT NOT NULL,
                    spec_text TEXT NOT NULL DEFAULT '',
                    source_path TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_benchmark ON tasks(benchmark);

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    objective TEXT NOT NULL DEFAULT '',
                    path TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    producer TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    schema_version INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id);

                CREATE TABLE IF NOT EXISTS features (
                    record_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    run_id TEXT NOT NULL DEFAULT '',
                    parent_id TEXT NOT NULL DEFAULT '',
                    producer TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    features_json TEXT NOT NULL,
                    confidence_json TEXT NOT NULL DEFAULT '{}',
                    overall_confidence TEXT NOT NULL DEFAULT '',
                    optimization_target TEXT NOT NULL DEFAULT '',
                    source_path TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_features_task ON features(task_id);

                CREATE TABLE IF NOT EXISTS selections (
                    record_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    run_id TEXT NOT NULL DEFAULT '',
                    parent_id TEXT NOT NULL DEFAULT '',
                    producer TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    rule_fired TEXT NOT NULL DEFAULT '',
                    confidence TEXT NOT NULL DEFAULT '',
                    tier INTEGER NOT NULL DEFAULT 0,
                    reason TEXT NOT NULL DEFAULT '',
                    token_usage_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_selections_run ON selections(run_id);

                CREATE TABLE IF NOT EXISTS actions (
                    record_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    run_id TEXT NOT NULL DEFAULT '',
                    parent_id TEXT NOT NULL DEFAULT '',
                    producer TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    benchmark TEXT NOT NULL DEFAULT '',
                    objective TEXT NOT NULL DEFAULT '',
                    path TEXT NOT NULL DEFAULT '',
                    region_id TEXT NOT NULL DEFAULT '',
                    region_type TEXT NOT NULL DEFAULT '',
                    transform_name TEXT NOT NULL DEFAULT '',
                    selected INTEGER NOT NULL DEFAULT 0,
                    applied_successfully INTEGER,
                    planning_score REAL,
                    -- expected_metric_gain mixes units: percent on AREA runs,
                    -- picoseconds on TIMING runs.  It is kept populated because
                    -- existing readers and exported CSVs depend on it; new
                    -- aggregates should read the two typed columns instead.
                    expected_metric_gain REAL,
                    expected_area_gain_pct REAL,
                    expected_timing_gain_ps REAL,
                    confidence REAL,
                    context_json TEXT NOT NULL DEFAULT '{}',
                    outcome_json TEXT NOT NULL DEFAULT '{}',
                    source_path TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_actions_lookup
                    ON actions(objective, region_type, transform_name);
                CREATE INDEX IF NOT EXISTS idx_actions_run ON actions(run_id);

                CREATE TABLE IF NOT EXISTS evaluations (
                    record_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    run_id TEXT NOT NULL DEFAULT '',
                    parent_id TEXT NOT NULL DEFAULT '',
                    producer TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    benchmark TEXT NOT NULL DEFAULT '',
                    objective TEXT NOT NULL DEFAULT '',
                    path TEXT NOT NULL DEFAULT '',
                    correctness_status TEXT NOT NULL DEFAULT 'unknown',
                    synthesis_status TEXT NOT NULL DEFAULT 'unknown',
                    area REAL,
                    baseline_area REAL,
                    area_gain_pct REAL,
                    data_arrival_time_ps REAL,
                    baseline_data_arrival_time_ps REAL,
                    -- Positive critical-path delay reduction (ps); the legacy
                    -- column name is retained for on-disk compatibility.
                    timing_gain_ps REAL,
                    slack_ps REAL,
                    baseline_slack_ps REAL,
                    slack_status TEXT NOT NULL DEFAULT '',
                    token_usage_json TEXT NOT NULL DEFAULT '{}',
                    failure_reason TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    source_path TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id);
                CREATE INDEX IF NOT EXISTS idx_evaluations_objective ON evaluations(objective);

                CREATE TABLE IF NOT EXISTS failures (
                    record_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    run_id TEXT NOT NULL DEFAULT '',
                    parent_id TEXT NOT NULL DEFAULT '',
                    producer TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    benchmark TEXT NOT NULL DEFAULT '',
                    module TEXT NOT NULL DEFAULT '',
                    failure_type TEXT NOT NULL DEFAULT '',
                    failure_stage TEXT NOT NULL DEFAULT '',
                    context_json TEXT NOT NULL DEFAULT '{}',
                    fix_applied TEXT NOT NULL DEFAULT '',
                    fix_succeeded INTEGER,
                    retries_needed INTEGER NOT NULL DEFAULT 0,
                    source_path TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_failures_type ON failures(failure_type, failure_stage);

                CREATE TABLE IF NOT EXISTS artifacts (
                    record_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    run_id TEXT NOT NULL DEFAULT '',
                    parent_id TEXT NOT NULL DEFAULT '',
                    producer TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS trajectories (
                    record_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    run_id TEXT NOT NULL DEFAULT '',
                    parent_id TEXT NOT NULL DEFAULT '',
                    producer TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    path TEXT NOT NULL DEFAULT '',
                    events_json TEXT NOT NULL DEFAULT '[]',
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    succeeded INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS import_sources (
                    source_key TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    importer TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    record_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS path_policies (
                    policy_key TEXT PRIMARY KEY,
                    policy_id TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    conditions_json TEXT NOT NULL DEFAULT '{}',
                    path_scores_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    evidence_count INTEGER NOT NULL DEFAULT 0,
                    evidence_json TEXT NOT NULL DEFAULT '{}',
                    reason TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    version INTEGER NOT NULL DEFAULT 1,
                    model TEXT NOT NULL DEFAULT '',
                    token_usage_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_path_policies_lookup
                    ON path_policies(objective, enabled, policy_id);

                -- Spec latent vectors for the adaptive router's cosine
                -- retrieval. Kept in its own table rather than as columns on
                -- `tasks` so that a spec can carry several vectors at once
                -- (different model / output_dim / protocol) and so that
                -- CREATE TABLE IF NOT EXISTS is a complete migration for
                -- databases written before latents existed.
                --
                -- `latent_vector` is raw little-endian float32, output_dim
                -- values, L2-normalised at write time. `protocol_sha256`
                -- pins the exact encoding recipe: vectors from two different
                -- protocols are NOT comparable and must never be mixed in one
                -- cosine ranking.
                CREATE TABLE IF NOT EXISTS spec_latents (
                    latent_key TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL DEFAULT '',
                    benchmark TEXT NOT NULL DEFAULT '',
                    spec_id TEXT NOT NULL DEFAULT '',
                    spec_sha256 TEXT NOT NULL,
                    model_alias TEXT NOT NULL,
                    model_id TEXT NOT NULL DEFAULT '',
                    output_dim INTEGER NOT NULL,
                    protocol_sha256 TEXT NOT NULL,
                    latent_vector BLOB NOT NULL,
                    vector_sha256 TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_spec_latents_protocol
                    ON spec_latents(protocol_sha256, model_alias, output_dim);
                CREATE INDEX IF NOT EXISTS idx_spec_latents_task
                    ON spec_latents(task_id);
                CREATE INDEX IF NOT EXISTS idx_spec_latents_spec
                    ON spec_latents(spec_sha256);
                """
            )
            self._migrate_action_gain_columns(db)

    # (column, sql type, objective whose expected_metric_gain uses this unit)
    _ACTION_GAIN_COLUMNS = (
        ("expected_area_gain_pct", "REAL", "AREA"),
        ("expected_timing_gain_ps", "REAL", "TIMING"),
    )

    def _migrate_action_gain_columns(self, db: sqlite3.Connection) -> None:
        """Add the typed expected-gain columns to an already-existing table.

        `CREATE TABLE IF NOT EXISTS` is a no-op once `actions` exists, so a new
        column on an old database needs an explicit ALTER.  The PRAGMA guard
        makes this idempotent, and the backfill runs only inside the branch that
        just added the column -- simply re-opening the store must never rewrite
        rows.  Rows whose objective is empty or unrecognised stay NULL: there is
        no unit to attribute their number to.
        """
        existing = {row["name"] for row in db.execute("PRAGMA table_info(actions)")}
        if not existing:
            return
        for column, sql_type, objective in self._ACTION_GAIN_COLUMNS:
            if column in existing:
                continue
            db.execute(f"ALTER TABLE actions ADD COLUMN {column} {sql_type}")
            db.execute(
                f"UPDATE actions SET {column} = expected_metric_gain "
                "WHERE expected_metric_gain IS NOT NULL "
                "AND UPPER(TRIM(objective)) = ?",
                (objective,),
            )

    def upsert_task(self, record: TaskRecord) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    benchmark=excluded.benchmark,
                    spec_text=CASE WHEN excluded.spec_text != '' THEN excluded.spec_text ELSE tasks.spec_text END,
                    source_path=CASE WHEN excluded.source_path != '' THEN excluded.source_path ELSE tasks.source_path END,
                    metadata_json=excluded.metadata_json
                """,
                (
                    record.task_id, record.benchmark, record.spec_text,
                    record.source_path, _json(record.metadata),
                    record.schema_version, record.created_at,
                ),
            )

    def upsert_run(self, record: RunRecord) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    objective=excluded.objective,
                    path=CASE WHEN excluded.path != '' THEN excluded.path ELSE runs.path END,
                    status=excluded.status,
                    metadata_json=excluded.metadata_json,
                    completed_at=excluded.completed_at
                """,
                (
                    record.run_id, record.task_id, record.objective, record.path,
                    record.status, record.producer, _json(record.metadata),
                    record.schema_version, record.started_at, record.completed_at,
                ),
            )

    def _upsert(self, table: str, columns: Iterable[str], values: Iterable[Any]) -> None:
        column_list = list(columns)
        placeholders = ", ".join("?" for _ in column_list)
        updates = ", ".join(
            f"{column}=excluded.{column}" for column in column_list if column != "record_id"
        )
        with self.connect() as db:
            db.execute(
                f"INSERT INTO {table} ({', '.join(column_list)}) VALUES ({placeholders}) "
                f"ON CONFLICT(record_id) DO UPDATE SET {updates}",
                tuple(values),
            )

    def upsert_feature(self, r: FeatureRecord) -> None:
        self._upsert("features", (
            "record_id", "task_id", "run_id", "parent_id", "producer", "timestamp",
            "schema_version", "features_json", "confidence_json", "overall_confidence",
            "optimization_target", "source_path",
        ), (
            r.record_id, r.task_id, r.run_id, r.parent_id, r.producer, r.timestamp,
            r.schema_version, _json(r.features), _json(r.confidence),
            r.overall_confidence, r.optimization_target, r.source_path,
        ))

    def upsert_selection(self, r: SelectionRecord) -> None:
        self._upsert("selections", (
            "record_id", "task_id", "run_id", "parent_id", "producer", "timestamp",
            "schema_version", "path", "rule_fired", "confidence", "tier", "reason",
            "token_usage_json",
        ), (
            r.record_id, r.task_id, r.run_id, r.parent_id, r.producer, r.timestamp,
            r.schema_version, r.path, r.rule_fired, r.confidence, r.tier, r.reason,
            _json(r.token_usage),
        ))

    def upsert_action(self, r: ActionRecord) -> None:
        success = None if r.applied_successfully is None else int(r.applied_successfully)
        area_gain, timing_gain = r.resolved_expected_gains()
        self._upsert("actions", (
            "record_id", "task_id", "run_id", "parent_id", "producer", "timestamp",
            "schema_version", "benchmark", "objective", "path", "region_id",
            "region_type", "transform_name", "selected", "applied_successfully",
            "planning_score", "expected_metric_gain", "expected_area_gain_pct",
            "expected_timing_gain_ps", "confidence", "context_json",
            "outcome_json", "source_path",
        ), (
            r.record_id, r.task_id, r.run_id, r.parent_id, r.producer, r.timestamp,
            r.schema_version, r.benchmark, r.objective, r.path, r.region_id,
            r.region_type, r.transform_name, int(r.selected), success,
            r.planning_score, r.expected_metric_gain, area_gain, timing_gain,
            r.confidence, _json(r.context), _json(r.outcome), r.source_path,
        ))

    def annotate_action_context(self, record_id: str, extra: Dict[str, Any]) -> None:
        """Merge extra keys into one action's stored context without rewriting it."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT context_json FROM actions WHERE record_id = ?", (record_id,)
            ).fetchone()
            if row is None:
                return
            context = _from_json(row["context_json"], {})
            if not isinstance(context, dict):
                context = {}
            context.update(extra)
            conn.execute(
                "UPDATE actions SET context_json = ? WHERE record_id = ?",
                (_json(context), record_id),
            )

    def upsert_evaluation(self, r: EvaluationRecord) -> None:
        self._upsert("evaluations", (
            "record_id", "task_id", "run_id", "parent_id", "producer", "timestamp",
            "schema_version", "benchmark", "objective", "path", "correctness_status",
            "synthesis_status", "area", "baseline_area", "area_gain_pct",
            "data_arrival_time_ps", "baseline_data_arrival_time_ps", "timing_gain_ps",
            "slack_ps", "baseline_slack_ps", "slack_status", "token_usage_json",
            "failure_reason", "metadata_json", "source_path",
        ), (
            r.record_id, r.task_id, r.run_id, r.parent_id, r.producer, r.timestamp,
            r.schema_version, r.benchmark, r.objective, r.path, r.correctness_status,
            r.synthesis_status, r.area, r.baseline_area, r.area_gain_pct,
            r.data_arrival_time_ps, r.baseline_data_arrival_time_ps, r.timing_gain_ps,
            r.slack_ps, r.baseline_slack_ps, r.slack_status, _json(r.token_usage),
            r.failure_reason, _json(r.metadata), r.source_path,
        ))

    def upsert_failure(self, r: FailureRecord) -> None:
        fixed = None if r.fix_succeeded is None else int(r.fix_succeeded)
        self._upsert("failures", (
            "record_id", "task_id", "run_id", "parent_id", "producer", "timestamp",
            "schema_version", "benchmark", "module", "failure_type", "failure_stage",
            "context_json", "fix_applied", "fix_succeeded", "retries_needed", "source_path",
        ), (
            r.record_id, r.task_id, r.run_id, r.parent_id, r.producer, r.timestamp,
            r.schema_version, r.benchmark, r.module, r.failure_type, r.failure_stage,
            _json(r.context), r.fix_applied, fixed, r.retries_needed, r.source_path,
        ))

    def upsert_artifact(self, r: ArtifactRecord) -> None:
        self._upsert("artifacts", (
            "record_id", "task_id", "run_id", "parent_id", "producer", "timestamp",
            "schema_version", "artifact_type", "path", "sha256", "metadata_json",
        ), (
            r.record_id, r.task_id, r.run_id, r.parent_id, r.producer, r.timestamp,
            r.schema_version, r.artifact_type, r.path, r.sha256, _json(r.metadata),
        ))

    def upsert_trajectory(self, r: TrajectoryRecord) -> None:
        self._upsert("trajectories", (
            "record_id", "task_id", "run_id", "parent_id", "producer", "timestamp",
            "schema_version", "path", "events_json", "total_tokens", "succeeded",
            "metadata_json",
        ), (
            r.record_id, r.task_id, r.run_id, r.parent_id, r.producer, r.timestamp,
            r.schema_version, r.path, _json(r.events), r.total_tokens,
            int(r.succeeded), _json(r.metadata),
        ))

    def upsert_path_policy(
        self,
        policy: Dict[str, Any],
        *,
        model: str = "",
        token_usage: Optional[Dict[str, Any]] = None,
        updated_at: str,
    ) -> None:
        policy_id = str(policy["policy_id"])
        objective = str(policy["objective"])
        policy_key = f"{policy_id}:{objective}"
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO path_policies (
                    policy_key, policy_id, objective, conditions_json,
                    path_scores_json, confidence, evidence_count, evidence_json,
                    reason, source, enabled, version, model, token_usage_json,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(policy_key) DO UPDATE SET
                    conditions_json=excluded.conditions_json,
                    path_scores_json=excluded.path_scores_json,
                    confidence=excluded.confidence,
                    evidence_count=excluded.evidence_count,
                    evidence_json=excluded.evidence_json,
                    reason=excluded.reason,
                    source=excluded.source,
                    enabled=excluded.enabled,
                    version=path_policies.version + 1,
                    model=excluded.model,
                    token_usage_json=excluded.token_usage_json,
                    updated_at=excluded.updated_at
                """,
                (
                    policy_key,
                    policy_id,
                    objective,
                    _json(policy.get("conditions", {})),
                    _json(policy.get("path_scores", {})),
                    float(policy.get("confidence", 0.0)),
                    int(policy.get("evidence_count", 0)),
                    _json(policy.get("evidence", {})),
                    str(policy.get("reason", "")),
                    str(policy.get("source", "")),
                    int(bool(policy.get("enabled", True))),
                    1,
                    model,
                    _json(token_usage or {}),
                    updated_at,
                ),
            )

    def path_policy_rows(
        self,
        objective: str = "",
        *,
        enabled_only: bool = True,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if objective:
            clauses.append("objective = ?")
            params.append(objective)
        if enabled_only:
            clauses.append("enabled = 1")
        query = "SELECT * FROM path_policies"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY policy_id"
        rows = self.rows(query, params)
        for row in rows:
            row["conditions"] = _from_json(row.pop("conditions_json"), {})
            row["path_scores"] = _from_json(row.pop("path_scores_json"), {})
            row["evidence"] = _from_json(row.pop("evidence_json"), {})
            row["token_usage"] = _from_json(row.pop("token_usage_json"), {})
        return rows

    def rows(self, query: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with self.connect() as db:
            return [dict(row) for row in db.execute(query, tuple(params)).fetchall()]

    # ── Spec latents (adaptive router cosine retrieval) ────────────────

    @staticmethod
    def latent_key(
        spec_sha256: str, model_alias: str, output_dim: int, protocol_sha256: str
    ) -> str:
        """Identity of one (spec, encoding protocol) pair.

        The protocol is part of the key on purpose: re-encoding the same spec
        under a new protocol must add a row, not overwrite the old vector, so
        that an in-flight experiment keeps a consistent comparison basis.
        """
        return (
            f"{spec_sha256}__{model_alias}__d{int(output_dim)}__"
            f"{protocol_sha256[:16]}"
        )

    def upsert_spec_latent(
        self,
        *,
        spec_sha256: str,
        model_alias: str,
        output_dim: int,
        protocol_sha256: str,
        latent_vector: bytes,
        task_id: str = "",
        benchmark: str = "",
        spec_id: str = "",
        model_id: str = "",
        vector_sha256: str = "",
        metadata: Dict[str, Any] | None = None,
        created_at: str = "",
    ) -> str:
        """Store one L2-normalised float32 latent vector. Returns its key."""
        expected_bytes = int(output_dim) * 4
        if len(latent_vector) != expected_bytes:
            raise ValueError(
                f"latent_vector is {len(latent_vector)} bytes, expected "
                f"{expected_bytes} for {output_dim} float32 values"
            )
        key = self.latent_key(spec_sha256, model_alias, output_dim, protocol_sha256)
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO spec_latents (
                    latent_key, task_id, benchmark, spec_id, spec_sha256,
                    model_alias, model_id, output_dim, protocol_sha256,
                    latent_vector, vector_sha256, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(latent_key) DO UPDATE SET
                    task_id=CASE WHEN excluded.task_id != '' THEN excluded.task_id
                                 ELSE spec_latents.task_id END,
                    benchmark=CASE WHEN excluded.benchmark != ''
                                   THEN excluded.benchmark
                                   ELSE spec_latents.benchmark END,
                    spec_id=CASE WHEN excluded.spec_id != '' THEN excluded.spec_id
                                 ELSE spec_latents.spec_id END,
                    latent_vector=excluded.latent_vector,
                    vector_sha256=excluded.vector_sha256,
                    metadata_json=excluded.metadata_json
                """,
                (
                    key, task_id, benchmark, spec_id, spec_sha256,
                    model_alias, model_id, int(output_dim), protocol_sha256,
                    sqlite3.Binary(latent_vector), vector_sha256,
                    _json(metadata or {}),
                    created_at or datetime.now(timezone.utc).isoformat(),
                ),
            )
        return key

    def spec_latent_rows(
        self,
        *,
        model_alias: str = "",
        output_dim: int = 0,
        protocol_sha256: str = "",
        exclude_spec_sha256: str = "",
    ) -> List[Dict[str, Any]]:
        """Every stored latent matching one encoding protocol.

        Filtering on the protocol is not an optimisation -- vectors produced by
        different protocols live in different spaces and their cosine is
        meaningless, so an unfiltered ranking would be silently wrong.
        """
        clauses: List[str] = []
        params: List[Any] = []
        if model_alias:
            clauses.append("model_alias = ?")
            params.append(model_alias)
        if output_dim:
            clauses.append("output_dim = ?")
            params.append(int(output_dim))
        if protocol_sha256:
            clauses.append("protocol_sha256 = ?")
            params.append(protocol_sha256)
        if exclude_spec_sha256:
            clauses.append("spec_sha256 != ?")
            params.append(exclude_spec_sha256)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.rows(f"SELECT * FROM spec_latents{where}", params)
        for row in rows:
            row["metadata"] = _from_json(row.pop("metadata_json", "{}"), {})
        return rows

    def feature_rows(self) -> List[Dict[str, Any]]:
        rows = self.rows(
            """
            SELECT f.*, t.benchmark, r.path AS run_path, r.objective AS run_objective
            FROM features f
            JOIN tasks t ON t.task_id = f.task_id
            LEFT JOIN runs r ON r.run_id = f.run_id
            """
        )
        for row in rows:
            row["features"] = _from_json(row.pop("features_json"), {})
            row["confidence"] = _from_json(row.pop("confidence_json"), {})
        return rows

    def action_rows(self, objective: str = "") -> List[Dict[str, Any]]:
        query = "SELECT * FROM actions"
        params: List[Any] = []
        if objective:
            query += " WHERE objective = ?"
            params.append(objective)
        rows = self.rows(query, params)
        for row in rows:
            row["context"] = _from_json(row.pop("context_json"), {})
            row["outcome"] = _from_json(row.pop("outcome_json"), {})
        return rows

    def failure_rows(self) -> List[Dict[str, Any]]:
        rows = self.rows("SELECT * FROM failures")
        for row in rows:
            row["context"] = _from_json(row.pop("context_json"), {})
        return rows

    def get_run_experience(self, run_id: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"run_id": run_id}
        for table in (
            "runs", "features", "selections", "actions", "evaluations",
            "failures", "artifacts", "trajectories",
        ):
            key = "run_id"
            rows = self.rows(f"SELECT * FROM {table} WHERE {key} = ?", (run_id,))
            result[table] = rows
        return result

    def status(self) -> Dict[str, int]:
        tables = (
            "tasks", "runs", "features", "selections", "actions",
            "evaluations", "failures", "artifacts", "trajectories",
            "path_policies", "spec_latents",
        )
        with self.connect() as db:
            return {
                table: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in tables
            }
