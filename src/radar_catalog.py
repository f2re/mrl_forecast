"""SQLite catalog for raw radar observations and prepared datasets."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import re
import sqlite3
from typing import Any, Optional

from metadata_utils import load_metadata

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "radar_catalog.sqlite3"
NEXRAD_TIMESTAMP = re.compile(r"^[A-Z0-9]{4}(\d{8})_(\d{6})")
DWD_TIMESTAMP = re.compile(r"DBZH_00-(\d{12})", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_from_filename(filename: str) -> Optional[str]:
    nexrad = NEXRAD_TIMESTAMP.match(filename.upper())
    if nexrad:
        value = datetime.datetime.strptime("".join(nexrad.groups()), "%Y%m%d%H%M%S")
        return value.replace(tzinfo=datetime.UTC).isoformat()
    dwd = DWD_TIMESTAMP.search(filename)
    if dwd:
        value = datetime.datetime.strptime(dwd.group(1), "%Y%m%d%H%M")
        return value.replace(tzinfo=datetime.UTC).isoformat()
    return None


class RadarCatalog:
    """Keep a small, queryable index without replacing source metadata files."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = pathlib.Path(db_path or os.environ.get("MRL_CATALOG_DB", DEFAULT_DB_PATH))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_archives (
                    path TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    station TEXT NOT NULL,
                    observed_date TEXT,
                    status TEXT NOT NULL,
                    file_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS observations (
                    path TEXT PRIMARY KEY,
                    archive_path TEXT NOT NULL,
                    source TEXT NOT NULL,
                    station TEXT NOT NULL,
                    timestamp_utc TEXT,
                    format TEXT,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    qc_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(archive_path) REFERENCES raw_archives(path) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_observations_source_station_time
                ON observations(source, station, timestamp_utc);

                CREATE TABLE IF NOT EXISTS datasets (
                    path TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    station TEXT NOT NULL,
                    pipeline_version TEXT NOT NULL,
                    time_step_minutes INTEGER,
                    sample_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    class_counts_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _file_metadata(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for item in metadata.get("files", []):
            if isinstance(item, str):
                result[pathlib.PurePath(item).name] = {"filename": pathlib.PurePath(item).name}
            elif isinstance(item, dict):
                filename = item.get("filename") or item.get("source_file")
                if filename:
                    result[pathlib.PurePath(str(filename)).name] = dict(item)
        return result

    def index_archive(self, archive_dir: str) -> dict[str, Any]:
        archive = pathlib.Path(archive_dir).resolve()
        metadata = load_metadata(str(archive))
        if not metadata or metadata.get("type") != "raw_data":
            raise ValueError(f"Raw archive metadata is missing in {archive}")

        source = str(metadata.get("source", "unknown"))
        station = str(metadata.get("station", "unknown")).upper()
        files = sorted(
            path
            for path in archive.iterdir()
            if path.is_file() and path.name != "metadata.json" and not path.name.endswith("_MDM")
        )
        file_metadata = self._file_metadata(metadata)
        now = _utc_now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO raw_archives(
                    path, source, station, observed_date, status,
                    file_count, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    source=excluded.source,
                    station=excluded.station,
                    observed_date=excluded.observed_date,
                    status=excluded.status,
                    file_count=excluded.file_count,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(archive), source, station, metadata.get("date"),
                    str(metadata.get("status", "unknown")), len(files), _json(metadata), now,
                ),
            )

            current_paths = set()
            for path in files:
                current_paths.add(str(path.resolve()))
                item = file_metadata.get(path.name, {})
                timestamp = item.get("timestamp_utc") or timestamp_from_filename(path.name)
                checksum = str(item.get("sha256") or _sha256(path))
                file_format = str(metadata.get("format") or path.suffix.lstrip(".") or source)
                connection.execute(
                    """
                    INSERT INTO observations(
                        path, archive_path, source, station, timestamp_utc, format,
                        size_bytes, sha256, qc_json, provenance_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        archive_path=excluded.archive_path,
                        source=excluded.source,
                        station=excluded.station,
                        timestamp_utc=excluded.timestamp_utc,
                        format=excluded.format,
                        size_bytes=excluded.size_bytes,
                        sha256=excluded.sha256,
                        qc_json=excluded.qc_json,
                        provenance_json=excluded.provenance_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        str(path.resolve()), str(archive), source, station, timestamp,
                        file_format, path.stat().st_size, checksum,
                        _json(item.get("qc", {})), _json(item), now,
                    ),
                )

            existing = connection.execute(
                "SELECT path FROM observations WHERE archive_path=?", (str(archive),)
            ).fetchall()
            for row in existing:
                if row["path"] not in current_paths:
                    connection.execute("DELETE FROM observations WHERE path=?", (row["path"],))

        return {
            "archive": str(archive),
            "source": source,
            "station": station,
            "file_count": len(files),
        }

    def index_dataset(self, dataset_dir: str) -> dict[str, Any]:
        dataset = pathlib.Path(dataset_dir).resolve()
        metadata = load_metadata(str(dataset))
        if not metadata or metadata.get("type") != "dataset":
            raise ValueError(f"Dataset metadata is missing in {dataset}")
        manifest_path = dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        pipeline = metadata.get("pipeline", {})
        source = str(metadata.get("source_type", "unknown"))
        station = str(metadata.get("station", "unknown")).upper()
        now = _utc_now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets(
                    path, source, station, pipeline_version, time_step_minutes,
                    sample_count, status, class_counts_json, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    source=excluded.source,
                    station=excluded.station,
                    pipeline_version=excluded.pipeline_version,
                    time_step_minutes=excluded.time_step_minutes,
                    sample_count=excluded.sample_count,
                    status=excluded.status,
                    class_counts_json=excluded.class_counts_json,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(dataset), source, station,
                    str(pipeline.get("pipeline_version", "unknown")),
                    pipeline.get("time_step_minutes"), int(metadata.get("sample_count", 0)),
                    str(metadata.get("status", "unknown")),
                    _json(metadata.get("class_counts", {})), _json(metadata), now,
                ),
            )

            source_path = metadata.get("source_path")
            if source_path:
                archive = pathlib.Path(str(source_path))
                archive = (ROOT / archive).resolve() if not archive.is_absolute() else archive.resolve()
                for frame in manifest.get("frames", []):
                    filename = frame.get("source_file")
                    if not filename:
                        continue
                    observation_path = str((archive / str(filename)).resolve())
                    connection.execute(
                        """
                        UPDATE observations
                        SET qc_json=?, provenance_json=?, updated_at=?
                        WHERE path=?
                        """,
                        (
                            _json(frame.get("qc", {})),
                            _json(frame.get("provenance", {})), now, observation_path,
                        ),
                    )

        return {
            "dataset": str(dataset),
            "source": source,
            "station": station,
            "sample_count": int(metadata.get("sample_count", 0)),
        }

    def rebuild(self, raw_root: str, datasets_root: str) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("DELETE FROM observations")
            connection.execute("DELETE FROM raw_archives")
            connection.execute("DELETE FROM datasets")

        archives = 0
        datasets = 0
        errors = []
        raw_path = pathlib.Path(raw_root)
        if raw_path.exists():
            for child in sorted(raw_path.iterdir()):
                if not child.is_dir():
                    continue
                try:
                    self.index_archive(str(child))
                    archives += 1
                except Exception as exc:
                    errors.append({"path": str(child), "error": str(exc)})

        dataset_path = pathlib.Path(datasets_root)
        if dataset_path.exists():
            for child in sorted(dataset_path.iterdir()):
                if not child.is_dir():
                    continue
                try:
                    self.index_dataset(str(child))
                    datasets += 1
                except Exception as exc:
                    errors.append({"path": str(child), "error": str(exc)})
        return {"archives": archives, "datasets": datasets, "errors": errors, **self.summary()}

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            archives = connection.execute("SELECT COUNT(*) FROM raw_archives").fetchone()[0]
            observations = connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            datasets = connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
            stations = connection.execute(
                "SELECT COUNT(DISTINCT source || ':' || station) FROM observations"
            ).fetchone()[0]
            source_rows = connection.execute(
                """
                SELECT source, COUNT(*) AS observations, COUNT(DISTINCT station) AS stations,
                       MIN(timestamp_utc) AS first_time_utc, MAX(timestamp_utc) AS last_time_utc
                FROM observations GROUP BY source ORDER BY source
                """
            ).fetchall()
        return {
            "archives": int(archives),
            "observations": int(observations),
            "datasets": int(datasets),
            "stations": int(stations),
            "sources": [dict(row) for row in source_rows],
            "db_path": str(self.db_path),
        }

    def list_observations(
        self,
        *,
        source: Optional[str] = None,
        station: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = []
        values: list[Any] = []
        if source:
            clauses.append("source=?")
            values.append(source)
        if station:
            clauses.append("station=?")
            values.append(station.upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, min(int(limit), 1000)))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT path, archive_path, source, station, timestamp_utc, format,
                       size_bytes, sha256, qc_json, provenance_json
                FROM observations {where}
                ORDER BY timestamp_utc DESC LIMIT ?
                """,
                values,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["qc"] = json.loads(item.pop("qc_json"))
            item["provenance"] = json.loads(item.pop("provenance_json"))
            result.append(item)
        return result
