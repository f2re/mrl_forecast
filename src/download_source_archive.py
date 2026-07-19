#!/usr/bin/env python3
"""Download files from a registered source into a provenance-preserving archive."""

from __future__ import annotations

import argparse
import datetime
import pathlib
import re
from typing import Any

from international_sources import KnmiRadarSource
from metadata_utils import save_metadata
from radar_catalog import RadarCatalog
from source_registry import build_default_source_registry


ODIM_SOURCES = {"dwd-open-data", "fmi-s3", "dmi-radar", "knmi-radar"}


def _date_range(value: str):
    if not value:
        return None, None
    date = datetime.date.fromisoformat(value)
    start = datetime.datetime.combine(date, datetime.time.min, tzinfo=datetime.UTC)
    return start, start + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)


def _safe_token(value: str, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return token or fallback


def _create_source(registry, args):
    if args.source == "knmi-radar":
        return registry.create(
            args.source,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
        )
    return registry.create(args.source)


def download_registered_source(args) -> str:
    registry = build_default_source_registry()
    capabilities = registry.capabilities(args.source)
    if not capabilities.download_supported:
        raise ValueError(
            f"Source {args.source} has no automated downloader. "
            f"Run: python mrl.py sources --action info --source {args.source}"
        )

    source = _create_source(registry, args)
    if not hasattr(source, "list_files") or not hasattr(source, "download"):
        raise ValueError(f"Source {args.source} does not implement list_files/download")

    start, end = _date_range(args.date)
    list_kwargs: dict[str, Any] = {
        "limit": args.count,
        "prefix": args.prefix,
        "station": args.station,
        "collection": args.collection,
        "start": start,
        "end": end,
    }
    list_kwargs = {
        key: value
        for key, value in list_kwargs.items()
        if value not in (None, "")
    }
    files = source.list_files(**list_kwargs)
    if not files:
        raise RuntimeError(f"Source {args.source} returned no files for the requested selection")

    selected = files[: max(1, min(args.count, len(files)))]
    station_token = _safe_token(args.station, "MULTI")
    date_token = args.date or datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    session_id = (
        f"{_safe_token(args.source, 'SOURCE').upper()}_"
        f"{station_token.upper()}_{date_token}_"
        f"{datetime.datetime.now().strftime('%H%M%S')}"
    )
    output_dir = pathlib.Path(args.output) / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "type": "raw_data",
        "source": args.source,
        "station": args.station.upper() if args.station else "MULTI",
        "date": args.date or None,
        "format": capabilities.native_format,
        "decoder": "odim_h5" if args.source in ODIM_SOURCES else "auto",
        "access": capabilities.to_metadata(),
        "selection": {
            "prefix": args.prefix,
            "collection": args.collection,
            "dataset_name": args.dataset_name if args.source == "knmi-radar" else None,
            "dataset_version": args.dataset_version if args.source == "knmi-radar" else None,
        },
        "requested_count": len(selected),
        "downloaded_count": 0,
        "status": "downloading",
        "files": [],
        "errors": [],
    }
    save_metadata(str(output_dir), metadata)

    for index, remote in enumerate(selected, start=1):
        try:
            result = source.download(remote, str(output_dir))
            metadata["files"].append(
                {
                    **remote.to_metadata(),
                    **result,
                }
            )
            metadata["downloaded_count"] = len(metadata["files"])
            print(f"[{index}/{len(selected)}] downloaded {remote.filename}")
        except Exception as exc:
            metadata["errors"].append(
                {"file_id": remote.file_id, "filename": remote.filename, "error": str(exc)}
            )
            print(f"[{index}/{len(selected)}] failed {remote.filename}: {exc}")
        save_metadata(str(output_dir), metadata)

    if not metadata["files"]:
        metadata["status"] = "failed"
        metadata["error"] = "No files were downloaded"
        save_metadata(str(output_dir), metadata)
        raise RuntimeError(metadata["error"])

    metadata["status"] = "completed"
    metadata["error_count"] = len(metadata["errors"])
    save_metadata(str(output_dir), metadata)
    try:
        RadarCatalog().index_archive(str(output_dir))
        metadata["catalog_indexed"] = True
    except Exception as exc:
        metadata["catalog_indexed"] = False
        metadata["catalog_error"] = str(exc)
    save_metadata(str(output_dir), metadata)
    print(f"Downloaded {metadata['downloaded_count']} files to {output_dir}")
    return str(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--station", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--collection", default="volume")
    parser.add_argument("--dataset-name", default=KnmiRadarSource.DEFAULT_DATASET)
    parser.add_argument("--dataset-version", default=KnmiRadarSource.DEFAULT_VERSION)
    parser.add_argument("--output", default="data/raw/archive")
    args = parser.parse_args()
    download_registered_source(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
