#!/usr/bin/env python3
"""Download NOAA NEXRAD Level II observations and preserve source provenance."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
from datetime import datetime
from typing import Any

import nexradaws

from metadata_utils import save_metadata
from radar_catalog import RadarCatalog


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _success_record(item: Any) -> dict:
    scan = getattr(item, "scan", None)
    filepath = pathlib.Path(getattr(item, "filepath", ""))
    filename = getattr(scan, "filename", None) or filepath.name
    scan_time = getattr(scan, "scan_time", None)
    return {
        "filename": filename,
        "timestamp_utc": scan_time.isoformat() if scan_time is not None else None,
        "size_bytes": filepath.stat().st_size,
        "sha256": _sha256(filepath),
    }


def download_nexrad_data(
    station: str,
    start_date: str,
    end_file_count: int = 100,
    output_root: str = "data/raw/archive",
) -> str:
    """Download one UTC day of public NEXRAD Level II data."""

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["AWS_REGION"] = "us-east-1"
    station = station.upper()
    start = datetime.strptime(start_date, "%Y-%m-%d")
    session_id = f"{station}_{start_date}_{datetime.now().strftime('%H%M%S')}"
    output_dir = pathlib.Path(output_root) / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "type": "raw_data",
        "station": station,
        "source": "aws",
        "format": "NEXRAD_LEVEL2",
        "bucket": "unidata-nexrad-level2",
        "region": "us-east-1",
        "date": start_date,
        "requested_count": int(end_file_count),
        "downloaded_count": 0,
        "status": "downloading",
        "files": [],
        "errors": [],
    }
    save_metadata(str(output_dir), metadata)

    connection = nexradaws.NexradAwsInterface()
    print(f"Searching for scans for station {station} on {start_date}...")
    scans = [
        scan
        for scan in connection.get_avail_scans(start.year, start.month, start.day, station)
        if not scan.filename.endswith("_MDM")
    ]
    if not scans:
        metadata["status"] = "failed"
        metadata["error"] = "No observed scans found"
        save_metadata(str(output_dir), metadata)
        raise ValueError(metadata["error"])

    selected = scans[: min(len(scans), max(1, int(end_file_count)))]
    print(f"Found {len(scans)} observed scans. Downloading {len(selected)}...")
    results = connection.download(selected, str(output_dir))

    for item in results.success:
        try:
            metadata["files"].append(_success_record(item))
        except Exception as exc:
            metadata["errors"].append({"file": str(item), "error": str(exc)})
    for item in getattr(results, "failed", []):
        metadata["errors"].append({"file": str(item), "error": "download failed"})

    metadata["downloaded_count"] = len(metadata["files"])
    metadata["error_count"] = len(metadata["errors"])
    if not metadata["files"]:
        metadata["status"] = "failed"
        metadata["error"] = "NOAA download returned no usable files"
        save_metadata(str(output_dir), metadata)
        raise RuntimeError(metadata["error"])

    metadata["status"] = "completed"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NEXRAD Level II archive data from AWS")
    parser.add_argument("--station", default="KOKX")
    parser.add_argument("--date", required=True, help="UTC date in YYYY-MM-DD format")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", default="data/raw/archive")
    args = parser.parse_args()
    download_nexrad_data(args.station, args.date, args.count, args.output)
