#!/usr/bin/env python3
"""Download open DWD ODIM HDF5 reflectivity sweeps without altering source files."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import pathlib

from dwd_source import DWDOpenDataAdapter
from metadata_utils import save_metadata
from radar_catalog import RadarCatalog


def download_dwd_data(
    station: str,
    date: str,
    count: int = 100,
    output_root: str = "data/raw/archive",
) -> str:
    requested_date = datetime.date.fromisoformat(date)
    station_code = station.strip().lower()
    adapter = DWDOpenDataAdapter()
    scans = [scan for scan in adapter.list_scans(station_code) if scan.scan_time.date() == requested_date]
    if not scans:
        raise ValueError(f"DWD has no indexed scans for {station_code.upper()} on {date}")

    selected = scans[: max(1, min(count, len(scans)))]
    session_id = (
        f"DWD_{station_code.upper()}_{date}_"
        f"{datetime.datetime.now().strftime('%H%M%S')}"
    )
    output_dir = pathlib.Path(output_root) / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "raw_data",
        "source": "dwd-open-data",
        "format": "ODIM_H5",
        "product": "sweep_pcp_z_dbzh_00",
        "filter": adapter.filter_name,
        "station": station_code.upper(),
        "date": date,
        "native_time_step_minutes": 5,
        "requested_count": len(selected),
        "downloaded_count": 0,
        "status": "downloading",
        "files": [],
        "errors": [],
    }
    save_metadata(str(output_dir), metadata)

    for index, scan in enumerate(selected, start=1):
        try:
            response = adapter.session.get(scan.url, timeout=adapter.timeout_seconds)
            response.raise_for_status()
            content = response.content
            target = output_dir / scan.filename
            target.write_bytes(content)
            metadata["files"].append(
                {
                    "filename": scan.filename,
                    "timestamp_utc": scan.scan_time.isoformat(),
                    "url": scan.url,
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
            metadata["downloaded_count"] = len(metadata["files"])
            print(f"[{index}/{len(selected)}] downloaded {scan.filename}")
        except Exception as exc:
            metadata["errors"].append({"filename": scan.filename, "error": str(exc)})
            print(f"[{index}/{len(selected)}] failed {scan.filename}: {exc}")
        save_metadata(str(output_dir), metadata)

    if not metadata["files"]:
        metadata["status"] = "failed"
        metadata["error"] = "No DWD files were downloaded"
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

    print(f"Downloaded {metadata['downloaded_count']} DWD files to {output_dir}")
    return str(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DWD ODIM HDF5 radar sweeps")
    parser.add_argument("--station", default="ess")
    parser.add_argument("--date", required=True, help="UTC date in YYYY-MM-DD format")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", default="data/raw/archive")
    args = parser.parse_args()
    download_dwd_data(args.station, args.date, args.count, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
