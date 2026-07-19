#!/usr/bin/env python3
"""Read-only health check for the DWD ODIM HDF5 radar adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwd_source import DWDOpenDataAdapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default="ess", help="Three-character DWD radar code")
    parser.add_argument("--decode-one", action="store_true")
    parser.add_argument("--list-stations", action="store_true")
    args = parser.parse_args()

    adapter = DWDOpenDataAdapter()
    report = {
        "source": adapter.CAPABILITIES.to_metadata(),
        "station": args.station.lower(),
    }
    try:
        if args.list_stations:
            report["stations"] = adapter.list_stations()
        scans = adapter.list_scans(args.station)
        report["scan_count"] = len(scans)
        report["first_scan_utc"] = scans[0].scan_time.isoformat() if scans else None
        report["last_scan_utc"] = scans[-1].scan_time.isoformat() if scans else None
        report["last_filename"] = scans[-1].filename if scans else None
        if args.decode_one and scans:
            sequence = adapter.get_latest_sequence(1, station_code=args.station)
            frame = sequence.frames[0]
            report["decoded_frame"] = {
                "shape": list(frame.data.shape),
                "timestamp_utc": frame.timestamp_utc.isoformat(),
                "qc": frame.qc,
                "provenance": frame.provenance,
            }
    except Exception as exc:
        report["error"] = str(exc)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("scan_count", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
