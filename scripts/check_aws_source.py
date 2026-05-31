#!/usr/bin/env python3
"""Read-only NOAA NEXRAD AWS health check."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters import configure_public_aws_region  # noqa: E402
from radar_pipeline import RadarPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default="KOKX")
    parser.add_argument("--date", required=True, help="UTC date in YYYY-MM-DD format")
    parser.add_argument("--decode-one", action="store_true")
    args = parser.parse_args()

    configure_public_aws_region()
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    import nexradaws

    date = datetime.datetime.strptime(args.date, "%Y-%m-%d")
    connection = nexradaws.NexradAwsInterface()
    scans = connection.get_avail_scans(date.year, date.month, date.day, args.station.upper())
    observed = [scan for scan in scans if not scan.filename.endswith("_MDM")]
    report = {
        "station": args.station.upper(),
        "date": args.date,
        "bucket": "unidata-nexrad-level2",
        "region": "us-east-1",
        "scan_count": len(scans),
        "observed_scan_count": len(observed),
        "first_scan": observed[0].filename if observed else None,
        "last_scan": observed[-1].filename if observed else None,
    }
    if args.decode_one and observed:
        with tempfile.TemporaryDirectory() as directory:
            results = connection.download([observed[0]], directory)
            if not results.success:
                raise RuntimeError("AWS returned no downloaded file")
            frame = RadarPipeline().process_file(
                results.success[0].filepath,
                timestamp_utc=observed[0].scan_time,
                station=args.station,
                source="aws",
            )
            report["decoded_frame"] = {
                "shape": list(frame.data.shape),
                "qc": frame.qc,
            }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if observed else 1


if __name__ == "__main__":
    raise SystemExit(main())

