#!/usr/bin/env python3
"""Build and inspect the local radar observation catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar_catalog import RadarCatalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("rebuild", "summary", "list"))
    parser.add_argument("--raw-root", default=str(ROOT / "data" / "raw" / "archive"))
    parser.add_argument("--datasets-root", default=str(ROOT / "data" / "processed_archive"))
    parser.add_argument("--source")
    parser.add_argument("--station")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    catalog = RadarCatalog()
    if args.command == "rebuild":
        result = catalog.rebuild(args.raw_root, args.datasets_root)
    elif args.command == "list":
        result = catalog.list_observations(
            source=args.source,
            station=args.station,
            limit=args.limit,
        )
    else:
        result = catalog.summary()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
