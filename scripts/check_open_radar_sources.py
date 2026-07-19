#!/usr/bin/env python3
"""Read-only health check for open radar discovery and visual sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_sources import MeteoinfoVisualSource, RainViewerMetadataSource, Wis2RadarCatalog  # noqa: E402


def _record_title(feature):
    properties = feature.get("properties", {})
    return properties.get("title") or feature.get("id") or "unknown"


def check_wis2(limit: int) -> dict:
    source = Wis2RadarCatalog()
    records = source.search_russian_radar(limit=limit)
    return {
        "ok": True,
        "capabilities": source.CAPABILITIES.to_metadata(),
        "candidate_count": len(records),
        "candidates": [
            {"id": record.get("id"), "title": _record_title(record)}
            for record in records[:10]
        ],
        "note": "Candidates remain discovery-only until format and data access are verified.",
    }


def check_meteoinfo() -> dict:
    source = MeteoinfoVisualSource()
    return {
        "ok": True,
        "capabilities": source.CAPABILITIES.to_metadata(),
        "image_url": source.discover_image_url(),
    }


def check_rainviewer() -> dict:
    source = RainViewerMetadataSource()
    frames = source.latest_radar_frames()
    return {
        "ok": True,
        "capabilities": source.CAPABILITIES.to_metadata(),
        "frame_count": len(frames),
        "latest_frame": frames[-1] if frames else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("all", "wis2", "meteoinfo", "rainviewer"), default="all")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    checks = {
        "wis2": lambda: check_wis2(args.limit),
        "meteoinfo": check_meteoinfo,
        "rainviewer": check_rainviewer,
    }
    selected = checks if args.source == "all" else {args.source: checks[args.source]}
    report = {}
    success = True
    for name, function in selected.items():
        try:
            report[name] = function()
        except Exception as exc:
            success = False
            report[name] = {"ok": False, "error": str(exc)}

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
