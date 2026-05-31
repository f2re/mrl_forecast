#!/usr/bin/env python3
"""Local environment diagnostics for MRL Forecast."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = [
    "torch",
    "numpy",
    "flask",
    "xarray",
    "matplotlib",
    "pyart",
    "nexradaws",
    "boto3",
    "pyproj",
]


def check_modules() -> dict:
    missing = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            missing.append({"module": name, "error": str(exc)})
    return {"ok": not missing, "missing": missing}


def check_writable_directories() -> dict:
    directories = [ROOT / "data", ROOT / "models"]
    failures = [str(path) for path in directories if not path.exists() or not os.access(path, os.W_OK)]
    return {"ok": not failures, "failures": failures}


def check_disk() -> dict:
    usage = shutil.disk_usage(ROOT)
    return {
        "ok": usage.free >= 2 * 1024**3,
        "free_gb": round(usage.free / 1024**3, 2),
    }


def check_aws(station: str, date: str) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "check_aws_source.py"),
        "--station",
        station,
        "--date",
        date,
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-aws", action="store_true")
    parser.add_argument("--station", default="KOKX")
    parser.add_argument("--date", default="2024-05-20")
    args = parser.parse_args()

    report = {
        "python": {"ok": sys.version_info >= (3, 10), "version": sys.version},
        "modules": check_modules(),
        "directories": check_writable_directories(),
        "disk": check_disk(),
    }
    if args.check_aws:
        report["aws"] = check_aws(args.station, args.date)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all(item["ok"] for item in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

