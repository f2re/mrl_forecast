#!/usr/bin/env python3
"""Summarize processed radar datasets and registry models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_arrays(directory: Path) -> dict:
    files = sorted(directory.glob("*.npy"))
    if not files:
        return {"directory": str(directory), "sample_count": 0}
    arrays = [np.load(path) for path in files]
    return {
        "directory": str(directory),
        "sample_count": len(files),
        "shape": list(arrays[0].shape),
        "dtype": str(arrays[0].dtype),
        "min_dbz": float(min(np.nanmin(array) for array in arrays)),
        "max_dbz": float(max(np.nanmax(array) for array in arrays)),
        "mean_dbz": float(np.mean([np.nanmean(array) for array in arrays])),
        "zero_fraction": float(np.mean([np.mean(array == 0) for array in arrays])),
        "nan_count": int(sum(np.isnan(array).sum() for array in arrays)),
    }


def registry_models(directory: Path) -> list[dict]:
    models = []
    for metadata_path in sorted(directory.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        models.append(
            {
                "model_id": metadata.get("model_id", metadata_path.parent.name),
                "status": metadata.get("status", "unknown"),
                "pipeline_version": metadata.get("pipeline_version", "legacy"),
                "metrics": metadata.get("metrics", {}),
                "has_learning_curve": (metadata_path.parent / "learning_curve.png").exists(),
            }
        )
    return models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", action="append", default=[])
    parser.add_argument("--models-dir", default="models/registry")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = {
        "datasets": [summarize_arrays(Path(path)) for path in args.dataset_dir],
        "models": registry_models(Path(args.models_dir)),
    }
    content = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(content + "\n", encoding="utf-8")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

