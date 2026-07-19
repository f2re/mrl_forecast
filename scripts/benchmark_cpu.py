#!/usr/bin/env python3
"""Measure CPU latency and memory for a registered radar model."""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_runtime import ModelRuntime  # noqa: E402


def _checkpoint_path(value: str) -> Path:
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if path.is_dir():
        path = path / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _fixture(runtime: ModelRuntime) -> tuple[np.ndarray, np.ndarray]:
    height, width = runtime.expected_grid_shape() or (256, 256)
    yy, xx = np.mgrid[0:height, 0:width]
    values = []
    for index in range(runtime.input_length):
        center_x = width * (0.35 + 0.015 * index)
        center_y = height * (0.50 - 0.008 * index)
        echo = 45.0 * np.exp(
            -(
                ((xx - center_x) / max(width * 0.08, 1.0)) ** 2
                + ((yy - center_y) / max(height * 0.06, 1.0)) ** 2
            )
        )
        values.append(echo.astype(np.float32))
    array = np.stack(values)
    return array, np.ones_like(array, dtype=bool)


def _max_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / 1024**2
    return value / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    parser.add_argument("--save", action="store_true", help="Save cpu_benchmark.json next to the model")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    runtime = ModelRuntime(torch.device("cpu"))
    checkpoint = _checkpoint_path(args.model_path)
    info = runtime.load(str(checkpoint))
    values, masks = _fixture(runtime)

    for _ in range(max(0, args.warmup)):
        runtime.predict(values, masks)

    durations = []
    for _ in range(max(1, args.repeats)):
        started = time.perf_counter()
        runtime.predict(values, masks)
        durations.append((time.perf_counter() - started) * 1000.0)

    sorted_durations = sorted(durations)
    p95_index = min(len(sorted_durations) - 1, int(round(0.95 * (len(sorted_durations) - 1))))
    report = {
        "model_id": info["model_id"],
        "model_architecture": runtime.architecture,
        "pipeline_version": info["pipeline_version"],
        "grid": runtime.grid,
        "input_length": runtime.input_length,
        "target_length": runtime.target_length,
        "forecast_step_minutes": runtime.forecast_step_minutes,
        "threads": torch.get_num_threads(),
        "warmup_runs": max(0, args.warmup),
        "measured_runs": len(durations),
        "latency_ms": {
            "mean": statistics.fmean(durations),
            "p50": statistics.median(durations),
            "p95": sorted_durations[p95_index],
            "min": min(durations),
            "max": max(durations),
        },
        "max_rss_mb": _max_rss_mb(),
        "device": "cpu",
        "fixture": "synthetic_echo_for_performance_only",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.save:
        destination = checkpoint.parent / "cpu_benchmark.json"
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
