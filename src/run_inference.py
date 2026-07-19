#!/usr/bin/env python3
"""Run registered radar models from the terminal using the shared runtime."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

from adapters import DemoRadarAdapter, LocalDirectoryAdapter, NOAAAWSAdapter  # noqa: E402
from config import MAX_DBZ  # noqa: E402
from diagnostic_visualization import render_evolution_layers, render_quality_layers  # noqa: E402
from forecast_quality import summarize_forecast  # noqa: E402
from map_visualization import generate_sequence_plots  # noqa: E402
from model_runtime import ModelRuntime  # noqa: E402


def _checkpoint_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.is_dir():
        path = path / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Модель не найдена: {path}")
    return path


def _source_sequence(runtime: ModelRuntime, args):
    grid_shape = runtime.expected_grid_shape() or (256, 256)
    pipeline = runtime.pipeline()
    if args.source == "aws":
        adapter = NOAAAWSAdapter(grid_size=grid_shape, pipeline=pipeline)
        return adapter.get_latest_sequence(runtime.input_length, station_code=args.station)
    if args.source == "local":
        adapter = LocalDirectoryAdapter(args.local_dir, grid_size=grid_shape, pipeline=pipeline)
        return adapter.get_latest_sequence(runtime.input_length)
    return DemoRadarAdapter(grid_size=grid_shape).get_latest_sequence(runtime.input_length)


def _evolution_summary(diagnostics: Dict[str, np.ndarray]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    motion = diagnostics.get("motion")
    if motion is not None:
        result["mean_motion_pixels"] = float(
            np.mean(np.sqrt(motion[:, 0] ** 2 + motion[:, 1] ** 2))
        )
    for name in ("growth", "decay", "uncertainty"):
        if name in diagnostics:
            result[f"mean_{name}"] = float(np.mean(diagnostics[name]))
    return result


def _quality_summary(quality: Dict[str, np.ndarray]) -> Dict[str, float]:
    valid = np.asarray(quality["valid_mask"], dtype=bool)
    coverage = np.asarray(quality["coverage_mask"], dtype=bool)
    clutter = np.asarray(quality["clutter_mask"], dtype=bool)
    weights = np.asarray(quality["interpolation_weight"], dtype=np.float32)
    return {
        "valid_fraction": float(valid.mean()),
        "coverage_fraction": float(coverage.mean()),
        "clutter_fraction": float(clutter.mean()),
        "mean_interpolation_weight": (
            float(weights[coverage].mean()) if np.any(coverage) else 0.0
        ),
    }


def _save_pngs(
    output_dir: pathlib.Path,
    station: str,
    history: np.ndarray,
    forecast: np.ndarray,
    timestamps,
    runtime: ModelRuntime,
    diagnostics: Dict[str, np.ndarray],
    forecast_quality: Dict[str, np.ndarray],
) -> None:
    range_km = float(runtime.grid.get("radius_km", 250.0))
    images = generate_sequence_plots(
        history,
        forecast,
        runtime.input_length,
        station_code=station,
        start_datetime=timestamps[-1],
        history_timestamps=timestamps,
        interval_minutes=runtime.forecast_step_minutes,
        max_range_km=range_km,
    )
    for index, image in enumerate(images):
        if index < runtime.input_length:
            name = f"{station}_history_{index:02d}.png"
        else:
            name = f"{station}_forecast_{index - runtime.input_length + 1:02d}.png"
        (output_dir / name).write_bytes(image)

    lead_times = [runtime.forecast_step_minutes * (index + 1) for index in range(runtime.target_length)]
    rendered = render_evolution_layers(diagnostics, lead_times, range_km)
    rendered.update(render_quality_layers(forecast_quality, lead_times, range_km))
    for layer, layer_images in rendered.items():
        for index, image in enumerate(layer_images):
            (output_dir / f"{station}_{layer}_{lead_times[index]:03d}min.png").write_bytes(image)


def main() -> int:
    parser = argparse.ArgumentParser(description="Экспериментальный прогноз радиоэха МРЛ")
    parser.add_argument("--model-path", required=True, help="best_model.pt или каталог модели")
    parser.add_argument("--station", default="kokx", help="Код радиолокатора")
    parser.add_argument("--source", choices=("aws", "local", "demo"), default="aws")
    parser.add_argument("--local-dir", default="data/processed", help="Каталог для source=local")
    parser.add_argument("--output-dir", default="data/predictions")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = ModelRuntime(device)
    model_info = runtime.load(str(_checkpoint_path(args.model_path)))

    print(
        f"Модель: {model_info['model_id']} · {runtime.architecture} · "
        f"{runtime.forecast_step_minutes} мин · {device}"
    )
    sequence = _source_sequence(runtime, args)
    values = sequence.stack(require_observed=sequence.status == "observed")
    quality = sequence.quality_arrays()
    timestamps = sequence.timestamps[-runtime.input_length:]
    print(f"Источник: {sequence.message}")

    result = runtime.predict(
        values,
        valid_mask=quality["valid_mask"],
        coverage_mask=quality["coverage_mask"],
        clutter_mask=quality["clutter_mask"],
        interpolation_weight=quality["interpolation_weight"],
    )
    history = result["input"]
    forecast = result["forecast"]
    diagnostics = result["diagnostics"]
    forecast_quality = result["forecast_quality_masks"]
    forecast_diagnostics = summarize_forecast(forecast * MAX_DBZ)
    if forecast_diagnostics["uniform_field_anomaly"]:
        print(f"Прогноз отклонён quality gate: {forecast_diagnostics}")
        return 2

    lead_times = np.asarray(
        [runtime.forecast_step_minutes * (index + 1) for index in range(runtime.target_length)],
        dtype=np.int16,
    )
    arrays: Dict[str, Any] = {
        "history_dbz": history * MAX_DBZ,
        "forecast_dbz": forecast * MAX_DBZ,
        "history_timestamps_utc": np.asarray([value.isoformat() for value in timestamps], dtype="U32"),
        "lead_times_minutes": lead_times,
    }
    for name, value in result["quality_masks"].items():
        arrays[f"history_{name}"] = value
    for name, value in forecast_quality.items():
        arrays[f"forecast_{name}"] = value
    for name, value in diagnostics.items():
        if isinstance(value, np.ndarray):
            arrays[name] = value
    np.savez_compressed(output_dir / f"{args.station}_forecast.npz", **arrays)

    _save_pngs(
        output_dir,
        args.station.lower(),
        history,
        forecast,
        timestamps,
        runtime,
        diagnostics,
        forecast_quality,
    )
    report = {
        "product": "experimental_radar_reflectivity_nowcast",
        "not_official_warning": True,
        "source": args.source,
        "station": args.station.upper(),
        "base_time_utc": timestamps[-1].isoformat(),
        "lead_times_minutes": lead_times.tolist(),
        "model": model_info,
        "forecast_diagnostics": forecast_diagnostics,
        "quality_diagnostics": _quality_summary(forecast_quality),
        "evolution_diagnostics": _evolution_summary(diagnostics),
    }
    (output_dir / f"{args.station}_forecast.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Результаты сохранены: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
