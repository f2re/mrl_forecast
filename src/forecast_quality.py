"""Forecast baselines and lightweight anomaly checks."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.ndimage import shift


def persistence_forecast(history: np.ndarray, output_steps: int) -> np.ndarray:
    """Repeat the most recent observed frame for every forecast lead time."""
    if history.ndim < 3 or history.shape[0] == 0:
        raise ValueError("history must contain at least one radar frame")
    return np.repeat(history[-1:,...], output_steps, axis=0)


def advection_forecast(
    history: np.ndarray,
    output_steps: int,
    *,
    search_radius: int = 6,
) -> np.ndarray:
    """Extrapolate the latest frame with a simple constant-motion baseline."""
    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("history must have shape [T, H, W] with at least two frames")
    previous, current = values[-2], values[-1]
    best_error = float("inf")
    best_motion = (0, 0)
    for delta_y in range(-search_radius, search_radius + 1):
        for delta_x in range(-search_radius, search_radius + 1):
            moved = shift(previous, (delta_y, delta_x), order=0, mode="constant", cval=0.0)
            error = float(np.mean((moved - current) ** 2))
            if error < best_error:
                best_error = error
                best_motion = (delta_y, delta_x)
    return np.stack(
        [
            shift(
                current,
                (best_motion[0] * lead_time, best_motion[1] * lead_time),
                order=0,
                mode="constant",
                cval=0.0,
            )
            for lead_time in range(1, output_steps + 1)
        ],
        axis=0,
    )


def threshold_metrics_by_lead_time(
    forecast: np.ndarray,
    target: np.ndarray,
    *,
    thresholds=(5.0, 10.0, 20.0, 30.0),
) -> Dict[str, list[Dict[str, float]]]:
    """Calculate categorical precipitation metrics for each lead time."""
    predicted = np.asarray(forecast)
    observed = np.asarray(target)
    if predicted.shape != observed.shape or predicted.ndim not in (3, 4):
        raise ValueError("forecast and target must have shape [T, H, W] or [N, T, H, W]")
    if predicted.ndim == 3:
        predicted = predicted[np.newaxis, ...]
        observed = observed[np.newaxis, ...]
    report = {}
    for threshold in thresholds:
        lead_metrics = []
        for lead_time in range(predicted.shape[1]):
            predicted_rain = predicted[:, lead_time] >= threshold
            observed_rain = observed[:, lead_time] >= threshold
            hits = int(np.sum(predicted_rain & observed_rain))
            misses = int(np.sum(~predicted_rain & observed_rain))
            false_alarms = int(np.sum(predicted_rain & ~observed_rain))
            csi_denominator = hits + misses + false_alarms
            pod_denominator = hits + misses
            far_denominator = hits + false_alarms
            lead_metrics.append(
                {
                    "hits": hits,
                    "misses": misses,
                    "false_alarms": false_alarms,
                    "csi": hits / csi_denominator if csi_denominator else 1.0,
                    "pod": hits / pod_denominator if pod_denominator else 1.0,
                    "far": false_alarms / far_denominator if far_denominator else 0.0,
                }
            )
        report[str(float(threshold))] = lead_metrics
    return report


def is_uniform_forecast(
    forecast: np.ndarray,
    *,
    min_mean_dbz: float = 1.0,
    max_spatial_std_dbz: float = 1.0,
    min_covered_fraction: float = 0.95,
) -> bool:
    """Detect nearly constant precipitation layers such as the green-layer failure."""
    values = np.asarray(forecast, dtype=np.float32)
    if values.ndim < 3 or values.size == 0:
        raise ValueError("forecast must have shape [T, H, W]")
    covered_fraction = float(np.mean(values >= min_mean_dbz))
    spatial_std = float(np.mean(np.std(values, axis=(-2, -1))))
    mean_dbz = float(np.mean(values))
    return (
        mean_dbz >= min_mean_dbz
        and covered_fraction >= min_covered_fraction
        and spatial_std <= max_spatial_std_dbz
    )


def mse_by_lead_time(forecast: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return mean squared error for each forecast lead time."""
    if forecast.shape != target.shape:
        raise ValueError("forecast and target shapes must match")
    return np.mean((forecast - target) ** 2, axis=(-2, -1))


def summarize_forecast(forecast: np.ndarray) -> Dict[str, Any]:
    """Return compact diagnostics suitable for metadata and API responses."""
    values = np.asarray(forecast, dtype=np.float32)
    return {
        "min_dbz": float(np.min(values)),
        "max_dbz": float(np.max(values)),
        "mean_dbz": float(np.mean(values)),
        "covered_fraction_1dbz": float(np.mean(values >= 1.0)),
        "uniform_field_anomaly": is_uniform_forecast(values),
    }
