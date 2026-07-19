"""Forecast baselines and lightweight verification metrics."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, shift, uniform_filter, zoom


def persistence_forecast(history: np.ndarray, output_steps: int) -> np.ndarray:
    """Repeat the most recent observed frame for every forecast lead time."""
    if history.ndim < 3 or history.shape[0] == 0:
        raise ValueError("history must contain at least one radar frame")
    return np.repeat(history[-1:, ...], output_steps, axis=0)


def advection_forecast(
    history: np.ndarray,
    output_steps: int,
    *,
    search_radius: int = 6,
) -> np.ndarray:
    """Extrapolate the latest frame with one constant motion vector."""
    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("history must have shape [T,H,W] with at least two frames")
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


def block_motion_field(
    history: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    downsample: int = 8,
    block_size: int = 8,
    search_radius: int = 3,
) -> np.ndarray:
    """Estimate a smooth local displacement field by coarse block matching."""

    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("history must have shape [T,H,W] with at least two frames")
    factor = max(1, int(downsample))
    previous = np.nan_to_num(values[-2], nan=0.0)[::factor, ::factor]
    current = np.nan_to_num(values[-1], nan=0.0)[::factor, ::factor]

    if valid_mask is None:
        previous_valid = np.isfinite(values[-2])[::factor, ::factor]
        current_valid = np.isfinite(values[-1])[::factor, ::factor]
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must match history shape")
        previous_valid = mask[-2][::factor, ::factor]
        current_valid = mask[-1][::factor, ::factor]

    candidates = []
    for delta_y in range(-search_radius, search_radius + 1):
        for delta_x in range(-search_radius, search_radius + 1):
            moved = shift(previous, (delta_y, delta_x), order=1, mode="constant", cval=0.0)
            moved_valid = shift(
                previous_valid.astype(np.float32),
                (delta_y, delta_x),
                order=0,
                mode="constant",
                cval=0.0,
            ) >= 0.5
            candidates.append((delta_y, delta_x, moved, moved_valid))

    height, width = current.shape
    coarse_y = np.zeros((height, width), dtype=np.float32)
    coarse_x = np.zeros((height, width), dtype=np.float32)
    block = max(2, int(block_size))
    for y0 in range(0, height, block):
        y1 = min(y0 + block, height)
        for x0 in range(0, width, block):
            x1 = min(x0 + block, width)
            target = current[y0:y1, x0:x1]
            target_valid = current_valid[y0:y1, x0:x1]
            minimum_valid = max(4, int(target_valid.size * 0.20))
            best_error = float("inf")
            best_motion = (0.0, 0.0)
            for delta_y, delta_x, moved, moved_valid in candidates:
                local_valid = target_valid & moved_valid[y0:y1, x0:x1]
                if int(local_valid.sum()) < minimum_valid:
                    continue
                difference = moved[y0:y1, x0:x1] - target
                weights = 1.0 + 2.0 * (target >= 20.0)
                error = float(np.mean((difference[local_valid] ** 2) * weights[local_valid]))
                if error < best_error:
                    best_error = error
                    best_motion = (float(delta_y), float(delta_x))
            coarse_y[y0:y1, x0:x1] = best_motion[0]
            coarse_x[y0:y1, x0:x1] = best_motion[1]

    sigma = max(block / 3.0, 1.0)
    coarse_y = gaussian_filter(coarse_y, sigma=sigma)
    coarse_x = gaussian_filter(coarse_x, sigma=sigma)
    full_height, full_width = values.shape[-2:]
    scale = (full_height / height, full_width / width)
    flow_y = zoom(coarse_y, scale, order=1)[:full_height, :full_width] * factor
    flow_x = zoom(coarse_x, scale, order=1)[:full_height, :full_width] * factor
    return np.stack([flow_y, flow_x], axis=0).astype(np.float32)


def block_motion_forecast(
    history: np.ndarray,
    output_steps: int,
    *,
    valid_mask: Optional[np.ndarray] = None,
    downsample: int = 8,
    block_size: int = 8,
    search_radius: int = 3,
) -> np.ndarray:
    """Extrapolate the latest frame with a locally varying motion field."""

    values = np.asarray(history, dtype=np.float32)
    flow = block_motion_field(
        values,
        valid_mask=valid_mask,
        downsample=downsample,
        block_size=block_size,
        search_radius=search_radius,
    )
    current = np.nan_to_num(values[-1], nan=0.0)
    yy, xx = np.mgrid[0:current.shape[0], 0:current.shape[1]].astype(np.float32)
    forecasts = []
    for lead in range(1, output_steps + 1):
        source_y = yy - flow[0] * lead
        source_x = xx - flow[1] * lead
        forecasts.append(
            map_coordinates(
                current,
                [source_y, source_x],
                order=1,
                mode="constant",
                cval=0.0,
            )
        )
    return np.stack(forecasts, axis=0).astype(np.float32)


def _normalise_forecast_arrays(
    forecast: np.ndarray,
    target: np.ndarray,
    valid_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(forecast, dtype=np.float32)
    observed = np.asarray(target, dtype=np.float32)
    if predicted.shape != observed.shape or predicted.ndim not in (3, 4):
        raise ValueError("forecast and target must have shape [T,H,W] or [N,T,H,W]")
    if predicted.ndim == 3:
        predicted = predicted[np.newaxis, ...]
        observed = observed[np.newaxis, ...]
    if valid_mask is None:
        valid = np.isfinite(observed)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.ndim == 3:
            valid = valid[np.newaxis, ...]
        if valid.shape != observed.shape:
            raise ValueError("valid_mask must match forecast and target shape")
    return predicted, observed, valid


def threshold_metrics_by_lead_time(
    forecast: np.ndarray,
    target: np.ndarray,
    *,
    thresholds=(5.0, 10.0, 20.0, 30.0),
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, list[Dict[str, Any]]]:
    """Calculate categorical metrics for each lead time over valid pixels."""

    predicted, observed, valid = _normalise_forecast_arrays(forecast, target, valid_mask)
    report = {}
    for threshold in thresholds:
        lead_metrics = []
        for lead_time in range(predicted.shape[1]):
            lead_valid = valid[:, lead_time]
            predicted_echo = predicted[:, lead_time] >= threshold
            observed_echo = observed[:, lead_time] >= threshold
            hits = int(np.sum(predicted_echo & observed_echo & lead_valid))
            misses = int(np.sum(~predicted_echo & observed_echo & lead_valid))
            false_alarms = int(np.sum(predicted_echo & ~observed_echo & lead_valid))
            correct_negatives = int(np.sum(~predicted_echo & ~observed_echo & lead_valid))
            total = hits + misses + false_alarms + correct_negatives
            csi_denominator = hits + misses + false_alarms
            pod_denominator = hits + misses
            far_denominator = hits + false_alarms
            random_hits = (
                (hits + misses) * (hits + false_alarms) / total if total else 0.0
            )
            ets_denominator = hits + misses + false_alarms - random_hits
            lead_metrics.append(
                {
                    "valid_pixels": int(np.sum(lead_valid)),
                    "hits": hits,
                    "misses": misses,
                    "false_alarms": false_alarms,
                    "correct_negatives": correct_negatives,
                    "csi": hits / csi_denominator if csi_denominator else None,
                    "pod": hits / pod_denominator if pod_denominator else None,
                    "far": false_alarms / far_denominator if far_denominator else None,
                    "frequency_bias": (
                        (hits + false_alarms) / pod_denominator if pod_denominator else None
                    ),
                    "ets": (hits - random_hits) / ets_denominator if ets_denominator else None,
                }
            )
        report[str(float(threshold))] = lead_metrics
    return report


def fractions_skill_score_by_lead_time(
    forecast: np.ndarray,
    target: np.ndarray,
    *,
    thresholds=(20.0, 30.0),
    scales=(1, 4, 8, 16),
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, list[Optional[float]]]]:
    """Return Fractions Skill Score for several thresholds and spatial scales."""

    predicted, observed, valid = _normalise_forecast_arrays(forecast, target, valid_mask)
    result: Dict[str, Dict[str, list[Optional[float]]]] = {}
    for threshold in thresholds:
        threshold_result: Dict[str, list[Optional[float]]] = {}
        for scale in scales:
            lead_scores: list[Optional[float]] = []
            window = max(1, int(scale))
            size = (1, window, window)
            for lead in range(predicted.shape[1]):
                lead_valid = valid[:, lead].astype(np.float32)
                local_valid = uniform_filter(lead_valid, size=size, mode="constant")
                predicted_fraction = uniform_filter(
                    ((predicted[:, lead] >= threshold) * lead_valid).astype(np.float32),
                    size=size,
                    mode="constant",
                ) / np.maximum(local_valid, 1e-6)
                observed_fraction = uniform_filter(
                    ((observed[:, lead] >= threshold) * lead_valid).astype(np.float32),
                    size=size,
                    mode="constant",
                ) / np.maximum(local_valid, 1e-6)
                usable = local_valid >= 0.5
                if not np.any(usable):
                    lead_scores.append(None)
                    continue
                numerator = float(np.mean((predicted_fraction[usable] - observed_fraction[usable]) ** 2))
                denominator = float(
                    np.mean(predicted_fraction[usable] ** 2 + observed_fraction[usable] ** 2)
                )
                lead_scores.append(1.0 - numerator / denominator if denominator > 0 else None)
            threshold_result[str(window)] = lead_scores
        result[str(float(threshold))] = threshold_result
    return result


def spatial_metrics_by_lead_time(
    forecast: np.ndarray,
    target: np.ndarray,
    *,
    thresholds=(20.0, 30.0, 40.0),
    valid_mask: Optional[np.ndarray] = None,
) -> list[Dict[str, Any]]:
    """Summarise peak intensity and echo-area bias for each lead time."""

    predicted, observed, valid = _normalise_forecast_arrays(forecast, target, valid_mask)
    metrics = []
    for lead in range(predicted.shape[1]):
        maximum_errors = []
        for sample in range(predicted.shape[0]):
            sample_valid = valid[sample, lead]
            if np.any(sample_valid):
                maximum_errors.append(
                    float(np.max(predicted[sample, lead][sample_valid]) - np.max(observed[sample, lead][sample_valid]))
                )
        area_bias = {}
        for threshold in thresholds:
            lead_valid = valid[:, lead]
            predicted_area = int(np.sum((predicted[:, lead] >= threshold) & lead_valid))
            observed_area = int(np.sum((observed[:, lead] >= threshold) & lead_valid))
            area_bias[str(float(threshold))] = (
                predicted_area / observed_area if observed_area else None
            )
        metrics.append(
            {
                "max_dbz_error": float(np.mean(maximum_errors)) if maximum_errors else None,
                "area_bias": area_bias,
            }
        )
    return metrics


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
        raise ValueError("forecast must have shape [T,H,W]")
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
