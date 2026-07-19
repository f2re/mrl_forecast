"""Simple radar sequence classification for cataloguing and train balancing."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Optional

import numpy as np

EVENT_CLASSES = (
    "dry_valid",
    "weak_echo",
    "precipitation",
    "convective",
    "severe_core",
    "invalid",
)


def summarize_sequence(reflectivity_dbz: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Any]:
    """Return compact, reproducible statistics for a radar sequence."""

    values = np.asarray(reflectivity_dbz, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.ndim != 3 or valid.shape != values.shape:
        raise ValueError("reflectivity_dbz and valid_mask must have shape [T,H,W]")

    valid_count = int(valid.sum())
    total_count = int(valid.size)
    statistics: Dict[str, Any] = {
        "valid_fraction": valid_count / total_count if total_count else 0.0,
        "valid_pixel_count": valid_count,
    }
    if valid_count == 0:
        statistics.update({"min_dbz": None, "max_dbz": None, "mean_dbz": None})
        for threshold in (5, 20, 30, 35, 40, 45):
            statistics[f"pixel_count_ge_{threshold}dbz"] = 0
            statistics[f"fraction_ge_{threshold}dbz"] = 0.0
        statistics["area_trend_ge_20dbz"] = 0.0
        statistics["event_class"] = "invalid"
        return statistics

    valid_values = values[valid]
    statistics.update(
        {
            "min_dbz": float(valid_values.min()),
            "max_dbz": float(valid_values.max()),
            "mean_dbz": float(valid_values.mean()),
        }
    )
    for threshold in (5, 20, 30, 35, 40, 45):
        count = int(np.sum((values >= threshold) & valid))
        statistics[f"pixel_count_ge_{threshold}dbz"] = count
        statistics[f"fraction_ge_{threshold}dbz"] = count / valid_count

    first_valid = valid[0]
    last_valid = valid[-1]
    first_area = int(np.sum((values[0] >= 20.0) & first_valid))
    last_area = int(np.sum((values[-1] >= 20.0) & last_valid))
    normalization = max(int(last_valid.sum()), 1)
    statistics["area_trend_ge_20dbz"] = (last_area - first_area) / normalization
    statistics["event_class"] = classify_statistics(statistics)
    return statistics


def classify_statistics(statistics: Dict[str, Any]) -> str:
    """Classify a sequence using transparent reflectivity thresholds."""

    if statistics.get("valid_fraction", 0.0) < 0.70:
        return "invalid"
    if statistics.get("pixel_count_ge_5dbz", 0) < 4:
        return "dry_valid"
    if statistics.get("pixel_count_ge_20dbz", 0) < 4:
        return "weak_echo"
    if statistics.get("pixel_count_ge_35dbz", 0) < 4:
        return "precipitation"
    if statistics.get("pixel_count_ge_45dbz", 0) < 4:
        return "convective"
    return "severe_core"


def dry_echo_balance_weights(classes: Iterable[str]) -> Optional[list[float]]:
    """Return weights giving dry and echo groups equal sampling probability."""

    class_list = list(classes)
    if not class_list:
        return None
    dry_indices = [index for index, value in enumerate(class_list) if value == "dry_valid"]
    echo_indices = [
        index
        for index, value in enumerate(class_list)
        if value in {"weak_echo", "precipitation", "convective", "severe_core"}
    ]
    if not dry_indices or not echo_indices:
        return None

    weights = [0.0] * len(class_list)
    dry_weight = 0.5 / len(dry_indices)
    echo_weight = 0.5 / len(echo_indices)
    for index in dry_indices:
        weights[index] = dry_weight
    for index in echo_indices:
        weights[index] = echo_weight
    return weights


def class_counts(classes: Iterable[str]) -> Dict[str, int]:
    counts = Counter(classes)
    return {name: int(counts.get(name, 0)) for name in EVENT_CLASSES}
