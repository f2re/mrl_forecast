"""Render interpretable motion, growth, decay and uncertainty layers."""

from __future__ import annotations

import io
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _render_scalar(
    values: np.ndarray,
    title: str,
    range_km: float,
    unit: str,
    cmap: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> bytes:
    array = np.asarray(values, dtype=np.float32)
    if maximum is None:
        finite = array[np.isfinite(array)]
        maximum = float(np.percentile(finite, 99)) if finite.size else 1.0
        maximum = max(maximum, minimum + 1e-6)

    figure, axis = plt.subplots(figsize=(7, 7), dpi=100)
    image = axis.imshow(
        array,
        origin="lower",
        extent=(-range_km, range_km, -range_km, range_km),
        cmap=cmap,
        vmin=minimum,
        vmax=maximum,
    )
    axis.axhline(0.0, linewidth=0.4, alpha=0.35)
    axis.axvline(0.0, linewidth=0.4, alpha=0.35)
    axis.plot(0.0, 0.0, "k+", markersize=9)
    axis.set_title(title)
    axis.set_xlabel("км")
    axis.set_ylabel("км")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.78)
    colorbar.set_label(unit)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def render_evolution_layers(
    diagnostics: Dict[str, np.ndarray],
    lead_times_minutes: List[int],
    range_km: float,
) -> Dict[str, List[bytes]]:
    """Return per-lead PNG images for all available diagnostic heads."""

    if not {"motion", "growth", "decay", "uncertainty"}.issubset(diagnostics):
        return {}

    motion = diagnostics["motion"]
    motion_magnitude = np.sqrt(motion[:, 0] ** 2 + motion[:, 1] ** 2)
    layer_specs = {
        "motion": (motion_magnitude, "Скорость переноса", "пикселей/шаг", "viridis"),
        "growth": (diagnostics["growth"][:, 0], "Рост радиоэха", "proxy/шаг", "magma"),
        "decay": (diagnostics["decay"][:, 0], "Распад радиоэха", "proxy/шаг", "Blues"),
        "uncertainty": (diagnostics["uncertainty"][:, 0], "Неопределённость", "норм. ед.", "cividis"),
    }
    rendered: Dict[str, List[bytes]] = {}
    for layer_name, (values, title, unit, cmap) in layer_specs.items():
        common_max = float(np.percentile(values[np.isfinite(values)], 99)) if np.any(np.isfinite(values)) else 1.0
        common_max = max(common_max, 1e-6)
        rendered[layer_name] = [
            _render_scalar(
                values[index],
                f"{title} · T+{lead_times_minutes[index]} мин",
                range_km,
                unit,
                cmap,
                maximum=common_max,
            )
            for index in range(min(len(values), len(lead_times_minutes)))
        ]
    return rendered
