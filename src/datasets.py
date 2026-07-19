"""Training datasets for masked radar sequences."""

from __future__ import annotations

import json
import pathlib
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from config import MAX_DBZ
from event_catalog import dry_echo_balance_weights


class RadarSequenceDataset(Dataset):
    """Load new masked NPZ sequences and legacy NPY sequences."""

    def __init__(self, data_dir: str, input_length: int = 4, target_length: int = 4):
        self.data_dir = pathlib.Path(data_dir)
        npz_files = sorted(self.data_dir.glob("*.npz"))
        files = npz_files or sorted(self.data_dir.glob("*.npy"))
        catalog = self._load_event_catalog()

        selected = [
            (path, catalog.get(path.name, "unknown"))
            for path in files
            if catalog.get(path.name, "unknown") != "invalid"
        ]
        self.files = [path for path, _event_class in selected]
        self.sample_classes = [event_class for _path, event_class in selected]
        self.input_length = input_length
        self.target_length = target_length
        self.required_length = input_length + target_length

        if self.files:
            values, _ = self._load_arrays(self.files[0])
            if values.shape[0] < self.required_length:
                raise ValueError(
                    f"Dataset sequences (len={values.shape[0]}) are shorter than "
                    f"requested input+target ({self.required_length})"
                )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        values, valid_mask = self._load_arrays(self.files[index])
        if values.shape[0] < self.required_length:
            raise ValueError(f"Sequence {self.files[index]} is shorter than {self.required_length} frames")

        values = np.clip(np.nan_to_num(values, nan=0.0), 0.0, MAX_DBZ).astype(np.float32)
        values = values[:, np.newaxis, :, :] / MAX_DBZ
        valid_mask = valid_mask[:, np.newaxis, :, :].astype(np.float32)

        split = self.input_length
        end = split + self.target_length
        return (
            torch.from_numpy(values[:split]),
            torch.from_numpy(values[split:end]),
            torch.from_numpy(valid_mask[:split]),
            torch.from_numpy(valid_mask[split:end]),
        )

    def _load_event_catalog(self) -> dict[str, str]:
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {
            str(item.get("file")): str(item.get("event_class", "unknown"))
            for item in manifest.get("sequences", [])
            if item.get("file")
        }

    @staticmethod
    def _load_arrays(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
        if path.suffix == ".npz":
            with np.load(path) as payload:
                if "reflectivity" in payload:
                    values = payload["reflectivity"]
                elif "arr_0" in payload:
                    values = payload["arr_0"]
                else:
                    raise ValueError(f"NPZ sequence {path} has no reflectivity array")
                valid_mask = payload["valid_mask"] if "valid_mask" in payload else np.isfinite(values)
        else:
            values = np.load(path)
            valid_mask = np.isfinite(values)

        values = np.asarray(values, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if values.ndim != 3:
            raise ValueError(f"Expected [T,H,W] sequence in {path}, got shape {values.shape}")
        if valid_mask.shape != values.shape:
            raise ValueError(f"valid_mask shape {valid_mask.shape} does not match {values.shape} in {path}")
        return values, valid_mask


def sample_classes_for_dataset(dataset: Dataset) -> list[str]:
    """Resolve event classes through Subset/ConcatDataset wrappers."""

    if isinstance(dataset, RadarSequenceDataset):
        return list(dataset.sample_classes)
    if isinstance(dataset, Subset):
        parent_classes = sample_classes_for_dataset(dataset.dataset)
        return [parent_classes[index] for index in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        classes: list[str] = []
        for part in dataset.datasets:
            classes.extend(sample_classes_for_dataset(part))
        return classes
    return ["unknown"] * len(dataset)


def balanced_sample_weights(dataset: Dataset) -> Optional[list[float]]:
    """Return 50/50 dry-vs-echo weights when both groups are available."""

    return dry_echo_balance_weights(sample_classes_for_dataset(dataset))
