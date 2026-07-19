"""Training datasets for masked radar sequences."""

from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import MAX_DBZ


class RadarSequenceDataset(Dataset):
    """Load new masked NPZ sequences and legacy NPY sequences."""

    def __init__(self, data_dir: str, input_length: int = 4, target_length: int = 4):
        self.data_dir = pathlib.Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz")) + sorted(self.data_dir.glob("*.npy"))
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
