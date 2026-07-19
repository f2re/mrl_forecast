#!/usr/bin/env python3
"""Train radar nowcasting baselines and the physics-guided evolution model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import traceback
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from convlstm import ConvLSTM, ConvLSTMCell
from datasets import RadarSequenceDataset, balanced_sample_weights, sample_groups_for_dataset
from forecast_quality import (
    advection_forecast,
    block_motion_forecast,
    fractions_skill_score_by_lead_time,
    is_uniform_forecast,
    spatial_metrics_by_lead_time,
    threshold_metrics_by_lead_time,
)
from losses import MaskedMSELoss, PhysicsEvolutionLoss, masked_mse
from metadata_utils import load_metadata, save_metadata
from phys_evolution import MRLPhysEvolution

MAX_DBZ = 70.0


def temporal_split_indices(sample_count: int, overlap_frames: int, val_fraction: float):
    """Chronological split with a gap between overlapping windows."""

    if sample_count < 2:
        raise ValueError("At least two samples are required for train/validation split")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    boundary = int(sample_count * (1.0 - val_fraction))
    train_indices = list(range(max(boundary - overlap_frames, 0)))
    validation_indices = list(range(boundary, sample_count))
    if not train_indices or not validation_indices:
        raise ValueError("Dataset is too small for a leakage-free temporal split")
    return train_indices, validation_indices


def grouped_temporal_split_indices(
    groups: list[str],
    val_fraction: float,
    purge_groups: int = 1,
):
    """Split complete chronological blocks and remove adjacent boundary blocks."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    ordered_groups = list(dict.fromkeys(groups))
    validation_count = max(1, int(math.ceil(len(ordered_groups) * val_fraction)))
    boundary = len(ordered_groups) - validation_count
    train_stop = boundary - max(0, int(purge_groups))
    if train_stop < 1 or boundary >= len(ordered_groups):
        raise ValueError("Not enough chronological groups for a purged split")

    train_groups = set(ordered_groups[:train_stop])
    validation_groups = set(ordered_groups[boundary:])
    train_indices = [index for index, group in enumerate(groups) if group in train_groups]
    validation_indices = [index for index, group in enumerate(groups) if group in validation_groups]
    if not train_indices or not validation_indices:
        raise ValueError("Grouped split produced an empty partition")
    return train_indices, validation_indices


def grouped_train_validation_test_indices(
    groups: list[str],
    val_fraction: float,
    test_fraction: float,
    purge_groups: int = 1,
):
    """Create chronological train/validation/test blocks with purged boundaries."""

    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("val_fraction and test_fraction must be between 0 and 1")
    if val_fraction + test_fraction >= 0.8:
        raise ValueError("Validation and test fractions leave too little training data")

    ordered_groups = list(dict.fromkeys(groups))
    validation_count = max(1, int(math.ceil(len(ordered_groups) * val_fraction)))
    test_count = max(1, int(math.ceil(len(ordered_groups) * test_fraction)))
    purge = max(0, int(purge_groups))
    test_start = len(ordered_groups) - test_count
    validation_stop = test_start - purge
    validation_start = validation_stop - validation_count
    train_stop = validation_start - purge
    if train_stop < 1 or validation_start < 1 or test_start >= len(ordered_groups):
        raise ValueError("Not enough chronological groups for independent test data")

    train_groups = set(ordered_groups[:train_stop])
    validation_groups = set(ordered_groups[validation_start:validation_stop])
    test_groups = set(ordered_groups[test_start:])
    train_indices = [index for index, group in enumerate(groups) if group in train_groups]
    validation_indices = [index for index, group in enumerate(groups) if group in validation_groups]
    test_indices = [index for index, group in enumerate(groups) if group in test_groups]
    if not train_indices or not validation_indices or not test_indices:
        raise ValueError("Three-way grouped split produced an empty partition")
    return train_indices, validation_indices, test_indices


def build_temporal_datasets(datasets, val_fraction, test_fraction):
    train_parts = []
    validation_parts = []
    test_parts = []
    split_report = []
    for dataset in datasets:
        groups = sample_groups_for_dataset(dataset)
        unique_groups = list(dict.fromkeys(groups))
        strategy = "purged_train_validation_test"
        test_indices = []
        try:
            if len(unique_groups) < 5:
                raise ValueError("too few groups")
            train_indices, validation_indices, test_indices = grouped_train_validation_test_indices(
                groups,
                val_fraction,
                test_fraction,
                purge_groups=1,
            )
        except ValueError:
            strategy = "window_overlap_gap_no_test"
            overlap = dataset.input_length + dataset.target_length - 1
            train_indices, validation_indices = temporal_split_indices(
                len(dataset),
                overlap,
                val_fraction,
            )
        train_parts.append(Subset(dataset, train_indices))
        validation_parts.append(Subset(dataset, validation_indices))
        if test_indices:
            test_parts.append(Subset(dataset, test_indices))
        split_report.append(
            {
                "dataset": dataset.data_dir.name,
                "strategy": strategy,
                "group_count": len(unique_groups),
                "train_samples": len(train_indices),
                "validation_samples": len(validation_indices),
                "test_samples": len(test_indices),
            }
        )
    test_dataset = ConcatDataset(test_parts) if test_parts else None
    return ConcatDataset(train_parts), ConcatDataset(validation_parts), test_dataset, split_report


def _forward_model(model, x, x_mask, architecture):
    if architecture == "phys-evolution":
        return model(x, x_mask)
    prediction, states = model(x)
    return prediction, {"states": states}


def train_epoch(model, dataloader, criterion, optimizer, device, architecture):
    model.train()
    total_loss = 0.0
    for x, y, x_mask, y_mask in dataloader:
        x, y = x.to(device), y.to(device)
        x_mask, y_mask = x_mask.to(device), y_mask.to(device)
        optimizer.zero_grad()
        prediction, diagnostics = _forward_model(model, x, x_mask, architecture)
        loss = criterion(prediction, y, y_mask, diagnostics)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device, architecture):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y, x_mask, y_mask in dataloader:
            x, y = x.to(device), y.to(device)
            x_mask, y_mask = x_mask.to(device), y_mask.to(device)
            prediction, diagnostics = _forward_model(model, x, x_mask, architecture)
            total_loss += criterion(prediction, y, y_mask, diagnostics).item()
    return total_loss / len(dataloader)


def quality_gate_passes(metrics: Dict[str, float]) -> bool:
    model_mse = metrics["model_mse"]
    baselines = [
        metrics["persistence_mse"],
        metrics.get("advection_mse", float("inf")),
        metrics.get("block_motion_mse", float("inf")),
    ]
    return all(model_mse < baseline for baseline in baselines) and not metrics["uniform_field_anomaly"]


def _masked_numpy_mse(prediction, target, valid_mask):
    valid = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return None
    return float(np.mean((np.asarray(prediction)[valid] - np.asarray(target)[valid]) ** 2))


def evaluate_model_quality(model, dataloader, device, architecture):
    """Compare model, persistence, global shift and local block motion."""

    model.eval()
    model_losses = []
    persistence_losses = []
    advection_losses = []
    block_motion_losses = []
    forecasts = []
    targets = []
    masks = []
    uniform_anomaly = False
    diagnostic_values = {"motion_pixels": [], "growth_proxy": [], "decay_proxy": [], "uncertainty": []}

    with torch.no_grad():
        for x, y, x_mask, y_mask in dataloader:
            x, y = x.to(device), y.to(device)
            x_mask, y_mask = x_mask.to(device), y_mask.to(device)
            prediction, diagnostics = _forward_model(model, x, x_mask, architecture)
            persistence = x[:, -1:].expand_as(y)
            model_losses.append(masked_mse(prediction, y, y_mask).item())
            persistence_losses.append(masked_mse(persistence, y, y_mask).item())

            if architecture == "phys-evolution":
                motion = diagnostics["motion"]
                diagnostic_values["motion_pixels"].append(
                    torch.sqrt(motion[:, :, 0] ** 2 + motion[:, :, 1] ** 2).mean().item()
                )
                diagnostic_values["growth_proxy"].append(diagnostics["growth"].mean().item())
                diagnostic_values["decay_proxy"].append(diagnostics["decay"].mean().item())
                diagnostic_values["uncertainty"].append(diagnostics["uncertainty"].mean().item())

            for history, history_mask, forecast, target, target_mask in zip(
                x.cpu().numpy(),
                x_mask.cpu().numpy().astype(bool),
                prediction.cpu().numpy(),
                y.cpu().numpy(),
                y_mask.cpu().numpy().astype(bool),
            ):
                valid = target_mask[:, 0]
                forecast_dbz = forecast[:, 0] * MAX_DBZ
                history_dbz = history[:, 0] * MAX_DBZ
                history_valid = history_mask[:, 0]
                uniform_anomaly = uniform_anomaly or is_uniform_forecast(
                    np.where(valid, forecast_dbz, 0.0)
                )
                advection = advection_forecast(
                    history_dbz,
                    forecast.shape[0],
                ) / MAX_DBZ
                block_motion = block_motion_forecast(
                    history_dbz,
                    forecast.shape[0],
                    valid_mask=history_valid,
                ) / MAX_DBZ
                advection_loss = _masked_numpy_mse(advection, target[:, 0], valid)
                block_motion_loss = _masked_numpy_mse(block_motion, target[:, 0], valid)
                if advection_loss is not None:
                    advection_losses.append(advection_loss)
                if block_motion_loss is not None:
                    block_motion_losses.append(block_motion_loss)
                forecasts.append(forecast_dbz)
                targets.append(target[:, 0] * MAX_DBZ)
                masks.append(valid)

    forecast_array = np.stack(forecasts)
    target_array = np.stack(targets)
    mask_array = np.stack(masks)
    metrics = {
        "model_mse": float(np.mean(model_losses)),
        "persistence_mse": float(np.mean(persistence_losses)),
        "advection_mse": float(np.mean(advection_losses)) if advection_losses else float("inf"),
        "block_motion_mse": (
            float(np.mean(block_motion_losses)) if block_motion_losses else float("inf")
        ),
        "uniform_field_anomaly": bool(uniform_anomaly),
        "threshold_metrics": threshold_metrics_by_lead_time(
            forecast_array,
            target_array,
            valid_mask=mask_array,
        ),
        "fractions_skill_score": fractions_skill_score_by_lead_time(
            forecast_array,
            target_array,
            valid_mask=mask_array,
        ),
        "spatial_metrics": spatial_metrics_by_lead_time(
            forecast_array,
            target_array,
            valid_mask=mask_array,
        ),
    }
    if architecture == "phys-evolution":
        metrics["evolution_diagnostics"] = {
            name: float(np.mean(values)) if values else None
            for name, values in diagnostic_values.items()
        }
    metrics["quality_gate_passed"] = quality_gate_passes(metrics)
    return metrics


def save_history(model_dir, history):
    with open(os.path.join(model_dir, "history.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, values in enumerate(zip(history["train_loss"], history["val_loss"]), start=1):
            writer.writerow([epoch, *values])

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, "learning_curve.png"))
    plt.close()


def _load_datasets(data_dirs, input_length, target_length):
    datasets = []
    provenance = []
    pipeline_versions = set()
    time_steps = set()
    grids = {}
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue
        metadata = load_metadata(data_dir) or {}
        pipeline = metadata.get("pipeline", {})
        pipeline_version = pipeline.get("pipeline_version")
        if metadata.get("status") != "completed" or not pipeline_version:
            print(f"Skipping incompatible dataset {data_dir}: completed pipeline metadata is required")
            continue
        dataset = RadarSequenceDataset(data_dir, input_length, target_length)
        if len(dataset) == 0:
            continue
        datasets.append(dataset)
        pipeline_versions.add(pipeline_version)
        if pipeline.get("time_step_minutes") is not None:
            time_steps.add(int(pipeline["time_step_minutes"]))
        grid = pipeline.get("grid", {})
        grids[json.dumps(grid, sort_keys=True)] = grid
        provenance.append(
            {
                "dataset_id": os.path.basename(data_dir),
                "station": metadata.get("station", "unknown"),
                "samples": len(dataset),
                "pipeline_version": pipeline_version,
                "time_step_minutes": pipeline.get("time_step_minutes"),
                "sample_format": metadata.get("sample_format", "legacy-npy"),
                "class_counts": metadata.get("class_counts", {}),
                "split_group_count": metadata.get("split_group_count", 0),
            }
        )
        print(f"Loaded dataset from {data_dir} ({len(dataset)} samples)")
    return datasets, provenance, pipeline_versions, time_steps, grids


def _build_model(args, device):
    if args.architecture == "phys-evolution":
        config = {
            "input_channels": 3,
            "base_channels": args.base_channels,
            "hidden_channels": args.hidden_channels,
            "output_steps": args.target_length,
            "max_motion_pixels": args.max_motion_pixels,
            "max_evolution_per_step": args.max_evolution_per_step,
        }
        return MRLPhysEvolution(**config).to(device), PhysicsEvolutionLoss(), config

    config = {
        "input_channels": 1,
        "hidden_channels": [args.hidden_channels, args.hidden_channels],
        "kernel_size": (3, 3),
        "output_steps": args.target_length,
    }
    return ConvLSTM(**config).to(device), MaskedMSELoss(), config


def _save_checkpoint(
    path,
    model,
    args,
    model_config,
    pipeline_version,
    forecast_step_minutes,
    grid,
    val_loss,
    epoch,
    metrics,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hyperparameters": vars(args),
            "input_length": args.input_length,
            "target_length": args.target_length,
            "model_config": model_config,
            "pipeline_version": pipeline_version,
            "forecast_step_minutes": forecast_step_minutes,
            "grid": grid,
            "model_architecture": args.architecture,
            "loss": "physics_evolution" if args.architecture == "phys-evolution" else "masked_mse",
            "metrics": {"best_val_loss": val_loss, "epoch": epoch, **metrics},
        },
        path,
    )


def _load_checkpoint(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", required=True, help="Comma-separated processed datasets")
    parser.add_argument("--architecture", choices=("phys-evolution", "convlstm"), default="phys-evolution")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--input-length", type=int, default=4)
    parser.add_argument("--target-length", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--max-motion-pixels", type=float, default=14.0)
    parser.add_argument("--max-evolution-per-step", type=float, default=0.08)
    parser.add_argument("--output-dir", default="models/registry")
    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance dry and echo sequences 50/50 when classes are available",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets, provenance, versions, time_steps, grids = _load_datasets(
        args.data_dirs.split(","),
        args.input_length,
        args.target_length,
    )
    if not datasets:
        print("No datasets found")
        return
    if len(versions) != 1 or len(time_steps) != 1 or len(grids) != 1:
        print(
            f"Datasets must share one pipeline, time step and grid: "
            f"{sorted(versions)}, {sorted(time_steps)}, {len(grids)} grids"
        )
        return
    pipeline_version = next(iter(versions))
    forecast_step_minutes = next(iter(time_steps))
    model_grid = next(iter(grids.values()))

    try:
        train_dataset, validation_dataset, test_dataset, split_report = build_temporal_datasets(
            datasets,
            args.val_split,
            args.test_split,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    weights = balanced_sample_weights(train_dataset) if args.balanced_sampling else None
    sampler = None
    if weights is not None:
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        print("Using 50/50 dry-vs-echo train sampling")
    elif args.balanced_sampling:
        print("Balanced sampling unavailable: both dry and echo classes are required")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size) if test_dataset else None

    model, criterion, model_config = _build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = os.path.join(args.output_dir, model_id)
    checkpoint_path = os.path.join(model_dir, "best_model.pt")
    os.makedirs(model_dir, exist_ok=True)
    metadata = {
        "type": "model",
        "model_id": model_id,
        "model_architecture": args.architecture,
        "model_config": model_config,
        "loss": "physics_evolution" if args.architecture == "phys-evolution" else "masked_mse",
        "sampling": "dry_echo_50_50" if sampler is not None else "natural",
        "split": split_report,
        "quality_gate_dataset": "independent_test" if test_loader else "validation_fallback",
        "hyperparameters": vars(args),
        "training_data": provenance,
        "pipeline_version": pipeline_version,
        "forecast_step_minutes": forecast_step_minutes,
        "grid": model_grid,
        "horizon_minutes": args.target_length * forecast_step_minutes,
        "status": "training",
        "timestamp_created": datetime.now().isoformat(),
    }
    save_metadata(model_dir, metadata)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_validation_metrics = None
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, args.architecture
            )
            val_loss = validate_epoch(
                model, validation_loader, criterion, device, args.architecture
            )
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch}/{args.epochs}: train={train_loss:.6f}, val={val_loss:.6f}")
            save_history(model_dir, history)
            metadata["current_epoch"] = epoch

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_validation_metrics = evaluate_model_quality(
                    model, validation_loader, device, args.architecture
                )
                metadata["metrics"] = {
                    "best_val_loss": val_loss,
                    "best_epoch": epoch,
                    **best_validation_metrics,
                }
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    args,
                    model_config,
                    pipeline_version,
                    forecast_step_minutes,
                    model_grid,
                    val_loss,
                    epoch,
                    best_validation_metrics,
                )
                print(
                    f"Validation baselines: persistence={best_validation_metrics['persistence_mse']:.6f}, "
                    f"block_motion={best_validation_metrics['block_motion_mse']:.6f}"
                )
            save_metadata(model_dir, metadata)

        if best_validation_metrics is None:
            raise RuntimeError("Training did not produce a valid checkpoint")

        checkpoint = _load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        quality_loader = test_loader or validation_loader
        quality_metrics = evaluate_model_quality(
            model, quality_loader, device, args.architecture
        )
        metadata["quality_gate_metrics"] = quality_metrics
        metadata["status"] = (
            "completed" if quality_metrics["quality_gate_passed"] else "rejected_quality_gate"
        )
        checkpoint["quality_gate_dataset"] = metadata["quality_gate_dataset"]
        checkpoint["quality_gate_metrics"] = quality_metrics
        torch.save(checkpoint, checkpoint_path)
        save_metadata(model_dir, metadata)
        print(
            f"Quality gate ({metadata['quality_gate_dataset']}): "
            f"model={quality_metrics['model_mse']:.6f}, "
            f"persistence={quality_metrics['persistence_mse']:.6f}, "
            f"block_motion={quality_metrics['block_motion_mse']:.6f}"
        )
        print(f"Training complete. Model saved to {model_dir}")
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        metadata["traceback"] = traceback.format_exc()
        save_metadata(model_dir, metadata)
        raise


if __name__ == "__main__":
    main()
