#!/usr/bin/env python3
"""Train and validate the ConvLSTM radar nowcasting baseline."""

import argparse
import csv
import os
import traceback
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from config import FORECAST_STEP_MINUTES, MAX_DBZ
from datasets import RadarSequenceDataset
from forecast_quality import advection_forecast, is_uniform_forecast, threshold_metrics_by_lead_time
from losses import MaskedMSELoss, masked_mse
from metadata_utils import load_metadata, save_metadata
from radar_pipeline import PIPELINE_VERSION


def temporal_split_indices(sample_count: int, overlap_frames: int, val_fraction: float):
    """Split sliding-window samples chronologically and drop the overlap boundary."""
    if sample_count < 2:
        raise ValueError("At least two samples are required for train/validation split")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    boundary = int(sample_count * (1.0 - val_fraction))
    train_stop = max(boundary - overlap_frames, 0)
    train_indices = list(range(train_stop))
    validation_indices = list(range(boundary, sample_count))
    if not train_indices or not validation_indices:
        raise ValueError("Dataset is too small for a leakage-free temporal split")
    return train_indices, validation_indices


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined_conv = self.conv(torch.cat([input_tensor, h_cur], dim=1))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        shape = (batch_size, self.hidden_dim, height, width)
        return (
            torch.zeros(*shape, device=self.conv.weight.device),
            torch.zeros(*shape, device=self.conv.weight.device),
        )


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=None,
        kernel_size=(3, 3),
        num_layers=None,
        batch_first=True,
        bias=True,
        input_channels=None,
        hidden_channels=None,
        output_steps=None,
    ):
        super().__init__()
        self.input_dim = input_channels if input_channels is not None else input_dim
        self.hidden_dim = hidden_channels if hidden_channels is not None else hidden_dim
        if self.hidden_dim is None:
            self.hidden_dim = [16, 32]
        self.num_layers = len(self.hidden_dim)
        self.batch_first = batch_first
        self.bias = bias
        self.output_steps = output_steps
        self.kernel_size = self._check_kernel_size_consistency(kernel_size)
        self.kernel_size = self._extend_for_multilayer(self.kernel_size, self.num_layers)
        if not len(self.kernel_size) == len(self.hidden_dim) == self.num_layers:
            raise ValueError("Inconsistent list lengths")

        cells = []
        for index in range(self.num_layers):
            current_input_dim = self.input_dim if index == 0 else self.hidden_dim[index - 1]
            cells.append(
                ConvLSTMCell(
                    input_dim=current_input_dim,
                    hidden_dim=self.hidden_dim[index],
                    kernel_size=self.kernel_size[index],
                    bias=self.bias,
                )
            )
        self.cells = nn.ModuleList(cells)
        self.output_conv = nn.Conv2d(self.hidden_dim[-1], self.input_dim, kernel_size=1)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        batch_size, sequence_length, _, height, width = input_tensor.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=batch_size, image_size=(height, width))

        layer_output_list = []
        last_state_list = []
        current_layer_input = input_tensor
        for layer_index in range(self.num_layers):
            h_state, c_state = hidden_state[layer_index]
            output_inner = []
            for time_index in range(sequence_length):
                h_state, c_state = self.cells[layer_index](
                    input_tensor=current_layer_input[:, time_index],
                    cur_state=[h_state, c_state],
                )
                output_inner.append(h_state)
            layer_output = torch.stack(output_inner, dim=1)
            current_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h_state, c_state])

        if self.output_steps is not None:
            predictions = []
            current_h = layer_output_list[-1][:, -1]
            for _ in range(self.output_steps):
                predicted_frame = torch.sigmoid(self.output_conv(current_h))
                predictions.append(predicted_frame)
                previous_layer_h = predicted_frame
                for layer_index in range(self.num_layers):
                    h_state, c_state = last_state_list[layer_index]
                    h_state, c_state = self.cells[layer_index](
                        input_tensor=previous_layer_h,
                        cur_state=[h_state, c_state],
                    )
                    last_state_list[layer_index] = [h_state, c_state]
                    previous_layer_h = h_state
                current_h = previous_layer_h
            prediction_tensor = torch.stack(predictions, dim=1)
            if not self.batch_first:
                return prediction_tensor.permute(1, 0, 2, 3, 4), last_state_list
            return prediction_tensor, last_state_list

        final_output = torch.sigmoid(
            self.output_conv(layer_output_list[-1].view(-1, self.hidden_dim[-1], height, width))
        )
        final_output = final_output.view(
            batch_size,
            sequence_length,
            self.input_dim,
            height,
            width,
        )
        if not self.batch_first:
            return final_output.permute(1, 0, 2, 3, 4), last_state_list
        return final_output, last_state_list

    def _init_hidden(self, batch_size, image_size):
        return [cell.init_hidden(batch_size, image_size) for cell in self.cells]

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if isinstance(kernel_size, int):
            return (kernel_size, kernel_size)
        if isinstance(kernel_size, tuple):
            return kernel_size
        if isinstance(kernel_size, list):
            if all(isinstance(element, tuple) for element in kernel_size):
                return kernel_size
            if all(isinstance(element, int) for element in kernel_size):
                return [(element, element) for element in kernel_size]
        raise ValueError("kernel_size must be int, tuple or list of tuples/ints")

    @staticmethod
    def _extend_for_multilayer(parameter, number_of_layers):
        if not isinstance(parameter, list):
            parameter = [parameter] * number_of_layers
        return parameter


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y, _x_mask, y_mask in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output, y, y_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y, _x_mask, y_mask in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)
            output, _ = model(x)
            total_loss += criterion(output, y, y_mask).item()
    return total_loss / len(dataloader)


def quality_gate_passes(metrics: Dict[str, float]) -> bool:
    """Only publish models that beat persistence and avoid uniform precipitation."""
    return (
        metrics["model_mse"] < metrics["persistence_mse"]
        and metrics["model_mse"] < metrics.get("advection_mse", float("inf"))
        and not metrics["uniform_field_anomaly"]
    )


def _masked_numpy_mse(prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray):
    mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(mask):
        return None
    return float(np.mean((np.asarray(prediction)[mask] - np.asarray(target)[mask]) ** 2))


def evaluate_model_quality(model, dataloader, device):
    """Compare the trained model with baselines on valid validation pixels."""
    model.eval()
    model_losses = []
    persistence_losses = []
    advection_losses = []
    uniform_field_anomaly = False
    all_forecasts = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for x, y, _x_mask, y_mask in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)
            output, _ = model(x)
            persistence = x[:, -1:, ...].expand_as(y)
            model_losses.append(masked_mse(output, y, y_mask).item())
            persistence_losses.append(masked_mse(persistence, y, y_mask).item())

            output_values = output.cpu().numpy()
            target_values = y.cpu().numpy()
            history_values = x.cpu().numpy()
            mask_values = y_mask.cpu().numpy().astype(bool)
            for history, forecast, target, target_mask in zip(
                history_values,
                output_values,
                target_values,
                mask_values,
            ):
                valid = target_mask[:, 0]
                values_dbz = forecast[:, 0] * MAX_DBZ
                checked_values = np.where(valid, values_dbz, 0.0)
                uniform_field_anomaly = uniform_field_anomaly or is_uniform_forecast(checked_values)
                advection = advection_forecast(history[:, 0] * MAX_DBZ, forecast.shape[0]) / MAX_DBZ
                advection_loss = _masked_numpy_mse(advection, target[:, 0], valid)
                if advection_loss is not None:
                    advection_losses.append(advection_loss)
                all_forecasts.append(values_dbz)
                all_targets.append(target[:, 0] * MAX_DBZ)
                all_masks.append(valid)

    metrics = {
        "model_mse": float(np.mean(model_losses)),
        "persistence_mse": float(np.mean(persistence_losses)),
        "advection_mse": float(np.mean(advection_losses)) if advection_losses else float("inf"),
        "uniform_field_anomaly": bool(uniform_field_anomaly),
        "threshold_metrics": threshold_metrics_by_lead_time(
            np.stack(all_forecasts, axis=0),
            np.stack(all_targets, axis=0),
            valid_mask=np.stack(all_masks, axis=0),
        ),
    }
    metrics["quality_gate_passed"] = quality_gate_passes(metrics)
    return metrics


def build_temporal_datasets(datasets, val_split):
    """Create leakage-resistant temporal subsets for every source dataset."""
    train_parts = []
    validation_parts = []
    for dataset in datasets:
        overlap_frames = dataset.input_length + dataset.target_length - 1
        train_indices, validation_indices = temporal_split_indices(
            len(dataset),
            overlap_frames,
            val_split,
        )
        train_parts.append(Subset(dataset, train_indices))
        validation_parts.append(Subset(dataset, validation_indices))
    return ConcatDataset(train_parts), ConcatDataset(validation_parts)


def save_plots(model_dir, history):
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, "learning_curve.png"))
    plt.close()


def save_history(model_dir, history):
    """Persist epoch metrics incrementally so interrupted runs remain inspectable."""
    path = os.path.join(model_dir, "history.csv")
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, (train_loss, val_loss) in enumerate(
            zip(history["train_loss"], history["val_loss"]),
            start=1,
        ):
            writer.writerow([epoch, train_loss, val_loss])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", required=True, help="Comma-separated processed dataset directories")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--input-length", type=int, default=4)
    parser.add_argument("--target-length", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--output-dir", default="models/registry")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_datasets = []
    provenance_data = []
    pipeline_versions = set()
    for data_dir in args.data_dirs.split(","):
        if not os.path.exists(data_dir):
            continue
        metadata = load_metadata(data_dir)
        pipeline_version = (metadata or {}).get("pipeline", {}).get("pipeline_version")
        if (metadata or {}).get("status") != "completed" or not pipeline_version:
            print(f"Skipping incompatible dataset {data_dir}: completed pipeline metadata is required")
            continue
        pipeline_versions.add(pipeline_version)
        dataset = RadarSequenceDataset(data_dir, args.input_length, args.target_length)
        all_datasets.append(dataset)
        provenance_data.append(
            {
                "dataset_id": os.path.basename(data_dir),
                "station": metadata.get("station", "unknown"),
                "samples": len(dataset),
                "pipeline_version": pipeline_version,
                "time_step_minutes": metadata.get("pipeline", {}).get("time_step_minutes"),
                "sample_format": metadata.get("sample_format", "legacy-npy"),
            }
        )
        print(f"Loaded dataset from {data_dir} ({len(dataset)} samples)")

    if not all_datasets:
        print("No datasets found!")
        return
    if pipeline_versions != {PIPELINE_VERSION}:
        print(f"Incompatible pipeline versions: {sorted(pipeline_versions)}")
        return

    full_dataset = ConcatDataset(all_datasets)
    if len(full_dataset) == 0:
        print("Error: Total number of samples is 0. Cannot proceed with training")
        return

    try:
        train_dataset, validation_dataset = build_temporal_datasets(all_datasets, args.val_split)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    hidden_channels = args.hidden_channels
    model = ConvLSTM(
        input_dim=1,
        hidden_dim=[hidden_channels, hidden_channels],
        kernel_size=(3, 3),
        num_layers=2,
        batch_first=True,
        output_steps=args.target_length,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = MaskedMSELoss()

    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = os.path.join(args.output_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    metadata = {
        "type": "model",
        "model_id": model_id,
        "model_architecture": "convlstm_baseline",
        "loss": "masked_mse",
        "hyperparameters": vars(args),
        "training_data": provenance_data,
        "pipeline_version": PIPELINE_VERSION,
        "forecast_step_minutes": FORECAST_STEP_MINUTES,
        "horizon_minutes": args.target_length * FORECAST_STEP_MINUTES,
        "status": "training",
        "timestamp_created": datetime.now().isoformat(),
    }
    save_metadata(model_dir, metadata)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_quality_metrics = None

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            validation_loss = validate_epoch(model, validation_loader, criterion, device)
            quality_metrics = evaluate_model_quality(model, validation_loader, device)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(validation_loss)
            print(
                f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.6f}, "
                f"val_loss={validation_loss:.6f}, "
                f"persistence_mse={quality_metrics['persistence_mse']:.6f}"
            )
            save_plots(model_dir, history)
            save_history(model_dir, history)
            metadata["current_epoch"] = epoch

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_quality_metrics = quality_metrics
                checkpoint_path = os.path.join(model_dir, "best_model.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "hyperparameters": vars(args),
                        "input_length": args.input_length,
                        "target_length": args.target_length,
                        "hidden_channels": [hidden_channels, hidden_channels],
                        "pipeline_version": PIPELINE_VERSION,
                        "forecast_step_minutes": FORECAST_STEP_MINUTES,
                        "model_architecture": "convlstm_baseline",
                        "loss": "masked_mse",
                        "metrics": {
                            "best_val_loss": best_val_loss,
                            "epoch": epoch,
                            **quality_metrics,
                        },
                    },
                    checkpoint_path,
                )
                metadata["metrics"] = {
                    "best_val_loss": best_val_loss,
                    "best_epoch": epoch,
                    **quality_metrics,
                }
            save_metadata(model_dir, metadata)

        if best_quality_metrics is None:
            raise RuntimeError("Training did not produce a valid checkpoint")
        metadata["status"] = (
            "completed" if best_quality_metrics["quality_gate_passed"] else "rejected_quality_gate"
        )
        save_metadata(model_dir, metadata)
        print(f"Training complete. Model saved to {model_dir}")
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        metadata["traceback"] = traceback.format_exc()
        save_metadata(model_dir, metadata)
        raise


if __name__ == "__main__":
    main()
