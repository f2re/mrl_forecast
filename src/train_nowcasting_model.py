#!/usr/bin/env python3
"""
train_nowcasting_model.py (Pro Version)
=======================================
Advanced multi-dataset training for precipitation nowcasting.
"""
import argparse
import csv
import os
import pathlib
import traceback
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from metadata_utils import save_metadata, load_metadata
from forecast_quality import advection_forecast, is_uniform_forecast, threshold_metrics_by_lead_time
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

class RadarSequenceDataset(Dataset):
    def __init__(self, data_dir: str, input_length: int = 4, target_length: int = 4):
        self.data_dir = pathlib.Path(data_dir)
        self.files = sorted(list(self.data_dir.glob('*.npy')))
        self.input_length = input_length
        self.target_length = target_length

        if len(self.files) > 0:
            # Check the first file to verify sequence length
            test_data = np.load(self.files[0])
            actual_len = test_data.shape[0]
            if actual_len < (input_length + target_length):
                raise ValueError(f"Dataset sequences (len={actual_len}) are shorter than requested input+target ({input_length}+{target_length})")

    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        # Load [T, H, W]
        data = np.load(self.files[idx]).astype(np.float32)
        # Convert to [T, C, H, W] where C=1
        data = np.clip(data, 0.0, 70.0)
        data = data[:, np.newaxis, :, :]
        # Normalize (0-70 dBZ -> 0-1)
        data = data / 70.0
        
        # Split history and target
        x = data[:self.input_length]
        y = data[self.input_length : self.input_length + self.target_length]
        return torch.from_numpy(x), torch.from_numpy(y)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
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
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=None, kernel_size=(3, 3), num_layers=None, batch_first=True, bias=True,
                 input_channels=None, hidden_channels=None, output_steps=None):
        super(ConvLSTM, self).__init__()
        
        # Flexibility for different arg names
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
            raise ValueError('Inconsistent list lengths.')
            
        cells = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cells.append(ConvLSTMCell(input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i], bias=self.bias))
        self.cells = nn.ModuleList(cells)
        
        # Final convolution to map hidden state back to input dimension
        self.output_conv = nn.Conv2d(in_channels=self.hidden_dim[-1], out_channels=self.input_dim, kernel_size=1)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            
        b, t, _, h, w = input_tensor.size()
        
        # Initialize hidden state
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
            
        layer_output_list = []
        last_state_list = []
        
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        # Encoding/Processing Phase
        for layer_idx in range(self.num_layers):
            h_state, c_state = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h_state, c_state = self.cells[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h_state, c_state])
                output_inner.append(h_state)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h_state, c_state])
            
        # Prediction Phase (Autoregressive)
        if self.output_steps is not None:
            predictions = []
            # We take the last hidden state and output to start prediction
            current_h = layer_output_list[-1][:, -1, :, :, :]
            
            for _ in range(self.output_steps):
                # Apply output_conv to the last hidden state to get the predicted frame
                # which will be used as input for the next step
                pred_frame = torch.sigmoid(self.output_conv(current_h))
                predictions.append(pred_frame)
                
                # Feed the predicted frame back through all layers
                prev_layer_h = pred_frame
                for layer_idx in range(self.num_layers):
                    h_s, c_s = last_state_list[layer_idx]
                    h_s, c_s = self.cells[layer_idx](input_tensor=prev_layer_h, cur_state=[h_s, c_s])
                    last_state_list[layer_idx] = [h_s, c_s]
                    prev_layer_h = h_s
                
                current_h = prev_layer_h
                
            prediction_tensor = torch.stack(predictions, dim=1)
            if not self.batch_first:
                return prediction_tensor.permute(1, 0, 2, 3, 4), last_state_list
            return prediction_tensor, last_state_list

        # If not predicting steps, apply output_conv to the entire sequence
        final_output = torch.sigmoid(
            self.output_conv(layer_output_list[-1].view(-1, self.hidden_dim[-1], h, w))
        )
        final_output = final_output.view(b, seq_len, self.input_dim, h, w)
        
        if not self.batch_first:
            return final_output.permute(1, 0, 2, 3, 4), last_state_list
        return final_output, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cells[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if isinstance(kernel_size, int):
            return (kernel_size, kernel_size)
        if isinstance(kernel_size, tuple):
            return kernel_size
        if isinstance(kernel_size, list):
            if all(isinstance(elem, tuple) for elem in kernel_size):
                return kernel_size
            if all(isinstance(elem, int) for elem in kernel_size):
                return [(e, e) for e in kernel_size]
        raise ValueError('`kernel_size` must be int, tuple or list of tuples/ints')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def quality_gate_passes(metrics: Dict[str, float]) -> bool:
    """Only publish models that beat persistence and avoid uniform precipitation."""
    return (
        metrics["model_mse"] < metrics["persistence_mse"]
        and metrics["model_mse"] < metrics.get("advection_mse", float("inf"))
        and not metrics["uniform_field_anomaly"]
    )


def evaluate_model_quality(model, dataloader, device):
    """Compare the trained model with persistence on leakage-free validation data."""
    model.eval()
    model_losses = []
    persistence_losses = []
    advection_losses = []
    uniform_field_anomaly = False
    all_forecasts = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            persistence = x[:, -1:, ...].expand_as(y)
            model_losses.append(torch.mean((output - y) ** 2).item())
            persistence_losses.append(torch.mean((persistence - y) ** 2).item())
            output_values = output.cpu().numpy()
            target_values = y.cpu().numpy()
            history_values = x.cpu().numpy()
            for history, forecast, target in zip(history_values, output_values, target_values):
                values_dbz = forecast[:, 0] * 70.0
                uniform_field_anomaly = uniform_field_anomaly or is_uniform_forecast(values_dbz)
                advection = advection_forecast(history[:, 0] * 70.0, forecast.shape[0]) / 70.0
                advection_losses.append(float(np.mean((advection - target[:, 0]) ** 2)))
                all_forecasts.append(values_dbz)
                all_targets.append(target[:, 0] * 70.0)
    metrics = {
        "model_mse": float(np.mean(model_losses)),
        "persistence_mse": float(np.mean(persistence_losses)),
        "advection_mse": float(np.mean(advection_losses)),
        "uniform_field_anomaly": bool(uniform_field_anomaly),
        "threshold_metrics": threshold_metrics_by_lead_time(
            np.stack(all_forecasts, axis=0),
            np.stack(all_targets, axis=0),
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
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, 'learning_curve.png'))
    plt.close()


def save_history(model_dir, history):
    """Persist epoch metrics incrementally so interrupted runs remain inspectable."""
    path = os.path.join(model_dir, 'history.csv')
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
        for epoch, (train_loss, val_loss) in enumerate(
            zip(history['train_loss'], history['val_loss']),
            start=1,
        ):
            writer.writerow([epoch, train_loss, val_loss])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dirs', required=True, help="Comma-separated directories with processed datasets")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--input-length', type=int, default=4)
    parser.add_argument('--target-length', type=int, default=4)
    parser.add_argument('--hidden-channels', type=int, default=32)
    parser.add_argument('--output-dir', default='models/registry')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load multiple datasets
    data_dirs = args.data_dirs.split(',')
    all_datasets = []
    provenance_data = []
    pipeline_versions = set()
    for d in data_dirs:
        if os.path.exists(d):
            meta = load_metadata(d)
            pipeline_version = (meta or {}).get('pipeline', {}).get('pipeline_version')
            if (meta or {}).get('status') != 'completed' or not pipeline_version:
                print(f"Skipping incompatible dataset {d}: completed pipeline metadata is required")
                continue
            pipeline_versions.add(pipeline_version)
            ds = RadarSequenceDataset(d, args.input_length, args.target_length)
            all_datasets.append(ds)
            provenance_data.append({
                "dataset_id": os.path.basename(d),
                "station": meta.get('station', 'unknown') if meta else 'unknown',
                "samples": len(ds),
                "pipeline_version": pipeline_version,
            })
            print(f"Loaded dataset from {d} ({len(ds)} samples)")

    if not all_datasets:
        print("No datasets found!")
        return
    if pipeline_versions != {PIPELINE_VERSION}:
        print(f"Incompatible pipeline versions: {sorted(pipeline_versions)}")
        return

    full_dataset = ConcatDataset(all_datasets)
    total_samples = len(full_dataset)
    
    if total_samples == 0:
        print("Error: Total number of samples is 0. Cannot proceed with training.")
        return

    try:
        train_ds, val_ds = build_temporal_datasets(all_datasets, args.val_split)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Initialize model
    hidden_channels = args.hidden_channels
    model = ConvLSTM(
        input_dim=1, 
        hidden_dim=[hidden_channels, hidden_channels],
        kernel_size=(3, 3), 
        num_layers=2, 
        batch_first=True,
        output_steps=args.target_length
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Setup output
    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = os.path.join(args.output_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    metadata = {
        'type': 'model',
        'model_id': model_id,
        'hyperparameters': vars(args),
        'training_data': provenance_data,
        'pipeline_version': PIPELINE_VERSION,
        'status': 'training',
        'timestamp_created': datetime.now().isoformat()
    }
    save_metadata(model_dir, metadata)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate_epoch(model, val_loader, criterion, device)
            quality_metrics = evaluate_model_quality(model, val_loader, device)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            print(
                f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}, persistence_mse={quality_metrics['persistence_mse']:.6f}"
            )
            save_plots(model_dir, history)
            save_history(model_dir, history)
            metadata['current_epoch'] = epoch
            metadata['metrics'] = {
                'best_val_loss': min(best_val_loss, val_loss),
                'best_epoch': metadata.get('metrics', {}).get('best_epoch', epoch),
                **quality_metrics,
            }
            save_metadata(model_dir, metadata)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(model_dir, 'best_model.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'hyperparameters': vars(args),
                    'input_length': args.input_length,
                    'target_length': args.target_length,
                    'hidden_channels': [hidden_channels, hidden_channels],
                    'pipeline_version': PIPELINE_VERSION,
                    'metrics': {'best_val_loss': best_val_loss, 'epoch': epoch, **quality_metrics},
                }, checkpoint_path)
                metadata['metrics'] = {
                    'best_val_loss': best_val_loss,
                    'best_epoch': epoch,
                    **quality_metrics,
                }
                save_metadata(model_dir, metadata)
        metadata['status'] = (
            'completed'
            if metadata['metrics']['quality_gate_passed']
            else 'rejected_quality_gate'
        )
        save_metadata(model_dir, metadata)
        print(f"Training complete. Model saved to {model_dir}")
    except Exception as exc:
        metadata['status'] = 'failed'
        metadata['error'] = str(exc)
        metadata['traceback'] = traceback.format_exc()
        save_metadata(model_dir, metadata)
        raise

if __name__ == '__main__':
    main()
