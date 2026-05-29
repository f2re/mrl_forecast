#!/usr/bin/env python3
"""
train_nowcasting_model.py
=========================

This module provides a minimal framework for training a precipitation nowcasting
model from meteorological radar data.  The primary target architecture is a
convolutional long short‑term memory (ConvLSTM) network, which has been
demonstrated by numerous studies to effectively capture the spatio‑temporal
dynamics of precipitation fields【16572649923328†L27-L33】.  The script also defines
a flexible PyTorch dataset for loading preprocessed radar sequences and
implements a simple training loop.

The training pipeline assumes that radar observations are available on a
regular grid and stored as NumPy arrays or netCDF files on disk.  For
production use you should integrate BUFR decoding (via eccodes) and
incorporate the recommended preprocessing steps – clipping, scaling,
sampling and sliding windows – described in the literature【64329450166814†L60-L80】.  For the
purposes of this example the dataset class expects that each file contains a
tensor of shape `(T, H, W)` where `T` is the number of time steps in the
sequence (for example, 8 frames representing two hours of history and two
hours of future at 15‑minute resolution).  The first half of the tensor
provides the input context and the second half the ground truth targets.

Example usage:
```
python train_nowcasting_model.py \
    --data-dir /path/to/preprocessed_sequences \
    --epochs 20 --batch-size 4 --lr 1e-4 --model convlstm
```

The script will save model checkpoints to `checkpoints/` and metrics to
`training_log.csv` in the working directory.  See the README.md in this
repository for further instructions on how to convert BUFR data to the
expected format.
"""

import argparse
import os
import pathlib
from datetime import datetime
import numpy as np
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from metadata_utils import save_metadata, load_metadata


class RadarSequenceDataset(Dataset):
    """PyTorch Dataset for radar sequences.

    Each file in the supplied directory must contain a NumPy array with shape
    `(sequence_length, height, width)`.  The first `input_length` frames of
    the array are used as input and the remaining `target_length` frames as
    targets.  Arrays are automatically scaled to the range [0, 1] based on
    their maximum value.  To incorporate additional preprocessing steps such
    as clipping and sampling, override the ``_preprocess`` method.
    """

    def __init__(self, data_dir: str, input_length: int, target_length: int):
        self.data_dir = pathlib.Path(data_dir)
        self.input_length = input_length
        self.target_length = target_length
        self.files = sorted([p for p in self.data_dir.iterdir() if p.suffix in ('.npy', '.npz')])
        if not self.files:
            raise ValueError(f"No .npy or .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _preprocess(self, array: np.ndarray) -> np.ndarray:
        # Clip negative or extreme values (e.g. clutter) and scale to [0,1]
        # Using 70 dBZ as the standard maximum for normalization
        MAX_DBZ = 70.0
        array = np.clip(array, a_min=0.0, a_max=MAX_DBZ)
        return array / MAX_DBZ

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        if path.suffix == '.npz':
            data = np.load(path)['arr_0']
        else:
            data = np.load(path)
        # Expect shape (T, H, W)
        data = self._preprocess(data)
        input_seq = data[: self.input_length]
        target_seq = data[self.input_length : self.input_length + self.target_length]
        # Add channel dimension
        input_tensor = torch.from_numpy(input_seq).unsqueeze(1).float()
        target_tensor = torch.from_numpy(target_seq).unsqueeze(1).float()
        return input_tensor, target_tensor


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell.

    Based on the formulation in Shi et al. (2015)【64329450166814†L283-L293】, each gate is a 2D
    convolution instead of a fully connected linear layer.  The cell maintains
    hidden and cell states of equal spatial dimensions to the input.
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )
        self.hidden_channels = hidden_channels

    def forward(
        self,
        x: torch.Tensor,
        h_cur: torch.Tensor,
        c_cur: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """A multi‑layer ConvLSTM network.

    Args:
        input_channels: Number of channels in the input frames (e.g. 1 for
            reflectivity).
        hidden_channels: List of channels for hidden states of each layer.
        kernel_size: Convolution kernel size for all layers.
        num_layers: Depth of the ConvLSTM (must match length of hidden_channels).
        output_steps: Number of frames to predict in the future.

    The network operates in an encoder–decoder fashion: the encoder ingests
    the input sequence and updates the hidden states, then the decoder
    autoregressively generates future frames by feeding its own outputs back
    into the network.
    """

    def __init__(self, input_channels: int, hidden_channels: List[int], kernel_size: int = 3, output_steps: int = 4):
        super().__init__()
        assert len(hidden_channels) > 0, "At least one hidden layer required"
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.output_steps = output_steps
        self.num_layers = len(hidden_channels)

        # Construct cells for each layer
        cells = []
        for i in range(self.num_layers):
            in_ch = self.input_channels if i == 0 else hidden_channels[i - 1]
            cells.append(ConvLSTMCell(in_ch, hidden_channels[i], kernel_size))
        self.cells = nn.ModuleList(cells)

        # Convolution to map the last hidden state to output frame
        self.output_conv = nn.Conv2d(hidden_channels[-1], input_channels, kernel_size=1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        # input_sequence: (batch, time_steps, channels, height, width)
        b, t, c, h, w = input_sequence.size()
        # Initialize hidden and cell states
        hidden = [torch.zeros(b, hc, h, w, device=input_sequence.device) for hc in self.hidden_channels]
        cell = [torch.zeros(b, hc, h, w, device=input_sequence.device) for hc in self.hidden_channels]

        # Encoder phase: iterate over input frames
        for step in range(t):
            x = input_sequence[:, step]
            for i, cell_layer in enumerate(self.cells):
                h_cur, c_cur = hidden[i], cell[i]
                h_next, c_next = cell_layer(x, h_cur, c_cur)
                hidden[i], cell[i] = h_next, c_next
                x = h_next  # output of this layer becomes input of next

        # Decoder phase: generate future frames
        outputs = []
        x = input_sequence[:, -1]  # start with last input frame
        for _ in range(self.output_steps):
            for i, cell_layer in enumerate(self.cells):
                h_cur, c_cur = hidden[i], cell[i]
                h_next, c_next = cell_layer(x, h_cur, c_cur)
                hidden[i], cell[i] = h_next, c_next
                x = h_next
            out_frame = self.output_conv(x)
            outputs.append(out_frame)
            # feed the output as next input
            x = out_frame
        # Stack along time dimension
        return torch.stack(outputs, dim=1)


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)  # shape (B, input_length, 1, H, W)
        targets = targets.to(device)  # shape (B, target_length, 1, H, W)
        optimizer.zero_grad()
        outputs = model(inputs)  # shape (B, target_length, 1, H, W)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train a precipitation nowcasting model.")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing preprocessed radar sequences (.npy or .npz)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--input-length', type=int, default=4, help='Number of past frames used as input')
    parser.add_argument('--target-length', type=int, default=4, help='Number of future frames to predict')
    parser.add_argument('--hidden-channels', type=str, default='16,32', help='Comma‑separated list of hidden channel sizes for each ConvLSTM layer')
    parser.add_argument('--kernel-size', type=int, default=3, help='Convolutional kernel size')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Directory to save model checkpoints and logs')
    args = parser.parse_args()

    # Prepare dataset and dataloaders
    dataset = RadarSequenceDataset(args.data_dir, args.input_length, args.target_length)
    # Split dataset into training and validation sets (80/20)
    indices = list(range(len(dataset)))
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Build model
    hidden_channels = [int(h) for h in args.hidden_channels.split(',')]
    model = ConvLSTM(input_channels=1, hidden_channels=hidden_channels, kernel_size=args.kernel_size, output_steps=args.target_length)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer and loss (MSE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup output directory
    dataset_meta = load_metadata(args.data_dir)
    station = dataset_meta.get('station', 'unknown') if dataset_meta else 'unknown'
    model_id = f"model_{station}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = os.path.join(args.output_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    metadata = {
        'type': 'model',
        'model_id': model_id,
        'dataset_path': args.data_dir,
        'station': station,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'hidden_channels': args.hidden_channels,
        'status': 'training',
        'best_val_loss': float('inf')
    }
    save_metadata(model_dir, metadata)

    log_path = os.path.join(model_dir, 'training_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,timestamp\n')

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        timestamp = datetime.utcnow().isoformat()
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        # Append to log
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{timestamp}\n")
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(model_dir, 'best_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'input_length': args.input_length,
                'target_length': args.target_length,
                'hidden_channels': hidden_channels,
            }, checkpoint_path)
            
            metadata['best_val_loss'] = val_loss
            metadata['best_epoch'] = epoch
            save_metadata(model_dir, metadata)

    metadata['status'] = 'completed'
    save_metadata(model_dir, metadata)
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()