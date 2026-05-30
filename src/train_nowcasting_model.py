#!/usr/bin/env python3
"""
train_nowcasting_model.py (Pro Version)
=======================================
Advanced multi-dataset training for precipitation nowcasting.
"""
import argparse
import os
import pathlib
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from metadata_utils import save_metadata, load_metadata

class RadarSequenceDataset(Dataset):
    def __init__(self, data_dir: str, input_length: int = 4, target_length: int = 4):
        self.data_dir = pathlib.Path(data_dir)
        self.files = sorted(list(self.data_dir.glob('*.npy')))
        self.input_length = input_length
        self.target_length = target_length
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load [T, H, W]
        data = np.load(self.files[idx]).astype(np.float32)
        # Convert to [T, C, H, W] where C=1
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
            self.hidden_dim = [16, 1]
            
        self.num_layers = num_layers if num_layers is not None else len(self.hidden_dim)
        self.batch_first = batch_first
        self.bias = bias
        self.output_steps = output_steps
        
        self._check_kernel_size_consistency(kernel_size)
        self.kernel_size = self._extend_for_multilayer(kernel_size, self.num_layers)
        self.hidden_dim = self._extend_for_multilayer(self.hidden_dim, self.num_layers)
        
        if not len(self.kernel_size) == len(self.hidden_dim) == self.num_layers:
            raise ValueError('Inconsistent list lengths.')
            
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i], bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

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
                h_state, c_state = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h_state, c_state])
                output_inner.append(h_state)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h_state, c_state])
            
        # Prediction Phase (Autoregressive)
        # If output_steps is defined, we generate future steps
        if self.output_steps is not None:
            predictions = []
            # We take the last output of the encoding phase as our first input for prediction
            # Or use the last frame of the input tensor if input_dim matches output_dim
            # Here we assume last layer output is what we want to predict
            current_input = layer_output_list[-1][:, -1, :, :, :]
            
            for _ in range(self.output_steps):
                # Feed it back through all layers
                prev_h = current_input
                for layer_idx in range(self.num_layers):
                    h_s, c_s = last_state_list[layer_idx]
                    h_s, c_s = self.cell_list[layer_idx](input_tensor=prev_h, cur_state=[h_s, c_s])
                    last_state_list[layer_idx] = [h_s, c_s]
                    prev_h = h_s
                
                # The output of the last layer is our prediction for this step
                predictions.append(prev_h)
                current_input = prev_h
                
            prediction_tensor = torch.stack(predictions, dim=1)
            if not self.batch_first:
                return prediction_tensor.permute(1, 0, 2, 3, 4), last_state_list
            return prediction_tensor, last_state_list

        if not self.batch_first:
            return layer_output_list[-1].permute(1, 0, 2, 3, 4), last_state_list
        return layer_output_list[-1], last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

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
    for d in data_dirs:
        if os.path.exists(d):
            meta = load_metadata(d)
            ds = RadarSequenceDataset(d, args.input_length, args.target_length)
            all_datasets.append(ds)
            provenance_data.append({
                "dataset_id": os.path.basename(d),
                "station": meta.get('station', 'unknown') if meta else 'unknown',
                "samples": len(ds)
            })
            print(f"Loaded dataset from {d} ({len(ds)} samples)")

    if not all_datasets:
        print("No datasets found!")
        return

    full_dataset = ConcatDataset(all_datasets)
    num_val = int(len(full_dataset) * args.val_split)
    num_train = len(full_dataset) - num_val
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [num_train, num_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Initialize model
    hidden_channels = args.hidden_channels
    model = ConvLSTM(
        input_dim=1, 
        hidden_dim=[hidden_channels, 1], 
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
        'status': 'training',
        'timestamp_created': datetime.now().isoformat()
    }
    save_metadata(model_dir, metadata)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(model_dir, 'best_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': vars(args),
                'metrics': {'best_val_loss': best_val_loss, 'epoch': epoch}
            }, checkpoint_path)
            
            metadata['metrics'] = {'best_val_loss': best_val_loss, 'best_epoch': epoch}
            save_metadata(model_dir, metadata)

    metadata['status'] = 'completed'
    save_metadata(model_dir, metadata)
    save_plots(model_dir, history)
    print(f"Training complete. Model saved to {model_dir}")

if __name__ == '__main__':
    main()
