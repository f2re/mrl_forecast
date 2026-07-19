"""Compact ConvLSTM baseline kept separate from the training workflow."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, input_tensor, state):
        hidden, cell = state
        gates = self.conv(torch.cat([input_tensor, hidden], dim=1))
        input_gate, forget_gate, output_gate, candidate = torch.split(
            gates,
            self.hidden_dim,
            dim=1,
        )
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        next_cell = forget_gate * cell + input_gate * candidate
        next_hidden = output_gate * torch.tanh(next_cell)
        return next_hidden, next_cell

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        shape = (batch_size, self.hidden_dim, height, width)
        return (
            torch.zeros(shape, device=self.conv.weight.device),
            torch.zeros(shape, device=self.conv.weight.device),
        )


class ConvLSTM(nn.Module):
    """Autoregressive ConvLSTM baseline with a stable legacy constructor."""

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
        del num_layers
        self.input_dim = input_channels if input_channels is not None else input_dim
        self.hidden_dim = hidden_channels if hidden_channels is not None else hidden_dim
        self.hidden_dim = self.hidden_dim or [16, 32]
        self.batch_first = batch_first
        self.output_steps = output_steps
        kernels = self._normalise_kernels(kernel_size, len(self.hidden_dim))

        cells = []
        for index, channels in enumerate(self.hidden_dim):
            current_input = self.input_dim if index == 0 else self.hidden_dim[index - 1]
            cells.append(ConvLSTMCell(current_input, channels, kernels[index], bias=bias))
        self.cells = nn.ModuleList(cells)
        self.output_conv = nn.Conv2d(self.hidden_dim[-1], self.input_dim, kernel_size=1)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        batch_size, sequence_length, _, height, width = input_tensor.shape
        states = hidden_state or [
            cell.init_hidden(batch_size, (height, width))
            for cell in self.cells
        ]

        current = input_tensor
        last_states = []
        for layer_index, cell in enumerate(self.cells):
            hidden, cell_state = states[layer_index]
            outputs = []
            for time_index in range(sequence_length):
                hidden, cell_state = cell(current[:, time_index], (hidden, cell_state))
                outputs.append(hidden)
            current = torch.stack(outputs, dim=1)
            last_states.append([hidden, cell_state])

        if self.output_steps is None:
            output = torch.sigmoid(
                self.output_conv(current.reshape(-1, self.hidden_dim[-1], height, width))
            )
            output = output.reshape(batch_size, sequence_length, self.input_dim, height, width)
        else:
            predictions = []
            current_hidden = current[:, -1]
            for _ in range(self.output_steps):
                predicted = torch.sigmoid(self.output_conv(current_hidden))
                predictions.append(predicted)
                layer_input = predicted
                for layer_index, cell in enumerate(self.cells):
                    hidden, cell_state = last_states[layer_index]
                    hidden, cell_state = cell(layer_input, (hidden, cell_state))
                    last_states[layer_index] = [hidden, cell_state]
                    layer_input = hidden
                current_hidden = layer_input
            output = torch.stack(predictions, dim=1)

        if not self.batch_first:
            output = output.permute(1, 0, 2, 3, 4)
        return output, last_states

    @staticmethod
    def _normalise_kernels(kernel_size, layers):
        if isinstance(kernel_size, int):
            kernels = [(kernel_size, kernel_size)] * layers
        elif isinstance(kernel_size, tuple):
            kernels = [kernel_size] * layers
        elif isinstance(kernel_size, list) and all(isinstance(item, int) for item in kernel_size):
            kernels = [(item, item) for item in kernel_size]
        elif isinstance(kernel_size, list) and all(isinstance(item, tuple) for item in kernel_size):
            kernels = kernel_size
        else:
            raise ValueError("kernel_size must be int, tuple or list of ints/tuples")
        if len(kernels) != layers:
            raise ValueError("kernel_size and hidden channel layer counts differ")
        return kernels
