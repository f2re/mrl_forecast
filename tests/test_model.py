import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from train_nowcasting_model import ConvLSTM, RadarSequenceDataset, quality_gate_passes


class TestModel(unittest.TestCase):
    def test_convlstm_shape(self):
        batch_size = 2
        input_length = 4
        target_length = 4
        height, width = 64, 64
        model = ConvLSTM(input_channels=1, hidden_channels=[16, 1], output_steps=target_length)

        input_tensor = torch.randn(batch_size, input_length, 1, height, width)
        output, _ = model(input_tensor)

        self.assertEqual(output.shape, (batch_size, target_length, 1, height, width))

    def test_convlstm_predictions_are_normalized(self):
        model = ConvLSTM(input_channels=1, hidden_channels=[4, 4], output_steps=2)
        output, _ = model(torch.randn(1, 3, 1, 8, 8))

        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))

    def test_dataset_clips_reflectivity_and_preserves_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            data = np.zeros((8, 4, 4), dtype=np.float32)
            valid_mask = np.ones_like(data, dtype=bool)
            data[0, 0, 0] = -10.0
            data[0, 0, 1] = 100.0
            valid_mask[0, 0, 2] = False
            np.savez_compressed(
                Path(directory) / "seq_0000.npz",
                reflectivity=data,
                valid_mask=valid_mask,
            )

            history, _target, history_mask, _target_mask = RadarSequenceDataset(directory)[0]

            self.assertEqual(history[0, 0, 0, 0].item(), 0.0)
            self.assertEqual(history[0, 0, 0, 1].item(), 1.0)
            self.assertEqual(history_mask[0, 0, 0, 2].item(), 0.0)

    def test_quality_gate_rejects_model_worse_than_persistence(self):
        metrics = {
            "model_mse": 0.2,
            "persistence_mse": 0.1,
            "uniform_field_anomaly": False,
        }

        self.assertFalse(quality_gate_passes(metrics))


if __name__ == "__main__":
    unittest.main()
