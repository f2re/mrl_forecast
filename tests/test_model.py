import unittest
import torch
import sys
import os
import tempfile
from pathlib import Path

import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_nowcasting_model import ConvLSTM, RadarSequenceDataset, quality_gate_passes

class TestModel(unittest.TestCase):
    def test_convlstm_shape(self):
        batch_size = 2
        input_length = 4
        target_length = 4
        h, w = 64, 64
        model = ConvLSTM(input_channels=1, hidden_channels=[16, 1], output_steps=target_length)
        
        # (batch, time, channels, h, w)
        input_tensor = torch.randn(batch_size, input_length, 1, h, w)
        output, _ = model(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, target_length, 1, h, w))

    def test_convlstm_predictions_are_normalized(self):
        model = ConvLSTM(input_channels=1, hidden_channels=[4, 4], output_steps=2)
        output, _ = model(torch.randn(1, 3, 1, 8, 8))

        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))

    def test_dataset_clips_reflectivity_before_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            data = np.zeros((8, 4, 4), dtype=np.float32)
            data[0, 0, 0] = -10.0
            data[0, 0, 1] = 100.0
            np.save(Path(directory) / "seq_0000.npy", data)

            history, _ = RadarSequenceDataset(directory)[0]

            self.assertEqual(history[0, 0, 0, 0].item(), 0.0)
            self.assertEqual(history[0, 0, 0, 1].item(), 1.0)

    def test_quality_gate_rejects_model_worse_than_persistence(self):
        metrics = {
            "model_mse": 0.2,
            "persistence_mse": 0.1,
            "uniform_field_anomaly": False,
        }

        self.assertFalse(quality_gate_passes(metrics))

if __name__ == '__main__':
    unittest.main()
