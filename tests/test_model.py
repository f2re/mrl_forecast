import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_nowcasting_model import ConvLSTM, RadarSequenceDataset

class TestModel(unittest.TestCase):
    def test_convlstm_shape(self):
        batch_size = 2
        input_length = 4
        target_length = 4
        h, w = 64, 64
        model = ConvLSTM(input_channels=1, hidden_channels=[16, 32], output_steps=target_length)
        
        # (batch, time, channels, h, w)
        input_tensor = torch.randn(batch_size, input_length, 1, h, w)
        output = model(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, target_length, 1, h, w))

if __name__ == '__main__':
    unittest.main()
