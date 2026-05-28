import os
import numpy as np
import pathlib
import argparse
from datetime import datetime
from bufr_decoder import MRLBufrDecoder

def make_dataset(bufr_dir, output_dir, sequence_length=8):
    """
    Scans bufr_dir for BUFR files, decodes them, and groups into sequences.
    """
    decoder = MRLBufrDecoder()
    bufr_files = sorted([p for p in pathlib.Path(bufr_dir).iterdir() if p.is_file()])
    
    if not bufr_files:
        print(f"No files found in {bufr_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Simple sliding window approach
    for i in range(len(bufr_files) - sequence_length + 1):
        sequence = []
        for j in range(sequence_length):
            file_path = bufr_files[i + j]
            grid = decoder.decode(str(file_path))
            sequence.append(grid)
        
        sequence_array = np.stack(sequence, axis=0)
        output_path = os.path.join(output_dir, f'sequence_{i:04d}.npy')
        np.save(output_path, sequence_array)
        print(f"Saved {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bufr-dir', required=True)
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--seq-len', type=int, default=8)
    args = parser.parse_args()
    
    make_dataset(args.bufr_dir, args.output_dir, args.seq_len)
