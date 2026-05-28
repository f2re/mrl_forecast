import os
import numpy as np
import pathlib
import argparse
from datetime import datetime
import pyart

def process_archive_directory(archive_dir, output_dir, sequence_length=8, grid_shape=(1, 256, 256), grid_limits=((0, 10000), (-250000.0, 250000.0), (-250000.0, 250000.0))):
    """
    Scans a directory of NEXRAD Level II files, grids them using Py-ART, 
    and groups them into contiguous sequences.
    """
    files = sorted([p for p in pathlib.Path(archive_dir).iterdir() if p.is_file() and not p.name.endswith('_MDM')])
    
    if not files:
        print(f"No valid radar files found in {archive_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files into grids first (or on the fly)
    print(f"Found {len(files)} files. Gridding data...")
    all_grids = []
    valid_files = []
    
    for f in files:
        try:
            radar = pyart.io.read(str(f))
            # Grid the data to Cartesian (250km range, 256x256 grid)
            grid = pyart.map.grid_from_radars(
                (radar,),
                grid_shape=grid_shape,
                grid_limits=grid_limits,
                fields=['reflectivity'],
                weighting_function='Barnes2' # Fast and reasonable
            )
            # Extract the 2D array for reflectivity
            data = grid.fields['reflectivity']['data'][0] # Take lowest z level
            
            # Mask handling
            if hasattr(data, 'mask'):
                data = data.filled(0)
            data = np.nan_to_num(data)
            data = np.clip(data, 0, 70)
            
            all_grids.append(data)
            valid_files.append(f)
            print(f"Successfully gridded {f.name}")
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
            
    if len(all_grids) < sequence_length:
        print("Not enough valid files to form a sequence.")
        return
        
    print("Forming sequences...")
    # Simple sliding window approach. In reality, we should check timestamps 
    # to ensure they are continuous (~10 min apart max).
    for i in range(len(all_grids) - sequence_length + 1):
        sequence = all_grids[i:i + sequence_length]
        sequence_array = np.stack(sequence, axis=0)
        
        # Use timestamp of the last input frame as the ID
        target_file = valid_files[i + (sequence_length // 2) - 1]
        ts = target_file.name.split('_')[1] # e.g. KOKX20240520_002009_V06 -> 002009
        
        output_path = os.path.join(output_dir, f'seq_{target_file.name[:15]}_{i:04d}.npy')
        np.save(output_path, sequence_array)
        print(f"Saved {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-dir', required=True, help="Directory with raw NEXRAD Level II files")
    parser.add_argument('--output-dir', default='data/processed_archive')
    parser.add_argument('--seq-len', type=int, default=8)
    args = parser.parse_args()
    
    process_archive_directory(args.archive_dir, args.output_dir, args.seq_len)
