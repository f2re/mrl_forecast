import os
import numpy as np
import pathlib
import argparse
from datetime import datetime
import pyart
from metadata_utils import save_metadata, load_metadata

def process_archive_directory(archive_dir, output_root, sequence_length=8, grid_shape=(1, 256, 256), grid_limits=((0, 10000), (-250000.0, 250000.0), (-250000.0, 250000.0))):
    """
    Scans a directory of NEXRAD Level II files, grids them, and saves as a structured dataset with metadata.
    """
    files = sorted([p for p in pathlib.Path(archive_dir).iterdir() if p.is_file() and not p.name.endswith('_MDM') and not p.name.endswith('.json')])
    
    if not files:
        print(f"No valid radar files found in {archive_dir}")
        return

    # Создаем уникальную папку для датасета
    source_meta = load_metadata(archive_dir)
    station = source_meta.get('station', 'unknown') if source_meta else 'unknown'
    dataset_id = f"dataset_{station}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(output_root, dataset_id)
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        'type': 'dataset',
        'source_path': archive_dir,
        'station': station,
        'sequence_length': sequence_length,
        'grid_shape': grid_shape,
        'status': 'processing',
        'sample_count': 0
    }
    save_metadata(output_dir, metadata)

    # Process all files into grids first
    print(f"Found {len(files)} files. Gridding data...")
    all_grids = []
    valid_files = []
    
    for f in files:
        try:
            radar = pyart.io.read(str(f))
            grid = pyart.map.grid_from_radars(
                (radar,),
                grid_shape=grid_shape,
                grid_limits=grid_limits,
                fields=['reflectivity'],
                weighting_function='Barnes2'
            )
            data = grid.fields['reflectivity']['data'][0]
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
        metadata['status'] = 'failed'
        metadata['error'] = 'Not enough valid files'
        save_metadata(output_dir, metadata)
        return
        
    print("Forming sequences...")
    sample_count = 0
    for i in range(len(all_grids) - sequence_length + 1):
        sequence = all_grids[i:i + sequence_length]
        sequence_array = np.stack(sequence, axis=0)
        target_file = valid_files[i + (sequence_length // 2) - 1]
        output_path = os.path.join(output_dir, f'seq_{i:04d}.npy')
        np.save(output_path, sequence_array)
        sample_count += 1

    metadata['status'] = 'completed'
    metadata['sample_count'] = sample_count
    save_metadata(output_dir, metadata)
    print(f"Saved {sample_count} sequences to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-dir', required=True, help="Directory with raw NEXRAD Level II files")
    parser.add_argument('--output-dir', default='data/processed_archive')
    parser.add_argument('--seq-len', type=int, default=8)
    args = parser.parse_args()
    
    process_archive_directory(args.archive_dir, args.output_dir, args.seq_len)
