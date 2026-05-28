import numpy as np
import os
import argparse

def generate_dummy_data(output_dir, num_samples=10, sequence_length=8, height=128, width=128):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        # Create a simple moving blob
        data = np.zeros((sequence_length, height, width))
        start_x = np.random.randint(20, height - 20)
        start_y = np.random.randint(20, width - 20)
        vx = np.random.randint(-2, 3)
        vy = np.random.randint(-2, 3)
        
        for t in range(sequence_length):
            x = start_x + vx * t
            y = start_y + vy * t
            # Gaussian blob
            yy, xx = np.mgrid[0:height, 0:width]
            blob = np.exp(-((xx - x)**2 + (yy - y)**2) / 100.0)
            data[t] = blob * np.random.uniform(20, 50) # dBZ-like values
            
        np.save(os.path.join(output_dir, f'sample_{i}.npy'), data)
    print(f"Generated {num_samples} dummy samples in {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='data/processed')
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()
    generate_dummy_data(args.output_dir, args.num_samples)
