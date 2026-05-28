import nexradaws
import os
import argparse
from datetime import datetime, timedelta

def download_nexrad_data(station, start_date, end_file_count=100, output_dir='data/raw/archive'):
    """
    Downloads historical Level III NEXRAD data from AWS.
    """
    # Force us-east-1 region for NOAA open data bucket to avoid local config conflicts
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    os.environ['AWS_REGION'] = 'us-east-1'
    
    conn = nexradaws.NexradAwsInterface()
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    # Let's just download one day for start
    end = start + timedelta(days=1)
    
    print(f"Searching for scans for station {station} on {start_date}...")
    scans = conn.get_avail_scans(start.year, start.month, start.day, station)
    
    if not scans:
        print(f"No scans found for {station} on {start_date}")
        return

    print(f"Found {len(scans)} scans. Downloading first {min(len(scans), end_file_count)}...")
    
    os.makedirs(output_dir, exist_ok=True)
    results = conn.download(scans[:end_file_count], output_dir)
    
    print(f"Downloaded {len(results.success)} files to {output_dir}")
    for scan in results.success:
        print(f"  - {scan.filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download NEXRAD Level II archive data from AWS.")
    parser.add_argument('--station', type=str, default='KOKX', help='Station ID (e.g. KOKX)')
    parser.add_argument('--date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--count', type=int, default=50, help='Number of files to download')
    parser.add_argument('--output', type=str, default='data/raw/archive', help='Output directory')
    args = parser.parse_args()
    
    download_nexrad_data(args.station, args.date, args.count, args.output)
