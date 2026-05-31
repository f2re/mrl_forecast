import nexradaws
import os
import argparse
from datetime import datetime, timedelta
from metadata_utils import save_metadata

def download_nexrad_data(station, start_date, end_file_count=100, output_root='data/raw/archive'):
    """
    Downloads historical Level II NEXRAD data from AWS and saves metadata.
    """
    # Force us-east-1 region for NOAA open data bucket to avoid local config conflicts
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    os.environ['AWS_REGION'] = 'us-east-1'
    
    conn = nexradaws.NexradAwsInterface()
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    # Let's just download one day for start
    end = start + timedelta(days=1)
    
    # Создаем уникальную папку для сессии скачивания
    session_id = f"{station}_{start_date}_{datetime.now().strftime('%H%M%S')}"
    output_dir = os.path.join(output_root, session_id)
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        'type': 'raw_data',
        'station': station,
        'source': 'aws',
        'bucket': 'unidata-nexrad-level2',
        'region': 'us-east-1',
        'date': start_date,
        'requested_count': end_file_count,
        'status': 'downloading',
        'files': []
    }
    save_metadata(output_dir, metadata)
    
    print(f"Searching for scans for station {station} on {start_date}...")
    scans = conn.get_avail_scans(start.year, start.month, start.day, station)
    
    if not scans:
        print(f"No scans found for {station} on {start_date}")
        metadata['status'] = 'failed'
        metadata['error'] = 'No scans found'
        save_metadata(output_dir, metadata)
        return

    count_to_download = min(len(scans), end_file_count)
    print(f"Found {len(scans)} scans. Downloading first {count_to_download}...")
    
    results = conn.download(scans[:count_to_download], output_dir)
    
    metadata['status'] = 'completed'
    metadata['downloaded_count'] = len(results.success)
    metadata['files'] = [scan.filename for scan in results.success]
    save_metadata(output_dir, metadata)

    print(f"Downloaded {len(results.success)} files to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download NEXRAD Level II archive data from AWS.")
    parser.add_argument('--station', type=str, default='KOKX', help='Station ID (e.g. KOKX)')
    parser.add_argument('--date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--count', type=int, default=50, help='Number of files to download')
    parser.add_argument('--output', type=str, default='data/raw/archive', help='Output directory')
    args = parser.parse_args()
    
    download_nexrad_data(args.station, args.date, args.count, args.output)
