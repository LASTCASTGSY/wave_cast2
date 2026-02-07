#!/usr/bin/env python3
"""
Preprocessing script for NDBC buoy data (Batch Version).
Converts raw NDBC text files to processed pickle files for training.

Usage:
    python preprocess_ndbc_data.py --input_dir ./raw_data --output ./data/wave --dataset_type NDBC1 --station "42001,42020,42035"
"""

import argparse
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dataset_wave import preprocess_wave_data

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess NDBC buoy data for CSDI wave height model (Batch Support)"
    )
    # Changed from --input (single file) to --input_dir (directory)
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Directory containing raw NDBC data files (e.g., ./raw_data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/wave",
        help="Output directory for processed data (default: ./data/wave)"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="NDBC1",
        choices=["NDBC1", "NDBC2"],
        help="Dataset type: NDBC1 (10-min) or NDBC2 (hourly)"
    )
    # Modified to accept comma-separated list
    parser.add_argument(
        "--station",
        type=str,
        default="42001",
        help="Buoy station IDs, comma separated (e.g. '42001,42020,42035')"
    )
    
    args = parser.parse_args()
    
    # Parse station list
    station_list = [s.strip() for s in args.station.split(',')]
    
    print("="*60)
    print("NDBC Buoy Data Preprocessing (Batch Mode)")
    print("="*60)
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Stations to process: {station_list}")
    print("="*60 + "\n")
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    # Loop through stations
    for station_id in station_list:
        print(f"Processing Station: {station_id}...")
        
        # Construct expected filename (assuming format: raw_{station}.txt)
        #  adjust this naming convention if your files are named differently
        input_file = Path(args.input_dir) / f"raw_{station_id}.txt"
        
        if not input_file.exists():
            # Fallback: try just the station ID as filename
            input_file = Path(args.input_dir) / f"{station_id}.txt"
            
        if not input_file.exists():
            print(f"  [SKIPPED] Input file not found for station {station_id}")
            print(f"            Checked: {Path(args.input_dir) / f'raw_{station_id}.txt'}")
            continue
            
        try:
            preprocess_wave_data(
                raw_file=str(input_file),
                output_path=args.output,
                dataset_type=args.dataset_type,
                station=station_id
            )
            print(f"  [SUCCESS] {station_id} processed.")
            success_count += 1
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {station_id}: {e}")
            # We continue to the next station instead of exiting
            continue

    print("\n" + "="*60)
    print("BATCH PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Successfully processed {success_count} / {len(station_list)} stations.")
    
    if success_count > 0:
        print(f"Processed data saved to: {args.output}")
        print("\nYou can now run training with:")
        print(f"  python exe_wave.py --dataset_type {args.dataset_type} "
              f"--station \"{args.station}\" --data_path {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()