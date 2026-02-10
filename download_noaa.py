import os
import requests
import gzip
import shutil
from pathlib import Path

# Configuration
STATIONS = ["42001", "42002", "42012", "42019", "42020", "42035", "42055"]
YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025,2026]  
OUTPUT_DIR = "./raw_data"

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        return True
    return False

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading data for {len(STATIONS)} stations, years {YEARS}...")
    
    for station in STATIONS:
        print(f"\nProcessing Station {station}...")
        merged_file_path = Path(OUTPUT_DIR) / f"raw_{station}.txt"
        
        # Open the merged file in write mode
        with open(merged_file_path, 'wb') as outfile:
            first_file = True
            
            for year in YEARS:
                filename = f"{station}h{year}.txt.gz"
                url = f"https://www.ndbc.noaa.gov/view_text_file.php?filename={filename}&dir=data/historical/stdmet/"
                temp_gz = Path(OUTPUT_DIR) / filename
                
                print(f"  Fetching {year}...", end=" ", flush=True)
                
                if download_file(url, temp_gz):
                    try:
                        # 1. Check if the file is already plain text (NDBC server fallback)
                        try:
                            with gzip.open(temp_gz, 'rb') as infile:
                                content = infile.read()
                        except OSError:
                            # If gzip fails (Not a gzipped file), read the file directly as plain binary
                            with open(temp_gz, 'rb') as infile:
                                content = infile.read()
                        
                        lines = content.splitlines()
                        
                        if len(lines) > 2:
                            # Determine if we need to write the header
                            if first_file:
                                # Write everything including headers for the first year
                                outfile.write(b'\n'.join(lines) + b'\n')
                                first_file = False
                            else:
                                # Skip the first 2 lines (headers) for subsequent years
                                outfile.write(b'\n'.join(lines[2:]) + b'\n')
                            print("OK")
                        else:
                            print("Empty/Short file")
                        
                    except Exception as e:
                        print(f"Error processing file: {e}")
                    
                    # Cleanup (must be outside the try/except block that processes the file)
                    if temp_gz.exists():
                        os.remove(temp_gz)
                else:
                    print(f"Failed (404/Network)")
        
        print(f"  -> Saved to {merged_file_path}")

if __name__ == "__main__":
    main()
