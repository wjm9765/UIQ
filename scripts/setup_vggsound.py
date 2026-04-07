#!/usr/bin/env -S uv run
import os
from pathlib import Path
import subprocess

def download_csv():
    # Make sure input dir exists
    Path("input").mkdir(parents=True, exist_ok=True)
    csv_path = "input/vggsound.csv"
    
    if not os.path.exists(csv_path):
        print("Downloading VGGSound CSV...")
        url = "https://raw.githubusercontent.com/hche11/VGGSound/master/data/vggsound.csv"
        subprocess.run(["curl", "-o", csv_path, url], check=True)
        print(f"Downloaded VGGSound CSV to {csv_path}")
    else:
        print(f"VGGSound CSV already exists at {csv_path}")

    print("\nNext steps:")
    print("1. Set up audio downloads for the required class samples. Due to size limitations, using a script (e.g. yt-dlp) via custom parallel downloads is recommended.")
    print("2. Put downloaded .wav files into the 'input/audio' directory format: '{youtube_id}_{start_time:06d}.wav'.")

if __name__ == "__main__":
    download_csv()
