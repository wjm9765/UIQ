#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas",
#     "pyyaml",
#     "tqdm",
#     "yt-dlp",
# ]
# ///
import os
import subprocess
import yaml
from pathlib import Path
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import time
import random

def download_csv():
    Path("input/audio").mkdir(parents=True, exist_ok=True)
    csv_path = "input/vggsound.csv"
    
    if not os.path.exists(csv_path):
        print("Downloading VGGSound CSV...")
        url = "https://raw.githubusercontent.com/hche11/VGGSound/master/data/vggsound.csv"
        subprocess.run(["curl", "-o", csv_path, url], check=True)
        print(f"Downloaded VGGSound CSV to {csv_path}")
    else:
        print(f"VGGSound CSV already exists at {csv_path}")
    return csv_path

def download_audio(y_id, st_time):
    out_path = f"input/audio/{y_id}_{st_time:06d}.wav"
    
    # 이미 다운로드 되어있으면 스킵
    if os.path.exists(out_path):
        return True
        
    url = f"https://www.youtube.com/watch?v={y_id}"
    
    # yt-dlp로 10초(시작 시간 ~ 시작 시간 + 10) 오디오만 잘라서 .wav 포맷으로 추출
    cmd = [
        "yt-dlp",
        "-f", "ba",
        "-x", "--audio-format", "wav",
        "--download-sections", f"*{st_time}-{st_time+10}",
        "--quiet", "--no-warnings",
        "-o", out_path,
        url
    ]
    
    try:
        # 유튜브 Bot 탐지를 피하기 위해 요청 전 1~3초 사이의 무작위 딜레이 추가
        time.sleep(random.uniform(1.0, 3.0))
        
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"Error {y_id}: {res.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception {y_id}: {e}")
        return False

def main():
    csv_path = download_csv()
    
    # config.yaml에서 클래스당 샘플 수 읽기
    config_path = "config.yaml"
    stream_audio = False
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ds_config = config.get("dataset", {})
        samples_per_class = ds_config.get("samples_per_class", 100)
        stream_audio = ds_config.get("stream_audio", False)
    else:
        samples_per_class = 100
        
    if stream_audio:
        print("config.yaml에서 'stream_audio: true'로 설정되어 있습니다. 인메모리 스트리밍을 사용하므로 오디오 다운로드를 건너뜁니다.")
        return
        
    print(f"Reading CSV... Targeting {samples_per_class} samples per class.")
    df = pd.read_csv(csv_path, header=None, names=["youtube_id", "start_time", "label", "split"])
    filtered_df = df.groupby("label").head(samples_per_class).reset_index(drop=True)
    
    # 작업할 다운로드 리스트 (youtube_id, start_time) 추출
    tasks = [(row["youtube_id"], int(row["start_time"])) for _, row in filtered_df.iterrows()]
    total_tasks = len(tasks)
    
    print(f"Starting parallel download of {total_tasks} audio files to input/audio/...")
    print("This may take some time depending on your network speed (Warning: Some YouTube videos may be unavailable).")
    
    success_count = 0
    # ThreadPoolExecutor를 사용해 제한된 멀티스레딩으로 다운로드 (동시 2개 작업 처리하여 차단 방지)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # 진행 상황을 tqdm으로 표시
        futures = [executor.submit(download_audio, y_id, st_time) for y_id, st_time in tasks]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="Downloading Data"):
            if future.result():
                success_count += 1
                
    print(f"\nDownload Phase Finished! Successfully downloaded {success_count}/{total_tasks} files.")
    print("Now you can run the evaluation script.")

if __name__ == "__main__":
    main()
