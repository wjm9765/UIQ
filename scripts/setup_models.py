#!/usr/bin/env -S uv run
import os
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

def setup_repos():
    base_dir = Path("models_third_party")
    base_dir.mkdir(exist_ok=True)
    
    repos = {
        "m2d": "https://github.com/nttcslab/m2d.git",
        "MGA-CLAP": "https://github.com/Ming-er/MGA-CLAP.git"
    }
    
    for name, url in repos.items():
        repo_path = base_dir / name
        if not repo_path.exists():
            print(f"Cloning {name} from {url}...")
            subprocess.run(["git", "clone", url, str(repo_path)], check=True)
        else:
            print(f"Repository {name} already exists at {repo_path}.")

def setup_checkpoints_dir():
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    
    print("\n[INFO] Checkpoint directory created at 'checkpoints/'.")
    print("[INFO] Downloading MGA and M2D weights from HuggingFace dataset repo: wjm9765/clap_weights")
    print("       (Make sure you have HF_TOKEN environment variable set if it's a private repo!)")
    
    repo_id = "wjm9765/clap_weights"
    repo_type = "dataset"
    
    # 1. Download MGA weight
    try:
        mga_file = "mga-clap.pt"
        print(f"Downloading {mga_file}...")
        hf_hub_download(repo_id=repo_id, filename=mga_file, repo_type=repo_type, local_dir=ckpt_dir)
    except Exception as e:
        print(f"Failed to download MGA weight: {e}")

    # 2. Download M2D weight
    try:
        m2d_file = "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth"
        print(f"Downloading {m2d_file}...")
        hf_hub_download(repo_id=repo_id, filename=m2d_file, repo_type=repo_type, local_dir=ckpt_dir)
    except Exception as e:
        print(f"Failed to download M2D weight: {e}")

if __name__ == "__main__":
    setup_repos()
    setup_checkpoints_dir()
