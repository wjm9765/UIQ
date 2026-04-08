#!/usr/bin/env -S uv run
import os
import yaml
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
    
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"[WARN] {config_path} not found. Skipping weight downloads.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("[INFO] Checking models in config.yaml for HuggingFace dynamic download...")
    print("       (Make sure HF_TOKEN environment variable is set if dealing with private repos!)")
    
    models = config.get("models", [])
    for model in models:
        hf_dl = model.get("hf_download")
        if hf_dl:
            repo_id = hf_dl.get("repo_id")
            filename = hf_dl.get("filename")
            repo_type = hf_dl.get("repo_type", "dataset") # Usually model or dataset
            
            if repo_id and filename:
                print(f"Downloading {filename} from {repo_type} repo: {repo_id}...")
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id, 
                        filename=filename, 
                        repo_type=repo_type, 
                        local_dir=ckpt_dir
                    )
                    print(f"Successfully downloaded to {downloaded_path}")
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    setup_repos()
    setup_checkpoints_dir()
