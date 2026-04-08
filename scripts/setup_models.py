#!/usr/bin/env -S uv run
import os
import subprocess
from pathlib import Path

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
    print("Please manually download the respective `.pth` or `.pt` weights for M2D and MGA from their repositories")
    print("and place them inside the 'checkpoints/' folder. Update config.yaml accordingly.")

if __name__ == "__main__":
    setup_repos()
    setup_checkpoints_dir()
