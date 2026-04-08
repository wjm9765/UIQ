#!/usr/bin/env -S uv run
import sys
from pathlib import Path
import argparse


import os

# Add src to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ["HF_HOME"] = str(Path(__file__).parent.parent / "input")
os.environ["HF_DATASETS_CACHE"] = str(Path(__file__).parent.parent / "input" / "datasets")

import os
os.environ["HF_HOME"] = str(Path(__file__).parent.parent / "input")
os.environ["HF_DATASETS_CACHE"] = str(Path(__file__).parent.parent / "input" / "datasets")

from clap_eval.config import Config
from clap_eval.pipeline import EvaluationPipeline

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLAP Models on VGGSound")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config = Config.load(args.config)
    
    # Initialize and run pipeline
    pipeline = EvaluationPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
