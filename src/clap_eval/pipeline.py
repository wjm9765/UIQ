import os
import json
from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np

from .config import Config
from .dataset import VGGSoundDataset
from .models import get_model
from .utils import stream_youtube_audio_memory

class EvaluationPipeline:
    def __init__(self, config: Config):
        self.config = config
        ds_cfg = self.config.dataset
        exec_cfg = self.config.execution
        
        self.dataset = VGGSoundDataset(
            csv_path=ds_cfg.get("csv_path", "input/vggsound.csv"),
            audio_dir=ds_cfg.get("audio_dir", "input/audio"),
            samples_per_class=ds_cfg.get("samples_per_class", 100),
        )
        self.stream_audio = ds_cfg.get("stream_audio", True)
        
        self.batch_size = exec_cfg.get("batch_size", 8)
        self.device = exec_cfg.get("device", "cuda")
        self.output_dir = Path(exec_cfg.get("output_dir", "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_accelerate = exec_cfg.get("use_accelerate", False)

        if self.use_accelerate:
            try:
                from accelerate import Accelerator
                self.accelerator = Accelerator()
                self.device = self.accelerator.device
                print(f"Using accelerate. Assigned device: {self.device}")
            except ImportError:
                print("accelerate library not found. Falling back to default device logic.")
                self.accelerator = None
        else:
            self.accelerator = None

        # Determine strategy
        self.strategy = exec_cfg.get("strategy", "sequential")
        
        # Don't load all models at once if sequential. 
        # Define model configs to load later.
        self.model_configs = []
        for m_cfg in self.config.models:
            if m_cfg.get("enabled", True):
                self.model_configs.append(m_cfg)

    def load_audio(self, item: dict):
        if self.stream_audio:
            # Memory string logic (Lazy loading)
            return stream_youtube_audio_memory(item['youtube_id'], item['start_time'])
        else:
            # Fallback to local files
            audio_path = item["audio_path"]
            if not os.path.exists(audio_path):
                return None, None
            try:
                return librosa.load(audio_path, sr=48000, mono=True)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                return None, None

    def run(self):
        print(f"Total samples to execute: {len(self.dataset)}")
        dataset_list = list(self.dataset) # Convert to list for batching

        if self.strategy == "simultaneous":
            # Load all at once
            models = {cfg["name"]: get_model(cfg["name"], cfg, self.device) for cfg in self.model_configs}
        else:
            models = {} # We'll load per loop

        for m_cfg in self.model_configs:
            model_name = m_cfg["name"]
            model_type = m_cfg.get("type", "unknown")
            print(f"\n=== Starting evaluation for model: {model_name} ({model_type}) ===")
            
            if self.strategy == "sequential":
                # Load one model to maximize VRAM availability
                model = get_model(model_name, m_cfg, self.device)
            else:
                model = models[model_name]
            
            output_file = self.output_dir / f"{model_name}_results.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i in tqdm(range(0, len(dataset_list), self.batch_size), desc=f"Evaluating {model_name}"):
                    batch = dataset_list[i : i + self.batch_size]

                    results = []
                    for item in batch:
                        aud, sr = self.load_audio(item)
                        
                        if aud is None:
                            # Not available or taken down on youtube
                            embed = None
                        else:
                            embed_arr = model.get_audio_embedding(aud, sr)
                            embed = embed_arr.flatten().tolist()
                            
                        result = {
                            "youtube_id": item["youtube_id"],
                            "start_time": item["start_time"],
                            "label": item["label"],
                            "embedding": embed
                        }
                        results.append(result)
                        
                    for res in results:
                        f.write(json.dumps(res) + "\n")
            
            print(f"[{model_name}] Finished writing outputs to {output_file}")
            
            if self.strategy == "sequential":
                # Free memory
                del model
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
