import os
import json
from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np

from .config import Config
from .dataset import VGGSoundDataset
from .models import get_model

class EvaluationPipeline:
    def __init__(self, config: Config):
        self.config = config
        ds_cfg = self.config.dataset
        exec_cfg = self.config.execution
        
        self.dataset = VGGSoundDataset(
            hf_repo=ds_cfg.get("hf_repo", "txya900619/vggsound-16k"),
            split="test",
            samples_per_class=ds_cfg.get("samples_per_class", 1),
            streaming=ds_cfg.get("streaming", True)
        )
        
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
        audio_array = item.get("audio_array")
        orig_sr = item.get("sampling_rate", 16000)
        
        if audio_array is None:
            return None, None
            
        # CLAP models generally expect 48kHz audio. 
        # If the input is different, we resample it.
        target_sr = 48000
        if orig_sr != target_sr:
            import resampy
            audio_array = resampy.resample(audio_array, orig_sr, target_sr)
            
        return audio_array, target_sr

    def run(self):
        # We don't convert the streaming dataset to a list because 31,000 samples 
        # would consume ~50GB of RAM (Out of Memory). Instead, we will iterate 
        # the dataset generator for each model.
        print(f"Dataset streaming limit set to {self.dataset.samples_per_class} per class.")

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
            
            # Since dataset is an iterator that stops, we need to rebuild the iterator 
            # for each model if we do sequential looping!
            current_dataset_iter = iter(self.dataset)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                batch = []
                for item in tqdm(current_dataset_iter, desc=f"Evaluating {model_name}"):
                    batch.append(item)
                    
                    if len(batch) == self.batch_size:
                        self.process_batch(batch, model, f)
                        batch = []
                
                # Process remaining
                if len(batch) > 0:
                    self.process_batch(batch, model, f)
            
            print(f"[{model_name}] Finished writing outputs to {output_file}")
            
            if self.strategy == "sequential":
                # Free memory
                del model
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def process_batch(self, batch, model, f):
        results = []
        for item in batch:
            aud, sr = self.load_audio(item)
            
            if aud is None:
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
            
        import json
        for res in results:
            f.write(json.dumps(res) + "\n")
