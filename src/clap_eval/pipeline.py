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
        
        self.dataset_config = ds_cfg
        requested_split = ds_cfg.get("split", "train")
        if requested_split.lower() == "all":
            self.splits_to_run = ["train", "test"]
        else:
            self.splits_to_run = [requested_split]
            
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
            
        # Dataset is 16kHz. Keep it as 16kHz here, and let the model wrappers upsample if needed.
        target_sr = 16000
        if orig_sr != target_sr:
            import torch
            import torchaudio.transforms as T
            
            device = self.device if hasattr(self, 'device') else "cuda" if torch.cuda.is_available() else "cpu"
            
            # Cache the resampler instance to avoid memory leak / hanging by creating thousands of torchaudio resamplers
            if getattr(self, '_resampler', None) is None or self._resampler.orig_freq != orig_sr:
                self._resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr).to(device)
            
            audio_tensor = torch.from_numpy(audio_array).float().to(device)
            
            is_1d = (audio_tensor.ndim == 1)
            if is_1d: 
                audio_tensor = audio_tensor.unsqueeze(0)
                
            with torch.no_grad():
                audio_tensor = self._resampler(audio_tensor)
            
            if is_1d:
                audio_tensor = audio_tensor.squeeze(0)
                
            audio_array = audio_tensor.cpu().numpy()
            
        return audio_array, target_sr

    def run(self):
        for split in self.splits_to_run:
            print(f"\n===========================================")
            print(f"   STARTING EVALUATION FOR SPLIT: {split.upper()}")
            print(f"===========================================")
            
            # Load dataset for the current split
            self.dataset = VGGSoundDataset(
                hf_repo=self.dataset_config.get("hf_repo", "txya900619/vggsound-16k"),
                split=split,
                samples_per_class=self.dataset_config.get("samples_per_class", 1),
                streaming=self.dataset_config.get("streaming", True),
                cache_dir=self.dataset_config.get("cache_dir", "input")
            )
            
            print("Dataset will process items sequentially...")

            if self.strategy == "simultaneous":
                # Load all at once
                models = {cfg["name"]: get_model(cfg["name"], cfg, self.device) for cfg in self.model_configs}
            else:
                models = {} # We'll load per loop

            for m_cfg in self.model_configs:
                model_name = m_cfg["name"]
                model_type = m_cfg.get("type", "unknown")
                print(f"\n=== Starting evaluation for model: {model_name} ({model_type}) on split: {split} ===")
                
                if self.strategy == "sequential":
                    # Load one model to maximize VRAM availability
                    model = get_model(model_name, m_cfg, self.device)
                else:
                    model = models[model_name]
                
                output_file = self.output_dir / f"{model_name}_{split}_results.jsonl"
                
                # Since dataset is an iterator that stops, we need to rebuild the iterator 
                # for each model if we do sequential looping!
                current_dataset_iter = iter(self.dataset)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    batch = []
                    for idx, item in enumerate(tqdm(current_dataset_iter, desc=f"Evaluating {model_name}", total=len(self.dataset))):
                        item["dataset_index"] = idx  # 원본 데이터셋의 인덱스 저장
                        batch.append(item)
                        
                        if len(batch) == self.batch_size:
                            self.process_batch(batch, model, f, split)
                            batch = []
                    
                    # Process remaining
                    if len(batch) > 0:
                        self.process_batch(batch, model, f, split)
                
                print(f"[{model_name}] Finished writing outputs to {output_file}")
                
                if self.strategy == "sequential":
                    # Free memory
                    del model
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def process_batch(self, batch, model, f, split_name):
        import json
        auds = []
        texts = []
        valid_indices = []
        sr_batch = 16000 # default
        
        for idx, item in enumerate(batch):
            aud, sr = self.load_audio(item)
            if aud is not None:
                auds.append(aud)
                valid_indices.append(idx)
                sr_batch = sr
            
            text_val = item.get("label") or item.get("caption") or item.get("text") or ""
            texts.append(str(text_val))
            
        audio_embeds = [None] * len(batch)
        if auds:
            batch_audio_embeds = model.get_audio_embedding(auds, sr_batch)
            for val_idx, embed in zip(valid_indices, batch_audio_embeds):
                audio_embeds[val_idx] = embed.flatten().tolist()
                
        text_embeds = [None] * len(batch)
        if any(texts):
            batch_text_embeds = model.get_text_embedding(texts)
            for i, embed in enumerate(batch_text_embeds):
                text_embeds[i] = embed.flatten().tolist()
                
        for i, item in enumerate(batch):
            text_val = texts[i]
            result = {
                "index": item.get("dataset_index"),
                "split": split_name,
                "youtube_id": item.get("youtube_id", ""),
                "start_time": item.get("start_time", 0.0),
                "label": text_val,
                "embedding": audio_embeds[i],
                "text_embedding": text_embeds[i] if text_val else None
            }
            f.write(json.dumps(result) + "\n")
