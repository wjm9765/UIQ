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
            split=ds_cfg.get("split", "train"),  # config.yaml의 split 설정을 따름 (기본값 test)
            samples_per_class=ds_cfg.get("samples_per_class", 1),
            streaming=ds_cfg.get("streaming", True),
            cache_dir=ds_cfg.get("cache_dir", "input")
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
        # We don't convert the streaming dataset to a list because 31,000 samples 
        # would consume ~50GB of RAM (Out of Memory). Instead, we will iterate 
        # the dataset generator for each model.
        print("Dataset will process items sequentially...")

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
                for idx, item in enumerate(tqdm(current_dataset_iter, desc=f"Evaluating {model_name}", total=len(self.dataset))):
                    item["dataset_index"] = idx  # 원본 데이터셋의 인덱스 저장
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
                
            text_val = item.get("label") or item.get("caption") or item.get("text") or ""
            if text_val:
                text_embed_arr = model.get_text_embedding([str(text_val)])
                text_embed = text_embed_arr.flatten().tolist()
            else:
                text_embed = None
                
            result = {
                "index": item.get("dataset_index"),  # 데이터셋의 원본 인덱스
                "youtube_id": item["youtube_id"],
                "start_time": item["start_time"],
                "label": text_val,
                "embedding": embed,
                "text_embedding": text_embed
            }
            results.append(result)
            
        import json
        for res in results:
            f.write(json.dumps(res) + "\n")
