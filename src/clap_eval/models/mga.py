import torch
import numpy as np
import sys
import yaml
from pathlib import Path
from .base import BaseClapModel

class MGAClapModel(BaseClapModel):
    def _load_model(self):
        print(f"Loading '{self.name}' from local path: {self.checkpoint_path} on {self.device}")
        
        # Add the external repository to Python's sys.path dynamically
        repo_abs_path = None
        if self.repo_path:
            repo_abs_path = str(Path(self.repo_path).resolve())
            if repo_abs_path not in sys.path:
                sys.path.insert(0, repo_abs_path)
                print(f"Added {repo_abs_path} to sys.path")
        
        try:
            from models.ase_model import ASE
            import torchaudio.transforms as T
            from ruamel.yaml import YAML

            # Load configuration file required by MGA-CLAP
            config_path = Path(repo_abs_path) / "settings" / "inference_example.yaml"
            with open(config_path, "r") as f:
                yaml = YAML(typ='safe', pure=True)
                config = yaml.load(f)

            config["device"] = self.device
            self.model = ASE(config)
            self.model.to(self.device)

            # Load weights
            cp = torch.load(self.checkpoint_path, map_location=self.device,weights_only=False)
            # Some checkpoints dict load `model` key
            state_dict = cp['model'] if 'model' in cp else cp
            # strict=False 로 변경하여 사소한 키 불일치를 무시합니다.
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            print(f"Model weights loaded from {self.checkpoint_path}")
            
            self.target_sr = config.get("audio_args", {}).get("sr", 32000)
            self.resampler_cache = {}

        except Exception as e:
            print(f"Failed to import/load MGA model code: {e}")
            raise e

    @torch.no_grad()
    def get_audio_embedding(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        if self.model is None:
            return np.zeros((1, 512))

        # Expected to be (mono) and float32. Numpy array [len] -> Torch [1, len]
        if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
            # We assume it's shape (channels, length) -> simply average them
            audio_data = audio_data.mean(axis=0)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_data).float()

        if sr != self.target_sr:
            if sr not in self.resampler_cache:
                import torchaudio.transforms as T
                self.resampler_cache[sr] = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            audio_tensor = self.resampler_cache[sr](audio_tensor)
            
        # MGA-CLAP (HTSAT 기반)은 고정된 최대 버퍼 크기를 갖습니다.
        # inference_example.yaml 설정 기준 최대 10초(10 * 32000 = 320000 샘플)
        max_length = 10 * self.target_sr
        if audio_tensor.shape[-1] > max_length:
            audio_tensor = audio_tensor[:max_length]
        
        # Add batch dimension: [1, seq_len]
        audio_tensor = audio_tensor.unsqueeze(0).to(self.device, non_blocking=True)

        _, frame_embeds = self.model.encode_audio(audio_tensor)
        audio_embeds = self.model.msc(frame_embeds, self.model.codebook)
        
        # Return numpy array
        return audio_embeds.cpu().numpy()

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(texts), 512))
        
        _, word_embeds, attn_mask = self.model.encode_text(texts)
        text_embeds = self.model.msc(word_embeds, self.model.codebook, attn_mask)
        
        return text_embeds.cpu().numpy()

