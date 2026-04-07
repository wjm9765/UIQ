import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
import librosa
from .base import BaseClapModel

class LaionClapModel(BaseClapModel):
    def _load_model(self):
        print(f"Loading '{self.name}' from HuggingFace Hub: {self.hf_model_id} on {self.device}")
        self.model = ClapModel.from_pretrained(self.hf_model_id).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(self.hf_model_id)

    @torch.no_grad()
    def get_audio_embedding(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
            audio_data = librosa.to_mono(audio_data)

        # Standard LAION CLAP sample rate is mostly 48kHz
        target_sr = self.processor.feature_extractor.sampling_rate
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        
        inputs = self.processor(audios=audio_data, sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, 'to')}
        embeddings = self.model.get_audio_features(**inputs)
        embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, 'to')}
        embeddings = self.model.get_text_features(**inputs)
        embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
