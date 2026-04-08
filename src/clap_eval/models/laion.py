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
        self.resampler_cache = {}
        
    @torch.no_grad()
    def get_audio_embedding(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
            audio_data = audio_data.mean(axis=0)

        # Standard LAION CLAP sample rate is mostly 48kHz
        target_sr = self.processor.feature_extractor.sampling_rate
        if sr != target_sr:
            import torch
            import torchaudio.transforms as T
            if sr not in self.resampler_cache:
                self.resampler_cache[sr] = T.Resample(orig_freq=sr, new_freq=target_sr).to(self.device)
            audio_tensor = torch.from_numpy(audio_data).float().to(self.device)
            audio_tensor = self.resampler_cache[sr](audio_tensor)
            audio_data = audio_tensor.cpu().numpy()
        
        inputs = self.processor(audio=audio_data, sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, 'to')}
        
        outputs = self.model.get_audio_features(**inputs)
        # HF ClapModel get_audio_features sometimes returns BaseModelOutputWithPooling 
        # where the projected feature is in pooler_output
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        elif hasattr(outputs, "audio_features"):
            embeddings = outputs.audio_features
        elif isinstance(outputs, tuple):
            embeddings = outputs[0]
        else:
            embeddings = outputs
            
        embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, 'to')}
        
        outputs = self.model.get_text_features(**inputs)
        # Similar to audio features, might return BaseModelOutputWithPooling
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        elif hasattr(outputs, "text_features"):
            embeddings = outputs.text_features
        elif hasattr(outputs, "text_embeds"):
            embeddings = outputs.text_embeds
        elif isinstance(outputs, tuple):
            embeddings = outputs[0]
        else:
            embeddings = outputs
            
        embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
