import torch
import numpy as np
from .base import BaseClapModel

class M2DClapModel(BaseClapModel):
    def _load_model(self):
        print(f"Loading '{self.name}' from HuggingFace Hub: {self.hf_model_id} on {self.device}")
        # Insert M2D specific loading logic.
        # IF M2D is supported in Transformers, you can use:
        # self.model = AutoModel.from_pretrained(self.hf_model_id)
        self.model = None

    @torch.no_grad()
    def get_audio_embedding(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        return np.zeros((1, 768)) # dummy shape

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 768)) # dummy shape
