import torch
import numpy as np
import sys
from pathlib import Path
from .base import BaseClapModel

class M2DClapModel(BaseClapModel):
    def _load_model(self):
        print(f"Loading '{self.name}' from local path: {self.checkpoint_path} on {self.device}")
        
        # Add the external repository to Python's sys.path dynamically
        if self.repo_path:
            repo_abs_path = str(Path(self.repo_path).resolve())
            if repo_abs_path not in sys.path:
                sys.path.insert(0, repo_abs_path)
                print(f"Added {repo_abs_path} to sys.path")
            
        # Example loading for M2D model
        try:
            # Note: Replace `models` with the exact module name or inference class 
            # provided by the m2d repository.
            # Example:
            # import models
            # self.model = models.build_model(...)
            # state_dict = torch.load(self.checkpoint_path, map_location='cpu')
            # self.model.load_state_dict(state_dict)
            # self.model.to(self.device).eval()
            print("[INFO] Please update M2DClapModel._load_model() in m2d.py with the exact import paths from the m2d repository.")
            pass
        except ImportError as e:
            print(f"Failed to import M2D model code: {e}")
        self.model = None

    @torch.no_grad()
    def get_audio_embedding(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        return np.zeros((1, 768)) # dummy shape

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 768)) # dummy shape
