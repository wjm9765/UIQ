import abc
import numpy as np

class BaseClapModel(abc.ABC):
    """
    Abstract base class for CLAP wrappers (LAION, MGA, M2D).
    All models should implement this interface to be integrated into the test pipeline.
    """
    def __init__(self, name: str, hf_model_id: str, device: str):
        self.name = name
        self.hf_model_id = hf_model_id
        self.device = device
        self._load_model()

    @abc.abstractmethod
    def _load_model(self):
        """Load model weights and configs from HuggingFace via hf_model_id."""
        pass

    @abc.abstractmethod
    def get_audio_embedding(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Takes raw audio signal (numpy array) and sample rate, and outputs the audio embedding.
        Must handle its own required sequence length, resampling, or padding.
        """
        pass

    @abc.abstractmethod
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        """
        Takes a list of text string inputs and outputs the text embeddings.
        """
        pass
