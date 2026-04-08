import os
from pathlib import Path
from datasets import load_dataset
import resampy

class VGGSoundDataset:
    def __init__(self, hf_repo: str = "txya900619/vggsound-16k", split: str = "test", samples_per_class: int = 100, streaming: bool = True, cache_dir: str = "input"):
        self.hf_repo = hf_repo
        # We respect the split passed from config if we want to run only on test data
        self.split = split
        self.samples_per_class = samples_per_class
        self.streaming = streaming
        self.cache_dir = cache_dir
        
        # User requested to just do simple input->output mapping without hanging logic.
        # But samples_per_class is important, so we use it as a multiplier.
        # VGGSound has approx 309 classes. We multiply to get the total target limit.
        self.target_total = self.samples_per_class * 309
        
        print(f"Loading {self.hf_repo} split={self.split} with streaming={self.streaming}")
        print(f"Total target evaluation limit: {self.target_total} (samples_per_class {self.samples_per_class} * 309 classes)")
        self.dataset = load_dataset(self.hf_repo, split=self.split, streaming=self.streaming, cache_dir=self.cache_dir, trust_remote_code=True)
            
    def __len__(self):
        try:
            # If not streaming or length is known, we can return the minimum of actual length or target total
            return min(len(self.dataset), self.target_total)
        except TypeError:
            # For streaming, we can return the target total as a rough estimate for tqdm
            return self.target_total
            
    def __iter__(self):
        total_collected = 0
        # Simply yield everything in the split until we reach the target_total limit.
        # This acts effectively as tracking batch sizing/total samples without skipping heavily.
        for item in self.dataset:
            if total_collected >= self.target_total:
                break
                
            audio_dict = item.get("audio", {})
            y_id = item.get("youtube_id", "unknown_id")
            st_time = item.get("start_time", 0)
            
            yield {
                "youtube_id": y_id,
                "start_time": st_time,
                "label": item.get("label"),
                "caption": item.get("caption"),
                "text": item.get("text"),
                "audio_array": audio_dict.get("array"),
                "sampling_rate": audio_dict.get("sampling_rate", 16000)
            }
            total_collected += 1
