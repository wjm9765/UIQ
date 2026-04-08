import os
from pathlib import Path
from datasets import load_dataset
import resampy

class VGGSoundDataset:
    def __init__(self, hf_repo: str = "txya900619/vggsound-16k", split: str = "test", samples_per_class: int = 100, streaming: bool = True, cache_dir: str = "input"):
        self.hf_repo = hf_repo
        self.split = split
        self.samples_per_class = samples_per_class
        self.streaming = streaming
        self.cache_dir = cache_dir
        # Load dataset
        self.dataset = load_dataset(self.hf_repo, split=self.split, streaming=self.streaming, cache_dir=self.cache_dir, trust_remote_code=True)

    
    def __len__(self):
        # We don't know the exact length upfront due to dynamic streaming & class limits.
        # This will be handled during iteration, but returning a rough estimate or 0 is fine.
        # Tqdm will not show a strict percentage, just a count.
        return 309 * self.samples_per_class

    def __iter__(self):
        label_counts = {}
        total_collected = 0
        target_total = 309 * self.samples_per_class # Approx 309 classes in VGGSound
        
        for item in self.dataset:
            label = item.get("label")
            # Some datasets have string labels, some integer IDs. 
            # We track them as-is.
            if label not in label_counts:
                label_counts[label] = 0
                
            if label_counts[label] < self.samples_per_class:
                label_counts[label] += 1
                total_collected += 1
                
                # audio dict has 'array' and 'sampling_rate'
                audio_dict = item.get("audio", {})
                y_id = item.get("youtube_id", "unknown_id")
                st_time = item.get("start_time", 0)
                
                yield {
                    "youtube_id": y_id,
                    "start_time": st_time,
                    "label": label,
                    "audio_array": audio_dict.get("array"),
                    "sampling_rate": audio_dict.get("sampling_rate", 16000)
                }
            
            # If we think we hit all classes 
            if total_collected >= target_total:
                break
