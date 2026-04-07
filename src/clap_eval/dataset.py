import os
from pathlib import Path
import pandas as pd

class VGGSoundDataset:
    def __init__(self, csv_path: str, audio_dir: str, samples_per_class: int = 100):
        self.csv_path = Path(csv_path)
        self.audio_dir = Path(audio_dir)
        self.samples_per_class = samples_per_class
        
        self.data = self._load_and_filter()

    def _load_and_filter(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"VGGSound csv file not found at {self.csv_path}")

        # The original vggsound.csv does not have headers, but typical format is:
        # youtube_id, start_time, label, split
        df = pd.read_csv(self.csv_path, header=None, names=["youtube_id", "start_time", "label", "split"])
        
        # Filter samples to only what we need per class
        # This takes the top `samples_per_class` rows (0 to N-1) for each label group
        filtered_df = df.groupby("label").head(self.samples_per_class).reset_index(drop=True)
        return filtered_df

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for _, row in self.data.iterrows():
            y_id = row['youtube_id']
            st_time = row['start_time']
            label = row['label']
            
            # The audio filename convention depends on download script. E.g. {y_id}_{st_time}.wav
            # Customize if your actual download script saves differently
            audio_path = self.audio_dir / f"{y_id}_{st_time:06d}.wav"
            
            # Note: During actual processing, if audio file doesn't exist, we might skip or warn
            
            yield {
                "youtube_id": y_id,
                "start_time": st_time,
                "label": label,
                "audio_path": str(audio_path),
            }
