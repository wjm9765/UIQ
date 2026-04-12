import os
import io
import soundfile as sf
from pathlib import Path
from datasets import load_dataset, Audio
import resampy
import numpy as np

class VGGSoundDataset:
    def __init__(self, hf_repo: str = "txya900619/vggsound-16k", split: str = "test", samples_per_class: str | int = 100, streaming: bool = True, cache_dir: str = "input"):
        self.hf_repo = hf_repo
        self.split = split
        self.samples_per_class = samples_per_class
        self.streaming = streaming
        self.cache_dir = cache_dir
        
        # [핵심 최적화 부분 1]
        # 전체 다운로드를 피하기 위해 사용자의 streaming 설정(True 권장)을 준수합니다.
        self.dataset = load_dataset(self.hf_repo, split=self.split, streaming=self.streaming, cache_dir=self.cache_dir, trust_remote_code=True)
        
        self.is_all_samples = (str(self.samples_per_class).lower() == "all")
        
        # [핵심 최적화 부분 2]
        # streaming 모드에서 건너뛸(skip) 데이터의 오디오까지 매번 디코딩하면 18만개를 읽는데 엄청난 시간(수 시간)이 낭비됩니다.
        # 따라서 오디오 디코딩을 강제로 지연(decode=False)시켜 메타데이터만 0.001초만에 가져도록 합니다.
        self.dataset = self.dataset.cast_column('audio', Audio(decode=False))

        if self.is_all_samples:
            try:
                self.target_total = len(self.dataset)
            except TypeError:
                self.target_total = float('inf')
        else:
            self.target_total = int(self.samples_per_class) * 309
            print(f"Loading {self.hf_repo} split={self.split} -> Searching for {self.samples_per_class} samples per class (Target Total ~{self.target_total})...")

    def __len__(self):
        try: return self.target_total
        except Exception: return 0
            
    def __iter__(self):
        class_counts = {}
        total_collected = 0
        
        # streaming을 순회하되 건너뛰는 데이터는 오디오를 일절 로드하지 않습니다.
        for item in self.dataset:
            if not self.is_all_samples:
                # [핵심 오류 수정] vggsound-16k 데이터셋은 'label' 컬럼이 없고 'caption' 컬럼이 클래스명입니다!
                # 기존에는 label이 없어서 모두 "unknown"으로 인식되었고, 결국 처음 10개만 평가하고 종류되는 버그가 있었습니다.
                target_class = item.get("caption", "unknown")
                
                if class_counts.get(target_class, 0) >= int(self.samples_per_class):
                    continue # 이미 수집된 클래스는 단 0.001초만에 스킵!
                    
                class_counts[target_class] = class_counts.get(target_class, 0) + 1
            
                if total_collected >= self.target_total:
                    print(f"\n[Completed] Collected total limit of {self.target_total} effectively mapping 309 classes!")
                    break
            
            # [수동 디코딩] 스킵되지 않고 조건에 맞는 N개 데이터만 딱 여기서 1회성으로 직접 디코딩합니다!
            audio_dict = item.get("audio", {})
            try:
                bytes_data = audio_dict.get("bytes")
                y, sr = sf.read(io.BytesIO(bytes_data))
                # y가 다채널(Stereo)일 경우 Mono로 투영
                if len(y.shape) > 1:
                    y = y.mean(axis=1)
            except Exception:
                y, sr = np.zeros(16000), 16000
                
            yield {
                "youtube_id": item.get("youtube_id", "unknown_id"),
                "start_time": item.get("start_time", 0),
                "label": item.get("caption"), # 모델 파이프라인과 호환성을 위해 label 필드에도 기입
                "caption": item.get("caption"),
                "text": item.get("text"), # 만약 text가 없다면 pipeline에서 caption을 쓰도록 폴백
                "audio_array": y,
                "sampling_rate": sr
            }
            total_collected += 1
