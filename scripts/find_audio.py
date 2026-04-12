#!/usr/bin/env -S uv run python
import json
import yaml
from pathlib import Path
import os
import shutil
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    eval_config = config.get("evaluation", {})
    
    # 새롭게 생성된 구조화된 리포트 읽기
    json_path = Path("temp/analysis_output/evaluation_collapse_report.json")
    if not json_path.exists():
        print(f"결과 파일 {json_path} 이 없습니다. 먼저 temp/analyze_all_embeddings.py 를 실행하세요.")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
        
    cache_dir = config.get("dataset", {}).get("cache_dir", "input")
    save_dir = Path("temp/analysis_output/vulnerable_audio_samples")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 빠른 로딩을 위한 split 캐싱
    print(f"\n🎧 로컬에 다운로드된 VGGSound 데이터셋을 로드합니다... (cache_dir: {cache_dir})")
    ds_cache = {}
    
    def get_split(split_name):
        if split_name not in ds_cache:
            print(f"Loading '{split_name}' split...")
            ds_cache[split_name] = load_dataset("txya900619/vggsound-16k", split=split_name, cache_dir=cache_dir)
        return ds_cache[split_name]

    print("오디오 파일 추출 시작...\n")
    
    # 모델 -> 도메인 -> Worst 5 추출
    for model_name, domain_data in report_data.items():
        print(f"🚀 처리 중인 모델: {model_name.upper()}")
        for domain, data in domain_data.items():
            worst_samples = data.get("worst_samples", [])
            if not worst_samples: continue
                
            model_domain_dir = save_dir / f"{model_name.upper()}_{domain.replace('/', '_')}"
            model_domain_dir.mkdir(parents=True, exist_ok=True)
            
            for rank, sample in enumerate(worst_samples, 1):
                margin = sample.get("margin", 0.0)
                
                # 1. 원본(정답) 오디오
                orig = sample["original"]
                orig_split = orig.get("split", "test").replace("val", "test")
                orig_idx = orig.get("index")
                orig_label = orig.get("label", "unknown").replace(' ', '_').replace('/', '_')
                
                if orig_idx is not None:
                    orig_ds = get_split(orig_split)
                    if orig_idx < len(orig_ds):
                        orig_audio = orig_ds[int(orig_idx)]["audio"]
                        orig_path = model_domain_dir / f"Rank{rank}_Orig_(Split-{orig_split}_Idx-{orig_idx})_{orig_label}.wav"
                        sf.write(str(orig_path), orig_audio["array"], orig_audio["sampling_rate"])

                # 2. 오답(Confused/Collapsed) 오디오
                conf = sample["confused_with"]
                conf_split = conf.get("split", "test").replace("val", "test")
                conf_idx = conf.get("index")
                conf_label = conf.get("label", "unknown").replace(' ', '_').replace('/', '_')
                
                if conf_idx is not None:
                    conf_ds = get_split(conf_split)
                    if conf_idx < len(conf_ds):
                        conf_audio = conf_ds[int(conf_idx)]["audio"]
                        conf_path = model_domain_dir / f"Rank{rank}_ConfusedWith_(Split-{conf_split}_Idx-{conf_idx})_{conf_label}_Margin{margin:.2f}.wav"
                        sf.write(str(conf_path), conf_audio["array"], conf_audio["sampling_rate"])

    print(f"\n✅ 완료되었습니다! 추출된 오디오는 '{save_dir}' 에서 직접 들어보실 수 있습니다.")

if __name__ == "__main__":
    main()
