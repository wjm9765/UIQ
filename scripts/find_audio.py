#!/usr/bin/env -S uv run python
import json
import yaml
from pathlib import Path
import os
import shutil

# datasets 라이브러리를 통해 오디오 데이터 접근
try:
    from datasets import load_dataset
except ImportError:
    print("datasets 라이브러리가 필요합니다. 'uv pip install datasets librosa soundfile' 실행 요망")
    import sys
    sys.exit(1)

import soundfile as sf

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    eval_config = config.get("evaluation", {})
    results_dir = eval_config.get("results_dir", "eval_outputs_sample")
    
    json_path = Path(results_dir) / "evaluation_results.json"
    if not json_path.exists():
        print(f"결과 파일 {json_path} 이 없습니다. 먼저 evaluate_all_claps.py 를 실행하세요.")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    metrics = data.get("metrics", {})
    retrieval_analysis = data.get("retrieval_analysis", {})
    
    # 붕괴된 클래스 수집 (각 모델별 Top 5)
    vulnerable_classes = set()
    for model, m_data in metrics.items():
        vul_list = m_data.get("Vulnerable_Classes", [])
        for vc in vul_list:
            vulnerable_classes.add(vc[0])  # vc는 [label, margin_score] 형태
            
    if not vulnerable_classes:
        print("붕괴된 클래스를 찾을 수 없습니다.")
        return
        
    print(f"🔍 붕괴가 관찰된 클래스들 ({len(vulnerable_classes)}개): {', '.join(vulnerable_classes)}")
    
    # VGG Sound 데이터셋 로드
    cache_dir = config.get("dataset", {}).get("cache_dir", "input")
    split = config.get("dataset", {}).get("split", "test")
    print(f"\n🎧 로컬에 다운로드된 VGGSound 데이터셋을 로드합니다... (split: {split}, cache_dir: {cache_dir})")
    dataset = load_dataset(
        "txya900619/vggsound-16k", 
        split=split,  # config.yaml의 split 값을 사용
        cache_dir=cache_dir
    )
    
    save_dir = Path(results_dir) / "vulnerable_audio_samples"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # OOM(메모리 초과) 및 무한 로딩 방지를 위해 list(dataset)을 사용하지 않고 다이렉트 인덱싱
    print("빠른 검색을 위해 데이터셋 메타데이터만 확인 중...")
    
    # txya900619/vggsound-16k 데이터셋은 'caption' (실제 라벨 역할)과 'audio' 컬럼만 가짐.
    all_labels = dataset["caption"]  
    
    print("오디오 파일 추출 시작...\n")
    for cls in vulnerable_classes:
        cls_dir = save_dir / cls.replace(' ', '_').replace('/', '_')
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 대상 클래스의 정답 오디오(Ground Truth) 1개 다이렉트 접근 및 저장
        try:
            gt_idx = all_labels.index(cls)  # 'caption' 컬럼에서 클래스명과 완벽히 일치하는 인덱스 찾기
            gt_row = dataset[gt_idx]
            audio_data = gt_row["audio"]["array"]
            sr = gt_row["audio"]["sampling_rate"]
            
            # youtube_id/start_time은 메타데이터에 없으므로 인덱스로 파일명 지정
            gt_path = cls_dir / f"GroundTruth_index{gt_idx}.wav"
            sf.write(str(gt_path), audio_data, sr)
        except ValueError:
            print(f"⚠️ [{cls}] 데이터셋에서 정답 오디오(caption: {cls})를 찾을 수 없습니다.")
                
        # 2. 각 모델이 이 클래스라고 검색한 최상위(Top-1) 오디오 다이렉트 저장
        for model_name, r_data in retrieval_analysis.items():
            if cls in r_data:
                top_1 = r_data[cls][0]
                yt_id = top_1.get("youtube_id", "unknownID")
                st = top_1.get("start_time", "0")
                retrieved_label = top_1.get("label", "unknownLabel")
                idx = top_1.get("index") # JSON에 저장된 index
                
                row = None
                if idx is not None and idx < len(dataset):
                    row = dataset[idx]
                
                if row:
                    audio_data = row["audio"]["array"]
                    sr = row["audio"]["sampling_rate"]
                    
                    status = "Correct" if retrieved_label == cls else "Wrong"
                    safe_ret_label = retrieved_label.replace(' ', '_').replace('/', '_')
                    
                    # youtube_id/start_time 대신, 안전하게 index로 파일명 명명 및 표시 보완
                    out_path = cls_dir / f"{model_name}_{status}_(Actual_{safe_ret_label})_index{idx}.wav"
                    sf.write(str(out_path), audio_data, sr)
                else:
                    print(f"[{model_name}] {cls} 에 대해 검색된 오디오(index: {idx})를 찾지 못했습니다.")

    print(f"\n✅ 완료되었습니다! 추출된 오디오는 '{save_dir}' 에서 확인할 수 있습니다.")

if __name__ == "__main__":
    main()
