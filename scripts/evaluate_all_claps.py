#!/usr/bin/env -S uv run python
import json
import torch
import numpy as np
import yaml
from pathlib import Path

# config.yaml에서 파라미터 로드
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

eval_config = config.get("evaluation", {})
results_dir = eval_config.get("results_dir", "eval_outputs_sample")
TOP_K_LIST = eval_config.get("top_k_list", [1, 5, 10])

# 모델별 결과 파일 경로 하드코딩 제거 및 동적 매핑
OUTPUT_FILES = {
    "M2D-CLAP": str(Path(results_dir) / "m2d_results.jsonl"),
    "MGA-CLAP": str(Path(results_dir) / "mga_results.jsonl"),
    "LAION-CLAP": str(Path(results_dir) / "laion_results.jsonl")
}

def load_data(jsonl_path):
    audio_embeddings = []
    text_embeddings_dict = {}
    true_labels = []
    audio_metadata = []
    
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {jsonl_path}")
        
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        print(f"[{Path(jsonl_path).name}] 데이터 로딩 중 (JSONL 파싱)...")
        lines = f.readlines()
        total_lines = len(lines)
        
        for i, line in enumerate(lines):
            if not line.strip(): continue
            
            if i > 0 and i % 5000 == 0:
                print(f"  ... {i}/{total_lines} 줄 처리 중 ...")
                
            data = json.loads(line.strip())
            
            label = data.get('label')
            aud_emb = data.get('embedding')
            txt_emb = data.get('text_embedding')
            yt_id = data.get('youtube_id')
            start_time = data.get('start_time')
            idx = data.get('index', i) # index가 없으면 기본 줄 번호 사용
            
            if aud_emb is not None and label is not None:
                audio_embeddings.append(aud_emb)
                true_labels.append(label)
                audio_metadata.append({"index": idx, "youtube_id": yt_id, "start_time": start_time, "label": label})
            
            if label is not None and txt_emb is not None and label not in text_embeddings_dict:
                text_embeddings_dict[label] = txt_emb
                
    if not audio_embeddings:
        raise ValueError(f"{jsonl_path}에 유효한 데이터가 없습니다.")
        
    unique_labels = sorted(list(text_embeddings_dict.keys()))
    if not unique_labels:
        raise ValueError(f"{jsonl_path}에 텍스트 임베딩 정보가 포함되어 있지 않습니다.")
        
    text_embeddings = [text_embeddings_dict[l] for l in unique_labels]
    
    return (
        torch.tensor(audio_embeddings).float(),
        torch.tensor(text_embeddings).float(),
        true_labels,
        unique_labels,
        audio_metadata
    )

def calculate_metrics(audio_embeds, text_embeds, true_labels, unique_labels, audio_metadata):
    # 정규화
    audio_embeds = torch.nn.functional.normalize(audio_embeds, p=2, dim=-1)
    text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
    
    # 유사도 계산 (Audio to Text)
    similarity = audio_embeds @ text_embeds.T
    sorted_indices = torch.argsort(similarity, dim=-1, descending=True)
    N, C = similarity.shape
    
    # Text to Audio 유사도 계산
    t2a_similarity = similarity.T  # [C, N]
    t2a_sorted_indices = torch.argsort(t2a_similarity, dim=-1, descending=True)
    
    metrics = {f"R@{k}": 0.0 for k in TOP_K_LIST}
    metrics["MRR"] = 0.0
    intra_sims, inter_sims = [], []
    class_margins = {label: [] for label in unique_labels}
    
    print(f"총 {N}개의 오디오에 대해 지표 계산 중...")

    for i in range(N):
        if i > 0 and i % 5000 == 0:
            print(f"  ... 지표 계산 중: {i}/{N}")
            
        target_label = true_labels[i]
        try:
            target_idx = unique_labels.index(target_label)
        except ValueError:
            continue
            
        rank = sorted_indices[i].tolist().index(target_idx) + 1
        
        for k in TOP_K_LIST:
            if rank <= k: metrics[f"R@{k}"] += 1
        metrics["MRR"] += 1.0 / rank

        target_sim = similarity[i, target_idx].item()
        mask = torch.ones(C, dtype=torch.bool)
        mask[target_idx] = False
        negative_sims = similarity[i, mask]
        
        intra_sims.append(target_sim)
        inter_sims.append(negative_sims.mean().item())
        class_margins[target_label].append(target_sim - negative_sims.max().item())

    # Text to Audio 상위 매칭 결과 저장 (분석용)
    retrieved_audios = {}
    for c_idx, label in enumerate(unique_labels):
        top_k_indices = t2a_sorted_indices[c_idx][:5].tolist() # Top 5
        retrieved_audios[label] = [audio_metadata[idx] for idx in top_k_indices]

    # 평균 지표 계산
    Valid_N = len(intra_sims)
    if Valid_N == 0:
        return metrics, retrieved_audios
    
    for k in TOP_K_LIST: metrics[f"R@{k}"] = (metrics[f"R@{k}"] / Valid_N) * 100
    metrics["MRR"] /= Valid_N
    metrics["Intra_Sim_Mean"] = np.mean(intra_sims)
    metrics["Intra_Sim_Std"] = np.std(intra_sims)
    metrics["Inter_Sim_Mean"] = np.mean(inter_sims)
    metrics["Sim_Margin_Mean"] = np.mean(intra_sims) - np.mean(inter_sims)
    
    avg_margins = {l: np.mean(m) for l, m in class_margins.items() if len(m) > 0}
    metrics["Vulnerable_Classes"] = sorted(avg_margins.items(), key=lambda x: x[1])[:10]  # Top 10 vulnerable
    
    return metrics, retrieved_audios

if __name__ == "__main__":
    results = {}
    all_retrieved_audios = {}
    for model_name, file_path in OUTPUT_FILES.items():
        print(f"\n{'='*60}\n▶ {model_name} 분석 시작\n{'='*60}")
        try:
            audio_embeds, text_embeds, true_labels, unique_labels, audio_metadata = load_data(file_path)
            metrics, retrieved_audios = calculate_metrics(audio_embeds, text_embeds, true_labels, unique_labels, audio_metadata)
            results[model_name] = metrics
            all_retrieved_audios[model_name] = retrieved_audios
            print(f"✅ {model_name} 분석 완료")
        except FileNotFoundError:
            print(f"🚨 파일을 찾을 수 없음: {file_path}")
        except Exception as e:
            print(f"🚨 에러 발생 ({model_name}): {e}")

    if results:
        # JSON 형태로 구조화하여 저장
        structured_output = {
            "metrics": results,
            "retrieval_analysis": all_retrieved_audios
        }
        
        json_file_path = Path(results_dir) / "evaluation_results.json"
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(structured_output, f, indent=4, ensure_ascii=False)
            
        print(f"\n✅ 구조화된 평가 결과({json_file_path})가 저장되었습니다.")
        
        # 사람이 읽기 편한 TXT 리포트 생성
        output_lines = []
        output_lines.append("\n" + "="*80)
        output_lines.append("🏆 Final Evaluation & Representation Collapse Report")
        output_lines.append("="*80)
        
        output_lines.append("\n[ 1. Standard Retrieval Performance ]")
        output_lines.append(f"{'Model':<15} | {'R@1 (%)':<8} | {'R@5 (%)':<8} | {'R@10 (%)':<8} | {'MRR':<6}")
        output_lines.append("-" * 65)
        for name, m in results.items():
            output_lines.append(f"{name:<15} | {m['R@1']:<8.2f} | {m['R@5']:<8.2f} | {m['R@10']:<8.2f} | {m['MRR']:<6.4f}")
            
        output_lines.append("\n[ 2. Embedding Separation Analysis (Intra vs Inter) ]")
        output_lines.append("-" * 80)
        output_lines.append(f"{'Model':<15} | {'Intra_Sim (Mean±Std)':<25} | {'Inter_Sim (Mean)':<15} | {'Sim_Margin':<10}")
        for name, m in results.items():
            intra_str = f"{m['Intra_Sim_Mean']:.4f} ± {m['Intra_Sim_Std']:.4f}"
            output_lines.append(f"{name:<15} | {intra_str:<25} | {m['Inter_Sim_Mean']:<15.4f} | {m['Sim_Margin_Mean']:<10.4f}")
            
        output_lines.append("\n[ 3. Top-10 Vulnerable Categories Analysis ]")
        output_lines.append("-" * 80)
        for name, m in results.items():
            output_lines.append(f"\n[{name}] 취약/붕괴 클래스 TOP 10:")
            for i, (vul_cls, margin) in enumerate(m["Vulnerable_Classes"], 1):
                # vul_cls는 string, margin은 score
                top_retrieved = all_retrieved_audios[name][vul_cls]
                top1 = top_retrieved[0]
                status = "✅ 정답" if top1['label'] == vul_cls else "❌ 오답"
                output_lines.append(f"  {i}. {vul_cls:<30} (Margin: {margin:.4f})")
                output_lines.append(f"     └─ 모델이 1순위로 찾은 오디오: 실제 라벨='{top1['label']}' -> {status}")
                output_lines.append(f"        (youtube_id: {top1['youtube_id']}, start_time: {top1['start_time']}, index: {top1.get('index', 'N/A')})")
                
        output_lines.append("\n" + "="*80 + "\n")
        
        full_report = "\n".join(output_lines)
        print(full_report)
        
        txt_file_path = Path(results_dir) / "evaluation_report.txt"
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(full_report)
            
        print(f"✅ 분석 리포트({txt_file_path})도 함께 저장되었습니다. (사람 확인용)")

