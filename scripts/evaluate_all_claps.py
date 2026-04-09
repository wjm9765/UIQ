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
    
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {jsonl_path}")
        
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            
            label = data.get('label')
            aud_emb = data.get('embedding')
            txt_emb = data.get('text_embedding')
            
            if aud_emb is not None and label is not None:
                audio_embeddings.append(aud_emb)
                true_labels.append(label)
            
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
        unique_labels
    )

def calculate_metrics(audio_embeds, text_embeds, true_labels, unique_labels):
    # 정규화
    audio_embeds = torch.nn.functional.normalize(audio_embeds, p=2, dim=-1)
    text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
    
    # 유사도 계산
    similarity = audio_embeds @ text_embeds.T
    sorted_indices = torch.argsort(similarity, dim=-1, descending=True)
    N, C = similarity.shape
    
    metrics = {f"R@{k}": 0.0 for k in TOP_K_LIST}
    metrics["MRR"] = 0.0
    intra_sims, inter_sims = [], []
    class_margins = {label: [] for label in unique_labels}

    for i in range(N):
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

    # 평균 지표 계산
    Valid_N = len(intra_sims)
    if Valid_N == 0:
        return metrics
    
    for k in TOP_K_LIST: metrics[f"R@{k}"] = (metrics[f"R@{k}"] / Valid_N) * 100
    metrics["MRR"] /= Valid_N
    metrics["Intra_Sim_Mean"] = np.mean(intra_sims)
    metrics["Intra_Sim_Std"] = np.std(intra_sims)
    metrics["Inter_Sim_Mean"] = np.mean(inter_sims)
    metrics["Sim_Margin_Mean"] = np.mean(intra_sims) - np.mean(inter_sims)
    
    avg_margins = {l: np.mean(m) for l, m in class_margins.items() if len(m) > 0}
    metrics["Vulnerable_Classes"] = sorted(avg_margins.items(), key=lambda x: x[1])[:5]
    
    return metrics

if __name__ == "__main__":
    results = {}
    for model_name, file_path in OUTPUT_FILES.items():
        print(f"\n{'='*60}\n▶ {model_name} 분석 시작\n{'='*60}")
        try:
            audio_embeds, text_embeds, true_labels, unique_labels = load_data(file_path)
            metrics = calculate_metrics(audio_embeds, text_embeds, true_labels, unique_labels)
            results[model_name] = metrics
            print(f"✅ {model_name} 분석 완료")
        except FileNotFoundError:
            print(f"🚨 파일을 찾을 수 없음: {file_path}")
        except Exception as e:
            print(f"🚨 에러 발생 ({model_name}): {e}")

    if results:
        print("\n\n" + "#"*80)
        print("🏆 Final Evaluation & Representation Collapse Report")
        print("#"*80)
        
        print("\n[ 1. Standard Retrieval Performance ]")
        print(f"{'Model':<15} | {'R@1 (%)':<8} | {'R@5 (%)':<8} | {'R@10 (%)':<8} | {'MRR':<6}")
        print("-" * 60)
        for name, m in results.items():
            print(f"{name:<15} | {m['R@1']:<8.2f} | {m['R@5']:<8.2f} | {m['R@10']:<8.2f} | {m['MRR']:<6.4f}")
            
        print("\n[ 2. Embedding Separation Analysis (Intra vs Inter) ]")
        print("-" * 80)
        print(f"{'Model':<15} | {'Intra_Sim (Mean±Std)':<25} | {'Inter_Sim (Mean)':<15} | {'Sim_Margin':<10}")
        for name, m in results.items():
            intra_str = f"{m['Intra_Sim_Mean']:.4f} ± {m['Intra_Sim_Std']:.4f}"
            print(f"{name:<15} | {intra_str:<25} | {m['Inter_Sim_Mean']:<15.4f} | {m['Sim_Margin_Mean']:<10.4f}")

        print("\n[ 3. Top-5 Vulnerable Categories (High Confusion / Severe Collapse) ]")
        print("-" * 80)
        for name, m in results.items():
            vul_str = ", ".join([f"{l} ({mar:.3f})" for l, mar in m["Vulnerable_Classes"]])
            print(f"[{name}]\n ⚠️ {vul_str}\n")
        print("#"*80)
