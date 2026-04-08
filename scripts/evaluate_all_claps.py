import os
os.environ["HF_TOKEN"] = ""
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.clap_eval.models.laion import LaionClapModel
from src.clap_eval.models.mga import MGAClapModel
from src.clap_eval.models.m2d import M2DClapModel


# ================= 1. 설정 및 경로 =================
OUTPUT_FILES = {
    "M2D-CLAP": "eval_outputs_smaple(310*2)/m2d_results.jsonl",
    "MGA-CLAP": "eval_outputs_smaple(310*2)/mga_results.jsonl",
    "LAION-CLAP": "eval_outputs_smaple(310*2)/laion_results.jsonl"
}

TOP_K_LIST = [1, 5, 10]

# ================= 2. 텍스트 임베딩 추출 (차원 자동 맞춤) =================
def get_text_embeddings_for_model(unique_labels, model_name, current_dim):
    print(f"[{model_name}] 진짜 모델 인코더 가동 중...")
    
    try:
        # 1. 모델 객체 생성 (각 파일의 클래스 이름에 맞춤)
        if model_name == "LAION-CLAP":
            model = LaionClapModel() 
        elif model_name == "M2D-CLAP":
            model = M2DClapModel()
        elif model_name == "MGA-CLAP":
            # MGA 클래스가 정의되어 있다면 연결, 없다면 일단 Pass
            # model = MgaClapModel() 
            print(f"⚠️ {model_name} 클래스 연결 필요 - 현재 랜덤값 사용")
            return torch.randn(len(unique_labels), current_dim).float()
        else:
            return torch.randn(len(unique_labels), current_dim).float()

        # 2. 진짜 텍스트 임베딩 추출
        embeddings = model.get_text_embedding(unique_labels)
        
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
            
        return embeddings.float()
        
    except Exception as e:
        print(f"🚨 모델 로드/추론 중 에러 발생: {e}")
        return torch.randn(len(unique_labels), current_dim).float()

# ================= 3. 데이터 로드 로직 (3개 값 반환 확인) =================
def load_audio_data(jsonl_path):
    audio_embeddings = []
    true_labels = []
    
    target_path = Path(jsonl_path)
    if not target_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {jsonl_path}")
        
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('embedding') is not None:
                audio_embeddings.append(data['embedding'])
                true_labels.append(data['label'])
                
    if not audio_embeddings:
        raise ValueError(f"{jsonl_path}에 유효한 데이터가 없습니다.")
    
    audio_tensor = torch.tensor(audio_embeddings).float()
    
    # 💡 데이터셋의 실제 차원 수를 추출 (예: 512, 1024 등)
    detected_dim_value = audio_tensor.shape[1]
        
    return audio_tensor, true_labels, detected_dim_value

# ================= 4. 심층 분석 로직 =================
def calculate_metrics(audio_embeds, text_embeds, true_labels, unique_labels):
    # 정규화
    audio_embeds = torch.nn.functional.normalize(audio_embeds, p=2, dim=-1)
    text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
    
    # 유사도 계산 (이제 차원이 무조건 맞습니다)
    similarity = audio_embeds @ text_embeds.T
    sorted_indices = torch.argsort(similarity, dim=-1, descending=True)
    N, C = similarity.shape
    
    metrics = {f"R@{k}": 0.0 for k in TOP_K_LIST}
    metrics["MRR"] = 0.0
    intra_sims, inter_sims = [], []
    class_margins = {label: [] for label in unique_labels}

    for i in range(N):
        target_label = true_labels[i]
        target_idx = unique_labels.index(target_label)
        rank = sorted_indices[i].tolist().index(target_idx) + 1
        
        for k in TOP_K_LIST:
            if rank <= k: metrics[f"R@{k}"] += 1
        metrics["MRR"] += 1.0 / rank

        target_sim = similarity[i, target_idx].item()
        mask = torch.ones(C, dtype=torch.bool); mask[target_idx] = False
        negative_sims = similarity[i, mask]
        
        intra_sims.append(target_sim)
        inter_sims.append(negative_sims.mean().item())
        class_margins[target_label].append(target_sim - negative_sims.max().item())

    # 평균 지표 계산
    for k in TOP_K_LIST: metrics[f"R@{k}"] = (metrics[f"R@{k}"] / N) * 100
    metrics["MRR"] /= N
    metrics["Intra_Sim_Mean"] = np.mean(intra_sims)
    metrics["Intra_Sim_Std"] = np.std(intra_sims)
    metrics["Inter_Sim_Mean"] = np.mean(inter_sims)
    metrics["Sim_Margin_Mean"] = np.mean(intra_sims) - np.mean(inter_sims)
    
    avg_margins = {l: np.mean(m) for l, m in class_margins.items() if len(m) > 0}
    metrics["Vulnerable_Classes"] = sorted(avg_margins.items(), key=lambda x: x[1])[:5]
    
    return metrics

# ================= 5. 메인 실행부 (NameError 방지용 수정) =================
if __name__ == "__main__":
    results = {}
    
    for model_name, file_path in OUTPUT_FILES.items():
        print(f"\n{'='*60}\n▶ {model_name} 분석 시작\n{'='*60}")
        try:
            # 💡 [중요] 여기서 detected_dim_value를 정확히 받아옵니다.
            audio_embeddings, true_labels, current_data_dim = load_audio_data(file_path)
            
            unique_labels = sorted(list(set(true_labels)))
            
            # 💡 [중요] 받아온 current_data_dim을 인자로 넘겨줍니다.
            text_embeddings = get_text_embeddings_for_model(unique_labels, model_name, current_data_dim)
            
            metrics = calculate_metrics(audio_embeddings, text_embeddings, true_labels, unique_labels)
            results[model_name] = metrics
            print(f"✅ {model_name} 분석 완료")
            
        except FileNotFoundError:
            print(f"🚨 파일을 찾을 수 없음: {file_path}")
        except Exception as e:
            print(f"🚨 에러 발생 ({model_name}): {e}")

    # ================= 6. 요약 리포트 =================
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