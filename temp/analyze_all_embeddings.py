#!/usr/bin/env -S uv run python
import os
import glob
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

MODELS = ["laion", "m2d", "mga", "msclap"]

def get_domain(label):
    if not label: return "Unknown"
    label = label.lower()
    
    if any(w in label for w in ['speak', 'talk', 'cough', 'sneeze', 'laugh', 'cry', 'breath', 'human', 'person', 'footstep', 'cheer', 'beatbox', 'whistl', 'sing', 'burp', 'hiccup', 'snore', 'clear throat', 'yell', 'scream', 'gargling', 'babbl', 'chatter']):
        return "Human"
    if any(w in label for w in ['dog', 'cat', 'bird', 'horse', 'pig', 'cow', 'lion', 'animal', 'bark', 'meow', 'roar', 'chirp', 'duck', 'goose', 'chicken', 'sheep', 'frog', 'insect', 'fly', 'mosquito', 'bee', 'crocodile', 'elephant', 'monkey', 'wolf', 'bear', 'tiger', 'snake', 'owl', 'pigeon']):
        return "Animal"
    if any(w in label for w in ['rain', 'wind', 'thunder', 'water', 'ocean', 'wave', 'fire', 'storm', 'stream', 'river', 'weather', 'tornado', 'crackling', 'splashing']):
        return "Nature"
    if any(w in label for w in ['play', 'guitar', 'piano', 'drum', 'violin', 'cello', 'harp', 'flute', 'trumpet', 'saxophone', 'keyboard', 'synthesizer', 'accordion', 'bass', 'music', 'singing', 'choir', 'orchestra', 'dj', 'chime', 'bell']):
        return "Music"
    if any(w in label for w in ['engine', 'car', 'train', 'plane', 'aircraft', 'siren', 'motor', 'helicopter', 'tractor', 'boat', 'ship', 'bicycle', 'bus', 'truck', 'brake', 'accelerat', 'vehicle', 'drive']):
        return "Machine_Vehicle"
    if any(w in label for w in ['drill', 'saw', 'tool', 'hammer', 'wrench', 'typing', 'typewriter', 'machine', 'printer', 'gear', 'ratchet', 'scissor', 'cut', 'weld', 'grind', 'sewing']):
        return "Tool_Mechanism"
    if any(w in label for w in ['door', 'window', 'clock', 'alarm', 'phone', 'ring', 'toilet', 'flush', 'shower', 'vacuum', 'blender', 'microwave', 'fridge', 'cook', 'frying', 'boiling', 'pour', 'glass', 'smash', 'bottle', 'chop']):
        return "Home_Everyday"
    if any(w in label for w in ['ball', 'tennis', 'golf', 'basketball', 'soccer', 'billiard', 'bowling', 'skat', 'swim', 'dive', 'jump', 'run', 'walk', 'stunt', 'hit']):
        return "Sports_Action"
    return "Other"

def analyze_all(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_results = {}
    
    for model in MODELS:
        print(f"\n======================================")
        print(f"📊 Analyzing Model: {model.upper()}")
        print(f"======================================")
        
        files_to_load = glob.glob(f"{input_dir}/{model}_*_results.jsonl")
        
        if not files_to_load:
            print(f"⚠️ No files found for {model}. Skipping.")
            continue
            
        all_items = []
        for filepath in files_to_load:
            print(f"  -> Loading {filepath}...")
            with open(filepath, 'r') as f:
                for line in tqdm(f, desc="Reading JSONL"):
                    try:
                        item = json.loads(line)
                        label = item.get("label", "")
                        audio_emb = item.get("embedding")
                        text_emb = item.get("text_embedding")
                        
                        if audio_emb is not None and text_emb is not None:
                            all_items.append({
                                "label": label,
                                "audio_emb": np.array(audio_emb, dtype=np.float32),
                                "text_emb": np.array(text_emb, dtype=np.float32)
                            })
                    except: continue
        
        if not all_items:
            continue
            
        print("  -> Processing embeddings...")
        
        labels = np.array([x["label"] for x in all_items])
        unique_labels = np.unique(labels)
        print(f"  -> Found {len(unique_labels)} unique domains/captions")
        
        audio_embs = np.stack([x["audio_emb"] for x in all_items])
        audio_embs = audio_embs / (np.linalg.norm(audio_embs, axis=1, keepdims=True) + 1e-8)
        
        label_to_text_emb = {}
        for x in all_items:
            if x["label"] not in label_to_text_emb:
                tb = x["text_emb"]
                label_to_text_emb[x["label"]] = tb / (np.linalg.norm(tb) + 1e-8)
                
        text_embs_unique = np.stack([label_to_text_emb[lbl] for lbl in unique_labels])
        
        print("  -> Computing Audio->Caption (A->T) Metrics...")
        sim_a2t = np.dot(audio_embs, text_embs_unique.T)
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        true_indices_a2t = np.array([label_to_idx[lbl] for lbl in labels])
        
        sorted_indices_a2t = np.argsort(-sim_a2t, axis=1)
        ranks_a2t = np.where(sorted_indices_a2t == true_indices_a2t[:, None])[1] + 1
        
        print("  -> Computing Caption->Audio (T->A) Metrics...")
        sim_t2a = sim_a2t.T
        sorted_indices_t2a = np.argsort(-sim_t2a, axis=1)
        
        domain_metrics = {}
        
        for i, lbl in enumerate(unique_labels):
            true_mask = (labels == lbl)
            num_correct = np.sum(true_mask)
            if num_correct == 0: continue
            
            label_ranks = ranks_a2t[true_mask]
            a2t_r1 = float(np.mean(label_ranks == 1))
            a2t_mrr = float(np.mean(1.0 / label_ranks))
            
            binary_hits = true_mask[sorted_indices_t2a[i]]
            precisions = np.cumsum(binary_hits) / np.arange(1, len(binary_hits) + 1)
            map_score = float(np.sum(precisions * binary_hits) / num_correct)
            
            t2a_r10 = float(np.sum(binary_hits[:10]) / num_correct)
            t2a_r50 = float(np.sum(binary_hits[:50]) / num_correct)
            t2a_r100 = float(np.sum(binary_hits[:100]) / num_correct)
            
            intra_sims = sim_t2a[i][true_mask]
            inter_sims = sim_t2a[i][~true_mask]
            
            intra_mean = float(np.mean(intra_sims)) if len(intra_sims) > 0 else 0.0
            inter_mean = float(np.mean(inter_sims)) if len(inter_sims) > 0 else 0.0
            intra_inter_ratio = float(intra_mean / (inter_mean + 1e-8)) if inter_mean > -1e-8 else 0.0
            
            domain_metrics[lbl] = {
                "Macro_Condition": get_domain(lbl),
                "Count": int(num_correct),
                "A2T_Recall@1": a2t_r1,
                "A2T_MRR": a2t_mrr,
                "T2A_MAP": map_score,
                "T2A_Recall@10": t2a_r10,
                "T2A_Recall@50": t2a_r50,
                "T2A_Recall@100": t2a_r100,
                "Intra_Sim": intra_mean,
                "Inter_Sim": inter_mean,
                "Intra_Inter_Ratio": intra_inter_ratio
            }
            
        overall_a2t_r1 = float(np.mean(ranks_a2t == 1))
        overall_a2t_mrr = float(np.mean(1.0 / ranks_a2t))
        overall_t2a_map = float(np.mean([m["T2A_MAP"] for m in domain_metrics.values()]))
        
        model_results[model] = {
            "Overall": {
                "Count": len(all_items),
                "A2T_Recall@1": overall_a2t_r1,
                "A2T_MRR": overall_a2t_mrr,
                "T2A_MAP": overall_t2a_map
            },
            "Domains": domain_metrics
        }
            
    report_path = os.path.join(output_dir, "evaluation_retrieval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=4)
        
    print(f"\n✅ All Models Evaluation Report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="results/eval_outputs", help="Directory containing eval output jsonl files")
    parser.add_argument("--output_dir", default="temp/analysis_output", help="Directory to save the analysis output")
    args = parser.parse_args()
    
    analyze_all(args.input_dir, args.output_dir)
