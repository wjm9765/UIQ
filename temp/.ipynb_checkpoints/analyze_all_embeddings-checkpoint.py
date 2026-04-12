#!/usr/bin/env -S uv run python
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def calculate_alignment(features_a, features_b):
    return float(np.mean(np.linalg.norm(features_a - features_b, axis=1)**2))

def calculate_uniformity(features, t=2):
    if len(features) > 2000:
        idx = np.random.choice(len(features), 2000, replace=False)
        features = features[idx]
    sq_pdist = np.sum((features[:, None, :] - features[None, :, :])**2, axis=-1)
    return float(np.log(np.mean(np.exp(-t * sq_pdist))))

def analyze_all(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_results = {}
    
    # Track cross-model comparison stats
    model_uniformities = defaultdict(dict)

    for model in MODELS:
        print(f"\n======================================")
        print(f"📊 Analyzing Model: {model.upper()}")
        print(f"======================================")
        
        data_by_domain = defaultdict(list)
        files_to_load = glob.glob(f"{input_dir}/{model}_*_results.jsonl")
        
        if not files_to_load:
            print(f"⚠️ No files found for {model}. Skipping.")
            continue
            
        for filepath in files_to_load:
            print(f"  -> Loading {filepath}...")
            with open(filepath, 'r') as f:
                for line in tqdm(f, desc="Reading JSONL"):
                    try:
                        item = json.loads(line)
                    except: continue
                    label = item.get("label", "")
                    domain = get_domain(label)
                    audio_emb = item.get("embedding")
                    text_emb = item.get("text_embedding")
                    
                    if audio_emb is not None and text_emb is not None:
                        data_by_domain[domain].append({
                            "index": item.get("index"),
                            "split": item.get("split", "train"),
                            "youtube_id": item.get("youtube_id", ""),
                            "label": label,
                            "audio_emb": np.array(audio_emb, dtype=np.float32),
                            "text_emb": np.array(text_emb, dtype=np.float32)
                        })
        
        model_results[model] = {}
        print("  -> Calculating Embedding Collapse Metrics per Domain...")
        
        for domain, items in data_by_domain.items():
            n_samples = len(items)
            if n_samples < 2: continue
            
            audio_embs = np.stack([x["audio_emb"] for x in items])
            text_embs = np.stack([x["text_emb"] for x in items])
            
            audio_embs_norm = audio_embs / (np.linalg.norm(audio_embs, axis=1, keepdims=True) + 1e-8)
            text_embs_norm = text_embs / (np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-8)
            
            alignment_val = calculate_alignment(audio_embs_norm, text_embs_norm)
            audio_uniformity = calculate_uniformity(audio_embs_norm)
            
            model_uniformities[domain][model] = audio_uniformity
            
            # Use max 2000 for similarity matrix to prevent OOM
            n_sim_samples = min(n_samples, 2000)
            sim_indices = np.random.choice(n_samples, n_sim_samples, replace=False)
            
            sub_audio = audio_embs_norm[sim_indices]
            sub_text = text_embs_norm[sim_indices]
            
            sim_matrix = np.dot(sub_audio, sub_text.T)
            pos_sims = np.diag(sim_matrix)
            
            sim_matrix_masked = sim_matrix.copy()
            np.fill_diagonal(sim_matrix_masked, -np.inf)
            
            # Find the most confused instance correctly
            max_neg_idx_sub = np.argmax(sim_matrix_masked, axis=1)
            max_neg_sims = sim_matrix_masked[np.arange(n_sim_samples), max_neg_idx_sub]
            
            margins = pos_sims - max_neg_sims
            worst_idx_local = np.argsort(margins)[:5]
            
            worst_samples = []
            for idx in worst_idx_local:
                global_idx = sim_indices[idx]
                conflicting_global_idx = sim_indices[max_neg_idx_sub[idx]]
                
                orig_item = items[global_idx]
                conflicting_item = items[conflicting_global_idx]
                
                worst_samples.append({
                    "original": {
                        "index": orig_item["index"],
                        "split": orig_item["split"],
                        "label": orig_item["label"],
                        "youtube_id": orig_item["youtube_id"]
                    },
                    "confused_with": {
                        "index": conflicting_item["index"],
                        "split": conflicting_item["split"],
                        "label": conflicting_item["label"],
                        "youtube_id": conflicting_item["youtube_id"]
                    },
                    "margin": float(margins[idx]),
                    "pos_sim": float(pos_sims[idx]),
                    "neg_sim": float(max_neg_sims[idx])
                })
                
            model_results[model][domain] = {
                "count": n_samples,
                "alignment": alignment_val,
                "uniformity": audio_uniformity,
                "avg_margin": float(np.mean(margins)),
                "worst_samples": worst_samples
            }
            
    # Save Joint Report
    report_path = os.path.join(output_dir, "evaluation_collapse_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=4)
        
    print(f"\n✅ All Models Collapse Report saved to {report_path}")

    # Plot Multi-Model Uniformity Comparison
    print("🎨 Generating Cross-Model Collapsing Charts...")
    plt.figure(figsize=(14, 8))
    
    domains_plot = list(model_uniformities.keys())
    x = np.arange(len(domains_plot))
    width = 0.2
    
    for i, model in enumerate(MODELS):
        if model not in model_results: continue
        y_vals = [model_uniformities[d].get(model, 0) for d in domains_plot]
        plt.bar(x + (i * width) - (width * len(MODELS)/2.0) + width/2., y_vals, width, label=model.upper())
        
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Uniformity Score (Closer to 0 / Positive = Severe Collapse)')
    plt.title('Embedding Space Collapse Comparison by Model & Domain')
    plt.xticks(x, domains_plot, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "models_uniformity_comparison.png"))
    plt.close()
    
    print(f"✅ Charts saved in {output_dir}/models_uniformity_comparison.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="results/eval_outputs", help="Directory containing eval output jsonl files")
    parser.add_argument("--output_dir", default="temp/analysis_output", help="Directory to save the analysis output")
    args = parser.parse_args()
    
    analyze_all(args.input_dir, args.output_dir)
