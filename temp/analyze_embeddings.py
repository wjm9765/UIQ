#!/usr/bin/env -S uv run python
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from collections import defaultdict


def get_domain(label):
    if not label:
        return "Unknown"
        
    label = label.lower()
    
    # Keyword based mapping covering VGGSound 310 classes
    human_keywords = ['speak', 'talk', 'cough', 'sneeze', 'laugh', 'cry', 'breath', 'human', 'person', 'footstep', 'cheer', 'beatbox', 'whistl', 'sing', 'burp', 'hiccup', 'snore', 'clear throat', 'yell', 'scream', 'gargling', 'babbl', 'chatter']
    if any(w in label for w in human_keywords):
        return "Human"
        
    animal_keywords = ['dog', 'cat', 'bird', 'horse', 'pig', 'cow', 'lion', 'animal', 'bark', 'meow', 'roar', 'chirp', 'duck', 'goose', 'chicken', 'sheep', 'frog', 'insect', 'fly', 'mosquito', 'bee', 'crocodile', 'elephant', 'monkey', 'wolf', 'bear', 'tiger', 'snake', 'owl', 'pigeon']
    if any(w in label for w in animal_keywords):
        return "Animal"
        
    nature_keywords = ['rain', 'wind', 'thunder', 'water', 'ocean', 'wave', 'fire', 'storm', 'stream', 'river', 'weather', 'tornado', 'crackling', 'splashing']
    if any(w in label for w in nature_keywords):
        return "Nature"
        
    music_keywords = ['play', 'guitar', 'piano', 'drum', 'violin', 'cello', 'harp', 'flute', 'trumpet', 'saxophone', 'keyboard', 'synthesizer', 'accordion', 'bass', 'music', 'singing', 'choir', 'orchestra', 'dj', 'chime', 'bell']
    if any(w in label for w in music_keywords):
        return "Music"
        
    vehicle_keywords = ['engine', 'car', 'train', 'plane', 'aircraft', 'siren', 'motor', 'helicopter', 'tractor', 'boat', 'ship', 'bicycle', 'bus', 'truck', 'brake', 'accelerat', 'vehicle', 'drive']
    if any(w in label for w in vehicle_keywords):
        return "Machine/Vehicle"
        
    tool_keywords = ['drill', 'saw', 'tool', 'hammer', 'wrench', 'typing', 'typewriter', 'machine', 'printer', 'gear', 'ratchet', 'scissor', 'cut', 'weld', 'grind', 'sewing']
    if any(w in label for w in tool_keywords):
        return "Tool/Mechanism"
        
    home_keywords = ['door', 'window', 'clock', 'alarm', 'phone', 'ring', 'toilet', 'flush', 'shower', 'vacuum', 'blender', 'microwave', 'fridge', 'cook', 'frying', 'boiling', 'pour', 'glass', 'smash', 'bottle', 'chop']
    if any(w in label for w in home_keywords):
        return "Home/Everyday"
        
    sport_keywords = ['ball', 'tennis', 'golf', 'basketball', 'soccer', 'billiard', 'bowling', 'skat', 'swim', 'dive', 'jump', 'run', 'walk', 'stunt', 'hit']
    if any(w in label for w in sport_keywords):
        return "Sports/Action"
        
    return "Other"

def calculate_alignment(features_a, features_b):
    """Calculate Alignment (Wang & Isola, 2020)"""
    return float(np.mean(np.linalg.norm(features_a - features_b, axis=1)**2))

def calculate_uniformity(features, t=2):
    """Calculate Uniformity (Wang & Isola, 2020)"""
    # Use random subset if features are too large to avoid OOM
    if len(features) > 2000:
        idx = np.random.choice(len(features), 2000, replace=False)
        features = features[idx]
        
    sq_pdist = np.sum((features[:, None, :] - features[None, :, :])**2, axis=-1)
    return float(np.log(np.mean(np.exp(-t * sq_pdist))))

def analyze_results(jsonl_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {jsonl_path}...")
    data_by_domain = defaultdict(list)
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            label = item.get("label", "")
            domain = get_domain(label)
            
            audio_emb = item.get("embedding")
            text_emb = item.get("text_embedding")
            
            # Skip if embeddings are missing
            if audio_emb is not None and text_emb is not None:
                data_by_domain[domain].append({
                    "youtube_id": item.get("youtube_id", ""),
                    "start_time": item.get("start_time", 0.0),
                    "label": label,
                    "audio_emb": np.array(audio_emb, dtype=np.float32),
                    "text_emb": np.array(text_emb, dtype=np.float32)
                })

    print("Calculating domain metrics...")
    results_summary = {}
    
    for domain, items in data_by_domain.items():
        n_samples = len(items)
        if n_samples < 2:
            continue
            
        print(f"  -> Processing {domain} ({n_samples} samples)")
            
        audio_embs = np.stack([x["audio_emb"] for x in items])
        text_embs = np.stack([x["text_emb"] for x in items])
        
        # L2 Normalize Embeddings
        audio_embs_norm = audio_embs / (np.linalg.norm(audio_embs, axis=1, keepdims=True) + 1e-8)
        text_embs_norm = text_embs / (np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-8)
        
        # 1. Alignment & Uniformity
        alignment_val = calculate_alignment(audio_embs_norm, text_embs_norm)
        audio_uniformity = calculate_uniformity(audio_embs_norm)
        
        # 2. Similarity and Margins
        # Sub-sample to max 1000 for matrix math speed if large
        n_sim_samples = min(n_samples, 1000)
        sim_indices = np.random.choice(n_samples, n_sim_samples, replace=False)
        
        sub_audio = audio_embs_norm[sim_indices]
        sub_text = text_embs_norm[sim_indices]
        
        sim_matrix = np.dot(sub_audio, sub_text.T) # (N x N)
        pos_sims = np.diag(sim_matrix)
        
        # Max Negative Sims
        mask = np.ones(sim_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        
        neg_sims_flat = sim_matrix[mask]
        
        sim_matrix_masked = sim_matrix.copy()
        np.fill_diagonal(sim_matrix_masked, -np.inf)
        max_neg_sims = np.max(sim_matrix_masked, axis=1)
        
        margins = pos_sims - max_neg_sims
        
        # Worst Samples (Margin is lowest/negative = Embedding Collapsed area)
        worst_idx_local = np.argsort(margins)[:5]
        worst_samples = []
        for idx in worst_idx_local:
            global_idx = sim_indices[idx]
            worst_samples.append({
                "youtube_id": items[global_idx]["youtube_id"],
                "start_time": items[global_idx]["start_time"],
                "label": items[global_idx]["label"],
                "margin": float(margins[idx]),
                "pos_sim": float(pos_sims[idx]),
                "max_neg_sim": float(max_neg_sims[idx])
            })
            
        avg_pos_sim = float(np.mean(pos_sims))
        avg_neg_sim = float(np.mean(neg_sims_flat))
            
        results_summary[domain] = {
            "count": n_samples,
            "alignment": alignment_val,
            "audio_uniformity": audio_uniformity,
            "avg_pos_sim": avg_pos_sim,
            "avg_neg_sim": avg_neg_sim,
            "avg_margin": avg_pos_sim - avg_neg_sim,
            "worst_samples": worst_samples
        }
        
        # 3. Visualization: Histogram of Similarities
        plt.figure(figsize=(8, 5))
        sns.histplot(pos_sims, color='blue', label='Positive (Match)', kde=True, stat='density', alpha=0.5)
        # Randomly sample negatives to plot quickly
        plot_negs = np.random.choice(neg_sims_flat, min(5000, len(neg_sims_flat)), replace=False)
        sns.histplot(plot_negs, color='red', label='Negative (Mismatch)', kde=True, stat='density', alpha=0.5)
        plt.title(f"Similarity Dist: {domain} | Uniformity: {audio_uniformity:.3f}")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Density")
        plt.legend()
        
        safe_domain_name = domain.replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"sim_dist_{safe_domain_name}.png"))
        plt.close()

    # Save summary Report JSON
    report_path = os.path.join(output_dir, "domain_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4)
        
    print("\n" + "="*50)
    print("📋 [ANALYSIS REPORT SUMMARY]")
    print("="*50)
    for domain, metrics in sorted(results_summary.items(), key=lambda x: x[1]['audio_uniformity'], reverse=True):
        print(f"[{domain}] (n={metrics['count']})")
        print(f"   ▶ Uniformity (Higher is worse/collapsed): {metrics['audio_uniformity']:.4f}")
        print(f"   ▶ Alignment (Lower is better):            {metrics['alignment']:.4f}")
        print(f"   ▶ Pos Sim vs Neg (Margin):                {metrics['avg_pos_sim']:.4f} vs {metrics['avg_neg_sim']:.4f} (Avg Margin: {metrics['avg_margin']:.4f})")
        if metrics['worst_samples']:
            worst = metrics['worst_samples'][0]
            print(f"   ▶ Worst confused sample: '{worst['label']}' (ID: {worst['youtube_id']}) -> Margin: {worst['margin']:.4f}")
        print("-" * 50)
    print(f"\n✅ Full report saved to: {report_path}")
    print(f"✅ Histograms saved in:  {output_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/eval_outputs/laion_test_results.jsonl")
    parser.add_argument("--output_dir", default="temp/analysis_output")
    args = parser.parse_args()
    
    analyze_results(args.input, args.output_dir)
    
    analyze_results(args.input, args.output_dir)
