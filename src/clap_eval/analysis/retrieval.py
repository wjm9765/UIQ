from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from tqdm import tqdm


def get_domain(label: str) -> str:
    if not label:
        return "Unknown"
    label = label.lower()

    if any(w in label for w in [
        "speak", "talk", "cough", "sneeze", "laugh", "cry", "breath", "human", "person",
        "footstep", "cheer", "beatbox", "whistl", "sing", "burp", "hiccup", "snore",
        "clear throat", "yell", "scream", "gargling", "babbl", "chatter",
    ]):
        return "Human"
    if any(w in label for w in [
        "dog", "cat", "bird", "horse", "pig", "cow", "lion", "animal", "bark", "meow",
        "roar", "chirp", "duck", "goose", "chicken", "sheep", "frog", "insect", "fly",
        "mosquito", "bee", "crocodile", "elephant", "monkey", "wolf", "bear", "tiger",
        "snake", "owl", "pigeon",
    ]):
        return "Animal"
    if any(w in label for w in [
        "rain", "wind", "thunder", "water", "ocean", "wave", "fire", "storm", "stream",
        "river", "weather", "tornado", "crackling", "splashing",
    ]):
        return "Nature"
    if any(w in label for w in [
        "play", "guitar", "piano", "drum", "violin", "cello", "harp", "flute", "trumpet",
        "saxophone", "keyboard", "synthesizer", "accordion", "bass", "music", "singing",
        "choir", "orchestra", "dj", "chime", "bell",
    ]):
        return "Music"
    if any(w in label for w in [
        "engine", "car", "train", "plane", "aircraft", "siren", "motor", "helicopter",
        "tractor", "boat", "ship", "bicycle", "bus", "truck", "brake", "accelerat",
        "vehicle", "drive",
    ]):
        return "Machine_Vehicle"
    if any(w in label for w in [
        "drill", "saw", "tool", "hammer", "wrench", "typing", "typewriter", "machine",
        "printer", "gear", "ratchet", "scissor", "cut", "weld", "grind", "sewing",
    ]):
        return "Tool_Mechanism"
    if any(w in label for w in [
        "door", "window", "clock", "alarm", "phone", "ring", "toilet", "flush", "shower",
        "vacuum", "blender", "microwave", "fridge", "cook", "frying", "boiling", "pour",
        "glass", "smash", "bottle", "chop",
    ]):
        return "Home_Everyday"
    if any(w in label for w in [
        "ball", "tennis", "golf", "basketball", "soccer", "billiard", "bowling", "skat",
        "swim", "dive", "jump", "run", "walk", "stunt", "hit",
    ]):
        return "Sports_Action"
    return "Other"


def _discover_result_files(input_dir: Path) -> Dict[str, List[Path]]:
    model_to_files: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(input_dir.glob("*_results.jsonl")):
        stem = path.stem
        if not stem.endswith("_results"):
            continue
        prefix = stem[: -len("_results")]
        model_name = prefix.rsplit("_", 1)[0] if "_" in prefix else prefix
        if model_name:
            model_to_files[model_name].append(path)
    return dict(model_to_files)


def _load_items(files: Iterable[Path]) -> List[dict]:
    all_items: List[dict] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {file_path.name}"):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                label = item.get("label", "")
                audio_emb = item.get("embedding")
                text_emb = item.get("text_embedding")
                if audio_emb is None or text_emb is None:
                    continue

                all_items.append(
                    {
                        "label": label,
                        "audio_emb": np.array(audio_emb, dtype=np.float32),
                        "text_emb": np.array(text_emb, dtype=np.float32),
                    }
                )
    return all_items


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)


def _compute_metrics(items: List[dict]) -> dict:
    labels = np.array([x["label"] for x in items])
    unique_labels = np.unique(labels)

    audio_embs = _normalize_rows(np.stack([x["audio_emb"] for x in items]))

    label_to_text_emb = {}
    for x in items:
        if x["label"] not in label_to_text_emb:
            tb = x["text_emb"]
            label_to_text_emb[x["label"]] = tb / (np.linalg.norm(tb) + 1e-8)

    text_embs_unique = np.stack([label_to_text_emb[lbl] for lbl in unique_labels])

    sim_a2t = np.dot(audio_embs, text_embs_unique.T)
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    true_indices_a2t = np.array([label_to_idx[lbl] for lbl in labels])

    sorted_indices_a2t = np.argsort(-sim_a2t, axis=1)
    ranks_a2t = np.where(sorted_indices_a2t == true_indices_a2t[:, None])[1] + 1

    sim_t2a = sim_a2t.T
    sorted_indices_t2a = np.argsort(-sim_t2a, axis=1)

    domain_metrics = {}
    for i, lbl in enumerate(unique_labels):
        true_mask = labels == lbl
        num_correct = int(np.sum(true_mask))
        if num_correct == 0:
            continue

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
            "Count": num_correct,
            "A2T_Recall@1": a2t_r1,
            "A2T_MRR": a2t_mrr,
            "T2A_MAP": map_score,
            "T2A_Recall@10": t2a_r10,
            "T2A_Recall@50": t2a_r50,
            "T2A_Recall@100": t2a_r100,
            "Intra_Sim": intra_mean,
            "Inter_Sim": inter_mean,
            "Intra_Inter_Ratio": intra_inter_ratio,
        }

    overall_a2t_r1 = float(np.mean(ranks_a2t == 1))
    overall_a2t_mrr = float(np.mean(1.0 / ranks_a2t))
    overall_t2a_map = float(np.mean([m["T2A_MAP"] for m in domain_metrics.values()])) if domain_metrics else 0.0

    return {
        "Overall": {
            "Count": len(items),
            "A2T_Recall@1": overall_a2t_r1,
            "A2T_MRR": overall_a2t_mrr,
            "T2A_MAP": overall_t2a_map,
        },
        "Domains": domain_metrics,
    }


def analyze_all_models(input_dir: str | Path, output_dir: str | Path) -> Path:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_to_files = _discover_result_files(input_path)
    model_results = {}

    for model_name, files in model_to_files.items():
        items = _load_items(files)
        if not items:
            continue
        model_results[model_name] = _compute_metrics(items)

    report_path = output_path / "evaluation_retrieval_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=4)

    return report_path
