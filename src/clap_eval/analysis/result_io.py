from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .retrieval import get_domain

MODEL_ORDER = ["LAION", "M2D", "MGA", "MSCLAP"]
META_DOMAIN_ORDER = [
    "Human",
    "Animal",
    "Nature",
    "Music",
    "Machine_Vehicle",
    "Tool_Mechanism",
    "Home_Everyday",
    "Sports_Action",
    "Other",
    "Unknown",
]


@dataclass(frozen=True)
class ResultRecord:
    model: str
    split: str
    index: int | None
    youtube_id: str
    start_time: float
    label: str
    meta_domain: str
    audio_embedding: np.ndarray
    text_embedding: np.ndarray | None


class ResultFileDiscovery:
    def discover(self, input_dir: str | Path) -> Dict[str, List[Path]]:
        input_path = Path(input_dir)
        model_to_files: Dict[str, List[Path]] = {}
        for path in sorted(input_path.glob("*_results.jsonl")):
            model_name = self._infer_model_name(path)
            model_to_files.setdefault(model_name, []).append(path)
        return model_to_files

    @staticmethod
    def _infer_model_name(path: Path) -> str:
        stem = path.stem
        prefix = stem[: -len("_results")] if stem.endswith("_results") else stem
        return prefix.rsplit("_", 1)[0] if "_" in prefix else prefix


class ResultDataLoader:
    def __init__(self, model_order: Sequence[str] | None = None):
        self.model_order = [m.upper() for m in (model_order or MODEL_ORDER)]

    def load(self, source: str | Path) -> pd.DataFrame:
        source_path = Path(source)
        if source_path.is_file():
            rows = self._load_cache_file(source_path)
        else:
            discovery = ResultFileDiscovery()
            rows = []
            for raw_model, files in tqdm(discovery.discover(source_path).items(), desc="Models", unit="model"):
                model_name = self._normalize_model_name(raw_model)
                for file_path in tqdm(files, desc=f"Files[{model_name}]", unit="file", leave=False):
                    rows.extend(self._load_raw_file(model_name, file_path))

        if not rows:
            raise ValueError(f"No valid result records were found under: {source}")

        return pd.DataFrame.from_records(rows)

    def _load_raw_file(self, model_name: str, file_path: Path) -> List[dict]:
        rows: List[dict] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {file_path.name}", unit="line", leave=False):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                audio_embedding = item.get("embedding")
                if audio_embedding is None:
                    continue

                text_embedding = item.get("text_embedding")
                label = str(item.get("label", ""))
                rows.append(
                    {
                        "model": model_name,
                        "split": str(item.get("split", "")),
                        "index": item.get("index", item.get("dataset_index")),
                        "youtube_id": str(item.get("youtube_id", "")),
                        "start_time": float(item.get("start_time", 0.0)),
                        "label": label,
                        "meta_domain": get_domain(label),
                        "audio_embedding": np.asarray(audio_embedding, dtype=np.float32),
                        "text_embedding": None if text_embedding is None else np.asarray(text_embedding, dtype=np.float32),
                    }
                )

        return rows

    def _load_cache_file(self, file_path: Path) -> List[dict]:
        rows: List[dict] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {file_path.name}", unit="line", leave=False):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                audio_embedding = item.get("embedding") or item.get("audio_embedding")
                if audio_embedding is None:
                    continue

                text_embedding = item.get("text_embedding")
                label = str(item.get("label", ""))
                model_name = self._normalize_model_name(item.get("model", ""))
                rows.append(
                    {
                        "model": model_name,
                        "split": str(item.get("split", "")),
                        "index": item.get("index", item.get("dataset_index")),
                        "youtube_id": str(item.get("youtube_id", "")),
                        "start_time": float(item.get("start_time", 0.0)),
                        "label": label,
                        "meta_domain": str(item.get("meta_domain", get_domain(label))),
                        "sample_id": str(item.get("sample_id", build_sample_id(item))),
                        "audio_embedding": np.asarray(audio_embedding, dtype=np.float32),
                        "text_embedding": None if text_embedding is None else np.asarray(text_embedding, dtype=np.float32),
                    }
                )

        return rows

    @staticmethod
    def _normalize_model_name(raw_model: str) -> str:
        model_name = str(raw_model).strip().upper()
        aliases = {
            "LAION": "LAION",
            "M2D": "M2D",
            "MGA": "MGA",
            "MSCLAP": "MSCLAP",
        }
        return aliases.get(model_name, model_name)


class ResultCacheBuilder:
    def __init__(self, loader: ResultDataLoader | None = None):
        self.loader = loader or ResultDataLoader()

    def build(self, source_dir: str | Path, cache_file: str | Path, force: bool = False) -> Path:
        source_path = Path(source_dir)
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if not force and cache_path.exists():
            cache_mtime = cache_path.stat().st_mtime
            latest_source_mtime = max((path.stat().st_mtime for path in source_path.glob("*_results.jsonl")), default=0.0)
            if cache_mtime >= latest_source_mtime:
                print(f"[cache] Reusing up-to-date cache: {cache_path}")
                return cache_path

        print(f"[cache] Building cache from raw results: {source_path} -> {cache_path}")
        rows = self.loader.load(source_path)
        self._write_cache(rows, cache_path)
        print(f"[cache] Cache build completed: {cache_path}")
        return cache_path

    def _write_cache(self, rows: pd.DataFrame, cache_file: Path) -> None:
        with cache_file.open("w", encoding="utf-8") as f:
            for record in tqdm(rows.to_dict(orient="records"), total=len(rows), desc="Writing eval cache", unit="row"):
                payload = {
                    "model": record.get("model"),
                    "split": record.get("split"),
                    "index": record.get("index"),
                    "youtube_id": record.get("youtube_id"),
                    "start_time": record.get("start_time"),
                    "label": record.get("label"),
                    "meta_domain": record.get("meta_domain"),
                    "sample_id": record.get("sample_id", build_sample_id(record)),
                    "embedding": record.get("audio_embedding").tolist() if record.get("audio_embedding") is not None else None,
                    "text_embedding": record.get("text_embedding").tolist() if record.get("text_embedding") is not None else None,
                }
                f.write(json.dumps(payload) + "\n")


def build_sample_id(row: pd.Series | dict) -> str:
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = row
    return "|".join(
        [
            str(data.get("split", "")),
            str(data.get("index", "")),
            str(data.get("youtube_id", "")),
            str(data.get("start_time", "")),
            str(data.get("label", "")),
        ]
    )


def compute_group_margins(samples: pd.DataFrame, category_col: str) -> pd.DataFrame:
    rows: List[dict] = []

    grouped = samples.groupby(["model", category_col], dropna=False)
    for (model_name, category), group in tqdm(grouped, desc=f"Margin groups[{category_col}]", unit="group"):
        group = group.copy()
        if len(group) < 2:
            for _, row in group.iterrows():
                rows.append(
                    {
                        "model": model_name,
                        "category": category,
                        "sample_id": build_sample_id(row),
                        "margin": np.nan,
                    }
                )
            continue

        audio = np.stack(group["audio_embedding"].tolist())
        text = np.stack(group["text_embedding"].tolist()) if group["text_embedding"].notna().all() else None
        if text is None:
            continue

        audio = audio / (np.linalg.norm(audio, axis=1, keepdims=True) + 1e-8)
        text = text / (np.linalg.norm(text, axis=1, keepdims=True) + 1e-8)
        sim_matrix = audio @ text.T
        positive = np.diag(sim_matrix)
        neg_matrix = sim_matrix.copy()
        np.fill_diagonal(neg_matrix, -np.inf)
        max_negative = np.max(neg_matrix, axis=1)
        margins = positive - max_negative

        for row, margin in zip(group.to_dict("records"), margins):
            rows.append(
                {
                    "model": model_name,
                    "category": category,
                    "sample_id": build_sample_id(row),
                    "margin": float(margin),
                }
            )

    return pd.DataFrame.from_records(rows)


def normalized_mean_embeddings(embeddings: Iterable[np.ndarray]) -> np.ndarray:
    vectors = np.stack(list(embeddings))
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)


def compute_uniformity(embeddings: Iterable[np.ndarray], temperature: float = 2.0, max_samples: int = 2000) -> float:
    vectors = normalized_mean_embeddings(embeddings)
    if len(vectors) < 2:
        return float("nan")

    if len(vectors) > max_samples:
        indices = np.random.choice(len(vectors), max_samples, replace=False)
        vectors = vectors[indices]

    sq_dist = np.sum((vectors[:, None, :] - vectors[None, :, :]) ** 2, axis=-1)
    upper = sq_dist[np.triu_indices(len(vectors), k=1)]
    if upper.size == 0:
        return float("nan")
    return float(np.log(np.mean(np.exp(-temperature * upper))))
