from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .result_io import ResultDataLoader, META_DOMAIN_ORDER


@dataclass(frozen=True)
class UniformityRecord:
    model: str
    level: str
    category: str
    sample_count: int
    uniformity: float


class UniformityCalculator:
    def __init__(self, temperature: float = 2.0, max_samples_per_category: int = 2000):
        self.temperature = temperature
        self.max_samples_per_category = max_samples_per_category

    def compute(self, embeddings: Sequence[np.ndarray]) -> float:
        if len(embeddings) < 2:
            return float("nan")

        vectors = np.stack(list(embeddings))
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        if len(vectors) > self.max_samples_per_category:
            indices = np.random.choice(len(vectors), self.max_samples_per_category, replace=False)
            vectors = vectors[indices]

        sq_distances = np.sum((vectors[:, None, :] - vectors[None, :, :]) ** 2, axis=-1)
        upper = sq_distances[np.triu_indices(len(vectors), k=1)]
        if upper.size == 0:
            return float("nan")

        return float(np.log(np.mean(np.exp(-self.temperature * upper))))


class CategoryUniformityAnalyzer:
    def __init__(self, calculator: UniformityCalculator | None = None):
        self.calculator = calculator or UniformityCalculator()

    def build_table(self, samples: pd.DataFrame, level: str) -> pd.DataFrame:
        category_column = "meta_domain" if level == "meta_domain" else "label"
        rows: List[UniformityRecord] = []

        grouped = samples.groupby(["model", category_column], dropna=False)
        for (model_name, category), group in tqdm(grouped, desc=f"Uniformity groups[{level}]", unit="group"):
            uniformity = self.calculator.compute(group["audio_embedding"].tolist())
            if np.isnan(uniformity):
                continue

            rows.append(
                UniformityRecord(
                    model=str(model_name),
                    level=level,
                    category=str(category),
                    sample_count=int(len(group)),
                    uniformity=uniformity,
                )
            )

        table = pd.DataFrame([record.__dict__ for record in rows])
        if table.empty:
            return table

        table["rank_within_model"] = table.groupby("model")["uniformity"].rank(method="dense", ascending=False).astype(int)
        table = table.sort_values(["model", "uniformity", "category"], ascending=[True, False, True])
        return table

    @staticmethod
    def top_k_collapse(table: pd.DataFrame, k: int = 20) -> pd.DataFrame:
        if table.empty:
            return table
        return (
            table.sort_values(["model", "uniformity", "sample_count", "category"], ascending=[True, False, False, True])
            .groupby("model", as_index=False, group_keys=False)
            .head(k)
            .reset_index(drop=True)
        )


class CategoryUniformityReporter:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, full_table: pd.DataFrame, top_tables: Dict[str, pd.DataFrame]) -> Dict[str, Path]:
        outputs: Dict[str, Path] = {}

        full_path = self.output_dir / "category_uniformity_full.csv"
        full_table.to_csv(full_path, index=False)
        outputs["full_csv"] = full_path

        for level, table in top_tables.items():
            level_path = self.output_dir / f"category_uniformity_top20_{level}.csv"
            table.to_csv(level_path, index=False)
            outputs[f"top20_{level}"] = level_path

        return outputs

    @staticmethod
    def print_top_tables(top_tables: Dict[str, pd.DataFrame]) -> None:
        for level, table in top_tables.items():
            if table.empty:
                print(f"\n[{level}] No valid categories with at least 2 samples.")
                continue

            print(f"\n[{level}] Top categories closest to zero (collapsed first)")
            for model_name, model_table in table.groupby("model"):
                print(f"\n  Model: {model_name}")
                print(model_table[["category", "sample_count", "uniformity"]].to_string(index=False))


def generate_category_uniformity_report(
    input_dir: str | Path,
    output_dir: str | Path,
    max_samples_per_category: int = 2000,
) -> Dict[str, Path]:
    loader = ResultDataLoader()
    samples = loader.load(input_dir)

    calculator = UniformityCalculator(max_samples_per_category=max_samples_per_category)
    analyzer = CategoryUniformityAnalyzer(calculator=calculator)

    level1_table = analyzer.build_table(samples, level="meta_domain")
    level2_table = analyzer.build_table(samples, level="fine_category")

    top_tables = {
        "level1": analyzer.top_k_collapse(level1_table, k=20),
        "level2": analyzer.top_k_collapse(level2_table, k=20),
    }

    reporter = CategoryUniformityReporter(output_dir)
    outputs = reporter.save(
        full_table=pd.concat([level1_table, level2_table], ignore_index=True),
        top_tables=top_tables,
    )
    reporter.print_top_tables(top_tables)
    return outputs
