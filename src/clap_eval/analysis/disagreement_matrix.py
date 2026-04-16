from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .result_io import MODEL_ORDER, ResultDataLoader, compute_group_margins


class DisagreementMatrixAnalyzer:
    def __init__(self, samples: pd.DataFrame, model_order: Sequence[str] | None = None):
        if samples.empty:
            raise ValueError("No result samples were loaded from the input directory.")

        self.samples = samples
        self.model_order = [m.upper() for m in (model_order or MODEL_ORDER)]

    def compute_pair_disagreement(self, level: str = "meta_domain") -> pd.DataFrame:
        category_col = "meta_domain" if level == "meta_domain" else "label"
        margins = compute_group_margins(self.samples, category_col=category_col)
        if margins.empty:
            return pd.DataFrame()

        results: Dict[str, Dict[str, int]] = {}
        for category, group in margins.groupby("category", dropna=False):
            pivot = group.pivot_table(index="sample_id", columns="model", values="margin")
            row: Dict[str, int] = {}
            for a in self.model_order:
                for b in self.model_order:
                    if a == b:
                        continue
                    a_vals = pivot[a] if a in pivot.columns else pd.Series(np.nan, index=pivot.index)
                    b_vals = pivot[b] if b in pivot.columns else pd.Series(np.nan, index=pivot.index)
                    cond = (a_vals < 0) & (b_vals > 0)
                    row[f"{a} fail & {b} pass"] = int(cond.fillna(False).sum())
            results[str(category)] = row

        df = pd.DataFrame.from_dict(results, orient="index").fillna(0).astype(int)
        col_order = [f"{a} fail & {b} pass" for a in self.model_order for b in self.model_order if a != b]
        return df[[c for c in col_order if c in df.columns]]

    def compute_unique_success(self, level: str = "meta_domain") -> pd.DataFrame:
        category_col = "meta_domain" if level == "meta_domain" else "label"
        margins = compute_group_margins(self.samples, category_col=category_col)
        if margins.empty:
            return pd.DataFrame()

        results: Dict[str, Dict[str, int]] = {}
        for category, group in margins.groupby("category", dropna=False):
            pivot = group.pivot_table(index="sample_id", columns="model", values="margin")
            row: Dict[str, int] = {}
            for model_name in self.model_order:
                if model_name not in pivot.columns:
                    row[f"only {model_name} pass"] = 0
                    continue
                others = [m for m in self.model_order if m != model_name and m in pivot.columns]
                cond = pivot[model_name] > 0
                for other in others:
                    cond = cond & (pivot[other] <= 0)
                row[f"only {model_name} pass"] = int(cond.fillna(False).sum())
            results[str(category)] = row

        df = pd.DataFrame.from_dict(results, orient="index").fillna(0).astype(int)
        col_order = [f"only {m} pass" for m in self.model_order]
        return df[[c for c in col_order if c in df.columns]]


class DisagreementMatrixRenderer:
    def __init__(self, cmap: str = "YlOrRd"):
        self.cmap = cmap

    def save_heatmap(self, matrix: pd.DataFrame, output_path: str | Path, title: str, figsize: tuple[float, float] | None = None) -> None:
        if matrix.empty:
            return

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if figsize is None:
            rows = len(matrix.index)
            cols = max(1, len(matrix.columns))
            width = max(8.0, cols * 1.8 + 4.0)
            height = max(5.0, min(rows * 0.33 + 2.0, 120.0))
            figsize = (width, height)

        plt.figure(figsize=figsize)
        sns.heatmap(
            matrix,
            cmap=self.cmap,
            annot=True,
            fmt="d",
            linewidths=0.2,
            linecolor="#f2f2f2",
            cbar_kws={"label": "Count"},
        )
        plt.title(title)
        plt.xlabel("Model Disagreement Pair")
        plt.ylabel("Category")
        plt.tight_layout()
        plt.savefig(output, dpi=300)
        plt.close()


def generate_disagreement_matrix_report_from_results(
    input_dir: str | Path,
    output_dir: str | Path,
    model_order: Sequence[str] | None = None,
    cmap: str = "YlOrRd",
) -> Dict[str, Path]:
    samples = ResultDataLoader(model_order=model_order).load(input_dir)
    analyzer = DisagreementMatrixAnalyzer(samples=samples, model_order=model_order)

    level1 = analyzer.compute_pair_disagreement(level="meta_domain")
    level2 = analyzer.compute_pair_disagreement(level="fine_category")
    unique_success = analyzer.compute_unique_success(level="meta_domain")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    renderer = DisagreementMatrixRenderer(cmap=cmap)
    outputs: Dict[str, Path] = {}

    level1_path = output_path / "disagreement_matrix_level1.png"
    renderer.save_heatmap(level1, level1_path, "Model Disagreement Matrix (Meta Domain)")
    outputs["level1_heatmap"] = level1_path

    level2_path = output_path / "disagreement_matrix_level2.png"
    renderer.save_heatmap(level2, level2_path, "Model Disagreement Matrix (Fine Category)")
    outputs["level2_heatmap"] = level2_path

    unique_path = output_path / "unique_success_matrix_level1.png"
    renderer.save_heatmap(unique_success, unique_path, "Unique Model Success (Meta Domain)")
    outputs["unique_success_level1"] = unique_path

    table_output = {
        "level1": level1.reset_index().rename(columns={"index": "category"}).to_dict(orient="records"),
        "level2": level2.reset_index().rename(columns={"index": "category"}).to_dict(orient="records"),
        "unique_success_level1": unique_success.reset_index().rename(columns={"index": "category"}).to_dict(orient="records"),
    }
    table_path = output_path / "disagreement_matrix_tables.json"
    with table_path.open("w", encoding="utf-8") as f:
        json.dump(table_output, f, ensure_ascii=False, indent=2)
    outputs["tables_json"] = table_path

    return outputs
