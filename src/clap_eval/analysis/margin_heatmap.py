from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .result_io import META_DOMAIN_ORDER, MODEL_ORDER, ResultDataLoader, compute_group_margins


class MarginHeatmapAnalyzer:
    def __init__(self, samples: pd.DataFrame, preferred_model_order: Sequence[str] | None = None):
        if samples.empty:
            raise ValueError("No result samples were loaded from the input directory.")

        self.samples = samples
        self.preferred_model_order = [m.upper() for m in (preferred_model_order or MODEL_ORDER)]

    def build_level1_matrix(self) -> pd.DataFrame:
        margins = compute_group_margins(self.samples, category_col="meta_domain")
        matrix = self._aggregate_matrix(margins)
        index_order = [d for d in META_DOMAIN_ORDER if d in matrix.index] + [d for d in matrix.index if d not in META_DOMAIN_ORDER]
        return matrix.reindex(index=index_order)

    def build_level2_matrix(self) -> pd.DataFrame:
        margins = compute_group_margins(self.samples, category_col="label")
        matrix = self._aggregate_matrix(margins)
        if not matrix.empty:
            order = matrix.min(axis=1, skipna=True).sort_values().index
            matrix = matrix.loc[order]
        return matrix

    def filter_negative_rows(self, matrix: pd.DataFrame) -> pd.DataFrame:
        if matrix.empty:
            return matrix
        return matrix.loc[(matrix < 0).any(axis=1)]

    def _aggregate_matrix(self, margins: pd.DataFrame) -> pd.DataFrame:
        if margins.empty:
            return pd.DataFrame()

        matrix = margins.groupby(["category", "model"], dropna=False)["margin"].mean().unstack("model")
        columns = [m for m in self.preferred_model_order if m in matrix.columns]
        extras = [m for m in matrix.columns if m not in columns]
        return matrix.reindex(columns=columns + sorted(extras))


class HeatmapRenderer:
    def __init__(self, cmap: str = "RdBu"):
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

        values = matrix.to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        max_abs = float(np.max(np.abs(finite_values))) if finite_values.size > 0 else 1.0

        plt.figure(figsize=figsize)
        sns.heatmap(
            matrix,
            cmap=self.cmap,
            center=0.0,
            vmin=-max_abs,
            vmax=max_abs,
            linewidths=0.2,
            linecolor="#f2f2f2",
            cbar_kws={"label": "Average Margin"},
        )
        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel("Category")
        plt.tight_layout()
        plt.savefig(output, dpi=300)
        plt.close()


def generate_margin_heatmap_report(
    input_dir: str | Path,
    output_dir: str | Path,
    cmap: str = "RdBu",
    model_order: Sequence[str] | None = None,
) -> Dict[str, Path]:
    samples = ResultDataLoader(model_order=model_order).load(input_dir)
    analyzer = MarginHeatmapAnalyzer(samples=samples, preferred_model_order=model_order)

    level1_matrix = analyzer.build_level1_matrix()
    level2_matrix = analyzer.build_level2_matrix()
    level2_negative_matrix = analyzer.filter_negative_rows(level2_matrix)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    renderer = HeatmapRenderer(cmap=cmap)
    outputs: Dict[str, Path] = {}

    level1_path = output_path / "margin_heatmap_level1.png"
    renderer.save_heatmap(level1_matrix, level1_path, "Average Margin by Meta Domain (Level 1)")
    outputs["level1_heatmap"] = level1_path

    level2_path = output_path / "margin_heatmap_level2_all.png"
    renderer.save_heatmap(level2_matrix, level2_path, "Average Margin by Fine Category (Level 2)")
    outputs["level2_heatmap_all"] = level2_path

    if not level2_negative_matrix.empty:
        negative_path = output_path / "margin_heatmap_level2_negative_only.png"
        renderer.save_heatmap(level2_negative_matrix, negative_path, "Collapsed Categories Only (Margin < 0)")
        outputs["level2_heatmap_negative_only"] = negative_path

    table_output = {
        "level1": level1_matrix.reset_index().rename(columns={"meta_domain": "category"}).to_dict(orient="records"),
        "level2": level2_matrix.reset_index().rename(columns={"label": "category"}).to_dict(orient="records"),
    }
    table_path = output_path / "margin_heatmap_tables.json"
    with table_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(table_output, f, ensure_ascii=False, indent=2)
    outputs["tables_json"] = table_path

    return outputs
