#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "input"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "input" / "datasets"))

from clap_eval.analysis import (
    generate_category_uniformity_report,
    generate_disagreement_matrix_report_from_results,
    generate_margin_heatmap_report,
)
from clap_eval.analysis.result_io import ResultCacheBuilder
from clap_eval.config import Config


def _load_config(argv: list[str]) -> tuple[Config, argparse.Namespace]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="config.yaml")
    known_args, remaining = config_parser.parse_known_args(argv)
    config = Config.load(known_args.config)
    return config, remaining


def run_uniformity(input_source: str, output_dir: str, max_samples_per_category: int) -> None:
    outputs = generate_category_uniformity_report(
        input_dir=input_source,
        output_dir=output_dir,
        max_samples_per_category=max_samples_per_category,
    )
    for key, path in outputs.items():
        print(f"[{key}] {path}")


def run_margin_heatmap(input_source: str, output_dir: str, cmap: str) -> None:
    outputs = generate_margin_heatmap_report(
        input_dir=input_source,
        output_dir=output_dir,
        cmap=cmap,
    )
    for key, path in outputs.items():
        print(f"[{key}] {path}")


def run_disagreement_matrix_from_results(input_source: str, output_dir: str) -> None:
    outputs = generate_disagreement_matrix_report_from_results(
        input_dir=input_source,
        output_dir=output_dir,
    )
    for key, path in outputs.items():
        print(f"[{key}] {path}")


def run_all(args: argparse.Namespace, analysis_cfg: dict) -> None:
    input_dir = args.input_dir or analysis_cfg.get("input_dir", "results/eval_outputs")
    output_dir = args.output_dir or analysis_cfg.get("output_dir", "analysis_output")
    cache_filename = analysis_cfg.get("cache_filename", "eval.jsonl")
    cache_force = bool(analysis_cfg.get("force_rebuild_cache", False))
    cache_path = Path(output_dir) / cache_filename

    print(f"[0/3] Preparing analysis cache from: {input_dir}")
    cache_builder = ResultCacheBuilder()
    cache_file = cache_builder.build(input_dir, cache_path, force=cache_force)
    print(f"[0/3] Cache ready: {cache_file}")

    print("[1/3] Running category-wise uniformity analysis...")
    run_uniformity(str(cache_file), output_dir, args.max_samples_per_category)

    print("[2/3] Running margin heatmap analysis...")
    run_margin_heatmap(str(cache_file), output_dir, args.cmap)

    print("[3/3] Running disagreement matrix analysis...")
    run_disagreement_matrix_from_results(str(cache_file), output_dir)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    config, remaining_argv = _load_config(argv)
    analysis_cfg = config.analysis
    config_path = next((argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--config"), "config.yaml")

    parser = argparse.ArgumentParser(description="CLAP analysis entrypoint for saved evaluation outputs")
    parser.add_argument("--config", default=config_path, help="Path to config file")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "uniformity", "margin-heatmap", "disagreement"],
        help="Execution mode",
    )
    parser.add_argument("--input_dir", default=analysis_cfg.get("input_dir", "results/eval_outputs"), help="Directory with evaluation JSONL outputs")
    parser.add_argument("--output_dir", default=analysis_cfg.get("output_dir", "analysis_output"), help="Output directory for analysis artifacts")
    parser.add_argument(
        "--cmap",
        default=analysis_cfg.get("cmap", "RdBu"),
        help="Colormap for margin heatmaps",
    )
    parser.add_argument(
        "--max_samples_per_category",
        type=int,
        default=int(analysis_cfg.get("max_samples_per_category", 2000)),
        help="Max samples per category when computing uniformity",
    )
    args = parser.parse_args(remaining_argv)

    if args.mode == "all":
        run_all(args, analysis_cfg)
    elif args.mode == "uniformity":
        cache_file = ResultCacheBuilder().build(args.input_dir, Path(args.output_dir) / analysis_cfg.get("cache_filename", "eval.jsonl"), force=bool(analysis_cfg.get("force_rebuild_cache", False)))
        run_uniformity(str(cache_file), args.output_dir, args.max_samples_per_category)
    elif args.mode == "margin-heatmap":
        cache_file = ResultCacheBuilder().build(args.input_dir, Path(args.output_dir) / analysis_cfg.get("cache_filename", "eval.jsonl"), force=bool(analysis_cfg.get("force_rebuild_cache", False)))
        run_margin_heatmap(str(cache_file), args.output_dir, args.cmap)
    elif args.mode == "disagreement":
        cache_file = ResultCacheBuilder().build(args.input_dir, Path(args.output_dir) / analysis_cfg.get("cache_filename", "eval.jsonl"), force=bool(analysis_cfg.get("force_rebuild_cache", False)))
        run_disagreement_matrix_from_results(str(cache_file), args.output_dir)


if __name__ == "__main__":
    main()
