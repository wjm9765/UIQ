#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML(typ='safe')
yaml.default_flow_style = False

def _replace_suffix(path_str: str, suffix: str) -> str:
    path = Path(path_str)
    return str(path.parent / f"{path.name}_{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a base config.yaml into per-model config files")
    parser.add_argument("--config", default="config.yaml", help="Base config file path")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f)

    models = config.get('models', [])
    for model in models:
        name = str(model.get('name', 'model')).strip()
        model_key = name.lower()

        new_conf = deepcopy(config)

        # Keep only one model and ensure it is enabled.
        model_copy = deepcopy(model)
        model_copy['enabled'] = True
        new_conf['models'] = [model_copy]

        # Keep evaluation/analysis paths consistent with the single-model setup.
        exec_cfg = deepcopy(new_conf.get('execution', {}))
        eval_cfg = deepcopy(new_conf.get('evaluation', {}))
        analysis_cfg = deepcopy(new_conf.get('analysis', {}))

        base_output_dir = exec_cfg.get('output_dir', 'results/eval_outputs')
        model_output_dir = _replace_suffix(base_output_dir, model_key)
        exec_cfg['output_dir'] = model_output_dir

        eval_cfg['results_dir'] = model_output_dir

        analysis_cfg['input_dir'] = model_output_dir
        analysis_cfg['model_order'] = [model_key]
        cache_name = analysis_cfg.get('cache_filename', 'eval.jsonl')
        analysis_cfg['cache_filename'] = f"{Path(cache_name).stem}_{model_key}.jsonl"

        new_conf['execution'] = exec_cfg
        new_conf['evaluation'] = eval_cfg
        new_conf['analysis'] = analysis_cfg

        out_file = Path(args.config).with_name(f"config_{model_key}.yaml")
        with open(out_file, 'w', encoding='utf-8') as fw:
            yaml.dump(new_conf, fw)


if __name__ == "__main__":
    main()
