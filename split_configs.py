#!/usr/bin/env -S uv run python
import sys
from ruamel.yaml import YAML

yaml = YAML(typ='safe')
yaml.default_flow_style = False

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f)

models = config.get('models', [])
for m in models:
    m['enabled'] = True
    name = m['name']
    new_conf = dict(config)
    new_conf['models'] = [m]
    
    with open(f'config_{name}.yaml', 'w', encoding='utf-8') as fw:
        yaml.dump(new_conf, fw)
