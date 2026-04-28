"""
Configuration loader.

Reads config.yaml if it exists, otherwise falls back to config.default.yaml.
Copy config.default.yaml to config.yaml and fill in your instance-specific
values to get started. config.yaml is gitignored.
"""
import os
import yaml

_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH   = os.path.join(_DIR, 'config.yaml')
_DEFAULT_PATH  = os.path.join(_DIR, 'config.default.yaml')


def _load() -> dict:
    path = _CONFIG_PATH if os.path.exists(_CONFIG_PATH) else _DEFAULT_PATH
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


_cfg = _load()

jira:  dict = _cfg['jira']
epics: dict = _cfg['epics']
