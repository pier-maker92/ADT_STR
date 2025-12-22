import yaml
from typing import Dict


def load_config_from_yaml(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def deep_merge_dicts(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base and return the merged dict.

    - For dict values: merge recursively
    - For other types: override replaces base
    """
    for key, override_value in (override or {}).items():
        base_value = base.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            base[key] = deep_merge_dicts(dict(base_value), override_value)
        else:
            base[key] = override_value
    return base
