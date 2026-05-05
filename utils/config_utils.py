from omegaconf import OmegaConf
from typing import Dict, Any

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    # Return a primitive dictionary after loading to maintain backwards compatibility
    conf = OmegaConf.load(config_path)
    return OmegaConf.to_container(conf, resolve=True)

def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    # Merge using OmegaConf and return primitive dict
    base_conf = OmegaConf.create(base)
    override_conf = OmegaConf.create(override)
    merged_conf = OmegaConf.merge(base_conf, override_conf)
    return OmegaConf.to_container(merged_conf, resolve=True)
