import os
import torch
import logging
from pathlib import Path
from model import ADTModel
from config import ADTModelConfig
from utils.config_utils import load_config_from_yaml, deep_merge_dicts
from safetensors.torch import load_file

def build_model(config_path: str, device: str = "cpu"):
    """
    Builds and loads the ADT model from a configuration file and its associated checkpoint.
    
    Args:
        config_path: Path to the experiment YAML configuration file.
        device: Device to load the model on ('cpu', 'cuda', 'mps').
        
    Returns:
        model: The loaded ADTModel instance.
        tokenizer: The associated MidiTokenizer instance.
    """
    device = torch.device(device)
    
    # Load and merge configurations
    base_dir = Path(__file__).parent
    default_config_path = base_dir / "configs" / "config_default.yaml"
    cfg = load_config_from_yaml(str(default_config_path))
    experiment_cfg = load_config_from_yaml(config_path)
    merged_cfg = deep_merge_dicts(cfg, experiment_cfg)
    
    # Extract model and inference sections
    model_section = merged_cfg.get("model", {})
    inference_section = merged_cfg.get("inference", {})
    checkpoint_path = inference_section.get("checkpoint_path")
    
    if not checkpoint_path:
        raise ValueError("inference.checkpoint_path is required in the configuration file.")
    
    # Prepare model configuration
    model_section["enc_lr"] = merged_cfg.get("training", {}).get("learning_rate", 1e-4)
    model_section["dec_lr"] = merged_cfg.get("training", {}).get("learning_rate", 1e-4)
    model_section.update(merged_cfg.get("shared", {}))
    model_config = ADTModelConfig(**model_section)
    
    # Initialize model
    model = ADTModel(model_config)
    
    # Load weights
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    
    state_dict = None
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found at {checkpoint_path}")
    
    # Handle different checkpoint formats (nested state_dicts)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, merged_cfg
