import os
import torch
from pathlib import Path
from model import ADTModel
from config import ADTModelConfig
from utils.config_utils import load_config, deep_merge_dicts
from safetensors.torch import load_file


def resolve_checkpoint_path(checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        return checkpoint_path

    named_checkpoint = Path(__file__).parent / "checkpoints" / checkpoint
    if named_checkpoint.exists():
        return named_checkpoint

    raise FileNotFoundError(
        f"Checkpoint {checkpoint!r} not found as a path or under checkpoints/."
    )


def _load_merged_config(config_path: str) -> dict:
    base_dir = Path(__file__).parent
    default_config_path = base_dir / "configs" / "config_default.yaml"
    cfg = load_config(str(default_config_path))
    experiment_cfg = load_config(config_path)
    return deep_merge_dicts(cfg, experiment_cfg)


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

    merged_cfg = _load_merged_config(config_path)

    # Extract model and inference sections
    model_section = merged_cfg.get("model", {})
    inference_section = merged_cfg.get("inference", {})
    checkpoint_path = inference_section.get("checkpoint_path")

    if not checkpoint_path:
        raise ValueError(
            "inference.checkpoint_path is required in the configuration file."
        )

    # Prepare model configuration
    model_section["enc_lr"] = merged_cfg.get("training", {}).get("learning_rate", 1e-4)
    model_section["dec_lr"] = merged_cfg.get("training", {}).get("learning_rate", 1e-4)
    model_section.update(merged_cfg.get("shared", {}))
    model_config = ADTModelConfig(**model_section)

    # Initialize model
    model = ADTModel(model_config)

    # Load weights
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    safetensor_path = os.path.join(checkpoint_path, "model.safetensor")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    state_dict = None
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    elif os.path.exists(safetensor_path):
        state_dict = load_file(safetensor_path)
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


def build_model_from_checkpoint(checkpoint: str, device: str = "cpu"):
    checkpoint_path = resolve_checkpoint_path(checkpoint)
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} is required. Write the checkpoint inference config as JSON first."
        )
    return build_model(str(config_path), device=device)
