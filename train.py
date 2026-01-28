import gc
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import torch
import argparse
from transformers import Trainer, TrainingArguments

# Local imports
from model import ADTModel
from config import ADTModelConfig
from modules.synthetiser import SynthDrum, SynthDrumConfig
from utils.config_utils import load_config_from_yaml, deep_merge_dicts
from modules.midi_tokenizer import MidiTokenizer, MidiTokenizerConfig
from model import ComputeMelSpectrogram, create_mask, create_mask_plain
from data_modules.train_dataset import LakhDataset, collate_fn, LakhDatasetConfig


class ADTTrainer(Trainer):
    """Custom Trainer class for ADT model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute the training loss for the ADT model."""
        model.train()

        # Extract inputs
        tokens = inputs["tokens"]  # [batch_size, seq_len]
        wavs = inputs["wavs"]  # [batch_size, wav_len]
        token_lengths = inputs["token_lengths"]

        # Move to device
        tokens = tokens.to(self.device)
        wavs = wavs.to(self.device)
        token_lengths = token_lengths.to(self.device)

        # Create target input and output for teacher forcing
        tgt_input = tokens[:, :-1]
        labels = tokens[:, 1:]

        # Create masks
        tgt_seq_len = tgt_input.size(1)
        tgt_mask, tgt_padding_mask = create_mask_plain(tgt_seq_len, token_lengths, self.device)

        # Forward pass
        loss = model(src=wavs, tgt=tgt_input, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask, labels=labels)

        # Clean up tensors to save memory
        del tokens, wavs, token_lengths, tgt_input, labels, tgt_mask, tgt_padding_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (loss, None) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to handle validation properly."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            return {}

        model = self.model
        model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataset:
                # Extract batch data
                tokens = batch["tokens"].to(self.device)
                wavs = batch["wavs"].to(self.device)
                token_lengths = batch["token_lengths"].to(self.device)
                wav_lengths = batch["wav_lengths"].to(self.device)

                # Create masks for validation (full sequence)
                tgt_seq_len = tokens.size(1)
                tgt_mask = create_mask(tgt_seq_len, token_lengths, self.device)
                src_mask = None

                # Forward pass
                loss = model(src=wavs, tgt=tokens, src_mask=src_mask, tgt_mask=tgt_mask)

                total_loss += loss.item()
                num_batches += 1

                # Clean up tensors
                del tokens, wavs, token_lengths, wav_lengths, tgt_mask, loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        self.log(metrics)

        return metrics


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=getattr(logging, log_level.upper()),
    )
    logger = logging.getLogger(__name__)
    return logger


def create_model(config: ADTModelConfig, device: torch.device) -> ADTModel:
    """Create and initialize the ADT model."""
    # Create spectrogram computation module
    compute_spectrogram = ComputeMelSpectrogram(
        sample_rate=config.sample_rate,
        win_length=config.win_length,
        time_res=config.time_res,
        n_mels=config.n_mels,
        device=device,
    )

    # Create the main model
    model = ADTModel(config=config, compute_spectrogram=compute_spectrogram)

    return model


def create_training_arguments(config: Dict[str, Any]) -> TrainingArguments:
    """Create TrainingArguments from config."""

    experiment_cfg = config.get("experiment", {})
    training_cfg = config.get("training", {})
    logging_cfg = config.get("logging", {})
    checkpoint_cfg = config.get("checkpoint", {})

    run_name = experiment_cfg.get("run_name", "default")
    output_dir = logging_cfg.get("output_dir", "./outputs")

    # Create output directory
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine checkpoint path
    resume_from_checkpoint = None
    if checkpoint_cfg.get("resume_from_checkpoint"):
        resume_from_checkpoint = checkpoint_cfg.get("resume_from_checkpoint")
    elif checkpoint_cfg.get("auto_resume"):
        # Find latest checkpoint
        checkpoint_dir = output_path
        if checkpoint_dir.exists():
            checkpoint_pattern = "checkpoint-epoch-*-step-*"
            checkpoints = list(checkpoint_dir.glob(checkpoint_pattern))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                resume_from_checkpoint = str(latest_checkpoint)

    # FIXME just pass **training_cfg to the TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=training_cfg.get("num_epochs"),
        per_device_train_batch_size=training_cfg.get("batch_size"),
        per_device_eval_batch_size=training_cfg.get("batch_size"),
        learning_rate=float(training_cfg.get("learning_rate")),
        weight_decay=float(training_cfg.get("weight_decay")),
        warmup_ratio=training_cfg.get("warmup_ratio"),
        logging_steps=logging_cfg.get("logging_steps"),
        save_steps=logging_cfg.get("save_every_n_steps"),
        save_strategy="steps" if logging_cfg.get("save_every_n_steps") else "epoch",
        eval_strategy=training_cfg.get("eval_strategy"),
        eval_steps=logging_cfg.get("eval_every_n_steps"),
        save_total_limit=checkpoint_cfg.get("max_checkpoints"),
        fp16=training_cfg.get("mixed_precision") == "fp16",
        bf16=training_cfg.get("mixed_precision") == "bf16",
        dataloader_num_workers=min(os.cpu_count(), training_cfg.get("max_dataloader_num_workers")),
        dataloader_pin_memory=True,
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps"),
        gradient_checkpointing=False,
        max_grad_norm=float(training_cfg.get("max_grad_norm")),
        optim=training_cfg.get("optim"),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type"),
        report_to="wandb" if experiment_cfg.get("use_wandb") else "none",
        run_name=run_name,
        seed=experiment_cfg.get("seed"),
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return training_args


def train(config: Dict[str, Any]):
    """Main training function using Transformers Trainer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logger = setup_logging(config.get("logging", {}).get("log_level", "INFO"))

    # Initialize wandb if requested
    use_wandb = bool(config.get("experiment", {}).get("use_wandb"))
    if use_wandb:
        wandb.init(
            project=config.get("experiment").get("project_name"),
            config=config,
            name=config.get("experiment").get("run_name"),
        )

    # config
    config_tokenizer = config["tokenizer"]
    config_dataset = config["TrainDatasetConfig"]
    config_dataset.update(config["shared"])

    if config_dataset["dataset_name"] == "Lakh":
        config_synthetiser = config.get("synthetiser", None)
        assert config_synthetiser is not None, "Synthetiser is required for Lakh dataset"
        config_synthetiser["ADTOF_mapping"] = config_tokenizer["ADTOF_mapping"]
        config_synthetiser.update(config["shared"])
        synthetiser = SynthDrum(SynthDrumConfig(**config_synthetiser))
    else:
        config_synthetiser = None
    # load modules
    tokenizer = MidiTokenizer(MidiTokenizerConfig(**config_tokenizer))
    if config_dataset["dataset_name"] == "Lakh":
        dataset = LakhDataset(LakhDatasetConfig(**config_dataset), tokenizer, synthetiser)
    elif config_dataset["dataset_name"] == "TMIDT":
        dataset = TMIDTDataset(TMIDTDatasetConfig(**config_dataset), tokenizer)
    else:
        raise ValueError(f"Dataset name {config_dataset['dataset_name']} not supported")

    logger.info("Creating model...")
    model_section = config.get("model")
    model_section["enc_lr"] = config.get("training").get("learning_rate")
    model_section["dec_lr"] = config.get("training").get("learning_rate")
    model_section.update(config.get("shared"))
    model_config = ADTModelConfig(**model_section)
    model = create_model(model_config, device)

    # Create training arguments
    logger.info("Creating training arguments...")
    training_args = create_training_arguments(config)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = ADTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collate_fn,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()

    logger.info("Training completed!")

    if use_wandb:
        wandb.finish()


def _substitute_env_vars(content: str) -> str:
    """Substitute environment variables in config."""
    import re

    def replace(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))

    content = re.sub(r"\$\{oc\.env:([^}]+)\}", replace, content)
    content = re.sub(r"\$\{([^}]+)\}", replace, content)
    return content


parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()
if __name__ == "__main__":
    default_config_path = Path(__file__).parent / "configs" / "config_default.yaml"
    # config
    cfg = load_config_from_yaml(default_config_path)
    experiment_cfg = load_config_from_yaml(args.config)
    merged_cfg = deep_merge_dicts(cfg, experiment_cfg)

    train(merged_cfg)
