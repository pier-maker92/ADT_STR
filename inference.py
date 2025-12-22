import gc
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import torch
import argparse
import mir_eval
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader
from safetensors.torch import load_file

# Local imports
from model import ADTModel
from config import ADTModelConfig
from model import ComputeMelSpectrogram
from torch.nn.utils.rnn import pad_sequence
from utils.mapping_utils import MappingUtils
from modules.midi_tokenizer import MidiTokenizer, MidiTokenizerConfig
from utils.config_utils import load_config_from_yaml, deep_merge_dicts
from data_modules.eval_dataset import ENSTDataset, ENSTDatasetConfig, MDBDataset, MDBDatasetConfig


class DrumConfusionMatrix:
    def __init__(self, label_mapping, onset_tolerance=0.05):
        base = list(label_mapping.values())
        if "Other" not in base:
            base.append("Other")
        self.labels = base
        self.mapping = label_mapping
        self.onset_tolerance = onset_tolerance
        rows = self.labels + ["False Positive"]
        cols = self.labels + ["False Negative"]
        self.matrix = pd.DataFrame(0, index=rows, columns=cols)

    def _label(self, pitch):
        return self.mapping.get(int(pitch), "Other")

    def update(self, ref_notes, pred_notes):
        refs = [(i, float(r[0]), int(r[2])) for i, r in enumerate(ref_notes)]
        preds = [(j, float(p[0]), int(p[2])) for j, p in enumerate(pred_notes)]

        # Build candidate lists within tolerance
        cand = {}  # i -> list of (|dt|, j)
        for i, r_on, r_pi in refs:
            c = []
            for j, p_on, p_pi in preds:
                dt = abs(p_on - r_on)
                if dt <= self.onset_tolerance:
                    c.append((dt, j))
            cand[i] = sorted(c)  # by |dt|

        matched_ref = {}
        matched_pred = {}

        # PASS 1: exact pitch matches (closest first)
        exact_edges = []
        for i, r_on, r_pi in refs:
            for dt, j in cand[i]:
                if preds[j][2] == r_pi:
                    exact_edges.append((dt, i, j))
        exact_edges.sort(key=lambda x: x[0])
        for _, i, j in exact_edges:
            if i not in matched_ref and j not in matched_pred:
                matched_ref[i] = j
                matched_pred[j] = i

        # PASS 2: remaining refs matched by closest onset (pitch may differ → confusion)
        for i, r_on, r_pi in refs:
            if i in matched_ref:
                continue
            for dt, j in cand[i]:
                if j not in matched_pred:
                    matched_ref[i] = j
                    matched_pred[j] = i
                    break

        # Tally
        used_preds = set(matched_pred.keys())
        for i, r_on, r_pi in refs:
            r_lbl = self._label(r_pi)
            if i in matched_ref:
                j = matched_ref[i]
                p_lbl = self._label(preds[j][2])
                self.matrix.loc[r_lbl, p_lbl] += 1
            else:
                self.matrix.loc[r_lbl, "False Negative"] += 1

        fp_count = 0
        for j, p_on, p_pi in preds:
            if j not in used_preds:
                self.matrix.loc["False Positive", self._label(p_pi)] += 1
                fp_count += 1
        if fp_count > 100:
            print(f"False Positive count: {fp_count}")

    def _metrics_for_label(self, lbl):
        m, L = self.matrix, self.labels
        tp = int(m.loc[lbl, lbl])
        fn = int(m.loc[lbl, "False Negative"] + m.loc[lbl, L].sum() - tp)
        fp = int(m.loc["False Positive", lbl] + m.loc[L, lbl].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return prec, rec, f1, tp + fn

    def to_csv(self, path):
        df = self.matrix.copy()
        prec, rec, f1, sup = [], [], [], []
        for lbl in df.index:
            if lbl in self.labels:
                p, r, f, s = self._metrics_for_label(lbl)
                prec.append(p)
                rec.append(r)
                f1.append(f)
                sup.append(s)
            else:
                prec.append("")
                rec.append("")
                f1.append("")
                sup.append("")
        df["precision"] = prec
        df["recall"] = rec
        df["f1"] = f1
        df["support"] = sup
        df.to_csv(path, index=True)

    def get_matrix(self):
        return self.matrix


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=getattr(logging, log_level.upper()),
    )
    logger = logging.getLogger(__name__)
    return logger


def load_model(
    checkpoint_path: str,
    config: ADTModelConfig,
    device: torch.device,
    logger: logging.Logger = None,
) -> ADTModel:
    """Load trained model from checkpoint."""
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

    # Load checkpoint from safetensors
    safetensors_path = f"{checkpoint_path}/model.safetensors"
    pytorch_path = f"{checkpoint_path}/pytorch_model.bin"

    # Try to load from safetensors first, then fallback to pytorch format
    if os.path.exists(safetensors_path):
        try:
            state_dict = load_file(safetensors_path)
            model.load_state_dict(state_dict)
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load from safetensors ({safetensors_path}): {e}")
                logger.info("Falling back to pytorch format...")
            checkpoint = torch.load(pytorch_path, map_location=device)

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                # Assume the checkpoint is just the state dict
                model.load_state_dict(checkpoint)
    elif os.path.exists(pytorch_path):
        checkpoint = torch.load(pytorch_path, map_location=device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Looked for both model.safetensors and pytorch_model.bin"
        )

    model.to(device)
    model.eval()

    return model


def compute_metrics(
    ref_notes: torch.FloatTensor,
    est_notes: torch.FloatTensor,
) -> Tuple[int, int, int]:
    """Compute precision, recall, F1 using mir_eval."""

    if len(ref_notes) == 0 and len(est_notes) == 0:
        # Both empty - perfect match
        return 0, 0, 0
    elif len(ref_notes) == 0:
        # Reference empty but estimation not - all false positives
        return 0, 0, len(est_notes)
    elif len(est_notes) == 0:
        # Estimation empty but reference not - all false negatives
        return 0, len(ref_notes), 0

    # Extract intervals and pitches
    ref_intervals = ref_notes[:, 0:2]
    ref_pitches = ref_notes[:, 2] * 1000
    est_intervals = est_notes[:, 0:2]
    est_pitches = est_notes[:, 2] * 1000

    # Compute metrics without offset tolerance
    matching = mir_eval.transcription.match_notes(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=0.05,
        offset_ratio=None,
        pitch_tolerance=1.0,
    )
    TP = len(matching)
    FN = len(ref_notes) - TP
    FP = len(est_notes) - TP
    return TP, FN, FP


def compute_per_label_metrics(pred_notes: np.ndarray, gt_notes: np.ndarray, per_label_metrics: dict) -> dict:
    """Select a subset of the predictions."""
    ADTOF_label_mapping = MappingUtils().ADTOF_label_mapping
    for pitch, label in ADTOF_label_mapping.items():
        if label == "Other":
            continue
        if len(pred_notes):
            pred_notes_label = pred_notes[pred_notes[:, 2] == pitch]
        else:
            pred_notes_label = []
        if len(gt_notes):
            gt_notes_label = gt_notes[gt_notes[:, 2] == pitch]
        else:
            gt_notes_label = []
        tp, fn, fp = compute_metrics(gt_notes_label, pred_notes_label)
        per_label_metrics[label]["tp"] += tp
        per_label_metrics[label]["fn"] += fn
        per_label_metrics[label]["fp"] += fp

    return per_label_metrics


def run_inference(
    model: ADTModel,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer: MidiTokenizer,
    beam_size: int = 5,
    use_beam_search: bool = True,
    output_path: str = None,
) -> Dict[str, float]:
    """Run inference and compute metrics."""
    TP, FN, FP = 0, 0, 0

    def aggregate_metrics(TP, FN, FP):
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f_measure = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f_measure": f_measure,
        }

    model.eval()

    per_label_metrics = defaultdict(lambda: defaultdict(int))

    # Prepare per-label metrics containers
    mu = MappingUtils()
    confusion_matrix = DrumConfusionMatrix(
        mu.ADTOF_label_mapping if tokenizer.ADTOF_mapping else mu.GM_reduced_name_convention
    )

    # Track where GT HH notes get misclassified
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            # Extract batch data
            gt_notes = [b.cpu().numpy() for b in batch["notes"]]
            # get wavs
            wavs = pad_sequence(batch["wavs"], batch_first=True).to(device)
            if wavs.shape[-1] < 1024:
                continue
            batch_size = wavs.shape[0]

            # Generate predictions
            kwargs = {
                "src": wavs,
                "src_mask": None,
                "tgt_mask": None,
                "max_length": 1024,
            }
            if use_beam_search:
                kwargs["beam_size"] = beam_size
            sample_fn = model.sample if not use_beam_search else model.beam_search
            tokens_pred = sample_fn(**kwargs)

            # Process each item in the batch
            for i in range(batch_size):
                # Extract individual sequences
                pred_tokens = tokens_pred[i].cpu().numpy()
                # Remove padding from predictions (stop at first EOS token=3 or PAD token=1)
                eos_indices = np.where((pred_tokens == 3) | (pred_tokens == 1))[0]
                if len(eos_indices) > 0:
                    pred_tokens = pred_tokens[: eos_indices[0]]

                gt = gt_notes[i]
                if gt.shape[-1] == 0:
                    gt = []
                pred_notes = tokenizer.decode(pred_tokens)
                if not pred_notes.shape[-1] == 0:
                    pred_notes = pred_notes[pred_notes[:, 3] >= 0]
                pred_notes = pred_notes.detach().cpu().numpy()
                pred_notes = np.unique(pred_notes, axis=0)

                # Compute metrics
                cur_TP, cur_FN, cur_FP = compute_metrics(gt, pred_notes)
                TP += cur_TP
                FN += cur_FN
                FP += cur_FP

                # Per-label metrics for this item
                per_label_metrics = compute_per_label_metrics(pred_notes, gt, per_label_metrics)
                confusion_matrix.update(gt, pred_notes)
                if not os.path.exists(output_path):
                    os.makedirs(output_path, exist_ok=True)
                confusion_matrix.to_csv(
                    f"{output_path}/confusion_matrix.csv",
                )
            # Memory cleanup
            del wavs, tokens_pred

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Aggregate metrics
    final_metrics_no_offset = aggregate_metrics(TP, FN, FP)
    # Combine both metrics
    combined_metrics = defaultdict(dict)
    for key, value in final_metrics_no_offset.items():
        combined_metrics["all"][f"{key}"] = value
    for label, metrics in per_label_metrics.items():
        per_label_Aggregate_metrics = aggregate_metrics(metrics["tp"], metrics["fn"], metrics["fp"])
        for key, value in per_label_Aggregate_metrics.items():
            combined_metrics[label][f"{key}"] = value

    return combined_metrics


def inference(config: Dict[str, Any]):
    """Main inference function."""

    # Setup logging
    logger = setup_logging(config.get("logging", {}).get("log_level", "INFO"))

    # Extract configuration sections
    model_section = config.get("model", {})
    inference_section = config.get("inference", {})

    # Validate required paths
    checkpoint_path = inference_section.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("inference.checkpoint_path is required")
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading model
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model_section = config.get("model")
    model_section["enc_lr"] = config.get("training").get("learning_rate")
    model_section["dec_lr"] = config.get("training").get("learning_rate")
    model_section.update(config.get("shared"))
    model_config = ADTModelConfig(**model_section)
    model = load_model(checkpoint_path, model_config, device, logger)  # synth_config

    # Loading tokenizer
    logger.info("loading tokenizer...")
    tokenizer_config = MidiTokenizerConfig(**config.get("tokenizer"))
    tokenizer = MidiTokenizer(tokenizer_config)

    # Loading dataset
    data_section = config.get("EvalDatasetConfig")
    logger.info(f"Creating dataloader from: {data_section.get('dataset_path')}")
    if data_section.get("dataset_name") == "ENST":
        DatasetConfigClass, DatasetClass = ENSTDatasetConfig, ENSTDataset
    elif data_section.get("dataset_name") == "MDB":
        DatasetConfigClass, DatasetClass = MDBDatasetConfig, MDBDataset
    else:
        raise ValueError(f"Dataset name {data_section.get('dataset_name')} not supported")
    DatasetConfigItem = DatasetConfigClass(
        **data_section,
        **config.get("shared"),
    )
    num_workers = min(os.cpu_count(), 16)
    batch_size = inference_section.get("batch_size")
    dataloader = DatasetClass(DatasetConfigItem, tokenizer).get_dataloader(
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    logger.info(f"Dataset size: {len(dataloader.dataset)}")
    logger.info(f"Number of batches: {len(dataloader)}")

    # Run inference
    logger.info("Starting inference...")
    metrics = run_inference(
        model=model,
        dataloader=dataloader,
        device=device,
        beam_size=inference_section.get("beam_size"),
        use_beam_search=inference_section.get("use_beam_search"),
        tokenizer=tokenizer,
        output_path=inference_section.get("output_path"),
    )
    # save metrics to json
    with open(inference_section.get("output_path") + "/metrics.json", "w") as f:
        json.dump(metrics, f)

    # Print results
    logger.info("Inference completed!")
    logger.info("=" * 50)
    logger.info("RESULTS:")
    logger.info("=" * 50)

    logger.info("\nMetrics without offset tolerance:")
    logger.info(f"Precision: {metrics['all']['precision']:.4f}")
    logger.info(f"Recall:    {metrics['all']['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['all']['f_measure']:.4f}")
    return metrics


parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="Path to config file")
args = parser.parse_args()
if __name__ == "__main__":

    default_config_path = Path(__file__).parent / "configs" / "config_default.yaml"
    # config
    cfg = load_config_from_yaml(default_config_path)
    experiment_cfg = load_config_from_yaml(args.config)
    merged_cfg = deep_merge_dicts(cfg, experiment_cfg)
    # merged_cfg["inference"]["checkpoint_path"] = args.checkpoint_path
    # print("\n *** Evaluating checkpoint: ", args.checkpoint_path, " ***\n")
    inference(merged_cfg)

# usage: python inference.py configs/eval/MDBinference.yaml
