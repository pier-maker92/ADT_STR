import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import torchaudio
import audio_process
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path
from config import ClapConfig
from clap_encoder import ClapWrapper
from config.utils import load_yaml, recursive_merge


def sort_paths_by_parent_folder(file_paths):
    def sort_key(path):
        parent_name = Path(path).parent.name
        try:
            parent_index = int(parent_name)
            return (0, parent_index, Path(path).name.lower())
        except ValueError:
            return (1, parent_name, Path(path).name.lower())

    return sorted(file_paths, key=sort_key)


def load_audio(path, target_sample_rate: int):
    # load audio and convert to mono
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # resample audio according to target sample rate
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    return waveform


parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="Path to the config file")
parser.add_argument(
    "--num_bins", type=int, default=10, help="Number of bins for discretization (must evenly divide 100)"
)
if __name__ == "__main__":
    args = parser.parse_args()
    config_path = args.config_path
    num_bins = args.num_bins
    if num_bins <= 0 or 100 % num_bins != 0:
        parser.error("--num_bins must be a positive integer that divides 100 evenly")

    config_path = Path(config_path)
    default_path = Path(__file__).parent / "config_default.yaml"

    # Load and merge configurations
    cfg = load_yaml(default_path)
    experiment_cfg = load_yaml(config_path)
    clap_cfg = recursive_merge(cfg, experiment_cfg)["clap_config"]
    clap_cfg.update(cfg["shared"])
    clap_cfg = ClapConfig(**clap_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clap_wrapper = ClapWrapper(device=device, model_name=clap_cfg.model_name, sample_rate=clap_cfg.sample_rate)

    # let's go!
    sample_pack_root = clap_cfg.sample_pack_root
    wav_files = glob(f"{sample_pack_root}/**/*.[Ww][Aa][Vv]", recursive=True)
    print(f"Total: {len(wav_files)}")

    reference_root = clap_cfg.reference_root
    reference_files = glob(f"{reference_root}/**/*.[Ww][Aa][Vv]", recursive=True)
    reference_files = sort_paths_by_parent_folder(reference_files)
    print(f"Total: {len(reference_files)}")

    # Encode reference files with CLAP

    reference_dict = {k: [] for k in range(35, 82)}
    reference_dict.update({421: []})  # electric hi-hat
    pbar = tqdm(total=len(reference_files), desc="Encoding reference files")
    for i in range(0, len(reference_files), clap_cfg.batch_size):
        batch = [
            audio_process.normalize(load_audio(file, clap_cfg.sample_rate))
            for file in reference_files[i : i + clap_cfg.batch_size]
        ]
        embeddings = clap_wrapper.get_audio_features(batch)
        for file, embedding in zip(reference_files[i : i + clap_cfg.batch_size], embeddings):
            reference_dict[int(Path(file).parent.name)].append(embedding.detach())
        pbar.update(len(batch))
    pbar.close()
    # compute means only for non-empty reference groups
    non_empty_keys = [k for k, v in reference_dict.items() if len(v) > 0]
    if len(non_empty_keys) == 0:
        raise RuntimeError("No reference embeddings found. Please check reference_root.")
    mean_reference_embeddings = {k: torch.mean(torch.stack(reference_dict[k]), dim=0) for k in non_empty_keys}
    reference_embeddings = torch.stack([mean_reference_embeddings[k] for k in non_empty_keys])
    print(f"Reference embeddings shape: {reference_embeddings.shape}")

    # Encode sample packs with CLAP
    sample_pack_embeddings = []
    pbar_sp = tqdm(total=len(wav_files), desc="Encoding sample pack files")
    for i in range(0, len(wav_files), clap_cfg.batch_size):
        batch = [
            audio_process.normalize(load_audio(file, clap_cfg.sample_rate))
            for file in wav_files[i : i + clap_cfg.batch_size]
        ]
        embeddings = clap_wrapper.get_audio_features(batch)
        for file, embedding in zip(wav_files[i : i + clap_cfg.batch_size], embeddings):
            sample_pack_embeddings.append(embedding.detach())
        pbar_sp.update(len(batch))
    pbar_sp.close()
    sample_pack_embeddings = torch.stack(sample_pack_embeddings)
    print(f"Sample pack embeddings shape: {sample_pack_embeddings.shape}")

    # Compute cosine similarity for each single embedding in reference_embeddings
    similarity = []
    scores = []
    pbar_sim = tqdm(total=len(reference_embeddings) * len(wav_files), desc="Computing cosine similarity")
    for embedding, ref_label in zip(reference_embeddings, non_empty_keys):
        similarity = torch.nn.functional.cosine_similarity(sample_pack_embeddings, embedding, dim=1)
        sim_list = similarity.tolist()
        for sample_idx, score_value in enumerate(sim_list):
            scores.append((ref_label, wav_files[sample_idx], float(score_value)))
        pbar_sim.update(len(sim_list))
    pbar_sim.close()
    # sort scores by score
    scores.sort(key=lambda x: x[2], reverse=True)

    # Prepare augmented output root
    augmented_root = Path(f"{reference_root}_clap_augmented")
    if augmented_root.exists():
        shutil.rmtree(augmented_root)
    augmented_root.mkdir(parents=True, exist_ok=True)

    # define helper to bin scores into string labels
    bin_size = 100 // num_bins

    def score_to_bin_label(score_value: float) -> str:
        # convert cosine similarity [-1, 1] to percentage [0, 100]
        pct = int(round((max(min(score_value, 1.0), -1.0) + 1.0) * 50.0))
        # map to bin index [0, num_bins-1]
        bin_idx = min(pct // bin_size, num_bins - 1)
        lower = bin_idx * bin_size
        upper = (bin_idx + 1) * bin_size
        return f"{upper}-{lower}"

    # copy files in descending score order, ensuring each source file is copied at most once
    seen_sample_paths = set()
    copied_count = 0
    skipped_count = 0
    # add tqdm progress bar
    pbar = tqdm(total=len(scores), desc="Copying files")
    for ref_label, sample_path, score_value in scores:
        if sample_path in seen_sample_paths:
            skipped_count += 1
            continue
        bin_label = score_to_bin_label(score_value)
        dest_dir = augmented_root / str(ref_label) / bin_label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / Path(sample_path).name
        try:
            shutil.copy2(sample_path, dest_path)
            seen_sample_paths.add(sample_path)
            copied_count += 1
        except Exception as e:
            print(f"Failed to copy {sample_path} -> {dest_path}: {e}")
        pbar.update(1)
    pbar.close()
    print(f"Copied: {copied_count}, Skipped (duplicates): {skipped_count}")
