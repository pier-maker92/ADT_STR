import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from glob import glob

import h5py
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from utils.audio_utils import normalize


def load_audio(path: str, target_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    return waveform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_root",
        type=str,
        help="Path to augmented dataset root (e.g., /path/to/GM_Mapped_Reduced_clap_augmented)",
    )
    parser.add_argument(
        "output_hdf5",
        type=str,
        help="Path to output HDF5 file (will be created)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Target sample rate for audio resampling (default: 44100)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing output HDF5 file",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    output_path = Path(args.output_hdf5)
    if output_path.exists():
        if args.overwrite:
            output_path.unlink()
        else:
            raise FileExistsError(f"Output file exists: {output_path}. Use --overwrite to replace.")

    wav_files = glob(str(input_root / "**" / "*.[Ww][Aa][Vv]"), recursive=True)
    wav_files.sort()
    print(f"Found {len(wav_files)} WAV files under {input_root}")

    # Create HDF5 with gzip compression; one fixed-length 1D float32 dataset per file
    with h5py.File(output_path, "w") as h5:
        index_group = h5.create_group("index")
        index_paths = index_group.create_dataset(
            "paths", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(), compression="gzip"
        )
        index_labels = index_group.create_dataset(
            "labels", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(), compression="gzip"
        )
        index_bins = index_group.create_dataset(
            "bins", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(), compression="gzip"
        )
        index_sample_rates = index_group.create_dataset(
            "sample_rates", shape=(0,), maxshape=(None,), dtype="i4", compression="gzip"
        )
        index_lengths = index_group.create_dataset(
            "lengths", shape=(0,), maxshape=(None,), dtype="i8", compression="gzip"
        )

        num_written = 0
        pb = tqdm(wav_files, desc="Converting to HDF5")
        for wav_path in pb:
            rel = Path(wav_path).relative_to(input_root)
            # Expect structure: <label>/<bin>/<filename>
            if len(rel.parts) < 3:
                # Skip unexpected structure
                continue
            instrument_label = rel.parts[0]
            bin_label = rel.parts[1]
            file_stem = Path(wav_path).stem

            try:
                waveform = load_audio(wav_path, args.sample_rate)
                waveform = normalize(waveform)
                audio_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
            except Exception as e:
                print(f"Failed to load '{wav_path}': {e}")
                continue

            # Ensure groups exist
            grp_label = h5.require_group(instrument_label)
            grp_bin = grp_label.require_group(bin_label)

            # Ensure unique dataset name
            ds_name = file_stem
            suffix_idx = 1
            while ds_name in grp_bin:
                suffix_idx += 1
                ds_name = f"{file_stem}_{suffix_idx}"

            ds = grp_bin.create_dataset(ds_name, data=audio_np, dtype="float32", compression="gzip")
            ds.attrs["sample_rate"] = args.sample_rate
            ds.attrs["path"] = str(rel)
            ds.attrs["label"] = instrument_label
            ds.attrs["bin"] = bin_label
            ds.attrs["num_samples"] = int(audio_np.shape[0])

            # Append to flat index
            new_size = num_written + 1
            index_paths.resize((new_size,))
            index_labels.resize((new_size,))
            index_bins.resize((new_size,))
            index_sample_rates.resize((new_size,))
            index_lengths.resize((new_size,))
            index_paths[num_written] = str(rel)
            index_labels[num_written] = instrument_label
            index_bins[num_written] = bin_label
            index_sample_rates[num_written] = int(args.sample_rate)
            index_lengths[num_written] = int(audio_np.shape[0])
            num_written = new_size

            pb.set_postfix({"written": num_written})
        pb.close()

    print(f"Done. Wrote {num_written} items to {output_path}")


if __name__ == "__main__":
    main()
