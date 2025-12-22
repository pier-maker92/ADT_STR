import subprocess
import os
from glob import glob
from pathlib import Path


def separate_drums(files, output_dir="demucs_output", model="htdemucs"):
    """
    Separate only the drums stem from a list of audio files using Demucs.

    Parameters
    ----------
    files : list of str
        List of paths to input audio files.
    output_dir : str
        Directory where separated stems will be saved.
    model : str
        Demucs model to use (default: htdemucs).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        file_path = Path(file)
        print(f"Processing: {file_path}")

        # Run Demucs
        subprocess.run(
            [
                "demucs",
                "-n",
                model,
                "--two-stems",
                "drums",  # Extract only drums
                "-o",
                str(output_dir),
                str(file_path),
            ],
            check=True,
        )

        # Move/rename result for convenience
        song_name = file_path.stem
        drum_path = output_dir / model / song_name / "drums.wav"
        final_path = output_dir / f"{song_name}_drums.wav"

        if drum_path.exists():
            drum_path.rename(final_path)
            print(f"Saved drums: {final_path}")
        else:
            print(f"Warning: Drums not found for {file_path}")


ENST_SPLITS = [
    "162_MIDI-minus-one_fusion-125_sticks",
    "126_minus-one_salsa_sticks",
    "127_minus-one_rock-60s_sticks",
    "128_minus-one_metal_sticks",
    "129_minus-one_musette_sticks",
    "130_minus-one_funky_sticks",
    "131_minus-one_funk_sticks",
    "132_minus-one_charleston_sticks",
    "133_minus-one_celtic-rock_sticks",
    "134_minus-one_bossa_sticks",
    "140_MIDI-minus-one_bigband_sticks",
    "142_MIDI-minus-one_blues-102_sticks",
    "144_MIDI-minus-one_country-120_sticks",
    "146_MIDI-minus-one_disco-108_sticks",
    "148_MIDI-minus-one_funk-101_sticks",
    "150_MIDI-minus-one_grunge_sticks",
    "152_MIDI-minus-one_nu-soul_sticks",
    "154_MIDI-minus-one_rock-113_sticks",
    "156_MIDI-minus-one_rock'n'roll-188_sticks",
    "158_MIDI-minus-one_soul-120-marvin-gaye_sticks",
    "160_MIDI-minus-one_soul-98_sticks",
]
# Example usage
if __name__ == "__main__":
    all_files = glob("/home/ach18017ws/MDBDrums/MDB Drums/audio/full_mix/*.wav", recursive=True)
    # files = [f for f in all_files if any(split == f.split("/")[-1].split(".")[0] for split in ENST_SPLITS)]
    separate_drums(all_files, output_dir="/home/ach18017ws/MDBDrums/MDB Drums/audio/demucs_separated")
