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

# Example usage
if __name__ == "__main__":
    all_files = glob("path/to/your/audio/files/*.wav", recursive=True)
    separate_drums(all_files, output_dir="path/to/your/output/directory")
