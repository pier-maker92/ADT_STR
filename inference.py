import os
import torch
import argparse
import torchaudio
import pretty_midi
from pathlib import Path

from adt_transcriber import ADTTranscriber
from utils.utils import select_inference_device


def main():
    parser = argparse.ArgumentParser(description="Minimal ADT Inference Script")
    parser.add_argument("positional", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument(
        "--input",
        "--input_path",
        dest="input_path",
        type=str,
        default=None,
        help="Path to input audio file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint name under checkpoints/ or a checkpoint directory path.",
    )
    parser.add_argument(
        "--config",
        "--config_path",
        dest="config_path",
        type=str,
        default=None,
        help="Path to model config YAML/JSON. Kept for backward compatibility.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="outputs/",
        help="Directory to save output files",
    )
    parser.add_argument(
        "-s",
        "--synthetise_transcription",
        action="store_true",
        help="Resynthesize the drum transcription",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for inference (overrides config)",
    )
    args = parser.parse_args()

    if args.positional:
        if len(args.positional) != 2:
            parser.error(
                "Use either: python inference.py --checkpoint <name> --input <audio> "
                "or the legacy form: python inference.py <input_path> <config_path>"
            )
        if args.input_path is None:
            args.input_path = args.positional[0]
        if args.config_path is None and args.checkpoint is None:
            args.config_path = args.positional[1]

    if not args.input_path:
        parser.error("--input is required")
    if not args.checkpoint and not args.config_path:
        parser.error("--checkpoint or --config is required")

    os.makedirs(args.output_path, exist_ok=True)
    input_stem = Path(args.input_path).stem

    device = select_inference_device()
    print(f"Using device: {device}")

    if args.checkpoint:
        transcriber = ADTTranscriber.from_checkpoint(args.checkpoint, device=device)
    else:
        transcriber = ADTTranscriber.from_config(args.config_path, device=device)

    print(f"Loading audio: {args.input_path}")
    midi_path = transcriber.transcribe(
        args.input_path,
        output_dir=args.output_path,
        batch_size=args.batch_size,
    )
    notes = pretty_midi.PrettyMIDI(str(midi_path)).instruments[0].notes
    print(f"Transcription saved to: {midi_path} ({len(notes)} notes)")

    if args.synthetise_transcription:
        if len(notes) == 0:
            print("No notes transcribed, skipping synthesis.")
        else:
            print("Synthesizing transcription...")
            from modules.synthetiser import SynthDrum, SynthDrumConfig

            synth_section = transcriber.cfg.get("synthetiser", {})
            synth_section.update(transcriber.cfg.get("shared", {}))

            try:
                synth_config = SynthDrumConfig(**synth_section)
                synthesizer = SynthDrum(synth_config)
                note_rows = [
                    [note.start, note.end, note.pitch, note.velocity] for note in notes
                ]
                resynth_audio = synthesizer(torch.tensor(note_rows))
                resynth_path = os.path.join(
                    args.output_path, f"{input_stem}_resynth.wav"
                )
                torchaudio.save(
                    resynth_path,
                    resynth_audio.unsqueeze(0),
                    transcriber.sample_rate,
                )
                print(f"Resynthesized audio saved to: {resynth_path}")
            except Exception as e:
                print(f"Synthesis failed: {e}")


if __name__ == "__main__":
    main()
