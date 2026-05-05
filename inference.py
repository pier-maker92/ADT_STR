import os
import torch
import torchaudio
import argparse
import numpy as np
import pretty_midi
from pathlib import Path
from build_model import build_model
from modules.midi_tokenizer import MidiTokenizer, MidiTokenizerConfig
from modules.synthetiser import SynthDrum, SynthDrumConfig
from utils.utils import select_inference_device


def save_midi(notes, output_path):
    """Saves transcription notes to a MIDI file."""
    midi = pretty_midi.PrettyMidi()
    drums = pretty_midi.Instrument(program=0, is_drum=True)

    for note_data in notes:
        if len(note_data) < 4:
            continue
        onset, offset, pitch, velocity = note_data
        note = pretty_midi.Note(
            velocity=int(max(0, min(127, velocity))),
            pitch=int(pitch),
            start=float(onset),
            end=float(offset),
        )
        drums.notes.append(note)

    midi.instruments.append(drums)
    midi.write(output_path)


def _chunk_audio(wav, chunk_samples):
    """Divide audio into chunks of fixed size, padding the last one if needed."""
    chunks = []
    n_samples = wav.shape[-1]
    for start in range(0, n_samples, chunk_samples):
        end = min(start + chunk_samples, n_samples)
        chunk = wav[:, start:end]
        if chunk.shape[-1] < chunk_samples:
            pad = torch.zeros(
                (chunk.shape[0], chunk_samples - chunk.shape[-1]), device=chunk.device
            )
            chunk = torch.cat([chunk, pad], dim=-1)
        chunks.append((start, chunk))
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Minimal ADT Inference Script")
    parser.add_argument("input_path", type=str, help="Path to input audio file")
    parser.add_argument("config_path", type=str, help="Path to model config YAML")
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
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    input_stem = Path(args.input_path).stem

    device = select_inference_device()
    print(f"Using device: {device}")

    model, cfg = build_model(args.config_path, device=device)

    tokenizer_config = MidiTokenizerConfig(**cfg.get("tokenizer"))
    tokenizer = MidiTokenizer(tokenizer_config)

    print(f"Loading audio: {args.input_path}")
    waveform, sample_rate = torchaudio.load(args.input_path)
    shared = cfg.get("shared", {})
    target_sr = shared.get("sample_rate", 44100)
    input_sec = float(shared.get("input_sec", 2.56))
    chunk_samples = int(round(input_sec * target_sr))

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.to(device)

    # Chunk audio
    chunks = _chunk_audio(waveform, chunk_samples)
    all_notes = []

    print(f"Running inference on {len(chunks)} chunks...")
    with torch.no_grad():
        for start_sample, chunk in chunks:
            tokens = model.sample(
                src=chunk,
                src_mask=None,
                tgt_mask=None,
                max_length=cfg.get("inference", {}).get("max_length", 1024),
                start_token=tokenizer.BOS_token,
                end_token=tokenizer.EOS_token,
            )

            # Decode tokens and shift time
            tokens_np = tokens[0].cpu().numpy()
            chunk_notes = tokenizer.decode(tokens_np)

            if chunk_notes.numel() > 0:
                t0 = start_sample / target_sr
                chunk_notes[:, 0:2] += t0
                all_notes.append(chunk_notes)

    if len(all_notes) > 0:
        notes = torch.cat(all_notes, dim=0)
        # Remove duplicates
        notes_np = notes.cpu().numpy()
        unique_notes = np.unique(notes_np, axis=0)
        notes = torch.from_numpy(unique_notes)
    else:
        notes = torch.zeros((0, 4))

    midi_path = os.path.join(args.output_path, f"{input_stem}.mid")
    save_midi(notes.cpu().numpy(), midi_path)
    print(f"Transcription saved to: {midi_path} ({len(notes)} notes)")

    if args.synthetise_transcription:
        if len(notes) == 0:
            print("No notes transcribed, skipping synthesis.")
        else:
            print("Synthesizing transcription...")
            synth_section = cfg.get("synthetiser", {})
            synth_section.update(shared)

            try:
                synth_config = SynthDrumConfig(**synth_section)
                synthesizer = SynthDrum(synth_config)
                resynth_audio = synthesizer(notes)
                resynth_path = os.path.join(
                    args.output_path, f"{input_stem}_resynth.wav"
                )
                torchaudio.save(resynth_path, resynth_audio.unsqueeze(0), target_sr)
                print(f"Resynthesized audio saved to: {resynth_path}")
            except Exception as e:
                print(f"Synthesis failed: {e}")


if __name__ == "__main__":
    main()
