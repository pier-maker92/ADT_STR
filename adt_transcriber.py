from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torchaudio
from huggingface_hub import snapshot_download

from build_model import build_model, build_model_from_checkpoint
from model import ADTModel
from modules.midi_tokenizer import MidiTokenizer, MidiTokenizerConfig
from utils.config_utils import load_config
from utils.utils import select_inference_device


def save_midi(notes, output_path):
    """Save transcription notes to a MIDI file."""
    midi = pretty_midi.PrettyMIDI()
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
    """Divide audio into fixed-size chunks, padding the last one if needed."""
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


class ADTTranscriber:
    """Small inference helper for local checkpoints and Hugging Face model repos."""

    def __init__(self, model, cfg, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.cfg = cfg
        self.tokenizer = MidiTokenizer(MidiTokenizerConfig(**cfg.get("tokenizer")))
        shared = cfg.get("shared", {})
        self.sample_rate = int(shared.get("sample_rate"))
        self.input_sec = float(shared.get("input_sec"))
        self.chunk_samples = int(round(self.input_sec * self.sample_rate))

    @classmethod
    def from_checkpoint(cls, checkpoint, device=None):
        device = device or select_inference_device()
        model, cfg = build_model_from_checkpoint(checkpoint, device=device)
        return cls(model, cfg, device=device)

    @classmethod
    def from_config(cls, config_path, device=None):
        device = device or select_inference_device()
        model, cfg = build_model(config_path, device=device)
        return cls(model, cfg, device=device)

    @classmethod
    def from_pretrained(cls, repo_id, variant=None, device=None, revision=None):
        device = device or select_inference_device()
        repo_path = Path(repo_id)
        local_dir = repo_path if repo_path.exists() else Path(
            snapshot_download(repo_id=repo_id, revision=revision)
        )
        model_dir = local_dir / variant if variant else local_dir
        cfg_path = model_dir / "adt_config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Expected full ADT config at {cfg_path}. "
                "Use a repo exported with push_to_hf.py."
            )
        cfg = load_config(str(cfg_path))
        model = ADTModel.from_pretrained(str(model_dir))
        return cls(model, cfg, device=device)

    def _load_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(str(audio_path))
        return self._prepare_waveform(waveform, sample_rate)

    def _prepare_waveform(self, waveform, sample_rate):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)
        if waveform.dim() != 2:
            raise ValueError(
                "Expected audio tensor with shape (samples,), (channels, samples), "
                "or batch tensor with shape (batch, samples)."
            )
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.to(self.device)

    def _transcribe_waveform(self, waveform, batch_size=None, max_length=None):
        chunks = _chunk_audio(waveform, self.chunk_samples)
        all_notes = []
        batch_size = batch_size or self.cfg.get("inference", {}).get("batch_size", 8)
        max_length = max_length or self.cfg.get("inference", {}).get("max_length", 1024)

        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_src = torch.cat([c[1] for c in batch_chunks], dim=0)
                tokens_batch = self.model.sample(
                    src=batch_src,
                    src_mask=None,
                    tgt_mask=None,
                    max_length=max_length,
                    start_token=self.tokenizer.BOS_token,
                    end_token=self.tokenizer.EOS_token,
                )

                for start_sample, tokens in zip([c[0] for c in batch_chunks], tokens_batch):
                    chunk_notes = self.tokenizer.decode(tokens.cpu().numpy())
                    if chunk_notes.numel() > 0:
                        chunk_notes[:, 0:2] += start_sample / self.sample_rate
                        all_notes.append(chunk_notes)

        if not all_notes:
            return torch.zeros((0, 4))

        notes = torch.cat(all_notes, dim=0)
        return torch.from_numpy(np.unique(notes.cpu().numpy(), axis=0))

    def _transcribe_path(
        self,
        audio_path,
        midi_path,
        batch_size=None,
        max_length=None,
    ):
        notes = self._transcribe_waveform(
            self._load_audio(audio_path),
            batch_size=batch_size,
            max_length=max_length,
        )
        save_midi(notes.cpu().numpy(), midi_path)
        return midi_path

    def transcribe(
        self,
        audio,
        output_dir="outputs",
        sample_rate=None,
        lengths=None,
        batch_size=None,
        max_length=None,
    ):
        """Transcribe a path, list of paths, or padded audio tensor batch to MIDI."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(audio, (str, Path)):
            midi_path = output_dir / f"{Path(audio).stem}.mid"
            return self._transcribe_path(
                audio,
                midi_path,
                batch_size=batch_size,
                max_length=max_length,
            )

        if isinstance(audio, Sequence) and not isinstance(audio, torch.Tensor):
            midi_paths = []
            for idx, item in enumerate(audio):
                midi_path = output_dir / f"{idx:03d}_{Path(item).stem}.mid"
                midi_paths.append(
                    self._transcribe_path(
                        item,
                        midi_path,
                        batch_size=batch_size,
                        max_length=max_length,
                    )
                )
            return midi_paths

        if not isinstance(audio, torch.Tensor):
            raise TypeError("audio must be a path, a list of paths, or a torch.Tensor.")

        tensor = audio
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 3 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.dim() != 2:
            raise ValueError(
                "Tensor input must have shape (samples,), (batch, samples), "
                "or (batch, 1, samples)."
            )

        sample_rate = sample_rate or self.sample_rate
        lengths = lengths or [tensor.shape[-1]] * tensor.shape[0]
        midi_paths = []
        for idx, (waveform, length) in enumerate(zip(tensor, lengths)):
            waveform = waveform[: int(length)].unsqueeze(0)
            waveform = self._prepare_waveform(waveform, sample_rate)
            notes = self._transcribe_waveform(
                waveform, batch_size=batch_size, max_length=max_length
            )
            midi_path = output_dir / f"audio_{idx:03d}.mid"
            save_midi(notes.cpu().numpy(), midi_path)
            midi_paths.append(midi_path)
        return midi_paths[0] if audio.dim() == 1 else midi_paths

    def _render_midi_preview(self, midi_path, sample_rate=44100):
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        notes = [note for inst in midi.instruments for note in inst.notes]
        if not notes:
            return np.zeros(sample_rate, dtype=np.float32)

        length = int((max(note.end for note in notes) + 0.75) * sample_rate)
        audio = np.zeros(length, dtype=np.float32)
        rng = np.random.default_rng(1234)

        def envelope(size, decay, attack=0.001):
            t = np.arange(size) / sample_rate
            return np.exp(-decay * t) * np.minimum(1.0, t / attack)

        def add(start, signal):
            idx = max(0, int(start * sample_rate))
            end = min(length, idx + len(signal))
            if end > idx:
                audio[idx:end] += signal[: end - idx]

        def noise(size):
            x = rng.normal(0, 1, size)
            kernel = max(2, int(sample_rate * 0.0015))
            return x - np.convolve(x, np.ones(kernel) / kernel, mode="same")

        for note in notes:
            velocity = max(0.05, min(1.0, note.velocity / 127.0))
            pitch = int(note.pitch)
            if pitch in (35, 36):
                size = int(0.38 * sample_rate)
                t = np.arange(size) / sample_rate
                freq = 38 + (95 - 38) * np.exp(-18 * t)
                signal = np.sin(2 * np.pi * np.cumsum(freq) / sample_rate)
                signal = signal * envelope(size, 7.5) * velocity
            elif pitch in (42, 44, 46):
                size = int((0.36 if pitch == 46 else 0.11) * sample_rate)
                signal = noise(size) * envelope(size, 9 if pitch == 46 else 38) * velocity * 0.22
            elif pitch in (49, 51, 52, 53, 55, 57, 59):
                size = int(0.9 * sample_rate)
                signal = noise(size) * envelope(size, 2.5) * velocity * 0.28
            else:
                size = int(0.28 * sample_rate)
                t = np.arange(size) / sample_rate
                tone = np.sin(2 * np.pi * 185 * t) * envelope(size, 15)
                signal = (0.65 * noise(size) * envelope(size, 12) + 0.35 * tone) * velocity * 0.7
            add(note.start, signal)

        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = np.tanh(audio * 1.4)
            audio = audio / max(float(np.max(np.abs(audio))), 1e-8) * 0.9
        return audio.astype(np.float32)

    def play(self, midi_path, sample_rate=44100, output_path=None):
        """Return an IPython audio widget, optionally saving a WAV preview."""
        audio = self._render_midi_preview(midi_path, sample_rate=sample_rate)
        if output_path is not None:
            torchaudio.save(
                str(output_path),
                torch.from_numpy(audio).unsqueeze(0),
                sample_rate,
            )
        try:
            from IPython.display import Audio

            return Audio(audio, rate=sample_rate)
        except ImportError:
            return audio, sample_rate
