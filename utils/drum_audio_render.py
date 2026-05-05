"""Drum preview: PrettyMIDI export + one-shot waveform rendering.

Each note is rendered only if a matching WAV exists under ``one-shot-rendering/<pitch>/``.
Notes without a sample are skipped (silent). MIDI export uses ``pretty_midi``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union
import glob
import os

import numpy as np
import torch


try:
    import pretty_midi
except ImportError:  # pragma: no cover
    pretty_midi = None  # type: ignore


def _velocity_to_midi(vel: float) -> int:
    v = float(vel)
    if v <= 1.0:
        v = int(round(v * 127))
    else:
        v = int(round(v))
    return int(np.clip(v, 1, 127))


def notes_to_pretty_midi(
    notes: Union[np.ndarray, torch.Tensor],
    name: str = "ADT drums",
) -> "pretty_midi.PrettyMIDI":
    if pretty_midi is None:
        raise RuntimeError("pretty_midi is required for MIDI export.")
    arr = (
        notes.detach().cpu().numpy()
        if isinstance(notes, torch.Tensor)
        else np.asarray(notes, dtype=np.float64)
    )
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True, name=name)
    for row in arr:
        onset, offset, pitch, vel = (
            float(row[0]),
            float(row[1]),
            int(row[2]),
            float(row[3]),
        )
        if offset <= onset:
            offset = onset + 0.05
        inst.notes.append(
            pretty_midi.Note(
                velocity=_velocity_to_midi(vel),
                pitch=int(np.clip(pitch, 0, 127)),
                start=onset,
                end=offset,
            )
        )
    pm.instruments.append(inst)
    return pm


def save_drum_midi(
    notes: Union[np.ndarray, torch.Tensor], path: Union[str, Path]
) -> None:
    pm = notes_to_pretty_midi(notes)
    pm.write(str(path))


import wave

_ONESHOT_CACHE = {}


def get_oneshot_waveform(
    pitch: int, sample_rate: int, apply_mapping: bool = True
) -> Optional[np.ndarray]:
    global _ONESHOT_CACHE
    if not _ONESHOT_CACHE:
        try:
            from .mapping_utils import MappingUtils
            import torchaudio

            base_dir = Path(__file__).parent.parent / "one-shot-rendering"
            if base_dir.exists():
                for d in os.listdir(base_dir):
                    if not d.isdigit():
                        continue
                    gm_custom = int(d)
                    wavs = glob.glob(str(base_dir / d / "*.wav"))
                    if wavs:
                        wav_path = wavs[0]
                        data, sr = torchaudio.load(wav_path)

                        # Convert to mono if needed
                        if data.shape[0] > 1:
                            data = data.mean(dim=0, keepdim=True)

                        # Resample if needed
                        if sr != sample_rate:
                            resampler = torchaudio.transforms.Resample(
                                orig_freq=sr, new_freq=sample_rate
                            )
                            data = resampler(data)

                        _ONESHOT_CACHE[gm_custom] = (
                            data.squeeze().numpy().astype(np.float32)
                        )
        except Exception as e:
            print(f"Failed to load oneshot samples: {e}")
            pass

    custom_pitch = pitch
    if apply_mapping:
        try:
            from .mapping_utils import MappingUtils

            mapping = MappingUtils().GM_standard_midi_to_Gm_custom_Mapping
            custom_pitch = mapping.get(pitch, pitch)
        except Exception:
            pass

    return _ONESHOT_CACHE.get(custom_pitch, None)


def synthesize_drums_procedural(
    notes: Union[np.ndarray, torch.Tensor],
    num_samples: int,
    sample_rate: int,
    apply_mapping: bool = True,
) -> np.ndarray:
    """Sum one-shot WAV hits into a mono float32 buffer. Skips notes with no sample."""
    arr = (
        notes.detach().cpu().numpy()
        if isinstance(notes, torch.Tensor)
        else np.asarray(notes, dtype=np.float64)
    )
    if arr.size == 0:
        return np.zeros(num_samples, dtype=np.float32)
    buf = np.zeros(num_samples, dtype=np.float32)
    max_s = num_samples / float(sample_rate)

    for row in arr:
        onset, offset, pitch, vel = (
            float(row[0]),
            float(row[1]),
            int(row[2]),
            float(row[3]),
        )
        if onset >= max_s:
            continue

        i0 = int(onset * sample_rate)
        if i0 >= num_samples:
            continue

        hit = get_oneshot_waveform(pitch, sample_rate, apply_mapping=apply_mapping)
        if hit is None:
            continue

        n = min(len(hit), num_samples - i0)
        if n > 0:
            g = float(np.clip(vel if vel > 1.0 else vel * 127.0, 1.0, 127.0)) / 127.0
            buf[i0 : i0 + n] += hit[:n] * g

    peak = np.abs(buf).max()
    if peak > 1e-6:
        buf *= min(1.0, 0.98 / peak)
    return buf


def render_drum_preview(
    notes: Union[np.ndarray, torch.Tensor],
    num_samples: int,
    sample_rate: int,
    midi_path: Optional[Union[str, Path]] = None,
    apply_mapping: bool = False,
) -> Tuple[torch.Tensor, str]:
    """
    Optionally write MIDI; audio is one-shot WAVs only (missing samples are silent).

    Returns
    -------
    waveform : (num_samples,) float32 torch tensor
    mode : "oneshot"
    """
    if midi_path is not None:
        save_drum_midi(notes, midi_path)
    wav = synthesize_drums_procedural(
        notes, num_samples, sample_rate, apply_mapping=apply_mapping
    )
    return torch.from_numpy(wav), "oneshot"
