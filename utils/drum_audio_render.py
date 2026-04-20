"""Drum preview: PrettyMIDI export + audio without HDF5/oneshot samples.

Recommended GM-capable .sf2 soundfonts (better drums than pretty_midi's tiny default):

- **GeneralUser GS** (~32 MB): very FluidSynth-friendly, multiple drum kits, good
  balance for GM playback. Download the repo file:
  https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/GeneralUser-GS.sf2
  (or clone https://github.com/mrbumpy409/GeneralUser-GS )

- **Matrix SoundFont** (~1 GB): large free GM/GS-oriented bank with several drum kits
  (incl. acoustic-style kits). Page with downloads:
  https://musical-artifacts.com/artifacts/3912

- **FluidR3_GM** (~142 MB “Professional”): classic FluidSynth-era GM+many drum kits;
  widely used reference. Example listing:
  https://musical-artifacts.com/artifacts/738

Use with the YouTube pipeline: ``pip install pyfluidsynth`` and a system FluidSynth
library (e.g. ``brew install fluid-synth`` on macOS), then
``--soundfont /path/to/GeneralUser-GS.sf2``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

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
    arr = notes.detach().cpu().numpy() if isinstance(notes, torch.Tensor) else np.asarray(notes, dtype=np.float64)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True, name=name)
    for row in arr:
        onset, offset, pitch, vel = float(row[0]), float(row[1]), int(row[2]), float(row[3])
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


def save_drum_midi(notes: Union[np.ndarray, torch.Tensor], path: Union[str, Path]) -> None:
    pm = notes_to_pretty_midi(notes)
    pm.write(str(path))


def _hit_waveform(pitch: int, n: int, sr: int, vel: float, rng: np.random.Generator) -> np.ndarray:
    """Short GM-ish hit waveform (preview only, not real samples)."""
    t = np.arange(n, dtype=np.float64) / sr
    g = float(np.clip(vel if vel > 1.0 else vel * 127.0, 1.0, 127.0)) / 127.0
    p = int(pitch)

    if p in (35, 36):
        f0 = 60.0
        return (g * np.sin(2 * np.pi * f0 * t) * np.exp(-12.0 * t)).astype(np.float32)
    if p in (37, 38, 39, 40):
        noise = rng.standard_normal(n).astype(np.float32)
        env = np.exp(-22.0 * t).astype(np.float32)
        tone = np.sin(2 * np.pi * 200.0 * t) * np.exp(-14.0 * t)
        return (g * (0.55 * noise * env + 0.35 * tone.astype(np.float32))).astype(np.float32)
    if p in (41, 43, 45, 47, 48, 50):
        f0 = 100.0 + (p % 7) * 15.0
        return (g * np.sin(2 * np.pi * f0 * t) * np.exp(-10.0 * t)).astype(np.float32)
    if p in (42, 44, 46):
        noise = rng.standard_normal(n).astype(np.float32)
        prev = np.concatenate([[0.0], noise[:-1]]) if n > 1 else np.zeros_like(noise)
        hp = (noise - prev).astype(np.float32)
        env = np.exp(-45.0 * t).astype(np.float32)
        return (g * 0.35 * hp * env).astype(np.float32)
    # cymbals / other
    noise = rng.standard_normal(n).astype(np.float32)
    env = np.exp(-4.5 * t).astype(np.float32)
    return (g * 0.25 * noise * env).astype(np.float32)


def synthesize_drums_procedural(
    notes: Union[np.ndarray, torch.Tensor],
    num_samples: int,
    sample_rate: int,
    seed: int = 0,
) -> np.ndarray:
    """Sum elementary hits into a mono float32 buffer in [-1, 1]."""
    arr = notes.detach().cpu().numpy() if isinstance(notes, torch.Tensor) else np.asarray(notes, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(num_samples, dtype=np.float32)
    rng = np.random.default_rng(seed)
    buf = np.zeros(num_samples, dtype=np.float32)
    max_s = num_samples / float(sample_rate)

    for row in arr:
        onset, offset, pitch, vel = float(row[0]), float(row[1]), int(row[2]), float(row[3])
        if onset >= max_s:
            continue
        dur = max(offset - onset, 2.0 / sample_rate)
        dur = min(dur, 0.35)
        i0 = int(onset * sample_rate)
        n = int(np.ceil(dur * sample_rate))
        if i0 >= num_samples:
            continue
        n = min(n, num_samples - i0)
        if n <= 0:
            continue
        hit = _hit_waveform(pitch, n, sample_rate, vel, rng)
        buf[i0 : i0 + n] += hit

    peak = np.abs(buf).max()
    if peak > 1e-6:
        buf *= min(1.0, 0.98 / peak)
    return buf


def synthesize_drums_fluidsynth(
    notes: Union[np.ndarray, torch.Tensor],
    num_samples: int,
    sample_rate: int,
    soundfont_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Return audio if pyfluidsynth + soundfont work; otherwise None."""
    if pretty_midi is None:
        return None
    try:
        pm = notes_to_pretty_midi(notes)
        audio = pm.fluidsynth(fs=sample_rate, synthesizer=soundfont_path)
    except Exception:
        return None
    if audio is None or audio.size == 0:
        return None
    if audio.shape[0] < num_samples:
        out = np.zeros(num_samples, dtype=np.float32)
        out[: audio.shape[0]] = audio.astype(np.float32)
    else:
        out = audio[:num_samples].astype(np.float32)
    peak = np.abs(out).max()
    if peak > 1e-6:
        out *= min(1.0, 0.98 / peak)
    return out


def render_drum_preview(
    notes: Union[np.ndarray, torch.Tensor],
    num_samples: int,
    sample_rate: int,
    midi_path: Optional[Union[str, Path]] = None,
    soundfont_path: Optional[str] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, str]:
    """
    Optionally write MIDI; audio is fluidsynth when possible else procedural synthesis.

    Returns
    -------
    waveform : (num_samples,) float32 torch tensor
    mode : "fluidsynth" | "procedural"
    """
    if midi_path is not None:
        save_drum_midi(notes, midi_path)
    wav = synthesize_drums_fluidsynth(notes, num_samples, sample_rate, soundfont_path=soundfont_path)
    if wav is not None:
        return torch.from_numpy(wav), "fluidsynth"
    wav = synthesize_drums_procedural(notes, num_samples, sample_rate, seed=seed)
    return torch.from_numpy(wav), "procedural"
