#!/usr/bin/env python3
"""
Pipeline: YouTube URL -> audio -> Demucs (drums only) -> ADT inference -> drum preview.

Requirements:
  - yt-dlp (+ ffmpeg) for YouTube download
  - demucs CLI (same usage as data_modules/demucs_seaprate.py)
  - model checkpoint
  - pretty_midi (writes `predicted_drums.mid`; preview audio via lightweight synthesis or fluidsynth if available)

--synth-mapping only remaps pitches to GM / ADTOF before MIDI+audio (to match the tokenizer):
  adtof       -> ADTOF pitch clusters (ADTOF_mapping tokens)
  gm_reduced  -> reduced GM pitches (recommended for GM drum map / fluidsynth)

Uses the same YAML merge as inference.py (config_default + experiment config).
"""

from __future__ import annotations

import argparse
import gc
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ADTModelConfig
from data_modules.demucs_seaprate import separate_drums
from inference import load_model
from model import ADTModel
from modules.midi_tokenizer import MidiTokenizer, MidiTokenizerConfig
from utils.audio_utils import load_and_resample, normalize
from utils.config_utils import deep_merge_dicts, load_config_from_yaml
from utils.drum_audio_render import render_drum_preview
from utils.mapping_utils import MappingUtils
from utils.utils import empty_accelerator_cache, select_inference_device


def _ensure_training_lr(config: Dict[str, Any]) -> None:
    """inference.load_model expects learning_rate in the merged config (same as train.py)."""
    t = config.setdefault("training", {})
    if t.get("learning_rate") is None:
        t["learning_rate"] = 1e-4


def _download_youtube(url: str, out_dir: Path, filename_stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{filename_stem}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--force-overwrites",
        "--no-playlist",
        "-o",
        pattern,
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "yt-dlp not found on PATH. Install it (e.g. pip install yt-dlp) and ensure ffmpeg is available."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed: {e.stderr or e.stdout}") from e

    candidates = sorted(out_dir.glob(f"{filename_stem}.*"))
    wavs = [p for p in candidates if p.suffix.lower() == ".wav"]
    if not wavs:
        raise FileNotFoundError(
            f"No WAV produced in {out_dir} with prefix {filename_stem!r}. Found: {candidates}"
        )
    return wavs[0]


def _load_mono_wav(path: Path, target_sr: int) -> torch.Tensor:
    wav = load_and_resample(str(path), target_sr)
    return normalize(wav)


def _chunk_audio(
    wav: torch.Tensor, chunk_samples: int, hop_samples: Optional[int] = None
) -> List[Tuple[int, torch.Tensor]]:
    """Return (start_sample, chunk) pairs; last chunk is zero-padded to chunk_samples."""
    if hop_samples is None:
        hop_samples = chunk_samples
    out: List[Tuple[int, torch.Tensor]] = []
    n = wav.numel()
    start = 0
    while start < n:
        end = min(start + chunk_samples, n)
        piece = wav[start:end].clone()
        if piece.numel() < chunk_samples:
            pad = torch.zeros(chunk_samples - piece.numel(), dtype=piece.dtype, device=piece.device)
            piece = torch.cat([piece, pad], dim=0)
        out.append((start, piece))
        start += hop_samples
    return out


def _run_model_on_chunks(
    model: ADTModel,
    tokenizer: MidiTokenizer,
    wav: torch.Tensor,
    sample_rate: int,
    input_sec: float,
    device: torch.device,
    beam_size: int,
    use_beam_search: bool,
    max_decode_tokens: int,
    min_audio_samples: int = 1024,
) -> torch.Tensor:
    chunk_samples = int(round(input_sec * sample_rate))
    chunks = _chunk_audio(wav, chunk_samples)
    all_notes: List[List[float]] = []

    sample_fn = model.beam_search if use_beam_search else model.sample
    model.eval()

    with torch.no_grad():
        for start_sample, chunk in tqdm(chunks, desc="Running model on chunks"):
            if chunk.numel() < min_audio_samples:
                continue
            src = chunk.unsqueeze(0).to(device)
            kwargs: Dict[str, Any] = {
                "src": src,
                "src_mask": None,
                "tgt_mask": None,
                "max_length": max(2, int(max_decode_tokens)),
            }
            if use_beam_search:
                kwargs["beam_size"] = beam_size
            tokens_pred = sample_fn(**kwargs)
            pred_tokens = tokens_pred[0].cpu().numpy()
            eos_idx = np.where((pred_tokens == tokenizer.EOS_token) | (pred_tokens == tokenizer.pad_token))[0]
            if len(eos_idx) > 0:
                pred_tokens = pred_tokens[: int(eos_idx[0])]

            pred_notes = tokenizer.decode(pred_tokens)
            if pred_notes.numel() == 0:
                continue
            pred_notes = pred_notes[pred_notes[:, 3] >= 0]
            t0 = start_sample / float(sample_rate)
            for row in pred_notes:
                onset = float(row[0]) + t0
                offset = float(row[1]) + t0
                pitch = float(row[2])
                vel = float(row[3])
                all_notes.append([onset, offset, pitch, vel])

            del src, tokens_pred
            gc.collect()
            empty_accelerator_cache(device)

    if not all_notes:
        return torch.zeros((0, 4), dtype=torch.float32)
    notes = torch.tensor(all_notes, dtype=torch.float32)
    # Drop identical duplicates (same as inference)
    uniq = np.unique(notes.numpy(), axis=0)
    return torch.from_numpy(uniq).float()


def _remap_notes_for_synth(
    notes: torch.Tensor,
    tokenizer_uses_adtof: bool,
    synth_uses_adtof: bool,
    mu: MappingUtils,
) -> torch.Tensor:
    """Remap note pitches for MIDI export / preview audio (GM vs ADTOF)."""
    if notes.numel() == 0:
        return notes
    out = notes.clone()
    if tokenizer_uses_adtof and not synth_uses_adtof:
        for i in range(out.shape[0]):
            p = int(out[i, 2].item())
            if p in mu.ADTOF_inverse_mapping:
                out[i, 2] = float(mu.ADTOF_inverse_mapping[p][0])
    elif (not tokenizer_uses_adtof) and synth_uses_adtof:
        new_p = [float(mu.ADTOF_map.get(int(p), int(p))) for p in out[:, 2].tolist()]
        out[:, 2] = torch.tensor(new_p, dtype=out.dtype)
    return out


def _filter_valid_synth_notes(notes: torch.Tensor) -> torch.Tensor:
    if notes.numel() == 0:
        return notes
    rows = []
    for row in notes:
        onset, offset, pitch, vel = row.tolist()
        p = int(pitch)
        if 35 <= p <= 61 and offset >= onset:
            rows.append([onset, offset, p, vel])
    if not rows:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YouTube / audio -> Demucs drums -> ADT -> drum preview")
    p.add_argument("--url", type=str, default=None, help="YouTube URL (optional if --input-audio)")
    p.add_argument("--input-audio", type=str, default=None, help="Skip download: use this audio file")
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Eval YAML (e.g. configs/eval/ENSTinference.yaml), merged with configs/config_default.yaml",
    )
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/setting-tau-0.4/model.safetensors",
        help="Override inference.checkpoint_path from the YAML",
    )
    p.add_argument("--output-dir", type=str, default="youtube_pipeline_out", help="Output directory")
    p.add_argument(
        "--synth-mapping",
        type=str,
        choices=("adtof", "gm_reduced"),
        default="gm_reduced",
        help="Pitch remap for MIDI/preview: gm_reduced=standard GM (default); adtof=ADTOF clusters",
    )
    p.add_argument(
        "--soundfont",
        type=str,
        default=None,
        help=(
            "Path to a GM .sf2 for pyfluidsynth (needs: pip install pyfluidsynth; system fluid-synth). "
            "Suggested banks: GeneralUser-GS, Matrix SoundFont, FluidR3_GM — see utils/drum_audio_render.py "
            "module docstring for download URLs. If omitted, procedural preview is used."
        ),
    )
    p.add_argument("--demucs-model", type=str, default="htdemucs", help="Demucs model name (-n)")
    p.add_argument(
        "--skip-demucs",
        action="store_true",
        help="Skip Demucs (use --input-audio or the downloaded mix as drums)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stem-name", type=str, default="youtube_track", help="Intermediate filename stem for Demucs")
    p.add_argument(
        "--max-decode-tokens",
        type=int,
        default=256,
        metavar="N",
        help="Max tokens to generate per chunk (sample/beam); lowers risk of long stalls (default 256).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("youtube_drum_pipeline")

    if not args.url and not args.input_audio:
        raise SystemExit("Provide --url or --input-audio")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    default_cfg_path = ROOT / "configs" / "config_default.yaml"
    merged = deep_merge_dicts(load_config_from_yaml(str(default_cfg_path)), load_config_from_yaml(args.config))
    _ensure_training_lr(merged)

    if args.checkpoint_path:
        merged.setdefault("inference", {})["checkpoint_path"] = args.checkpoint_path

    inf = merged.get("inference", {})
    checkpoint_path = inf.get("checkpoint_path")
    if not checkpoint_path:
        raise SystemExit("Missing checkpoint_path: set it in the YAML or use --checkpoint-path")

    shared = merged.get("shared", {})
    sample_rate = int(shared["sample_rate"])
    input_sec = float(shared["input_sec"])

    device = select_inference_device()
    log.info("Device: %s", device)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)

    # 1) Source audio
    if args.input_audio:
        src_audio = Path(args.input_audio).resolve()
        if not src_audio.is_file():
            raise SystemExit(f"Audio file not found: {src_audio}")
        dl_path = src_audio
    else:
        dl_path = _download_youtube(args.url, work, args.stem_name)

    # 2) Demucs
    if args.skip_demucs:
        drums_wav = dl_path
        log.info("Skipping Demucs; using: %s", drums_wav)
    else:
        demucs_out = work / "demucs"
        stem = Path(dl_path).stem
        separate_drums([str(dl_path)], output_dir=str(demucs_out), model=args.demucs_model)
        drums_wav = demucs_out / f"{stem}_drums.wav"
        if not drums_wav.is_file():
            raise FileNotFoundError(
                f"Drums stem not found after Demucs: expected {drums_wav} (check that demucs finished)."
            )

    # 3) Model + tokenizer
    model_section = dict(merged["model"])
    model_section["enc_lr"] = merged["training"]["learning_rate"]
    model_section["dec_lr"] = merged["training"]["learning_rate"]
    model_section.update(shared)
    model_config = ADTModelConfig(**model_section)
    model = load_model(checkpoint_path, model_config, device)

    tok_cfg = MidiTokenizerConfig(**merged["tokenizer"])
    tokenizer = MidiTokenizer(tok_cfg)
    tokenizer_uses_adtof = bool(tok_cfg.ADTOF_mapping)

    beam_size = int(inf.get("beam_size", 5))
    use_beam_search = bool(inf.get("use_beam_search", False))
    max_decode = max(2, int(args.max_decode_tokens))
    if inf.get("max_length") is not None:
        max_decode = min(max_decode, max(2, int(inf["max_length"])))
    log.info("max_length (decode) per chunk: %d", max_decode)

    wav = _load_mono_wav(drums_wav, sample_rate)
    torchaudio.save(str(out_dir / "input_drums_resampled.wav"), wav.unsqueeze(0), sample_rate)

    log.info("Chunked inference (input_sec=%.3f, sr=%d)", input_sec, sample_rate)
    notes = _run_model_on_chunks(
        model,
        tokenizer,
        wav,
        sample_rate,
        input_sec,
        device,
        beam_size=beam_size,
        use_beam_search=use_beam_search,
        max_decode_tokens=max_decode,
    )
    np.save(str(out_dir / "predicted_notes.npy"), notes.numpy())

    synth_adtof = args.synth_mapping == "adtof"
    mu = MappingUtils()
    notes_synth = _remap_notes_for_synth(notes, tokenizer_uses_adtof, synth_adtof, mu)
    notes_synth = _filter_valid_synth_notes(notes_synth)
    log.info("Notes after validity filter: %d", notes_synth.shape[0])
    if notes_synth.numel() == 0:
        log.warning("No notes: empty MIDI and silent WAV.")

    n_audio = wav.numel()
    midi_path = out_dir / "predicted_drums.mid"
    rendered, render_mode = render_drum_preview(
        notes_synth,
        n_audio,
        sample_rate,
        midi_path=midi_path,
        soundfont_path=args.soundfont,
        seed=args.seed,
    )
    log.info("Drum preview (%s): MIDI %s", render_mode, midi_path)

    out_wav = out_dir / "synthesized_drums.wav"
    torchaudio.save(str(out_wav), rendered.unsqueeze(0).float(), sample_rate)
    log.info("Done. Output: %s", out_wav)


if __name__ == "__main__":
    main()
