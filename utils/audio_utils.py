import torch
import random
import torchaudio
import numpy as np
from typing import Union
from config import SharedConfig
import torchaudio.transforms as T


def load_and_resample(wav_file: str, target_sr: Union[int, None]):
    wav_seg, orig_sr = torchaudio.load(wav_file)
    wav_seg = wav_seg.mean(0)
    if target_sr is None:
        return wav_seg
    return resample(wav_seg, orig_sr, target_sr)


def resample(wav_seg: torch.Tensor, orig_sr: int, target_sr: int):
    resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(wav_seg)


def normalize(wav_seg: torch.Tensor):
    return wav_seg / wav_seg.abs().max()


def instrument_to_release_time(instrument):  # determine release time depends on instrument
    instrument = instrument.strip()  # NOTE there is a trailing space in the instrument name
    if instrument == "keyboard":
        return random.uniform(0.1, 1.0)
    if instrument == "mallet":
        return random.uniform(0.1, 1.0)
    if instrument == "organ":
        return random.uniform(0.1, 1.0)
    if instrument == "guitar":
        return random.uniform(0.1, 0.5)
    if instrument == "bass":
        return random.uniform(0.1, 0.2)
    if instrument == "string":
        return random.uniform(0.8, 1.0)
    if instrument == "brass":
        return random.uniform(0.1, 1.0)
    if instrument == "reed":
        return random.uniform(0.8, 1.0)
    if instrument == "flute":
        return random.uniform(0.8, 1.0)
    if instrument == "synth":
        return random.uniform(0.1, 1.0)
    if instrument == "vocal":
        return random.uniform(0.1, 1.0)
    if instrument == "other":
        return random.uniform(0.1, 1.0)
    if instrument == "drums":
        return 0
    raise ValueError(f"Unknown instrument: {instrument}")


class AudioSegmenter:
    def __init__(self, config: SharedConfig):
        self.input_sec = config.input_sec

    def chunk_audio_and_notes(self, waveform: np.ndarray):
        waveform = torch.from_numpy(waveform)
        assert waveform.ndim == 1
        wave_length = waveform.shape[0]
        chunks = list(waveform.split(int(self.input_sec * self.sample_rate), dim=0))
        return chunks
