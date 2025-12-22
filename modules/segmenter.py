import torch
import numpy as np
from config import SharedConfig
from typing import Union, Optional


class Segmenter:
    def __init__(self, config: SharedConfig):
        self.config = config

    def chunk_notes(self, notes: np.ndarray):
        # create a mock audio

        audio_length = notes[:, 1].max()  # get the max offset
        n_chunks = int(audio_length // self.config.input_sec) + 1
        notes_chunks = [[] for _ in range(n_chunks)]
        for note in notes:
            onset, offset, pitch, velocity = note
            on_idx = int(onset // self.config.input_sec)
            off_idx = int(offset // self.config.input_sec)
            onset = onset % self.config.input_sec
            offset = offset % self.config.input_sec
            if on_idx == off_idx:
                notes_chunks[on_idx].append(np.array([onset, offset, pitch, velocity]))
        return notes_chunks

    def chunk_audio_and_notes(self, audio: torch.FloatTensor, notes: np.ndarray, audio_file: Optional[str] = None):
        assert audio.ndim == 1, "audio must be a 1D tensor"
        audio_chunks = list(audio.split(int(self.config.input_sec * self.config.sample_rate), dim=0))
        notes_chunks = [[] for _ in range(len(audio_chunks))]
        for note in notes:
            onset, offset, pitch, velocity = note
            on_idx = int(onset // self.config.input_sec)
            off_idx = int(offset // self.config.input_sec)
            onset = onset % self.config.input_sec
            offset = offset % self.config.input_sec
            if on_idx == off_idx:
                notes_chunks[on_idx].append(np.array([onset, offset, pitch, velocity]))

        return audio_chunks, notes_chunks
