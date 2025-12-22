import os
import sys
import torch
from dataclasses import dataclass
from collections import defaultdict
from utils.mapping_utils import MappingUtils


@dataclass
class MidiTokenizerConfig:
    ADTOF_mapping: bool
    BOS_token: int
    EOS_token: int
    pad_token: int
    silence_token: int
    add_velocity: bool


class MidiTokenizer:
    def __init__(self, config: MidiTokenizerConfig):
        self.ADTOF_mapping = config.ADTOF_mapping
        self.ADTOF_map = MappingUtils().ADTOF_mapping
        self.GM_standard_midi_to_Gm_custom_map = MappingUtils().GM_standard_midi_to_Gm_custom_Mapping

        self.adt_tokens_offset_dict = {
            "time": 4,
            "pitch": 300,
            "velocity": 400,
        }
        self.BOS_token = config.BOS_token
        self.EOS_token = config.EOS_token
        self.pad_token = config.pad_token
        self.silence_token = config.silence_token
        self.add_velocity = config.add_velocity

    def map_notes_to_Gm_custom(self, notes, random_velocity=False):
        if self.ADTOF_mapping:
            notes[:, 2] = torch.tensor(
                [self.ADTOF_map[self.GM_standard_midi_to_Gm_custom_map[int(key)]] for key in notes[:, 2].tolist()]
            )
        else:
            notes[:, 2] = torch.tensor(
                [self.GM_standard_midi_to_Gm_custom_map[int(key)] for key in notes[:, 2].tolist()]
            )
        if random_velocity:
            notes[:, 3] = torch.randint(10, 127, (notes.shape[0],))
        return notes

    def notes_to_adt_tokens(self, notes, **kwargs):
        "Notes is intended to be all the notes in one segment"
        tokens = [self.BOS_token]  # BOS token
        for note in notes:
            onset, _, pitch, velocity = note
            onset = int(onset * 100)  # 100ms resolution
            time = onset + self.adt_tokens_offset_dict["time"]
            assert time < self.adt_tokens_offset_dict["pitch"], "Time token is out of range"
            pitch = pitch + self.adt_tokens_offset_dict["pitch"]
            tokens.extend([time, pitch])
            if self.add_velocity:
                velocity = velocity + self.adt_tokens_offset_dict["velocity"]
                tokens.extend([velocity])
        tokens.append(self.EOS_token)  # EOS token
        tokens = torch.tensor(tokens)
        return tokens

    def empty_adt_tokens(self):
        return torch.tensor([self.BOS_token, self.silence_token, self.EOS_token])

    def decode(self, tokens):
        onsets = defaultdict(float)
        pitches = defaultdict(float)
        velocities = defaultdict(float)
        notes = []
        for i, token in enumerate(tokens):
            if token in [self.BOS_token, self.EOS_token]:
                continue
            if token < self.adt_tokens_offset_dict["pitch"] and token >= self.adt_tokens_offset_dict["time"]:
                onset = (token - self.adt_tokens_offset_dict["time"]) / 100
                onsets[i] = onset
            elif token >= self.adt_tokens_offset_dict["pitch"] and token < self.adt_tokens_offset_dict["velocity"]:
                pitch = token - self.adt_tokens_offset_dict["pitch"]
                if self.ADTOF_mapping:
                    pitch = self.ADTOF_map[pitch]
                if i - 1 not in onsets:
                    continue
                pitches[i - 1] = pitch
            elif token >= self.adt_tokens_offset_dict["velocity"]:
                velocity = token - self.adt_tokens_offset_dict["velocity"]
                if i - 2 not in onsets:
                    continue
                velocities[i - 2] = velocity

        if len(velocities.keys()) == 0:
            velocities = defaultdict(float)
            for i in range(len(onsets)):
                velocities[i] = 100

        for onset, pitch, velocity in zip(onsets.values(), pitches.values(), velocities.values()):
            notes.append([onset, onset + 0.1, pitch, velocity])
        return torch.tensor(notes)

    def batch_decode(self, tokens):
        return [self.decode(token) for token in tokens]
