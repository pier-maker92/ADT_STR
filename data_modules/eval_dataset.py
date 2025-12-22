from calendar import c
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional
from config import SharedConfig
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from utils.audio_utils import resample, normalize
from utils.config_utils import load_config_from_yaml
from modules.midi_tokenizer import MidiTokenizer, MidiTokenizerConfig


@dataclass
class EvalDatasetConfig(SharedConfig):
    dataset_path: str
    dataset_name: str


@dataclass
class ENSTDatasetConfig(EvalDatasetConfig):
    drummers: Optional[list[int]]
    minus_one: bool
    splits: Optional[list[int]]


@dataclass
class MDBDatasetConfig(EvalDatasetConfig):
    splits: Optional[list[int]]
    demucs_separated: bool


def collate_fn(batch):
    audio = [item[0] for item in batch]
    notes = [item[1] for item in batch]
    return {
        "wavs": audio,
        "notes": notes,
    }


class EvalDataset(Dataset):
    def __init__(self, config: EvalDatasetConfig, tokenizer: MidiTokenizer):
        self.sample_rate = config.sample_rate
        self.dataset_path = config.dataset_path
        self.dataset = load_dataset(
            "parquet",
            data_files=self.dataset_path,
            split="train",
        )

        self.tokenizer = tokenizer

    def _binary_to_torch(self, binary_data):
        return torch.from_numpy(np.frombuffer(binary_data, dtype=np.float32))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        audio = self._binary_to_torch(item["audio"])
        audio = resample(audio, orig_sr=item["sample_rate"], target_sr=self.sample_rate)
        audio = normalize(audio)
        notes = torch.stack(self._binary_to_torch(item["notes"]).split(4))  # onset, offset, pitch, velocity
        if not notes.shape[-1]:
            return audio, notes
        notes = self.tokenizer.map_notes_to_Gm_custom(notes)
        return audio, notes  # onset, offset, pitch, velocity

    def get_dataloader(self, batch_size: int, shuffle: bool, num_workers: int):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)


class ENSTDataset(EvalDataset):
    def __init__(self, config: ENSTDatasetConfig, tokenizer: MidiTokenizer):
        super().__init__(config, tokenizer)
        self.ENST_SPLITS = {
            0: [
                "143_MIDI-minus-one_fusion-125_sticks",
                "107_minus-one_salsa_sticks",
                "108_minus-one_rock-60s_sticks",
                "109_minus-one_metal_sticks",
                "110_minus-one_musette_brushes",
                "111_minus-one_funky_rods",
                "112_minus-one_funk_rods",
                "113_minus-one_charleston_sticks",
                "114_minus-one_celtic-rock_brushes",
                "115_minus-one_bossa_brushes",
                "121_MIDI-minus-one_bigband_brushes",
                "123_MIDI-minus-one_blues-102_sticks",
                "125_MIDI-minus-one_country-120_brushes",
                "127_MIDI-minus-one_disco-108_sticks",
                "129_MIDI-minus-one_funk-101_sticks",
                "131_MIDI-minus-one_grunge_sticks",
                "133_MIDI-minus-one_nu-soul_sticks",
                "135_MIDI-minus-one_rock-113_sticks",
                "137_MIDI-minus-one_rock'n'roll-188_sticks",
                "139_MIDI-minus-one_soul-120-marvin-gaye_sticks",
                "141_MIDI-minus-one_soul-98_sticks",
            ],
            1: [
                "152_MIDI-minus-one_fusion-125_sticks",
                "115_minus-one_salsa_sticks",
                "116_minus-one_rock-60s_sticks",
                "117_minus-one_metal_sticks",
                "118_minus-one_musette_brushes",
                "119_minus-one_funky_sticks",
                "120_minus-one_funk_sticks",
                "121_minus-one_charleston_sticks",
                "122_minus-one_celtic-rock_sticks",
                "123_minus-one_celtic-rock-better-take_sticks",
                "124_minus-one_bossa_sticks",
                "130_MIDI-minus-one_bigband_sticks",
                "132_MIDI-minus-one_blues-102_sticks",
                "134_MIDI-minus-one_country-120_sticks",
                "136_MIDI-minus-one_disco-108_sticks",
                "138_MIDI-minus-one_funk-101_sticks",
                "140_MIDI-minus-one_grunge_sticks",
                "142_MIDI-minus-one_nu-soul_sticks",
                "144_MIDI-minus-one_rock-113_sticks",
                "146_MIDI-minus-one_rock'n'roll-188_sticks",
                "148_MIDI-minus-one_soul-120-marvin-gaye_sticks",
                "150_MIDI-minus-one_soul-98_sticks",
            ],
            2: [
                "162_MIDI-minus-one_fusion-125_sticks",
                "126_minus-one_salsa_sticks",
                "127_minus-one_rock-60s_sticks",
                "128_minus-one_metal_sticks",
                "129_minus-one_musette_sticks",
                "130_minus-one_funky_sticks",
                "131_minus-one_funk_sticks",
                "132_minus-one_charleston_sticks",
                "133_minus-one_celtic-rock_sticks",
                "134_minus-one_bossa_sticks",
                "140_MIDI-minus-one_bigband_sticks",
                "142_MIDI-minus-one_blues-102_sticks",
                "144_MIDI-minus-one_country-120_sticks",
                "146_MIDI-minus-one_disco-108_sticks",
                "148_MIDI-minus-one_funk-101_sticks",
                "150_MIDI-minus-one_grunge_sticks",
                "152_MIDI-minus-one_nu-soul_sticks",
                "154_MIDI-minus-one_rock-113_sticks",
                "156_MIDI-minus-one_rock'n'roll-188_sticks",
                "158_MIDI-minus-one_soul-120-marvin-gaye_sticks",
                "160_MIDI-minus-one_soul-98_sticks",
            ],
        }
        if config.minus_one:
            self.dataset = self.dataset.filter(self._filter_minus_one)
        if config.splits is not None:
            self.splits = config.splits
            self.dataset = self.dataset.filter(self._filter_splits)
        self.drummers = config.drummers
        if self.drummers is not None:
            self.dataset = self.dataset.filter(self._filter_drummers)

    def _filter_drummers(self, example):
        return any(str(drummer) in example["drummer"] for drummer in self.drummers)

    def _filter_minus_one(self, example):
        return "minus-one" in example["audio_id"]

    def _filter_splits(self, example):
        return any(
            split_name in example["audio_id"]
            for split_index in self.splits
            for split_name in self.ENST_SPLITS[split_index]
        )


class MDBDataset(EvalDataset):
    def __init__(self, config: MDBDatasetConfig, tokenizer: MidiTokenizer):
        super().__init__(config, tokenizer)
        self.MDB_SPLITS = {
            0: [
                "MusicDelta_Punk",
                "MusicDelta_CoolJazz",
                "MusicDelta_Disco",
                "MusicDelta_SwingJazz",
                "MusicDelta_Rockabilly",
                "MusicDelta_Gospel",
                "MusicDelta_BebopJazz",
            ],
            1: [
                "MusicDelta_FunkJazz",
                "MusicDelta_FreeJazz",
                "MusicDelta_Reggae",
                "MusicDelta_LatinJazz",
                "MusicDelta_Britpop",
                "MusicDelta_FusionJazz",
                "MusicDelta_Shadows",
                "MusicDelta_80sRock",
            ],
            2: [
                "MusicDelta_Beatles",
                "MusicDelta_Grunge",
                "MusicDelta_Zeppelin",
                "MusicDelta_ModalJazz",
                "MusicDelta_Country1",
                "MusicDelta_SpeedMetal",
                "MusicDelta_Rock",
                "MusicDelta_Hendrix",
            ],
        }
        if config.splits is not None:
            self.splits = config.splits
            self.dataset = self.dataset.filter(self._filter_splits)
        if config.demucs_separated:
            self.dataset = self.dataset.filter(self._filter_demucs_separated)
        else:
            self.dataset = self.dataset.filter(self._filter_not_demucs_separated)

    def _filter_splits(self, example):
        return any(name in example["audio_id"] for split_index in self.splits for name in self.MDB_SPLITS[split_index])

    def _filter_demucs_separated(self, example):
        return example["is_demucs_separated"] == True

    def _filter_not_demucs_separated(self, example):
        return example["is_demucs_separated"] == False


parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()
if __name__ == "__main__":
    config = load_config_from_yaml(args.config)
    tokenizer = MidiTokenizer(MidiTokenizerConfig(**config["tokenizer"]))
    config_plain = config["shared"]
    config_plain.update(config["EvalDatasetConfig"])
    if config_plain.get("dataset_name") == "ENST":
        dataset = ENSTDataset(ENSTDatasetConfig(**config_plain), tokenizer)
    elif config_plain.get("dataset_name") == "MDB":
        dataset = MDBDataset(MDBDatasetConfig(**config_plain), tokenizer)
    else:
        raise ValueError(f"Dataset name {config_plain.get('dataset_name')} not supported")

    dataloader = dataset.get_dataloader(batch_size=4, shuffle=False, num_workers=4)

    for batch in tqdm(dataloader, desc="Processing dataset"):
        wavs = batch["wavs"]
        notes = batch["notes"]
        breakpoint()
        pass
# dataset["train"]["audio"][0]
