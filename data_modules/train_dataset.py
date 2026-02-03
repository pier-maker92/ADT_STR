import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import logging
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm
from config import SharedConfig
from dataclasses import dataclass
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from utils.config_utils import load_config_from_yaml
from modules.synthetiser import SynthDrum, SynthDrumConfig
from modules.midi_tokenizer import MidiTokenizer, MidiTokenizerConfig


@dataclass
class TrainDatasetConfig(SharedConfig):
    dataset_path: str
    empty_tokens_percentage: float
    random_velocity_prob: float
    dataset_name: str


@dataclass
class LakhDatasetConfig(TrainDatasetConfig):
    partitions: list[str]


@dataclass
class TMIDTDatasetConfig(TrainDatasetConfig):
    pass


def collate_fn(batch):
    pad_token = 1
    wavs = [item[0] for item in batch]
    token_lengths = [len(item[1]) for item in batch]
    tokens = [torch.tensor(item[1]) for item in batch]
    max_value = max(token_lengths) if len(token_lengths) > 0 else 0
    # Decrease lengths that are equal to the maximum length (to match reference behavior)
    if max_value > 0:
        for i, length in enumerate(token_lengths):
            if length == max_value:
                token_lengths[i] = length - 1
    return {
        "wavs": pad_sequence(wavs, batch_first=True, padding_value=0.0),
        "tokens": pad_sequence(tokens, batch_first=True, padding_value=pad_token).long(),
        "token_lengths": torch.tensor(token_lengths).long(),
    }


class TrainDataset(Dataset):
    def __init__(self, config: TrainDatasetConfig, tokenizer: MidiTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.empty_tokens_percentage = config.empty_tokens_percentage
        self.sample_rate = config.sample_rate

    def _resample_audio(self, audio: torch.Tensor, sr: int, target_sr: int):
        return torchaudio.transforms.Resample(sr, target_sr)(audio)

    def _map_notes_to_Gm_custom(self, example):
        notes = torch.stack(self._binary_to_torch(example["notes"]).split(4))  # onset, offset, pitch, velocity
        example["notes"] = self.tokenizer.map_notes_to_Gm_custom(notes)
        return example

    def _tokenize_notes(self, example):
        example["tokens"] = self.tokenizer.notes_to_adt_tokens(example["notes"])
        return example

    def __len__(self):
        return len(self.dataset)

    def _binary_to_torch(self, binary_data):
        return torch.from_numpy(np.frombuffer(binary_data, dtype=np.float32))

    def _empty_wav(self):
        return torch.zeros(int(self.config.input_sec * self.config.sample_rate))

    def __getitem__(self, index):
        pass

    def get_dataloader(self, batch_size: int, shuffle: bool, num_workers: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

class TMIDTDataset(TrainDataset):
    def __init__(self, config: TMIDTDatasetConfig, tokenizer: MidiTokenizer):
        super().__init__(config, tokenizer)
        self.config = config
        self.sample_rate = config.sample_rate
        self.dataset_path = config.dataset_path
        self.dataset = load_dataset(
            config.dataset_path,
        )["train"]
        self.tokenizer = tokenizer
        num_proc = min(32, os.cpu_count())

        if not config.random_velocity_prob:
            # map notes to Gm_custom
            self.dataset = self.dataset.map(self._map_notes_to_Gm_custom, num_proc=num_proc)
            # convert notes to tokens
            self.dataset = self.dataset.map(self._tokenize_notes, num_proc=num_proc)
        else:
            logging.info(
                "When the flag random_velocity_prob is True, the velocity will be randomly generated each time when an item is drawn, and the tokenization preprocess will be skipped in favour of a on-the-fly tokenization."
            )

        #self.dataset = self.dataset.map(self._get_audio_for_mapping, num_proc=num_proc)

    def _map_notes_to_Gm_custom(
        self, example
    ):  # we need to define here this method otherwise the map method will not work
        notes = torch.from_numpy(np.frombuffer(example["notes"], dtype=np.float32))
        notes = torch.stack(notes.split(4))  # onset, offset, pitch, velocity
        example["notes"] = self.tokenizer.map_notes_to_Gm_custom(notes)
        return example

    def _get_audio_for_mapping(self, example):
        audio = torch.from_numpy(np.frombuffer(example["audio"], dtype=np.float32))
        sample_rate = example["sample_rate"]
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            audio = resampler(audio)
        # normalize audio
        example["audio"] = audio / (audio.abs().max() + 1e-8)
        return example
    
    def _get_audio(self, audio, sample_rate):
        audio = torch.from_numpy(np.frombuffer(audio, dtype=np.float32))
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            audio = resampler(audio)
        # normalize audio
        audio = audio / (audio.abs().max() + 1e-8)
        return audio

    def __getitem__(self, index):
        if random.random() < self.empty_tokens_percentage:
            return self._empty_wav(), self.tokenizer.empty_adt_tokens()
        item = self.dataset[index]
        if self.config.random_velocity_prob:
            # get the notes with random velocity
            notes = torch.stack(self._binary_to_torch(item["notes"]).split(4))
            notes = self.tokenizer.map_notes_to_Gm_custom(
                notes, random_velocity=random.random() < self.config.random_velocity_prob
            )
            # get tokens
            tokens = self.tokenizer.notes_to_adt_tokens(notes)
        else:
            notes = item["notes"]
            tokens = item["tokens"]
        wavs = torch.tensor(self._get_audio(item["audio"], item["sample_rate"])) # item["audio"] 
        return wavs, tokens

    def get_dataloader(self, batch_size: int, shuffle: bool, num_workers: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )


class LakhDataset(Dataset):
    def __init__(self, config: LakhDatasetConfig, tokenizer: MidiTokenizer, synthetiser: SynthDrum):
        self.config = config
        self.sample_rate = config.sample_rate
        self.dataset_path = config.dataset_path
        partitions = config.partitions
        num_proc = min(32, os.cpu_count())
        if partitions is None:
            partitions = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        else:
            assert isinstance(partitions, list)
            for partition in partitions:
                assert isinstance(partition, str)
                assert len(partition) == 1
                assert partition in [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        data_path_list = [f"{self.dataset_path}/{partition}.parquet" for partition in partitions]
        self.dataset = load_dataset(
            "parquet",
            data_files=data_path_list,
            split="train",
        )
        self.tokenizer = tokenizer
        self.synthetiser = synthetiser

        if not config.random_velocity_prob:
            # map notes to Gm_custom
            self.dataset = self.dataset.map(self._map_notes_to_Gm_custom, num_proc=num_proc)
            # convert notes to tokens
            self.dataset = self.dataset.map(self._tokenize_notes, num_proc=num_proc)
        else:
            logging.info(
                "When the flag random_velocity_prob is True, the velocity will be randomly generated each time when an item is drawn, and the tokenization preprocess will be skipped in favour of a on-the-fly tokenization."
            )
        self.empty_tokens_percentage = config.empty_tokens_percentage

    def __getitem__(self, index):
        if random.random() < self.empty_tokens_percentage:
            return self._empty_wav(), self.tokenizer.empty_adt_tokens()
        item = self.dataset[index]
        if self.config.random_velocity_prob:
            # get the notes with random velocity
            notes = torch.stack(self._binary_to_torch(item["notes"]).split(4))
            notes = self.tokenizer.map_notes_to_Gm_custom(
                notes, random_velocity=random.random() < self.config.random_velocity_prob
            )
            # get tokens
            tokens = self.tokenizer.notes_to_adt_tokens(notes)
        else:
            notes = item["notes"]
            tokens = item["tokens"]
        wavs = self.synthetiser(notes)
        return wavs, tokens


parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()
if __name__ == "__main__":
    # config
    config = load_config_from_yaml(args.config)
    config_tokenizer = config["tokenizer"]
    config_dataset = config["TrainDatasetConfig"]
    config_dataset.update(config["shared"])

    if config_dataset["dataset_name"] == "Lakh":
        config_synthetiser = config.get("synthetiser", None)
        assert config_synthetiser is not None, "Synthetiser is required for Lakh dataset"
        config_synthetiser["ADTOF_mapping"] = config_tokenizer["ADTOF_mapping"]
        config_synthetiser.update(config["shared"])
        synthetiser = SynthDrum(SynthDrumConfig(**config_synthetiser))
    else:
        config_synthetiser = None
    # load modules
    tokenizer = MidiTokenizer(MidiTokenizerConfig(**config_tokenizer))
    if config_dataset["dataset_name"] == "Lakh":
        dataset = LakhDataset(LakhDatasetConfig(**config_dataset), tokenizer, synthetiser)
    elif config_dataset["dataset_name"] == "TMIDT":
        dataset = TMIDTDataset(TMIDTDatasetConfig(**config_dataset), tokenizer)
    else:
        raise ValueError(f"Dataset name {config_dataset['dataset_name']} not supported")
    dataloader = dataset.get_dataloader(batch_size=4, shuffle=False, num_workers=4)

    counter = 0
    sanity_check_path = "/home/ec2-user/ADT_STR/sanity_check"
    os.makedirs(sanity_check_path, exist_ok=True)
    for batch in tqdm(dataloader, desc="Processing dataset"):
        # save wav in sanity check using torchaudio
        wavs = batch["wavs"]
        tokens = batch["tokens"]
        for i, (wav, token) in enumerate(zip(wavs, tokens)):
            torchaudio.save(f"{sanity_check_path}/{counter}_{i}.wav", wav.unsqueeze(0), config_dataset["sample_rate"])
        counter += 1
        if counter >= 10:
            break
