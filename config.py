import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

PBS_O_WORKDIR = os.getenv("PBS_O_WORKDIR")


@dataclass
class SharedConfig:
    input_sec: float
    time_res: float
    win_length: int
    sample_rate: int


@dataclass
class ClapConfig(SharedConfig):
    model_name: str
    batch_size: int
    sample_pack_root: str
    reference_root: str


@dataclass
class PreprocessConfig(SharedConfig):
    midi_root: str
    dataset_name: str
    dump_path: str
    separate: int
    sepa_tgt: List[int]
    manual_file_range: List[int]
    seed_fix: bool

    ignore_silent_p: float
    balance_thr: int
    balance_p: float
    only_drum: bool

    n_jobs: int
    pre_dispatch: str
    files_per_chunk: int

    limit_thr: float

    partitions: Optional[List[str]]


@dataclass
class DatasetBuilderConfig(SharedConfig):
    ignore_silent_p: float
    transposes: List[int]
    balance_thr: int
    balance_p: float
    dataset_path: str


@dataclass
class MidiConfig(SharedConfig):
    variation: str


@dataclass
class SynthDrumConfig(MidiConfig):
    segment_type: int
    dr_oneshot_path: str
    limit_thr: float
    dr_insttoken_offset: int
    limit_thrs: List[float]
    limit_p: float
    mixup_range: float
    tolerance_thr: float


@dataclass
class ENSTDrumConfig(MidiConfig):
    variation: str


@dataclass
class ADTModelConfig(SharedConfig):
    enc_layers: int
    dec_layers: int
    nhead: int
    d_query: int
    dropout: float
    tgt_vocab_size: int
    enc_lr: float
    dec_lr: float
    plain: bool
    n_mels: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enc_layers": self.enc_layers,
            "dec_layers": self.dec_layers,
            "nhead": self.nhead,
            "d_query": self.d_query,
            "dropout": self.dropout,
            "tgt_vocab_size": self.tgt_vocab_size,
            "enc_lr": self.enc_lr,
            "dec_lr": self.dec_lr,
            "plain": self.plain,
            "n_mels": self.n_mels,
        }
