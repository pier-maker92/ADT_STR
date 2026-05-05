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


from transformers import PretrainedConfig

class ADTModelConfig(PretrainedConfig):
    model_type = "adt_model"

    def __init__(
        self,
        input_sec: float = 0.0,
        time_res: float = 0.0,
        win_length: int = 0,
        sample_rate: int = 0,
        enc_layers: int = 0,
        dec_layers: int = 0,
        nhead: int = 0,
        d_query: int = 0,
        dropout: float = 0.0,
        tgt_vocab_size: int = 0,
        enc_lr: float = 0.0,
        dec_lr: float = 0.0,
        plain: bool = False,
        n_mels: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_sec = input_sec
        self.time_res = time_res
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.nhead = nhead
        self.d_query = d_query
        self.dropout = dropout
        self.tgt_vocab_size = tgt_vocab_size
        self.enc_lr = enc_lr
        self.dec_lr = dec_lr
        self.plain = plain
        self.n_mels = n_mels

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
