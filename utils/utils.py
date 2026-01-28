import yaml
import random
import torch
import torchaudio
import numpy as np
from pathlib import Path
from collections import Counter
from fast_transformers.masking import TriangularCausalMask, BaseMask, LengthMask


class myLengthMask(BaseMask):
    """Provide a BaseMask interface for lengths. Mostly to be used with
    sequences of different lengths.
    """
    def __init__(self, lengths=None, max_len=None, device=None, manual_bool_matrix=None):
        self._device = device or (lengths.device if lengths is not None else None)

        # 初期化時に `bool_matrix` が渡されていれば、それを使用
        if manual_bool_matrix is not None:
            self._bool_matrix = manual_bool_matrix.to(self._device)
        elif lengths is not None:
            self._lengths = lengths.clone().to(self._device)
            self._max_len = max_len or self._lengths.max()
            self._bool_matrix = None  # `lengths` から `bool_matrix` を計算
        else:
            raise ValueError("Either `lengths` or `bool_matrix` must be specified")

    @property
    def bool_matrix(self):
        """
        If `bool_matrix` is already set, use it.
        Otherwise, compute it based on the `lengths`.
        """
        if self._bool_matrix is None:
            self._bool_matrix = create_bool_matrix(self._lengths, self._max_len, mask_prob=0.0)
        return self._bool_matrix

def create_mask_plain(tgt_seq_len, tgt_lengths = None, device = None):
    tgt_mask = TriangularCausalMask(tgt_seq_len, device = device)
    if tgt_lengths is None:
        return tgt_mask 
    tgt_padding_mask = LengthMask(torch.tensor(tgt_lengths), device = device)
    return tgt_mask, tgt_padding_mask



def create_bool_matrix(lengths, max_len=None, mask_prob=0.0):
    """
    Create a `bool_matrix` based on `lengths`, 
    and add functionality to optionally mask tokens at positions 5k+3 with probability `mask_prob`.

    Arguments:
        lengths: List or PyTorch Tensor (N,) indicating the valid length for each sequence.
        max_len: Maximum sequence length (default is lengths.max()).
        mask_prob: Probability to mask tokens at positions 5k+3 (default: 95%).

    Returns:
        bool_matrix: PyTorch boolean mask of shape (N, max_len)
    """
    # list を Tensor に変換
    if isinstance(lengths, list):
        lengths = torch.tensor(lengths, dtype=torch.int64)

    max_len = max_len or lengths.max()
    batch_size = lengths.size(0)

    with torch.no_grad():  # 勾配を記録せずにメモリ効率よく計算
        indices = torch.arange(max_len, device=lengths.device)
        bool_matrix = (indices.view(1, -1) < lengths.view(-1, 1))  # 通常のマスク

        # 5k+3 の位置のトークン(inst_token)を特定
        mask_positions = (indices % 5 == 3).view(1, -1)

        # 5k+3 の位置を mask_prob の確率でマスク
        random_mask = torch.bernoulli(torch.full(
            (batch_size, max_len), mask_prob, dtype=torch.float32, device=lengths.device
        )).bool()

        bool_matrix = bool_matrix & ~(mask_positions & random_mask)

    return bool_matrix

def create_mask(tgt_seq_len, tgt_lengths = None, device = None, inst_mask_proba=0.0):
    tgt_mask = TriangularCausalMask(tgt_seq_len-1, device = device)
    if tgt_lengths is None:
        return tgt_mask 
    tgt_bool_matrix = create_bool_matrix(tgt_lengths, mask_prob=inst_mask_proba)
    tgt_input_bool_matrix = tgt_bool_matrix[:, :-1]
    tgt_out_padding_mask = tgt_bool_matrix[:, 1:]
    tgt_input_padding_mask = myLengthMask(device=device, manual_bool_matrix=tgt_input_bool_matrix)
    
    return tgt_mask, tgt_input_padding_mask, tgt_out_padding_mask


def calc_separate_idx(total_files, separate, sepa_tgt, manual_file_range):
    if manual_file_range is not None:
        return np.array([manual_file_range]), [0, 1]

    if sepa_tgt is None:
        sepa_tgt = [0, separate]

    files_per_hdffile = total_files // separate

    file_range = np.zeros([separate, 2], dtype=int)
    for i in range(separate):
        file_range[i, 0] = i * files_per_hdffile
        if i < separate - 1:
            file_range[i, 1] = file_range[i, 0] + files_per_hdffile
        else:
            file_range[i, 1] = total_files

    return file_range, sepa_tgt


def myrelu(x, thr=0, rev=False):
    if rev:
        return x if x < thr else thr
    else:
        return x if x > thr else thr


def save_to_hdf(hf, inst_group, wav_seg, tokens):
    if inst_group is not None:
        if inst_group not in hf:
            group = hf.create_group(inst_group)
        else:
            group = hf[inst_group]
    else:
        group = hf
    seg_num = len(group.keys())
    seg_group = group.create_group(f"seg{seg_num}")

    seg_group.create_dataset("wav", data=wav_seg)
    if tokens is not None:
        seg_group.create_dataset("midi", data=tokens)


def save_to_hdf_variations(hf, inst_group, wav_seg, tokens):
    if inst_group is not None:
        if inst_group not in hf:
            group = hf.create_group(inst_group)
        else:
            group = hf[inst_group]
    else:
        group = hf
    seg_num = len(group.keys())
    seg_group = group.create_group(f"seg{seg_num}")

    seg_group.create_dataset("wav", data=wav_seg)

    if tokens is not None:
        for i in range(3):
            seg_group.create_dataset(f"midi{i}", data=tokens[i])


def save_tokens_hdf_variations(hf, inst_group, tokens):
    if inst_group is not None:
        if inst_group not in hf:
            group = hf.create_group(inst_group)
        else:
            group = hf[inst_group]
    else:
        group = hf
    seg_num = len(group.keys())
    seg_group = group.create_group(f"seg{seg_num}")

    if tokens is not None:
        for i in range(3):
            seg_group.create_dataset(f"midi{i}", data=tokens[i])


def file_shuffle(files, seed_fix):
    files = sorted(files, key=lambda x: str(Path(x)))
    if seed_fix:
        random.seed(1)
        random.shuffle(files)
    random.seed(None)

    return files


def debug_message(msg):
    with open("debug.txt", "w") as file:
        file.write(f"{msg}\n")
    assert False, "Debug file created !"


def my_vstack(array1, array2):
    if len(array1) == 0:
        return array2
    if len(array2) == 0:
        return array1
    else:
        return np.vstack((array1, array2))


def get_random_mode(lst):  # リストから最頻値を取得（重複がある場合はそこからランダム）
    counter = Counter(lst)
    max_count = max(counter.values())
    modes = [key for key, count in counter.items() if count == max_count]
    return random.choice(modes)


def create_one_hot(length, index):
    one_hot = torch.zeros(length, dtype=torch.float32)
    one_hot[index] = 1.0
    return one_hot


def pad_arrays(array1, array2):
    len1, len2 = len(array1), len(array2)
    if len1 > len2:
        array2 = np.pad(array2, (0, len1 - len2), mode="constant")
    elif len2 > len1:
        array1 = np.pad(array1, (0, len2 - len1), mode="constant")
    return array1, array2


def load_audio(path, target_sample_rate: int):
    # load audio and convert to mono
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # resample audio according to target sample rate
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    return waveform


def recursive_merge(default_cfg, experiment_cfg):
    for key, value in experiment_cfg.items():
        if isinstance(value, dict):
            default_cfg[key] = recursive_merge(default_cfg[key], value)
        else:
            default_cfg[key] = value
    return default_cfg


def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def draw_from_normal_distribution(std: float, mean: float, high_bound: float, low_bound: float):
    return torch.clamp(
        torch.clamp(torch.randn(1) * std + mean, -1.0, 1.0).abs() * high_bound, low_bound, high_bound
    ).item()