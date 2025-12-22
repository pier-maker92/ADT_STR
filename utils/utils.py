import yaml
import random
import torch
import torchaudio
import numpy as np
from pathlib import Path
from collections import Counter


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


# def print_hdf_structure(file_path):
#     """
#     HDF5ファイルのグループ構造と属性を出力する関数。

#     Parameters:
#         file_path (str): HDF5ファイルのパス。
#     """
#     def explore_group(group, depth=0):
#         """
#         グループ内の構造と属性を再帰的に探索。
#         """
#         indent = "  " * depth  # インデントを深さに応じて調整
#         print(f"{indent}Group: {group.name}")

#         # グループの属性を表示
#         if group.attrs:
#             print(f"{indent}  Attributes:")
#             for attr_name, attr_value in group.attrs.items():
#                 print(f"{indent}    {attr_name}: {attr_value}")

#         # サブグループとデータセットを探索
#         keys = list(group.keys())
#         if len(keys) > 4:
#             # 多すぎる場合は例として3〜4つ表示
#             print(f"{indent}  Contains {len(keys)} items. Showing first 4 examples:")
#             keys_to_show = keys[:4]
#         else:
#             keys_to_show = keys

#         for key in keys_to_show:
#             item = group[key]
#             if isinstance(item, h5py.Group):
#                 # サブグループを再帰的に探索
#                 explore_group(item, depth + 1)
#             elif isinstance(item, h5py.Dataset):
#                 # データセットの情報を表示
#                 print(f"{indent}  Dataset: {key}, shape: {item.shape}, dtype: {item.dtype}")

#         if len(keys) > 4:
#             print(f"{indent}  ... and {len(keys) - 4} more items.")

#     # HDF5ファイルを開いてトップレベルグループを探索
#     with h5py.File(file_path, 'r') as hdf_file:
#         print(f"Exploring HDF5 file: {file_path}")
#         explore_group(hdf_file)

# file_path = "/media/data/sato/datasets/drumshot_train.hdf5"  # あなたのHDF5ファイルのパスを指定してください
# print_hdf_structure(file_path)


# def download_m4a(youtube_url, dump_path):
#     """
#     YouTubeのオーディオをダウンロードし、M4Aが利用できなければWebM、さらに利用できなければbestaudioを取得する。

#     Args:
#         youtube_url (str): YouTubeの動画URL
#         dump_path (str): 保存するファイルのベース名（拡張子なし）

#     Returns:
#         str: ダウンロードされたオーディオファイルのパス（None の場合は失敗）
#     """
#     base_path = "/groups/gcc50559/sato/gaps_v1/audio/"
#     formats = [("m4a", "bestaudio[ext=m4a]"), ("webm", "bestaudio[ext=webm]"), ("best", "bestaudio")]

#     for ext, fmt in formats:
#         audio_path = f"{base_path}{dump_path}.{ext}"
#         ydl_opts = {
#             "format": fmt,
#             "outtmpl": audio_path,
#         }

#         try:
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 ydl.extract_info(youtube_url, download=True)
#                 print(f"✅ {ext.upper()} 形式でダウンロード成功: {audio_path}")
#                 return audio_path
#         except Exception as e:
#             print(f"⚠️ {ext.upper()} 形式でのダウンロードに失敗: {e}")

#     print("❌ すべてのフォーマットでダウンロードに失敗しました。")
#     return None


# def gaps_create_audio(csv_file):
#     """
#     CSVファイルから `scorehash` と `yt_id` の列を取得し、
#     [[scorehash1, yt_id1], [scorehash2, yt_id2], ...] のリストを作成する。

#     Args:
#         csv_file (str): CSVファイルのパス

#     Returns:
#         list: 形状 [2, n] のリスト
#     """
#     datanames = []

#     with open(csv_file, newline='', encoding='utf-8') as file:
#         reader = csv.DictReader(file)  # ヘッダーをキーとして辞書形式で読み込む
#         for row in reader:
#             if 'scorehash' in row and 'yt_id' in row:
#                 datanames.append([row['scorehash'], row['yt_id']])

#     for dataname in datanames:
#         dump_path = dataname[0]
#         yt_url = f"https://www.youtube.com/watch?v={dataname[1]}"

#         download_m4a(yt_url, dump_path)


# # 使用例
# csv_file_path = "/groups/gcc50559/sato/gaps_v1/gaps_v1_metadata.csv"  # CSVファイルのパスを指定
# gaps_create_audio(csv_file_path)


# import h5py
# import numpy as np
# def analyze_midi_lengths(hdf5_file_path):
#     """
#     HDF5ファイルを再帰的に探索し、'midi0' から 'midi5' までのデータセットの長さの統計を出力する。

#     Args:
#         hdf5_file_path (str): HDF5ファイルのパス
#     """
#     midi_keys = [f'midi{i}' for i in range(6)]
#     midi_lengths = {key: [] for key in midi_keys}

#     def recursive_search(group):
#         """HDF5ファイルを再帰的に探索してmidiデータの長さを取得"""
#         for key in group.keys():
#             item = group[key]
#             if isinstance(item, h5py.Group):
#                 recursive_search(item)  # 再帰的に探索
#             elif isinstance(item, h5py.Dataset) and key in midi_lengths:
#                 midi_lengths[key].append(len(item))

#     # HDF5ファイルを開いて探索
#     with h5py.File(hdf5_file_path, 'r') as f:
#         recursive_search(f)

#     # 統計情報を表示
#     for key, lengths in midi_lengths.items():
#         if lengths:
#             lengths = np.array(lengths)
#             print(f"{key}: count={len(lengths)}, min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.2f}, std={lengths.std():.2f}")
#         else:
#             print(f"{key}: No data found.")

# # 使用例
# hdf5_file_path = "synthetic_data/train0.hdf5"  # 実際のHDF5ファイルのパスに変更
# analyze_midi_lengths(hdf5_file_path)
