import math
import h5py
import torch
import random
import numpy as np
from config import SharedConfig
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from utils.mapping_utils import MappingUtils
from utils.utils import draw_from_normal_distribution
from pedalboard import Pedalboard, Reverb, Compressor, Limiter


@dataclass
class SynthDrumConfig(SharedConfig):
    oneshot_path: str
    similarity_threshold: float
    max_hat_std_velocity: float
    max_hat_mean_velocity: float
    max_cymbals_std_velocity: float
    max_cymbals_mean_velocity: float
    ADTOF_mapping: bool
    mixup_range: float
    use_fx_prob: float
    use_reverb_prob: float
    use_limiter_prob: float
    use_compression_prob: float


class BoardChain:
    def __init__(
        self,
        sample_rate: int,
        use_reverb_prob: float,
        use_compression_prob: float,
        use_limiter_prob: float,
    ):
        self.sample_rate = sample_rate
        self.board = Pedalboard([])
        self.use_reverb_prob = use_reverb_prob
        self.use_compression_prob = use_compression_prob
        self.use_limiter_prob = use_limiter_prob

    def _add_reverb(self):
        room_size = random.uniform(0.2, 0.8)
        damping = random.uniform(0.2, 0.8)
        wet_level = random.uniform(0.1, 0.4)
        dry_level = 1 - wet_level
        width = random.uniform(0.6, 1.0)
        freeze_mode = 0.0
        self.board.append(
            Reverb(
                room_size=room_size,
                damping=damping,
                wet_level=wet_level,
                dry_level=dry_level,
                width=width,
                freeze_mode=freeze_mode,
            )
        )

    def _add_compression(self):
        threshold = -draw_from_normal_distribution(std=0.15, mean=0.5, high_bound=10, low_bound=0)
        ratio = draw_from_normal_distribution(std=0.15, mean=0.5, high_bound=10, low_bound=1.0)
        attack = draw_from_normal_distribution(std=0.05, mean=0.1, high_bound=1000, low_bound=0)
        release = draw_from_normal_distribution(std=0.15, mean=0.2, high_bound=1000, low_bound=0)
        self.board.append(
            Compressor(
                threshold_db=threshold,
                ratio=ratio,
                attack_ms=attack,
                release_ms=release,
            )
        )

    def _add_limiter(self):
        threshold = -draw_from_normal_distribution(std=0.2, mean=0.4, high_bound=3, low_bound=0)
        self.board.append(Limiter(threshold_db=threshold))

    def get_board(self):
        if random.random() < self.use_reverb_prob:
            self._add_reverb()
        if random.random() < self.use_compression_prob:
            self._add_compression()
        if random.random() < self.use_limiter_prob:
            self._add_limiter()
        return self.board


class VolumeMixer:
    def __init__(
        self,
        wave_length: int,
        sample_rate: int,
        use_fx_prob: float,
        use_reverb_prob: float,
        use_compression_prob: float,
        use_limiter_prob: float,
        ADTOF_mapping: bool,
    ):
        self.wave_length = wave_length
        self.mapping_utils = MappingUtils()
        # FIXME: get these from config
        self.volume_per_instrument = {
            "BD": 1.0,  # random.uniform(0.6, 1),
            "SD": 1.0,  # random.uniform(0.6, 1),
            "TT": 1.0,  # random.uniform(0.6, 1),
            "HH": 0.7,  # random.uniform(0.1, 0.6),
            "CY + RD": 0.7,  # random.uniform(0.1, 0.6),
            "Cowbell": 0.7,  # random.uniform(0.3, 0.8),
            "Claves": 0.7,  # random.uniform(0.3, 0.8),
            "Other": 1.0,  # random.uniform(0.3, 1),
        }
        self.ADTOF_mapping = ADTOF_mapping
        self.ADTOF_map = MappingUtils().ADTOF_mapping
        self.ADTOF_label_mapping = MappingUtils().ADTOF_label_mapping
        self.sample_rate = sample_rate
        self.use_fx_prob = use_fx_prob
        self.board_chain = BoardChain(sample_rate, use_reverb_prob, use_compression_prob, use_limiter_prob)

    def _add_fx(
        self,
        x: torch.Tensor,
    ):
        x = x.detach().cpu().numpy()
        # x: mono (S,) or (C, S). Returns same shape.
        board = self.board_chain.get_board()

        mono = x.ndim == 1
        if mono:
            x_proc = x[None, :]  # (1, S)
        else:
            x_proc = x  # (C, S)

        y = board(x_proc.T.astype(np.float32), sample_rate=self.sample_rate).T  # (C, S)
        y = torch.from_numpy(y)
        return y[0] if mono else y

    def _valid_note(self, note):
        return note[2].item() >= 35 and note[2].item() <= 61 and note[1].item() >= note[0].item()

    def _normalize_audio(self, wav_seg):
        wav_seg = wav_seg / wav_seg.abs().max()
        return wav_seg

    def init_tracks(self, notes: torch.Tensor):
        return {note[2].item(): torch.zeros(self.wave_length) for note in notes if self._valid_note(note)}

    def instrument_mixer(self, tracks: dict, max_volume: float):
        wav = torch.zeros(self.wave_length)
        for instrument in tracks:
            key = self.ADTOF_map[instrument] if not self.ADTOF_mapping else instrument
            wav += tracks[instrument] * self.volume_per_instrument[self.ADTOF_label_mapping[key]]
        if random.random() < self.use_fx_prob:
            wav = self._add_fx(wav)
        return self._normalize_audio(wav) * max_volume


class SynthDrum:
    def __init__(self, config: SynthDrumConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.oneshot_path = f"{config.oneshot_path}@{self.sample_rate}.hdf5"
        self.similarity_threshold = config.similarity_threshold
        self.mapping_utils = MappingUtils()
        self.ADTOF_mapping = config.ADTOF_mapping

    def floor_to_tenth(self, x: float) -> float:
        return math.floor(x * 10) / 10

    def tolerance_thr_to_h5_group(self):
        trh_map = {
            1.0: "gold",
            0.9: "100-90",
            0.8: "90-80",
            0.7: "80-70",
            0.6: "70-60",
            0.5: "60-50",
            0.4: "50-40",
            0.3: "40-30",
            0.2: "30-20",
            0.1: "20-10",
            0.0: "10-0",
        }
        iter_thr = 1.0
        groups = []
        while iter_thr >= (self.floor_to_tenth(self.similarity_threshold)):
            groups.append(trh_map[round(iter_thr, 1)])
            iter_thr -= 0.1
        return groups

    def random_choice_timbre(self, ost, group):
        if self.ADTOF_mapping:
            group = random.choice(self.mapping_utils.ADTOF_inverse_mapping[group])
        thr_groups = self.tolerance_thr_to_h5_group()
        valid_groups = [thr_group for thr_group in thr_groups if f"{int(group)}/{thr_group}" in ost]
        thr_groups = random.choice(valid_groups)

        timbre = random.choice(list(ost[str(int(group))][thr_groups].keys()))
        timbre_path = f"{int(group)}/{thr_groups}/{timbre}"

        return timbre_path

    def _vel_to_vol(self, velocity, min_volume=0.1, max_volume=1.0, base=6):
        if velocity == 0:
            return 0
        velocity = torch.clamp(torch.tensor(velocity), 0, 127)
        # 正規化（0〜127を0〜1に変換）
        normalized_velocity = velocity / 127.0
        # 指数関数でマッピング
        volume = min_volume + (max_volume - min_volume) * (base**normalized_velocity - 1) / (base - 1)
        return volume

    def drum_rendering(
        self, wav_seg, onset, velocity, oneshot, sub_oneshot
    ):  # render oneshot audio of a note into the wav_segment
        mixup = random.uniform(0, self.config.mixup_range)
        oneshot, sub_oneshot = pad_sequence(
            [torch.tensor(oneshot), torch.tensor(sub_oneshot)], batch_first=True
        )  # ワンショットの長さを比較して長い方に合わせるようにパディング

        vol = self._vel_to_vol(velocity)  # ベロシティから音量(振幅)を計算
        oneshot = oneshot * (1 - mixup) + mixup * sub_oneshot
        # normalize oneshot
        oneshot = oneshot / oneshot.abs().max()
        # multiply vol
        oneshot = oneshot * vol

        note_start = int((onset) * self.sample_rate)  # 開始時刻を計算
        shot_len = len(oneshot)  # ワンショット全体をレンダリング

        if note_start + shot_len > len(wav_seg):  # もしワンショットの終了時刻ががオーディオ全体の長さを超える場合
            oneshot = oneshot[: len(wav_seg) - note_start]  # オーディオの最後に時刻までにワンショットの長さを調整
            wav_seg[note_start:] += oneshot  # レンダリング

        else:  # 超えない場合
            wav_seg[note_start : note_start + shot_len] += oneshot  # 普通にレンダリング

        return wav_seg

    def _get_volume_mixer(self, wav_seg_end: float):
        return VolumeMixer(
            int(wav_seg_end * self.config.sample_rate),
            self.config.sample_rate,
            self.config.use_fx_prob,
            self.config.use_reverb_prob,
            self.config.use_compression_prob,
            self.config.use_limiter_prob,
            self.ADTOF_mapping,
        )

    def _valid_note(self, note):
        return note[2].item() >= 35 and note[2].item() <= 61 and note[1].item() >= note[0].item()

    def __call__(self, notes, eval_rendering=False):
        # Use the same synthesis strategy as tokenize_and_rendering_0308
        if len(notes) == 0:
            return torch.zeros(int(self.config.input_sec * self.config.sample_rate))
        notes = torch.tensor(notes)

        # Determine end time by maximum note offset
        wav_seg_end = max(notes[:, 1].max() + 0.1, self.config.input_sec)
        mixer = self._get_volume_mixer(wav_seg_end)
        tracks = mixer.init_tracks(notes)
        perc_notelist = {}
        mex_velocity = 0
        for note in notes:
            onset, offset, pitch, velocity = note
            mex_velocity = max(mex_velocity, velocity)
            if not self._valid_note(note):
                raise ValueError(f"Invalid note: {note}")
            instrument = int(pitch.item())
            with h5py.File(self.oneshot_path, "r") as ost:
                if str(instrument) in perc_notelist:
                    timbre_path = perc_notelist[str(instrument)][0]
                    sub_timbre_path = perc_notelist[str(instrument)][1]
                else:
                    timbre_path = self.random_choice_timbre(ost, instrument)
                    sub_timbre_path = self.random_choice_timbre(ost, instrument)
                    # add timbre and sub_timbre to perc_notelist
                    perc_notelist[str(instrument)] = [timbre_path, sub_timbre_path]

                main_timbre = ost[timbre_path][...]
                sub_timbre = ost[sub_timbre_path][...]

                if eval_rendering:
                    main_timbre = ost[f"{instrument}/{self.default_timbre_path(instrument)}"][...]
                    sub_timbre = ost[f"{instrument}/{self.default_timbre_path(instrument)}"][...]

                tracks[instrument] = self.drum_rendering(tracks[instrument], onset, velocity, main_timbre, sub_timbre)

        return mixer.instrument_mixer(tracks, max_volume=self._vel_to_vol(mex_velocity))
