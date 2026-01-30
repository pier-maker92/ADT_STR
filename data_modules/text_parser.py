import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from pathlib import Path
from typing import Optional
from config import SharedConfig
from dataclasses import dataclass
from utils.midi_utils import MidiUtils
from modules.segmenter import Segmenter
from utils.mapping_utils import MappingUtils
from utils.audio_utils import load_and_resample
from utils.config_utils import load_config_from_yaml

import pyarrow.parquet as pq
import pyarrow as pa


@dataclass
class DrumTextParserConfig(SharedConfig):
    dataset_path: str
    output_path: str
    dataset_name: str
    audio_extension: str


class DrumTextParser:
    def __init__(
        self,
        config: DrumTextParserConfig,
    ):
        self.dataset_path = config.dataset_path
        self.dataset_name = config.dataset_name
        self.audio_data_files = glob(
            os.path.join(config.dataset_path, f"**/*.{config.audio_extension}"), recursive=True
        )
        self.audio_data_files.sort()

        self.dump_path = config.output_path
        self.parquet_path = os.path.join(self.dump_path, self.dataset_name, f"data@{config.sample_rate}.parquet")
        os.makedirs(os.path.dirname(self.parquet_path), exist_ok=True)

        self.midi_utils = MidiUtils()
        self.segmenter = Segmenter(config)
        self.config = config

    def _estimate_batch_size(self, batch_rows):
        """Estimate the size of batch_rows in bytes."""
        total_size = 0
        for key, values in batch_rows.items():
            if not values:
                continue
            if key == "audio":
                # Sum of all audio binary data
                total_size += sum(len(v) for v in values)
            elif key == "notes":
                # Sum of all notes binary data
                total_size += sum(len(v) for v in values)
            else:
                # Estimate size for other fields (strings, ints, etc.)
                # Rough estimate: 50 bytes per entry for metadata
                total_size += len(values) * 50
        return total_size

    def _write_chunk(self, batch_rows, schema, chunk_index):
        """Write a chunk of data to a parquet file."""
        if not batch_rows or not any(batch_rows.values()):
            return

        # Generate chunk filename
        base_path = self.parquet_path.replace(".parquet", "")
        os.makedirs(base_path, exist_ok=True)
        chunk_path = f"{base_path}/{chunk_index:04d}.parquet"

        table = pa.table(batch_rows, schema=schema)
        pq.write_table(table, chunk_path)

        # Clear the batch_rows for next chunk
        for key in batch_rows:
            batch_rows[key] = []


@dataclass
class TMIDTTextParserConfig(DrumTextParserConfig):
    dataset_size: str
    drums_only: bool
    chunk_size_mb: int = 512
    num_audio_files: Optional[int] = None

    def __post_init__(self):
        if self.dataset_size not in ["m", "l"]:
            raise ValueError("dataset_size must be either 'm' or 'l'")


class TMIDTTextParser(DrumTextParser):
    def __init__(self, config: TMIDTTextParserConfig):
        super().__init__(config)

        self.dataset_size = config.dataset_size
        self.chunk_size_mb = config.chunk_size_mb
        self.chunk_size_bytes = self.chunk_size_mb * 1024 * 1024

        if config.num_audio_files:
            print(
                f"WARNING: num_audio_files is set to {config.num_audio_files}, this should be used for debugging only"
            )
            self.audio_data_files = self.audio_data_files[: config.num_audio_files]

        self.schema = pa.schema(
            [
                pa.field("audio_id", pa.string()),
                pa.field("audio", pa.binary()),  # raw wav bytes
                pa.field("sample_rate", pa.int32()),
                pa.field("notes", pa.binary()),
            ]
        )
        self.annotation_path = os.path.join(self.dataset_path, "annotations")
        if config.drums_only:
            self.audio_data_files = [f for f in self.audio_data_files if "_accomp" not in f]
        filtered_audio_data_files = []
        for f in self.audio_data_files:
            if os.path.exists(
                os.path.join(self.annotation_path, f"drums_{self.dataset_size}", Path(f).name.replace(".mp3", ".txt"))
            ):
                filtered_audio_data_files.append(f)
        self.audio_data_files = filtered_audio_data_files
        self.audio_data_files.sort()

        self.mapping = MappingUtils().TMIDT_to_Standard_MIDI

    def parse(self):
        # parse audio segment and notes into parquet
        batch_rows = {
            "audio_id": [],
            "sample_rate": [],
            "audio": [],
            "notes": [],
        }
        chunk_index = 0
        for audio_file in tqdm(self.audio_data_files, desc="Parsing audio files"):
            notes = []
            audio_id = Path(audio_file).name
            with open(
                os.path.join(self.annotation_path, f"drums_{self.dataset_size}", audio_id.replace(".mp3", ".txt")), "r"
            ) as f:
                for line in f.readlines():
                    content = line.split()
                    if len(content):
                        start, label = content
                        label = int(label)
                        start = float(start)
                        notes.append([start, start + 0.1, self.mapping[label], 100])
            notes = sorted(notes, key=lambda x: (x[0], x[1]))  # sort by onset and offset
            try:
                audio = load_and_resample(audio_file, self.config.sample_rate)
                audio_chunks, notes_chunks = self.segmenter.chunk_audio_and_notes(audio, notes, audio_id)
            except Exception as e:
                print(e)
                continue
            for audio_chunk, notes_chunk in zip(audio_chunks, notes_chunks):
                if not len(notes_chunk):
                    continue
                wav_bin = audio_chunk.numpy().astype(np.float16).tobytes()
                notes_bin = np.array(notes_chunk, dtype=np.float32).tobytes()
                batch_rows["audio_id"].append(audio_id)
                batch_rows["audio"].append(wav_bin)
                batch_rows["notes"].append(notes_bin)
                batch_rows["sample_rate"].append(self.config.sample_rate)

                # Check if we need to write a chunk
                if self._estimate_batch_size(batch_rows) >= self.chunk_size_bytes:
                    self._write_chunk(batch_rows, self.schema, chunk_index)
                    chunk_index += 1

        # Write the final chunk
        if any(batch_rows.values()):
            self._write_chunk(batch_rows, self.schema, chunk_index)


@dataclass
class MDBDrumTextParserConfig(DrumTextParserConfig):
    pass


class MDBDrumTextParser(DrumTextParser):
    def __init__(self, config: MDBDrumTextParserConfig):
        super().__init__(config)

        self.audio_data_files = [
            f
            for f in self.audio_data_files
            if any(name in f for name in ["drum_only", "demucs_separated"]) and "no_drums" not in f
        ]
        self.audio_data_files.sort()
        self.schema = pa.schema(
            [
                pa.field("audio_id", pa.string()),
                pa.field("audio", pa.binary()),  # raw wav bytes
                pa.field("sample_rate", pa.int32()),
                pa.field("notes", pa.binary()),
                pa.field("split", pa.int32()),
                pa.field("is_demucs_separated", pa.bool_()),
            ]
        )
        self.annotation_path = os.path.join(self.dataset_path, "annotations", "subclass")
        self.mapping = MappingUtils().MDB_to_Standard_MIDI
        self.MDB_SPLITS = {
            0: [
                "MusicDelta_Punk_Drum",
                "MusicDelta_CoolJazz_Drum",
                "MusicDelta_Disco_Drum",
                "MusicDelta_SwingJazz_Drum",
                "MusicDelta_Rockabilly_Drum",
                "MusicDelta_Gospel_Drum",
                "MusicDelta_BebopJazz_Drum",
            ],
            1: [
                "MusicDelta_FunkJazz_Drum",
                "MusicDelta_FreeJazz_Drum",
                "MusicDelta_Reggae_Drum",
                "MusicDelta_LatinJazz_Drum",
                "MusicDelta_Britpop_Drum",
                "MusicDelta_FusionJazz_Drum",
                "MusicDelta_Shadows_Drum",
                "MusicDelta_80sRock_Drum",
            ],
            2: [
                "MusicDelta_Beatles_Drum",
                "MusicDelta_Grunge_Drum",
                "MusicDelta_Zeppelin_Drum",
                "MusicDelta_ModalJazz_Drum",
                "MusicDelta_Country1_Drum",
                "MusicDelta_SpeedMetal_Drum",
                "MusicDelta_Rock_Drum",
                "MusicDelta_Hendrix_Drum",
            ],
        }

    def get_split(self, audio_file):
        for split, genres in self.MDB_SPLITS.items():
            if any(genre in audio_file for genre in genres):
                return split
        return -1

    def parse(self):
        # parse audio segment and notes into parquet
        batch_rows = {
            "audio_id": [],
            "sample_rate": [],
            "audio": [],
            "notes": [],
            "split": [],
            "is_demucs_separated": [],
        }
        chunk_index = 0
        for audio_file in tqdm(self.audio_data_files, desc="Parsing audio files"):
            notes = []
            audio_id = Path(audio_file).name
            is_demucs_separated = "_MIX_drums.wav" in audio_file
            audio_id = audio_id.replace("_MIX_drums.wav", "_Drum.wav")
            with open(os.path.join(self.annotation_path, audio_id.replace("_Drum.wav", "_subclass.txt")), "r") as f:
                for line in f.readlines():
                    content = line.split()
                    if len(content):
                        start, label = content
                    if self.midi_utils.valid_note_per_instrument("drums", self.mapping[label]):
                        notes.append([float(start), float(start) + 0.1, self.mapping[label], 100])
            notes = sorted(notes, key=lambda x: (x[0], x[1]))  # sort by onset and offset
            audio = load_and_resample(audio_file, self.config.sample_rate)
            audio_chunks, notes_chunks = self.segmenter.chunk_audio_and_notes(audio, notes)
            for audio_chunk, notes_chunk in zip(audio_chunks, notes_chunks):
                wav_bin = audio_chunk.numpy().astype(np.float32).tobytes()
                notes_bin = np.array(notes_chunk, dtype=np.float32).tobytes()
                batch_rows["audio_id"].append(audio_id)
                batch_rows["audio"].append(wav_bin)
                batch_rows["notes"].append(notes_bin)
                batch_rows["sample_rate"].append(self.config.sample_rate)
                batch_rows["split"].append(self.get_split(audio_file))
                batch_rows["is_demucs_separated"].append(is_demucs_separated)

                # Check if we need to write a chunk
                if self._estimate_batch_size(batch_rows) >= self.chunk_size_bytes:
                    self._write_chunk(batch_rows, self.schema, chunk_index)
                    chunk_index += 1

        # Write the final chunk
        if any(batch_rows.values()):
            self._write_chunk(batch_rows, self.schema, chunk_index)


@dataclass
class ENSTDrumTextParserConfig(DrumTextParserConfig):
    drummers: list[int]


class ENSTDrumTextParser(DrumTextParser):
    def __init__(self, config: ENSTDrumTextParserConfig):
        super().__init__(config)
        self.audio_data_files = [f for f in self.audio_data_files if "wet_mix" in f]
        if config.drummers:
            drummers = [f"drummer_{drummer}" for drummer in config.drummers]
            self.audio_data_files = [f for f in self.audio_data_files if any(drummer in f for drummer in drummers)]
        self.audio_data_files.sort()
        self.schema = pa.schema(
            [
                pa.field("audio_id", pa.string()),
                pa.field("drummer", pa.string()),
                pa.field("audio", pa.binary()),  # raw wav bytes
                pa.field("sample_rate", pa.int32()),
                pa.field("notes", pa.binary()),
            ]
        )
        self.mapping = MappingUtils().ENST_to_Standard_MIDI

    def search_for_string_in_path(self, path, string):
        for part in path.split("/"):
            if string in part:
                return part
        return ""

    def create_audio_folderwith_metadata(self):
        # copy audio in destination folder and create a csv with aggregated onsets and labels per file
        records = []
        for audio_file in tqdm(self.audio_data_files):
            file_name = Path(audio_file).name
            drummer = self.search_for_string_in_path(audio_file, "drummer")
            text_file = os.path.join(self.dataset_path, drummer, "annotation", file_name.replace(".wav", ".txt"))
            accompaniment = self.search_for_string_in_path(audio_file, "accompaniment")
            base_name = Path(text_file).with_suffix("").name
            file_name = f"{drummer}_{accompaniment}_{base_name}.wav"
            shutil.copy(audio_file, os.path.join(self.dump_path, file_name))

            onsets = []
            labels = []
            with open(text_file, "r") as f:
                for line in f.readlines():
                    content = line.split()
                    if len(content) >= 2:
                        start, label = content[0], content[1]
                        onsets.append(float(start))
                        labels.append(label)

            # sort pairs by onset while preserving alignment between onsets and labels
            if onsets:
                sorted_pairs = sorted(zip(onsets, labels), key=lambda x: x[0])
                onsets, labels = [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]

            records.append(
                {
                    "file_name": file_name,
                    "onsets_and_labels": sorted_pairs,
                    "drummer": drummer,
                }
            )

        df = pd.DataFrame.from_records(records, columns=["file_name", "onsets_and_labels", "drummer"])
        df.to_csv(os.path.join(self.dump_path, "metadata.csv"), index=False)

    def parse(self):
        # parse audio segment and notes into parquet
        batch_rows = {
            "audio_id": [],
            "sample_rate": [],
            "audio": [],
            "notes": [],
            "drummer": [],
        }
        chunk_index = 0
        for audio_file in tqdm(self.audio_data_files, desc="Parsing audio files"):
            notes = []
            drummer = self.search_for_string_in_path(audio_file, "drummer")
            file_name = Path(audio_file).name
            audio_id = f"{drummer}_{file_name}"
            text_file = os.path.join(self.dataset_path, drummer, "annotation", file_name.replace(".wav", ".txt"))
            with open(text_file, "r") as f:
                for line in f.readlines():
                    content = line.split()
                    if len(content):
                        start, label = content
                    if self.midi_utils.valid_note_per_instrument("drums", self.mapping[label]):
                        notes.append([float(start), float(start) + 0.1, self.mapping[label], 100])
            notes = sorted(notes, key=lambda x: (x[0], x[1]))  # sort by onset and offset
            audio = load_and_resample(audio_file)
            try:
                audio_chunks, notes_chunks = self.segmenter.chunk_audio_and_notes(audio, notes)
            except ValueError as e:
                print(e)
                continue
            for audio_chunk, notes_chunk in zip(audio_chunks, notes_chunks):
                wav_bin = audio_chunk.numpy().astype(np.float32).tobytes()
                notes_bin = np.array(notes_chunk, dtype=np.float32).tobytes()
                batch_rows["audio_id"].append(audio_id)
                batch_rows["drummer"].append(drummer)
                batch_rows["audio"].append(wav_bin)
                batch_rows["notes"].append(notes_bin)
                batch_rows["sample_rate"].append(self.config.sample_rate)

                # Check if we need to write a chunk
                if self._estimate_batch_size(batch_rows) >= self.chunk_size_bytes:
                    self._write_chunk(batch_rows, self.schema, chunk_index)
                    chunk_index += 1

        # Write the final chunk
        if any(batch_rows.values()):
            self._write_chunk(batch_rows, self.schema, chunk_index)


# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config_from_yaml(args.config_path)
    plain_config = config["shared"]
    plain_config.update(config["EvalDataPreprocess"])
    if plain_config.get("dataset_name") == "ENST":
        text_parser = ENSTDrumTextParser(config=ENSTDrumTextParserConfig(**plain_config))
    elif plain_config.get("dataset_name") == "MDB":
        text_parser = MDBDrumTextParser(config=MDBDrumTextParserConfig(**plain_config))
    elif plain_config.get("dataset_name") == "TMIDT":
        text_parser = TMIDTTextParser(config=TMIDTTextParserConfig(**plain_config))
    else:
        raise ValueError(f"Dataset name {config.dataset_name} not supported")
    text_parser.parse()
