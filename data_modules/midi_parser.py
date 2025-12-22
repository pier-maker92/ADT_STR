import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pretty_midi
import numpy as np
from tqdm import tqdm
from pathlib import Path
from config import PreprocessConfig
from joblib import Parallel, delayed
from utils.midi_utils import MidiUtils
from modules.segmenter import Segmenter
from utils.config_utils import load_config_from_yaml, deep_merge_dicts

import pyarrow as pa
import pyarrow.parquet as pq


class LakhMidiParser:
    def __init__(
        self,
        config: PreprocessConfig,
    ):
        self.config = config
        self.midi_root = config.midi_root
        self.only_drums = config.only_drum
        self.dataset_name = config.dataset_name
        if self.dataset_name != "lakh_matched":  # TODO check if also lakh is supported
            raise NotImplementedError("Only Lakh matched dataset is supported for now")
        if not self.only_drums:
            raise NotImplementedError("only the drums parsing is supported for now")

        # create output path
        self.config.dump_path = os.path.join(self.config.dump_path, self.config.dataset_name)
        os.makedirs(self.config.dump_path, exist_ok=True)

        # check if custom partition
        partitions = self.config.partitions
        if partitions is None:
            self.partitions = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        else:
            assert isinstance(partitions, list)
            for partition in partitions:
                assert isinstance(partition, str)
                assert len(partition) == 1
                assert partition in [chr(c) for c in range(ord("A"), ord("Z") + 1)]
            self.partitions = partitions
        print(f"Partitions: {self.partitions}")

        # create utils
        self.midi_utils = MidiUtils()
        self.segmenter = Segmenter(config)

        # create schema
        self.schema = pa.schema(
            [
                pa.field("midi_id", pa.string()),
                pa.field("segment_number", pa.int32()),
                pa.field("notes", pa.binary()),
            ]
        )

    def _get__instrument(self, inst):
        if inst.is_drum:
            return "drums"
        else:
            return self.midi_utils._program_to_group(inst.program)

    def _get_midi_data(self, midi_path: str):
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        except:
            midi_data = None
        return midi_data

    def create_dataset(self):
        for partition in tqdm(self.partitions, desc="Processing partitions"):
            self.process_partition(partition=partition)

    def process_partition(self, partition: str):
        midi_partition_path = os.path.join(self.midi_root, partition)
        midi_files = list(Path(midi_partition_path).rglob("*.mid"))
        batch_rows = {
            "notes": [],
            "midi_id": [],
            "segment_number": [],
        }
        results = Parallel(n_jobs=self.config.n_jobs, return_as="generator")(
            delayed(self.parse_midi)(midi_file)
            for midi_file in (tqdm(midi_files, desc=f"Parsing midi files for partition {partition}"))
        )
        for local_batch in results:
            if local_batch:  # Only process if we got valid results
                for key in batch_rows:
                    batch_rows[key].extend(local_batch[key])
        table = pa.table(batch_rows, schema=self.schema)
        pq.write_table(table, os.path.join(self.config.dump_path, f"{partition}.parquet"))

    def parse_midi(self, midi_file: Path):
        batch_row = {
            "notes": [],
            "midi_id": [],
            "segment_number": [],
        }
        midi_data = self._get_midi_data(midi_file)
        if midi_data is None:
            return None
        for inst in midi_data.instruments:
            # get the instrument of the instrument
            instrument = self._get__instrument(inst)
            if self.only_drums and instrument != "drums":
                continue
            # parse the notes of the instrument, filtering out invalid notes
            notes = [
                [note.start, note.start + 0.1, note.pitch, note.velocity]
                for note in inst.notes
                if self.midi_utils.valid_note_per_instrument(instrument, note.pitch)
            ]
            notes = sorted(notes, key=lambda x: (x[0], x[1]))  # sort by onset and offset
            if len(notes):
                notes_chunks = self.segmenter.chunk_notes(notes=np.array(notes))
                for i, notes_chunk in enumerate(notes_chunks):
                    if len(notes_chunk):
                        notes_bin = np.array(notes_chunk, dtype=np.float32).tobytes()
                        batch_row["midi_id"].append(midi_file.stem)
                        batch_row["notes"].append(notes_bin)
                        batch_row["segment_number"].append(i)
        return batch_row


parser = argparse.ArgumentParser()
parser.add_argument(
    "config_path",
    type=str,
    help="Path to the config file. If not provided, the default config will be used.",
)
if __name__ == "__main__":
    args = parser.parse_args()
    default_path = "/home/ach18017ws/AFAMT_share_250331/codes/ADT/config_default.yaml"

    cfg = load_config_from_yaml(default_path)
    if args.config_path:
        user_cfg_path = Path(args.config_path)
        user_cfg = load_config_from_yaml(user_cfg_path)
        cfg = deep_merge_dicts(cfg, user_cfg)
    preprocess_config = cfg["preprocess"]
    preprocess_config.update(cfg["shared"])
    preprocess_config = PreprocessConfig(**preprocess_config)
    LakhMidiParser = LakhMidiParser(preprocess_config)
    LakhMidiParser.create_dataset()
