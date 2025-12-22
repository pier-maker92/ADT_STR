## ADT - Automatic Drums Transcription

End-to-end pipeline for building a tokenized drum dataset from MIDI, training the ADT model with on-the-fly audio synthesis, and running inference.


### What you get
- **Dataset creation from MIDI**: fast, parallel preprocessing of the Lakh MIDI matched dataset into Parquet shards.
- **Token-only dataset**: no pre-rendered audio stored; audio is synthesized on-the-fly during training to enable augmentation and timbral diversity.
- **High-throughput dataloading**: on-the-fly synthesis enables mini-batches of 128 samples at ~3 it/s (hardware-dependent).
- **Config-first workflow**: a default YAML plus small per-experiment YAMLs that override only what you need.


## 1) Dataset creation (from Lakh MIDI)

The dataset is created from a path to a MIDI corpus—by default, the Lakh MIDI matched dataset—then saved to Parquet shards. Each shard contains only the MIDI tokens; no audio is stored.

- Script: `create_midi_dataset.py`
- Config: `config_default.yaml` under the `preprocess:` and `shared:` sections
- Input root: `preprocess.midi_root` (e.g., the Lakh MIDI matched path)
- Output root: `preprocess.dump_path` + `preprocess.dataset_name`
- Output layout:
  - `${dump_path}/${dataset_name}/A/*.parquet`, `.../B/*.parquet`, ..., `.../Z/*.parquet`
  - Each row: `midi_id`, `instrument`, `midi_variation` (0/1/2), `segment_number`, `tokens`

Parallelization and speed:
- CPU-intensive steps are parallelized with `joblib.Parallel`, so each partition (A, B, C, ...) typically processes in about 1–2 minutes in our environment.
- All three token variants are emitted by the preprocessor:
  - `0`: noinst (no instrument tokens)
  - `1`: inst (instrument tokens present)
  - `2`: sharem (shared-main timbre)

Run dataset creation:
```bash
cd /home/ach18017ws/AFAMT_share_250331/codes/ADT
python create_midi_dataset.py
```
Configuration keys to check before running (in `config_default.yaml`):
- `preprocess.midi_root`: path to your MIDI dataset root (e.g., Lakh MIDI matched)
- `preprocess.dataset_name`: a short name like "lakh_matched"
- `preprocess.dump_path`: base output directory for shards
- `preprocess.n_jobs`: CPU parallelism for tokenization
- `shared.sample_rate`, `shared.input_sec`, `shared.time_res`, `shared.win_length`: shared timing/audio params used downstream


## 2) Loading data: `MyDataset` and on-the-fly synthesis with `SynthDrum`

- `MyDataset` (in `data_utils_parquet.py`)
  - Reads Parquet shards via `datasets.load_dataset("parquet", ...)` from a glob string (e.g., `root/**/*.parquet`) or a list of shard paths.
  - Filters rows by `midi_variation` and optionally `instrument`.
  - Crops to `token_max_length` if provided.
  - Returns `(midi_tokens, wav_segment)` pairs where audio is created on demand by `SynthDrum`.

- `SynthDrum` (in `data_utils_parquet.py`)
  - Converts tokens → note events → waveform per segment on the fly.
  - Timbre is sourced from a one-shot drum HDF5: `synth.dr_oneshot_path` with a sample-rate suffix (e.g., `...@16000.hdf5`). This HDF5 corresponds to an organic one-shot set previously created; more one-shot sets can be added.
  - Applies limiter/mixup and random timbre selection to deliver strong augmentation and timbral diversity without inflating storage.
  - Supported `variation` values for synthesis: `0` (noinst) and `1` (inst). `2` (sharem) exists in preprocessing but is not yet enabled in synthesis.

- `collate_fn`
  - Pads token sequences with PAD=1 and audio with zeros.
  - Returns a dict with `tokens`, `wavs`, `token_lengths`, `wav_lengths`.

- Partition control
  - Point the loader at the full root (e.g., `.../lakh_matched/**/*.parquet`) or pass an explicit list of shard paths.
  - Training/inference configs allow specifying `A`, `B`, `C`, ... in `train_partitions` / `test_partitions` to limit which partitions to load.

Throughput:
- With synthesis on-the-fly, the dataloader synthesizes mini-batches of 128 samples at roughly ~3 iterations/sec (hardware-dependent).


## 3) Training with a custom Transformers Trainer

- Entry point: refer to `train_new.py` as `train.py` when invoking.
- Trainer: `ADTTrainer` (subclass of `transformers.Trainer`) computes loss by teacher-forcing tokens and synthesizing spectrograms on the fly.
- Datasets: created via `MyDataset` + `collate_fn` with configurable partitions/variations.
- Checkpointing/logging: handled by standard `TrainingArguments`. Auto-resume is supported.

Run training:
```bash
cd /home/ach18017ws/AFAMT_share_250331/codes/ADT
python train.py experiments/setting-1.yaml
```
Key configuration (YAML):
- `data.*`
  - `train_dataset_path`: root of your Parquet shards (e.g., `.../codes/ADT/data/lakh_matched`)
  - `val_dataset_path`: optional validation root
  - `variation`: `0` (noinst) or `1` (inst)
  - `instrument`: set to `"drums"` (multi-instrument training not implemented yet)
  - `train_partitions`: optional list like `[A, B, C]`
- `synth.*`
  - `dr_oneshot_path`: base path to the one-shot drum HDF5 (without the `@sr.hdf5` suffix; code appends `@${shared.sample_rate}.hdf5`)
  - `segment_type`, `limit_thr`, `limit_thrs`, `limit_p`, `mixup_range`, `dr_insttoken_offset`: synthesis/augmentation controls
- `shared.*`
  - `sample_rate`, `input_sec`, `time_res`, `win_length`: shared timing/audio params
- `model.*`
  - Transformer sizes, vocab size, `n_mels`, and `token_max_length`
- `training.*`
  - `batch_size`, `num_epochs`, `learning_rate`, `mixed_precision` (`no`/`fp16`/`bf16`), `gradient_accumulation_steps`, `eval_strategy`, `max_dataloader_num_workers`, etc.
- `logging.*`
  - `output_dir`, `logging_steps`, `save_every_n_steps`, optional `save_every_n_epochs`
- `experiment.*`
  - `run_name`, `project_name`, `use_wandb`, `seed`
- `checkpoint.*`
  - `resume_from_checkpoint`, `auto_resume`, `max_checkpoints`

Environment variables:
- YAML supports `${oc.env:VARNAME}` substitution for paths like `logging.output_dir`.


## 4) Inference

- Script: `inference.py`
- Uses the same Parquet reader and on-the-fly synthesis for evaluation dataloading.
- Metrics via `mir_eval` (precision/recall/F1 with and without offset tolerance).

Run inference:
```bash
cd /home/ach18017ws/AFAMT_share_250331/codes/ADT
python inference.py experiments/inference_config.yaml
```
Important inference keys:
- `inference.checkpoint_path`: path to a saved checkpoint directory
- `inference.test_dataset_path`: Parquet root used for evaluation
- `inference.variation`: `0` or `1`
- `inference.test_partitions`: optional subset like `[A]`
- `inference.batch_size`, `inference.max_length`, `inference.beam_size`, `inference.use_beam_search`


## 5) Config pattern: defaults + experiment overrides

This project uses a "defaults + overrides" configuration pattern:
- Base defaults live in `config_default.yaml` (preprocessing, shared audio params, synthesis, model, training, logging, experiment metadata, checkpoint policy, inference).
- Each run defines a small YAML (e.g., `experiments/setting-1.yaml`) that overrides only the needed keys.
- At runtime, the two YAMLs are merged recursively; environment variables are expanded.

Example minimal experiment overrides (training):
```yaml
experiment:
  project_name: "adt-training"
  run_name: "adt-setting-1"
  use_wandb: true

training:
  num_epochs: 3
  learning_rate: 1e-3
  batch_size: 96
  mixed_precision: "bf16"

logging:
  output_dir: "${oc.env:PBS_O_WORKDIR}/checkpoints"
  save_every_n_steps: 1000
  logging_steps: 1

model:
  enc_layers: 4
  dec_layers: 4
  d_query: 128
  dropout: 0.1

data:
  train_dataset_path: "/home/ach18017ws/AFAMT_share_250331/codes/ADT/data/lakh_matched"
  instrument: "drums"
  # train_partitions: [A, B, C]

checkpoint:
  max_checkpoints: 3
```

Example minimal experiment overrides (inference):
```yaml
inference:
  checkpoint_path: "/home/ach18017ws/checkpoints/adt-setting-1/checkpoint-epoch-0-step-5000"
  batch_size: 8
  max_length: 1000
  beam_size: 5
  use_beam_search: false
  test_dataset_path: "/home/ach18017ws/AFAMT_share_250331/codes/ADT/data/lakh_matched"
  variation: 0
  test_partitions: [A]
  instrument: "drums"

logging:
  log_level: "INFO"
```


## 6) Quick start

1. Prepare a MIDI root (e.g., Lakh MIDI matched) and set `preprocess.midi_root`, `preprocess.dump_path`, and `preprocess.dataset_name` in `config_default.yaml`.
2. Build Parquet shards:
   ```bash
   python create_midi_dataset.py
   ```
3. Ensure the one-shot HDF5 exists (e.g., organic one-shots) and set `synth.dr_oneshot_path` (base path; the code appends `@${shared.sample_rate}.hdf5`).
4. Create a small experiment YAML under `experiments/` overriding only what you need (dataset paths, batch size, run name, etc.).
5. Train the model:
   ```bash
   python train.py experiments/your_experiment.yaml
   ```
6. Optionally evaluate:
   ```bash
   python inference.py experiments/inference_config.yaml
   ```

### Tips
- Use `train_partitions` / `test_partitions` with letters like `A`, `B`, `C` to subset shards.
- Leave `val_dataset_path` unset to disable validation.
- Set `training.mixed_precision` to `bf16`/`fp16` if your hardware supports it.
- Use `checkpoint.auto_resume: true` to auto-pick the latest checkpoint in `logging.output_dir/run_name`.
