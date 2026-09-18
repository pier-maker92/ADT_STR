# ADT_STR

Version: `v1.0.0`

## Overview

ADT_STR is an automatic drum transcription model. Given an input audio file, the
model predicts a drum MIDI sequence containing note onsets, pitches, and
velocities.

The repository provides:

- Transformer-based drum transcription model code.
- Checkpoint-based inference with per-checkpoint JSON configuration.
- Training scripts based on Hugging Face `Trainer`.
- Evaluation utilities for drum transcription benchmarks.

Supplementary material is available at the
[demo page](https://devsupplementary.github.io/2026ADTsupplementaryMaterial/).

Available local checkpoints:

| Checkpoint | Weight file | Configuration |
| --- | --- | --- |
| `setting-tau-0.4` | `checkpoints/setting-tau-0.4/model.safetensors` | `checkpoints/setting-tau-0.4/config.json` |
| `setting-tau-0.6` | `checkpoints/setting-tau-0.6/model.safetensors` | `checkpoints/setting-tau-0.6/config.json` |
| `setting-tau-0.8` | `checkpoints/setting-tau-0.8/model.safetensor` | `checkpoints/setting-tau-0.8/config.json` |

Each checkpoint directory contains the model weights and the JSON configuration
needed for inference. The default audio configuration uses 24 kHz audio and
2.56-second chunks.

## Results

The figure below reports F1 scores on ENST and MDB across the full drum
instrument vocabulary.

![F1 scores across the full drum vocabulary](assets/f1_scores_bars_full_vocab.png)

Evaluations were computed with the
[mir_eval](https://mir-eval.readthedocs.io/latest/) package using a 50 ms
tolerance window.

The tau checkpoints correspond to different training settings:

- `setting-tau-0.4`
- `setting-tau-0.6`
- `setting-tau-0.8`

In the reported aggregate scores, `setting-tau-0.8` gives the strongest overall
performance on both ENST and MDB.

### Confusion matrices

The confusion matrices below report class-level transcription behavior for each
tau checkpoint on ENST and MDB.

| Checkpoint | ENST | MDB |
| --- | --- | --- |
| `setting-tau-0.4` | ![ENST confusion matrix for tau 0.4](assets/confusion_matrix_ENST_tau_0.4.png) | ![MDB confusion matrix for tau 0.4](assets/confusion_matrix_MDB_tau_0.4.png) |
| `setting-tau-0.6` | ![ENST confusion matrix for tau 0.6](assets/confusion_matrix_ENST_tau_0.6.png) | ![MDB confusion matrix for tau 0.6](assets/confusion_matrix_MDB_tau_0.6.png) |
| `setting-tau-0.8` | ![ENST confusion matrix for tau 0.8](assets/confusion_matrix_ENST_tau_0.8.png) | ![MDB confusion matrix for tau 0.8](assets/confusion_matrix_MDB_tau_0.8.png) |

## Install

This project is configured with `pyproject.toml` and can be installed with
`uv`.

Create a clean environment:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r pyproject.toml
```

The core inference dependencies include PyTorch, TorchAudio, TorchCodec,
Transformers, SafeTensors, OmegaConf, PrettyMIDI, NumPy, and tqdm.

## Inference

### Hugging Face

Use the Hugging Face Hub repo as a self-contained inference bundle:

```python
from huggingface_hub import snapshot_download
import sys

repo_dir = snapshot_download("your-username/adt-str")
sys.path.insert(0, repo_dir)

from adt_transcriber import ADTTranscriber

transcriber = ADTTranscriber.from_pretrained(repo_dir, variant="setting-tau-0.8")
midi_path = transcriber.transcribe("path/to/audio.wav", output_dir="outputs")
transcriber.play(midi_path, output_path="outputs/preview.wav")
```

`transcribe` also accepts a list of audio paths:

```python
midi_paths = transcriber.transcribe(
    ["path/to/audio_1.wav", "path/to/audio_2.wav"],
    output_dir="outputs",
)
```

or a padded tensor batch:

```python
midi_paths = transcriber.transcribe(
    padded_audio_batch,
    sample_rate=24000,
    lengths=valid_lengths,
    output_dir="outputs",
)
```

Use `variant="setting-tau-0.4"`, `variant="setting-tau-0.6"`, or
`variant="setting-tau-0.8"` to select a checkpoint.

### Local checkpoint

Run inference by passing a checkpoint name and an input audio file:

```bash
python inference --checkpoint setting-tau-0.8 --input path/to/audio.wav
```

The checkpoint can be either:

- a local checkpoint name under `checkpoints/`, for example `setting-tau-0.8`
- a direct path to a checkpoint directory

Example with an explicit output directory:

```bash
python inference \
  --checkpoint setting-tau-0.8 \
  --input path/to/audio.wav \
  --output_path outputs/inference_tau08
```

The script writes a MIDI file named after the input audio stem:

```text
outputs/inference_tau08/audio.mid
```

You can select another checkpoint by changing the checkpoint name:

```bash
python inference --checkpoint setting-tau-0.4 --input path/to/audio.wav
python inference --checkpoint setting-tau-0.6 --input path/to/audio.wav
python inference --checkpoint setting-tau-0.8 --input path/to/audio.wav
```

The inference script reads the model architecture, tokenizer settings, audio
settings, and decoding parameters from:

```text
checkpoints/<checkpoint-name>/config.json
```

Legacy YAML-based inference is still supported:

```bash
python inference.py path/to/audio.wav configs/model_inference.yaml
```

## Train

Training uses `train.py` and YAML experiment configurations under
`configs/train/`.

Single-process training:

```bash
python train.py configs/train/setting-tau-0.8.yaml
```

Distributed or multi-GPU training with Accelerate:

```bash
accelerate launch train.py configs/train/setting-tau-0.8.yaml
```

Training configuration files define:

- model architecture parameters
- dataset paths
- tokenizer settings
- audio settings
- optimizer and scheduler settings
- checkpoint/output directories

The available tau training configurations are:

```text
configs/train/setting-tau-0.4.yaml
configs/train/setting-tau-0.6.yaml
configs/train/setting-tau-0.8.yaml
```

After training, place the exported weights and a matching `config.json` in a
checkpoint directory:

```text
checkpoints/<checkpoint-name>/
  config.json
  model.safetensors
```

The checkpoint can then be used directly with:

```bash
python inference --checkpoint <checkpoint-name> --input path/to/audio.wav
```

## Versioning

This repository uses semantic versioning. The current release is `v1.0.0`.

## License

This work is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
