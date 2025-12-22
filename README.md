# ADT_STR

Automatic Drum Transcription model.

## Project Structure

```
ADT_STR/
├── train.py              # Training script
├── inference.py          # Inference and evaluation script
├── model.py              # Model architecture
├── config.py             # Configuration dataclasses
├── configs/              # Configuration files
│   ├── config_default.yaml
│   ├── train/            # Training configs
│   └── eval/             # Evaluation configs
├── modules/              # Core modules
│   ├── midi_tokenizer.py
│   ├── synthetiser.py
│   └── segmenter.py
├── data_modules/         # Dataset and data processing
│   ├── train_dataset.py
│   └── eval_dataset.py
└── utils/                # Utility functions
```

## Configuration

The configuration system uses YAML files with a default configuration (`configs/config_default.yaml`) that gets merged with experiment-specific configs.

Training configs are placed in `configs/train/`, evaluation configs in `configs/eval/`.

## Training

The training script uses HuggingFace's `Trainer` and is designed to work with `accelerate` for multi-GPU training.

### Single GPU

```bash
python train.py configs/train/setting-1.yaml
```

### Multi-GPU with Accelerate

```bash
accelerate launch train.py configs/train/setting-1.yaml
```

The training config should specify:
- `training.batch_size`: Per-device batch size
- `training.num_epochs`: Number of training epochs
- `training.learning_rate`: Learning rate
- `LakhDatasetConfig.dataset_path`: Path to the training dataset
- `synthetiser.oneshot_path`: Path to drum oneshot samples
- `logging.output_dir`: Output directory for checkpoints

## Inference

Run evaluation on ENST or MDB datasets:

```bash
python inference.py configs/eval/ENSTinference.yaml
```

```bash
python inference.py configs/eval/MDBinference.yaml
```

The inference config should specify:
- `inference.checkpoint_path`: Path to the model checkpoint
- `EvalDatasetConfig.dataset_path`: Path to the evaluation dataset
- `inference.output_path`: Output directory for results

## Results

![Results on ENST and MDB datasets](assets/results_ENST_MDB.png)

