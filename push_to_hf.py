import argparse
import shutil
import tempfile
from pathlib import Path

import yaml
from huggingface_hub import HfApi, create_repo

from build_model import build_model
from utils.config_utils import deep_merge_dicts, load_config


DEFAULT_CONFIG = Path("configs/model_inference.yaml")
DEFAULT_EXPORT_DIR = Path("hf_export")


def find_checkpoint_dirs(root: Path) -> list[Path]:
    checkpoint_dirs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if any(
            (child / filename).exists()
            for filename in ("model.safetensors", "model.safetensor", "pytorch_model.bin")
        ):
            checkpoint_dirs.append(child)
    return checkpoint_dirs


def write_temp_config(base_config: Path, checkpoint_dir: Path) -> Path:
    checkpoint_config = checkpoint_dir / "config.json"
    cfg = load_config(str(checkpoint_config if checkpoint_config.exists() else base_config))
    cfg = deep_merge_dicts(cfg, {"inference": {"checkpoint_path": str(checkpoint_dir)}})
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    return Path(tmp.name)


def merged_config_for_export(config_path: Path) -> dict:
    default_cfg = load_config("configs/config_default.yaml")
    experiment_cfg = load_config(str(config_path))
    return deep_merge_dicts(default_cfg, experiment_cfg)


def export_variant(checkpoint_dir: Path, base_config: Path, export_dir: Path) -> dict:
    tmp_config = write_temp_config(base_config, checkpoint_dir)
    try:
        model, merged_cfg = build_model(str(tmp_config), device="cpu")
    finally:
        tmp_config.unlink(missing_ok=True)

    variant_dir = export_dir / checkpoint_dir.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(variant_dir, safe_serialization=True)

    with (variant_dir / "adt_config.yaml").open("w") as f:
        yaml.safe_dump(merged_cfg, f, sort_keys=False)

    params = sum(p.numel() for p in model.parameters())
    return {
        "name": checkpoint_dir.name,
        "path": str(variant_dir),
        "parameters": params,
        "checkpoint": str(checkpoint_dir),
    }


def copy_runtime_files(export_dir: Path) -> None:
    for filename in (
        "adt_transcriber.py",
        "build_model.py",
        "config.py",
        "inference",
        "inference.py",
        "model.py",
        "pyproject.toml",
    ):
        path = Path(filename)
        if path.exists():
            shutil.copy2(path, export_dir / path.name)

    for dirname in ("modules", "utils"):
        source = Path(dirname)
        if source.exists():
            shutil.copytree(
                source,
                export_dir / dirname,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )


def cleanup_export_dir(export_dir: Path) -> None:
    for path in export_dir.rglob(".DS_Store"):
        path.unlink(missing_ok=True)
    for path in sorted(export_dir.rglob("__pycache__"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)


def write_model_card(
    export_dir: Path,
    variants: list[dict],
    default_variant: str,
    repo_id: str | None = None,
) -> None:
    repo_id = repo_id or "your-username/adt-str"
    rows = "\n".join(
        f"| `{variant['name']}` | `{variant['name']}` | {variant['parameters']:,} |"
        for variant in variants
    )
    readme = f"""---
library_name: transformers
tags:
- audio
- automatic-drum-transcription
- music-information-retrieval
---

# ADT_STR

Automatic Drum Transcription model exported from this repository.

Default variant: `{default_variant}`.

## Quick start

```python
from huggingface_hub import snapshot_download
import sys

repo_dir = snapshot_download("{repo_id}")
sys.path.insert(0, repo_dir)

from adt_transcriber import ADTTranscriber

transcriber = ADTTranscriber.from_pretrained(repo_dir, variant="{default_variant}")
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

## Variants

| Variant | Folder | Parameters |
| --- | --- | ---: |
{rows}

## Local loading

```python
from adt_transcriber import ADTTranscriber

transcriber = ADTTranscriber.from_pretrained(".", variant="{default_variant}")
```

You can choose `setting-tau-0.4`, `setting-tau-0.6`, or `setting-tau-0.8` with
the `variant` argument.
"""
    (export_dir / "README.md").write_text(readme)


def copy_default_variant(export_dir: Path, default_variant: str) -> None:
    default_dir = export_dir / default_variant
    for path in default_dir.iterdir():
        target = export_dir / path.name
        if path.is_file():
            shutil.copy2(path, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ADT_STR checkpoints in Hugging Face format and optionally upload them."
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional Hugging Face repo id, e.g. username/adt-drum-transcription.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints",
        type=Path,
        help="Directory containing checkpoint variant folders.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        type=Path,
        help="Base inference config used for architecture/tokenizer/shared settings.",
    )
    parser.add_argument(
        "--export-dir",
        default=DEFAULT_EXPORT_DIR,
        type=Path,
        help="Local directory to write Hugging Face export files.",
    )
    parser.add_argument(
        "--variant",
        default="all",
        help="Checkpoint folder name to export, or 'all'.",
    )
    parser.add_argument(
        "--default-variant",
        default="setting-tau-0.8",
        help="Variant copied to the export root as the default model.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/upload to a private Hugging Face repository.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dirs = find_checkpoint_dirs(args.checkpoint_root)
    if args.variant != "all":
        checkpoint_dirs = [path for path in checkpoint_dirs if path.name == args.variant]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found for variant={args.variant!r}")

    args.export_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        export_variant(checkpoint_dir, args.config, args.export_dir)
        for checkpoint_dir in checkpoint_dirs
    ]

    exported_names = {variant["name"] for variant in variants}
    if args.default_variant not in exported_names:
        raise ValueError(
            f"Default variant {args.default_variant!r} was not exported. "
            f"Available: {sorted(exported_names)}"
        )
    copy_default_variant(args.export_dir, args.default_variant)
    copy_runtime_files(args.export_dir)
    write_model_card(args.export_dir, variants, args.default_variant, args.repo_id)
    cleanup_export_dir(args.export_dir)

    print("Exported variants:")
    for variant in variants:
        print(
            f"- {variant['name']}: {variant['parameters']:,} parameters -> {variant['path']}"
        )

    if args.repo_id:
        create_repo(args.repo_id, private=args.private, exist_ok=True)
        HfApi().upload_folder(
            repo_id=args.repo_id,
            folder_path=str(args.export_dir),
            repo_type="model",
            ignore_patterns=[".DS_Store", "**/.DS_Store", "__pycache__/*", "**/__pycache__/*"],
        )
        print(f"Uploaded to https://huggingface.co/{args.repo_id}")
    else:
        print(f"Local export ready at: {args.export_dir}")
        print("Pass --repo-id username/repo-name to upload.")


if __name__ == "__main__":
    main()
