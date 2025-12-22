import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shutil
from pathlib import Path
from utils import load_yaml, recursive_merge
from config import ClapConfig
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing 'gold' directories under each instrument label",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    default_path = Path(__file__).parent / "config_default.yaml"

    cfg = load_yaml(default_path)
    experiment_cfg = load_yaml(config_path)
    clap_cfg = recursive_merge(cfg, experiment_cfg)["clap_config"]
    clap_cfg.update(cfg["shared"])
    clap_cfg = ClapConfig(**clap_cfg)

    reference_root = Path(clap_cfg.reference_root)
    if not reference_root.exists():
        raise FileNotFoundError(f"reference_root does not exist: {reference_root}")

    augmented_root = Path(f"{reference_root}_clap_augmented")
    augmented_root.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_count = 0
    pb = tqdm(reference_root.iterdir())
    for item in pb:
        if not item.is_dir():
            continue
        instrument_label = item.name
        destination_gold = augmented_root / instrument_label / "gold"
        if destination_gold.exists():
            if args.overwrite:
                shutil.rmtree(destination_gold)
            else:
                print(f"Destination already exists, skipping copy: {destination_gold}. Use --overwrite to replace.")
                skipped_count += 1
                continue
        destination_gold.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(item, destination_gold)
            copied_count += 1
            pb.set_description(f"Copied '{instrument_label}' to: {destination_gold}")
        except Exception as e:
            print(f"Failed to copy '{item}' -> '{destination_gold}': {e}")
            continue
    pb.close()
    print(f"Finished. Copied: {copied_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
