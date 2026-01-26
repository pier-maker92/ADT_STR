# This script converts the TMIDT dataset to a parquet file
# you can download the dataset from http://ifs.tuwien.ac.at/~vogl/dafx2018/

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from config import TMIDTToParquetConfig
from utils.config_utils import load_config_from_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config file")
    args = parser.parse_args()
    config = load_config_from_yaml(args.config_path)
    config_plain = config["shared"]
    config_plain.update(config["TMIDTToParquet"])
    config = TMIDTToParquetConfig(**config_plain)
