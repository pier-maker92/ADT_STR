#!/usr/bin/env bash
# Use conda env adt3
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate adt3
pip install -r requirements.txt
