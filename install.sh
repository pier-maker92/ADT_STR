#!/usr/bin/env bash
# Install deps. pytorch-fast-transformers needs torch at build time; pip's
# isolated build doesn't see it, so we install it with --no-build-isolation.
set -e
pip install -r requirements.txt
pip install --no-build-isolation pytorch-fast-transformers
