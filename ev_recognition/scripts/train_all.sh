#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m evr.validate_dataset
python -m evr.train_vision --data data/images --epochs 15
python -m evr.train_audio --data data/audio
