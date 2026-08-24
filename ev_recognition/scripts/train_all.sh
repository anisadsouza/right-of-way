#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m evr.train_vision --data data/images --epochs 12
python -m evr.train_audio --data data/audio
