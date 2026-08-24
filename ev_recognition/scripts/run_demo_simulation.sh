#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m evr.demo_simulation --config config.yaml --speed 4
