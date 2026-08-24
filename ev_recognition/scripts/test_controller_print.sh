#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHONPATH=src python3 -m evr.test_controller --config config.yaml --mode print
