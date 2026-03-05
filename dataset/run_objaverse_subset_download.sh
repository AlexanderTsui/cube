#!/usr/bin/env bash
set -euo pipefail

source /etc/network_turbo
export PYTHONUNBUFFERED=1
python -u /root/cube/dataset/download_objaverse_subset.py --output-root /root/autodl-tmp "$@"
