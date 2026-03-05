#!/usr/bin/env bash
set -euo pipefail

TARGET_GB="${1:-38}"
MIN_FREE_GB="${2:-8}"
MAX_RETRIES="${3:-8}"

LOG_DIR="/root/autodl-tmp/objaverse_subset"
LOG_FILE="${LOG_DIR}/download.log"

mkdir -p "${LOG_DIR}"
echo "" >> "${LOG_FILE}"
echo "===== restart $(date -u '+%F %T UTC') =====" >> "${LOG_FILE}"

nohup bash -lc "cd /root/cube && /root/cube/dataset/run_objaverse_subset_download.sh --target-gb ${TARGET_GB} --min-free-gb ${MIN_FREE_GB} --max-retries ${MAX_RETRIES}" >> "${LOG_FILE}" 2>&1 &
echo "Started PID: $!"
echo "Log file: ${LOG_FILE}"
