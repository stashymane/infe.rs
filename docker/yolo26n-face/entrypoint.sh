#!/usr/bin/env bash
set -euo pipefail

IMGSZ="${IMGSZ:-192}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

echo "YOLO26n-face ExecuTorch export (IMGSZ=${IMGSZ}, OUTPUT_DIR=${OUTPUT_DIR})"
exec python3 /usr/local/share/infers/yolo26n-face/export.py
