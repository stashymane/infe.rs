#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: docker run ... <executorch|yolo26n-face> [args...]" >&2
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

op="$1"
shift

case "${op}" in
    executorch)
        export PATH="/opt/et-venv/bin:${PATH}"
        exec /usr/local/bin/executorch-entrypoint.sh "$@"
        ;;
    yolo26n-face)
        export PATH="/opt/export-venv/bin:${PATH}"
        exec /usr/local/bin/yolo26n-face-export-entrypoint.sh "$@"
        ;;
    *)
        echo "Unknown operation: ${op}" >&2
        usage
        exit 1
        ;;
esac
