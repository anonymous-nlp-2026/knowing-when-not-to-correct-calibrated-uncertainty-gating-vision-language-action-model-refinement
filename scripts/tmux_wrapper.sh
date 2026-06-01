#!/bin/bash
EXP_ID="$1"
shift
SENTINEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/logs/sentinels"
mkdir -p "$SENTINEL_DIR"

eval "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS" > "$SENTINEL_DIR/${EXP_ID}.sentinel"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT_CODE=0" >> "$SENTINEL_DIR/${EXP_ID}.sentinel"
else
    echo "FAILED" > "$SENTINEL_DIR/${EXP_ID}.sentinel"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT_CODE=$EXIT_CODE" >> "$SENTINEL_DIR/${EXP_ID}.sentinel"
fi

exit $EXIT_CODE
