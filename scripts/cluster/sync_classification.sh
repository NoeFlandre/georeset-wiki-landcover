#!/usr/bin/env bash
set -uo pipefail

ACCESS_HOST="${G5K_ACCESS_HOST:-nflandre@access.grid5000.fr}"
SITE="${G5K_SITE:-nancy}"
REMOTE_DIR="${G5K_REMOTE_DIR:-georeset}"
TASK="${GEORESET_CLASSIFICATION_TASK:?Set GEORESET_CLASSIFICATION_TASK}"
TEXT_SOURCE="${GEORESET_CLASSIFICATION_TEXT_SOURCE:?Set GEORESET_CLASSIFICATION_TEXT_SOURCE}"
OUTPUT_PREFIX="data/classification/${TASK}_${TEXT_SOURCE}"
INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-20}"

mkdir -p "$(dirname "${OUTPUT_PREFIX}_predictions.json")"

echo "Syncing ${OUTPUT_PREFIX}_predictions.json and _metrics.json from ${SITE} every ${INTERVAL_SECONDS}s"

while true; do
  for suffix in predictions metrics; do
    remote_path="/home/nflandre/${REMOTE_DIR}/${OUTPUT_PREFIX}_${suffix}.json"
    local_path="${OUTPUT_PREFIX}_${suffix}.json"
    tmp_path="${local_path}.tmp"
    if ssh -o BatchMode=yes "${ACCESS_HOST}" "ssh ${SITE} 'test -s ${remote_path}'"; then
      if ssh -o BatchMode=yes "${ACCESS_HOST}" "ssh ${SITE} 'cat ${remote_path}'" > "${tmp_path}" \
        && python -m json.tool "${tmp_path}" >/dev/null 2>&1; then
        mv "${tmp_path}" "${local_path}"
        python - <<PY
import json
import time

with open("${local_path}") as f:
    data = json.load(f)
n = len(data)
errors = sum(1 for v in data.values() if isinstance(v, dict) and v.get("parse_status") == "error")
print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} synced entries={n} errors={errors}", flush=True)
PY
      else
        echo "$(date '+%Y-%m-%d %H:%M:%S') skipped incomplete remote JSON: ${remote_path}"
      fi
    fi
    rm -f "${tmp_path}"
  done
  sleep "${INTERVAL_SECONDS}"
done
