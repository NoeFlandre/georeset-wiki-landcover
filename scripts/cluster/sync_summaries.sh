#!/usr/bin/env bash
set -uo pipefail

ACCESS_HOST="${G5K_ACCESS_HOST:-nflandre@access.grid5000.fr}"
SITE="${G5K_SITE:-nancy}"
REMOTE_DIR="${G5K_REMOTE_DIR:-georeset}"
OUTPUT_PATH="${GEORESET_SUMMARY_OUTPUT:-data/wiki/article_summaries.json}"
INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-20}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "Syncing /home/nflandre/${REMOTE_DIR}/${OUTPUT_PATH} from ${SITE} every ${INTERVAL_SECONDS}s"

while true; do
  tmp_path="${OUTPUT_PATH}.tmp"
  if ssh -o BatchMode=yes "${ACCESS_HOST}" "ssh ${SITE} 'test -s /home/nflandre/${REMOTE_DIR}/${OUTPUT_PATH}'"; then
    if ssh -o BatchMode=yes "${ACCESS_HOST}" "ssh ${SITE} 'cat /home/nflandre/${REMOTE_DIR}/${OUTPUT_PATH}'" > "${tmp_path}" \
      && python -m json.tool "${tmp_path}" >/dev/null 2>&1; then
      mv "${tmp_path}" "${OUTPUT_PATH}"
      python - <<PY
import json
import time

with open("${OUTPUT_PATH}") as f:
    data = json.load(f)
has_thinking = any("thinking" in v for v in data.values() if isinstance(v, dict))
print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} synced entries={len(data)} has_thinking={has_thinking}", flush=True)
PY
    else
      echo "$(date '+%Y-%m-%d %H:%M:%S') skipped incomplete remote JSON"
    fi
  fi
  rm -f "${tmp_path}"
  sleep "${INTERVAL_SECONDS}"
done
