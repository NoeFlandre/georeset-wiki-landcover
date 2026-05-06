#!/usr/bin/env bash
set -euo pipefail

ACCESS_HOST="${G5K_ACCESS_HOST:-nflandre@access.grid5000.fr}"
SITE="${G5K_SITE:-nancy}"
REMOTE_DIR="${G5K_REMOTE_DIR:-georeset}"
OUTPUT_PATH="data/wiki/article_summaries.json"
JOB_SCRIPT="scripts/grid5000/run_summarization_job.sh"

mkdir -p data/wiki

echo "Preparing remote directory ${SITE}/${REMOTE_DIR}"
ssh -o BatchMode=yes "${ACCESS_HOST}" "mkdir -p ${SITE}/${REMOTE_DIR}"

echo "Syncing repository to Grid5000"
rsync -az --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'build' \
  --exclude 'dist' \
  --exclude '*.egg-info' \
  --exclude 'data/wiki/article_summaries.json' \
  --exclude 'data/corine' \
  --exclude 'data/maps' \
  --exclude 'data/osm' \
  --exclude 'data/distribution' \
  --exclude '*.db' \
  ./ "${ACCESS_HOST}:${SITE}/${REMOTE_DIR}/"

echo "Submitting OAR job"
SUBMIT_OUTPUT="$(ssh -o BatchMode=yes "${ACCESS_HOST}" "ssh ${SITE} 'cd ${REMOTE_DIR} && chmod +x ${JOB_SCRIPT} && oarsub -S ./${JOB_SCRIPT}'")"
echo "${SUBMIT_OUTPUT}"

JOB_ID="$(printf '%s\n' "${SUBMIT_OUTPUT}" | sed -n 's/.*OAR_JOB_ID=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [ -z "${JOB_ID}" ]; then
  echo "Could not parse OAR job id from submission output." >&2
  exit 1
fi

echo "Submitted OAR job ${JOB_ID}"
echo "Watch status: ssh -o BatchMode=yes ${ACCESS_HOST} \"ssh ${SITE} 'oarstat -j ${JOB_ID}'\""
echo "Watch stderr: ssh -o BatchMode=yes ${ACCESS_HOST} \"ssh ${SITE} 'tail -f /home/nflandre/${REMOTE_DIR}/OAR_${JOB_ID}.err'\""
echo "Syncing ${OUTPUT_PATH}; press Ctrl+C to stop after the job finishes."

while true; do
  if ssh -o BatchMode=yes "${ACCESS_HOST}" "ssh ${SITE} 'test -f /home/nflandre/${REMOTE_DIR}/${OUTPUT_PATH}'"; then
    ssh -o BatchMode=yes "${ACCESS_HOST}" "ssh ${SITE} 'cat /home/nflandre/${REMOTE_DIR}/${OUTPUT_PATH}'" > "./${OUTPUT_PATH}.tmp"
    mv "./${OUTPUT_PATH}.tmp" "./${OUTPUT_PATH}"
  fi
  sleep 30
done
