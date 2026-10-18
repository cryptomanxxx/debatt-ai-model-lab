#!/usr/bin/env bash
# Upload an export directory as a private Kaggle dataset version and wait until it is ready.
set -euo pipefail

dir="$1"
: "${KAGGLE_USERNAME:?KAGGLE_USERNAME must be set}"
: "${GITHUB_SHA:?GITHUB_SHA must be set}"
: "${EXPERIMENT_RUN_ID:?EXPERIMENT_RUN_ID must be set}"
dataset="${KAGGLE_USERNAME}/debatt-ai-eval-data"

# The pilot refuses any dataset version whose run_id differs, so a stale version can never be evaluated.
printf '{"run_id": "%s", "git_sha": "%s"}\n' "$EXPERIMENT_RUN_ID" "$GITHUB_SHA" > "$dir/provenance.json"
printf '{"title": "Debatt AI eval data", "id": "%s", "licenses": [{"name": "other"}]}\n' "$dataset" > "$dir/dataset-metadata.json"

# Kaggle CLI versions can print errors yet exit 0; fail closed on error text too.
upload() {
  if ! "$@" > upload.log 2>&1 || grep -Eiq 'error|failed|traceback' upload.log; then
    cat upload.log
    return 1
  fi
  cat upload.log
}
if kaggle datasets status "$dataset" > status.log 2>&1 && grep -Eiq '^(ready|pending)' status.log; then
  upload kaggle datasets version -p "$dir" -m "Export for run ${EXPERIMENT_RUN_ID}" || { echo "::error::Dataset version upload failed"; exit 1; }
else
  upload kaggle datasets create -p "$dir" || { echo "::error::Dataset creation failed"; exit 1; }
fi

for i in $(seq 1 40); do
  status="$(kaggle datasets status "$dataset" 2>&1 || true)"
  echo "$status"
  if printf '%s\n' "$status" | grep -Eiq '^ready'; then
    exit 0
  fi
  sleep 15
done
echo "::error::Kaggle dataset did not become ready"
exit 1
