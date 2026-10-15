#!/usr/bin/env bash
set -euo pipefail

kernel="${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}"
: "${GITHUB_SHA:?GITHUB_SHA must be set}"
mkdir -p kaggle-results

# CLI versions can print "Kernel push error" yet return exit status 0.
# Fail closed unless the CLI explicitly confirms a successful new version.
if ! kaggle kernels push -p kaggle-submission --accelerator NvidiaTeslaT4 > kaggle-results/submission.log 2>&1; then
  cat kaggle-results/submission.log
  echo "::error::Kaggle submission command failed"
  exit 1
fi
cat kaggle-results/submission.log
if grep -Eiq 'kernel push error|error pushing|failed|traceback' kaggle-results/submission.log; then
  echo "::error::Kaggle rejected the submission"
  exit 1
fi
if ! grep -Eiq 'kernel version [0-9]+ successfully pushed|kernel version [0-9]+ pushed' kaggle-results/submission.log; then
  echo "::error::Kaggle submission did not confirm a new kernel version"
  exit 1
fi

# Only the exact completed state qualifies; never trust a previous version's
# result merely because status reports COMPLETE.
completed=0
for i in $(seq 1 100); do
  status="$(kaggle kernels status "$kernel" 2>&1)" || {
    echo "$status" | tee -a kaggle-results/status.log
    exit 1
  }
  echo "$status" | tee -a kaggle-results/status.log
  normalized="$(printf '%s' "$status" | tr '[:upper:]' '[:lower:]')"
  if echo "$normalized" | grep -Eq '(^|[^a-z])error([^a-z]|$)|(^|[^a-z])failed([^a-z]|$)|(^|[^a-z])cancelled([^a-z]|$)'; then
    echo "::error::Kaggle run failed"
    kaggle kernels output "$kernel" -p kaggle-results --force > kaggle-results/output-download.log 2>&1 || true
    exit 1
  fi
  if echo "$normalized" | grep -Eq '(^|[^a-z])complete([^a-z]|$)'; then
    completed=1
    break
  fi
  sleep 60
done

if [ "$completed" -ne 1 ]; then
  echo "::error::Kaggle run timed out"
  exit 1
fi

kaggle kernels output "$kernel" -p kaggle-results --force > kaggle-results/output-download.log 2>&1 || {
  cat kaggle-results/output-download.log
  exit 1
}

# Provenance is written by the remote runner AFTER the model test succeeds.
# A stale result from a previous Kaggle kernel version must be rejected.
python - "$GITHUB_SHA" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path("kaggle-results")
provenance = result_dir / "experiment_provenance.json"
result = result_dir / "smollm3_smoketest.json"
if not provenance.is_file() or not result.is_file():
    raise SystemExit("Missing experiment result or provenance: run not verified")
record = json.loads(provenance.read_text(encoding="utf-8"))
if record.get("git_sha") != sys.argv[1]:
    raise SystemExit("Kaggle result SHA does not match the submitted GitHub commit")
data = json.loads(result.read_text(encoding="utf-8"))
if not data.get("cuda_available") or not data.get("response"):
    raise SystemExit("Kaggle GPU result is invalid or incomplete")
print("Verified Kaggle result provenance for", sys.argv[1])
PY
