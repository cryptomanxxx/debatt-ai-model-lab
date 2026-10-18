#!/usr/bin/env bash
set -euo pipefail

kernel="${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}"
# The task's result file; the smoke test stays the default.
result_file="${KAGGLE_RESULT_FILE:-smollm3_smoketest.json}"
# Kaggle run time limit (seconds) and how many one-minute status polls to wait; training needs more than evaluation.
kernel_timeout="${KAGGLE_KERNEL_TIMEOUT:-5700}"
max_polls="${KAGGLE_MAX_POLLS:-100}"
: "${GITHUB_SHA:?GITHUB_SHA must be set}"
: "${EXPERIMENT_RUN_ID:?EXPERIMENT_RUN_ID must be set}"
mkdir -p kaggle-results

# CLI versions can print "Kernel push error" yet return exit status 0.
# Fail closed unless the CLI explicitly confirms a successful new version.
if ! kaggle kernels push -p kaggle-submission --accelerator NvidiaTeslaT4 --timeout "$kernel_timeout" > kaggle-results/submission.log 2>&1; then
  cat kaggle-results/submission.log
  echo "::error::Kaggle submission command failed"
  exit 1
fi
cat kaggle-results/submission.log
if grep -Eiq '^[[:space:]]*(kernel push error|error pushing|failed:|traceback)' kaggle-results/submission.log; then
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
for i in $(seq 1 "$max_polls"); do
  status="$(kaggle kernels status "$kernel" 2>&1)" || {
    echo "$status" | tee -a kaggle-results/status.log
    exit 1
  }
  echo "$status" | tee -a kaggle-results/status.log
  # Extract the CLI's quoted worker-status field only. Never match username,
  # kernel slug or a progress URL that happens to contain "complete".
  worker_status="$(printf '%s\n' "$status" | grep -Eo 'KernelWorkerStatus\.[A-Z_]+' | head -n 1 || true)"
  case "$worker_status" in
    KernelWorkerStatus.COMPLETE)
      completed=1
      break
      ;;
    KernelWorkerStatus.ERROR|KernelWorkerStatus.FAILED|KernelWorkerStatus.CANCELED|KernelWorkerStatus.CANCELLED|KernelWorkerStatus.CANCEL_ACKNOWLEDGED|KernelWorkerStatus.CANCELLING|KernelWorkerStatus.CANCELING)
      echo "::error::Kaggle run failed: $worker_status"
      kaggle kernels output "$kernel" -p kaggle-results --force > kaggle-results/output-download.log 2>&1 || true
      exit 1
      ;;
    KernelWorkerStatus.QUEUED|KernelWorkerStatus.RUNNING)
      ;;
    *)
      echo "::error::Unknown or missing Kaggle worker status: $status"
      exit 1
      ;;
  esac
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
python - "$GITHUB_SHA" "$EXPERIMENT_RUN_ID" "$result_file" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path("kaggle-results")
provenance = result_dir / "experiment_provenance.json"
result = result_dir / sys.argv[3]
if not provenance.is_file() or not result.is_file():
    raise SystemExit("Missing experiment result or provenance: run not verified")
record = json.loads(provenance.read_text(encoding="utf-8"))
if record.get("git_sha") != sys.argv[1] or record.get("run_id") != sys.argv[2]:
    raise SystemExit("Kaggle result provenance does not match the GitHub commit and workflow run")
data = json.loads(result.read_text(encoding="utf-8"))
if not data.get("cuda_available") or (sys.argv[3] == "smollm3_smoketest.json" and not data.get("response")):
    raise SystemExit("Kaggle GPU result is invalid or incomplete")
print("Verified Kaggle result provenance for", sys.argv[1], sys.argv[2])
PY
