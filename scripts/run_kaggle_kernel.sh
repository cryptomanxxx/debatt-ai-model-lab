#!/usr/bin/env bash
set -euo pipefail

kernel="${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}"
mkdir -p kaggle-results

# Save logs even when submission or run fails.
kaggle kernels push -p kaggle-submission --accelerator NvidiaTeslaT4 2>&1 | tee kaggle-results/submission.log

# API status refers to the most recent kernel version. The workflow uses a
# serialized concurrency group to avoid overlapping submissions from this repo.
completed=0
for i in $(seq 1 100); do
  status="$(kaggle kernels status "$kernel" 2>&1)" || {
    echo "$status" | tee -a kaggle-results/status.log
    exit 1
  }
  echo "$status" | tee -a kaggle-results/status.log
  normalized="$(printf '%s' "$status" | tr '[:upper:]' '[:lower:]')"
  if echo "$normalized" | grep -Eq 'complete|success'; then
    completed=1
    break
  fi
  if echo "$normalized" | grep -Eq 'error|failed|cancel'; then
    echo "::error::Kaggle run failed"
    kaggle kernels logs "$kernel" > kaggle-results/kernel.log 2>&1 || true
    exit 1
  fi
  sleep 60
done

if [ "$completed" -ne 1 ]; then
  echo "::error::Kaggle run timed out"
  exit 1
fi

kaggle kernels output "$kernel" -p kaggle-results --force
if [ ! -s kaggle-results/smollm3_smoketest.json ]; then
  echo "::error::Expected experiment JSON missing or empty"
  kaggle kernels logs "$kernel" > kaggle-results/kernel.log 2>&1 || true
  exit 1
fi
python -m json.tool kaggle-results/smollm3_smoketest.json >/dev/null
echo "GPU smoke test completed; result downloaded."
