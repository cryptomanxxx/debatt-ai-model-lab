"""Runs scripts/run_kaggle_kernel.sh against a fake kaggle CLI, so no Kaggle account is needed."""
import json
import os
import stat
import subprocess
from pathlib import Path

FAKE_KAGGLE = """#!/usr/bin/env bash
echo "$@" >> "$FAKE_LOG"
case "$1 $2" in
  "kernels status") echo "user/slug has status \\"KernelWorkerStatus.COMPLETE\\"" ;;
  "kernels output")
    mkdir -p "$5"
    printf '{"git_sha": "%s", "run_id": "%s"}' "$FAKE_SHA" "$FAKE_RUN" > "$5/experiment_provenance.json"
    echo '{"cuda_available": true}' > "$5/train_results.json" ;;
  *) echo "unexpected: $@" >&2; exit 1 ;;
esac
"""


def run(tmp_path: Path, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "kaggle"
    fake.write_text(FAKE_KAGGLE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    script = Path("scripts/run_kaggle_kernel.sh").resolve()
    base = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "KAGGLE_USERNAME": "user", "KAGGLE_KERNEL_SLUG": "slug",
            "KAGGLE_RESULT_FILE": "train_results.json", "GITHUB_SHA": "a" * 40, "FAKE_LOG": str(tmp_path / "calls")}
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=base | env, capture_output=True, text=True)
    return result, (tmp_path / "calls").read_text()


def test_retrieval_fetches_an_earlier_run_without_pushing(tmp_path):
    result, calls = run(tmp_path, KAGGLE_RETRIEVE_ONLY="true", EXPERIMENT_RUN_ID="5-1", EXPECTED_SHA="b" * 40,
                        FAKE_SHA="b" * 40, FAKE_RUN="5-1")
    assert result.returncode == 0, result.stderr
    assert "push" not in calls and "kernels output" in calls
    assert "Verified Kaggle result provenance for " + "b" * 40 in result.stdout
    assert json.loads((tmp_path / "kaggle-results/train_results.json").read_text())["cuda_available"]


def test_retrieval_rejects_a_result_from_another_run(tmp_path):
    result, _ = run(tmp_path, KAGGLE_RETRIEVE_ONLY="true", EXPERIMENT_RUN_ID="5-1", EXPECTED_SHA="b" * 40,
                    FAKE_SHA="b" * 40, FAKE_RUN="6-1")
    assert result.returncode != 0 and "provenance does not match" in result.stderr
