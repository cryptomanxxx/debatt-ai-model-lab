"""Prepare the Kaggle kernel using the exact Git commit under test."""
import argparse
import json
import re
from pathlib import Path

RUNNER = """import json
import subprocess
from pathlib import Path

repo = "/kaggle/working/debatt-ai-model-lab"
sha = __COMMIT_SHA__
run_id = __RUN_ID__
subprocess.run(["git", "clone", "--quiet", "https://github.com/cryptomanxxx/debatt-ai-model-lab.git", repo], check=True)
subprocess.run(["git", "-C", repo, "checkout", "--detach", sha], check=True)
# Kaggle provides a GPU-enabled PyTorch build; avoid overwriting it with an incompatible wheel.
subprocess.run(["python", "-m", "pip", "install", "transformers==4.53.3", "accelerate==1.9.0"], check=True)
subprocess.run(
    ["python", "scripts/smollm3_smoketest.py", "--output", "/kaggle/working/smollm3_smoketest.json"],
    cwd=repo, check=True,
)
# Written only after successful experiment execution.
with open("/kaggle/working/experiment_provenance.json", "w", encoding="utf-8") as fp:
    json.dump({"git_sha": sha, "run_id": run_id}, fp)
"""

def prepare(sha: str, username: str, output: Path, run_id: str = "local-test") -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Git commit must be a full 40-character SHA")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("Invalid Kaggle username")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run identifier")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.py").write_text(RUNNER.replace("__COMMIT_SHA__", repr(sha)).replace("__RUN_ID__", repr(run_id)), encoding="utf-8")
    metadata = {
        "id": f"{username}/debatt-ai-smollm3-gpu",
        "title": "Debatt AI SmolLM3 GPU",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    prepare(args.sha, args.username, args.output, args.run_id)
