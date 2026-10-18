import json
from pathlib import Path

import pytest
from scripts.prepare_kaggle_kernel import prepare


def test_prepare_kernel(tmp_path: Path):
    sha = "a" * 40
    prepare(sha, "researcher", tmp_path, "98765-2")
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    script = (tmp_path / "run.py").read_text()
    assert metadata["id"] == "researcher/debatt-ai-smollm3-gpu"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert sha in script
    assert "98765-2" in script
    # Versions live only in requirements-kaggle.txt, and Kaggle's torch is kept.
    assert '"-r", "requirements-kaggle.txt"' in script
    assert "==" not in script
    pins = [line for line in Path("requirements-kaggle.txt").read_text().splitlines() if not line.startswith("#")]
    assert not any(line.startswith("torch") for line in pins)


def test_reject_invalid_sha(tmp_path: Path):
    with pytest.raises(ValueError):
        prepare("main", "researcher", tmp_path)


def test_reject_invalid_username(tmp_path: Path):
    with pytest.raises(ValueError):
        prepare("a" * 40, "bad/name", tmp_path)


def test_invalid_run_identifier(tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid run identifier"):
        prepare("a" * 40, "researcher", tmp_path, "bad/run")


def test_prepare_pilot_kernel(tmp_path: Path):
    prepare("b" * 40, "researcher", tmp_path, "555-1", task="pilot")
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    script = (tmp_path / "run.py").read_text()
    assert metadata["id"] == "researcher/debatt-ai-pilot-gpu"
    assert metadata["dataset_sources"] == ["researcher/debatt-ai-eval-data"]
    assert "scripts/pilot_eval.py" in script and "'--run-id', '555-1'" in script
    assert "smollm3_smoketest.py" not in script


def test_smoketest_kernel_has_no_dataset(tmp_path: Path):
    prepare("b" * 40, "researcher", tmp_path, "555-1")
    assert json.loads((tmp_path / "kernel-metadata.json").read_text())["dataset_sources"] == []


def test_unknown_task_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown task"):
        prepare("b" * 40, "researcher", tmp_path, "555-1", task="finetune")


def test_prepare_training_kernel(tmp_path: Path):
    prepare("c" * 40, "researcher", tmp_path, "777-1", task="train")
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    script = (tmp_path / "run.py").read_text()
    assert metadata["id"] == "researcher/debatt-ai-train-editor-gpu"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == ["researcher/debatt-ai-eval-data"]
    assert "scripts/train_editor.py" in script and "'--run-id', '777-1'" in script
