import json
from pathlib import Path

import pytest
from scripts.prepare_kaggle_kernel import prepare


def test_prepare_kernel(tmp_path: Path):
    sha = "a" * 40
    prepare(sha, "researcher", tmp_path)
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    script = (tmp_path / "run.py").read_text()
    assert metadata["id"] == "researcher/debatt-ai-smollm3-gpu"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert sha in script


def test_reject_invalid_sha(tmp_path: Path):
    with pytest.raises(ValueError):
        prepare("main", "researcher", tmp_path)


def test_reject_invalid_username(tmp_path: Path):
    with pytest.raises(ValueError):
        prepare("a" * 40, "bad/name", tmp_path)
