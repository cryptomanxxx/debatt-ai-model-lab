import json
from pathlib import Path

import pytest
from scripts.smollm3_smoketest import load_config, require_cuda


def test_smoketest_configuration():
    config = load_config("configs/smollm3_smoketest.json")
    assert config["model_id"] == "HuggingFaceTB/SmolLM3-3B"
    assert len(config["model_revision"]) == 40
    assert config["enable_thinking"] is False
    assert isinstance(config["prompt"], str) and config["prompt"]
    assert config["max_new_tokens"] > 0


def test_missing_key_rejected(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"model_id": "dummy", "prompt": "hello"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing config keys"):
        load_config(str(path))


def test_cuda_required():
    class NoCuda:
        class cuda:
            @staticmethod
            def is_available():
                return False
    with pytest.raises(RuntimeError, match="CUDA GPU required"):
        require_cuda(NoCuda)


def test_invalid_revision_rejected(tmp_path: Path):
    config = load_config("configs/smollm3_smoketest.json")
    config["model_revision"] = "main"
    path = tmp_path / "bad_revision.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        load_config(str(path))
