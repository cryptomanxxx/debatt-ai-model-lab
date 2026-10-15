import json
from pathlib import Path

from scripts.smollm3_smoketest import load_config


def test_smoketest_configuration():
    config = load_config("configs/smollm3_smoketest.json")
    assert config["model_id"] == "HuggingFaceTB/SmolLM3-3B"
    assert isinstance(config["prompt"], str) and config["prompt"]
    assert config["max_new_tokens"] > 0


def test_missing_key_rejected(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"model_id": "dummy", "prompt": "hello"}), encoding="utf-8")
    try:
        load_config(str(path))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected missing key to be rejected")
