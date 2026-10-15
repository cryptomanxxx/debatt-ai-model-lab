"""Tests for automated, provenance-checked result archival."""
import json
from pathlib import Path

import pytest

from scripts.archive_experiment import archive


SHA = "a" * 40
RUN = "123456-1"
URL = "https://github.com/cryptomanxxx/debatt-ai-model-lab/actions/runs/123456"


def fixture(tmp_path):
    source = tmp_path / "kaggle-results"
    source.mkdir()
    sample = {
        "model_id": "HuggingFaceTB/SmolLM3-3B",
        "model_revision": "b" * 40,
        "prompt": "Why?",
        "rendered_prompt": "<|im_start|>user\nWhy?<|im_end|>",
        "response": "Because science.",
        "gpu_name": "Tesla T4",
        "torch_version": "2.11.0",
        "cuda_runtime_version": "12.8",
        "packages": {"transformers": "4.53.3"},
        "generation_settings": {"max_new_tokens": 64, "do_sample": False},
        "generated_tokens": 4,
        "finish_reason": "eos",
        "cuda_available": True,
    }
    (source / "smollm3_smoketest.json").write_text(json.dumps(sample))
    (source / "experiment_provenance.json").write_text(json.dumps({"git_sha": SHA, "run_id": RUN}))
    return source


def test_archives_compact_record_with_rendered_prompt(tmp_path):
    source = fixture(tmp_path)
    root = archive(source, tmp_path / "results", SHA, RUN, URL)
    record = json.loads((root / "result.json").read_text())
    assert record["inference"]["rendered_prompt"].startswith("<|im_start|>")
    assert record["source"]["run_attempt"] == 1
    assert record["software"]["pytorch_cuda_build_version"] == "12.8"
    assert (root / "experiment-report.md").is_file()


def test_rejects_wrong_provenance(tmp_path):
    source = fixture(tmp_path)
    with pytest.raises(ValueError, match="Provenance mismatch"):
        archive(source, tmp_path / "results", "c" * 40, RUN, URL)


def test_never_overwrites_existing_run(tmp_path):
    source = fixture(tmp_path)
    archive(source, tmp_path / "results", SHA, RUN, URL)
    with pytest.raises(FileExistsError):
        archive(source, tmp_path / "results", SHA, RUN, URL)


def test_rejects_incomplete_gpu_result(tmp_path):
    source = fixture(tmp_path)
    data_file = source / "smollm3_smoketest.json"
    data = json.loads(data_file.read_text())
    data["cuda_available"] = False
    data_file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="did not complete"):
        archive(source, tmp_path / "results", SHA, RUN, URL)
