import json
from pathlib import Path

import pytest
from scripts.smollm3_smoketest import fixed_strftime, load_config, require_cuda, validate_generation


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

def test_completed_generation():
    assert validate_generation([42, 2], 2, 64, "Valid answer.") == "eos"


@pytest.mark.parametrize("ids,response", [([], ""), ([2], ""), ([42, 42], "Unfinished")])
def test_incomplete_response_rejected(ids, response):
    with pytest.raises(RuntimeError, match="Smoke test failed"):
        validate_generation(ids, 2, 2, response)


def test_missing_eos_rejected_even_under_budget():
    with pytest.raises(RuntimeError, match="did not terminate with EOS"):
        validate_generation([42], 2, 64, "Some text")


def test_invalid_template_date_rejected(tmp_path: Path):
    config = load_config("configs/smollm3_smoketest.json")
    config["template_date"] = "today"
    path = tmp_path / "bad_date.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="template_date"):
        load_config(str(path))


def test_fixed_strftime_ignores_current_day():
    assert fixed_strftime("2026-01-01")("%d %B %Y") == "01 January 2026"


def test_template_date_overrides_chat_template_clock():
    pytest.importorskip("jinja2")
    chat_template_utils = pytest.importorskip("transformers.utils.chat_template_utils")
    template = '{%- set today = strftime_now("%d %B %Y") -%}Today Date: {{ today }}'
    rendered, _ = chat_template_utils.render_jinja_template(
        conversations=[[{"role": "user", "content": "hi"}]], chat_template=template,
        strftime_now=fixed_strftime("2026-01-01"),
    )
    assert rendered == ["Today Date: 01 January 2026"]
