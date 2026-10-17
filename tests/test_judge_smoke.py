"""Tests for the judge API smoke test; no network access."""
import json

from scripts.judge_smoke import SmokeError, check_judge, summary_markdown

JUDGE = {"provider": "hip", "base_url": "https://example.test/v1", "secret": "HIP_API_KEY", "model": "Kimi-K2.6"}
KEY = "secret-key-123"


def fake(models=("Kimi-K2.6", "other"), content="OK", finish="stop", fail=None):
    calls = []

    def request(base_url, key, path, payload=None):
        calls.append((path, payload))
        if fail and path == fail[0]:
            raise SmokeError(fail[1])
        if path == "/models":
            return {"data": [{"id": m} for m in models]}
        return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
    return request, calls


def test_working_judge():
    request, calls = fake()
    result = check_judge("primary", JUDGE, KEY, request)
    assert result["ok"] and result["status"] == "ok"
    assert result["models_listed"] == ["Kimi-K2.6", "other"]
    assert calls[1][1]["model"] == "Kimi-K2.6" and calls[1][1]["temperature"] == 0


def test_missing_key_makes_no_request():
    request, calls = fake()
    result = check_judge("primary", JUDGE, "", request)
    assert result["status"] == "missing_or_invalid_HIP_API_KEY" and not calls


def test_model_not_listed():
    request, _ = fake(models=("other",))
    assert check_judge("primary", JUDGE, KEY, request)["status"] == "model_not_listed"


def test_http_error_code_reported():
    request, _ = fake(fail=("/models", "http_401"))
    assert check_judge("primary", JUDGE, KEY, request)["status"] == "http_401"


def test_unexpected_or_truncated_reply():
    assert check_judge("primary", JUDGE, KEY, fake(content="Sure! OK")[0])["status"] == "unexpected_completion"
    assert check_judge("primary", JUDGE, KEY, fake(finish="length")[0])["status"] == "output_truncated"
    assert check_judge("primary", JUDGE, KEY, fake(content="ok.")[0])["ok"]


def test_key_never_appears_in_results_or_summary():
    request, _ = fake(models=("Kimi-K2.6", "echo-" + KEY))
    result = check_judge("primary", JUDGE, KEY, request)
    assert result["status"] == "invalid_model_list"
    assert KEY not in json.dumps(result) + summary_markdown([result])


def test_committed_judge_config():
    judges = json.loads(open("configs/judges.json", encoding="utf-8").read())
    assert {j["secret"] for j in judges.values()} == {"HIP_API_KEY", "BLABLADOR_API_KEY"}
    assert all(j["base_url"].startswith("https://") for j in judges.values())
