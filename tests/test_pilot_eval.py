"""CPU-only tests for the pilot evaluation logic; model inference runs on Kaggle."""
import json
import re

import pytest

from scripts.pilot_eval import (baselines, brier, check_provenance, editor_items, editor_metrics, editor_prompt,
                                find_dataset, load_config, market_items, market_metrics, market_prompt,
                                parse_probability, parse_scores, spearman, train_mean_scores)

FROZEN = {"test_from": "2026-09-01", "test_until": "2026-10-16"}


def article(i, split="test", **scores):
    return {"id": i, "split": split, "rubrik": f"R{i}", "artikel": "x" * 10,
            **({k: 5 for k in ("arg", "ori", "rel", "tro")} | scores)}


def test_committed_pilot_config():
    config = load_config("configs/pilot.json")
    assert [m["id"] for m in config["models"]] == ["Qwen/Qwen3-4B", "HuggingFaceTB/SmolLM3-3B"]


def test_unpinned_revision_must_be_sha_or_null(tmp_path):
    config = json.loads(open("configs/pilot.json", encoding="utf-8").read())
    config["models"][0]["revision"] = "main"
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="commit SHA or null"):
        load_config(str(path))


def test_editor_sample_is_deterministic_test_only():
    articles = [article(i) for i in range(50)] + [article(99, split="train"), article(100, arg=None)]
    first, second = editor_items(articles, 10, 7), editor_items(list(reversed(articles)), 10, 7)
    assert [a["id"] for a in first] == [a["id"] for a in second]
    assert all(a["split"] == "test" and a["id"] < 50 for a in first)


def test_only_resolved_test_markets_are_scored():
    markets = [{"id": 1, "split": "test", "status": "avgjord", "utfall": "ja"},
               {"id": 2, "split": "test", "status": "öppen", "utfall": None},
               {"id": 3, "split": "train", "status": "avgjord", "utfall": "nej"}]
    assert [m["id"] for m in market_items(markets)] == [1]


def test_prompts_truncate_and_date_markets():
    assert editor_prompt(article(1) | {"artikel": "y" * 100}, 20).endswith("y" * 20)
    market = {"skapad": "2026-09-14T23:30:00-02:00", "titel": "T", "beskrivning": None, "deadline": "2026-12-31"}
    assert "Dagens datum är 2026-09-15" in market_prompt(market)


def test_parsers_accept_json_and_reject_out_of_range():
    assert parse_scores('Svar: {"arg": 8, "ori": 7, "rel": 9, "tro": 6}') == {"arg": 8, "ori": 7, "rel": 9, "tro": 6}
    assert parse_scores('{"arg": 11, "ori": 7, "rel": 9, "tro": 6}') is None
    assert parse_scores("ingen json") is None
    assert parse_probability('{"sannolikhet": 72}') == 0.72
    assert parse_probability('{"sannolikhet": 140}') is None
    assert parse_probability('{"sannolikhet": true}') is None


def test_metrics():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert spearman([5, 5, 5], [1, 2, 3]) is None
    assert brier([1.0, 0.0], [True, False]) == 0
    items = [article(1, arg=8), article(2, arg=4), article(3, arg=6)]
    preds = [{"arg": 7, "ori": 5, "rel": 5, "tro": 5}, None, {"arg": 6, "ori": 5, "rel": 5, "tro": 5}]
    fallback = {"arg": 5, "ori": 5, "rel": 5, "tro": 5}
    metrics = editor_metrics(items, preds, fallback)
    # The unparsable reply stays in the denominator with the fallback score: (1 + |4-5| + 0) / 3.
    assert metrics["parsed"] == 2 and metrics["n"] == 3 and metrics["arg"]["mae"] == pytest.approx(2 / 3)
    markets = [{"id": 1, "utfall": "ja"}, {"id": 2, "utfall": "nej"}]
    result = market_metrics(markets, [1.0, None], {2})
    assert (result["parsed"], result["brier"], result["brier_consensus_markets"]) == (1, 0.125, 0.25)
    assert result["distinct_predictions"] == 2


def test_train_mean_fallback():
    articles = [article(1, split="train", arg=4), article(2, split="train", arg=8), article(3, arg=0)]
    assert train_mean_scores(articles)["arg"] == 6


def test_baselines_use_train_rate_and_test_bets():
    markets = [{"id": 1, "split": "train", "utfall": "ja"}, {"id": 2, "split": "train", "utfall": "nej"},
               {"id": 3, "split": "test", "status": "avgjord", "utfall": "ja", "skapad": "2026-09-10T08:00:00+00:00"}]
    day = "2026-09-10T20:00:00+00:00"
    bets = [{"market_id": 3, "split": "test", "sannolikhet": 80, "skapad": day},
            {"market_id": 3, "split": "test", "sannolikhet": 60, "skapad": day},
            # Placed days later, with information the model never had: excluded from the consensus.
            {"market_id": 3, "split": "test", "sannolikhet": 99, "skapad": "2026-09-20T08:00:00+00:00"},
            {"market_id": 1, "split": "train", "sannolikhet": 10, "skapad": day}]
    result = baselines(markets, bets, market_items(markets))
    assert result["train_base_rate"] == 0.5 and result["base_rate_brier"] == 0.25
    assert result["agent_consensus_brier"] == pytest.approx(0.09) and result["agent_consensus_markets"] == 1
    assert result["agent_consensus_market_ids"] == [3] and result["base_rate_brier_consensus_markets"] == 0.25


def test_dataset_discovery_and_provenance(tmp_path):
    data = tmp_path / "input" / "datasets" / "user" / "debatt-ai-eval-data"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text(json.dumps(FROZEN))
    (data / "provenance.json").write_text(json.dumps({"run_id": "1-1"}))
    assert find_dataset(tmp_path / "input") == data
    check_provenance(data, "1-1", FROZEN)
    with pytest.raises(RuntimeError, match="different workflow run"):
        check_provenance(data, "2-1", FROZEN)
    with pytest.raises(RuntimeError, match="not the frozen test set"):
        check_provenance(data, "1-1", {"test_from": "2026-01-01", "test_until": "2026-10-16"})


def test_constant_answers_are_exposed():
    items = [article(i, arg=v) for i, v in enumerate([6, 7, 8, 9])]
    copied = {"arg": 8, "ori": 7, "rel": 9, "tro": 8}
    metrics = editor_metrics(items, [copied] * 4, copied)
    assert metrics["distinct_predictions"] == 1
    assert metrics["arg"]["prediction_std"] == 0 and metrics["arg"]["truth_std"] > 0
    assert metrics["arg"]["spearman"] is None
    markets = [{"id": i, "utfall": "ja"} for i in range(3)]
    assert market_metrics(markets, [0.55] * 3, set())["prediction_std"] == 0


def test_prompts_contain_no_copyable_example_values():
    prompt = editor_prompt(article(1), 100)
    assert '"arg": A' in prompt and not re.search(r'"(arg|ori|rel|tro)":\s*\d', prompt)
    market = {"skapad": "2026-09-14T10:00:00+00:00", "titel": "T", "beskrivning": "B", "deadline": "2026-12-31"}
    assert not re.search(r'"sannolikhet":\s*\d', market_prompt(market))
