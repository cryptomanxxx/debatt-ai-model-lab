"""CPU-only tests for the editor training data and comparison logic; training runs on Kaggle."""
import json

import pytest

from scripts.train_editor import (compare, encode_example, group_size, load_config, paired_bootstrap, split_training, target_text,
                                  training_examples)

SCORES = ("arg", "ori", "rel", "tro")


def submission(i, beslut, day, split="train", **extra):
    return {"id": i, "beslut": beslut, "skapad": f"{day}T12:00:00+00:00", "split": split,
            **dict.fromkeys(SCORES, 7), **extra}


def test_committed_training_config():
    config = load_config("configs/train_editor.json")
    pilot = json.loads(open("configs/pilot.json", encoding="utf-8").read())
    # Same model, revision and test samples as the evaluation pilot, so results are comparable.
    assert config["model"] == pilot["models"][0]
    assert (config["seed"], config["editor_sample"], config["submission_sample"]) == (
        pilot["seed"], pilot["editor_sample"], pilot["submission_sample"])


def test_training_never_uses_test_rows_and_holds_out_validation_days():
    rows = [submission(1, "publicera", "2026-07-01"), submission(2, "avvisa", "2026-08-20"),
            submission(3, "publicera", "2026-09-02", split="test"), submission(4, "revidera", "2026-08-05"),
            submission(5, "publicera", "2026-07-02", filmrecension=True), submission(6, None, "2026-07-03")]
    train, validation = split_training(rows, "2026-09-01", 21)
    assert [r["id"] for r in train] == [1, 4]
    assert [r["id"] for r in validation] == [2]


def test_held_submissions_are_weighted_up():
    rows = [submission(i, "publicera", "2026-07-01") for i in range(20)] + [submission(99, "avvisa", "2026-07-01")]
    examples = training_examples(rows, published_sample=5, held_weight=4, seed=1)
    assert len(examples) == 9 and sum(r["id"] == 99 for r in examples) == 4
    assert [r["id"] for r in examples] == [r["id"] for r in training_examples(rows, 5, 4, 1)]


def test_loss_covers_the_answer_only():
    assert target_text(submission(1, "publicera", "2026-07-01")) == '{"arg": 7, "ori": 7, "rel": 7, "tro": 7}'
    example = encode_example([1, 2, 3], [8, 9], 10)
    assert example == {"input_ids": [1, 2, 3, 8, 9], "labels": [-100, -100, -100, 8, 9]}
    # Too long: dropped rather than cutting the chat template's answer prefix.
    assert encode_example(list(range(9)), [8, 9], 10) is None


def test_paired_bootstrap_sign_and_interval():
    truth, base, tuned = [5] * 40, [8] * 40, [6] * 40
    result = paired_bootstrap(truth, base, tuned, 200, 3)
    assert result["mae_diff"] == pytest.approx(-2.0)
    assert result["ci95"] == pytest.approx([-2.0, -2.0])
    items = [dict.fromkeys(SCORES, 5)] * 4
    diff = compare(items, [None] * 4, [dict.fromkeys(SCORES, 5)] * 4, dict.fromkeys(SCORES, 7), 100, 1)
    assert diff["arg"]["mae_diff"] == pytest.approx(-2.0)


def test_final_partial_group_is_averaged_over_its_own_size():
    assert [group_size(i, 10, 4) for i in range(1, 11)] == [4] * 8 + [2, 2]
    assert [group_size(i, 3, 8) for i in (1, 2, 3)] == [3, 3, 3]
