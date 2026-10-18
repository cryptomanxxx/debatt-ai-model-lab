"""Pilot evaluation on the frozen test set: AI editor scores, publish decisions and prediction markets, GPU-only."""
import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smollm3_smoketest import fixed_strftime, require_cuda  # noqa: E402

SCORES = ("arg", "ori", "rel", "tro")
DECISIONS = ("publicera", "revidera", "avvisa")
# Debatt-AI publishes only when all four scores are at least 6 (debatt-ai app/api/agent/submit/route.js).
PUBLISH_THRESHOLD = 6

# The criteria are copied from the Debatt-AI editor (debatt-ai app/client.js); the answer is reduced to the
# four scores so a small model spends its tokens on the judgment, not on the rest of the editor JSON.
EDITOR_PROMPT = """Du är chefredaktör för en svensk debattajt. Bedöm artikeln på fyra kriterier (heltal 0-10):
1. Argumentationsklarhet – Är argumenten tydliga och logiskt uppbyggda?
2. Originalitet – Tillför artikeln något nytt till debatten?
3. Samhällsrelevans – Är ämnet viktigt och aktuellt?
4. Trovärdighet – Är faktapåståendena rimliga och välgrundade?

Svara ENDAST med JSON (inga andra tecken):
{{"arg": A, "ori": O, "rel": R, "tro": T}}
där A, O, R och T är dina egna heltal 0-10 för respektive kriterium.

Rubrik: {rubrik}

{artikel}"""

MARKET_PROMPT = """Dagens datum är {today}. Uppskatta sannolikheten att följande händer före deadline.

Fråga: {titel}
Beskrivning: {beskrivning}
Deadline: {deadline}

Svara ENDAST med JSON (inga andra tecken):
{{"sannolikhet": P}}
där P är din egen bedömning i procent, ett heltal 0-100."""


def load_config(path: str) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not config.get("models"):
        raise ValueError("At least one model is required")
    for model in config["models"]:
        revision = model.get("revision")
        if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{model['id']}: revision must be a 40-character commit SHA or null")
    for key in ("editor_sample", "submission_sample"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    return config


def utc_day(created: str) -> str:
    return datetime.fromisoformat(created).astimezone(timezone.utc).date().isoformat()


def find_dataset(root: Path) -> Path:
    """Kaggle mounts datasets under /kaggle/input with a layout that has changed over time; find the manifest."""
    manifests = sorted(root.rglob("manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected exactly one exported dataset under {root}, found {len(manifests)}")
    return manifests[0].parent


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_provenance(data_dir: Path, run_id: str, frozen: dict) -> None:
    """Refuse a stale Kaggle dataset version or an export that is not the frozen test set."""
    provenance = json.loads((data_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("run_id") != run_id:
        raise RuntimeError("Dataset belongs to a different workflow run; refusing stale data")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("test_from"), manifest.get("test_until")) != (frozen["test_from"], frozen["test_until"]):
        raise RuntimeError("Dataset is not the frozen test set")


def editor_items(articles: list, sample: int, seed: int) -> list:
    scored = [a for a in articles if a["split"] == "test" and all(isinstance(a.get(k), int) for k in SCORES)]
    scored.sort(key=lambda a: a["id"])
    return random.Random(seed).sample(scored, min(sample, len(scored)))


def submission_items(submissions: list, sample: int, seed: int) -> list:
    """Every test submission the editor did not publish, topped up with a seeded sample of published ones.

    Published submissions outnumber the rest about six to one, so a plain sample would hold few rejections.
    The metrics that use the decision are per class, so the mix does not bias them."""
    scored = sorted((s for s in submissions if s["split"] == "test" and s.get("beslut") in DECISIONS
                     and all(isinstance(s.get(k), int) for k in SCORES)), key=lambda s: s["id"])
    held = [s for s in scored if s["beslut"] != "publicera"]
    published = [s for s in scored if s["beslut"] == "publicera"]
    rng = random.Random(seed)
    if len(held) >= sample:
        return sorted(rng.sample(held, sample), key=lambda s: s["id"])
    return sorted(held + rng.sample(published, min(sample - len(held), len(published))), key=lambda s: s["id"])


def market_items(markets: list) -> list:
    return sorted((m for m in markets if m["split"] == "test" and m.get("status") == "avgjord"
                   and m.get("utfall") in ("ja", "nej")), key=lambda m: m["id"])


def editor_prompt(article: dict, max_chars: int) -> str:
    return EDITOR_PROMPT.format(rubrik=article.get("rubrik") or "", artikel=(article.get("artikel") or "")[:max_chars])


def market_prompt(market: dict) -> str:
    return MARKET_PROMPT.format(today=utc_day(market["skapad"]), titel=market.get("titel") or "",
                                beskrivning=market.get("beskrivning") or "", deadline=market.get("deadline") or "")


def parse_json_object(text: str):
    """Return the first JSON object in a model reply, or None."""
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except ValueError:
        return None


def parse_scores(text: str):
    obj = parse_json_object(text)
    if not isinstance(obj, dict):
        return None
    values = [obj.get(k) for k in SCORES]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 10 for v in values):
        return None
    return dict(zip(SCORES, values))


def parse_probability(text: str):
    obj = parse_json_object(text)
    value = obj.get("sannolikhet") if isinstance(obj, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 100:
        return None
    return value / 100


def ranks(values: list) -> list:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2  # ties share their average rank
        i = j + 1
    return result


def spearman(x: list, y: list):
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return None if var == 0 else cov / var


def train_mean_scores(articles: list) -> dict:
    train = [a for a in articles if a["split"] == "train" and all(isinstance(a.get(k), int) for k in SCORES)]
    return {k: sum(a[k] for a in train) / len(train) if train else 5 for k in SCORES}


def std(values: list):
    if not values:
        return None
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def editor_metrics(items: list, predictions: list, fallback: dict) -> dict:
    """Score every item: an unparsable reply gets the training-mean scores, so failures cannot improve a model.

    Spread and distinct-answer counts expose a model that repeats one answer: its MAE can look fine
    while it has learned nothing about individual articles."""
    guesses = [pred if pred is not None else fallback for pred in predictions]
    # Diagnostics describe the model's own answers only; the fallback is used for scoring, not counted as an answer.
    parsed = [pred for pred in predictions if pred is not None]
    metrics = {"n": len(items), "parsed": len(parsed),
               "distinct_predictions": len({tuple(p[k] for k in SCORES) for p in parsed}) if parsed else None}
    for k in SCORES:
        truth, guess = [item[k] for item in items], [g[k] for g in guesses]
        metrics[k] = {
            "mae": sum(abs(a - b) for a, b in zip(truth, guess)) / len(items) if items else None,
            "spearman": spearman(truth, guess),
            "truth_std": std(truth),
            "prediction_std": std([p[k] for p in parsed]),
        }
    return metrics


def would_publish(scores: dict) -> bool:
    return all(scores[k] >= PUBLISH_THRESHOLD for k in SCORES)


def decision_metrics(items: list, predictions: list, fallback: dict) -> dict:
    """Publish/hold decision implied by the scores, compared with the editor's beslut, per class.

    Balanced accuracy does not depend on the sample's class mix; always publishing scores 0.5."""
    guesses = [pred if pred is not None else fallback for pred in predictions]
    truth = [item["beslut"] == "publicera" for item in items]
    predicted = [would_publish(g) for g in guesses]

    def recall(label):
        hits = [p == label for t, p in zip(truth, predicted) if t == label]
        return sum(hits) / len(hits) if hits else None
    published, held = recall(True), recall(False)
    return {"published": sum(truth), "not_published": len(truth) - sum(truth),
            "recall_published": published, "recall_not_published": held,
            "balanced_accuracy": (published + held) / 2 if published is not None and held is not None else None}


def submission_metrics(items: list, predictions: list, fallback: dict) -> dict:
    return editor_metrics(items, predictions, fallback) | {"decision": decision_metrics(items, predictions, fallback)}


def brier(probabilities: list, outcomes: list):
    return sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / len(outcomes) if outcomes else None


def market_metrics(items: list, predictions: list, consensus_ids: set) -> dict:
    """Score every market: an unparsable reply counts as an uninformed 50% forecast.

    The second Brier score uses only the markets the agent consensus covers, so the two are comparable."""
    probabilities = [0.5 if p is None else p for p in predictions]
    outcomes = [m["utfall"] == "ja" for m in items]
    subset = [(p, o) for m, p, o in zip(items, probabilities, outcomes) if m["id"] in consensus_ids]
    parsed = [p for p in predictions if p is not None]
    return {"n": len(items), "parsed": len(parsed),
            "distinct_predictions": len(set(parsed)) if parsed else None, "prediction_std": std(parsed),
            "brier": brier(probabilities, outcomes),
            "brier_consensus_markets": brier([p for p, _ in subset], [o for _, o in subset])}


def baselines(markets: list, bets: list, items: list) -> dict:
    """Agent consensus and the training base rate, scored on the same resolved test markets."""
    train = [m for m in markets if m["split"] == "train" and m.get("utfall") in ("ja", "nej")]
    base_rate = sum(m["utfall"] == "ja" for m in train) / len(train) if train else 0.5
    # The model forecasts as of the market's creation day, so only bets placed by that day may inform the consensus.
    created = {m["id"]: utc_day(m["skapad"]) for m in items}
    by_market = {}
    for bet in bets:
        if (bet["split"] == "test" and bet["market_id"] in created and isinstance(bet.get("sannolikhet"), (int, float))
                and utc_day(bet["skapad"]) <= created[bet["market_id"]]):
            by_market.setdefault(bet["market_id"], []).append(bet["sannolikhet"] / 100)
    outcomes = [m["utfall"] == "ja" for m in items]
    covered = [(sum(by_market[m["id"]]) / len(by_market[m["id"]]), m["utfall"] == "ja")
               for m in items if by_market.get(m["id"])]
    return {
        "train_base_rate": base_rate,
        "base_rate_brier": brier([base_rate] * len(items), outcomes),
        "base_rate_brier_consensus_markets": brier([base_rate] * len(covered), [o for _, o in covered]),
        "agent_consensus_brier": brier([p for p, _ in covered], [o for _, o in covered]),
        "agent_consensus_markets": len(covered),
        "agent_consensus_market_ids": sorted(m["id"] for m in items if by_market.get(m["id"])),
    }


def resolve_revision(model_id: str, revision):
    if revision:
        return revision
    from huggingface_hub import HfApi
    return HfApi().model_info(model_id).sha


def run_model(spec: dict, config: dict, editor: list, submissions: list, markets: list, fallback: dict,
              submission_fallback: dict, consensus_ids: set, torch) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = resolve_revision(spec["id"], spec.get("revision"))
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=revision)
    model = AutoModelForCausalLM.from_pretrained(spec["id"], revision=revision, torch_dtype=torch.float16,
                                                 device_map="auto")

    def generate(prompt: str, today: str, max_new_tokens: int) -> str:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False, strftime_now=fixed_strftime(today),
        )
        inputs = tokenizer(rendered, return_tensors="pt").input_ids.to(model.device)
        with torch.inference_mode():
            output = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0, inputs.shape[-1]:], skip_special_tokens=True)

    # Each prompt is dated at its item's creation, so the "today" a template inserts matches what the item knew.
    editor_raw = [generate(editor_prompt(a, config["max_article_chars"]), utc_day(a["skapad"]), 64) for a in editor]
    submission_raw = [generate(editor_prompt(s, config["max_article_chars"]), utc_day(s["skapad"]), 64)
                      for s in submissions]
    market_raw = [generate(market_prompt(m), utc_day(m["skapad"]), 32) for m in markets]
    editor_pred = [parse_scores(text) for text in editor_raw]
    submission_pred = [parse_scores(text) for text in submission_raw]
    market_pred = [parse_probability(text) for text in market_raw]
    del model
    torch.cuda.empty_cache()
    return {
        "id": spec["id"], "revision": revision,
        "editor": editor_metrics(editor, editor_pred, fallback),
        "submissions": submission_metrics(submissions, submission_pred, submission_fallback),
        "markets": market_metrics(markets, market_pred, consensus_ids),
        "predictions": {
            "editor": [{"id": a["id"], "raw": raw, "scores": p} for a, raw, p in zip(editor, editor_raw, editor_pred)],
            # No raw replies: a model may echo the unpublished submission, and the results go to a public artifact.
            "submissions": [{"id": s["id"], "scores": p} for s, p in zip(submissions, submission_pred)],
            "markets": [{"id": m["id"], "raw": raw, "probability": p} for m, raw, p in zip(markets, market_raw, market_pred)],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--frozen", type=Path, default=Path("evals/frozen/v1.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/pilot_results.json"))
    args = parser.parse_args()
    config = load_config(args.config)

    import torch
    require_cuda(torch)

    data_dir = find_dataset(args.data_root)
    check_provenance(data_dir, args.run_id, json.loads(args.frozen.read_text(encoding="utf-8")))
    articles = read_jsonl(data_dir / "artiklar.jsonl")
    markets_all = read_jsonl(data_dir / "markets.jsonl")
    bets = read_jsonl(data_dir / "agent_bets.jsonl")
    submissions_all = read_jsonl(data_dir / "inlamningar.jsonl")
    editor = editor_items(articles, config["editor_sample"], config["seed"])
    submissions = submission_items(submissions_all, config["submission_sample"], config["seed"])
    markets = market_items(markets_all)
    fallback = train_mean_scores(articles)
    submission_fallback = train_mean_scores(submissions_all)
    market_baselines = baselines(markets_all, bets, markets)
    consensus_ids = set(market_baselines["agent_consensus_market_ids"])

    results = {
        "schema_version": 1, "run_id": args.run_id, "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "config": config, "editor_items": len(editor), "submission_items": len(submissions),
        "market_items": len(markets),
        "baselines": market_baselines | {
            "editor_train_mean": editor_metrics(editor, [None] * len(editor), fallback),
            "submissions_train_mean": submission_metrics(submissions, [None] * len(submissions), submission_fallback),
        },
        "models": [run_model(spec, config, editor, submissions, markets, fallback, submission_fallback, consensus_ids,
                             torch) for spec in config["models"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {m["id"]: {"editor": {k: m["editor"][k] for k in ("parsed", *SCORES)},
                         "submissions": {k: m["submissions"][k] for k in ("parsed", *SCORES, "decision")},
                         "markets": m["markets"]}
               for m in results["models"]}
    print(json.dumps({"baselines": results["baselines"], "models": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
