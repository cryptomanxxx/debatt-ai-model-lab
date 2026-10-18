"""QLoRA fine-tuning of the AI editor on training-split submissions, evaluated on the frozen test set.

Only the training split is used for training; the last days before the test window are held out as validation.
Submissions the editor did not publish are rare in training, so they are repeated (weighted up)."""
import argparse
import json
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_eval import (SCORES, check_provenance, editor_items, editor_metrics, editor_prompt,  # noqa: E402
                        film_review_items, find_dataset, is_film_review, parse_scores, read_jsonl,
                        submission_items, submission_metrics, train_mean_scores, utc_day)
from smollm3_smoketest import fixed_strftime, require_cuda  # noqa: E402

GENERATION_TOKENS = 64


def load_config(path: str) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", config["model"].get("revision") or ""):
        raise ValueError("model.revision must be a 40-character commit SHA")
    for key in ("train_published_sample", "held_weight", "validation_days", "epochs", "grad_accum", "max_seq_tokens",
                "editor_sample", "submission_sample", "validation_sample", "bootstrap_samples"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    return config


def scored_submission(row: dict) -> bool:
    return row.get("beslut") in ("publicera", "revidera", "avvisa") and all(isinstance(row.get(k), int) for k in SCORES)


def split_training(submissions: list, test_from: str, validation_days: int) -> tuple:
    """Training-split submissions, divided by date: the last validation_days before test_from are validation."""
    cutoff = (date.fromisoformat(test_from) - timedelta(days=validation_days)).isoformat()
    usable = sorted((s for s in submissions if s["split"] == "train" and not is_film_review(s) and scored_submission(s)),
                    key=lambda s: s["id"])
    return ([s for s in usable if utc_day(s["skapad"]) < cutoff], [s for s in usable if utc_day(s["skapad"]) >= cutoff])


def training_examples(rows: list, published_sample: int, held_weight: int, seed: int) -> list:
    """A seeded sample of published submissions plus every held one repeated held_weight times, shuffled."""
    rng = random.Random(seed)
    published = [r for r in rows if r["beslut"] == "publicera"]
    held = [r for r in rows if r["beslut"] != "publicera"]
    examples = rng.sample(published, min(published_sample, len(published))) + held * held_weight
    rng.shuffle(examples)
    return examples


def target_text(row: dict) -> str:
    return json.dumps({k: row[k] for k in SCORES})


def encode_example(prompt_ids: list, target_ids: list, max_tokens: int):
    """Loss on the answer only. Too long returns None: cutting the prompt would cut the chat template's answer prefix."""
    if len(prompt_ids) + len(target_ids) > max_tokens:
        return None
    return {"input_ids": prompt_ids + target_ids, "labels": [-100] * len(prompt_ids) + target_ids}


def group_size(index: int, total: int, accum: int) -> int:
    """Size of the gradient-accumulation group that 1-based example index belongs to."""
    start = (index - 1) // accum * accum
    return min(accum, total - start)


def paired_bootstrap(truth: list, base: list, tuned: list, samples: int, seed: int) -> dict:
    """MAE(tuned) - MAE(base) with a 95% percentile interval; negative means the tuned model is better."""
    rng = random.Random(seed)
    n = len(truth)
    gain = [abs(t - b) - abs(t - u) for t, b, u in zip(truth, base, tuned)]

    def diff(idx):
        return -sum(gain[i] for i in idx) / len(idx)
    stats = sorted(diff([rng.randrange(n) for _ in range(n)]) for _ in range(samples))
    return {"mae_diff": diff(range(n)), "ci95": [stats[int(0.025 * samples)], stats[int(0.975 * samples) - 1]]}


def compare(items: list, base_pred: list, tuned_pred: list, fallback: dict, samples: int, seed: int) -> dict:
    fill = lambda preds: [p if p is not None else fallback for p in preds]  # noqa: E731
    base, tuned = fill(base_pred), fill(tuned_pred)
    return {k: paired_bootstrap([i[k] for i in items], [b[k] for b in base], [t[k] for t in tuned], samples, seed)
            for k in SCORES}


def run(config: dict, data_dir: Path, test_from: str, torch, output: Path) -> dict:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    seed, chars = config["seed"], config["max_article_chars"]
    random.seed(seed)
    torch.manual_seed(seed)
    articles = read_jsonl(data_dir / "artiklar.jsonl")
    submissions_all = read_jsonl(data_dir / "inlamningar.jsonl")
    train_rows, validation_rows = split_training(submissions_all, test_from, config["validation_days"])
    examples = training_examples(train_rows, config["train_published_sample"], config["held_weight"], seed)
    # Validation uses the submission sampler on the held-out days; it never touches the test split.
    validation = submission_items([dict(r, split="test") for r in validation_rows], config["validation_sample"], seed)
    editor = editor_items(articles, config["editor_sample"], seed)
    films = film_review_items(articles)
    submissions = submission_items(submissions_all, config["submission_sample"], seed)
    fallback = train_mean_scores(articles)
    submission_fallback = train_mean_scores([dict(r, split="train") for r in train_rows])

    spec = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"], revision=spec["revision"], device_map="auto", torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))

    def prompt_ids(row: dict) -> list:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": editor_prompt(row, chars)}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False, strftime_now=fixed_strftime(utc_day(row["skapad"])))
        return tokenizer(rendered, add_special_tokens=False).input_ids

    def predict(rows: list) -> list:
        model.eval()
        preds = []
        for row in rows:
            ids = torch.tensor([prompt_ids(row)], device=model.device)
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=GENERATION_TOKENS, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            preds.append(parse_scores(tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)))
        return preds

    def evaluate() -> dict:
        preds = {"validation": predict(validation), "editor": predict(editor), "film_reviews": predict(films),
                 "submissions": predict(submissions)}
        return {
            "predictions": preds,
            "metrics": {
                "validation": submission_metrics(validation, preds["validation"], submission_fallback),
                "editor": editor_metrics(editor, preds["editor"], fallback),
                "film_reviews": editor_metrics(films, preds["film_reviews"], fallback),
                "submissions": submission_metrics(submissions, preds["submissions"], submission_fallback),
            },
        }

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"], task_type="CAUSAL_LM"))
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()  # fp32 adapter weights, so the fp16 grad scaler can unscale them

    results = {"counts": {"train_rows": len(train_rows), "train_examples": len(examples),
                          "train_held_rows": sum(r["beslut"] != "publicera" for r in train_rows),
                          "validation_items": len(validation), "editor_items": len(editor),
                          "film_review_items": len(films), "submission_items": len(submissions)}}
    if config["evaluate_base"]:
        with model.disable_adapter():
            base = evaluate()

    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    encoded = [encode_example(prompt_ids(r), tokenizer(target_text(r), add_special_tokens=False).input_ids + [end],
                              config["max_seq_tokens"]) for r in examples]
    results["counts"]["train_examples_too_long"] = sum(e is None for e in encoded)
    encoded = [e for e in encoded if e is not None]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"])
    # The loop flushes a final partial group, so each epoch takes ceil(len / grad_accum) optimizer steps.
    total = config["epochs"] * -(-len(encoded) // config["grad_accum"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / max(1, total)))
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    losses, step = [], 0
    for _ in range(config["epochs"]):
        for i, example in enumerate(encoded, 1):
            # Average over the actual group, so a final partial group weighs as much as a full one.
            group = group_size(i, len(encoded), config["grad_accum"])
            ids = torch.tensor([example["input_ids"]], device=model.device)
            labels = torch.tensor([example["labels"]], device=model.device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, labels=labels).loss / group
            scaler.scale(loss).backward()
            losses.append(loss.item() * group)
            if i % config["grad_accum"] == 0 or i == len(encoded):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                print(f"step {step}/{total} loss {sum(losses[-group:]) / group:.4f}", flush=True)
    # The adapter stays in the private Kaggle output; the public workflow artifact uploads JSON and logs only.
    model.save_pretrained(output.parent / "editor-adapter")
    tuned = evaluate()

    results["training"] = {"steps": step, "loss_first_50": sum(losses[:50]) / max(1, len(losses[:50])),
                           "loss_last_50": sum(losses[-50:]) / max(1, len(losses[-50:]))}
    results["tuned"] = tuned["metrics"]
    results["tuned_predictions"] = {k: [{"id": r["id"], "scores": p} for r, p in zip(rows, tuned["predictions"][k])]
                                    for k, rows in (("validation", validation), ("editor", editor),
                                                    ("film_reviews", films), ("submissions", submissions))}
    if config["evaluate_base"]:
        results["base"] = base["metrics"]
        samples = config["bootstrap_samples"]
        results["tuned_minus_base"] = {
            "validation": compare(validation, base["predictions"]["validation"], tuned["predictions"]["validation"],
                                  submission_fallback, samples, seed),
            "editor": compare(editor, base["predictions"]["editor"], tuned["predictions"]["editor"], fallback,
                              samples, seed),
            "film_reviews": compare(films, base["predictions"]["film_reviews"], tuned["predictions"]["film_reviews"],
                                    fallback, samples, seed),
            "submissions": compare(submissions, base["predictions"]["submissions"],
                                   tuned["predictions"]["submissions"], submission_fallback, samples, seed),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_editor.json")
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--frozen", type=Path, default=Path("evals/frozen/v1.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/train_results.json"))
    args = parser.parse_args()
    config = load_config(args.config)

    import torch
    require_cuda(torch)

    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    data_dir = find_dataset(args.data_root)
    check_provenance(data_dir, args.run_id, frozen)
    # run_kaggle_kernel.sh accepts a result only if it records that the GPU was used.
    results = {"schema_version": 1, "run_id": args.run_id, "cuda_available": True, "config": config,
               "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__}
    results |= run(config, data_dir, frozen["test_from"], torch, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("counts", "training", "tuned_minus_base") if k in results}, indent=2))


if __name__ == "__main__":
    main()
