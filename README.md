# Debatt-AI Model Lab

An open research workspace for experiments with open-weight language models: inference, evaluation, training, and fine-tuning.

## First experiment: SmolLM3 smoke test

The first baseline runs **GPU-only inference**, not training, with a pinned revision of [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B).

### Layout

- `configs/` — versioned experiment settings and immutable model revision
- `scripts/` — runnable experiments
- `tests/` — CPU-only unit tests
- `notebooks/` — Kaggle setup instructions
- `results/` — local experiment JSON output
- `.github/workflows/` — lightweight CI

### Run

Use an enabled CUDA GPU and adequate memory:

```bash
pip install -r requirements.txt
python scripts/smollm3_smoketest.py
```

#### Reference environment

The **reference baseline is the Kaggle T4 GPU image**, not a local install. On Kaggle only `requirements-kaggle.txt` (Transformers, Accelerate) is installed, on top of Kaggle's preinstalled GPU PyTorch and Python. Experiment 001 therefore ran with Python 3.13 and PyTorch 2.11.0+cu128; the exact versions of each run are recorded in its result JSON. `requirements.txt` adds `torch` and `pytest` for local runs and CI (Python 3.11); local results are not the reference baseline.

The script writes to `results/smollm3_smoketest.json` and fails immediately when CUDA is unavailable. It records the pinned model commit, full rendered prompt, input token IDs, exact package versions, platform/GPU information and deterministic generation settings. Extended thinking is disabled for this short smoke response.

SmolLM3's chat template inserts "Today Date" into the system prompt. The script pins that date to `template_date` in the config, so the rendered prompt and `input_sha256` are identical on every day. Experiment 001 predates this and contains its real run date.

**Environment limitations:** `requirements-kaggle.txt` and `requirements.txt` pin direct dependencies, but is not a complete transitive lock file. Package metadata in the JSON supports auditing but does not guarantee byte-identical installation across Python platforms or GPU drivers. To obtain stronger reproducibility, capture a full resolved environment/lock from the validated Kaggle runtime before declaring the baseline frozen.

See [notebooks/README.md](notebooks/README.md) for Kaggle setup. A successful CI run does **not** prove a GPU smoke test succeeded.

## Roadmap

1. Validate the pinned SmolLM3 inference baseline on Kaggle GPU.
2. Freeze the verified full environment, report results and evaluation metrics.
3. Add controlled LoRA/QLoRA fine-tuning and compare with the unmodified model.
4. Explore automated Kaggle submission and artifact retrieval within access quotas.

Keep secrets, large model weights and private data out of Git.
