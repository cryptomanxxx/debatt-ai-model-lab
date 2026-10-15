# Debatt-AI Model Lab

An open research workspace for reproducible experiments with open-weight language models, including inference, evaluation, training, and fine-tuning.

## First experiment: SmolLM3 smoke test

The initial reproducible baseline runs **inference only** with [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B). It checks whether the model can be loaded and generate a response in a GPU-capable environment. It does **not** train a model yet.

### Layout

- `configs/` — versioned experiment settings
- `scripts/` — runnable Python experiments
- `tests/` — fast CI checks that do not download models
- `notebooks/` — Kaggle setup instructions
- `results/` — generated local outputs (do not commit large weights or datasets)
- `.github/workflows/` — lightweight CI

### Run

Use a Python 3.11 environment with sufficient GPU memory:

```bash
pip install -r requirements.txt
python scripts/smollm3_smoketest.py
```

The script writes a JSON record to `results/smollm3_smoketest.json`. For Kaggle instructions, see [notebooks/README.md](notebooks/README.md).

### CI

```bash
pip install pytest
python -m pytest -q
```

GitHub Actions checks syntax and configuration without requiring a GPU. A successful CI run **does not** mean Kaggle GPU execution has succeeded.

## Roadmap

1. Reproduce and document the first SmolLM3 inference smoke test.
2. Add explicit experiment metadata, metrics and artifact handling.
3. Introduce a small, controlled fine-tuning baseline (e.g. LoRA/QLoRA) and evaluation against the unmodified model.
4. Automate Kaggle notebook submission and artifact retrieval if supported by account permissions and quotas.

Keep credentials in secret managers, not in Git, and document experimental limitations.
