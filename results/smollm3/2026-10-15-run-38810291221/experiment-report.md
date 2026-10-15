# Experiment 001 — SmolLM3-3B GPU smoke test

**Outcome:** completed technical smoke test; **not** a model-quality benchmark.

## Provenance

- **GitHub Actions run:** [38810291221](https://github.com/cryptomanxxx/debatt-ai-model-lab/actions/runs/38810291221)
- **Immutable source commit:** `10c5c6a9290e78c51a5e623a0529391f6ca7a038`
- **Unique experiment run:** `38810291221-1`
- **Full original output:** [GitHub Actions artifact 12007783263](https://github.com/cryptomanxxx/debatt-ai-model-lab/actions/runs/38810291221/artifacts/12007783263)
- **Model:** `HuggingFaceTB/SmolLM3-3B` at revision `1c00fc78bd9cf90108046bc433cb34992480f1c1`

## Verified result

- **GPU:** NVIDIA Tesla T4 (`cuda_available: true`)
- **Runtime:** CUDA 12.8, PyTorch 2.11.0+cu128, Transformers 4.53.3
- **Configuration:** deterministic greedy decoding (`do_sample: false`), thinking disabled, 64-token generation cap
- **Generated tokens:** 44; **finish reason:** `eos`
- **Prompt:** Explain in one sentence why reproducible scientific experiments matter.
- **Response:**

> Reproducible scientific experiments matter because they ensure that results can be consistently verified and trusted, allowing the scientific community to build upon and validate findings, ultimately advancing knowledge and reducing the risk of errors or misinterpretations.

The GitHub Actions runner confirmed Kaggle's `KernelWorkerStatus.COMPLETE`, downloaded outputs, and verified their source commit and unique run ID before uploading the artifact. The curated `result.json` is derived from the original `smollm3_smoketest.json` and `experiment_provenance.json`; it intentionally omits the full package inventory and token IDs to keep Git history small.

## Limitations

A successful single-prompt response verifies execution and provenance, not accuracy, robustness, repeatability across independent runs, or improved model performance. No fine-tuning occurred. Kaggle hardware and installed packages can change between runs. Keep the original immutable experiment artifact as the source of record, subject to GitHub Actions artifact retention.
