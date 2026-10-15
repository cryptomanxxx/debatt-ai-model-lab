# Research result registry

Small, reviewed, non-sensitive experiment summaries are versioned here alongside the code. Organize each experiment under `results/<model>/<date>-run-<github-actions-run-id>/` with a compact `result.json` and a human-readable `experiment-report.md`.

## Rules

1. Trace every result to the exact source commit, model revision, GitHub Actions run/attempt and original artifact.
2. Record parameters, hardware/software versions, observable outputs, and limitations. Never label a smoke test a scientific benchmark.
3. Do not commit large files, raw logs, model weights, private prompts, API tokens, or full package inventories.
4. Treat GitHub Actions artifacts as original execution outputs and the Git-tracked records as reviewed summaries. Artifacts expire according to repository retention policy.
5. Store large training outputs, model weights, and versioned public datasets in dedicated storage such as Hugging Face repositories when publication and licensing permit.
6. Do not let GPU workflows commit directly to protected `main`. Future automated publication should create a new PR for review and CI validation.

## Experiments

| Experiment | Model | Run | Result |
| --- | --- | --- | --- |
| [001 – GPU smoke test](smollm3/2026-10-15-run-38810291221/experiment-report.md) | SmolLM3-3B | [38810291221](https://github.com/cryptomanxxx/debatt-ai-model-lab/actions/runs/38810291221) | Completed; technical verification only |
