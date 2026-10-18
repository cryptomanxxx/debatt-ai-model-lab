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

## Data export

The manual **Export Debatt-AI Data** workflow reads public Debatt-AI data (articles, direct debates, prediction markets and bets) via Supabase REST, splits it by date into train/test, and uploads it with a manifest as an Actions artifact. Exports are never committed.

Once `evals/frozen/v1.json` exists, a run with empty dates reuses its test window and fails if any frozen test row was added, removed or edited. To freeze a new window, run with `test_from`, `test_until` (a past date) and `freeze`; the ids and fingerprints are printed on the `FROZEN_TEST_SET` log line and saved in the artifact.

Set the Actions secret `SUPABASE_ANON_KEY` to Debatt-AI's public anon key. The script refuses service-role and secret keys for the public tables, so they only show what Row Level Security exposes publicly.

Also set `SUPABASE_SERVICE_ROLE_KEY` (Supabase → Settings → API Keys, the site project; prefer a dedicated `sb_secret_` key that can be revoked on its own). It exports the AI agents' submissions from `inlamningar`, including rejected ones. They are part of the frozen test set v1, so exports with empty dates and the pilot fail early without it; only an export with an explicit new window works without it. The key is used for that table only, with GET requests only: the script never writes to the database. Only rows with `kalla = 'ai'` are requested, and the export fails if any other row comes back, so human submissions never leave the database. The submission texts go to the private Kaggle dataset only, not to the Actions artifact, because this repository is public.

Optionally set the Actions variable `SUPABASE_URL`.

## Pilot evaluation

The manual **AI Model Evaluation (Kaggle GPU)** workflow:
1. exports and verifies the frozen test set;
2. uploads it as the private Kaggle dataset `debatt-ai-eval-data`, tagged with the workflow run;
3. runs `scripts/pilot_eval.py` on a Kaggle T4 for every model in `configs/pilot.json`.

Three tasks are scored, plus film reviews on their own:
- **Editor:** a seeded sample of test articles is scored on `arg`/`ori`/`rel`/`tro`. The prompt uses the Debatt-AI editor's criteria, and predictions are compared with the stored editor scores (MAE and Spearman). The training-mean scores are reported as a baseline.
- **Film reviews:** published test film reviews (`filmrecension`) are scored like the editor task but reported separately, and excluded from the editor and submission samples. The training split holds none, so mixing them in would test an unseen genre.
- **Submissions:** every test submission the editor did not publish (`revidera`, `avvisa`), topped up with a seeded sample of published ones, is scored the same way. The scores also imply a decision (publish if all four are at least 6, as on Debatt-AI), reported as recall per class and balanced accuracy against the editor's `beslut`. The training mean of the submissions is the baseline.
- **Markets:** every resolved test market gets a probability, dated at market creation, and is scored with Brier. The training base rate and the agents' consensus are scored as baselines; the consensus counts only bets placed by the market's creation day.

Every prompt is dated at its item's creation, including the date a chat template inserts. An unparsable reply is scored rather than dropped: the training mean for the editor, 50% for markets.

The job summary shows the tables; the artifact holds every raw reply, except for submissions, where only the parsed scores are kept so that an echoed unpublished text cannot reach the public artifact. A model with `"revision": null` is resolved to its current Hugging Face commit at run time, and that commit is recorded in the results; pin it in the config afterwards.

Needs the Kaggle secrets above, `SUPABASE_ANON_KEY` and `SUPABASE_SERVICE_ROLE_KEY`; the workflow checks all of them before it starts.

## Editor training

The manual **AI Model Training (Kaggle GPU)** workflow fine-tunes Qwen3-4B as the AI editor with QLoRA (4-bit base, LoRA adapter) on a Kaggle T4. It runs `scripts/train_editor.py` with `configs/train_editor.json`.

- **Training data:** training-split AI-agent submissions only, never test rows. The last `validation_days` before the test window are held out as validation for tuning.
- **Weighting:** submissions the editor did not publish are rare in training (48 of about 1,590), so each is repeated `held_weight` times, next to a seeded sample of `train_published_sample` published ones.
- **Loss:** on the four-score JSON answer only, with the same prompt and chat template as the evaluation.
- **Evaluation:** the untuned 4-bit base (adapter disabled) and the tuned model are scored on validation and on the frozen test tasks (editor articles, film reviews, submissions). The MAE difference per score gets a paired 95% bootstrap interval.
- **Privacy:** the adapter stays in the private Kaggle output. The public workflow artifact holds JSON and logs only; predictions are parsed scores, never raw replies.

### Progress while a run is going

While a Kaggle run is running, the workflow log shows its latest progress line after each status check, for example:

```
PROGRESS run=39119404135-1 phase 6/9 training steps 40/111 | phase left ~38m10s | total left ~1h24m | elapsed 1h02m
```

The phase estimate uses the phase's own pace. The total also prices the remaining phases at the last measured pace of their kind (evaluation or training), and reads "not known yet" until every remaining kind has been measured once. Time spent in Kaggle's queue is not counted.

## Judge API smoke test

**Judge API Smoke Test** checks `HIP_API_KEY` and `BLABLADOR_API_KEY` (Actions secrets). For the judges in `configs/judges.json`, it lists each provider's models and sends one synthetic prompt per judge. Errors are reported as short codes only, so keys cannot leak into logs.

## Roadmap

See [docs/evaluation-plan.md](docs/evaluation-plan.md) for the goal of Debatt-AI model 1.0, the base model choice, metrics and release criteria.


1. Validate the pinned SmolLM3 inference baseline on Kaggle GPU.
2. Freeze the verified full environment, report results and evaluation metrics.
3. Add controlled LoRA/QLoRA fine-tuning and compare with the unmodified model.
4. Explore automated Kaggle submission and artifact retrieval within access quotas.

Keep secrets, large model weights and private data out of Git.
