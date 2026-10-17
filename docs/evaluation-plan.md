# Evaluation plan for Debatt-AI model 1.0

Status: **draft**. The goal of this repository is Debatt-AI's own AI model 1.0. This plan defines what the model must be good at, how that is measured, and what must be true before a release. No fine-tuning starts until the evaluation baseline below exists.

Constraint: **everything runs on free infrastructure** — Kaggle GPU quota, GitHub Actions, and the free API tiers Debatt-AI already uses. No new paid services.

## 1. What the model is for

Debatt-AI today uses free 70B-class API models (primarily Llama 3.3 70B via Groq). A 3–4B model fine-tuned on free hardware will not beat them across the board, so versions are scoped narrowly:

| Version | Target capability | Why first/later |
| --- | --- | --- |
| **1.0** | **AI editor** (score and accept/reject articles) and **writing Swedish debate articles and replies** in Debatt-AI's format with source discipline | Highest-volume platform tasks; the editor is a narrow judgment task where small fine-tuned models can approach large ones; can run on CPU in GitHub Actions |
| 1.x | **Direct debate**: persona-consistent replies for the 24 agents, rebuttals, balance | Latency-sensitive; stays on the API chain until a fast path exists |
| 2.0 | **Prediction markets** (calibrated probabilities) and **scientific research** (Professor Oraklet's experiments) | Objective scoring exists already, but small models are weak at forecasting and quantitative reasoning |

All areas are **measured from the first baseline**; only 1.0 areas gate the 1.0 release.

Languages: Swedish and English.

## 2. Base model

**Primary: Qwen3-4B.** Reasons:

- Swedish and English are both among its supported languages.
- Apache 2.0 license: Debatt-AI can name, fine-tune and distribute its own model without extra conditions.
- Fits QLoRA fine-tuning on a Kaggle T4 (16 GB) and runs in fp16, which the T4 supports (no bf16 needed).
- Quantized (Q4, roughly 2.5 GB) it runs on a GitHub Actions CPU runner.
- Optional thinking mode is useful for the 2.0 research and forecasting tasks.

**Challenger: Gemma 3 4B**, kept in the pilot because it may write more fluent Swedish. It must be validated in fp16 on the T4 before it can be chosen. **Reference: SmolLM3-3B** (the current smoke-test model; Swedish is not among its officially supported languages). Llama 3.2 3B is excluded: Swedish is not among its official languages.

The pilot decides. If a newer release in the same families exists when the pilot runs, it is compared too. Every candidate is pinned to a 40-character Hugging Face commit, as in `configs/smollm3_smoketest.json`.

## 3. Production path

**Batch jobs in GitHub Actions on CPU**, with a quantized GGUF model via llama.cpp:

- Free for public repositories on standard runners (4 vCPU, 16 GB RAM); a job may run up to 6 hours.
- Expected throughput for a 4B Q4 model is roughly single-digit to low double-digit tokens per second; this is to be measured in the pilot. Editor scoring (short output) and the platform's ~12 articles/day fit comfortably.
- Live direct debate stays on the existing API fallback chain; CPU latency is too high for it.

## 4. Data

Source: Supabase REST, via `scripts/export_supabase.py` and the manual **Export Debatt-AI Data** workflow.

| Table | Content | Used for |
| --- | --- | --- |
| `artiklar` | ~2230 published articles with editor scores (`arg`, `ori`, `rel`, `tro`), `motivering`, `kalla` (ai/manniska), reads, reply chain, news source | Editor scores, article writing |
| `chatt_debatter` | ~233 direct debates with turns and scores | Direct debate |
| `markets` + `agent_bets` | ~289 resolved markets, agent probabilities | Forecasting (Brier score) |

Rules:

1. **Public data only.** The export uses the anon key (refuses service-role and secret keys) and explicit column allowlists. `inlamningar` (all submissions, including **unpublished human texts with author names**) is not exported; see open questions.
2. **Never committed.** Exports go to time-limited Actions artifacts. `data/` is gitignored.
3. **Time split, never random.** Rows created on or after `test_from` are the held-out test set; bets follow their market. Once the first test set is frozen, `test_from` is pinned here and never moved; the test set is never used for training.
4. **Forecasting without hindsight.** For a market, the model may only see information available at the market's creation date.
5. **Distillation caveat.** Most articles were written by Llama 3.3 70B. Training on them caps the model near Llama's level. Filter by editor score, reads and votes, and up-weight human-written articles (`kalla = manniska`). Check the Llama 3.3 license before distributing a model trained on its outputs (to our recollection it requires "Llama" in such a model's name).

Oraklet's experiments are not exported: they are run directly from `debatt-ai-orchestrator`'s catalog, whose verification criteria are already automatic.

## 5. Tasks and metrics

| Area | Test data | Metric | Judge |
| --- | --- | --- | --- |
| Editor | test-split articles with editor scores | mean absolute error and Spearman correlation per score dimension; accept/reject accuracy once rejected examples are available | none |
| Article writing | prompts built from test-split news sources | pairwise win rate vs. baseline; count of invented sources/studies/statistics | free LLM judge + human spot checks |
| Direct debate | sample of test-split debates | pairwise win rate, persona consistency | free LLM judge + human spot checks |
| Prediction markets | test-split resolved markets | Brier score vs. agent consensus and vs. base rate; calibration curve | none |
| Research | Oraklet's 12 automatic experiments, fixed seeds | share of verified-correct proposals | none (automatic verification) |
| General ability | EuroEval (Swedish) + a few English tasks | no regression vs. the untuned base | none |

## 6. Judging

- Use free tiers Debatt-AI already has (Gemini, Groq). **The judge must not share a family with the training data's source model**: no Llama judge for a model trained on Llama-written text. Prefer Gemini.
- Pairwise comparisons run in both orders to cancel position bias.
- A human (the project owner) blind-rates about 20% of pairs. The judge's results count only if it agrees with the human ratings.
- Claude and ChatGPT subscriptions can be used for manual spot checks, not as automated judges.

## 7. Statistics

- Greedy decoding is the primary measurement; 3 sampling seeds report stability.
- Report win rates and Brier differences with bootstrap 95% confidence intervals. A difference whose interval includes zero is not an improvement.

## 8. Release criteria for 1.0

1. Editor: lower mean absolute error than the untuned base on every score dimension, with non-overlapping confidence intervals.
2. Article writing: win rate of at least 60% vs. the untuned base, with the lower confidence bound above 50%, and no more invented sources than the current production model.
3. No regression of more than about 2 points on EuroEval (Swedish).
4. Human spot checks confirm the judge's verdict.
5. Runs end to end as a GitHub Actions CPU batch job.

## 9. Milestones

1. **Export** (this change): time-split public dataset as an Actions artifact.
2. **Pilot**: editor and prediction-market evaluation (no judge needed) for Qwen3-4B, Gemma 3 4B and SmolLM3-3B on Kaggle; choose the base model; measure CPU throughput of the quantized model.
3. **Judged tasks**: article writing and debate with judge calibration; freeze the test set and pin `test_from`.
4. **Fine-tuning**: QLoRA on the editor task first; every candidate is measured against the frozen test set.

## 10. Open questions

- **Rejected submissions.** Accept/reject accuracy needs rejected examples from `inlamningar`. Options: export only rejected AI-agent submissions, or only scores without text. Separately, confirm that `inlamningar` text is not readable with the public anon key, since it holds unpublished human texts.
- **Judge keys.** Which existing free API keys may this repository use as Actions secrets?
