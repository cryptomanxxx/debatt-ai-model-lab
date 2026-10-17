# Kaggle notebook workflow

The Python script in `scripts/smollm3_smoketest.py` is the source of truth. Kaggle is the execution environment, not the code editor. For automated runs, prefer the GitHub Action described in [docs/kaggle-automation.md](../docs/kaggle-automation.md).

1. Create a Kaggle notebook with GPU acceleration and Internet access where available.
2. Clone this repository inside the notebook:
   `!git clone https://github.com/cryptomanxxx/debatt-ai-model-lab.git`
3. Enter the cloned directory and install dependencies. Use `requirements-kaggle.txt`, not `requirements.txt`, so Kaggle's preinstalled GPU PyTorch is not replaced:
   `%cd debatt-ai-model-lab`
   `!pip install -r requirements-kaggle.txt`
4. Run the smoke test:
   `!python scripts/smollm3_smoketest.py`
5. Inspect `results/smollm3_smoketest.json` and export results as a Kaggle notebook output.

To test an unmerged change, `git checkout` its branch or commit after cloning; the branch must already be pushed to GitHub.

GPU quota, network access, and downloads depend on Kaggle settings. The first run downloads model weights and needs sufficient GPU memory. No API keys should be committed to this repository.
