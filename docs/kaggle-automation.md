# Kaggle GPU automation (manual GitHub Action)

This workflow is **opt-in** using GitHub Actions → **Kaggle GPU Smoke Test** → **Run workflow**.
It uploads a private GPU kernel, runs the script from the exact public repository commit,
polls the kernel status, downloads its JSON output and saves it as a GitHub Actions artifact.

## Configure credentials

Set these GitHub **Actions repository secrets** in Settings → Secrets and variables → Actions:

- `KAGGLE_USERNAME`: your Kaggle account username (not your email)
- `KAGGLE_KEY`: the Kaggle legacy API key/token accepted by `KAGGLE_USERNAME` / `KAGGLE_KEY` authentication

Never paste tokens in the repository, PR comments, or Actions logs. If your Kaggle account
uses the newer API token scheme instead, adjust authentication before running; the
workflow deliberately uses the legacy environment-variable pair.

## Notes and limits

- Requires an available Kaggle GPU quota, access to the T4 GPU, and permitted Internet access.
- The Kaggle kernel is private, but GitHub Actions artifacts may contain generated model
  outputs and environment metadata. Never include private prompts or credentials.
- Running this workflow may download several GB of model weights.
- GitHub Actions executes no GPU inference. Kaggle is the GPU runtime.
- The Kaggle script clones the *exact commit SHA* of the triggered GitHub workflow,
  so a workflow from an unmerged PR cannot clone its SHA via the public repository unless
  the branch commit has already been pushed to this GitHub repository.
- The existing experiment records exact installed package versions, but full transitive
  dependency locking remains future work.
- Runs are serialized within this workflow. Avoid manually launching a different Kaggle
  run under the same kernel slug during an active workflow because the status and output
  CLI commands address the latest version.
- Check the workflow's artifact and Kaggle logs on failure. The initial integration
  should be validated once before relying on it for automated research.
