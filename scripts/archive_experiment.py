"""Create compact, validated, versioned records from a successful Kaggle run."""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def archive(source: Path, destination: Path, sha: str, run_id: str, run_url: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid source Git SHA")
    if not re.fullmatch(r"[0-9]+-[0-9]+", run_id):
        raise ValueError("Invalid experiment run ID")
    raw = json.loads((source / "smollm3_smoketest.json").read_text(encoding="utf-8"))
    provenance = json.loads((source / "experiment_provenance.json").read_text(encoding="utf-8"))
    if provenance.get("git_sha") != sha or provenance.get("run_id") != run_id:
        raise ValueError("Provenance mismatch; refusing to archive stale results")
    for name in ("model_id", "model_revision", "prompt", "rendered_prompt", "response", "gpu_name", "torch_version"):
        if not isinstance(raw.get(name), str) or not raw[name]:
            raise ValueError(f"Missing or invalid {name}")
    if not raw.get("cuda_available") or raw.get("finish_reason") != "eos":
        raise ValueError("GPU smoke test did not complete successfully")
    if not re.fullmatch(r"[0-9a-f]{40}", raw["model_revision"]):
        raise ValueError("Model revision must be pinned")
    if not isinstance(raw.get("generated_tokens"), int) or raw["generated_tokens"] <= 0:
        raise ValueError("Invalid generated token count")
    if not isinstance(raw.get("generation_settings"), dict):
        raise ValueError("Missing generation settings")

    run_number, attempt = run_id.split("-")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    root = destination / "smollm3" / f"{date}-run-{run_number}-attempt-{attempt}"
    record = {
        "schema_version": 1,
        "experiment_id": f"smollm3-smoketest-{run_id}",
        "status": "completed",
        "source": {
            "repository": "cryptomanxxx/debatt-ai-model-lab",
            "git_sha": sha,
            "github_actions_run_id": int(run_number),
            "run_attempt": int(attempt),
            "github_actions_url": run_url,
        },
        "model": {"id": raw["model_id"], "revision": raw["model_revision"]},
        "inference": {
            "prompt": raw["prompt"],
            "rendered_prompt": raw["rendered_prompt"],
            "response": raw["response"],
            "generated_tokens": raw["generated_tokens"],
            "finish_reason": raw["finish_reason"],
            "enable_thinking": raw.get("enable_thinking"),
            "template_date": raw.get("template_date"),
            "generation_settings": raw["generation_settings"],
            "input_sha256": raw.get("input_sha256"),
        },
        "hardware": {"cuda_available": True, "gpu_name": raw["gpu_name"]},
        "software": {
            "python_version": raw.get("python_version"),
            "torch_version": raw["torch_version"],
            "pytorch_cuda_build_version": raw.get("cuda_runtime_version"),
            "transformers_version": raw.get("packages", {}).get("transformers"),
        },
        "interpretation": {
            "technical_smoke_test_passed": True,
            "scientific_quality_benchmark": False,
            "notes": "CUDA build version is from torch.version.cuda, not the host runtime or driver. Full unfiltered outputs remain in the time-limited GitHub Actions artifact.",
        },
    }

    report = (
        f"# SmolLM3 GPU smoke test — {run_id}\n\n"
        f"**Outcome:** successful technical test, not a scientific-quality benchmark.\n\n"
        f"- Source commit: `{sha}`\n"
        f"- [GitHub Actions run]({run_url})\n"
        f"- Model: `{raw['model_id']}` at `{raw['model_revision']}`\n"
        f"- GPU: {raw['gpu_name']}\n"
        f"- Generated tokens: {raw['generated_tokens']} (EOS terminated)\n\n"
        f"## Prompt\n\n{raw['prompt']}\n\n"
        f"## Response\n\n> {raw['response'].replace(chr(10), chr(10) + '> ')}\n\n"
        f"## Reproducibility\n\nSee `result.json` for the exact rendered input, model settings and environment metadata. "
        f"The original full artifact is attached to the linked run, subject to GitHub retention.\n"
    )
    # Search across dates, since a retry can happen after midnight UTC.
    existing = sorted((destination / "smollm3").glob(f"????-??-??-run-{run_number}-attempt-{attempt}"))
    if existing:
        if len(existing) != 1:
            raise ValueError(f"Multiple archived results found for {run_id}")
        previous = existing[0]
        expected_json = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
        if ((previous / "result.json").is_file()
                and (previous / "experiment-report.md").is_file()
                and (previous / "result.json").read_text(encoding="utf-8") == expected_json
                and (previous / "experiment-report.md").read_text(encoding="utf-8") == report):
            print(f"Archive already published and identical: {previous}")
            return previous
        raise ValueError(f"Archived experiment {run_id} differs from the current result")
    root.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "experiment-report.md").write_text(report, encoding="utf-8")
    print(root)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("kaggle-results"))
    parser.add_argument("--destination", type=Path, default=Path("results"))
    parser.add_argument("--sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-url", required=True)
    args = parser.parse_args()
    archive(args.source, args.destination, args.sha, args.run_id, args.run_url)
