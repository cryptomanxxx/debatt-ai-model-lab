"""GPU-only inference smoke test for a pinned SmolLM3 revision."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path


def load_config(path: str) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"model_id", "model_revision", "prompt", "max_new_tokens", "enable_thinking"}
    if not required.issubset(config):
        raise ValueError(f"Missing config keys: {sorted(required - set(config))}")
    if not re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]):
        raise ValueError("model_revision must be an immutable 40-character commit SHA")
    if not isinstance(config["max_new_tokens"], int) or isinstance(config["max_new_tokens"], bool) or config["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if not isinstance(config["enable_thinking"], bool):
        raise ValueError("enable_thinking must be boolean")
    return config


def require_cuda(torch_module) -> None:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA GPU required; enable the Kaggle GPU accelerator before running.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smollm3_smoketest.json")
    parser.add_argument("--output", default="results/smollm3_smoketest.json")
    args = parser.parse_args()
    config = load_config(args.config)

    import torch
    require_cuda(torch)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = config["model_id"]
    revision = config["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype="auto", device_map="auto"
    )
    messages = [{"role": "user", "content": config["prompt"]}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=config["enable_thinking"],
    )
    inputs = tokenizer(rendered_prompt, return_tensors="pt").input_ids.to(model.device)
    token_ids = inputs[0].tolist()
    with torch.inference_mode():
        output = model.generate(
            inputs, max_new_tokens=config["max_new_tokens"], do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = output[0, inputs.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    packages = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()
                if d.metadata.get("Name")}
    result = {
        "model_id": model_id,
        "model_revision": revision,
        "prompt": config["prompt"],
        "rendered_prompt": rendered_prompt,
        "input_token_ids": token_ids,
        "input_sha256": hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest(),
        "enable_thinking": config["enable_thinking"],
        "response": response,
        "generated_tokens": int(generated_tokens.numel()),
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "packages": dict(sorted(packages.items())),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
