"""Small, reproducible inference smoke test for SmolLM3 on a GPU runtime.

No model training or fine-tuning is performed by this script.
"""
import argparse
import json
from pathlib import Path


def load_config(path: str) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"model_id", "prompt", "max_new_tokens"}
    if not required.issubset(config):
        raise ValueError(f"Missing config keys: {sorted(required - set(config))}")
    if not isinstance(config["max_new_tokens"], int) or config["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smollm3_smoketest.json")
    parser.add_argument("--output", default="results/smollm3_smoketest.json")
    args = parser.parse_args()
    config = load_config(args.config)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    messages = [{"role": "user", "content": config["prompt"]}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            inputs,
            max_new_tokens=config["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = output[0, inputs.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    result = {
        "model_id": model_id,
        "prompt": config["prompt"],
        "response": response,
        "generated_tokens": int(generated_tokens.numel()),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
