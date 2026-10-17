"""Check that the judge API keys work: list models, then one short synthetic chat call per judge."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = "Reply with exactly the word OK and nothing else."
MAX_RESPONSE_BYTES = 1024 * 1024


class SmokeError(Exception):
    """Carries only a short code: provider errors and exceptions can echo credentials, so they are never shown."""


def request_json(base_url: str, key: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url + path, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise SmokeError(f"http_{error.code}") from None
    except Exception as error:
        raise SmokeError(f"transport_{type(error).__name__}") from None
    if len(body) > MAX_RESPONSE_BYTES:
        raise SmokeError("response_too_large")
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError:
        raise SmokeError("invalid_json") from None


def check_judge(role: str, judge: dict, key: str, request=request_json) -> dict:
    result = {"role": role, "provider": judge["provider"], "model": judge["model"], "ok": False,
              "status": "not_started", "models_listed": None, "latency_seconds": None}
    if len(key) < 8 or any(ch.isspace() for ch in key):
        result["status"] = f"missing_or_invalid_{judge['secret']}"
        return result
    try:
        listed = request(judge["base_url"], key, "/models")
        ids = sorted({row["id"] for row in listed.get("data", []) if isinstance(row, dict) and isinstance(row.get("id"), str)})
        if not ids or any(key in model_id for model_id in ids):
            raise SmokeError("invalid_model_list")
        result["models_listed"] = ids
        if judge["model"] not in ids:
            raise SmokeError("model_not_listed")
        start = time.monotonic()
        reply = request(judge["base_url"], key, "/chat/completions", {
            "model": judge["model"], "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 1024, "temperature": 0, "stream": False,
        })
        result["latency_seconds"] = round(time.monotonic() - start, 2)
        choice = (reply.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        if choice.get("finish_reason") == "length":
            raise SmokeError("output_truncated")
        if not isinstance(content, str) or content.strip().rstrip(".").upper() != "OK":
            raise SmokeError("unexpected_completion")
        result.update(ok=True, status="ok")
    except SmokeError as error:
        result["status"] = str(error)
    return result


def summary_markdown(results: list) -> str:
    lines = ["| Role | Provider | Model | Result | Latency (s) | Models listed |", "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        listed = len(r["models_listed"]) if r["models_listed"] is not None else "-"
        lines.append(f"| {r['role']} | {r['provider']} | `{r['model']}` | {'✅' if r['ok'] else '❌'} {r['status']} "
                     f"| {r['latency_seconds'] if r['latency_seconds'] is not None else '-'} | {listed} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/judges.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/judge-smoke.json"))
    args = parser.parse_args()
    judges = json.loads(args.config.read_text(encoding="utf-8"))
    results = [check_judge(role, judge, os.environ.get(judge["secret"], "")) for role, judge in judges.items()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    table = summary_markdown(results)
    print(table)
    for r in results:
        print(f"{r['provider']} models: {', '.join(r['models_listed'] or [])}")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write("## Judge API smoke test\n\n" + table)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
