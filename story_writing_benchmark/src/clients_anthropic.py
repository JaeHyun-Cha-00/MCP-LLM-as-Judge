import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from anthropic import Anthropic, RateLimitError

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs"

MODEL = "claude-sonnet-4-6"


def _model_log_path(model: str) -> Path:
    safe_name = model.replace("/", "_").replace(" ", "_")
    return LOGS_DIR / f"llm_calls_{safe_name}.jsonl"


class AnthropicClient:
    """Wrapper around the Anthropic API for Claude models."""

    def __init__(self, model: str = MODEL, temperature: float = 0.0, api_key: str | None = None):
        self._client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._model = model
        self._temperature = temperature
        self._log_path = _model_log_path(model)

    def _append_jsonl_log(self, entry: dict) -> None:
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")
        except Exception as log_error:
            print(f"[WARNING] Failed to write log: {log_error}", file=sys.stderr, flush=True)

    def chat(self, *, system_prompt: str, user_prompt: str, request_tag: str | None = None) -> str:
        started_at = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    temperature=self._temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                break
            except RateLimitError as e:
                wait = 2 ** attempt * 10
                print(f"[WARN] Rate limit hit, retrying in {wait}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr, flush=True)
                time.sleep(wait)
                if attempt == max_retries - 1:
                    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
                    self._append_jsonl_log({"timestamp_utc": timestamp, "request_tag": request_tag, "model": self._model, "latency_ms": latency_ms, "status": "error", "error": str(e)})
                    raise
            except Exception as api_error:
                latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
                self._append_jsonl_log({"timestamp_utc": timestamp, "request_tag": request_tag, "model": self._model, "latency_ms": latency_ms, "status": "error", "error": str(api_error)})
                raise

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        text = response.content[0].text.strip() if response.content else ""
        usage = response.usage

        self._append_jsonl_log({
            "timestamp_utc": timestamp,
            "request_tag": request_tag,
            "model": self._model,
            "temperature": self._temperature,
            "latency_ms": latency_ms,
            "status": "ok",
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
            "response_chars": len(text),
            "finish_reason": response.stop_reason,
        })

        print(
            f"[API] Request completed (tag={request_tag}, latency_ms={latency_ms}, "
            f"input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens})",
            file=sys.stderr,
            flush=True,
        )
        return text
