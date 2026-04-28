import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI
from config import WOLVERINE_SETTINGS

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs"


def _model_log_path(model: str) -> Path:
    safe_name = model.split("/")[-1].replace(" ", "_")
    return LOGS_DIR / f"llm_calls_{safe_name}.jsonl"


class WolverineClient:
    """Lightweight wrapper around the Wolverine OpenAI-compatible endpoint."""

    def __init__(self):
        s = WOLVERINE_SETTINGS
        self._client = OpenAI(base_url=s.base_url, api_key=s.api_key)
        self._model = s.model
        self._temperature = s.temperature
        self._log_path = _model_log_path(s.model)

    def _append_jsonl_log(self, entry: dict) -> None:
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")
        except Exception as log_error:
            print(f"[WARNING] Failed to write LLM call log: {log_error}", file=sys.stderr, flush=True)

    def chat(self, *, system_prompt: str, user_prompt: str, request_tag: str | None = None) -> str:
        """Send a chat request to the model, log usage/latency, and return text."""
        started_at = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        extra_kwargs = {}
        if "qwen" in self._model.lower():
            extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                **extra_kwargs,
            )
        except Exception as api_error:
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            self._append_jsonl_log({
                "timestamp_utc": timestamp,
                "request_tag": request_tag,
                "model": self._model,
                "temperature": self._temperature,
                "latency_ms": latency_ms,
                "status": "error",
                "error": str(api_error),
            })
            raise

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        response = (completion.choices[0].message.content or "").strip()
        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        finish_reason = getattr(completion.choices[0], "finish_reason", None) if completion.choices else None

        self._append_jsonl_log({
            "timestamp_utc": timestamp,
            "request_tag": request_tag,
            "model": self._model,
            "temperature": self._temperature,
            "latency_ms": latency_ms,
            "status": "ok",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "response_chars": len(response),
            "finish_reason": finish_reason,
        })

        print(
            f"[API] Request completed (tag={request_tag}, latency_ms={latency_ms}, "
            f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens})",
            file=sys.stderr,
            flush=True,
        )
        return response
