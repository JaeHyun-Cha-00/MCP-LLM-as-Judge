import sys
from openai import OpenAI
from config import WOLVERINE_SETTINGS


class WolverineClient:
    """Lightweight wrapper around the Wolverine OpenAI-compatible endpoint."""

    def __init__(self):
        s = WOLVERINE_SETTINGS
        self._client = OpenAI(base_url=s.base_url, api_key=s.api_key)
        self._model = s.model
        self._temperature = s.temperature

    def chat(self, *, system_prompt: str, user_prompt: str) -> str:
        """Send a chat request to the model and return the text content."""
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
        )
        response = (completion.choices[0].message.content or "").strip()
        print(f"[API] Request completed - Response length: {len(response)} chars", file=sys.stderr, flush=True)
        return response