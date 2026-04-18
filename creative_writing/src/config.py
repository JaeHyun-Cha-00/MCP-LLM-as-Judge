from dataclasses import dataclass

@dataclass(frozen=True)
class WolverineSettings:
    """Runtime configuration for the Wolverine OpenAI-compatible endpoint."""

    base_url: str = "http://localhost:8000/v1"  # vLLM Server Address - http://localhost:8000/v1
    api_key: str = ""  # vLLM does not require a real key
    model: str = "mistralai/Ministral-3-3B-Instruct-2512"  # Model Name

    temperature: float = 0.0

WOLVERINE_SETTINGS = WolverineSettings()

#  In short: Llama-3.2-3B returned malformed JSON for 2 stories, triggering the fallback logic    
# that makes individual per-category API calls, resulting in 44 extra logged entries. The other
# models parsed successfully every time. 