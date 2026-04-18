from dataclasses import dataclass

@dataclass(frozen=True)
class WolverineSettings:
    """Runtime configuration for the Wolverine OpenAI-compatible endpoint."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model: str = "mistralai/Ministral-3-3B-Instruct-2512"
    temperature: float = 0.0

WOLVERINE_SETTINGS = WolverineSettings()
