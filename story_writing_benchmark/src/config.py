from dataclasses import dataclass

@dataclass(frozen=True)
class WolverineSettings:
    """Runtime configuration for the Wolverine OpenAI-compatible endpoint."""

    base_url: str = "http://localhost:8001/v1"
    api_key: str = ""
    # model: str = "google/gemma-4-E2B-it"
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    temperature: float = 0.0

WOLVERINE_SETTINGS = WolverineSettings()
