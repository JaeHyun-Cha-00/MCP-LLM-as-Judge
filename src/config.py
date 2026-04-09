from dataclasses import dataclass

@dataclass(frozen=True)
class WolverineSettings:
    """Runtime configuration for the Wolverine OpenAI-compatible endpoint."""

    base_url: str = "SERVER ADDRESS"  # vLLM Server Address - http://localhost:8000/v1
    api_key: str = "EMPTY"
    model: str = "MODEL NAME"  # Model Name
    # meta-llama/Llama-3.1-8B-Instruct
    # meta-llama/Llama-3.2-3B-Instruct
    # google/gemma-3-4b-it
    # google/gemma-3-12b-it
    # Qwen/Qwen3-4B-Instruct-2507
    # Qwen/Qwen2.5-7B-Instruct

    temperature: float = 0.0

WOLVERINE_SETTINGS = WolverineSettings()