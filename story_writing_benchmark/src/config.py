import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

@dataclass(frozen=True)
class WolverineSettings:
    """Runtime configuration for the Wolverine OpenAI-compatible endpoint."""

    # base_url: str = "http://localhost:8001/v1"  # vLLM Server Address
    # model: str = ""  # Model Name
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    model: str = "anthropic/claude-sonnet-4-6"

    temperature: float = 0.0

WOLVERINE_SETTINGS = WolverineSettings()
