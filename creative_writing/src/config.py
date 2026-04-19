import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

@dataclass(frozen=True)
class WolverineSettings:
    """Runtime configuration for the Wolverine OpenAI-compatible endpoint."""

    # base_url: str = "http://localhost:8000/v1"  # vLLM Server Address
    # api_key: str = ""  # vLLM does not require a real key
    # model: str = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"  # Model Name
    # base_url: str = "https://tritonai-api.ucsd.edu"  # TRITONAI
    # api_key: str = os.getenv("TRITON_API_KEY", "")
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    model: str = "anthropic/claude-sonnet-4-6"

    temperature: float = 0.0

WOLVERINE_SETTINGS = WolverineSettings()

#  In short: Llama-3.2-3B returned malformed JSON for 2 stories, triggering the fallback logic    
# that makes individual per-category API calls, resulting in 44 extra logged entries. The other
# models parsed successfully every time. 