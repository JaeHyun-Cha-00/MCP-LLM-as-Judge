"""
MCP server that generates one tool per category, routing each to a
designated open-weight model endpoint (OpenAI-compatible API).

Reads a routing_config.json produced by generate_config.py.

Usage:
    python routed_mcp_server.py --config ../configs/routing_config.json
    python routed_mcp_server.py --config ../configs/routing_config.json --transport sse --port 8080
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Core: build a handler for one category entry
# ---------------------------------------------------------------------------

def _make_handler(category: str, entry: dict[str, Any]):
    """Return an async callable that invokes the judge model for `category`."""
    ep = entry["endpoint"]
    ev = entry["evaluation"]
    ds = entry["dataset"]

    api_key = os.environ.get(ep.get("api_key_env", ""), ep.get("api_key", "EMPTY")) or "EMPTY"
    client = AsyncOpenAI(base_url=ep["base_url"], api_key=api_key)

    model       = ep["model"]
    max_tokens  = ep.get("max_tokens", 1024)
    temperature = ep.get("temperature", 0.0)
    system_prompt    = ev["system_prompt"]
    prompt_template  = ev["user_prompt_template"]
    text_column      = ds.get("text_column", "story")

    async def handler(
        story: str,
        system_override: str | None = None,
        max_tokens_override: int | None = None,
    ) -> str:
        """
        Evaluate `story` across all rubrics defined for this category.

        Args:
            story: The story text to evaluate.
            system_override: Optional replacement for the default system prompt.
            max_tokens_override: Override the default max_tokens for this call.

        Returns:
            Raw JSON string from the judge model containing a 'scores' object.
        """
        user_prompt = prompt_template.replace(f"{{{text_column}}}", story)
        messages = [
            {"role": "system", "content": system_override or system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens_override or max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    return handler


# ---------------------------------------------------------------------------
# Build server from config dict
# ---------------------------------------------------------------------------

def build_mcp_server(config: dict[str, Any]) -> FastMCP:
    server_name = config.get("server_name", "llm-judge-router")
    mcp = FastMCP(server_name)

    for category, entry in config["categories"].items():
        tool_name   = f"evaluate_{category}"
        description = entry.get("description", f"Evaluate a {category} story.")
        handler     = _make_handler(category, entry)

        handler.__name__ = tool_name
        handler.__doc__  = f"{description}\n\n" + (handler.__doc__ or "")

        mcp.tool(name=tool_name, description=description)(handler)

    return mcp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Routed LLM-Judge MCP server")
    parser.add_argument(
        "--config",
        metavar="FILE",
        default=str(Path(__file__).parent.parent / "configs" / "routing_config.json"),
        help="Path to routing_config.json (default: ../configs/routing_config.json)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with open(args.config) as f:
        config = json.load(f)

    mcp = build_mcp_server(config)

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
