"""
Demo: print the MCP tool definitions generated from the category mapping,
and simulate a routed call with a mock endpoint (no live server needed).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from routed_mcp_server import CATEGORY_ENDPOINT_MAP, build_mcp_server


def show_tool_definitions(mapping: dict) -> None:
    mcp = build_mcp_server(mapping)
    tools = asyncio.run(mcp.get_tools())

    print("=== Generated MCP Tools ===\n")
    for name, tool in tools.items():
        print(f"Tool : {name}")
        print(f"  Description : {tool.description}")
        schema = tool.parameters
        props = schema.get("properties", {})
        for param, info in props.items():
            required = param in schema.get("required", [])
            print(f"  Param: {param!r}  type={info.get('type','?')}  required={required}")
        print()


async def simulate_call(category: str, prompt: str) -> None:
    """Call the generated tool with a mocked OpenAI client response."""
    mock_choice = MagicMock()
    mock_choice.message.content = f"[MOCK {category.upper()} RESPONSE] Answered: {prompt!r}"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch("routed_mcp_server.AsyncOpenAI", return_value=mock_client):
        mcp = build_mcp_server(CATEGORY_ENDPOINT_MAP)
        tools = await mcp.get_tools()
        tool_name = f"ask_{category}"
        if tool_name not in tools:
            print(f"No tool found for category '{category}'")
            return
        result = await mcp.call_tool(tool_name, {"prompt": prompt})
        print(f"=== Simulated call: {tool_name} ===")
        print(f"  Prompt : {prompt!r}")
        print(f"  Result : {result[0].text}")
        print()


if __name__ == "__main__":
    show_tool_definitions(CATEGORY_ENDPOINT_MAP)

    print("=== Simulated Calls ===\n")
    asyncio.run(simulate_call("code", "Write a binary search in Python."))
    asyncio.run(simulate_call("math", "What is the integral of x^2?"))
    asyncio.run(simulate_call("creative_writing", "Write a haiku about winter."))
