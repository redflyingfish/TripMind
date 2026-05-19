from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import ExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class McpTravelClient:
    """Synchronous facade over TripMind's MCP travel-tools server."""

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return asyncio.run(_call_tool(name, arguments))


async def _call_tool(name: str, arguments: dict[str, Any]) -> Any:
    server = StdioServerParameters(command=sys.executable, args=["-m", "tripmind.mcp_server"])
    with ExitStack() as stack:
        errlog = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
        async with stdio_client(server, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return _parse_tool_result(result)


def _parse_tool_result(result) -> Any:
    if getattr(result, "isError", False):
        raise RuntimeError(f"MCP tool call failed: {result}")
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and "result" in structured:
        return structured["result"]
    content = getattr(result, "content", [])
    if not content:
        return None
    if len(content) > 1:
        return [json.loads(item.text) for item in content if getattr(item, "text", None) is not None]
    first = content[0]
    text = getattr(first, "text", None)
    if text is None:
        return first
    return json.loads(text)
