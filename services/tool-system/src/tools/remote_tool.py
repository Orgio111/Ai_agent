"""Remote tool: calls external HTTP endpoints registered dynamically."""
from __future__ import annotations

from typing import Any

import httpx


class RemoteTool:
    def __init__(
        self, name: str, description: str, category: str,
        endpoint: str, parameters: dict, required_role: str = "agent"
    ) -> None:
        self.name = name
        self.description = description
        self.category = category
        self.endpoint = endpoint
        self.parameters = parameters
        self.required_role = required_role
        self.sandboxed = False

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.endpoint, json=args)
            resp.raise_for_status()
            return resp.json()
