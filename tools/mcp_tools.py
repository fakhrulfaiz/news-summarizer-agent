from __future__ import annotations

DDG_MCP_CONFIG = {
    "duckduckgo": {
        "transport": "stdio",
        "command": "docker",
        "args": ["run", "-i", "--rm", "mcp/duckduckgo"],
    }
}
