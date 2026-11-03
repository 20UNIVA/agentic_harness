"""Shared helpers for the example MCP servers.

The servers use the official MCP Python SDK (`pip install mcp`), so they behave like any third-party server the
harness might connect to. Works with SDK 2.x (`MCPServer`) and 1.x (`FastMCP`).
The `mcp` package is imported lazily so task generators can import the world builders without it.

State lives in a JSON file (`--state`), one per episode, so every rollout starts from a fresh, seed-determined world
and post-episode checks can inspect the result through the server's own tools.
"""
from __future__ import annotations

import argparse
import json
import os
import threading


class StateFile:
    def __init__(self, path: str, build_world, seed: int):
        self.path = path
        self.lock = threading.Lock()
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._write(build_world(seed))

    def _write(self, data):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def read(self) -> dict:
        with open(self.path) as f:
            return json.load(f)

    def update(self, fn):
        """Apply fn(state) -> result atomically and persist the mutated state."""
        with self.lock:
            st = self.read()
            out = fn(st)
            self._write(st)
            return out


def make_server(name: str, instructions: str):
    try:
        from mcp.server.mcpserver import MCPServer      # SDK 2.x
        return MCPServer(name=name, instructions=instructions, version="0.1.0")
    except ImportError:
        from mcp.server.fastmcp import FastMCP          # SDK 1.x
        return FastMCP(name=name, instructions=instructions)


def tool_error(msg: str) -> Exception:
    """Exception that the SDK turns into an `isError: true` tool result shown to the model."""
    try:
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        from mcp.server.fastmcp.exceptions import ToolError
    return ToolError(msg)


def read_only():
    from mcp.types import ToolAnnotations
    return ToolAnnotations(readOnlyHint=True)


def write_tool(destructive: bool = False):
    from mcp.types import ToolAnnotations
    return ToolAnnotations(readOnlyHint=False, destructiveHint=destructive)


def parse_args(default_state: str):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=default_state, help="JSON state file (created from --seed if missing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--transport", default="stdio", choices=["stdio", "http"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    return ap.parse_args()


def run(server, args) -> None:
    if args.transport == "stdio":
        server.run("stdio")
        return
    try:
        server.run("streamable-http", host=args.host, port=args.port)
    except TypeError:                                   # SDK 1.x: host/port live in settings
        server.settings.host, server.settings.port = args.host, args.port
        server.run("streamable-http")
