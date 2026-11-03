"""External MCP server registration (config files) and per-episode toolsets.

Config format = the common `mcpServers` JSON used by Claude Desktop / Claude Code, plus optional harness keys:

{
  "mcpServers": {
    "kanban": {
      "command": "${PYTHON}",
      "args": ["${CONFIG_DIR}/../mcp_servers/kanban_server.py", "--state", "${EPISODE_DIR}/kanban.json",
               "--seed", "${TASK_SEED}"],
      "env": {"SOME_VAR": "x"},          # optional; only PATH/HOME/USER/LANG... are inherited unless "inherit_env": true
      "cwd": "${EPISODE_DIR}",            # optional
      "scope": "episode",                 # "episode" (fresh process + dir per episode, default) | "run" (shared)
      "timeout_s": 30,                    # per tools/call
      "allowed_tools": ["list_cards"],    # optional allow-list (server tool names)
      "disabled_tools": [],               # optional deny-list
      "description": "free text shown in docs"
    },
    "remote": {"type": "http", "url": "http://127.0.0.1:8765/mcp", "headers": {"Authorization": "Bearer ${env:TOKEN}"},
               "scope": "run"}
  }
}

Placeholders: ${PYTHON} ${HARNESS_ROOT} ${CONFIG_DIR} ${EPISODE_DIR} ${TASK_SEED} ${TASK_ID} ${env:NAME}.
Model-facing tool names are `mcp__<server>__<tool>` (sanitized, <=64 chars).
SECURITY: stdio servers run arbitrary local commands with the harness user's permissions and are NOT network-sandboxed.
Register only servers you trust.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading

from ..core.types import InfraError, sha256_of
from .client import HttpTransport, McpClient, McpError, StdioTransport

HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_examples_dir() -> str:
    """The repo's examples/ folder: next to the package (source checkout or editable install), else ./examples
    (a non-editable install used from a checkout)."""
    for root in (HARNESS_ROOT, os.getcwd()):
        d = os.path.join(root, "examples")
        if os.path.isdir(os.path.join(d, "skills")) and os.path.isdir(os.path.join(d, "mcp_servers")):
            return d
    return os.path.join(HARNESS_ROOT, "examples")


EXAMPLES_DIR = _find_examples_dir()
PREFIX = "mcp__"
_PH = re.compile(r"\$\{([^}]+)\}")


def tool_alias(server: str, tool: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]", "_", f"{PREFIX}{server}__{tool}")
    return name[:64]


def simplify_schema(schema):
    """Model-facing schema cleanup: drop pydantic `title`s and `$schema`, and collapse Optional[X]
    (anyOf: [X, {type: null}]) to X. The server still validates against its own schema."""
    if isinstance(schema, list):
        return [simplify_schema(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in ("title", "$schema"):
            continue
        out[k] = {pk: simplify_schema(pv) for pk, pv in v.items()} if k in ("properties", "$defs", "definitions") \
            else simplify_schema(v)
    for key in ("anyOf", "oneOf"):
        alts = out.get(key)
        if isinstance(alts, list):
            non_null = [a for a in alts if not (isinstance(a, dict) and a.get("type") == "null")]
            if len(non_null) == 1 and len(non_null) < len(alts):
                merged = {**{k: v for k, v in out.items() if k != key}, **non_null[0]}
                return merged
    return out


def _subst(value, ctx: dict):
    if isinstance(value, str):
        def rep(m):
            key = m.group(1)
            if key.startswith("env:"):
                return os.environ.get(key[4:], "")
            if key not in ctx:
                raise KeyError(f"unknown placeholder ${{{key}}}")
            return str(ctx[key])
        return _PH.sub(rep, value)
    if isinstance(value, list):
        return [_subst(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: _subst(v, ctx) for k, v in value.items()}
    return value


class McpToolset:
    """Connected MCP server as seen by one episode."""

    def __init__(self, server: str, client: McpClient, tools: list[dict], spec: dict, owned: bool):
        self.server = server
        self.client = client
        self.spec = spec
        self.owned = owned                       # episode-scoped clients are closed with the episode
        allow, deny = spec.get("allowed_tools"), set(spec.get("disabled_tools") or [])
        self.tools = [t for t in tools if (allow is None or t["name"] in allow) and t["name"] not in deny]
        self.by_alias = {tool_alias(server, t["name"]): t for t in self.tools}
        self.timeout = float(spec.get("timeout_s", 30))

    def tool_schemas(self) -> list[dict]:
        out = []
        for alias, t in self.by_alias.items():
            params = simplify_schema(t.get("inputSchema") or {"type": "object", "properties": {}})
            params.setdefault("type", "object")
            params.setdefault("properties", {})
            desc = (t.get("description") or t.get("title") or t["name"]).strip()
            out.append({"type": "function", "function": {"name": alias, "description": f"[{self.server}] {desc}",
                                                         "parameters": params}})
        return out

    def handles(self, alias: str) -> bool:
        return alias in self.by_alias

    def is_mutating(self, alias: str) -> bool:
        ann = (self.by_alias.get(alias) or {}).get("annotations") or {}
        return not ann.get("readOnlyHint", False)

    def call(self, alias: str, args: dict) -> tuple[str, bool]:
        t = self.by_alias[alias]
        try:
            text, is_err, _ = self.client.call_tool(t["name"], args, timeout=self.timeout)
        except McpError as e:
            raise InfraError(f"MCP server {self.server}: {e}")
        return text, is_err

    def call_raw(self, tool: str, args: dict) -> tuple[str, bool, dict]:
        """Direct call by server tool name (used by setup steps and post-episode checks)."""
        try:
            return self.client.call_tool(tool, args, timeout=self.timeout)
        except McpError as e:
            raise InfraError(f"MCP server {self.server}: {e}")

    def provenance(self) -> dict:
        return {"server": self.server, "server_info": self.client.server_info, "protocol": self.client.protocol_version,
                "scope": self.spec.get("scope", "episode"),
                "tools_hash": sha256_of([[t["name"], t.get("inputSchema")] for t in self.tools]),
                "tools": [t["name"] for t in self.tools]}

    def close(self):
        if self.owned:
            self.client.close()


class McpRegistry:
    def __init__(self, config_paths: list[str] | None = None):
        self.specs: dict[str, dict] = {}
        self.config_dirs: dict[str, str] = {}
        self._shared: dict[str, tuple[McpClient, list]] = {}
        self._lock = threading.Lock()
        for p in config_paths or []:
            self.add_config(p)

    def add_config(self, path: str) -> None:
        with open(path) as f:
            cfg = json.load(f)
        servers = cfg.get("mcpServers", cfg)
        for name, spec in servers.items():
            if name.startswith("_"):
                continue
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(f"invalid MCP server name {name!r}")
            if name in self.specs:
                raise ValueError(f"duplicate MCP server {name!r} (from {path})")
            if "command" not in spec and "url" not in spec:
                raise ValueError(f"MCP server {name!r} needs 'command' (stdio) or 'url' (http)")
            self.specs[name] = spec
            self.config_dirs[name] = os.path.dirname(os.path.abspath(path))

    def names(self) -> list[str]:
        return sorted(self.specs)

    def _connect(self, name: str, ctx: dict) -> tuple[McpClient, list[dict]]:
        spec = self.specs[name]
        full = {"PYTHON": sys.executable, "HARNESS_ROOT": HARNESS_ROOT, "CONFIG_DIR": self.config_dirs[name], **ctx}
        tr = None
        try:
            if "url" in spec:
                tr = HttpTransport(_subst(spec["url"], full), _subst(spec.get("headers") or {}, full))
            else:
                log = os.path.join(ctx["EPISODE_DIR"], f"mcp_{name}.stderr.log") if ctx.get("EPISODE_DIR") else None
                tr = StdioTransport(_subst(spec["command"], full), _subst(spec.get("args") or [], full),
                                    env=_subst(spec.get("env") or {}, full),
                                    cwd=_subst(spec["cwd"], full) if spec.get("cwd") else None, stderr_path=log,
                                    inherit_env=bool(spec.get("inherit_env", False)))
            client = McpClient(tr, name=name, timeout=float(spec.get("timeout_s", 30)),
                               init_timeout=float(spec.get("init_timeout_s", 60)))
            client.initialize()
            return client, client.list_tools()
        except (McpError, OSError, KeyError) as e:
            tail = ""
            if tr is not None:
                try:
                    tail = tr.stderr_tail() if isinstance(tr, StdioTransport) else ""
                    tr.close()
                except Exception:
                    pass
            raise InfraError(f"failed to start MCP server {name!r}: {e}" + (f"\nserver stderr:\n{tail}" if tail else ""))

    def open(self, names: list[str], ctx: dict) -> list[McpToolset]:
        """Connect the given servers for one episode. ctx provides EPISODE_DIR / TASK_SEED / TASK_ID."""
        out = []
        try:
            for n in names:
                if n not in self.specs:
                    raise InfraError(f"MCP server {n!r} is not registered (registered: {self.names()})")
                spec = self.specs[n]
                if spec.get("scope", "episode") == "run":
                    with self._lock:
                        if n not in self._shared:
                            self._shared[n] = self._connect(n, {k: v for k, v in ctx.items() if k not in ("EPISODE_DIR",)})
                        client, tools = self._shared[n]
                    out.append(McpToolset(n, client, tools, spec, owned=False))
                else:
                    client, tools = self._connect(n, ctx)
                    out.append(McpToolset(n, client, tools, spec, owned=True))
        except Exception:
            for ts in out:
                ts.close()
            raise
        return out

    def close(self):
        for client, _ in self._shared.values():
            client.close()
        self._shared.clear()
