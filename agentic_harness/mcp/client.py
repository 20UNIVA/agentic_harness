"""Minimal Model Context Protocol client (stdlib only): JSON-RPC 2.0 over stdio or Streamable HTTP.

Implements the initialize-handshake protocol revisions (2024-11-05 .. 2025-11-25):
initialize -> notifications/initialized -> tools/list (paginated) -> tools/call.
Server->client requests are answered minimally (ping -> {}, roots/list -> [], others -> method-not-found)
because the harness does not offer sampling/elicitation to tool servers.
Not implemented: the stateless 2026-07-28 envelope, resources/prompts, OAuth.
"""
from __future__ import annotations

import collections
import json
import os
import queue
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

from ..core.types import HARNESS_VERSION

CLIENT_PROTOCOL_VERSION = "2025-11-25"
HANDSHAKE_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")


class McpError(Exception):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


class McpTimeout(McpError):
    pass


def _server_request_reply(msg: dict) -> dict:
    method = msg.get("method")
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
    if method == "roots/list":
        return {"jsonrpc": "2.0", "id": msg["id"], "result": {"roots": []}}
    return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"client does not support {method}"}}


# Like the official SDKs, stdio servers only inherit a minimal environment by default, so secrets such as API keys
# in the harness environment are not passed to third-party servers. Opt in with `inherit_env: true`.
DEFAULT_INHERITED_ENV = ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")


class StdioTransport:
    def __init__(self, command: str, args: list[str], env: dict | None = None, cwd: str | None = None,
                 stderr_path: str | None = None, inherit_env: bool = False):
        self.stderr_f = open(stderr_path, "ab") if stderr_path else None
        self.stderr_lines: collections.deque = collections.deque(maxlen=20)
        full_env = dict(os.environ) if inherit_env else {k: os.environ[k] for k in DEFAULT_INHERITED_ENV if k in os.environ}
        full_env.update(env or {})
        self.proc = subprocess.Popen([command, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env=full_env, cwd=cwd, start_new_session=True)
        self.q: queue.Queue = queue.Queue()
        self.noise: list[str] = []
        self._stderr_done = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stderr(self):
        for line in self.proc.stderr:
            self.stderr_lines.append(line.decode("utf-8", "replace").rstrip())
            if self.stderr_f is not None:
                self.stderr_f.write(line)
                self.stderr_f.flush()
        self._stderr_done.set()

    def stderr_tail(self, n: int = 5) -> str:
        """Last stderr lines of the server (for error messages), waiting briefly if the process has exited."""
        if self.proc.poll() is not None:
            self._stderr_done.wait(1.0)
        return "\n".join(list(self.stderr_lines)[-n:])

    def _read(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.q.put(json.loads(line))
            except json.JSONDecodeError:
                self.noise.append(line[:200].decode("utf-8", "replace"))   # servers must not log to stdout
        self.q.put(None)   # EOF

    def send(self, msg: dict) -> None:
        if self.proc.poll() is not None:
            raise McpError(f"server process exited with code {self.proc.returncode}")
        self.proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode())
        self.proc.stdin.flush()

    def request(self, msg: dict, timeout: float) -> dict:
        self.send(msg)
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                raise McpTimeout(f"timeout after {timeout}s waiting for {msg.get('method')}")
            try:
                m = self.q.get(timeout=left)
            except queue.Empty:
                continue
            if m is None:
                raise McpError(f"server closed stdout (exit code {self.proc.poll()})")
            if "method" in m and "id" in m:            # server -> client request
                self.send(_server_request_reply(m))
            elif "method" in m:                        # notification (logging, list_changed, progress)
                continue
            elif m.get("id") == msg["id"]:
                return m

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.close()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                    try:
                        self.proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        self._stderr_done.wait(1.0)
        if self.stderr_f is not None:
            self.stderr_f.close()


class HttpTransport:
    """Streamable HTTP transport: POST JSON-RPC, accept application/json or text/event-stream responses."""

    def __init__(self, url: str, headers: dict | None = None):
        self.url = url
        self.headers = dict(headers or {})
        self.session_id: str | None = None
        self.protocol_version: str | None = None

    def _post(self, msg: dict, timeout: float):
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **self.headers}
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            h["MCP-Protocol-Version"] = self.protocol_version
        req = urllib.request.Request(self.url, data=json.dumps(msg).encode(), headers=h, method="POST")
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise McpError(f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}")
        except (urllib.error.URLError, TimeoutError) as e:
            raise McpTimeout(f"HTTP transport error: {e}")

    def send(self, msg: dict) -> None:
        r = self._post(msg, 30)
        r.read()
        r.close()

    def request(self, msg: dict, timeout: float) -> dict:
        r = self._post(msg, timeout)
        sid = r.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip()
        try:
            if ctype == "application/json":
                data = json.loads(r.read() or b"null")
                for m in (data if isinstance(data, list) else [data]):
                    if isinstance(m, dict) and m.get("id") == msg["id"] and "method" not in m:
                        return m
                raise McpError("no matching JSON-RPC response in HTTP body")
            if ctype == "text/event-stream":
                data_lines = []
                for raw in r:
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                    elif line == "" and data_lines:
                        m = json.loads("\n".join(data_lines))
                        data_lines = []
                        if "method" in m and "id" in m:
                            self.send(_server_request_reply(m))
                        elif m.get("id") == msg["id"] and "method" not in m:
                            return m
                raise McpError("event stream ended without a response")
            raise McpError(f"unexpected content-type {ctype!r}")
        finally:
            r.close()

    def close(self):
        if self.session_id:
            try:
                h = {**self.headers, "Mcp-Session-Id": self.session_id}
                urllib.request.urlopen(urllib.request.Request(self.url, headers=h, method="DELETE"), timeout=5).close()
            except Exception:
                pass


class McpClient:
    def __init__(self, transport, *, name: str = "server", timeout: float = 30.0, init_timeout: float = 60.0):
        self.t = transport
        self.name = name
        self.timeout = timeout
        self.init_timeout = init_timeout
        self._id = 0
        self._lock = threading.Lock()
        self.server_info: dict = {}
        self.instructions: str | None = None
        self.protocol_version: str | None = None

    def _rpc(self, method: str, params: dict | None = None, timeout: float | None = None) -> dict:
        with self._lock:
            self._id += 1
            msg = {"jsonrpc": "2.0", "id": self._id, "method": method}
            if params is not None:
                msg["params"] = params
            resp = self.t.request(msg, timeout or self.timeout)
        if "error" in resp:
            e = resp["error"]
            raise McpError(f"{method}: {e.get('message')}", e.get("code"))
        return resp.get("result") or {}

    def initialize(self) -> dict:
        res = self._rpc("initialize", {"protocolVersion": CLIENT_PROTOCOL_VERSION, "capabilities": {},
                                       "clientInfo": {"name": "agentic-harness", "version": HARNESS_VERSION}},
                        timeout=self.init_timeout)
        pv = res.get("protocolVersion")
        if pv not in HANDSHAKE_VERSIONS:
            raise McpError(f"server {self.name} negotiated unsupported protocol version {pv!r}")
        self.protocol_version = pv
        if isinstance(self.t, HttpTransport):
            self.t.protocol_version = pv
        self.server_info = res.get("serverInfo") or {}
        self.instructions = res.get("instructions")
        self.t.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return res

    def list_tools(self) -> list[dict]:
        tools, cursor = [], None
        for _ in range(100):
            res = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            tools += res.get("tools") or []
            cursor = res.get("nextCursor")
            if not cursor:
                return tools
        raise McpError("tools/list pagination did not terminate")

    def call_tool(self, name: str, arguments: dict, timeout: float | None = None) -> tuple[str, bool, dict]:
        """Returns (text_for_model, is_error, raw_result). JSON-RPC level errors (e.g. unknown tool,
        invalid params) are returned as tool errors so the model can recover."""
        try:
            res = self._rpc("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)
        except McpTimeout:
            raise
        except McpError as e:
            if e.code is not None:
                return json.dumps({"error": str(e)}, ensure_ascii=False), True, {}
            raise
        return content_to_text(res), bool(res.get("isError")), res

    def close(self):
        self.t.close()


def content_to_text(res: dict) -> str:
    parts = []
    for c in res.get("content") or []:
        typ = c.get("type")
        if typ == "text":
            parts.append(c.get("text", ""))
        elif typ in ("image", "audio"):
            parts.append(f"[{typ} omitted: {c.get('mimeType', '?')}]")
        elif typ == "resource":
            r = c.get("resource") or {}
            parts.append(r.get("text") if "text" in r else f"[resource {r.get('uri')}]")
        elif typ == "resource_link":
            parts.append(f"[resource link {c.get('uri')}]")
        else:
            parts.append(json.dumps(c, ensure_ascii=False))
    if not parts and res.get("structuredContent") is not None:
        parts.append(json.dumps(res["structuredContent"], ensure_ascii=False))
    return "\n".join(p for p in parts if p is not None)
