"""A minimal MCP stdio server written with the standard library only (no SDK). Used by the tests to check the client
against a non-SDK implementation: it sends a log notification and a ping request to the client before answering."""
import json
import os
import sys

TOOLS = [
    {"name": "get_env", "description": "Return an environment variable of the server process.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
     "annotations": {"readOnlyHint": True}},
    {"name": "big", "description": "Return a long text.", "inputSchema": {"type": "object", "properties": {"n": {"type": "integer"}}}},
]


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    msg = json.loads(line)
    method, mid = msg.get("method"), msg.get("id")
    if mid is None and method:                       # notifications
        continue
    if method is None:                               # response to our ping
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                                                      "serverInfo": {"name": "plain", "version": "0.0.1"},
                                                      "instructions": "Plain test server."}})
    elif method == "tools/list":
        cursor = (msg.get("params") or {}).get("cursor")
        if cursor is None:                           # paginate: one tool per page
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS[:1], "nextCursor": "p2"}})
        else:
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS[1:]}})
    elif method == "tools/call":
        send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "working"}})
        send({"jsonrpc": "2.0", "id": "srv-ping-1", "method": "ping"})
        p = msg["params"]
        if p["name"] == "get_env":
            text = os.environ.get(p["arguments"]["name"], "")
        elif p["name"] == "big":
            text = "x" * int(p["arguments"].get("n", 100))
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"Unknown tool: {p['name']}"}})
            continue
        send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}], "isError": False}})
    else:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found"}})
