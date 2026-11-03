# External MCP servers

Any [Model Context Protocol](https://modelcontextprotocol.io) server can provide tools. The harness has its own
stdlib MCP client (JSON-RPC 2.0 over **stdio** or **Streamable HTTP**, initialize-handshake protocol revisions
2024-11-05 to 2025-11-25), tested against servers built with the official Python SDK (1.x and 2.x).

## Registering servers

The config is the common `mcpServers` JSON (as in Claude Desktop / Claude Code) plus optional harness keys:

```json
{
  "mcpServers": {
    "kanban": {
      "command": "${PYTHON}",
      "args": ["${CONFIG_DIR}/../mcp_servers/kanban_server.py", "--state", "${EPISODE_DIR}/kanban.json", "--seed", "${TASK_SEED}"],
      "env": {},
      "scope": "episode",
      "timeout_s": 30
    },
    "remote_wiki": {
      "type": "http",
      "url": "https://wiki.example.com/mcp",
      "headers": {"Authorization": "Bearer ${env:WIKI_TOKEN}"},
      "scope": "run",
      "allowed_tools": ["search_docs", "get_doc"]
    }
  }
}
```

| key | meaning |
|---|---|
| `command`, `args`, `env`, `cwd` | stdio server process. Only a minimal environment (`PATH`, `HOME`, `USER`, `LANG`, ...) is inherited plus `env`, so API keys in your shell are not passed to servers; set `"inherit_env": true` to pass everything |
| `type: "http"`, `url`, `headers` | Streamable HTTP server (JSON or SSE responses, `Mcp-Session-Id` handled) |
| `scope` | `episode` (default): a new process and a new `${EPISODE_DIR}` per episode, so no state leaks between rollouts. `run`: one shared connection for the whole run (read-only or remote services) |
| `timeout_s`, `init_timeout_s` | per tool call / initialize |
| `allowed_tools`, `disabled_tools` | filter the server's tools before the model sees them |
| `description` | free text |

Placeholders: `${PYTHON}` (current interpreter), `${CONFIG_DIR}` (folder of the config file; use it for paths next to
the config), `${HARNESS_ROOT}` (folder that contains the `agentic_harness` package),
`${EPISODE_DIR}`, `${TASK_SEED}`, `${TASK_ID}`, `${env:NAME}`.

```bash
agentic-harness resources --mcp-config my_servers.json                 # connect to each server and list its tools
agentic-harness run --mcp-config my_servers.json --extra-mcp remote_wiki ...
```

Which servers an episode connects to: the task's `mcp_servers` plus `--extra-mcp`. The repo's
`examples/mcp/servers.json` is registered by default from a source checkout (`--no-example-resources` disables it).

## What happens in an episode

1. The server is started (stdio) or connected (HTTP); `initialize`, `notifications/initialized` and a paginated
   `tools/list` run; the server's `instructions` are appended to the system prompt.
2. Each tool is exposed as `mcp__<server>__<tool>` with the server's JSON Schema. For a shorter prompt, pydantic
   `title`s are dropped and `Optional[X]` (`anyOf: [X, null]`) is collapsed to `X`; the server still validates
   arguments against its own schema.
3. The task's `mcp_setup` calls run (seed data through the server's own tools; works with any server).
4. Model calls are routed to the server. Text content is returned to the model; images/audio become placeholders;
   `isError` results and JSON-RPC errors (unknown tool, invalid params) are returned as tool errors so the model can
   recover.
5. After the episode, the task's `mcp_call` checks query the server, then the server is stopped and the episode
   directory is deleted (`--keep-episode-dirs` keeps it for debugging; stderr of stdio servers is logged there).
6. Start-up failures, timeouts and crashes are infrastructure failures (`reward=None`), never model failures.

Server-to-client requests are answered minimally (`ping` → `{}`, `roots/list` → `[]`, others → method not found);
the harness does not offer sampling or elicitation to servers.

Trajectories record, per server: server name/version, negotiated protocol, scope and a hash of the tool list.

## Example servers

`examples/mcp_servers/` contains three servers written with the official MCP Python SDK (`pip install mcp`), so they
behave like third-party servers. Their initial data comes from a seed (`worlds.py`), which makes tasks verifiable.

| server | tools | kind |
|---|---|---|
| `kanban` | `list_members`, `list_cards`, `get_card`, `create_card`, `move_card`, `assign_card`, `add_comment` | stateful board |
| `wiki` | `list_spaces`, `search_docs`, `get_doc` | read-only knowledge base |
| `memory` | `create_entities`, `create_relations`, `add_observations`, `delete_*`, `read_graph`, `search_nodes`, `open_nodes` | knowledge-graph memory (tool names follow the reference `server-memory`) |

Run one standalone, over HTTP:

```bash
python examples/mcp_servers/wiki_server.py --seed 7 --state /tmp/wiki.json --transport http --port 8772
agentic-harness resources --mcp-config examples/mcp/servers_http.example.json
```

`examples/mcp/official_servers.example.json` shows how to register official reference servers (time, git, filesystem,
memory) through `uvx` / `npx`; they download packages on first use and are not covered by the test suite.

## Writing MCP tasks

```json
{"instruction": "The task 'Migrate CI to new runners' is finished. Move its card to done.",
 "env": {"domain": "none", "seed": 7}, "mcp_servers": ["kanban"],
 "verify": {"checks": [{"type": "mcp_call", "server": "kanban", "tool": "get_card", "arguments": {"card_id": "CARD-103"},
                        "expect_json": {"column": "done"}}]}}
```

Tasks can combine MCP servers with built-in envs and user toolkits (see `mcp-plus-tools-budget` in
`examples/tasks/quickstart.json` and the `mcp_examples.cross_meeting_card` benchmark template).

## Security and limits

- stdio servers run local commands **with your permissions** and are **not network-sandboxed** by the harness.
  Register only servers you trust, point file access at `${EPISODE_DIR}`, and use containers for untrusted servers.
- Each `episode`-scoped server costs a process start (about 0.4 s for the Python examples); prefer `scope: "run"`
  for read-only servers in large RL runs.
- Not supported yet: the stateless 2026-07-28 protocol envelope, resources/prompts, sampling/elicitation, OAuth flows.
