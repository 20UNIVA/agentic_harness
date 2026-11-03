# Examples

```
examples/
  tasks/
    quickstart.json        11 user tasks: function tools, stateful toolkits, a built-in env + skill, MCP servers
                           (incl. mcp_setup), MCP + toolkit together, a skill that runs a script in the shell workspace,
                           Python verifiers, and an unscored open-ended task
    verifiers.py           Python verifiers used by quickstart.json
  tools/
    finance_tools.py       toolkit "finance": plain @tool functions (currency, loan payment, compound growth)
    todo_toolkit.py        toolkit "todo": stateful Toolkit class, verified with "state" checks
    library_toolkit.py     toolkit "library": stateful, own system prompt and custom check()
  skills/
    fiscal-calendar/       knowledge skill + reference CSV                (benchmark: data_sql.skill_*)
    file-archiving/        convention skill                               (benchmark: filesystem.skill_*)
    travel-api/            private API docs in reference/endpoints.md     (benchmark: travel_api.skill_*)
    csv-insights/          skill with a script, needs the shell toolkit   (quickstart: shell-skill-csv)
  mcp/
    servers.json                    default registry: kanban, wiki, memory (stdio, fresh process per episode)
    servers_http.example.json       a running Streamable HTTP server shared by all episodes
    official_servers.example.json   official reference servers (time, git, filesystem, memory) via uvx / npx
                                    (downloaded on first use; not covered by the tests)
  mcp_servers/
    kanban_server.py   wiki_server.py   memory_server.py   (official MCP Python SDK, stdio or --transport http)
    worlds.py          seed-deterministic initial data shared by servers and task generators
    _common.py         SDK 1.x/2.x compatibility, state file, runner
  configs/
    oracle_quickstart.json  openai_quickstart.json  anthropic_quickstart.json  hf_local_cpu.json  vllm_benchmark.json
```

Try them:

```bash
pip install -e ".[mcp-examples]"
agentic-harness resources                                        # skills + live MCP servers and their tools
agentic-harness run --config examples/configs/oracle_quickstart.json
agentic-harness view runs/quickstart_oracle/trajectories.jsonl --task-id mcp-plus-tools-budget
```

Docs: [tasks](../docs/tasks.md) · [tools](../docs/tools.md) · [skills](../docs/skills.md) · [MCP](../docs/mcp.md) ·
[models](../docs/models.md).
