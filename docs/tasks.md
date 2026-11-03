# Tasks and verification

A task is one problem for the model: a first user message, the capabilities the model may use, and how to check the
result. Task files are JSON (`{"tasks": [...]}` or a list), JSONL, or YAML (with PyYAML installed).
`agentic-harness run --tasks <file>` accepts both user tasks (this page) and the generated benchmark tasks.

## Fields

Only `instruction` is required.

| field | type | meaning |
|---|---|---|
| `id` | str | unique id (default `task-<n>`) |
| `instruction` | str | the first user message |
| `system_prompt` | str | default: the env's prompt, or a generic tool-using assistant prompt |
| `env` | str or `{"domain", "seed"}` | a built-in environment (`retail`, `workspace`, `filesystem`, `travel_api`, `data_sql`) |
| `toolkits` | [str] | user toolkits ([tools.md](tools.md)) or built-in ones (`shell`) |
| `skills` | [str] | skills to offer ([skills.md](skills.md)) |
| `mcp_servers` | [str] | MCP servers to connect ([mcp.md](mcp.md)) |
| `mcp_setup` | [{`server`, `tool`, `arguments`}] | calls made by the harness before the episode, e.g. to seed a server |
| `files` | {path: text} | files written into the shell workspace |
| `followups` | [str] | scripted user replies; one is sent each time the assistant answers without calling a tool |
| `max_turns` | int | assistant turns before the episode is stopped (default 12) |
| `gold_actions` | [{`name`, `arguments`}] | reference solution: replayed by the `oracle` policy; env steps also define the expected env state |
| `gold_final` | str | the oracle's final answer |
| `verify` | object | see below; without it the task is run and saved but marked `unscored` |
| `category`, `difficulty`, `lang`, `metadata` | any | labels used in reports |

Unknown fields are rejected (typos fail loudly).

## Verification

All listed conditions must hold for `reward = 1`.

```json
"verify": {
  "answers": ["Paris", 42, {"number": 1520.06, "tol": 0.5}, {"regex": "^Yes\\b"}, {"contains": "aisle", "alts": ["통로"]}],
  "answer_forbidden": ["112,000"],
  "checks": [
    {"type": "state", "toolkit": "todo", "json_subset": {"items": [{"text": "Book dentist", "done": false}]}},
    {"type": "toolkit", "toolkit": "library", "kind": "on_loan", "book_id": "B2", "member": "carol"},
    {"type": "mcp_call", "server": "kanban", "tool": "get_card", "arguments": {"card_id": "CARD-103"},
     "expect_json": {"column": "done"}},
    {"type": "mcp_call", "server": "memory", "tool": "open_nodes", "arguments": {"names": ["Jordan"]},
     "expect_contains": ["vegetarian"], "expect_not_contains": ["deleted"]},
    {"type": "skill_loaded", "name": "fiscal-calendar"}
  ],
  "forbidden_actions": [{"name": "cancel_pending_order", "match": {"order_id": "#W123"}},
                        {"name": "http_request", "match": {"method": "POST", "body.item_id": "HT009"}}],
  "expect_no_state_change": true,
  "python": "verifiers.py:my_check"
}
```

| key | checks |
|---|---|
| `answers` | the final assistant message. Strings: case-insensitive substring (`alts` = accepted alternatives); an ISO date such as `"2026-07-13"` also matches `2026/7/13`, `2026.07.13`, `2026년 7월 13일`, `July 13, 2026` and `13 Jul 2026`. Numbers: any number in the text within `tol` (commas ignored). `regex`: `re.search` (case-insensitive). |
| `answer_forbidden` | none of these strings may appear in the final message (information leaks) |
| `checks` | post-episode state. `state`: subset match against a toolkit's (or the env's) `state()`. `toolkit`: the toolkit's own `check()`. `mcp_call`: call a server tool after the episode and match its result (`expect_contains`, `expect_not_contains`, `expect_json` subset). `skill_loaded`: the model loaded a skill. Built-in env checks are also allowed. |
| `forbidden_actions` | no executed call may match (`match` keys may use dots for nested arguments; string values match path suffixes) |
| `expect_no_state_change` | the built-in env state hash must be unchanged (refusal tasks) |
| `python` | `"file.py:function"` (relative to the task file) or `"package.module:function"`. Called as `fn(task=, final_text=, session=, executed_calls=)`. Returns `bool`, a float in `[0, 1]` (partial credit) or `{"reward": x, ...info}`. `session.check({...})` and `session.call_mcp(server, tool, args)` are available. |

With `gold_actions` and an `env`, the expected env state is computed by replaying the env steps, so state-changing tasks
are verified by comparing the final state (any action order that reaches the same state passes).

## Outcomes

| status | reward | meaning |
|---|---|---|
| `ok` | 1.0 (or partial) | all conditions hold and the model ended with a final answer |
| `policy_fail` | 0.0 | `failure_kind`: `wrong_state`, `wrong_answer`, `forbidden_action`, `leak`, `check_failed`, `python_verifier`, `format_*` |
| `max_turns`, `loop_abort`, `truncated`, `format_fail` | 0.0 | the episode did not end normally |
| `unscored` | None | the task has no verification |
| `infra_fail` | None | harness / server / backend failure (MCP server crash, API error, verifier exception, tool code bug); excluded from scores and counted separately |

## Episode rules

- The model may call several tools in one turn; results are returned in order.
- A turn without tool calls is the final answer, unless `followups` remain (then the next follow-up is sent).
- A malformed tool call gets an error message back (`--on-parse-error feedback`, default) or ends the episode (`end`).
- Four identical consecutive calls stop the episode (`loop_abort`); `max_turns` bounds the rest.

## Example

[examples/tasks/quickstart.json](../examples/tasks/quickstart.json) has 11 tasks covering function tools, stateful
toolkits, a built-in env with a skill, MCP servers (with `mcp_setup`), a skill that runs a script in the shell
workspace, Python verifiers and an unscored open-ended task. Validate a task file without a model:

```bash
agentic-harness run --provider oracle --tasks my_tasks.json --toolkits my_tools.py --out runs/check
```
