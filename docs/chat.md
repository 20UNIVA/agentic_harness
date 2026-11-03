# Chat with your agent

`agentic-harness chat` lets you play the user: you type messages, the model answers with the same tools, skills and
MCP servers as in `run`, and every conversation is saved as a trajectory. Use it to try a model by hand, to collect
human-in-the-loop trajectories, or to turn real requests into tasks.

## Start

```bash
# a model behind an OpenAI-compatible server (vLLM, SGLang, Ollama, LM Studio, ...)
agentic-harness chat --provider openai_compatible --base-url http://127.0.0.1:8000/v1 --model my-model \
    --toolkits examples/tools/finance_tools.py --use-mcp kanban

# the same model / resources as a run config (its "tasks" and "out" are ignored)
agentic-harness chat --config examples/configs/openai_quickstart.json

# Claude, attached to the built-in retail environment (seed 3)
agentic-harness chat --provider anthropic --model claude-opus-5-5 --env retail --seed 3 --use-mcp --use-skills

# a local Hugging Face model, no server
agentic-harness chat --provider hf_local --model /path/to/model --device cpu --toolkits my_tools.py
```

What the agent gets:

| flag | default |
|---|---|
| `--toolkits FILE` (repeatable) | registers toolkits; all registered toolkits are offered (pick some with `--use-toolkits NAME ...`) |
| `--skills-dir DIR` | registers skills; all are offered (`--use-skills NAME ...`, or `--use-skills` alone for none) |
| `--mcp-config FILE` | registers MCP servers; all are connected (`--use-mcp NAME ...`, or `--use-mcp` alone for none) |
| `--env DOMAIN --seed N` | attaches a built-in environment (retail, workspace, filesystem, travel_api, data_sql) |
| `--allow-shell` | adds the `shell` toolkit (not a sandbox) |
| `--system-prompt TEXT` / `@file` | replaces the generic system prompt (toolkit, skill and MCP sections are still appended) |
| `--max-steps N` | model turns allowed per user message (default 12) |
| `--think`, `--show-reasoning` | hybrid-thinking models: enable thinking / print the reasoning |

The repo's example skills and MCP servers are registered unless `--no-example-resources` is given, so a bare
`chat` offers them all; the MCP examples need `pip install -e ".[mcp-examples]"`.

## In the browser

Add `--ui` to any of the commands above and open the printed address (default http://127.0.0.1:7860; `--port`,
`--host`):

```bash
agentic-harness chat --ui --provider openai_compatible --base-url http://127.0.0.1:8000/v1 --model my-model \
    --toolkits examples/tools/finance_tools.py --use-mcp kanban
```

![chat UI](images/chat-ui.png)

- Messages stream in as the agent works: each tool call is a card (`running` → `result` / `error`, click to see the
  arguments and the full result; error cards open by themselves).
- The side panel has 👍/👎 and notes for the conversation, the tool calls that really ran, the conversations saved
  in this browser session, the tool list and the system prompt.
- **Save & new** saves the conversation and starts a fresh one; **Discard** drops it; **Stop** ends the agent's answer
  before its next model call. Enter sends (not while a Korean/Japanese/Chinese IME is composing), Shift+Enter adds a
  line. Sending an empty box after an error or Stop asks the model again.
- Each browser tab keeps its own conversation (session id in `localStorage`, or `?sid=name` in the URL); reloading the
  page restores it. Model calls are served one at a time.
- Stopping the server (Ctrl-C or SIGTERM) saves every open conversation.
- The page is a single HTML file served by the Python standard library; no extra packages. There is **no
  authentication**: keep the default `--host 127.0.0.1` (the agent's tools run with your permissions).

## In the terminal

```
model: openai_compatible my-model | tools: 10 (mcp:kanban: 7, toolkit:finance: 3) | saving to runs/chat/trajectories.jsonl
you> 'Upgrade database driver' 카드 담당자가 누구고 지금 어느 컬럼에 있어?
  -> mcp__kanban__list_cards {"column": "todo"}
  <- {"card_id": "CARD-101", "title": "Upgrade database driver", "column": "todo", "assignee": "david", ...}
agent> "Upgrade database driver" (CARD-101)는 david에게 할당되어 있고, 지금 todo 컬럼에 있습니다.
you> 그 카드를 in_progress로 옮겨줘.
agent> 카드는 in_progress 컬럼으로 이동되었습니다.
you> /calls
  ok  mcp__kanban__list_cards {"column": "todo"}
you> /bad claimed the move without calling move_card
```

| command | |
|---|---|
| `/good [note]`, `/bad [note]` | rate the conversation (stored in `human_feedback`) |
| `/note TEXT` | attach a note |
| `/calls` | tool calls that really ran in this conversation (useful to catch claimed-but-not-done actions) |
| `/tools`, `/system` | what the agent sees |
| `/new` | save this conversation and start a fresh one (fresh toolkit state and MCP server processes) |
| `/discard` | drop the conversation |
| `/retry` | ask the model again after a backend error or Ctrl-C |
| `/quit`, Ctrl-D | save and exit |

Ctrl-C while the agent is working stops the current answer; the conversation is kept.

## What is saved

`<out>/trajectories.jsonl` (default `runs/chat/`) gets one line per conversation in the normal
[trajectory format](trajectories.md), with `category: "chat"`, `outcome.status: "unscored"` (there is no verifier) and:

```json
"human_feedback": {"rating": "good", "notes": ["correct and concise"], "rated_by": "human"},
"chat": {"n_user_messages": 2, "started_at": "...", "ended_at": "...", "final_status": "answered"}
```

`<out>/tasks.jsonl` gets the conversation's user turns as a task (first message = `instruction`, later ones =
`followups`) with the same toolkits / skills / MCP servers. Add a `verify` block to make it a scored task, or replay the
same requests with other models:

```bash
agentic-harness view runs/chat/trajectories.jsonl --limit 20                  # read the conversations
agentic-harness run --tasks runs/chat/tasks.jsonl --provider vllm --model other-model --out runs/chat_replay \
    --toolkits examples/tools/finance_tools.py
agentic-harness export sft --traj runs/chat/trajectories.jsonl --out chat_sft.jsonl   # conversations rated /good
```

`export sft` treats a `good` rating as success and always drops `bad` ones; unrated chats are not exported.

## Python

```python
from agentic_harness import ChatSession

chat = ChatSession(model={"provider": "openai_compatible", "base_url": "http://127.0.0.1:8000/v1", "model": "m"},
                   toolkit_files=["examples/tools/finance_tools.py"], out_dir="runs/chat")
for m in chat.send("Monthly payment on $300,000 at 4.5% for 30 years?"):
    print(m["role"], m.get("content") or m.get("tool_calls"))
chat.send("And over 15 years?")
chat.rate("good")
traj = chat.save()          # returns the trajectory dict and appends it to runs/chat/
chat.close()
```

`on_message=callback` streams each new message (assistant turns, tool results) as it is produced, which is how the CLI
prints them and how the browser UI streams them (`agentic_harness/web/server.py`, a small JSON / NDJSON API you can
also call from your own front end). `request_stop()` ends the current answer before the next model call.
