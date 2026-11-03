# Trajectory format

`trajectories.jsonl` has one JSON object per episode (`schema_version: "agentharness.traj.v1"`).

| field | content |
|---|---|
| `episode_id`, `task_id`, `task_hash`, `attempt` | identifiers; `task_hash` is the content hash of the task definition |
| `domain`, `category`, `difficulty`, `split`, `lang` | task labels |
| `policy` | backend info (provider, model, sampling, `enable_thinking`, chat-template hash for local models) |
| `skills`, `mcp_servers`, `required_skill` | what the task asked for |
| `tools` | the exact tool schemas shown to the model (OpenAI function format) |
| `messages` | the conversation: `system`, `user`, `assistant` (`content`, `reasoning_content`, `tool_calls` with parsed `arguments`), `tool` (`tool_call_id`, `name`, `content`, `is_error`) |
| `turns` | one entry per model call (below) |
| `executed_calls` | every tool call that ran: `name`, `arguments`, `ok`, `source` (`env`, `toolkit:<name>`, `skill`, `mcp:<server>`) |
| `events` | ordered event log: `episode.started`, `model.turn`, `tool.call`, `user.followup`, `user.message` (chat), `harness.parse_feedback`, `episode.finished` |
| `outcome` | `reward` (float or null), `status`, `failure_kind`, `components` (state_match, answer_match, checks_passed, forbidden_called, leaked, action_recall, python_reward, counters, termination) |
| `human_feedback`, `chat` | only for `agentic-harness chat` conversations: the person's rating / notes, and the number of user messages, timestamps and final status ([chat.md](chat.md)) |
| `provenance` | harness version, env version, task generator, timestamp, `skills` (offered with content hashes, loaded, files read), `mcp` (server info, protocol, scope, tool-list hash), `toolkits` (class, version, tool-schema hash) |

`turns[]`:

| field | content |
|---|---|
| `raw_text` | the model output as text (for API backends: content plus rendered tool calls) |
| `content`, `reasoning`, `tool_calls`, `parse_errors`, `finish_reason`, `think_unclosed` | parsed turn |
| `prompt_token_ids`, `completion_token_ids`, `completion_logprobs` | when the backend provides them (`vllm_completions`: ids + logprobs; `hf_local`: ids) |
| `prefix_mismatch` | token-level backends: whether the re-rendered prompt failed to extend the previous one |
| `latency_s`, `provider_meta` | timing; provider data such as the answering model, usage, stop reason, request id |
| `error` | backend error text (episode becomes `infra_fail`) |

Token ids and logprobs are stored unrounded. Tool results are in `messages` (role `tool`) and are never part of the
model's completion tokens.

Useful commands:

```bash
agentic-harness view runs/x/trajectories.jsonl --failed --limit 5 --out failures.md
agentic-harness analyze runs/x/trajectories.jsonl > analysis.md
```
