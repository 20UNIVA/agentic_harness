# From trajectories to training

## SFT (rejection sampling)

```bash
agentic-harness run --config my_teacher.json --tasks data/tasks_v0.2/train.jsonl --n-attempts 4 --out runs/teacher
agentic-harness export sft --traj runs/teacher/trajectories.jsonl --out data/sft.jsonl --max-per-task 2 [--keep-reasoning]
```

Selection keeps successful episodes (`reward == 1`) without tool-call parse errors (`--allow-parse-errors N` to relax),
at most `--max-per-task` per task, optionally excluding categories. Human ratings from `agentic-harness chat` override
the verifier: `good` conversations are kept, `bad` ones are always dropped ([chat.md](chat.md)). Output rows use the ms-swift agent format:

```json
{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", "content": "text before the call"},
              {"role": "tool_call", "content": "{\"name\": ..., \"arguments\": {...}}"},
              {"role": "tool_response", "content": "..."}, ..., {"role": "assistant", "content": "final answer"}],
 "tools": "<JSON string of the tool schemas>"}
```

`export openai` writes the same episodes as OpenAI chat messages (`tool_calls` with JSON-string arguments) for other
trainers. The test suite checks that, for Qwen3.5-format models, ms-swift's rendering of exported rows equals the
inference chat template's rendering (exception: conversations with several real user turns, where the inference
template drops earlier reasoning blocks).

The `oracle` policy produces gold demonstrations for every generated task; their natural-language parts are templated,
so they are best used to teach the tool-use procedure, combined with teacher-model rollouts for fluent responses.

## Multi-turn RL with ms-swift GRPO

```bash
agentic-harness export grpo --tasks data/tasks_v0.2/train.jsonl --out data/grpo.jsonl \
    [--toolkits my_tools.py --skills-dir my_skills --mcp-config my_servers.json --allow-shell]

swift rlhf --rlhf_type grpo --model <sft checkpoint> --template <template> --dataset data/grpo.jsonl \
    --external_plugins agentic_harness/integrations/swift_plugin.py \
    --multi_turn_scheduler agent_harness --max_turns 12 \
    --reward_funcs harness_reward [harness_format] --use_vllm true ...
```

- Each row carries the full task (`harness_task`). The scheduler rebuilds a ToolSession per rollout (env + toolkits +
  skills + fresh MCP servers), overwrites the system prompt and `tools` with the session's (so training sees exactly
  the evaluation tool list), parses each completion with the harness parser, executes tool calls, appends results as
  `tool` messages (rendered in user turns, not trained on), and scores the finished episode with the same scorer as
  evaluation. The score is passed in `rollout_infos["harness"]`.
- `harness_reward` returns that score (infrastructure failures give 0.0 and are reported in `rollout_infos`).
  `harness_format` is an optional shaping reward (-1 per parse error).
- Registries for the plugin come from environment variables (os.pathsep-separated): `AGENTIC_HARNESS_SKILLS_DIRS`,
  `AGENTIC_HARNESS_MCP_CONFIGS`, `AGENTIC_HARNESS_TOOLKITS`, `AGENTIC_HARNESS_SKILLS_MODE`,
  `AGENTIC_HARNESS_ALLOW_SHELL=1`. Without them the repo examples are used.
- Tested with ms-swift 4.5.x by driving `MultiTurnScheduler.run()` with a scripted engine (see `tests/`).

## Step-level DPO (experimental)

```bash
agentic-harness export dpo --traj runs/a/trajectories.jsonl runs/b/trajectories.jsonl --out data/dpo.jsonl
```

For tasks with both a successful and a failed episode, the pair is cut at the first assistant turn whose *action*
differs (tool name + arguments, or tool call vs. final answer); turns that differ only in wording before the same call
are treated as equal. Rows: `{"messages": prefix + chosen turn, "rejected_response": rejected turn, "tools": ...}`.
The "first differing action is the mistake" assumption does not always hold; inspect pairs before training.

## Token-level data

With `vllm_completions`, every turn stores prompt/completion token ids and logprobs, suitable for off-policy
corrections or custom trainers. `prefix_mismatch` marks turns where the conversation could not be extended
token-exactly (filter or split them).
