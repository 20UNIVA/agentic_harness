# Changelog

## 0.2.0
- Browser chat UI (`agentic-harness chat --ui`): stdlib server + one HTML page; streamed tool cards, ratings / notes,
  executed-call list, save / discard / stop, per-tab sessions; open conversations are saved on shutdown.
- Tool messages in trajectories carry `is_error`.
- The `scripts/*.py` wrappers are gone; use the `agentic-harness` command (`scripts/serve_vllm.sh` remains).
- Interactive chat (`agentic-harness chat`, `agentic_harness.ChatSession`): a person types the user turns; tool calls
  are shown live (`/calls` lists what really ran), conversations are saved as trajectories with `human_feedback`
  ratings (used by `export sft`) and as replayable tasks (`tasks.jsonl`).
- Skills: SKILL.md registry, progressive disclosure (`load_skill`, `read_skill_file`), inline/off modes, content hashes.
- External MCP servers: stdlib client (stdio + Streamable HTTP), `mcpServers` config with scope / filters / placeholders,
  per-episode server processes, `mcp_call` checks, `mcp_setup`; three example servers (kanban, wiki, memory).
- User tools: `@tool` with schema inference, stateful `Toolkit` classes, loading from files/modules; built-in opt-in `shell` toolkit.
- User task format with answers / regex / state / toolkit / MCP checks, forbidden actions, Python verifiers, `unscored` outcome.
- Model providers: provider-neutral OpenAI-compatible client, vLLM presets, Anthropic (official SDK), custom Python policies,
  `serve-hf` OpenAI-compatible server for local models.
- `agentic-harness` CLI (init, run, gen-tasks, export, resources, view, analyze, serve-hf) with JSON/YAML run configs,
  and a Python API (`agentic_harness.run_tasks`, `summarize`).
- Safety: stdio MCP servers inherit only a minimal environment (opt in with `inherit_env`); tool results are capped
  (`--max-tool-result-chars`); the shell toolkit is opt-in.
- Works as a regular (non-editable) install: example skills / MCP servers are found next to the package or in
  `./examples`, example configs name their `skills_dir` / `mcp_config`, and `servers.json` uses `${CONFIG_DIR}`.
- MCP startup failures include the server's last stderr lines; `gen-tasks` checks the MCP example prerequisites;
  the test suite runs without the optional `mcp` SDK (MCP tests are skipped).
- Answer checks accept common renderings of ISO dates (`2026년 7월 13일`, `July 13, 2026`, `2026/7/13`).
- Benchmark v0.2: 80 templates (skills, MCP, cross env+MCP), leakage-free splits (test 800 / dev 320 / train 11,706).

## 0.1.0
- Episode loop, five deterministic environments, 59 task templates (en/ko), validation gates, metrics (pass@k, pass^k),
  HF / vLLM backends with token tracking, SFT / GRPO / DPO export, ms-swift GRPO plugin.
