# Architecture and design decisions

```
 TaskSpec (frozen JSON, content hash)
     │
     ▼
 Episode loop (core/runner.py) ──────────── Policy.generate_batch(active episodes)
     │   messages, tools                      openai / vllm / vllm_completions / anthropic / hf_local / python
     ▼
 ToolSession (core/session.py)
     ├─ built-in env (envs/*)              deterministic state, schema-validated tools
     ├─ toolkits (tools.py, builtin_toolkits.py)
     ├─ skills (skills/registry.py)        load_skill / read_skill_file, or inline
     └─ MCP servers (mcp/*)                stdio / Streamable HTTP, per-episode processes
     │
     ▼
 Scorer (scoring/verify.py) ─► Outcome(reward | None, status, failure_kind, components)
     │
     ▼
 Trajectory (jsonl) ─► metrics/report · view/analyze · export (SFT, OpenAI chat, DPO) · GRPO plugin (same session + scorer)
```

## Principles

1. **One code path for evaluation, data generation and RL.** The offline runner and the ms-swift GRPO scheduler use the
   same ToolSession, tool-call parser and scorer, so an RL reward means exactly what the evaluation score means.
2. **Verify outcomes, not transcripts.** Success is decided by final state (env state hash, toolkit state, MCP server
   queried after the episode) and answer checks, so any valid sequence of actions passes.
3. **Typed outcomes.** `reward=None` for infrastructure failures (server crash, API error, verifier bug, tool code bug)
   and for unscored tasks; these never count as model failures.
4. **Deterministic, generated tasks with gates.** Tasks come from seeds; reference solutions are executed to compute
   expected results; gold must pass, do-nothing must fail, splits must not share prompts.
5. **Isolation per episode.** Fresh toolkit instances, fresh env state, fresh MCP server processes and a fresh episode
   directory for every episode (shared `scope: "run"` servers are an explicit opt-in).
6. **Provenance.** Every trajectory records the tool schemas shown, skill hashes, MCP server versions and tool-list
   hashes, toolkit versions, policy settings and harness version.
7. **No required dependencies.** The core, the MCP client and the OpenAI-compatible client use only the standard
   library; model SDKs and servers are optional extras.
8. **Opt-in danger.** Shell execution is off unless enabled; MCP servers and toolkits run only if registered.

## Decisions worth knowing

| topic | decision | why |
|---|---|---|
| tool-call parsing for raw-text backends | one parser (Qwen3.5 / Qwen3-Coder XML with schema-based type conversion, `anyOf` aware, JSON fallback) shared by `hf_local`, `vllm_completions` and the GRPO plugin | training and evaluation must parse identically |
| token-level backend | client-side chat template, token ids appended turn by turn, `prefix_mismatch` flagged instead of silent re-tokenization | exact data for RL; no hidden train/inference drift |
| Anthropic backend | official SDK, append-only history, original content blocks (thinking included) replayed unchanged, no sampling params, fallbacks off by default | API requirements for multi-turn tool use; evaluation fidelity |
| skills | progressive disclosure by default; inline and off modes for ablations; content hashes in provenance | measure skill use and its effect |
| MCP | own stdlib client, `mcpServers` config compatible with common clients, `mcp__server__tool` names, schema simplification for small models | no dependency; existing configs work |
| user tools | `@tool` with schema inference; `ToolError` for model-visible errors, other exceptions are infra failures | simple to write, bugs never look like model mistakes |
| loops and limits | `max_turns`, per-turn token limits, identical-call loop detection | small models often repeat calls until they run out of turns |
| benchmark leakage | test → dev → train generation with visible-prompt exclusion | the first version had 44% of test prompts also in train |

## Related harnesses

The design borrows from NVIDIA NeMo Gym (resources servers with `verify`, unbiased pass@k, gates) and Prime Intellect
verifiers / prime-rl (typed episodes, token-in/token-out with branch on prefix mismatch, skills as run variables),
and avoids some of their pitfalls (different parsers for train and eval, zero-score sentinels for infrastructure
errors, cloud-by-default sandboxes, unbounded waits). It is intentionally smaller: an in-process loop, JSONL
trajectories and plain Python extension points.
