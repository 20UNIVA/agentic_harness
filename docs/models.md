# Models (policies)

A policy turns the current episode (messages + tool schemas) into the next assistant turn. All policies share the
same episode loop, tools and scorer, so results are comparable across backends.

Choose one with `--provider` flags or a `"model"` block in a run config:

```json
{"model": {"provider": "vllm", "model": "my-model", "base_url": "http://127.0.0.1:8000",
           "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 2048}, "concurrency": 64}}
```

| provider | required | optional | tool calls parsed by |
|---|---|---|---|
| `openai` | `model` | `api_key_env` (default `OPENAI_API_KEY`), `base_url`, `sampling`, `extra_body`, `max_tokens_field` | the API |
| `openai_compatible` | `model`, `base_url` | `api_key_env`, `headers`, `sampling`, `extra_body` | the server |
| `vllm` | `model` | `base_url` (default `http://127.0.0.1:8000`), `sampling` (incl. `top_k`) | vLLM `--tool-call-parser` |
| `vllm_completions` | `model`, `tokenizer` | `base_url`, `sampling` | the harness (client-side chat template) |
| `anthropic` | - | `model` (default `claude-opus-5-5`), `effort`, `thinking`, `max_tokens`, `fallbacks`, `base_url`, `api_key` / `api_key_env` | the API |
| `hf_local` | `model` (HF path) | `device` (`cpu`, `cuda:0`, `mps`), `threads`, `sampling` | the harness |
| `python` | `class` (`"file.py:Class"` or `"pkg.mod:Class"`) | `kwargs` | your code |
| `oracle`, `null`, `noisy_oracle` | - | `p`, `seed` (noisy) | - |

`sampling` keys: `temperature`, `top_p`, `top_k`, `max_new_tokens`. Only keys you set are sent. Common keys:
`concurrency` (parallel requests), `timeout`. Errors from the backend (HTTP errors, timeouts) make the episode an
infrastructure failure (`reward=None`); client errors (4xx except 408/409/429) are not retried.

## OpenAI-compatible servers

Works with OpenAI, vLLM, SGLang, Ollama, LM Studio, llama.cpp server and hosted OpenAI-compatible APIs.
Base URLs with or without the trailing `/v1` are accepted. Only standard fields are sent by default; put
server-specific options in `extra_body` (e.g. `{"top_k": 20, "repetition_penalty": 1.05}`). The `vllm` preset also sends
`chat_template_kwargs.enable_thinking` from `--think`, for hybrid-thinking models.

Serving a model with vLLM: `scripts/serve_vllm.sh` (pick the `TOOL_PARSER` that matches the model's chat template).

## Token-level data for RL (`vllm_completions`)

The harness renders the chat template itself (with the model's tokenizer), sends token ids to `/v1/completions`, and
keeps every turn's prompt and completion token ids and logprobs in the trajectory. Assistant turns are appended as the
exact generated tokens; if re-rendering the conversation does not reproduce the previous prompt as a prefix (e.g. the
template drops earlier reasoning), the turn is flagged `prefix_mismatch` instead of being silently re-tokenized.
Tool calls are parsed by the harness parser (Qwen3.5 / Qwen3-Coder XML; JSON-style calls are accepted as a fallback),
which is also what the ms-swift GRPO plugin uses, so RL and evaluation parse identically.

## Anthropic

Uses the official `anthropic` SDK (`pip install anthropic`); credentials come from `ANTHROPIC_API_KEY` or any source
the SDK resolves. Tools are sent with `input_schema`; `tool_use` blocks become tool calls and results go back as
`tool_result` blocks (all results of a turn in one user message, `is_error` on failures). Assistant turns are replayed
with their original content blocks, thinking blocks included and unchanged, and the history is append-only.
Sampling parameters are not sent (current Claude models reject them); use `effort` (`low` ... `max`).
Server-side refusal fallbacks are off by default so that an evaluation measures the configured model; enable them with
`"fallbacks": "default"`, and check `provider_meta.model` in the trajectory to see which model answered each turn.

## Local Hugging Face models

`hf_local` loads the model with transformers and batches all active episodes into one `generate` call per turn.
It renders the model's chat template and parses tool calls with the harness parser, so it suits models whose template
uses the Qwen3.5 / Qwen3-Coder XML tool format. On CPU, Triton-only kernels (e.g. flash-linear-attention) are disabled
and the pure-torch path is used. For other model families, serve them with vLLM/SGLang and use `openai_compatible`.

To run on CPU next to a GPU job, hide the GPUs and bound the CPU share, e.g.
`CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 0-9 agentic-harness run --provider hf_local --model <path> --device cpu --threads 10 ...`.

`agentic-harness serve-hf --model <path> --device cpu --port 8008` exposes the same generation behind an
OpenAI-compatible `/v1/chat/completions` endpoint (non-streaming, batched), for laptops or CI.

## Hybrid thinking

`--think` sets `enable_thinking=True` for chat templates that support it (`hf_local`, `vllm`, `vllm_completions`).
Reasoning is stored per turn (`turns[].reasoning`, `messages[].reasoning_content`); unclosed `<think>` blocks are
counted in the report (`think_unclosed_turn_rate`).

## Custom policies

```python
from agentic_harness.policies.base import Policy, turn_from_raw

class MyPolicy(Policy):
    name = "my_policy"

    def generate_batch(self, episodes):
        turns = []
        for ep in episodes:          # ep.messages (OpenAI-style), ep.tools (function schemas), ep.policy_state (scratch)
            text = my_model(ep.messages, ep.tools)
            turns.append(turn_from_raw(text, ep.tools, prompt_opened_think=False, finish_reason="stop"))
        return turns
```

Return `Turn(tool_calls=[{"id", "name", "arguments"}], content=..., ...)` directly if your model already produces
structured tool calls. `{"provider": "python", "class": "my_policy.py:MyPolicy", "kwargs": {...}}` loads it.
