"""Anthropic Messages API backend (official `anthropic` SDK, optional dependency: `pip install anthropic`).

- Tools are sent as {name, description, input_schema}; the model's `tool_use` blocks become harness tool calls and the
  results go back as `tool_result` blocks, all results of a turn in one user message.
- The conversation is rebuilt append-only every turn and assistant turns are replayed with their *original* content
  blocks (thinking blocks included, unchanged), as required for multi-turn tool use with thinking models.
- Sampling parameters (temperature/top_p/top_k) are not sent: current Claude models reject them. Depth/cost is
  controlled with `effort` (output_config.effort) instead.
- Server-side refusal fallbacks are OFF by default so that an evaluation measures exactly the configured model.
  Enable with fallbacks="default" (beta); every turn records the model that actually answered.
Credentials: ANTHROPIC_API_KEY (or any source the SDK resolves, e.g. an `ant auth login` profile).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

from ..core.toolcall import render_tool_calls
from ..core.types import Turn
from .base import Policy

DEFAULT_MODEL = "claude-opus-5-5"


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append({"name": fn["name"], "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}})
    return out


def _fallback_blocks(msg: dict) -> list[dict]:
    """Assistant turn produced by another policy (no raw blocks): rebuild text + tool_use blocks."""
    blocks = []
    if (msg.get("content") or "").strip():
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                       "input": tc["function"]["arguments"]})
    return blocks or [{"type": "text", "text": ""}]


def build_messages(episode) -> tuple[str, list[dict]]:
    """Episode messages -> (system, Anthropic messages). Deterministic, so the prefix never changes across turns."""
    system = episode.messages[0]["content"]
    raw = episode.policy_state.get("anthropic_raw", {})
    out: list[dict] = []

    def push(role, blocks):
        if out and out[-1]["role"] == role:          # merge consecutive same-role turns (e.g. tool results + follow-up)
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    ti = 0
    for m in episode.messages[1:]:
        if m["role"] == "user":
            push("user", [{"type": "text", "text": m["content"]}])
        elif m["role"] == "assistant":
            push("assistant", raw.get(ti) or _fallback_blocks(m))
            ti += 1
        elif m["role"] == "tool":
            if m.get("tool_call_id") == "parse_error":
                push("user", [{"type": "text", "text": m["content"]}])
            else:
                is_err = m.get("is_error", m["content"].startswith('{"error"'))
                push("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"],
                               **({"is_error": True} if is_err else {})}])
    return system, out


class AnthropicPolicy(Policy):
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, *, max_tokens: int = 16000, effort: str | None = None,
                 thinking: str | None = None, fallbacks: str | list | None = None, base_url: str | None = None,
                 api_key: str | None = None, concurrency: int = 8, timeout: float = 600.0, max_retries: int = 2,
                 extra_body: dict | None = None):
        import anthropic  # optional dependency
        self.anthropic = anthropic
        kw = {"timeout": timeout, "max_retries": max_retries}
        if base_url:
            kw["base_url"] = base_url
        if api_key:
            kw["api_key"] = api_key
        self.client = anthropic.Anthropic(**kw)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking
        self.fallbacks = fallbacks
        self.extra_body = dict(extra_body or {})
        self.pool = ThreadPoolExecutor(concurrency)

    def info(self):
        return {"name": self.name, "model": self.model, "max_tokens": self.max_tokens, "effort": self.effort,
                "thinking": self.thinking, "fallbacks": self.fallbacks, "sdk_version": self.anthropic.__version__}

    def generate_batch(self, episodes):
        return list(self.pool.map(self._safe_one, episodes))

    def _safe_one(self, ep):
        try:
            return self._one(ep)
        except Exception as e:    # API/network errors -> infra_fail (reward=None), never a policy failure
            return Turn(error=f"{type(e).__name__}: {str(e)[:400]}")

    def _one(self, ep):
        system, messages = build_messages(ep)
        req = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": messages}
        if ep.tools:
            req["tools"] = _to_anthropic_tools(ep.tools)
            req["tool_choice"] = {"type": "auto"}
        if self.effort:
            req["output_config"] = {"effort": self.effort}
        if self.thinking:
            req["thinking"] = {"type": self.thinking}
        if self.extra_body:
            req["extra_body"] = self.extra_body
        t0 = time.time()
        if self.fallbacks:
            beta = "server-side-fallback-2026-07-01" if self.fallbacks == "default" else "server-side-fallback-2026-06-01"
            resp = self.client.beta.messages.create(**req, betas=[beta], fallbacks=self.fallbacks)
        else:
            resp = self.client.messages.create(**req)
        blocks = [b.model_dump(exclude_none=True) for b in resp.content]
        ep.policy_state.setdefault("anthropic_raw", {})[len(ep.turns)] = blocks
        text = "".join(b.get("text", "") for b in blocks if b["type"] == "text").strip()
        thinking = "\n".join(b.get("thinking", "") for b in blocks if b["type"] == "thinking").strip() or None
        calls = [{"id": b["id"], "name": b["name"], "arguments": b.get("input") or {}} for b in blocks if b["type"] == "tool_use"]
        finish = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop",
                  "refusal": "refusal"}.get(resp.stop_reason, resp.stop_reason)
        usage = resp.usage.model_dump(exclude_none=True) if getattr(resp, "usage", None) else None
        return Turn(raw_text=render_tool_calls([{"name": c["name"], "arguments": c["arguments"]} for c in calls], text),
                    content=text, reasoning=thinking, tool_calls=calls, parse_errors=[], finish_reason=finish,
                    latency_s=round(time.time() - t0, 3),
                    provider_meta={"model": getattr(resp, "model", self.model), "stop_reason": resp.stop_reason,
                                   "usage": usage, "id": getattr(resp, "id", None),
                                   "stop_details": json.loads(json.dumps(getattr(resp, "stop_details", None), default=str))})
