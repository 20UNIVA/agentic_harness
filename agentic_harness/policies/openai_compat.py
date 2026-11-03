"""OpenAI-compatible HTTP backends (stdlib only, urllib).

- OpenAIChatPolicy: /v1/chat/completions with `tools`; tool calls are parsed by the server. Works with OpenAI, vLLM
  (--enable-auto-tool-choice --tool-call-parser ...), SGLang, Ollama, LM Studio, llama.cpp server, Together, etc.
  Only standard fields are sent by default; server-specific options go in `extra_body`
  (the "vllm" preset in policies/__init__.py adds top_k and chat_template_kwargs.enable_thinking).
- OpenAICompletionsPolicy: client-side chat-template rendering + token-id prompt to /v1/completions (vLLM), returning
  completion token ids and logprobs (token-in/token-out, needed for exact RL data). Uses the harness tool-call parser.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ..core.toolcall import render_tool_calls
from ..core.types import Turn
from .base import Policy, PromptTracker, turn_from_raw


class _HTTPBase(Policy):
    def __init__(self, base_url: str, model: str, *, api_key: str | None = None, api_key_env: str | None = None,
                 temperature: float | None = None, top_p: float | None = None, top_k: int | None = None,
                 max_new_tokens: int | None = 1024, max_tokens_field: str = "max_tokens", extra_body: dict | None = None,
                 headers: dict | None = None, concurrency: int = 16, timeout: float = 600, retries: int = 2):
        base_url = base_url.rstrip("/")
        self.base_url = base_url[:-3] if base_url.endswith("/v1") else base_url
        self.model = model
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None) or "EMPTY"
        self.sampling = {k: v for k, v in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k)) if v is not None}
        if max_new_tokens is not None:
            self.sampling[max_tokens_field] = max_new_tokens
        self.extra_body = dict(extra_body or {})
        self.headers = dict(headers or {})
        self.pool = ThreadPoolExecutor(concurrency)
        self.timeout = timeout
        self.retries = retries

    def info(self):
        return {"name": self.name, "base_url": self.base_url, "model": self.model, "sampling": self.sampling,
                "extra_body": self.extra_body}

    def _post(self, path: str, payload: dict) -> dict:
        err = None
        for i in range(self.retries + 1):
            req = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode(), method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.api_key}", **self.headers})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                body = e.read()[:500].decode("utf-8", "replace")
                err = RuntimeError(f"HTTP {e.code}: {body}")
                if e.code < 500 and e.code not in (408, 409, 429):
                    break                                 # client errors are not retried
            except Exception as e:
                err = e
            time.sleep(min(2 ** i, 10))
        raise RuntimeError(f"{type(err).__name__}: {err}")

    def generate_batch(self, episodes):
        return list(self.pool.map(self._safe_one, episodes))

    def _safe_one(self, ep):
        try:
            return self._one(ep)
        except Exception as e:     # transport/server errors -> infra_fail (reward=None), never a 0 score
            return Turn(error=str(e)[:500])


class OpenAICompletionsPolicy(_HTTPBase):
    name = "openai_completions"

    def __init__(self, base_url, model, *, tokenizer_path: str, **kw):
        super().__init__(base_url, model, **kw)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        self.stop_ids = [i for i in (self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>"))
                         if isinstance(i, int) and i >= 0]

    def _one(self, ep):
        tr = ep.policy_state.setdefault("tracker", PromptTracker(self.tok))
        ids, mm = tr.build(ep)
        t0 = time.time()
        resp = self._post("/v1/completions", {
            "model": self.model, "prompt": ids, "logprobs": 1, "return_token_ids": True, "skip_special_tokens": False,
            "stop_token_ids": self.stop_ids, "include_stop_str_in_output": True, **self.sampling, **self.extra_body})
        ch = resp["choices"][0]
        gen = ch.get("token_ids")
        lps = (ch.get("logprobs") or {}).get("token_logprobs")
        text = ch["text"]
        if gen is None:     # server without return_token_ids: re-tokenize (flagged; not RL-grade)
            gen = self.tok.encode(text, add_special_tokens=False)
            mm = True
        if gen and gen[-1] in self.stop_ids and not text.endswith(("<|im_end|>", "<|endoftext|>")):
            text += self.tok.decode([gen[-1]], skip_special_tokens=False)
        tr.record(gen, text)
        n = sum(len(t.tool_calls) for t in ep.turns)
        return turn_from_raw(text, ep.tools, prompt_opened_think=ep.enable_thinking, finish_reason=ch.get("finish_reason"),
                             call_offset=n, prompt_token_ids=ids, completion_token_ids=gen, completion_logprobs=lps,
                             prefix_mismatch=mm, latency_s=round(time.time() - t0, 3),
                             provider_meta={"model": resp.get("model"), "usage": resp.get("usage")})


def to_openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "assistant":
            mm = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                mm["tool_calls"] = [{"id": tc["id"], "type": "function",
                                     "function": {"name": tc["function"]["name"],
                                                  "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                                    for tc in m["tool_calls"]]
            out.append(mm)
        elif m["role"] == "tool":
            if m.get("tool_call_id") == "parse_error":       # orphan feedback -> user message for strict APIs
                out.append({"role": "user", "content": m["content"]})
            else:
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class OpenAIChatPolicy(_HTTPBase):
    name = "openai_chat"

    def __init__(self, base_url, model, *, send_thinking_flag: bool = False, **kw):
        """send_thinking_flag: also send chat_template_kwargs.enable_thinking (vLLM/SGLang hybrid-thinking models)."""
        super().__init__(base_url, model, **kw)
        self.send_thinking_flag = send_thinking_flag

    def _one(self, ep):
        payload = {"model": self.model, "messages": to_openai_messages(ep.messages), **self.sampling}
        if ep.tools:
            payload["tools"] = ep.tools
            payload["tool_choice"] = "auto"
        body = dict(self.extra_body)
        if self.send_thinking_flag:
            body.setdefault("chat_template_kwargs", {})
            body["chat_template_kwargs"] = {**body["chat_template_kwargs"], "enable_thinking": ep.enable_thinking}
        payload.update(body)
        t0 = time.time()
        resp = self._post("/v1/chat/completions", payload)
        ch = resp["choices"][0]
        msg = ch.get("message") or {}
        calls, errors = [], []
        n = sum(len(t.tool_calls) for t in ep.turns)
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            try:
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                errors.append("unparseable_tool_call")
                continue
            calls.append({"id": tc.get("id") or f"call_{n + i}", "name": fn.get("name", ""), "arguments": args})
        content = msg.get("content") or ""
        if not calls and "<tool_call>" in content:
            errors.append("unparseable_tool_call")        # the server's tool parser failed to extract a call
        raw = render_tool_calls([{"name": c["name"], "arguments": c["arguments"]} for c in calls], content)
        return Turn(raw_text=raw, content=content.strip(), reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
                    tool_calls=calls, parse_errors=errors, finish_reason=ch.get("finish_reason"),
                    latency_s=round(time.time() - t0, 3),
                    provider_meta={"model": resp.get("model"), "usage": resp.get("usage"), "id": resp.get("id")})
