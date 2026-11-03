"""Model backends ("policies") and a factory that builds one from a config dict.

    {"provider": "openai",            "model": "gpt-...",  "api_key_env": "OPENAI_API_KEY"}
    {"provider": "openai_compatible", "model": "my-model", "base_url": "http://host:8000/v1", "api_key_env": "..."}
    {"provider": "vllm",              "model": "served-name", "base_url": "http://127.0.0.1:8000"}
    {"provider": "vllm_completions",  "model": "served-name", "base_url": "...", "tokenizer": "/path/to/hf/model"}
    {"provider": "anthropic",         "model": "claude-opus-5-5", "effort": "medium"}
    {"provider": "hf_local",          "model": "/path/to/hf/model", "device": "cpu", "threads": 16}
    {"provider": "python",            "class": "my_pkg.my_policy:MyPolicy", "kwargs": {...}}   # or "file.py:MyPolicy"
    {"provider": "oracle" | "null" | "noisy_oracle"}                                           # testing

Common optional keys: "sampling": {"temperature", "top_p", "top_k", "max_new_tokens"}, "extra_body", "concurrency",
"timeout", "headers". A custom policy subclasses `agentic_harness.policies.base.Policy` and implements
`generate_batch(episodes) -> list[Turn]`.
"""
from __future__ import annotations

import importlib
import importlib.util
import os

from .base import Policy  # noqa: F401
from .oracle import NoisyOraclePolicy, NullPolicy, OraclePolicy

PROVIDERS = ("openai", "openai_compatible", "openai_chat", "vllm", "vllm_completions", "openai_completions",
             "anthropic", "hf_local", "python", "oracle", "null", "noisy_oracle")


def _load_attr(ref: str):
    mod_ref, attr = ref.rsplit(":", 1)
    if mod_ref.endswith(".py"):
        spec = importlib.util.spec_from_file_location("ah_policy_" + str(abs(hash(mod_ref))), os.path.abspath(mod_ref))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_ref)
    return getattr(mod, attr)


def make_policy(cfg: dict) -> Policy:
    cfg = dict(cfg)
    p = cfg.pop("provider", None)
    if p not in PROVIDERS:
        raise ValueError(f"unknown provider {p!r}; choose one of {PROVIDERS}")
    samp = dict(cfg.pop("sampling", {}) or {})
    common = {k: cfg[k] for k in ("concurrency", "timeout", "headers", "extra_body", "api_key", "api_key_env",
                                  "max_tokens_field") if k in cfg}
    if p in ("oracle", "null"):
        return OraclePolicy() if p == "oracle" else NullPolicy()
    if p == "noisy_oracle":
        return NoisyOraclePolicy(p=cfg.get("p", 0.3), seed=cfg.get("seed", 0))
    if p in ("openai", "openai_compatible", "openai_chat", "vllm"):
        from .openai_compat import OpenAIChatPolicy
        if p == "openai":
            cfg.setdefault("base_url", "https://api.openai.com/v1")
            common.setdefault("api_key_env", "OPENAI_API_KEY")
        if p == "vllm":
            cfg.setdefault("base_url", "http://127.0.0.1:8000")
        if "base_url" not in cfg:
            raise ValueError(f"provider {p!r} needs base_url")
        return OpenAIChatPolicy(cfg["base_url"], cfg["model"], send_thinking_flag=(p == "vllm"), **samp, **common)
    if p in ("vllm_completions", "openai_completions"):
        from .openai_compat import OpenAICompletionsPolicy
        return OpenAICompletionsPolicy(cfg.get("base_url", "http://127.0.0.1:8000"), cfg["model"],
                                       tokenizer_path=cfg.get("tokenizer") or cfg["model"], **samp, **common)
    if p == "anthropic":
        from .anthropic import DEFAULT_MODEL, AnthropicPolicy
        kw = {k: cfg[k] for k in ("max_tokens", "effort", "thinking", "fallbacks", "base_url", "api_key", "concurrency",
                                  "timeout", "max_retries", "extra_body") if k in cfg}
        if "api_key_env" in cfg and "api_key" not in kw:
            kw["api_key"] = os.environ.get(cfg["api_key_env"])
        return AnthropicPolicy(cfg.get("model", DEFAULT_MODEL), **kw)
    if p == "hf_local":
        from .hf_local import HFLocalPolicy
        kw = {k: cfg[k] for k in ("device", "threads", "seed") if k in cfg}
        return HFLocalPolicy(cfg["model"], **kw, **samp)
    if p == "python":
        cls = _load_attr(cfg["class"])
        return cls(**(cfg.get("kwargs") or {}))
    raise AssertionError(p)
