"""Policy interface + shared helpers (raw-text parsing, client-side prompt tracking)."""
from __future__ import annotations

import hashlib

from ..core.toolcall import parse_assistant
from ..core.types import Turn


class Policy:
    name = "base"

    def info(self) -> dict:
        return {"name": self.name}

    def generate_batch(self, episodes) -> list[Turn]:
        raise NotImplementedError


def turn_from_raw(raw_text: str, tools: list[dict], *, prompt_opened_think: bool, finish_reason: str | None,
                  call_offset: int = 0, **extra) -> Turn:
    pr = parse_assistant(raw_text, tools, prompt_opened_think=prompt_opened_think)
    calls = [{"id": f"call_{call_offset + i}", "name": c.name, "arguments": c.arguments} for i, c in enumerate(pr.tool_calls)]
    return Turn(raw_text=raw_text, content=pr.content, reasoning=pr.reasoning, tool_calls=calls, parse_errors=pr.errors,
                finish_reason=finish_reason, think_unclosed=pr.think_unclosed, **extra)


def template_messages(episode) -> list[dict]:
    """Messages for client-side rendering: assistant turns carry their *raw generated text* so that
    re-rendering reproduces the exact prefix the model saw (token-in/out)."""
    out, ti = [], 0
    for m in episode.messages:
        if m["role"] == "assistant":
            raw = episode.turns[ti].raw_text if ti < len(episode.turns) else m.get("content", "")
            ti += 1
            out.append({"role": "assistant", "content": _strip_stop(raw)})
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _strip_stop(s: str) -> str:
    for stop in ("<|im_end|>", "<|endoftext|>"):
        if s.endswith(stop):
            s = s[: -len(stop)]
    return s


class PromptTracker:
    """Per-episode client-side renderer that splices token ids across turns and flags prefix breaks."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.text: str | None = None
        self.ids: list[int] | None = None
        self.last_completion_text = ""
        self.last_completion_ids: list[int] = []

    def build(self, episode) -> tuple[list[int], bool | None]:
        full = self.tok.apply_chat_template(template_messages(episode), tools=episode.tools, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=episode.enable_thinking)
        mismatch = None
        if self.text is not None:
            expect = self.text + self.last_completion_text
            if full.startswith(expect):
                ids = self.ids + self.last_completion_ids + self.tok.encode(full[len(expect):], add_special_tokens=False)
                mismatch = False
            else:
                ids = self.tok.encode(full, add_special_tokens=False)
                mismatch = True
        else:
            ids = self.tok.encode(full, add_special_tokens=False)
        self.text, self.ids = full, ids
        return ids, mismatch

    def record(self, completion_ids: list[int], completion_text: str) -> None:
        self.last_completion_ids = list(completion_ids)
        self.last_completion_text = completion_text


def file_sha256(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return "sha256:" + hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None
