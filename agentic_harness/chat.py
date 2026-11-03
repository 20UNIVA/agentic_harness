"""Interactive chat with an agent: a person types the user turns, the model answers with tools / skills / MCP servers,
and every conversation is saved as one trajectory (same schema as `agentic-harness run`).

    from agentic_harness.chat import ChatSession

    chat = ChatSession(model={"provider": "openai_compatible", "base_url": "http://127.0.0.1:8000/v1", "model": "m"},
                       toolkit_files=["examples/tools/finance_tools.py"], out_dir="runs/chat")
    for msg in chat.send("What is the monthly payment on $300k at 4.5% for 30 years?"):
        print(msg["role"], msg.get("content"))
    chat.rate("good", "correct and concise")
    chat.save()          # -> runs/chat/trajectories.jsonl (+ the user turns as a replayable task in tasks.jsonl)
    chat.close()

Each saved conversation is `unscored` (no verifier); the person's rating is stored in `human_feedback` and is used by
`agentic-harness export sft` (rating "good" counts as success, "bad" is excluded).
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Callable

from .core.runner import Episode
from .core.session import Resources, build_resources
from .core.taskspec import normalize_task
from .policies import make_policy
from .policies.base import Policy

RATINGS = ("good", "bad")
_HANGUL = re.compile(r"[가-힣]")


class ChatModelError(RuntimeError):
    """The model backend failed (connection refused, HTTP error, ...). The conversation is kept; call retry()."""


class ChatSession:
    def __init__(self, model: dict | Policy, *, resources: Resources | None = None, toolkit_files=(), skills_dirs=(),
                 mcp_configs=(), allow_shell: bool = False, skills_mode: str = "tool", toolkits: list[str] | None = None,
                 skills: list[str] | None = None, mcp_servers: list[str] | None = None, env: str | dict | None = None,
                 system_prompt: str | None = None, enable_thinking: bool = False, max_steps: int = 12,
                 out_dir: str | None = None, on_message: Callable[[dict], None] | None = None):
        """toolkits / skills / mcp_servers: names to offer (None = everything registered). env: built-in environment
        ("retail" or {"domain": "retail", "seed": 3}). max_steps: model turns allowed per user message."""
        self.policy = model if isinstance(model, Policy) else make_policy(model)
        self._own_res = resources is None
        self.res = resources or build_resources(skills_dirs=list(skills_dirs) or None, mcp_configs=list(mcp_configs) or None,
                                                toolkit_specs=list(toolkit_files) or None, skills_mode=skills_mode,
                                                allow_shell=allow_shell)
        if toolkits is None:
            toolkits = sorted(self.res.toolkits) + (["shell"] if self.res.allow_shell else [])
        if skills is None:
            skills = self.res.skills.names() if self.res.skills is not None and self.res.skills_mode != "off" else []
        if mcp_servers is None:
            mcp_servers = self.res.mcp.names() if self.res.mcp is not None else []
        self.spec = {"toolkits": list(toolkits), "skills": list(skills), "mcp_servers": list(mcp_servers),
                     "max_turns": int(max_steps), "category": "chat"}
        if env:
            self.spec["env"] = env
        if system_prompt:
            self.spec["system_prompt"] = system_prompt
        self.enable_thinking = enable_thinking
        self.max_steps = int(max_steps)
        self.out_dir = out_dir
        self.on_message = on_message
        self.ep: Episode | None = None
        self.user_messages: list[str] = []
        self.feedback: dict = {}
        self.started_at: str | None = None
        self._task_id = ""
        self._stop_requested = False

    # ------------------------------------------------------------------ conversation state
    def _episode(self) -> Episode:
        if self.ep is None:
            task = normalize_task({**self.spec, "id": "chat", "instruction": "(chat)"})
            self.ep = Episode(task, enable_thinking=self.enable_thinking, resources=self.res)
            if self.ep.infra_error:
                err, self.ep = self.ep.infra_error, None
                raise RuntimeError(f"could not start the tool session: {err}")
            self.user_messages, self.feedback = [], {}
            self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._task_id = f"chat-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        return self.ep

    @property
    def tools(self) -> list[dict]:
        return self._episode().tools

    @property
    def system_prompt(self) -> str:
        return self._episode().messages[0]["content"]

    @property
    def messages(self) -> list[dict]:
        return self.ep.messages if self.ep and self.user_messages else []

    @property
    def status(self) -> str:
        """'idle' (no message yet), 'awaiting_user', 'running', or why the agent stopped
        (max_turns, loop_abort, truncated, interrupted, infra_fail)."""
        ep = self.ep
        if ep is None or not self.user_messages:
            return "idle"
        if ep.infra_error:
            return "infra_fail"
        if not ep.done:
            return "running"
        return "awaiting_user" if ep.ended_normally else ep.stats.get("termination", "stopped")

    # ------------------------------------------------------------------ turns
    def send(self, text: str) -> list[dict]:
        """Add a user message and let the agent work until it answers (or stops). Returns the new messages."""
        ep = self._episode()
        if ep.infra_error:
            raise RuntimeError("this conversation ended with an infrastructure error; save() or discard() it first")
        start = len(ep.messages)
        if not self.user_messages:
            ep.messages[1]["content"] = text
            ep.max_turns = self.max_steps
            ep._ev("user.message", content=text)
            start = 2
        else:
            ep.add_user_message(text, self.max_steps)
        self.user_messages.append(text)
        return self._run(start)

    def retry(self) -> list[dict]:
        """Ask the model again after a ChatModelError (or an interrupted generation)."""
        ep = self._episode()
        if ep.done and ep.stats.get("termination") == "interrupted":
            ep.done = False
            ep.stats.pop("termination", None)
        return self._run(len(ep.messages))

    def request_stop(self) -> None:
        """Stop the agent before its next model turn (thread-safe; used by the web UI's Stop button)."""
        self._stop_requested = True

    def _run(self, start: int) -> list[dict]:
        ep = self.ep
        self._stop_requested = False
        try:
            while not ep.done:
                if self._stop_requested:
                    raise KeyboardInterrupt
                turn = self.policy.generate_batch([ep])[0]
                if turn.error:                        # keep the conversation; the person can retry or save
                    raise ChatModelError(turn.error)
                n = len(ep.messages)
                ep.consume(turn)
                if self.on_message:
                    for m in ep.messages[n:]:
                        self.on_message(m)
        except KeyboardInterrupt:
            ep._stop("interrupted")
            ep._ev("user.interrupt")
        return ep.messages[start:]

    # ------------------------------------------------------------------ feedback / saving
    def rate(self, rating: str | None, note: str | None = None) -> None:
        if rating not in RATINGS + (None,):
            raise ValueError(f"rating must be one of {RATINGS} or None")
        self.feedback["rating"] = rating
        if note:
            self.note(note)

    def note(self, text: str) -> None:
        self.feedback.setdefault("notes", []).append(text)

    def task_spec(self) -> dict:
        """The conversation's user turns as a task in the user task format (replay it with `agentic-harness run`)."""
        spec = {k: v for k, v in self.spec.items() if k != "max_turns"}
        spec.update(id=self._task_id, instruction=self.user_messages[0], followups=self.user_messages[1:],
                    lang="ko" if _HANGUL.search(self.user_messages[0]) else "en", max_turns=self.max_steps)
        return spec

    def save(self) -> dict | None:
        """Finish the current conversation and return its trajectory (written to out_dir if set). Empty conversations
        are discarded. The next send() starts a new conversation."""
        ep = self.ep
        if ep is None:
            return None
        if not self.user_messages:
            self.discard()
            return None
        spec = self.task_spec()
        task = normalize_task(spec)
        task["limits"]["max_turns"] = ep.max_turns
        ep.task = task
        if not ep.done:
            ep._stop("interrupted")
        traj = ep.trajectory(self.policy.info())
        traj["human_feedback"] = {"rating": self.feedback.get("rating"), "notes": self.feedback.get("notes", []),
                                  "rated_by": "human" if self.feedback else None}
        traj["chat"] = {"n_user_messages": len(self.user_messages), "started_at": self.started_at,
                        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "final_status": self._final_status(ep)}
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(os.path.join(self.out_dir, "trajectories.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            with open(os.path.join(self.out_dir, "tasks.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(spec, ensure_ascii=False) + "\n")
        self.ep, self.user_messages, self.feedback = None, [], {}
        return traj

    @staticmethod
    def _final_status(ep: Episode) -> str:
        if ep.infra_error:
            return "infra_fail"
        return "answered" if ep.ended_normally else ep.stats.get("termination", "stopped")

    def discard(self) -> None:
        """Drop the current conversation without saving it."""
        if self.ep is not None and self.ep.session is not None:
            self.ep.session.close()
        self.ep, self.user_messages, self.feedback = None, [], {}

    def close(self) -> None:
        """Discard the current (unsaved) conversation and release resources. Call save() first to keep it."""
        self.discard()
        if self._own_res:
            self.res.close()
