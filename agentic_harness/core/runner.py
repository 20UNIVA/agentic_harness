"""Episode loop shared by data generation, evaluation and (via the swift plugin) RL."""
from __future__ import annotations

import json
import time
import uuid
from collections import deque
from typing import Callable, Iterable

from ..scoring.verify import score
from .session import Resources, ToolSession
from .types import HARNESS_VERSION, TRAJ_SCHEMA, InfraError, Outcome, Turn, canonical_json

FATAL_PARSE_ERRORS = {"unparseable_tool_call", "function_tag_without_tool_call", "tool_call_inside_unclosed_think"}


class Episode:
    def __init__(self, task: dict, *, attempt: int = 0, enable_thinking: bool = False, on_parse_error: str = "feedback",
                 loop_limit: int = 4, max_turns: int | None = None, resources: Resources | None = None):
        self.task = task
        self.attempt = attempt
        self.enable_thinking = enable_thinking
        self.on_parse_error = on_parse_error
        self.loop_limit = loop_limit
        self.max_turns = max_turns or task["limits"]["max_turns"]
        self.session = None
        self.infra_error = None
        try:
            self.session = ToolSession(task, resources)
            self.env = self.session.env
            self.tools = self.session.tool_schemas()
            system = self.session.system_prompt()
        except InfraError as e:          # e.g. an MCP server failed to start: not the policy's fault
            self.infra_error = str(e)
            self.env, self.tools, system = None, [], task["system_prompt"]
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": task["instruction"]}]
        self.followups = deque(task.get("user_followups") or [])
        self.turns: list[Turn] = []
        self.events: list[dict] = []
        self.executed: list[dict] = []
        self.stats = {"n_turns": 0, "n_tool_calls": 0, "n_parse_errors": 0, "n_tool_errors": 0, "n_repeat_calls": 0,
                      "n_user_followups": 0, "n_prefix_mismatch": 0}
        self.done = self.infra_error is not None
        self.ended_normally = False
        self.final_text = ""
        self.policy_state: dict = {}           # per-episode scratch for policies (token tracker, oracle cursor)
        self.episode_id = uuid.uuid4().hex[:16]
        self._last_sig = None
        self._repeat_run = 0
        self._ev("episode.started", task_id=task["task_id"])

    def _ev(self, typ: str, **kw):
        self.events.append({"seq": len(self.events), "t": round(time.time(), 3), "type": typ, **kw})

    def _stop(self, termination: str | None):
        self.done = True
        self.ended_normally = termination is None
        if termination:
            self.stats["termination"] = termination

    def consume(self, turn: Turn) -> None:
        self.turns.append(turn)
        self.stats["n_turns"] += 1
        if turn.prefix_mismatch:
            self.stats["n_prefix_mismatch"] += 1
        self._ev("model.turn", idx=len(self.turns) - 1, n_tool_calls=len(turn.tool_calls), parse_errors=turn.parse_errors,
                 finish_reason=turn.finish_reason)
        if turn.error:
            self.infra_error = turn.error
            self._stop("infra_fail")
            return
        msg = {"role": "assistant", "content": turn.content}
        if turn.reasoning:
            msg["reasoning_content"] = turn.reasoning
        if turn.tool_calls:
            msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                  "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in turn.tool_calls]
        self.messages.append(msg)
        self.stats["n_parse_errors"] += len([e for e in turn.parse_errors if e != "text_after_tool_call"])
        if turn.tool_calls:
            for tc in turn.tool_calls:
                try:
                    result, is_err = self.session.call(tc["name"], tc["arguments"])
                except InfraError as e:
                    self.infra_error = str(e)
                    self._stop("infra_fail")
                    return
                src = self.session.source(tc["name"])
                self.stats["n_tool_calls"] += 1
                self.stats["n_tool_errors"] += int(is_err)
                if src != "env":
                    key = {"skill": "n_skill_calls"}.get(src, "n_mcp_calls" if src.startswith("mcp") else "n_toolkit_calls")
                    self.stats[key] = self.stats.get(key, 0) + 1
                self.executed.append({"name": tc["name"], "arguments": tc["arguments"], "ok": not is_err, "source": src})
                self.messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": result,
                                      "is_error": bool(is_err)})
                self._ev("tool.call", name=tc["name"], source=src, arguments=tc["arguments"], error=is_err, result_chars=len(result))
                sig = canonical_json([tc["name"], tc["arguments"]])
                if sig == self._last_sig:
                    self._repeat_run += 1
                    self.stats["n_repeat_calls"] += 1
                else:
                    self._repeat_run = 0
                self._last_sig = sig
            if self._repeat_run >= self.loop_limit - 1:
                self._stop("loop_abort")
        elif any(e in FATAL_PARSE_ERRORS for e in turn.parse_errors):
            if self.on_parse_error == "feedback":
                self.messages.append({"role": "tool", "tool_call_id": "parse_error", "name": "harness",
                                      "content": json.dumps({"error": "could not parse your tool call (" + ", ".join(turn.parse_errors)
                                                             + "). Use the exact <tool_call><function=...><parameter=...> format."})})
                self._ev("harness.parse_feedback", errors=turn.parse_errors)
            else:
                self._stop("format_fail")
        elif turn.finish_reason == "length":
            self._stop("truncated")
        else:
            if self.followups:
                fu = self.followups.popleft()
                self.messages.append({"role": "user", "content": fu})
                self.stats["n_user_followups"] += 1
                self._ev("user.followup", content=fu)
            else:
                self.final_text = turn.content
                self._stop(None)
        if not self.done and self.stats["n_turns"] >= self.max_turns:
            self._stop("max_turns")

    def add_user_message(self, content: str, max_new_turns: int | None = None) -> None:
        """Continue a finished (or stopped) episode with a live user message (interactive chat)."""
        if self.infra_error:
            raise RuntimeError(f"episode ended with an infrastructure error: {self.infra_error}")
        self.messages.append({"role": "user", "content": content})
        self.stats["n_user_followups"] += 1
        self.stats.pop("termination", None)
        self._ev("user.message", content=content)
        self.done, self.ended_normally, self.final_text = False, False, ""
        self._last_sig, self._repeat_run = None, 0
        self.max_turns = self.stats["n_turns"] + (max_new_turns or self.task["limits"]["max_turns"])

    def trajectory(self, policy_info: dict) -> dict:
        outcome = None
        if not self.infra_error:
            try:
                outcome = score(self.task, self.session, self.executed, self.final_text, ended_normally=self.ended_normally,
                                stats=self.stats)
            except InfraError as e:      # e.g. MCP server died before the post-episode check
                self.infra_error = str(e)
        if self.infra_error:
            outcome = Outcome(reward=None, status="infra_fail", failure_kind="infra", components={"error": self.infra_error, **self.stats})
        session_prov = self.session.provenance() if self.session else None
        if self.session:
            self.session.close()
        self._ev("episode.finished", reward=outcome.reward, status=outcome.status)
        t = self.task
        return {
            "schema_version": TRAJ_SCHEMA, "episode_id": self.episode_id, "task_id": t["task_id"], "task_hash": t["content_hash"],
            "domain": t["domain"], "category": t["category"], "difficulty": t["difficulty"], "split": t["split"], "lang": t["lang"],
            "attempt": self.attempt, "policy": {**policy_info, "enable_thinking": self.enable_thinking},
            "skills": t.get("skills") or [], "mcp_servers": t.get("mcp_servers") or [],
            "required_skill": (t.get("expected") or {}).get("skill_required"),
            "tools": self.tools, "messages": self.messages, "turns": [x.to_dict() for x in self.turns], "events": self.events,
            "executed_calls": self.executed, "outcome": outcome.to_dict(),
            "provenance": {"harness_version": HARNESS_VERSION, "env_version": self.env.version if self.env else None,
                           "skills": (session_prov or {}).get("skills"), "mcp": (session_prov or {}).get("mcp"),
                           "toolkits": (session_prov or {}).get("toolkits"),
                           "generator": t.get("generator"), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        }


def run_episodes(tasks: Iterable[dict], policy, *, n_attempts: int = 1, concurrency: int = 8, enable_thinking: bool = False,
                 on_parse_error: str = "feedback", sink: Callable[[dict], None] | None = None, log_every: int = 20,
                 resources: Resources | None = None) -> list[dict]:
    """Continuous batching at episode granularity: keep `concurrency` episodes alive, call policy.generate_batch per round."""
    queue = deque((t, a) for t in tasks for a in range(n_attempts))
    total = len(queue)
    active: list[Episode] = []
    out = []
    t0 = time.time()
    while queue or active:
        while queue and len(active) < concurrency:
            task, a = queue.popleft()
            active.append(Episode(task, attempt=a, enable_thinking=enable_thinking, on_parse_error=on_parse_error,
                                  resources=resources))
        live = [ep for ep in active if not ep.done]       # episodes that failed to set up are finished immediately
        if live:
            for ep, turn in zip(live, policy.generate_batch(live)):
                ep.consume(turn)
        still = []
        for ep in active:
            if ep.done:
                traj = ep.trajectory(policy.info())
                out.append(traj)
                if sink:
                    sink(traj)
                if log_every and len(out) % log_every == 0:
                    ok = sum(1 for x in out if x["outcome"]["reward"] == 1.0)
                    print(f"[run] {len(out)}/{total} done, success={ok}/{len(out)}, {time.time() - t0:.0f}s", flush=True)
            else:
                still.append(ep)
        active = still
    return out
