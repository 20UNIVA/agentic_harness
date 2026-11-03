"""ms-swift GRPO integration. Load with:

  swift rlhf --rlhf_type grpo ... \
      --external_plugins <repo>/agentic_harness/integrations/swift_plugin.py \
      --multi_turn_scheduler agent_harness --max_turns 12 \
      --reward_funcs harness_reward [harness_format] \
      --dataset <exported grpo jsonl with columns messages/tools/harness_task>

The scheduler uses the SAME parser (core.toolcall), Env and scorer (scoring.verify) as the offline runner,
so RL reward == evaluation reward. Tool results are appended as {"role": "tool"} messages, which the qwen3_5
template renders as <tool_response> inside a user turn (not trained on).
Tested offline with a scripted engine (tests/test_swift_plugin.py); not yet run inside a real GRPO job.
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agentic_harness.core.runner import FATAL_PARSE_ERRORS  # noqa: E402
from agentic_harness.core.session import ToolSession, build_resources, default_resources  # noqa: E402
from agentic_harness.core.toolcall import parse_assistant  # noqa: E402
from agentic_harness.core.types import InfraError, canonical_json  # noqa: E402
from agentic_harness.scoring.verify import score  # noqa: E402
from swift.rewards.orm import ORM, orms  # noqa: E402
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns  # noqa: E402


_RESOURCES = None


def _resources():
    """Skills / MCP / toolkit registries for GRPO rollouts. Override with env vars (os.pathsep-separated lists):
    AGENTIC_HARNESS_SKILLS_DIRS, AGENTIC_HARNESS_MCP_CONFIGS, AGENTIC_HARNESS_TOOLKITS,
    AGENTIC_HARNESS_SKILLS_MODE (tool|inline|off), AGENTIC_HARNESS_ALLOW_SHELL=1."""
    global _RESOURCES
    if _RESOURCES is None:
        env = os.environ.get
        sk, mc, tk = env("AGENTIC_HARNESS_SKILLS_DIRS"), env("AGENTIC_HARNESS_MCP_CONFIGS"), env("AGENTIC_HARNESS_TOOLKITS")
        mode, shell = env("AGENTIC_HARNESS_SKILLS_MODE", "tool"), env("AGENTIC_HARNESS_ALLOW_SHELL") == "1"
        if sk is None and mc is None and tk is None:
            _RESOURCES = default_resources()
            _RESOURCES.skills_mode, _RESOURCES.allow_shell = mode, shell
        else:
            split = lambda v: v.split(os.pathsep) if v else None  # noqa: E731
            _RESOURCES = build_resources(skills_dirs=split(sk), mcp_configs=split(mc), toolkit_specs=split(tk),
                                         skills_mode=mode, allow_shell=shell)
    return _RESOURCES


class _EpisodeState:
    def __init__(self, task: dict):
        self.task = task
        self.session = ToolSession(task, _resources())     # env + skills + MCP servers (fresh per trajectory)
        self.tools = self.session.tool_schemas()
        self.followups = deque(task.get("user_followups") or [])
        self.executed: List[dict] = []
        self.pending: List[dict] = []
        self.stats = {"n_turns": 0, "n_tool_calls": 0, "n_parse_errors": 0, "n_tool_errors": 0, "n_repeat_calls": 0,
                      "n_user_followups": 0}
        self.last_sig = None
        self.repeat_run = 0


class AgentHarnessScheduler(MultiTurnScheduler):
    loop_limit = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eps: Dict[str, _EpisodeState] = {}

    async def on_trajectory_start(self, requests) -> None:
        for req in requests:
            raw = (req.data_dict or {}).get("harness_task")
            if raw is None:
                raise ValueError("dataset row lacks 'harness_task' (export with agentic_harness.export.swift.to_swift_grpo_row)")
            task = json.loads(raw) if isinstance(raw, str) else raw
            try:
                st = _EpisodeState(task)
            except InfraError as e:
                self._eps[req.uuid] = None
                self._infra = getattr(self, "_infra", {})
                self._infra[req.uuid] = str(e)
                continue
            self._eps[req.uuid] = st
            # the session is authoritative for what the model sees: system prompt (+skills/MCP sections) and tools
            if req.messages and req.messages[0].get("role") == "system":
                req.messages[0]["content"] = st.session.system_prompt()
            req.tools = st.tools

    async def on_turn_end(self, infer_request, response_choice, current_turn: int) -> Dict[str, Any]:
        st = self._eps.get(infer_request.uuid)
        if st is None:
            err = getattr(self, "_infra", {}).pop(infer_request.uuid, "episode state missing")
            return {"done": True, "rollout_infos": {"harness": {"reward": None, "status": "infra_fail", "error": err}}}
        st.stats["n_turns"] += 1
        text = response_choice.message.content or ""
        enable_thinking = bool((infer_request.chat_template_kwargs or {}).get("enable_thinking", False)) \
            if hasattr(infer_request, "chat_template_kwargs") else False
        pr = parse_assistant(text, st.tools, prompt_opened_think=enable_thinking)
        st.stats["n_parse_errors"] += len([e for e in pr.errors if e != "text_after_tool_call"])
        termination = "pending"
        if pr.tool_calls:
            for c in pr.tool_calls:
                try:
                    result, is_err = st.session.call(c.name, c.arguments)
                except InfraError as e:
                    self._eps.pop(infer_request.uuid, None)
                    st.session.close()
                    return {"done": True, "rollout_infos": {"harness": {"reward": None, "status": "infra_fail", "error": str(e)}}}
                st.stats["n_tool_calls"] += 1
                st.stats["n_tool_errors"] += int(is_err)
                st.executed.append({"name": c.name, "arguments": c.arguments, "ok": not is_err})
                st.pending.append({"role": "tool", "content": result})
                sig = canonical_json([c.name, c.arguments])
                if sig == st.last_sig:
                    st.repeat_run += 1
                    st.stats["n_repeat_calls"] += 1
                else:
                    st.repeat_run = 0
                st.last_sig = sig
            if st.repeat_run >= self.loop_limit - 1:
                termination = "loop_abort"
        elif any(e in FATAL_PARSE_ERRORS for e in pr.errors):
            st.pending.append({"role": "tool", "content": json.dumps(
                {"error": "could not parse your tool call (" + ", ".join(pr.errors) + ")."})})
        elif response_choice.finish_reason == "length":
            termination = "truncated"
        elif st.followups:
            st.pending.append({"role": "user", "content": st.followups.popleft()})
            st.stats["n_user_followups"] += 1
        else:
            termination = None   # normal end: final answer
            final_text = pr.content
        max_turns = self.max_turns or st.task["limits"]["max_turns"]
        if termination == "pending" and current_turn >= max_turns:
            termination = "max_turns"
        if termination == "pending":
            return {"done": False}
        if termination is not None:
            st.stats["termination"] = termination
        try:
            out = score(st.task, st.session, st.executed, final_text if termination is None else "",
                        ended_normally=termination is None, stats=st.stats)
        except InfraError as e:
            self._eps.pop(infer_request.uuid, None)
            st.session.close()
            return {"done": True, "rollout_infos": {"harness": {"reward": None, "status": "infra_fail", "error": str(e)}}}
        self._eps.pop(infer_request.uuid, None)
        st.session.close()
        return {"done": True, "rollout_infos": {"harness": {"reward": out.reward, "status": out.status,
                                                            "failure_kind": out.failure_kind, "components": out.components}}}

    def check_finished(self, infer_request, response_choice, current_turn: int) -> bool:
        return False   # termination is decided in on_turn_end (its 'done' overrides this)

    def step(self, infer_request, response_choice, current_turn: int) -> Dict:
        st = self._eps.get(infer_request.uuid)
        if st is not None:
            infer_request.messages.extend(st.pending)
            st.pending = []
        return {"infer_request": infer_request}


def _infos(kwargs, n):
    infos = kwargs.get("rollout_infos") or [None] * n
    return [((i or {}).get("harness") or {}) for i in infos]


class HarnessReward(ORM):
    """Binary task success from the harness scorer. Unscored (infra) episodes get 0.0 and are counted in logs."""

    def __call__(self, completions, **kwargs) -> List[float]:
        return [float(h.get("reward") or 0.0) for h in _infos(kwargs, len(completions))]


class HarnessFormat(ORM):
    """Optional shaping: -1 per parse error turn (clipped to [-1, 0]). Use with a small weight, e.g. 0.1."""

    def __call__(self, completions, **kwargs) -> List[float]:
        return [max(-1.0, -float((h.get("components") or {}).get("n_parse_errors", 0)))
                for h in _infos(kwargs, len(completions))]


multi_turns["agent_harness"] = AgentHarnessScheduler
orms["harness_reward"] = HarnessReward
orms["harness_format"] = HarnessFormat
