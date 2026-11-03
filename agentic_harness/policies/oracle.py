"""Oracle policies: replay gold steps (pipeline validation, gold SFT demos) and a noisy variant for scorer tests."""
from __future__ import annotations

import json
import random

from ..core.toolcall import render_tool_calls
from .base import Policy, turn_from_raw


class OraclePolicy(Policy):
    name = "oracle"

    def generate_batch(self, episodes):
        return [self._one(ep) for ep in episodes]

    def _next_step(self, ep):
        i = ep.policy_state.get("i", 0)
        gold = ep.task["gold_actions"]
        ep.policy_state["i"] = i + 1
        return gold[i] if i < len(gold) else None

    def _one(self, ep):
        step = self._next_step(ep)
        if step is None:
            raw = ep.task["gold_final"]
        elif "ask" in step:
            raw = step["ask"]
        else:
            raw = render_tool_calls([step])
        n = sum(len(t.tool_calls) for t in ep.turns)
        return turn_from_raw(raw, ep.tools, prompt_opened_think=False, finish_reason="stop", call_offset=n)


class NoisyOraclePolicy(OraclePolicy):
    """Oracle with injected faults: malformed format, wrong argument, hallucinated tool, premature final answer."""
    name = "noisy_oracle"

    def __init__(self, p: float = 0.3, seed: int = 0):
        self.p = p
        self.rng = random.Random(seed)

    def info(self):
        return {"name": self.name, "p": self.p}

    def _one(self, ep):
        step = self._next_step(ep)
        n = sum(len(t.tool_calls) for t in ep.turns)
        if step is None or "ask" in step:
            raw = ep.task["gold_final"] if step is None else step["ask"]
            return turn_from_raw(raw, ep.tools, prompt_opened_think=False, finish_reason="stop", call_offset=n)
        fault = self.rng.choice(["format", "arg", "tool", "stop"]) if self.rng.random() < self.p else None
        ep.policy_state.setdefault("faults", []).append(fault)
        if fault == "format":
            raw = f"<tool_call>\n{json.dumps(step)[:-3]}\n</tool_call>"          # broken JSON-ish call
        elif fault == "arg":
            bad = dict(step["arguments"])
            k = next(iter(bad), None)
            if k is not None:
                bad[k] = "WRONG" if isinstance(bad[k], str) else 999999
            raw = render_tool_calls([{"name": step["name"], "arguments": bad}])
        elif fault == "tool":
            raw = render_tool_calls([{"name": step["name"] + "_v2", "arguments": step["arguments"]}])
        elif fault == "stop":
            raw = "I have completed your request."
        else:
            raw = render_tool_calls([step])
        return turn_from_raw(raw, ep.tools, prompt_opened_think=False, finish_reason="stop", call_offset=n)


class NullPolicy(Policy):
    """Never calls a tool; answers immediately. Used by the 'do-nothing must fail' gate."""
    name = "null"

    def generate_batch(self, episodes):
        return [turn_from_raw("I'm sorry, I can't help with that.", ep.tools, prompt_opened_think=False, finish_reason="stop")
                for ep in episodes]
