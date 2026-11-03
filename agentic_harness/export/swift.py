"""Exporters: trajectory -> ms-swift SFT row / GRPO row / step-level DPO pair. Pure functions.

ms-swift agent format (`swift sft --dataset <file>`):
  {"messages": [{"role":"system"}, {"role":"user"}, {"role":"assistant","content":"(optional text before calls)"},
                {"role":"tool_call","content":"{\"name\":..,\"arguments\":{..}}"}, {"role":"tool_response","content":"..."},
                ..., {"role":"assistant","content":"final"}],
   "tools": "<json string of OpenAI function schemas>"}
"""
from __future__ import annotations

import json
from collections import defaultdict

from ..core.toolcall import render_tool_calls


def _assistant_text(msg: dict, keep_reasoning: bool) -> str:
    content = msg.get("content") or ""
    if keep_reasoning and msg.get("reasoning_content"):
        return f"<think>\n{msg['reasoning_content'].strip()}\n</think>\n\n{content}"
    return content


def to_swift_sft(traj: dict, *, keep_reasoning: bool = False, include_system: bool = True) -> dict:
    out = []
    for m in traj["messages"]:
        r = m["role"]
        if r == "system":
            if include_system:
                out.append({"role": "system", "content": m["content"]})
        elif r == "user":
            out.append({"role": "user", "content": m["content"]})
        elif r == "assistant":
            text = _assistant_text(m, keep_reasoning)
            if m.get("tool_calls"):
                if text.strip():
                    out.append({"role": "assistant", "content": text})
                for tc in m["tool_calls"]:
                    out.append({"role": "tool_call", "content": json.dumps(
                        {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}, ensure_ascii=False)})
            else:
                out.append({"role": "assistant", "content": text})
        elif r == "tool":
            if m.get("tool_call_id") == "parse_error":
                raise ValueError("trajectory contains harness parse-error feedback; filter it before SFT export")
            out.append({"role": "tool_response", "content": m["content"]})
    return {"messages": out, "tools": json.dumps(traj["tools"], ensure_ascii=False)}


def to_openai_chat(traj: dict) -> dict:
    """Generic OpenAI chat format (tool_calls with JSON-string arguments) for other trainers (e.g. prime-rl SFT, TRL)."""
    from ..policies.openai_compat import to_openai_messages
    return {"messages": to_openai_messages(traj["messages"]), "tools": traj["tools"]}


def select_for_sft(trajs, *, require_success=True, max_parse_errors=0, allow_tool_errors=True, max_per_task=1,
                   exclude_categories=(), max_prefix_mismatch=None):
    """Rejection-sampling filter. Keeps up to `max_per_task` successful, clean trajectories per task.
    A human rating (chat conversations) overrides the verifier: "good" counts as success, "bad" is always excluded."""
    kept, per_task = [], defaultdict(int)
    for t in trajs:
        o = t["outcome"]
        c = o["components"]
        rating = (t.get("human_feedback") or {}).get("rating")
        if rating == "bad" or (o["status"] == "infra_fail"):
            continue
        if require_success and o["reward"] != 1.0 and rating != "good":
            continue
        if c.get("n_parse_errors", 0) > max_parse_errors:
            continue
        if not allow_tool_errors and c.get("n_tool_errors", 0) > 0:
            continue
        if t["category"] in exclude_categories:
            continue
        if max_prefix_mismatch is not None and c.get("n_prefix_mismatch", 0) > max_prefix_mismatch:
            continue
        if per_task[t["task_id"]] >= max_per_task:
            continue
        per_task[t["task_id"]] += 1
        kept.append(t)
    return kept


_VIEW_CACHE: dict = {}


def _session_view(task: dict, resources=None) -> tuple[list, str]:
    """(tool schemas, system-prompt addendum) as the model will see them, including skills and MCP tools.
    Cached per tool signature so MCP servers are started once per distinct configuration, not per task."""
    from ..core.session import ToolSession
    key = (task["domain"], tuple(task.get("tools") or []), tuple(task.get("skills") or []),
           tuple(task.get("mcp_servers") or []), id(resources))
    if key not in _VIEW_CACHE:
        sess = ToolSession({**task, "mcp_setup": []}, resources)
        try:
            sp = sess.system_prompt()
            _VIEW_CACHE[key] = (sess.tool_schemas(), sp[len(task["system_prompt"]):])
        finally:
            sess.close()
    return _VIEW_CACHE[key]


def to_swift_grpo_row(task: dict, resources=None) -> dict:
    """GRPO dataset row; the AgentHarnessScheduler rebuilds env + skills + MCP from `harness_task` (and re-applies the
    exact system prompt / tools at rollout start)."""
    tools, addendum = _session_view(task, resources)
    return {"messages": [{"role": "system", "content": task["system_prompt"] + addendum},
                         {"role": "user", "content": task["instruction"]}],
            "tools": json.dumps(tools, ensure_ascii=False),
            "harness_task": json.dumps(task, ensure_ascii=False)}


def _turn_text(msg: dict) -> str:
    calls = [{"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]} for tc in msg.get("tool_calls") or []]
    return render_tool_calls(calls, msg.get("content") or "", style="swift")


def _calls_key(msg: dict):
    return json.dumps([[tc["function"]["name"], tc["function"]["arguments"]] for tc in msg.get("tool_calls") or []],
                      sort_keys=True, ensure_ascii=False)


def to_step_dpo_pairs(trajs):
    """Step-level DPO (experimental): pair a successful and a failed trajectory of the same task at the first assistant
    turn whose *action* differs (tool name+arguments, or tool call vs. final answer). Turns that differ only in free text
    before an identical call are treated as equivalent, so the pair isolates a behavioural decision.
    Output rows use ms-swift DPO format: {"messages": prefix + chosen turn, "rejected_response": text, "tools"}."""
    by_task = defaultdict(lambda: ([], []))
    for t in trajs:
        if t["outcome"]["reward"] is None:
            continue
        by_task[t["task_id"]][0 if t["outcome"]["reward"] == 1.0 else 1].append(t)
    rows = []
    for tid, (good, bad) in by_task.items():
        if not good or not bad:
            continue
        g, b = good[0], bad[0]
        ga = [i for i, m in enumerate(g["messages"]) if m["role"] == "assistant"]
        ba = [i for i, m in enumerate(b["messages"]) if m["role"] == "assistant"]
        k = 0
        while k < min(len(ga), len(ba)) and _calls_key(g["messages"][ga[k]]) == _calls_key(b["messages"][ba[k]]) \
                and (g["messages"][ga[k]].get("tool_calls") or b["messages"][ba[k]].get("tool_calls")):
            k += 1
        if k >= len(ga) or k >= len(ba):
            continue
        gm, bm = g["messages"], b["messages"]
        gi, bi = ga[k], ba[k]
        if not gm[gi].get("tool_calls") and not bm[bi].get("tool_calls") and gm[gi].get("content") == bm[bi].get("content"):
            continue
        prefix = to_swift_sft({"messages": gm[:gi], "tools": g["tools"]})["messages"]
        chosen = to_swift_sft({"messages": [gm[gi]], "tools": g["tools"]})["messages"]
        rows.append({"messages": prefix + chosen, "rejected_response": _turn_text(bm[bi]),
                     "tools": json.dumps(g["tools"], ensure_ascii=False), "task_id": tid, "turn_index": k})
    return rows
