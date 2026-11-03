"""User-facing task format -> internal TaskSpec.

A task file is JSON (a list or {"tasks": [...]}), JSONL (one task per line) or YAML (if PyYAML is installed).
Internal tasks produced by `agentic-harness gen-tasks` (they carry "schema_version") are passed through unchanged.

User task fields (only `instruction` is required):

    id               str      default "task-<n>"
    instruction      str      first user message
    system_prompt    str      default: the env's prompt, or a generic tool-using assistant prompt
    env              str | {"domain": "retail", "seed": 3}     optional built-in environment
    toolkits         [str]    user toolkits (registered with --toolkits)
    skills           [str]    skills to offer (registered with --skills-dir)
    mcp_servers      [str]    MCP servers to connect (registered with --mcp-config)
    mcp_setup        [{"server", "tool", "arguments"}]          seed MCP servers before the episode
    followups        [str]    scripted user replies, given when the assistant answers without calling a tool
    files            {path: text}   files written into the shell workspace (needs the `shell` toolkit)
    max_turns        int      default 12
    gold_actions     [{"name", "arguments"}]   optional reference solution (env steps also define the expected env state)
    verify:
      answers          [str | number | {"contains"|"number"|"regex": ..., "tol": .., "alts": [..]}]   all must match
      answer_forbidden [str]                    must NOT appear in the final answer
      checks           [{"type": "mcp_call" | "state" | "toolkit" | "skill_loaded" | <env check>, ...}]
      forbidden_actions [{"name": tool, "match": {arg: value}}]
      expect_no_state_change  bool              built-in env state must be unchanged
      python           "file.py:function" or "module:function"  -> bool | float in [0,1] | {"reward": x, ...}
    category, difficulty, lang, metadata      free-form labels used in reports

Tasks without any verification are run and saved but get reward=None, status "unscored".
"""
from __future__ import annotations

import json
import os

from ..envs import make_env
from ..envs.none import NoneEnv
from .types import TASK_SCHEMA, finalize_task


def _answer(a) -> dict:
    if isinstance(a, bool):
        return {"kind": "text", "value": str(a).lower(), "alts": []}
    if isinstance(a, (int, float)):
        return {"kind": "number", "value": a, "tol": 1e-6 if isinstance(a, int) else 0.01}
    if isinstance(a, str):
        return {"kind": "text", "value": a, "alts": []}
    if "number" in a:
        return {"kind": "number", "value": a["number"], "tol": a.get("tol", 0.01)}
    if "regex" in a:
        return {"kind": "regex", "value": a["regex"]}
    if "contains" in a:
        return {"kind": "text", "value": a["contains"], "alts": list(a.get("alts") or [])}
    if a.get("kind") in ("text", "number", "regex"):
        return a
    raise ValueError(f"unsupported answer spec: {a!r}")


def normalize_task(d: dict, index: int = 0, base_dir: str | None = None) -> dict:
    if "schema_version" in d and "world" in d:          # already an internal task
        return d
    if not d.get("instruction"):
        raise ValueError(f"task #{index}: 'instruction' is required")
    unknown = set(d) - {"id", "instruction", "system_prompt", "env", "toolkits", "skills", "mcp_servers", "mcp_setup",
                        "followups", "max_turns", "gold_actions", "gold_final", "verify", "category", "difficulty",
                        "lang", "metadata", "files"}
    if unknown:
        raise ValueError(f"task #{index} ({d.get('id')}): unknown fields {sorted(unknown)}")
    env_spec = d.get("env")
    if isinstance(env_spec, str):
        env_spec = {"domain": env_spec}
    domain = (env_spec or {}).get("domain", "none")
    seed = int((env_spec or {}).get("seed", 0))
    env = make_env(domain, {"seed": seed, "params": {}})
    v = d.get("verify") or {}
    unknown_v = set(v) - {"answers", "answer_forbidden", "checks", "forbidden_actions", "expect_no_state_change", "python"}
    if unknown_v:
        raise ValueError(f"task #{index}: unknown verify fields {sorted(unknown_v)}")
    gold = list(d.get("gold_actions") or [])
    state_hash = None
    if domain != "none" and (gold or v.get("expect_no_state_change")):
        for step in gold:
            if "name" in step and step["name"] in env.tool_names():
                out, err = env.call(step["name"], step.get("arguments") or {})
                if err and not step.get("expect_error"):
                    raise ValueError(f"task {d.get('id')}: gold step {step['name']} failed: {out[:200]}")
        state_hash = env.state_hash()
    py = v.get("python")
    if py and base_dir and ":" in py:
        path, fn = py.rsplit(":", 1)
        if path.endswith(".py") and not os.path.isabs(path):
            py = os.path.join(base_dir, path) + ":" + fn
    task = {
        "schema_version": TASK_SCHEMA,
        "task_id": str(d.get("id") or f"task-{index}"),
        "domain": domain, "category": d.get("category", "custom"), "difficulty": d.get("difficulty", 0),
        "split": "custom", "lang": d.get("lang", "en"),
        "world": {"seed": seed, "params": {}},
        "system_prompt": d.get("system_prompt") or (env.system_prompt if domain != "none" else NoneEnv.system_prompt),
        "instruction": d["instruction"],
        "user_followups": list(d.get("followups") or []),
        "tools": env.tool_names() if domain != "none" else [],
        "toolkits": list(d.get("toolkits") or []),
        "skills": list(d.get("skills") or []),
        "mcp_servers": list(d.get("mcp_servers") or []),
        "mcp_setup": list(d.get("mcp_setup") or []),
        "gold_actions": gold,
        "gold_final": d.get("gold_final", ""),
        "expected": {"state_hash": state_hash, "state_changes": None,
                     "answers": [_answer(a) for a in v.get("answers") or []],
                     "answer_forbidden": list(v.get("answer_forbidden") or []),
                     "checks": list(v.get("checks") or []), "python": py, "skill_required": None},
        "forbidden_actions": list(v.get("forbidden_actions") or []),
        "limits": {"max_turns": int(d.get("max_turns", 12))},
        "generator": {"name": "user", "version": "0", "gold_mutates": None},
        "metadata": d.get("metadata") or {},
        "files": dict(d.get("files") or {}),
    }
    return finalize_task(task)


def load_task_file(path: str) -> list[dict]:
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml  # optional dependency
        data = yaml.safe_load(raw)
    elif path.endswith(".jsonl"):
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("tasks", [data])
    tasks = [normalize_task(d, i, base) for i, d in enumerate(data)]
    ids = [t["task_id"] for t in tasks]
    dup = {x for x in ids if ids.count(x) > 1}
    if dup:
        raise ValueError(f"duplicate task ids in {path}: {sorted(dup)}")
    return tasks
