"""Task construction helpers: execute gold steps on a fresh env to derive the expected state."""
from __future__ import annotations

import json

from ..core.types import finalize_task
from ..envs import make_env

STATUS_ALTS = {
    "pending": ["pending", "대기", "보류"],
    "processed": ["processed", "처리"],
    "delivered": ["delivered", "배송 완료", "배송완료", "배달 완료", "배달완료"],
    "cancelled": ["cancelled", "canceled", "취소"],
}

SPLIT_SEED_BASE = {"train": 1_000_000, "dev": 8_000_000, "test": 9_000_000}


PREFIX = {"en": ["", "", "", "Hi. ", "Hello! ", "Hey, ", "Good morning. ", "Hi there, "],
          "ko": ["", "", "", "안녕하세요. ", "저기요, ", "수고 많으십니다. "]}
SUFFIX = {"en": ["", "", "", " Thanks!", " Thank you.", " Appreciate it.", " Thanks in advance."],
          "ko": ["", "", "", " 감사합니다.", " 고마워요!", " 부탁드려요."]}
GREETINGS = ("hi", "hello", "hey", "안녕")


def stylize(rng, lang: str, instruction: str) -> str:
    """Light surface variation (greeting / closing) so identical templates do not yield identical prompts."""
    pre = "" if instruction.lower().startswith(GREETINGS) else rng.choice(PREFIX[lang])
    return pre + instruction + rng.choice(SUFFIX[lang])


def visible_hash(task: dict) -> str:
    """Hash of what the model actually sees (system + first user message + scripted follow-ups)."""
    import hashlib
    return hashlib.sha1(json.dumps([task["system_prompt"], task["instruction"], task["user_followups"]],
                                   ensure_ascii=False).encode()).hexdigest()


SKILL_TOOLS = ("load_skill", "read_skill_file")
EXAMPLE_SKILLS = ["file-archiving", "fiscal-calendar", "travel-api"]   # examples/skills; offered together in skill tasks


def _is_env_step(name: str) -> bool:
    return name not in SKILL_TOOLS and not name.startswith("mcp__")


def load_skill(name):
    return {"name": "load_skill", "arguments": {"name": name}}


def read_skill_file(name, path):
    return {"name": "read_skill_file", "arguments": {"name": name, "path": path}}


def mcp(server, tool, **arguments):
    """Gold step calling an MCP tool (model-facing alias mcp__<server>__<tool>)."""
    return {"name": f"mcp__{server}__{tool}", "arguments": arguments}


def mcp_check(server, tool, arguments=None, *, contains=None, not_contains=None, json_subset=None):
    chk = {"type": "mcp_call", "server": server, "tool": tool, "arguments": arguments or {}}
    if contains:
        chk["expect_contains"] = list(contains)
    if not_contains:
        chk["expect_not_contains"] = list(not_contains)
    if json_subset is not None:
        chk["expect_json"] = json_subset
    return chk


class GenSkip(Exception):
    """World does not contain a suitable entity for this template; try another seed."""


def num(value, tol=0.01):
    return {"kind": "number", "value": value, "tol": tol}


def text(value, alts=None):
    return {"kind": "text", "value": value, "alts": list(alts or [])}


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def call_err(name, **arguments):
    """Gold step that is expected to return an error (recovery demonstrations)."""
    return {"name": name, "arguments": arguments, "expect_error": True}


def ask(text_):
    return {"ask": text_}


def L(lang, en, ko):
    return ko if lang == "ko" else en


def build_task(*, domain: str, category: str, difficulty: int, seed: int, lang: str, split: str,
               instruction: str, gold: list, final: str, answers: list | None = None,
               forbidden: list | None = None, answer_forbidden: list | None = None,
               followups: list | None = None, checks: list | None = None, max_turns: int = 12,
               gen_name: str = "", world_params: dict | None = None, skills: list | None = None,
               mcp_servers: list | None = None, mcp_setup: list | None = None, system_prompt: str | None = None,
               env_tools: list | None = None) -> dict:
    """Skill (`load_skill`, `read_skill_file`) and MCP (`mcp__*`) gold steps are not executed here: they need the
    run-time registries and live servers, and are validated by the oracle gate (`agentic-harness gen-tasks`)."""
    world = {"seed": seed, "params": world_params or {}}
    env = make_env(domain, world)
    initial_hash = env.state_hash()
    for step in gold:
        if "name" not in step or not _is_env_step(step["name"]):
            continue
        out, is_err = env.call(step["name"], step["arguments"])
        if is_err != bool(step.get("expect_error")):
            raise GenSkip(f"gold step {step['name']} error={is_err}: {out[:200]}")
    for chk in checks or []:
        if chk["type"] in ("mcp_call", "skill_loaded"):
            continue
        if not env.check(chk):
            raise GenSkip(f"gold fails check {chk}")
    mutating = any((env.is_mutating(s["name"]) or s["name"].startswith("mcp__")) and not s.get("expect_error")
                   for s in gold if "name" in s)
    task = {
        "task_id": f"{domain}-{category}-{split}-{seed}",
        "domain": domain, "category": category, "difficulty": difficulty, "split": split, "lang": lang,
        "world": world,
        "system_prompt": system_prompt if system_prompt is not None else env.system_prompt,
        "instruction": instruction,
        "user_followups": list(followups or []),
        "tools": env.tool_names() if env_tools is None else list(env_tools),
        "skills": list(skills or []),
        "mcp_servers": list(mcp_servers or []),
        "mcp_setup": list(mcp_setup or []),
        "gold_actions": gold,
        "gold_final": final,
        "expected": {
            "state_hash": env.state_hash(),
            "state_changes": env.state_hash() != initial_hash,
            "answers": list(answers or []),
            "answer_forbidden": list(answer_forbidden or []),
            "checks": list(checks or []),
            "skill_required": next((st["arguments"]["name"] for st in gold if st.get("name") == "load_skill"), None)
            if category == "skill" else None,
        },
        "forbidden_actions": list(forbidden or []),
        "limits": {"max_turns": max_turns},
        "generator": {"name": gen_name or f"{domain}.{category}", "version": "0.1", "gold_mutates": mutating},
    }
    json.dumps(task)  # must be serializable
    return finalize_task(task)
