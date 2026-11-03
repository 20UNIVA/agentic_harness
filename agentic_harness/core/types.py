"""Shared data types and hashing helpers (stdlib only)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

HARNESS_VERSION = "0.2.0"
TASK_SCHEMA = "agentharness.task.v1"
TRAJ_SCHEMA = "agentharness.traj.v1"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_of(obj: Any) -> str:
    data = obj if isinstance(obj, (bytes, str)) else canonical_json(obj)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def task_content_hash(task: dict) -> str:
    """Hash of everything that defines the task, excluding identifiers and split label."""
    body = {k: v for k, v in task.items() if k not in ("task_id", "content_hash", "split")}
    return sha256_of(body)


def finalize_task(task: dict) -> dict:
    task.setdefault("schema_version", TASK_SCHEMA)
    task["content_hash"] = task_content_hash(task)
    return task


@dataclass
class Turn:
    """One policy generation."""
    raw_text: str = ""
    content: str = ""
    reasoning: str | None = None
    tool_calls: list[dict] = field(default_factory=list)      # [{"id","name","arguments"}]
    parse_errors: list[str] = field(default_factory=list)
    finish_reason: str | None = None
    think_unclosed: bool = False
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    completion_logprobs: list[float] | None = None
    prefix_mismatch: bool | None = None
    latency_s: float | None = None
    error: str | None = None                                  # backend/infra error text
    provider_meta: dict | None = None                         # e.g. answering model, usage, stop reason, request id

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Outcome:
    reward: float | None
    status: str                       # ok | policy_fail | format_fail | max_turns | loop_abort | infra_fail
    failure_kind: str | None = None
    components: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ToolError(Exception):
    """Raised by tools for user-visible errors (returned to the model as {"error": ...})."""


class InfraError(Exception):
    """Harness/backend failure that must not be scored as a policy failure (reward=None)."""
