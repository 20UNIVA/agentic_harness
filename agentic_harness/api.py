"""Python API.

    from agentic_harness import run_tasks, summarize

    trajs = run_tasks(
        tasks="examples/tasks/quickstart.json",             # path, or a list of task dicts (user format or generated)
        model={"provider": "openai_compatible", "base_url": "http://127.0.0.1:8000/v1", "model": "my-model"},
        toolkits=["examples/tools/finance_tools.py"],        # files / modules
        skills_dirs=["examples/skills"], mcp_configs=["examples/mcp/servers.json"],
        allow_shell=False, n_attempts=1, concurrency=8,
    )
    print(summarize(trajs)["overall"])
"""
from __future__ import annotations

from typing import Callable, Iterable

from .core.runner import run_episodes
from .core.session import build_resources
from .core.taskspec import load_task_file, normalize_task
from .policies import make_policy
from .policies.base import Policy


def run_tasks(tasks, model: dict | Policy, *, toolkits: Iterable[str] = (), skills_dirs: Iterable[str] = (),
              mcp_configs: Iterable[str] = (), skills_mode: str = "tool", extra_skills: Iterable[str] = (),
              extra_mcp: Iterable[str] = (), extra_toolkits: Iterable[str] = (), allow_shell: bool = False,
              n_attempts: int = 1, concurrency: int = 8, enable_thinking: bool = False,
              max_tool_result_chars: int = 20000, sink: Callable[[dict], None] | None = None,
              log_every: int = 0) -> list[dict]:
    """Run tasks with a model and return trajectories (dicts, same schema as trajectories.jsonl)."""
    if isinstance(tasks, str):
        task_list = load_task_file(tasks)
    else:
        task_list = [normalize_task(t, i) for i, t in enumerate(tasks)]
    res = build_resources(skills_dirs=list(skills_dirs) or None, mcp_configs=list(mcp_configs) or None,
                          toolkit_specs=list(toolkits) or None, skills_mode=skills_mode, extra_skills=list(extra_skills),
                          extra_mcp=list(extra_mcp), extra_toolkits=list(extra_toolkits), allow_shell=allow_shell,
                          max_tool_result_chars=max_tool_result_chars)
    policy = model if isinstance(model, Policy) else make_policy(model)
    try:
        return run_episodes(task_list, policy, n_attempts=n_attempts, concurrency=concurrency,
                            enable_thinking=enable_thinking, sink=sink, log_every=log_every, resources=res)
    finally:
        res.close()
