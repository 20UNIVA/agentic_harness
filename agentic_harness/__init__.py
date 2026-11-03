"""agentic-harness: run any LLM with tools, skills and MCP servers on verifiable tasks and save trajectories."""
from .core.types import HARNESS_VERSION as __version__  # noqa: F401


def run_tasks(*args, **kwargs):
    """See agentic_harness.api.run_tasks."""
    from .api import run_tasks as _run
    return _run(*args, **kwargs)


def summarize(trajectories, ks=(1,)):
    """pass@k / pass^k and breakdowns for a list of trajectories (see agentic_harness.scoring.metrics)."""
    from .scoring.metrics import summarize as _summarize
    return _summarize(trajectories, ks=ks)


def __getattr__(name):
    if name == "ChatSession":                 # lazy: keeps `import agentic_harness` light
        from .chat import ChatSession
        return ChatSession
    raise AttributeError(name)
