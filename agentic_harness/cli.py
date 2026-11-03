"""agentic-harness command line.

  agentic-harness init       scaffold a project (tools, tasks, run configs)
  agentic-harness chat       talk to an agent yourself (tools / skills / MCP); every conversation is saved as a trajectory
  agentic-harness run        let a model solve tasks with tools / skills / MCP and save trajectories (+ report)
  agentic-harness gen-tasks  generate the built-in benchmark task splits (with validation gates)
  agentic-harness export     trajectories -> SFT / OpenAI-chat / step-DPO rows, tasks -> GRPO rows
  agentic-harness resources  list registered skills and connect to every MCP server (config check)
  agentic-harness view       render trajectories as Markdown transcripts
  agentic-harness analyze    behavioural failure analysis of a run
  agentic-harness serve-hf   minimal OpenAI-compatible server for a Hugging Face model
"""
import importlib
import sys

COMMANDS = {"init": "init", "chat": "chat", "run": "run", "gen-tasks": "gen_tasks", "export": "export", "resources": "resources", "view": "view",
            "analyze": "analyze", "serve-hf": "serve_hf"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-V", "--version"):
        from .core.types import HARNESS_VERSION
        print(HARNESS_VERSION)
        return 0
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        if argv and argv[0] not in ("-h", "--help"):
            print(f"unknown command {argv[0]!r}")
            return 2
        return 0
    mod = importlib.import_module(f"agentic_harness.commands.{COMMANDS[argv[0]]}")
    mod.main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
