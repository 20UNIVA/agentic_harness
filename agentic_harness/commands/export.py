"""Export trajectories / tasks to training formats.

  # SFT (rejection sampling: successful, no parse errors, 1 per task)
  agentic-harness export sft --traj runs/X/trajectories.jsonl --out data/sft/harness_sft.jsonl [--keep-reasoning] [--max-per-task 2]
  # GRPO prompts (one row per task; AgentHarnessScheduler rebuilds the env from harness_task)
  agentic-harness export grpo --tasks data/tasks_v0.2/train.jsonl --out data/grpo/harness_train.jsonl
  # step-level DPO pairs (experimental)
  agentic-harness export dpo --traj runs/X/trajectories.jsonl --out data/dpo/harness_dpo.jsonl
"""
import argparse
import collections
import json
import os

from ..export.swift import (select_for_sft, to_openai_chat, to_step_dpo_pairs, to_swift_grpo_row,
                                          to_swift_sft)


def read(path):
    return [json.loads(line) for line in open(path)]


def write(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentic-harness export")
    ap.add_argument("kind", choices=["sft", "openai", "grpo", "dpo"])
    ap.add_argument("--traj", nargs="*", default=[])
    ap.add_argument("--tasks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-reasoning", action="store_true")
    ap.add_argument("--max-per-task", type=int, default=1)
    ap.add_argument("--allow-parse-errors", type=int, default=0)
    ap.add_argument("--exclude-categories", nargs="*", default=[])
    ap.add_argument("--skills-dir", action="append", help="for grpo rows: skills registry (default: examples/skills)")
    ap.add_argument("--mcp-config", action="append", help="for grpo rows: MCP registry (default: examples/mcp/servers.json)")
    ap.add_argument("--toolkits", action="append", help="for grpo rows: user toolkits")
    ap.add_argument("--allow-shell", action="store_true")
    args = ap.parse_args(argv)
    if args.kind == "grpo":
        from ..core.taskspec import load_task_file
        from .run import DEFAULTS, build_run_resources
        res = build_run_resources({**DEFAULTS, "skills_dir": args.skills_dir, "mcp_config": args.mcp_config,
                                   "toolkits": args.toolkits, "allow_shell": args.allow_shell})
        write([to_swift_grpo_row(t, res) for t in load_task_file(args.tasks)], args.out)
        res.close()
        return
    trajs = [t for p in args.traj for t in read(p)]
    if args.kind == "dpo":
        write(to_step_dpo_pairs(trajs), args.out)
        return
    sel = select_for_sft(trajs, max_parse_errors=args.allow_parse_errors, max_per_task=args.max_per_task,
                         exclude_categories=tuple(args.exclude_categories))
    print(f"selected {len(sel)}/{len(trajs)} trajectories; by category:",
          dict(collections.Counter(t["category"] for t in sel)))
    fn = (lambda t: to_swift_sft(t, keep_reasoning=args.keep_reasoning)) if args.kind == "sft" else to_openai_chat
    write([fn(t) for t in sel], args.out)


if __name__ == "__main__":
    main()
