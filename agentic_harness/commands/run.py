"""`agentic-harness run`: let a model solve tasks with tools / skills / MCP servers and save trajectories.

The same command is used for evaluation (summary.json, report.md) and for data generation (trajectories.jsonl).

  agentic-harness run --config examples/configs/openai_quickstart.json
  agentic-harness run --provider vllm --model my-model --base-url http://127.0.0.1:8000 \
      --tasks examples/tasks/quickstart.json --toolkits examples/tools/finance_tools.py --out runs/qs
  agentic-harness run --provider oracle --tasks examples/tasks/quickstart.json --allow-shell --out runs/oracle

Config file (JSON or YAML) keys mirror the long flags (dashes -> underscores), plus "model": {provider config}.
Flags given on the command line override the file.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import random
import sys
import time

from ..core.runner import run_episodes
from ..core.session import DEFAULT_MCP_CONFIG, DEFAULT_SKILLS_DIR, build_resources
from ..core.taskspec import load_task_file
from ..policies import PROVIDERS, NullPolicy, make_policy
from ..scoring.metrics import summarize, to_markdown

LIST_KEYS = ("skills_dir", "mcp_config", "toolkits", "extra_skills", "extra_mcp", "extra_toolkits", "domains",
             "categories", "templates")


def add_model_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("model")
    g.add_argument("--provider", "--policy", dest="provider", choices=PROVIDERS)
    g.add_argument("--model", help="model name / HF path / served model name")
    g.add_argument("--base-url")
    g.add_argument("--api-key-env", help="environment variable holding the API key")
    g.add_argument("--tokenizer", help="HF tokenizer path (vllm_completions)")
    g.add_argument("--device", help="hf_local device (cpu, cuda:0, ...)")
    g.add_argument("--threads", type=int)
    g.add_argument("--effort", help="anthropic: output_config.effort")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-p", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--max-new-tokens", type=int)
    g.add_argument("--model-config", help="JSON string or file with a full provider config (see agentic_harness.policies)")


def add_resource_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("resources")
    g.add_argument("--skills-dir", action="append", help="skills directory (repeatable)")
    g.add_argument("--mcp-config", action="append", help="mcpServers JSON (repeatable)")
    g.add_argument("--toolkits", action="append", help="python file / module with @tool functions or Toolkit classes (repeatable)")
    g.add_argument("--skills-mode", choices=["tool", "inline", "off"])
    g.add_argument("--allow-shell", action="store_true", default=None, help="enable the built-in shell toolkit (NOT a sandbox)")
    g.add_argument("--no-example-resources", action="store_true", default=None,
                   help="do not register the repo's examples/skills and examples/mcp/servers.json by default")
    g.add_argument("--keep-episode-dirs", action="store_true", default=None)
    g.add_argument("--max-tool-result-chars", type=int, help="truncate longer tool results (default 20000; 0 = no limit)")
    return g


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="agentic-harness run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="JSON/YAML run config; command-line flags override it")
    add_model_args(ap)
    g = ap.add_argument_group("tasks")
    g.add_argument("--tasks", help="task file: generated tasks (.jsonl) or user tasks (.json/.jsonl/.yaml)")
    g.add_argument("--limit", type=int)
    g.add_argument("--per-template", type=int, help="stratified sample: N tasks per generator template")
    g.add_argument("--sample-seed", type=int)
    g.add_argument("--domains", nargs="*")
    g.add_argument("--categories", nargs="*")
    g.add_argument("--templates", nargs="*", help="generator template globs, e.g. 'mcp_examples.*' '*.skill_*'")
    g.add_argument("--task-ids-from", help="reuse the exact task set of an earlier run (trajectories.jsonl)")
    g.add_argument("--lang")
    g = add_resource_args(ap)
    g.add_argument("--extra-skills", nargs="*", help="skills offered in every episode")
    g.add_argument("--extra-mcp", nargs="*", help="MCP servers connected in every episode")
    g.add_argument("--extra-toolkits", nargs="*", help="toolkits instantiated in every episode")
    g = ap.add_argument_group("run")
    g.add_argument("--out", help="output directory")
    g.add_argument("--n-attempts", type=int)
    g.add_argument("--concurrency", type=int)
    g.add_argument("--think", action="store_true", default=None, help="enable_thinking=True for hybrid-thinking models")
    g.add_argument("--on-parse-error", choices=["feedback", "end"])
    g.add_argument("--resume", action="store_true", default=None)
    g.add_argument("--null-baseline", action="store_true", default=None, help="also score a do-nothing policy on the same tasks")
    return ap


DEFAULTS = {"n_attempts": 1, "concurrency": 8, "think": False, "on_parse_error": "feedback", "resume": False,
            "skills_mode": "tool", "allow_shell": False, "no_example_resources": False, "keep_episode_dirs": False,
            "null_baseline": False, "sample_seed": 0, "max_tool_result_chars": 20000}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        cfg = yaml.safe_load(raw) or {}
    else:
        cfg = json.loads(raw)
    base = os.path.dirname(os.path.abspath(path))
    for k in ("tasks", "out", "task_ids_from"):                # paths in a config are relative to the config file
        if isinstance(cfg.get(k), str) and not os.path.isabs(cfg[k]):
            cfg[k] = os.path.normpath(os.path.join(base, cfg[k]))
    for k in ("skills_dir", "mcp_config", "toolkits"):
        if k in cfg:
            vals = cfg[k] if isinstance(cfg[k], list) else [cfg[k]]
            cfg[k] = [v if os.path.isabs(v) or not os.path.exists(os.path.join(base, v)) else os.path.normpath(os.path.join(base, v))
                      for v in vals]
    return cfg


def resolve(args, *, need_tasks: bool = True, defaults: dict | None = None) -> dict:
    """Merge defaults < config file < command-line flags. The model config is merged key by key."""
    file_cfg = load_config(args.config) if args.config else {}
    cfg = dict(defaults or DEFAULTS)
    cfg.update({k.replace("-", "_"): v for k, v in file_cfg.items()})
    fm = file_cfg.get("model")
    model = dict(fm) if isinstance(fm, dict) else ({"model": fm} if isinstance(fm, str) else {})
    for k, v in vars(args).items():
        if v is not None and k not in ("config", "model", "model_config"):
            cfg[k] = v
    if args.model_config:
        mc = args.model_config
        model.update(json.load(open(mc)) if os.path.exists(mc) else json.loads(mc))
    if args.provider:
        model["provider"] = args.provider
    if args.model:
        model["model"] = args.model
    for k in ("base_url", "api_key_env", "tokenizer", "device", "threads", "effort"):
        if getattr(args, k) is not None:
            model[k] = getattr(args, k)
    samp = dict(model.get("sampling") or {})
    for k in ("temperature", "top_p", "top_k", "max_new_tokens"):
        if getattr(args, k) is not None:
            samp[k] = getattr(args, k)
    if samp:
        model["sampling"] = samp
    if "concurrency" not in model and model.get("provider") not in ("hf_local", None) and "concurrency" in cfg:
        model["concurrency"] = cfg["concurrency"]
    cfg["model"] = model
    if not model.get("provider"):
        raise SystemExit("no model provider given (--provider or config 'model.provider')")
    if need_tasks and (not cfg.get("tasks") or not cfg.get("out")):
        raise SystemExit("--tasks and --out are required (or set them in the config)")
    return cfg


def select_tasks(tasks: list, cfg: dict) -> list:
    if cfg.get("task_ids_from"):
        want = {json.loads(line)["task_id"] for line in open(cfg["task_ids_from"])}
        tasks = [t for t in tasks if t["task_id"] in want]
    if cfg.get("domains"):
        tasks = [t for t in tasks if t["domain"] in cfg["domains"]]
    if cfg.get("categories"):
        tasks = [t for t in tasks if t["category"] in cfg["categories"]]
    if cfg.get("templates"):
        tasks = [t for t in tasks if any(fnmatch.fnmatch(t["generator"]["name"], p) for p in cfg["templates"])]
    if cfg.get("lang"):
        tasks = [t for t in tasks if t["lang"] == cfg["lang"]]
    rng = random.Random(cfg.get("sample_seed", 0))
    if cfg.get("per_template"):
        rng.shuffle(tasks)
        cnt, keep = {}, []
        for t in tasks:
            g = t["generator"]["name"]
            if cnt.get(g, 0) < cfg["per_template"]:
                cnt[g] = cnt.get(g, 0) + 1
                keep.append(t)
        tasks = sorted(keep, key=lambda t: t["task_id"])
    if cfg.get("limit"):
        rng.shuffle(tasks)
        tasks = sorted(tasks[: cfg["limit"]], key=lambda t: t["task_id"])
    return tasks


def build_run_resources(cfg: dict):
    examples = not cfg.get("no_example_resources")
    skills_dirs = list(cfg.get("skills_dir") or []) + ([DEFAULT_SKILLS_DIR] if examples and os.path.isdir(DEFAULT_SKILLS_DIR) else [])
    mcp_cfgs = list(cfg.get("mcp_config") or []) + ([DEFAULT_MCP_CONFIG] if examples and os.path.exists(DEFAULT_MCP_CONFIG) else [])
    skills_dirs = list({os.path.realpath(p): p for p in skills_dirs}.values())
    mcp_cfgs = list({os.path.realpath(p): p for p in mcp_cfgs}.values())
    return build_resources(skills_dirs=skills_dirs or None, mcp_configs=mcp_cfgs or None,
                           toolkit_specs=cfg.get("toolkits"), skills_mode=cfg.get("skills_mode", "tool"),
                           extra_skills=cfg.get("extra_skills") or [], extra_mcp=cfg.get("extra_mcp") or [],
                           extra_toolkits=cfg.get("extra_toolkits") or [], allow_shell=bool(cfg.get("allow_shell")),
                           keep_episode_dirs=bool(cfg.get("keep_episode_dirs")),
                           max_tool_result_chars=int(cfg.get("max_tool_result_chars", 20000)))


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = resolve(args)
    os.makedirs(cfg["out"], exist_ok=True)
    tasks = select_tasks(load_task_file(cfg["tasks"]), cfg)
    resources = build_run_resources(cfg)
    traj_path = os.path.join(cfg["out"], "trajectories.jsonl")
    done = set()
    if cfg["resume"] and os.path.exists(traj_path):
        done = {(t["task_id"], t["attempt"]) for t in map(json.loads, open(traj_path))}
    safe_cfg = {k: v for k, v in cfg.items() if "key" not in k.lower() or k == "api_key_env"}
    safe_cfg["model"] = {k: v for k, v in cfg["model"].items() if k != "api_key"}
    json.dump({**safe_cfg, "n_tasks": len(tasks), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv},
              open(os.path.join(cfg["out"], "run_config.json"), "w"), indent=2, default=str)
    policy = make_policy(cfg["model"])
    n_att = int(cfg["n_attempts"])
    todo = [t for t in tasks if any((t["task_id"], a) not in done for a in range(n_att))]
    fout = open(traj_path, "a" if cfg["resume"] else "w")

    def sink(tr):
        if (tr["task_id"], tr["attempt"]) not in done:
            fout.write(json.dumps(tr, ensure_ascii=False) + "\n")
            fout.flush()

    t0 = time.time()
    print(f"[run] {len(todo)} tasks x {n_att} attempts, provider={cfg['model']['provider']} -> {traj_path}", flush=True)
    try:
        run_episodes(todo, policy, n_attempts=n_att, concurrency=int(cfg["concurrency"]), enable_thinking=bool(cfg["think"]),
                     on_parse_error=cfg["on_parse_error"], sink=sink, log_every=max(1, int(cfg["concurrency"])),
                     resources=resources)
    finally:
        fout.close()
    trajs = [json.loads(line) for line in open(traj_path)]
    summary = summarize(trajs, ks=tuple(k for k in (1, 2, 4, 8) if k <= n_att))
    null_sum = None
    if cfg.get("null_baseline"):
        null_sum = summarize(run_episodes(tasks, NullPolicy(), concurrency=8, log_every=0, resources=resources), ks=(1,))
    resources.close()
    summary["wall_seconds"] = round(time.time() - t0, 1)
    json.dump(summary, open(os.path.join(cfg["out"], "summary.json"), "w"), ensure_ascii=False, indent=2)
    md = to_markdown(summary, f"Agentic harness report: {os.path.basename(cfg['out'].rstrip('/'))}",
                     [f"- policy: `{json.dumps(policy.info(), ensure_ascii=False, default=str)}`",
                      f"- tasks: `{cfg['tasks']}` ({len(tasks)} tasks x {n_att} attempts), enable_thinking={cfg['think']}, "
                      f"skills_mode={cfg['skills_mode']}, allow_shell={cfg['allow_shell']}",
                      f"- wall time: {summary['wall_seconds']}s"] +
                     (["- `null pass@1`: score of a policy that never calls a tool (refusal-only)."] if null_sum else []),
                     null_baseline=null_sum)
    open(os.path.join(cfg["out"], "report.md"), "w").write(md)
    print(md)
    return summary


if __name__ == "__main__":
    main()
