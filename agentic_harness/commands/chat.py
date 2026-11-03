"""`agentic-harness chat`: talk to an agent yourself; every conversation is saved as a trajectory.

  agentic-harness chat --provider openai_compatible --base-url http://127.0.0.1:8000/v1 --model my-model \\
      --toolkits examples/tools/finance_tools.py --out runs/chat
  agentic-harness chat --config examples/configs/openai_quickstart.json        # reuse a run config (model + resources)
  agentic-harness chat --provider anthropic --model claude-opus-5-5 --env retail --seed 3
  agentic-harness chat --ui --config examples/configs/openai_quickstart.json   # same, in the browser (127.0.0.1:7860)

The agent gets every registered toolkit, skill and MCP server unless you pick some with --use-toolkits / --use-skills /
--use-mcp. Type /help in the chat for commands (/good, /bad, /new, /tools, ...). Conversations are appended to
<out>/trajectories.jsonl (view them with `agentic-harness view`, export rated ones with `agentic-harness export sft`)
and their user turns to <out>/tasks.jsonl (replay them with another model via `agentic-harness run --tasks`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ..chat import ChatModelError, ChatSession
from .run import add_model_args, add_resource_args, build_run_resources, resolve

HELP = """commands:
  /good [note]   rate this conversation good (used as an SFT example)   /bad [note]  rate it bad
  /note <text>   attach a note                                           /retry       ask the model again after an error
  /new           save this conversation and start a new one              /discard     drop it without saving
  /tools         list the tools the agent can call                       /system      show the system prompt
  /calls         tool calls actually executed in this conversation (check claims like "done!")
  /quit          save and exit (Ctrl-D also works; Ctrl-C stops the agent's current answer)"""

DEFAULTS = {"skills_mode": "tool", "allow_shell": False, "no_example_resources": False, "keep_episode_dirs": False,
            "max_tool_result_chars": 20000, "think": False, "out": "runs/chat", "max_steps": 12}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="agentic-harness chat", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="JSON/YAML run config (model and resources are used; tasks are ignored)")
    add_model_args(ap)
    add_resource_args(ap)
    g = ap.add_argument_group("chat")
    g.add_argument("--use-toolkits", nargs="*", help="toolkit names to offer (default: all registered)")
    g.add_argument("--use-skills", nargs="*", help="skill names to offer (default: all registered)")
    g.add_argument("--use-mcp", nargs="*", help="MCP server names to connect (default: all registered)")
    g.add_argument("--env", help="built-in environment to attach (retail, workspace, filesystem, travel_api, data_sql)")
    g.add_argument("--seed", type=int, help="world seed of --env and of the example MCP servers (default 0)")
    g.add_argument("--system-prompt", help="system prompt text, or @file to read it from a file")
    g.add_argument("--max-steps", type=int, help="model turns allowed per user message (default 12)")
    g.add_argument("--think", action="store_true", default=None, help="enable_thinking=True for hybrid-thinking models")
    g.add_argument("--show-reasoning", action="store_true", help="print the model's reasoning text")
    g.add_argument("--result-chars", type=int, default=600, help="tool result characters to print (the full result is saved)")
    g.add_argument("--out", help="output directory (default runs/chat; a config file's \"out\" is not used)")
    g = ap.add_argument_group("web UI")
    g.add_argument("--ui", action="store_true", help="serve a browser chat page instead of the terminal prompt")
    g.add_argument("--host", default="127.0.0.1", help="UI bind address (default 127.0.0.1; there is no authentication)")
    g.add_argument("--port", type=int, default=7860)
    return ap


class _Printer:
    def __init__(self, out, show_reasoning: bool, result_chars: int):
        self.out, self.show_reasoning, self.result_chars = out, show_reasoning, result_chars
        color = hasattr(out, "isatty") and out.isatty() and not os.environ.get("NO_COLOR")
        self.dim = (lambda s: f"\033[2m{s}\033[0m") if color else (lambda s: s)
        self.bold = (lambda s: f"\033[1m{s}\033[0m") if color else (lambda s: s)

    def _clip(self, s: str) -> str:
        s = s or ""
        return s if len(s) <= self.result_chars else s[: self.result_chars] + f" ... [{len(s) - self.result_chars} more chars]"

    def __call__(self, m: dict) -> None:
        w = self.out.write
        if m["role"] == "assistant":
            if self.show_reasoning and m.get("reasoning_content"):
                w(self.dim("  (thinking) " + self._clip(m["reasoning_content"]).replace("\n", "\n  ")) + "\n")
            if (m.get("content") or "").strip():
                w(self.bold("agent> ") + m["content"].strip() + "\n")
            for tc in m.get("tool_calls") or []:
                args = json.dumps(tc["function"]["arguments"], ensure_ascii=False)
                w(self.dim(f"  -> {tc['function']['name']} {args}") + "\n")
        elif m["role"] == "tool":
            prefix = "  ! harness: " if m.get("name") == "harness" else "  <- "
            w(self.dim(prefix + self._clip(m["content"]).replace("\n", "\n     ")) + "\n")
        self.out.flush()


def _describe(chat: ChatSession, cfg: dict) -> str:
    by_src: dict = {}
    sess = chat._episode().session
    for t in chat.tools:
        src = sess.source(t["function"]["name"])
        by_src.setdefault(src, []).append(t["function"]["name"])
    parts = [f"{src}: {len(names)}" for src, names in sorted(by_src.items())]
    m = cfg["model"]
    return (f"model: {m.get('provider')} {m.get('model') or ''}".rstrip() + f" | tools: {len(chat.tools)} (" +
            (", ".join(parts) or "none") + f") | saving to {os.path.join(cfg['out'], 'trajectories.jsonl')}")


def main(argv=None, *, input_fn=input, out=None):
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    cfg = resolve(args, need_tasks=False, defaults=DEFAULTS)
    cfg["out"] = args.out or "runs/chat"        # a run config's "out" holds batch results; keep chats separate
    sp = cfg.get("system_prompt")
    if sp and sp.startswith("@"):
        sp = open(sp[1:], encoding="utf-8").read()
    env = None
    if cfg.get("env") or cfg.get("seed") is not None:
        env = {"domain": cfg.get("env") or "none", "seed": int(cfg.get("seed") or 0)}
    resources = build_run_resources(cfg)

    def make_session(on_message=None, model=None):
        return ChatSession(model or cfg["model"], resources=resources, toolkits=cfg.get("use_toolkits"),
                           skills=cfg.get("use_skills"), mcp_servers=cfg.get("use_mcp"), env=env, system_prompt=sp,
                           enable_thinking=bool(cfg.get("think")), max_steps=int(cfg.get("max_steps") or 12),
                           out_dir=cfg["out"], on_message=on_message)
    if args.ui:
        return _serve_ui(args, cfg, make_session, resources, out)
    printer = _Printer(out, bool(cfg.get("show_reasoning")), int(cfg.get("result_chars", 600)))
    chat = make_session(printer)
    w = out.write
    saved = 0
    echo = input_fn is input and not sys.stdin.isatty()     # piped input: show the user turns in the transcript

    def save():
        nonlocal saved
        rated = chat.feedback.get("rating")
        tr = chat.save()
        if tr:
            saved += 1
            w(f"[saved {tr['task_id']}: {tr['chat']['n_user_messages']} user messages, rating={rated}]\n")
        return tr

    try:
        header = _describe(chat, cfg)
    except RuntimeError as e:
        chat.close()
        resources.close()
        w(f"[error: {e}]\nCheck the setup with `agentic-harness resources`; connect fewer MCP servers with --use-mcp NAME ... "
          "(or --use-mcp with no names), or skip the bundled examples with --no-example-resources.\n")
        return 0
    try:
        w(header + "\nType a message, or /help.\n")
        while True:
            try:
                line = input_fn("you> ")
            except EOFError:
                w("\n")
                break
            text = line.strip()
            if echo:
                w(text + "\n")
            if not text:
                continue
            if text.startswith("/"):
                cmd, _, rest = text.partition(" ")
                rest = rest.strip()
                if cmd in ("/quit", "/exit"):
                    break
                if cmd == "/help":
                    w(HELP + "\n")
                elif cmd in ("/good", "/bad"):
                    chat.rate(cmd[1:], rest or None)
                    w(f"[rated {cmd[1:]}]\n")
                elif cmd == "/note" and rest:
                    chat.note(rest)
                    w("[noted]\n")
                elif cmd in ("/new", "/save"):
                    if not save():
                        w("[nothing to save]\n")
                    w(_describe(chat, cfg) + "\n")
                elif cmd == "/discard":
                    chat.discard()
                    w("[discarded]\n")
                elif cmd == "/tools":
                    for t in chat.tools:
                        f = t["function"]
                        w(f"  {f['name']}: {(f.get('description') or '').splitlines()[0][:100] if f.get('description') else ''}\n")
                elif cmd == "/calls":
                    calls = chat.ep.executed if chat.ep and chat.user_messages else []
                    for c in calls:
                        w(f"  {'ok ' if c['ok'] else 'ERR'} {c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}\n")
                    if not calls:
                        w("  (no tool calls yet)\n")
                elif cmd == "/system":
                    w(chat.system_prompt + "\n")
                elif cmd == "/retry":
                    try:
                        chat.retry()
                    except ChatModelError as e:
                        w(f"[model error: {e}]\n")
                else:
                    w(f"unknown command {cmd!r}\n{HELP}\n")
                continue
            try:
                chat.send(text)
            except ChatModelError as e:
                w(f"[model error: {e}] (/retry to ask again)\n")
                continue
            except RuntimeError as e:
                w(f"[error: {e}]\n")
                continue
            st = chat.status
            if st == "interrupted":
                w("[stopped by you; send a message to continue, or /retry]\n")
            elif st == "infra_fail":
                w(f"[tool infrastructure error: {chat.ep.infra_error}]\n")
                save()
            elif st != "awaiting_user":
                w(f"[agent stopped: {st}; send a message to continue]\n")
    except KeyboardInterrupt:
        w("\n")
    finally:
        save()
        chat.close()
        resources.close()
        w(f"[{saved} conversation(s) saved to {cfg['out']}]\n")
    return saved


def _serve_ui(args, cfg, make_session, resources, out):
    from ..policies import make_policy
    from ..web.server import ChatApp, serve
    policy = make_policy(cfg["model"])                  # one model client shared by all browser sessions
    m = cfg["model"]
    label = f"{m.get('provider')} {m.get('model') or ''}".strip()
    app = ChatApp(lambda: make_session(model=policy), model_label=label, out_dir=cfg["out"])
    try:
        app.state("probe")                                # start the tool session once so setup errors show up now
        app.sessions.pop("probe").close()
        app.saved.pop("probe", None)
    except RuntimeError as e:
        resources.close()
        out.write(f"[error: {e}]\n")
        return 0
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        out.write("WARNING: the chat UI has no authentication and the agent's tools run with your permissions.\n")

    def ready(srv):
        host, port = srv.server_address[:2]
        out.write(f"chat UI: http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/  "
                  f"(model: {label}; saving to {os.path.join(cfg['out'], 'trajectories.jsonl')}; Ctrl-C to stop)\n")
        out.flush()
    try:
        serve(app, args.host, args.port, ready=ready)
    finally:
        n = app.close()
        resources.close()
        out.write(f"[{n} open conversation(s) saved to {cfg['out']}]\n")
    return n


if __name__ == "__main__":
    main()
