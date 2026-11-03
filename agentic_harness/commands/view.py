"""`agentic-harness view`: render trajectories as readable Markdown transcripts.

  agentic-harness view runs/x/trajectories.jsonl                 # first trajectory
  agentic-harness view runs/x/trajectories.jsonl --task-id t1 --attempt 0
  agentic-harness view runs/x/trajectories.jsonl --failed --limit 5 --out failures.md
"""
from __future__ import annotations

import argparse
import json


def _clip(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + f" ...[{len(s) - n} more chars]"


def render(t: dict, max_chars: int = 1500, show_system: bool = False) -> str:
    o = t["outcome"]
    lines = [f"## {t['task_id']} (attempt {t['attempt']})",
             f"- outcome: **{o['status']}**, reward={o['reward']}, failure={o['failure_kind']}",
             f"- policy: `{json.dumps(t['policy'], ensure_ascii=False)[:300]}`",
             f"- tools offered ({len(t['tools'])}): " + ", ".join(f"`{x['function']['name']}`" for x in t["tools"])]
    hf = t.get("human_feedback") or {}
    if hf.get("rating") or hf.get("notes"):
        lines.append(f"- human feedback: **{hf.get('rating')}**" + ("; " + " / ".join(hf["notes"]) if hf.get("notes") else ""))
    prov = t.get("provenance") or {}
    if (prov.get("skills") or {}).get("offered"):
        lines.append("- skills: " + ", ".join(s["name"] for s in prov["skills"]["offered"]) +
                     f" (loaded: {prov['skills'].get('loaded')})")
    if prov.get("mcp"):
        lines.append("- MCP servers: " + ", ".join(f"{m['server']} ({m.get('protocol')})" for m in prov["mcp"]))
    if prov.get("toolkits"):
        lines.append("- toolkits: " + ", ".join(k["name"] for k in prov["toolkits"]))
    lines.append("")
    for m in t["messages"]:
        r = m["role"]
        if r == "system":
            if show_system:
                lines += ["**system**", "", "~~~~~", _clip(m["content"], max_chars * 2), "~~~~~", ""]
        elif r == "user":
            lines += [f"**user**: {_clip(m['content'], max_chars)}", ""]
        elif r == "assistant":
            if m.get("reasoning_content"):
                lines += [f"> _reasoning_: {_clip(m['reasoning_content'], 400)}", ""]
            if (m.get("content") or "").strip():
                lines += [f"**assistant**: {_clip(m['content'], max_chars)}", ""]
            for tc in m.get("tool_calls") or []:
                args = json.dumps(tc["function"]["arguments"], ensure_ascii=False)
                lines += [f"**assistant -> `{tc['function']['name']}`** `{_clip(args, 600)}`", ""]
        elif r == "tool":
            body = _clip(m["content"], max_chars)
            fence = "~~~~~" if "```" in body else "```"
            lines += ["<details><summary>tool " + ("error" if m.get("is_error") else "result") +
                      (f" ({m.get('name')})" if m.get("name") else "") + "</summary>",
                      "", fence, body, fence, "</details>", ""]
    comp = {k: v for k, v in (o.get("components") or {}).items() if v not in (None, 0, False, [], {})}
    lines += [f"- components: `{json.dumps(comp, ensure_ascii=False, default=str)[:800]}`", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentic-harness view")
    ap.add_argument("trajectories")
    ap.add_argument("--task-id")
    ap.add_argument("--attempt", type=int)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--failed", action="store_true", help="only trajectories with reward != 1")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--system", action="store_true", help="also print the system prompt")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    trajs = [json.loads(line) for line in open(a.trajectories)]
    if a.task_id:
        trajs = [t for t in trajs if t["task_id"] == a.task_id and (a.attempt is None or t["attempt"] == a.attempt)]
    if a.failed:
        trajs = [t for t in trajs if t["outcome"]["reward"] != 1.0]
    if a.index is not None:
        trajs = trajs[a.index:a.index + 1]
    md = "\n---\n\n".join(render(t, a.max_chars, a.system) for t in trajs[:a.limit if a.index is None else 1])
    if a.out:
        open(a.out, "w").write(md + "\n")
        print(f"wrote {a.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
