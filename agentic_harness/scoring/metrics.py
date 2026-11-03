"""Aggregate metrics: pass@1, unbiased pass@k, tau-bench pass^k, breakdowns and failure taxonomy."""
from __future__ import annotations

from collections import Counter, defaultdict
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """tau-bench pass^k: probability that k i.i.d. attempts all succeed = C(c,k)/C(n,k)."""
    if n < k:
        return float("nan")
    return comb(c, k) / comb(n, k)


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(trajs: list[dict], ks=(1, 2, 4)) -> dict:
    scored = [t for t in trajs if t["outcome"]["reward"] is not None]
    infra = sum(1 for t in trajs if t["outcome"]["status"] == "infra_fail")
    unscored = sum(1 for t in trajs if t["outcome"]["status"] == "unscored")
    by_task = defaultdict(list)
    for t in scored:
        by_task[t["task_id"]].append(t["outcome"]["reward"] == 1.0)
    meta = {t["task_id"]: (t["domain"], t["category"], t["difficulty"], t["lang"]) for t in trajs}

    def block(task_ids):
        res = {"n_tasks": len(task_ids), "n_episodes": sum(len(by_task[x]) for x in task_ids)}
        for k in ks:
            res[f"pass@{k}"] = _mean([pass_at_k(len(by_task[x]), sum(by_task[x]), k) for x in task_ids])
            if k > 1:
                res[f"pass^{k}"] = _mean([pass_hat_k(len(by_task[x]), sum(by_task[x]), k) for x in task_ids])
        return res

    ids = sorted(by_task)
    out = {"overall": block(ids), "n_infra_fail": infra, "n_unscored": unscored}
    for name, idx in (("domain", 0), ("category", 1), ("difficulty", 2), ("lang", 3)):
        groups = defaultdict(list)
        for x in ids:
            groups[meta[x][idx]].append(x)
        out[f"by_{name}"] = {str(g): block(v) for g, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}
    # behaviour diagnostics (per turn / per episode)
    turns = [tu for t in scored for tu in t["turns"]]
    n_turns = max(len(turns), 1)
    calls = [c for t in scored for c in t.get("executed_calls", [])]
    tool_names = {t["task_id"]: {s["function"]["name"] for s in t["tools"]} for t in scored}
    unknown = sum(1 for t in scored for c in t.get("executed_calls", []) if c["name"] not in tool_names[t["task_id"]])
    irr = [t for t in scored if t["category"] == "irrelevant"]
    out["behaviour"] = {
        "turn_parse_error_rate": sum(1 for tu in turns if any(e != "text_after_tool_call" for e in tu["parse_errors"])) / n_turns,
        "turns_with_tool_call_rate": sum(1 for tu in turns if tu["tool_calls"]) / n_turns,
        "tool_error_rate": (sum(1 for c in calls if not c["ok"]) / len(calls)) if calls else float("nan"),
        "unknown_tool_rate": (unknown / len(calls)) if calls else float("nan"),
        "think_unclosed_turn_rate": sum(1 for tu in turns if tu.get("think_unclosed")) / n_turns,
        "prefix_mismatch_turn_rate": sum(1 for tu in turns if tu.get("prefix_mismatch")) / n_turns,
        "mean_turns": _mean([t["outcome"]["components"].get("n_turns", 0) for t in scored]),
        "mean_tool_calls": _mean([t["outcome"]["components"].get("n_tool_calls", 0) for t in scored]),
        "irrelevant_any_tool_call_rate": _mean([1.0 if t.get("executed_calls") else 0.0 for t in irr]),
        "mean_completion_tokens_per_turn": _mean([len(tu["completion_token_ids"]) for tu in turns if tu.get("completion_token_ids")]),
    }
    by_src = Counter(c.get("source", "env").split(":")[0] for t in scored for c in t.get("executed_calls", []))
    mcp_calls = [c for t in scored for c in t.get("executed_calls", []) if c.get("source", "").startswith("mcp")]
    need = [t for t in scored if t.get("required_skill")]
    out["behaviour"].update({
        "calls_by_source": " ".join(f"{k}={v}" for k, v in sorted(by_src.items())) or "-",
        "mcp_tool_error_rate": (sum(1 for c in mcp_calls if not c["ok"]) / len(mcp_calls)) if mcp_calls else float("nan"),
        "skill_task_load_rate": _mean([1.0 if any(c["name"] == "load_skill" and (c.get("arguments") or {}).get("name") == t["required_skill"]
                                                  for c in t.get("executed_calls", [])) else 0.0 for t in need]),
        "skill_task_pass@1_when_loaded": _mean([1.0 if t["outcome"]["reward"] == 1.0 else 0.0 for t in need
                                                 if any(c["name"] == "load_skill" for c in t.get("executed_calls", []))]),
    })
    usage = Counter()
    for t in trajs:
        for tu in t["turns"]:
            u = (tu.get("provider_meta") or {}).get("usage") or {}
            usage["input_tokens"] += int(u.get("input_tokens") or u.get("prompt_tokens") or 0)
            usage["output_tokens"] += int(u.get("output_tokens") or u.get("completion_tokens") or 0)
            usage["cache_read_input_tokens"] += int(u.get("cache_read_input_tokens") or 0)
    if any(usage.values()):
        out["usage"] = dict(usage)
    out["status"] = dict(Counter(t["outcome"]["status"] for t in trajs))
    out["failure_kind"] = dict(Counter(t["outcome"]["failure_kind"] for t in scored if t["outcome"]["reward"] != 1.0))
    return out


def _fmt(v):
    return "-" if v is None or v != v else (f"{v:.3f}" if isinstance(v, float) else str(v))


def to_markdown(summary: dict, title: str, extra_lines: list[str] | None = None, null_baseline: dict | None = None) -> str:
    ks = [k for k in summary["overall"] if k.startswith("pass")]
    lines = [f"# {title}", ""] + (extra_lines or []) + [""]
    lines += ["| group | n_tasks | n_ep | " + " | ".join(ks) + (" | null pass@1 |" if null_baseline else " |"),
              "|---|---|---|" + "---|" * len(ks) + ("---|" if null_baseline else "")]

    def row(label, b, nb=None):
        r = f"| {label} | {b['n_tasks']} | {b['n_episodes']} | " + " | ".join(_fmt(b[k]) for k in ks) + " |"
        if null_baseline is not None:
            r += f" {_fmt(nb['pass@1']) if nb else '-'} |"
        return r

    lines.append(row("**overall**", summary["overall"], (null_baseline or {}).get("overall")))
    for sec in ("by_domain", "by_category", "by_difficulty", "by_lang"):
        for g, b in summary[sec].items():
            nb = (null_baseline or {}).get(sec, {}).get(g)
            lines.append(row(f"{sec[3:]}={g}", b, nb))
    lines += ["", "## Behaviour", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in summary["behaviour"].items()]
    lines += ["", "## Termination / failure kinds", "", f"- status: {summary['status']}", f"- failure_kind: {summary['failure_kind']}",
              f"- infra_fail (excluded from denominators): {summary['n_infra_fail']}",
              f"- unscored (task has no verification): {summary.get('n_unscored', 0)}"]
    if summary.get("usage"):
        lines += [f"- token usage (from provider responses): {summary['usage']}"]
    return "\n".join(lines) + "\n"
