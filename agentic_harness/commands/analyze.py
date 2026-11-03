"""Behavioural failure analysis of a rollout: classify each failed episode and print examples.

  agentic-harness analyze runs/<name>/trajectories.jsonl [--examples 2] > runs/<name>/analysis.md
"""
import argparse
import collections
import json


def classify(t):
    o = t["outcome"]
    c = o["components"]
    if o["reward"] == 1.0:
        return "success"
    if o["reward"] is None:
        return "infra"
    calls = t.get("executed_calls", [])
    if o["status"] == "loop_abort":
        return "loop(identical calls)"
    if o["status"] == "max_turns":
        return "max_turns(never finished)"
    if o["status"] == "truncated":
        return "truncated(max_new_tokens)"
    if c.get("forbidden_called"):
        return "policy_violation(forbidden call)"
    if c.get("leaked"):
        return "policy_violation(leaked info)"
    if not calls and t["category"] not in ("irrelevant", "policy"):
        return "no_tool_use(answered/refused without tools)"
    if c.get("n_parse_errors"):
        return "format_error(+wrong outcome)"
    if t["category"] == "irrelevant":
        return "irrelevant(wrong answer or used tools)"
    if c.get("state_match") is False:
        if t["outcome"]["components"].get("action_recall", 1.0) == 0.0 and not any(
                x["ok"] for x in calls if x["name"] not in ("get_order_details",)):
            return "wrong_state(write never succeeded)"
        return "wrong_state(wrong/missing/extra write)"
    if c.get("answer_match") is False:
        return "wrong_answer(after tool use)"
    return "other:" + str(o["failure_kind"])


def snippet(t, n_turns=4):
    out = [f"- task `{t['task_id']}` ({t['lang']}): {t['messages'][1]['content'][:220]}"]
    for tu in t["turns"][:n_turns]:
        out.append(f"  - turn: `{tu['raw_text'][:260]!r}`")
    out.append(f"  - outcome: {t['outcome']['status']} / {t['outcome']['failure_kind']} / "
               f"{ {k: v for k, v in t['outcome']['components'].items() if k in ('state_match', 'answer_match', 'n_tool_errors', 'n_parse_errors', 'n_turns')} }")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentic-harness analyze")
    ap.add_argument("traj")
    ap.add_argument("--examples", type=int, default=2)
    args = ap.parse_args(argv)
    trajs = [json.loads(line) for line in open(args.traj)]
    cls = collections.defaultdict(list)
    for t in trajs:
        cls[classify(t)].append(t)
    print(f"# Failure analysis: {args.traj}\n\n| class | n | share |\n|---|---|---|")
    for k, v in sorted(cls.items(), key=lambda kv: -len(kv[1])):
        print(f"| {k} | {len(v)} | {len(v) / len(trajs):.2f} |")
    by_cat = collections.defaultdict(collections.Counter)
    for k, v in cls.items():
        for t in v:
            by_cat[t["category"]][k] += 1
    print("\n## by category\n")
    for cat, cnt in sorted(by_cat.items()):
        print(f"- **{cat}**: {dict(cnt.most_common())}")
    errs = collections.Counter(e for t in trajs for tu in t["turns"] for e in tu["parse_errors"])
    tool_errs = collections.Counter()
    for t in trajs:
        for m in t["messages"]:
            if m["role"] == "tool" and m["content"].startswith('{"error"'):
                tool_errs[json.loads(m["content"])["error"].split(":")[0][:60]] += 1
    print(f"\n## parse errors (turn level)\n\n{dict(errs)}\n\n## tool error messages (top)\n")
    for k, v in tool_errs.most_common(12):
        print(f"- {v} × `{k}`")
    print("\n## examples\n")
    for k, v in sorted(cls.items(), key=lambda kv: -len(kv[1])):
        if k == "success":
            continue
        print(f"### {k}\n")
        for t in v[: args.examples]:
            print(snippet(t) + "\n")


if __name__ == "__main__":
    main()
