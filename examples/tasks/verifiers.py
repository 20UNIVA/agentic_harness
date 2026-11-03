"""Example Python verifiers referenced from quickstart.json ("verify": {"python": "verifiers.py:<fn>"}).

A verifier is called after the episode as fn(task=..., final_text=..., session=..., executed_calls=...)
and returns a bool, a float in [0, 1] (partial credit) or {"reward": x, ...extra info}.
`session.check({...})` runs any built-in check (mcp_call / state / toolkit) and `session.call_mcp(server, tool, args)`
calls an MCP tool directly.
"""
import re


def three_primes(task, final_text, session, executed_calls):
    nums = {int(x) for x in re.findall(r"\b\d+\b", final_text or "")}
    primes = {n for n in nums if 100 < n < 150 and all(n % d for d in range(2, int(n ** 0.5) + 1))}
    return {"reward": 1.0 if len(primes) >= 3 else 0.0, "primes_found": sorted(primes)}


def budget_saved(task, final_text, session, executed_calls):
    # 900 USD at the finance toolkit's fixed rate (1 USD = 1380 KRW) = 1,242,000 KRW, stored on Lucia Rossi.
    ok = session.check({"type": "mcp_call", "server": "memory", "tool": "open_nodes", "arguments": {"names": ["Lucia Rossi"]},
                        "expect_contains": ["travel budget"]})
    text = session.call_mcp("memory", "open_nodes", {"names": ["Lucia Rossi"]})[0] if ok else ""
    amounts = [float(x.replace(",", "")) for x in re.findall(r"travel budget ([\d,\.]+) KRW", text)]
    return any(abs(a - 1242000) < 1 for a in amounts)
