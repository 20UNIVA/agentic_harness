"""Outcome scoring: state match + answer checks + forbidden actions."""
from __future__ import annotations

import re

import importlib
import importlib.util
import os

from ..core.types import InfraError, Outcome

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _numbers(s: str) -> list[float]:
    out = []
    for m in NUM_RE.findall(s):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december")


def _date_pattern(y: str, m: str, d: str) -> re.Pattern:
    """Common renderings of an ISO date: 2026-07-13, 2026/7/13, 2026.07.13, 2026년 7월 13일, July 13, 2026, 13 Jul 2026."""
    mi, di = int(m), int(d)
    mm, dd = f"0?{mi}", f"0?{di}"
    name = MONTHS[mi - 1]
    mon = f"(?:{name}|{name[:3]}\\.?)"
    alts = [
        rf"{y}\s*[-/.]\s*{mm}\s*[-/.]\s*{dd}(?!\d)",
        rf"{y}\s*년\s*{mm}\s*월\s*{dd}\s*일",
        rf"{mon}\s+{dd}(?:st|nd|rd|th)?,?\s+{y}",
        rf"(?<!\d){dd}(?:st|nd|rd|th)?\s+(?:of\s+)?{mon},?\s+{y}",
    ]
    return re.compile(r"(?<!\d)(?:" + "|".join(alts) + ")", re.IGNORECASE)


def answer_ok(spec: dict, final_text: str) -> bool:
    t = final_text or ""
    if spec["kind"] == "regex":
        return re.search(spec["value"], t, re.IGNORECASE | re.DOTALL) is not None
    if spec["kind"] == "number":
        return any(abs(x - float(spec["value"])) <= float(spec.get("tol", 0.01)) + 1e-9 for x in _numbers(t))
    cands = [spec["value"]] + list(spec.get("alts") or [])
    low = t.lower()
    for c in cands:
        if str(c).lower() in low:
            return True
        m = ISO_DATE_RE.fullmatch(str(c))
        if m and _date_pattern(*m.groups()).search(t):
            return True
    return False


def _get(args: dict, key: str):
    cur = args
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _match_value(actual, expected) -> bool:
    if isinstance(actual, list):
        return any(_match_value(a, expected) for a in actual)
    if isinstance(actual, str) and isinstance(expected, str):
        a, e = actual.strip().lower().lstrip("./"), expected.strip().lower().lstrip("./")
        return a == e or a.endswith("/" + e) or e.endswith("/" + a)
    return actual == expected


def forbidden_hit(rule: dict, call: dict) -> bool:
    if call["name"] != rule["name"]:
        return False
    for k, v in (rule.get("match") or {}).items():
        if not _match_value(_get(call.get("arguments") or {}, k), v):
            return False
    return True


_PY_CACHE: dict = {}


def load_verifier(ref: str):
    """'path/to/file.py:fn' or 'package.module:fn' -> callable."""
    if ref in _PY_CACHE:
        return _PY_CACHE[ref]
    mod_ref, fn_name = ref.rsplit(":", 1)
    if mod_ref.endswith(".py"):
        spec = importlib.util.spec_from_file_location("ah_verifier_" + str(abs(hash(mod_ref))), os.path.abspath(mod_ref))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_ref)
    _PY_CACHE[ref] = getattr(mod, fn_name)
    return _PY_CACHE[ref]


def has_verification(task: dict) -> bool:
    exp = task.get("expected") or {}
    return bool(exp.get("state_hash") or exp.get("answers") or exp.get("answer_forbidden") or exp.get("checks")
                or exp.get("python") or task.get("forbidden_actions"))


def score(task: dict, env, executed_calls: list[dict], final_text: str, *, ended_normally: bool,
          stats: dict) -> Outcome:
    """`env` is anything with state_hash() / check() / is_mutating() (a ToolSession or an Env)."""
    exp = task["expected"]
    comp = dict(stats)
    if not has_verification(task):
        comp["termination"] = stats.get("termination") or ("ended" if ended_normally else None)
        return Outcome(reward=None, status="unscored", failure_kind=None, components=comp)
    comp["state_match"] = (env.state_hash() == exp["state_hash"]) if exp.get("state_hash") else None
    ans = exp.get("answers") or []
    comp["answer_match"] = all(answer_ok(a, final_text) for a in ans) if ans else None
    bad_ans = [s for s in exp.get("answer_forbidden") or [] if s and str(s).lower() in (final_text or "").lower()]
    comp["leaked"] = bool(bad_ans)
    hits = [c for c in executed_calls if any(forbidden_hit(r, c) for r in task.get("forbidden_actions") or [])]
    comp["forbidden_called"] = bool(hits)
    checks = exp.get("checks") or []
    comp["checks_passed"] = sum(1 for c in checks if env.check(c)) if checks else None
    gold_writes = [s for s in task["gold_actions"] if "name" in s and env.is_mutating(s["name"]) and not s.get("expect_error")]
    if gold_writes:
        got = sum(1 for g in gold_writes if any(c["name"] == g["name"] and c.get("ok") and
                                                 all(_match_value(_get(c["arguments"], k), v) for k, v in g["arguments"].items()
                                                     if isinstance(v, (str, int, float)))
                                                 for c in executed_calls))
        comp["action_recall"] = got / len(gold_writes)
    ok = (comp["state_match"] is not False and comp["answer_match"] is not False and not comp["leaked"]
          and not comp["forbidden_called"] and (comp["checks_passed"] is None or comp["checks_passed"] == len(checks)))
    py_reward = None
    if exp.get("python"):
        try:
            r = load_verifier(exp["python"])(task=task, final_text=final_text, session=env, executed_calls=executed_calls)
        except Exception as e:                 # verifier bugs are harness failures, not policy failures
            raise InfraError(f"python verifier {exp['python']} raised {type(e).__name__}: {e}")
        if isinstance(r, dict):
            py_reward = float(r.get("reward", 0.0))
            comp["python_info"] = {k: v for k, v in r.items() if k != "reward"}
        else:
            py_reward = float(r)
        comp["python_reward"] = py_reward
    if not ended_normally:
        status = stats.get("termination", "max_turns")
        return Outcome(reward=0.0, status=status, failure_kind=status, components=comp)
    if ok and (py_reward is None or py_reward > 0):
        return Outcome(reward=1.0 if py_reward is None else max(0.0, min(1.0, py_reward)), status="ok", components=comp)
    if ok and py_reward is not None:
        return Outcome(reward=0.0, status="policy_fail", failure_kind="python_verifier", components=comp)
    kind = ("forbidden_action" if comp["forbidden_called"] else "leak" if comp["leaked"] else
            "wrong_state" if comp["state_match"] is False else "wrong_answer" if comp["answer_match"] is False else "check_failed")
    if stats.get("n_parse_errors") and not executed_calls:
        kind = "format_" + kind
    return Outcome(reward=0.0, status="policy_fail", failure_kind=kind, components=comp)
