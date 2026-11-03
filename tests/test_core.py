"""Core tests: parser, environments, generation gates, metrics, export, skills, MCP, ms-swift plugin.

Run with `pytest tests/` or the stdlib runner `python tests/run_tests.py [--full]`.
Tests that need a Qwen3.5-format checkpoint/tokenizer (HARNESS_TEST_CKPT), ms-swift or the `mcp` SDK skip themselves.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _m in ("fla", "causal_conv1d"):
    sys.modules.setdefault(_m, None)

from agentic_harness.core.runner import run_episodes  # noqa: E402
from agentic_harness.core.toolcall import parse_assistant, render_tool_calls  # noqa: E402
from agentic_harness.export.swift import select_for_sft, to_step_dpo_pairs, to_swift_grpo_row, to_swift_sft  # noqa: E402
from agentic_harness.policies.oracle import NoisyOraclePolicy, NullPolicy, OraclePolicy  # noqa: E402
from agentic_harness.scoring.metrics import pass_at_k, pass_hat_k, summarize  # noqa: E402
from agentic_harness.tasks import DOMAIN_GENERATORS, generate  # noqa: E402

from _util import CK, HAS_MCP_SDK, needs_ckpt, needs_mcp_sdk, needs_swift  # noqa: E402
SCHEMA = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
    "s": {"type": "string"}, "i": {"type": "integer"}, "n": {"type": "number"}, "b": {"type": "boolean"},
    "a": {"type": "array"}, "o": {"type": "object"}}}}}]
_TASKS = None


def all_tasks():
    global _TASKS
    if _TASKS is None:
        _TASKS = list(generate("dev", 2))
    return _TASKS


def tasks():
    """Generated tasks that can run here: tasks using the example MCP servers need the `mcp` SDK."""
    return all_tasks() if HAS_MCP_SDK else [t for t in all_tasks() if not t.get("mcp_servers")]


# ----------------------------------------------------------------------------- parser
def test_parse_roundtrip_types():
    args = {"s": "hello\nworld", "i": 3, "n": 2.5, "b": True, "a": [1, "x"], "o": {"k": [1]}}
    raw = render_tool_calls([{"name": "f", "arguments": args}], "Let me check.")
    pr = parse_assistant(raw + "<|im_end|>", SCHEMA)
    assert pr.content == "Let me check.", pr.content
    assert len(pr.tool_calls) == 1 and pr.tool_calls[0].arguments == args, pr.tool_calls
    assert not pr.errors, pr.errors


def test_parse_multiple_and_missing_close():
    raw = render_tool_calls([{"name": "f", "arguments": {"i": 1}}, {"name": "f", "arguments": {"i": 2}}])
    pr = parse_assistant(raw, SCHEMA)
    assert [c.arguments["i"] for c in pr.tool_calls] == [1, 2]
    pr = parse_assistant("<tool_call>\n<function=f>\n<parameter=i>\n7\n</parameter>\n</function>", SCHEMA)  # no </tool_call>
    assert pr.tool_calls and pr.tool_calls[0].arguments == {"i": 7}


def test_parse_think_and_errors():
    pr = parse_assistant("plan...\n</think>\n\nanswer", SCHEMA, prompt_opened_think=True)
    assert pr.reasoning == "plan..." and pr.content == "answer"
    pr = parse_assistant("still thinking <tool_call>", SCHEMA, prompt_opened_think=True)
    assert pr.think_unclosed and "tool_call_inside_unclosed_think" in pr.errors and not pr.tool_calls
    pr = parse_assistant('<tool_call>\n{"name": "f", "arguments": {"i": 1}}\n</tool_call>', SCHEMA)
    assert pr.tool_calls and "json_style_tool_call" in pr.errors
    pr = parse_assistant("<function=f>\n</function>", SCHEMA)
    assert not pr.tool_calls and "function_tag_without_tool_call" in pr.errors


def test_bad_types_reach_env_validation():
    pr = parse_assistant("<tool_call>\n<function=f>\n<parameter=i>\nthree\n</parameter>\n</function>\n</tool_call>", SCHEMA)
    assert pr.tool_calls[0].arguments["i"] == "three"     # kept as string; Env reports the type error


# ----------------------------------------------------------------------------- gates
def test_gold_passes_all_templates():
    trajs = run_episodes(tasks(), OraclePolicy(), concurrency=128, log_every=0)
    bad = [t["task_id"] for t in trajs if t["outcome"]["reward"] != 1.0]
    assert not bad, bad
    n_tpl = sum(len(v) for v in DOMAIN_GENERATORS.values())
    assert len(all_tasks()) == 2 * n_tpl and len(trajs) == len(tasks())


def test_null_policy_fails_except_policy_category():
    trajs = run_episodes(tasks(), NullPolicy(), concurrency=128, log_every=0)
    passed = [t for t in trajs if t["outcome"]["reward"] == 1.0]
    assert all(t["category"] == "policy" for t in passed), [t["task_id"] for t in passed if t["category"] != "policy"]


def test_noisy_oracle_detected():
    trajs = run_episodes(tasks(), NoisyOraclePolicy(p=1.0, seed=3), concurrency=128, log_every=0)
    rate = sum(t["outcome"]["reward"] == 1.0 for t in trajs) / len(trajs)
    assert rate < 0.6, rate


def test_split_disjoint_and_deterministic():
    a = {t["content_hash"] for t in generate("train", 2)}
    b = {t["content_hash"] for t in generate("test", 2)}
    assert not (a & b)
    assert [t["content_hash"] for t in generate("test", 1)] == [t["content_hash"] for t in generate("test", 1)]


def test_loop_abort():
    t = next(x for x in tasks() if x["domain"] == "retail" and x["category"] == "lookup")

    class Looper(OraclePolicy):
        def _one(self, ep):
            from agentic_harness.policies.base import turn_from_raw
            return turn_from_raw(render_tool_calls([{"name": "list_product_types", "arguments": {}}]), ep.tools,
                                 prompt_opened_think=False, finish_reason="stop", call_offset=len(ep.turns))
    tr = run_episodes([t], Looper(), log_every=0)[0]
    assert tr["outcome"]["status"] == "loop_abort", tr["outcome"]


# ----------------------------------------------------------------------------- metrics / export
def test_answer_date_formats():
    from agentic_harness.scoring.verify import answer_ok
    spec = {"kind": "text", "value": "2026-07-13"}
    for text in ("2026-07-13", "2026년 7월 13일", "July 13, 2026", "13 Jul 2026", "2026/7/13", "2026.07.13"):
        assert answer_ok(spec, f"released on {text}."), text
    for text in ("2026년 7월 3일", "June 13, 2026", "July 1, 2026", "2026-07-14"):
        assert not answer_ok(spec, f"released on {text}."), text


def test_metrics_formulas():
    assert abs(pass_at_k(4, 1, 1) - 0.25) < 1e-9 and abs(pass_at_k(4, 1, 2) - 0.5) < 1e-9
    assert pass_hat_k(4, 4, 4) == 1.0 and pass_hat_k(4, 3, 4) == 0.0 and abs(pass_hat_k(4, 2, 2) - 1 / 6) < 1e-9


def test_exports():
    trajs = run_episodes(tasks(), OraclePolicy(), concurrency=128, log_every=0)
    trajs += run_episodes(tasks(), NoisyOraclePolicy(p=1.0, seed=5), concurrency=128, log_every=0)
    sel = select_for_sft(trajs)
    rows = [to_swift_sft(t) for t in sel]
    assert rows and all(r["messages"][-1]["role"] == "assistant" for r in rows)
    assert any(m["role"] == "tool_call" for r in rows for m in r["messages"])
    json.loads(rows[0]["tools"])
    grpo = to_swift_grpo_row(tasks()[0])
    assert json.loads(grpo["harness_task"])["task_id"] == tasks()[0]["task_id"]
    assert to_step_dpo_pairs(trajs)
    s = summarize(trajs, ks=(1, 2))
    assert s["overall"]["n_tasks"] == len(tasks())


# ----------------------------------------------------------------------------- full (tokenizer / swift)
def test_prefix_tracking_with_real_template():
    needs_ckpt()
    """Feeding the oracle's raw text as 'generated' tokens must reproduce the re-rendered prompt exactly
    (no prefix mismatch) for single-query episodes; clarify episodes are allowed to branch."""
    from transformers import AutoTokenizer

    from agentic_harness.policies.base import PromptTracker, turn_from_raw
    tok = AutoTokenizer.from_pretrained(CK)

    class TokenOracle(OraclePolicy):
        def _one(self, ep):
            base = super()._one(ep)
            tr = ep.policy_state.setdefault("tracker", PromptTracker(tok))
            ids, mm = tr.build(ep)
            text = base.raw_text + "<|im_end|>"
            gen = tok.encode(text, add_special_tokens=False)
            tr.record(gen, text)
            return turn_from_raw(text, ep.tools, prompt_opened_think=False, finish_reason="stop",
                                 call_offset=len(ep.turns), prompt_token_ids=ids, completion_token_ids=gen, prefix_mismatch=mm)
    trajs = run_episodes(tasks(), TokenOracle(), concurrency=64, log_every=0)
    assert all(t["outcome"]["reward"] == 1.0 for t in trajs)
    bad = [t["task_id"] for t in trajs if t["category"] != "clarify" and t["outcome"]["components"]["n_prefix_mismatch"]]
    assert not bad, bad
    # spliced ids must equal tokenization of the full rendered text for the final turn (string-level identity)
    t = next(x for x in trajs if x["outcome"]["components"]["n_turns"] >= 3 and x["category"] != "clarify")
    last = t["turns"][-1]
    assert tok.decode(last["prompt_token_ids"]).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_swift_plugin_offline():
    needs_ckpt()
    needs_swift()
    """Drive AgentHarnessScheduler.run() with a scripted engine that replays oracle text."""
    from transformers import AutoTokenizer

    import agentic_harness.integrations.swift_plugin as sp
    from swift.infer_engine.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage,
                                             RequestConfig, RolloutInferRequest, UsageInfo)
    tok = AutoTokenizer.from_pretrained(CK)
    picked = [t for t in tasks() if t["category"] in ("write", "multi_write", "clarify", "recover", "lookup")][:12]

    class Engine:
        def __init__(self):
            self.cursor = {}

        async def infer_async(self, req, cfg, **kw):
            task = json.loads(req.data_dict["harness_task"])
            i = self.cursor.get(req.uuid, 0)
            self.cursor[req.uuid] = i + 1
            gold = task["gold_actions"]
            if i < len(gold):
                st = gold[i]
                text = st["ask"] if "ask" in st else render_tool_calls([st])
            else:
                text = task["gold_final"]
            ch = ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content=text), finish_reason="stop",
                                              token_ids=tok.encode(text + "<|im_end|>", add_special_tokens=False))
            return ChatCompletionResponse(model="fake", choices=[ch], usage=UsageInfo(0, 0, 0))

    sched = sp.AgentHarnessScheduler(infer_engine=Engine(), max_turns=14, tokenizer=tok)

    async def go():
        outs = []
        for k, t in enumerate(picked):
            row = to_swift_grpo_row(t)
            req = RolloutInferRequest(messages=row["messages"], tools=json.loads(row["tools"]),
                                      data_dict={"harness_task": row["harness_task"]}, uuid=f"u{k}")
            outs.append(await sched.run(req, RequestConfig(max_tokens=512)))
        return outs
    outs = asyncio.run(go())
    rewards = sp.HarnessReward()([""] * len(outs), rollout_infos=[o.rollout_infos for o in outs])
    assert rewards == [1.0] * len(outs), [(t["task_id"], o.rollout_infos) for t, o in zip(picked, outs)]
    # tool results are appended as role=tool and never part of the trained response ids
    assert any(m["role"] == "tool" for m in outs[0].messages)
    assert all(len(ids) == len(mask) for o in outs for ids, mask in zip(o.response_token_ids, o.response_loss_mask))


# ----------------------------------------------------------------------------- skills
def test_skill_registry_and_modes():
    import tempfile
    from agentic_harness.core.session import DEFAULT_SKILLS_DIR, ToolSession, build_resources
    from agentic_harness.skills.registry import Skill, SkillRegistry, parse_frontmatter
    reg = SkillRegistry([DEFAULT_SKILLS_DIR])
    assert reg.names() == ["csv-insights", "file-archiving", "fiscal-calendar", "travel-api"], reg.names()
    assert all(s.content_hash.startswith("sha256:") for s in reg.skills.values())
    # fallback frontmatter parser (no PyYAML)
    saved = sys.modules.get("yaml")
    sys.modules["yaml"] = None
    try:
        meta, body = parse_frontmatter("---\nname: x-y\ndescription: >\n  folded line one\n  line two\n---\nBody")
    finally:
        if saved is not None:
            sys.modules["yaml"] = saved
        else:
            sys.modules.pop("yaml", None)
    assert meta["name"] == "x-y" and meta["description"] == "folded line one line two" and body == "Body", meta
    # invalid skills are rejected
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "Bad_Name"))
    open(os.path.join(d, "Bad_Name", "SKILL.md"), "w").write("---\nname: Bad_Name\ndescription: x\n---\nhi")
    try:
        Skill.load(os.path.join(d, "Bad_Name"))
        raise AssertionError("invalid name accepted")
    except ValueError:
        pass
    task = next(t for t in tasks() if t["category"] == "skill" and t["domain"] == "travel_api")
    for mode, n_skill_tools, body_in_prompt in (("tool", 2, False), ("inline", 0, True), ("off", 0, False)):
        sess = ToolSession(task, build_resources(skills_dirs=[DEFAULT_SKILLS_DIR], skills_mode=mode))
        assert "- csv-insights" not in sess.system_prompt()       # only the task's skills are offered
        names = [x["function"]["name"] for x in sess.tool_schemas()]
        assert sum(n in ("load_skill", "read_skill_file") for n in names) == n_skill_tools, (mode, names)
        assert ("Read `reference/endpoints.md`" in sess.system_prompt()) == body_in_prompt, mode
        if mode == "tool":
            assert "- travel-api:" in sess.system_prompt()
            out, err = sess.call("read_skill_file", {"name": "travel-api", "path": "../../../etc/passwd"})
            assert err
            out, err = sess.call("load_skill", {"name": "travel-api"})
            assert not err and "reference/endpoints.md" in out and sess.skills.loaded == ["travel-api"]
        sess.close()


# ----------------------------------------------------------------------------- MCP
def test_mcp_stdio_session_and_checks():
    needs_mcp_sdk()
    from agentic_harness.core.session import ToolSession, default_resources
    task = next(t for t in all_tasks() if t["generator"]["name"] == "mcp_examples.kanban_move")
    sess = ToolSession(task, default_resources())
    proc = sess.mcp_sets[0].client.t.proc
    chk = task["expected"]["checks"][0]
    assert not sess.check(chk)                                  # not moved yet
    for st in task["gold_actions"]:
        out, err = sess.call(st["name"], st["arguments"])
        assert not err, out
    assert sess.check(chk)
    assert sess.is_mutating("mcp__kanban__move_card") and not sess.is_mutating("mcp__kanban__list_cards")
    d = sess.episode_dir
    sess.close()
    assert proc.poll() is not None and not os.path.exists(d)    # server stopped, sandbox removed


def test_mcp_http_transport():
    needs_mcp_sdk()
    import socket
    import subprocess
    import tempfile
    import time
    from agentic_harness.core.session import ToolSession, build_resources
    from agentic_harness.mcp.registry import HARNESS_ROOT
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    d = tempfile.mkdtemp()
    srv = subprocess.Popen([sys.executable, os.path.join(HARNESS_ROOT, "examples/mcp_servers/wiki_server.py"), "--seed", "7",
                            "--state", os.path.join(d, "wiki.json"), "--transport", "http", "--port", str(port)],
                           stdout=subprocess.DEVNULL, stderr=open(os.path.join(d, "err.log"), "w"))
    try:
        cfg = os.path.join(d, "http.json")
        json.dump({"mcpServers": {"wiki_http": {"type": "http", "url": f"http://127.0.0.1:{port}/mcp", "scope": "run"}}},
                  open(cfg, "w"))
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        res = build_resources(mcp_configs=[cfg])
        task = {"task_id": "http", "domain": "none", "world": {"seed": 7}, "system_prompt": "S", "tools": [],
                "mcp_servers": ["wiki_http"]}
        s1, s2 = ToolSession(task, res), ToolSession(task, res)
        assert s1.mcp_sets[0].client is s2.mcp_sets[0].client       # scope=run -> one shared connection
        out, err = s1.call("mcp__wiki_http__search_docs", {"query": "expense"})
        assert not err and "Expense policy" in out, out
        assert s1.mcp_sets[0].client.t.session_id is not None
        s1.close(), s2.close()
        res.close()
    finally:
        srv.terminate()
        srv.wait(timeout=10)


def test_mcp_server_failure_is_infra_not_policy():
    import tempfile
    from agentic_harness.core.session import build_resources
    d = tempfile.mkdtemp()
    cfg = os.path.join(d, "bad.json")
    json.dump({"mcpServers": {"broken": {"command": sys.executable, "args": ["-c", "import sys; sys.exit('no module named foo')"]}}},
              open(cfg, "w"))
    res = build_resources(mcp_configs=[cfg])
    task = dict(next(t for t in all_tasks() if t["generator"]["name"] == "mcp_examples.wiki_expense"), mcp_servers=["broken"])
    tr = run_episodes([task], OraclePolicy(), log_every=0, resources=res)[0]
    assert tr["outcome"]["reward"] is None and tr["outcome"]["status"] == "infra_fail", tr["outcome"]
    assert "no module named foo" in tr["outcome"]["components"]["error"], tr["outcome"]   # server stderr is surfaced


def test_swift_plugin_skills_and_mcp():
    needs_ckpt()
    needs_swift()
    needs_mcp_sdk()
    """GRPO scheduler with skill + MCP tasks: tools/system come from the session, reward from live MCP checks."""
    from transformers import AutoTokenizer

    import agentic_harness.integrations.swift_plugin as sp
    from swift.infer_engine.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage,
                                             RequestConfig, RolloutInferRequest, UsageInfo)
    tok = AutoTokenizer.from_pretrained(CK)
    names = ["mcp_examples.cross_meeting_card", "mcp_examples.memory_relation", "mcp_examples.kanban_bulk_move",
             "travel_api.skill_book_cheapest", "filesystem.skill_archive_csvs", "data_sql.skill_fiscal_quarter_sales"]
    picked = [next(t for t in all_tasks() if t["generator"]["name"] == n) for n in names]

    class Engine:
        def __init__(self):
            self.cursor = {}

        async def infer_async(self, req, cfg, **kw):
            task = json.loads(req.data_dict["harness_task"])
            i = self.cursor.get(req.uuid, 0)
            self.cursor[req.uuid] = i + 1
            gold = task["gold_actions"]
            text = (gold[i]["ask"] if "ask" in gold[i] else render_tool_calls([gold[i]])) if i < len(gold) else task["gold_final"]
            ch = ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content=text), finish_reason="stop",
                                              token_ids=tok.encode(text + "<|im_end|>", add_special_tokens=False))
            return ChatCompletionResponse(model="fake", choices=[ch], usage=UsageInfo(0, 0, 0))

    sched = sp.AgentHarnessScheduler(infer_engine=Engine(), max_turns=14, tokenizer=tok)

    async def go():
        outs = []
        for k, t in enumerate(picked):
            row = to_swift_grpo_row(t)
            req = RolloutInferRequest(messages=row["messages"], tools=json.loads(row["tools"]),
                                      data_dict={"harness_task": row["harness_task"]}, uuid=f"m{k}")
            out = await sched.run(req, RequestConfig(max_tokens=512))
            outs.append((req, out))
        return outs
    outs = asyncio.run(go())
    rewards = sp.HarnessReward()([""] * len(outs), rollout_infos=[o.rollout_infos for _, o in outs])
    assert rewards == [1.0] * len(outs), [(t["task_id"], o.rollout_infos) for t, (_, o) in zip(picked, outs)]
    tool_names = [{x["function"]["name"] for x in r.tools} for r, _ in outs]
    assert "mcp__kanban__create_card" in tool_names[0] and "list_events" in tool_names[0]
    assert "load_skill" in tool_names[3] and "# Skills" in outs[3][0].messages[0]["content"]
