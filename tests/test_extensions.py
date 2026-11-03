"""Tests for user toolkits, the user task format, verifiers, the shell toolkit, model providers (against fake
OpenAI / Anthropic servers), the CLI and rendering consistency."""
import importlib.util
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from _util import CK, ROOT, needs_ckpt, needs_mcp_sdk, needs_swift, skip

from agentic_harness.commands import run as run_cmd
from agentic_harness.core.runner import run_episodes
from agentic_harness.core.session import ToolSession, build_resources
from agentic_harness.core.taskspec import load_task_file, normalize_task
from agentic_harness.policies import make_policy
from agentic_harness.policies.oracle import OraclePolicy
from agentic_harness.tools import Toolkit, infer_schema, load_toolkits, tool

EX = os.path.join(ROOT, "examples")
QS = os.path.join(EX, "tasks", "quickstart.json")
TOOLKITS = [os.path.join(EX, "tools", f) for f in ("finance_tools.py", "todo_toolkit.py", "library_toolkit.py")]


def _res(**kw):
    return build_resources(skills_dirs=[os.path.join(EX, "skills")], mcp_configs=[os.path.join(EX, "mcp", "servers.json")],
                           toolkit_specs=TOOLKITS, **kw)


# ----------------------------------------------------------------------------- tool definitions
def test_tool_decorator_schema_inference():
    from typing import Literal, Optional

    @tool(mutating=True, name="book")
    def f(city: str, nights: int, kind: Literal["hotel", "hostel"] = "hotel", tags: Optional[list[str]] = None,
          budget: float = 100.0):
        """Book a stay.

        Args:
            city: Destination city.
            nights: Number of nights.
        """
    meta = f._tooldef
    assert meta["name"] == "book" and meta["mutating"] and meta["description"] == "Book a stay."
    p = meta["params"]
    assert p["city"] == {"type": "string", "description": "Destination city."}
    assert p["nights"]["type"] == "integer" and p["kind"]["enum"] == ["hotel", "hostel"]
    assert p["tags"] == {"type": "array", "items": {"type": "string"}} and p["budget"]["default"] == 100.0
    assert meta["required"] == ["city", "nights"]
    props, req, desc = infer_schema(lambda self, x, y=1: None)
    assert req == ["x"] and "self" not in props


def test_user_toolkits_load_and_validate():
    tks = {}
    for spec in TOOLKITS:
        tks.update(load_toolkits(spec))
    assert sorted(tks) == ["finance", "library", "todo"]
    fin = tks["finance"]()
    fin.reset({"seed": 0})
    out, err = fin.call("loan_payment", {"principal": 1000, "annual_rate_pct": 0, "years": 1})
    assert not err and json.loads(out)["monthly_payment"] == 83.33
    out, err = fin.call("loan_payment", {"principal": "a lot", "annual_rate_pct": 1, "years": 1})
    assert err and "invalid type" in out                       # schema validation before the function runs
    lib = tks["library"]()
    lib.reset({"seed": 0})
    assert lib.call("checkout_book", {"book_id": "B3", "member": "bob"})[1]   # ToolError -> is_error
    assert lib.system_prompt.startswith("Library rule")

    class Broken(Toolkit):
        name = "broken"

        @tool
        def boom(self) -> str:
            """Always fails."""
            raise RuntimeError("bug in tool code")
    res = build_resources()
    res.toolkits["broken"] = Broken
    task = normalize_task({"instruction": "x", "toolkits": ["broken"], "verify": {"answers": ["y"]}})
    sess = ToolSession(task, res)
    try:
        sess.call("boom", {})
        raise AssertionError("tool bug must surface as InfraError")
    except Exception as e:
        assert type(e).__name__ == "InfraError"
    sess.close()


# ----------------------------------------------------------------------------- task format and verification
def test_task_format_validation():
    for bad, msg in (({"id": "x"}, "instruction"), ({"instruction": "a", "colour": 1}, "unknown fields"),
                     ({"instruction": "a", "verify": {"answer": ["x"]}}, "unknown verify")):
        try:
            normalize_task(bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError as e:
            assert msg in str(e), e
    t = normalize_task({"instruction": "cancel", "env": {"domain": "retail", "seed": 3},
                        "verify": {"expect_no_state_change": True}})
    assert t["expected"]["state_hash"] and t["tools"] and t["system_prompt"].startswith("You are a customer service agent")
    t2 = normalize_task({"instruction": "q", "verify": {"answers": [42, "Paris", {"regex": "^ok"}, {"number": 3.5, "tol": 0.1}]}})
    kinds = [a["kind"] for a in t2["expected"]["answers"]]
    assert kinds == ["number", "text", "regex", "number"]
    tasks = load_task_file(QS)
    assert all(os.path.isabs(t["expected"]["python"].split(":")[0]) for t in tasks if t["expected"].get("python"))


def test_quickstart_oracle_end_to_end_via_cli():
    needs_mcp_sdk()
    out = tempfile.mkdtemp()
    cfg = os.path.join(out, "cfg.json")
    json.dump({"model": {"provider": "oracle"}, "tasks": QS, "out": os.path.join(out, "run"), "toolkits": TOOLKITS,
               "allow_shell": True, "concurrency": 4}, open(cfg, "w"))
    summary = run_cmd.main(["--config", cfg])
    assert summary["overall"]["pass@1"] == 1.0 and summary["n_unscored"] == 1 and summary["n_infra_fail"] == 0, summary
    trajs = [json.loads(line) for line in open(os.path.join(out, "run", "trajectories.jsonl"))]
    t = next(x for x in trajs if x["task_id"] == "shell-skill-csv")
    assert t["provenance"]["skills"]["loaded"] == ["csv-insights"] and t["provenance"]["toolkits"][0]["name"] == "shell"
    from agentic_harness.commands.view import render
    md = render(t)
    assert "run_command" in md and "csv-insights" in md
    assert os.path.exists(os.path.join(out, "run", "report.md")) and os.path.exists(os.path.join(out, "run", "run_config.json"))


def test_shell_toolkit_is_opt_in_and_confined():
    task = normalize_task({"instruction": "x", "toolkits": ["shell"], "files": {"a/b.txt": "hello"}, "verify": {"answers": ["x"]}})
    tr = run_episodes([task], OraclePolicy(), log_every=0, resources=build_resources())[0]
    assert tr["outcome"]["status"] == "infra_fail" and "disabled" in tr["outcome"]["components"]["error"]
    sess = ToolSession(task, build_resources(allow_shell=True))
    assert json.loads(sess.call("run_command", {"command": "cat a/b.txt && pwd"})[0])["stdout"].startswith("hello")
    out, err = sess.call("read_text_file", {"path": "../../etc/passwd"})
    assert err and "inside the workspace" in out
    out, err = sess.call("run_command", {"command": "sleep 5", "timeout_s": 1})
    assert err and "timed out" in out
    ws = sess.toolkits[0].ws
    sess.close()
    assert not os.path.exists(ws)


def test_python_verifier_partial_and_errors():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "v.py"), "w").write(
        "def half(task, final_text, session, executed_calls):\n    return 0.5\n"
        "def boom(task, final_text, session, executed_calls):\n    raise ValueError('bad verifier')\n")
    tasks = [normalize_task({"id": "p", "instruction": "q", "gold_final": "a", "verify": {"python": "v.py:half"}}, 0, d),
             normalize_task({"id": "b", "instruction": "q", "gold_final": "a", "verify": {"python": "v.py:boom"}}, 1, d),
             normalize_task({"id": "u", "instruction": "q", "gold_final": "a"}, 2, d)]
    out = {t["task_id"]: t["outcome"] for t in run_episodes(tasks, OraclePolicy(), log_every=0, resources=build_resources())}
    assert out["p"]["reward"] == 0.5 and out["b"]["status"] == "infra_fail" and out["u"]["status"] == "unscored", out


# ----------------------------------------------------------------------------- model providers (fake servers)
class _Fake:
    """Tiny scripted HTTP server. `responses` is a list of dicts returned in order; requests are recorded."""

    def __init__(self, responses):
        self.responses, self.requests, self.headers = list(responses), [], []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                fake.requests.append((self.path, body))
                fake.headers.append(dict(self.headers))
                if not fake.responses:              # script exhausted: behave like a broken backend
                    self.send_response(400)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                resp = json.dumps(fake.responses.pop(0)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def _fin_task():
    return normalize_task({"id": "fin", "instruction": "Monthly payment for $1000 at 0% over 1 year?", "toolkits": ["finance"],
                           "verify": {"answers": [{"number": 83.33, "tol": 0.01}]}})


def test_openai_compatible_policy_against_fake_server():
    fake = _Fake([
        {"id": "c1", "model": "m", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "", "tool_calls": [{"id": "call_a", "type": "function", "function": {
                "name": "loan_payment", "arguments": json.dumps({"principal": 1000, "annual_rate_pct": 0, "years": 1})}}]}}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        {"id": "c2", "model": "m", "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "It is $83.33 per month."}}]}])
    os.environ["AH_TEST_KEY"] = "sk-test-123"
    try:
        pol = make_policy({"provider": "openai_compatible", "base_url": f"http://127.0.0.1:{fake.port}/v1", "model": "m",
                           "api_key_env": "AH_TEST_KEY", "sampling": {"temperature": 0.2, "max_new_tokens": 64}})
        tr = run_episodes([_fin_task()], pol, log_every=0, resources=_res())[0]
    finally:
        fake.close()
    assert tr["outcome"]["reward"] == 1.0, tr["outcome"]
    (p1, r1), (p2, r2) = fake.requests
    assert p1 == "/v1/chat/completions" and r1["temperature"] == 0.2 and r1["max_tokens"] == 64 and "top_k" not in r1
    assert r1["tools"][0]["function"]["name"] and r1["tool_choice"] == "auto" and "chat_template_kwargs" not in r1
    assert fake.headers[0]["Authorization"] == "Bearer sk-test-123"
    tool_msg = next(m for m in r2["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_a" and "83.33" in tool_msg["content"]
    assert r2["messages"][2]["tool_calls"][0]["function"]["arguments"] == json.dumps(
        {"principal": 1000, "annual_rate_pct": 0, "years": 1})
    assert tr["turns"][0]["provider_meta"]["usage"]["prompt_tokens"] == 10


def test_anthropic_policy_against_fake_server():
    if importlib.util.find_spec("anthropic") is None:
        skip("anthropic SDK not installed")
    thinking = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    fake = _Fake([
        {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "stop_reason": "tool_use",
         "content": [thinking, {"type": "text", "text": "Let me compute."},
                     {"type": "tool_use", "id": "toolu_1", "name": "loan_payment",
                      "input": {"principal": 1000, "annual_rate_pct": 0, "years": 1}}],
         "usage": {"input_tokens": 20, "output_tokens": 9}},
        {"id": "msg_2", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "stop_reason": "end_turn",
         "content": [{"type": "text", "text": "$83.33 per month."}], "usage": {"input_tokens": 40, "output_tokens": 5}}])
    try:
        pol = make_policy({"provider": "anthropic", "base_url": f"http://127.0.0.1:{fake.port}", "api_key": "test",
                           "effort": "low", "max_retries": 0})
        tr = run_episodes([_fin_task()], pol, log_every=0, resources=_res())[0]
    finally:
        fake.close()
    assert tr["outcome"]["reward"] == 1.0, tr["outcome"]
    (p1, r1), (p2, r2) = fake.requests
    assert p1.endswith("/v1/messages") and r1["model"] == "claude-opus-5-5" and r1["output_config"] == {"effort": "low"}
    assert not ({"temperature", "top_p", "top_k"} & set(r1)), "sampling params must not be sent"
    assert r1["tools"][0].keys() >= {"name", "description", "input_schema"} and r1["system"]
    assert r2["messages"][1] == {"role": "assistant", "content": [thinking, {"type": "text", "text": "Let me compute."},
                                                                  {"type": "tool_use", "id": "toolu_1", "name": "loan_payment",
                                                                   "input": {"principal": 1000, "annual_rate_pct": 0, "years": 1}}]}
    res_block = r2["messages"][2]["content"][0]
    assert res_block["type"] == "tool_result" and res_block["tool_use_id"] == "toolu_1" and "83.33" in res_block["content"]
    assert r1["messages"] == r2["messages"][:1]                      # append-only history
    assert tr["turns"][0]["provider_meta"]["stop_reason"] == "tool_use"


def test_custom_python_policy():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "pol.py"), "w").write(
        "from agentic_harness.policies.base import Policy, turn_from_raw\n"
        "class Echo(Policy):\n    name = 'echo'\n"
        "    def __init__(self, answer):\n        self.answer = answer\n"
        "    def generate_batch(self, eps):\n"
        "        return [turn_from_raw(self.answer, ep.tools, prompt_opened_think=False, finish_reason='stop') for ep in eps]\n")
    pol = make_policy({"provider": "python", "class": os.path.join(d, "pol.py") + ":Echo", "kwargs": {"answer": "Paris"}})
    task = normalize_task({"instruction": "Capital of France?", "verify": {"answers": ["Paris"]}})
    assert run_episodes([task], pol, log_every=0)[0]["outcome"]["reward"] == 1.0


# ----------------------------------------------------------------------------- training-format consistency
def test_swift_render_matches_inference_template():
    needs_ckpt()
    needs_swift()
    from swift.model import get_processor
    from swift.template import get_template

    from agentic_harness.export.swift import to_swift_sft
    from agentic_harness.policies.openai_compat import to_openai_messages
    from agentic_harness.tasks import generate
    proc = get_processor(CK)
    tpl = get_template(proc, template_type="qwen3_5")
    tpl.set_mode("train")
    tok = proc.tokenizer
    tasks = [t for t in generate("dev", 1) if t["category"] != "clarify" and not t.get("mcp_servers")]
    diff = []
    for t in run_episodes(tasks, OraclePolicy(), concurrency=64, log_every=0):
        row = to_swift_sft(t)
        s = tok.decode(tpl.encode({"messages": row["messages"], "tools": row["tools"]})["input_ids"], skip_special_tokens=False)
        msgs = to_openai_messages(t["messages"])
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
        j = tok.apply_chat_template(msgs, tools=t["tools"], tokenize=False)
        if s.strip() != j.strip():
            diff.append(t["task_id"])
    assert not diff, diff


# ----------------------------------------------------------------------------- MCP client robustness
def test_mcp_plain_server_env_isolation_and_truncation():
    d = tempfile.mkdtemp()
    server = os.path.join(ROOT, "tests", "fixtures", "plain_mcp_server.py")
    cfg = os.path.join(d, "servers.json")
    json.dump({"mcpServers": {
        "plain": {"command": "${PYTHON}", "args": [server], "env": {"VISIBLE": "yes"}},
        "plain_inherit": {"command": "${PYTHON}", "args": [server], "inherit_env": True}}}, open(cfg, "w"))
    os.environ["AH_TEST_SECRET"] = "s3cret"
    res = build_resources(mcp_configs=[cfg], max_tool_result_chars=1000)
    task = normalize_task({"instruction": "x", "mcp_servers": ["plain", "plain_inherit"], "verify": {"answers": ["x"]}})
    sess = ToolSession(task, res)
    names = [s["function"]["name"] for s in sess.tool_schemas()]
    assert "mcp__plain__get_env" in names and "mcp__plain__big" in names        # pagination followed
    assert "Plain test server." in sess.system_prompt()
    assert sess.call("mcp__plain__get_env", {"name": "VISIBLE"}) == ("yes", False)
    assert sess.call("mcp__plain__get_env", {"name": "AH_TEST_SECRET"}) == ("", False)    # secrets are not inherited
    assert sess.call("mcp__plain_inherit__get_env", {"name": "AH_TEST_SECRET"}) == ("s3cret", False)
    text, err = sess.call("mcp__plain__big", {"n": 5000})
    assert not err and len(text) < 1100 and "truncated: 4000 more characters" in text
    assert not sess.is_mutating("mcp__plain__get_env") and sess.is_mutating("mcp__plain__big")
    sess.close()


def test_third_party_skill_layout():
    d = tempfile.mkdtemp()
    sk = os.path.join(d, "pdf-tools")
    os.makedirs(os.path.join(sk, "scripts"))
    open(os.path.join(sk, "SKILL.md"), "w").write(
        "---\nname: pdf-tools\ndescription: Work with PDF files. Use when the user mentions PDFs.\nlicense: Apache-2.0\n"
        "metadata:\n  version: \"1.2\"\n---\n# PDF tools\nRun scripts/extract.py <file>.\n")
    open(os.path.join(sk, "scripts", "extract.py"), "w").write("print('ok')\n")
    open(os.path.join(sk, "scripts", ".hidden"), "w").write("x")
    from agentic_harness.skills.registry import SkillRegistry
    reg = SkillRegistry([d])
    s = reg.get("pdf-tools")
    assert s.files == ["scripts/extract.py"] and s.metadata.get("license") == "Apache-2.0"
    task = normalize_task({"instruction": "x", "skills": ["pdf-tools"], "toolkits": ["shell"], "verify": {"answers": ["x"]}})
    sess = ToolSession(task, build_resources(skills_dirs=[d], allow_shell=True))
    assert "skills/pdf-tools/" in sess.call("load_skill", {"name": "pdf-tools"})[0]
    assert json.loads(sess.call("run_command", {"command": "python3 skills/pdf-tools/scripts/extract.py"})[0])["stdout"] == "ok\n"
    sess.close()


def test_python_api():
    import agentic_harness
    trajs = agentic_harness.run_tasks(
        [{"id": "fx", "instruction": "250 EUR in KRW?", "toolkits": ["finance"],
          "gold_actions": [{"name": "convert_currency", "arguments": {"amount": 250, "source": "EUR", "target": "KRW"}}],
          "gold_final": "375,000 KRW", "verify": {"answers": [{"number": 375000, "tol": 1}]}}],
        model={"provider": "oracle"}, toolkits=[TOOLKITS[0]], n_attempts=2)
    s = agentic_harness.summarize(trajs, ks=(1, 2))
    assert s["overall"]["pass@1"] == 1.0 and s["overall"]["pass^2"] == 1.0 and agentic_harness.__version__ == "0.2.0"


def test_init_scaffold_runs():
    from agentic_harness.commands import init as init_cmd
    d = os.path.join(tempfile.mkdtemp(), "proj")
    init_cmd.main([d])
    summary = run_cmd.main(["--config", os.path.join(d, "run_oracle.json")])
    assert summary["overall"]["pass@1"] == 1.0 and summary["overall"]["n_tasks"] == 2


def test_vllm_completions_token_path_against_fake_server():
    needs_ckpt()
    from transformers import AutoTokenizer

    from agentic_harness.core.toolcall import render_tool_calls
    tok = AutoTokenizer.from_pretrained(CK)
    call = render_tool_calls([{"name": "loan_payment", "arguments": {"principal": 1000, "annual_rate_pct": 0, "years": 1}}])
    outs = [call + "<|im_end|>", "The payment is $83.33 per month.<|im_end|>"]
    resps = []
    for text in outs:
        ids = tok.encode(text, add_special_tokens=False)
        resps.append({"id": "cmpl", "model": "m", "choices": [{"index": 0, "text": text[:-len("<|im_end|>")], "token_ids": ids,
                                                                "logprobs": {"token_logprobs": [-0.1] * len(ids)},
                                                                "finish_reason": "stop"}]})
    fake = _Fake(resps)
    try:
        pol = make_policy({"provider": "vllm_completions", "base_url": f"http://127.0.0.1:{fake.port}", "model": "m",
                           "tokenizer": CK, "sampling": {"max_new_tokens": 64}})
        tr = run_episodes([_fin_task()], pol, log_every=0, resources=_res())[0]
    finally:
        fake.close()
    assert tr["outcome"]["reward"] == 1.0, tr["outcome"]
    (_, r1), (_, r2) = fake.requests
    assert isinstance(r1["prompt"], list) and all(isinstance(x, int) for x in r1["prompt"])
    t0, t1 = tr["turns"]
    assert t0["completion_logprobs"] == [-0.1] * len(t0["completion_token_ids"])
    assert r2["prompt"][: len(r1["prompt"]) + len(t0["completion_token_ids"])] == r1["prompt"] + t0["completion_token_ids"]
    assert t1["prefix_mismatch"] is False


# ----------------------------------------------------------------------------- interactive chat
def _chat_responses():
    call = {"id": "c1", "model": "m", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "", "tool_calls": [{"id": "call_a", "type": "function", "function": {
            "name": "loan_payment", "arguments": json.dumps({"principal": 1000, "annual_rate_pct": 0, "years": 1})}}]}}]}

    def say(text):
        return {"id": "c", "model": "m", "choices": [{"index": 0, "finish_reason": "stop",
                                                     "message": {"role": "assistant", "content": text}}]}
    return call, say


def test_chat_session_saves_rated_trajectory_and_replayable_task():
    from agentic_harness.chat import ChatSession
    from agentic_harness.export.swift import select_for_sft, to_swift_sft
    call, say = _chat_responses()
    fake = _Fake([call, say("It is $83.33 per month."), say("You're welcome.")])
    out = tempfile.mkdtemp()
    seen = []
    try:
        model = {"provider": "openai_compatible", "base_url": f"http://127.0.0.1:{fake.port}/v1", "model": "m"}
        chat = ChatSession(model, resources=_res(), mcp_servers=[], out_dir=out, on_message=seen.append)
        assert {"loan_payment", "load_skill"} <= {t["function"]["name"] for t in chat.tools} and chat.status == "idle"
        new = chat.send("Monthly payment for $1000 at 0% over 1 year?")
        assert [m["role"] for m in new] == ["assistant", "tool", "assistant"] and chat.status == "awaiting_user"
        assert new[-1]["content"] == "It is $83.33 per month." and seen == new
        chat.send("thanks!")
        chat.rate("good", "correct")
        tr = chat.save()
        assert chat.status == "idle" and chat.save() is None
        chat.close()
    finally:
        fake.close()
    assert [m["role"] for m in tr["messages"]] == ["system", "user", "assistant", "tool", "assistant", "user", "assistant"]
    assert tr["outcome"]["status"] == "unscored" and tr["human_feedback"]["rating"] == "good"
    assert tr["chat"]["n_user_messages"] == 2 and tr["executed_calls"][0]["name"] == "loan_payment"
    assert tr["provenance"]["toolkits"] and tr["category"] == "chat" and tr["messages"][3]["is_error"] is False
    saved = [json.loads(line) for line in open(os.path.join(out, "trajectories.jsonl"))]
    assert saved[0]["task_id"] == tr["task_id"]
    assert select_for_sft(saved) and to_swift_sft(saved[0])["messages"][-1]["content"] == "You're welcome."
    assert not select_for_sft([{**saved[0], "human_feedback": {"rating": "bad"}}])
    assert not select_for_sft([{**saved[0], "human_feedback": {}}])          # unrated chats are not training data
    # the user turns replay as a normal task with another model
    task = load_task_file(os.path.join(out, "tasks.jsonl"))[0]
    assert task["instruction"].startswith("Monthly") and task["user_followups"] == ["thanks!"]
    fake = _Fake([call, say("83.33 per month"), say("bye")])
    try:
        pol = make_policy({"provider": "openai_compatible", "base_url": f"http://127.0.0.1:{fake.port}/v1", "model": "m2"})
        rep = run_episodes([task], pol, log_every=0, resources=_res())[0]
    finally:
        fake.close()
    assert [m["role"] for m in rep["messages"]][1:] == ["user", "assistant", "tool", "assistant", "user", "assistant"]


def test_chat_cli_commands_and_model_error():
    from agentic_harness.commands import chat as chat_cmd
    import io
    call, say = _chat_responses()
    fake = _Fake([call, say("It is $83.33 per month.")])
    out = tempfile.mkdtemp()
    lines = iter(["/help", "/tools", "What is the payment on $1000 at 0% for a year?", "/calls", "/good looks right", "/new",
                  "second conversation", "/retry", "/discard", "/quit"])

    def fake_input(prompt):
        return next(lines)
    buf = io.StringIO()
    try:
        n = chat_cmd.main(["--provider", "openai_compatible", "--base-url", f"http://127.0.0.1:{fake.port}/v1", "--model", "m",
                           "--toolkits", TOOLKITS[0], "--use-mcp", "--out", out], input_fn=fake_input, out=buf)
    finally:
        fake.close()
    text = buf.getvalue()
    assert n == 1, text
    assert "-> loan_payment" in text and "agent> It is $83.33 per month." in text and "[rated good]" in text
    assert "ok  loan_payment" in text        # /calls lists what was really executed
    assert "model error" in text            # the fake server has no more responses: reported, conversation kept
    trajs = [json.loads(line) for line in open(os.path.join(out, "trajectories.jsonl"))]
    assert len(trajs) == 1 and trajs[0]["human_feedback"] == {"rating": "good", "notes": ["looks right"], "rated_by": "human"}
    from agentic_harness.commands.view import render
    assert "human feedback: **good**" in render(trajs[0])


def test_chat_web_ui_api_streams_and_saves():
    import urllib.error
    import urllib.request
    from agentic_harness.chat import ChatSession
    from agentic_harness.web.server import ChatApp, serve
    call, say = _chat_responses()
    fake = _Fake([call, say("It is $83.33 per month.")])
    out = tempfile.mkdtemp()
    pol = make_policy({"provider": "openai_compatible", "base_url": f"http://127.0.0.1:{fake.port}/v1", "model": "m"})
    res = _res()
    app = ChatApp(lambda: ChatSession(pol, resources=res, mcp_servers=[], out_dir=out), model_label="fake m", out_dir=out)
    box = {}
    ready = threading.Event()

    def on_ready(srv):
        box["srv"] = srv
        ready.set()
    threading.Thread(target=serve, args=(app, "127.0.0.1", 0), kwargs={"ready": on_ready}, daemon=True).start()
    ready.wait(10)
    base = f"http://127.0.0.1:{box['srv'].server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=30).read().decode()
    try:
        page = urllib.request.urlopen(base + "/", timeout=10).read().decode()
        assert "<title>agentic-harness chat</title>" in page and "/api/send" in page
        st = json.loads(urllib.request.urlopen(base + "/api/state?sid=t1", timeout=30).read())
        assert st["status"] == "idle" and any(t["name"] == "loan_payment" and t["source"] == "toolkit:finance" for t in st["tools"])
        events = [json.loads(line) for line in post("/api/send", {"sid": "t1", "text": "Payment on $1000 at 0% for 1 year?"}).splitlines()]
        kinds = [e["type"] for e in events]
        assert kinds == ["message", "message", "message", "done"], kinds
        assert [e["message"]["role"] for e in events[:3]] == ["assistant", "tool", "assistant"]
        done = events[-1]["state"]
        assert done["status"] == "awaiting_user" and done["calls"][0]["name"] == "loan_payment" and len(done["messages"]) == 4
        assert json.loads(post("/api/rate", {"sid": "t1", "rating": "good"}))["rating"] == "good"
        post("/api/note", {"sid": "t1", "note": "from the browser"})
        ev = [json.loads(line) for line in post("/api/send", {"sid": "t1", "text": "again"}).splitlines()]
        assert ev[0]["type"] == "error" and ev[0]["retry"] and ev[-1]["state"]["status"] == "running"   # backend exhausted
        saved = json.loads(post("/api/new", {"sid": "t1"}))
        assert saved["saved"]["rating"] == "good" and saved["state"]["status"] == "idle"
        assert json.loads(post("/api/stop", {"sid": "t1"}))["ok"]
        try:
            post("/api/send", {"sid": "../bad", "text": "x"})
            raise AssertionError("invalid sid accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        box["srv"].shutdown()
        app.close()
        res.close()
        fake.close()
    tr = [json.loads(line) for line in open(os.path.join(out, "trajectories.jsonl"))]
    assert len(tr) == 1 and tr[0]["human_feedback"]["notes"] == ["from the browser"]
    assert [m["role"] for m in tr[0]["messages"]][-1] == "user"        # the unanswered message is kept
