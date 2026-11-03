"""`agentic-harness serve-hf`: minimal OpenAI-compatible server for a Hugging Face chat model (transformers).

For machines without vLLM (CPU, laptops) or for quick tests. Implements POST /v1/chat/completions (with `tools`,
returns `tool_calls`; non-streaming) and GET /v1/models. Requests are batched (up to --max-batch within --batch-window-ms).

Tool calls are parsed from the generated text with the harness parser (Qwen3.5 / Qwen3-Coder XML format; JSON-style
calls are also accepted), so it is meant for models whose chat template uses that format. For production serving
use vLLM / SGLang.

  agentic-harness serve-hf --model /path/to/hf_model --device cpu --port 8008
  agentic-harness run --provider openai_compatible --base-url http://127.0.0.1:8008/v1 --model local ...
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..core.toolcall import parse_assistant


class Generator:
    def __init__(self, model_path, device, threads, max_batch, window_ms):
        if device == "cpu":
            for m in ("fla", "causal_conv1d", "triton"):
                sys.modules.setdefault(m, None)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device).eval()
        self.stop_ids = [i for i in (self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>"))
                         if isinstance(i, int) and i >= 0] or [self.tok.eos_token_id]
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stop_ids[0]
        self.q: queue.Queue = queue.Queue()
        self.max_batch, self.window = max_batch, window_ms / 1000
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, prompt_ids, sampling) -> tuple:
        ev, box = threading.Event(), {}
        self.q.put((prompt_ids, sampling, ev, box))
        ev.wait()
        if "error" in box:
            raise RuntimeError(box["error"])
        return box["ids"], box["finish"]

    def _loop(self):
        while True:
            items = [self.q.get()]
            t_end = time.time() + self.window
            while len(items) < self.max_batch and time.time() < t_end:
                try:
                    items.append(self.q.get(timeout=max(0.0, t_end - time.time())))
                except queue.Empty:
                    break
            groups = {}
            for it in items:     # one generate() call per distinct sampling config
                groups.setdefault(json.dumps(it[1], sort_keys=True), []).append(it)
            for key, grp in groups.items():
                try:
                    self._run(grp, json.loads(key))
                except Exception as e:
                    for _, _, ev, box in grp:
                        box["error"] = f"{type(e).__name__}: {e}"
                        ev.set()

    def _run(self, grp, s):
        torch = self.torch
        L = max(len(g[0]) for g in grp)
        inp = torch.full((len(grp), L), self.pad, dtype=torch.long)
        att = torch.zeros((len(grp), L), dtype=torch.long)
        for i, g in enumerate(grp):
            inp[i, L - len(g[0]):] = torch.tensor(g[0])
            att[i, L - len(g[0]):] = 1
        temp = s.get("temperature", 0.7)
        kw = dict(max_new_tokens=s.get("max_tokens", 1024), do_sample=temp > 0, eos_token_id=self.stop_ids, pad_token_id=self.pad)
        if temp > 0:
            kw.update(temperature=temp, top_p=s.get("top_p", 1.0), top_k=s.get("top_k", 50))
        with torch.no_grad():
            out = self.model.generate(inp.to(self.model.device), attention_mask=att.to(self.model.device), **kw)
        for i, (_, _, ev, box) in enumerate(grp):
            gen, finish = out[i, L:].tolist(), "length"
            for j, t in enumerate(gen):
                if t in self.stop_ids:
                    gen, finish = gen[:j], "stop"
                    break
            box["ids"], box["finish"] = gen, finish
            ev.set()


def to_template_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m["tool_calls"] = [{"type": "function", "function": {"name": tc["function"]["name"], "arguments":
                                (json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str)
                                 else tc["function"]["arguments"])}} for tc in m["tool_calls"]]
        if isinstance(m.get("content"), list):
            m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
        out.append(m)
    return out


def make_handler(gen: Generator, served_name: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._send(200, {"object": "list", "data": [{"id": served_name, "object": "model"}]})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._send(404, {"error": {"message": "not found"}})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if req.get("stream"):
                    return self._send(400, {"error": {"message": "stream=true is not supported"}})
                tools = req.get("tools") or None
                think = bool((req.get("chat_template_kwargs") or {}).get("enable_thinking", False))
                text = gen.tok.apply_chat_template(to_template_messages(req["messages"]), tools=tools, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=think)
                ids = gen.tok.encode(text, add_special_tokens=False)
                samp = {k: req[k] for k in ("temperature", "top_p", "top_k") if req.get(k) is not None}
                samp["max_tokens"] = req.get("max_tokens") or req.get("max_completion_tokens") or 1024
                out_ids, finish = gen.submit(ids, samp)
                raw = gen.tok.decode(out_ids, skip_special_tokens=False)
                pr = parse_assistant(raw, tools or [], prompt_opened_think=think)
                msg = {"role": "assistant", "content": pr.content}
                if pr.reasoning:
                    msg["reasoning_content"] = pr.reasoning
                if pr.tool_calls:
                    msg["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                                          "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                                         for c in pr.tool_calls]
                    finish = "tool_calls"
                self._send(200, {"id": f"chatcmpl-{uuid.uuid4().hex[:16]}", "object": "chat.completion",
                                 "created": int(time.time()), "model": served_name,
                                 "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                 "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out_ids),
                                           "total_tokens": len(ids) + len(out_ids)}})
            except Exception as e:
                self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentic-harness serve-hf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-name", default="local")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--batch-window-ms", type=int, default=50)
    a = ap.parse_args(argv)
    gen = Generator(a.model, a.device, a.threads, a.max_batch, a.batch_window_ms)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(gen, a.served_name))
    print(f"[serve-hf] {a.model} on http://{a.host}:{a.port}/v1 (served name {a.served_name!r})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
