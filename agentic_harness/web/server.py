"""Browser UI for `agentic-harness chat --ui`: the same ChatSession as the terminal chat, served with the stdlib.

One ChatSession per browser session id. Model calls are serialized (one answer at a time) so local backends such as
hf_local are never used from two threads at once. No authentication: it binds to 127.0.0.1 by default, and the agent's
tools run with your permissions, so do not expose it on a shared network.

API (JSON; `sid` identifies the browser session):
  GET  /                       the page
  GET  /api/state?sid=         model, tools, system prompt, current messages, status, rating, executed calls, saved list
  POST /api/send   {sid, text} NDJSON stream: {"type": "message", "message": {...}} ... {"type": "done", "state": {...}}
  POST /api/retry  {sid}       same stream, after a model error or Stop
  POST /api/stop   {sid}       stop before the next model turn
  POST /api/rate   {sid, rating: "good"|"bad"|null, note?}
  POST /api/note   {sid, note}
  POST /api/new    {sid}       save the conversation and start a new one
  POST /api/discard {sid}
"""
from __future__ import annotations

import json
import os
import re
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ..chat import ChatModelError, ChatSession

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.html")
_SID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ChatApp:
    def __init__(self, make_session, *, title: str = "agentic-harness chat", model_label: str = "", out_dir: str = ""):
        self.make_session = make_session          # () -> ChatSession
        self.title, self.model_label, self.out_dir = title, model_label, out_dir
        self.sessions: dict[str, ChatSession] = {}
        self.saved: dict[str, list[dict]] = {}
        self.lock = threading.Lock()              # protects the dicts
        self.gen_lock = threading.Lock()          # one model answer at a time

    def session(self, sid: str) -> ChatSession:
        with self.lock:
            if sid not in self.sessions:
                self.sessions[sid] = self.make_session()
                self.saved[sid] = []
            return self.sessions[sid]

    def state(self, sid: str) -> dict:
        chat = self.session(sid)
        st = {"title": self.title, "model": self.model_label, "out_dir": self.out_dir, "saved": self.saved.get(sid, [])}
        try:
            sess = chat._episode().session
            st["tools"] = [{"name": t["function"]["name"], "description": (t["function"].get("description") or "").strip(),
                            "source": sess.source(t["function"]["name"])} for t in chat.tools]
            st["system_prompt"] = chat.system_prompt
        except RuntimeError as e:
            st.update(tools=[], system_prompt="", setup_error=str(e))
        st.update(status=chat.status, messages=chat.messages[1:], rating=chat.feedback.get("rating"),
                  notes=chat.feedback.get("notes", []), calls=list(chat.ep.executed) if chat.ep and chat.user_messages else [],
                  infra_error=chat.ep.infra_error if chat.ep else None)
        return st

    def save(self, sid: str) -> dict | None:
        chat = self.session(sid)
        rating = chat.feedback.get("rating")
        tr = chat.save()
        if not tr:
            return None
        info = {"task_id": tr["task_id"], "rating": rating, "n_user_messages": tr["chat"]["n_user_messages"],
                "first_message": tr["messages"][1]["content"][:120], "final_status": tr["chat"]["final_status"]}
        self.saved[sid].append(info)
        return info

    def close(self) -> int:
        """Save every open conversation (server shutdown)."""
        n = 0
        for sid in list(self.sessions):
            if self.save(sid):
                n += 1
            self.sessions[sid].close()
        return n


def make_handler(app: ChatApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "agentic-harness-chat"

        def log_message(self, *a):
            pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        @staticmethod
        def _sid(v) -> str:
            v = (v or "").strip()
            if not _SID.match(v):
                raise ValueError("missing or invalid sid")
            return v

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path in ("/", "/index.html"):
                    body = open(PAGE, "rb").read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif u.path == "/api/state":
                    self._json(200, app.state(self._sid(parse_qs(u.query).get("sid", [""])[0])))
                else:
                    self._json(404, {"error": "not found"})
            except ValueError as e:
                self._json(400, {"error": str(e)})

        def _stream(self, sid: str, fn) -> None:
            """Run fn(chat) under the generation lock, streaming each new message as one NDJSON line."""
            chat = app.session(sid)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.close_connection = True

            def emit(obj):
                try:
                    self.wfile.write((json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode())
                    self.wfile.flush()
                except OSError:                      # browser went away; keep running, the state is kept
                    pass
            if not app.gen_lock.acquire(blocking=False):
                emit({"type": "info", "info": "waiting for another conversation's answer to finish"})
                app.gen_lock.acquire()
            try:
                chat.on_message = lambda m: emit({"type": "message", "message": m})
                try:
                    fn(chat)
                except ChatModelError as e:
                    emit({"type": "error", "error": f"model error: {e}", "retry": True})
                except RuntimeError as e:
                    emit({"type": "error", "error": str(e)})
                finally:
                    chat.on_message = None
            finally:
                app.gen_lock.release()
            emit({"type": "done", "state": app.state(sid)})

        def do_POST(self):
            u = urlparse(self.path)
            try:
                d = self._body()
                sid = self._sid(d.get("sid"))
                if u.path == "/api/send":
                    text = (d.get("text") or "").strip()
                    if not text:
                        raise ValueError("empty message")
                    self._stream(sid, lambda chat: chat.send(text))
                elif u.path == "/api/retry":
                    self._stream(sid, lambda chat: chat.retry())
                elif u.path == "/api/stop":
                    app.session(sid).request_stop()
                    self._json(200, {"ok": True})
                elif u.path == "/api/rate":
                    app.session(sid).rate(d.get("rating"), d.get("note") or None)
                    self._json(200, app.state(sid))
                elif u.path == "/api/note":
                    if d.get("note"):
                        app.session(sid).note(str(d["note"]))
                    self._json(200, app.state(sid))
                elif u.path == "/api/new":
                    info = app.save(sid)
                    self._json(200, {"saved": info, "state": app.state(sid)})
                elif u.path == "/api/discard":
                    app.session(sid).discard()
                    self._json(200, app.state(sid))
                else:
                    self._json(404, {"error": "not found"})
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e)})
    return Handler


def serve(app: ChatApp, host: str = "127.0.0.1", port: int = 7860, *, ready=None) -> None:
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    srv.daemon_threads = True
    if threading.current_thread() is threading.main_thread():     # SIGTERM -> same clean shutdown as Ctrl-C (saves chats)
        def _term(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _term)
    if ready:
        ready(srv)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
