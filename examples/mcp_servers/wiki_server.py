#!/usr/bin/env python
"""Example MCP server #2: a read-only company wiki / knowledge base.

  python wiki_server.py --seed 7                                   # stdio
  python wiki_server.py --seed 7 --transport http --port 8772      # Streamable HTTP (shared, scope "run")

Tools: search_docs, get_doc, list_spaces. All tools are read-only.
Every fact is generated from the seed, so answers can be checked exactly.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worlds import build_wiki as build_world  # noqa: E402
from _common import StateFile, make_server, parse_args, read_only, run, tool_error  # noqa: E402



def create_server(state: StateFile):
    srv = make_server("wiki", "Acme internal wiki. Search first with search_docs, then read the full page with get_doc. "
                              "Spaces: policies, oncall, offices, products.")

    @srv.tool(annotations=read_only())
    def list_spaces() -> list[str]:
        """List wiki spaces."""
        return sorted({d["space"] for d in state.read()["docs"]})

    @srv.tool(annotations=read_only())
    def search_docs(query: str, limit: int = 5) -> list[dict]:
        """Keyword search over titles and bodies. Returns doc ids, titles and a snippet."""
        words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 1]
        if not words:
            raise tool_error("query must contain at least one word")
        scored = []
        for d in state.read()["docs"]:
            text = (d["title"] + " " + d["body"]).lower()
            score = sum(text.count(w) for w in words) + 3 * sum(w in d["title"].lower() for w in words)
            if score:
                scored.append((-score, d["doc_id"], d))
        scored.sort()
        return [{"doc_id": d["doc_id"], "title": d["title"], "space": d["space"], "snippet": d["body"][:80] + "..."}
                for _, _, d in scored[:max(1, min(limit, 10))]]

    @srv.tool(annotations=read_only())
    def get_doc(doc_id: str) -> dict:
        """Read a full wiki page by id (e.g. DOC-003)."""
        for d in state.read()["docs"]:
            if d["doc_id"] == doc_id.strip().upper():
                return d
        raise tool_error(f"doc {doc_id!r} not found")

    return srv


if __name__ == "__main__":
    a = parse_args("wiki_state.json")
    run(create_server(StateFile(a.state, build_world, a.seed)), a)
