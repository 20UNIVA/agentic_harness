#!/usr/bin/env python
"""Example MCP server #1: a Kanban project board (stateful).

  python kanban_server.py --state /tmp/kanban.json --seed 7                 # stdio (default)
  python kanban_server.py --state /tmp/kanban.json --transport http --port 8771

Tools: list_members, list_cards, get_card, create_card, move_card, assign_card, add_comment.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worlds import COLUMNS, build_kanban as build_world  # noqa: E402
from _common import StateFile, make_server, parse_args, read_only, run, tool_error, write_tool  # noqa: E402



def create_server(state: StateFile):
    srv = make_server("kanban", "Kanban board of the Acme platform team. Columns are todo, in_progress, review and done. "
                                "Assignees are referenced by username (see list_members). Card ids look like CARD-101.")

    def _card(st, card_id):
        c = st["cards"].get(card_id.strip().upper())
        if not c:
            raise tool_error(f"card {card_id!r} not found")
        return c

    def _member(st, username):
        u = username.strip().lower()
        if u not in {m["username"] for m in st["members"]}:
            raise tool_error(f"unknown member {username!r}; use list_members")
        return u

    @srv.tool(annotations=read_only())
    def list_members() -> list[dict]:
        """List team members (username and full name)."""
        return state.read()["members"]

    @srv.tool(annotations=read_only())
    def list_cards(column: str | None = None, assignee: str | None = None) -> list[dict]:
        """List cards, optionally filtered by column (todo|in_progress|review|done) and/or assignee username."""
        st = state.read()
        out = []
        for c in st["cards"].values():
            if column and c["column"] != column.strip().lower():
                continue
            if assignee and (c["assignee"] or "") != assignee.strip().lower():
                continue
            out.append({k: c[k] for k in ("card_id", "title", "column", "assignee", "due")})
        return out

    @srv.tool(annotations=read_only())
    def get_card(card_id: str) -> dict:
        """Get one card with labels, due date and comments."""
        return _card(state.read(), card_id)

    @srv.tool(annotations=write_tool())
    def create_card(title: str, column: str = "todo", assignee: str | None = None, labels: list[str] | None = None,
                    due: str | None = None) -> dict:
        """Create a card. due is YYYY-MM-DD. Returns the new card (with its card_id)."""
        def fn(st):
            col = column.strip().lower()
            if col not in COLUMNS:
                raise tool_error(f"column must be one of {COLUMNS}")
            cid = f"CARD-{st['next_id']}"
            st["next_id"] += 1
            st["cards"][cid] = {"card_id": cid, "title": title.strip(), "column": col,
                                "assignee": _member(st, assignee) if assignee else None,
                                "labels": sorted(set(labels or [])), "due": due, "comments": []}
            return st["cards"][cid]
        return state.update(fn)

    @srv.tool(annotations=write_tool())
    def move_card(card_id: str, column: str) -> dict:
        """Move a card to another column (todo|in_progress|review|done)."""
        def fn(st):
            col = column.strip().lower()
            if col not in COLUMNS:
                raise tool_error(f"column must be one of {COLUMNS}")
            c = _card(st, card_id)
            c["column"] = col
            return c
        return state.update(fn)

    @srv.tool(annotations=write_tool())
    def assign_card(card_id: str, assignee: str) -> dict:
        """Assign a card to a member by username."""
        def fn(st):
            c = _card(st, card_id)
            c["assignee"] = _member(st, assignee)
            return c
        return state.update(fn)

    @srv.tool(annotations=write_tool())
    def add_comment(card_id: str, text: str) -> dict:
        """Add a comment to a card."""
        def fn(st):
            c = _card(st, card_id)
            c["comments"].append(text.strip())
            return c
        return state.update(fn)

    return srv


if __name__ == "__main__":
    a = parse_args("kanban_state.json")
    run(create_server(StateFile(a.state, build_world, a.seed)), a)
