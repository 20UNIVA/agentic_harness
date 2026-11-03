#!/usr/bin/env python
"""Example MCP server #3: a knowledge-graph memory (entities, relations, observations).

The tool names and argument shapes follow the reference `@modelcontextprotocol/server-memory` server, so tasks
written against this example can in principle target that server too (not tested against it here).

  python memory_server.py --state /tmp/memory.json --seed 7

Tools: create_entities, create_relations, add_observations, delete_entities, delete_observations,
       delete_relations, read_graph, search_nodes, open_nodes.
"""
from __future__ import annotations

import os
import sys

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worlds import build_memory as build_world  # noqa: E402
from _common import StateFile, make_server, parse_args, read_only, run, tool_error, write_tool  # noqa: E402

class Entity(BaseModel):
    name: str = Field(description="Unique entity name, e.g. a person's full name")
    entityType: str = Field(description="e.g. person, organization, project")
    observations: list[str] = Field(default_factory=list, description="Facts about the entity")


class Relation(BaseModel):
    model_config = {"populate_by_name": True}

    from_: str = Field(alias="from", description="Source entity name")
    to: str = Field(description="Target entity name")
    relationType: str = Field(description="Relation in active voice, e.g. manages, works_at")


class ObservationInput(BaseModel):
    entityName: str
    contents: list[str]


class ObservationDeletion(BaseModel):
    entityName: str
    observations: list[str]


def _rel(r) -> dict:
    d = r.model_dump(by_alias=True) if hasattr(r, "model_dump") else dict(r)
    return {"from": d.get("from") or d.get("from_"), "to": d["to"], "relationType": d["relationType"]}



def create_server(state: StateFile):
    srv = make_server("memory", "Long-term memory as a knowledge graph. Search before answering questions about "
                                "people; store new durable facts as observations on the person's entity.")

    def _find(st, name):
        for e in st["entities"]:
            if e["name"].lower() == name.strip().lower():
                return e
        return None

    @srv.tool(annotations=write_tool())
    def create_entities(entities: list[Entity]) -> list[dict]:
        """Create entities. Entities whose name already exists are skipped."""
        def fn(st):
            made = []
            for e in entities:
                if not _find(st, e.name):
                    d = {"name": e.name.strip(), "entityType": e.entityType.strip(), "observations": list(e.observations)}
                    st["entities"].append(d)
                    made.append(d)
            return made
        return state.update(fn)

    @srv.tool(annotations=write_tool())
    def create_relations(relations: list[Relation]) -> list[dict]:
        """Create relations between existing entities (duplicates are skipped)."""
        def fn(st):
            made = []
            for r in map(_rel, relations):
                for end in (r["from"], r["to"]):
                    if not _find(st, end):
                        raise tool_error(f"entity {end!r} does not exist")
                if r not in st["relations"]:
                    st["relations"].append(r)
                    made.append(r)
            return made
        return state.update(fn)

    @srv.tool(annotations=write_tool())
    def add_observations(observations: list[ObservationInput]) -> list[dict]:
        """Add observations (facts) to existing entities."""
        def fn(st):
            out = []
            for o in observations:
                e = _find(st, o.entityName)
                if not e:
                    raise tool_error(f"entity {o.entityName!r} not found")
                new = [c.strip() for c in o.contents if c.strip() not in e["observations"]]
                e["observations"] += new
                out.append({"entityName": e["name"], "addedObservations": new})
            return out
        return state.update(fn)

    @srv.tool(annotations=write_tool(destructive=True))
    def delete_entities(entityNames: list[str]) -> str:
        """Delete entities and their relations."""
        def fn(st):
            names = {n.strip().lower() for n in entityNames}
            st["entities"] = [e for e in st["entities"] if e["name"].lower() not in names]
            st["relations"] = [r for r in st["relations"] if r["from"].lower() not in names and r["to"].lower() not in names]
            return "Entities deleted successfully"
        return state.update(fn)

    @srv.tool(annotations=write_tool(destructive=True))
    def delete_observations(deletions: list[ObservationDeletion]) -> str:
        """Delete specific observations from entities."""
        def fn(st):
            for d in deletions:
                e = _find(st, d.entityName)
                if e:
                    drop = {x.strip().lower() for x in d.observations}
                    e["observations"] = [x for x in e["observations"] if x.lower() not in drop]
            return "Observations deleted successfully"
        return state.update(fn)

    @srv.tool(annotations=write_tool(destructive=True))
    def delete_relations(relations: list[Relation]) -> str:
        """Delete relations."""
        def fn(st):
            drop = [_rel(r) for r in relations]
            st["relations"] = [r for r in st["relations"] if r not in drop]
            return "Relations deleted successfully"
        return state.update(fn)

    @srv.tool(annotations=read_only())
    def read_graph() -> dict:
        """Read the entire knowledge graph."""
        return state.read()

    @srv.tool(annotations=read_only())
    def search_nodes(query: str) -> dict:
        """Find entities whose name, type or observations contain the query (case-insensitive)."""
        st, q = state.read(), query.strip().lower()
        ents = [e for e in st["entities"] if q in e["name"].lower() or q in e["entityType"].lower()
                or any(q in o.lower() for o in e["observations"])]
        names = {e["name"] for e in ents}
        return {"entities": ents, "relations": [r for r in st["relations"] if r["from"] in names or r["to"] in names]}

    @srv.tool(annotations=read_only())
    def open_nodes(names: list[str]) -> dict:
        """Get specific entities by exact name, with the relations between them."""
        st = state.read()
        want = {n.strip().lower() for n in names}
        ents = [e for e in st["entities"] if e["name"].lower() in want]
        got = {e["name"] for e in ents}
        return {"entities": ents, "relations": [r for r in st["relations"] if r["from"] in got and r["to"] in got]}

    return srv


if __name__ == "__main__":
    a = parse_args("memory_state.json")
    run(create_server(StateFile(a.state, build_world, a.seed)), a)
