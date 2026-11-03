"""Environment base class: tool registry, argument validation, state hashing."""
from __future__ import annotations

import collections.abc
import copy
import enum
import inspect
import json
import re
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable

from ..core.types import ToolError, sha256_of


@dataclass
class ToolDef:
    name: str
    description: str
    params: dict          # name -> JSON schema
    required: list
    mutating: bool
    fn: Callable

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": self.params, "required": list(self.required)},
            },
        }


def _doc_parts(fn) -> tuple[str, dict]:
    """(description, {param: description}) from a Google-style ("Args:") or Sphinx-style (":param x:") docstring."""
    doc = inspect.getdoc(fn) or ""
    params: dict = {}
    lines = doc.splitlines()
    desc_lines, i = [], 0
    while i < len(lines) and not re.match(r"^\s*(Args|Arguments|Parameters|Returns|Raises)\s*:", lines[i]) \
            and not lines[i].strip().startswith(":param"):
        desc_lines.append(lines[i])
        i += 1
    current = None
    for line in lines[i:]:
        m = re.match(r"^\s*:param\s+(?:\w+\s+)?(\w+):\s*(.*)$", line)
        if m:
            params[m.group(1)], current = m.group(2).strip(), m.group(1)
            continue
        if re.match(r"^\s*(Returns|Raises|Yields)\s*:", line):
            current = None
            continue
        m = re.match(r"^\s{2,}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
        if m:
            params[m.group(1)], current = m.group(2).strip(), m.group(1)
        elif current and line.strip() and not re.match(r"^\s*(Args|Arguments|Parameters)\s*:", line):
            params[current] = (params[current] + " " + line.strip()).strip()
    desc = "\n".join(desc_lines).strip().split("\n\n")[0].replace("\n", " ")
    return desc, params


def _type_schema(tp) -> dict:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if tp is inspect.Parameter.empty or tp is typing.Any or tp is object:
        return {}
    if origin in (typing.Union, types.UnionType):
        non_null = [a for a in args if a is not type(None)]
        return _type_schema(non_null[0]) if len(non_null) == 1 else {"anyOf": [_type_schema(a) for a in non_null]}
    if origin is typing.Literal:
        vals = list(args)
        base = {str: "string", int: "integer", float: "number", bool: "boolean"}.get(type(vals[0]), "string")
        return {"type": base, "enum": vals}
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return {"type": "string", "enum": [e.value for e in tp]}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}
    if tp in (list, tuple, set) or origin in (list, tuple, set, collections.abc.Sequence):
        return {"type": "array", "items": _type_schema(args[0])} if args else {"type": "array"}
    if tp is dict or origin in (dict, collections.abc.Mapping):
        return {"type": "object"}
    return {}


def infer_schema(fn) -> tuple[dict, list, str]:
    """JSON-schema properties, required names and description inferred from a function's signature and docstring."""
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    desc, pdocs = _doc_parts(fn)
    props, required = {}, []
    for pname, p in sig.parameters.items():
        if pname in ("self", "cls") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        sch = dict(_type_schema(hints.get(pname, p.annotation)))
        if pname in pdocs:
            sch["description"] = pdocs[pname]
        if p.default is inspect.Parameter.empty:
            required.append(pname)
        elif p.default is not None:
            try:
                json.dumps(p.default)
                sch["default"] = p.default
            except TypeError:
                pass
        props[pname] = sch
    return props, required, desc


def tool(arg=None, params: dict | None = None, required: list | None = None, mutating: bool = False, *,
         description: str | None = None, name: str | None = None):
    """Register a function or a Toolkit/Env method as a tool.

    Forms:
        @tool                                   # schema + description inferred from type hints and docstring
        @tool(mutating=True, name="x")          # same, with options
        @tool("Description", {"arg": {"type": "string"}}, required=["arg"], mutating=True)   # explicit schema
    """
    def register(fn, desc):
        if params is None:
            props, req, doc_desc = infer_schema(fn)
            req = list(required) if required is not None else req
        else:
            props, doc_desc = params, ""
            req = list(required if required is not None else params.keys())
        fn._tooldef = dict(description=(desc or doc_desc or fn.__name__).strip(), params=props, required=req,
                           mutating=mutating, name=name or fn.__name__)
        return fn

    if callable(arg) and not isinstance(arg, str):
        return register(arg, description)
    return lambda fn: register(fn, arg if isinstance(arg, str) else description)


_PY_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def _check_type(value: Any, schema: dict, path: str) -> None:
    t = schema.get("type")
    if t is None:
        return
    types = t if isinstance(t, list) else [t]
    if value is None and "null" in types:
        return
    ok = False
    for tt in types:
        py = _PY_TYPES.get(tt)
        if py is None:
            continue
        if tt in ("integer", "number") and isinstance(value, bool):
            continue
        if isinstance(value, py):
            ok = True
            break
    if not ok:
        raise ToolError(f"invalid type for '{path}': expected {t}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError(f"invalid value for '{path}': must be one of {schema['enum']}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            _check_type(v, schema["items"], f"{path}[{i}]")


class Env:
    name = "base"
    version = "0.1"
    system_prompt = ""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        for attr in dir(type(self)):
            fn = getattr(type(self), attr)
            meta = getattr(fn, "_tooldef", None)
            if meta:
                meta = dict(meta)
                tname = meta.pop("name", None) or attr
                self._tools[tname] = ToolDef(name=tname, fn=getattr(self, attr), **meta)
        self.db: dict = {}
        self.call_log: list[dict] = []

    # ---- lifecycle -------------------------------------------------------
    def reset(self, world: dict) -> None:
        raise NotImplementedError

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def tool_schemas(self, names: list[str] | None = None) -> list[dict]:
        names = names or self.tool_names()
        return [self._tools[n].schema() for n in names]

    def is_mutating(self, name: str) -> bool:
        return name in self._tools and self._tools[name].mutating

    # ---- execution -------------------------------------------------------
    def call(self, name: str, args: dict, allowed: list[str] | None = None) -> tuple[str, bool]:
        """Execute a tool. Returns (result_json_string, is_error)."""
        entry = {"name": name, "arguments": args, "error": None}
        self.call_log.append(entry)
        try:
            if name not in self._tools or (allowed is not None and name not in allowed):
                raise ToolError(f"unknown tool '{name}'")
            td = self._tools[name]
            if not isinstance(args, dict):
                raise ToolError("arguments must be an object")
            for r in td.required:
                if r not in args:
                    raise ToolError(f"missing required argument '{r}'")
            for k, v in args.items():
                if k not in td.params:
                    raise ToolError(f"unexpected argument '{k}'")
                _check_type(v, td.params[k], k)
            result = td.fn(**copy.deepcopy(args))
            return json.dumps(result, ensure_ascii=False), False
        except ToolError as e:
            entry["error"] = str(e)
            return json.dumps({"error": str(e)}, ensure_ascii=False), True

    # ---- state -----------------------------------------------------------
    def state(self) -> dict:
        """Normalized verifiable state (override to exclude read-only catalogs)."""
        return self.db

    def state_hash(self) -> str:
        return sha256_of(self.state())

    def check(self, chk: dict) -> bool:
        """Domain-specific declarative check (e.g. email body contains text)."""
        raise NotImplementedError(f"{self.name} has no check {chk.get('type')}")

    def snapshot(self) -> dict:
        return copy.deepcopy(self.db)
