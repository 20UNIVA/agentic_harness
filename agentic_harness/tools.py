"""Public API for user-defined tools.

Two ways to define tools:

1) Plain functions (stateless) - schema inferred from type hints and the docstring::

       from agentic_harness.tools import tool

       @tool
       def get_exchange_rate(base: str, quote: str) -> dict:
           \"\"\"Get the exchange rate between two ISO currencies.

           Args:
               base: Base currency, e.g. USD.
               quote: Quote currency, e.g. KRW.
           \"\"\"
           ...

   All decorated functions of a module become one toolkit named after the module file
   (or the module-level ``TOOLKIT_NAME``).

2) A stateful ``Toolkit`` subclass - fresh instance per episode, ``reset(world)`` builds the initial state,
   ``state()`` is what ``state`` checks inspect, ``check(chk)`` implements custom checks::

       from agentic_harness.tools import Toolkit, tool, ToolError

       class TodoList(Toolkit):
           name = "todo"
           def reset(self, world):
               self.db = {"items": []}
           @tool(mutating=True)
           def add_item(self, text: str) -> dict:
               \"\"\"Add a todo item.\"\"\"
               self.db["items"].append(text)
               return {"count": len(self.db["items"])}

Register them with ``--toolkits path/to/my_tools.py`` (or ``package.module``) and list the toolkit names in a task's
``toolkits`` field (or offer them in every episode with ``--extra-toolkits``).
Raise ``ToolError("message")`` for errors the model should see; any other exception is a harness failure.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import os
import sys

from .core.types import ToolError  # noqa: F401  (re-exported)
from .envs.base import Env, ToolDef, infer_schema, tool  # noqa: F401  (re-exported)

__all__ = ["tool", "Toolkit", "FunctionToolkit", "ToolError", "load_toolkits", "infer_schema"]


class Toolkit(Env):
    """Base class for stateful user toolkits (an Env with a default empty state)."""
    name = "toolkit"
    version = "0.1"
    system_prompt = ""

    def reset(self, world: dict) -> None:
        self.db = {}


class FunctionToolkit(Toolkit):
    """Wraps module-level @tool functions."""

    def __init__(self, name: str, functions: list):
        self.name = name
        super().__init__()
        for fn in functions:
            meta = dict(fn._tooldef)
            tname = meta.pop("name", None) or fn.__name__
            self._tools[tname] = ToolDef(name=tname, fn=fn, **meta)


def _import(spec: str):
    if spec.endswith(".py") or os.path.sep in spec:
        path = os.path.abspath(spec)
        if not os.path.exists(path):
            raise FileNotFoundError(f"toolkit file not found: {path}")
        modname = "ah_toolkit_" + hashlib.sha1(path.encode()).hexdigest()[:10]
        if modname in sys.modules:
            return sys.modules[modname], os.path.splitext(os.path.basename(path))[0]
        sp = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(sp)
        sys.modules[modname] = mod
        sp.loader.exec_module(mod)
        return mod, os.path.splitext(os.path.basename(path))[0]
    mod = importlib.import_module(spec)
    return mod, spec.rsplit(".", 1)[-1]


def load_toolkits(spec: str) -> dict:
    """Return {toolkit_name: factory} for every Toolkit/Env subclass defined in the module and, if the module has
    @tool functions, one FunctionToolkit (named TOOLKIT_NAME or the module name)."""
    attr = None
    if ":" in spec and not (len(spec) > 1 and spec[1] == ":"):      # "module:Class" (ignore Windows drive letters)
        spec, attr = spec.split(":", 1)
    mod, stem = _import(spec)
    out = {}
    if attr:
        obj = getattr(mod, attr)
        if not (inspect.isclass(obj) and issubclass(obj, Env)):
            raise TypeError(f"{spec}:{attr} is not a Toolkit/Env subclass")
        out[obj.name] = obj
        return out
    for _, obj in vars(mod).items():
        if inspect.isclass(obj) and issubclass(obj, Env) and obj.__module__ == mod.__name__ \
                and obj not in (Toolkit, FunctionToolkit):
            if obj.name in out:
                raise ValueError(f"duplicate toolkit name {obj.name!r} in {spec}")
            out[obj.name] = obj
    fns = [obj for obj in vars(mod).values() if inspect.isfunction(obj) and hasattr(obj, "_tooldef")
           and obj.__module__ == mod.__name__ and "." not in obj.__qualname__]
    if fns:
        name = getattr(mod, "TOOLKIT_NAME", stem)
        if name in out:
            raise ValueError(f"toolkit name {name!r} used by both a class and module functions in {spec}")
        out[name] = (lambda n=name, f=tuple(fns): FunctionToolkit(n, list(f)))
    if not out:
        raise ValueError(f"no toolkits or @tool functions found in {spec}")
    return out
