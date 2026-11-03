"""ToolSession: everything the model can call in one episode = built-in env tools + skills + external MCP tools.

Used by the offline runner (core/runner.py), task generation gates and the ms-swift GRPO plugin, so that
evaluation, data generation and RL see the same tools, routing and checks.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field

from ..envs import make_env
from ..mcp.registry import EXAMPLES_DIR, McpRegistry
from ..skills.registry import SkillRegistry, SkillToolset
from .types import InfraError, ToolError

DEFAULT_SKILLS_DIR = os.path.join(EXAMPLES_DIR, "skills")
DEFAULT_MCP_CONFIG = os.path.join(EXAMPLES_DIR, "mcp", "servers.json")


@dataclass
class Resources:
    """Run-level registries and options (skills, MCP servers)."""
    skills: SkillRegistry | None = None
    mcp: McpRegistry | None = None
    toolkits: dict = field(default_factory=dict)       # name -> zero-arg factory (agentic_harness.tools.load_toolkits)
    extra_toolkits: list = field(default_factory=list)  # instantiated in every episode (besides task["toolkits"])
    skills_mode: str = "tool"                          # tool | inline | off
    extra_skills: list = field(default_factory=list)   # offered in every episode (in addition to task["skills"])
    extra_mcp: list = field(default_factory=list)      # connected in every episode (in addition to task["mcp_servers"])
    episode_root: str | None = None                    # parent dir for per-episode sandboxes (default: system tmp)
    keep_episode_dirs: bool = False
    mcp_instructions: bool = True                      # append server `instructions` to the system prompt
    allow_shell: bool = False                          # enable the built-in `shell` toolkit (not a sandbox!)
    max_tool_result_chars: int = 20000                 # longer tool results are truncated before the model sees them

    def close(self):
        if self.mcp:
            self.mcp.close()


_DEFAULT: Resources | None = None


def default_resources() -> Resources:
    """Example skills + example MCP config shipped with the repo (loaded lazily, once)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = build_resources(skills_dirs=[DEFAULT_SKILLS_DIR], mcp_configs=[DEFAULT_MCP_CONFIG])
    return _DEFAULT


def build_resources(skills_dirs=None, mcp_configs=None, toolkit_specs=None, **kw) -> Resources:
    """skills_dirs: folders of SKILL.md skills; mcp_configs: mcpServers JSON files;
    toolkit_specs: python files / modules with @tool functions or Toolkit classes."""
    for d in skills_dirs or []:
        if d != DEFAULT_SKILLS_DIR and not os.path.isdir(d):
            raise FileNotFoundError(f"skills dir not found: {d}")
    for p in mcp_configs or []:
        if p != DEFAULT_MCP_CONFIG and not os.path.exists(p):
            raise FileNotFoundError(f"MCP config not found: {p}")
    skills = SkillRegistry([d for d in (skills_dirs or []) if os.path.isdir(d)]) if skills_dirs else None
    mcp = McpRegistry([p for p in (mcp_configs or []) if os.path.exists(p)]) if mcp_configs else None
    toolkits = {}
    if toolkit_specs:
        from ..tools import load_toolkits
        for spec in toolkit_specs:
            for name, factory in load_toolkits(spec).items():
                if name in toolkits:
                    raise ValueError(f"duplicate toolkit name {name!r} (from {spec})")
                toolkits[name] = factory
    return Resources(skills=skills, mcp=mcp, toolkits=toolkits, **kw)


def _subset_match(expected, actual) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and _subset_match(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and all(any(_subset_match(e, a) for a in actual) for e in expected)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


class ToolSession:
    def __init__(self, task: dict, resources: Resources | None = None):
        self.task = task
        self.res = resources if resources is not None else default_resources()
        self.env = make_env(task["domain"], task["world"])
        self.env_tools = list(task.get("tools") or [])
        self.episode_dir = None
        self.mcp_sets = []
        self.skills = None
        self.toolkits = []
        try:
            from ..builtin_toolkits import BUILTIN_TOOLKITS
            tk_names = list(dict.fromkeys(list(task.get("toolkits") or []) + list(self.res.extra_toolkits)))
            for n in tk_names:
                if n in self.res.toolkits:
                    factory = self.res.toolkits[n]
                elif n in BUILTIN_TOOLKITS:
                    if n == "shell" and not self.res.allow_shell:
                        raise InfraError("the 'shell' toolkit is disabled; enable it explicitly (--allow-shell)")
                    factory = BUILTIN_TOOLKITS[n]
                else:
                    raise InfraError(f"toolkit {n!r} is not registered (registered: {sorted(self.res.toolkits)}, "
                                     f"built-in: {sorted(BUILTIN_TOOLKITS)})")
                tk = factory()
                tk.reset(task["world"])
                self.toolkits.append(tk)
            skill_names = list(dict.fromkeys(list(task.get("skills") or []) + list(self.res.extra_skills)))
            if skill_names and self.res.skills_mode != "off":
                if self.res.skills is None:
                    raise InfraError(f"task needs skills {skill_names} but no skill registry is configured")
                try:
                    self.skills = SkillToolset(self.res.skills, skill_names, self.res.skills_mode)
                except KeyError as e:
                    raise InfraError(str(e))
            if any(getattr(tk, "uses_workspace", False) for tk in self.toolkits):
                self.episode_dir = tempfile.mkdtemp(prefix="ah-ep-", dir=self.res.episode_root)
                ws = os.path.join(self.episode_dir, "workspace")
                os.makedirs(ws)
                for tk in self.toolkits:
                    if getattr(tk, "uses_workspace", False):
                        tk.bind_workspace(ws, task, self.skills.skills if self.skills else [])
                if self.skills:
                    self.skills.workspace_hint = True
            mcp_names = list(dict.fromkeys(list(task.get("mcp_servers") or []) + list(self.res.extra_mcp)))
            if mcp_names:
                if self.res.mcp is None:
                    raise InfraError(f"task needs MCP servers {mcp_names} but no MCP config is registered")
                self.episode_dir = self.episode_dir or tempfile.mkdtemp(prefix="ah-ep-", dir=self.res.episode_root)
                self.mcp_sets = self.res.mcp.open(mcp_names, {"EPISODE_DIR": self.episode_dir,
                                                              "TASK_SEED": task["world"]["seed"], "TASK_ID": task["task_id"]})
                for step in task.get("mcp_setup") or []:      # harness-side initialization through the server's own tools
                    text, is_err, _ = self._mcp(step["server"]).call_raw(step["tool"], step.get("arguments") or {})
                    if is_err:
                        raise InfraError(f"mcp_setup {step['server']}.{step['tool']} failed: {text[:200]}")
            names = [s["function"]["name"] for s in self.tool_schemas()]
            dup = {n for n in names if names.count(n) > 1}
            if dup:
                raise InfraError(f"tool name collision: {sorted(dup)}")
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------------ prompt / schema
    def system_prompt(self) -> str:
        parts = [self.task["system_prompt"]] + [tk.system_prompt for tk in self.toolkits if tk.system_prompt]
        if self.skills:
            parts.append(self.skills.system_addendum())
        if self.res.mcp_instructions:
            ins = [(ts.server, ts.client.instructions) for ts in self.mcp_sets if ts.client.instructions]
            if ins:
                parts.append("# Connected MCP servers\n" + "\n".join(f"## {s}\n{i.strip()}" for s, i in ins))
        return "\n\n".join(p for p in parts if p)

    def tool_schemas(self) -> list[dict]:
        out = self.env.tool_schemas(self.env_tools) if self.env_tools else []
        for tk in self.toolkits:
            out += tk.tool_schemas()
        if self.skills:
            out += self.skills.tool_schemas()
        for ts in self.mcp_sets:
            out += ts.tool_schemas()
        return out

    # ------------------------------------------------------------------ routing
    def _mcp(self, server: str):
        for ts in self.mcp_sets:
            if ts.server == server:
                return ts
        raise InfraError(f"MCP server {server!r} not connected in this episode")

    def _toolkit_for(self, name: str):
        for tk in self.toolkits:
            if name in tk._tools:
                return tk
        return None

    def source(self, name: str) -> str:
        tk = self._toolkit_for(name)
        if tk is not None:
            return "toolkit:" + tk.name
        if self.skills and self.skills.handles(name):
            return "skill"
        for ts in self.mcp_sets:
            if ts.handles(name):
                return "mcp:" + ts.server
        return "env"

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Execute a model tool call. Returns (result_text, is_error). Raises InfraError for harness/server failures."""
        text, is_err = self._call(name, args)
        cap = self.res.max_tool_result_chars
        if cap and len(text) > cap:
            text = text[:cap] + f"\n...[truncated: {len(text) - cap} more characters]"
        return text, is_err

    def _call(self, name: str, args: dict) -> tuple[str, bool]:
        tk = self._toolkit_for(name)
        if tk is not None:
            try:
                return tk.call(name, args)
            except ToolError:
                raise
            except Exception as e:            # a bug in user tool code is not the policy's fault
                raise InfraError(f"toolkit {tk.name}.{name} raised {type(e).__name__}: {e}")
        if self.skills and self.skills.handles(name):
            try:
                return self.skills.call(name, args if isinstance(args, dict) else {}), False
            except ToolError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False), True
        for ts in self.mcp_sets:
            if ts.handles(name):
                if not isinstance(args, dict):
                    return json.dumps({"error": "arguments must be an object"}), True
                return ts.call(name, args)
        return self.env.call(name, args, allowed=self.env_tools)

    def is_mutating(self, name: str) -> bool:
        tk = self._toolkit_for(name)
        if tk is not None:
            return tk.is_mutating(name)
        if self.skills and self.skills.handles(name):
            return False
        for ts in self.mcp_sets:
            if ts.handles(name):
                return ts.is_mutating(name)
        return self.env.is_mutating(name)

    # ------------------------------------------------------------------ verification
    def state_hash(self) -> str:
        return self.env.state_hash()

    def call_mcp(self, server: str, tool: str, arguments: dict | None = None) -> tuple[str, bool]:
        """Call an MCP tool directly (by its server-side name). For verifiers and setup code, not the model."""
        text, is_err, _ = self._mcp(server).call_raw(tool, arguments or {})
        return text, is_err

    def check(self, chk: dict) -> bool:
        typ = chk["type"]
        if typ == "mcp_call":
            text, is_err, raw = self._mcp(chk["server"]).call_raw(chk["tool"], chk.get("arguments") or {})
            if is_err:
                return False
            low = text.lower()
            if any(str(s).lower() not in low for s in chk.get("expect_contains") or []):
                return False
            if any(str(s).lower() in low for s in chk.get("expect_not_contains") or []):
                return False
            if "expect_json" in chk:
                actual = raw.get("structuredContent")
                if actual is None:
                    try:
                        actual = json.loads(text)
                    except json.JSONDecodeError:
                        return False
                if not _subset_match(chk["expect_json"], actual):
                    return False
            return True
        if typ == "skill_loaded":
            return bool(self.skills) and chk["name"] in self.skills.loaded
        if typ in ("state", "toolkit"):
            tk = next((t for t in self.toolkits if t.name == chk.get("toolkit")), None)
            target = tk if tk is not None else (self.env if chk.get("toolkit") in (None, self.env.name) else None)
            if target is None:
                raise InfraError(f"check refers to unknown toolkit {chk.get('toolkit')!r}")
            if typ == "toolkit":
                return bool(target.check(chk))
            return _subset_match(chk["json_subset"], json.loads(json.dumps(target.state(), default=list)))
        return self.env.check(chk)

    def provenance(self) -> dict:
        from .types import sha256_of
        return {"skills": self.skills.provenance() if self.skills else None,
                "mcp": [ts.provenance() for ts in self.mcp_sets],
                "toolkits": [{"name": tk.name, "class": f"{type(tk).__module__}.{type(tk).__qualname__}",
                              "version": getattr(tk, "version", None),
                              "tools_hash": sha256_of([s["function"] for s in tk.tool_schemas()])} for tk in self.toolkits]}

    def close(self):
        for ts in self.mcp_sets:
            try:
                ts.close()
            except Exception:
                pass
        self.mcp_sets = []
        if self.episode_dir and not self.res.keep_episode_dirs:
            shutil.rmtree(self.episode_dir, ignore_errors=True)
