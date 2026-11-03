"""Built-in optional toolkits.

shell  - run shell commands and read/write text files inside a per-episode workspace directory.
         Needed for skills that ship scripts (Anthropic-style skills often say "run scripts/x.py").
         Offered skills are copied to <workspace>/skills/<name>/ and task `files` are written into the workspace.

SECURITY: the shell toolkit is DISABLED unless the run enables it (--allow-shell / Resources.allow_shell).
It is NOT a sandbox: commands run as the harness user with network access; only the working directory, a minimal
environment, a timeout and output truncation are applied. Use it with trusted models/tasks, or run the whole harness
inside a container/VM.
"""
from __future__ import annotations

import os
import shutil
import subprocess

from .core.types import ToolError
from .tools import Toolkit, tool

MAX_OUT = 8000


class ShellToolkit(Toolkit):
    name = "shell"
    version = "0.1"
    uses_workspace = True
    system_prompt = ("You have a shell workspace (the current directory of `run_command`). Skill files, if any, are under "
                     "skills/<skill-name>/. Paths in read_text_file / write_text_file are relative to the workspace.")

    def reset(self, world: dict) -> None:
        self.db = {"commands": 0}
        self.ws = None

    def bind_workspace(self, path: str, task: dict, skills=None) -> None:
        self.ws = os.path.realpath(path)
        for s in skills or []:
            shutil.copytree(s.root, os.path.join(self.ws, "skills", s.name), dirs_exist_ok=True)
        for rel, content in (task.get("files") or {}).items():
            self._write(rel, content)

    def _path(self, rel: str) -> str:
        if self.ws is None:
            raise ToolError("workspace not initialized")
        p = os.path.realpath(os.path.join(self.ws, rel))
        if not (p == self.ws or p.startswith(self.ws + os.sep)):
            raise ToolError("path must stay inside the workspace")
        return p

    def _write(self, rel: str, content: str) -> str:
        p = self._path(rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return p

    def state(self) -> dict:
        files = {}
        if self.ws:
            for d, _, fs in os.walk(self.ws):
                for fn in fs:
                    rel = os.path.relpath(os.path.join(d, fn), self.ws)
                    if not rel.startswith("skills" + os.sep):
                        try:
                            files[rel] = open(os.path.join(d, fn), encoding="utf-8", errors="replace").read()[:MAX_OUT]
                        except OSError:
                            pass
        return {"files": files, "commands": self.db["commands"]}

    @tool(mutating=True)
    def run_command(self, command: str, timeout_s: int = 30) -> dict:
        """Run a shell command (bash) in the workspace directory and return exit code, stdout and stderr.

        Args:
            command: The command line, e.g. "python skills/csv-insights/scripts/summarize.py data.csv".
            timeout_s: Seconds before the command is killed (max 120).
        """
        if self.ws is None:
            raise ToolError("workspace not initialized")
        self.db["commands"] += 1
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": self.ws, "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            r = subprocess.run(["bash", "-c", command], cwd=self.ws, env=env, capture_output=True, text=True,
                               timeout=max(1, min(int(timeout_s), 120)))
        except subprocess.TimeoutExpired:
            raise ToolError(f"command timed out after {timeout_s}s")
        return {"exit_code": r.returncode, "stdout": r.stdout[-MAX_OUT:], "stderr": r.stderr[-MAX_OUT // 2:]}

    @tool
    def read_text_file(self, path: str) -> str:
        """Read a text file from the workspace.

        Args:
            path: Path relative to the workspace.
        """
        p = self._path(path)
        if not os.path.isfile(p):
            raise ToolError(f"no such file: {path}")
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()[:MAX_OUT]

    @tool(mutating=True)
    def write_text_file(self, path: str, content: str) -> str:
        """Create or overwrite a text file in the workspace.

        Args:
            path: Path relative to the workspace.
            content: Full file content.
        """
        self._write(path, content)
        return f"wrote {len(content)} characters to {path}"


BUILTIN_TOOLKITS = {"shell": ShellToolkit}
