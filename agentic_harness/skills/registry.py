"""Skill discovery, validation and the skill toolset exposed to the model."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from ..core.types import ToolError, sha256_of

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_FILE_CHARS = 20_000
TEXT_EXT = {".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".py", ".sh", ".sql", ".toml", ".ini", ".html"}


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse a `---` delimited YAML frontmatter block. Uses PyYAML when available, otherwise a small
    `key: value` subset parser (enough for name / description / metadata scalars)."""
    if not text.startswith("---"):
        raise ValueError("SKILL.md must start with a '---' frontmatter block")
    end = text.find("\n---", 3)
    if end < 0:
        raise ValueError("unterminated frontmatter")
    raw, body = text[3:end].strip("\n"), text[end + 4:].lstrip("\n")
    try:
        import yaml  # type: ignore
        meta = yaml.safe_load(raw) or {}
    except ImportError:
        meta, key = {}, None
        for line in raw.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
            if m and not line.startswith((" ", "\t")):
                key, val = m.group(1), m.group(2).strip()
                if val in (">", "|", ">-", "|-"):
                    meta[key] = ""
                else:
                    meta[key] = val.strip("'\"")
            elif key is not None:            # continuation of a folded / literal block
                meta[key] = (meta[key] + " " + line.strip()).strip()
    if not isinstance(meta, dict):
        raise ValueError("frontmatter must be a mapping")
    return meta, body


@dataclass
class Skill:
    name: str
    description: str
    body: str
    root: str
    files: list[str] = field(default_factory=list)       # extra files relative to root (excluding SKILL.md)
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""

    @classmethod
    def load(cls, root: str) -> "Skill":
        path = os.path.join(root, "SKILL.md")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        meta, body = parse_frontmatter(text)
        name, desc = str(meta.get("name", "")).strip(), str(meta.get("description", "")).strip()
        if not NAME_RE.match(name) or len(name) > 64:
            raise ValueError(f"{path}: name must be lowercase letters/digits/hyphens (<=64 chars), got {name!r}")
        if os.path.basename(os.path.normpath(root)) != name:
            raise ValueError(f"{path}: directory name must equal skill name {name!r}")
        if not desc or len(desc) > 1024:
            raise ValueError(f"{path}: description is required (<=1024 chars)")
        files, blobs = [], {"SKILL.md": text}
        for d, _, fs in os.walk(root):
            for fn in sorted(fs):
                rel = os.path.relpath(os.path.join(d, fn), root)
                if rel == "SKILL.md" or rel.startswith(".") or "/." in rel:
                    continue
                files.append(rel)
                with open(os.path.join(d, fn), "rb") as fh:
                    blobs[rel] = fh.read().hex()
        files.sort()
        return cls(name=name, description=desc, body=body.strip(), root=os.path.abspath(root), files=files,
                   metadata={k: v for k, v in meta.items() if k not in ("name", "description")},
                   content_hash=sha256_of(blobs))


class SkillRegistry:
    """All skills found under one or more directories (each child directory with a SKILL.md is a skill)."""

    def __init__(self, dirs: list[str] | None = None):
        self.skills: dict[str, Skill] = {}
        self.dirs = []
        for d in dirs or []:
            self.add_dir(d)

    def add_dir(self, d: str) -> None:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"skills dir not found: {d}")
        self.dirs.append(d)
        cands = [d] if os.path.exists(os.path.join(d, "SKILL.md")) else \
            [os.path.join(d, x) for x in sorted(os.listdir(d)) if os.path.exists(os.path.join(d, x, "SKILL.md"))]
        for root in cands:
            s = Skill.load(root)
            if s.name in self.skills and self.skills[s.name].root != s.root:
                raise ValueError(f"duplicate skill name {s.name!r}: {self.skills[s.name].root} and {s.root}")
            self.skills[s.name] = s

    def get(self, name: str) -> Skill:
        return self.skills[name]

    def names(self) -> list[str]:
        return sorted(self.skills)


class SkillToolset:
    """Per-episode view of the offered skills + the `load_skill` / `read_skill_file` tools."""

    LOAD, READ = "load_skill", "read_skill_file"

    def __init__(self, registry: SkillRegistry, names: list[str], mode: str = "tool"):
        missing = [n for n in names if n not in registry.skills]
        if missing:
            raise KeyError(f"unknown skills {missing}; registered: {registry.names()}")
        self.mode = mode
        self.skills = [registry.get(n) for n in sorted(set(names))] if mode != "off" else []
        self.loaded: list[str] = []
        self.files_read: list[str] = []
        self.workspace_hint = False        # set by the session when skills are staged into a shell workspace

    # ---- prompt ----------------------------------------------------------------
    def system_addendum(self) -> str:
        if not self.skills:
            return ""
        if self.mode == "inline":
            parts = ["# Skills", "The following skills (procedures and reference knowledge) apply to this conversation."]
            for s in self.skills:
                parts.append(f"\n## Skill: {s.name}\n{s.body}")
            return "\n".join(parts)
        lines = ["# Skills",
                 "Skills are packaged instructions and reference knowledge. Their content is NOT included here.",
                 f"When a request relates to a skill below, call `{self.LOAD}` with its name before acting, and follow it.",
                 ""]
        lines += [f"- {s.name}: {s.description}" for s in self.skills]
        return "\n".join(lines)

    # ---- tools -----------------------------------------------------------------
    def tool_schemas(self) -> list[dict]:
        if self.mode != "tool" or not self.skills:
            return []
        names = [s.name for s in self.skills]
        return [
            {"type": "function", "function": {
                "name": self.LOAD, "description": "Load a skill's instructions (SKILL.md) by name.",
                "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": names}},
                               "required": ["name"]}}},
            {"type": "function", "function": {
                "name": self.READ, "description": "Read an additional file bundled with a skill (paths are listed by load_skill).",
                "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": names},
                                                                "path": {"type": "string"}},
                               "required": ["name", "path"]}}},
        ]

    def handles(self, tool: str) -> bool:
        return self.mode == "tool" and bool(self.skills) and tool in (self.LOAD, self.READ)

    def call(self, tool: str, args: dict) -> str:
        by_name = {s.name: s for s in self.skills}
        name = str(args.get("name", "")).strip()
        if name not in by_name:
            raise ToolError(f"unknown skill {name!r}; available: {sorted(by_name)}")
        s = by_name[name]
        if tool == self.LOAD:
            self.loaded.append(name)
            out = f"# Skill: {s.name}\n\n{s.body}"
            if s.files:
                out += "\n\n## Bundled files (use read_skill_file)\n" + "\n".join(f"- {f}" for f in s.files)
                if self.workspace_hint:
                    out += f"\n\nThese files are also available in your shell workspace under skills/{s.name}/."
            return out
        rel = os.path.normpath(str(args.get("path", "")).strip().lstrip("/"))
        if rel.startswith("..") or rel not in s.files:
            raise ToolError(f"file {rel!r} not found in skill {name!r}; files: {s.files}")
        if os.path.splitext(rel)[1].lower() not in TEXT_EXT:
            raise ToolError("only text files can be read")
        with open(os.path.join(s.root, rel), encoding="utf-8", errors="replace") as f:
            text = f.read()
        self.files_read.append(f"{name}/{rel}")
        return text[:MAX_FILE_CHARS] + ("\n...[truncated]" if len(text) > MAX_FILE_CHARS else "")

    def provenance(self) -> dict:
        return {"mode": self.mode, "offered": [{"name": s.name, "content_hash": s.content_hash} for s in self.skills],
                "loaded": list(self.loaded), "files_read": list(self.files_read)}


def skills_json(registry: SkillRegistry) -> str:
    return json.dumps([{"name": s.name, "description": s.description, "files": s.files, "hash": s.content_hash}
                       for s in registry.skills.values()], ensure_ascii=False, indent=2)
