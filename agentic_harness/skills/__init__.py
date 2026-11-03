"""Agent Skills: SKILL.md packages the model can discover and load at run time.

Format (compatible with the Anthropic Agent Skills layout):

    my-skill/
      SKILL.md          # YAML frontmatter (name, description) + markdown instructions
      reference/*.md    # optional extra files, readable through `read_skill_file`

Exposure modes (run-level):
  - "tool"   (default) progressive disclosure: the system prompt lists name + description;
             the model calls `load_skill(name)` to get the body and `read_skill_file(name, path)` for extra files.
  - "inline" every offered skill body is appended to the system prompt (no skill tools).
  - "off"    skills are not offered.
Every trajectory records the offered skills with their content hash (survey B14).
"""
from .registry import Skill, SkillRegistry, SkillToolset, parse_frontmatter  # noqa: F401
