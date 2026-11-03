# Skills

A skill is a folder of instructions and reference material the model can discover and load while it works.
The format is the Agent Skills layout used by Claude, so existing skill folders (your own or third-party ones) can be
registered as-is.

```
my-skill/
  SKILL.md              # YAML frontmatter + markdown instructions
  reference/api.md      # optional extra files (readable with read_skill_file)
  scripts/tool.py       # optional scripts (runnable only with the opt-in shell toolkit)
```

```markdown
---
name: my-skill
description: What the skill contains and WHEN to use it. This is all the model sees before loading it.
---

# Instructions ...
```

Rules checked at load time: `name` uses lowercase letters, digits and hyphens (max 64), equals the folder name and is
unique; `description` is non-empty (max 1024 chars). Other frontmatter keys (e.g. `license`) are kept as metadata.
Every skill gets a content hash over all its files; trajectories record which skills were offered (with hashes),
loaded and which files were read.

## Registering skills

```bash
agentic-harness run --skills-dir examples/skills --skills-dir /path/to/other/skills ...
agentic-harness resources --skills-dir /path/to/other/skills        # list and validate
```

A skills directory is either a folder containing skill folders or a single skill folder. The repo's `examples/skills`
is registered by default when you run from a source checkout (`--no-example-resources` disables that).
Third-party skill collections that follow the same layout (one folder per skill with a `SKILL.md`) can be cloned and
passed with `--skills-dir`.

Which skills an episode offers: the task's `skills` plus `--extra-skills`.

## How the model sees skills (`--skills-mode`)

| mode | system prompt | tools |
|---|---|---|
| `tool` (default) | a `# Skills` list: name + description | `load_skill(name)` returns SKILL.md and the list of bundled files; `read_skill_file(name, path)` returns a text file |
| `inline` | the full SKILL.md of every offered skill | none |
| `off` | nothing | none (ablation: does the model need the skill?) |

`read_skill_file` only serves files inside the skill folder (no `..`, text files only, truncated at 20k characters).

## Skills with scripts

Skills often say "run `scripts/x.py`". Enable the built-in shell toolkit for such tasks:

```bash
agentic-harness run --allow-shell ...       # and "toolkits": ["shell"] in the task (or --extra-toolkits shell)
```

Offered skills are copied into the episode workspace under `skills/<name>/`, `load_skill` tells the model where they
are, and `run_command` executes in that workspace. The shell toolkit is **not a sandbox** (see the README's Safety
section); run untrusted skills in a container or VM.

## Example skills

| skill | contents | used by |
|---|---|---|
| `fiscal-calendar` | company fiscal year/quarter rules + `reference/fy_months.csv` | `data_sql.skill_*` benchmark tasks |
| `file-archiving` | archive folder layout and index-file format | `filesystem.skill_*` |
| `travel-api` | the Travel API reference, moved out of the system prompt into `reference/endpoints.md` | `travel_api.skill_*` |
| `csv-insights` | a CSV summary script (`scripts/summarize.py`), needs the shell toolkit | `shell-skill-csv` in `examples/tasks/quickstart.json` |

Each benchmark skill task is unsolvable without its skill (company-specific conventions or a private API) and is
verified by the final state or answer, not by whether `load_skill` was called. Reports show `skill_task_load_rate`
separately, and `--skills-mode off` gives the no-skill baseline.

## Writing skill tasks

```json
{"instruction": "What were our total Widget sales in FY2026 Q2?", "env": {"domain": "data_sql", "seed": 7},
 "skills": ["fiscal-calendar", "file-archiving", "travel-api"],
 "gold_actions": [{"name": "load_skill", "arguments": {"name": "fiscal-calendar"}}, {"name": "run_sql", "arguments": {"query": "..."}}],
 "verify": {"answers": [{"number": 12541.23, "tol": 0.01}]}}
```

Offering unrelated skills as distractors tests whether the model picks the right one.
