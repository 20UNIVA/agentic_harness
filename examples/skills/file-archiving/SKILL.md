---
name: file-archiving
description: The team's convention for archiving files in the home directory (target folder layout and the archive index file). Use whenever the user asks to archive, put away or file away documents.
---

# File archiving convention

All paths below are relative to `/home/user`.

## Where archived files go

Move the file to `archive/<top-level folder>/<file name>`, where `<top-level folder>` is the first path component of the
original location. Keep the file name and the content unchanged.

| Original | Archived as |
|---|---|
| `downloads/export_2.csv` | `archive/downloads/export_2.csv` |
| `notes/meeting.txt` | `archive/notes/meeting.txt` |
| `projects/billing/README.md` | `archive/projects/README.md` |

Create the target folder first if it does not exist (`make_dir`).

## The archive index

Every archived file gets exactly one line in `archive/INDEX.txt`:

```
<original path> -> <archived path>
```

- Use the relative paths from the table above, separated by ` -> ` (space, arrow, space).
- If `archive/INDEX.txt` already exists, keep its existing lines and add the new ones after them.
- When archiving several files at once, add their lines in alphabetical order of the original path.
- The file ends with a newline.

Example `archive/INDEX.txt` after archiving two exports:

```
downloads/export_1.csv -> archive/downloads/export_1.csv
downloads/export_2.csv -> archive/downloads/export_2.csv
```

## Never archive

- anything under `protected/`
- files ending in `.lock`

If asked to archive one of these, refuse and explain why.
