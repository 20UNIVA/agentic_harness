---
name: csv-insights
description: Summarize a CSV file (row count, per-column min/max/mean/median, top values) with a bundled script instead of reading it by hand. Use when asked for statistics about a CSV file in the shell workspace. Requires the shell toolkit.
---

# CSV insights

Do not compute statistics by reading the CSV yourself. Run the bundled script from the shell workspace:

```
python skills/csv-insights/scripts/summarize.py <path/to/file.csv>
```

It prints JSON:

```json
{"rows": 12, "columns": {"amount": {"type": "number", "min": 3.5, "max": 99.0, "mean": 41.2, "median": 38.0, "sum": 494.4},
                         "region": {"type": "text", "distinct": 3, "top": [["north", 6], ["south", 4]]}}}
```

Options:

- `--column NAME` only summarize one column.
- `--where COLUMN=VALUE` only keep rows where COLUMN equals VALUE (repeatable).

Report the numbers exactly as printed (they are already rounded to 2 decimals).
