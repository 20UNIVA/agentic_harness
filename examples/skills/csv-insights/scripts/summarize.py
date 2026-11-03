#!/usr/bin/env python3
"""Summarize a CSV file as JSON (stdlib only). Part of the csv-insights example skill."""
import argparse
import csv
import json
import statistics
import sys
from collections import Counter


def num(v):
    try:
        return float(v.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--column")
    ap.add_argument("--where", action="append", default=[])
    a = ap.parse_args()
    with open(a.path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for cond in a.where:
        col, _, val = cond.partition("=")
        rows = [r for r in rows if (r.get(col) or "").strip() == val.strip()]
    cols = [a.column] if a.column else (list(rows[0].keys()) if rows else [])
    out = {"rows": len(rows), "columns": {}}
    for c in cols:
        vals = [r.get(c, "") for r in rows]
        nums = [x for x in map(num, vals) if x is not None]
        if vals and len(nums) == len(vals):
            out["columns"][c] = {"type": "number", "min": round(min(nums), 2), "max": round(max(nums), 2),
                                 "mean": round(statistics.mean(nums), 2), "median": round(statistics.median(nums), 2),
                                 "sum": round(sum(nums), 2)}
        else:
            cnt = Counter(v.strip() for v in vals)
            out["columns"][c] = {"type": "text", "distinct": len(cnt), "top": cnt.most_common(3)}
    json.dump(out, sys.stdout)
    print()


if __name__ == "__main__":
    main()
