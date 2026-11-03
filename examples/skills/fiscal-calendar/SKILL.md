---
name: fiscal-calendar
description: Acme's fiscal calendar (fiscal years, quarters and halves) and how to turn "FY2026 Q2"-style periods into calendar months for SQL. Use whenever a question mentions a fiscal year, fiscal quarter (FQ, Q1-Q4 in a fiscal sense) or fiscal half.
---

# Acme fiscal calendar

Acme's fiscal year does **not** follow the calendar year.

- A fiscal year runs from **July 1 to June 30**.
- It is named after the calendar year in which it **ends**: FY2026 = 2025-07-01 .. 2026-06-30.

## Quarters and halves

| Period | Months |
|---|---|
| Q1 | July, August, September (of the previous calendar year) |
| Q2 | October, November, December (of the previous calendar year) |
| Q3 | January, February, March |
| Q4 | April, May, June |
| H1 | Q1 + Q2 |
| H2 | Q3 + Q4 |

Example: **FY2026 Q2 = 2025-10, 2025-11, 2025-12**. FY2026 Q3 = 2026-01, 2026-02, 2026-03.

A per-month lookup table is bundled as `reference/fy_months.csv`.

## Querying

The `sales.month` column stores calendar months as `'YYYY-MM'` text. Convert the fiscal period to its months first,
then filter with `IN`:

```sql
SELECT ROUND(SUM(amount), 2)
FROM sales
WHERE product = 'Widget' AND month IN ('2025-10', '2025-11', '2025-12');   -- FY2026 Q2
```

Always state which calendar months you used in the final answer.
