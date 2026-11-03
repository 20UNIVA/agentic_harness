# Built-in benchmark

A generated, verifiable benchmark for tool use at small-model difficulty. Every task is produced from a seed by a
template, its reference solution is executed to compute the expected result, and the whole set passes validation
gates before it is written.

## Environments

| env | state | tools | verified by |
|---|---|---|---|
| `retail` | users, orders, products | user lookup, order/product details, cancel, address change, return, exchange, transfer to human | final DB state + policy (only the customer's own orders, refunds, $1000 rule) |
| `workspace` | contacts, calendar, inbox, sent mail | contact search, events (list/create/update/delete), email search/read/send | normalized calendar/sent state (ids ignored), email body checks, policy (no external email, own events only) |
| `filesystem` | virtual files under `/home/user` | list, read, find, search, write, replace, move, delete, mkdir | normalized file tree (JSON files compared semantically), protected paths |
| `travel_api` | bookings + read-only catalog | one generic `http_request(method, path, query, body)` + API docs | bookings + answers; tests composing REST calls from documentation |
| `data_sql` | read-only SQLite | list/describe tables, `run_sql` (SELECT only), calculator | numeric/text answers; confidentiality policy (no individual salaries) |
| `none` + MCP | example MCP servers (kanban, wiki, memory) | `mcp__*` | `mcp_call` checks against the live server after the episode |

## Templates

80 templates (en/ko, about 30% Korean):

| group | n | templates |
|---|---|---|
| `retail` | 14 | `cancel`, `cancel_by_product`, `clarify`, `exchange`, `irrelevant`, `latest_order`, `lookup_status`, `lookup_total`, `modify_address`, `policy_big_cancel`, `policy_other_user`, `price`, `recover`, `return` |
| `workspace` | 12 | `clarify`, `create_event`, `delete_own`, `email_attendees`, `email_code`, `event_time`, `irrelevant`, `move_notify`, `phone`, `policy_external`, `policy_not_organizer`, `recover` |
| `filesystem` | 15 | `append_todo`, `clarify`, `config_port`, `count_errors`, `create_note`, `csv_sum`, `edit_port`, `find_todo`, `irrelevant`, `organize_csv`, `policy_protected`, `recover`, `skill_archive_csvs`, `skill_archive_one`, `skill_archive_refuse` |
| `travel_api` | 13 | `book_flight_city`, `book_hotel`, `cheapest_flight`, `clarify`, `fx`, `hotel_in_krw`, `irrelevant`, `policy_over_budget`, `recover`, `skill_book_cheapest`, `skill_weather`, `trip`, `weather` |
| `data_sql` | 12 | `avg_salary`, `clarify`, `count_dept`, `irrelevant`, `manager_top_budget`, `policy_salary`, `recover`, `sales_growth`, `sales_total`, `skill_fiscal_half_top_product`, `skill_fiscal_quarter_sales`, `top_seller` |
| `mcp_examples` | 14 | `cross_meeting_card`, `kanban_bulk_move`, `kanban_comment`, `kanban_create_assign`, `kanban_move`, `kanban_who`, `memory_org`, `memory_recall`, `memory_relation`, `memory_store`, `wiki_expense`, `wiki_irrelevant`, `wiki_oncall`, `wiki_release` |

Categories:

| category | expected behaviour |
|---|---|
| `lookup` | answer from one or two reads |
| `chain` | later calls depend on earlier results |
| `write` / `multi_write` | change state correctly (one / several writes) |
| `policy` | refuse or escalate a request the system prompt forbids (doing nothing is often correct; read with the null baseline) |
| `irrelevant` | answer without tools (no state change) |
| `clarify` | ask for missing information; a scripted user answers once |
| `recover` | a first attempt naturally fails (wrong email, file name, column, missing IATA code); recover from the error |
| `skill` | needs knowledge that exists only in a skill (fiscal calendar, archiving convention, private API docs) |
| `cross` | combine a built-in env with an MCP server |

## Splits and gates

```bash
agentic-harness gen-tasks --out data/tasks_v0.2 --train 150 --dev 4 --test 10     # counts per template
```

| split | tasks | seeds |
|---|---|---|
| test | 800 | 9,000,000+ |
| dev | 320 | 8,000,000+ |
| train | 11,706 | 1,000,000+ |

- **Gold gate**: the oracle replays every reference solution (MCP tasks against live servers); all must score 1.0.
- **Null gate**: a policy that never calls a tool must fail every task outside the `policy` category.
- **No leakage**: splits are generated test → dev → train and each excludes the visible prompts (system prompt,
  instruction, follow-ups) of earlier splits; content hashes are disjoint. Within train, at most 3 tasks share a prompt.
- Generation is deterministic: the same command reproduces the same files. `manifest.json` records counts and gates.

Scope: test prompts are unseen, but worlds come from the same generators, so this measures in-distribution
generalization to new instances, not new domains. Use external benchmarks for that.

## Metrics

- `pass@1`: mean success over attempts. `pass@k`: unbiased estimator `1 - C(n-c, k)/C(n, k)`.
- `pass^k`: probability that k independent attempts all succeed, `C(c, k)/C(n, k)` (tau-bench), averaged over tasks.
- Breakdowns by domain, category, difficulty (1-4), language; `--null-baseline` adds the do-nothing score per group.
- Behaviour: turn-level parse error rate, tool error rate, unknown-tool rate, unclosed-think rate, prefix-mismatch
  rate, mean turns and tool calls, tool use on irrelevant tasks, calls by source, MCP tool error rate, skill load rate
  on skill tasks.
- Infrastructure failures (`reward=None`) are excluded from all denominators and counted separately.
