"""Tasks for the three example MCP servers (examples/mcp_servers: kanban, wiki, memory).

Ground truth comes from examples/mcp_servers/worlds.py (the same seed-deterministic builders the servers use), so no
server is started during generation; the oracle gate in `agentic-harness gen-tasks` replays every gold path against the live
servers. Post-episode verification uses `mcp_call` checks, i.e. the server's own read tools.
"""
from __future__ import annotations

import os
import sys

from ..envs import make_env
from ..envs.names import GENERAL_QA
from ..envs.workspace import WEEKDAY
from ..mcp.registry import EXAMPLES_DIR
from .common import GenSkip, L, call, mcp, mcp_check, num, text

sys.path.insert(0, os.path.join(EXAMPLES_DIR, "mcp_servers"))
try:
    import worlds  # noqa: E402
except ImportError:   # installed without the repo's examples/: the mcp_examples templates are unavailable
    worlds = None

KO_DAY = {"Monday": "월요일", "Tuesday": "화요일", "Wednesday": "수요일", "Thursday": "목요일", "Friday": "금요일"}
COL_KO = {"todo": "할 일(todo)", "in_progress": "진행 중(in_progress)", "review": "리뷰(review)", "done": "완료(done)"}


def _seed(env):
    return env.world["seed"]


def _unique_title_cards(k):
    titles = [c["title"] for c in k["cards"].values()]
    return [c for c in k["cards"].values() if titles.count(c["title"]) == 1]


def _member_name(k, username):
    return next(m["name"] for m in k["members"] if m["username"] == username)


# ---------------------------------------------------------------------------------------------- kanban
def g_kanban_who(rng, env, lang):
    k = worlds.build_kanban(_seed(env))
    cards = [c for c in _unique_title_cards(k) if c["assignee"]]
    c = rng.choice(cards)
    name = _member_name(k, c["assignee"])
    return dict(category="lookup", difficulty=2, mcp_servers=["kanban"],
                instruction=L(lang, f"Who is working on the '{c['title']}' card, and which column is it in?",
                              f"'{c['title']}' 카드 담당자가 누구고 지금 어느 컬럼에 있어요?"),
                gold=[mcp("kanban", "list_cards"), mcp("kanban", "list_members")],
                final=L(lang, f"{name} ({c['assignee']}) owns it; it is in {c['column']}.", f"담당자는 {name}({c['assignee']})이고 {c['column']} 컬럼에 있습니다."),
                answers=[text(name, [c["assignee"]]), text(c["column"], [c["column"].replace("_", " ")])])


def g_kanban_move(rng, env, lang):
    k = worlds.build_kanban(_seed(env))
    c = rng.choice([c for c in _unique_title_cards(k) if c["column"] != "done"])
    return dict(category="write", difficulty=2, mcp_servers=["kanban"],
                instruction=L(lang, f"'{c['title']}' is finished. Please move it to done on the board.",
                              f"'{c['title']}' 작업이 끝났어요. 보드에서 done으로 옮겨 주세요."),
                gold=[mcp("kanban", "list_cards"), mcp("kanban", "move_card", card_id=c["card_id"], column="done")],
                final=L(lang, f"Moved {c['card_id']} to done.", f"{c['card_id']}를 done으로 옮겼습니다."),
                checks=[mcp_check("kanban", "get_card", {"card_id": c["card_id"]}, json_subset={"column": "done"})])


def g_kanban_create_assign(rng, env, lang):
    k = worlds.build_kanban(_seed(env))
    m = rng.choice(k["members"])
    title = rng.choice(["Prepare Q4 capacity plan", "Fix flaky checkout test", "Update on-call runbook",
                        "Review vendor contract", "Add metrics to search API", "Clean up feature flags"])
    col = rng.choice(["todo", "in_progress"])
    return dict(category="multi_write", difficulty=3, mcp_servers=["kanban"],
                instruction=L(lang, f"Add a card '{title}' to the {col} column and assign it to {m['name']}.",
                              f"'{title}' 카드를 {COL_KO[col]} 컬럼에 만들고 {m['name']}에게 배정해 주세요."),
                gold=[mcp("kanban", "list_members"), mcp("kanban", "create_card", title=title, column=col, assignee=m["username"])],
                final=L(lang, f"Created '{title}' in {col}, assigned to {m['name']}.", f"'{title}' 카드를 만들고 {m['name']}에게 배정했습니다."),
                checks=[mcp_check("kanban", "list_cards", {"column": col, "assignee": m["username"]}, contains=[title])])


def g_kanban_bulk_move(rng, env, lang):
    k = worlds.build_kanban(_seed(env))
    by = {}
    for c in k["cards"].values():
        if c["column"] == "review" and c["assignee"]:
            by.setdefault(c["assignee"], []).append(c)
    if not by:
        raise GenSkip("nobody has review cards")
    u = rng.choice(sorted(by))
    cards = sorted(by[u], key=lambda c: c["card_id"])
    name = _member_name(k, u)
    gold = [mcp("kanban", "list_members"), mcp("kanban", "list_cards", column="review", assignee=u)]
    gold += [mcp("kanban", "move_card", card_id=c["card_id"], column="done") for c in cards]
    return dict(category="chain", difficulty=3, mcp_servers=["kanban"],
                instruction=L(lang, f"{name} approved everything they had in review. Move all of their review cards to done.",
                              f"{name}님이 리뷰 중이던 카드를 모두 승인했어요. {name}님의 review 카드를 전부 done으로 옮겨 주세요."),
                gold=gold,
                final=L(lang, f"Moved {len(cards)} card(s) to done.", f"카드 {len(cards)}개를 done으로 옮겼습니다."),
                checks=[mcp_check("kanban", "list_cards", {"column": "done", "assignee": u}, contains=[c["title"] for c in cards]),
                        mcp_check("kanban", "list_cards", {"column": "review", "assignee": u}, not_contains=[c["title"] for c in cards])])


def g_kanban_comment(rng, env, lang):
    k = worlds.build_kanban(_seed(env))
    c = rng.choice(_unique_title_cards(k))
    note = rng.choice(["Blocked by vendor API outage", "Waiting for design sign-off", "Needs security review",
                       "Pairing with Mina tomorrow"])
    return dict(category="write", difficulty=2, mcp_servers=["kanban"],
                instruction=L(lang, f"Leave a comment on the '{c['title']}' card saying: {note}",
                              f"'{c['title']}' 카드에 다음 코멘트를 남겨 주세요: {note}"),
                gold=[mcp("kanban", "list_cards"), mcp("kanban", "add_comment", card_id=c["card_id"], text=note)],
                final=L(lang, f"Comment added to {c['card_id']}.", f"{c['card_id']}에 코멘트를 남겼습니다."),
                checks=[mcp_check("kanban", "get_card", {"card_id": c["card_id"]}, contains=[note])])


# ---------------------------------------------------------------------------------------------- wiki
def _doc(w, prefix):
    return next(d for d in w["docs"] if d["title"].startswith(prefix))


def g_wiki_expense(rng, env, lang):
    w = worlds.build_wiki(_seed(env))
    d = _doc(w, "Expense policy")
    meal = int(d["body"].split("up to $")[1].split(" ")[0])
    hotel = int(d["body"].split("up to $")[2].split(" ")[0])
    which, v = rng.choice([("meal", meal), ("hotel", hotel)])
    return dict(category="lookup", difficulty=2, mcp_servers=["wiki"],
                instruction=L(lang, "What's the reimbursement limit per person for a team meal?" if which == "meal"
                              else "How much can I expense per night for a hotel?",
                              "팀 회식비는 1인당 얼마까지 정산돼요?" if which == "meal" else "호텔은 1박에 얼마까지 정산돼요?"),
                gold=[mcp("wiki", "search_docs", query="expense policy"), mcp("wiki", "get_doc", doc_id=d["doc_id"])],
                final=L(lang, f"Up to ${v} ({d['title']}).", f"${v}까지입니다({d['title']})."),
                answers=[num(v, 0)])


def g_wiki_oncall(rng, env, lang):
    w = worlds.build_wiki(_seed(env))
    d = rng.choice([d for d in w["docs"] if d["space"] == "oncall"])
    team = d["title"].split(" on-call")[0]
    primary = d["body"].split("primary on-call for ")[1].split(" is ")[1].split(" (phone")[0]
    phone = d["body"].split("(phone ")[1].split(")")[0]
    return dict(category="chain", difficulty=2, mcp_servers=["wiki"],
                instruction=L(lang, f"Who is the primary on-call for the {team} team this week, and what's their phone number?",
                              f"이번 주 {team} 팀 primary on-call이 누구고 전화번호가 뭐예요?"),
                gold=[mcp("wiki", "search_docs", query=f"{team} on-call"), mcp("wiki", "get_doc", doc_id=d["doc_id"])],
                final=L(lang, f"{primary}, {phone}.", f"{primary}, {phone}입니다."),
                answers=[text(primary), text(phone)])


def g_wiki_release(rng, env, lang):
    w = worlds.build_wiki(_seed(env))
    d = rng.choice([d for d in w["docs"] if d["space"] == "products"])
    product = d["title"].split(" release")[0]
    version = d["body"].split(f"{product} ")[1].split(" was released")[0]
    date = d["body"].split("released on ")[1].split(".")[0]
    return dict(category="lookup", difficulty=2, mcp_servers=["wiki"],
                instruction=L(lang, f"What is the latest version of {product}, and when was it released?",
                              f"{product} 최신 버전이 뭐고 언제 출시됐어요?"),
                gold=[mcp("wiki", "search_docs", query=f"{product} release"), mcp("wiki", "get_doc", doc_id=d["doc_id"])],
                final=L(lang, f"{product} {version}, released {date}.", f"{product} {version}, {date} 출시입니다."),
                answers=[text(version), text(date)])


def g_wiki_irrelevant(rng, env, lang):
    q, a = rng.choice(GENERAL_QA[lang])
    return dict(category="irrelevant", difficulty=1, mcp_servers=["wiki"], instruction=q, gold=[], final=a[0],
                answers=[text(a[0], a[1:])])


# ---------------------------------------------------------------------------------------------- memory
SEAT = {"prefers window seats": ("window", "창가"), "prefers aisle seats": ("aisle", "통로")}


def g_memory_recall(rng, env, lang):
    m = worlds.build_memory(_seed(env))
    people = [e for e in m["entities"] if e["entityType"] == "person" and any(o in SEAT for o in e["observations"])]
    if not people:
        raise GenSkip("no seat preference stored")
    e = rng.choice(people)
    obs = next(o for o in e["observations"] if o in SEAT)
    en, ko = SEAT[obs]
    return dict(category="lookup", difficulty=2, mcp_servers=["memory"],
                instruction=L(lang, f"I'm booking a flight for {e['name']}. Do we have a seat preference on file for them?",
                              f"{e['name']}님 항공권을 예약하려는데, 저장된 좌석 선호가 있나요?"),
                gold=[mcp("memory", "search_nodes", query=e["name"])],
                final=L(lang, f"Yes: {e['name']} {obs}.", f"네, {e['name']}님은 {ko} 좌석을 선호합니다."),
                answers=[text(en, [ko])])


def g_memory_store(rng, env, lang):
    m = worlds.build_memory(_seed(env))
    e = rng.choice([e for e in m["entities"] if e["entityType"] == "person"])
    fact = rng.choice([f for f in ("is allergic to shellfish", "prefers vegetarian meals", "moved to the Austin office",
                                   "will be out of office on 2026-11-20", "prefers calls after 2pm")
                       if f not in e["observations"]])
    return dict(category="write", difficulty=2, mcp_servers=["memory"],
                instruction=L(lang, f"Please remember this for later: {e['name']} {fact}.",
                              f"나중을 위해 기억해 두세요: {e['name']} {fact}."),
                gold=[mcp("memory", "search_nodes", query=e["name"]),
                      mcp("memory", "add_observations", observations=[{"entityName": e["name"], "contents": [fact]}])],
                final=L(lang, f"Saved: {e['name']} {fact}.", f"저장했습니다: {e['name']} {fact}."),
                checks=[mcp_check("memory", "open_nodes", {"names": [e["name"]]}, contains=[fact])])


def g_memory_relation(rng, env, lang):
    m = worlds.build_memory(_seed(env))
    people = [e["name"] for e in m["entities"] if e["entityType"] == "person"]
    a, b = rng.sample(people, 2)
    if {"from": a, "to": b, "relationType": "manages"} in m["relations"]:
        raise GenSkip("relation exists")
    return dict(category="write", difficulty=3, mcp_servers=["memory"],
                instruction=L(lang, f"Update the memory graph: {a} now manages {b}. Use the relation type 'manages'.",
                              f"메모리 그래프를 갱신해 주세요: 이제 {a}가 {b}를 관리합니다. 관계 이름은 'manages'로 해 주세요."),
                gold=[mcp("memory", "open_nodes", names=[a, b]),
                      mcp("memory", "create_relations", relations=[{"from": a, "to": b, "relationType": "manages"}])],
                final=L(lang, f"Recorded: {a} manages {b}.", f"{a} → {b} manages 관계를 기록했습니다."),
                checks=[mcp_check("memory", "open_nodes", {"names": [a, b]},
                                  json_subset={"relations": [{"from": a, "to": b, "relationType": "manages"}]})])


def g_memory_org(rng, env, lang):
    m = worlds.build_memory(_seed(env))
    orgs = [e["name"] for e in m["entities"] if e["entityType"] == "organization"]
    org = rng.choice(orgs)
    members = sorted(r["from"] for r in m["relations"] if r["to"] == org and r["relationType"] == "works_at")
    if not members:
        raise GenSkip("empty org")
    return dict(category="chain", difficulty=3, mcp_servers=["memory"],
                instruction=L(lang, f"According to our memory graph, who works at {org}?", f"메모리 그래프 기준으로 {org}에서 일하는 사람이 누구예요?"),
                gold=[mcp("memory", "search_nodes", query=org)],
                final=L(lang, ", ".join(members) + ".", ", ".join(members) + "입니다."),
                answers=[text(n) for n in members])


# ---------------------------------------------------------------------------------------------- built-in env + MCP
def g_cross_meeting_card(rng, env, lang):
    """Combines a built-in env (workspace calendar) with an external MCP server (kanban)."""
    seed = _seed(env)
    ws = make_env("workspace", env.world)
    k = worlds.build_kanban(seed)
    evs = [e for e in ws.db["events"].values()
           if sum(1 for x in ws.db["events"].values() if x["title"] == e["title"] and x["date"] == e["date"]) == 1]
    e = rng.choice(evs)
    m = rng.choice(k["members"])
    title = f"Prep: {e['title']}"
    day = L(lang, f"{WEEKDAY[e['date']]} ({e['date']})", f"{KO_DAY[WEEKDAY[e['date']]]}({e['date']})")
    return dict(domain="workspace", category="cross", difficulty=4, mcp_servers=["kanban"],
                instruction=L(lang, f"For the '{e['title']}' meeting on {day}, add a kanban card titled '{title}' in todo, "
                                    f"assigned to {m['name']}, due on the meeting date.",
                              f"{day} '{e['title']}' 회의 준비용으로 칸반에 '{title}' 카드를 todo에 만들어 {m['name']}에게 배정하고, "
                              f"마감일은 회의 날짜로 해 주세요."),
                gold=[call("list_events", date=e["date"]), mcp("kanban", "list_members"),
                      mcp("kanban", "create_card", title=title, column="todo", assignee=m["username"], due=e["date"])],
                final=L(lang, f"Created '{title}' (due {e['date']}) for {m['name']}.", f"'{title}' 카드를 만들었습니다(마감 {e['date']})."),
                checks=[mcp_check("kanban", "list_cards", {"column": "todo", "assignee": m["username"]}, contains=[title, e["date"]])])


GENERATORS = {
    "kanban_who": g_kanban_who, "kanban_move": g_kanban_move, "kanban_create_assign": g_kanban_create_assign,
    "kanban_bulk_move": g_kanban_bulk_move, "kanban_comment": g_kanban_comment,
    "wiki_expense": g_wiki_expense, "wiki_oncall": g_wiki_oncall, "wiki_release": g_wiki_release,
    "wiki_irrelevant": g_wiki_irrelevant,
    "memory_recall": g_memory_recall, "memory_store": g_memory_store, "memory_relation": g_memory_relation,
    "memory_org": g_memory_org, "cross_meeting_card": g_cross_meeting_card,
} if worlds is not None else {}
