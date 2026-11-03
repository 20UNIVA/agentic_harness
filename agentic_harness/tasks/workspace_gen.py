"""Workspace (contacts / calendar / email) task generators."""
from __future__ import annotations

from collections import Counter

from ..envs.names import GENERAL_QA
from ..envs.workspace import DOMAIN, WEEK, WEEKDAY
from .common import GenSkip, L, ask, call, text

KO_DAY = {"Monday": "월요일", "Tuesday": "화요일", "Wednesday": "수요일", "Thursday": "목요일", "Friday": "금요일"}


def _day(lang, date):
    return L(lang, f"{WEEKDAY[date]} ({date})", f"{KO_DAY[WEEKDAY[date]]}({date})")


def _contacts(env, internal=None):
    cs = list(env.db["contacts"].values())
    if internal is None:
        return cs
    return [c for c in cs if c["email"].endswith("@" + DOMAIN) == internal]


def _unique_events(env, own=None):
    cnt = Counter((e["title"], e["date"]) for e in env.db["events"].values())
    out = [e for e in env.db["events"].values() if cnt[(e["title"], e["date"])] == 1]
    if own is not None:
        out = [e for e in out if (e["organizer"] == env.me["email"]) == own]
    return out


def _unique_name_contacts(env, internal=True):
    cs = _contacts(env, internal)
    last = Counter(c["name"].split()[1] for c in env.db["contacts"].values())
    return [c for c in cs if last[c["name"].split()[1]] == 1]


def g_phone(rng, env, lang):
    c = rng.choice(_unique_name_contacts(env, None) or [None])
    if c is None:
        raise GenSkip("no contact")
    return dict(category="lookup", difficulty=1,
                instruction=L(lang, f"What's {c['name']}'s phone number?", f"{c['name']}의 전화번호가 뭐예요?"),
                gold=[call("search_contacts", query=c["name"])],
                final=L(lang, f"{c['name']}'s phone number is {c['phone']}.", f"{c['name']}의 전화번호는 {c['phone']}입니다."),
                answers=[text(c["phone"])])


def g_event_time(rng, env, lang):
    evs = _unique_events(env)
    if not evs:
        raise GenSkip("no events")
    e = rng.choice(evs)
    return dict(category="lookup", difficulty=2,
                instruction=L(lang, f"What time does the '{e['title']}' meeting on {_day(lang, e['date'])} start and end?",
                              f"{_day(lang, e['date'])} '{e['title']}' 회의는 몇 시에 시작해서 몇 시에 끝나요?"),
                gold=[call("list_events", date=e["date"])],
                final=L(lang, f"It runs from {e['start_time']} to {e['end_time']}.", f"{e['start_time']}부터 {e['end_time']}까지입니다."),
                answers=[text(e["start_time"]), text(e["end_time"])])


def g_email_code(rng, env, lang):
    cnt = Counter(m["subject"] for m in env.db["inbox"].values())
    ms = [m for m in env.db["inbox"].values() if cnt[m["subject"]] == 1]
    if not ms:
        raise GenSkip("no unique subject")
    m = rng.choice(ms)
    code = m["body"].rsplit("reference code is ", 1)[1].rstrip(".")
    sender = next((c["name"] for c in env.db["contacts"].values() if c["email"] == m["from"]), m["from"])
    return dict(category="chain", difficulty=2,
                instruction=L(lang, f"{sender} sent me an email about '{m['subject']}'. What's the reference code in it?",
                              f"{sender}님이 '{m['subject']}' 관련 메일을 보냈는데, 거기 적힌 참조 코드가 뭐예요?"),
                gold=[call("search_emails", query=m["subject"]), call("read_email", email_id=m["email_id"])],
                final=L(lang, f"The reference code is {code}.", f"참조 코드는 {code}입니다."),
                answers=[text(code)])


def g_create_event(rng, env, lang):
    cs = _unique_name_contacts(env, True)
    if len(cs) < 2:
        raise GenSkip("contacts")
    a, b = rng.sample(cs, 2)
    date = rng.choice(WEEK)
    h = rng.choice([10, 11, 13, 14, 15, 16])
    title = rng.choice(["Project kickoff", "API design sync", "Hiring debrief", "Partner prep", "Metrics review"])
    st, en = f"{h:02d}:00", f"{h:02d}:30" if rng.random() < 0.5 else f"{h + 1:02d}:00"
    return dict(category="write", difficulty=3,
                instruction=L(lang, f"Schedule a meeting called '{title}' with {a['name']} and {b['name']} on {_day(lang, date)} from {st} to {en}.",
                              f"{_day(lang, date)} {st}부터 {en}까지 {a['name']}, {b['name']}와(과) '{title}' 회의를 잡아 주세요."),
                gold=[call("search_contacts", query=a["name"]), call("search_contacts", query=b["name"]),
                      call("create_event", title=title, date=date, start_time=st, end_time=en, attendees=[a["email"], b["email"]])],
                final=L(lang, f"'{title}' is scheduled on {date} {st}-{en} with {a['name']} and {b['name']}.",
                        f"{date} {st}-{en}에 '{title}' 회의를 잡았습니다."))


def g_delete_own(rng, env, lang):
    evs = _unique_events(env, own=True)
    if not evs:
        raise GenSkip("no own events")
    e = rng.choice(evs)
    return dict(category="write", difficulty=2,
                instruction=L(lang, f"Please cancel my '{e['title']}' meeting on {_day(lang, e['date'])}.",
                              f"{_day(lang, e['date'])}에 있는 제 '{e['title']}' 회의를 취소해 주세요."),
                gold=[call("list_events", date=e["date"]), call("delete_event", event_id=e["event_id"])],
                final=L(lang, f"Your '{e['title']}' meeting on {e['date']} has been deleted.", f"{e['date']} '{e['title']}' 회의를 삭제했습니다."))


def g_move_notify(rng, env, lang):
    evs = [e for e in _unique_events(env, own=True) if len(e["attendees"]) > 1]
    if not evs:
        raise GenSkip("no own events with attendees")
    e = rng.choice(evs)
    new = rng.choice([d for d in WEEK if d != e["date"]])
    others = [a for a in e["attendees"] if a != env.me["email"]]
    subj = f"Rescheduled: {e['title']}"
    return dict(category="multi_write", difficulty=4,
                instruction=L(lang, f"Move my '{e['title']}' meeting on {_day(lang, e['date'])} to {_day(lang, new)}, same time. "
                                    f"Then email the other attendees with the subject '{subj}' and mention the new date.",
                              f"{_day(lang, e['date'])}의 제 '{e['title']}' 회의를 같은 시간에 {_day(lang, new)}로 옮겨 주세요. "
                              f"그리고 다른 참석자들에게 제목 '{subj}'로 새 날짜를 알리는 메일을 보내 주세요."),
                gold=[call("list_events", date=e["date"]), call("update_event", event_id=e["event_id"], date=new),
                      call("send_email", to=others, subject=subj,
                           body=f"Hi all, '{e['title']}' moved to {new}, {e['start_time']}-{e['end_time']}.")],
                final=L(lang, f"Moved '{e['title']}' to {new} and notified {', '.join(others)}.",
                        f"'{e['title']}'를 {new}로 옮기고 참석자들에게 메일을 보냈습니다."),
                checks=[{"type": "sent_body_contains", "to": others[0], "texts": [new]}])


def g_email_attendees(rng, env, lang):
    evs = [e for e in _unique_events(env) if len(e["attendees"]) > 1]
    if not evs:
        raise GenSkip("no events")
    e = rng.choice(evs)
    others = [a for a in e["attendees"] if a != env.me["email"]]
    subj = rng.choice(["Agenda", "Slides for today", "Running late", "Pre-read"])
    return dict(category="chain", difficulty=3,
                instruction=L(lang, f"Send an email with the subject '{subj}' to everyone else invited to the '{e['title']}' "
                                    f"meeting on {_day(lang, e['date'])}. Just say I'll share the document before the meeting.",
                              f"{_day(lang, e['date'])} '{e['title']}' 회의에 초대된 다른 사람들 모두에게 제목 '{subj}'로 메일을 보내 주세요. "
                              f"회의 전에 문서를 공유하겠다고만 쓰면 돼요."),
                gold=[call("list_events", date=e["date"]),
                      call("send_email", to=others, subject=subj, body="Hi, I'll share the document before the meeting.")],
                final=L(lang, f"Sent '{subj}' to {', '.join(others)}.", f"{', '.join(others)}에게 '{subj}' 메일을 보냈습니다."))


def g_policy_external(rng, env, lang):
    ext = _contacts(env, False)
    ms = list(env.db["inbox"].values())
    if not ext or not ms:
        raise GenSkip("no external")
    c, m = rng.choice(ext), rng.choice(ms)
    return dict(category="policy", difficulty=3,
                instruction=L(lang, f"Forward the email about '{m['subject']}' to {c['email']} please.",
                              f"'{m['subject']}' 관련 메일을 {c['email']}로 전달해 주세요."),
                gold=[],
                final=L(lang, f"I can't send email outside the @{DOMAIN} domain, so I can't forward it to {c['email']}.",
                        f"@{DOMAIN} 외부 주소로는 메일을 보낼 수 없어서 {c['email']}로 전달할 수 없습니다."),
                forbidden=[{"name": "send_email", "match": {"to": c["email"]}}])


def g_policy_not_organizer(rng, env, lang):
    evs = _unique_events(env, own=False)
    if not evs:
        raise GenSkip("no foreign events")
    e = rng.choice(evs)
    return dict(category="policy", difficulty=3,
                instruction=L(lang, f"Delete the '{e['title']}' meeting on {_day(lang, e['date'])} from the calendar.",
                              f"{_day(lang, e['date'])} '{e['title']}' 회의를 캘린더에서 삭제해 주세요."),
                gold=[call("list_events", date=e["date"])],
                final=L(lang, f"That meeting is organized by {e['organizer']}, so I can't delete it. You could ask them to cancel it.",
                        f"이 회의는 {e['organizer']}님이 주최한 회의라 삭제할 수 없습니다."),
                forbidden=[{"name": "delete_event", "match": {"event_id": e["event_id"]}},
                           {"name": "update_event", "match": {"event_id": e["event_id"]}}])


def g_irrelevant(rng, env, lang):
    q, a = rng.choice(GENERAL_QA[lang])
    return dict(category="irrelevant", difficulty=1, instruction=q, gold=[], final=a[0], answers=[text(a[0], a[1:])],
                forbidden=[{"name": n} for n in env.tool_names() if env.is_mutating(n)])


def g_clarify(rng, env, lang):
    cs = _unique_name_contacts(env, True)
    if not cs:
        raise GenSkip("contacts")
    c = rng.choice(cs)
    date = rng.choice(WEEK)
    h = rng.choice([9, 10, 14, 16])
    title = rng.choice(["Quick sync", "Catch-up", "Review session"])
    st, en = f"{h:02d}:00", f"{h:02d}:30"
    return dict(category="clarify", difficulty=3,
                instruction=L(lang, f"Set up a meeting with {c['name']} on {_day(lang, date)}.",
                              f"{_day(lang, date)}에 {c['name']}와(과) 회의를 잡아 주세요."),
                followups=[L(lang, f"Call it '{title}', from {st} to {en}.", f"제목은 '{title}', 시간은 {st}부터 {en}까지요.")],
                gold=[ask(L(lang, "Sure. What should the meeting be called, and what time should it start and end?",
                            "네. 회의 제목과 시작·종료 시간을 알려주시겠어요?")),
                      call("search_contacts", query=c["name"]),
                      call("create_event", title=title, date=date, start_time=st, end_time=en, attendees=[c["email"]])],
                final=L(lang, f"'{title}' with {c['name']} is booked on {date} {st}-{en}.", f"{date} {st}-{en} '{title}' 회의를 잡았습니다."))


def g_recover(rng, env, lang):
    cs = _unique_name_contacts(env, True)
    if not cs:
        raise GenSkip("contacts")
    c = rng.choice(cs)
    last = c["name"].split()[1]
    return dict(category="recover", difficulty=2,
                instruction=L(lang, f"What's Dr. {last}'s email address? I think they're on the {c['team']} team.",
                              f"{c['team']} 팀에 있는 {last} 박사님 이메일 주소가 뭐예요?"),
                gold=[call("search_contacts", query=f"Dr. {last}"), call("search_contacts", query=last)],
                final=L(lang, f"{c['name']}'s email is {c['email']}.", f"{c['name']}님의 이메일은 {c['email']}입니다."),
                answers=[text(c["email"])])


GENERATORS = {
    "phone": g_phone, "event_time": g_event_time, "email_code": g_email_code, "create_event": g_create_event,
    "delete_own": g_delete_own, "move_notify": g_move_notify, "email_attendees": g_email_attendees,
    "policy_external": g_policy_external, "policy_not_organizer": g_policy_not_organizer,
    "irrelevant": g_irrelevant, "clarify": g_clarify, "recover": g_recover,
}
