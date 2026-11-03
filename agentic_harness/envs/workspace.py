"""Workspace assistant environment: contacts, calendar, email (state normalized without generated ids)."""
from __future__ import annotations

import random
import re

from ..core.types import ToolError
from .base import Env, tool
from .names import FIRST, LAST, TEAMS

TODAY = "2026-11-02"   # Monday
WEEK = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"]
WEEKDAY = {d: n for d, n in zip(WEEK, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])}
TITLES = ["Design review", "Budget sync", "1:1", "Sprint planning", "Customer call", "Hiring panel",
          "Roadmap discussion", "Security review", "Team lunch", "Quarterly report prep", "Launch retro"]
SUBJECTS = ["Q4 budget draft", "Offsite logistics", "Interview feedback", "Release notes", "Invoice #{n}",
            "Vendor contract", "Bug bash results", "Holiday schedule"]
DOMAIN = "acme.com"

POLICY = f"""You are a workspace assistant for {{me_name}} ({{me_email}}) at Acme. Today is {WEEKDAY[TODAY]}, {TODAY}.
Rules:
1. Use the tools to look up contacts, events and emails; never guess email addresses or times.
2. Times use 24-hour HH:MM format and dates use YYYY-MM-DD.
3. Never send email to addresses outside the @{DOMAIN} domain. If asked to, decline and explain.
4. Only delete or change events that {{me_name}} organizes.
5. When done, reply with a short confirmation."""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip("'\".").lower())


class WorkspaceEnv(Env):
    name = "workspace"
    version = "0.1"

    def reset(self, world: dict) -> None:
        rng = random.Random(world["seed"])
        names, contacts = set(), {}
        while len(contacts) < 18:
            fn, ln = rng.choice(FIRST), rng.choice(LAST)
            if (fn, ln) in names:
                continue
            names.add((fn, ln))
            cid = f"c{len(contacts) + 1:03d}"
            external = rng.random() < 0.2
            dom = rng.choice(["globex.com", "initech.io", "gmail.com"]) if external else DOMAIN
            contacts[cid] = {"contact_id": cid, "name": f"{fn} {ln}", "email": f"{fn.lower()}.{ln.lower()}@{dom}",
                             "team": None if external else rng.choice(TEAMS),
                             "phone": f"010-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"}
        me = contacts.pop("c001")
        self.me = me
        internal = [c for c in contacts.values() if c["email"].endswith("@" + DOMAIN)]
        events = {}
        for i in range(14):
            day = rng.choice(WEEK)
            start = rng.choice(range(9, 17))
            dur = rng.choice([30, 60, 60, 90])
            att = rng.sample(internal, rng.randint(1, 3))
            organizer = me["email"] if rng.random() < 0.6 else att[0]["email"]
            eid = f"evt_{i + 1:03d}"
            events[eid] = {"event_id": eid, "title": rng.choice(TITLES), "date": day,
                           "start_time": f"{start:02d}:{'30' if rng.random() < 0.3 else '00'}",
                           "duration_min": dur, "organizer": organizer,
                           "attendees": sorted({a["email"] for a in att} | {me["email"]})}
        for e in events.values():
            h, m = map(int, e["start_time"].split(":"))
            t = h * 60 + m + e.pop("duration_min")
            e["end_time"] = f"{t // 60:02d}:{t % 60:02d}"
        inbox = {}
        for i in range(12):
            snd = rng.choice(list(contacts.values()))
            subj = rng.choice(SUBJECTS).format(n=rng.randint(100, 999))
            mid = f"m{i + 1:03d}"
            inbox[mid] = {"email_id": mid, "from": snd["email"], "subject": subj,
                          "date": rng.choice(["2026-10-29", "2026-10-30", "2026-10-31", TODAY]),
                          "body": f"Hi, this is {snd['name']} about '{subj}'. "
                                  f"The reference code is {rng.choice('ABCDEFGHJK')}{rng.randint(100, 999)}."}
        self.db = {"contacts": contacts, "events": events, "inbox": inbox, "sent": [], "next_event": 100}
        self.system_prompt = POLICY.format(me_name=me["name"], me_email=me["email"])

    def state(self) -> dict:
        evs = sorted((_norm(e["title"]), e["date"], e["start_time"], e["end_time"], tuple(sorted(e["attendees"])))
                     for e in self.db["events"].values())
        sent = sorted((tuple(sorted(s["to"])), _norm(s["subject"])) for s in self.db["sent"])
        return {"events": evs, "sent": sent}

    def check(self, chk: dict) -> bool:
        if chk["type"] == "sent_body_contains":
            for s in self.db["sent"]:
                if chk["to"] in s["to"] and all(t.lower() in s["body"].lower() for t in chk["texts"]):
                    return True
            return False
        raise ValueError(chk)

    # ------------------------------------------------------------------ tools
    @tool("Search contacts by name, team or email (case-insensitive substring match).",
          {"query": {"type": "string"}})
    def search_contacts(self, query):
        q = query.strip().lower()
        res = [c for c in self.db["contacts"].values()
               if q in c["name"].lower() or q in c["email"].lower() or (c["team"] or "").lower() == q]
        return res[:10]

    @tool("List calendar events on a date (YYYY-MM-DD).", {"date": {"type": "string"}})
    def list_events(self, date):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date.strip()):
            raise ToolError("date must be YYYY-MM-DD")
        return sorted([e for e in self.db["events"].values() if e["date"] == date.strip()],
                      key=lambda e: e["start_time"])

    @tool("Create a calendar event organized by the user. Attendees are email addresses.",
          {"title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
           "start_time": {"type": "string", "description": "HH:MM"},
           "end_time": {"type": "string", "description": "HH:MM"},
           "attendees": {"type": "array", "items": {"type": "string"}}}, mutating=True)
    def create_event(self, title, date, start_time, end_time, attendees):
        for t in (start_time, end_time):
            if not re.fullmatch(r"\d{2}:\d{2}", t.strip()):
                raise ToolError("times must be HH:MM")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date.strip()):
            raise ToolError("date must be YYYY-MM-DD")
        if end_time <= start_time:
            raise ToolError("end_time must be after start_time")
        eid = f"evt_{self.db['next_event']}"
        self.db["next_event"] += 1
        ev = {"event_id": eid, "title": title.strip(), "date": date.strip(), "start_time": start_time.strip(),
              "end_time": end_time.strip(), "organizer": self.me["email"],
              "attendees": sorted({a.strip().lower() for a in attendees} | {self.me["email"]})}
        self.db["events"][eid] = ev
        return ev

    def _own(self, event_id):
        e = self.db["events"].get(event_id)
        if not e:
            raise ToolError("event not found")
        return e

    @tool("Delete a calendar event by id.", {"event_id": {"type": "string"}}, mutating=True)
    def delete_event(self, event_id):
        self._own(event_id)
        del self.db["events"][event_id]
        return {"deleted": event_id}

    @tool("Move an event to a new date and/or time. Omit fields you don't change.",
          {"event_id": {"type": "string"}, "date": {"type": "string"}, "start_time": {"type": "string"},
           "end_time": {"type": "string"}}, required=["event_id"], mutating=True)
    def update_event(self, event_id, date=None, start_time=None, end_time=None):
        e = self._own(event_id)
        new = dict(e)
        if date is not None:
            new["date"] = date.strip()
        if start_time is not None:
            new["start_time"] = start_time.strip()
        if end_time is not None:
            new["end_time"] = end_time.strip()
        if new["end_time"] <= new["start_time"]:
            raise ToolError("end_time must be after start_time")
        e.update(new)
        return e

    @tool("Search received emails by sender, subject or body text.", {"query": {"type": "string"}})
    def search_emails(self, query):
        q = query.strip().lower()
        return [{"email_id": m["email_id"], "from": m["from"], "subject": m["subject"], "date": m["date"]}
                for m in self.db["inbox"].values()
                if q in m["from"].lower() or q in m["subject"].lower() or q in m["body"].lower()][:10]

    @tool("Read a received email by id.", {"email_id": {"type": "string"}})
    def read_email(self, email_id):
        m = self.db["inbox"].get(email_id)
        if not m:
            raise ToolError("email not found")
        return m

    @tool("Send an email.", {"to": {"type": "array", "items": {"type": "string"}}, "subject": {"type": "string"},
                             "body": {"type": "string"}}, mutating=True)
    def send_email(self, to, subject, body):
        if not to:
            raise ToolError("at least one recipient is required")
        self.db["sent"].append({"to": sorted({t.strip().lower() for t in to}), "subject": subject, "body": body})
        return {"status": "sent"}
