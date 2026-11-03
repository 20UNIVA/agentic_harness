"""Seed-deterministic initial worlds of the example MCP servers (stdlib only).

Imported by the servers and by the task generators (agentic_harness/tasks/mcp_examples_gen.py), so tasks know the
ground truth without starting a server.
"""
import random


# ---------------------------------------------------------------- kanban
COLUMNS = ["todo", "in_progress", "review", "done"]
MEMBERS = [("mina", "Mina Park"), ("jason", "Jason Lee"), ("sora", "Sora Kim"), ("david", "David Choi"),
           ("emily", "Emily Jung"), ("hyejin", "Hyejin Kang"), ("omar", "Omar Haddad"), ("lucia", "Lucia Rossi")]
TITLES = ["Fix login redirect loop", "Add CSV export to reports", "Upgrade database driver", "Write onboarding guide",
          "Rate-limit public API", "Migrate CI to new runners", "Dark mode for dashboard", "Audit IAM roles",
          "Cache product search results", "Remove legacy payment flow", "Add retry to webhook sender",
          "Localize checkout page (ko)", "Investigate memory leak in worker", "Set up error budget alerts",
          "Refactor notification service", "Load test the gateway", "Document release process", "Rotate API keys"]
LABELS = ["bug", "feature", "infra", "docs", "security", "perf"]


def build_kanban(seed: int) -> dict:
    rng = random.Random(f"kanban|{seed}")
    members = rng.sample(MEMBERS, 6)
    cards = {}
    for i, title in enumerate(rng.sample(TITLES, 12)):
        cid = f"CARD-{101 + i}"
        cards[cid] = {"card_id": cid, "title": title, "column": rng.choice(COLUMNS),
                      "assignee": rng.choice([m[0] for m in members] + [None]),
                      "labels": sorted(rng.sample(LABELS, rng.randint(1, 2))),
                      "due": f"2026-11-{rng.randint(3, 28):02d}", "comments": []}
    return {"members": [{"username": u, "name": n} for u, n in members], "cards": cards, "next_id": 101 + len(cards)}


# ---------------------------------------------------------------- wiki
WIKI_TEAMS = ["Platform", "Payments", "Search", "Mobile", "Data", "Security"]
WIKI_OFFICES = ["Seoul", "Busan", "Austin", "Berlin"]
WIKI_PEOPLE = ["Mina Park", "Jason Lee", "Sora Kim", "David Choi", "Emily Jung", "Hyejin Kang", "Omar Haddad", "Lucia Rossi",
          "Kenji Sato", "Priya Patel", "Lars Muller", "Yuna Lim"]
WIKI_PRODUCTS = ["Atlas", "Beacon", "Comet", "Delta", "Ember"]


def build_wiki(seed: int) -> dict:
    rng = random.Random(f"wiki|{seed}")
    docs = []

    def add(space, title, body, updated):
        docs.append({"doc_id": f"DOC-{len(docs) + 1:03d}", "space": space, "title": title, "body": body, "updated": updated})

    meal = rng.choice([30, 40, 50, 60])
    hotel = rng.choice([150, 180, 200, 250])
    add("policies", "Expense policy",
        f"Team meals are reimbursed up to ${meal} per person. Hotel stays are reimbursed up to ${hotel} per night. "
        f"Flights over {rng.choice([4, 5, 6])} hours may be booked in premium economy. Submit receipts within "
        f"{rng.choice([14, 30, 45])} days.", "2026-09-01")
    add("policies", "Remote work policy",
        f"Employees may work remotely up to {rng.choice([2, 3])} days per week. Core hours are "
        f"{rng.choice(['10:00-16:00', '11:00-16:00'])} KST.", "2026-08-12")
    add("policies", "Security incident reporting",
        f"Report suspected incidents in #sec-incidents within {rng.choice([1, 2, 4])} hours and page the Security on-call.",
        "2026-07-30")
    for team in rng.sample(WIKI_TEAMS, 4):
        primary, secondary = rng.sample(WIKI_PEOPLE, 2)
        add("oncall", f"{team} on-call rotation",
            f"This week the primary on-call for {team} is {primary} (phone 010-{rng.randint(1000, 9999)}-"
            f"{rng.randint(1000, 9999)}) and the secondary is {secondary}. Escalate after "
            f"{rng.choice([10, 15, 20, 30])} minutes without acknowledgement.", "2026-11-02")
    for office in rng.sample(WIKI_OFFICES, 3):
        add("offices", f"{office} office guide",
            f"The {office} office is on floor {rng.randint(3, 22)}. Guest Wi-Fi network: ACME-{office.upper()}-GUEST, "
            f"password rotates monthly. Badge desk opens at {rng.choice(['08:00', '08:30', '09:00'])}.", "2026-10-05")
    for p in rng.sample(WIKI_PRODUCTS, 3):
        major, minor = rng.randint(2, 6), rng.randint(0, 12)
        add("products", f"{p} release notes",
            f"{p} {major}.{minor} was released on 2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}. "
            f"The previous version was {major}.{max(0, minor - 1)}. Owner team: {rng.choice(WIKI_TEAMS)}.", "2026-10-20")
    return {"docs": docs}


# ---------------------------------------------------------------- memory
MEMORY_PEOPLE = ["Mina Park", "Jason Lee", "Sora Kim", "David Choi", "Emily Jung", "Hyejin Kang", "Omar Haddad", "Lucia Rossi"]
MEMORY_FACTS = ["prefers window seats", "prefers aisle seats", "is vegetarian", "is allergic to peanuts", "drinks only decaf",
         "likes morning meetings", "works from the Busan office", "works from the Seoul office", "speaks Japanese",
         "is on parental leave until 2026-12-15", "prefers email over chat", "plays tennis on Fridays"]
MEMORY_ORGS = ["Acme Platform team", "Acme Payments team", "Globex (customer)", "Initech (vendor)"]


def build_memory(seed: int) -> dict:
    rng = random.Random(f"memory|{seed}")
    people = rng.sample(MEMORY_PEOPLE, 5)
    ents = [{"name": p, "entityType": "person", "observations": rng.sample(MEMORY_FACTS, 2)} for p in people]
    orgs = rng.sample(MEMORY_ORGS, 2)
    ents += [{"name": o, "entityType": "organization", "observations": []} for o in orgs]
    rels = [{"from": p, "to": rng.choice(orgs), "relationType": "works_at"} for p in people]
    rels.append({"from": people[0], "to": people[1], "relationType": "manages"})
    return {"entities": ents, "relations": rels}
