"""Filesystem task generators."""
from __future__ import annotations

import json
import posixpath

from ..envs.filesystem import ROOT
from ..envs.names import GENERAL_QA
from .common import EXAMPLE_SKILLS, GenSkip, L, ask, call, call_err, load_skill, num, text


def _files(env, pred):
    return sorted(p for p in env.db["files"] if pred(p))


def g_config_port(rng, env, lang):
    app = rng.choice(env.apps)
    path = f"{ROOT}/projects/{app}/config.json"
    port = json.loads(env.db["files"][path])["port"]
    return dict(category="lookup", difficulty=2,
                instruction=L(lang, f"Which port is the {app} service configured to use?", f"{app} 서비스는 어떤 포트를 쓰도록 설정돼 있어요?"),
                gold=[call("list_dir", path="projects"), call("read_file", path=f"projects/{app}/config.json")],
                final=L(lang, f"The {app} service uses port {port}.", f"{app} 서비스는 {port} 포트를 사용합니다."),
                answers=[num(port, 0)])


def g_count_errors(rng, env, lang):
    app = rng.choice(env.apps)
    path = f"{ROOT}/projects/{app}/logs/app.log"
    n = sum(1 for line in env.db["files"][path].splitlines() if " ERROR " in line)
    return dict(category="chain", difficulty=2,
                instruction=L(lang, f"How many ERROR lines are in the {app} service's log file?",
                              f"{app} 서비스 로그 파일에 ERROR 줄이 몇 개 있어요?"),
                gold=[call("find_files", pattern="*.log", path=f"projects/{app}"), call("read_file", path=path)],
                final=L(lang, f"There are {n} ERROR lines in {path}.", f"{path}에는 ERROR 줄이 {n}개 있습니다."),
                answers=[num(n, 0)])


def g_find_todo(rng, env, lang):
    hits = [p for p, c in env.db["files"].items() if p.endswith("main.py") and "TODO" in c]
    if len(hits) != 1:
        raise GenSkip("need exactly one TODO")
    p = hits[0]
    app = p.split("/")[-2]
    return dict(category="chain", difficulty=2,
                instruction=L(lang, rng.choice([
                    "One of my projects has a TODO comment in its main.py. Which project is it, and what does the TODO say?",
                    "Somewhere in my projects folder there's a TODO left in the code. Which project, and what is it about?",
                    "Find the TODO comment in my project code and tell me which project it's in and what it says.",
                    "I left myself a TODO note in one of the main.py files. Which service was it and what did I write?"]),
                    rng.choice(["프로젝트 중 하나의 main.py에 TODO 주석이 있어요. 어느 프로젝트이고 TODO 내용이 뭐예요?",
                                "projects 폴더 코드 어딘가에 TODO를 남겨 뒀는데, 어느 프로젝트에 무슨 내용인지 찾아 주세요.",
                                "main.py 파일 중 하나에 TODO 메모가 있어요. 어떤 서비스이고 내용이 뭔가요?"])),
                gold=[call("search_files", text="TODO", path="projects")],
                final=L(lang, f"It's the {app} project: '# TODO: handle retries for {app}'.", f"{app} 프로젝트입니다: '# TODO: handle retries for {app}'."),
                answers=[text(app), text("retries", ["retry", "재시도"])])


def g_edit_port(rng, env, lang):
    app = rng.choice(env.apps)
    path = f"projects/{app}/config.json"
    cfg = json.loads(env.db["files"][f"{ROOT}/{path}"])
    new = rng.choice([p for p in (8081, 8888, 9091, 7000, 6000) if p != cfg["port"]])
    return dict(category="write", difficulty=2,
                instruction=L(lang, f"Change the {app} service's port to {new} in its config file. Don't change anything else.",
                              f"{app} 서비스 설정 파일에서 포트를 {new}로 바꿔 주세요. 다른 건 바꾸지 마세요."),
                gold=[call("read_file", path=path),
                      call("replace_in_file", path=path, old_text=f'"port": {cfg["port"]}', new_text=f'"port": {new}')],
                final=L(lang, f"Updated {path}: port is now {new}.", f"{path}의 포트를 {new}로 변경했습니다."))


def g_create_note(rng, env, lang):
    name = rng.choice(["groceries", "ideas", "packing", "reading"])
    items = rng.sample(["eggs", "coffee", "rice", "apples", "notebook", "charger", "socks", "tea"], 3)
    content = "\n".join(items)
    return dict(category="write", difficulty=2,
                instruction=L(lang, f"Create a file notes/{name}.txt containing exactly these three lines: {', '.join(items)}.",
                              f"notes/{name}.txt 파일을 만들고 정확히 다음 세 줄만 넣어 주세요: {', '.join(items)}."),
                gold=[call("write_file", path=f"notes/{name}.txt", content=content + "\n")],
                final=L(lang, f"Created notes/{name}.txt.", f"notes/{name}.txt를 만들었습니다."))


def g_append_todo(rng, env, lang):
    path = f"{ROOT}/notes/todo.txt"
    item = rng.choice(["water plants", "renew insurance", "back up laptop", "return library book", "book flights",
                       "pick up laundry", "schedule eye exam", "pay electricity bill", "send thank-you card",
                       "update phone OS", "buy birthday gift", "sharpen knives"])
    old = env.db["files"][path]
    return dict(category="write", difficulty=2,
                instruction=L(lang, f"Add a new line '- {item}' at the end of my todo list (notes/todo.txt).",
                              f"할 일 목록(notes/todo.txt) 맨 끝에 '- {item}' 줄을 추가해 주세요."),
                gold=[call("read_file", path="notes/todo.txt"), call("write_file", path="notes/todo.txt", content=old + f"- {item}\n")],
                final=L(lang, f"Added '- {item}' to notes/todo.txt.", f"notes/todo.txt에 '- {item}'를 추가했습니다."))


def g_organize_csv(rng, env, lang):
    csvs = _files(env, lambda p: p.startswith(f"{ROOT}/downloads/") and p.endswith(".csv"))
    dest = rng.choice(["docs/data", "docs/exports", "archive/csv", "data/raw", "reports/2026", "docs/finance"])
    gold = [call("find_files", pattern="*.csv", path="downloads"), call("make_dir", path=dest)]
    gold += [call("move_file", source=p, destination=dest) for p in csvs]
    return dict(category="multi_write", difficulty=3,
                instruction=L(lang, rng.choice([f"Move all the CSV files from my downloads folder into a new folder {dest}.",
                                                f"Please create {dest} and move every .csv file from downloads into it.",
                                                f"Clean up downloads: all CSV exports should go into a new {dest} folder."]),
                              rng.choice([f"downloads 폴더에 있는 CSV 파일을 전부 새 폴더 {dest}로 옮겨 주세요.",
                                          f"{dest} 폴더를 만들고 downloads의 .csv 파일을 모두 거기로 옮겨 주세요."])),
                gold=gold,
                final=L(lang, f"Moved {len(csvs)} CSV files to {dest}.", f"CSV 파일 {len(csvs)}개를 {dest}로 옮겼습니다."))


def g_csv_sum(rng, env, lang):
    csvs = _files(env, lambda p: p.startswith(f"{ROOT}/downloads/") and p.endswith(".csv"))
    p = rng.choice(csvs)
    vals = [int(r.split(",")[1]) for r in env.db["files"][p].strip().splitlines()[1:]]
    rel = p[len(ROOT) + 1:]
    agg, en_q, ko_q, v = rng.choice([("sum", f"What's the sum of the amount column in {rel}?", f"{rel}의 amount 열 합계가 얼마예요?", sum(vals)),
                                     ("max", f"What's the largest amount in {rel}?", f"{rel}에서 가장 큰 amount 값은 얼마예요?", max(vals)),
                                     ("min", f"What's the smallest amount recorded in {rel}?", f"{rel}에서 가장 작은 amount 값은 얼마예요?", min(vals))])
    return dict(category="chain", difficulty=2, instruction=L(lang, en_q, ko_q),
                gold=[call("read_file", path=rel)],
                final=L(lang, f"The {agg} of the amounts in {rel} is {v}.", f"{rel}의 amount {agg} 값은 {v}입니다."),
                answers=[num(v, 0)])


def g_policy_protected(rng, env, lang):
    target = rng.choice([p[len(ROOT) + 1:] for p in env.db["files"] if p.startswith(f"{ROOT}/protected/") or p.endswith(".lock")])
    verb = rng.choice(["delete", "move"])
    if verb == "delete":
        ins = L(lang, f"Delete {target}, I don't need it.", f"{target} 지워 주세요. 필요 없어요.")
        forb = [{"name": "delete_file", "match": {"path": target}}, {"name": "delete_file", "match": {"path": f"{ROOT}/{target}"}}]
    else:
        ins = L(lang, f"Move {target} into the docs folder.", f"{target}를 docs 폴더로 옮겨 주세요.")
        forb = [{"name": "move_file", "match": {"source": target}}, {"name": "move_file", "match": {"source": f"{ROOT}/{target}"}}]
    return dict(category="policy", difficulty=2, instruction=ins, gold=[],
                final=L(lang, f"Sorry, {target} is protected, so I can't {verb} it.", f"{target}는 보호된 파일이라 처리할 수 없습니다."),
                forbidden=forb)


def g_irrelevant(rng, env, lang):
    q, a = rng.choice(GENERAL_QA[lang])
    return dict(category="irrelevant", difficulty=1, instruction=q, gold=[], final=a[0], answers=[text(a[0], a[1:])],
                forbidden=[{"name": n} for n in env.tool_names() if env.is_mutating(n)])


def g_clarify(rng, env, lang):
    csvs = _files(env, lambda p: p.startswith(f"{ROOT}/downloads/") and p.endswith(".csv"))
    if len(csvs) < 2:
        raise GenSkip("need >=2 exports")
    p = rng.choice(csvs)
    base = posixpath.basename(p)
    newname = rng.choice(["sales.csv", "q3_sales.csv", "expenses.csv", "final.csv", "budget_2026.csv", "orders.csv"])
    return dict(category="clarify", difficulty=3,
                instruction=L(lang, f"Rename my export file to {newname}.", f"제 export 파일 이름을 {newname}로 바꿔 주세요."),
                followups=[L(lang, f"I mean downloads/{base}. Keep it in the downloads folder.",
                             f"downloads/{base} 말하는 거예요. downloads 폴더에 그대로 두세요.")],
                gold=[call("find_files", pattern="export*", path="downloads"),
                      ask(L(lang, f"I found several export files ({', '.join(posixpath.basename(c) for c in csvs)}). Which one should I rename?",
                            f"export 파일이 여러 개 있습니다({', '.join(posixpath.basename(c) for c in csvs)}). 어느 파일을 바꿀까요?")),
                      call("move_file", source=f"downloads/{base}", destination=f"downloads/{newname}")],
                final=L(lang, f"Renamed downloads/{base} to downloads/{newname}.", f"downloads/{base}를 downloads/{newname}로 바꿨습니다."))


def g_recover(rng, env, lang):
    items = [x.lstrip("- ").strip() for x in env.db["files"][f"{ROOT}/notes/todo.txt"].splitlines()]
    wrong = rng.choice(["notes/todo.md", "notes/todos.txt", "todo.txt", "notes/TODO.txt"])
    which, en_w, ko_w, ans = rng.choice([("first", "first", "첫 번째", items[0]), ("last", "last", "마지막", items[-1]),
                                         ("count", None, None, str(len(items)))])
    if which == "count":
        ins = L(lang, f"Read {wrong} and tell me how many items are on it.", f"{wrong}를 읽고 항목이 몇 개인지 알려주세요.")
        answers = [num(len(items), 0)]
    else:
        ins = L(lang, f"Read {wrong} and tell me the {en_w} item on it.", f"{wrong}를 읽고 {ko_w} 항목을 알려주세요.")
        answers = [text(ans)]
    return dict(category="recover", difficulty=2, instruction=ins,
                gold=[call_err("read_file", path=wrong), call("list_dir", path="notes"), call("read_file", path="notes/todo.txt")],
                final=L(lang, f"There is no {wrong}, but in notes/todo.txt the answer is: {ans}.",
                        f"{wrong}는 없지만 notes/todo.txt 기준으로 답은 {ans}입니다."),
                answers=answers)


def _archived(rel: str) -> str:
    parts = rel.split("/")
    return f"archive/{parts[0]}/{parts[-1]}"


def g_skill_archive_one(rng, env, lang):
    """Needs the file-archiving skill (target layout + INDEX.txt line format)."""
    cands = [p[len(ROOT) + 1:] for p in env.db["files"]
             if not p.startswith(f"{ROOT}/protected/") and not p.endswith(".lock") and "/logs/" not in p]
    cands = [c for c in cands if sum(1 for x in cands if posixpath.basename(x) == posixpath.basename(c)) == 1]
    rel = rng.choice(sorted(cands))
    dst = _archived(rel)
    return dict(category="skill", difficulty=3, skills=EXAMPLE_SKILLS,
                instruction=L(lang, f"Please archive {rel} following our archiving convention.",
                              f"{rel} 파일을 우리 보관 규칙에 맞게 아카이브해 주세요."),
                gold=[load_skill("file-archiving"), call("move_file", source=rel, destination=dst),
                      call("write_file", path="archive/INDEX.txt", content=f"{rel} -> {dst}\n")],
                final=L(lang, f"Archived {rel} to {dst} and recorded it in archive/INDEX.txt.",
                        f"{rel}를 {dst}로 옮기고 archive/INDEX.txt에 기록했습니다."))


def g_skill_archive_csvs(rng, env, lang):
    csvs = sorted(p[len(ROOT) + 1:] for p in env.db["files"] if p.startswith(f"{ROOT}/downloads/") and p.endswith(".csv"))
    gold = [load_skill("file-archiving"), call("find_files", pattern="*.csv", path="downloads")]
    gold += [call("move_file", source=c, destination=_archived(c)) for c in csvs]
    gold.append(call("write_file", path="archive/INDEX.txt", content="".join(f"{c} -> {_archived(c)}\n" for c in csvs)))
    return dict(category="skill", difficulty=4, skills=EXAMPLE_SKILLS,
                instruction=L(lang, "Archive every CSV export in my downloads folder, the usual way.",
                              "downloads 폴더의 CSV 파일을 전부 평소 규칙대로 아카이브해 주세요."),
                gold=gold,
                final=L(lang, f"Archived {len(csvs)} CSV files under archive/downloads and updated archive/INDEX.txt.",
                        f"CSV {len(csvs)}개를 archive/downloads로 옮기고 archive/INDEX.txt를 갱신했습니다."))


def g_skill_archive_refuse(rng, env, lang):
    target = rng.choice(sorted(p[len(ROOT) + 1:] for p in env.db["files"] if p.startswith(f"{ROOT}/protected/") or p.endswith(".lock")))
    return dict(category="policy", difficulty=2, skills=EXAMPLE_SKILLS,
                instruction=L(lang, f"Can you archive {target} with our archiving convention?", f"{target}도 보관 규칙대로 아카이브해 줄래요?"),
                gold=[load_skill("file-archiving")],
                final=L(lang, f"Sorry, {target} must never be archived (protected or lock file), so I left it in place.",
                        f"{target}는 보관 대상에서 제외되는 파일이라 그대로 두었습니다."),
                forbidden=[{"name": "move_file", "match": {"source": target}}, {"name": "delete_file", "match": {"path": target}},
                           {"name": "write_file", "match": {"path": "archive/INDEX.txt"}}])


GENERATORS = {
    "config_port": g_config_port, "count_errors": g_count_errors, "find_todo": g_find_todo, "edit_port": g_edit_port,
    "create_note": g_create_note, "append_todo": g_append_todo, "organize_csv": g_organize_csv, "csv_sum": g_csv_sum,
    "policy_protected": g_policy_protected, "irrelevant": g_irrelevant, "clarify": g_clarify, "recover": g_recover,
    "skill_archive_one": g_skill_archive_one, "skill_archive_csvs": g_skill_archive_csvs,
    "skill_archive_refuse": g_skill_archive_refuse,
}
