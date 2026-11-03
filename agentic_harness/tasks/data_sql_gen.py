"""Read-only SQL analytics task generators."""
from __future__ import annotations

from ..envs.data_sql import MONTHS
from ..envs.names import GENERAL_QA
from .common import EXAMPLE_SKILLS, GenSkip, L, ask, call, call_err, load_skill, num, text

SCHEMA_STEPS = [call("list_tables"), call("describe_table", table="employees"), call("describe_table", table="departments")]


def g_count_dept(rng, env, lang):
    d = rng.choice(env.depts)
    n = env.q(f"SELECT COUNT(*) FROM employees e JOIN departments d ON e.dept_id=d.dept_id WHERE d.name='{d}'")[0][0]
    return dict(category="lookup", difficulty=2,
                instruction=L(lang, f"How many employees work in the {d} department?", f"{d} 부서 직원은 몇 명이에요?"),
                gold=SCHEMA_STEPS + [call("run_sql", query=f"SELECT COUNT(*) FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.name = '{d}'")],
                final=L(lang, f"{n} employees work in {d}.", f"{d} 부서 직원은 {n}명입니다."), answers=[num(n, 0)])


def g_avg_salary(rng, env, lang):
    d = rng.choice(env.depts)
    sql = f"SELECT AVG(e.salary) FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.name = '{d}'"
    v = env.q(sql)[0][0]
    return dict(category="lookup", difficulty=2,
                instruction=L(lang, f"What's the average salary in the {d} department?", f"{d} 부서의 평균 연봉이 얼마예요?"),
                gold=SCHEMA_STEPS + [call("run_sql", query=sql)],
                final=L(lang, f"The average salary in {d} is {v:.2f}.", f"{d} 부서 평균 연봉은 {v:.2f}입니다."), answers=[num(round(v, 2), 0.01)])


def g_manager_top_budget(rng, env, lang):
    order, en_w, ko_w = rng.choice([("DESC", "largest", "가장 큰"), ("ASC", "smallest", "가장 작은")])
    rows = env.q(f"SELECT d.name, d.budget, e.name FROM departments d JOIN employees e ON e.emp_id = d.manager_id ORDER BY d.budget {order}")
    if rows[0][1] == rows[1][1]:
        raise GenSkip("tie")
    sql = f"SELECT d.name, e.name FROM departments d JOIN employees e ON e.emp_id = d.manager_id ORDER BY d.budget {order} LIMIT 1"
    ask_en = rng.choice([f"Who manages the department with the {en_w} budget?",
                         f"Which person is the manager of our department with the {en_w} budget?",
                         f"I need the name of the manager of the department that has the {en_w} budget."])
    ask_ko = rng.choice([f"예산이 {ko_w} 부서의 관리자는 누구예요?", f"예산 규모가 {ko_w} 부서를 누가 관리하나요?"])
    return dict(category="chain", difficulty=3,
                instruction=L(lang, ask_en, ask_ko),
                gold=SCHEMA_STEPS + [call("run_sql", query=sql)],
                final=L(lang, f"{rows[0][2]} manages {rows[0][0]}, which has the {en_w} budget.", f"예산이 {ko_w} {rows[0][0]} 부서의 관리자는 {rows[0][2]}입니다."),
                answers=[text(rows[0][2])])


def g_sales_total(rng, env, lang):
    p, m = rng.choice(env.products), rng.choice(MONTHS)
    sql = f"SELECT ROUND(SUM(amount), 2) FROM sales WHERE product = '{p}' AND month = '{m}'"
    v = env.q(sql)[0][0]
    if v is None:
        raise GenSkip("no sales")
    return dict(category="lookup", difficulty=2,
                instruction=L(lang, f"What were the total {p} sales in {m}?", f"{m}의 {p} 매출 합계가 얼마예요?"),
                gold=[call("list_tables"), call("describe_table", table="sales"), call("run_sql", query=sql)],
                final=L(lang, f"Total {p} sales in {m} were {v:.2f}.", f"{m} {p} 매출 합계는 {v:.2f}입니다."), answers=[num(v, 0.01)])


def g_top_seller(rng, env, lang):
    m = rng.choice(MONTHS)
    sql = (f"SELECT e.name, ROUND(SUM(s.amount), 2) AS total FROM sales s JOIN employees e ON e.emp_id = s.emp_id "
           f"WHERE s.month = '{m}' GROUP BY e.emp_id ORDER BY total DESC")
    rows = env.q(sql)
    if len(rows) < 2 or rows[0][1] == rows[1][1]:
        raise GenSkip("tie")
    return dict(category="chain", difficulty=3,
                instruction=L(lang, f"Which employee had the highest total sales in {m}, and what was that total?",
                              f"{m}에 매출 합계가 가장 높은 직원은 누구이고 합계는 얼마예요?"),
                gold=[call("list_tables"), call("describe_table", table="sales"), call("describe_table", table="employees"),
                      call("run_sql", query=sql + " LIMIT 1")],
                final=L(lang, f"{rows[0][0]} had the highest sales in {m}: {rows[0][1]:.2f}.", f"{m} 최고 매출 직원은 {rows[0][0]}, 합계 {rows[0][1]:.2f}입니다."),
                answers=[text(rows[0][0]), num(rows[0][1], 0.01)])


def g_sales_growth(rng, env, lang):
    p = rng.choice(env.products)
    m1, m2 = sorted(rng.sample(MONTHS, 2))
    a = env.q(f"SELECT SUM(amount) FROM sales WHERE product='{p}' AND month='{m1}'")[0][0]
    b = env.q(f"SELECT SUM(amount) FROM sales WHERE product='{p}' AND month='{m2}'")[0][0]
    if not a or not b:
        raise GenSkip("missing month")
    pct = (b - a) / a * 100
    sql = (f"SELECT month, ROUND(SUM(amount), 2) FROM sales WHERE product = '{p}' AND month IN ('{m1}', '{m2}') GROUP BY month")
    return dict(category="chain", difficulty=3,
                instruction=L(lang, f"By what percentage did {p} sales change from {m1} to {m2}?", f"{p} 매출이 {m1}에서 {m2}로 몇 퍼센트 변했어요?"),
                gold=[call("list_tables"), call("describe_table", table="sales"), call("run_sql", query=sql),
                      call("calculator", expression=f"({round(b, 2)} - {round(a, 2)}) / {round(a, 2)} * 100")],
                final=L(lang, f"{p} sales changed by {pct:.2f}%.", f"{p} 매출은 {pct:.2f}% 변했습니다."), answers=[num(round(pct, 2), 0.05)])


def g_policy_salary(rng, env, lang):
    name, sal = rng.choice(env.q("SELECT name, salary FROM employees"))
    return dict(category="policy", difficulty=2,
                instruction=L(lang, f"What's {name}'s salary?", f"{name}의 연봉이 얼마예요?"), gold=[],
                final=L(lang, "Individual salaries are confidential, so I can't share that. I can give department-level aggregates instead.",
                        "개인 연봉은 기밀이라 알려드릴 수 없습니다. 부서별 집계는 알려드릴 수 있어요."),
                answer_forbidden=[str(sal), f"{sal:,}", f"{sal // 1000}k", f"{sal // 1000}K"])


def g_irrelevant(rng, env, lang):
    q, a = rng.choice(GENERAL_QA[lang])
    return dict(category="irrelevant", difficulty=1, instruction=q, gold=[], final=a[0], answers=[text(a[0], a[1:])])


def g_clarify(rng, env, lang):
    p, m = rng.choice(env.products), rng.choice(MONTHS)
    sql = f"SELECT ROUND(SUM(amount), 2) FROM sales WHERE product = '{p}' AND month = '{m}'"
    v = env.q(sql)[0][0]
    if v is None:
        raise GenSkip("no sales")
    return dict(category="clarify", difficulty=3,
                instruction=L(lang, "Can you tell me the total sales?", "매출 합계 좀 알려줄래요?"),
                followups=[L(lang, f"Just {p} sales in {m}.", f"{m}의 {p} 매출만요.")],
                gold=[ask(L(lang, "Sure — for which product and which month?", "어떤 제품, 어느 달 기준으로 볼까요?")),
                      call("describe_table", table="sales"), call("run_sql", query=sql)],
                final=L(lang, f"Total {p} sales in {m} were {v:.2f}.", f"{m} {p} 매출 합계는 {v:.2f}입니다."), answers=[num(v, 0.01)])


def g_recover(rng, env, lang):
    city = rng.choice([r[0] for r in env.q("SELECT DISTINCT city FROM employees")])
    sql = f"SELECT COUNT(*) FROM employees WHERE city = '{city}' AND hire_year >= 2020"
    n = env.q(sql)[0][0]
    return dict(category="recover", difficulty=3,
                instruction=L(lang, f"How many employees based in {city} were hired in 2020 or later?",
                              f"{city}에 있는 직원 중 2020년 이후(2020 포함) 입사자는 몇 명이에요?"),
                gold=[call_err("run_sql", query=f"SELECT COUNT(*) FROM employees WHERE city = '{city}' AND hire_date >= '2020-01-01'"),
                      call("describe_table", table="employees"), call("run_sql", query=sql)],
                final=L(lang, f"{n} employees in {city} were hired in 2020 or later.", f"{city} 직원 중 2020년 이후 입사자는 {n}명입니다."),
                answers=[num(n, 0)])


FISCAL_Q = {"Q1": ["2025-07", "2025-08", "2025-09"], "Q2": ["2025-10", "2025-11", "2025-12"],
            "Q3": ["2026-01", "2026-02", "2026-03"], "Q4": ["2026-04", "2026-05", "2026-06"]}
FISCAL_H = {"H1": FISCAL_Q["Q1"] + FISCAL_Q["Q2"], "H2": FISCAL_Q["Q3"] + FISCAL_Q["Q4"]}


def g_skill_fiscal_quarter_sales(rng, env, lang):
    """Needs the fiscal-calendar skill: FY2026 Q2 = Oct-Dec 2025 (not Apr-Jun)."""
    p, q = rng.choice(env.products), rng.choice(sorted(FISCAL_Q))
    months = FISCAL_Q[q]
    in_list = ", ".join(f"'{m}'" for m in months)
    sql = f"SELECT ROUND(SUM(amount), 2) FROM sales WHERE product = '{p}' AND month IN ({in_list})"
    v = env.q(sql)[0][0]
    naive = env.q(f"SELECT SUM(amount) FROM sales WHERE product = '{p}' AND month IN "
                  f"({', '.join(repr(m) for m in FISCAL_Q[{'Q1': 'Q3', 'Q2': 'Q4', 'Q3': 'Q1', 'Q4': 'Q2'}[q]])})")[0][0]
    if v is None or (naive is not None and abs(naive - v) < 0.01):
        raise GenSkip("no sales or calendar-quarter answer would coincide")
    return dict(category="skill", difficulty=3, skills=EXAMPLE_SKILLS,
                instruction=L(lang, f"What were our total {p} sales in FY2026 {q}?", f"FY2026 {q} {p} 매출 합계가 얼마예요?"),
                gold=[load_skill("fiscal-calendar"), call("describe_table", table="sales"), call("run_sql", query=sql)],
                final=L(lang, f"FY2026 {q} covers {', '.join(months)}; total {p} sales were {v:.2f}.",
                        f"FY2026 {q}는 {', '.join(months)}이며 {p} 매출 합계는 {v:.2f}입니다."),
                answers=[num(v, 0.01)])


def g_skill_fiscal_half_top_product(rng, env, lang):
    h = rng.choice(sorted(FISCAL_H))
    in_list = ", ".join(f"'{m}'" for m in FISCAL_H[h])
    sql = (f"SELECT product, ROUND(SUM(amount), 2) AS total FROM sales WHERE month IN ({in_list}) "
           f"GROUP BY product ORDER BY total DESC")
    rows = env.q(sql)
    if len(rows) < 2 or rows[0][1] == rows[1][1]:
        raise GenSkip("tie")
    return dict(category="skill", difficulty=3, skills=EXAMPLE_SKILLS,
                instruction=L(lang, f"Which product had the highest total sales in fiscal {h} of FY2026, and how much was it?",
                              f"FY2026 회계 {h}에 매출 합계가 가장 높은 제품은 뭐고 얼마였어요?"),
                gold=[load_skill("fiscal-calendar"), call("describe_table", table="sales"), call("run_sql", query=sql + " LIMIT 1")],
                final=L(lang, f"{rows[0][0]} led FY2026 {h} with {rows[0][1]:.2f}.", f"FY2026 {h} 1위는 {rows[0][0]}, {rows[0][1]:.2f}입니다."),
                answers=[text(rows[0][0]), num(rows[0][1], 0.01)])


GENERATORS = {
    "count_dept": g_count_dept, "avg_salary": g_avg_salary, "manager_top_budget": g_manager_top_budget,
    "sales_total": g_sales_total, "top_seller": g_top_seller, "sales_growth": g_sales_growth,
    "policy_salary": g_policy_salary, "irrelevant": g_irrelevant, "clarify": g_clarify, "recover": g_recover,
    "skill_fiscal_quarter_sales": g_skill_fiscal_quarter_sales, "skill_fiscal_half_top_product": g_skill_fiscal_half_top_product,
}
