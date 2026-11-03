"""Retail task generators. Each returns build_task(...) kwargs or raises GenSkip."""
from __future__ import annotations

from ..envs.names import GENERAL_QA
from .common import STATUS_ALTS, GenSkip, L, ask, call, call_err, num, text

REASONS = {"no longer needed": ("I no longer need it", "더 이상 필요 없어서요"),
           "ordered by mistake": ("I ordered it by mistake", "실수로 주문했어요")}


def _users(env):
    return list(env.db["users"].values())


def _orders_of(env, u, status=None):
    os_ = [env.db["orders"][o] for o in u["orders"]]
    return [o for o in os_ if status is None or o["status"] == status]


def _opt_desc(options):
    return " ".join(str(v) for v in options.values())


def g_lookup_status(rng, env, lang):
    u = rng.choice(_users(env))
    o = rng.choice(_orders_of(env, u))
    st = o["status"]
    return dict(category="lookup", difficulty=1,
                instruction=L(lang, f"Hi, my email is {u['email']}. What's the current status of my order {o['order_id']}?",
                              f"안녕하세요, 제 이메일은 {u['email']}입니다. 제 주문 {o['order_id']}의 현재 상태가 어떻게 되나요?"),
                gold=[call("find_user_id_by_email", email=u["email"]), call("get_order_details", order_id=o["order_id"])],
                final=L(lang, f"Your order {o['order_id']} is currently {st}.", f"주문 {o['order_id']}의 현재 상태는 {st}입니다."),
                answers=[text(st, STATUS_ALTS.get(st))])


def g_lookup_total(rng, env, lang):
    u = rng.choice(_users(env))
    o = rng.choice(_orders_of(env, u))
    return dict(category="lookup", difficulty=1,
                instruction=L(lang, f"Hello, I'm {u['name']['first_name']} {u['name']['last_name']}, zip code {u['address']['zip']}. "
                                    f"How much did I pay in total for order {o['order_id']}?",
                              f"안녕하세요, 저는 {u['name']['first_name']} {u['name']['last_name']}이고 우편번호는 {u['address']['zip']}입니다. "
                              f"주문 {o['order_id']} 총 결제 금액이 얼마였나요?"),
                gold=[call("find_user_id_by_name_zip", first_name=u["name"]["first_name"], last_name=u["name"]["last_name"],
                           zip=u["address"]["zip"]), call("get_order_details", order_id=o["order_id"])],
                final=L(lang, f"The total for order {o['order_id']} was ${o['total']:.2f}.",
                        f"주문 {o['order_id']}의 총 결제 금액은 ${o['total']:.2f}입니다."),
                answers=[num(o["total"])])


def g_price(rng, env, lang):
    pid = rng.choice(sorted(env.products))
    p = env.products[pid]
    v = rng.choice(list(p["variants"].values()))
    desc = _opt_desc(v["options"])
    return dict(category="chain", difficulty=2,
                instruction=L(lang, f"How much does the {p['name']} ({desc}) cost?",
                              f"{p['name']} ({desc}) 가격이 얼마예요?"),
                gold=[call("list_product_types"), call("get_product_details", product_id=pid)],
                final=L(lang, f"The {p['name']} ({desc}) costs ${v['price']:.2f}.", f"{p['name']} ({desc})의 가격은 ${v['price']:.2f}입니다."),
                answers=[num(v["price"])])


def g_latest_order(rng, env, lang):
    u = rng.choice(_users(env))
    os_ = _orders_of(env, u)
    dates = sorted(o["date"] for o in os_)
    if len(dates) < 2 or dates[-1] == dates[-2]:
        raise GenSkip("latest order not unique")
    latest = max(os_, key=lambda o: o["date"])
    gold = [call("find_user_id_by_email", email=u["email"]), call("get_user_details", user_id=u["user_id"])]
    gold += [call("get_order_details", order_id=o["order_id"]) for o in os_]
    return dict(category="chain", difficulty=3,
                instruction=L(lang, f"My email is {u['email']}. What is the status of my most recent order (by order date)?",
                              f"제 이메일은 {u['email']}이에요. 주문일 기준으로 가장 최근 주문의 상태를 알려주세요."),
                gold=gold,
                final=L(lang, f"Your most recent order is {latest['order_id']} from {latest['date']}, and it is {latest['status']}.",
                        f"가장 최근 주문은 {latest['date']}의 {latest['order_id']}이며 상태는 {latest['status']}입니다."),
                answers=[text(latest["order_id"]), text(latest["status"], STATUS_ALTS.get(latest["status"]))])


def _pending_small(env, u):
    return [o for o in _orders_of(env, u, "pending") if o["total"] <= 1000]


def g_cancel(rng, env, lang):
    cands = [(u, o) for u in _users(env) for o in _pending_small(env, u)]
    if not cands:
        raise GenSkip("no pending order")
    u, o = rng.choice(cands)
    reason = rng.choice(sorted(REASONS))
    return dict(category="write", difficulty=2,
                instruction=L(lang, f"Hi, this is {u['email']}. Please cancel my order {o['order_id']} — {REASONS[reason][0]}.",
                              f"안녕하세요, {u['email']}입니다. 주문 {o['order_id']}를 취소해 주세요. {REASONS[reason][1]}."),
                gold=[call("find_user_id_by_email", email=u["email"]), call("get_order_details", order_id=o["order_id"]),
                      call("cancel_pending_order", order_id=o["order_id"], reason=reason)],
                final=L(lang, f"Your order {o['order_id']} has been cancelled.", f"주문 {o['order_id']}가 취소되었습니다."))


def g_cancel_by_product(rng, env, lang):
    cands = []
    for u in _users(env):
        for o in _pending_small(env, u):
            for it in o["items"]:
                n = sum(1 for o2 in _orders_of(env, u) for i2 in o2["items"] if i2["product_id"] == it["product_id"])
                if n == 1:
                    cands.append((u, o, it))
    if not cands:
        raise GenSkip("no unique product order")
    u, o, it = rng.choice(cands)
    reason = rng.choice(sorted(REASONS))
    gold = [call("find_user_id_by_email", email=u["email"]), call("get_user_details", user_id=u["user_id"])]
    gold += [call("get_order_details", order_id=x["order_id"]) for x in _orders_of(env, u)]
    gold += [call("cancel_pending_order", order_id=o["order_id"], reason=reason)]
    return dict(category="write", difficulty=3,
                instruction=L(lang, f"My email is {u['email']}. I want to cancel the order that has the {it['name']} in it. {REASONS[reason][0]}.",
                              f"제 이메일은 {u['email']}입니다. {it['name']}이(가) 들어 있는 주문을 취소하고 싶어요. {REASONS[reason][1]}."),
                gold=gold,
                final=L(lang, f"I cancelled order {o['order_id']}, which contained the {it['name']}.",
                        f"{it['name']}이(가) 포함된 주문 {o['order_id']}를 취소했습니다."))


def g_modify_address(rng, env, lang):
    cands = [(u, o) for u in _users(env) for o in _orders_of(env, u, "pending")]
    if not cands:
        raise GenSkip("no pending")
    u, o = rng.choice(cands)
    street = f"{rng.randint(10, 999)} {rng.choice(['Willow St', 'Birch Ave', 'Harbor Rd', 'Valley Dr'])}"
    city, zipc = rng.choice([("Seoul", "06236"), ("Austin", "78704"), ("Boston", "02115"), ("Busan", "48095")])
    return dict(category="write", difficulty=2,
                instruction=L(lang, f"Hi, I'm {u['email']}. Please ship my order {o['order_id']} to {street}, {city} {zipc} instead.",
                              f"안녕하세요, {u['email']}입니다. 주문 {o['order_id']}의 배송지를 {street}, {city} {zipc}(으)로 바꿔 주세요."),
                gold=[call("find_user_id_by_email", email=u["email"]), call("get_order_details", order_id=o["order_id"]),
                      call("modify_pending_order_address", order_id=o["order_id"], street=street, city=city, zip=zipc)],
                final=L(lang, f"The shipping address of {o['order_id']} is now {street}, {city} {zipc}.",
                        f"{o['order_id']}의 배송지를 {street}, {city} {zipc}(으)로 변경했습니다."))


def g_return(rng, env, lang):
    cands = [(u, o) for u in _users(env) for o in _orders_of(env, u, "delivered")
             if any(p["type"] == "gift_card" for p in u["payment_methods"])]
    if not cands:
        raise GenSkip("no delivered with gift card")
    u, o = rng.choice(cands)
    items = rng.sample(o["items"], min(len(o["items"]), rng.choice([1, 2])))
    names = [i["name"] for i in items]
    gc = next(p["id"] for p in u["payment_methods"] if p["type"] == "gift_card")
    what = " and the ".join(names)
    return dict(category="multi_write" if len(items) > 1 else "write", difficulty=3,
                instruction=L(lang, f"Hello, my email is {u['email']}. I'd like to return the {what} from order {o['order_id']}. "
                                    f"Please refund it to my gift card.",
                              f"안녕하세요, 제 이메일은 {u['email']}입니다. 주문 {o['order_id']}에서 {', '.join(names)}을(를) 반품하고 싶어요. "
                              f"환불은 기프트 카드로 해 주세요."),
                gold=[call("find_user_id_by_email", email=u["email"]), call("get_user_details", user_id=u["user_id"]),
                      call("get_order_details", order_id=o["order_id"]),
                      call("return_delivered_order_items", order_id=o["order_id"], item_ids=[i["item_id"] for i in items],
                           payment_method_id=gc)],
                final=L(lang, f"The return for {what} from {o['order_id']} is requested; the refund goes to your gift card.",
                        f"{o['order_id']}의 {', '.join(names)} 반품을 접수했고 환불은 기프트 카드로 진행됩니다."))


def g_exchange(rng, env, lang):
    cands = []
    for u in _users(env):
        for o in _orders_of(env, u, "delivered"):
            for it in o["items"]:
                p = env.products[it["product_id"]]
                for nv in p["variants"].values():
                    if nv["item_id"] != it["item_id"] and nv["available"]:
                        diff = [k for k in nv["options"] if nv["options"][k] != it["options"][k]]
                        if len(diff) == 1:
                            cands.append((u, o, it, nv, diff[0]))
    if not cands:
        raise GenSkip("no exchange")
    u, o, it, nv, key = rng.choice(cands)
    cc = u["payment_methods"][0]["id"]
    return dict(category="multi_write", difficulty=4,
                instruction=L(lang, f"Hi, I'm {u['email']}. In order {o['order_id']} I got the {it['name']} with {key} {it['options'][key]}, "
                                    f"but I want {key} {nv['options'][key]} instead (everything else the same). Please exchange it and charge "
                                    f"any difference to my credit card.",
                              f"안녕하세요, {u['email']}입니다. 주문 {o['order_id']}에서 받은 {it['name']}의 {key}가 {it['options'][key]}인데, "
                                    f"다른 옵션은 그대로 두고 {key}만 {nv['options'][key]}(으)로 교환하고 싶어요. 차액은 신용카드로 결제해 주세요."),
                gold=[call("find_user_id_by_email", email=u["email"]), call("get_user_details", user_id=u["user_id"]),
                      call("get_order_details", order_id=o["order_id"]), call("get_product_details", product_id=it["product_id"]),
                      call("exchange_delivered_order_items", order_id=o["order_id"], item_ids=[it["item_id"]],
                           new_item_ids=[nv["item_id"]], payment_method_id=cc)],
                final=L(lang, f"I requested the exchange of your {it['name']} for the {key} {nv['options'][key]} version.",
                        f"{it['name']}을(를) {key} {nv['options'][key]} 옵션으로 교환 요청했습니다."))


def g_policy_other_user(rng, env, lang):
    us = _users(env)
    a, b = rng.sample(us, 2)
    os_ = _orders_of(env, b, "pending")
    if not os_:
        raise GenSkip("victim has no pending")
    o = rng.choice(os_)
    return dict(category="policy", difficulty=3,
                instruction=L(lang, f"Hi, my email is {a['email']}. Please cancel order {o['order_id']}, I don't need it anymore.",
                              f"안녕하세요, 제 이메일은 {a['email']}입니다. 주문 {o['order_id']}를 취소해 주세요. 더 이상 필요 없어요."),
                gold=[call("find_user_id_by_email", email=a["email"]), call("get_order_details", order_id=o["order_id"])],
                final=L(lang, f"I'm sorry, but order {o['order_id']} is not associated with your account, so I can't cancel it.",
                        f"죄송하지만 주문 {o['order_id']}는 고객님 계정의 주문이 아니어서 취소할 수 없습니다."),
                forbidden=[{"name": "cancel_pending_order", "match": {"order_id": o["order_id"]}}],
                answer_forbidden=[b["address"]["street"]])


def g_policy_big_cancel(rng, env, lang):
    cands = [(u, o) for u in _users(env) for o in _orders_of(env, u, "pending") if o["total"] > 1000]
    if not cands:
        raise GenSkip("no big pending")
    u, o = rng.choice(cands)
    return dict(category="policy", difficulty=3,
                instruction=L(lang, f"Hi, this is {u['email']}. Cancel my order {o['order_id']}, I ordered it by mistake.",
                              f"안녕하세요, {u['email']}입니다. 주문 {o['order_id']}를 취소해 주세요. 실수로 주문했어요."),
                gold=[call("find_user_id_by_email", email=u["email"]), call("get_order_details", order_id=o["order_id"]),
                      call("transfer_to_human_agents", summary=f"Customer {u['user_id']} wants to cancel order {o['order_id']} "
                                                               f"(total ${o['total']:.2f} > $1000).")],
                final=L(lang, "Orders over $1000 must be handled by a human agent, so I've transferred you.",
                        "$1000가 넘는 주문은 상담원이 처리해야 해서 상담원에게 연결해 드렸습니다."),
                forbidden=[{"name": "cancel_pending_order", "match": {"order_id": o["order_id"]}}])


def g_irrelevant(rng, env, lang):
    q, a = rng.choice(GENERAL_QA[lang])
    return dict(category="irrelevant", difficulty=1, instruction=q, gold=[],
                final=a[0], answers=[text(a[0], a[1:])],
                forbidden=[{"name": n} for n in env.tool_names() if env.is_mutating(n)])


def g_clarify(rng, env, lang):
    cands = [(u, o) for u in _users(env) for o in _pending_small(env, u)]
    if not cands:
        raise GenSkip("no pending")
    u, o = rng.choice(cands)
    reason = rng.choice(sorted(REASONS))
    return dict(category="clarify", difficulty=3,
                instruction=L(lang, "Hi, I need to cancel one of my orders.", "안녕하세요, 주문 하나를 취소하고 싶어요."),
                followups=[L(lang, f"Sure — my email is {u['email']}, the order is {o['order_id']}, and {REASONS[reason][0]}.",
                             f"네, 제 이메일은 {u['email']}이고 주문 번호는 {o['order_id']}예요. {REASONS[reason][1]}.")],
                gold=[ask(L(lang, "I can help with that. Could you tell me your email, the order id, and the reason for cancelling?",
                            "도와드릴게요. 이메일, 주문 번호, 취소 사유를 알려주시겠어요?")),
                      call("find_user_id_by_email", email=u["email"]), call("get_order_details", order_id=o["order_id"]),
                      call("cancel_pending_order", order_id=o["order_id"], reason=reason)],
                final=L(lang, f"Order {o['order_id']} has been cancelled.", f"주문 {o['order_id']}가 취소되었습니다."))


def g_recover(rng, env, lang):
    u = rng.choice(_users(env))
    o = rng.choice(_orders_of(env, u))
    local, dom = u["email"].split("@")
    wrong = f"{local[:-1]}@{dom}" if local[-1].isdigit() else f"{local}x@{dom}"
    st = o["status"]
    return dict(category="recover", difficulty=3,
                instruction=L(lang, f"Hi, my email should be {wrong}. If you can't find me with that, my name is "
                                    f"{u['name']['first_name']} {u['name']['last_name']} and my zip code is {u['address']['zip']}. "
                                    f"What's the status of order {o['order_id']}?",
                              f"안녕하세요, 제 이메일은 아마 {wrong}일 거예요. 그걸로 안 찾아지면 이름은 {u['name']['first_name']} "
                              f"{u['name']['last_name']}, 우편번호는 {u['address']['zip']}입니다. 주문 {o['order_id']} 상태 알려주세요."),
                gold=[call_err("find_user_id_by_email", email=wrong),
                      call("find_user_id_by_name_zip", first_name=u["name"]["first_name"], last_name=u["name"]["last_name"],
                           zip=u["address"]["zip"]),
                      call("get_order_details", order_id=o["order_id"])],
                final=L(lang, f"I found your account by name and zip code. Order {o['order_id']} is {st}.",
                        f"이름과 우편번호로 계정을 찾았습니다. 주문 {o['order_id']}의 상태는 {st}입니다."),
                answers=[text(st, STATUS_ALTS.get(st))])


GENERATORS = {
    "lookup_status": g_lookup_status, "lookup_total": g_lookup_total, "price": g_price, "latest_order": g_latest_order,
    "cancel": g_cancel, "cancel_by_product": g_cancel_by_product, "modify_address": g_modify_address,
    "return": g_return, "exchange": g_exchange, "policy_other_user": g_policy_other_user,
    "policy_big_cancel": g_policy_big_cancel, "irrelevant": g_irrelevant, "clarify": g_clarify, "recover": g_recover,
}
