"""Travel REST API task generators (single http_request tool)."""
from __future__ import annotations

from ..envs.names import GENERAL_QA
from ..envs.travel_api import AIRPORTS, DATES
from .common import EXAMPLE_SKILLS, GenSkip, L, ask, call, call_err, load_skill, num, read_skill_file, text

TRAVELERS = ["Mina Park", "Jason Lee", "Sora Kim", "David Choi", "Emily Jung", "Hyejin Kang"]


def http(method, path, query=None, body=None):
    args = {"method": method, "path": path}
    if query:
        args["query"] = query
    if body:
        args["body"] = body
    return call("http_request", **args)


def _flights(env, o, d, date):
    return sorted([f for f in env.catalog["flights"].values() if f["origin"] == o and f["destination"] == d and f["date"] == date],
                  key=lambda f: f["depart"])


def _cheapest_route(rng, env, max_price=None):
    routes = {}
    for f in env.catalog["flights"].values():
        routes.setdefault((f["origin"], f["destination"], f["date"]), []).append(f)
    cands = []
    for (o, d, date), fl in routes.items():
        avail = [f for f in fl if f["seats_left"] > 0]
        if not avail:
            continue
        best = min(f["price_usd"] for f in avail)
        winners = [f for f in avail if f["price_usd"] == best]
        if len(winners) == 1 and (max_price is None or best <= max_price):
            cands.append((o, d, date, winners[0], any(f["seats_left"] == 0 and f["price_usd"] < best for f in fl)))
    if not cands:
        raise GenSkip("no route")
    # prefer routes where a cheaper sold-out flight exists (tests the sold-out rule)
    tricky = [c for c in cands if c[4]]
    return rng.choice(tricky if tricky and rng.random() < 0.5 else cands)


def g_weather(rng, env, lang):
    city, date = rng.choice(list(AIRPORTS)), rng.choice(DATES)
    w = env.catalog["weather"][(city, date)]
    return dict(category="lookup", difficulty=1,
                instruction=L(lang, f"What's the forecast high temperature in {city} on {date}?", f"{date}에 {city} 최고 기온 예보가 몇 도예요?"),
                gold=[http("GET", "/weather", {"city": city, "date": date})],
                final=L(lang, f"The high in {city} on {date} is {w['high_c']}°C ({w['condition']}).",
                        f"{date} {city}의 최고 기온은 {w['high_c']}°C({w['condition']})입니다."),
                answers=[num(w["high_c"], 0)])


def g_fx(rng, env, lang):
    b, q = rng.sample(["USD", "KRW", "JPY", "EUR", "SGD"], 2)
    rate = env.catalog["fx"][(b, q)]
    return dict(category="lookup", difficulty=1,
                instruction=L(lang, f"What's the current {b} to {q} exchange rate?", f"지금 {b} 대 {q} 환율이 얼마예요?"),
                gold=[http("GET", "/fx/rate", {"base": b, "quote": q})],
                final=L(lang, f"1 {b} = {rate} {q}.", f"1 {b} = {rate} {q}입니다."),
                answers=[num(rate, max(abs(rate) * 1e-3, 1e-4))])


def g_cheapest_flight(rng, env, lang):
    o, d, date, f, _ = _cheapest_route(rng, env)
    return dict(category="chain", difficulty=2,
                instruction=L(lang, f"What's the cheapest flight with seats available from {o} to {d} on {date}? Give me the flight id and price.",
                              f"{date}에 {o}에서 {d}로 가는 좌석 남은 항공편 중 가장 싼 건 뭐예요? 항공편 ID와 가격을 알려주세요."),
                gold=[http("GET", "/flights/search", {"origin": o, "destination": d, "date": date})],
                final=L(lang, f"The cheapest available flight is {f['flight_id']} at ${f['price_usd']}.",
                        f"가장 싼 항공편은 {f['flight_id']}이고 가격은 ${f['price_usd']}입니다."),
                answers=[text(f["flight_id"]), num(f["price_usd"], 0)])


def _city_of(code):
    return next(c for c, codes in AIRPORTS.items() if code in codes)


def g_book_flight_city(rng, env, lang):
    for _ in range(50):
        o, d, date, f, _x = _cheapest_route(rng, env, max_price=2000)
        oc, dc = _city_of(o), _city_of(d)
        if len(AIRPORTS[oc]) == 1 and len(AIRPORTS[dc]) == 1:
            break
    else:
        raise GenSkip("no single-airport route")
    who = rng.choice(TRAVELERS)
    return dict(category="write", difficulty=3,
                instruction=L(lang, f"Book the cheapest available flight from {oc} to {dc} on {date} for {who}.",
                              f"{date}에 {oc}에서 {dc}로 가는 가장 싼 항공편을 {who} 이름으로 예약해 주세요."),
                gold=[http("GET", "/airports", {"city": oc}), http("GET", "/airports", {"city": dc}),
                      http("GET", "/flights/search", {"origin": o, "destination": d, "date": date}),
                      http("POST", "/bookings", body={"type": "flight", "item_id": f["flight_id"], "traveler": who})],
                final=L(lang, f"Booked {f['flight_id']} (${f['price_usd']}) for {who}.", f"{who} 님 이름으로 {f['flight_id']}(${f['price_usd']})를 예약했습니다."))


def _best_hotel(env, city, budget):
    hs = [h for h in env.catalog["hotels"].values() if h["city"] == city and h["price_per_night_usd"] <= budget]
    if not hs:
        return None
    top = max(h["rating"] for h in hs)
    w = [h for h in hs if h["rating"] == top]
    return w[0] if len(w) == 1 else None


def g_book_hotel(rng, env, lang):
    city = rng.choice(list(AIRPORTS))
    budget = rng.choice([150, 200, 250, 300])
    h = _best_hotel(env, city, budget)
    if not h:
        raise GenSkip("no hotel")
    i = rng.randint(0, 2)
    ci, co = DATES[i], DATES[i + rng.randint(1, 2)]
    who = rng.choice(TRAVELERS)
    return dict(category="write", difficulty=3,
                instruction=L(lang, f"Book the highest-rated hotel in {city} that costs at most ${budget} per night, for {who}, "
                                    f"checking in {ci} and checking out {co}.",
                              f"{city}에서 1박 ${budget} 이하인 호텔 중 평점이 가장 높은 곳을 {who} 이름으로 예약해 주세요. "
                              f"체크인 {ci}, 체크아웃 {co}입니다."),
                gold=[http("GET", "/hotels/search", {"city": city, "checkin": ci, "checkout": co}),
                      http("POST", "/bookings", body={"type": "hotel", "item_id": h["hotel_id"], "traveler": who, "checkin": ci, "checkout": co})],
                final=L(lang, f"Booked {h['name']} (rating {h['rating']}, ${h['price_per_night_usd']}/night) for {who}.",
                        f"{who} 님 이름으로 {h['name']}(평점 {h['rating']}, 1박 ${h['price_per_night_usd']})을 예약했습니다."))


def g_trip(rng, env, lang):
    o, d, date, f, _ = _cheapest_route(rng, env, max_price=2000)
    city = _city_of(d)
    i = DATES.index(date)
    if i >= len(DATES) - 1:
        raise GenSkip("no checkout date")
    co = DATES[i + 1]
    budget = rng.choice([200, 300])
    h = _best_hotel(env, city, budget)
    if not h:
        raise GenSkip("no hotel")
    who = rng.choice(TRAVELERS)
    return dict(category="multi_write", difficulty=4,
                instruction=L(lang, f"Plan a trip for {who}: book the cheapest available flight from {o} to {d} on {date}, and the "
                                    f"highest-rated hotel in {city} under ${budget}/night for one night ({date} to {co}).",
                              f"{who} 님 여행 예약: {date} {o}→{d} 좌석 남은 가장 싼 항공편, 그리고 {city}에서 1박 ${budget} 이하 "
                              f"평점 최고 호텔을 {date}부터 {co}까지 1박 예약해 주세요."),
                gold=[http("GET", "/flights/search", {"origin": o, "destination": d, "date": date}),
                      http("POST", "/bookings", body={"type": "flight", "item_id": f["flight_id"], "traveler": who}),
                      http("GET", "/hotels/search", {"city": city, "checkin": date, "checkout": co}),
                      http("POST", "/bookings", body={"type": "hotel", "item_id": h["hotel_id"], "traveler": who, "checkin": date, "checkout": co})],
                final=L(lang, f"Booked flight {f['flight_id']} and {h['name']} for {who}.", f"{who} 님의 항공편 {f['flight_id']}와 {h['name']}을 예약했습니다."))


def g_hotel_in_krw(rng, env, lang):
    city = rng.choice(list(AIRPORTS))
    hs = [h for h in env.catalog["hotels"].values() if h["city"] == city]
    cheapest = min(h["price_per_night_usd"] for h in hs)
    if sum(h["price_per_night_usd"] == cheapest for h in hs) != 1:
        raise GenSkip("tie")
    h = next(h for h in hs if h["price_per_night_usd"] == cheapest)
    rate = env.catalog["fx"][("USD", "KRW")]
    krw = cheapest * rate
    return dict(category="chain", difficulty=3,
                instruction=L(lang, f"How much is one night at the cheapest hotel in {city} (check-in {DATES[0]}, check-out {DATES[1]}) in Korean won?",
                              f"{city}에서 가장 싼 호텔 1박({DATES[0]} 체크인, {DATES[1]} 체크아웃)은 원화로 얼마예요?"),
                gold=[http("GET", "/hotels/search", {"city": city, "checkin": DATES[0], "checkout": DATES[1]}),
                      http("GET", "/fx/rate", {"base": "USD", "quote": "KRW"})],
                final=L(lang, f"{h['name']} costs ${cheapest}/night, about {krw:,.0f} KRW.", f"{h['name']}은 1박 ${cheapest}, 약 {krw:,.0f}원입니다."),
                answers=[num(round(krw), max(krw * 0.01, 1))])


def g_policy_over_budget(rng, env, lang):
    hs = [h for h in env.catalog["hotels"].values() if h["price_per_night_usd"] * 4 > 2000]
    if not hs:
        raise GenSkip("no expensive hotel")
    h = rng.choice(hs)
    who = rng.choice(TRAVELERS)
    return dict(category="policy", difficulty=3,
                instruction=L(lang, f"Book {h['name']} ({h['hotel_id']}) for {who} from {DATES[0]} to {DATES[4]}.",
                              f"{h['name']}({h['hotel_id']})을 {who} 이름으로 {DATES[0]}부터 {DATES[4]}까지 예약해 주세요."),
                gold=[http("GET", "/hotels/search", {"city": h["city"], "checkin": DATES[0], "checkout": DATES[4]})],
                final=L(lang, f"That stay would cost ${h['price_per_night_usd'] * 4}, over the $2000 single-booking limit, so I can't book it.",
                        f"총 ${h['price_per_night_usd'] * 4}로 1건당 한도 $2000를 넘어서 예약할 수 없습니다."),
                forbidden=[{"name": "http_request", "match": {"method": "POST", "body.item_id": h["hotel_id"]}}])


def g_irrelevant(rng, env, lang):
    q, a = rng.choice(GENERAL_QA[lang])
    return dict(category="irrelevant", difficulty=1, instruction=q, gold=[], final=a[0], answers=[text(a[0], a[1:])],
                forbidden=[{"name": "http_request", "match": {"method": "POST"}}, {"name": "http_request", "match": {"method": "DELETE"}}])


def g_clarify(rng, env, lang):
    o, d, date, f, _ = _cheapest_route(rng, env, max_price=2000)
    who = rng.choice(TRAVELERS)
    return dict(category="clarify", difficulty=3,
                instruction=L(lang, f"Book me a flight from {o} to {d}. My name is {who}.", f"{o}에서 {d} 가는 항공편 예약해 주세요. 제 이름은 {who}예요."),
                followups=[L(lang, f"On {date}, please — the cheapest one that still has seats.", f"{date}요. 좌석 남은 것 중 가장 싼 걸로요.")],
                gold=[ask(L(lang, "Which date would you like to fly, and do you have a preference among flights?",
                            "어느 날짜로 가실 건가요? 선호하는 항공편 조건이 있나요?")),
                      http("GET", "/flights/search", {"origin": o, "destination": d, "date": date}),
                      http("POST", "/bookings", body={"type": "flight", "item_id": f["flight_id"], "traveler": who})],
                final=L(lang, f"Booked {f['flight_id']} on {date} for {who}.", f"{date} {f['flight_id']}를 예약했습니다."))


def g_recover(rng, env, lang):
    for _ in range(50):
        o, d, date, f, _x = _cheapest_route(rng, env)
        oc, dc = _city_of(o), _city_of(d)
        if len(AIRPORTS[oc]) == 1 and len(AIRPORTS[dc]) == 1:
            break
    else:
        raise GenSkip("no single-airport route")
    return dict(category="recover", difficulty=3,
                instruction=L(lang, f"Search flights from {oc} to {dc} on {date} and tell me the cheapest one that has seats.",
                              f"{date} {oc}→{dc} 항공편을 찾아서 좌석 남은 것 중 가장 싼 걸 알려주세요."),
                gold=[call_err("http_request", method="GET", path="/flights/search", query={"origin": oc, "destination": dc, "date": date}),
                      http("GET", "/airports", {"city": oc}), http("GET", "/airports", {"city": dc}),
                      http("GET", "/flights/search", {"origin": o, "destination": d, "date": date})],
                final=L(lang, f"The cheapest available flight is {f['flight_id']} at ${f['price_usd']}.",
                        f"가장 싼 항공편은 {f['flight_id']}(${f['price_usd']})입니다."),
                answers=[text(f["flight_id"]), num(f["price_usd"], 0)])


SKILL_SYSTEM = ("You are a travel assistant with access to the Travel API through the `http_request` tool. "
                "The API documentation is not included here. When done, reply with a short summary.")


def g_skill_book_cheapest(rng, env, lang):
    """API reference lives only in the travel-api skill (reference/endpoints.md)."""
    o, d, date, f, _ = _cheapest_route(rng, env, max_price=2000)
    who = rng.choice(TRAVELERS)
    return dict(category="skill", difficulty=4, skills=EXAMPLE_SKILLS, system_prompt=SKILL_SYSTEM,
                instruction=L(lang, f"Book the cheapest flight that still has seats from {o} to {d} on {date} for {who}.",
                              f"{date} {o}→{d} 좌석 남은 가장 싼 항공편을 {who} 이름으로 예약해 주세요."),
                gold=[load_skill("travel-api"), read_skill_file("travel-api", "reference/endpoints.md"),
                      http("GET", "/flights/search", {"origin": o, "destination": d, "date": date}),
                      http("POST", "/bookings", body={"type": "flight", "item_id": f["flight_id"], "traveler": who})],
                final=L(lang, f"Booked {f['flight_id']} (${f['price_usd']}) for {who}.", f"{who} 님 이름으로 {f['flight_id']}(${f['price_usd']})를 예약했습니다."))


def g_skill_weather(rng, env, lang):
    city, date = rng.choice(list(AIRPORTS)), rng.choice(DATES)
    w = env.catalog["weather"][(city, date)]
    return dict(category="skill", difficulty=2, skills=EXAMPLE_SKILLS, system_prompt=SKILL_SYSTEM,
                instruction=L(lang, f"What's the low temperature forecast for {city} on {date}?", f"{date} {city} 최저 기온 예보가 몇 도예요?"),
                gold=[load_skill("travel-api"), read_skill_file("travel-api", "reference/endpoints.md"),
                      http("GET", "/weather", {"city": city, "date": date})],
                final=L(lang, f"The low in {city} on {date} is {w['low_c']}°C.", f"{date} {city} 최저 기온은 {w['low_c']}°C입니다."),
                answers=[num(w["low_c"], 0)])


GENERATORS = {
    "weather": g_weather, "fx": g_fx, "cheapest_flight": g_cheapest_flight, "book_flight_city": g_book_flight_city,
    "book_hotel": g_book_hotel, "trip": g_trip, "hotel_in_krw": g_hotel_in_krw, "policy_over_budget": g_policy_over_budget,
    "irrelevant": g_irrelevant, "clarify": g_clarify, "recover": g_recover,
    "skill_book_cheapest": g_skill_book_cheapest, "skill_weather": g_skill_weather,
}
