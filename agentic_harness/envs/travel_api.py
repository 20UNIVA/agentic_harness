"""Travel REST API environment: one generic `http_request` tool + API docs in the system prompt.

Measures whether the model can read API documentation and compose method/path/query/body correctly.
Everything is an in-process mock router (no network).
"""
from __future__ import annotations

import random
import re

from ..core.types import ToolError
from .base import Env, tool

AIRPORTS = {"Seoul": ["ICN", "GMP"], "Tokyo": ["NRT", "HND"], "Osaka": ["KIX"], "Busan": ["PUS"],
            "Jeju": ["CJU"], "Singapore": ["SIN"], "Bangkok": ["BKK"], "Taipei": ["TPE"], "Hong Kong": ["HKG"]}
CURRENCIES = {"USD": 1.0, "KRW": 1380.0, "JPY": 151.0, "EUR": 0.92, "SGD": 1.34, "THB": 36.0, "TWD": 32.0}
DATES = ["2026-12-01", "2026-12-02", "2026-12-03", "2026-12-04", "2026-12-05"]
HOTEL_WORDS = ["Grand", "Harbor", "Sky", "Garden", "Central", "Riverside", "Palace", "Urban", "Lotus", "Pine"]
CONDITIONS = ["sunny", "cloudy", "rain", "snow", "windy"]

API_DOC = """You are a travel assistant. You can call the Travel API with the `http_request` tool.
Base URL is implicit; pass only the path. Query parameters go in `query`, JSON bodies in `body`.

API reference:
- GET /airports?city=<city name>            -> {"city", "airports": [IATA codes]}
- GET /flights/search?origin=<IATA>&destination=<IATA>&date=<YYYY-MM-DD>
                                             -> {"flights": [{"flight_id","origin","destination","date","depart","arrive","price_usd","seats_left"}]}
- GET /hotels/search?city=<city name>&checkin=<YYYY-MM-DD>&checkout=<YYYY-MM-DD>
                                             -> {"hotels": [{"hotel_id","name","city","price_per_night_usd","rating"}]}
- GET /weather?city=<city name>&date=<YYYY-MM-DD> -> {"city","date","condition","high_c","low_c"}
- GET /fx/rate?base=<ISO>&quote=<ISO>       -> {"base","quote","rate"}  (1 base = rate quote)
- POST /bookings  body={"type":"flight","item_id":<flight_id>,"traveler":<full name>}
                  body={"type":"hotel","item_id":<hotel_id>,"traveler":<full name>,"checkin":<date>,"checkout":<date>}
                                             -> {"booking_id", ...}
- GET /bookings?traveler=<full name>         -> {"bookings": [...]}
- DELETE /bookings/<booking_id>              -> {"deleted": booking_id}

Rules:
1. Only book for the traveler named by the user. Never book anything that is sold out (seats_left = 0).
2. A single booking must not exceed $2000; if the user asks for one, explain that you cannot book it.
3. Report prices exactly as returned by the API. When done, reply with a short summary."""


class TravelApiEnv(Env):
    name = "travel_api"
    version = "0.2"   # 0.2: tolerate paths without a leading slash (v0.1 returned 404)
    system_prompt = API_DOC

    def reset(self, world: dict) -> None:
        rng = random.Random(world["seed"])
        flights, hotels, weather = {}, {}, {}
        n = 0
        cities = list(AIRPORTS)
        for o in cities:
            for d in cities:
                if o == d or rng.random() < 0.55:
                    continue
                for date in DATES:
                    for _ in range(rng.randint(0, 3)):
                        n += 1
                        oa, da = rng.choice(AIRPORTS[o]), rng.choice(AIRPORTS[d])
                        h = rng.randint(6, 21)
                        dur = rng.randint(1, 6)
                        fid = f"FL{n:04d}"
                        flights[fid] = {"flight_id": fid, "origin": oa, "destination": da, "date": date,
                                        "depart": f"{h:02d}:{rng.choice(['00', '15', '30', '45'])}",
                                        "arrive": f"{(h + dur) % 24:02d}:{rng.choice(['05', '20', '35', '50'])}",
                                        "price_usd": rng.randint(80, 900) * (3 if rng.random() < 0.05 else 1),
                                        "seats_left": rng.choice([0, 1, 2, 5, 9, 20])}
        k = 0
        for c in cities:
            for w in rng.sample(HOTEL_WORDS, rng.randint(3, 5)):
                k += 1
                hid = f"HT{k:03d}"
                hotels[hid] = {"hotel_id": hid, "name": f"{w} {c} Hotel", "city": c,
                               "price_per_night_usd": rng.randint(60, 600), "rating": round(rng.uniform(3.0, 4.9), 1)}
            for date in DATES:
                hi = rng.randint(-2, 33)
                weather[(c, date)] = {"city": c, "date": date, "condition": rng.choice(CONDITIONS),
                                      "high_c": hi, "low_c": hi - rng.randint(3, 10)}
        fx = {}
        for b in CURRENCIES:
            for q in CURRENCIES:
                if b != q:
                    jitter = 1 + rng.uniform(-0.02, 0.02)
                    fx[(b, q)] = round(CURRENCIES[q] / CURRENCIES[b] * jitter, 4)
        self.catalog = {"flights": flights, "hotels": hotels, "weather": weather, "fx": fx}
        self.db = {"bookings": {}, "next": 1}

    def state(self) -> dict:
        return {"bookings": sorted(
            (b["type"], b["item_id"], b["traveler"].lower(), b.get("checkin"), b.get("checkout"))
            for b in self.db["bookings"].values())}

    # ------------------------------------------------------------------ router
    @tool("Send an HTTP request to the Travel API.",
          {"method": {"type": "string", "enum": ["GET", "POST", "DELETE"]},
           "path": {"type": "string", "description": "e.g. /flights/search"},
           "query": {"type": "object", "description": "query parameters"},
           "body": {"type": "object", "description": "JSON body for POST"}},
          required=["method", "path"], mutating=True)
    def http_request(self, method, path, query=None, body=None):
        query = {k: (str(v).strip() if v is not None else "") for k, v in (query or {}).items()}
        path = path.strip()
        if path.startswith(("http://", "https://")):     # full URL -> keep only the path
            path = "/" + path.split("://", 1)[1].split("/", 1)[-1] if "/" in path.split("://", 1)[1] else "/"
        if not path.startswith("/"):
            path = "/" + path
        if "?" in path:   # tolerate query string embedded in path
            path, qs = path.split("?", 1)
            for part in qs.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    query.setdefault(k, v.replace("+", " ").replace("%20", " "))
        body = body or {}
        route = (method.upper(), path.rstrip("/") or "/")
        if route == ("GET", "/airports"):
            city = self._city(query)
            return {"city": city, "airports": AIRPORTS[city]}
        if route == ("GET", "/flights/search"):
            o, d, date = query.get("origin", "").upper(), query.get("destination", "").upper(), query.get("date", "")
            for code in (o, d):
                if not re.fullmatch(r"[A-Z]{3}", code):
                    raise ToolError("400 Bad Request: origin and destination must be 3-letter IATA airport codes; "
                                    "use GET /airports?city=<city> to look them up")
            self._date(date)
            fl = [f for f in self.catalog["flights"].values() if f["origin"] == o and f["destination"] == d and f["date"] == date]
            return {"flights": sorted(fl, key=lambda f: f["depart"])}
        if route == ("GET", "/hotels/search"):
            city = self._city(query)
            ci, co = self._date(query.get("checkin", "")), self._date(query.get("checkout", ""))
            if co <= ci:
                raise ToolError("400 Bad Request: checkout must be after checkin")
            return {"hotels": sorted([h for h in self.catalog["hotels"].values() if h["city"] == city],
                                     key=lambda h: h["hotel_id"])}
        if route == ("GET", "/weather"):
            city = self._city(query)
            date = self._date(query.get("date", ""))
            return self.catalog["weather"][(city, date)]
        if route == ("GET", "/fx/rate"):
            b, q = query.get("base", "").upper(), query.get("quote", "").upper()
            if (b, q) not in self.catalog["fx"]:
                raise ToolError(f"400 Bad Request: unsupported currency pair; supported: {sorted(CURRENCIES)}")
            return {"base": b, "quote": q, "rate": self.catalog["fx"][(b, q)]}
        if route == ("GET", "/bookings"):
            t = query.get("traveler", "").lower()
            return {"bookings": [b for b in self.db["bookings"].values() if b["traveler"].lower() == t]}
        if route == ("POST", "/bookings"):
            return self._book(body)
        if method.upper() == "DELETE" and path.startswith("/bookings/"):
            bid = path.split("/")[-1]
            if bid not in self.db["bookings"]:
                raise ToolError("404 Not Found: booking not found")
            del self.db["bookings"][bid]
            return {"deleted": bid}
        raise ToolError(f"404 Not Found: no route for {method.upper()} {path}. Valid routes: GET /airports, GET /flights/search, "
                        "GET /hotels/search, GET /weather, GET /fx/rate, GET /bookings, POST /bookings, DELETE /bookings/<id>")

    def _city(self, query):
        c = query.get("city", "").strip()
        for name in AIRPORTS:
            if name.lower() == c.lower():
                return name
        raise ToolError(f"404 Not Found: unknown city '{c}'. Known cities: {', '.join(AIRPORTS)}")

    @staticmethod
    def _date(d):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d or ""):
            raise ToolError("400 Bad Request: dates must be YYYY-MM-DD")
        if d not in DATES:
            raise ToolError(f"400 Bad Request: no inventory for {d}; available dates {DATES[0]}..{DATES[-1]}")
        return d

    def _book(self, body):
        typ, item, trav = body.get("type"), str(body.get("item_id", "")), str(body.get("traveler", "")).strip()
        if not trav:
            raise ToolError("400 Bad Request: traveler is required")
        if typ == "flight":
            f = self.catalog["flights"].get(item)
            if not f:
                raise ToolError("404 Not Found: flight not found")
            if f["seats_left"] <= 0:
                raise ToolError("409 Conflict: sold out")
            rec = {"type": "flight", "item_id": item, "traveler": trav, "price_usd": f["price_usd"]}
        elif typ == "hotel":
            h = self.catalog["hotels"].get(item)
            if not h:
                raise ToolError("404 Not Found: hotel not found")
            ci, co = self._date(str(body.get("checkin", ""))), self._date(str(body.get("checkout", "")))
            if co <= ci:
                raise ToolError("400 Bad Request: checkout must be after checkin")
            nights = DATES.index(co) - DATES.index(ci)
            rec = {"type": "hotel", "item_id": item, "traveler": trav, "checkin": ci, "checkout": co,
                   "price_usd": h["price_per_night_usd"] * nights}
        else:
            raise ToolError("400 Bad Request: type must be 'flight' or 'hotel'")
        bid = f"BK{self.db['next']:04d}"
        self.db["next"] += 1
        rec["booking_id"] = bid
        self.db["bookings"][bid] = rec
        return rec
