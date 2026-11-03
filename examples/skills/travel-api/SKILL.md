---
name: travel-api
description: How to use the Travel API through the http_request tool (search flights and hotels, weather, exchange rates, bookings) and the booking rules. Use for any travel search or booking request.
---

# Travel API

The Travel API is reached only through the `http_request` tool. Pass the path (for example `/flights/search`),
query parameters in `query` and JSON bodies in `body`. The base URL is implicit.

**Read `reference/endpoints.md` before your first API call.** It lists every route with its parameters and
response fields.

## Workflow

1. Flights need IATA airport codes. If the user gives city names, look them up with `GET /airports?city=<city>` first.
   A city can have more than one airport; search each pair if needed.
2. Search, then compare results yourself: "cheapest available" means the lowest `price_usd` among flights with
   `seats_left > 0`.
3. Book with `POST /bookings` using the exact traveler name the user gave.
4. Report ids and prices exactly as returned.

## Booking rules

- Never book a sold-out flight (`seats_left = 0`).
- A single booking must not exceed $2000 (hotel price = nightly rate x nights). If it would, do not book; explain why.
- Only book for the traveler named by the user.
