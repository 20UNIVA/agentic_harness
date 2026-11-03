# Travel API reference

All dates are `YYYY-MM-DD`. Inventory exists for 2026-12-01 .. 2026-12-05.

| Method | Path | Parameters | Response |
|---|---|---|---|
| GET | `/airports` | query `city` (city name) | `{"city", "airports": [IATA codes]}` |
| GET | `/flights/search` | query `origin`, `destination` (IATA codes), `date` | `{"flights": [{"flight_id", "origin", "destination", "date", "depart", "arrive", "price_usd", "seats_left"}]}` |
| GET | `/hotels/search` | query `city` (city name), `checkin`, `checkout` | `{"hotels": [{"hotel_id", "name", "city", "price_per_night_usd", "rating"}]}` |
| GET | `/weather` | query `city`, `date` | `{"city", "date", "condition", "high_c", "low_c"}` |
| GET | `/fx/rate` | query `base`, `quote` (ISO currency codes) | `{"base", "quote", "rate"}` meaning 1 base = rate quote |
| POST | `/bookings` | body `{"type": "flight", "item_id": <flight_id>, "traveler": <full name>}` | booking record with `booking_id` |
| POST | `/bookings` | body `{"type": "hotel", "item_id": <hotel_id>, "traveler": <full name>, "checkin", "checkout"}` | booking record with `booking_id` |
| GET | `/bookings` | query `traveler` | `{"bookings": [...]}` |
| DELETE | `/bookings/<booking_id>` | none | `{"deleted": booking_id}` |

Errors come back as `{"error": "<status> <reason>: <detail>"}`, for example
`400 Bad Request: origin and destination must be 3-letter IATA airport codes`.

Example call:

```
http_request(method="GET", path="/flights/search", query={"origin": "PUS", "destination": "CJU", "date": "2026-12-02"})
```
