"""Retail customer-service environment (tau-bench-like, simplified and deterministic)."""
from __future__ import annotations

import random

from ..core.types import ToolError
from .base import Env, tool
from .names import CITIES, FIRST, LAST, STREETS

PRODUCTS = {
    "T-Shirt": {"color": ["black", "white", "blue", "red"], "size": ["S", "M", "L", "XL"], "base": 25},
    "Running Shoes": {"color": ["black", "white", "gray"], "size": ["8", "9", "10", "11"], "base": 120},
    "Backpack": {"color": ["black", "green", "navy"], "capacity": ["20L", "30L"], "base": 70},
    "Water Bottle": {"color": ["silver", "black", "pink"], "capacity": ["500ml", "1L"], "base": 18},
    "Desk Lamp": {"color": ["white", "black"], "brightness": ["standard", "high"], "base": 45},
    "Bluetooth Speaker": {"color": ["black", "blue"], "battery": ["10h", "20h"], "base": 90},
    "Mechanical Keyboard": {"switch": ["red", "brown", "blue"], "layout": ["full", "tenkeyless"], "base": 140},
    "Laptop": {"cpu": ["i5", "i7", "i9"], "ram": ["16GB", "32GB"], "base": 1100},
    "Office Chair": {"color": ["black", "gray"], "armrest": ["fixed", "adjustable"], "base": 260},
    "Coffee Maker": {"type": ["drip", "espresso"], "color": ["black", "silver"], "base": 85},
}

POLICY = """You are a customer service agent for an online store. Follow this policy strictly:
1. Before taking any action on an account, identify the customer by email, or by first name + last name + zip code.
2. You may only read or change orders that belong to the identified customer. Never act on another customer's order and never reveal another customer's information.
3. Only orders with status "pending" can be cancelled or have their shipping address changed. Valid cancellation reasons are "no longer needed" and "ordered by mistake".
4. Only orders with status "delivered" can have items returned or exchanged. An exchange must be for a different option of the same product.
5. Refunds go to the payment method the customer chooses among the methods on their account.
6. If the customer asks to cancel a pending order whose total is more than $1000, do not cancel it yourself: transfer the customer to a human agent.
7. When the request is completed, reply with a short summary. Do not make up information; use the tools."""


class RetailEnv(Env):
    name = "retail"
    version = "0.1"
    system_prompt = POLICY

    def reset(self, world: dict) -> None:
        rng = random.Random(world["seed"])
        n_users = world.get("params", {}).get("n_users", 12)
        products, item_index = {}, {}
        for pi, (pname, opts) in enumerate(PRODUCTS.items()):
            pid = f"P{1001 + pi}"
            keys = [k for k in opts if k != "base"]
            variants = {}
            for ii, (a, b) in enumerate([(x, y) for x in opts[keys[0]] for y in opts[keys[1]]]):
                iid = f"I{pid[1:]}{ii:02d}"
                price = round(opts["base"] * rng.uniform(0.85, 1.25), 2)
                variants[iid] = {"item_id": iid, "options": {keys[0]: a, keys[1]: b}, "price": price,
                                 "available": rng.random() > 0.15}
                item_index[iid] = pid
            products[pid] = {"product_id": pid, "name": pname, "variants": variants}
        users, orders = {}, {}
        used = set()
        order_no = rng.randint(1000000, 8000000)
        for _ in range(n_users):
            while True:
                fn, ln = rng.choice(FIRST), rng.choice(LAST)
                if (fn, ln) not in used:
                    used.add((fn, ln))
                    break
            num = rng.randint(1000, 9999)
            uid = f"{fn.lower()}_{ln.lower()}_{num}"
            city, zipc = rng.choice(CITIES)
            pms = [{"id": f"credit_card_{rng.randint(1000000, 9999999)}", "type": "credit_card",
                    "last4": f"{rng.randint(0, 9999):04d}"}]
            if rng.random() < 0.6:
                pms.append({"id": f"gift_card_{rng.randint(1000000, 9999999)}", "type": "gift_card",
                            "balance": round(rng.uniform(10, 300), 2)})
            if rng.random() < 0.4:
                pms.append({"id": f"paypal_{rng.randint(1000000, 9999999)}", "type": "paypal"})
            user = {"user_id": uid, "name": {"first_name": fn, "last_name": ln},
                    "email": f"{fn.lower()}.{ln.lower()}{num % 100}@example.com",
                    "address": {"street": f"{rng.randint(1, 999)} {rng.choice(STREETS)}", "city": city, "zip": zipc},
                    "payment_methods": pms, "orders": []}
            for _k in range(rng.randint(2, 4)):
                order_no += rng.randint(7, 997)
                oid = f"#W{order_no}"
                n_items = rng.choice([1, 1, 2, 2, 3])
                items = []
                for pid in rng.sample(sorted(products), n_items):
                    iid = rng.choice(sorted(products[pid]["variants"]))
                    v = products[pid]["variants"][iid]
                    items.append({"item_id": iid, "product_id": pid, "name": products[pid]["name"],
                                  "options": dict(v["options"]), "price": v["price"]})
                status = rng.choices(["pending", "processed", "delivered", "cancelled"], [4, 2, 4, 1])[0]
                orders[oid] = {"order_id": oid, "user_id": uid, "status": status, "items": items,
                               "date": f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}",
                               "payment_method_id": pms[0]["id"], "address": dict(user["address"]),
                               "total": round(sum(i["price"] for i in items), 2)}
                user["orders"].append(oid)
            users[uid] = user
        self.products = products
        self.item_index = item_index
        self.db = {"users": users, "orders": orders, "transfers": []}

    def state(self) -> dict:
        return {"orders": self.db["orders"], "addresses": {u: d["address"] for u, d in self.db["users"].items()},
                "transfers": len(self.db["transfers"])}

    # ---------------------------------------------------------------- read tools
    @tool("Find a customer's user id by email address.", {"email": {"type": "string", "description": "Customer email"}})
    def find_user_id_by_email(self, email):
        for u in self.db["users"].values():
            if u["email"].lower() == email.strip().lower():
                return u["user_id"]
        raise ToolError("user not found")

    @tool("Find a customer's user id by first name, last name and zip code.",
          {"first_name": {"type": "string"}, "last_name": {"type": "string"}, "zip": {"type": "string"}})
    def find_user_id_by_name_zip(self, first_name, last_name, zip):
        for u in self.db["users"].values():
            if (u["name"]["first_name"].lower() == first_name.strip().lower()
                    and u["name"]["last_name"].lower() == last_name.strip().lower()
                    and u["address"]["zip"] == zip.strip()):
                return u["user_id"]
        raise ToolError("user not found")

    @tool("Get a customer's profile: name, email, address, payment methods and order ids.",
          {"user_id": {"type": "string"}})
    def get_user_details(self, user_id):
        u = self.db["users"].get(user_id)
        if not u:
            raise ToolError("user not found")
        return u

    @tool("Get the details of an order: status, items (with item ids and prices), total, address, date.",
          {"order_id": {"type": "string", "description": "Order id such as '#W1234567'"}})
    def get_order_details(self, order_id):
        oid = order_id if order_id.startswith("#") else "#" + order_id
        o = self.db["orders"].get(oid)
        if not o:
            raise ToolError("order not found")
        return o

    @tool("Get a product and all its variants (item ids, options, price, availability).",
          {"product_id": {"type": "string"}})
    def get_product_details(self, product_id):
        p = self.products.get(product_id)
        if not p:
            raise ToolError("product not found")
        return p

    @tool("List all product types as a mapping from product name to product id.", {})
    def list_product_types(self):
        return {p["name"]: pid for pid, p in sorted(self.products.items())}

    # --------------------------------------------------------------- write tools
    def _order(self, order_id):
        oid = order_id if order_id.startswith("#") else "#" + order_id
        o = self.db["orders"].get(oid)
        if not o:
            raise ToolError("order not found")
        return o

    @tool("Cancel a pending order.",
          {"order_id": {"type": "string"},
           "reason": {"type": "string", "enum": ["no longer needed", "ordered by mistake"]}}, mutating=True)
    def cancel_pending_order(self, order_id, reason):
        o = self._order(order_id)
        if o["status"] != "pending":
            raise ToolError("non-pending order cannot be cancelled")
        o["status"] = "cancelled"
        o["cancel_reason"] = reason
        return {"order_id": o["order_id"], "status": "cancelled"}

    @tool("Change the shipping address of a pending order.",
          {"order_id": {"type": "string"}, "street": {"type": "string"}, "city": {"type": "string"},
           "zip": {"type": "string"}}, mutating=True)
    def modify_pending_order_address(self, order_id, street, city, zip):
        o = self._order(order_id)
        if o["status"] != "pending":
            raise ToolError("non-pending order cannot be modified")
        o["address"] = {"street": street.strip(), "city": city.strip(), "zip": zip.strip()}
        return {"order_id": o["order_id"], "address": o["address"]}

    def _pm(self, o, payment_method_id):
        u = self.db["users"][o["user_id"]]
        if payment_method_id not in {p["id"] for p in u["payment_methods"]}:
            raise ToolError("payment method not found on this account")

    @tool("Return items of a delivered order and refund to a payment method of the customer.",
          {"order_id": {"type": "string"}, "item_ids": {"type": "array", "items": {"type": "string"}},
           "payment_method_id": {"type": "string"}}, mutating=True)
    def return_delivered_order_items(self, order_id, item_ids, payment_method_id):
        o = self._order(order_id)
        if o["status"] != "delivered":
            raise ToolError("only delivered orders can be returned")
        have = [i["item_id"] for i in o["items"]]
        if not item_ids or any(i not in have for i in item_ids):
            raise ToolError("some item ids are not in this order")
        self._pm(o, payment_method_id)
        o["status"] = "return requested"
        o["return"] = {"item_ids": sorted(item_ids), "payment_method_id": payment_method_id}
        return {"order_id": o["order_id"], "status": o["status"], "return": o["return"]}

    @tool("Exchange items of a delivered order for other available options of the same products.",
          {"order_id": {"type": "string"}, "item_ids": {"type": "array", "items": {"type": "string"}},
           "new_item_ids": {"type": "array", "items": {"type": "string"}},
           "payment_method_id": {"type": "string"}}, mutating=True)
    def exchange_delivered_order_items(self, order_id, item_ids, new_item_ids, payment_method_id):
        o = self._order(order_id)
        if o["status"] != "delivered":
            raise ToolError("only delivered orders can be exchanged")
        have = [i["item_id"] for i in o["items"]]
        if not item_ids or len(item_ids) != len(new_item_ids) or any(i not in have for i in item_ids):
            raise ToolError("item_ids must be items of this order and match new_item_ids in length")
        for old, new in zip(item_ids, new_item_ids):
            pid = self.item_index.get(new)
            if pid is None or pid != self.item_index.get(old) or new == old:
                raise ToolError(f"{new} is not a different option of the same product as {old}")
            if not self.products[pid]["variants"][new]["available"]:
                raise ToolError(f"{new} is not available")
        self._pm(o, payment_method_id)
        o["status"] = "exchange requested"
        o["exchange"] = {"pairs": sorted(zip(item_ids, new_item_ids)), "payment_method_id": payment_method_id}
        return {"order_id": o["order_id"], "status": o["status"], "exchange": o["exchange"]}

    @tool("Transfer the customer to a human agent with a short summary of the issue.",
          {"summary": {"type": "string"}}, mutating=True)
    def transfer_to_human_agents(self, summary):
        self.db["transfers"].append(summary)
        return "Transfer successful"
