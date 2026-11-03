"""Example toolkit #2: a stateful Toolkit class (fresh state per episode). Registered as toolkit "todo".

Tasks can verify the final state with  {"type": "state", "toolkit": "todo", "json_subset": {...}}.
"""
from agentic_harness.tools import Toolkit, ToolError, tool


class TodoToolkit(Toolkit):
    name = "todo"
    version = "0.1"

    def reset(self, world: dict) -> None:
        # world["seed"] could be used to build a seed-dependent initial state; this example starts with two items.
        self.db = {"items": [{"id": 1, "text": "Renew passport", "done": False, "priority": "high"},
                             {"id": 2, "text": "Buy groceries", "done": True, "priority": "low"}], "next_id": 3}

    def state(self) -> dict:
        return {"items": [{k: v for k, v in it.items() if k != "id"} for it in self.db["items"]]}

    def _get(self, item_id: int) -> dict:
        for it in self.db["items"]:
            if it["id"] == item_id:
                return it
        raise ToolError(f"no item with id {item_id}")

    @tool
    def list_items(self, include_done: bool = False) -> list:
        """List todo items.

        Args:
            include_done: Also list completed items.
        """
        return [it for it in self.db["items"] if include_done or not it["done"]]

    @tool(mutating=True)
    def add_item(self, text: str, priority: str = "normal") -> dict:
        """Add a todo item.

        Args:
            text: What needs to be done.
            priority: low, normal or high.
        """
        if priority not in ("low", "normal", "high"):
            raise ToolError("priority must be low, normal or high")
        it = {"id": self.db["next_id"], "text": text.strip(), "done": False, "priority": priority}
        self.db["next_id"] += 1
        self.db["items"].append(it)
        return it

    @tool(mutating=True)
    def complete_item(self, item_id: int) -> dict:
        """Mark an item as done.

        Args:
            item_id: Id from list_items.
        """
        it = self._get(item_id)
        it["done"] = True
        return it
