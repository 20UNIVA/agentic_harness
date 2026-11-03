"""Example toolkit #3: stateful toolkit with its own system prompt, a policy rule and a custom `check`.

Registered as toolkit "library". Custom checks: {"type": "toolkit", "toolkit": "library", "kind": "on_loan", ...}.
"""
from agentic_harness.tools import Toolkit, ToolError, tool

BOOKS = [
    ("B1", "The Pragmatic Programmer", "Hunt & Thomas", "software"),
    ("B2", "Designing Data-Intensive Applications", "Kleppmann", "software"),
    ("B3", "The Little Prince", "Saint-Exupery", "fiction"),
    ("B4", "Sapiens", "Harari", "history"),
    ("B5", "Clean Code", "Martin", "software"),
]


class LibraryToolkit(Toolkit):
    name = "library"
    version = "0.1"
    system_prompt = "Library rule: a member may have at most 2 books on loan at the same time."

    def reset(self, world: dict) -> None:
        self.db = {"loans": {"B3": "alice"}, "members": ["alice", "bob", "carol"]}

    def state(self) -> dict:
        return {"loans": dict(sorted(self.db["loans"].items()))}

    def check(self, chk: dict) -> bool:
        if chk.get("kind") == "on_loan":
            return self.db["loans"].get(chk["book_id"]) == chk["member"]
        if chk.get("kind") == "available":
            return chk["book_id"] not in self.db["loans"]
        raise ValueError(f"unknown library check {chk}")

    @tool
    def search_books(self, query: str) -> list:
        """Search the catalog by title, author or topic (case-insensitive) and show availability.

        Args:
            query: Words to search for.
        """
        q = query.lower()
        return [{"book_id": b, "title": t, "author": a, "topic": tp, "available": b not in self.db["loans"]}
                for b, t, a, tp in BOOKS if q in t.lower() or q in a.lower() or q in tp]

    @tool(mutating=True)
    def checkout_book(self, book_id: str, member: str) -> dict:
        """Lend a book to a member.

        Args:
            book_id: Catalog id such as B2.
            member: Member username.
        """
        if member not in self.db["members"]:
            raise ToolError(f"unknown member {member!r}")
        if book_id not in {b[0] for b in BOOKS}:
            raise ToolError(f"unknown book {book_id!r}")
        if book_id in self.db["loans"]:
            raise ToolError("book is already on loan")
        self.db["loans"][book_id] = member      # the 2-book limit is a policy the model must follow, not enforced here
        return {"book_id": book_id, "member": member, "status": "on loan"}

    @tool(mutating=True)
    def return_book(self, book_id: str) -> dict:
        """Return a book.

        Args:
            book_id: Catalog id.
        """
        if book_id not in self.db["loans"]:
            raise ToolError("book is not on loan")
        del self.db["loans"][book_id]
        return {"book_id": book_id, "status": "available"}
