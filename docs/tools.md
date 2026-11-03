# Tools and toolkits

Everything the model can call in an episode comes from a **ToolSession**: the built-in env of the task (if any),
user and built-in **toolkits**, **skills** ([skills.md](skills.md)) and **MCP servers** ([mcp.md](mcp.md)).
Tool names must be unique across all of them; a collision is reported before the episode starts.

## Function tools

```python
from agentic_harness.tools import tool, ToolError

TOOLKIT_NAME = "finance"          # optional; default = module file name

@tool
def convert_currency(amount: float, source: str, target: str) -> dict:
    """Convert an amount between currencies.

    Args:
        amount: Amount in the source currency.
        source: ISO code of the source currency.
        target: ISO code of the target currency.
    """
    if source not in RATES:
        raise ToolError(f"unsupported currency {source}")   # shown to the model as {"error": ...}
    return {"amount": ...}
```

The JSON schema is inferred from the signature and docstring:

| Python | JSON schema |
|---|---|
| `str`, `int`, `float`, `bool` | `string`, `integer`, `number`, `boolean` |
| `list[X]`, `tuple`, `set` | `array` (with `items`) |
| `dict` | `object` |
| `Literal["a", "b"]`, `Enum` | `enum` |
| `Optional[X]` / `X \| None` | schema of `X` |
| parameter without default | listed in `required` |
| default value | `default` |

Descriptions come from the first docstring paragraph and Google-style `Args:` or Sphinx `:param x:` entries.

Decorator forms:

```python
@tool                                              # everything inferred
@tool(mutating=True, name="book_room")             # options
@tool("Explicit description", {"city": {"type": "string"}}, required=["city"], mutating=True)   # explicit schema
```

`mutating=True` marks tools that change state (used for action-recall metrics and reports).

## Stateful toolkits

```python
from agentic_harness.tools import Toolkit, tool, ToolError

class LibraryToolkit(Toolkit):
    name = "library"
    version = "0.1"
    system_prompt = "Library rule: at most 2 books per member."   # appended to the task's system prompt

    def reset(self, world):                 # fresh instance per episode; world = {"seed": ..., "params": ...}
        self.db = {"loans": {}}

    def state(self):                        # what {"type": "state", "toolkit": "library"} checks inspect
        return {"loans": self.db["loans"]}

    def check(self, chk):                   # custom {"type": "toolkit", "toolkit": "library", ...} checks
        return self.db["loans"].get(chk["book_id"]) == chk["member"]

    @tool(mutating=True)
    def checkout_book(self, book_id: str, member: str) -> dict:
        """Lend a book to a member."""
        ...
```

## Registering toolkits

```bash
agentic-harness run --toolkits examples/tools/finance_tools.py --toolkits my_pkg.my_tools --toolkits my_file.py:MyToolkit ...
```

- A file or module contributes every `Toolkit` subclass defined in it, plus one function toolkit if it has
  module-level `@tool` functions.
- Tasks list the toolkit names they need in `toolkits`; `--extra-toolkits name` offers a toolkit in every episode.
- Arguments are validated against the schema before your function runs (types, enums, required, unknown arguments).
- Tool results longer than `--max-tool-result-chars` (default 20000) are truncated with a notice before the model sees them.
- `ToolError` messages go back to the model. Any other exception is a bug in tool code and makes the episode an
  infrastructure failure (`reward=None`), so tool bugs never look like model mistakes.

## Built-in toolkits

| name | tools | notes |
|---|---|---|
| `shell` | `run_command`, `read_text_file`, `write_text_file` | **Disabled unless `--allow-shell`.** Runs `bash -c` in a per-episode workspace with a timeout and output truncation. Offered skills are copied to `skills/<name>/` and task `files` are written into the workspace. Not a security boundary. |

## Built-in environments

`retail`, `workspace`, `filesystem`, `travel_api` and `data_sql` are deterministic mock services used by the benchmark
([benchmark.md](benchmark.md)). A user task can use one with `"env": {"domain": "retail", "seed": 3}`; the env's tools
and system prompt are then included, and env checks / state hashes are available for verification.
