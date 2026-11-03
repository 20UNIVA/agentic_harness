"""`agentic-harness init <dir>`: scaffold a project with a toolkit, tasks and run configs.

  agentic-harness init my_agent_eval && cd my_agent_eval
  agentic-harness run --config run_oracle.json          # validates the tasks without a model
  agentic-harness run --config run_model.json           # edit the model block first
"""
from __future__ import annotations

import argparse
import json
import os

TOOLS = '''"""Your tools. Every @tool function becomes a tool of the toolkit named TOOLKIT_NAME."""
from agentic_harness.tools import Toolkit, ToolError, tool

TOOLKIT_NAME = "weather"

_FORECAST = {"seoul": {"high_c": 21, "condition": "sunny"}, "busan": {"high_c": 24, "condition": "cloudy"}}


@tool
def get_forecast(city: str) -> dict:
    """Tomorrow's forecast for a city.

    Args:
        city: City name, e.g. Seoul.
    """
    f = _FORECAST.get(city.strip().lower())
    if f is None:
        raise ToolError(f"no forecast for {city!r}; known cities: {sorted(_FORECAST)}")
    return {"city": city, **f}


class Notes(Toolkit):
    """A stateful toolkit: a fresh instance per episode."""
    name = "notes"

    def reset(self, world):
        self.db = {"notes": []}

    def state(self):
        return {"notes": self.db["notes"]}

    @tool(mutating=True)
    def add_note(self, text: str) -> dict:
        """Save a note.

        Args:
            text: The note text.
        """
        self.db["notes"].append(text.strip())
        return {"count": len(self.db["notes"])}
'''

TASKS = {"tasks": [
    {"id": "forecast", "instruction": "Will it be warmer in Seoul or Busan tomorrow? Give both highs.",
     "toolkits": ["weather"],
     "gold_actions": [{"name": "get_forecast", "arguments": {"city": "Seoul"}},
                      {"name": "get_forecast", "arguments": {"city": "Busan"}}],
     "gold_final": "Busan (24°C) will be warmer than Seoul (21°C).",
     "verify": {"answers": ["Busan", {"number": 24, "tol": 0}, {"number": 21, "tol": 0}]}},
    {"id": "note", "instruction": "Check tomorrow's weather in Seoul and save a note 'Seoul: <condition>'.",
     "toolkits": ["weather", "notes"],
     "gold_actions": [{"name": "get_forecast", "arguments": {"city": "Seoul"}}, {"name": "add_note", "arguments": {"text": "Seoul: sunny"}}],
     "gold_final": "Saved the note 'Seoul: sunny'.",
     "verify": {"checks": [{"type": "state", "toolkit": "notes", "json_subset": {"notes": ["Seoul: sunny"]}}]}},
]}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentic-harness init")
    ap.add_argument("directory")
    a = ap.parse_args(argv)
    d = os.path.abspath(a.directory)
    if os.path.exists(d) and os.listdir(d):
        raise SystemExit(f"{d} exists and is not empty")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "my_tools.py"), "w").write(TOOLS)
    json.dump(TASKS, open(os.path.join(d, "tasks.json"), "w"), indent=1)
    base = {"tasks": "tasks.json", "toolkits": ["my_tools.py"], "no_example_resources": True}
    json.dump({"model": {"provider": "oracle"}, **base, "out": "runs/oracle"}, open(os.path.join(d, "run_oracle.json"), "w"), indent=1)
    json.dump({"model": {"provider": "openai_compatible", "base_url": "http://127.0.0.1:8000/v1", "model": "YOUR_MODEL",
                         "api_key_env": "OPENAI_API_KEY", "sampling": {"temperature": 0.2, "max_new_tokens": 1024}},
               **base, "n_attempts": 2, "out": "runs/model"}, open(os.path.join(d, "run_model.json"), "w"), indent=1)
    print(f"created {d}:\n  my_tools.py  tasks.json  run_oracle.json  run_model.json\n"
          f"next:\n  cd {a.directory}\n  agentic-harness run --config run_oracle.json\n"
          f"  # edit the model block in run_model.json, then\n  agentic-harness run --config run_model.json\n"
          f"  agentic-harness chat --config run_model.json     # talk to the agent yourself; chats go to runs/chat")


if __name__ == "__main__":
    main()
