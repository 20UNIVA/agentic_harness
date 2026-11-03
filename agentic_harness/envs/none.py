"""Empty environment for tasks whose tools come only from skills and/or external MCP servers."""
from .base import Env


class NoneEnv(Env):
    name = "none"
    version = "0.1"
    system_prompt = "You are a helpful assistant that completes tasks by calling the available tools. When done, reply with a short summary."

    def reset(self, world: dict) -> None:
        self.world = world          # kept so generators can build companion worlds from the same seed
        self.db = {}
