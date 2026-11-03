from .base import Env, tool  # noqa: F401
from .data_sql import DataSqlEnv
from .filesystem import FilesystemEnv
from .none import NoneEnv
from .retail import RetailEnv
from .travel_api import TravelApiEnv
from .workspace import WorkspaceEnv

ENVS = {e.name: e for e in (RetailEnv, WorkspaceEnv, FilesystemEnv, TravelApiEnv, DataSqlEnv, NoneEnv)}


def make_env(domain: str, world: dict | None = None):
    env = ENVS[domain]()
    if world is not None:
        env.reset(world)
    return env
