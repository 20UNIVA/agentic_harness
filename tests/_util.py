"""Shared test helpers (work with pytest and with the stdlib runner)."""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
for _m in ("fla", "causal_conv1d"):          # CPU-only tests: never import Triton kernels
    sys.modules.setdefault(_m, None)

CK = os.environ.get("HARNESS_TEST_CKPT", "")


class Skip(Exception):
    pass


def skip(msg):
    if os.environ.get("PYTEST_CURRENT_TEST"):     # running under pytest
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


def needs_ckpt():
    if not CK or not os.path.isdir(CK):
        skip("set HARNESS_TEST_CKPT to a Qwen3.5-format HF checkpoint/tokenizer dir")


def needs_swift():
    if importlib.util.find_spec("swift") is None:
        skip("ms-swift not installed")


HAS_MCP_SDK = importlib.util.find_spec("mcp") is not None


def needs_mcp_sdk():
    if not HAS_MCP_SDK:
        skip("the `mcp` Python SDK (needed by the example MCP servers) is not installed")
