#!/usr/bin/env python
"""Stdlib test runner (no pytest needed): runs every test_* function in tests/test_*.py.

  python tests/run_tests.py            # all tests; slow ones (model/tokenizer) skip unless HARNESS_TEST_CKPT is set
  python tests/run_tests.py -k mcp     # only tests whose name contains "mcp"
"""
import importlib
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _util import Skip  # noqa: E402


def main():
    pat = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    mods = sorted(f[:-3] for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
    passed = failed = skipped = 0
    for m in mods:
        mod = importlib.import_module(m)
        for name in [n for n in vars(mod) if n.startswith("test_") and callable(getattr(mod, n))]:
            if pat and pat not in name:
                continue
            t0 = time.time()
            try:
                getattr(mod, name)()
                passed += 1
                print(f"PASS {m}::{name} ({time.time() - t0:.1f}s)", flush=True)
            except Skip as e:
                skipped += 1
                print(f"SKIP {m}::{name}: {e}", flush=True)
            except Exception:
                failed += 1
                print(f"FAIL {m}::{name}", flush=True)
                traceback.print_exc()
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
