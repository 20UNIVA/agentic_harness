# Contributing

- Run the tests before sending a change: `pytest tests/` (or `python tests/run_tests.py`). Tests that need a model
  checkpoint (`HARNESS_TEST_CKPT`), ms-swift or the `mcp` SDK skip themselves when those are missing.
- New benchmark templates must pass the gates in `agentic-harness gen-tasks` (gold passes, do-nothing fails, no
  prompt overlap between splits). Add them to the generator's `GENERATORS` dict.
- New environments subclass `agentic_harness.envs.base.Env` (deterministic `reset(world)`, `@tool` methods, a
  normalized `state()` for verification) and are registered in `agentic_harness/envs/__init__.py`.
- Keep the core free of required dependencies; optional integrations import their packages lazily.
- Infrastructure errors must surface as `InfraError` (reward `None`), never as a zero score.
