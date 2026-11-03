"""Task generation entry point."""
from __future__ import annotations

import random

from ..envs import make_env
from . import data_sql_gen, filesystem_gen, mcp_examples_gen, retail_gen, travel_gen, workspace_gen
from .common import SPLIT_SEED_BASE, GenSkip, build_task, stylize, visible_hash

DOMAIN_GENERATORS = {
    "retail": retail_gen.GENERATORS,
    "workspace": workspace_gen.GENERATORS,
    "filesystem": filesystem_gen.GENERATORS,
    "travel_api": travel_gen.GENERATORS,
    "data_sql": data_sql_gen.GENERATORS,
    "mcp_examples": mcp_examples_gen.GENERATORS,     # tasks for the example MCP servers (needs examples/mcp/servers.json)
}
GROUP_ENV = {"mcp_examples": "none"}                 # generator group -> env domain (templates may override "domain")


def generate_one(domain: str, template: str, seed: int, lang: str, split: str) -> dict:
    """Build one task from (domain, template, world seed). Raises GenSkip if the world doesn't fit."""
    env = make_env(GROUP_ENV.get(domain, domain), {"seed": seed, "params": {}})
    rng = random.Random(f"{domain}|{template}|{seed}|{lang}")
    spec = DOMAIN_GENERATORS[domain][template](rng, env, lang)
    env_domain = spec.pop("domain", GROUP_ENV.get(domain, domain))
    spec.setdefault("max_turns", 14 if spec.get("difficulty", 1) >= 3 else 10)
    if spec["category"] != "irrelevant":
        spec["instruction"] = stylize(rng, lang, spec["instruction"])
    return build_task(domain=env_domain, seed=seed, lang=lang, split=split, gen_name=f"{domain}.{template}", **spec)


def generate(split: str, n_per_template: int, domains=None, ko_ratio: float = 0.3, templates=None,
             seed_offset: int = 0, max_tries: int = 20, exclude_visible: set | None = None, max_visible_dup: int = 0):
    """exclude_visible: visible-prompt hashes that must not appear (e.g. train prompts when building test).
    max_visible_dup: cap on tasks sharing the same visible prompt within this split (0 = no cap)."""
    exclude_visible = exclude_visible if exclude_visible is not None else set()
    vis_count: dict = {}
    """Yield tasks round-robin over (domain, template). World seeds are disjoint per split."""
    base = SPLIT_SEED_BASE[split] + seed_offset
    domains = domains or list(DOMAIN_GENERATORS)
    skipped = {}
    for domain in domains:
        for ti, template in enumerate(sorted(DOMAIN_GENERATORS[domain])):
            if templates and template not in templates:
                continue
            made, k, last = 0, 0, ""
            while made < n_per_template and k < n_per_template * max_tries:
                seed = base + ti * 100_000 + k
                k += 1
                lang = "ko" if random.Random(f"lang|{seed}|{template}").random() < ko_ratio else "en"
                try:
                    task = generate_one(domain, template, seed, lang, split)
                    vh = visible_hash(task)
                    if vh in exclude_visible:
                        raise GenSkip("visible prompt seen in another split")
                    if max_visible_dup and vis_count.get(vh, 0) >= max_visible_dup:
                        raise GenSkip("visible prompt duplicate cap")
                    vis_count[vh] = vis_count.get(vh, 0) + 1
                    yield task
                    made += 1
                except GenSkip as e:
                    skipped[(domain, template)] = skipped.get((domain, template), 0) + 1
                    last = str(e)
            if made < n_per_template:
                print(f"[generate] WARN {domain}.{template}: only {made}/{n_per_template} ({last})")
    return skipped
