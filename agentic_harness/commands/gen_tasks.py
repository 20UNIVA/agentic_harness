"""Generate task splits + run validation gates (gold must pass, null must fail outside `policy`, no cross-split hashes).

  agentic-harness gen-tasks --out data/tasks_v0.2 --train 150 --dev 4 --test 10
(counts are per template; there are 80 templates over 5 environments + the example MCP servers)
"""
import argparse
import collections
import importlib.util
import json
import os
import time

from ..core.runner import run_episodes
from ..core.types import HARNESS_VERSION
from ..policies.oracle import NullPolicy, OraclePolicy
from ..tasks import DOMAIN_GENERATORS, generate
from ..tasks.common import visible_hash


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentic-harness gen-tasks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--dev", type=int, default=4)
    ap.add_argument("--test", type=int, default=10)
    ap.add_argument("--ko-ratio", type=float, default=0.3)
    ap.add_argument("--domains", nargs="*")
    ap.add_argument("--train-max-dup", type=int, default=3, help="max train tasks sharing one visible prompt")
    ap.add_argument("--train-mcp", type=int, default=None, help="per-template train count for the mcp_examples group "
                    "(default: same as --train; each MCP task starts server processes during the gate)")
    ap.add_argument("--gate-mcp-concurrency", type=int, default=8)
    args = ap.parse_args(argv)
    if "mcp_examples" in (args.domains or DOMAIN_GENERATORS) and not DOMAIN_GENERATORS["mcp_examples"]:
        ap.error("the mcp_examples templates need the repo's examples/ folder (run from a source checkout), "
                 "or pass --domains without mcp_examples")
    if "mcp_examples" in (args.domains or DOMAIN_GENERATORS) and importlib.util.find_spec("mcp") is None:
        ap.error("the mcp_examples tasks run the example MCP servers, which need the `mcp` SDK: "
                 "pip install -e \".[mcp-examples]\", or pass --domains without mcp_examples")
    os.makedirs(args.out, exist_ok=True)
    manifest = {"harness_version": HARNESS_VERSION, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "ko_ratio": args.ko_ratio,
                "templates": {d: sorted(g) for d, g in DOMAIN_GENERATORS.items()}, "splits": {}}
    hashes = {}
    seen_visible = set()      # evaluation splits are generated first; later splits exclude their visible prompts (test -> dev -> train)
    for split, n in (("test", args.test), ("dev", args.dev), ("train", args.train)):
        if n <= 0:
            continue
        t0 = time.time()
        kw = dict(ko_ratio=args.ko_ratio, exclude_visible=set(seen_visible),
                  max_visible_dup=args.train_max_dup if split == "train" else 1)
        doms = args.domains or list(DOMAIN_GENERATORS)
        tasks = list(generate(split, n, domains=[d for d in doms if d != "mcp_examples"], **kw))
        if "mcp_examples" in doms:
            n_mcp = args.train_mcp if (split == "train" and args.train_mcp is not None) else n
            tasks += list(generate(split, n_mcp, domains=["mcp_examples"], **kw))
        seen_visible |= {visible_hash(t) for t in tasks}
        # dedupe within split by content hash
        seen, uniq = set(), []
        for t in tasks:
            if t["content_hash"] not in seen:
                seen.add(t["content_hash"])
                uniq.append(t)
        def gate(policy, tasks=uniq):
            plain = [t for t in tasks if not t.get("mcp_servers")]
            withm = [t for t in tasks if t.get("mcp_servers")]
            return (run_episodes(plain, policy, concurrency=256, log_every=0) +
                    run_episodes(withm, policy, concurrency=args.gate_mcp_concurrency, log_every=0))
        gold = gate(OraclePolicy())
        gold_fail = [t["task_id"] for t in gold if t["outcome"]["reward"] != 1.0]
        infra = [t["task_id"] for t in gold if t["outcome"]["reward"] is None]
        assert not infra, f"infra failures during gate (MCP servers?): {infra[:5]}"
        null = gate(NullPolicy())
        null_pass = [t["task_id"] for t in null if t["outcome"]["reward"] == 1.0 and t["category"] != "policy"]
        assert not gold_fail, f"gold failed: {gold_fail[:5]}"
        assert not null_pass, f"null passed: {null_pass[:5]}"
        path = os.path.join(args.out, f"{split}.jsonl")
        with open(path, "w") as f:
            for t in uniq:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        hashes[split] = seen
        manifest["splits"][split] = {
            "path": path, "n": len(uniq), "dupes_removed": len(tasks) - len(uniq),
            "by_domain": dict(collections.Counter(t["domain"] for t in uniq)),
            "by_category": dict(collections.Counter(t["category"] for t in uniq)),
            "by_lang": dict(collections.Counter(t["lang"] for t in uniq)),
            "unique_visible_prompts": len({visible_hash(t) for t in uniq}),
            "gate_gold_pass": f"{len(uniq) - len(gold_fail)}/{len(uniq)}",
            "gate_null_pass_policy_only": sum(1 for t in null if t["outcome"]["reward"] == 1.0),
            "seconds": round(time.time() - t0, 1)}
        print(split, manifest["splits"][split], flush=True)
    names = list(hashes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = hashes[names[i]] & hashes[names[j]]
            assert not inter, f"content_hash overlap {names[i]}/{names[j]}: {len(inter)}"
    manifest["cross_split_overlap"] = 0
    manifest["cross_split_visible_overlap"] = "0 by construction (generated test -> dev -> train, each excluding earlier visible prompts)"
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
