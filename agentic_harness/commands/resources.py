"""Inspect / validate registered skills and MCP servers (connects to every server and lists its tools).

  agentic-harness resources                                   # repo examples
  agentic-harness resources --skills-dir my_skills --mcp-config my_servers.json --servers kanban
"""
import argparse
import json
import sys
import tempfile
import time

from ..core.session import DEFAULT_MCP_CONFIG, DEFAULT_SKILLS_DIR, build_resources


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentic-harness resources")
    ap.add_argument("--skills-dir", action="append")
    ap.add_argument("--mcp-config", action="append")
    ap.add_argument("--servers", nargs="*", help="only these MCP servers (default: all registered)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = build_resources(skills_dirs=a.skills_dir or [DEFAULT_SKILLS_DIR], mcp_configs=a.mcp_config or [DEFAULT_MCP_CONFIG])
    report = {"skills": [], "mcp": []}
    for s in (res.skills.skills.values() if res.skills else []):
        report["skills"].append({"name": s.name, "description": s.description, "files": s.files, "hash": s.content_hash})
    for name in (a.servers or (res.mcp.names() if res.mcp else [])):
        d = tempfile.mkdtemp(prefix="ah-check-")
        t0 = time.time()
        try:
            ts = res.mcp.open([name], {"EPISODE_DIR": d, "TASK_SEED": a.seed, "TASK_ID": "resources-check"})[0]
            report["mcp"].append({"server": name, "ok": True, "startup_s": round(time.time() - t0, 2),
                                  "server_info": ts.client.server_info, "protocol": ts.client.protocol_version,
                                  "tools": [{"name": f["function"]["name"], "read_only": not ts.is_mutating(f["function"]["name"])}
                                            for f in ts.tool_schemas()]})
            ts.close()
        except Exception as e:
            report["mcp"].append({"server": name, "ok": False, "error": str(e)})
    res.close()
    if a.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(f"Skills ({len(report['skills'])}):")
    for s in report["skills"]:
        print(f"  - {s['name']}: {s['description'][:90]}{'...' if len(s['description']) > 90 else ''}  files={s['files']}")
    print(f"MCP servers ({len(report['mcp'])}):")
    for m in report["mcp"]:
        if m["ok"]:
            print(f"  - {m['server']}: OK {m['server_info']} protocol={m['protocol']} startup={m['startup_s']}s")
            for t in m["tools"]:
                print(f"      {t['name']}{'  (read-only)' if t['read_only'] else ''}")
        else:
            print(f"  - {m['server']}: FAILED {m['error']}")
    sys.exit(0 if all(m["ok"] for m in report["mcp"]) else 1)


if __name__ == "__main__":
    main()
