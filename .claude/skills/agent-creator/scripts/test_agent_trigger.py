#!/usr/bin/env python3
"""Trigger test for a Claude Code subagent definition.

Given an agent name (the file `.claude/agents/<name>.md` must exist) and a
list of test queries, spawn `claude -p` for each query and watch the
stream-json output to see whether the main session decides to spawn the
target agent via the `Task` tool with a matching `subagent_type`.

Outputs a JSON report to stdout with per-query trigger rates and a summary.

Usage:
    python -m scripts.test_agent_trigger \\
        --agent-name pr-summarizer \\
        --queries-file /tmp/queries.json \\
        --runs-per-query 3

Queries file format:
    [
      {"query": "...", "should_trigger": true},
      {"query": "...", "should_trigger": false}
    ]

The script intentionally does not require any project-specific dependencies
(no Anthropic SDK, no third-party libraries). It speaks to `claude -p` via
subprocess and parses the stream-json output line by line.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def find_project_root() -> Path:
    """Walk up from cwd looking for a directory containing `.claude/`."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".claude").is_dir():
            return parent
    return current


def parse_agent_md(agent_path: Path) -> tuple[str, str]:
    """Return (name, description) parsed from an agent markdown frontmatter."""
    text = agent_path.read_text()
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        raise ValueError(f"No YAML frontmatter found in {agent_path}")
    front = match.group(1)

    name_match = re.search(r"^name:\s*(.+)$", front, re.MULTILINE)
    desc_match = re.search(r"^description:\s*(.+(?:\n  .+)*)", front, re.MULTILINE)

    if not name_match or not desc_match:
        raise ValueError(f"Missing `name` or `description` in frontmatter of {agent_path}")

    name = name_match.group(1).strip()
    # Multi-line YAML descriptions get indented continuation lines; flatten them.
    description = re.sub(r"\n\s+", " ", desc_match.group(1)).strip()
    return name, description


def run_single_query(
    query: str,
    agent_name: str,
    timeout: int,
    project_root: str,
    model: str | None = None,
) -> bool:
    """Run a single query against `claude -p` and return whether the agent was spawned.

    Detection: looks for a `Task` tool call in the assistant's stream output
    whose `subagent_type` input matches `agent_name`. Uses partial-message
    streaming so we can return as soon as the spawn decision is visible,
    without waiting for the agent's actual execution.
    """
    cmd = [
        "claude",
        "-p", query,
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
    ]
    if model:
        cmd.extend(["--model", model])

    # Drop CLAUDECODE so we can nest claude -p inside an interactive session.
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=project_root,
        env=env,
    )

    triggered = False
    start_time = time.time()
    buffer = ""
    pending_tool_name: str | None = None
    accumulated_json = ""

    try:
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                remaining = process.stdout.read()
                if remaining:
                    buffer += remaining.decode("utf-8", errors="replace")
                break

            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            if not ready:
                continue

            chunk = os.read(process.stdout.fileno(), 8192)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Stream events: detect Task tool spawn early.
                if event.get("type") == "stream_event":
                    se = event.get("event", {})
                    se_type = se.get("type", "")

                    if se_type == "content_block_start":
                        cb = se.get("content_block", {})
                        if cb.get("type") == "tool_use":
                            pending_tool_name = cb.get("name", "")
                            accumulated_json = ""

                    elif se_type == "content_block_delta" and pending_tool_name == "Task":
                        delta = se.get("delta", {})
                        if delta.get("type") == "input_json_delta":
                            accumulated_json += delta.get("partial_json", "")
                            # Parse opportunistically; subagent_type may arrive in pieces.
                            if f'"subagent_type"' in accumulated_json and agent_name in accumulated_json:
                                # Crude but effective: subagent_type appears near the agent name.
                                # We confirm by checking the field-value pair shape.
                                if re.search(
                                    r'"subagent_type"\s*:\s*"' + re.escape(agent_name) + r'"',
                                    accumulated_json,
                                ):
                                    return True

                    elif se_type in ("content_block_stop", "message_stop"):
                        if pending_tool_name == "Task" and accumulated_json:
                            if re.search(
                                r'"subagent_type"\s*:\s*"' + re.escape(agent_name) + r'"',
                                accumulated_json,
                            ):
                                return True
                        pending_tool_name = None
                        accumulated_json = ""
                        if se_type == "message_stop":
                            return triggered

                # Fallback: full assistant message.
                elif event.get("type") == "assistant":
                    message = event.get("message", {})
                    for content_item in message.get("content", []):
                        if content_item.get("type") != "tool_use":
                            continue
                        if content_item.get("name") != "Task":
                            continue
                        tool_input = content_item.get("input", {})
                        if tool_input.get("subagent_type") == agent_name:
                            return True
                    return triggered

                elif event.get("type") == "result":
                    return triggered
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    return triggered


def run_eval(
    queries: list[dict],
    agent_name: str,
    runs_per_query: int,
    num_workers: int,
    timeout: int,
    project_root: Path,
    trigger_threshold: float,
    model: str | None,
) -> dict:
    """Run all queries N times in parallel, return per-query stats + summary."""
    results: list[dict] = []
    query_triggers: dict[str, list[bool]] = {}
    query_meta: dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for item in queries:
            for _ in range(runs_per_query):
                fut = executor.submit(
                    run_single_query,
                    item["query"],
                    agent_name,
                    timeout,
                    str(project_root),
                    model,
                )
                futures[fut] = item

        for fut in as_completed(futures):
            item = futures[fut]
            q = item["query"]
            query_meta[q] = item
            query_triggers.setdefault(q, [])
            try:
                query_triggers[q].append(fut.result())
            except Exception as exc:
                print(f"warning: query failed: {exc}", file=sys.stderr)
                query_triggers[q].append(False)

    for q, triggers in query_triggers.items():
        item = query_meta[q]
        rate = sum(triggers) / len(triggers)
        should = item["should_trigger"]
        passed = (rate >= trigger_threshold) if should else (rate < trigger_threshold)
        results.append({
            "query": q,
            "should_trigger": should,
            "trigger_rate": rate,
            "triggers": sum(triggers),
            "runs": len(triggers),
            "pass": passed,
        })

    passed_count = sum(1 for r in results if r["pass"])
    return {
        "agent_name": agent_name,
        "results": results,
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Trigger test for a Claude Code subagent")
    p.add_argument("--agent-name", required=True,
                   help="Name of the agent (matches `name:` in frontmatter and filename without .md)")
    p.add_argument("--queries-file", required=True,
                   help="Path to a JSON file containing [{query, should_trigger}, ...]")
    p.add_argument("--agents-dir", default=None,
                   help="Override agents directory (default: <project_root>/.claude/agents)")
    p.add_argument("--runs-per-query", type=int, default=3,
                   help="How many times to run each query (default: 3)")
    p.add_argument("--num-workers", type=int, default=5,
                   help="Parallel workers (default: 5)")
    p.add_argument("--timeout", type=int, default=60,
                   help="Per-query timeout in seconds (default: 60)")
    p.add_argument("--trigger-threshold", type=float, default=0.5,
                   help="Trigger rate threshold for pass/fail (default: 0.5)")
    p.add_argument("--model", default=None,
                   help="Override model for `claude -p` (e.g. claude-sonnet-4-6)")
    p.add_argument("--verbose", action="store_true", help="Print per-query results to stderr")
    args = p.parse_args()

    project_root = find_project_root()
    agents_dir = Path(args.agents_dir) if args.agents_dir else project_root / ".claude" / "agents"
    agent_path = agents_dir / f"{args.agent_name}.md"
    if not agent_path.exists():
        print(f"error: agent file not found: {agent_path}", file=sys.stderr)
        sys.exit(1)

    name, description = parse_agent_md(agent_path)
    if name != args.agent_name:
        print(f"warning: filename `{args.agent_name}.md` does not match frontmatter name `{name}`",
              file=sys.stderr)

    queries = json.loads(Path(args.queries_file).read_text())
    if not isinstance(queries, list) or not queries:
        print("error: queries file must contain a non-empty JSON array", file=sys.stderr)
        sys.exit(1)
    for item in queries:
        if "query" not in item or "should_trigger" not in item:
            print("error: each query item needs `query` and `should_trigger` fields", file=sys.stderr)
            sys.exit(1)

    if args.verbose:
        print(f"agent: {name}", file=sys.stderr)
        print(f"description: {description[:120]}...", file=sys.stderr)
        print(f"queries: {len(queries)} x {args.runs_per_query} runs", file=sys.stderr)

    output = run_eval(
        queries=queries,
        agent_name=name,
        runs_per_query=args.runs_per_query,
        num_workers=args.num_workers,
        timeout=args.timeout,
        project_root=project_root,
        trigger_threshold=args.trigger_threshold,
        model=args.model,
    )

    if args.verbose:
        s = output["summary"]
        print(f"results: {s['passed']}/{s['total']} passed", file=sys.stderr)
        for r in output["results"]:
            status = "PASS" if r["pass"] else "FAIL"
            print(f"  [{status}] rate={r['triggers']}/{r['runs']} expected={r['should_trigger']}: "
                  f"{r['query'][:70]}", file=sys.stderr)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
