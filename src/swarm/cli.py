"""CLI entrypoint.

    swarm "add retry with exponential backoff to the http client"

Durable by default: state is checkpointed to sqlite after every node, so a
crashed or Ctrl-C'd run resumes with --resume <thread-id> instead of
restarting from zero. That is the main reason this is a LangGraph app and
not a for-loop.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from .config import SETTINGS
from .graph import build_graph


def _checkpointer():
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        from langgraph.checkpoint.memory import InMemorySaver

        print("! langgraph-checkpoint-sqlite not installed - using in-memory "
              "checkpoints (no resume across processes)", file=sys.stderr)
        return None, InMemorySaver()

    db = Path(SETTINGS.repo_path) / SETTINGS.checkpoint_db
    db.parent.mkdir(parents=True, exist_ok=True)
    ctx = SqliteSaver.from_conn_string(str(db))
    return ctx, ctx.__enter__()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swarm")
    parser.add_argument("objective", nargs="?", help="what the swarm should accomplish")
    parser.add_argument("--repo", default=None, help="target git repo (default: $SWARM_REPO or cwd)")
    parser.add_argument("--resume", default=None, help="thread id of a previous run")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)

    if not args.objective and not args.resume:
        parser.error("give an objective, or --resume <thread-id>")

    if args.repo:
        import os

        os.environ["SWARM_REPO"] = args.repo

    ctx, saver = _checkpointer()
    graph = build_graph(checkpointer=saver)

    thread_id = args.resume or str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    print(f"» thread {thread_id}  (resume with: swarm --resume {thread_id})\n")

    payload = None if args.resume else {"objective": args.objective}
    seen = 0
    final = {}
    try:
        for chunk in graph.stream(payload, config=config, stream_mode="values"):
            final = chunk
            events = chunk.get("events", [])
            for line in events[seen:]:
                print(f"  · {line}")
            seen = len(events)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    print()
    print(f"» {final.get('outcome', 'no outcome recorded')}")
    for branch in final.get("merged_branches", []):
        print(f"  merged: {branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
