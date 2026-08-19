"""Integrator node - merge verified branches back, in dependency order."""

from __future__ import annotations

from typing import Any

from ..config import SETTINGS
from ..state import SwarmState, TaskRecord
from ..worktree import cleanup_all, ensure_repo, merge_branch


def _topo_order(tasks: dict[str, TaskRecord]) -> list[str]:
    """Simple Kahn ordering; cycles fall back to insertion order."""
    remaining = {tid: set(t.get("depends_on", [])) & set(tasks) for tid, t in tasks.items()}
    ordered: list[str] = []
    while remaining:
        ready = sorted([tid for tid, deps in remaining.items() if not deps - set(ordered)])
        if not ready:
            ordered.extend(sorted(remaining))
            break
        for tid in ready:
            ordered.append(tid)
            remaining.pop(tid)
    return ordered


def integrate_node(state: SwarmState) -> dict[str, Any]:
    repo = ensure_repo(SETTINGS.repo_path)
    tasks = state.get("tasks", {})
    verified = {tid: t for tid, t in tasks.items() if t.get("status") == "verified"}

    merged: list[str] = []
    events: list[str] = []

    for task_id in _topo_order(verified):
        branch = verified[task_id].get("branch")
        if not branch:
            continue
        ok, message = merge_branch(repo, branch)
        if ok:
            merged.append(branch)
            events.append(f"merged {branch}")
        else:
            events.append(f"MERGE CONFLICT on {branch} - left unmerged: {message.splitlines()[0]}")

    abandoned = [tid for tid, t in tasks.items() if t.get("status") == "abandoned"]
    outcome = (
        f"{len(merged)}/{len(tasks)} task(s) merged"
        + (f"; abandoned: {', '.join(abandoned)}" if abandoned else "")
    )

    if merged:
        cleanup_all(repo, SETTINGS.worktree_root)
        events.append("cleaned up worktrees")

    return {"merged_branches": merged, "outcome": outcome, "events": events}
