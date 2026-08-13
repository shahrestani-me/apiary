"""Verifier node - the quality gate.

Generation is cheap; verification is the bottleneck. This node is the reason
the system produces trustworthy output, so it runs a real command in a real
checkout and believes only the exit code. No LLM judges code here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import SETTINGS
from ..state import SwarmState, TaskRecord


def _run_verify(cwd: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            SETTINGS.verify_command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=SETTINGS.verify_timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"verify timed out after {SETTINGS.verify_timeout_s}s"
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, output[-4000:]


def verify_node(state: SwarmState) -> dict:
    """Verify every task currently in 'running'."""
    tasks = state.get("tasks", {})
    updates: dict[str, TaskRecord] = {}
    events: list[str] = []
    verified = state.get("verified_count", 0)

    for task_id, task in tasks.items():
        if task.get("status") != "running":
            continue
        wt = task.get("worktree")
        if not wt or not Path(wt).exists():
            updates[task_id] = {**task, "status": "failed", "last_error": "worktree missing"}
            continue

        ok, output = _run_verify(Path(wt))
        if ok:
            updates[task_id] = {**task, "status": "verified", "last_error": ""}
            verified += 1
            events.append(f"[{task_id}] PASS")
        else:
            exhausted = task.get("attempts", 0) >= SETTINGS.max_attempts_per_task
            updates[task_id] = {
                **task,
                "status": "abandoned" if exhausted else "failed",
                "last_error": output,
            }
            events.append(
                f"[{task_id}] FAIL{' (attempts exhausted)' if exhausted else ''}: "
                f"{output.splitlines()[-1] if output else 'no output'}"
            )

    return {"tasks": updates, "verified_count": verified, "events": events}
