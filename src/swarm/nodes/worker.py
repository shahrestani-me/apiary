"""Worker node - one isolated coding agent, one git worktree.

Design note: this does NOT run a free-form tool-calling agent loop. Local
models under ~10B are unreliable at multi-turn tool use, and a runaway loop
costs you wall-clock time even when tokens are free. Instead:

    read the listed files -> emit complete replacement files (schema-forced)
    -> write them -> commit

Deterministic, bounded, and it works on a 2B model. Swap this node for a
real agentic worker (Claude Agent SDK, Codex SDK, `ollama` + a tool loop)
once the rest of the machinery is proven - the graph does not change.
"""

from __future__ import annotations

from pathlib import Path

from ..config import SETTINGS
from ..llm import structured, worker_llm
from ..state import SwarmState, TaskRecord, WorkerOutput
from ..worktree import GitError, commit_all, create_worktree, ensure_repo

SYSTEM = """You are a careful software engineer working alone on ONE task.

You will be given a goal and the current contents of the files you may edit.
Return the COMPLETE new contents of every file you change - never a diff,
never a fragment, never "... rest unchanged".

Only touch the files listed. If a file should be created, include it with
its full contents. Keep changes minimal and focused on the goal."""

MAX_FILE_CHARS = 20_000


def _read_context(root: Path, files: list[str]) -> str:
    chunks: list[str] = []
    for rel in files:
        fp = root / rel
        if fp.exists() and fp.is_file():
            text = fp.read_text(encoding="utf-8", errors="replace")
            truncated = text[:MAX_FILE_CHARS]
            suffix = "\n... [truncated]" if len(text) > MAX_FILE_CHARS else ""
            chunks.append(f"--- {rel} ---\n{truncated}{suffix}")
        else:
            chunks.append(f"--- {rel} --- (does not exist yet)")
    return "\n\n".join(chunks) if chunks else "(no files provided)"


def worker_node(state: SwarmState, config=None) -> dict:
    """Invoked once per task, in parallel, via Send()."""
    task: TaskRecord = state["task"]  # type: ignore[typeddict-item]
    task_id = task["id"]
    repo = ensure_repo(SETTINGS.repo_path)

    try:
        wt = create_worktree(
            repo, SETTINGS.worktree_root, task_id, base=state.get("base_branch", "HEAD")
        )
    except GitError as exc:
        return {
            "tasks": {task_id: {**task, "status": "failed", "last_error": f"worktree: {exc}"}},
            "events": [f"[{task_id}] worktree failed: {exc}"],
        }

    context = _read_context(wt.path, task.get("files", []))
    human = (
        f"Task goal:\n{task['goal']}\n\n"
        f"Files you may edit: {', '.join(task.get('files', [])) or '(none listed)'}\n\n"
        f"Current contents:\n{context}"
    )

    try:
        llm = structured(worker_llm(), WorkerOutput)
        out: WorkerOutput = llm.invoke([("system", SYSTEM), ("human", human)])
    except Exception as exc:  # noqa: BLE001 - local model failures are varied
        return {
            "tasks": {task_id: {**task, "status": "failed", "last_error": f"model: {exc}"}},
            "events": [f"[{task_id}] model error: {exc}"],
        }

    allowed = set(task.get("files", []))
    written: list[str] = []
    for edit in out.edits:
        # Guard rail: never let a worker escape its declared file set.
        if allowed and edit.path not in allowed:
            continue
        target = (wt.path / edit.path).resolve()
        if not str(target).startswith(str(wt.path)):
            continue  # path traversal attempt
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit.content, encoding="utf-8")
        written.append(edit.path)

    if not written:
        return {
            "tasks": {
                task_id: {
                    **task,
                    "status": "failed",
                    "attempts": task.get("attempts", 0) + 1,
                    "last_error": "worker produced no valid edits",
                    "worktree": str(wt.path),
                    "branch": wt.branch,
                }
            },
            "events": [f"[{task_id}] no edits produced"],
        }

    committed = commit_all(wt, f"swarm[{task_id}]: {task['goal'][:60]}")

    return {
        "tasks": {
            task_id: {
                **task,
                "status": "running",
                "attempts": task.get("attempts", 0) + 1,
                "worktree": str(wt.path),
                "branch": wt.branch,
                "notes": out.notes,
            }
        },
        "events": [
            f"[{task_id}] wrote {len(written)} file(s): {', '.join(written)}"
            + ("" if committed else " (nothing to commit)")
        ],
    }
