"""The worker: one container, one issue, one verified commit.

`docs/architecture-v2.md` replaces v1's git worktrees with containers, so this
package is the inside of one. It is the only part of the system that executes
code it did not write, and the only part that holds a token that can push -
which is why the pieces are split the way they are:

- `edit` reads the files a contract declares, asks the worker model for
  whole-file replacements, and writes back **only** the paths the contract
  authorised. Every guard rail lives there.
- `entrypoint` is the process: clone, contract, edit, verify, commit. It owns
  the exit codes the orchestrator reads (`docs/issue-contract.md` §4).
- `pr` (#17) pushes the branch and opens the pull request.

Nothing here imports the LangGraph orchestrator, and nothing in the
orchestrator imports this: the two halves meet through GitHub and a container
exit code, and that seam is what makes a worker disposable.
"""

from __future__ import annotations

from .edit import EditError, apply_edits, gather_context, propose_edits, read_writable
from .entrypoint import (
    EXIT_INFRASTRUCTURE,
    EXIT_OK,
    EXIT_TASK_FAILED,
    InfrastructureError,
    WorkerResult,
    main,
    run_worker,
)

__all__ = [
    "EXIT_INFRASTRUCTURE",
    "EXIT_OK",
    "EXIT_TASK_FAILED",
    "EditError",
    "InfrastructureError",
    "WorkerResult",
    "apply_edits",
    "gather_context",
    "main",
    "propose_edits",
    "read_writable",
    "run_worker",
]
