"""The scheduler: the half of the swarm that decides, rather than does.

`docs/architecture-v2.md` describes one cycle - read GitHub, compute readiness,
dispatch, observe, judge - and this package is where the steps that are neither
a GitHub call nor a Docker call live. It reaches the control plane through
`swarm.github` and the execution plane through `swarm.containers`, and it is
the only place that reads from both: a module that talked to GitHub *and*
Docker anywhere else would be a second dispatcher, with its own idea of the
cap and its own answer to "is that file already being edited".

`dispatcher.py` (#21) is the first tenant. The reconciler (#22, #23) and the
recovery sweep (#35) join it and read the same shapes.

`goal.py` is the last step of a run rather than a step of a cycle: it asks
whether the objective the run was given is actually met once the plan has
nothing live left in it, and appends issues when it is not. It belongs here
for the same reason everything else does - it reads the ledger and writes the
tracker, and a module elsewhere doing that would be a second scheduler.
"""

from __future__ import annotations

from .dispatcher import (
    CLAIMED,
    MIN_WORKERS,
    ORCHESTRATOR_SLOTS,
    REVIEW,
    Capacity,
    Deferred,
    DispatchFailure,
    DispatchPlan,
    DispatchReport,
    Dispatched,
    Labeller,
    Spawner,
    claim,
    container_memory_cap,
    dispatch,
    held_files,
    normalise,
    plan_dispatch,
)

__all__ = [
    "CLAIMED",
    "MIN_WORKERS",
    "ORCHESTRATOR_SLOTS",
    "REVIEW",
    "Capacity",
    "Deferred",
    "DispatchFailure",
    "DispatchPlan",
    "DispatchReport",
    "Dispatched",
    "Labeller",
    "Spawner",
    "claim",
    "container_memory_cap",
    "dispatch",
    "held_files",
    "normalise",
    "plan_dispatch",
]
