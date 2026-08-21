"""Which tasks the run can still reach, and which are stranded behind a failure.

**The deadlock this exists to break.** `CycleReport.exhausted` is `live == 0`,
and `live` counts every task whose state is not `landed` or `needs-human`. A
task escalated to `needs-human` is terminal, so it stops being counted - but its
*issue stays open*, and `readiness` only discharges a dependency whose issue is
closed as completed. So every task behind the failure is held at `blocked`
forever, `blocked` is not terminal, and the ledger can never exhaust.

That matters because `exhausted` is the only door to `goal.close_the_loop` - the
one piece of the system that can revive a failed task or extend the plan, which
is to say the only thing that could rescue the very task that closed the door.
Measured over sixteen recorded runs: 1,211 cycles, **zero** goal-gate calls, and
every run ending with live work outstanding. The replan is unreachable for a
second reason (`judge.Verdict.should_replan`), but this is the one that makes a
single escalation permanent.

**Stranded is not a state; it is arithmetic over the plan.** ADR 0001's claim is
that apiary's workflow is derived rather than stored, and a sixth stored state
for "waiting on something that will never finish" would be a cache with no
reader and one more thing to keep in step. What a task *is* stays `blocked` -
that is true and readable, and it is what the board should show. What changes is
whether the run counts it as work it can still do, and the answer is no: nothing
inside this run can make it eligible.

**In-flight work is never stranded, whatever its dependencies say.** A task with
a live container or an open pull request is real work in progress, and counting
it as unreachable would let a run end - and the reaper dispose a worker - under a
container that was mid-edit. `replan.py` refuses to abandon `claimed` and
`review` tasks for the same reason and states it more strongly: that decision
belongs to a human. `NEVER_STRANDED` is that rule, and it is written as an
exclusion for a reason the constant explains at length: the allow-list version
of it is a no-op in production.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping

from .derived import CLAIMED, LANDED, NEEDS_HUMAN, REVIEW

if TYPE_CHECKING:  # pragma: no cover - import cycle, and only for the annotation
    from ..github.ledger import Ledger

__all__ = ["NEVER_STRANDED", "stranded"]

#: The states that hold a task open however dead its dependencies are, written
#: as the exclusion rather than as the set of strandable states. **That
#: direction is load-bearing and the first draft got it backwards.**
#:
#: An allow-list of `{blocked, eligible}` reads correctly and does nothing: a
#: task waiting on an unmet dependency is not believed `blocked`, it is believed
#: `""`. `authority.believe` only records a state for a task the resolver
#: returned a verdict about, and `Belief.state` answers `""` for the rest -
#: which the live count deliberately treats as live ("a cycle that resolved
#: nothing must not read as a finished plan"). So the one state the production
#: deadlock actually presents in was outside the allow-list, and the fix was a
#: no-op everywhere except its own unit tests.
#:
#: Excluding instead means a state this module has never heard of strands, which
#: is the safe direction here: the cost of wrongly stranding is a run that ends
#: early and reports why, and the cost of wrongly holding is the deadlock.
#:
#: `claimed` and `review` are the two that matter. A live container or an open
#: pull request is work in progress, and ending a run under it would let the
#: reaper dispose a worker mid-edit - `replan.py` refuses to abandon these two
#: for the same reason and puts it more strongly: that call belongs to a human.
#: `""` is safe to strand despite meaning "no verdict", because the resolver
#: reads containers and pull requests directly: a task it saw a container for is
#: `claimed`, so an unresolved task is one nothing is running.
NEVER_STRANDED = frozenset({CLAIMED, REVIEW, LANDED, NEEDS_HUMAN})


def stranded(ledger: Ledger, belief: Any) -> frozenset[str]:
    """Task ids that can never become eligible again in this run.

    Pure, and recomputed from scratch every cycle rather than folded forward:
    the goal gate's revival turns a `needs-human` task back into work, and a
    fold would have to remember to un-strand everything behind it. Recomputing
    is O(edges) on a plan of a dozen tasks, and it cannot go stale.

    Walks *forward from* the failures over the reverse dependency edges, so a
    task two hops behind an escalation is stranded as surely as its immediate
    dependent - which is the case that produced the deadlock in the wild:
    `integrate-*` tasks depend on the feature tasks, not on the bootstrap that
    actually died.
    """
    entries = ledger.entries
    if not entries:
        return frozenset()

    # dependency -> the tasks waiting on it. Built from `depends_on`, which
    # `load_ledger` has already resolved from `## Blocked by` refs to task ids,
    # so an unresolvable ref is absent here rather than a `KeyError`: a task
    # blocked on a ref that names nothing is a contract error, reported by the
    # ledger, and not this module's to re-diagnose.
    waiters: dict[str, list[str]] = {}
    for task_id, entry in entries.items():
        for dependency in entry.depends_on:
            waiters.setdefault(dependency, []).append(task_id)

    frontier = [
        task_id
        for task_id, entry in entries.items()
        if belief.state(entry.task_id) == NEEDS_HUMAN
    ]
    found: set[str] = set()
    while frontier:
        blocker = frontier.pop()
        for waiter in waiters.get(blocker, ()):
            if waiter in found:
                continue
            found.add(waiter)
            # Pushed whatever its own state is, because reachability is a
            # property of the *chain*: a claimed task is not stranded itself,
            # but the tasks waiting on it still cannot start if the thing it is
            # waiting on is dead.
            frontier.append(waiter)

    return frozenset(
        task_id
        for task_id in found
        if belief.state(entries[task_id].task_id) not in NEVER_STRANDED
    )


def reasons(ledger: Ledger, belief: Any, ids: Iterable[str]) -> Mapping[str, str]:
    """One sentence per stranded task, naming the dependency that stranded it.

    For the report a human reads, not for any decision. Names the *nearest*
    dead dependency rather than the whole chain: "blocked on #4, which needs a
    human" is actionable, and the transitive closure of that is a paragraph
    nobody finishes.
    """
    entries = ledger.entries
    out: dict[str, str] = {}
    for task_id in ids:
        entry = entries.get(task_id)
        if entry is None:
            continue
        dead = [
            dependency
            for dependency in entry.depends_on
            if dependency in entries
            and belief.state(entries[dependency].task_id) == NEEDS_HUMAN
        ]
        if dead:
            named = ", ".join(f"{entries[d].ref} ({d})" for d in dead)
            out[task_id] = f"waiting on {named}, which needs a human"
        else:
            out[task_id] = "waiting on a task that is itself stranded behind a failure"
    return out
