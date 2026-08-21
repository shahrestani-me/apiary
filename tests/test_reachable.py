"""Unit tests for the reachable plan (#293).

One property carries this file: **a task that can never become eligible again
must stop being counted as work the run can still do.** That is what lets the
ledger exhaust, which is the only door to the goal gate, which is the only thing
that can revive the failure that closed the door in the first place. Sixteen
recorded runs went 1,211 cycles without ever opening it.

The belief is a plain mapping here rather than a real `Belief`. Everything under
test is graph reachability over `depends_on`, and building a resolver to answer
`state()` would make the arithmetic depend on the one thing these tests are not
about - `orchestrator/derived.py` has its own suite for that.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from swarm.github.ledger import Ledger, LedgerEntry
from swarm.orchestrator.derived import BLOCKED, CLAIMED, ELIGIBLE, LANDED, NEEDS_HUMAN, REVIEW
from swarm.orchestrator.reachable import reasons, stranded


class Says:
    """The states a cycle believes, as a value a test can read at a glance."""

    def __init__(self, **states: str) -> None:
        # Written `task_2=NEEDS_HUMAN` and read `task-2`, because a task id is
        # kebab-case everywhere else and a keyword argument cannot be.
        self._states = {task_id.replace("_", "-"): state for task_id, state in states.items()}

    def state(self, task_id: str) -> str:
        # `""` is the default because that is production's default: `believe`
        # records a state only for a task the resolver returned a verdict about,
        # and a task waiting on an unmet dependency gets no verdict. A fake that
        # defaulted to `blocked` would have let the allow-list bug through.
        return self._states.get(task_id, "")


def task(number: int, *, depends_on: tuple[str, ...] = ()) -> LedgerEntry:
    return LedgerEntry(
        number=number,
        title=f"issue {number}",
        task_id=f"task-{number}",
        attempt=0,
        goal="do the thing",
        files=(f"src/mod{number}.py",),
        verify="python -m pytest -q",
        blocked_by=(),
        labels=frozenset(),
        depends_on=depends_on,
    )


def plan(*entries: LedgerEntry) -> Ledger:
    return Ledger(entries={item.task_id: item for item in entries})


# --------------------------------------------------------------------------
# What is stranded
# --------------------------------------------------------------------------


def test_a_dependent_of_a_failed_task_is_stranded():
    """The deadlock, at its smallest. #1 escalated; #2 waits on it.

    `readiness` discharges a dependency only when its issue is closed as
    completed, and an escalated issue stays *open* - so #2 is held at `blocked`
    for the rest of the run, and counting it as live is what kept `exhausted`
    false forever.
    """
    book = plan(task(1), task(2, depends_on=("task-1",)))

    assert stranded(book, Says(task_1=NEEDS_HUMAN)) == {"task-2"}


def test_a_task_two_hops_behind_a_failure_is_stranded_too():
    """The case that actually produced the deadlock in the wild.

    The task that dies is the bootstrap; the task that strands the run is the
    `integrate-*` one, which depends on the feature tasks and not on the
    bootstrap at all. A rule that only looked at direct dependents would leave
    it live and change nothing.
    """
    book = plan(
        task(1),
        task(2, depends_on=("task-1",)),
        task(3, depends_on=("task-2",)),
    )

    assert stranded(book, Says(task_1=NEEDS_HUMAN)) == {"task-2", "task-3"}


def test_a_dependency_that_landed_does_not_rescue_a_dependency_that_failed():
    """One unmet dependency is enough. A task waiting on #1 and #2 cannot run
    when #2 is dead, however finished #1 is."""
    book = plan(
        task(1),
        task(2),
        task(3, depends_on=("task-1", "task-2")),
    )

    assert stranded(book, Says(task_1=LANDED, task_2=NEEDS_HUMAN)) == {"task-3"}


def test_nothing_is_stranded_while_nothing_has_failed():
    book = plan(task(1), task(2, depends_on=("task-1",)))

    assert stranded(book, Says(task_1=CLAIMED)) == frozenset()


def test_an_empty_ledger_strands_nothing():
    """A cycle that resolved nothing must not read as a finished plan."""
    assert stranded(plan(), Says()) == frozenset()


# --------------------------------------------------------------------------
# What is never stranded, and why that matters more than the rest
# --------------------------------------------------------------------------


@pytest.mark.parametrize("state", [CLAIMED, REVIEW])
def test_work_in_flight_is_never_stranded(state: str):
    """A live container or an open pull request is real work in progress.

    Counting it as unreachable would let the run end - and the reaper dispose a
    worker - under a container that was mid-edit. `replan.py` refuses to abandon
    these two states for the same reason and puts it more strongly: that call
    belongs to a human.
    """
    book = plan(task(1), task(2, depends_on=("task-1",)))

    assert stranded(book, Says(task_1=NEEDS_HUMAN, task_2=state)) == frozenset()


def test_a_task_that_already_landed_is_not_reported():
    """Not because it is reachable, but because it is *finished*: the live count
    already excludes it, and naming it here would put a landed task in a human's
    report about what is stuck."""
    book = plan(task(1), task(2, depends_on=("task-1",)))

    assert stranded(book, Says(task_1=NEEDS_HUMAN, task_2=LANDED)) == frozenset()


def test_the_failed_task_itself_is_not_stranded():
    """It is the failure, not something behind it. `needs-human` is already
    terminal, so the live count excludes it without any help from here."""
    book = plan(task(1), task(2, depends_on=("task-1",)))

    assert "task-1" not in stranded(book, Says(task_1=NEEDS_HUMAN))


@pytest.mark.parametrize("state", ["", BLOCKED, ELIGIBLE])
def test_every_waiting_state_behind_a_failure_is_stranded(state: str):
    """Including `""`, which is the only one that occurs in practice.

    `believe` records a state for a task the resolver had a verdict about, and
    a task held behind an unmet dependency gets none - so the deadlock presents
    as `""`, not as `blocked`. The first version of this rule was an allow-list
    of `{blocked, eligible}`: correct-reading, fully unit-tested, and a complete
    no-op on every real run. `NEVER_STRANDED` is written as an exclusion so the
    state nobody anticipated is the one that strands.

    `blocked` and `eligible` are parametrised alongside it because the resolver
    and readiness can disagree for a cycle - a dependency closed as
    `not_planned` is one live way - and the answer must not depend on which.
    """
    book = plan(task(1), task(2, depends_on=("task-1",)))

    assert stranded(book, Says(task_1=NEEDS_HUMAN, task_2=state)) == {"task-2"}


def test_a_revival_unstrands_everything_behind_it():
    """Recomputed from scratch every cycle, never folded forward.

    The goal gate's whole purpose is to turn a `needs-human` task back into
    work, and a fold would have to remember to un-strand the chain behind it.
    Same ledger, one different belief, and the answer inverts.
    """
    book = plan(task(1), task(2, depends_on=("task-1",)))

    assert stranded(book, Says(task_1=NEEDS_HUMAN)) == {"task-2"}
    assert stranded(book, Says(task_1=ELIGIBLE)) == frozenset()


def test_a_dependency_naming_nothing_is_not_this_modules_problem():
    """A `## Blocked by` ref that resolves to no task is a contract error the
    ledger already reports. Reachability must not raise on it."""
    book = plan(task(2, depends_on=("task-does-not-exist",)))

    assert stranded(book, Says()) == frozenset()


def test_a_dependency_ring_terminates():
    """`readiness` aborts the cycle on a ring (`DependencyCycleError`), so this
    is unreachable in production - but a reachability walk that could loop
    forever on malformed input is a hang, not a failed assertion."""
    book = plan(
        task(1, depends_on=("task-3",)),
        task(2, depends_on=("task-1",)),
        task(3, depends_on=("task-2",)),
    )

    assert stranded(book, Says(task_1=NEEDS_HUMAN)) == {"task-2", "task-3"}


# --------------------------------------------------------------------------
# The sentence a human reads
# --------------------------------------------------------------------------


def test_the_reason_names_the_nearest_dead_dependency():
    book = plan(task(1), task(2, depends_on=("task-1",)))

    said: Mapping[str, str] = reasons(book, Says(task_1=NEEDS_HUMAN), ["task-2"])

    assert said["task-2"] == "waiting on #1 (task-1), which needs a human"


def test_a_task_further_down_the_chain_says_so_rather_than_naming_nothing():
    """Its own dependencies are all alive; what is dead is further up. The
    alternative was the transitive closure, which is a paragraph nobody
    finishes."""
    book = plan(
        task(1),
        task(2, depends_on=("task-1",)),
        task(3, depends_on=("task-2",)),
    )

    said = reasons(book, Says(task_1=NEEDS_HUMAN), ["task-3"])

    assert said["task-3"] == "waiting on a task that is itself stranded behind a failure"
