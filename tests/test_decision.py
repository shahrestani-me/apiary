"""Unit tests for the decision report (#293).

The property: **a run that stopped because it needs a human says so, names the
tasks, and prints the command.** Before this, an escalation produced one line in
`events.jsonl` and a comment on the issue, and the run's own last word was the
same whether it was waiting for a decision or had merely hit its cycle cap.

Reports are namespaces rather than real `CycleReport`s. Everything here is a fold
over `escalated`, `ledger` and `belief`, and building three real cycles to
produce them would make these tests fail for reasons that belong to
`test_reconcile.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from swarm.github.ledger import Ledger, LedgerEntry
from swarm.github.refs import task_ref
from swarm.orchestrator.decision import UNCLASSIFIED, classify, decisions
from swarm.orchestrator.derived import NEEDS_HUMAN

REPO = "kamyar-finlex/fantasy-bestiary-2"

# The reasons the real rules write, quoted rather than paraphrased: this table is
# the contract between `decision.CAUSES` and the modules that escalate, and a
# paraphrase here would let the two drift without a test noticing.
INFRASTRUCTURE = (
    "3 consecutive infrastructure failures, most recently 'model call failed: "
    "ReadTimeoutError'; this is not a coding problem and no attempt was ever "
    "consumed for it (APIARY_MAX_INFRASTRUCTURE to change)"
)
GATE_RED = (
    "worker exit 1: the verify command failed; 3 attempt(s) made, the last 3 "
    "failing the same way against a cap of 3"
)
FOREIGN = (
    "CI failed in tests/test_other.py, which is outside this issue's ## Files - "
    "no attempt of this issue could fix it"
)
NO_CHECKS = (
    "no check run was ever created for #5 in 900s - nothing gated this pull "
    "request, so nothing verified it"
)


def escalation(number: int, reason: str, task_id: str | None = None) -> Any:
    return SimpleNamespace(
        ref=task_ref(number),
        task_id=task_id or f"task-{number}",
        reason=reason,
        to_state=NEEDS_HUMAN,
    )


def task(number: int, *, depends_on: tuple[str, ...] = ()) -> LedgerEntry:
    return LedgerEntry(
        number=number,
        title=f"issue {number}",
        task_id=f"task-{number}",
        attempt=0,
        goal="do the thing",
        files=(f"src/mod{number}.py",),
        verify="python3 -m pytest -q",
        blocked_by=(),
        labels=frozenset(),
        depends_on=depends_on,
    )


class Says:
    def __init__(self, **states: str) -> None:
        self._states = {k.replace("_", "-"): v for k, v in states.items()}

    def state(self, task_id: str) -> str:
        return self._states.get(task_id, "")


def cycle(*escalated: Any, entries: tuple[LedgerEntry, ...] = (), belief: Any = None) -> Any:
    return SimpleNamespace(
        escalated=escalated,
        ledger=Ledger(entries={item.task_id: item for item in entries}),
        belief=belief if belief is not None else Says(),
    )


# --------------------------------------------------------------------------
# Nothing to decide
# --------------------------------------------------------------------------


def test_a_run_with_no_escalations_says_nothing():
    """Printed unconditionally by the caller, so silence has to be the default -
    a clean run must not grow a section explaining that it is clean."""
    report = decisions([cycle()], repo=REPO)

    assert not report
    assert report.text() == ""


def test_a_run_with_no_cycles_at_all_says_nothing():
    """`--max-cycles 0` is neither met nor failed, and has nothing to report."""
    assert decisions([], repo=REPO).text() == ""


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason, expected_fragment",
    [
        (INFRASTRUCTURE, "the machine, not the code"),
        (GATE_RED, "the gate went red the same way"),
        (FOREIGN, "a file this task was never allowed to edit"),
        (NO_CHECKS, "nothing gated the pull request"),
    ],
)
def test_each_real_escalation_reason_gets_an_actionable_summary(reason, expected_fragment):
    """The two that matter most are the first two. An unsatisfiable task and a
    broken machine are indistinguishable in a status field and call for opposite
    responses - fix the plan, or fix the plumbing."""
    report = decisions([cycle(escalation(2, reason))], repo=REPO)

    assert expected_fragment in report.decisions[0].cause


def test_an_unrecognised_reason_says_so_rather_than_guessing():
    """A rule this table has not been taught is a gap in the table. Inventing a
    cause for it would be worse than admitting it, because the cause is the one
    line an operator acts on."""
    assert classify("something no rule has ever written") == UNCLASSIFIED


def test_the_rules_own_words_survive_alongside_the_summary():
    """The summary is this module's paraphrase. An operator has to be able to see
    what was actually decided, not only how it was labelled."""
    text = decisions([cycle(escalation(2, GATE_RED))], repo=REPO).text()

    assert "3 attempt(s) made, the last 3 failing the same way" in text


# --------------------------------------------------------------------------
# What the operator is told to do
# --------------------------------------------------------------------------


def test_the_report_prints_the_command_that_resolves_it():
    """`swarm reset` already exists, which is why this module does not grow a
    second way to do it - it just has to be named."""
    text = decisions([cycle(escalation(2, GATE_RED))], repo=REPO).text()

    assert f"swarm reset '#2' --repo {REPO}" in text
    assert "needs a decision on 1 task(s)" in text


def test_a_task_escalated_twice_is_explained_by_the_failure_that_ended_it():
    """The goal gate can revive a failed task, so a run can escalate the same
    task twice for different reasons. The second one is the one still true."""
    report = decisions(
        [
            cycle(escalation(2, INFRASTRUCTURE)),
            cycle(escalation(2, GATE_RED)),
        ],
        repo=REPO,
    )

    assert len(report.decisions) == 1
    assert report.decisions[0].reason == GATE_RED


def test_escalations_from_every_cycle_are_gathered_not_just_the_last():
    """An escalation is permanent, and the cycle it happened in is long gone by
    the time the run ends."""
    report = decisions(
        [cycle(escalation(2, GATE_RED)), cycle(), cycle(escalation(5, INFRASTRUCTURE))],
        repo=REPO,
    )

    assert [str(ref) for ref in report.refs] == ["#2", "#5"]


# --------------------------------------------------------------------------
# What is stranded behind the failure
# --------------------------------------------------------------------------


def test_the_work_stranded_behind_a_failure_is_named_under_it():
    """The operator's question is "what does resetting this buy me", and this is
    the answer. #4 waits on #2, so #2's decision is worth two tasks."""
    report = decisions(
        [
            cycle(
                escalation(2, GATE_RED),
                entries=(task(2), task(4, depends_on=("task-2",))),
                belief=Says(task_2=NEEDS_HUMAN),
            )
        ],
        repo=REPO,
    )

    assert report.decisions[0].stranded[0][0] == task_ref(4)
    assert "#4" in report.text()
    assert "which needs a human" in report.text()


def test_a_failure_that_stranded_nothing_says_nothing_extra():
    """Common, and it means the plan lost a leaf rather than a branch."""
    report = decisions(
        [
            cycle(
                escalation(2, GATE_RED),
                entries=(task(2), task(4)),
                belief=Says(task_2=NEEDS_HUMAN),
            )
        ],
        repo=REPO,
    )

    assert report.decisions[0].stranded == ()
    assert "#4" not in report.text()


def test_stranding_is_read_from_where_the_run_stopped():
    """A task stranded in cycle 0 and revived by cycle 1 is not stranded. Only
    the final ledger and belief know that, which is why they are the pair read.
    """
    entries = (task(2), task(4, depends_on=("task-2",)))
    report = decisions(
        [
            cycle(escalation(2, GATE_RED), entries=entries, belief=Says(task_2=NEEDS_HUMAN)),
            cycle(entries=entries, belief=Says(task_2="eligible")),
        ],
        repo=REPO,
    )

    assert report.decisions[0].stranded == ()
