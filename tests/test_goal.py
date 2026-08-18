"""Tests for the question a run asks last: was the objective met, or the plan?

Four properties carry this file, and each of them is a way the swarm stops in
the wrong place rather than a way it computes wrong.

**A finished plan is not a met objective, and the loop must know the
difference.** `CycleReport.exhausted` and `CycleReport.finished` used to be one
property; splitting them is what lets a gate append work and have the loop carry
on rather than exit reporting success over a half-built repository. Both
directions are asserted, because a `finished` that is always true stops early
and one that is never true never stops at all.

**A follow-up round may only ever add.** `write_plan` retires every ledger entry
the plan does not contain, and a follow-up plan contains none of them by
construction - so `retire_dropped=False` is the single line standing between
this feature and a round that closes every issue the run just merged. It is
asserted twice: once against a writer spy, and once for real against the fake
transport, where the test reads back the issues afterwards and insists nothing
was closed.

**The model is asked as rarely as possible, and never when its answer could not
be acted on.** Every arithmetic refusal below runs against an oracle that raises
if it is invoked - `test_replan.py`'s `Never`, and for its reason: an
orchestrator call is a ~6.7 s model swap, and a gate that consults one to be
told the ledger is empty has spent the run's budget on arithmetic.

**A model that cannot be reached does not end the run and does not extend it.**
An unreachable Ollama reads as unresolved, exactly as it does for the judge and
for a worker's exit 2, and the run stops saying so rather than declaring victory
or planning follow-ups from a verdict nobody gave.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pytest

from fixtures.github import REPO, response
from swarm.github.ledger import Ledger, LedgerEntry
from swarm.nodes.planner import PlanError, PlanReport, render_body, write_plan
from swarm.orchestrator.goal import (
    EMPTY,
    EXHAUSTED,
    FAILED as FAILED_REASON,
    IN_FLIGHT,
    MAX_ROUNDS,
    NO_TASKS,
    Assessment,
    assess,
    close_the_loop,
    shipped,
)
from swarm.state import ObjectiveAssessment, Plan, PlannedTask

READY = "swarm:ready"
CLAIMED = "swarm:claimed"
REVIEW = "swarm:review"
DONE = "swarm:done"
FAILED = "swarm:failed"

VERIFY = "python -m pytest -q"
OBJECTIVE = "a trip planner that plans a trip"
BASE = f"/repos/{REPO}/issues"


# --------------------------------------------------------------------------
# Ledgers and oracles
# --------------------------------------------------------------------------


def entry(
    number: int,
    *,
    task_id: str = "",
    label: str = DONE,
    goal: str = "do the thing",
    files: Sequence[str] = (),
    attempt: int = 0,
    blocker: str = "",
    streak: int | None = None,
) -> LedgerEntry:
    return LedgerEntry(
        number=number,
        title=f"issue {number}",
        task_id=task_id or f"task-{number}",
        attempt=attempt,
        blocker=blocker,
        streak=streak,
        goal=goal,
        files=tuple(files) or (f"src/mod{number}.py",),
        verify=VERIFY,
        blocked_by=(),
        state_label=label,
        labels=frozenset({label}),
    )


def ledger(*entries: LedgerEntry) -> Ledger:
    return Ledger(entries={item.task_id: item for item in entries})


class ModelCalled(BaseException):
    """Not an `Exception`: `assess` catches those and reports unresolved, which
    would swallow the very assertion this class exists to make."""


class Never:
    """An oracle that fails the test if anything asks it a question."""

    def invoke(self, messages: Sequence[tuple[str, str]]) -> Any:
        raise ModelCalled(f"a model was asked: {messages!r}")


@dataclass
class Says:
    """An oracle with a scripted answer, which records what it was shown."""

    answer: Any
    asked: list[Any] = field(default_factory=list)

    def invoke(self, messages: Sequence[tuple[str, str]]) -> Any:
        self.asked.append(messages)
        return self.answer


class Unreachable:
    """Ollama restarting, a socket refusing, a model that is not pulled."""

    def invoke(self, messages: Sequence[tuple[str, str]]) -> Any:
        raise RuntimeError("connection refused")


@dataclass
class Writer:
    """A `write_plan` spy. Records the call it is asked to make, writes nothing."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    created: tuple[int, ...] = (11,)
    error: Exception | None = None

    def __call__(self, client: Any, plan: Plan, **kwargs: Any) -> PlanReport:
        self.calls.append({"plan": plan, **kwargs})
        if self.error is not None:
            raise self.error
        from swarm.nodes.planner import IssueAction

        return PlanReport(
            REPO,
            tuple(
                IssueAction("created", f"new-{number}", number) for number in self.created
            ),
        )


@dataclass
class RevivalClient:
    """The four calls a gate revival makes, recorded, with no HTTP anywhere.

    `list_issues` answers `resolve_states`' open/closed read; the label and
    comment calls are the revival's writes. Everything else is absent on
    purpose - a gate that reached for more than these has grown a write this
    suite did not agree to.
    """

    issues: list[dict[str, Any]] = field(default_factory=list)
    added: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)
    removed: list[tuple[int, str]] = field(default_factory=list)
    comments: list[tuple[int, str]] = field(default_factory=list)

    def list_issues(self, *, state: str = "open", **_: Any) -> list[dict[str, Any]]:
        return [dict(issue) for issue in self.issues]

    def add_labels(self, number: int, labels: Any) -> list[dict[str, Any]]:
        self.added.append((number, tuple(labels)))
        return []

    def remove_label(self, number: int, label: str) -> bool:
        self.removed.append((number, label))
        return True

    def create_issue_comment(self, number: int, body: str) -> dict[str, Any]:
        self.comments.append((number, body))
        return {"id": len(self.comments)}

    @property
    def wrote(self) -> bool:
        return bool(self.added or self.removed or self.comments)


def open_issue(number: int) -> dict[str, Any]:
    return {"number": number, "state": "open", "state_reason": None}


def closed_issue(number: int, reason: str = "not_planned") -> dict[str, Any]:
    return {"number": number, "state": "closed", "state_reason": reason}


def met(reason: str = "everything is there") -> ObjectiveAssessment:
    return ObjectiveAssessment(objective_met=True, reason=reason)


def unmet(*missing: str) -> ObjectiveAssessment:
    return ObjectiveAssessment(
        objective_met=False, missing=list(missing), reason="something is missing"
    )


def follow_up(*ids: str) -> Plan:
    return Plan(
        tasks=[
            PlannedTask(id=task_id, goal=f"build {task_id}", files=[f"src/{task_id}.py"])
            for task_id in ids
        ]
    )


# --------------------------------------------------------------------------
# The arithmetic refusals
# --------------------------------------------------------------------------


def test_an_empty_ledger_is_not_assessed() -> None:
    """Nothing was planned, so there is no evidence to assess an objective on."""
    verdict = assess(OBJECTIVE, Ledger(), oracle=Never())

    assert not verdict.met
    assert not verdict.consulted
    assert verdict.reason == EMPTY


def test_work_in_flight_is_not_assessed() -> None:
    """Half a run is not an answer about the objective, and asking mid-run would
    make the gate's verdict depend on which cycle happened to call it."""
    verdict = assess(OBJECTIVE, ledger(entry(1, label=DONE), entry(2, label=REVIEW)), oracle=Never())

    assert not verdict.met
    assert not verdict.consulted
    assert IN_FLIGHT in verdict.reason
    assert "#2" in verdict.reason


def test_an_abandoned_task_stops_the_gate_and_is_named() -> None:
    """`swarm:failed` is the swarm saying a human is needed. Planning more work
    on top of it stacks tasks onto a repository whose last known state is broken."""
    verdict = assess(OBJECTIVE, ledger(entry(1, label=DONE), entry(2, label=FAILED)), oracle=Never())

    assert not verdict.met
    assert not verdict.consulted
    assert not verdict.actionable
    assert verdict.abandoned == (2,)
    assert verdict.reason == FAILED_REASON
    assert any("#2" in line for line in verdict.missing)


def test_a_finished_plan_is_assessed_against_the_objective() -> None:
    """The one case worth a model swap: everything merged, nothing abandoned."""
    oracle = Says(met("the library and its CLI are both there"))

    verdict = assess(OBJECTIVE, ledger(entry(1), entry(2)), oracle=oracle)

    assert verdict.met
    assert verdict.consulted
    assert len(oracle.asked) == 1
    shown = oracle.asked[0][1][1]
    # The goals, not the titles: the goal is the sentence the contract calls
    # load-bearing, and "issue 1" tells a model nothing about coverage.
    assert "do the thing" in shown
    assert OBJECTIVE in shown


def test_an_unreachable_model_is_unresolved_not_unmet() -> None:
    """The judge's rule and §4's exit 2: infrastructure is not an answer."""
    verdict = assess(OBJECTIVE, ledger(entry(1)), oracle=Unreachable())

    assert not verdict.met
    assert verdict.unresolved
    assert not verdict.actionable
    assert "connection refused" in verdict.reason


# --------------------------------------------------------------------------
# Reviving: the gate does what the human used to do
# --------------------------------------------------------------------------


def test_the_gate_revives_an_open_failed_task_with_budget_remaining() -> None:
    """The live gap: #23 and #24 merged, #21 sat failed at attempt 5 of a 9
    hard cap, and the run exited failed asking a human to do the one relabel
    the orchestrator knew was safe."""
    client = RevivalClient(issues=[open_issue(2)])
    writer = Writer()

    report = close_the_loop(
        client,
        ledger(entry(1, label=DONE), entry(2, label=FAILED, attempt=5, streak=3)),
        OBJECTIVE,
        oracle=Never(),
        proposer=Never(),
        writer=writer,
        max_attempts=3,
        max_total_attempts=9,
    )

    # The run continues: revived work is dispatchable, not a resignation.
    assert not report.done
    assert not report.extended
    assert report.rounds == 0
    assert [action.number for action in report.revived] == [2]
    assert report.reason == "revived 1 failed task(s) with budget remaining: #2"
    assert "revived 1 failed task(s)" in report.summary()
    assert "the run continues" in report.summary()
    # The writes: ready on, failed off, in that order (two labels beats none),
    # and the marker is never touched - no update_issue method even exists on
    # this double, so a body write would have been an AttributeError.
    assert client.added == [(2, (READY,))]
    assert client.removed == [(2, FAILED)]
    number, text = client.comments[0]
    assert number == 2
    assert text.startswith("apiary: the goal gate found the objective unmet")
    assert "returned to `swarm:ready`" in text
    assert "streak 3 of 3, total 5 of 9" in text
    # No follow-up was planned and no model consulted: the revival is
    # arithmetic, and the objective is re-assessed only after the retry runs.
    assert not writer.calls


def test_a_closed_failed_issue_is_never_revived() -> None:
    """GitHub wins: a closed issue was shut by a human or retired by the
    planner, and the gate arguing with that would be the one overrule this
    system promises never to make."""
    client = RevivalClient(issues=[closed_issue(2)])

    report = close_the_loop(
        client,
        ledger(entry(1, label=DONE), entry(2, label=FAILED, attempt=1)),
        OBJECTIVE,
        oracle=Never(),
        proposer=Never(),
        writer=Writer(),
    )

    assert report.done
    assert report.revived == ()
    assert report.reason == FAILED_REASON
    assert not client.wrote


def test_a_budget_spent_failed_task_is_not_revived() -> None:
    """`planner.revive` owns the budget refusal: a spent task answers with a
    retained action and no writes, so the gate resigns exactly as it always
    did - the give-up comment on the issue already told a human what to do."""
    client = RevivalClient(issues=[open_issue(2)])

    report = close_the_loop(
        client,
        ledger(entry(1, label=DONE), entry(2, label=FAILED, attempt=9, streak=3)),
        OBJECTIVE,
        oracle=Never(),
        proposer=Never(),
        writer=Writer(),
        max_attempts=3,
        max_total_attempts=9,
    )

    assert report.done
    assert report.revived == ()
    assert report.reason == FAILED_REASON
    assert "still missing" in report.summary() or "stopping" in report.summary()
    assert not client.wrote


def test_only_the_revivable_failed_tasks_are_revived() -> None:
    client = RevivalClient(issues=[open_issue(2), open_issue(3), closed_issue(4)])

    report = close_the_loop(
        client,
        ledger(
            entry(1, label=DONE),
            entry(2, label=FAILED, attempt=5, streak=3),
            entry(3, label=FAILED, attempt=9, streak=3),
            entry(4, label=FAILED, attempt=1),
        ),
        OBJECTIVE,
        oracle=Never(),
        proposer=Never(),
        writer=Writer(),
        max_attempts=3,
        max_total_attempts=9,
    )

    assert not report.done
    assert [action.number for action in report.revived] == [2]
    assert report.reason == "revived 1 failed task(s) with budget remaining: #2"
    assert client.added == [(2, (READY,))]
    assert [number for number, _ in client.comments] == [2]


# --------------------------------------------------------------------------
# Extending
# --------------------------------------------------------------------------


def test_a_gap_becomes_follow_up_issues() -> None:
    writer = Writer(created=(11, 12))

    report = close_the_loop(
        object(),
        ledger(entry(1), entry(2)),
        OBJECTIVE,
        verify=VERIFY,
        oracle=Says(unmet("there is no CLI", "nothing reads a timetable")),
        proposer=Says(follow_up("add-cli", "read-timetable")),
        writer=writer,
    )

    assert report.extended
    assert not report.done
    assert report.rounds == 1
    assert [action.number for action in report.created] == [11, 12]
    assert "planned 2 follow-up task(s)" in report.summary()


def test_a_follow_up_round_never_retires_anything() -> None:
    """The line this whole feature rests on. `write_plan` retires every entry a
    plan omits, and a follow-up plan omits all of them by construction."""
    writer = Writer()

    close_the_loop(
        object(),
        ledger(entry(1), entry(2)),
        OBJECTIVE,
        verify=VERIFY,
        oracle=Says(unmet("there is no CLI")),
        proposer=Says(follow_up("add-cli")),
        writer=writer,
    )

    assert writer.calls[0]["retire_dropped"] is False
    # And the run's own command, not `SETTINGS.verify_command`: a generated
    # repository has no way to run the default.
    assert writer.calls[0]["verify"] == VERIFY


def test_the_shipped_work_is_shown_to_the_planner() -> None:
    """A follow-up plan that cannot see what shipped re-emits it under new ids,
    which is a second issue for work that already merged."""
    proposer = Says(follow_up("add-cli"))

    close_the_loop(
        object(),
        ledger(entry(1, task_id="core", goal="implement the core library")),
        OBJECTIVE,
        verify=VERIFY,
        oracle=Says(unmet("there is no CLI")),
        proposer=proposer,
        writer=Writer(),
    )

    prompt = proposer.asked[0][0][1]
    assert "core" in prompt
    assert "implement the core library" in prompt
    assert "there is no CLI" in prompt


def test_a_met_objective_extends_nothing() -> None:
    writer = Writer()

    report = close_the_loop(
        object(),
        ledger(entry(1)),
        OBJECTIVE,
        oracle=Says(met()),
        proposer=Never(),
        writer=writer,
    )

    assert report.met
    assert report.done
    assert not writer.calls


def test_an_unmet_objective_with_no_named_gap_extends_nothing() -> None:
    """A model that says "not met" and names nothing has given the planner
    nothing to decompose; asking anyway plans its guess at its own answer."""
    writer = Writer()

    report = close_the_loop(
        object(),
        ledger(entry(1)),
        OBJECTIVE,
        oracle=Says(ObjectiveAssessment(objective_met=False, reason="not sure")),
        proposer=Never(),
        writer=writer,
    )

    assert not report.extended
    assert not writer.calls
    assert "not sure" in report.summary()


def test_the_rounds_are_bounded() -> None:
    writer = Writer()

    report = close_the_loop(
        object(),
        ledger(entry(1)),
        OBJECTIVE,
        rounds=MAX_ROUNDS,
        oracle=Says(unmet("there is no CLI")),
        proposer=Never(),
        writer=writer,
    )

    assert not report.extended
    assert report.done
    assert EXHAUSTED in report.reason
    assert not writer.calls
    assert "still missing" in report.summary()


def test_a_plan_that_normalises_to_nothing_is_refused() -> None:
    """`write_plan` is not reached, so the round is not charged: a model that
    answered with no usable task has not spent a follow-up round."""
    writer = Writer()

    report = close_the_loop(
        object(),
        ledger(entry(1)),
        OBJECTIVE,
        oracle=Says(unmet("there is no CLI")),
        proposer=Says(Plan(tasks=[])),
        writer=writer,
    )

    assert not report.extended
    assert report.rounds == 0
    assert report.reason == NO_TASKS
    assert not writer.calls


def test_an_unwritable_plan_leaves_the_tracker_alone() -> None:
    report = close_the_loop(
        object(),
        ledger(entry(1)),
        OBJECTIVE,
        oracle=Says(unmet("there is no CLI")),
        proposer=Says(follow_up("add-cli")),
        writer=Writer(error=PlanError("a ring in the dependency graph")),
    )

    assert not report.extended
    assert "not writable" in report.reason


def test_a_write_that_created_nothing_is_not_an_extension() -> None:
    """Every draft rejected by the planner's self-check. Reporting an extension
    would send the loop round again to find the same nothing."""
    report = close_the_loop(
        object(),
        ledger(entry(1)),
        OBJECTIVE,
        oracle=Says(unmet("there is no CLI")),
        proposer=Says(follow_up("add-cli")),
        writer=Writer(created=()),
    )

    assert not report.extended
    assert report.done
    assert "created no issue" in report.reason


def test_an_unreachable_planner_does_not_spend_a_round() -> None:
    report = close_the_loop(
        object(),
        ledger(entry(1)),
        OBJECTIVE,
        oracle=Says(unmet("there is no CLI")),
        proposer=Unreachable(),
        writer=Writer(),
    )

    assert not report.extended
    assert report.rounds == 0
    assert "could not be reached" in report.reason


def test_shipped_reads_the_label_not_the_issue_state() -> None:
    done = ledger(entry(1, label=DONE), entry(2, label=FAILED), entry(3, label=CLAIMED))

    assert [item.number for item in shipped(done)] == [1]


# --------------------------------------------------------------------------
# The real write path, once
# --------------------------------------------------------------------------


def test_the_real_writer_creates_and_closes_nothing(fake_github) -> None:
    """The whole feature against the transport, and the one thing it must not do.

    Two issues are already on the tracker, both `swarm:done` and both closed.
    The follow-up round must create a third and leave those two exactly as they
    are - no `PATCH` closing them, no `state_reason` of `not_planned`.
    """
    existing = [
        {
            "number": number,
            "title": f"issue {number}",
            "state": "closed",
            "state_reason": "completed",
            "labels": [{"name": DONE}],
            "body": render_body(
                f"task-{number}",
                goal="do the thing",
                files=[f"src/mod{number}.py"],
                verify=VERIFY,
            ),
        }
        for number in (1, 2)
    ]

    def answer(request) -> Any:
        if request.method == "GET" and request.path.startswith(BASE):
            return response(200, existing)
        if request.method == "POST" and request.path == BASE:
            return response(
                201, {"number": 3, "title": "add a CLI", "labels": [{"name": READY}]}
            )
        # A `PATCH` here is the failure this test exists to catch: it is how
        # `write_plan` retires an entry the plan does not contain.
        raise AssertionError(f"a follow-up round wrote {request.method} {request.path}")

    client, transport, _ = fake_github(handler=answer)

    report = close_the_loop(
        client,
        ledger(
            entry(1, task_id="task-1", label=DONE),
            entry(2, task_id="task-2", label=DONE),
        ),
        OBJECTIVE,
        verify=VERIFY,
        oracle=Says(unmet("there is no CLI")),
        proposer=Says(follow_up("add-cli")),
        writer=write_plan,
    )

    assert report.extended
    assert [action.number for action in report.created] == [3]
    assert [sent.method for sent in transport.sent].count("POST") == 1
    assert "PATCH" not in [sent.method for sent in transport.sent]
