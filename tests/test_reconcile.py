"""Tests for the reconcile loop.

Five properties carry this file, and each of them is something the run gets
wrong rather than something it computes wrong.

**GitHub wins, within one cycle.** Closing an issue by hand disposes its worker
and the run carries on - #22's acceptance criterion - and so do relabelling it,
stripping its `swarm:*` label, and deleting it out from under a running
container. None of those raises.

**A finished worker's exit code decides the attempt, not the label.** Exit 1
consumes an attempt and gives up at the cap, exit 2 does not consume one, and
exit 0 moves no label at all because `claimed -> review` belongs to the worker.

**Absence of evidence is not evidence.** A `swarm:review` issue whose pull
request could not be listed must not be read as a pull request that was closed.

**The budget is asserted, not asserted about.** A run of N cycles over a
repository of 24 issues makes the same number of requests as over 4, and the
repeat read carries `If-None-Match` - which is the difference between a loop
that costs O(N) and one that costs O(N x issues).

**Writes fail one at a time.** A human deleting an issue between the read and
the write costs that issue's transition and nothing else.

Hermetic throughout. The behavioural tests drive a plain fake client - the
`Snapshot` falls through to it, which is also how the probes for the two
methods `GitHubClient` does not have yet are exercised - and the budget test
drives a real `GitHubClient` over `fixtures.github`'s scripted transport, so
the request count is the client's own and not a stub's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from types import SimpleNamespace

import pytest

from fixtures.github import SentRequest, not_modified, page, response
from swarm.containers.manager import DockerError, Handle
from swarm.github.client import GitHubHTTPError
from swarm.github.ledger import ContractError, LabelRepair, Ledger, LedgerEntry, render_marker
from swarm.github.readiness import IssueState
from swarm.orchestrator.dispatcher import CLAIMED, REVIEW, Capacity
from swarm.orchestrator.reconcile import (
    COMMENT_METHOD,
    PULLS_METHOD,
    DONE,
    FAILED,
    READY,
    Reconciler,
    Snapshot,
    Transition,
    apply_plan,
    fold,
    plan_reconcile,
    rewrite_marker,
)
from swarm.run import Run
from swarm.worker.result import ResultRecord, write_result

REPO = "shahrestani-me/apiary"
RUN_ID = "apiary-20260814-142530-k3f9qz"
BASE_COMMIT = "9f2c1ab3d4e5f60718293a4b5c6d7e8f90a1b2c3"
BLOCKED = "swarm:blocked"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def entry(
    number: int,
    *files: str,
    label: str = READY,
    attempt: int = 0,
) -> LedgerEntry:
    return LedgerEntry(
        number=number,
        title=f"issue {number}",
        task_id=f"task-{number}",
        attempt=attempt,
        goal="do the thing",
        files=files or (f"src/mod{number}.py",),
        verify="python -m pytest -q",
        blocked_by=(),
        state_label=label,
        labels=frozenset({label}),
    )


def ledger(*entries: LedgerEntry, **kwargs: Any) -> Ledger:
    return Ledger(entries={item.task_id: item for item in entries}, **kwargs)


def closed(number: int, reason: str | None = "completed") -> IssueState:
    return IssueState(number=number, state="closed", state_reason=reason)


def record(issue: int, exit_code: int, *, attempt: int = 0, reason: str = "") -> ResultRecord:
    return ResultRecord(
        run_id=RUN_ID,
        issue=issue,
        attempt=attempt,
        exit_code=exit_code,
        reason=reason or "the verify command failed",
    )


def body(task_id: str, *, attempt: int = 0, blocked_by: Iterable[int] = ()) -> str:
    refs = "\n".join(f"- #{ref}" for ref in blocked_by) or "_none._"
    return "\n".join(
        [
            render_marker(task_id, attempt),
            "",
            "## Goal",
            "Do the thing.",
            "",
            "## Files",
            f"- src/{task_id}.py",
            "",
            "## Verify",
            "python -m pytest -q",
            "",
            "## Blocked by",
            refs,
        ]
    )


def issue_payload(
    number: int,
    *,
    label: str = READY,
    state: str = "open",
    state_reason: str | None = None,
    task_id: str | None = None,
    attempt: int = 0,
    body_text: str | None = None,
) -> dict[str, Any]:
    task_id = task_id or f"task-{number}"
    return {
        "number": number,
        "title": f"issue {number}",
        "state": state,
        "state_reason": state_reason,
        "labels": [{"name": label}],
        "body": body_text if body_text is not None else body(task_id, attempt=attempt),
    }


@dataclass
class FakeClient:
    """Every call a cycle makes, recorded, with no HTTP anywhere.

    Deliberately *without* `list_pull_requests` and `create_issue_comment`:
    that is the state `GitHubClient` is actually in, and the subclasses below
    are what the reconciler will see once #23 and §1.4's comment method land.
    """

    issues: dict[int, dict[str, Any]] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    fail_labels_on: set[int] = field(default_factory=set)

    def list_issues(self, *, state: str = "open", **kwargs: Any) -> list[dict[str, Any]]:
        self.log.append(f"list_issues {state}")
        return [dict(payload) for payload in self.issues.values()]

    def get_issue(self, number: int) -> dict[str, Any]:
        self.log.append(f"get_issue #{number}")
        if number not in self.issues:
            raise GitHubHTTPError(404, "GET", f"/issues/{number}", b'{"message":"Not Found"}')
        return dict(self.issues[number])

    def update_issue(self, number: int, **kwargs: Any) -> dict[str, Any]:
        self.log.append(f"update_issue #{number}")
        self.issues[number].update(kwargs)
        return dict(self.issues[number])

    def add_labels(self, number: int, labels: Iterable[str]) -> Any:
        names = list(labels)
        self.log.append(f"+{','.join(names)} #{number}")
        if number in self.fail_labels_on:
            raise GitHubHTTPError(404, "POST", f"/issues/{number}/labels", b'{"message":"gone"}')
        current = self.issues.get(number, {}).setdefault("labels", [])
        current.extend({"name": name} for name in names)
        return current

    def remove_label(self, number: int, label: str) -> bool:
        self.log.append(f"-{label} #{number}")
        payload = self.issues.get(number)
        if payload is None:
            return False
        payload["labels"] = [item for item in payload["labels"] if item["name"] != label]
        return True

    # --- what the assertions read ---------------------------------------

    def labels_on(self, number: int) -> set[str]:
        return {item["name"] for item in self.issues[number]["labels"]}


@dataclass
class PullAwareClient(FakeClient):
    """The client once #23 has grown the listing method the loop needs."""

    open_pulls: tuple[str, ...] = ()

    def list_pull_requests(self, *, state: str = "open") -> list[dict[str, Any]]:
        self.log.append(f"list_pull_requests {state}")
        return [{"number": 900 + i, "head": {"ref": ref}} for i, ref in enumerate(self.open_pulls)]


@dataclass
class CommentingClient(FakeClient):
    """The client once §1.4's comment method exists."""

    comments: list[tuple[int, str]] = field(default_factory=list)

    def create_issue_comment(self, number: int, text: str) -> dict[str, Any]:
        self.comments.append((number, text))
        return {"id": len(self.comments)}


@dataclass
class FakeFleet:
    """`Fleet`: the containers a run has, and what happened to them."""

    handles: dict[int, Handle] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    dispose_error: Exception | None = None

    def find(self, *, issue: int | None = None) -> list[Handle]:
        self.log.append("find")
        found = list(self.handles.values())
        return [h for h in found if issue is None or h.issue == issue]

    def dispose(self, handle: Handle) -> str:
        self.log.append(f"dispose #{handle.issue}")
        if self.dispose_error is not None:
            raise self.dispose_error
        self.handles.pop(handle.issue, None)
        return ""

    def spawn(self, issue: int, base_commit: str) -> Handle:
        self.log.append(f"spawn #{issue}")
        handle = Handle(id=f"{issue:0>64x}", run_id=RUN_ID, issue=issue)
        self.handles[issue] = handle
        return handle

    @property
    def disposed(self) -> list[int]:
        return [int(line.split("#")[1]) for line in self.log if line.startswith("dispose")]

    @property
    def spawned(self) -> list[int]:
        return [int(line.split("#")[1]) for line in self.log if line.startswith("spawn")]


def running(*issues: int) -> dict[int, Handle]:
    return {n: Handle(id=f"{n:0>64x}", run_id=RUN_ID, issue=n) for n in issues}


def reconciler(client: Any, fleet: Any = None, **kwargs: Any) -> Reconciler:
    kwargs.setdefault("capacity", Capacity(slots=3, configured=2))
    # Nothing sleeps for real: the pacing test injects its own recorder, and
    # every other test here would otherwise pay `DEFAULT_INTERVAL_S` per cycle
    # to assert something that has nothing to do with the clock.
    kwargs.setdefault("sleep", lambda _seconds: None)
    return Reconciler(
        run=Run.start(REPO, "reconcile the ledger", run_id=RUN_ID),
        client=client,
        base_commit=BASE_COMMIT,
        fleet=fleet,
        **kwargs,
    )


# --------------------------------------------------------------------------
# GitHub wins
# --------------------------------------------------------------------------


def test_an_issue_closed_by_hand_is_taken_out_of_the_run_and_its_worker_disposed():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)),
        states={4: closed(4)},
        running=[4],
    )

    # The headline of #22: a human closing an issue mid-run is a feature of
    # putting the ledger on GitHub, and the container has to go with it.
    assert [str(t) for t in plan.transitions] == [
        "#4: swarm:claimed -> swarm:done (closed as completed on GitHub)"
    ]
    assert [d.number for d in plan.disposals] == [4]


def test_a_merged_pull_request_is_read_from_the_issue_it_closed():
    plan = plan_reconcile(ledger(entry(4, label=REVIEW)), states={4: closed(4)}, running=[4])

    # `Closes #<n>` means the merge closes the issue, so the cheap read already
    # carries the signal and nothing has to page through every PR ever opened.
    assert plan.transitions[0].to_label == DONE
    assert plan.disposals[0].number == 4


def test_an_issue_closed_as_not_planned_is_failed_rather_than_done():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)),
        states={4: closed(4, "not_planned")},
        running=[4],
    )

    # The same judgement readiness makes about a dependency: work somebody
    # explicitly decided not to do did not happen.
    assert plan.transitions[0].to_label == FAILED
    assert "not planned" in plan.transitions[0].reason


def test_an_issue_a_human_already_marked_done_keeps_no_container():
    plan = plan_reconcile(ledger(entry(4, label=DONE)), running=[4])

    assert plan.transitions == ()
    assert [d.number for d in plan.disposals] == [4]


def test_a_container_whose_issue_left_the_ledger_is_disposed():
    # A human deleted the issue, or stripped its `swarm:*` label - either way
    # nothing will look at that work again and the clone is held for nobody.
    plan = plan_reconcile(ledger(entry(4, label=CLAIMED)), states={4: closed(4)}, running=[4, 9])

    assert sorted(d.number for d in plan.disposals) == [4, 9]
    assert "no longer in the ledger" in next(d for d in plan.disposals if d.number == 9).reason


def test_two_state_labels_are_repaired_from_what_the_loader_reported():
    repair = LabelRepair(number=4, kept=DONE, removed=(CLAIMED,))

    plan = plan_reconcile(ledger(entry(4, label=DONE), repairs=(repair,)))

    # The loader only reports the fault (§3); removing the losing label is a
    # write, and writes are this module's.
    assert plan.repairs == (repair,)


# --------------------------------------------------------------------------
# Finished workers
# --------------------------------------------------------------------------


def test_a_failed_worker_consumes_an_attempt_and_goes_back_to_ready():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=0)),
        results={4: record(4, 1)},
        running=[4],
        max_attempts=3,
    )

    transition = plan.transitions[0]
    assert (transition.to_label, transition.attempt) == (READY, 1)
    assert [d.number for d in plan.disposals] == [4]


def test_a_failed_worker_at_the_cap_is_handed_to_a_human():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=2)),
        results={4: record(4, 1, attempt=2)},
        max_attempts=3,
    )

    transition = plan.transitions[0]
    assert (transition.to_label, transition.attempt) == (FAILED, 3)
    # §5: the counter is an upper bound on attempts made. Giving up early puts
    # a human in front of the problem; looping forever looks healthy.
    assert "cap of 3" in transition.reason
    assert transition.comment


def test_an_infrastructure_failure_does_not_consume_an_attempt():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=1)),
        results={4: record(4, 2, attempt=1, reason="ollama refused the connection")},
        max_attempts=3,
    )

    transition = plan.transitions[0]
    # A broken Ollama would otherwise burn every task's budget before anyone
    # noticed. `attempt=None` means the counter is not written at all.
    assert (transition.to_label, transition.attempt) == (READY, None)


def test_an_unknown_exit_code_is_charged_like_a_failure():
    # 137 from the OOM killer, 143 from a stop. `ResultRecord.consumes_attempt`
    # already settled this; the reconciler must not re-decide it.
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)),
        results={4: record(4, 137)},
        max_attempts=3,
    )

    assert plan.transitions[0].attempt == 1


def test_a_worker_that_published_moves_no_label():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)),
        results={4: record(4, 0)},
        running=[4],
    )

    # `claimed -> review` is the worker's row (§4): it knows the PR exists at
    # the instant it does, and taking the write here would race it. An exit 0
    # whose label did not stick is #35's stale claim, not this module's.
    assert plan.transitions == ()
    assert [d.number for d in plan.disposals] == [4]


def test_a_records_verdict_is_not_applied_twice():
    # The counter moves and the artifact does not, so the record of attempt 0
    # must stop counting once the issue is on attempt 1.
    plan = plan_reconcile(
        ledger(entry(4, label=READY, attempt=1)),
        results={4: record(4, 1, attempt=0)},
    )

    assert plan.transitions == ()


# --------------------------------------------------------------------------
# Pull requests
# --------------------------------------------------------------------------


def test_a_pull_request_closed_without_merging_returns_the_issue_to_the_pool():
    plan = plan_reconcile(
        ledger(entry(4, label=REVIEW, attempt=0)),
        open_branches=frozenset(),
        running=[4],
        max_attempts=3,
    )

    transition = plan.transitions[0]
    # The attempt is consumed: the work was done and rejected, and a retry that
    # costs nothing can be rejected forever.
    assert (transition.to_label, transition.attempt) == (READY, 1)
    assert [d.number for d in plan.disposals] == [4]
    assert plan.blind is False


def test_an_open_pull_request_is_left_alone():
    plan = plan_reconcile(
        ledger(entry(4, label=REVIEW)),
        open_branches=frozenset({"swarm/issue-4"}),
    )

    assert plan.transitions == ()


def test_a_published_workers_container_is_disposed_without_waiting_for_the_merge():
    plan = plan_reconcile(
        ledger(entry(4, label=REVIEW)),
        open_branches=frozenset({"swarm/issue-4"}),
        results={4: record(4, 0)},
        running=[4],
    )

    # The record is the worker's last act, so the container is only holding a
    # clone. Waiting for the merge would leak one per task for the length of
    # the review, and the label is already where the worker put it.
    assert plan.transitions == ()
    assert [d.number for d in plan.disposals] == [4]


def test_pull_requests_that_could_not_be_listed_are_not_read_as_closed():
    plan = plan_reconcile(ledger(entry(4, label=REVIEW)), open_branches=None)

    # The whole reason `open_branches` is `None` rather than an empty set. An
    # empty set means every review PR is gone; None means we did not look, and
    # conflating them relabels the entire review queue.
    assert plan.transitions == ()
    assert plan.blind is True


# --------------------------------------------------------------------------
# Malformed contracts
# --------------------------------------------------------------------------


def test_a_malformed_issue_is_failed_and_carries_the_reason_as_a_comment():
    error = ContractError(7, "Verify", "section is missing")

    plan = plan_reconcile(ledger(errors=(error,)), labels={7: frozenset({READY})})

    transition = plan.transitions[0]
    assert (transition.number, transition.from_label, transition.to_label) == (7, READY, FAILED)
    # §1.4: the parse failure is posted back on the issue that failed it.
    assert "section is missing" in transition.comment


def test_a_malformed_issue_already_failed_is_not_failed_again():
    error = ContractError(7, "Verify", "section is missing")

    plan = plan_reconcile(ledger(errors=(error,)), labels={7: frozenset({FAILED})})

    # Otherwise every cycle re-labels it and comments on it again forever.
    assert plan.transitions == ()


def test_a_malformed_issue_outside_the_ledger_is_left_alone():
    error = ContractError(7, "Goal", "section is empty")

    plan = plan_reconcile(ledger(errors=(error,)), labels={7: frozenset({"area/docs"})})

    # No `swarm:*` label means not part of the ledger at all (§1.4). Humans use
    # the tracker too.
    assert plan.transitions == ()


# --------------------------------------------------------------------------
# The attempt counter
# --------------------------------------------------------------------------


def test_the_counter_is_rewritten_and_every_other_byte_survives():
    original = "\n".join(
        [render_marker("add-retry-logic", 1), "", "## Goal", "A human's prose.", "  indented  "]
    )

    updated = rewrite_marker(original, "add-retry-logic", 2)

    assert updated.splitlines()[0] == render_marker("add-retry-logic", 2)
    assert updated.splitlines()[1:] == original.splitlines()[1:]


def test_a_body_with_no_marker_gets_one_without_losing_the_body():
    updated = rewrite_marker("## Goal\nwritten by a human", "adopted-task", 1)

    assert updated.startswith(render_marker("adopted-task", 1))
    assert "written by a human" in updated


def test_a_marker_quoted_inside_a_fence_is_not_the_one_rewritten():
    original = "\n".join(
        ["```markdown", render_marker("add-retry-logic", 0), "```", "", "## Goal", "x"]
    )

    updated = rewrite_marker(original, "real-task", 2)

    # `ledger._parse_marker` ignores fenced markers, so rewriting one would set
    # a counter nothing ever reads while leaving the real identity untouched.
    assert updated.splitlines()[0] == render_marker("real-task", 2)
    assert render_marker("add-retry-logic", 0) in updated


# --------------------------------------------------------------------------
# Folding
# --------------------------------------------------------------------------


def test_a_landed_transition_is_folded_into_the_ledger_rather_than_re_read():
    entries = ledger(entry(4, label=CLAIMED, attempt=0))

    folded = fold(entries, [Transition(4, CLAIMED, READY, "exit 1", task_id="task-4", attempt=1)])

    # A second listing to observe our own writes is the one request that buys
    # nothing, and the dispatcher needs the freed capacity this cycle.
    updated = folded.entries["task-4"]
    assert (updated.state_label, updated.attempt) == (READY, 1)
    assert updated.labels == frozenset({READY})
    assert entries.entries["task-4"].state_label == CLAIMED


def test_folding_a_transition_for_an_issue_outside_the_ledger_changes_nothing():
    entries = ledger(entry(4))

    assert fold(entries, [Transition(9, READY, FAILED, "malformed")]).entries == entries.entries


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def test_the_counter_is_persisted_before_the_label_goes_back_to_ready():
    client = FakeClient(issues={4: issue_payload(4, label=CLAIMED)})
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)), results={4: record(4, 1)}, max_attempts=3
    )

    apply_plan(client, plan)

    # §5: a crash between the write and the re-dispatch must cost an attempt,
    # not grant a free one. And add-before-remove, because two state labels are
    # repairable by §3's precedence and none is not.
    assert client.log == [
        "get_issue #4",
        "update_issue #4",
        f"+{READY} #4",
        f"-{CLAIMED} #4",
    ]
    assert "attempt=1" in client.issues[4]["body"]


def test_the_body_is_re_read_immediately_before_the_counter_is_patched():
    client = FakeClient(issues={4: issue_payload(4, label=CLAIMED)})
    client.issues[4]["body"] = body("task-4") + "\n\nA human typed this mid-cycle."
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)), results={4: record(4, 1)}, max_attempts=3
    )

    apply_plan(client, plan)

    # GitHub's last-write-wins gives no help here, so the fresh read is the
    # whole protection for somebody editing prose while a counter moves.
    assert "A human typed this mid-cycle." in client.issues[4]["body"]


def test_one_issue_failing_does_not_cost_the_others_their_transition():
    client = FakeClient(
        issues={4: issue_payload(4, label=CLAIMED), 5: issue_payload(5, label=CLAIMED)},
        fail_labels_on={4},
    )
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED), entry(5, label=CLAIMED)),
        states={4: closed(4), 5: closed(5)},
    )

    report = apply_plan(client, plan)

    # A human deleting an issue between the read and the write lands here. It
    # is a fact about one issue, not a reason to abandon the cycle.
    assert [t.number for t in report.applied] == [5]
    assert [f.number for f in report.failures] == [4]
    assert report.ok is False


def test_a_container_that_will_not_die_does_not_stop_the_labels_from_moving():
    client = FakeClient(issues={4: issue_payload(4, label=CLAIMED)})
    fleet = FakeFleet(handles=running(4))
    fleet.dispose_error = DockerError(["docker", "rm"], 1, "daemon is not responding")
    plan = plan_reconcile(ledger(entry(4, label=CLAIMED)), states={4: closed(4)}, running=[4])

    report = apply_plan(client, plan, fleet=fleet, handles=running(4))

    assert [t.to_label for t in report.applied] == [DONE]
    assert report.disposed == ()
    assert "daemon is not responding" in str(report.failures[0])


def test_a_dry_run_writes_nothing_at_all():
    client = FakeClient(issues={4: issue_payload(4, label=CLAIMED)})
    fleet = FakeFleet(handles=running(4))
    plan = plan_reconcile(ledger(entry(4, label=CLAIMED)), states={4: closed(4)}, running=[4])

    report = apply_plan(client, plan, fleet=fleet, handles=running(4), dry_run=True)

    assert client.log == []
    assert fleet.log == []
    assert (report.applied, report.disposed) == ((), ())


def test_a_client_with_no_comment_method_prints_the_reason_and_reports_the_gap(capsys):
    client = FakeClient(issues={7: issue_payload(7)})
    plan = plan_reconcile(
        ledger(errors=(ContractError(7, "Verify", "section is missing"),)),
        labels={7: frozenset({READY})},
    )

    report = apply_plan(client, plan)

    # §1.4 wants the text on the issue and `GitHubClient` has no method for it.
    # Printing is not a substitute; it is what stops the reason being lost.
    assert report.uncommented == (7,)
    assert COMMENT_METHOD in report.summary()
    assert "section is missing" in capsys.readouterr().err


def test_the_comment_is_posted_once_the_client_can_post_one():
    client = CommentingClient(issues={7: issue_payload(7)})
    plan = plan_reconcile(
        ledger(errors=(ContractError(7, "Verify", "section is missing"),)),
        labels={7: frozenset({READY})},
    )

    report = apply_plan(client, plan)

    assert report.uncommented == ()
    assert client.comments[0][0] == 7
    assert "section is missing" in client.comments[0][1]


# --------------------------------------------------------------------------
# The cycle
# --------------------------------------------------------------------------


def test_closing_an_issue_mid_run_disposes_its_worker_and_the_run_continues():
    client = FakeClient(
        issues={
            4: issue_payload(4, label=CLAIMED, state="closed", state_reason="not_planned"),
            5: issue_payload(5, label=READY),
        }
    )
    fleet = FakeFleet(handles=running(4))

    report = reconciler(client, fleet).cycle()

    # #22's "done when", end to end: the worker for the cancelled issue is
    # gone, the issue is out of the ledger's live set, and the cycle went on to
    # dispatch the work that was waiting behind it.
    assert fleet.disposed == [4]
    assert client.labels_on(4) == {FAILED}
    assert fleet.spawned == [5]
    assert report.live == 1


def test_a_cycle_reads_the_issue_list_once_however_many_collaborators_want_it():
    client = FakeClient(issues={n: issue_payload(n) for n in range(4, 12)})

    reconciler(client, FakeFleet()).cycle()

    # The loader, the readiness pass and this module's own rules all want it.
    assert client.log.count("list_issues all") == 1


def test_a_dependency_cycle_is_reported_every_cycle_rather_than_killing_the_run():
    payloads = {
        4: issue_payload(4, body_text=body("task-4", blocked_by=[5])),
        5: issue_payload(5, body_text=body("task-5", blocked_by=[4])),
    }
    client = FakeClient(issues=payloads)
    fleet = FakeFleet()

    report = reconciler(client, fleet).cycle()

    # Nothing was written - readiness detects the ring before its first call -
    # and nothing was dispatched over a graph whose prerequisites cannot land.
    assert "dependency cycle" in report.cycle_error
    assert report.dispatched is None
    assert fleet.spawned == []


def test_a_review_issue_whose_pull_request_vanished_is_retried_once_prs_are_readable():
    client = PullAwareClient(
        issues={4: issue_payload(4, label=REVIEW)},
        open_pulls=(),
    )

    fleet = FakeFleet()
    report = reconciler(client, fleet).cycle()

    # Freed and re-dispatched inside the one cycle: reconciling before
    # dispatching is what returns the capacity, and §5's increment is on the
    # issue before the second container exists.
    assert report.plan.blind is False
    assert "attempt=1" in client.issues[4]["body"]
    assert client.labels_on(4) == {CLAIMED}
    assert fleet.spawned == [4]


def test_a_client_that_cannot_list_pull_requests_leaves_the_review_queue_alone():
    client = FakeClient(issues={4: issue_payload(4, label=REVIEW)})

    report = reconciler(client, FakeFleet()).cycle()

    assert report.plan.blind is True
    assert client.labels_on(4) == {REVIEW}


def test_a_finished_worker_is_observed_from_its_artifact_and_not_from_docker(tmp_path):
    client = FakeClient(issues={4: issue_payload(4, label=CLAIMED)})
    fleet = FakeFleet(handles=running(4))
    write_result(record(4, 1, reason="the verify command failed"), tmp_path)

    report = reconciler(client, fleet, artifacts=tmp_path).cycle()

    # The worker writes its record last, so a record is the evidence that the
    # container finished - no blocking `docker wait`, no extra API call. Its
    # container is disposed and the retry goes out in the same cycle.
    assert report.result.applied[0].attempt == 1
    assert fleet.disposed == [4]
    assert fleet.spawned == [4]


def test_the_loop_stops_when_nothing_is_live():
    client = FakeClient(issues={4: issue_payload(4, label=DONE)})

    reports = reconciler(client, FakeFleet()).loop(cycles=5)

    assert len(reports) == 1
    assert reports[0].finished is True


def test_the_interval_is_a_floor_between_cycle_starts_not_a_delay_after_each():
    client = FakeClient(issues={4: issue_payload(4)})
    slept: list[float] = []
    ticks = iter([0.0, 4.0, 100.0, 130.0, 200.0, 200.5])

    loop = reconciler(
        client,
        FakeFleet(),
        interval_s=10.0,
        sleep=slept.append,
        clock=lambda: next(ticks),
    )
    loop.loop(cycles=3)

    # A cycle that took 4s waits 6 more; one that took 30s starts the next
    # immediately rather than sleeping on top of its own duration. And nothing
    # sleeps after the last cycle - the only thing that would wait for is the
    # caller's return.
    assert slept == [6.0]


# --------------------------------------------------------------------------
# The polling budget
# --------------------------------------------------------------------------


def scripted_repo(count: int) -> Callable[[SentRequest], Any]:
    """A repository of `count` well-formed, ready, unblocked issues.

    Answers the listing with an `ETag` and honours `If-None-Match`, because the
    claim being tested is about conditional requests and a fake that always
    returned 200 would prove only half of it.
    """
    payloads = [issue_payload(number) for number in range(1, count + 1)]

    def handle(request: SentRequest) -> Any:
        if request.method == "GET" and request.path.endswith("/issues"):
            if request.header("If-None-Match") == '"ledger-v1"':
                return not_modified()
            return page(payloads, ETag='"ledger-v1"')
        if request.method == "GET" and request.path.endswith("/pulls"):
            # The second of the cycle's two reads. It appeared when
            # `GitHubClient` grew `list_pull_requests`: before that the
            # reconciler was blind to PRs and deliberately decided nothing
            # about them. Two constant reads per cycle is still O(cycles),
            # which is what this section is about - the failure it guards
            # against is a read *per issue*.
            return page([], ETag='"pulls-v1"')
        raise AssertionError(f"unbudgeted request: {request.method} {request.path}")

    return handle


@pytest.mark.parametrize("issues", [4, 24])
def test_a_run_of_n_cycles_costs_o_n_requests_not_o_n_times_issues(fake_github, issues):
    gh, transport, _ = fake_github(handler=scripted_repo(issues))

    reports = reconciler(gh).loop(cycles=3, until=lambda report: False)

    # #22's second acceptance criterion. Six times the issues, the same number
    # of requests: two listings per cycle - issues and pulls - each shared by
    # every collaborator in that cycle. The claim is that the count follows
    # the number of cycles and not the size of the ledger, so it is asserted
    # as a multiple of the cycles rather than as a constant.
    assert len(reports) == 3
    assert len(transport.sent) == 3 * 2
    assert transport.sent[2].header("If-None-Match") == '"ledger-v1"'


def test_the_repeat_read_is_a_304_and_still_produces_the_whole_ledger(fake_github):
    gh, _, _ = fake_github(handler=scripted_repo(6))

    reports = reconciler(gh).loop(cycles=2, until=lambda report: False)

    # A 304 carries no body. Serving it from the cache is what makes the cheap
    # cycle a real read rather than an empty ledger that looks like a finished
    # run.
    assert [report.live for report in reports] == [6, 6]


def test_a_reconciler_holds_nothing_a_restart_would_need(fake_github):
    gh, _, _ = fake_github(handler=scripted_repo(3))
    first = reconciler(gh).cycle()

    gh2, _, _ = fake_github(handler=scripted_repo(3))
    second = reconciler(gh2).cycle()

    # "Restartable at any point": both processes ask GitHub, so both get the
    # same answer, and neither carries anything across the cycle boundary.
    assert first.summary() == second.summary()


def test_the_snapshot_falls_through_to_the_client_for_anything_it_does_not_shape():
    client = PullAwareClient(issues={4: issue_payload(4)}, open_pulls=("swarm/issue-4",))
    snapshot = Snapshot(client)

    # The probe for a method the client has not grown yet must see the client's
    # own answer. A wrapper that answered for it would turn a method that is
    # merely missing into one that can never be found.
    assert snapshot.open_branches() == frozenset({"swarm/issue-4"})
    with pytest.raises(AttributeError):
        getattr(snapshot, COMMENT_METHOD)

    # The listing is wrapped rather than delegated, so a cycle's collaborators
    # share one read - but the wrapper exists only because this client does
    # have the method, and it answers with the client's own data.
    assert [p["head"]["ref"] for p in getattr(snapshot, PULLS_METHOD)()] == ["swarm/issue-4"]


def test_a_client_that_cannot_list_pull_requests_still_cannot():
    """The cached wrapper must not invent a capability.

    `Snapshot.pull_requests` tells "this client cannot look" from "nothing is
    open" by probing for the method, and `open_branches` turns the first into
    `None` - which is what stops a cycle relabelling the entire review queue.
    A wrapper that existed unconditionally would answer `[]` and erase that
    distinction.
    """
    class Blind:
        def list_issues(self, **kwargs: Any) -> list[dict[str, Any]]:
            return [issue_payload(4)]

    snapshot = Snapshot(Blind())
    with pytest.raises(AttributeError):
        getattr(snapshot, PULLS_METHOD)
    assert snapshot.open_branches() is None


def test_recovery_is_handed_containers_not_issue_numbers(fake_github):
    """`_handles()` is a mapping keyed by issue number.

    Passing it straight to `Recovery.sweep` iterates the *keys*, so
    `recovery.holders` asked an int for its `.issue` and the whole run died on
    an AttributeError in the first cycle. Caught only by running it: every CLI
    test used --plan-only, so the loop this sits in had no coverage at all.
    """
    seen: dict = {}

    class SpyRecovery:
        def sweep(self, ledger, *, containers=None, states=None, open_branches=None):
            seen["containers"] = list(containers or ())
            return SimpleNamespace(result=SimpleNamespace(applied=()))

    gh, _, _ = fake_github(handler=scripted_repo(1))
    subject = reconciler(gh)
    subject.recovery = SpyRecovery()

    subject.cycle()

    assert "containers" in seen, "recovery was never swept"
    assert all(not isinstance(c, int) for c in seen["containers"]), (
        f"recovery received issue numbers, not handles: {seen['containers']}"
    )
