"""Tests for stale-claim recovery.

This module takes `swarm:claimed` off an issue and returns it to the pool, so
these tests are about two opposite mistakes and the line between them.

**A claim with nothing behind it must not survive.** The dispatcher claims
before it spawns, so a crash in that window leaves a label no other component
will ever remove and an issue that is undispatchable while looking perfectly
healthy. The ticket's done-when is here twice: once as the startup sweep it
asks for, and once as the mid-cycle case that actually happened - three issues
claimed, one spawned, the process interrupted before the rest.

**A claim somebody is honouring must not be stolen.** Liveness is #20's rule
and not a second one: a container speaks for a claim when its `apiary.run` is
live, and a container of a run this process does not answer to speaks for
nothing because that process is gone. A sibling declared through
`live_run_ids`, and a container wearing a run label this system could not have
minted, both keep their claims.

**The counter is the difference between recovering and looping.** An issue that
crashes the orchestrator is the issue that will crash it again, so every
release consumes an attempt and the cap ends in `swarm:failed`. The `review`
path consumes none: that worker finished.

Hermetic throughout. The container listings go through a `Runner` double that
parses `--filter` for real, so `find_containers`' own behaviour is under test
rather than mocked, and the GitHub side is a plain fake client. The handful of
tests that want a real daemon carry the `docker` marker (tests/conftest.py),
mint their own run ids, and use issue numbers no ledger here shares - and
nothing in this file removes a container on any path.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

import pytest

from swarm.containers.manager import (
    ISSUE_LABEL,
    ContainerError,
    ContainerManager,
    DockerCLI,
    Handle,
    Limits,
)
from swarm.github.branches import task_branch
from swarm.github.client import GitHubHTTPError
from swarm.github.ledger import Ledger, LedgerEntry, render_marker
from swarm.github.readiness import BLOCKED, READY, IssueState
from swarm.github.refs import task_ref as ref
from swarm.orchestrator.dispatcher import CLAIMED, REVIEW, claim
from swarm.orchestrator.reconcile import DONE, FAILED
from swarm.orchestrator.recovery import (
    Held,
    Recovery,
    RecoveryPlan,
    holders,
    in_flight,
    live_runs,
    plan_recovery,
    unrecognised,
)
from swarm.run import RUN_LABEL, Run
from swarm.orchestrator.derived import ELIGIBLE, NEEDS_HUMAN
from swarm.orchestrator.derived import CLAIMED as CLAIMED_STATE
from swarm.orchestrator.derived import REVIEW as REVIEW_STATE

REPO = "shahrestani-me/apiary"
OBJECTIVE = "recover the claims a killed orchestrator left on the tracker"
BASE_COMMIT = "9f2c1ab3d4e5f60718293a4b5c6d7e8f90a1b2c3"

#: A run whose orchestrator is gone. Ids are never reused, so nothing this
#: process does can ever make this one live again - which is the entire signal.
DEAD_RUN = "apiary-20260814-142530-k3f9qz"
SIBLING_RUN = "apiary-20260814-151500-qq8w2r"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def entry(number: int, *, label: str = CLAIMED, attempt: int = 0) -> LedgerEntry:
    return LedgerEntry(
        number=number,
        title=f"issue {number}",
        task_id=f"task-{number}",
        attempt=attempt,
        goal="do the thing",
        files=(f"src/mod{number}.py",),
        verify="python -m pytest -q",
        blocked_by=(),
        state_label=label,
        labels=frozenset({label}),
    )


def branch(number: int, attempt: int = 0) -> str:
    """The head ref a worker for `(number, attempt)` pushed - built, not spelled.

    Spelling it out would make these tests assert the encoding rather than the
    behaviour, and the encoding is `test_branches.py`'s subject.
    """
    return task_branch(ref(number), attempt)


def ledger(*entries: LedgerEntry, **kwargs: Any) -> Ledger:
    return Ledger(entries={item.task_id: item for item in entries}, **kwargs)


def handle(issue: int, run_id: str) -> Handle:
    return Handle(id=f"{issue:0>64x}", run_id=run_id, issue=issue)


def closed(number: int, reason: str | None = "completed") -> IssueState:
    return IssueState(ref=ref(number), state="closed", state_reason=reason)


def body(task_id: str, *, attempt: int = 0) -> str:
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
            "_none._",
        ]
    )


def issue_payload(number: int, *, label: str = CLAIMED, attempt: int = 0) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"issue {number}",
        "state": "open",
        "state_reason": None,
        "labels": [{"name": label}],
        "body": body(f"task-{number}", attempt=attempt),
    }


@dataclass
class FakeClient:
    """Every GitHub call a sweep makes, recorded, with no HTTP anywhere.

    Without `list_pull_requests`, which is the state `GitHubClient` is actually
    in: the sweep has to work blind today and get better when #23 lands, and
    both halves of that are asserted below.
    """

    issues: dict[int, dict[str, Any]] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    fail_labels_on: set[int] = field(default_factory=set)

    def list_issues(self, *, state: str = "open", **kwargs: Any) -> list[dict[str, Any]]:
        self.log.append(f"list_issues {state}")
        return [dict(payload) for payload in self.issues.values()]

    def get_issue(self, number: int) -> dict[str, Any]:
        self.log.append(f"get_issue #{number}")
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

    def attempt_on(self, number: int) -> int:
        marker = self.issues[number]["body"].splitlines()[0]
        return int(marker.split("attempt=")[1].split()[0])


@dataclass
class PullAwareClient(FakeClient):
    """The client once #23 has grown the listing the sweep would rather have."""

    open_pulls: tuple[str, ...] = ()

    def list_pull_requests(self, *, state: str = "open") -> list[dict[str, Any]]:
        self.log.append(f"list_pull_requests {state}")
        return [{"number": 900 + i, "head": {"ref": ref}} for i, ref in enumerate(self.open_pulls)]


@dataclass
class Daemon:
    """A `Runner` that answers `docker ps` from a container table.

    It parses `--filter` for real, and refuses a listing that carries no label
    filter rather than answering it: a bare `docker ps -a` on a development
    machine returns the human's databases and editors, and a sweep that decides
    which claims to release from that list is deciding from noise.

    It answers `rm` too - and no test here ever provokes one. Removing
    containers is #20's half of this fault, and doing it from here would put
    two modules on one `docker rm`.
    """

    containers: list[Handle] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)

    def __call__(
        self, argv: Sequence[str], *, timeout_s: float | None, merge: bool
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        if argv[1] != "ps":
            raise AssertionError(f"recovery issued an unexpected {argv[1]!r}: {list(argv)}")
        filters = [argv[i + 1] for i, part in enumerate(argv) if part == "--filter"]
        if not any(f.startswith("label=") for f in filters):
            raise AssertionError(f"a listing with no label filter: {list(argv)}")
        rows = [
            "\t".join(
                [
                    c.id,
                    c.name or "probe",
                    "apiary-worker",
                    c.run_id,
                    str(c.issue or ""),
                    # `docker ps --format {{.State}}` (#187). `exited` unless a
                    # test says otherwise, because `holders` counts a stopped
                    # container of a live run as a claim somebody is honouring
                    # - reading its exit code is #22's row, not this module's.
                    c.state or "exited",
                ]
            )
            for c in self.containers
            if all(_matches(c, f) for f in filters)
        ]
        stdout = "\n".join(rows) + ("\n" if rows else "")
        return subprocess.CompletedProcess(list(argv), 0, stdout, "")

    @property
    def commands(self) -> list[str]:
        return [call[1] for call in self.calls]


def _matches(container: Handle, spec: str) -> bool:
    labels = {RUN_LABEL: container.run_id}
    if container.issue is not None:
        labels[ISSUE_LABEL] = str(container.issue)
    key = spec.split("=", 1)[1]
    if "=" in key:
        name, value = key.split("=", 1)
        return labels.get(name) == value
    return key in labels


def make_run(run_id: str | None = None) -> Run:
    return Run.start(REPO, OBJECTIVE, run_id=run_id)


def recovery(client: Any, daemon: Daemon | None = None, **kwargs: Any) -> Recovery:
    kwargs.setdefault("run", make_run())
    return Recovery(client=client, docker=DockerCLI(runner=daemon or Daemon()), **kwargs)


# --------------------------------------------------------------------------
# The claim with nothing behind it
# --------------------------------------------------------------------------


def test_a_claim_with_no_container_behind_it_returns_to_the_pool():
    plan = plan_recovery(ledger(entry(4, attempt=0)), max_attempts=3)

    # The ticket. Nothing else in the system takes this label off, so an issue
    # left here is undispatchable forever while looking perfectly healthy.
    transition = plan.transitions[0]
    assert (transition.from_state, transition.to_state) == (CLAIMED_STATE, ELIGIBLE)
    assert transition.attempt == 1
    assert plan.held == ()


def test_the_release_consumes_an_attempt_so_a_second_crash_gives_up():
    first = plan_recovery(ledger(entry(4, attempt=0)), max_attempts=2)
    second = plan_recovery(ledger(entry(4, attempt=1)), max_attempts=2)

    # "An issue that crashes the orchestrator twice should reach swarm:failed,
    # not loop." §5's counter is an upper bound on attempts made: over-counting
    # puts a human in front of the problem, under-counting loops while looking
    # healthy.
    assert (first.transitions[0].to_state, first.transitions[0].attempt) == (ELIGIBLE, 1)
    assert (second.transitions[0].to_state, second.transitions[0].attempt) == (NEEDS_HUMAN, 2)
    assert "cap of 2" in second.transitions[0].reason
    assert second.transitions[0].comment


def test_only_claimed_issues_are_considered_at_all():
    entries = ledger(
        entry(1, label=READY),
        entry(2, label=BLOCKED),
        entry(3, label=REVIEW),
        entry(5, label=DONE),
        entry(6, label=FAILED),
    )

    plan = plan_recovery(entries)

    # Every other label is somebody else's row of §4, and none of them means
    # "a container should be running this". Relabelling one because no
    # container was found would yank issues out of states nobody asked about.
    assert plan == RecoveryPlan(blind=True)


def test_an_issue_closed_by_hand_is_left_to_the_reconciler():
    plan = plan_recovery(ledger(entry(4)), states={ref(4): closed(4)})

    # GitHub wins, but what a closed issue becomes - `done` or `failed` - is
    # #22's judgement, and answering it from here would be a second opinion on
    # one fact.
    assert plan.transitions == ()
    assert [item.ref for item in plan.held] == [ref(4)]
    assert "reconciler" in plan.held[0].reason


# --------------------------------------------------------------------------
# Whose claim it is
# --------------------------------------------------------------------------


def test_a_claim_a_live_container_is_holding_is_never_touched():
    run = make_run()

    plan = plan_recovery(
        ledger(entry(4)),
        containers=[handle(4, run.id)],
        live_run_ids=live_runs(run),
    )

    assert plan.transitions == ()
    assert plan.held == (Held(ref(4), f"a container of run {run.id!r} is holding it"),)


def test_a_container_that_exited_still_holds_the_claim_of_a_live_run():
    """The other side of #187: this module must keep reading stopped containers.

    `find_containers` can now be asked for running containers only, and asking
    here would be wrong. Reading the exit code and moving the label is #22's
    row; a second module deciding the same issue from "the process is gone" is
    how two of them end up writing one label.
    """
    run = make_run()
    daemon = Daemon([Handle(id="a" * 64, run_id=run.id, issue=4, state="exited")])

    plan = recovery(FakeClient(), daemon, run=run).plan(ledger(entry(4)))

    assert plan.transitions == ()
    assert [item.ref for item in plan.held] == [ref(4)]
    # And it was listed at all, which is the half a `status=running` filter
    # would have removed.
    assert not [part for call in daemon.calls for part in call if part.startswith("status=")]


def test_a_sibling_run_the_caller_declared_live_does_not_lose_its_claim():
    run = make_run()

    plan = plan_recovery(
        ledger(entry(4), entry(5)),
        containers=[handle(4, SIBLING_RUN), handle(5, DEAD_RUN)],
        live_run_ids=live_runs(run, {SIBLING_RUN}),
    )

    # "One orchestrator per repository" is the default, not a law. A scheduler
    # driving several names them here, and stealing a running sibling's claim
    # would put a second container on the same file set.
    assert plan.refs == (ref(5),)
    assert [item.ref for item in plan.held] == [ref(4)]


def test_a_container_of_a_dead_run_speaks_for_nothing_and_is_not_removed():
    daemon = Daemon([handle(4, DEAD_RUN)])
    client = FakeClient(issues={4: issue_payload(4)})

    recovery(client, daemon).sweep(ledger(entry(4)))

    # The reaper's rule, reused rather than re-invented: ids are never reused,
    # so a label naming a run this process is not names a process that is gone.
    # Its container is #20's to remove, and this module removes none.
    assert client.labels_on(4) == {READY}
    assert daemon.commands == ["ps"]


def test_a_container_wearing_an_id_this_system_could_not_have_minted_keeps_its_claim():
    plan = plan_recovery(
        ledger(entry(4)),
        containers=[Handle(id="foreign", run_id="Not A Run Id", issue=4)],
        live_run_ids=live_runs(make_run()),
    )

    # Something else is wearing our labels. The reaper spares it because it is
    # not ours to remove; the claim over it is not ours to release either, and
    # the reason is reported rather than swallowed.
    assert plan.transitions == ()
    assert [item.ref for item in plan.held] == [ref(4)]


def test_a_container_carrying_no_issue_label_holds_no_claim():
    run = make_run()

    assert holders([Handle(id="bare", run_id=run.id)], live_runs(run)) == {}


def test_without_a_run_of_its_own_only_declared_runs_are_live():
    # The `swarm reap`-shaped call: a sweep run by something that is not an
    # orchestrator has no containers of its own to spare.
    assert live_runs(None, {SIBLING_RUN}) == frozenset({SIBLING_RUN})
    assert live_runs(None) == frozenset()


# --------------------------------------------------------------------------
# What the branch names alone say (#144)
# --------------------------------------------------------------------------


def test_in_flight_reconstructs_tasks_and_attempts_from_branch_names_alone():
    """ADR 0001's promise, with nothing else in the room: no ledger, no label,
    no local memory - a list of names off a remote, and the pairs come back."""
    found = in_flight([branch(4, 0), branch(7, 2), "main"])

    assert {ref: found[ref].attempt for ref in found} == {ref(4): 0, ref(7): 2}


def test_the_furthest_attempt_wins_because_a_task_owns_one_branch_per_attempt():
    """A retry is a new branch (#144), so a ref legitimately owns several and
    only the newest speaks for where the task is now. An older attempt's branch
    outliving its pull request is history, not a contradiction to resolve."""
    found = in_flight([branch(4, 2), branch(4, 0), branch(4, 1)])

    assert found[ref(4)].attempt == 2


def test_branches_from_before_this_ticket_are_reported_rather_than_parsed():
    """The migration case. A repository that was mid-run when the naming changed
    holds `swarm/issue-<n>` branches, and a sweep that silently ignored them is
    indistinguishable from one that found nothing in flight at all."""
    names = ["swarm/issue-4", branch(7, 0), "renovate/urllib3-2.x"]

    assert in_flight(names).keys() == {ref(7)}
    assert unrecognised(names) == ("renovate/urllib3-2.x", "swarm/issue-4")


def test_a_sweep_says_how_many_branch_names_it_could_not_read():
    plan = plan_recovery(
        ledger(entry(4)), open_branches=frozenset({"swarm/issue-4"})
    )

    assert plan.unrecognised == ("swarm/issue-4",)
    assert "1 branch name(s) not apiary's" in plan.summary()
    # And the claim is still released, because nothing readable is behind it.
    assert plan.released == plan.transitions


# --------------------------------------------------------------------------
# The worker that finished and died
# --------------------------------------------------------------------------


def test_a_claim_is_matched_to_its_pull_request_by_ref_not_by_current_attempt():
    """The reason the name carries a pair rather than a ref alone.

    `LedgerEntry.branch` is rebuilt from a counter, and this sweep runs after a
    crash - the one moment the counter and the branch on the remote can disagree.
    Matching by name would leave a claim looking abandoned while its finished
    work sits in an open pull request, and release it: a second container over
    the top of a PR somebody may already be reading."""
    plan = plan_recovery(
        ledger(entry(4, attempt=2)),
        open_branches=frozenset({branch(4, attempt=1)}),
    )

    assert plan.published == plan.transitions
    assert plan.transitions[0].to_state == REVIEW_STATE
    # The name in the reason is the one on the remote, not the entry's, because
    # that is the branch a human goes and looks at.
    assert branch(4, attempt=1) in plan.transitions[0].reason




def test_a_claim_with_an_open_pull_request_moves_forward_to_review():
    plan = plan_recovery(
        ledger(entry(4, attempt=1)),
        open_branches=frozenset({branch(4, attempt=1)}),
    )

    transition = plan.transitions[0]
    # `worker/pr.py` writes its label after the PR exists, deliberately, so a
    # worker that died in between leaves exactly this. The work is done: moving
    # the label forward is all that is left, and the attempt is not consumed
    # because charging one would give up on a task that succeeded.
    assert (transition.to_state, transition.attempt) == (REVIEW_STATE, None)
    assert plan.published == plan.transitions
    assert plan.released == ()


def test_a_live_container_outranks_an_open_pull_request():
    run = make_run()

    plan = plan_recovery(
        ledger(entry(4)),
        containers=[handle(4, run.id)],
        live_run_ids=live_runs(run),
        open_branches=frozenset({branch(4)}),
    )

    # A worker that has pushed may still be running - it labels, it writes its
    # record, and only then does it exit. Nothing here is stale.
    assert plan.transitions == ()
    assert [item.ref for item in plan.held] == [ref(4)]


def test_an_unreadable_pull_request_list_does_not_leave_the_claim_unreachable():
    plan = plan_recovery(ledger(entry(4)), open_branches=None, max_attempts=3)

    # The asymmetry with #22, which refuses to act while blind: there the
    # subject is a `swarm:review` issue whose state is already correct, here it
    # is a claim already known to have nothing behind it. Holding would leave
    # the ticket unmet with today's client, which has no `list_pull_requests`
    # at all; releasing costs at most one redundant attempt, because the branch
    # is derived from the issue number and GitHub refuses a second PR for it.
    assert plan.blind is True
    assert (plan.transitions[0].to_state, plan.transitions[0].attempt) == (ELIGIBLE, 1)


def test_the_review_path_is_taken_once_pull_requests_can_be_listed():
    client = PullAwareClient(
        issues={4: issue_payload(4, attempt=1)},
        open_pulls=(branch(4, attempt=1),),
    )

    report = recovery(client).startup()

    assert client.labels_on(4) == {REVIEW}
    # No body PATCH at all: `attempt=None` means the counter is left alone,
    # which is not the same as writing back the value it already had.
    assert "update_issue #4" not in client.log
    assert client.attempt_on(4) == 1
    assert report.plan.blind is False


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def test_the_counter_is_persisted_before_the_label_goes_back_to_ready():
    client = FakeClient(issues={4: issue_payload(4)})

    recovery(client).sweep(ledger(entry(4)))

    # §5: a crash between the write and the re-dispatch must cost an attempt,
    # not grant a free one. And add-before-remove, because two state labels are
    # repairable by §3's precedence and none is not.
    assert client.log[-4:] == ["get_issue #4", "update_issue #4", f"+{READY} #4", f"-{CLAIMED} #4"]
    assert client.attempt_on(4) == 1


def test_one_issue_that_cannot_be_relabelled_does_not_cost_the_others_their_recovery():
    client = FakeClient(
        issues={4: issue_payload(4), 5: issue_payload(5)},
        fail_labels_on={4},
    )

    report = recovery(client).sweep(ledger(entry(4), entry(5)))

    # A human deleting or relabelling an issue between the read and the write
    # lands here. It is a fact about one issue, not a reason to abandon a sweep
    # that was about to free four others.
    assert report.refs == (ref(5),)
    assert [failure.ref for failure in report.result.failures] == [ref(4)]
    assert report.ok is False


def test_a_dry_run_writes_nothing_and_still_says_what_it_would_do():
    client = FakeClient(issues={4: issue_payload(4)})

    report = recovery(client, dry_run=True).startup()

    assert client.log == ["list_issues all"]
    assert client.labels_on(4) == {CLAIMED}
    assert report.plan.refs == (ref(4),)
    assert report.applied == ()


def test_the_startup_sweep_reads_the_issue_list_once():
    client = FakeClient(issues={n: issue_payload(n) for n in range(4, 12)})

    recovery(client).startup()

    # The ledger, each issue's open/closed state and the pull-request probe all
    # come off one listing, exactly as they do inside a cycle.
    assert client.log.count("list_issues all") == 1


# --------------------------------------------------------------------------
# The done-when
# --------------------------------------------------------------------------


def test_a_claim_left_by_a_killed_orchestrator_is_back_at_ready_on_restart():
    """The ticket's done-when, with the process that made the claim gone.

    `kill -9` between claim and spawn, then a restart: a fresh `Run` whose id
    the leftover container - if there even is one - cannot be wearing.
    """
    client = FakeClient(issues={4: issue_payload(4, attempt=0)})
    daemon = Daemon([handle(4, DEAD_RUN)])

    report = Recovery(
        client=client,
        run=make_run(),
        docker=DockerCLI(runner=daemon),
        max_attempts=3,
    ).startup()

    assert client.labels_on(4) == {READY}
    assert client.attempt_on(4) == 1
    assert report.ok and report.refs == (ref(4),)


def test_a_second_sweep_finds_nothing_left_to_do():
    client = FakeClient(issues={4: issue_payload(4)})
    sweeper = recovery(client)

    sweeper.startup()
    again = sweeper.startup()

    # Nothing is carried between passes, so a sweep is safe to run at startup
    # and then at the top of every cycle. The second one reads what the first
    # one wrote and has no opinion about it.
    assert again.plan == RecoveryPlan(blind=True)
    assert client.labels_on(4) == {READY}
    assert client.attempt_on(4) == 1


def test_a_mid_cycle_sweep_recovers_the_claims_a_spawn_never_reached():
    """Beyond the ticket: the window that opens without anybody restarting.

    Three issues claimed, the first spawned, the process interrupted before the
    other two reached `manager.spawn`. A startup-only sweep would not run again
    until somebody noticed; this is the same sweep, at the top of the next
    cycle.
    """
    run = make_run()
    client = FakeClient(issues={n: issue_payload(n, label=READY) for n in (4, 5, 6)})
    entries = [entry(n, label=READY) for n in (4, 5, 6)]
    for item in entries:
        claim(client, item)
    spawned = handle(4, run.id)

    plan = plan_recovery(
        ledger(*(entry(item.number) for item in entries)),
        containers=[spawned],
        live_run_ids=live_runs(run),
        max_attempts=3,
    )

    assert [item.ref for item in plan.held] == [ref(4)]
    assert plan.refs == (ref(5), ref(6))
    assert {transition.to_state for transition in plan.transitions} == {ELIGIBLE}


def test_a_sweep_that_frees_nothing_says_which_claims_it_left_and_why():
    run = make_run()

    plan = plan_recovery(
        ledger(entry(4), entry(5)),
        containers=[handle(4, run.id), handle(5, SIBLING_RUN)],
        live_run_ids=live_runs(run, {SIBLING_RUN}),
    )

    # "Recovery ran and released nothing" and "recovery ran and decided both
    # claims were somebody else's" are different facts, and only the second one
    # is reassuring to a human wondering why an issue is stuck.
    assert plan.changed is False
    assert plan.summary().startswith("released 0 stale claim(s), 0 to review, 2 held")
    assert str(plan.held[0]) == f"#4: held (a container of run {run.id!r} is holding it)"


# --------------------------------------------------------------------------
# Against a real daemon
# --------------------------------------------------------------------------


CANDIDATE_IMAGES = ("apiary-worker", "busybox", "alpine", "python:3.12-slim")

#: Issue numbers no ledger in this file shares with the repository's own
#: backlog. A sweep lists every apiary container on the machine - that is the
#: point - so a real orchestrator's containers may well be in the listing, and
#: these numbers are what keeps them out of the assertions.
DEAD_ISSUE = 90210
LIVE_ISSUE = 90211


@pytest.fixture(scope="module")
def trivial_image() -> str:
    """A locally present image with a shell. Nothing is pulled."""
    docker = DockerCLI()
    for name in CANDIDATE_IMAGES:
        try:
            docker("image", "inspect", "--format", "{{.Id}}", name)
        except ContainerError:
            continue
        return name
    pytest.skip(
        "no local image to spawn a probe container from; build one with "
        "`docker build -f Dockerfile.worker -t apiary-worker .`"
    )


def make_manager(image: str, run: Run) -> ContainerManager:
    return ContainerManager(
        run=run,
        image=image,
        env={},
        limits=Limits(cpus=1.0, memory="256m", pids=64),
        timeout_s=60,
    )


@pytest.fixture()
def two_runs(trivial_image: str) -> Iterator[tuple[ContainerManager, ContainerManager]]:
    """Two managers under two freshly minted run ids, cleaned up either way."""
    dead = make_manager(trivial_image, make_run())
    live = make_manager(trivial_image, make_run())
    try:
        yield dead, live
    finally:
        for manager in (dead, live):
            for found in manager.find():
                manager.dispose(found)


@pytest.mark.docker
def test_a_real_dead_runs_container_does_not_hold_its_claim_and_a_live_ones_does(two_runs):
    dead, live = two_runs
    dead.spawn(
        ref(DEAD_ISSUE), BASE_COMMIT, issue=DEAD_ISSUE,
        entrypoint="/bin/sh", command=["-c", "sleep 300"],
    )
    live.spawn(
        ref(LIVE_ISSUE), BASE_COMMIT, issue=LIVE_ISSUE,
        entrypoint="/bin/sh", command=["-c", "sleep 300"],
    )
    client = FakeClient(
        issues={
            DEAD_ISSUE: issue_payload(DEAD_ISSUE),
            LIVE_ISSUE: issue_payload(LIVE_ISSUE),
        }
    )

    report = Recovery(client=client, run=live.run).startup()

    # The done-when against the real listing: the killed run's claim is back in
    # the pool with its attempt consumed, and the running worker's is untouched.
    assert report.ok, report.summary()
    assert client.labels_on(DEAD_ISSUE) == {READY}
    assert client.attempt_on(DEAD_ISSUE) == 1
    assert client.labels_on(LIVE_ISSUE) == {CLAIMED}
    assert [item.ref for item in report.plan.held] == [ref(LIVE_ISSUE)]


@pytest.mark.docker
def test_recovering_a_claim_leaves_the_orphaned_container_for_the_reaper(two_runs):
    dead, _ = two_runs
    dead.spawn(
        ref(DEAD_ISSUE), BASE_COMMIT, issue=DEAD_ISSUE,
        entrypoint="/bin/sh", command=["-c", "sleep 300"],
    )
    client = FakeClient(issues={DEAD_ISSUE: issue_payload(DEAD_ISSUE)})

    Recovery(client=client, run=make_run()).startup()

    # Both halves of the fault are needed and they belong to different tickets.
    # A second module issuing `docker rm` is how a container gets removed out
    # from under the sweep that was capturing its logs.
    assert [found.issue for found in dead.find()] == [DEAD_ISSUE]
    assert client.labels_on(DEAD_ISSUE) == {READY}
