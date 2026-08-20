"""Unit tests for readiness: `blocked` <-> `eligible`.

The five graphs `docs/architecture-v2.md`'s orchestration loop can meet are all
here - a linear chain, a diamond, a cycle, a dangling reference and a
dependency somebody closed as not planned - plus the writes each of them should
and should not produce.

Two of those are the failure modes this module exists for, and both are silent
without a test: a cycle, where every issue reads as legitimately blocked and
the run waits forever looking healthy, and a cancelled prerequisite, where
"closed" is mistaken for "done" and the dependants run against a foundation
nobody built.

No network and no token. `FakeClient` replays issue payloads shaped like
GitHub's; the ledgers are built by running the real parser over real issue
bodies, so a body that would not parse cannot become a fixture here. #31 will
lift the double into the shared fixture set.

The double still records label writes even though #152 left this module with
none to make: an empty `writes` is the assertion that the readiness pass is
still the read-only thing that ticket made it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pytest

from swarm.github.client import GitHubHTTPError
from swarm.github.ledger import Ledger, load_ledger, render_marker
from swarm.github.readiness import (
    BLOCKED,
    READY,
    DependencyCycleError,
    IssueState,
    apply_readiness,
    compute_readiness,
    find_cycle,
    referenced_refs,
    resolve_states,
)
from swarm.github.refs import issue_number
from swarm.github.refs import task_ref as ref
from swarm.orchestrator.authority import Belief
from swarm.orchestrator.derived import LANDED, STATES
from swarm.taskref import TaskRef
from fixtures.belief import fixture_belief


def believed_for(book) -> Belief:
    """The cycle's belief, from the state each fixture declares.

    Readiness was told which entries were its to speak about by reading
    `LedgerEntry.state_label`; since #152 the caller says, and what it says
    comes off the cycle's `Belief`. The fixtures stash their declaration on
    `labels` (`fixtures.belief`), next to a real repository label an issue
    would also carry, so the state is picked out by name rather than taken as
    whichever member of a `frozenset` came first.
    """
    entries = getattr(book, "entries", book)
    states = {
        task_id: next((name for name in getattr(entry, "labels", ()) if name in STATES), "")
        for task_id, entry in entries.items()
    }
    return fixture_belief(book, states=states, stored=dict(states), previous=dict(states))


# What a caller hands `compute_readiness` now that no label carries it: the
# waiting entries, and what they are believed to be waiting in.
def _compute_readiness_(book, *args, **kwargs):
    believed = believed_for(book)
    kwargs.setdefault("transitionable", believed.waiting())
    kwargs.setdefault("current", believed.states)
    return compute_readiness(book, *args, **kwargs)



# --------------------------------------------------------------------------
# Fixtures and helpers (#31 will lift these into the shared set)
# --------------------------------------------------------------------------


@dataclass
class FakeClient:
    """The four calls readiness makes, and a record of every label write.

    `hidden` holds payloads `list_issues` does not return - pull requests,
    which the real client filters out - so a `get_issue` fallback has something
    to find. Anything in neither list 404s, which is how a dangling reference
    reaches the code under test.
    """

    issues: list[dict[str, Any]]
    hidden: list[dict[str, Any]] = field(default_factory=list)
    writes: list[tuple[str, int, str]] = field(default_factory=list)
    listed: list[str] = field(default_factory=list)
    fetched: list[int] = field(default_factory=list)
    repo: str = "shahrestani-me/apiary"

    def list_issues(self, *, state: str = "open", **_: Any) -> list[dict[str, Any]]:
        self.listed.append(state)
        return [dict(issue) for issue in self.issues]

    def get_issue(self, number: int) -> dict[str, Any]:
        self.fetched.append(number)
        for payload in (*self.issues, *self.hidden):
            if payload["number"] == number:
                return dict(payload)
        raise GitHubHTTPError(404, "GET", f"/issues/{number}", b'{"message": "Not Found"}')

    def add_labels(self, number: int, labels: Iterable[str]) -> list[dict[str, Any]]:
        names = list(labels)
        self.writes.extend(("add", number, name) for name in names)
        return [{"name": name} for name in names]

    def remove_label(self, number: int, label: str) -> bool:
        self.writes.append(("remove", number, label))
        return True


def body(task_id: str, refs: Sequence[int] = ()) -> str:
    """A contract body in §6's shape, with `## Blocked by` filled in.

    The prose form is the backlog's own `_none — …_`, which parses to no
    dependencies at all rather than to a malformed section.
    """
    blocked = "\n".join(f"- #{ref}" for ref in refs) or "_none — this is the root._"
    return (
        f"{render_marker(task_id)}\n\n"
        f"## Goal\nShip {task_id}.\n\n"
        f"## Files\n- src/swarm/{task_id}.py\n\n"
        f"## Verify\npython -m pytest -q tests/test_{task_id}.py\n\n"
        f"## Blocked by\n{blocked}\n"
    )


def task(
    number: int,
    refs: Sequence[int] = (),
    *,
    label: str = BLOCKED,
    state: str = "open",
    state_reason: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """One issue payload, ledger-shaped and GitHub-shaped at the same time."""
    task_id = task_id or f"task-{number}"
    return {
        "number": number,
        "title": f"Task {number}",
        "body": body(task_id, refs),
        "labels": [{"name": label}, {"name": "area/control-plane"}],
        "state": state,
        "state_reason": state_reason,
    }


def done(number: int, refs: Sequence[int] = (), **kwargs: Any) -> dict[str, Any]:
    """A merged, closed-as-completed task - the only kind that unblocks."""
    kwargs.setdefault("label", LANDED)
    return task(number, refs, state="closed", state_reason="completed", **kwargs)


def plan_for(client: FakeClient, ledger: Ledger | None = None):
    """The whole path - parse, resolve, decide. There is nothing after it.

    `apply_readiness` wrote the two waiting labels until #152 and now returns
    the plan and stops, so this helper is the whole module rather than the part
    before the writes.
    """
    ledger = load_ledger(client) if ledger is None else ledger
    believed = believed_for(ledger)
    return apply_readiness(
        client,
        ledger=ledger,
        transitionable=believed.waiting(),
        current=believed.states,
    )


def verdicts_by_number(plan) -> dict[int, str]:
    """Verdicts back in issue-number terms, so the assertions read as before.

    The plan itself is keyed on `TaskRef` (#142); un-minting here rather than
    spelling `ref(41)` in thirty assertions keeps these tests about readiness.
    """
    return {issue_number(verdict.ref): verdict.state for verdict in plan.verdicts}


def refs(numbers) -> tuple[TaskRef, ...]:
    return tuple(ref(number) for number in numbers)


def verdict_for(plan, number: int):
    return next(verdict for verdict in plan.verdicts if verdict.ref == ref(number))


# --------------------------------------------------------------------------
# What satisfies a dependency
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state,reason,satisfied",
    [
        ("closed", "completed", True),
        # Issues closed before GitHub shipped `state_reason` carry none; reading
        # those as unfinished would block their dependants forever.
        ("closed", None, True),
        ("open", None, False),
        # Reopened: closed once, open now, and the work is not in the base branch.
        ("open", "reopened", False),
        ("closed", "not_planned", False),
        ("closed", "duplicate", False),
        # An unfamiliar reason GitHub adds later fails towards "wait", not "go".
        ("closed", "moved_to_another_planet", False),
    ],
)
def test_only_a_completed_closure_satisfies(state, reason, satisfied):
    assert IssueState(ref(7), state, reason).satisfied is satisfied


def test_a_missing_issue_satisfies_nothing_and_says_why():
    missing = IssueState.missing(ref(404))

    assert missing.satisfied is False
    assert missing.resolvable is False
    assert missing.reason == "#404 does not exist"


def test_a_pull_request_is_not_an_issue():
    """A merged PR is closed with no state reason, so the naive read says done."""
    payload = {"number": 37, "state": "closed", "state_reason": None, "pull_request": {}}

    state = IssueState.from_payload(payload)

    assert state.satisfied is False
    assert "pull request" in state.reason


# --------------------------------------------------------------------------
# The five graphs
# --------------------------------------------------------------------------


def test_a_linear_chain_releases_one_link_at_a_time():
    """#40 -> #41 -> #42: closing the root readies exactly one successor."""
    client = FakeClient([done(40), task(41, [40]), task(42, [41])])

    plan = plan_for(client)

    assert verdicts_by_number(plan) == {41: READY, 42: BLOCKED}
    assert plan.ready == refs([41])
    # The middle link's own dependency is met; #42 waits on #41 being *closed*,
    # which readiness never does - the merge does.
    assert [unmet.reason for unmet in verdict_for(plan, 42).unmet] == ["#41 is open"]


def test_closing_the_middle_link_advances_the_chain():
    client = FakeClient([done(40), done(41, [40]), task(42, [41])])

    assert plan_for(client).ready == refs([42])


def test_a_diamond_waits_for_both_sides():
    """The backlog's own diamond: #16 -> {#9, #31} -> #7.

    Closing the base readies both sides at once and the apex not at all. This
    is the shape v1's `fan_out` got right by accident, because a set
    intersection has no way to be half-satisfied - the version that reads
    GitHub can be, and must not be.
    """
    client = FakeClient(
        [done(7), task(9, [7]), task(31, [7]), task(16, [9, 31])]
    )

    plan = plan_for(client)

    assert verdicts_by_number(plan) == {9: READY, 31: READY, 16: BLOCKED}


def test_a_diamond_apex_waits_for_the_slower_side():
    client = FakeClient([done(7), done(9, [7]), task(31, [7]), task(16, [9, 31])])

    plan = plan_for(client)

    assert verdicts_by_number(plan) == {31: READY, 16: BLOCKED}
    assert [unmet.ref for unmet in verdict_for(plan, 16).unmet] == [ref(31)]


def test_a_cycle_raises_instead_of_reporting_everything_blocked():
    """The worst failure in the system: nothing ready, nothing wrong-looking.

    Every issue in the ring is honestly blocked, so a plan describing them is
    indistinguishable from a healthy backlog mid-run. There is no partial
    answer worth returning.
    """
    client = FakeClient([task(50, [51]), task(51, [50])])

    with pytest.raises(DependencyCycleError) as caught:
        plan_for(client)

    assert caught.value.cycle == refs([50, 51, 50])
    assert "#50 -> #51 -> #50" in str(caught.value)


def test_a_longer_ring_names_its_whole_path():
    client = FakeClient([task(50, [52]), task(51, [50]), task(52, [51])])

    with pytest.raises(DependencyCycleError) as caught:
        plan_for(client)

    assert caught.value.cycle == refs([50, 52, 51, 50])


def test_an_issue_blocked_by_itself_is_a_cycle():
    client = FakeClient([task(53, [53])])

    with pytest.raises(DependencyCycleError) as caught:
        plan_for(client)

    assert caught.value.cycle == refs([53, 53])


def test_a_cycle_reached_from_an_acyclic_prefix_is_still_found():
    """A ring nothing outside it depends on still hangs the issues inside it."""
    client = FakeClient([done(40), task(41, [40]), task(50, [51]), task(51, [50])])

    with pytest.raises(DependencyCycleError):
        plan_for(client)


def test_a_ring_of_completed_work_is_not_a_cycle():
    """An edge onto a closed dependency is discharged and hangs nothing.

    Two finished issues that reference each other are a tidy-up job, not a
    reason to fail a run whose backlog is in fact making progress.
    """
    client = FakeClient([done(50, [51]), done(51, [50]), task(52, [50])])

    plan = plan_for(client)

    assert plan.ready == refs([52])


def test_a_dangling_reference_blocks_and_is_reported():
    """Not satisfied, not silently dropped, and not this module's to fail.

    `swarm:failed` belongs to the reconciler (`docs/issue-contract.md` §4), so
    readiness holds the issue at `swarm:blocked` and hands the error up.
    """
    client = FakeClient([task(60, [999])])

    plan = plan_for(client)

    assert verdicts_by_number(plan) == {60: BLOCKED}
    assert [(e.task, e.ref) for e in plan.errors] == [(ref(60), ref(999))]
    assert verdict_for(plan, 60).errors[0].reason == "#999 does not exist"


def test_a_dangling_reference_pulls_a_ready_issue_back():
    """A ref retyped by hand must un-ready the issue, not be ignored as noise."""
    client = FakeClient([task(60, [999], label=READY)])

    plan = plan_for(client)

    assert [(v.ref, v.current_state, v.state) for v in plan.transitions] == [
        (ref(60), READY, BLOCKED)
    ]


def test_a_reference_to_a_pull_request_never_satisfies():
    """`#37` may be a merged PR: closed, no state reason, and not a task."""
    merged_pr = {
        "number": 37,
        "state": "closed",
        "state_reason": None,
        "pull_request": {"merged_at": "2026-08-01T00:00:00Z"},
    }
    client = FakeClient([task(61, [37])], hidden=[merged_pr])

    plan = plan_for(client)

    assert plan.ready == ()
    assert "pull request" in plan.errors[0].reason


def test_a_dependency_closed_as_not_planned_does_not_unblock():
    """"Closed" is not "done", and this is where the difference bites.

    A cancelled prerequisite that unblocks its dependants dispatches work
    against a foundation somebody explicitly decided not to build.
    """
    client = FakeClient(
        [task(70, state="closed", state_reason="not_planned"), task(71, [70])]
    )

    plan = plan_for(client)

    assert verdicts_by_number(plan) == {71: BLOCKED}
    assert verdict_for(plan, 71).unmet[0].reason == "#70 was closed as not planned"
    # A permanent block, but not an error: the ref resolves, and a human
    # repointing or reopening #70 is the fix.
    assert plan.errors == ()


def test_the_same_dependency_closed_as_completed_does_unblock():
    """The contrast that proves the test above is testing `state_reason`."""
    client = FakeClient(
        [task(70, state="closed", state_reason="completed"), task(71, [70])]
    )

    assert plan_for(client).ready == refs([71])


# --------------------------------------------------------------------------
# Which entries the pass speaks about, and which it moves
# --------------------------------------------------------------------------


def test_only_the_disagreements_are_transitions():
    client = FakeClient(
        [
            done(40),
            task(41, [40], label=BLOCKED),   # met, mislabelled: becomes ready
            task(42, [41], label=READY),     # unmet, mislabelled: becomes blocked
            task(43, [40], label=READY),     # met and already ready: no write
        ]
    )

    plan = plan_for(client)

    assert [(v.ref, v.state) for v in plan.transitions] == [(ref(41), READY), (ref(42), BLOCKED)]
    assert len(plan.verdicts) == 3


# Deleted with #152: `test_a_label_is_added_before_the_old_one_is_removed`
# pinned the order of readiness's two label calls - add `swarm:ready`, then
# remove `swarm:blocked` - so that a crash between them left an issue wearing
# two labels rather than none. `_relabel` is gone and `apply_readiness` makes no
# write at all, so there is no pair of calls left to order and no way for this to
# fail. What replaces it is the assertion below that the pass writes nothing.


@pytest.mark.parametrize("state", ["claimed", "review", "landed", "needs-human"])
def test_issues_in_another_state_get_no_verdict(state):
    """Readiness owns two rows of §4's table and speaks about nothing else.

    A claimed issue whose dependency graph changed still has a container
    working on it; a verdict about it would be this module telling the cycle
    that an issue somebody is running is merely waiting.
    """
    client = FakeClient([task(80, [999], label=state)])

    plan = plan_for(client)

    assert plan.verdicts == ()
    # The dangling ref is still reported - the entry is out of scope for the
    # label, not for the error.
    assert [e.task for e in plan.errors] == [ref(80)]


def test_a_task_a_human_closed_is_never_readied():
    """`docs/architecture-v2.md`: closing a task mid-run is a supported edit.

    Its dependency is met and the cycle still believes it is waiting, so the
    naive pass calls it ready and the dispatcher resurrects work somebody
    cancelled on purpose.
    """
    client = FakeClient(
        [done(40), task(41, [40], state="closed", state_reason="not_planned")]
    )

    plan = plan_for(client)

    assert plan.verdicts == ()


# Deleted with #152: `test_a_cycle_is_detected_before_anything_is_written` ran a
# ring through `apply_readiness` and asserted the label writes never happened.
# The pass has no writes to suppress, and the assertion it would become - that a
# function which never writes did not write - cannot fail. That a ring raises
# instead of returning a plan is asserted by
# `test_a_cycle_raises_instead_of_reporting_everything_blocked` and
# `test_a_cycle_reached_from_an_acyclic_prefix_is_still_found`, both over the
# same whole path; that the pass writes nothing is
# `test_a_pass_computes_the_plan_and_writes_nothing`.


def test_a_pass_computes_the_plan_and_writes_nothing():
    """#152's whole claim about this module, and the reason `dry_run` went.

    The parameter guarded the two label calls; with those gone every pass is
    what `dry_run=True` used to be, so the property is asserted unconditionally
    - the double still records writes, and a regression that put one back would
    land here.
    """
    client = FakeClient([done(40), task(41, [40])])

    plan = plan_for(client)

    assert plan.ready == refs([41])
    assert plan.transitions[0].changed is True
    assert client.writes == []


def test_a_prebuilt_ledger_is_not_re_read():
    """The reconcile loop (#22) already holds a ledger; readiness reuses it."""
    client = FakeClient([done(40), task(41, [40])])
    ledger = load_ledger(client)
    client.listed.clear()

    plan_for(client, ledger)

    # One list call, for resolving the referenced issues' open/closed state.
    assert client.listed == ["all"]


# --------------------------------------------------------------------------
# Resolving state from the tracker
# --------------------------------------------------------------------------


def test_resolution_reads_closed_issues_too():
    """`state="open"` would report every met dependency as unmet."""
    client = FakeClient([done(40), task(41, [40])])

    plan_for(client)

    assert set(client.listed) == {"all"}


def test_a_reference_outside_the_listing_is_fetched_once():
    client = FakeClient([task(61, [37])], hidden=[{"number": 37, "state": "open"}])

    states = resolve_states(client, refs([37, 61]))

    assert client.fetched == [37]
    assert states[ref(37)].satisfied is False
    assert states[ref(61)].state == "open"


def test_a_404_becomes_a_missing_issue_rather_than_an_exception():
    client = FakeClient([task(60, [999])])

    states = resolve_states(client, referenced_refs(load_ledger(client)))

    assert states[ref(999)].exists is False


def test_other_http_errors_are_not_swallowed():
    """A 403 is "the token cannot see this", which is not "it is not there"."""

    class Forbidden(FakeClient):
        def get_issue(self, number: int) -> dict[str, Any]:
            raise GitHubHTTPError(403, "GET", f"/issues/{number}", b'{"message": "Forbidden"}')

    with pytest.raises(GitHubHTTPError):
        resolve_states(Forbidden([task(60, [999])]), refs([999]))


def test_nothing_referenced_means_no_api_call_at_all():
    client = FakeClient([task(40)])

    assert resolve_states(client, []) == {}
    assert client.listed == [] and client.fetched == []


def test_find_cycle_returns_none_for_a_forest():
    forest = {ref(1): refs([2]), ref(2): refs([3]), ref(3): (), ref(4): refs([3])}
    assert find_cycle(forest) is None


def test_a_reference_nobody_resolved_blocks_rather_than_passes():
    """`compute_readiness` is pure, and its unknowns fail towards "wait".

    A caller that hands over a states map missing a reference has a bug, and
    the reading that survives one is the one where the issue does not run.
    """
    client = FakeClient([task(41, [40])])
    ledger = load_ledger(client)

    plan = _compute_readiness_(ledger, {})

    assert plan.blocked == refs([41])
    assert plan.errors[0].ref == ref(40)


# --------------------------------------------------------------------------
# The real corpus: this repository's own v2 backlog
# --------------------------------------------------------------------------

# Every `## Blocked by` edge in issues #6-#35 as the backlog carried them, with
# the state each issue was in. One root (#6), a chain through #7 and #8, and the
# diamond #16 -> {#9, #31} -> #7. Closed entries are closed as completed.
BACKLOG: Mapping[int, tuple[int, ...]] = {
    6: (), 7: (6,), 8: (7,), 9: (6, 7), 10: (9, 31), 11: (9,), 12: (7,),
    13: (12, 32), 14: (12,), 15: (14, 33), 16: (9, 14, 31), 17: (7, 16),
    18: (16,), 19: (15,), 20: (15, 33), 21: (11, 15, 16), 22: (21,),
    23: (17, 22), 24: (10, 22), 25: (7, 8), 26: (25,), 27: (24, 26, 29, 34),
    28: (12, 16), 29: (15, 18, 33), 31: (7,), 32: (7, 31), 33: (9,),
    34: (23,), 35: (20, 22),
}
MERGED = (6, 7, 8, 9, 12)


def backlog_client(*, merged: Sequence[int] = MERGED) -> FakeClient:
    return FakeClient(
        [
            done(number, refs) if number in merged else task(number, refs)
            for number, refs in BACKLOG.items()
        ]
    )


def test_the_real_backlog_is_acyclic():
    """The corpus the contract doc names, walked by the real cycle detector."""
    assert find_cycle({ref(n): refs(edges) for n, edges in BACKLOG.items()}) is None


def test_the_real_backlog_readies_exactly_the_issues_whose_work_landed():
    """With #6-#9 and #12 merged, five issues come free and the rest wait.

    #11 is in that set and is this ticket: its only dependency is #9, the
    ledger loader, which merged. #10 is not, because #31 has not.
    """
    plan = plan_for(backlog_client())

    assert plan.ready == refs([11, 14, 25, 31, 33])
    assert ref(10) in plan.blocked and ref(16) in plan.blocked
    assert plan.errors == ()


def test_a_cancelled_root_freezes_everything_below_it_in_the_real_backlog():
    """#31 closed as not planned: #10, #16 and #32 stay blocked, not released."""
    client = backlog_client()
    for payload in client.issues:
        if payload["number"] == 31:
            payload.update(state="closed", state_reason="not_planned")

    plan = plan_for(client)

    assert ref(31) not in plan.ready
    assert set(refs([10, 16, 32])).issubset(set(plan.blocked))


def test_one_bad_reference_in_the_backlog_does_not_stop_the_others():
    """§1.4's policy, applied to references: report it, keep the cycle going."""
    client = backlog_client()
    client.issues.append(task(36, [12345]))

    plan = plan_for(client)

    assert [e.task for e in plan.errors] == [ref(36)]
    assert ref(11) in plan.ready


# --------------------------------------------------------------------------
# Identity is opaque: a ref this adapter did not mint
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ForeignEntry:
    """A ledger entry from a tracker whose ids are not numbers (#142).

    Deliberately *not* a `LedgerEntry`: that type is the GitHub adapter's
    record and mints its ref from an issue number, so building the graph out of
    one could never demonstrate the property under test. This carries only the
    three fields readiness reads, which is itself the assertion - anything more
    the graph reached for would be a piece of GitHub it has no business
    knowing.

    `labels` is the fourth and is not one of them: since #152 it is the
    fixture's *declaration* of what state the entry is in, which `believed_for`
    turns into the belief the caller passes. Readiness never looks at it.
    """

    ref: TaskRef
    task_id: str
    labels: frozenset[str] = frozenset()
    blocked_by: tuple[TaskRef, ...] = ()


def linear_ledger():
    """`ENG-9 -> ENG-10 -> ENG-11`, with the root already delivered."""
    entries = [
        ForeignEntry(TaskRef("ENG-9"), "eng-9", frozenset({LANDED})),
        ForeignEntry(TaskRef("ENG-10"), "eng-10", frozenset({BLOCKED}), (TaskRef("ENG-9"),)),
        ForeignEntry(TaskRef("ENG-11"), "eng-11", frozenset({BLOCKED}), (TaskRef("ENG-10"),)),
    ]
    return Ledger(entries={entry.task_id: entry for entry in entries})


def test_a_non_numeric_ref_survives_the_readiness_graph():
    """The property this ticket buys: nothing in the graph reads a ref.

    Edges, cycle detection, the satisfied test, the verdicts and the plan's
    `ready`/`blocked` all run over `ENG-*`, which no amount of `int()` would
    survive.
    """
    states = {
        TaskRef("ENG-9"): IssueState(TaskRef("ENG-9"), "closed", "completed"),
        TaskRef("ENG-10"): IssueState(TaskRef("ENG-10")),
        TaskRef("ENG-11"): IssueState(TaskRef("ENG-11")),
    }

    plan = _compute_readiness_(linear_ledger(), states)

    assert plan.ready == (TaskRef("ENG-10"),)
    assert plan.blocked == (TaskRef("ENG-11"),)
    assert [str(verdict) for verdict in plan.verdicts] == [
        "ENG-10 eng-10: ready",
        "ENG-11 eng-11: blocked - ENG-10 is open",
    ]


def test_a_cycle_of_non_numeric_refs_is_still_a_cycle():
    """The one failure that must never pass silently, on a foreign tracker."""
    entries = [
        ForeignEntry(TaskRef("ENG-1"), "eng-1", frozenset({BLOCKED}), (TaskRef("ENG-2"),)),
        ForeignEntry(TaskRef("ENG-2"), "eng-2", frozenset({BLOCKED}), (TaskRef("ENG-1"),)),
    ]
    ledger = Ledger(entries={entry.task_id: entry for entry in entries})

    with pytest.raises(DependencyCycleError) as caught:
        _compute_readiness_(ledger, {})

    assert caught.value.cycle == (TaskRef("ENG-1"), TaskRef("ENG-2"), TaskRef("ENG-1"))
    assert "ENG-1 -> ENG-2 -> ENG-1" in str(caught.value)


def test_a_foreign_ref_never_reaches_the_github_api_by_accident():
    """`resolve_states` is the adapter half, and it says so rather than guessing.

    A ref another tracker minted has no issue number in it, and the failure
    mode of inventing one - addressing an unrelated issue in this repository -
    is worse than the exception.
    """
    with pytest.raises(ValueError, match="not minted by the GitHub adapter"):
        resolve_states(FakeClient([]), [TaskRef("ENG-9")])
