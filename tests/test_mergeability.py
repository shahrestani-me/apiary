"""Tests for the base-branch gate.

Five properties carry this file, and each of them is a way the run stalls rather
than a way it computes wrong.

**A green pull request behind its base is updated, not merged.** The issue's own
acceptance criterion. Checks that passed against a commit that is no longer the
base have verified nothing about the merge, and admitting one is the bug this
module exists to prevent - so the merge #23 planned is taken away and the branch
is dragged forward instead.

**Merges are serialised.** Under `strict_required_status_checks_policy` every
merge invalidates every other open pull request, so two green ones in a cycle
means one merge and one deliberate wait. The second is *held*, not failed: it is
fine, it is just not this cycle's.

**A conflict is not a CI failure.** Retrying the same diff cannot fix it, so the
issue goes back to `swarm:ready` with the attempt counter bumped and the base it
conflicted with written onto the body, which is what makes the next attempt
start from a fresh base rather than reproduce the same collision.

**Nothing parks in `swarm:review` forever.** The update-and-recheck loop is
capped per pull request, and a branch that has been invalidated by faster
siblings that many times goes to a human with `swarm:failed`. The cap is charged
for every update this cycle *planned*, including the one a client with no update
method could not make - otherwise a missing method is an infinite wait charged to
nobody.

**Absence of evidence is not evidence.** Mergeability GitHub has not finished
computing is never read as a clearance, and a cycle that could not look admits
no merge at all.

Hermetic throughout: a plain fake client, no HTTP, no daemon. The one test that
touches git uses `fixtures/repo.py`'s bare-repo origin, and it is there to prove
that the states the fake payloads simulate are the states git actually produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, cast

import pytest

from swarm.github.branches import task_branch
from swarm.github.client import GitHubHTTPError
from swarm.github.ledger import Ledger, LedgerEntry, render_marker
from swarm.github.refs import task_ref
from swarm.orchestrator.checks import (
    CheckSet,
    Merge,
    Outcome,
    PullState,
    UnresolvedJoin,
    plan_checks,
    summarise_checks,
)
from swarm.orchestrator.dispatcher import CLAIMED, REVIEW
from swarm.orchestrator.mergeability import (
    BEHIND,
    COMPUTING,
    CONFLICT_CLOSE,
    CONFLICT_OPEN,
    CONFLICTED,
    FRESH,
    ROUNDS_ENV,
    UPDATE_METHODS,
    Mergeability,
    MergeabilityPlan,
    UpdateBudget,
    UpdatePolicy,
    _admit,
    apply_mergeability,
    conflict_context,
    plan_mergeability,
    read_conflict,
    read_mergeability,
    read_touched_files,
    run_mergeability,
    write_conflict,
)
from swarm.orchestrator.reconcile import DONE, FAILED, READY
from swarm.taskref import TaskRef


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def entry(
    number: int,
    *files: str,
    label: str = REVIEW,
    attempt: int = 0,
) -> LedgerEntry:
    return LedgerEntry(
        number=number,
        title=f"issue {number}",
        task_id=f"task-{number}",
        attempt=attempt,
        goal="do the thing",
        files=files or (f"src/mod{number}.py", f"tests/test_mod{number}.py"),
        verify="python -m pytest -q",
        blocked_by=(),
        state_label=label,
        labels=frozenset({label}),
    )


def ledger(*entries: LedgerEntry) -> Ledger:
    return Ledger(entries={item.task_id: item for item in entries})


#: Pull request numbers, handed out in the order `green()` is given issues, so
#: #23 is always PR #101 and #24 always PR #102. The two numberings are kept
#: visibly apart because `docs/issue-contract.md` §2 makes the issue number the
#: addressing key and a test that used one where it meant the other would pass
#: for the wrong reason.
PULL_NUMBERS = (101, 102, 103, 104)


def branch(number: int, attempt: int = 0) -> str:
    """The head ref a worker for `(number, attempt)` pushed - built, not spelled.

    Spelling the name out would make these tests assert #144's encoding rather
    than the merge gate's behaviour; the encoding is `test_branches.py`'s.
    """
    return task_branch(task_ref(number), attempt)


def pull(number: int, *, issue: int, attempt: int = 0) -> PullState:
    # The attempt has to match the ledger entry's: a branch name carries it
    # (#144) and the gate finds a pull request by `LedgerEntry.branch`. That is
    # the invariant, not a test detail - a task in `swarm:review` still carries
    # the attempt its open pull request was pushed from, because an exit 0
    # moves no counter (`reconcile._observe`).
    return PullState(number=number, branch=branch(issue, attempt), sha=f"{issue:0>40x}")


def pulls(*states: PullState) -> dict[str, PullState]:
    return {state.branch: state for state in states}


def state(
    pull_number: int,
    *,
    issue: int,
    mergeable: bool | None = True,
    mergeable_state: str = "clean",
) -> Mergeability:
    return Mergeability(
        number=pull_number,
        branch=branch(issue),
        mergeable=mergeable,
        state=mergeable_state,
        base="main",
        base_sha="abcdef1234567890",
        head_sha=f"{issue:0>40x}",
    )


def behind(pull_number: int, *, issue: int) -> Mergeability:
    """GitHub's answer for a branch under a strict policy whose base has moved."""
    return state(pull_number, issue=issue, mergeable=True, mergeable_state="behind")


def conflicted(pull_number: int, *, issue: int) -> Mergeability:
    return state(pull_number, issue=issue, mergeable=False, mergeable_state="dirty")


def green(*numbers: int, ledger_: Ledger | None = None, states: dict[TaskRef, Mergeability]):
    """#23's plan for green pull requests, plus this module's gate on it.

    Every interesting case in this file is "checks passed, now what does the base
    say", so building the two plans together is the shape most tests want.
    """
    tasks = ledger_ or ledger(*(entry(number) for number in numbers))
    # Read back off the ledger rather than assumed to be 0, so a test that
    # hands in an entry mid-budget gets a pull request on that attempt's branch.
    attempts = {item.number: item.attempt for item in tasks.entries.values()}
    checks = plan_checks(
        tasks,
        pulls=pulls(*(
            pull(pr, issue=number, attempt=attempts.get(number, 0))
            for pr, number in zip(PULL_NUMBERS, numbers)
        )),
        checks={task_ref(number): summarise_checks([{"name": "test", "status": "completed",
                                                    "conclusion": "success"}])
                for number in numbers},
    )
    return tasks, checks


def issue_payload(number: int, *, label: str = REVIEW, attempt: int = 0) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"issue {number}",
        "labels": [{"name": label}],
        "body": "\n".join(
            [
                render_marker(f"task-{number}", attempt),
                "",
                "## Goal",
                "Do the thing.",
                "",
                "## Files",
                f"- src/mod{number}.py",
                "",
                "## Verify",
                "python -m pytest -q",
                "",
                "## Blocked by",
                "_none._",
            ]
        ),
    }


@dataclass
class FakeClient:
    """Every call this module makes, recorded, with no HTTP anywhere.

    Deliberately without `update_branch` and without `create_issue_comment`:
    that is the state `GitHubClient` is actually in, so the default client here
    exercises the degraded path and the subclass below is what lands once
    `client.py` grows the method.
    """

    issues: dict[int, dict[str, Any]] = field(default_factory=dict)
    open_pulls: tuple[tuple[int, str], ...] = ()
    payloads: dict[int, dict[str, Any]] = field(default_factory=dict)
    files: dict[int, list[str]] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    # --- reads -----------------------------------------------------------

    def list_pull_requests(self, *, state: str = "open") -> list[dict[str, Any]]:
        self.log.append(f"list_pull_requests {state}")
        return [{"number": number, "head": {"ref": ref}} for number, ref in self.open_pulls]

    def get_pull_request(self, number: int) -> dict[str, Any]:
        self.log.append(f"get_pull_request #{number}")
        payload = self.payloads.get(number)
        if payload is None:
            raise GitHubHTTPError(404, "GET", f"/pulls/{number}", b'{"message":"Not Found"}')
        return dict(payload)

    def list_pull_request_files(self, number: int) -> list[dict[str, Any]]:
        self.log.append(f"list_pull_request_files #{number}")
        return [{"filename": name} for name in self.files.get(number, ())]

    def get_issue(self, number: int) -> dict[str, Any]:
        self.log.append(f"get_issue #{number}")
        if number not in self.issues:
            raise GitHubHTTPError(404, "GET", f"/issues/{number}", b'{"message":"Not Found"}')
        return dict(self.issues[number])

    # --- writes ----------------------------------------------------------

    def update_issue(self, number: int, **kwargs: Any) -> dict[str, Any]:
        self.log.append(f"update_issue #{number}")
        self.issues[number].update(kwargs)
        return dict(self.issues[number])

    def add_labels(self, number: int, labels: Iterable[str]) -> Any:
        names = list(labels)
        self.log.append(f"+{','.join(names)} #{number}")
        current = self.issues.setdefault(number, {}).setdefault("labels", [])
        current.extend({"name": name} for name in names)
        return current

    def remove_label(self, number: int, label: str) -> bool:
        self.log.append(f"-{label} #{number}")
        payload = self.issues.get(number)
        if payload is None:
            return False
        payload["labels"] = [item for item in payload.get("labels", []) if item["name"] != label]
        return True

    # --- what the assertions read ----------------------------------------

    def labels_on(self, number: int) -> set[str]:
        return {item["name"] for item in self.issues[number].get("labels", [])}


@dataclass
class UpdatingClient(FakeClient):
    """The client once `client.py` grows `PUT /pulls/{n}/update-branch`."""

    updated: list[int] = field(default_factory=list)
    update_error: Exception | None = None

    def update_branch(self, number: int) -> dict[str, Any]:
        self.log.append(f"update_branch #{number}")
        if self.update_error is not None:
            raise self.update_error
        self.updated.append(number)
        return {"message": "Updating pull request branch."}


@dataclass
class CommentingClient(UpdatingClient):
    """The client once §1.4's comment method exists."""

    comments: list[tuple[int, str]] = field(default_factory=list)

    def create_issue_comment(self, number: int, text: str) -> dict[str, Any]:
        self.comments.append((number, text))
        return {"id": len(self.comments)}


def pr_payload(
    number: int,
    *,
    issue: int,
    mergeable: bool | None = True,
    mergeable_state: str = "clean",
) -> dict[str, Any]:
    return {
        "number": number,
        "head": {"ref": branch(issue), "sha": f"{issue:0>40x}"},
        "base": {"ref": "main", "sha": "abcdef1234567890"},
        "mergeable": mergeable,
        "mergeable_state": mergeable_state,
    }


# --------------------------------------------------------------------------
# Reading what GitHub says about the merge
# --------------------------------------------------------------------------


def test_a_mergeability_github_has_not_computed_yet_decides_nothing():
    # `mergeable` is null on the first read of a pull request while a background
    # job runs. Reading that as "not mergeable" would re-dispatch a healthy
    # branch every time GitHub was slow.
    assert Mergeability(1, mergeable=None, state="unknown").verdict == COMPUTING
    assert Mergeability(1, mergeable=None, state="clean").verdict == COMPUTING
    assert Mergeability(1, unreadable=True).verdict == COMPUTING


def test_the_three_states_this_module_folds_and_the_ones_it_leaves_alone():
    assert Mergeability(1, mergeable=True, state="behind").verdict == BEHIND
    assert Mergeability(1, mergeable=False, state="dirty").verdict == CONFLICTED
    assert Mergeability(1, mergeable=True, state="clean").verdict == FRESH
    # `blocked` is a required check or review that has not landed and `unstable`
    # is a non-required check that failed. Both are #23's question, and two
    # modules with an opinion about CI is how an issue is relabelled twice.
    assert Mergeability(1, mergeable=True, state="blocked").verdict == FRESH
    assert Mergeability(1, mergeable=True, state="unstable").verdict == FRESH


def test_mergeability_comes_from_the_single_pull_request_read_not_the_listing():
    client = FakeClient(payloads={101: pr_payload(101, issue=23, mergeable_state="behind")})

    found = read_mergeability(client, pull(101, issue=23))

    # The listing carries no `mergeable` at all, which is why this costs one
    # request per review issue rather than riding along on #23's listing.
    assert found.verdict == BEHIND
    assert found.base_name == "main@abcdef1"
    assert client.log == ["get_pull_request #101"]


def test_a_pull_request_that_could_not_be_read_is_unreadable_not_fresh():
    found = read_mergeability(FakeClient(), pull(101, issue=23))

    assert found.unreadable
    assert found.verdict == COMPUTING


def test_the_touched_files_are_probed_for_and_absence_is_an_answer():
    client = FakeClient(files={101: ["src/mod23.py", "README.md"]})
    assert read_touched_files(client, pull(101, issue=23)) == ("src/mod23.py", "README.md")

    class NoFiles(FakeClient):
        list_pull_request_files = None  # type: ignore[assignment]

    assert read_touched_files(NoFiles(), pull(101, issue=23)) == ()


# --------------------------------------------------------------------------
# Behind the base
# --------------------------------------------------------------------------


def test_a_green_pull_request_behind_its_base_is_updated_and_not_merged():
    tasks, checks = green(23, states={})
    assert checks.merges  # #23 would have merged it

    plan = plan_mergeability(tasks, checks, states={task_ref(23): behind(101, issue=23)})

    # The headline. "Checks passed on a stale base" is not mergeable, however
    # green the pull request looks, because nothing verified it against this base.
    assert plan.merges == ()
    assert plan.held == (23,)
    assert [str(u) for u in plan.updates] == [
        f"#23: update PR #101 ({branch(23)}) from main, round 1 of 3"
    ]
    # And the `swarm:done` #23 planned goes with the merge: a label written for a
    # merge that did not happen is a lie nothing later can detect.
    assert plan.admitted.transitions == ()


def test_a_branch_behind_its_base_is_never_retried_only_updated():
    tasks, checks = green(23, states={})

    plan = plan_mergeability(tasks, checks, states={task_ref(23): behind(101, issue=23)})

    # The diff is fine; only its base is old. Throwing the work away to
    # reproduce it against a base that will have moved again is the expensive
    # way to make no progress.
    assert plan.transitions == ()


def test_a_fresh_pull_request_is_handed_straight_back_to_the_check_gate():
    tasks, checks = green(23, states={})

    plan = plan_mergeability(tasks, checks, states={task_ref(23): state(101, issue=23)})

    assert plan.updates == ()
    assert plan.held == ()
    assert [m.pull for m in plan.merges] == [101]
    assert plan.admitted.transitions[0].to_label == DONE


# --------------------------------------------------------------------------
# Serialising the merges
# --------------------------------------------------------------------------


def test_two_green_pull_requests_merge_one_at_a_time():
    tasks, checks = green(23, 24, states={})

    plan = plan_mergeability(
        tasks,
        checks,
        states={task_ref(23): state(101, issue=23), task_ref(24): state(102, issue=24)},
    )

    # Merging both would leave the second one stale the instant the first landed,
    # which under a strict status policy means it cannot merge at all.
    assert [m.number for m in plan.merges] == [23]
    assert plan.held == (24,)
    # Held is not failed: #24 is fine and goes next cycle.
    assert plan.transitions == ()
    assert task_ref(24) not in [t.ref for t in plan.admitted.transitions]


def test_the_pull_request_closest_to_starving_gets_the_merge_slot():
    tasks, checks = green(23, 24, states={})
    budget = UpdateBudget(cap=3, rounds={task_ref(24): 2})

    plan = plan_mergeability(
        tasks,
        checks,
        states={task_ref(23): state(101, issue=23), task_ref(24): state(102, issue=24)},
        budget=budget,
    )

    # #24 has been dragged forward twice already, so it is the one closest to
    # being given up on. Serving it first is what keeps the cap a bound nothing
    # normally reaches rather than the queue's natural end state.
    assert [m.number for m in plan.merges] == [24]
    assert plan.held == (23,)


def test_a_repository_without_a_strict_policy_can_merge_more_than_one():
    tasks, checks = green(23, 24, states={})

    plan = plan_mergeability(
        tasks,
        checks,
        states={task_ref(23): state(101, issue=23), task_ref(24): state(102, issue=24)},
        policy=UpdatePolicy(merges_per_cycle=2),
    )

    assert [m.number for m in plan.merges] == [23, 24]


def test_no_branch_is_updated_in_a_cycle_that_merges():
    tasks, checks = green(23, 24, states={})

    plan = plan_mergeability(
        tasks,
        checks,
        states={task_ref(23): state(101, issue=23), task_ref(24): behind(102, issue=24)},
    )

    # #24's update would be undone by #23's merge the moment it landed, and it
    # would charge a round for the privilege. Next cycle updates it against the
    # base this merge creates.
    assert [m.number for m in plan.merges] == [23]
    assert plan.updates == ()
    assert "deferred" in [d for d in plan.decisions if d.number == 24][0].detail


# --------------------------------------------------------------------------
# Conflicts
# --------------------------------------------------------------------------


def test_a_conflict_re_dispatches_from_a_fresh_base_rather_than_retrying_the_diff():
    tasks, checks = green(23, states={})

    plan = plan_mergeability(
        tasks,
        checks,
        states={task_ref(23): conflicted(101, issue=23)},
        files={task_ref(23): ("src/mod23.py",)},
    )
    decision = plan.decisions[0]

    assert plan.merges == ()
    assert decision.transition is not None
    assert decision.transition.to_label == READY
    # §5: the counter rides on the transition, so it is persisted before the
    # issue can be dispatched again. A conflict *is* the attempt's problem, which
    # is why it costs one - unlike the starvation case below.
    assert decision.transition.attempt == 1
    assert "fresh base commit" in decision.transition.reason
    # The half without which the next attempt collides all over again.
    assert "main@abcdef1" in decision.context
    assert "src/mod23.py" in decision.context


def test_a_conflict_at_the_attempt_cap_goes_to_a_human():
    tasks = ledger(entry(23, attempt=2))
    _, checks = green(23, ledger_=tasks, states={})

    plan = plan_mergeability(
        tasks, checks, states={task_ref(23): conflicted(101, issue=23)}, max_attempts=3
    )
    transition = plan.decisions[0].transition

    assert transition is not None
    assert transition.to_label == FAILED
    assert "against a cap of 3" in transition.reason


def test_the_conflict_context_names_the_declared_files_when_the_client_cannot_list_them():
    facts = conflicted(101, issue=23)

    context = conflict_context(entry(23, "src/mod23.py", "tests/test_mod23.py"), facts)

    # The declared set is the honest fallback rather than a guess: it is the only
    # set the worker is allowed to edit anyway.
    assert "src/mod23.py" in context and "tests/test_mod23.py" in context
    assert "re-apply the change on top" in context


# --------------------------------------------------------------------------
# Starvation
# --------------------------------------------------------------------------


def test_a_pull_request_invalidated_too_often_is_handed_over_rather_than_parked():
    tasks, checks = green(23, states={})
    budget = UpdateBudget(cap=3, rounds={task_ref(23): 3})

    plan = plan_mergeability(
        tasks, checks, states={task_ref(23): behind(101, issue=23)}, budget=budget
    )
    transition = plan.decisions[0].transition

    # The starvation case: three updates, three siblings merging first, no
    # progress. Parking it in `swarm:review` for the rest of the run is the
    # failure mode this ticket exists to remove.
    assert plan.updates == ()
    assert transition is not None
    assert transition.to_label == FAILED
    assert "starved" in transition.reason
    # The attempt budget is untouched: nothing about the work was wrong, and
    # charging it would take the retry away from whatever goes wrong next.
    assert transition.attempt is None


def test_the_round_cap_is_configurable_and_loud_about_a_value_it_cannot_read(monkeypatch):
    monkeypatch.setenv(ROUNDS_ENV, "5")
    assert UpdatePolicy.from_env().max_update_rounds == 5

    monkeypatch.setenv(ROUNDS_ENV, "none")
    # A silently defaulted cap is a starvation bound nobody chose.
    with pytest.raises(ValueError):
        UpdatePolicy.from_env()


def test_the_budget_forgets_a_pull_request_that_is_no_longer_in_review():
    budget = UpdateBudget(cap=3, rounds={task_ref(23): 2, task_ref(24): 1})

    budget.retain([task_ref(23)])

    # #24's next pull request must not inherit the rounds its last one spent.
    assert budget.spent(task_ref(24)) == 0
    assert budget.spent(task_ref(23)) == 2


# --------------------------------------------------------------------------
# What it refuses to decide
# --------------------------------------------------------------------------


def test_a_cycle_that_could_not_read_mergeability_admits_no_merge():
    tasks, checks = green(23, states={})

    plan = plan_mergeability(tasks, checks, states=None)

    # "We could not look" must never read as "the base has not moved". The pull
    # request keeps `swarm:review` and the next cycle asks again.
    assert plan.blind
    assert plan.merges == ()
    assert plan.transitions == ()
    assert plan.updates == ()
    assert "nothing merged" in plan.summary()


def test_mergeability_still_being_computed_holds_the_merge_without_failing_anything():
    tasks, checks = green(23, states={})

    plan = plan_mergeability(
        tasks,
        checks,
        states={
            task_ref(23): state(101, issue=23, mergeable=None, mergeable_state="unknown")
        },
    )

    assert plan.merges == ()
    assert plan.transitions == ()
    assert plan.decisions[0].verdict == COMPUTING


def test_an_issue_the_check_gate_already_decided_is_not_decided_twice():
    tasks = ledger(entry(23), entry(24, label=CLAIMED))
    checks = plan_checks(
        tasks,
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): summarise_checks([{"name": "test", "status": "completed",
                                       "conclusion": "failure",
                                       "output": {"summary": "FAILED tests/test_mod23.py"}}])},
    )
    assert checks.transitions[0].to_label == READY  # #23 is already being retried

    plan = plan_mergeability(tasks, checks, states={task_ref(23): conflicted(101, issue=23)})

    # A red pull request is already going back to `swarm:ready` with a fresh
    # base; adding a second transition for the same issue in the same cycle is
    # how a label moves twice.
    assert plan.decisions == ()
    assert plan.transitions == ()


def test_a_pull_request_with_no_planned_merge_is_held_back_by_nothing():
    tasks = ledger(entry(23))
    checks = plan_checks(tasks, pulls=pulls(pull(101, issue=23)), checks={task_ref(23): CheckSet(
        total=1, pending=("test",))})

    plan = plan_mergeability(tasks, checks, states={task_ref(23): state(101, issue=23)})

    # Its checks are still running, so there was no merge to hold and reporting
    # one would be noise on every cycle of every pull request.
    assert plan.held == ()


# --------------------------------------------------------------------------
# The joins: outcome -> ledger, and hold -> outcome (#174)
# --------------------------------------------------------------------------
#
# Two lookups decide whether this module runs at all, and both used to fail
# open in the direction that admits a merge:
#
# - `plan_mergeability` resolved each outcome against a private int index of
#   the ledger and `continue`d on a miss. An issue that fell out that way got
#   no `Decision`, so nothing held its merge, so #23's `swarm:done` and the
#   merge itself went through - the staleness gate switched off for that pull
#   request, with no log line saying so.
# - `_admit` matched the holds it *did* produce back onto #23's outcomes by
#   number. A hold that matched nothing was simply not applied, which is the
#   same merge admitted by a different route, and this one after the gate had
#   already decided against it.
#
# Neither is visible from outside: the merge looks clean, the issue looks done,
# and the only trace is a pull request that landed on a base it was never
# checked against. So every assertion below is on the miss.


def test_an_outcome_the_ledger_cannot_resolve_raises_rather_than_dropping_it():
    """The headline. #23 is green and #23's merge is planned; the ledger handed
    to this gate does not carry it. Skipping is the one thing that must not
    happen, because a skipped issue is a merge nothing here inspected.

    Restore the `entries.get(...) / continue` and this stops raising - and the
    plan it returns then carries the merge, which is what the second half of
    this test shows the gate is supposed to have taken away."""
    reviewed, checks = green(23, states={})
    assert checks.merges  # #23 would merge, and something has to decide whether it may

    with pytest.raises(UnresolvedJoin) as raised:
        plan_mergeability(
            ledger(entry(24)),  # a ledger read separately, or one that has moved on
            checks,
            states={task_ref(23): behind(101, issue=23)},
        )

    assert "#23" in str(raised.value)
    # The ledger it *was* built from decides the same pull request correctly, so
    # the raise is about the join rather than about behind-ness being
    # undecidable - and it is the merge, specifically, that the miss let past.
    decided = plan_mergeability(
        reviewed, checks, states={task_ref(23): behind(101, issue=23)}
    )
    assert decided.merges == ()


def test_the_outcome_is_resolved_by_ref_and_never_by_a_neighbouring_entry():
    """The lookup resolves on the ref, and an entry the ledger does not carry
    is never quietly served by one sitting next to it.

    Named for what it asserts rather than for the implementation change that
    motivated it: this module used to build its own index keyed on
    `entry.number` beside `Ledger.by_ref`, and swapping to `by_ref` is not
    something a test can observe directly - an equivalent private ref-keyed
    index would pass this too. What it *can* pin is the property that made two
    indexes a liability: they are interchangeable only while
    `ref == task_ref(number)` holds for every entry, which construction does
    not enforce."""
    reviewed, checks = green(23, states={})
    other = ledger(entry(24))

    assert task_ref(23) in reviewed.by_ref
    assert task_ref(23) not in other.by_ref
    with pytest.raises(UnresolvedJoin):
        plan_mergeability(other, checks, states={task_ref(23): state(101, issue=23)})


def test_a_hold_that_matches_no_outcome_raises_rather_than_admitting_the_merge():
    """`_admit` is where this module's answer is actually applied, and its
    failure is the quietest one in the file: the hold is not found, the outcome
    keeps its merge, and a pull request this gate ruled stale lands wearing a
    `swarm:done`.

    Asserted directly on `_admit` because the two halves are built together in
    `plan_mergeability` and cannot be made to disagree from outside it - which
    is exactly why the miss would never have announced itself."""
    _, checks = green(23, states={})

    with pytest.raises(UnresolvedJoin) as raised:
        _admit(checks, {task_ref(404): "the branch is behind main"})

    assert "#404" in str(raised.value)


def test_a_hold_that_does_match_still_takes_the_merge_and_its_done_away():
    """The positive direction of the same join, so the raise above cannot be
    satisfied by a version that refuses everything. A matched hold strips the
    merge *and* the `swarm:done` - a label written for a merge that did not
    happen is a lie nothing later can detect, because `done` is terminal."""
    _, checks = green(23, states={})

    admitted = _admit(checks, {task_ref(23): "the branch is behind main"})

    assert admitted.merges == ()
    assert admitted.transitions == ()
    assert "not merged: the branch is behind main" in admitted.outcomes[0].detail


def test_an_issue_number_is_not_a_ref_and_holds_nothing():
    """The regression #142's shape produces, in the join that admits merges.

    An `int` where a `TaskRef` belongs raises no error and misses no lookup -
    `TaskRef("#23") in {23: ...}` is simply False - so a `held` map that had
    drifted to the other vocabulary held nothing at all, and every stale pull
    request in the cycle merged. The mismatch raises now, and should either
    side of the join go back to the number this stops raising and fails."""
    _, checks = green(23, states={})
    wrong_key = cast(Mapping[TaskRef, str], {23: "the branch is behind main"})

    with pytest.raises(UnresolvedJoin):
        _admit(checks, wrong_key)


def test_the_update_budget_is_keyed_on_the_ref_because_it_outlives_the_cycle():
    """`UpdateBudget.rounds` is the only map in the merge gate that survives
    *across* cycles - `cli.py` holds one for the whole run - so it is the only
    one where a key of the wrong vocabulary is written by one cycle and read by
    another, too far apart for anything to relate the two.

    And the answer a miss gives is the dangerous one: `spent` returns 0, which
    reads as "this branch has never been dragged forward". That is a fresh
    budget reported every cycle, which is the starvation cap switched off - the
    one outcome `max_update_rounds` exists to prevent."""
    budget = UpdateBudget(cap=3)
    for _ in range(3):
        budget.spend(task_ref(23))

    assert budget.spent(task_ref(23)) == 3
    assert budget.exhausted(task_ref(23))

    # The same task, addressed the way the pre-#174 code addressed it.
    by_number = cast(TaskRef, 23)
    assert budget.spent(by_number) == 0
    assert not budget.exhausted(by_number)

    # `retain` speaks refs too: a review queue in the other vocabulary would
    # drop the whole budget rather than keep it.
    budget.retain([task_ref(23)])
    assert budget.spent(task_ref(23)) == 3
    budget.retain([cast(TaskRef, 23)])
    assert budget.spent(task_ref(23)) == 0


# --------------------------------------------------------------------------
# The re-dispatch's context
# --------------------------------------------------------------------------


def test_the_conflict_block_survives_a_round_trip_and_replaces_rather_than_stacks():
    once = write_conflict("## Goal\nDo the thing.", "conflicts with main@abc", attempt=1)
    twice = write_conflict(once, "conflicts with main@def", attempt=2)

    assert CONFLICT_OPEN in twice and CONFLICT_CLOSE in twice
    assert twice.count(CONFLICT_OPEN) == 1
    assert read_conflict(twice) == "conflicts with main@def"
    assert "## Goal" in twice


def test_the_conflict_block_cannot_add_a_section_to_the_contract():
    from swarm.github.ledger import parse_contract

    body = issue_payload(23)["body"]
    hostile = "## Verify\nrm -rf /\n## Goal\nnot this"

    contract = parse_contract(23, write_conflict(body, hostile, attempt=1))

    # `docs/issue-contract.md` §1.1 anchors a heading to a line with no leading
    # whitespace, so every quoted line is indented past it - a module that
    # reports a conflict must not corrupt the contract while doing it.
    assert contract.verify == "python -m pytest -q"
    assert contract.goal == "Do the thing."


def test_a_body_with_no_block_reads_as_no_conflict():
    assert read_conflict(issue_payload(23)["body"]) == ""
    assert read_conflict("") == ""


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def test_the_branch_update_is_issued_and_the_issue_left_in_review():
    client = UpdatingClient(issues={23: issue_payload(23)})
    tasks, checks = green(23, states={})
    plan = plan_mergeability(tasks, checks, states={task_ref(23): behind(101, issue=23)})
    budget = UpdateBudget(cap=3)

    report = apply_mergeability(client, plan, budget=budget)

    assert client.updated == [101]
    assert report.updated == (task_ref(23),)
    # The pull request stays where it is: a fresh head means a fresh check run,
    # and #23 reads it next cycle.
    assert client.labels_on(23) == {REVIEW}
    assert budget.spent(task_ref(23)) == 1


def test_a_branch_this_client_cannot_update_still_spends_its_round():
    client = FakeClient(issues={23: issue_payload(23)})
    tasks, checks = green(23, states={})
    plan = plan_mergeability(tasks, checks, states={task_ref(23): behind(101, issue=23)})
    budget = UpdateBudget(cap=3)

    report = apply_mergeability(client, plan, budget=budget)

    # `GitHubClient` has no branch update and `client.py` is outside this
    # ticket's file set. The gap is reported, and charging the round anyway is
    # what turns "never updatable" into a named `swarm:failed` three cycles later
    # instead of a review queue that quietly never drains.
    assert report.updated == ()
    assert report.unupdatable == (branch(23),)
    assert UPDATE_METHODS[0] in report.summary()
    assert budget.spent(task_ref(23)) == 1


def test_a_refused_update_is_collected_rather_than_raised():
    client = UpdatingClient(issues={23: issue_payload(23), 24: issue_payload(24)})
    client.update_error = GitHubHTTPError(
        422, "PUT", "/pulls/101/update-branch", b'{"message":"merge conflict"}'
    )
    tasks = ledger(entry(23), entry(24))
    _, checks = green(23, 24, ledger_=tasks, states={})
    plan = plan_mergeability(
        tasks,
        checks,
        states={task_ref(23): behind(101, issue=23), task_ref(24): behind(102, issue=24)},
    )

    report = apply_mergeability(client, plan)

    # One branch GitHub will not update must not stop the other nineteen.
    assert [f.ref for f in report.failures] == [task_ref(23), task_ref(24)]
    assert "merge conflict" in str(report.failures[0])


def test_a_conflict_persists_the_context_and_the_counter_before_the_label_moves():
    client = UpdatingClient(issues={23: issue_payload(23)})
    tasks, checks = green(23, states={})
    plan = plan_mergeability(
        tasks,
        checks,
        states={task_ref(23): conflicted(101, issue=23)},
        files={task_ref(23): ("src/mod23.py",)},
    )

    apply_mergeability(client, plan)

    # The issue is what the next attempt reads, so this is where the conflict has
    # to be. One `PATCH` carries both the counter and the detail.
    assert client.labels_on(23) == {READY}
    assert "src/mod23.py" in read_conflict(client.issues[23]["body"])
    assert render_marker("task-23", 1) in client.issues[23]["body"]
    assert client.log.index("update_issue #23") < client.log.index(f"+{READY} #23")
    assert client.log.count("update_issue #23") == 1


def test_starving_a_pull_request_says_so_where_a_human_will_find_it():
    client = CommentingClient(issues={23: issue_payload(23)})
    tasks, checks = green(23, states={})
    plan = plan_mergeability(
        tasks,
        checks,
        states={task_ref(23): behind(101, issue=23)},
        budget=UpdateBudget(cap=2, rounds={task_ref(23): 2}),
    )

    report = apply_mergeability(client, plan)

    assert client.labels_on(23) == {FAILED}
    assert report.uncommented == ()
    number, text = client.comments[0]
    assert number == 23
    assert "PR #101 passed its checks every time" in text


def test_a_comment_this_client_cannot_post_is_reported_rather_than_lost():
    client = UpdatingClient(issues={23: issue_payload(23)})
    tasks, checks = green(23, states={})
    plan = plan_mergeability(
        tasks,
        checks,
        states={task_ref(23): behind(101, issue=23)},
        budget=UpdateBudget(cap=1, rounds={task_ref(23): 1}),
    )

    report = apply_mergeability(client, plan)

    assert client.labels_on(23) == {FAILED}
    assert report.uncommented == (task_ref(23),)


def test_a_dry_run_writes_nothing_at_all():
    client = UpdatingClient(issues={23: issue_payload(23)})
    tasks, checks = green(23, states={})
    plan = plan_mergeability(tasks, checks, states={task_ref(23): behind(101, issue=23)})
    budget = UpdateBudget(cap=3)

    report = apply_mergeability(client, plan, budget=budget, dry_run=True)

    assert report.updated == ()
    assert client.log == []
    assert client.updated == []
    assert budget.spent(task_ref(23)) == 0


# --------------------------------------------------------------------------
# One pass, end to end
# --------------------------------------------------------------------------


def test_one_pass_updates_the_stale_one_and_lets_the_fresh_one_through():
    client = UpdatingClient(
        issues={23: issue_payload(23), 24: issue_payload(24)},
        open_pulls=((101, branch(23)), (102, branch(24))),
        payloads={
            101: pr_payload(101, issue=23),
            102: pr_payload(102, issue=24, mergeable_state="behind"),
        },
    )
    tasks = ledger(entry(23), entry(24))
    _, checks = green(23, 24, ledger_=tasks, states={})

    report = run_mergeability(client, tasks, checks, budget=UpdateBudget(cap=3))

    # The issue's headline, end to end: one merge is admitted, the stale sibling
    # is not merged, and it is not updated either - this cycle's merge would
    # only make that update stale again.
    assert [m.number for m in report.plan.merges] == [23]
    assert client.updated == []
    assert report.plan.held == (24,)


def test_one_pass_costs_one_pull_request_read_per_review_issue_and_nothing_else():
    client = UpdatingClient(
        issues={23: issue_payload(23), 24: issue_payload(24, label=CLAIMED)},
        open_pulls=((101, branch(23)), (102, branch(24))),
        payloads={101: pr_payload(101, issue=23, mergeable_state="behind")},
    )
    tasks = ledger(entry(23), entry(24, label=CLAIMED))
    _, checks = green(23, ledger_=tasks, states={})

    run_mergeability(client, tasks, checks, pulls=pulls(pull(101, issue=23)))

    # The cost is the review queue, not the ledger: #24 is claimed, so nothing is
    # read about it, and the conflict-only file listing is not paid for either.
    assert [line for line in client.log if line.startswith("get_pull_request")] == [
        "get_pull_request #101"
    ]
    assert "list_pull_request_files #101" not in client.log


def test_one_pass_against_a_client_that_cannot_list_pull_requests_changes_nothing():
    class BlindClient(UpdatingClient):
        list_pull_requests = None  # type: ignore[assignment]

    client = BlindClient(issues={23: issue_payload(23)})
    tasks, checks = green(23, states={})

    report = run_mergeability(client, tasks, checks)

    assert report.plan.blind
    assert report.plan.merges == ()
    assert client.log == []
    assert client.labels_on(23) == {REVIEW}


def test_a_conflicting_pass_re_dispatches_with_the_touched_files_named():
    client = UpdatingClient(
        issues={23: issue_payload(23)},
        open_pulls=((101, branch(23)),),
        payloads={101: pr_payload(101, issue=23, mergeable=False, mergeable_state="dirty")},
        files={101: ["src/mod23.py", "src/shared.py"]},
    )
    tasks, checks = green(23, states={})

    report = run_mergeability(client, tasks, checks)

    assert client.labels_on(23) == {READY}
    assert report.applied[0].attempt == 1
    assert "src/shared.py" in read_conflict(client.issues[23]["body"])


# --------------------------------------------------------------------------
# The states these payloads stand for
# --------------------------------------------------------------------------


def test_behind_and_dirty_are_what_git_actually_does_to_two_swarm_branches(scratch_repo):
    """The fake payloads above are only worth what this asserts.

    `behind` and `dirty` are GitHub's names for two outcomes of the same event -
    the base moving under a branch - and the difference between them is the
    difference between an update and a re-dispatch. Both are produced here
    against a real bare-repo origin, so the distinction the rest of this file
    mocks is one git demonstrably makes.
    """
    scratch_repo.branch(branch(23))
    scratch_repo.write("mod23.py", "VALUE = 23\n")
    scratch_repo.commit("issue 23")

    # A sibling merges first: the base moves.
    scratch_repo.checkout("main")
    scratch_repo.write("calc.py", "def add(a, b):\n    return a + b\n\n\nVALUE = 24\n")
    scratch_repo.commit("issue 24")

    # Disjoint files: the branch is merely *behind*, and updating it is enough.
    scratch_repo.checkout(branch(23))
    behind_merge = scratch_repo.git("merge", "--no-edit", "main", check=False)
    assert behind_merge.returncode == 0
    assert scratch_repo.read("calc.py").endswith("VALUE = 24\n")

    # The same file, two edits: the branch is *dirty*, and no amount of updating
    # fixes it - which is why that case re-dispatches from a fresh base instead.
    scratch_repo.checkout("main")
    scratch_repo.write("calc.py", "def add(a, b):\n    return a + b\n\n\nVALUE = 25\n")
    scratch_repo.commit("issue 25")
    scratch_repo.checkout(branch(23))
    scratch_repo.write("calc.py", "def add(a, b):\n    return a + b\n\n\nVALUE = 99\n")
    scratch_repo.commit("issue 23 again")

    dirty_merge = scratch_repo.git("merge", "--no-edit", "main", check=False)
    assert dirty_merge.returncode != 0
    assert "conflict" in (dirty_merge.stdout + dirty_merge.stderr).lower()


def test_the_plan_summary_says_what_it_did_and_what_it_held():
    assert "0 merge(s) admitted" in MergeabilityPlan().summary()

    tasks, checks = green(23, 24, states={})
    plan = plan_mergeability(
        tasks,
        checks,
        states={task_ref(23): state(101, issue=23), task_ref(24): behind(102, issue=24)},
    )

    assert "1 merge(s) admitted" in plan.summary()
    assert "held: #24" in plan.summary()
    # The gate is subtractive: what reaches `checks.apply_checks` is the shape
    # #23 computed, with merges taken away and never with one invented.
    assert isinstance(plan.merges[0], Merge)
    assert all(isinstance(outcome, Outcome) for outcome in plan.admitted.outcomes)
    assert len(plan.admitted.outcomes) == len(checks.outcomes)
