"""Tests for the check-run gate.

Five properties carry this file, and each of them is a way the run goes wrong
rather than a way it computes wrong.

**Green merges, red retries with the failure in hand, exhausted stops.** The
issue's own acceptance criterion, and the middle one is the load-bearing half: a
retry whose context does not carry the failure reproduces the failure, so the
output is asserted to be on the issue *before* the label goes back to
`swarm:ready`.

**Zero checks is its own answer.** An empty check set is never a merge and never
an endless wait: it is pending inside the grace period and a human's problem
outside it. Both directions are tested, because each of the two obvious
shortcuts - "no failures, ship it" and "wait for a check that is never coming" -
is a bug this file exists to keep out.

**The override is explicit.** Merging with an admin override bypasses the review
this repository's ruleset requires, so a policy with it turned off must merge
nothing at all, and the merge that does happen must carry the flag that says
under whose authority.

**A failure outside `## Files` is not the worker's to fix.** It is routed to a
human without consuming attempts, because feeding it back produces three
identical failures and a `swarm:failed` for a problem that was never in the
worker's code.

**Absence of evidence is not evidence.** A client that cannot list pull requests
decides nothing; check runs that could not be read are pending, not passed and
not failed.

Hermetic throughout: a plain fake client, no HTTP, no daemon, no clock that
moves on its own.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, cast

import pytest

FAILED = "needs-human"

DONE = "landed"
from fixtures import failures

from fixtures.store import RecordingStore
from swarm.store import STORE_DIR_ENV, StoreError
from swarm.github.branches import task_branch
from swarm.github.client import GitHubHTTPError
from swarm.github.ledger import Ledger, LedgerEntry, render_marker
from swarm.github.refs import pull_number, pull_ref, task_ref
from swarm.orchestrator.checks import (
    ADMIN_OVERRIDE_ENV,
    BRANCH_METHODS,
    EMPTY,
    FAILING,
    PASSED,
    PENDING,
    PULLS_METHOD,
    CheckSet,
    ChecksPlan,
    ChecksReport,
    MergePolicy,
    PullState,
    UnresolvedJoin,
    apply_checks,
    failing_paths,
    foreign_failure,
    plan_checks,
    read_pulls,
    run_checks,
    summarise_checks,
)
from swarm.nodes.judge import mentioned_paths
from swarm.orchestrator.dispatcher import CLAIMED, REVIEW
from swarm.orchestrator.reconcile import READY
from swarm.taskref import TaskRef
from swarm.orchestrator.authority import Belief
from swarm.orchestrator.derived import ELIGIBLE, LANDED, NEEDS_HUMAN
from swarm.orchestrator.derived import REVIEW as REVIEW_STATE
from fixtures.belief import fixture_belief


# The cycle's belief, supplied from what each fixture declares (see
# `fixtures.belief`). It was read off `LedgerEntry.state_label` until #152.
def _plan_checks_(book, *args, **kwargs):
    kwargs.setdefault("believed", fixture_belief(book))
    return plan_checks(book, *args, **kwargs)


NOW = dt.datetime(2026, 8, 14, 14, 25, 30, tzinfo=dt.timezone.utc)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


REPO = "shahrestani-me/apiary"


@pytest.fixture(autouse=True)
def store_root(tmp_path, monkeypatch):
    """Every store this module opens lands under `tmp_path`.

    `test_reconcile`'s fixture and its reason, which this module needed the day
    it started asserting against a store (ADR 0005): autouse and unconditional,
    because the failure it prevents is silent. A test that forgot to redirect
    the root would open the *operator's* store at `.swarm/store`, read a real
    project's retry budgets and write test judgments into them. Nothing would
    fail; the next real run would simply believe something untrue about its own
    history.
    """
    root = tmp_path / "store"
    monkeypatch.setenv(STORE_DIR_ENV, str(root))
    return root


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
        labels=frozenset({label}),
    )


def branch(number: int, attempt: int = 0) -> str:
    """The head ref a worker for `(number, attempt)` pushed - built, not spelled.

    Spelling the name out would make these tests assert #144's encoding rather
    than the gate's behaviour, and the encoding is `test_branches.py`'s subject.
    """
    return task_branch(task_ref(number), attempt)


def ledger(*entries: LedgerEntry) -> Ledger:
    return Ledger(entries={item.task_id: item for item in entries})


def pull(number: int, *, issue: int, age_s: float = 0.0, draft: bool = False,
         attempt: int = 0) -> PullState:
    # `attempt` has to match the entry's, because a branch name carries it
    # (#144) and the gate looks a pull request up by `LedgerEntry.branch`. That
    # is the real invariant, not a test detail: a task in `swarm:review` still
    # carries the attempt its open pull request was pushed from, because an
    # exit 0 moves no counter (`reconcile._observe`).
    return PullState(
        number=pull_ref(number),
        branch=branch(issue, attempt),
        sha=f"{issue:0>40x}",
        updated_at=NOW - dt.timedelta(seconds=age_s),
        draft=draft,
    )


def pulls(*states: PullState) -> dict[str, PullState]:
    return {state.branch: state for state in states}


def run(name: str, conclusion: str | None = "success", *, text: str = "") -> dict[str, Any]:
    """One check-run payload, in the shape `GET /commits/{ref}/check-runs` returns."""
    return {
        "name": name,
        "status": "completed" if conclusion else "in_progress",
        "conclusion": conclusion,
        "output": {"title": f"{name} {conclusion}", "summary": text},
        "details_url": f"https://github.com/shahrestani-me/apiary/runs/{name}",
    }


def failing(*paths: str, name: str = "ci") -> CheckSet:
    """A red check set whose output names `paths` the way pytest would."""
    lines = [f"FAILED {path}::test_thing - AssertionError" for path in paths]
    return summarise_checks([run(name, "failure", text="\n".join(lines))])


#: The world every #243 case shares: #23 in review by belief, wearing the label
#: a human typed onto it instead.
REVIEWED = Belief(states={"task-23": "review"})


def relabelled(label: str, attempt: int = 0) -> Ledger:
    """#23 as the gate sees it, wearing `swarm:done` because somebody typed it."""
    return ledger(entry(23, "src/mod23.py", "tests/test_mod23.py",
                        label=label, attempt=attempt))


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


@dataclass
class FakeClient:
    """Every call this module makes, recorded, with no HTTP anywhere.

    Carries `list_pull_requests` and `merge_pull_request` because both are what
    the module needs, and deliberately *not* `create_issue_comment` or a branch
    deleter: that is the state `GitHubClient` is actually in, so the default
    client here exercises the degraded path and the subclasses below are what
    lands once those methods exist.
    """

    issues: dict[int, dict[str, Any]] = field(default_factory=dict)
    open_pulls: tuple[tuple[int, str], ...] = ()
    checks: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    merge_error: Exception | None = None

    # --- reads -----------------------------------------------------------

    def list_pull_requests(self, *, state: str = "open") -> list[dict[str, Any]]:
        self.log.append(f"list_pull_requests {state}")
        return [
            {
                "number": number,
                "head": {"ref": ref, "sha": f"{number:0>40x}"},
                "updated_at": (NOW - dt.timedelta(hours=1)).isoformat(),
            }
            for number, ref in self.open_pulls
        ]

    def list_check_runs(self, ref: str) -> list[dict[str, Any]]:
        self.log.append(f"list_check_runs {ref}")
        return list(self.checks.get(ref, ()))

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
        # 404 on an issue that is not there, as GitHub does. It used to
        # `setdefault` the number into existence, which was invisible while
        # `get_issue` ran first on this path and 404'd for the same reason - the
        # marker's read-modify-write was the guard. ADR 0005 removed that read,
        # so a fake that quietly recreates a deleted issue would report a clean
        # relabel of something that no longer exists.
        if number not in self.issues:
            raise GitHubHTTPError(404, "POST", f"/issues/{number}/labels", b'{"message":"Not Found"}')
        current = self.issues[number].setdefault("labels", [])
        current.extend({"name": name} for name in names)
        return current

    def remove_label(self, number: int, label: str) -> bool:
        self.log.append(f"-{label} #{number}")
        payload = self.issues.get(number)
        if payload is None:
            return False
        payload["labels"] = [item for item in payload.get("labels", []) if item["name"] != label]
        return True

    def merge_pull_request(self, number: int, **kwargs: Any) -> dict[str, Any]:
        self.log.append(f"merge PR #{number} {kwargs.get('merge_method')} sha={kwargs.get('sha')}")
        if self.merge_error is not None:
            raise self.merge_error
        return {"merged": True, "sha": "deadbeef"}

    # --- what the assertions read ----------------------------------------

    def labels_on(self, number: int) -> set[str]:
        return {item["name"] for item in self.issues[number].get("labels", [])}


@dataclass
class DeletingClient(FakeClient):
    """The client once `client.py` grows the ref deletion a merge needs."""

    deleted: list[str] = field(default_factory=list)

    def delete_branch(self, branch: str) -> bool:
        self.deleted.append(branch)
        return True


@dataclass
class CommentingClient(FakeClient):
    """The client once §1.4's comment method exists.

    It reads comments back as well as writing them (`list_issue_comments`),
    which is not decoration: #248's assertion has to travel through
    `worker.entrypoint.fetch_feedback` - the thing that actually looks for the
    feedback - rather than round the back of it into `comments`. A test that
    inspected the comment body would have passed for the whole life of the bug,
    because the text was always correct and always written where nothing read.
    """

    comments: list[tuple[int, str]] = field(default_factory=list)

    def create_issue_comment(self, number: int, text: str) -> dict[str, Any]:
        # Narrated into `log` as well as collected, so the crash-ordering test
        # still has a *client* event to compare the store write against. Until
        # #152 that event was the label write; the retry feedback is what
        # `apply_checks` sends after the judgment now.
        self.log.append(f"comment #{number}")
        self.comments.append((number, text))
        return {"id": len(self.comments)}

    def list_issue_comments(self, number: int) -> list[dict[str, Any]]:
        """Oldest first, as GitHub returns them - `fetch_feedback` walks backwards."""
        return [{"body": text} for issue, text in self.comments if issue == number]


@dataclass
class RefusingClient(FakeClient):
    """A client that turns down *some* merges. `FakeClient.merge_error` is all
    or nothing, which cannot express the one case the refusal join is for: a
    cycle where one pull request lands and another does not."""

    refuses: set[int] = field(default_factory=set)

    def merge_pull_request(self, number: int, **kwargs: Any) -> dict[str, Any]:
        self.log.append(f"merge PR #{number}")
        if number in self.refuses:
            raise GitHubHTTPError(
                405, "PUT", f"/pulls/{number}/merge", b'{"message":"not mergeable"}'
            )
        return {"sha": f"{number:0>40x}"}


class BlindClient(FakeClient):
    """A client with no `list_pull_requests` - which is today's `GitHubClient`."""

    list_pull_requests = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Reading what CI said
# --------------------------------------------------------------------------


def test_a_check_still_running_is_pending_even_when_every_finished_one_passed():
    checks = summarise_checks([run("lint", None), run("test", "success")])

    # The safe direction: an unknown or unfinished status must never be read as
    # a clearance, because the merge it clears is not reversible.
    assert checks.verdict == PENDING
    assert checks.pending == ("lint",)


def test_neutral_and_skipped_conclusions_do_not_block_a_merge():
    checks = summarise_checks([run("test", "success"), run("docs", "skipped"), run("x", "neutral")])

    # A path-filtered job that decided it had nothing to do has not found a
    # fault, and treating it as one makes the PR unmergeable forever.
    assert checks.verdict == PASSED


def test_a_failure_beats_a_check_that_is_still_running():
    checks = summarise_checks([run("slow", None), run("test", "failure")])

    # The PR cannot pass, so waiting for the rest of the matrix buys wall clock
    # and nothing else.
    assert checks.verdict == FAILING
    assert checks.failed == ("test",)


def test_no_check_runs_at_all_is_empty_which_is_not_passed():
    # The distinction this whole module turns on: `empty` is a third answer, and
    # a `verdict` that collapsed it into `passed` would merge unverified code.
    assert summarise_checks([]).verdict == EMPTY
    assert CheckSet(unreadable=True).verdict == PENDING


def test_the_failure_output_carries_the_check_name_and_what_it_printed():
    checks = summarise_checks([run("ci", "failure", text="FAILED tests/test_a.py::test_x")])

    assert "ci: failure" in checks.output
    assert "FAILED tests/test_a.py::test_x" in checks.output


# --------------------------------------------------------------------------
# Whose failure is it
# --------------------------------------------------------------------------


def test_failing_paths_reads_both_pytest_shapes_and_nothing_else():
    text = "FAILED tests/test_a.py - boom\ntests/test_b.py::test_y failed\nsee src/thing.py for why"

    # The third line is prose about a file, not evidence about a failure, and a
    # false positive here escalates an issue a worker could have fixed.
    assert failing_paths(text) == ("tests/test_a.py", "tests/test_b.py")


def test_a_failure_inside_the_declared_files_is_the_workers_to_fix():
    task = entry(23, "src/mod23.py", "tests/test_mod23.py")

    assert foreign_failure(task, "FAILED tests/test_mod23.py::test_x") == ()
    # One inside is enough: there is something the next attempt can act on.
    assert foreign_failure(task, "FAILED tests/test_mod23.py\nFAILED tests/test_other.py") == ()


def test_a_failure_entirely_outside_the_declared_files_is_nobodys_to_retry():
    task = entry(23, "src/mod23.py", "tests/test_mod23.py")

    assert foreign_failure(task, "FAILED tests/test_other.py::test_x") == ("tests/test_other.py",)
    # No path named at all falls back to the ordinary retry - the escalation is
    # an optimisation on a correct default, never a precondition for one.
    assert foreign_failure(task, "the build died") == ()


# --------------------------------------------------------------------------
# The plan: green
# --------------------------------------------------------------------------


def test_a_green_pull_request_is_merged_and_its_issue_marked_done():
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): summarise_checks([run("test", "success")])},
        now=NOW,
    )

    assert [str(t) for t in plan.transitions] == [
        "#23: review -> landed (PR #101 merged: 1 passed)"
    ]
    assert plan.merges[0].pull == pull_ref(101)
    assert plan.merges[0].admin_override is True


def test_the_merge_deletes_the_branch_because_a_long_run_leaves_one_per_task():
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): summarise_checks([run("test", "success")])},
        now=NOW,
    )

    assert plan.merges[0].delete_branch is True
    assert plan.merges[0].branch == branch(23)


def test_with_the_override_off_a_green_pull_request_waits_for_a_human():
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): summarise_checks([run("test", "success")])},
        policy=MergePolicy(admin_override=False),
        now=NOW,
    )

    # The opt-in a target repository gets instead of having its review gate
    # bypassed: nothing is merged and nothing is relabelled, so the issue sits
    # in `swarm:review` until a person presses the button.
    assert plan.merges == ()
    assert plan.transitions == ()
    assert plan.outcomes[0].verdict == PASSED
    assert "waiting for a human" in plan.outcomes[0].detail


def test_a_draft_pull_request_is_not_merged_however_green_it_is():
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23, draft=True)),
        checks={task_ref(23): summarise_checks([run("test", "success")])},
        now=NOW,
    )

    assert plan.merges == ()
    assert plan.outcomes[0].verdict == PENDING


def test_the_policy_summary_says_out_loud_what_the_override_bypasses():
    assert "bypasses" in MergePolicy().summary()
    assert ADMIN_OVERRIDE_ENV in MergePolicy().summary()
    assert "will not merge" in MergePolicy(admin_override=False).summary()


def test_the_override_is_configurable_and_loud_about_a_value_it_cannot_read(monkeypatch):
    monkeypatch.setenv(ADMIN_OVERRIDE_ENV, "0")
    assert MergePolicy.from_env().admin_override is False

    monkeypatch.setenv(ADMIN_OVERRIDE_ENV, "flase")
    # Falling back to the default here would leave a repository merging without
    # review while somebody believed they had turned that off.
    with pytest.raises(ValueError):
        MergePolicy.from_env()


# --------------------------------------------------------------------------
# The plan: red
# --------------------------------------------------------------------------


def test_a_failing_check_retries_and_the_transition_carries_the_failure():
    plan = _plan_checks_(
        ledger(entry(23, attempt=0)),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): failing("tests/test_mod23.py")},
        now=NOW,
    )
    outcome = plan.outcomes[0]

    assert outcome.transition is not None
    assert outcome.transition.to_state == ELIGIBLE
    # §5: the counter rides on the transition, so it is persisted before the
    # issue can be dispatched again.
    assert outcome.transition.attempt == 1
    # The half without which the retry is theatre.
    assert "tests/test_mod23.py" in outcome.feedback


def test_the_last_attempt_gives_up_rather_than_retrying_forever():
    plan = _plan_checks_(
        ledger(entry(23, attempt=2)),
        pulls=pulls(pull(101, issue=23, attempt=2)),
        checks={task_ref(23): failing("tests/test_mod23.py")},
        max_attempts=3,
        now=NOW,
    )
    transition = plan.outcomes[0].transition

    assert transition is not None
    assert transition.to_state == NEEDS_HUMAN
    assert "against a cap of 3" in transition.reason


def test_a_failure_outside_the_declared_files_goes_to_a_human_not_to_a_retry():
    plan = _plan_checks_(
        ledger(entry(23, "src/mod23.py")),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): failing("tests/test_somebody_else.py")},
        max_attempts=3,
        now=NOW,
    )
    outcome = plan.outcomes[0]

    # Tonight's real failure: a PR passed its own `## Verify` and broke a test
    # in a file the worker is forbidden to edit. Three retries cannot fix that,
    # and spending them hides the diagnosis behind an exhausted budget.
    assert outcome.escalated
    assert outcome.transition is not None
    assert outcome.transition.to_state == NEEDS_HUMAN
    assert "tests/test_somebody_else.py" in outcome.transition.reason
    # The attempt budget is untouched: this was never the attempt's fault.
    assert outcome.transition.attempt is None
    assert plan.escalated == (23,)


# --------------------------------------------------------------------------
# The plan: zero checks
# --------------------------------------------------------------------------


def test_an_empty_check_set_is_pending_while_the_grace_period_runs():
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23, age_s=30)),
        checks={task_ref(23): CheckSet()},
        policy=MergePolicy(zero_check_grace_s=300),
        now=NOW,
    )

    # The seconds after a push, before GitHub has created anything. Failing here
    # would fail every PR the swarm ever opens.
    assert plan.outcomes[0].verdict == PENDING
    assert plan.transitions == ()
    assert plan.merges == ()


def test_an_empty_check_set_past_its_grace_goes_to_a_human_and_is_never_merged():
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23, age_s=3600)),
        checks={task_ref(23): CheckSet()},
        policy=MergePolicy(zero_check_grace_s=300),
        now=NOW,
    )
    outcome = plan.outcomes[0]

    # Neither of the two shortcuts: not merged, because nothing verified it, and
    # not parked, because nothing ever will.
    assert plan.merges == ()
    assert outcome.verdict == EMPTY
    assert outcome.transition is not None
    assert outcome.transition.to_state == NEEDS_HUMAN
    assert outcome.escalated
    # And not retried either: the next attempt touches the same paths and gets
    # the same empty set.
    assert outcome.transition.attempt is None


# --------------------------------------------------------------------------
# The join: `checks` against the ledger, keyed on the ref (#174)
# --------------------------------------------------------------------------
#
# These four pin the one lookup in this module that could not announce its own
# failure. `plan_checks` used to read `checks.get(entry.number, CheckSet())`,
# and every part of that line is load-bearing in the wrong direction: a miss
# returns rather than raises, the default is not neutral, and `CheckSet()`
# carries the verdict `EMPTY` - which past the grace period is `swarm:failed`,
# escalated, never merged and never retried. A task whose check runs could not
# be *looked up* was therefore handed to a human wearing exactly the label of a
# task whose attempts had genuinely run out, with nothing in the ledger, the
# plan or the log able to tell the two apart.
#
# So the assertions below are on the miss, not on the happy path. The happy
# path stayed green through the entire life of the defect; that is the whole
# problem with it.


def aged_pull_past_the_grace() -> dict[str, PullState]:
    """A pull request old enough that an empty check set escalates its issue.

    Named rather than inlined because it is the precondition that makes the
    fail-open dangerous: inside the grace period a missing check set is merely
    `PENDING`, and the tests below would pass for the wrong reason.
    """
    return pulls(pull(101, issue=23, age_s=3600))


def test_a_check_set_that_cannot_be_looked_up_raises_rather_than_escalating():
    """The headline. An entry the caller was supposed to read checks for and
    did not must stop the cycle, not quietly become a `swarm:failed`.

    Revert `plan_checks` to a defaulted `.get` and this stops raising: the plan
    comes back with one escalated outcome and a `swarm:review -> swarm:failed`
    transition for an issue whose pull request nobody looked at."""
    with pytest.raises(UnresolvedJoin) as raised:
        _plan_checks_(
            ledger(entry(23)),
            pulls=aged_pull_past_the_grace(),
            checks={},
            policy=MergePolicy(zero_check_grace_s=300),
            now=NOW,
        )

    # The message has to name the task, because the operator reading it has to
    # know which issue the cycle stopped on.
    assert "#23" in str(raised.value)


def test_an_issue_number_is_not_a_ref_and_does_not_resolve_a_check_set():
    """The regression #142's shape produces, asserted as the property that
    makes the key type load-bearing rather than decorative.

    An `int` where a `TaskRef` belongs is not a type error at runtime and not a
    lookup error either: `{23: green} [TaskRef("#23")]` simply misses. Before
    #174 that miss defaulted, and a caller half-migrated to refs escalated
    every green pull request it had just read a passing check set for. Now it
    raises - and should either side of this join go back to the number, this
    stops raising and fails."""
    green = summarise_checks([run("test", "success")])
    wrong_key = cast(Mapping[TaskRef, CheckSet], {23: green})

    with pytest.raises(UnresolvedJoin):
        _plan_checks_(
            ledger(entry(23)),
            pulls=aged_pull_past_the_grace(),
            checks=wrong_key,
            policy=MergePolicy(zero_check_grace_s=300),
            now=NOW,
        )


def test_a_check_set_read_for_another_task_never_answers_for_this_one():
    """Present but wrong is the same fault as absent, and the more likely one:
    a map built over a different selection of entries than the one this loop
    walks. #24's checks must not decide #23, and a non-empty map must not read
    as "the lookup worked"."""
    with pytest.raises(UnresolvedJoin):
        _plan_checks_(
            ledger(entry(23)),
            pulls=aged_pull_past_the_grace(),
            checks={task_ref(24): summarise_checks([run("test", "success")])},
            policy=MergePolicy(zero_check_grace_s=300),
            now=NOW,
        )


def test_a_genuinely_empty_check_set_still_escalates_after_its_grace():
    """The other direction, and the reason the fix is a raise rather than a
    softened verdict: `EMPTY` is a real answer GitHub gives, and a repository
    whose workflows never run on these paths must still reach a human.

    A fix that made an absent check set harmless by making an *empty* one
    harmless would delete this module's zero-check rule, so the miss and the
    empty answer are pinned apart on purpose - same ledger, same clock, same
    grace, one raising and one escalating."""
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=aged_pull_past_the_grace(),
        checks={task_ref(23): CheckSet()},
        policy=MergePolicy(zero_check_grace_s=300),
        now=NOW,
    )

    assert plan.escalated == (23,)
    assert plan.outcomes[0].verdict == EMPTY
    assert plan.merges == ()


def test_the_check_sets_a_run_reads_are_keyed_the_way_the_plan_looks_them_up():
    """End to end, and what makes the raise a wiring assertion rather than a
    trap for callers: `run_checks` builds the map and `plan_checks` consumes
    it, so if the two ever disagree about the key this is the test that goes
    red - in the pass that is supposed to be entirely ordinary."""
    client = FakeClient(
        issues={23: issue_payload(23)},
        open_pulls=((101, branch(23)),),
        checks={f"{101:0>40x}": [run("test", "success")]},
    )

    book = ledger(entry(23))
    report = run_checks(client, book, believed=fixture_belief(book))

    assert [outcome.verdict for outcome in report.plan.outcomes] == [PASSED]
    assert report.merged == (23,)


# --------------------------------------------------------------------------
# The plan: what it refuses to decide
# --------------------------------------------------------------------------


def test_a_client_that_cannot_list_pull_requests_decides_nothing():
    plan = _plan_checks_(ledger(entry(23)), pulls=None, checks={}, now=NOW)

    # "We could not look" must never read as "the checks are not there".
    assert plan.blind
    assert plan.outcomes == ()
    assert f"the client has no {PULLS_METHOD}" in plan.summary()


def test_the_pull_request_listing_is_probed_for_rather_than_assumed():
    # `GitHubClient` has no `list_pull_requests` and `client.py` is outside this
    # ticket's file set, so the probe is what keeps two tickets whose file sets
    # cannot reach each other from deadlocking.
    assert read_pulls(BlindClient()) is None
    found = read_pulls(FakeClient(open_pulls=((101, branch(23)),)))
    assert found is not None and found[branch(23)].number == pull_ref(101)


def test_check_runs_that_could_not_be_read_are_pending_not_failed():
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): CheckSet(unreadable=True)},
        now=NOW,
    )

    assert plan.outcomes[0].verdict == PENDING
    assert plan.transitions == ()


def test_only_review_issues_with_an_open_pull_request_are_decided():
    plan = _plan_checks_(
        ledger(entry(23), entry(24, label=CLAIMED), entry(25)),
        # #25 is in review with no open PR - that is #22's row, and two modules
        # writing one transition is how a label moves twice in a cycle.
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): summarise_checks([run("test", "success")])},
        now=NOW,
    )

    assert [outcome.number for outcome in plan.outcomes] == [23]


# --------------------------------------------------------------------------
# The retry's context
# --------------------------------------------------------------------------

def client_with(*, issues: dict[int, dict[str, Any]] | None = None, **kwargs: Any) -> FakeClient:
    return FakeClient(issues=issues or {}, **kwargs)


def issue_payload(number: int, *, label: str = REVIEW, attempt: int = 0) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"issue {number}",
        "labels": [{"name": label}],
        "body": body(f"task-{number}", attempt=attempt),
    }


def test_a_green_pull_request_merges_and_only_then_records_it_landed():
    client = DeletingClient(issues={23: issue_payload(23)})
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): summarise_checks([run("test", "success")])},
        now=NOW,
    )

    report = apply_checks(client, plan)

    assert report.merged == (23,)
    assert client.deleted == [branch(23)]
    # Merge first, and only what merged is recorded: a `landed` claimed over a
    # merge GitHub refused is a lie nothing later in the system can detect,
    # because `landed` is terminal. The transition is the record now - #152
    # deleted the label write that used to be the second half of this ordering -
    # so the ordering it asserted is asserted on the log the merge is the *only*
    # entry in, and "only what merged" is held by the refusal tests below.
    assert [transition.to_state for transition in report.applied] == [DONE]
    assert client.log == [f"merge PR #101 squash sha={23:0>40x}"]


def test_a_refused_merge_leaves_the_issue_in_review_for_the_next_cycle():
    client = client_with(issues={23: issue_payload(23)})
    client.merge_error = GitHubHTTPError(
        405, "PUT", "/pulls/101/merge", b'{"message":"not mergeable"}'
    )
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): summarise_checks([run("test", "success")])},
        now=NOW,
    )

    report = apply_checks(client, plan)

    assert report.merged == ()
    assert report.applied == ()
    assert [str(f) for f in report.failures] == [
        "#23: merging PR #101: PUT /pulls/101/merge -> 405: not mergeable"
    ]
    assert client.labels_on(23) == {REVIEW}


def green_plan(*numbers: int) -> ChecksPlan:
    """A plan that would merge every `numbers` on green. The refusal tests' input."""
    return _plan_checks_(
        ledger(*(entry(n) for n in numbers)),
        pulls=pulls(*(pull(100 + n, issue=n) for n in numbers)),
        checks={task_ref(n): summarise_checks([run("test", "success")]) for n in numbers},
        now=NOW,
    )


def filed_under_the_pull_request(plan: ChecksPlan, number: int) -> ChecksPlan:
    """`plan` with one merge's task identity replaced by its pull request's.

    The drift this join has to survive: `Merge` carries `number` (the issue)
    beside `pull` (the pull request), and a `Merge` whose first field holds the
    second's value mints a `TaskRef` for a pull request. Nothing else about the
    plan changes - the `Outcome` still answers for the issue, which is exactly
    what makes the two halves disagree.

    **The `pull_number` call is the point of #185, not an inconvenience.** This
    helper used to read `outcome.merge.pull` straight into `number`, because
    both were `int` and the module made that the easy mistake to make. It no
    longer type-checks: the value has to be un-minted, by name, before it will
    go in the other field. What is left after that is a *human* error - a caller
    who fetched the wrong number in the first place - and #184's guard is what
    catches those, which is why it stays and why this test does too.
    """
    outcomes = []
    for outcome in plan.outcomes:
        if outcome.number == number and outcome.merge is not None:
            drifted = replace(outcome.merge, number=pull_number(outcome.merge.pull))
            outcome = replace(outcome, merge=drifted)
        outcomes.append(outcome)
    return replace(plan, outcomes=tuple(outcomes))


def test_a_refused_merge_whose_identity_drifted_still_never_records_it_landed():
    """The headline (#181), asserted on the outcome rather than on the mechanism.

    `apply_checks` files a refusal under the merge's task and reads it under the
    outcome's. Let those two disagree and the refusal is filed under a key
    nothing looks up: the merge GitHub turned down is skipped, the transition is
    not, and `landed` - terminal, never re-read by any later cycle - is recorded
    for a merge that did not happen.

    Deliberately indifferent to *how* that is prevented, because the property is
    that nothing claims the merge and not which exception says so. Revert either
    half of the join to the number and this goes red: #23 comes back `landed`
    for a pull request still open.

    Asserted on the log rather than on a label since #152, and the assertion is
    the stronger of the two - a label could only say what the issue ended up
    wearing, whereas an empty log says the gate wrote nothing at all."""
    client = client_with(issues={23: issue_payload(23)})
    client.merge_error = GitHubHTTPError(
        405, "PUT", "/pulls/123/merge", b'{"message":"not mergeable"}'
    )

    report = None
    try:
        report = apply_checks(client, filed_under_the_pull_request(green_plan(23), 23))
    except UnresolvedJoin:
        pass

    assert client.log == []
    assert report is None or DONE not in [t.to_state for t in report.applied]


def test_a_merge_that_matches_no_outcome_stops_the_gate_before_it_merges():
    """And the mechanism, which is the same one #174 gave the other three joins.

    Checked before the first API call rather than at the join, because both
    sides are derived from `plan.outcomes` and so both exist before anything has
    happened. That is what keeps `Reconciler.cycle`'s reason for catching
    `UnresolvedJoin` true - no merge issued, no label written - so the assert on
    the empty log is part of the fix and not decoration."""
    client = client_with(issues={23: issue_payload(23)})

    with pytest.raises(UnresolvedJoin) as raised:
        apply_checks(client, filed_under_the_pull_request(green_plan(23), 23))

    # The ref the refusal would have been filed under, and the ones anything
    # reads. An operator needs both to see which way the two halves drifted.
    assert "#123" in str(raised.value)
    assert "#23" in str(raised.value)
    assert client.log == []


def test_a_dry_run_does_not_call_a_plan_the_gate_would_refuse_to_apply():
    """The check precedes the dry-run return, because the dry run's whole job is
    to answer "would this be applied" and a plan carrying this fault would
    not."""
    with pytest.raises(UnresolvedJoin):
        apply_checks(
            client_with(issues={23: issue_payload(23)}),
            filed_under_the_pull_request(green_plan(23), 23),
            dry_run=True,
        )


def test_a_refusal_is_matched_to_its_own_issue_and_not_to_the_others():
    """The join has to discriminate, which one refusal cannot show.

    A single-issue test passes for a `refused` that holds everything and for one
    that holds nothing - the first writes no label at all, the second writes
    them all, and with one issue in play only one of those is even visible. Two
    issues, one merge refused, separates them: #23 is left where it was and #24
    reaches `landed`."""
    client = RefusingClient(issues={23: issue_payload(23), 24: issue_payload(24)})
    client.refuses = {123}

    report = apply_checks(client, green_plan(23, 24))

    assert report.merged == (24,)
    # Both outcomes carried a `landed` transition. Exactly the one whose merge
    # GitHub accepted is applied, and it is #24's - a `refused` set holding
    # everything or nothing would show as no transition or as both.
    assert [transition.ref for transition in report.applied] == [task_ref(24)]
    assert [transition.to_state for transition in report.applied] == [DONE]


def test_a_branch_this_client_cannot_delete_is_reported_rather_than_swallowed():
    client = client_with(issues={23: issue_payload(23)})
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): summarise_checks([run("test", "success")])},
        now=NOW,
    )

    report = apply_checks(client, plan)

    # `GitHubClient` has no ref deletion and `client.py` is outside this
    # ticket's file set, so the gap degrades and says so.
    assert report.merged == (23,)
    assert report.undeleted == (branch(23),)
    assert BRANCH_METHODS[0] in report.summary()


def test_a_retry_persists_the_counter_before_it_offers_the_task_another_go():
    client = CommentingClient(issues={23: issue_payload(23)})
    plan = _plan_checks_(
        ledger(entry(23, "src/mod23.py", "tests/test_mod23.py")),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): failing("tests/test_mod23.py")},
        now=NOW,
    )

    with RecordingStore(REPO, client.log) as store:
        report = apply_checks(client, plan, store=store)
        held = store.read()[task_ref(23)]

    # ADR 0002's crash ordering, and it survived the counter changing address
    # twice over: the judgment is persisted before anything invites the task to
    # run again, so a crash between them costs an attempt rather than granting a
    # free one. What moved is where "persisted" points - the store, not a body
    # `PATCH` (ADR 0005) - and what comes after it: the label that re-readied
    # the task is gone (#152), and the retry feedback the worker greps for is
    # now the write on the far side of the boundary.
    assert [transition.to_state for transition in report.applied] == [READY]
    assert held.attempt == 1
    assert client.log.index(f"store {task_ref(23)} attempt=1") < client.log.index("comment #23")
    # And the customer's issue body was never opened, let alone written.
    assert not [line for line in client.log if "update_issue" in line or "get_issue" in line]


def test_giving_up_comments_the_failure_where_a_human_will_find_it():
    client = CommentingClient(issues={23: issue_payload(23, attempt=2)})
    plan = _plan_checks_(
        ledger(entry(23, "src/mod23.py", "tests/test_mod23.py", attempt=2)),
        pulls=pulls(pull(101, issue=23, attempt=2)),
        checks={task_ref(23): failing("tests/test_mod23.py")},
        max_attempts=3,
        now=NOW,
    )

    report = apply_checks(client, plan)

    assert [transition.to_state for transition in report.applied] == [FAILED]
    assert report.uncommented == ()
    number, text = client.comments[0]
    assert number == 23
    assert "giving up after 3 attempt(s)" in text
    assert "tests/test_mod23.py" in text


def test_a_comment_this_client_cannot_post_is_reported_rather_than_lost():
    client = client_with(issues={23: issue_payload(23, attempt=2)})
    plan = _plan_checks_(
        ledger(entry(23, attempt=2)),
        pulls=pulls(pull(101, issue=23, attempt=2)),
        checks={task_ref(23): failing("tests/test_mod23.py")},
        max_attempts=3,
        now=NOW,
    )

    report = apply_checks(client, plan)

    assert [transition.to_state for transition in report.applied] == [FAILED]
    assert report.uncommented == (23,)


def test_a_dry_run_writes_nothing_at_all():
    client = client_with(issues={23: issue_payload(23)})
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): summarise_checks([run("test", "success")])},
        now=NOW,
    )

    report = apply_checks(client, plan, dry_run=True)

    assert report.merged == ()
    assert client.log == []
    assert client.labels_on(23) == {REVIEW}


@dataclass
class RefusingStore:
    """A store that will not take one task's judgment, and takes the rest.

    This test used to delete #23's issue and let GitHub 404 the relabel, which
    is the write #152 removed - `apply_checks` no longer touches an issue's
    labels at all, so a deleted issue costs the gate nothing and the case could
    not fail. What is left inside the same `try` is the judgment write, so the
    store is where a single task's write can still go wrong while the other
    nineteen are nobody's fault.
    """

    refuses: set[TaskRef] = field(default_factory=set)
    written: list[TaskRef] = field(default_factory=list)

    def read(self) -> dict[TaskRef, Any]:
        return {}

    def write(self, judgement: Any) -> None:
        if judgement.ref in self.refuses:
            raise StoreError(f"cannot write {judgement.ref}")
        self.written.append(judgement.ref)

    def close(self) -> None:
        return None


def test_one_task_whose_judgment_the_store_refuses_does_not_cost_the_others():
    client = client_with(issues={23: issue_payload(23), 24: issue_payload(24)})
    plan = _plan_checks_(
        ledger(entry(23), entry(24)),
        pulls=pulls(pull(101, issue=23), pull(102, issue=24)),
        checks={
            task_ref(23): failing("tests/test_mod23.py"),
            task_ref(24): summarise_checks([run("test", "success")]),
        },
        now=NOW,
    )

    report = apply_checks(client, plan, store=RefusingStore(refuses={task_ref(23)}))

    # #23's retry is collected as one issue's failure rather than raised, and
    # #24's merge lands and is recorded regardless.
    assert [f.number for f in report.failures] == [23]
    assert report.merged == (24,)
    assert [transition.ref for transition in report.applied] == [task_ref(24)]
    assert [transition.to_state for transition in report.applied] == [DONE]


# --------------------------------------------------------------------------
# One pass, end to end
# --------------------------------------------------------------------------


def test_one_pass_reads_checks_only_for_review_issues_with_an_open_pull_request():
    client = DeletingClient(
        issues={23: issue_payload(23), 24: issue_payload(24, label=CLAIMED)},
        open_pulls=((101, branch(23)), (102, branch(24))),
        checks={f"{101:0>40x}": [run("test", "success")]},
    )

    book = ledger(entry(23), entry(24, label=CLAIMED))
    report = run_checks(client, book, believed=fixture_belief(book))

    # One listing plus one check read per review issue, and nothing per issue in
    # any other state: the cost is the review queue, not the ledger.
    assert [line for line in client.log if line.startswith("list_check_runs")] == [
        f"list_check_runs {101:0>40x}"
    ]
    assert report.merged == (23,)
    assert [transition.to_state for transition in report.applied] == [DONE]
    assert client.deleted == [branch(23)]


def test_one_pass_against_a_client_that_cannot_list_pull_requests_changes_nothing():
    client = BlindClient(issues={23: issue_payload(23)})

    book = ledger(entry(23))
    report = run_checks(client, book, believed=fixture_belief(book))

    assert report.plan.blind
    assert client.log == []
    # Nothing decided, so nothing recorded: "we could not look" must never read
    # as an answer about the checks.
    assert report.applied == ()
    assert report.merged == ()


def test_a_failing_pass_leaves_the_issue_ready_for_another_attempt():
    client = FakeClient(
        issues={23: issue_payload(23)},
        open_pulls=((101, branch(23)),),
        checks={
            f"{101:0>40x}": [
                run("test", "failure", text="FAILED tests/test_mod23.py::test_x - boom")
            ]
        },
    )

    book = ledger(entry(23, "src/mod23.py", "tests/test_mod23.py"))
    report = run_checks(client, book, believed=fixture_belief(book))

    # The issue's headline, end to end: a red PR is retried rather than merged.
    # It used to assert the failure text was written into the issue body too;
    # #152 removed that write, and the failing check is on the pull request.
    assert report.merged == ()
    assert [transition.to_state for transition in report.applied] == [READY]


# --------------------------------------------------------------------------
# Failing paths in more than one language (#93)
# --------------------------------------------------------------------------
#
# `foreign_failure` is the escalate-without-consuming-attempts valve: a failure
# whose paths lie entirely outside the issue's `## Files` is one no attempt can
# fix, so it goes to a human instead of burning three. Both patterns behind it
# were hardcoded to `\.py`, which turned that valve off for every non-Python
# stack - silently, with a green suite and a docstring blessing it as "allowed
# to find nothing".


@pytest.mark.parametrize("language", sorted(failures.SAMPLES))
def test_a_failing_path_is_found_in_every_language(language):
    found = failing_paths(failures.SAMPLES[language])

    assert found == failures.EXPECTED[language]


@pytest.mark.parametrize("language", sorted(failures.SAMPLES))
def test_a_foreign_failure_escalates_in_every_language(language):
    """The behaviour the paths exist for, end to end."""
    declared = failures.EXPECTED[language]
    mine = entry(23, *declared)
    someone_elses = entry(24, "src/unrelated/thing.py")
    text = failures.SAMPLES[language]

    # Named inside the declared set: the worker has something to act on.
    assert foreign_failure(mine, text) == ()
    # Named entirely outside it: no attempt can fix this, so a human gets it.
    assert foreign_failure(someone_elses, text) == declared


def test_pytest_extraction_is_unchanged():
    """The existing corpus, asserted again beside the new one.

    Widening the extension list must not change what pytest output yields, and
    this is the row that would notice.
    """
    text = "FAILED tests/test_a.py - boom\ntests/test_b.py::test_y failed\nsee src/thing.py for why"

    assert failing_paths(text) == ("tests/test_a.py", "tests/test_b.py")


def test_output_in_a_language_with_no_pattern_still_finds_nothing():
    """The documented default is correct; #93 widens it, it does not replace it.

    An ordinary retry is the right answer when nothing is named - escalation is
    an optimisation on top of a correct default, never a precondition for one.
    """
    task = entry(23, "src/mod23.py")

    assert failing_paths("Segmentation fault (core dumped)") == ()
    assert failing_paths("BUILD FAILED in 3s") == ()
    assert foreign_failure(task, "the build died") == ()


def test_a_frame_through_a_dependency_is_not_this_tasks_fault():
    """Dropped on `judge._FOREIGN`'s list, spelled the same way. A path the task
    could never have been given an answer about is not evidence either way."""
    text = (
        "  at Object.<anonymous> (node_modules/expect/build/index.js:12:5)\n"
        "  File \"/usr/lib/python3.12/unittest/case.py\", line 3\n"
        "  tests/test_calc.py:12: AssertionError\n"
    )

    assert failing_paths(text) == ("tests/test_calc.py",)


def test_an_absolute_path_is_not_a_repo_relative_one():
    assert failing_paths("thread 'x' panicked at /build/src/lib.rs:12:9:") == ()


def test_go_test_output_names_a_file_judge_cannot_see():
    """The one asymmetry between the two extractors, stated out loud.

    A Go package's tests run in that package's directory, so `go test` prints a
    bare `calc_test.go:12:`. `judge._PATH_RE` requires a slash - which is what
    stops an English sentence parsing as a file - so it cannot see this one, and
    that is a deliberate difference rather than a bug in either. `failing_paths`
    can accept it because `_test.go` is mandatory in Go and it only looks for it
    in the indented position under `--- FAIL:`.
    """
    assert failing_paths(failures.GO_TEST) == ("calc_test.go",)
    assert mentioned_paths(failures.GO_TEST) == ()


@pytest.mark.parametrize(
    "language", ["pytest", "node", "vitest", "jest", "go-build", "cargo"]
)
def test_the_two_extractors_agree_on_every_language(language):
    """`judge.mentioned_paths` and `checks.failing_paths` answer the same
    question about the same text, and this is what keeps them in step.

    `judge._PATH_RE` is **already** stack-agnostic and needed no change for
    #93. That is exactly why this test exists: nothing else would notice
    somebody narrowing it, and the two modules quietly disagreeing about which
    file a failure names is the kind of divergence that surfaces months later
    as an escalation that did not happen.
    """
    text = failures.SAMPLES[language]

    assert set(failing_paths(text)) <= set(mentioned_paths(text))
    for path in failures.EXPECTED[language]:
        assert path in mentioned_paths(text)


# --------------------------------------------------------------------------
# The commit a merge produced (#141)
# --------------------------------------------------------------------------


def test_the_merge_commit_is_kept_because_nothing_else_records_it():
    """`pr.merged` names it, and "which commit is this task now" is the one
    fact about a landed task the run directory could not otherwise recover -
    the answer to `PUT .../merge` used to be discarded on the line that made it."""
    client = FakeClient(issues={23: issue_payload(23)})
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): summarise_checks([run("test", "success")])},
        now=NOW,
    )

    report = apply_checks(client, plan)

    assert report.merged == (23,)
    assert report.commit_by_issue == {23: "deadbeef"}


def test_a_merge_that_answered_with_no_body_is_still_a_merge():
    """`merge_pull_request` is typed `Any` and a body-less 200 is a real
    answer. A merge that *landed* must not be reported as a failure because the
    commit it produced could not be read back."""

    @dataclass
    class SilentClient(FakeClient):
        def merge_pull_request(self, number: int, **kwargs: Any) -> Any:
            self.log.append(f"merge PR #{number}")
            return None

    client = SilentClient(issues={23: issue_payload(23)})
    plan = _plan_checks_(
        ledger(entry(23)),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): summarise_checks([run("test", "success")])},
        now=NOW,
    )

    report = apply_checks(client, plan)

    assert report.merged == (23,)
    assert report.ok
    assert report.commit_by_issue == {}


def test_the_report_stays_hashable():
    """Every collection on this frozen record is a tuple for this reason: a
    `dict` field makes the generated `__hash__` raise, and a frozen dataclass
    that cannot be hashed is frozen in name only."""
    assert hash(ChecksReport(plan=ChecksPlan())) == hash(ChecksReport(plan=ChecksPlan()))


# --------------------------------------------------------------------------
# The retry is told why (#248)
# --------------------------------------------------------------------------
#
# The failure text was always computed and always written - into a delimited
# block in the issue body, which nothing ever read. The worker looks in the
# comments. So a task re-dispatched because CI went red was charged an attempt
# and handed nothing, which is the one case this module's own docstring says is
# not worth creating: "a re-dispatch with identical context reproduces the
# identical result".


def test_a_worker_re_dispatched_by_a_red_check_is_told_what_failed():
    """End to end, through the reader rather than past it.

    `fetch_feedback` is the worker's own function, unmodified, run against the
    comments this cycle actually posted. That is the assertion #248 asks for: the
    bug was never that the text was wrong, it was that nothing could reach it, so
    a test reading `client.comments` would have been green throughout.
    """
    from swarm.worker.entrypoint import fetch_feedback

    client = CommentingClient(issues={23: issue_payload(23)})
    plan = _plan_checks_(
        ledger(entry(23, "src/mod23.py", "tests/test_mod23.py")),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): failing("tests/test_mod23.py")},
        now=NOW,
    )

    apply_checks(client, plan)

    delivered = fetch_feedback(client, 23)
    assert delivered.startswith("apiary: attempt 1 failed.")
    assert "failed on the pull request" in delivered
    assert "tests/test_mod23.py" in delivered


def test_the_retrys_comment_is_the_one_the_worker_greps_for():
    """The first line is a contract, and this pins it against the worker's own
    constant rather than against a copy of it here."""
    from swarm.worker.entrypoint import FEEDBACK_PREFIX

    client = CommentingClient(issues={23: issue_payload(23)})
    plan = _plan_checks_(
        ledger(entry(23, "src/mod23.py", "tests/test_mod23.py")),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): failing("tests/test_mod23.py")},
        now=NOW,
    )

    apply_checks(client, plan)

    assert [number for number, _ in client.comments] == [23]
    assert client.comments[0][1].startswith(FEEDBACK_PREFIX)


def test_a_give_up_is_not_offered_to_the_next_attempt():
    """The give-up comment must **not** match the prefix, and it does not.

    Nothing will re-dispatch this task, so a comment `fetch_feedback` would pick
    up is a comment that reaches a worker only if a human re-readies the issue -
    at which point "giving up after 3 attempt(s)" is the wrong thing to hand it.
    The two comments are deliberately different sentences.
    """
    from swarm.worker.entrypoint import fetch_feedback

    client = CommentingClient(issues={23: issue_payload(23, attempt=2)})
    plan = _plan_checks_(
        ledger(entry(23, "src/mod23.py", "tests/test_mod23.py", attempt=2)),
        pulls=pulls(pull(101, issue=23, attempt=2)),
        checks={task_ref(23): failing("tests/test_mod23.py")},
        max_attempts=3,
        now=NOW,
    )

    apply_checks(client, plan)

    assert client.comments and "giving up" in client.comments[0][1]
    assert fetch_feedback(client, 23) == ""


def test_the_ci_output_is_fenced_because_it_is_foreign_text():
    """A CI log line reading `## Verify` at column 0 would corrupt the contract.

    The module went to lengths about this for the body block it used to write;
    the comment carries the same text and inherits the same hazard, so the fence
    is asserted rather than assumed.
    """
    client = CommentingClient(issues={23: issue_payload(23)})
    plan = _plan_checks_(
        ledger(entry(23, "src/mod23.py", "tests/test_mod23.py")),
        pulls=pulls(pull(101, issue=23)),
        checks={task_ref(23): failing("tests/test_mod23.py")},
        now=NOW,
    )

    apply_checks(client, plan)

    assert "```" in client.comments[0][1]


#: Every rule in this module that builds a `Transition`, each in a world where
#: the carried label and the believed state disagree (#243, #152). `swarm:done`
#: is what a human typed onto #23 while its pull request was in flight; the
#: belief still says `review`, because the resolver reads that off the open pull
#: request rather than off the label.
#: Two labels per rule, never one - see `tests/test_reconcile.py`.
CARRIED = ("swarm:done", "swarm:blocked")

CARRIED_LABEL_RULES: tuple[tuple[str, Callable[[str], ChecksPlan], str], ...] = (
    (
        "CI failed outside this issue's files",
        lambda label: _plan_checks_(
            relabelled(label),
            pulls=pulls(pull(101, issue=23)),
            checks={task_ref(23): failing("tests/test_other.py")},
            believed=REVIEWED,
            now=NOW,
        ),
        NEEDS_HUMAN,
    ),
    (
        "no check run was ever created",
        lambda label: _plan_checks_(
            relabelled(label),
            pulls=pulls(pull(101, issue=23, age_s=3600)),
            checks={task_ref(23): CheckSet()},
            policy=MergePolicy(zero_check_grace_s=300),
            believed=REVIEWED,
            now=NOW,
        ),
        NEEDS_HUMAN,
    ),
    (
        "the checks passed and it merges",
        lambda label: _plan_checks_(
            relabelled(label),
            pulls=pulls(pull(101, issue=23)),
            checks={task_ref(23): summarise_checks([run("test", "success")])},
            believed=REVIEWED,
            now=NOW,
        ),
        LANDED,
    ),
    (
        "CI failed and the budget holds",
        lambda label: _plan_checks_(
            relabelled(label),
            pulls=pulls(pull(101, issue=23)),
            checks={task_ref(23): failing("tests/test_mod23.py")},
            max_attempts=3,
            believed=REVIEWED,
            now=NOW,
        ),
        ELIGIBLE,
    ),
    (
        "CI failed and the attempts are spent",
        lambda label: _plan_checks_(
            relabelled(label, attempt=2),
            pulls=pulls(pull(101, issue=23, attempt=2)),
            checks={task_ref(23): failing("tests/test_mod23.py")},
            max_attempts=3,
            believed=REVIEWED,
            now=NOW,
        ),
        NEEDS_HUMAN,
    ),
)


@pytest.mark.parametrize(
    "rule, world, decides", CARRIED_LABEL_RULES, ids=[case[0] for case in CARRIED_LABEL_RULES]
)
def test_no_rule_lets_the_label_the_issue_carries_change_what_it_writes(rule, world, decides):
    """#243, inverted by #152 - which is the whole point of the ticket.

    While the labels *were* the control plane, `from_state` answered "which
    label does this write have to remove", so a task the resolver believed
    `review` while the issue carried `swarm:done` had to name `swarm:done` or
    end up wearing two. `reconcile.write_labels` is deleted: nothing removes a
    label, so that question no longer has a subject, and `from_state` is now
    display-only and sourced from the belief through `authority.state_of`.

    What survives is the property #243 was really protecting, and it is the
    stronger direction: a label a human types onto an issue mid-review changes
    *nothing* the gate decides or writes. Both rows below are decided the same
    way and reported the same way whatever the issue is wearing, which is what
    #152 means by the label control plane being gone rather than merely unused.
    """
    written = set()
    for label in CARRIED:
        transition = world(label).transitions[0]

        assert transition.to_state == decides, f"{rule}: the rule stopped firing"
        # The belief, every time - never the label, and never a constant either:
        # `REVIEWED` is what `state_of` is asked and `review` is what it answers.
        assert transition.from_state == REVIEW_STATE, (
            f"{rule} carrying {label}: read the label instead of the belief"
        )
        written.add((transition.from_state, transition.to_state, transition.reason))

    # One outcome across both labels. A rule that had gone back to reading the
    # issue would produce two.
    assert len(written) == 1
