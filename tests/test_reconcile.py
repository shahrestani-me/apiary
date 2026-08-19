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

import datetime as dt
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable

from types import SimpleNamespace

import pytest

from fixtures.github import SentRequest, not_modified, page, response
from swarm.containers.manager import RUNNING_STATE, DockerError, Handle
from swarm.github.branches import task_branch
from swarm.github.client import GitHubHTTPError
from swarm.github.ledger import (
    ContractError,
    LabelRepair,
    Ledger,
    LedgerEntry,
    parse_contract,
    render_marker,
)
from swarm.github.readiness import IssueState
from swarm.github.refs import issue_number
from swarm.github.refs import task_ref as ref
from swarm.orchestrator.dispatcher import CLAIMED, REVIEW, Capacity
from swarm.orchestrator.reconcile import (
    COMMENT_TAIL_CHARS,
    DEFAULT_INFRASTRUCTURE_CAP,
    EMPTY_SIGNATURE,
    INFRASTRUCTURE_CAP_ENV,
    InfrastructurePolicy,
    infrastructure_streaks,
    observed_records,
    COMMENT_METHOD,
    PULLS_METHOD,
    DONE,
    FAILED,
    READY,
    CycleReport,
    Reconciler,
    ReconcilePlan,
    ReconcileReport,
    Snapshot,
    Transition,
    apply_plan,
    diagnose,
    fold,
    plan_reconcile,
    retry_comment,
    rewrite_marker,
    signature,
    write_labels,
)
from swarm.orchestrator.lifecycle import internal_state, lifecycle_events
from swarm.run import Run
from swarm.store import STORE_DIR_ENV, SqliteTaskStore, TaskJudgement
from swarm.state import ProgressJudgement
from swarm.taskref import TaskRef
from swarm.worker.result import ResultRecord, write_result
from swarm.orchestrator.authority import Belief
from swarm.orchestrator.derived import ELIGIBLE, LANDED, NEEDS_HUMAN
from swarm.orchestrator.derived import CLAIMED as CLAIMED_STATE
from swarm.orchestrator.derived import REVIEW as REVIEW_STATE

from fixtures.markers import legacy_marker

REPO = "shahrestani-me/apiary"
RUN_ID = "apiary-20260814-142530-k3f9qz"
BASE_COMMIT = "9f2c1ab3d4e5f60718293a4b5c6d7e8f90a1b2c3"
BLOCKED = "swarm:blocked"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def store_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every store this module opens lands under `tmp_path`.

    Autouse and unconditional, because the failure it prevents is silent: a
    test that forgot to redirect the root would open the *operator's* store at
    `.swarm/store`, read a real project's retry budgets and write test
    judgments into them. Nothing would fail; the next real run would simply
    believe something untrue about its own history.
    """
    root = tmp_path / "store"
    monkeypatch.setenv(STORE_DIR_ENV, str(root))
    return root


def task_store() -> SqliteTaskStore:
    """This repository's store, under whatever `store_root` redirected to."""
    return SqliteTaskStore.open(REPO)


def entry(
    number: int,
    *files: str,
    label: str = READY,
    attempt: int = 0,
    blocker: str = "",
    streak: int | None = None,
) -> LedgerEntry:
    return LedgerEntry(
        number=number,
        title=f"issue {number}",
        task_id=f"task-{number}",
        attempt=attempt,
        blocker=blocker,
        streak=streak,
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
    return IssueState(ref=ref(number), state="closed", state_reason=reason)


def record(
    issue: int,
    exit_code: int,
    *,
    attempt: int = 0,
    reason: str = "",
    verify_output: str = "",
) -> ResultRecord:
    return ResultRecord(
        run_id=RUN_ID,
        issue=issue,
        attempt=attempt,
        exit_code=exit_code,
        reason=reason or "the verify command failed",
        verify_output=verify_output,
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
    #: issue -> the attempt it was spawned with. See `spawn`.
    attempts: dict[int, int | None] = field(default_factory=dict)

    def find(self, *, ref: TaskRef | None = None) -> list[Handle]:
        # The fake keeps docker's int-keyed bookkeeping; only the seam changed.
        self.log.append("find")
        issue = None if ref is None else issue_number(ref)
        found = list(self.handles.values())
        return [h for h in found if issue is None or h.issue == issue]

    def dispose(self, handle: Handle) -> str:
        self.log.append(f"dispose #{handle.issue}")
        if self.dispose_error is not None:
            raise self.dispose_error
        self.handles.pop(handle.issue, None)
        return ""

    def spawn(
        self,
        task: TaskRef,
        base_commit: str,
        *,
        issue: int | None = None,
        image: str | None = None,
        attempt: int | None = None,
    ) -> Handle:
        assert issue is not None and task.label_value == str(issue)
        self.log.append(f"spawn #{issue}")
        # Recorded, not asserted here: which attempt a container was told is
        # `test_dispatcher`'s question. This fake only has to accept it, or every
        # cycle test in this file fails on a keyword.
        self.attempts[issue] = attempt
        handle = Handle(id=f"{issue:0>64x}", run_id=RUN_ID, issue=issue, image=image or "")
        self.handles[issue] = handle
        return handle

    @property
    def disposed(self) -> list[int]:
        return [int(line.split("#")[1]) for line in self.log if line.startswith("dispose")]

    @property
    def spawned(self) -> list[int]:
        return [int(line.split("#")[1]) for line in self.log if line.startswith("spawn")]


def running(*issues: int) -> dict[TaskRef, Handle]:
    """The handle map a cycle holds: keyed by task, valued by container."""
    return {ref(n): Handle(id=f"{n:0>64x}", run_id=RUN_ID, issue=n) for n in issues}


class ModelCalled(BaseException):
    """Not an `Exception`: `judge` and `assess` both catch those and report the
    cycle unresolved, which would swallow the assertion this exists to make."""


class Never:
    """An oracle that fails the test if a cycle asks it anything."""

    def invoke(self, messages: Any) -> Any:
        raise ModelCalled(f"a cycle asked a model: {messages!r}")


@dataclass
class Judged:
    """A scripted judge. Inert, and above all *injected*.

    Step 5 consults a model on a cycle that changed nothing while nothing is in
    flight, and this suite is full of those. On a host with Ollama running, a
    reconcile test that reached the real oracle would not fail - it would
    quietly spend a 31B inference per quiet cycle and pass slowly, which is how
    a 0.5 s suite became a 207 s one. So every reconciler built here is given
    one of these, and no test in this file can reach a model at all.
    """

    judgement: ProgressJudgement = field(
        default_factory=lambda: ProgressJudgement(
            request_satisfied=False,
            progress_being_made=True,
            in_loop=False,
            reason="the test's judge",
        )
    )
    asked: list[Any] = field(default_factory=list)

    def invoke(self, messages: Any) -> ProgressJudgement:
        self.asked.append(messages)
        return self.judgement


def reconciler(client: Any, fleet: Any = None, **kwargs: Any) -> Reconciler:
    kwargs.setdefault("capacity", Capacity(slots=3, configured=2))
    # Hermetic by default; the step-5 tests below script their own answers, and
    # the goal gate is opt-in because it writes issues.
    kwargs.setdefault("oracle", Judged())
    kwargs.setdefault("goal_gate", False)
    # Nothing sleeps for real: the pacing test injects its own recorder, and
    # every other test here would otherwise pay `DEFAULT_INTERVAL_S` per cycle
    # to assert something that has nothing to do with the clock.
    kwargs.setdefault("sleep", lambda _seconds: None)
    # Required rather than defaulted on `Reconciler`, so this is the one place
    # a test can forget it - and a forgotten store is a `TypeError` here rather
    # than a run that silently forgets what it decided.
    kwargs.setdefault("store", task_store())
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
        states={ref(4): closed(4)},
        running=[ref(4)],
    )

    # The headline of #22: a human closing an issue mid-run is a feature of
    # putting the ledger on GitHub, and the container has to go with it.
    assert [str(t) for t in plan.transitions] == [
        "#4: claimed -> landed (closed as completed on GitHub)"
    ]
    assert [d.ref for d in plan.disposals] == [ref(4)]


def test_a_merged_pull_request_is_read_from_the_issue_it_closed():
    plan = plan_reconcile(
        ledger(entry(4, label=REVIEW)), states={ref(4): closed(4)}, running=[ref(4)]
    )

    # `Closes #<n>` means the merge closes the issue, so the cheap read already
    # carries the signal and nothing has to page through every PR ever opened.
    assert plan.transitions[0].to_state == LANDED
    assert plan.disposals[0].ref == ref(4)


def test_an_issue_closed_as_not_planned_is_failed_rather_than_done():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)),
        states={ref(4): closed(4, "not_planned")},
        running=[ref(4)],
    )

    # The same judgement readiness makes about a dependency: work somebody
    # explicitly decided not to do did not happen.
    assert plan.transitions[0].to_state == NEEDS_HUMAN
    assert "not planned" in plan.transitions[0].reason


def test_an_issue_a_human_already_marked_done_keeps_no_container():
    plan = plan_reconcile(ledger(entry(4, label=DONE)), running=[ref(4)])

    assert plan.transitions == ()
    assert [d.ref for d in plan.disposals] == [ref(4)]


def test_a_container_whose_issue_left_the_ledger_is_disposed():
    # A human deleted the issue, or stripped its `swarm:*` label - either way
    # nothing will look at that work again and the clone is held for nobody.
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)), states={ref(4): closed(4)}, running=[ref(4), ref(9)]
    )

    assert sorted(d.ref for d in plan.disposals) == [ref(4), ref(9)]
    assert "no longer in the ledger" in next(d for d in plan.disposals if d.ref == ref(9)).reason


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
        results={ref(4): record(4, 1)},
        running=[ref(4)],
        max_attempts=3,
    )

    transition = plan.transitions[0]
    assert (transition.to_state, transition.attempt) == (ELIGIBLE, 1)
    assert [d.ref for d in plan.disposals] == [ref(4)]


def test_a_failed_worker_at_the_cap_is_handed_to_a_human():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=2)),
        results={ref(4): record(4, 1, attempt=2)},
        max_attempts=3,
    )

    transition = plan.transitions[0]
    assert (transition.to_state, transition.attempt) == (NEEDS_HUMAN, 3)
    # §5: the counter is an upper bound on attempts made. Giving up early puts
    # a human in front of the problem; looping forever looks healthy.
    assert "cap of 3" in transition.reason
    assert transition.comment


def test_an_infrastructure_failure_does_not_consume_an_attempt():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=1)),
        results={ref(4): record(4, 2, attempt=1, reason="ollama refused the connection")},
        max_attempts=3,
    )

    transition = plan.transitions[0]
    # A broken Ollama would otherwise burn every task's budget before anyone
    # noticed. `attempt=None` means the counter is not written at all.
    assert (transition.to_state, transition.attempt) == (ELIGIBLE, None)


def test_an_unknown_exit_code_is_charged_like_a_failure():
    # 137 from the OOM killer, 143 from a stop. `ResultRecord.consumes_attempt`
    # already settled this; the reconciler must not re-decide it.
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)),
        results={ref(4): record(4, 137)},
        max_attempts=3,
    )

    assert plan.transitions[0].attempt == 1


def test_a_worker_that_published_is_moved_to_review_here():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)),
        results={ref(4): record(4, 0)},
        running=[ref(4)],
    )

    # `claimed -> review` is this module's row since #148. It used to be the
    # worker's, on the argument that a container knows the PR exists at the
    # instant it does - but that was a tracker write issued from inside
    # model-generated code to announce a fact `derived.py` now reads off the
    # pull request itself, so the label is written from out here instead.
    transition = plan.transitions[0]
    assert (transition.from_state, transition.to_state) == (CLAIMED_STATE, REVIEW_STATE)
    # The attempt succeeded: nothing is charged for it, and `review -> ready` is
    # where a rejected pull request is accounted for.
    assert transition.attempt is None
    # And the record is retired, so a second cycle cannot re-observe it.
    assert transition.observed_record
    assert [d.ref for d in plan.disposals] == [ref(4)]


def test_a_records_verdict_is_not_applied_twice():
    # The counter moves and the artifact does not, so the record of attempt 0
    # must stop counting once the issue is on attempt 1.
    plan = plan_reconcile(
        ledger(entry(4, label=READY, attempt=1)),
        results={ref(4): record(4, 1, attempt=0)},
    )

    assert plan.transitions == ()


def test_a_record_behind_the_counter_is_discarded_and_the_claim_stands():
    """The live wedge, pinned as the guard's correct behaviour: a retry's
    model call blew up and the worker filed the exit-2 record under attempt 0
    against a ledger already on attempt 2. The staleness guard rightly
    discards it - a record behind the counter has already been acted on, for
    all this cycle can tell - so the issue stays claimed against an exited
    container, forever. The guard is not the bug; the record writer had to
    learn to tell the truth (`worker/entrypoint.py` stamps the real attempt,
    and `worker/result.py` files an unknowable one under the next free index,
    which is never behind the counter)."""
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=2)),
        results={
            ref(4): record(4, 2, attempt=0, reason="model call failed: OutputParserException")
        },
        running=[ref(4)],
    )

    assert plan.transitions == () and plan.disposals == ()


def test_the_corrected_record_is_observed_and_costs_no_attempt():
    """The same failure carrying its real attempt: the observation proceeds,
    the issue is re-readied, the container is disposed - and the budget is
    untouched, because an infrastructure verdict never consumes an attempt
    however late in the retry sequence it lands."""
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=2)),
        results={
            ref(4): record(4, 2, attempt=2, reason="model call failed: OutputParserException")
        },
        running=[ref(4)],
    )

    transition = plan.transitions[0]
    assert (transition.to_state, transition.attempt) == (ELIGIBLE, None)
    # Counted toward the infrastructure ceiling, not the task's budget.
    assert transition.infrastructure
    assert "OutputParserException" in transition.reason
    assert [d.ref for d in plan.disposals] == [ref(4)]


# --------------------------------------------------------------------------
# Retry feedback: diagnose() and the comment a retry leaves behind
# --------------------------------------------------------------------------

SQLALCHEMY_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "app/models.py", line 1, in <module>\n'
    "    import sqlalchemy\n"
    "ModuleNotFoundError: No module named 'sqlalchemy'\n"
)


def test_a_missing_module_is_diagnosed_by_name():
    finding = diagnose(SQLALCHEMY_TRACEBACK)

    assert "missing dependency 'sqlalchemy'" in finding
    assert "requirements.txt" in finding
    assert "standard library" in finding


def test_a_missing_submodule_names_the_top_level_package():
    # The distribution to declare is `sqlalchemy`, not `sqlalchemy.orm`.
    finding = diagnose("ModuleNotFoundError: No module named 'sqlalchemy.orm'")

    assert "missing dependency 'sqlalchemy'" in finding


def test_an_import_error_is_diagnosed_from_the_module_it_names():
    finding = diagnose("ImportError: cannot import name 'Session' from 'sqlalchemy.orm'")

    assert "missing dependency 'sqlalchemy'" in finding


def test_unrecognised_output_yields_no_diagnosis():
    # No guess: a wrong diagnosis is repeated to the next attempt as a fact.
    assert diagnose("FAILED tests/test_calc.py::test_add - assert 3 == 4") == ""
    assert diagnose("") == ""


#: The worker's pinned parse-gate failure, exactly as `worker.edit.
#: syntax_failure` writes it - the one SyntaxError shape `diagnose` may
#: recognise, because the worker authored the sentence.
WORKER_SYNTAX_FAILURE = (
    "python syntax error in tests/test_wallet.py, line 7: invalid syntax\n"
    "    amount=3.5 far, typo in my thought process again\n\n"
    "The verify command was not run: a file that does not parse fails every "
    "test that imports it."
)


def test_the_workers_pinned_syntax_failure_is_diagnosed_by_file():
    finding = diagnose(WORKER_SYNTAX_FAILURE)

    assert "syntax error in tests/test_wallet.py" in finding
    assert "must parse" in finding
    # No line number in the diagnosis: `signature` uses it as the failure's
    # identity, and the same broken file at a moved line is the same blocker.
    assert "line 7" not in finding


def test_a_raw_cpython_syntax_traceback_is_still_not_diagnosed():
    # Attributing a suite's own SyntaxError to a file would mean parsing
    # pytest's surrounding lines, which is a guess - the honesty rule holds.
    assert diagnose(SYNTAX_ERROR_OUTPUT) == ""


#: The worker's pinned context-overflow sentence, exactly as
#: `worker.edit.fit_context` writes it when the goal and the writable set
#: alone exceed the window - the other worker-authored line `diagnose` may
#: recognise, for the same honesty reason as the syntax one.
TOO_LARGE_OUTPUT = (
    "the task is too large for the worker's context window (~9210 tokens "
    "against a budget of 12224; SWARM_WORKER_CTX=16384), and the plan should "
    "split it into tasks with smaller file sets. The goal and the declared "
    "files alone overflow the window before any read-only context is added, "
    "so a retry with the same file set will overflow identically."
)


def test_the_workers_pinned_overflow_sentence_is_diagnosed_as_split_advice():
    finding = diagnose(TOO_LARGE_OUTPUT)

    assert "too large for the worker's context window" in finding
    assert "split" in finding
    assert "## Files" in finding
    # The numbers stay out of the diagnosis: `signature` uses it as the
    # failure's identity, and a retry whose folded-in feedback nudges the
    # token estimate is the same blocker, not budget-renewing progress.
    assert "9210" not in finding
    assert "12224" not in finding


def test_the_overflow_diagnosis_matches_the_workers_real_sentence():
    """Cross-module pin: the sentence `fit_context` writes today is the one
    this regex recognises, so neither side can be reworded alone."""
    from swarm.worker.edit import SourceFile, fit_context

    _, failure = fit_context("goal", (SourceFile("src/big.py", "x" * 60_000),), (), num_ctx=4096)

    assert failure is not None
    assert diagnose(failure) != ""


def test_two_overflows_with_different_numbers_sign_identically():
    moved = TOO_LARGE_OUTPUT.replace("~9210", "~9944")

    assert signature(TOO_LARGE_OUTPUT) == signature(moved)


def test_a_moved_syntax_error_in_the_same_file_signs_identically():
    moved = WORKER_SYNTAX_FAILURE.replace("line 7", "line 12")

    assert signature(WORKER_SYNTAX_FAILURE) == signature(moved)


def test_a_retried_issue_carries_the_failure_as_a_comment():
    """The defect this feature exists for: issue #21 of the first live run
    failed 3/3 attempts on the identical ModuleNotFoundError, because a retry
    posted nothing and the next worker saw only the issue body."""
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=0)),
        results={ref(4): record(4, 1, verify_output=SQLALCHEMY_TRACEBACK)},
        max_attempts=3,
    )

    transition = plan.transitions[0]
    assert transition.to_state == ELIGIBLE
    assert transition.comment.startswith("apiary: attempt 1 failed")
    assert "worker exit 1" in transition.comment
    assert "missing dependency 'sqlalchemy'" in transition.comment
    # The evidence travels in a fence, so the worker can read it and a human
    # can skim past it.
    assert "```" in transition.comment
    assert "ModuleNotFoundError" in transition.comment


def test_a_retry_comment_without_verify_output_still_states_the_reason():
    # The PR-closed-unmerged path has no record to quote; the reason is the
    # whole feedback, and there is no empty fence dangling under it.
    plan = plan_reconcile(
        ledger(entry(4, label=REVIEW, attempt=0)),
        open_branches=frozenset(),
        max_attempts=3,
    )

    transition = plan.transitions[0]
    assert transition.comment.startswith("apiary: attempt 1 failed")
    assert "closed without merging" in transition.comment
    assert "```" not in transition.comment


def test_the_retry_comments_tail_is_bounded():
    huge = "x" * (COMMENT_TAIL_CHARS * 3) + "\nModuleNotFoundError: No module named 'flask'"
    comment = retry_comment(1, "worker exit 1: the verify command failed", huge)

    assert len(comment) < COMMENT_TAIL_CHARS + 1_000
    assert "earlier characters elided" in comment
    # The diagnosis reads the whole output, not the clipped tail.
    assert "missing dependency 'flask'" in comment


def test_the_retry_comment_lands_on_the_issue_through_the_apply_path():
    client = CommentingClient(issues={4: issue_payload(4, label=CLAIMED)})
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)),
        results={ref(4): record(4, 1, verify_output=SQLALCHEMY_TRACEBACK)},
        max_attempts=3,
    )

    report = apply_plan(client, plan)

    assert report.uncommented == ()
    number, text = client.comments[0]
    assert number == 4
    assert text.startswith("apiary: attempt 1 failed")
    assert "missing dependency 'sqlalchemy'" in text


# --------------------------------------------------------------------------
# The retry budget is per blocker, not per task
# --------------------------------------------------------------------------

# Not diagnosable - `diagnose` recognises only the worker's own pinned
# syntax-failure line, never a raw CPython traceback - so this exercises the
# normalised-tail tier of `signature`.
SYNTAX_ERROR_OUTPUT = (
    '  File "tests/test_wallet.py", line 7\n'
    "    def test_balance(:\n"
    "                     ^\n"
    "SyntaxError: invalid syntax\n"
)


def test_the_same_failure_signs_identically_wherever_it_happened():
    # The diagnosis is the identity: the same missing module is the same
    # blocker even when the importing file and line have moved.
    moved = SQLALCHEMY_TRACEBACK.replace("line 1", "line 88").replace("app/models.py", "src/db.py")

    assert signature(SQLALCHEMY_TRACEBACK) == signature(moved)
    assert signature(SQLALCHEMY_TRACEBACK) == signature(SQLALCHEMY_TRACEBACK)


def test_paths_and_line_numbers_do_not_change_an_undiagnosed_signature():
    a = "E   SyntaxError: invalid syntax (tests/test_wallet.py, line 12)"
    b = "E   SyntaxError: invalid syntax (tests/test_ledger.py, line 30)"

    assert signature(a) == signature(b)


def test_different_failures_sign_differently():
    assert signature(SQLALCHEMY_TRACEBACK) != signature(SYNTAX_ERROR_OUTPUT)
    assert signature("E   AssertionError: assert 3 == 4") != signature(SYNTAX_ERROR_OUTPUT)


def test_empty_output_signs_as_the_fixed_sentinel():
    assert signature("") == EMPTY_SIGNATURE
    assert signature("   \n  ") == EMPTY_SIGNATURE


def test_a_different_failure_renews_the_retry_budget():
    """The live defect this feature exists for: issue #21's environment was
    fixed by hand, attempt 4 failed on a brand-new SyntaxError - proof the old
    blocker was gone - and the orchestrator gave up anyway because the counter
    was at its cap. A changed signature must renew the budget."""
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=2, blocker=signature(SQLALCHEMY_TRACEBACK), streak=2)),
        results={ref(4): record(4, 1, attempt=2, verify_output=SYNTAX_ERROR_OUTPUT)},
        max_attempts=3,
        max_total_attempts=9,
    )

    transition = plan.transitions[0]
    # Attempt 3 of a 3-cap would have failed under the old arithmetic.
    assert transition.to_state == ELIGIBLE
    # The total keeps counting for honesty; the streak restarts.
    assert (transition.attempt, transition.streak) == (3, 1)
    assert transition.blocker == signature(SYNTAX_ERROR_OUTPUT)
    # The machine-findable prefix is unchanged; the renewal is said out loud.
    assert transition.comment.startswith("apiary: attempt 3 failed")
    assert "previous blocker is gone" in transition.comment
    assert "renewed" in transition.comment
    assert "streak 1 of 3" in transition.comment
    assert "total 3 of 9" in transition.comment


def test_the_same_failure_repeating_gives_up_exactly_as_before():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=2, blocker=signature(SQLALCHEMY_TRACEBACK), streak=2)),
        results={ref(4): record(4, 1, attempt=2, verify_output=SQLALCHEMY_TRACEBACK)},
        max_attempts=3,
        max_total_attempts=9,
    )

    transition = plan.transitions[0]
    assert transition.to_state == NEEDS_HUMAN
    assert "cap of 3" in transition.reason
    assert transition.comment.startswith("apiary: giving up after 3 attempt(s)")
    assert "failed the same way" in transition.comment


def test_an_old_marker_without_a_signature_behaves_exactly_as_before():
    # Back-compat: no blocker recorded means no renewal, however new the
    # failure looks - the pre-signature arithmetic on the attempt counter.
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=2)),
        results={ref(4): record(4, 1, attempt=2, verify_output=SYNTAX_ERROR_OUTPUT)},
        max_attempts=3,
        max_total_attempts=9,
    )

    transition = plan.transitions[0]
    assert (transition.to_state, transition.attempt) == (NEEDS_HUMAN, 3)
    assert "cap of 3" in transition.reason


def test_a_renewed_blocker_that_then_repeats_burns_its_own_budget():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=3, blocker=signature(SYNTAX_ERROR_OUTPUT), streak=1)),
        results={ref(4): record(4, 1, attempt=3, verify_output=SYNTAX_ERROR_OUTPUT)},
        max_attempts=3,
        max_total_attempts=9,
    )

    transition = plan.transitions[0]
    assert transition.to_state == ELIGIBLE
    assert (transition.attempt, transition.streak) == (4, 2)
    # A repeat is not a renewal, so the comment must not claim progress.
    assert "renewed" not in transition.comment


def test_the_hard_cap_gives_up_whatever_the_signature_says():
    # Failures that keep changing renew the per-blocker budget, but the total
    # is bounded: a task failing a new way every time is not converging.
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=8, blocker=signature(SQLALCHEMY_TRACEBACK), streak=1)),
        results={ref(4): record(4, 1, attempt=8, verify_output=SYNTAX_ERROR_OUTPUT)},
        max_attempts=3,
        max_total_attempts=9,
    )

    transition = plan.transitions[0]
    assert (transition.to_state, transition.attempt) == (NEEDS_HUMAN, 9)
    assert "total cap of 9" in transition.reason
    assert "total retry budget is spent" in transition.comment


def test_a_cap_of_one_gives_up_even_on_a_renewed_failure():
    # max_attempts=1 is the operator saying "never retry"; a renewal restarts
    # the streak at 1, which is already at that cap.
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=1, blocker=signature(SQLALCHEMY_TRACEBACK), streak=1)),
        results={ref(4): record(4, 1, attempt=1, verify_output=SYNTAX_ERROR_OUTPUT)},
        max_attempts=1,
        max_total_attempts=9,
    )

    assert plan.transitions[0].to_state == NEEDS_HUMAN


def test_a_retry_with_no_output_signs_as_the_sentinel_and_burns_down():
    # The PR-closed-unmerged path has nothing to sign; the sentinel keeps it
    # consistent with itself, so closing the PR again is the same blocker.
    plan = plan_reconcile(
        ledger(entry(4, label=REVIEW, attempt=1, blocker=EMPTY_SIGNATURE, streak=1)),
        open_branches=frozenset(),
        max_attempts=3,
        max_total_attempts=9,
    )

    transition = plan.transitions[0]
    assert transition.to_state == ELIGIBLE
    assert (transition.blocker, transition.streak) == (EMPTY_SIGNATURE, 2)


def test_the_signature_is_persisted_in_the_store_before_the_relabel():
    """Where #154-#156 wrote the signature into the issue body, #159 writes it
    into apiary's own store. The ordering guarantee is the one that matters and
    it is unchanged: the record lands before the label goes back to ready, so a
    crash between the two costs an attempt with its signature intact."""
    client = FakeClient(issues={4: issue_payload(4, label=CLAIMED)})
    store = task_store()
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)),
        results={ref(4): record(4, 1, verify_output=SQLALCHEMY_TRACEBACK)},
        max_attempts=3,
    )

    apply_plan(client, plan, store=store)

    sig = signature(SQLALCHEMY_TRACEBACK)
    held = store.read()[ref(4)]
    assert (held.attempt, held.blocker, held.streak) == (1, sig, 1)
    # The judgment is durable before the counter is, and the counter before the
    # label: `update_issue` is the counter's write, so the store's must precede
    # a body that already carries the bump.
    assert "attempt=1" in client.issues[4]["body"]
    assert client.log.index("update_issue #4") < client.log.index(f"+{READY} #4")
    # And it is nowhere near the customer's issue - that is the whole point of
    # ADR 0002.
    body_text = client.issues[4]["body"]
    assert "blocker=" not in body_text
    assert "streak=" not in body_text
    contract = parse_contract(4, body_text)
    assert (contract.attempt, contract.blocker, contract.streak) == (1, "", None)


def test_folding_a_signature_transition_updates_the_in_memory_ledger():
    transition = Transition(
        ref=ref(4),
        from_state=CLAIMED_STATE,
        to_state=ELIGIBLE,
        reason="worker exit 1",
        task_id="task-4",
        attempt=1,
        blocker="ab12cd34ef",
        streak=1,
    )

    folded = fold(ledger(entry(4, label=CLAIMED)), [transition])

    after = folded.entries["task-4"]
    assert (after.attempt, after.blocker, after.streak) == (1, "ab12cd34ef", 1)


def test_a_counter_bump_without_a_signature_clears_the_stale_record():
    # checks, mergeability and recovery consume attempts through channels with
    # no verify output to sign; their transition carries no signature, so the
    # store is written with none and the next failure is judged by the old
    # arithmetic - the direction that can only give up early, never late (§5).
    client = FakeClient(issues={4: issue_payload(4, label=CLAIMED)})
    store = task_store()
    store.write(TaskJudgement(ref=ref(4), attempt=1, blocker="ab12cd34ef", streak=1))
    plan = ReconcilePlan(
        transitions=(
            Transition(
                ref=ref(4),
                from_state=CLAIMED_STATE,
                to_state=ELIGIBLE,
                reason="a stale claim, with nothing to sign",
                task_id="task-4",
                attempt=2,
            ),
        )
    )

    apply_plan(client, plan, store=store)

    held = store.read()[ref(4)]
    assert (held.attempt, held.blocker, held.streak) == (2, "", None)


def test_a_marker_still_carrying_an_older_builds_signature_sheds_it_on_the_next_bump():
    # The upgrade path: the parse still reads `blocker=`/`streak=` so a live
    # repository's budgets survive the change, and the first rewrite takes them
    # out of the body for good.
    original = legacy_marker("task-4", 1, blocker="ab12cd34ef", streak=1)

    updated = rewrite_marker(original, "task-4", 2)

    assert updated.splitlines()[0] == render_marker("task-4", 2)
    assert "blocker=" not in updated
    assert "streak=" not in updated


# --------------------------------------------------------------------------
# Pull requests
# --------------------------------------------------------------------------


def test_a_pull_request_closed_without_merging_returns_the_issue_to_the_pool():
    plan = plan_reconcile(
        ledger(entry(4, label=REVIEW, attempt=0)),
        open_branches=frozenset(),
        running=[ref(4)],
        max_attempts=3,
    )

    transition = plan.transitions[0]
    # The attempt is consumed: the work was done and rejected, and a retry that
    # costs nothing can be rejected forever.
    assert (transition.to_state, transition.attempt) == (ELIGIBLE, 1)
    assert [d.ref for d in plan.disposals] == [ref(4)]
    assert plan.blind is False


def test_an_open_pull_request_is_left_alone():
    plan = plan_reconcile(
        ledger(entry(4, label=REVIEW)),
        open_branches=frozenset({task_branch(ref(4), 0)}),
    )

    assert plan.transitions == ()


def test_a_published_workers_container_is_disposed_without_waiting_for_the_merge():
    plan = plan_reconcile(
        ledger(entry(4, label=REVIEW)),
        open_branches=frozenset({task_branch(ref(4), 0)}),
        results={ref(4): record(4, 0)},
        running=[ref(4)],
    )

    # The record is the worker's last act, so the container is only holding a
    # clone. Waiting for the merge would leak one per task for the length of
    # the review, and the label is already where the worker put it.
    assert plan.transitions == ()
    assert [d.ref for d in plan.disposals] == [ref(4)]


def test_pull_requests_that_could_not_be_listed_are_not_read_as_closed():
    plan = plan_reconcile(ledger(entry(4, label=REVIEW)), open_branches=None)

    # The whole reason `open_branches` is `None` rather than an empty set. An
    # empty set means every review PR is gone; None means we did not look, and
    # conflating them relabels the entire review queue.
    assert plan.transitions == ()
    assert plan.blind is True


# --------------------------------------------------------------------------
# What this function does not read
# --------------------------------------------------------------------------


def _decisions(plan: ReconcilePlan) -> tuple[Any, ...]:
    """A plan as the decisions it carries, with `from_state` struck out.

    `Transition.from_state` is the label the write has to *remove*, so it says
    what the issue was wearing rather than what was decided about it - two runs
    that decided identically over differently-labelled issues differ there and
    nowhere else. `test_authority.outcome` drops the label-write log for the
    same reason and says so at more length.
    """
    return (
        tuple(replace(t, from_state="") for t in plan.transitions),
        tuple(plan.disposals),
    )


def test_plan_reconcile_cannot_tell_blocked_from_ready():
    """`swarm:blocked`'s arm of the cutover pair can never fail, and this is why.

    #228 asked, of the three arms where the pair's `outcome()` equality was
    blind to a `plan_reconcile` regression, either to make the equality fail or
    to write down why it cannot. Two of the three - `swarm:claimed` and
    `swarm:review` - were a missing result record in the fixture, and they now
    fail. This one is not a fixture gap: it is a property of the function.

    Every rule in the per-entry loop is keyed on a *closed issue*, on
    `terminal`, on `claimed` or on `review`; the two that follow it read the
    parse errors and the containers no ledger entry claims, and neither asks
    what state an entry is in. `blocked` and `eligible` appear nowhere in any of
    them, so the two labels take the same branch at every combination of the
    facts the function is given - which is what this asserts, over the whole
    product of them rather than over one world. No fixture can make an arm
    sensitive to a distinction the function under test does not draw, so the
    `swarm:blocked` arm of the cutover pair in `tests/test_authority.py` costs
    the completeness claim one path: it is evidence about readiness and
    dispatch, and no evidence at all about reconcile.

    That is not the same as saying nothing checks the label there. Readiness
    owns both waiting states and recomputes them from the dependency graph every
    cycle, which is why `swarm:blocked` is the one wrong label the pre-#147
    machine already repaired by itself (`test_authority.WRONG_LABELS`).
    """
    branch = entry(4).branch
    worlds = product(
        [{}, {ref(4): IssueState(ref=ref(4))}, {ref(4): closed(4)}],
        [{}, {ref(4): record(4, 0)}, {ref(4): record(4, 1)}, {ref(4): record(4, 2)}],
        [None, (), (branch,)],
        [(), (ref(4),)],
    )
    decided = 0
    disposed = 0
    for states, results, branches, running in worlds:
        facts: dict[str, Any] = dict(
            states=states, results=results, open_branches=branches, running=running
        )
        # `believed=None` is the label reading - the authority a regression to
        # `entry.state_label` would restore, and the only one under which the
        # question is even askable.
        blocked = plan_reconcile(ledger(entry(4, label=BLOCKED)), **facts)
        ready = plan_reconcile(ledger(entry(4, label=READY)), **facts)

        assert _decisions(blocked) == _decisions(ready), facts
        decided += len(blocked.transitions)
        disposed += len(blocked.disposals)

    # And not a comparison of nothing with nothing, which is the failure mode a
    # test shaped like this one has. 24 of the 72 worlds decide something and 12
    # of those dispose a container - exactly the 24 where the issue is closed and
    # the 12 of those carrying a container, so every decision reached here came
    # from rule 1, the one rule that does not consult the state at all. The
    # other 48 produce nothing under either label, and that emptiness is the
    # finding rather than a hole in the matrix. A rule added later that *does*
    # read `blocked` moves these numbers as well as failing the equality above,
    # which is why they are pinned rather than bounded.
    assert (decided, disposed) == (24, 12)


# --------------------------------------------------------------------------
# Malformed contracts
# --------------------------------------------------------------------------


def test_a_malformed_issue_is_failed_and_carries_the_reason_as_a_comment():
    error = ContractError(7, "Verify", "section is missing")

    plan = plan_reconcile(ledger(errors=(error,)), labels={ref(7): frozenset({READY})})

    transition = plan.transitions[0]
    assert (transition.ref, transition.from_state, transition.to_state) == (ref(7), ELIGIBLE, NEEDS_HUMAN)
    # §1.4: the parse failure is posted back on the issue that failed it.
    assert "section is missing" in transition.comment


def test_a_malformed_issue_already_failed_is_not_failed_again():
    error = ContractError(7, "Verify", "section is missing")

    plan = plan_reconcile(ledger(errors=(error,)), labels={ref(7): frozenset({FAILED})})

    # Otherwise every cycle re-labels it and comments on it again forever.
    assert plan.transitions == ()


def test_a_malformed_issue_outside_the_ledger_is_left_alone():
    error = ContractError(7, "Goal", "section is empty")

    plan = plan_reconcile(ledger(errors=(error,)), labels={ref(7): frozenset({"area/docs"})})

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

    folded = fold(
        entries, [Transition(ref(4), CLAIMED_STATE, ELIGIBLE, "exit 1", task_id="task-4", attempt=1)]
    )

    # A second listing to observe our own writes is the one request that buys
    # nothing, and the dispatcher needs the freed capacity this cycle.
    updated = folded.entries["task-4"]
    assert (updated.state_label, updated.attempt) == (READY, 1)
    assert updated.labels == frozenset({READY})
    assert entries.entries["task-4"].state_label == CLAIMED


def test_folding_a_transition_for_an_issue_outside_the_ledger_changes_nothing():
    entries = ledger(entry(4))

    unrelated = [Transition(ref(9), ELIGIBLE, NEEDS_HUMAN, "malformed")]
    assert fold(entries, unrelated).entries == entries.entries


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def test_the_counter_is_persisted_before_the_label_goes_back_to_ready():
    client = FakeClient(issues={4: issue_payload(4, label=CLAIMED)})
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)), results={ref(4): record(4, 1)}, max_attempts=3
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
        ledger(entry(4, label=CLAIMED)), results={ref(4): record(4, 1)}, max_attempts=3
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
        states={ref(4): closed(4), ref(5): closed(5)},
    )

    report = apply_plan(client, plan)

    # A human deleting an issue between the read and the write lands here. It
    # is a fact about one issue, not a reason to abandon the cycle.
    assert [t.ref for t in report.applied] == [ref(5)]
    assert [f.ref for f in report.failures] == [ref(4)]
    assert report.ok is False


def test_a_container_that_will_not_die_does_not_stop_the_labels_from_moving():
    client = FakeClient(issues={4: issue_payload(4, label=CLAIMED)})
    fleet = FakeFleet(handles=running(4))
    fleet.dispose_error = DockerError(["docker", "rm"], 1, "daemon is not responding")
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)), states={ref(4): closed(4)}, running=[ref(4)]
    )

    report = apply_plan(client, plan, fleet=fleet, handles=running(4))

    assert [t.to_state for t in report.applied] == [LANDED]
    assert report.disposed == ()
    assert "daemon is not responding" in str(report.failures[0])


def test_a_dry_run_writes_nothing_at_all():
    client = FakeClient(issues={4: issue_payload(4, label=CLAIMED)})
    fleet = FakeFleet(handles=running(4))
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED)), states={ref(4): closed(4)}, running=[ref(4)]
    )

    report = apply_plan(client, plan, fleet=fleet, handles=running(4), dry_run=True)

    assert client.log == []
    assert fleet.log == []
    assert (report.applied, report.disposed) == ((), ())


def test_a_client_with_no_comment_method_prints_the_reason_and_reports_the_gap(capsys):
    client = FakeClient(issues={7: issue_payload(7)})
    plan = plan_reconcile(
        ledger(errors=(ContractError(7, "Verify", "section is missing"),)),
        labels={ref(7): frozenset({READY})},
    )

    report = apply_plan(client, plan)

    # §1.4 wants the text on the issue and `GitHubClient` has no method for it.
    # Printing is not a substitute; it is what stops the reason being lost.
    assert report.uncommented == (ref(7),)
    assert COMMENT_METHOD in report.summary()
    assert "section is missing" in capsys.readouterr().err


def test_the_comment_is_posted_once_the_client_can_post_one():
    client = CommentingClient(issues={7: issue_payload(7)})
    plan = plan_reconcile(
        ledger(errors=(ContractError(7, "Verify", "section is missing"),)),
        labels={ref(7): frozenset({READY})},
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
    client = PullAwareClient(issues={4: issue_payload(4)}, open_pulls=(task_branch(ref(4), 0),))
    snapshot = Snapshot(client)

    # The probe for a method the client has not grown yet must see the client's
    # own answer. A wrapper that answered for it would turn a method that is
    # merely missing into one that can never be found.
    assert snapshot.open_branches() == frozenset({task_branch(ref(4), 0)})
    with pytest.raises(AttributeError):
        getattr(snapshot, COMMENT_METHOD)

    # The listing is wrapped rather than delegated, so a cycle's collaborators
    # share one read - but the wrapper exists only because this client does
    # have the method, and it answers with the client's own data.
    assert [p["head"]["ref"] for p in getattr(snapshot, PULLS_METHOD)()] == [task_branch(ref(4), 0)]


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
        def sweep(
            self, ledger, *, containers=None, states=None, open_branches=None, believed=None
        ):
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


# --------------------------------------------------------------------------
# Step 5: judge, replan, and the goal gate
# --------------------------------------------------------------------------


def test_an_exhausted_ledger_is_not_the_end_of_the_run_if_the_gate_extends(monkeypatch):
    """The property this whole feature exists for.

    Every task is `swarm:done`, so the *plan* is finished - and until the goal
    gate existed that was also where the run stopped, which is a statement
    about the planner's first guess rather than about the objective. A gate
    that appends work must leave the loop running so the next cycle dispatches
    it.
    """
    calls: list = []

    def spy(client, ledger, objective, **kwargs):
        calls.append(objective)
        # Extends once, then reports the objective met, or this loop is infinite.
        if len(calls) == 1:
            return SimpleNamespace(
                done=False, rounds=1, summary=lambda: "planned 1 follow-up task(s)"
            )
        return SimpleNamespace(done=True, rounds=1, summary=lambda: "objective met")

    monkeypatch.setattr("swarm.orchestrator.goal.close_the_loop", spy)

    client = FakeClient(issues={4: issue_payload(4, label=DONE)})
    reports = reconciler(
        client, FakeFleet(), goal_gate=True, objective="make the thing work"
    ).loop(cycles=5)

    assert [report.exhausted for report in reports] == [True, True]
    assert [report.finished for report in reports] == [False, True]
    assert len(reports) == 2, "the loop stopped at plan exhaustion"
    assert calls == ["make the thing work"] * 2


def test_the_judge_is_handed_the_failure_text_the_cycle_just_read(tmp_path):
    """The seam a retype can break in silence, so it is pinned (#142).

    `_results()` and `judge.Observation.of` have to agree about what a task is
    keyed on. They agreed on `int` before and agree on `TaskRef` now, and if
    they ever stop, nothing raises: the lookup misses, every signal reads as
    "no evidence", and the judge calls a looping run healthy on the strength of
    a failure it could not see. Read off the observation the cycle stored,
    because that is the object handed to `judge` - and on a cycle that changed
    something, `judge` itself is deliberately not called.
    """
    client = FakeClient(issues={4: issue_payload(4, label=CLAIMED)})
    write_result(record(4, 1, reason="TypeError: cannot parse header"), tmp_path)

    loop = reconciler(client, FakeFleet(), artifacts=tmp_path)
    loop.cycle()

    observed = loop._previous
    assert observed is not None, "step 5 never observed the cycle"
    assert "TypeError: cannot parse header" in observed.signals["task-4"].evidence


def test_the_goal_gate_is_skipped_without_an_objective(monkeypatch, capsys):
    """Assessing against an empty objective asks a model whether nothing was
    delivered, and then plans follow-ups from whatever it answered."""
    monkeypatch.setattr(
        "swarm.orchestrator.goal.close_the_loop",
        lambda *a, **k: pytest.fail("the gate assessed an empty objective"),
    )

    client = FakeClient(issues={4: issue_payload(4, label=DONE)})
    reports = reconciler(client, FakeFleet(), goal_gate=True, objective="  ").loop(cycles=3)

    assert len(reports) == 1
    assert reports[0].finished is True
    assert "no objective" in capsys.readouterr().err


def test_a_dry_run_neither_judges_nor_extends(monkeypatch):
    """Both halves of step 5 write - one rewrites the tracker, one adds to it -
    and a command that promised to change nothing must do neither."""
    monkeypatch.setattr(
        "swarm.orchestrator.goal.close_the_loop",
        lambda *a, **k: pytest.fail("a dry run reached the goal gate"),
    )

    client = FakeClient(issues={4: issue_payload(4)})
    report = reconciler(
        client, FakeFleet(), goal_gate=True, objective="x", dry_run=True, oracle=Never()
    ).cycle()

    assert report.verdict is None
    assert report.goal is None


def test_a_stall_reaches_the_replanner_with_the_runs_own_verify_command(monkeypatch):
    """A replanned issue must carry the gate the original carried. Defaulting it
    inside the replanner re-points every task in a generated repository at
    `SETTINGS.verify_command`, which that repository has no way to run."""
    seen: dict = {}

    def spy(client, ledger, objective, verdict, **kwargs):
        seen.update(kwargs, objective=objective, verdict=verdict)
        return SimpleNamespace(replanned=False, replans=0, summary=lambda: "refused")

    monkeypatch.setattr("swarm.orchestrator.replan.replan", spy)

    stalled = Judged(
        ProgressJudgement(
            request_satisfied=False,
            progress_being_made=False,
            in_loop=True,
            reason="failed again identically",
        )
    )
    client = FakeClient(issues={4: issue_payload(4)})
    report = reconciler(
        client, None, objective="make the thing work", verify="pytest -q", oracle=stalled
    ).cycle()

    assert report.verdict is not None and report.verdict.stalled
    assert report.replanned is not None
    assert seen["verify"] == "pytest -q"
    assert seen["objective"] == "make the thing work"


def test_the_merge_policy_reaches_the_check_gate(monkeypatch):
    """`APIARY_MERGE_ADMIN_OVERRIDE=0` decides whether a human presses merge.
    A cycle that does not pass the policy down leaves that setting inert."""
    from swarm.orchestrator.checks import MergePolicy

    seen: dict = {}
    real = __import__("swarm.orchestrator.checks", fromlist=["plan_checks"]).plan_checks

    def spy(ledger, **kwargs):
        seen.update(kwargs)
        return real(ledger, **kwargs)

    monkeypatch.setattr("swarm.orchestrator.checks.plan_checks", spy)

    policy = MergePolicy(admin_override=False, merge_method="rebase")
    client = PullAwareClient(issues={4: issue_payload(4, label=REVIEW)})
    reconciler(client, FakeFleet(), merge_policy=policy).cycle()

    assert seen["policy"] is policy


def test_a_merge_gate_join_that_cannot_resolve_is_recorded_not_escaped(monkeypatch):
    """`UnresolvedJoin` reaches `cycle_error`, and the cycle still reports.

    The gate raising is asserted in `test_checks.py` and `test_mergeability.py`.
    What those cannot see is what the *cycle* does with it, and the answer is
    the one `DependencyCycleError` already had: record it, do not escape.

    The reason is specific to where this gate sits. It runs after `apply_plan`
    has written this cycle's labels and after the recovery sweep, so an
    exception leaving `cycle` is thrown before `CycleReport` is built -
    `on_cycle` never fires and the run directory never learns that those writes
    happened. #174 exists because a silent wrong answer is worse than a loud
    failure; a loud failure that erases its own evidence is not the trade it
    was asking for."""
    from swarm.orchestrator.checks import UnresolvedJoin

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise UnresolvedJoin("no check set for #4")

    monkeypatch.setattr("swarm.orchestrator.checks.plan_checks", refuse)

    seen: list[Any] = []
    client = PullAwareClient(issues={4: issue_payload(4, label=REVIEW)})
    reports = reconciler(client, FakeFleet(), on_cycle=seen.append).loop(cycles=1)

    # The fault is on the report a human and the run directory both read.
    report = reports[0]
    assert "no check set for #4" in report.cycle_error
    assert "no check set for #4" in report.summary()
    # And the gate decided nothing: no merge was issued, no admitted plan.
    assert report.checks is None
    assert report.mergeability is None
    # The half that an escaping exception destroyed: `loop` builds the report
    # and hands it to `on_cycle`, which is what writes `cycle.reconciled` into
    # the run directory (`cli._report_cycle`). Raising past `cycle` skipped
    # this entirely, so the cycle's already-written labels went unrecorded.
    assert seen == [report]


def test_a_failed_merge_gate_dispatches_nothing_that_cycle(monkeypatch):
    """A recorded fault must not read as a quiet cycle.

    Both faults that reach `cycle_error` say the same thing - the machinery
    deciding what may land is not answering - and a run that keeps spawning
    workers onto a review queue that cannot drain is precisely the "looks
    healthy while stuck" failure `mergeability.py`'s docstring is about. So the
    cycle reports and dispatches nothing, which is what `DependencyCycleError`
    already did for the readiness half."""
    from swarm.orchestrator.checks import UnresolvedJoin

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise UnresolvedJoin("no check set for #4")

    monkeypatch.setattr("swarm.orchestrator.checks.plan_checks", refuse)

    fleet = FakeFleet()
    client = PullAwareClient(
        issues={4: issue_payload(4, label=REVIEW), 5: issue_payload(5, label=READY)}
    )
    report = reconciler(client, fleet).cycle()

    assert report.cycle_error
    assert report.readiness is None
    assert report.dispatched is None
    assert fleet.spawned == []


def test_the_unresolved_join_is_not_a_lookup_error(monkeypatch):
    """And it must not be catchable as one.

    `UnresolvedJoin` reports a failed lookup, so `LookupError` is the base a
    reader reaches for - which is the trap: `KeyError` is a `LookupError`, so
    one `except LookupError` around a dict access anywhere upstream would
    swallow this and silently restore the default it exists to remove. The base
    class is therefore part of the fix, not an implementation detail, and this
    is what says so."""
    from swarm.orchestrator.checks import UnresolvedJoin

    assert not issubclass(UnresolvedJoin, LookupError)
    assert issubclass(UnresolvedJoin, RuntimeError)


# --------------------------------------------------------------------------
# A task that only ever fails as infrastructure (#91)
# --------------------------------------------------------------------------
#
# Exit 2 not consuming an attempt is right and unchanged (§4): a broken host
# must not burn every issue's retry budget before a human notices. This is the
# ceiling on it. Without one, a purely mechanical fault retries for free
# forever - and #90 widened what counts as mechanical.
#
# The only backstop before this was round-based stall detection, which routes
# to the *replanner*: a model, handed a broken socket as though it were a
# planning problem.


def infra(number: int = 4, *, reason: str = "docker: no such image") -> dict:
    return {ref(number): record(number, 2, attempt=1, reason=reason)}


def test_an_infrastructure_failure_below_the_cap_still_re_readies():
    """The rule this ticket adds a ceiling to, asserted at the boundary rather
    than only at zero."""
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=1)),
        results=infra(),
        max_attempts=3,
        infrastructure={ref(4): 1},
        infrastructure_policy=InfrastructurePolicy(cap=3),
    )

    transition = plan.transitions[0]
    assert (transition.to_state, transition.attempt) == (ELIGIBLE, None)
    assert transition.infrastructure


def test_the_nth_consecutive_infrastructure_failure_reaches_a_human():
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=1)),
        results=infra(reason="docker: no such image apiary-worker-node"),
        max_attempts=3,
        infrastructure={ref(4): 2},
        infrastructure_policy=InfrastructurePolicy(cap=3),
    )

    transition = plan.transitions[0]
    assert transition.to_state == NEEDS_HUMAN
    # The reason names the repeated verdict, not just a count: "3 failures" on
    # its own sends a human to the wrong place.
    assert "3 consecutive infrastructure failures" in transition.reason
    assert "no such image apiary-worker-node" in transition.reason
    assert transition.comment


def test_escalating_does_not_backdate_the_attempt_counter():
    """The attempts were never consumed, and writing one now would rewrite
    history to make this look like an exhausted budget rather than a machine
    fault - which is the diagnosis the whole exit-2 rule exists to protect."""
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=1)),
        results=infra(),
        max_attempts=3,
        infrastructure={ref(4): 2},
        infrastructure_policy=InfrastructurePolicy(cap=3),
    )

    assert plan.transitions[0].attempt is None


def test_a_real_task_failure_resets_the_streak():
    """The machine recovered and the run is back to arguing about code. Carrying
    the old streak over would escalate a task on the strength of a fault that is
    no longer happening."""
    before = {ref(4): 2}

    after = infrastructure_streaks(
        before,
        [Transition(ref=ref(4), from_state=CLAIMED_STATE, to_state=ELIGIBLE, reason="worker exit 1")],
    )

    assert ref(4) not in after


def test_consecutive_infrastructure_transitions_accumulate():
    streaks: dict[TaskRef, int] = {}
    for _ in range(3):
        streaks = infrastructure_streaks(
            streaks,
            [
                Transition(
                    ref=ref(4),
                    from_state=CLAIMED_STATE,
                    to_state=ELIGIBLE,
                    reason="infrastructure failure",
                    infrastructure=True,
                )
            ],
        )

    assert streaks == {ref(4): 3}


def test_one_issues_streak_is_not_another_issues():
    streaks = infrastructure_streaks(
        {ref(4): 2},
        [
            Transition(
                ref=ref(5),
                from_state=CLAIMED_STATE,
                to_state=ELIGIBLE,
                reason="infrastructure failure",
                infrastructure=True,
            )
        ],
    )

    assert streaks == {ref(4): 2, ref(5): 1}


def test_a_cycle_that_moved_nothing_counts_nothing():
    """The reason transitions are counted rather than result records.

    A result file is one per *attempt*, and an infrastructure verdict does not
    bump the attempt - so two mechanical failures in a row write the same
    filename. Re-reading an unchanged results directory must not look like a
    fourth failure.
    """
    assert infrastructure_streaks({4: 2}, []) == {4: 2}


def test_a_record_that_moved_nothing_is_not_retired():
    """`observed_records` folds from what landed, not from what was planned -
    `fold`'s rule and `infrastructure_streaks`' rule, for the same reason. A
    label write GitHub refused leaves the task where it was, and retiring its
    record would lose the only evidence the retry never happened."""
    moved_nothing = Transition(ref(4), CLAIMED_STATE, ELIGIBLE, "no record caused this")

    assert observed_records({}, []) == {}
    assert observed_records({ref(4): "kept"}, [moved_nothing]) == {ref(4): "kept"}


def test_two_verdicts_from_one_host_are_two_identities():
    """What actually keeps a genuine second infrastructure failure countable.

    Two mechanical failures in a row say the same thing for the same reason at
    the same attempt number, so the *only* field that separates them is when
    they finished - and `from_worker` and `synthesise` stamp that with
    `datetime.now` on every record real code writes. Asserted here rather than
    left to the loop test below, because it is the whole basis of the guard
    being safe to key on content.
    """
    moment = dt.datetime(2026, 8, 14, 14, 10, tzinfo=dt.timezone.utc)
    host_died = ResultRecord(
        run_id=RUN_ID, issue=4, attempt=0, exit_code=2, reason="docker: no such image"
    )
    first = replace(host_died, finished_at=moment)
    again = replace(host_died, finished_at=moment)
    later = replace(host_died, finished_at=moment + dt.timedelta(minutes=10))

    # One file read twice is one verdict, and that is the whole point.
    assert first.identity == again.identity
    # Two failures of the same host, indistinguishable in every other field.
    assert first.identity != later.identity


def test_the_cap_is_configurable_and_loud_on_garbage(monkeypatch):
    monkeypatch.delenv(INFRASTRUCTURE_CAP_ENV, raising=False)
    assert InfrastructurePolicy.from_env().cap == DEFAULT_INFRASTRUCTURE_CAP

    monkeypatch.setenv(INFRASTRUCTURE_CAP_ENV, "5")
    assert InfrastructurePolicy.from_env().cap == 5

    # A mistyped cap that silently fell back would leave a run looping on a
    # machine fault while somebody believed they had bounded it.
    monkeypatch.setenv(INFRASTRUCTURE_CAP_ENV, "three")
    with pytest.raises(ValueError):
        InfrastructurePolicy.from_env()


def test_a_cap_of_zero_or_less_is_honoured_as_written():
    """"Never escalate" is a legitimate thing to ask for while debugging a
    host, and the summary says out loud what it costs."""
    policy = InfrastructurePolicy(cap=0)

    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=1)),
        results=infra(),
        max_attempts=3,
        infrastructure={ref(4): 99},
        infrastructure_policy=policy,
    )

    assert plan.transitions[0].to_state == NEEDS_HUMAN  # cap 0 means the first one escalates
    assert "no ceiling" in InfrastructurePolicy(cap=-1).summary()


def test_the_escalation_needs_no_model_no_daemon_and_no_github():
    """"Pure arithmetic in the orchestrator, so invariant 2 holds. Keep it that
    way." - the ticket's own note, as an assertion."""
    referenced = set(infrastructure_streaks.__code__.co_names)

    assert not referenced & {"structured", "orchestrator_llm", "GitHubClient", "DockerCLI"}


def test_a_gate_that_extended_flushes_the_cache_so_the_next_read_sees_its_issues(monkeypatch):
    """GitHub's conditional cache lags its writes. The gate planned #15-#17,
    the next cycle's ledger read was answered 304 from the pre-write body, the
    ledger still looked exhausted - and the gate ran AGAIN: a second
    seven-minute assessment and a plan of near-duplicate follow-ups instead of
    a dispatch. Observed live. The fix is the one `invalidate_cache`'s own
    docstring prescribes for the planner, applied to the other writer."""
    from types import SimpleNamespace

    calls: list = []

    def spy(client, ledger, objective, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(done=False, extended=True, rounds=1,
                                   summary=lambda: "planned follow-ups")
        return SimpleNamespace(done=True, extended=False, rounds=1,
                               summary=lambda: "objective met")

    monkeypatch.setattr("swarm.orchestrator.goal.close_the_loop", spy)

    flushed: list[int] = []
    client = FakeClient(issues={4: issue_payload(4, label=DONE)})
    client.invalidate_cache = lambda: flushed.append(1)

    reconciler(client, FakeFleet(), goal_gate=True, objective="make it work").loop(cycles=5)

    assert flushed == [1], "flushed exactly once: after the extension, not after met"


def test_a_gate_that_revived_keeps_the_loop_running_and_flushes_the_cache(monkeypatch):
    """The revival's two loop-level obligations, asserted where they live.

    A gate revival is actionable progress, not an exhausted run: `GoalReport.
    done` is False, so `CycleReport.finished` is False and the loop carries on
    to dispatch the revived issue. And the revival is a label write through
    this client, so the conditional cache must be flushed for exactly the
    extension's reason - a 304 from the pre-revival body would read the issue
    as failed again and post a duplicate revival comment onto it."""
    from types import SimpleNamespace

    calls: list = []

    def spy(client, ledger, objective, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(
                done=False,
                extended=False,
                revived=(SimpleNamespace(number=4),),
                rounds=0,
                summary=lambda: "revived 1 failed task(s) with budget remaining: #4",
            )
        return SimpleNamespace(
            done=True, extended=False, revived=(), rounds=0, summary=lambda: "objective met"
        )

    monkeypatch.setattr("swarm.orchestrator.goal.close_the_loop", spy)

    flushed: list[int] = []
    client = FakeClient(issues={4: issue_payload(4, label=DONE)})
    client.invalidate_cache = lambda: flushed.append(1)

    reports = reconciler(client, FakeFleet(), goal_gate=True, objective="make it work").loop(
        cycles=5
    )

    assert [report.finished for report in reports] == [False, True]
    assert len(reports) == 2, "the loop stopped on the cycle that revived"
    assert flushed == [1], "flushed exactly once: after the revival, not after met"




# --------------------------------------------------------------------------
# The per-task lifecycle (#141)
# --------------------------------------------------------------------------
#
# A cycle is minutes long and `cycle.reconciled` is one sentence about it, so
# everything an operator came to see - which task became eligible, which
# container took it, what its gate said, when its pull request merged - used to
# live and die inside a cycle. `orchestrator/lifecycle.py` announces it, and it
# is driven from here because the loop is what produces the reports it projects.
#
# Three properties carry this section. **The announcement decides nothing**: it
# runs on a finished `CycleReport`, after the writes and after the judge, and
# the reconciler's plan is identical with and without it. **It never speaks the
# tracker's vocabulary**: every payload is keyed by the task ref, and no issue
# number or `swarm:*` label survives into one - `events.jsonl` is append-only,
# so a payload written in label terms would be invalidated the day epic #140
# removes the labels. And **derived facts and applied writes are sourced
# differently**: eligibility is recomputed every cycle (ADR 0001) so it is
# projected from the verdict and bounded by `once`, while a terminal label is
# projected only from a transition that actually landed.

MERGE_COMMIT = "5e1f00d" + "0" * 33


@dataclass
class LifecycleClient(FakeClient):
    """A client far enough along to take one task all the way to merged."""

    open_pulls: tuple[tuple[int, str], ...] = ()
    check_runs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    heads: dict[int, str] = field(default_factory=dict)
    merges: list[int] = field(default_factory=list)

    def head_of(self, number: int) -> str:
        return self.heads.get(number, f"{number:0>40x}")

    def list_pull_requests(self, *, state: str = "open") -> list[dict[str, Any]]:
        self.log.append(f"list_pull_requests {state}")
        return [
            {
                "number": number,
                "head": {"ref": ref, "sha": self.head_of(number)},
                "base": {"ref": "main", "sha": BASE_COMMIT},
                "updated_at": "2026-08-14T13:25:30+00:00",
            }
            for number, ref in self.open_pulls
        ]

    def get_pull_request(self, number: int) -> dict[str, Any]:
        payload = next(p for p in self.list_pull_requests() if p["number"] == number)
        return {**payload, "mergeable": True, "mergeable_state": "clean"}

    def list_check_runs(self, ref: str) -> list[dict[str, Any]]:
        self.log.append(f"list_check_runs {ref}")
        return list(self.check_runs.get(ref, ()))

    def merge_pull_request(self, number: int, **kwargs: Any) -> dict[str, Any]:
        self.merges.append(number)
        return {"merged": True, "sha": MERGE_COMMIT}


def green(name: str = "ci") -> list[dict[str, Any]]:
    return [{"name": name, "status": "completed", "conclusion": "success"}]


def pending(name: str = "ci") -> list[dict[str, Any]]:
    return [{"name": name, "status": "in_progress"}]


def failing(name: str = "ci") -> list[dict[str, Any]]:
    return [{"name": name, "status": "completed", "conclusion": "failure"}]


def recorder() -> tuple[list[tuple[str, dict[str, Any]]], Callable[..., None]]:
    """An `events` sink and the list it fills. The shape `RunArtifacts.event` has."""
    seen: list[tuple[str, dict[str, Any]]] = []

    def emit(name: str, **fields: Any) -> None:
        seen.append((name, fields))

    return seen, emit


def names(seen: Iterable[tuple[str, dict[str, Any]]]) -> list[str]:
    """The #141 lifecycle names a cycle announced, in order.

    `state.*` is filtered out because #146's shadow window emits through the
    same `events` seam and is a different concern with its own suite
    (`tests/test_shadow.py`): it announces once per cycle whether or not
    anything disagreed, so leaving it in would make every assertion here about
    a task's lifecycle also an assertion about how many cycles the fixture ran.
    Filtered by prefix rather than by name, so a lifecycle event added tomorrow
    is still visible to these tests and a shadow event added tomorrow is still
    not.
    """
    return [name for name, _ in lifecycle_only(seen)]


def lifecycle_only(
    seen: Iterable[tuple[str, dict[str, Any]]]
) -> list[tuple[str, dict[str, Any]]]:
    """`seen` without #146's `state.*` events. See `names`."""
    return [(name, fields) for name, fields in seen if not name.startswith("state.")]


#: Four digits, and deliberately not a number any payload field could coincide
#: with: the "no issue number survives" assertion below is only worth making
#: against a number that cannot also be an attempt, an exit code or a PR.
TASK_ISSUE = 4242
TASK_REF = f"task-{TASK_ISSUE}"
TASK_BRANCH = task_branch(ref(TASK_ISSUE), 0)
TASK_PULL = 900

#: The second task `alongside` adds, its branch and its pull request. Four
#: digits, for `TASK_ISSUE`'s stated reason, and far enough from it that a
#: reader of a payload can tell the two tasks apart at a glance.
OTHER_ISSUE = 4343
OTHER_BRANCH = task_branch(ref(OTHER_ISSUE), 0)
OTHER_PULL = 901


def a_lifecycle_run(
    label: str = READY, *, alongside: bool = False, artifacts: Path | None = None
) -> tuple[LifecycleClient, FakeFleet, Reconciler, list]:
    """One task, labelled the way the planner actually creates a dep-free one.

    `swarm:ready`, not `swarm:blocked` - `nodes/planner.py` writes the state
    label once, at creation, and picks `READY` whenever the blockers are
    already met. So the root task of every plan, and "a run with one task"
    verbatim from #141, never produces a readiness *transition* at all. A
    fixture that seeded `swarm:blocked` would test a shape the planner only
    emits for a task that has dependencies.

    `alongside` adds a **second** task, one stage further on: its worker
    published a green pull request and exited, so `OTHER_PULL` is open and its
    container is still there. Off by default, because the callers of this
    fixture - here and the fourteen in `tests/test_shadow.py` - assert on the
    announcements of a one-task run, and a second task would put its own
    lifecycle into every one of them.

    It is here for the cutover pair in `tests/test_authority.py`, which #202
    found could not fail on a `plan_reconcile` regression. Over a one-task run
    two of the four things `test_authority.outcome` compares are `[]` in every
    arm whatever the orchestrator decided: nothing is open to merge, and the
    cycle that could dispose a container runs before the one task is ever
    dispatched, so `apply_plan`'s disposal loop never has a handle to remove.
    This second task makes both live. The merge gate lands its pull request on
    the first cycle it sees it, and `Belief.landed` is a ratchet - so from the
    next cycle on the task is terminal with a container still against it, which
    is a disposal.

    `artifacts` is the directory the run reads result records from, and passing
    one is what completes the second task's story: a worker publishes its pull
    request and then writes its record, in that order and as the last thing it
    does (`worker/pr.py`, `worker/result.py`), so "published and exited" without
    a record on disk is half a worker. Threaded rather than made unconditional
    because a record needs somewhere to live and this helper has no `tmp_path`.

    That record is what #228 needed. Under the resolver it changes nothing here
    - the second task is `landed` by the second cycle and rule 2 disposes it
    before rule 3 is reached - but a *label* reader has no `landed`, so the
    record is the difference between `swarm:claimed` meaning "still holding it"
    and meaning "was claimed, has finished, observe the verdict", and between
    `swarm:review` meaning nothing and meaning "the container is only holding a
    clone". Both are §4 rows that a one-record-poorer world could not reach, so
    with it the cutover pair's `outcome()` equality can fail on those two arms
    rather than only on the two terminal ones. Measured, not argued: see
    `test_authority.a_run`.
    """
    client = LifecycleClient(issues={TASK_ISSUE: issue_payload(TASK_ISSUE, label=label)})
    fleet = FakeFleet()
    if alongside:
        client.issues[OTHER_ISSUE] = issue_payload(OTHER_ISSUE, label=REVIEW)
        client.open_pulls = ((OTHER_PULL, OTHER_BRANCH),)
        client.check_runs = {client.head_of(OTHER_PULL): green()}
        # The state is left empty, which `Handle.running` reads as false: this
        # is the exited container of a worker that has already published. A
        # `running` one would resolve the task `claimed` and the merge gate
        # would leave its pull request alone, which is not what an open pull
        # request means. The disposal loop still has a handle either way:
        # `Reconciler.cycle` passes `running=` the containers that *exist*,
        # which is the distinction `plan_dispatch` is written about.
        fleet.handles[OTHER_ISSUE] = Handle(
            id=f"{OTHER_ISSUE:0>64x}", run_id=RUN_ID, issue=OTHER_ISSUE
        )
        if artifacts is not None:
            # Exit 0 and attempt 0: the record a worker that published this
            # pull request would have left. `overwrite=False` so two runs
            # handed the same directory read one record rather than the second
            # landing at `attempt-1` and reading as a retry nobody made.
            write_result(
                record(OTHER_ISSUE, 0, attempt=0, reason="verified and committed"),
                artifacts,
                overwrite=False,
            )
    seen, emit = recorder()
    loop = reconciler(client, fleet, events=emit)
    if artifacts is not None:
        loop.artifacts = artifacts
    return client, fleet, loop, seen


def reaches_review(
    client: LifecycleClient, fleet: FakeFleet, checks: list, *, attempt: int = 0
) -> None:
    """The worker finished: it moved its own label (#17) and left an open PR.

    `attempt` names the branch the worker pushed, because #144 gives each
    attempt its own (`apiary/<ref>-attempt-<n>`). A caller driving a second
    attempt has to say so, exactly as a real second worker would.
    """
    client.issues[TASK_ISSUE]["labels"] = [{"name": REVIEW}]
    client.open_pulls = ((TASK_PULL, task_branch(ref(TASK_ISSUE), attempt)),)
    client.check_runs = {client.head_of(TASK_PULL): checks}
    fleet.handles.clear()


def test_one_task_announces_its_whole_lifecycle_in_order(tmp_path):
    """#141's first acceptance criterion, driven through the loop rather than
    asserted on a hand-built report."""
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path

    loop.cycle()
    reaches_review(client, fleet, green())
    write_result(record(TASK_ISSUE, 0, attempt=0, reason="verified and committed"), tmp_path)
    loop.cycle()

    assert names(seen) == [
        "task.eligible",
        "task.claimed",
        "task.result",
        "pr.opened",
        "pr.checks",
        "pr.merged",
        "task.landed",
    ]
    # "A reader builds a per-task timeline without joining on anything."
    assert {fields["task"] for _, fields in lifecycle_only(seen)} == {TASK_REF}
    assert client.merges == [TASK_PULL]


def test_a_task_created_ready_is_still_announced_eligible():
    """The bug a `swarm:blocked` fixture hides.

    `ReadinessPlan.transitions` is *verdicts that disagree with the label on the
    issue*, and the planner already labelled this one `swarm:ready` - so a
    projection of `transitions` announces nothing for the only task in a
    one-task run. Eligibility is derived and recomputed each cycle (ADR 0001),
    so it is projected from the verdict itself.
    """
    client, _, loop, seen = a_lifecycle_run(label=READY)

    loop.cycle()

    assert "task.eligible" in names(seen)


def test_a_task_that_was_blocked_is_announced_when_its_dependency_lands():
    """And the other direction still works: a task with a dependency is held at
    `swarm:blocked` until readiness moves it."""
    client, _, loop, seen = a_lifecycle_run(label=BLOCKED)
    client.issues[TASK_ISSUE]["body"] = body(TASK_REF, blocked_by=[7])
    client.issues[7] = issue_payload(7, label=DONE, state="closed", state_reason="completed")

    loop.cycle()

    eligible = [fields for name, fields in seen if name == "task.eligible"]
    assert eligible and eligible[0]["depends_on"] == ["task-7"]


def test_eligibility_is_announced_once_per_episode_not_once_per_cycle():
    """It is a standing fact - true every cycle until the task is claimed - so
    a projection with no memory would repeat it for the length of the queue."""
    client, _, loop, seen = a_lifecycle_run()
    # No fleet, so nothing is ever claimed and the task stays ready.
    loop.fleet = None

    loop.cycle()
    loop.cycle()
    loop.cycle()

    assert names(seen) == ["task.eligible"]


def test_an_infrastructure_retry_is_announced_eligible_again(tmp_path):
    """The episode, not the attempt. Exit 2 consumes no attempt (§4), so the
    re-dispatch is ready at the number already announced - a key on the attempt
    would have been silent for exactly the failure mode that repeats."""
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path

    loop.cycle()
    write_result(record(TASK_ISSUE, 2, attempt=0, reason="docker: no such image"), tmp_path)
    loop.cycle()

    assert names(seen) == [
        "task.eligible",
        "task.claimed",
        "task.result",
        "task.eligible",
        "task.claimed",
    ]
    # And the counter really did not move, which is what makes this the case a
    # key on the attempt could not see.
    assert [f["attempt"] for n, f in seen if n == "task.eligible"] == [0, 0]


def test_each_announcement_carries_what_its_reader_came_for(tmp_path):
    """The payloads, once, so the fields are pinned rather than implied by the
    sequence above."""
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path
    loop.cycle()
    reaches_review(client, fleet, green("build"))
    write_result(record(TASK_ISSUE, 0, attempt=0), tmp_path)
    loop.cycle()

    payloads = dict(seen)
    sha = client.head_of(TASK_PULL)

    # Every payload names the cycle it belongs to, so a reader never has to
    # bracket these between two `cycle.reconciled` lines to order them.
    assert all("cycle" in fields for fields in payloads.values())
    assert payloads["task.eligible"] == {
        "cycle": 0,
        "task": TASK_REF,
        "attempt": 0,
        # The root of a plan: nothing was ever blocking it.
        "depends_on": [],
    }
    assert payloads["task.claimed"]["container"] == f"{TASK_ISSUE:0>64x}"[:12]
    assert payloads["task.result"]["exit_code"] == 0
    assert payloads["task.result"]["outcome"]
    assert payloads["pr.opened"] == {
        "cycle": 1,
        "task": TASK_REF,
        "pull": TASK_PULL,
        "head_sha": sha,
    }
    assert payloads["pr.checks"] == {
        "cycle": 1,
        "task": TASK_REF,
        "pull": TASK_PULL,
        "head_sha": sha,
        "state": "passed",
        "check": "build",
    }
    assert payloads["pr.merged"] == {
        "cycle": 1,
        "task": TASK_REF,
        "pull": TASK_PULL,
        "merge_commit": MERGE_COMMIT,
    }
    assert payloads["task.landed"] == {"cycle": 1, "task": TASK_REF}


def test_no_announcement_carries_an_issue_number_or_a_label_name(tmp_path):
    """The reason #131 was retargeted into this ticket.

    `events.jsonl` is append-only and read back by `swarm show`, so a payload
    written in the vocabulary ADR 0001 removes would be invalidated by the rest
    of epic #140 - the recorded runs, the board reducer and the board together.
    Asserted over every value of every event of a task that *failed*, so the one
    field carrying free prose - `task.needs_human.reason` - is inspected too.
    """
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path
    loop.max_attempts = 1
    loop.cycle()
    reaches_review(client, fleet, failing())
    write_result(record(TASK_ISSUE, 1, attempt=0), tmp_path)
    loop.cycle()

    assert "task.needs_human" in names(seen)
    for name, fields in seen:
        for key, value in fields.items():
            assert value != TASK_ISSUE, f"{name}.{key} is the issue number"
            if isinstance(value, str):
                assert "swarm:" not in value, f"{name}.{key} names a label"
                assert TASK_BRANCH not in value, f"{name}.{key} names the branch"


def test_the_announcement_does_not_disturb_the_external_console_log(tmp_path):
    """`console_external` folds *every* event looking for four keys, so a task
    event that happened to carry one would print itself into the cycle log as
    though it were a cycle."""
    from swarm.console_external import _lines

    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path
    loop.cycle()
    fleet.handles.clear()
    write_result(record(TASK_ISSUE, 1, attempt=1), tmp_path)
    loop.cycle()

    reserved = {"summary", "gate", "failures", "goal"}
    assert seen
    assert all(reserved.isdisjoint(fields) for _, fields in seen)
    # And the same claim from the reader's side.
    log = tmp_path / "events.jsonl"
    log.write_text(
        "".join(f'{{"event": "{name}"}}\n' for name in names(seen)), encoding="utf-8"
    )
    assert _lines(tmp_path) == []


def test_the_announcement_reaches_events_jsonl_through_the_redactor(tmp_path, monkeypatch):
    """#141 asks for these to go through the *existing* `RunArtifacts.event`
    path, so they land in the run directory and are redacted like everything
    else - not through a second writer that would have to remember to."""
    import json

    from swarm.artifacts import RunArtifacts

    credential = "hunter2-not-a-recognisable-token-shape"
    monkeypatch.setenv("GITHUB_TOKEN", credential)
    run = Run.start(REPO, "prove the events land", run_id=RUN_ID)
    artifacts = RunArtifacts.open(run, root=tmp_path)
    client = LifecycleClient(issues={TASK_ISSUE: issue_payload(TASK_ISSUE, label=READY)})
    loop = reconciler(client, FakeFleet(), events=artifacts.event)
    loop.run = run

    loop.cycle()
    artifacts.event("task.needs_human", task="task-x", reason=f"token {credential} leaked")

    events = [json.loads(line) for line in artifacts.events_path.read_text().splitlines()]
    task_events = [e for e in events if e["event"].startswith(("task.", "pr."))]

    assert [e["event"] for e in task_events] == [
        "task.eligible",
        "task.claimed",
        "task.needs_human",
    ]
    # `run` and `ts` come from the writer, not from the projection.
    assert all(e["run"] == RUN_ID and e["ts"] for e in task_events)
    assert credential not in artifacts.events_path.read_text()


# --- every writer of a terminal label --------------------------------------
#
# ADR 0001 makes `needs-human` the one state reported outbound and the one a
# customer's tracker cannot infer, and four different modules write it. A
# substrate that hears it from only some of them is not one #147 can read from.


def a_report(
    *,
    applied: Iterable[Transition] = (),
    entries: Iterable[LedgerEntry] = (),
    recovered: Iterable[Transition] | None = None,
    mergeability: Iterable[Transition] | None = None,
) -> CycleReport:
    """One finished cycle, as far as the announcement is concerned."""
    return CycleReport(
        index=0,
        ledger=ledger(*entries),
        result=ReconcileReport(plan=ReconcilePlan(), applied=tuple(applied)),
        recovered=(
            None
            if recovered is None
            else SimpleNamespace(
                result=ReconcileReport(plan=ReconcilePlan(), applied=tuple(recovered))
            )
        ),
        mergeability=(
            None if mergeability is None else SimpleNamespace(applied=tuple(mergeability))
        ),
    )


def escalation(reason: str = "attempts exhausted", attempt: int | None = 3) -> Transition:
    return Transition(
        ref=ref(4),
        from_state=CLAIMED_STATE,
        to_state=NEEDS_HUMAN,
        reason=reason,
        task_id="task-4",
        attempt=attempt,
    )


@pytest.mark.parametrize(
    "where",
    ["applied", "recovered", "mergeability"],
    ids=["the reconciler", "the recovery sweep", "the merge-ability gate"],
)
def test_every_writer_of_a_terminal_label_is_announced(where):
    """The reconciler, the recovery sweep, mergeability and the check gate all
    write `swarm:failed`; the check gate has its own test above."""
    events = lifecycle_events(a_report(**{where: [escalation()]}))

    assert [event.name for event in events] == ["task.needs_human"]
    assert events[0].fields["task"] == "task-4"


def test_a_failing_task_says_why_it_needs_a_human_and_that_it_paid_for_it():
    """Exit 1 at the cap: the attempt was consumed, and the sentence says which
    budget ran out."""
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=2)),
        results={ref(4): record(4, 1, attempt=2)},
        max_attempts=3,
    )
    events = lifecycle_events(a_report(applied=plan.transitions, entries=[entry(4, label=FAILED)]))

    assert [event.name for event in events] == ["task.needs_human"]
    assert events[0].fields["task"] == "task-4"
    assert events[0].fields["attempt_consumed"] is True
    assert "attempt(s) made" in events[0].fields["reason"]


def test_an_infrastructure_escalation_says_no_attempt_was_ever_consumed():
    """§4's rule, announced rather than re-derived: exit 2 never consumes an
    attempt, so the escalation at the cap consumed none either - which is the
    difference between "this task is hard" and "this host is broken"."""
    plan = plan_reconcile(
        ledger(entry(4, label=CLAIMED, attempt=1)),
        results=infra(4),
        infrastructure={ref(4): DEFAULT_INFRASTRUCTURE_CAP - 1},
    )
    events = lifecycle_events(a_report(applied=plan.transitions))

    assert events[0].name == "task.needs_human"
    assert events[0].fields["attempt_consumed"] is False
    # No counter on the payload: `Transition.attempt` is the value about to be
    # written and `task.result.attempt` is the one that just ran, so carrying
    # both would give a reader two events about one attempt disagreeing by one.
    assert "attempt" not in events[0].fields
    assert "infrastructure" in events[0].fields["reason"]


def test_a_malformed_issue_reaching_a_human_is_deliberately_not_announced():
    """§1.4's escalation is the one `swarm:failed` with no task ref: the issue
    never parsed, so it is not in `Ledger.entries` at all. An event keyed on
    nothing is not a timeline entry, and inventing a key from the issue number
    is the thing this whole module refuses to do."""
    malformed = Transition(
        ref=ref(4), from_state=ELIGIBLE, to_state=NEEDS_HUMAN, reason="malformed contract: no ## Goal"
    )

    assert lifecycle_events(a_report(applied=[malformed])) == ()


def test_a_reason_that_quoted_a_branch_is_rewritten_into_the_task_ref():
    """The one place the label vocabulary leaks into prose: the merge gate's
    "no check run was ever created for <branch>" names a branch, and for this
    adapter a branch carries an issue number (#144 encodes `#4` as `%234`)."""
    failed = Transition(
        ref=ref(4),
        from_state=REVIEW_STATE,
        to_state=NEEDS_HUMAN,
        reason=(
            f"no check run was ever created for {task_branch(ref(4), 0)}; "
            f"move it back to {READY}"
        ),
        task_id="task-4",
    )

    events = lifecycle_events(a_report(applied=[failed], entries=[entry(4, label=REVIEW)]))

    assert events[0].fields["reason"] == (
        "no check run was ever created for task-4; move it back to eligible"
    )


def test_a_transition_github_refused_is_never_announced():
    """`fold`'s rule, and for the same reason: a label write GitHub refused left
    the task where it was, so announcing it would put a state in an append-only
    log that the control plane never reached."""
    planned = Transition(ref=ref(4), from_state=CLAIMED_STATE, to_state=LANDED, reason="x", task_id="task-4")

    # Planned, not applied.
    assert lifecycle_events(a_report(applied=[], entries=[entry(4, label=CLAIMED)])) == ()
    assert [e.name for e in lifecycle_events(a_report(applied=[planned]))] == ["task.landed"]


# --- announced once, but only while it is the same occurrence ---------------


def test_a_standing_fact_is_announced_once_rather_than_every_cycle(tmp_path):
    """The results directory still holds last cycle's record and the pull
    request is still open, so a projection with no memory would re-announce
    both every fifteen seconds for the length of a review."""
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path
    loop.cycle()
    # Pending, so nothing merges and the same facts survive into the next cycle.
    reaches_review(client, fleet, pending())
    write_result(record(TASK_ISSUE, 0, attempt=0), tmp_path)

    mark = len(seen)
    loop.cycle()
    first = lifecycle_only(seen[mark:])
    loop.cycle()

    assert names(first) == ["task.result", "pr.opened", "pr.checks"]
    assert lifecycle_only(seen[mark:]) == first


def test_a_check_name_is_announced_verbatim(tmp_path):
    """The one field here whose text apiary did not author.

    A check name is written by whoever wrote the target repository's workflow.
    It cannot carry an apiary issue number, and it is precisely the string a
    reader pastes into the CI UI - so it is not scrubbed, and a repository with
    a check named exactly like a worker branch sees that name and not a task ref.
    """
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path
    loop.cycle()
    reaches_review(client, fleet, pending(TASK_BRANCH))
    loop.cycle()

    checks = [f["check"] for name, f in seen if name == "pr.checks"]
    assert checks == [TASK_BRANCH]


def test_a_check_set_that_moved_is_announced_again(tmp_path):
    """Once per *state*, not once per pull request: "pending" turning into
    "failing" is the event somebody is waiting for."""
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path
    loop.cycle()
    reaches_review(client, fleet, pending())
    loop.cycle()
    client.check_runs = {client.head_of(TASK_PULL): failing()}
    loop.cycle()

    assert [fields["state"] for name, fields in seen if name == "pr.checks"] == [
        "pending",
        "failing",
    ]


def test_a_retry_pushing_a_new_head_gets_its_own_check_announcements(tmp_path):
    """One task's gate can report twice, and the second report must not be
    swallowed by the first. #144 changed how the second attempt gets there - it
    pushes `apiary/<ref>-attempt-1` and opens its own pull request rather than
    force-pushing the one attempt 0 opened - but the announcement key is what is
    under test, and a key without the head sha would announce the first
    attempt's gate and silently swallow every later one."""
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path
    loop.cycle()
    reaches_review(client, fleet, failing())
    loop.cycle()
    # Attempt two: the gate consumed an attempt and sent the issue back to
    # `swarm:ready`, and the next worker published from the attempt-1 branch.
    reaches_review(client, fleet, failing(), attempt=1)
    client.heads[TASK_PULL] = "b" * 40
    client.check_runs = {"b" * 40: failing()}
    loop.cycle()

    checks = [(f["head_sha"], f["state"]) for name, f in seen if name == "pr.checks"]
    assert checks == [(f"{TASK_PULL:0>40x}", "failing"), ("b" * 40, "failing")]


def test_a_second_infrastructure_failure_at_the_same_attempt_is_announced(tmp_path):
    """Exit 2 does not consume an attempt (§4), so the re-dispatch runs as the
    *same* attempt - and since #177 it writes a **new** record file beside the
    first rather than replacing it. A host that is broken three times over is
    exactly what an operator needs to see, and a key on the attempt alone
    reports it once because every failure of that attempt shares the number."""
    import datetime as dt

    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path

    def infra_record(minute: int) -> ResultRecord:
        moment = dt.datetime(2026, 8, 14, 14, minute, tzinfo=dt.timezone.utc)
        return ResultRecord(
            run_id=RUN_ID,
            issue=TASK_ISSUE,
            attempt=0,
            exit_code=2,
            reason="docker: no such image",
            started_at=moment,
            finished_at=moment,
        )

    write_result(infra_record(10), tmp_path)
    loop.cycle()
    write_result(infra_record(20), tmp_path)
    loop.cycle()

    assert names(seen).count("task.result") == 2


# --------------------------------------------------------------------------
# One exit 2 is one infrastructure verdict (#203)
# --------------------------------------------------------------------------
#
# The count, not the labels. A cycle that reads the previous attempt's record a
# second time produces the *same* `swarm:claimed -> swarm:ready`, on the same
# issue, at the same attempt number - so a fixture asserting on labels or on
# the plan's prose passes either way, which is how this survived a rewrite of
# everything around it.
#
# It takes three cycles to see. Cycle 1 dispatches, cycle 2 observes the exit 2
# and re-dispatches at the same attempt (§4 consumes none), and cycle 3 is the
# one that used to read the dead attempt's record again: a second
# `infrastructure=True` transition, a second increment, and the container cycle
# 2 had just spawned disposed out from under the retry.


#: The two moments this section's records finished at. Distinct, because that
#: is the one field two mechanical failures of the same host do not share.
_FIRST = dt.datetime(2026, 8, 14, 14, 10, tzinfo=dt.timezone.utc)
_LATER = dt.datetime(2026, 8, 14, 14, 20, tzinfo=dt.timezone.utc)


def dead_host() -> ResultRecord:
    """A worker's exit 2: the host failed, so the task never really ran.

    Timestamped, unlike the `record` helper at the top of this file, because
    the guard under test is keyed on the record's content and `from_worker`
    stamps `finished_at` on every record a real worker writes.
    """
    return ResultRecord(
        run_id=RUN_ID,
        issue=TASK_ISSUE,
        attempt=0,
        exit_code=2,
        reason="docker: no such image",
        started_at=_FIRST,
        finished_at=_FIRST,
    )


def alive(fleet: FakeFleet) -> None:
    """What `docker ps` says about a container the fleet just spawned.

    `FakeFleet.spawn` leaves `state` empty, exactly as the real `spawn` does -
    "a container that started and exited in the same breath is the ordinary
    case for a worker". The daemon fills it in on the *next* listing, and a
    retry that is genuinely in flight is listed `running`; without this the
    resolver reads every one of this harness's containers as finished and
    believes a claimed task eligible, which is a different bug's fixture.
    """
    fleet.handles = {
        issue: replace(handle, state=RUNNING_STATE) for issue, handle in fleet.handles.items()
    }


def test_one_exit_2_increments_the_infrastructure_streak_exactly_once(tmp_path):
    """One host failure, one verdict - across the cycle that re-dispatches it.

    `APIARY_MAX_INFRASTRUCTURE` is a ceiling on *consecutive mechanical
    failures*, and #197 made it authoritative for `needs-human`. Counting one
    failure twice halves the ceiling and escalates naming a streak the host
    never had.
    """
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path

    loop.cycle()
    write_result(dead_host(), tmp_path)
    loop.cycle()
    alive(fleet)
    loop.cycle()

    assert loop._infrastructure == {ref(TASK_ISSUE): 1}
    # And the second half of it: the retry cycle 2 spawned is still running.
    # The double count and the disposal are the same read of the same record.
    assert fleet.spawned == [TASK_ISSUE, TASK_ISSUE]
    assert fleet.disposed == [TASK_ISSUE]
    # §4 is untouched: a mechanical failure still costs no attempt, so the
    # retry was announced ready at the number the first attempt already had.
    assert [f["attempt"] for n, f in seen if n == "task.eligible"] == [0, 0]


def test_the_retrys_own_failure_is_a_second_verdict_and_counts(tmp_path):
    """The other direction, and the reason the guard is keyed on the record's
    content rather than on `(issue, attempt)`.

    A retry that fails mechanically too is a second verdict about the host, and
    it writes a record at the *same* attempt number - so a guard that retired
    an attempt would stop counting a host that is broken three times over,
    which is precisely the run the ceiling exists for.
    """
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path

    loop.cycle()
    write_result(dead_host(), tmp_path)
    loop.cycle()
    alive(fleet)
    loop.cycle()
    # The retry dies the same way and writes its own testimony: same attempt,
    # same reason, same everything a human would read - and a later
    # `finished_at`, which is what `from_worker` stamps and the only thing that
    # separates two failures of one host. `write_result` bumps the filename
    # rather than replacing the evidence of the attempt before it.
    write_result(replace(dead_host(), finished_at=_LATER), tmp_path)
    loop.cycle()

    assert loop._infrastructure == {ref(TASK_ISSUE): 2}


def test_eleven_mechanical_failures_do_not_silence_the_task(tmp_path):
    """The stall #218 is about, driven end to end. Eleven, because ten is fine.

    Exit 2 consumes no attempt, so every one of these records says `attempt: 0`
    and `write_result` bumps only the filename - `issue-N-attempt-0.json`,
    `-1`, ... `-10`. A sorted glob puts `-10` third, between `-1` and `-2`, so
    `RunSummary.latest` used to hand the reconciler the record from `-9` from
    the eleventh failure onward. #209's freshness guard had already stamped that
    record's identity, so `fresh` was `False` on that cycle and on every cycle
    after it: no transition, no disposal, no escalation, forever.

    **The failing direction is silence**, which is why the streak is asserted
    and why the loop runs past ten. A version with the bug reaches 10 and then
    stops counting while the host keeps dying; asserting anything about three
    records proves nothing at all.

    The cap is lifted out of the way rather than left at its default: this is a
    claim about the eleventh *observation*, and the default cap escalates at
    three, which is a different mechanism reaching a different right answer.
    Eleven mechanical failures on one task is what #203 existed to make
    visible, and a run reaches it - a human who moves an escalated issue back
    to `swarm:ready`, as the escalation comment tells them to, resets the
    streak and not the directory.
    """
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path
    loop.infrastructure_policy = InfrastructurePolicy(cap=99)

    def the_host_dies_again(minute: int) -> None:
        # A real worker stamps `finished_at` with `datetime.now`, so each of
        # these is its own verdict about the host - the property
        # `test_two_verdicts_from_one_host_are_two_identities` pins.
        record = replace(dead_host(), finished_at=_FIRST + dt.timedelta(minutes=minute))
        write_result(record, tmp_path)
        alive(fleet)
        loop.cycle()

    loop.cycle()  # dispatch, attempt 0
    for minute in range(11):
        the_host_dies_again(minute)

    # Eleven failures, eleven verdicts. Ten would pass with the bug in place.
    assert loop._infrastructure == {ref(TASK_ISSUE): 11}
    # And the run is still moving rather than sitting on a stale record: the
    # eleventh observation re-readied the task and disposed the container that
    # failed. Twelve announcements, because the first cycle announced the task
    # ready before dispatching it, and all at attempt 0 - §4 consumes none.
    assert [f["attempt"] for n, f in seen if n == "task.eligible"] == [0] * 12
    assert len(fleet.disposed) == 11
    # The whole reason the file order stopped being trustworthy, spelled out so
    # a reader does not have to believe the arithmetic above.
    names = sorted(p.name for p in tmp_path.glob("issue-*-attempt-*.json"))
    assert names[2] == f"issue-{TASK_ISSUE}-attempt-10.json"

    # The other half of the claim: the stall was permanent, not an off-by-one at
    # eleven. A stale pin never comes unstuck, because the identity it settled on
    # stays stamped - so a version with the bug counts 10 here too, whatever
    # happens to the host next.
    the_host_dies_again(11)
    the_host_dies_again(12)

    assert loop._infrastructure == {ref(TASK_ISSUE): 13}


# --- announcement only -----------------------------------------------------


def test_announcing_changes_nothing_the_reconciler_decides(tmp_path):
    """The whole claim of this ticket, asserted by running the cycle twice."""

    def run(events: Any) -> tuple[list[str], set[str], list[str]]:
        client = LifecycleClient(issues={TASK_ISSUE: issue_payload(TASK_ISSUE, label=READY)})
        fleet = FakeFleet()
        report = reconciler(client, fleet, events=events, artifacts=tmp_path).cycle()
        return (
            [str(t) for t in report.result.plan.transitions],
            client.labels_on(TASK_ISSUE),
            client.log,
        )

    quiet = run(None)
    loud = run(recorder()[1])

    assert quiet == loud


def test_a_run_that_merges_by_hand_still_records_that_a_task_reached_review():
    """`--no-merge` leaves the review queue to a human. The gate being off is
    not a reason for the run directory to stop recording `pr.opened` - but the
    check sets genuinely are not read, so `pr.checks` is silent."""
    client, fleet, loop, seen = a_lifecycle_run(label=REVIEW)
    client.open_pulls = ((TASK_PULL, TASK_BRANCH),)
    client.check_runs = {client.head_of(TASK_PULL): green()}
    loop.merge_gate = False

    loop.cycle()

    assert names(seen) == ["pr.opened"]
    assert client.merges == []


def test_a_dry_run_announces_the_derived_half_and_none_of_the_written_half():
    """A dry run promised to change nothing, and it does not: eligibility is a
    *derived* fact (ADR 0001 - "recomputed each cycle") and is true whether or
    not the label was written, while every event sourced from an applied
    transition is silent because nothing was applied. And it is bounded: two
    dry cycles compute the same verdict and say it once."""
    client, fleet, loop, seen = a_lifecycle_run()
    loop.dry_run = True

    loop.cycle()
    loop.cycle()

    assert names(seen) == ["task.eligible"]
    assert client.labels_on(TASK_ISSUE) == {READY}


# --------------------------------------------------------------------------
# A revival whose granted attempt leaves no artifact (#200)
# --------------------------------------------------------------------------
#
# The failure direction here is an **infinite loop**, not an error, which is
# why this is driven through `Reconciler.cycle` rather than asserted on a
# belief. `authority._budget_spent` used to lapse a revival grant on one thing
# only - the code host accounting for an attempt past the one it was granted at
# - and that needs a result record or an attempt-numbered branch on an *open*
# pull request. A granted attempt killed at `SWARM_WORKER_TIMEOUT`, or whose
# container was reaped mid-cycle, writes neither.
#
# Every other bound is inert on the same input, which is what made it a trap
# rather than a leak: `entry.attempt` moves only through `_retry_or_give_up`
# and the infrastructure streak only through `infrastructure_streaks(...,
# result.applied)`, and both need the artifact this failure is *defined by not
# producing*. So the task was re-dispatched every cycle, indefinitely, with
# nothing counting.


def a_revived_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any, list]:
    """One task apiary gave up on, which the goal gate then revives.

    `swarm:failed` at attempt 3 with no store judgment, which is ADR 0002's own
    fallback shape: the streak reads as the attempt counter, so the task is over
    the per-blocker cap and under the total one - exactly the task
    `goal._revive_abandoned` exists to return to a run.
    """
    from swarm.nodes.planner import IssueAction
    from swarm.orchestrator.recovery import Recovery

    client = LifecycleClient(
        issues={TASK_ISSUE: issue_payload(TASK_ISSUE, label=FAILED, attempt=3)}
    )
    fleet = FakeFleet()
    loop = reconciler(client, fleet, goal_gate=True, objective="make it work")
    loop.artifacts = tmp_path
    # The sweep is wired, because `cli.py` wires it unconditionally and it is
    # the thing that *does* move a counter on this input - `recovery._release`
    # consumes an attempt for a claim with nothing behind it. A harness without
    # it measures a configuration no real run has, and would overstate the bug.
    loop.recovery = Recovery(
        client=client, run=loop.run, store=loop.store, max_attempts=loop.max_attempts
    )
    calls: list[int] = []

    def gate(snapshot: Any, ledger_: Any, objective: str, **kwargs: Any) -> Any:
        calls.append(1)
        # What `planner.revive` does, and the whole of it: the state label goes
        # back to `swarm:ready` and *nothing else is reset* - not the attempt
        # counter, not the streak, not the blocker. Written through the client
        # rather than the `Snapshot` the gate is handed, which is the same
        # object one layer down.
        client.issues[TASK_ISSUE]["labels"] = [{"name": READY}]
        return SimpleNamespace(
            done=False,
            extended=False,
            rounds=0,
            revived=(IssueAction("revived", TASK_REF, TASK_ISSUE, reason="streak 3 of 3"),),
            summary=lambda: "revived 1 failed task(s) with budget remaining",
        )

    monkeypatch.setattr("swarm.orchestrator.goal.close_the_loop", gate)
    return client, fleet, loop, calls


def test_a_revived_attempt_that_leaves_no_result_lapses_instead_of_looping(
    tmp_path, monkeypatch
):
    """#200's acceptance criterion, and the loop is the assertion.

    Revive, dispatch, kill the container **without a result record**, then keep
    cycling. Before the fix the third cycle and every cycle after it dispatched
    the task again, because the grant could only lapse on evidence the killed
    worker never wrote.
    """
    client, fleet, loop, calls = a_revived_run(tmp_path, monkeypatch)

    # 1. The ledger is exhausted, so the gate runs and revives. Nothing is
    #    dispatched in the cycle that grants, because the revival happens in
    #    step 5 - after dispatch.
    loop.cycle()
    assert fleet.spawned == []

    # 2. The grant buys its one attempt: without it this cycle believes
    #    `needs-human` (the streak is 3 against a cap of 3) and dispatches
    #    nothing at all.
    loop.cycle()
    assert fleet.spawned == [TASK_ISSUE], "the revival bought exactly one attempt"

    # The worker is killed at `SWARM_WORKER_TIMEOUT`. The container is gone and
    # it wrote **nothing**: no result record under the run directory, no branch,
    # no pull request. Every counter in the system is exactly where it was.
    fleet.handles.clear()

    for _ in range(4):
        loop.cycle()
        # Every attempt dies the same way, so a loop keeps finding a fresh
        # container here rather than being held off by the last one.
        fleet.handles.clear()

    assert fleet.spawned == [TASK_ISSUE], (
        "the grant lapsed on the dispatch: one revival, one attempt, then the "
        "streak `planner.revive` never reset caps the task again"
    )
    assert calls == [1], "the gate ran once; a re-revival would be a second grant"
    # And it stopped in the state a human is asked about, rather than by being
    # quietly skipped: `needs-human` is what the dispatcher refuses to start.
    from swarm.orchestrator.derived import NEEDS_HUMAN

    assert loop._believed[ref(TASK_ISSUE)] == NEEDS_HUMAN


def test_a_revival_whose_spawn_never_ran_lapses_too(tmp_path, monkeypatch):
    """The arm a narrower reading of the signal leaves looping.

    `dispatcher.dispatch` records `claimed=False` for a spawn failure whose
    `release` provably undid the claim, and for a stack this host has no image
    for. So "the control plane is holding a claim" - which is what
    `_carry_forward` asks - is False, the issue goes back to `swarm:ready`, and
    a rule keyed on it would re-dispatch this task every cycle for the rest of
    the run: claim, release, claim, release, four label writes a cycle, with
    `DispatchReport.needs_judgement` False the whole time because a failed
    dispatch is not a stall.

    `_dispatch_attempted` counts it, because the grant buys one attempt and the
    dispatcher spending it on a host that could not run it is still spending it.
    The escalation is also the more useful answer: a human is told the task is
    stuck, which is true, instead of nothing at all.
    """
    client, fleet, loop, _ = a_revived_run(tmp_path, monkeypatch)

    def refuse(*args: Any, **kwargs: Any) -> Handle:
        fleet.log.append("spawn #%d" % kwargs["issue"])
        raise DockerError(["docker", "create"], 1, "no such image: apiary-worker:py")

    fleet.spawn = refuse  # type: ignore[method-assign]

    loop.cycle()
    for _ in range(4):
        loop.cycle()
        fleet.handles.clear()

    from swarm.orchestrator.derived import NEEDS_HUMAN

    assert fleet.spawned == [TASK_ISSUE], "one refused spawn spent the grant"
    assert loop._believed[ref(TASK_ISSUE)] == NEEDS_HUMAN


def test_a_revived_attempt_that_does_leave_a_result_is_unchanged(tmp_path, monkeypatch):
    """The other ending, pinned here because #200 must not have moved it.

    A revival that produces a result has always lapsed on the code host's own
    count: the record carries the granted attempt, `attempts_spent` moves past
    it, and the streak `planner.revive` never reset caps the task. Same fixture
    and the same one granted attempt as above - the only difference is that the
    worker wrote its record - so the two endings have to converge, and the
    arithmetic itself is pinned in `test_authority`.
    """
    client, fleet, loop, _ = a_revived_run(tmp_path, monkeypatch)

    loop.cycle()
    loop.cycle()
    assert fleet.spawned == [TASK_ISSUE]

    # The granted attempt fails the way a worker is supposed to fail: exit 1,
    # a result record carrying the attempt the revival granted.
    write_result(record(TASK_ISSUE, 1, attempt=3, reason="the verify command failed"), tmp_path)
    fleet.handles.clear()

    for _ in range(4):
        loop.cycle()

    from swarm.orchestrator.derived import NEEDS_HUMAN

    assert fleet.spawned == [TASK_ISSUE], "still exactly one attempt"
    assert loop._believed[ref(TASK_ISSUE)] == NEEDS_HUMAN


# --------------------------------------------------------------------------
# Storing a state as labels
# --------------------------------------------------------------------------
#
# `Transition` speaks ADR 0001's internal states since #152, so the `swarm:*`
# names appear at exactly one point in the transition path: `write_labels`.
# These are the properties that made it safe to move the vocabulary.


def test_a_transition_is_stored_as_the_label_that_holds_its_state():
    client = CommentingClient(issues={4: issue_payload(4, label=CLAIMED)})

    write_labels(client, Transition(ref(4), CLAIMED_STATE, LANDED, "merged"))

    assert client.labels_on(4) == {DONE}


def test_the_label_is_added_before_the_stale_one_is_removed():
    """`readiness._relabel`'s rule, held at the one place that now writes.

    GitHub has no transaction across two label calls. A crash between them leaves
    two state labels or none: two is repairable by §3's precedence, none puts the
    issue outside the ledger entirely, where nothing looks at it again.
    """
    client = CommentingClient(issues={4: issue_payload(4, label=CLAIMED)})

    write_labels(client, Transition(ref(4), CLAIMED_STATE, LANDED, "merged"))

    assert client.log == [f"+{DONE} #4", f"-{CLAIMED} #4"]


def test_a_transition_that_does_not_move_writes_no_removal():
    """Adding and removing the same name is a call that undoes itself."""
    client = CommentingClient(issues={4: issue_payload(4, label=CLAIMED)})

    write_labels(client, Transition(ref(4), CLAIMED_STATE, CLAIMED_STATE, "unchanged"))

    assert client.log == [f"+{CLAIMED} #4"]
    assert client.labels_on(4) == {CLAIMED}


def test_the_label_removed_is_the_one_the_issue_carries_not_the_one_believed():
    """Why `from_state` is built from the label and not from the belief (#152).

    A human relabels a claimed task `swarm:done` mid-run. The resolver still
    believes `claimed` - there is a container - and the cycle decides to move the
    task to `needs-human`. The write has to take **`swarm:done`** off, because
    that is what the issue is wearing; taking `swarm:claimed` off would leave the
    issue carrying two state labels, and §3 would then read the furthest-along of
    them and stop the task.

    `plan_reconcile` therefore passes `label_state(entry.state_label)` rather than
    the belief, and this asserts the consequence rather than the plumbing.
    """
    client = CommentingClient(issues={4: issue_payload(4, label=DONE)})

    # `label_state(swarm:done)` is `landed` - what the issue says, not what a
    # cycle watching the container would say.
    write_labels(client, Transition(ref(4), LANDED, NEEDS_HUMAN, "budget spent"))

    assert client.log == [f"+{FAILED} #4", f"-{DONE} #4"]
    assert client.labels_on(4) == {FAILED}


def test_a_transition_with_no_previous_state_removes_nothing():
    """The malformed-issue path: `from_state` is empty when nothing carried a state."""
    client = CommentingClient(issues={4: issue_payload(4, label=READY)})

    write_labels(client, Transition(ref(4), "", NEEDS_HUMAN, "malformed contract"))

    assert client.log == [f"+{FAILED} #4"]


def test_the_transition_path_writes_labels_in_exactly_one_place():
    """The property that makes the deletion in #152 a deletion rather than a hunt.

    Three modules used to carry their own copy of add-before-remove - here,
    `checks._apply` and `mergeability._apply` - so the label vocabulary had three
    exits from the transition path. Now it has one, and this is the test that
    fails if a fourth appears.

    Static, over the source, because a runtime probe only sees the path the test
    happened to take: the merge gate's copy fired on a green pull request and
    nothing else.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "swarm" / "orchestrator"
    writers = {}
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        if "add_labels" in calls:
            writers[path.name] = sorted(calls & {"add_labels", "remove_label"})

    # `dispatcher.py` writes the claim, which is not a `Transition` and is #152's
    # to delete separately; `reconcile.py` is `write_labels`. Nothing else.
    assert set(writers) == {"reconcile.py", "dispatcher.py"}, writers


# --------------------------------------------------------------------------
# `from_state` is the label the issue carries, at the sites that build it (#243)
# --------------------------------------------------------------------------
#
# `test_the_label_removed_is_the_one_the_issue_carries_not_the_one_believed`
# above pins the property *given* a `Transition`. These pin the other half: the
# transitions the reconciler actually builds carry the carried label in that
# field. Measured by mutation - replacing a construction site with a fixed
# state used to leave the suite green at four of eleven sites, which means four
# of the rules could write the wrong label with nothing noticing.
#
# **The worlds below all disagree with themselves on purpose.** With the
# carried label and the believed state equal, every one of these sites is
# indistinguishable from every wrong version of itself; the disagreement is
# what makes the assertion mean anything. A human relabelling a task mid-run is
# the case, and it is not hypothetical - it is the case `plan_reconcile`'s
# docstring cites for reading the label here in the first place.
#
# **And they all carry `swarm:done` rather than a state a rule here writes.**
# A world whose carried label happens to be the state a mutation substitutes is
# a world that cannot see that mutation: the first draft of these sweeps
# carried `swarm:blocked` in the merge gate's cases and reported three sites
# green against a `"blocked"` mutant that was, in those worlds, the original.
# `landed` is written by no rule under test, so no substitution collides with
# it by accident.


def believing(state: str, *, was: str | None = None, task: str = "task-4") -> Belief:
    """A cycle that believes `state` about one task, whatever its label says."""
    return Belief(states={task: state}, previous={task: was or state})


#: Every rule in `plan_reconcile` that builds a `Transition`, as a world where
#: the carried label and the believed state disagree. Keyed by the rule, and
#: the value is what that rule decides - so a case that stops reaching its rule
#: fails on `to_state` rather than passing vacuously on a transition list that
#: happens to be empty.
#: The labels each case is run with. Two, and never one: a world whose carried
#: label happens to equal the state a mutation substitutes cannot see that
#: mutation, so one world per site pins the site against every constant *except
#: its own*. Two worlds with different labels leave no such hole.
CARRIED = (DONE, BLOCKED)

#: The malformed-contract rule skips a terminal label on purpose - a task
#: nobody will run again is not failed a second time every cycle - so its two
#: are the non-terminal pair.
CARRIED_NON_TERMINAL = (CLAIMED, BLOCKED)

CARRIED_LABEL_RULES: tuple[
    tuple[str, Callable[[str], ReconcilePlan], str, tuple[str, ...]], ...
] = (
    (
        "a human closed the issue",
        lambda label: plan_reconcile(
            ledger(entry(4, label=label)),
            believed=believing(CLAIMED_STATE),
            states={ref(4): closed(4)},
        ),
        LANDED,
        CARRIED,
    ),
    (
        "the worker published its pull request",
        lambda label: plan_reconcile(
            ledger(entry(4, label=label)),
            believed=believing(CLAIMED_STATE),
            results={ref(4): record(4, 0)},
        ),
        REVIEW_STATE,
        CARRIED,
    ),
    (
        "the worker failed and the budget holds",
        lambda label: plan_reconcile(
            ledger(entry(4, label=label)),
            believed=believing(CLAIMED_STATE),
            results={ref(4): record(4, 1)},
            max_attempts=3,
        ),
        ELIGIBLE,
        CARRIED,
    ),
    (
        "the same failure hit the streak cap",
        lambda label: plan_reconcile(
            ledger(entry(4, label=label, attempt=2, streak=2, blocker=signature("boom"))),
            believed=believing(CLAIMED_STATE),
            results={ref(4): record(4, 1, attempt=2, verify_output="boom")},
            max_attempts=3,
        ),
        NEEDS_HUMAN,
        CARRIED,
    ),
    (
        "different failures spent the total cap",
        lambda label: plan_reconcile(
            ledger(entry(4, label=label, attempt=8)),
            believed=believing(REVIEW_STATE),
            open_branches=(),                    # the pull request is gone
            max_total_attempts=9,
        ),
        NEEDS_HUMAN,
        CARRIED,
    ),
    (
        "an infrastructure failure that costs nothing",
        lambda label: plan_reconcile(
            ledger(entry(4, label=label)),
            believed=believing(CLAIMED_STATE),
            results={ref(4): record(4, 2, reason="the network was unreachable")},
            infrastructure_policy=InfrastructurePolicy(cap=3),
        ),
        ELIGIBLE,
        CARRIED,
    ),
    (
        "infrastructure failures that hit their cap",
        lambda label: plan_reconcile(
            ledger(entry(4, label=label)),
            believed=believing(CLAIMED_STATE),
            results={ref(4): record(4, 2, reason="the network was unreachable")},
            infrastructure={ref(4): 2},
            infrastructure_policy=InfrastructurePolicy(cap=3),
        ),
        NEEDS_HUMAN,
        CARRIED,
    ),
    (
        # The one case that cannot carry `swarm:done`: this rule skips a
        # terminal label on purpose, so a task nobody is going to run again is
        # not failed a second time every cycle. `swarm:claimed` is the
        # hand-typed label here, and the belief disagrees just as loudly.
        "a malformed contract",
        lambda label: plan_reconcile(
            ledger(errors=(ContractError(4, "Verify", "section is missing"),)),
            believed=believing(ELIGIBLE),
            labels={ref(4): frozenset({label})},
        ),
        NEEDS_HUMAN,
        CARRIED_NON_TERMINAL,
    ),
)


@pytest.mark.parametrize(
    "rule, world, decides, carried",
    CARRIED_LABEL_RULES,
    ids=[case[0] for case in CARRIED_LABEL_RULES],
)
def test_every_rule_removes_the_label_the_issue_carries(rule, world, decides, carried):
    """One case per rule that builds a `Transition`, each in a world that
    disagrees with itself: the issue wears `swarm:done` because a human typed
    it there, and the cycle believes something else because the world says so.

    `from_state` names the label the write has to **remove**, so every one of
    these must be the internal state of the label the issue is wearing. Taking
    the
    believed label off instead leaves the issue with two state labels, and §3's
    precedence then reads the furthest-along of them and stops the task; it
    also feeds `fold`, which rebuilds the entry's label set from this field, so
    the cycle's own ledger would disagree with GitHub for long enough to
    dispatch a container against it. Both failures are silent.
    """
    for label in carried:
        transition = world(label).transitions[0]

        assert transition.to_state == decides, f"{rule}: the rule stopped firing"
        assert transition.from_state == internal_state(label), (
            f"{rule} carrying {label}: removed the believed label"
        )
