"""Tests for the progress ledger and for what a stall does to the tracker.

Four properties carry this file, and every one of them is a way a run fails
rather than a way it computes.

**The model is asked as rarely as possible, and the tests prove it by making a
model call an error.** Almost every judgement below runs against an oracle that
raises if it is invoked. On this host an orchestrator call is a ~6.7 s model
swap (`config.py`), the reconcile loop runs on a 15 s interval, and #21 went to
some trouble to make a dispatch-only cycle cost no model load; a judge that
asked on every cycle would hand all of that back. So "no model call" is an
assertion, not a comment.

**Rebasing is progress.** #34's whole job is keeping PRs mergeable against a
base that every merge invalidates, and a task can spend several cycles being
updated with nothing else about it moving. Reading that as a stall would replan
a run whose code was working, which is the expensive way to lose an afternoon.

**A task blocked on a file it may not edit is not a loop.** No decomposition of
the objective hands that task a fix, so a replan produces the same wall under a
new id. The judge distinguishes it and the replanner refuses; both halves are
asserted, because either one alone still parks the run in the wrong place.

**Nothing here touches a real tracker.** A replan closes issues and opens new
ones, so the write path is exercised exactly once - through #10's `write_plan`,
against an issue store behind #31's fake transport - and every other test drives
a `writer` spy that records the call it was never supposed to receive. There is
no code path in this file that can reach GitHub.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Sequence

import pytest

from fixtures.belief import fixture_belief
from fixtures.github import REPO, response
from fixtures.markers import legacy_marker
from swarm.github.client import GitHubError
from swarm.github.ledger import Ledger, LedgerEntry, load_ledger, render_marker
from swarm.github.refs import task_ref as ref
from swarm.nodes.judge import (
    Observation,
    Verdict,
    failure_signature,
    judge,
    judge_node,
    mentioned_paths,
)
from swarm.nodes.planner import render_body
from swarm.orchestrator.replan import (
    EXHAUSTED,
    NEEDS_HUMAN,
    NO_TASKS,
    PROGRESSING,
    SATISFIED,
    TOO_SOON,
    UNRESOLVED,
    brief,
    decide,
    replan,
)
from swarm.state import Plan, PlannedTask, ProgressJudgement, TaskRecord
from swarm.taskref import TaskRef
from swarm.worker.result import ResultRecord

# ADR 0001's internal states, which is what a fixture declares since #152 took
# the `swarm:*` control plane away. A state is derived per cycle and lives on
# the cycle's `Belief`; each `entry(label=...)` below stashes the declared state
# on `LedgerEntry.labels`, and `fixtures.belief.fixture_belief` turns the whole
# ledger's declarations into the `Belief` production now requires.
READY = "eligible"
BLOCKED = "blocked"
CLAIMED = "claimed"
REVIEW = "review"
DONE = "landed"
FAILED = "needs-human"

RUN_ID = "apiary-20260814-142530-k3f9qz"
VERIFY = "python -m pytest -q"
OBJECTIVE = "make the retry logic work"
BASE = f"/repos/{REPO}/issues"


# --------------------------------------------------------------------------
# Ledgers, results and oracles
# --------------------------------------------------------------------------


def entry(
    number: int,
    *,
    task_id: str = "",
    label: str = READY,
    attempt: int = 0,
    files: Sequence[str] = (),
    goal: str = "do the thing",
) -> LedgerEntry:
    return LedgerEntry(
        number=number,
        title=f"issue {number}",
        task_id=task_id or f"task-{number}",
        attempt=attempt,
        goal=goal,
        files=tuple(files) or (f"src/swarm/mod{number}.py",),
        verify=VERIFY,
        blocked_by=(),
        labels=frozenset({label}),
    )


def ledger(*entries: LedgerEntry) -> Ledger:
    return Ledger(entries={item.task_id: item for item in entries})


# The three seams #152 opened. A ledger no longer carries a state, so the state
# every one of these reads has to be handed in: `Observation.of` projects it
# through §3's table, and `replan`/`brief` ask `authority.state_of`, which
# raises rather than falling back to a label that is not written any more. Each
# wrapper supplies the belief the fixture's own declarations imply and changes
# nothing else, so a test that is not about the authority reads as it did.


def observe(book: Ledger, **kwargs: Any) -> Observation:
    """`Observation.of` over the states this ledger's fixtures declared."""
    kwargs.setdefault("states", fixture_belief(book).states)
    return Observation.of(book, **kwargs)


def _replan_(client: Any, book: Ledger, *args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("believed", fixture_belief(book))
    return replan(client, book, *args, **kwargs)


def _brief_(book: Ledger, *args: Any, **kwargs: Any) -> tuple[str, str]:
    kwargs.setdefault("believed", fixture_belief(book))
    return brief(book, *args, **kwargs)


def record(
    issue: int,
    *,
    attempt: int = 0,
    exit_code: int = 1,
    reason: str = "the verify command failed",
    output: str = "",
) -> ResultRecord:
    return ResultRecord(
        run_id=RUN_ID,
        issue=issue,
        attempt=attempt,
        exit_code=exit_code,
        reason=reason,
        verify_output=output,
    )


class ModelCalled(BaseException):
    """Raised by `Never`, and deliberately not an `Exception`.

    `judge` catches every `Exception` from an oracle and reports the cycle as
    unresolved, which is right for a restarting Ollama and useless for a test:
    a `Never` that raised one would be swallowed and the assertion it exists to
    make would silently never fire.
    """


class Never:
    """An oracle that fails the test if anything asks it a question.

    The deterministic ladder is the whole point of #24's "keep the
    short-circuits", and the only honest way to assert a model was not called
    is to make calling it an error.
    """

    def invoke(self, messages: Sequence[tuple[str, str]]) -> ProgressJudgement:
        raise ModelCalled(f"the judge asked a model: {messages!r}")


@dataclass
class Answers:
    """An oracle with a scripted answer, which records what it was shown."""

    judgement: ProgressJudgement
    asked: list[Any] = field(default_factory=list)

    def invoke(self, messages: Sequence[tuple[str, str]]) -> ProgressJudgement:
        self.asked.append(messages)
        return self.judgement


class Unreachable:
    """Ollama restarting, a socket refusing, a model that is not pulled."""

    def invoke(self, messages: Sequence[tuple[str, str]]) -> ProgressJudgement:
        raise RuntimeError("connection refused")


def verdict_of(
    current: Ledger,
    previous: Ledger | None = None,
    *,
    results: dict[TaskRef, ResultRecord] | None = None,
    prior_results: dict[TaskRef, ResultRecord] | None = None,
    churn: dict[TaskRef, int] | None = None,
    prior_churn: dict[TaskRef, int] | None = None,
    stalls: int = 0,
    oracle: Any = None,
) -> Verdict:
    """Judge one cycle against the one before it, with no I/O anywhere."""
    before = (
        None
        if previous is None
        else observe(previous, results=prior_results, churn=prior_churn)
    )
    return judge(
        observe(current, results=results, churn=churn),
        before,
        objective=OBJECTIVE,
        stalls=stalls,
        oracle=oracle if oracle is not None else Never(),
    )


# --------------------------------------------------------------------------
# The deterministic ladder
# --------------------------------------------------------------------------


def test_every_task_merged_is_satisfied_without_a_model():
    done = ledger(entry(1, label=DONE), entry(2, label=DONE))

    verdict = verdict_of(done)

    assert verdict.satisfied and not verdict.stalled
    assert verdict.judgement.progress_being_made and not verdict.judgement.in_loop
    assert not verdict.consulted


def test_a_task_that_needs_a_human_is_finished_but_never_satisfied():
    # `swarm:failed` is terminal, so the run is over - but §3 maps it to
    # `abandoned`, and a run that abandoned a task did not do what was asked.
    mixed = ledger(entry(1, label=DONE), entry(2, label=FAILED))

    verdict = verdict_of(mixed)

    assert verdict.observation is not None and verdict.observation.finished
    assert not verdict.satisfied and not verdict.consulted


def test_an_empty_ledger_is_a_stall_no_model_is_needed_for():
    verdict = verdict_of(ledger())

    assert verdict.stalled and verdict.stalls == 1
    assert not verdict.consulted


def test_a_label_that_moved_is_progress_without_a_model():
    before = ledger(entry(1, label=READY), entry(2, label=BLOCKED))
    after = ledger(entry(1, label=CLAIMED), entry(2, label=BLOCKED))

    verdict = verdict_of(after, before)

    assert not verdict.stalled and verdict.judgement.progress_being_made
    assert "task-1" in verdict.reason
    assert not verdict.consulted


def test_a_worker_that_is_still_running_is_progress_without_a_model():
    # Nothing changed between the two cycles, and nothing needs to have: a
    # worker takes minutes and a cycle takes seconds.
    working = ledger(entry(1, label=CLAIMED), entry(2, label=BLOCKED))

    verdict = verdict_of(working, working)

    assert not verdict.stalled and verdict.judgement.progress_being_made
    assert not verdict.consulted


def test_the_same_failure_twice_is_a_loop_without_a_model():
    # The two runs differ only in the digits pytest prints - a duration and a
    # line number - which is what `failure_signature` exists to see through.
    first = record(1, attempt=0, reason="AssertionError: expected 3, got 4", output="1 failed in 0.42s")
    second = record(1, attempt=1, reason="AssertionError: expected 3, got 4", output="1 failed in 0.51s")

    verdict = verdict_of(
        ledger(entry(1, attempt=2), entry(2, label=BLOCKED)),
        ledger(entry(1, attempt=1), entry(2, label=BLOCKED)),
        results={ref(1): second},
        prior_results={ref(1): first},
    )

    assert verdict.judgement.in_loop and verdict.stalled
    assert verdict.stalls == 1 and not verdict.consulted


def test_one_task_looping_while_another_lands_is_still_progress():
    # A repeat is always movement, so the loop rule has to ask whether it was
    # the *only* movement. Replanning here would discard a merged sibling.
    failure = dict(reason="AssertionError: expected 3, got 4")
    before = ledger(entry(1, attempt=1), entry(2, label=REVIEW))
    after = ledger(entry(1, attempt=2), entry(2, label=DONE))

    verdict = verdict_of(
        after,
        before,
        results={ref(1): record(1, attempt=1, **failure)},
        prior_results={ref(1): record(1, attempt=0, **failure)},
    )

    assert not verdict.judgement.in_loop and verdict.judgement.progress_being_made
    assert not verdict.stalled


def test_an_infrastructure_failure_is_not_the_tasks_failure():
    # Exit 2 does not consume an attempt (§4) and says nothing about the task,
    # so it must not become the signature two cycles are compared on.
    broken = record(1, exit_code=2, reason="ollama is unreachable")
    observation = observe(ledger(entry(1)), results={ref(1): broken})

    assert observation.signals["task-1"].failure == ""


# --------------------------------------------------------------------------
# #34: a pull request being rebased is not a stall
# --------------------------------------------------------------------------


def test_a_pull_request_being_rebased_is_progress_not_a_stall():
    # Nothing about the issue changes while #34 updates its branch: same label,
    # same attempt, same (absent) failure. Only the base-update count moves.
    review = ledger(entry(1, label=REVIEW), entry(2, label=DONE))

    verdict = verdict_of(review, review, churn={ref(1): 3}, prior_churn={ref(1): 2})

    assert not verdict.stalled and verdict.judgement.progress_being_made
    assert not verdict.consulted


def test_a_conflicted_pull_request_redispatched_is_not_read_as_a_loop():
    # #34's conflict path bumps the attempt and re-dispatches from a fresh base
    # with the same failure text still on file. The counter moved for a reason
    # that is not "it failed the same way again", and reading it as a loop would
    # replan a task that is being actively unstuck.
    conflict = dict(reason="merge conflict in src/swarm/mod1.py")
    verdict = verdict_of(
        ledger(entry(1, label=READY, attempt=2)),
        ledger(entry(1, label=REVIEW, attempt=1)),
        results={ref(1): record(1, attempt=1, **conflict)},
        prior_results={ref(1): record(1, attempt=0, **conflict)},
        churn={ref(1): 2},
        prior_churn={ref(1): 1},
    )

    assert not verdict.judgement.in_loop and not verdict.stalled


# --------------------------------------------------------------------------
# The one ambiguous case, and what the model is allowed to decide
# --------------------------------------------------------------------------


def test_a_cycle_that_learned_nothing_asks_the_model_once():
    # Everything ready, nothing dispatching, nothing changed. Arithmetic has no
    # answer to this one, which is what the model is for.
    idle = ledger(entry(1, label=READY), entry(2, label=READY))
    oracle = Answers(
        ProgressJudgement(
            request_satisfied=False,
            progress_being_made=False,
            in_loop=False,
            reason="nothing is being dispatched",
        )
    )

    verdict = verdict_of(idle, idle, oracle=oracle)

    assert verdict.consulted and len(oracle.asked) == 1
    assert verdict.stalled and verdict.stalls == 1
    assert OBJECTIVE in oracle.asked[0][1][1]


def test_the_model_cannot_declare_a_run_finished_while_issues_are_open():
    # §3: completion is the merge. A model that says otherwise is overruled by
    # the ledger, which is the only thing that knows what merged.
    open_work = ledger(entry(1, label=READY), entry(2, label=REVIEW))
    oracle = Answers(
        ProgressJudgement(
            request_satisfied=True, progress_being_made=True, in_loop=False, reason="looks done"
        )
    )

    verdict = judge(
        observe(open_work),
        observe(open_work),
        objective=OBJECTIVE,
        oracle=oracle,
    )

    assert not verdict.satisfied


def test_an_unreachable_model_is_not_a_stall():
    idle = ledger(entry(1, label=READY))

    verdict = verdict_of(idle, idle, stalls=1, oracle=Unreachable())

    assert verdict.unresolved and not verdict.stalled
    assert verdict.stalls == 1
    assert not verdict.should_replan()


def test_stalls_accumulate_until_a_replan_is_earned():
    empty = ledger()

    first = judge(observe(empty), stalls=0)
    second = judge(observe(empty), stalls=first.stalls)

    assert (first.stalls, second.stalls) == (1, 2)
    assert not first.should_replan(max_stalls=2)
    assert second.should_replan(max_stalls=2)


# --------------------------------------------------------------------------
# A wall the task was never allowed to touch
# --------------------------------------------------------------------------


def test_a_failure_outside_the_file_set_needs_a_human_not_a_replan():
    stuck = ledger(
        entry(1, attempt=2, files=("src/swarm/nodes/judge.py",)),
        entry(2, label=DONE),
    )
    outside = record(
        1,
        attempt=1,
        reason="ImportError: cannot import name 'Verdict'",
        output="src/swarm/orchestrator/reconcile.py:12: ImportError",
    )

    verdict = verdict_of(stuck, stuck, results={ref(1): outside}, prior_results={ref(1): outside})

    assert verdict.needs_human
    blocker = verdict.blockers[0]
    assert blocker.task_id == "task-1" and blocker.number == 1
    assert blocker.paths == ("src/swarm/orchestrator/reconcile.py",)
    # The refusal still holds - #1 is the only task not already finished, so
    # there is nothing a new decomposition could move - but it is `decide`'s to
    # make now rather than `should_replan`'s (#293). The stall budget *is* spent,
    # which is all that property answers.
    assert verdict.should_replan(max_stalls=1)
    assert decide(verdict) == NEEDS_HUMAN


def test_one_unhelpable_task_no_longer_vetoes_replanning_the_others():
    """The narrowing, as the case that used to be indistinguishable.

    #1 is failing on a file it may not edit, exactly as above, and no replan can
    help it. #3 is an ordinary stalled task, and a replan is precisely what it
    needs. The old rule read "some task is unhelpable" and refused the whole
    run, so #3 was never re-cut - and because a stalled run nearly always has
    one #1 in it, that made the replan close to unreachable.
    """
    stuck = ledger(
        entry(1, attempt=2, files=("src/swarm/nodes/judge.py",)),
        entry(3, attempt=2, files=("src/swarm/other.py",)),
    )
    outside = record(
        1,
        attempt=1,
        reason="ImportError: cannot import name 'Verdict'",
        output="src/swarm/orchestrator/reconcile.py:12: ImportError",
    )
    inside = record(
        3,
        attempt=1,
        reason="AssertionError",
        output="src/swarm/other.py:8: AssertionError",
    )

    verdict = verdict_of(
        stuck,
        stuck,
        results={ref(1): outside, ref(3): inside},
        prior_results={ref(1): outside, ref(3): inside},
    )

    assert verdict.needs_human
    assert [b.task_id for b in verdict.blockers] == ["task-1"]
    assert decide(verdict, max_stalls=1) == ""


def test_a_failure_inside_the_file_set_is_the_tasks_own_problem():
    own = ledger(entry(1, attempt=2, files=("src/swarm/nodes/judge.py",)))
    inside = record(
        1, attempt=1, reason="assert failed", output="src/swarm/nodes/judge.py:88: AssertionError"
    )

    assert observe(own, results={ref(1): inside}).blockers == ()


def test_one_attempt_is_not_yet_evidence_of_a_wall():
    # One failure is a data point; the second one having the same shape is the
    # evidence. Calling a human after the first would park every unlucky task.
    early = ledger(entry(1, attempt=1, files=("src/swarm/nodes/judge.py",)))
    outside = record(1, attempt=0, output="src/swarm/github/client.py:4: ImportError")

    assert observe(early, results={ref(1): outside}).blockers == ()


def test_only_paths_the_repository_could_own_are_evidence():
    text = (
        "/opt/homebrew/lib/python3.12/site-packages/pytest/__init__.py:1: in <module>\n"
        "  .venv/lib/site-packages/urllib3/util.py:9\n"
        "  src/swarm/github/client.py:120: GitHubError\n"
        "  see the README for details\n"
    )

    assert mentioned_paths(text) == ("src/swarm/github/client.py",)


def test_a_filename_with_two_extensions_survives_intact():
    """`src/calc.test.js`, not `src/calc.test`.

    The truncated form is a path that does not exist, so it compares equal to
    nothing in any `## Files` set - which made every double-extension file
    invisible to the "could the worker have fixed it" question, silently.
    `*.test.js` and `*.spec.tsx` are the dominant naming convention in exactly
    the stacks #87 adds. Surfaced by #93's agreement test.
    """
    assert mentioned_paths("❯ src/calc.test.ts:5:19") == ("src/calc.test.ts",)
    assert mentioned_paths("at (test/calc.test.js:5:10)") == ("test/calc.test.js",)
    assert mentioned_paths("FAIL app/routes/home.spec.tsx") == ("app/routes/home.spec.tsx",)


def test_a_leading_dot_slash_is_not_a_different_file():
    """Go build errors are spelled `./internal/calc/calc.go`, and a `## Files`
    set never is. `checks.failing_paths` has always stripped it."""
    assert mentioned_paths("./internal/calc/calc.go:12:2: undefined: helper") == (
        "internal/calc/calc.go",
    )


def test_a_workspace_absolute_path_is_the_repo_file_it_names():
    """Pinned: the two mangled extractions observed live, twice, on issue #40.

    The old lookbehind meant to refuse a match *continuing* an absolute path
    only shifted the match start by one character, so `/workspace/...` was
    extracted as `orkspace/...` and `/usr/local/...` as `sr/local/...` -
    paths that compared equal to nothing in any `## Files` set. Tracebacks
    always name workspace-absolute and stdlib paths, so the blocker veto fired
    on nearly every repeated pytest failure and suppressed replans that would
    have worked. A workspace-absolute path is the repo-relative file it names;
    an interpreter path is noise.
    """
    text = (
        'E   File "/workspace/issue-40/pyproject.toml", line 3\n'
        'E   File "/usr/local/lib/python3.12/importlib/__init__.py", line 90, in import_module'
    )

    paths = mentioned_paths(text)

    assert paths == ("pyproject.toml",)
    assert "orkspace/issue-40/pyproject.toml" not in paths
    assert "sr/local/lib/python3.12/importlib/__init__.py" not in paths


def test_interpreter_paths_alone_are_not_evidence_of_a_wall():
    # A traceback through the interpreter's own tree says nothing about
    # whether the task could reach its fix; after the workspace prefix is
    # stripped, a path still absolute is by construction not the repo's.
    stuck = ledger(entry(40, attempt=4, files=("pyproject.toml",)))
    noisy = record(
        40,
        attempt=3,
        output='File "/usr/local/lib/python3.12/importlib/__init__.py", line 90, in import_module',
    )

    assert observe(stuck, results={ref(40): noisy}).blockers == ()


def test_a_declared_file_failing_behind_the_workspace_prefix_does_not_veto():
    # `## Files` listed `pyproject.toml` and the failure named it as
    # `/workspace/issue-40/pyproject.toml` - the same file in the container's
    # coordinates. Observed live as "failed 4 time(s) on <files>, which its
    # ## Files does not list" on a file the issue did list.
    declared = ledger(entry(40, attempt=4, files=("pyproject.toml",)))
    failing = record(
        40,
        attempt=3,
        output=(
            'E   File "/workspace/issue-40/pyproject.toml", line 3\n'
            'E   File "/usr/local/lib/python3.12/importlib/__init__.py", line 90'
        ),
    )

    assert observe(declared, results={ref(40): failing}).blockers == ()


def test_an_undeclared_repo_file_behind_the_workspace_prefix_still_vetoes():
    # The veto still exists for the case it was written for: a repeated
    # failure entirely inside repo files the task was never given.
    stuck = ledger(entry(40, attempt=4, files=("pyproject.toml",)))
    failing = record(
        40,
        attempt=3,
        output=(
            "/workspace/issue-40/src/settings.py:7: ImportError\n"
            'File "/usr/local/lib/python3.12/importlib/__init__.py", line 90'
        ),
    )

    observation = observe(stuck, results={ref(40): failing})

    assert len(observation.blockers) == 1
    assert observation.blockers[0].paths == ("src/settings.py",)


def test_a_dotfile_directory_is_not_a_hostname():
    # The hostname heuristic drops `docs.pytest.org/...`; a leading dot is a
    # shape a hostname cannot take, and `lstrip("./")` used to rename the file.
    assert mentioned_paths("error in .github/workflows/ci.yml") == (".github/workflows/ci.yml",)


def test_two_runs_of_one_failure_share_a_signature():
    assert failure_signature("FAILED tests/test_a.py::test_x in 0.42s") == failure_signature(
        "FAILED tests/test_a.py::test_x in 11.7s"
    )
    assert failure_signature("expected 3") != failure_signature("no such file")


# --------------------------------------------------------------------------
# The v1 node still answers in the shape `graph.py` routes on
# --------------------------------------------------------------------------


def test_the_v1_node_still_answers_in_the_graphs_shape():
    state = {
        "objective": OBJECTIVE,
        "round": 2,
        "stalls": 0,
        "tasks": {
            "add-sub": TaskRecord(id="add-sub", status="verified", attempts=1, files=["calc.py"]),
        },
    }

    answer = judge_node(state)

    assert answer["round"] == 3 and answer["stalls"] == 0
    assert answer["last_judgement"].request_satisfied
    assert answer["events"] and "round 3" in answer["events"][0]


# --------------------------------------------------------------------------
# Deciding whether to replan at all
# --------------------------------------------------------------------------


def _verdict(
    *,
    satisfied: bool = False,
    progress: bool = True,
    in_loop: bool = False,
    stalls: int = 0,
    unresolved: bool = False,
) -> Verdict:
    return Verdict(
        judgement=ProgressJudgement(
            request_satisfied=satisfied,
            progress_being_made=progress,
            in_loop=in_loop,
            reason="fixture",
        ),
        stalls=stalls,
        observation=Observation(signals={}),
        unresolved=unresolved,
    )


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (_verdict(satisfied=True), SATISFIED),
        (_verdict(progress=True), PROGRESSING),
        (_verdict(progress=False, unresolved=True), UNRESOLVED),
        (_verdict(progress=False, stalls=1), TOO_SOON),
    ],
)
def test_decide_refuses_and_says_why(verdict: Verdict, expected: str):
    assert decide(verdict, max_stalls=2) == expected


def test_decide_stops_after_the_run_has_rewritten_itself_enough():
    stalled = _verdict(progress=False, stalls=2)

    assert decide(stalled, max_stalls=2, replans=0) == ""
    assert decide(stalled, max_stalls=2, replans=2) == EXHAUSTED


# --------------------------------------------------------------------------
# An issue store, and the one path that writes to it
# --------------------------------------------------------------------------


@dataclass
class Tracker:
    """The smallest store that can tell an update from a second issue.

    A sibling of `test_planner_issues.py`'s, kept to the four calls this file's
    write path makes: the round trip is that file's subject, and the question
    here is only whether a replan lands on the issues that already exist.
    """

    issues: dict[int, dict[str, Any]] = field(default_factory=dict)
    next_number: int = 1
    comments: list[tuple[int, str]] = field(default_factory=list)

    def add(
        self,
        *,
        body: str,
        labels: Sequence[str] = (READY,),
        title: str = "a task",
        state: str = "open",
    ) -> int:
        number = self.next_number
        self.next_number += 1
        self.issues[number] = {
            "number": number,
            "title": title,
            "body": body,
            "labels": [{"name": name} for name in labels],
            "state": state,
            "state_reason": None,
        }
        return number

    def __call__(self, request: Any) -> Any:
        assert request.path.startswith(BASE), f"unexpected path {request.path}"
        tail = request.path[len(BASE):]
        payload = request.json()

        if not tail:
            if request.method == "GET":
                wanted = request.query.get("state", "open")
                return response(
                    200,
                    [
                        dict(issue)
                        for issue in self.issues.values()
                        if wanted == "all" or issue["state"] == wanted
                    ],
                )
            if request.method == "POST":
                number = self.add(
                    title=payload["title"],
                    body=payload.get("body", ""),
                    labels=payload.get("labels", ()),
                )
                return response(201, dict(self.issues[number]))

        parts = [urllib.parse.unquote(part) for part in tail.strip("/").split("/")]
        issue = self.issues[int(parts[0])]
        if len(parts) == 1:
            if request.method == "GET":
                return response(200, dict(issue))
            if request.method == "PATCH":
                issue.update(payload)
                return response(200, dict(issue))
        elif parts[1] == "labels":
            # The revival path used to relabel failed -> ready through this
            # client. #152 removed every label write in the package, so the
            # handler is gone too: reaching it is the control plane growing
            # back, which is a louder failure than quietly recording the write.
            raise AssertionError(
                f"a label write reached the tracker: {request.method} {request.path}"
            )
        elif parts[1] == "comments" and request.method == "POST":
            self.comments.append((issue["number"], payload["body"]))
            return response(201, {"id": len(self.comments)})
        raise AssertionError(f"unhandled {request.method} {request.path}")


@dataclass
class Spy:
    """A `writer` that records the write it should never have been given."""

    calls: list[tuple[Any, ...]] = field(default_factory=list)

    def __call__(self, client: Any, plan: Plan, **kwargs: Any) -> Any:
        self.calls.append((client, plan, kwargs))
        raise AssertionError("the tracker was rewritten")


@dataclass
class Proposal:
    """A planner with a scripted answer, and a record of the prompt it saw."""

    plan: Plan
    asked: list[Any] = field(default_factory=list)

    def invoke(self, messages: Sequence[tuple[str, str]]) -> Plan:
        self.asked.append(messages)
        return self.plan


class NoPlanner:
    def invoke(self, messages: Sequence[tuple[str, str]]) -> Plan:
        raise RuntimeError("connection refused")


def task(task_id: str, *, goal: str = "", files: Sequence[str] = (), depends_on: Sequence[str] = ()):
    return PlannedTask(
        id=task_id,
        goal=goal or f"{task_id} is done",
        files=list(files) or [f"src/swarm/{task_id}.py"],
        depends_on=list(depends_on),
    )


@pytest.fixture()
def tracker(fake_github):
    """A real `GitHubClient` over #31's transport, backed by a `Tracker`."""

    def build() -> tuple[Any, Tracker]:
        store = Tracker()
        client, _, _ = fake_github(handler=store)
        return client, store

    return build


def stalled(stalls: int = 2) -> Verdict:
    return _verdict(progress=False, in_loop=True, stalls=stalls)


# --------------------------------------------------------------------------
# What a replan does to the tracker
# --------------------------------------------------------------------------


def test_a_stalled_run_rewrites_its_issues_without_duplicating_them(tracker):
    client, store = tracker()
    keep = store.add(
        body=render_body("keep-me", goal="Keep me", files=["src/swarm/a.py"], verify=VERIFY),
        labels=[READY],
    )
    drop = store.add(
        body=render_body("drop-me", goal="Drop me", files=["src/swarm/b.py"], verify=VERIFY),
        labels=[READY],
    )
    before = load_ledger(client)

    report = _replan_(
        client,
        before,
        OBJECTIVE,
        stalled(),
        proposer=Proposal(
            Plan(
                tasks=[
                    task("keep-me", goal="Keep me, differently", files=["src/swarm/a.py"]),
                    task("brand-new", files=["src/swarm/c.py"]),
                ]
            )
        ),
    )

    assert report.replanned
    after = load_ledger(client)
    # The surviving id kept its issue - a second issue for work that already
    # has one is the failure §2 exists to prevent.
    assert after.entries["keep-me"].number == keep
    assert after.entries["keep-me"].goal == "Keep me, differently"
    assert set(after.entries) == {"keep-me", "drop-me", "brand-new"}
    assert store.issues[drop]["state"] == "closed"
    # `not_planned`, so readiness never reads a cancellation as a dependency met.
    assert store.issues[drop]["state_reason"] == "not_planned"
    assert len(store.issues) == 3


def test_a_dropped_task_with_a_worker_on_it_is_left_alone_and_reported(tracker):
    client, store = tracker()
    running = store.add(
        body=render_body("in-flight", goal="Being worked on", files=["src/swarm/d.py"], verify=VERIFY),
        labels=[CLAIMED],
    )
    before = load_ledger(client)

    report = _replan_(
        client,
        before,
        OBJECTIVE,
        stalled(),
        proposer=Proposal(Plan(tasks=[task("something-else")])),
    )

    assert report.replanned
    assert store.issues[running]["state"] == "open"
    assert [action.task_id for action in report.retained] == ["in-flight"]
    # The reason names the state the authority answered in rather than the label
    # storing it (#212), which is also epic #140's other half: a `swarm:*` string
    # in a report is a storage detail leaking into what a human reads.
    assert "claimed" in report.retained[0].reason
    assert "swarm:" not in report.retained[0].reason


def test_a_kept_failed_task_is_revived_by_the_replan_that_kept_it(tracker):
    """The follow-up to the live stall this module replanned: the report read
    "0 created, 3 updated, 0 retired, 11 left alone" and the failed task whose
    chain blocked everything was left alone, so the run stayed 0-ready and
    re-stalled. A replan that keeps a failed task now revives it, budget
    intact - the failure signature is what makes that safe."""
    client, store = tracker()
    body = render_body("stuck", goal="Unblock the chain", files=["src/swarm/a.py"], verify=VERIFY, attempt=3)
    body = body.replace(
        render_marker("stuck", 3), legacy_marker("stuck", 3, blocker="ab12cd34ef", streak=3)
    )
    number = store.add(body=body, labels=[FAILED])
    before = load_ledger(client)

    report = _replan_(
        client,
        before,
        OBJECTIVE,
        stalled(),
        proposer=Proposal(
            Plan(tasks=[task("stuck", goal="Unblock the chain", files=["src/swarm/a.py"])])
        ),
    )

    assert report.replanned
    assert [action.number for action in report.revived] == [number]
    # The revival is the report and the comment, and since #152 nothing else:
    # this used to assert the issue had been relabelled `swarm:failed` ->
    # `swarm:ready`, and there is no label plane left to move it on. What the
    # next cycle acts on is the state it derives, so the issue is left exactly
    # as it was - a write here would be the control plane growing back.
    assert {label["name"] for label in store.issues[number]["labels"]} == {FAILED}
    # The marker survives verbatim: nothing is reset, the arithmetic guards.
    assert legacy_marker("stuck", 3, blocker="ab12cd34ef", streak=3) in store.issues[number]["body"]
    assert store.comments[0][0] == number
    assert store.comments[0][1].startswith("apiary: the replan retained this task")
    assert "revived" in report.summary()


def test_a_replan_resets_the_stall_count_and_counts_itself(tracker):
    client, _ = tracker()
    client_ledger = load_ledger(client)

    report = _replan_(
        client,
        client_ledger,
        OBJECTIVE,
        stalled(),
        replans=1,
        proposer=Proposal(Plan(tasks=[task("fresh-start")])),
    )

    assert report.replanned and report.stalls == 0 and report.replans == 2


def test_the_prompt_carries_the_failures_and_every_existing_id():
    tasks = ledger(
        entry(1, task_id="broken", files=("src/swarm/a.py",), attempt=2),
        entry(2, task_id="waiting", label=BLOCKED),
    )
    stall = Answers(
        ProgressJudgement(
            request_satisfied=False, progress_being_made=False, in_loop=False, reason="idle"
        )
    )
    verdict = verdict_of(
        tasks,
        tasks,
        results={ref(1): record(1, attempt=1, reason="ZeroDivisionError")},
        prior_results={ref(1): record(1, attempt=1, reason="ZeroDivisionError")},
        oracle=stall,
    )

    failures, tracked = _brief_(tasks, verdict)

    assert "broken" in failures and "ZeroDivisionError" in failures
    # Every id, not only the failing one: an id the model never sees is an id
    # it invents a replacement for.
    assert "broken" in tracked and "waiting" in tracked


def test_the_replan_shows_the_model_the_repositorys_tree(tracker):
    """A replan is asked for a *different* decomposition, and the strongest
    grounding for one is what the repository actually contains by now - the
    stalled attempt's own half-landed files included."""
    client, store = tracker()
    store.add(
        body=render_body("existing", goal="Existing", files=["src/swarm/a.py"], verify=VERIFY)
    )
    before = load_ledger(client)
    client.list_tree = lambda ref=None: ["src/swarm/a.py", "README.md"]
    proposer = Proposal(Plan(tasks=[task("existing", files=["src/swarm/a.py"])]))

    report = _replan_(client, before, OBJECTIVE, stalled(), proposer=proposer, verify=VERIFY)

    assert report.replanned
    human = dict(proposer.asked[0])["human"]
    assert "The repository currently contains these files" in human
    assert "README.md" in human


def test_a_tree_read_failure_does_not_fail_the_replan(tracker):
    """Pinned: the listing is advisory, never a blocker. A 502 from the trees
    API sends the replan prompt exactly as it was before listings existed,
    because a stalled run must never be made unfixable by a transient read."""
    client, store = tracker()
    store.add(
        body=render_body("existing", goal="Existing", files=["src/swarm/a.py"], verify=VERIFY)
    )
    before = load_ledger(client)

    def boom(ref=None):
        raise GitHubError("GET /git/trees/main -> 502")

    client.list_tree = boom
    proposer = Proposal(Plan(tasks=[task("existing", files=["src/swarm/a.py"])]))

    report = _replan_(client, before, OBJECTIVE, stalled(), proposer=proposer, verify=VERIFY)

    assert report.replanned
    assert dict(proposer.asked[0])["human"] == f"Objective:\n{OBJECTIVE}"


# --------------------------------------------------------------------------
# Every way a replan declines to write
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (_verdict(satisfied=True, progress=True), SATISFIED),
        (_verdict(progress=True), PROGRESSING),
        (_verdict(progress=False, stalls=1), TOO_SOON),
        (_verdict(progress=False, stalls=2, unresolved=True), UNRESOLVED),
    ],
)
def test_a_refusal_writes_nothing_and_asks_nobody(verdict: Verdict, expected: str):
    spy = Spy()
    proposer = Proposal(Plan(tasks=[task("never-written")]))

    report = _replan_(
        object(), ledger(entry(1)), OBJECTIVE, verdict, proposer=proposer, writer=spy
    )

    assert not report.replanned and report.reason == expected
    assert spy.calls == [] and proposer.asked == []


def test_a_plan_with_nothing_writable_in_it_is_refused_before_the_write():
    # `write_plan` retires every entry a plan does not contain, so an empty
    # answer would close the whole tracker in one call.
    spy = Spy()

    report = _replan_(
        object(),
        ledger(entry(1), entry(2)),
        OBJECTIVE,
        stalled(),
        proposer=Proposal(Plan(tasks=[PlannedTask(id="", goal="", files=[])])),
        writer=spy,
    )

    assert not report.replanned and report.reason == NO_TASKS
    assert spy.calls == []


def test_a_planner_that_cannot_be_reached_leaves_the_stall_standing():
    spy = Spy()

    report = _replan_(
        object(), ledger(entry(1)), OBJECTIVE, stalled(3), proposer=NoPlanner(), writer=spy
    )

    assert not report.replanned and "could not be reached" in report.reason
    # The budget is not restarted: the next cycle tries again from where this
    # one stood, rather than from zero.
    assert report.stalls == 3 and report.replans == 0
    assert spy.calls == []


def test_a_ring_in_the_proposed_plan_leaves_the_tracker_alone(tracker):
    client, store = tracker()
    store.add(body=render_body("existing", goal="Existing", files=["src/swarm/a.py"], verify=VERIFY))
    before = load_ledger(client)

    report = _replan_(
        client,
        before,
        OBJECTIVE,
        stalled(),
        proposer=Proposal(
            Plan(
                tasks=[
                    task("first", depends_on=["second"]),
                    task("second", depends_on=["first"]),
                ]
            )
        ),
    )

    assert not report.replanned and "not writable" in report.reason
    # `write_plan` orders before it writes, so nothing was created and nothing
    # was retired.
    assert len(store.issues) == 1 and store.issues[1]["state"] == "open"


def test_a_rewritten_issue_keeps_the_runs_own_verify_command(tracker):
    """A stall must not re-point the whole tracker at a command it cannot run.

    The command belongs to the repository - the scaffold's, in a generated one -
    and the run resolved it once at the top. Defaulting it here instead would
    hand every rewritten issue `SETTINGS.verify_command`, which is v1's pytest
    invocation and exactly the gate that was already red.
    """
    client, store = tracker()
    store.add(body=render_body("existing", goal="Existing", files=["src/swarm/a.py"], verify=VERIFY))
    scaffolded = "python3 -m unittest discover -q"

    report = _replan_(
        client,
        load_ledger(client),
        OBJECTIVE,
        stalled(),
        verify=scaffolded,
        proposer=Proposal(Plan(tasks=[task("existing"), task("second", files=["src/swarm/b.py"])])),
    )

    assert report.replanned
    after = load_ledger(client)
    assert {entry.verify for entry in after.entries.values()} == {scaffolded}


# --------------------------------------------------------------------------
# The acceptance criteria, end to end
# --------------------------------------------------------------------------


def test_a_repeatedly_failing_run_replans_instead_of_retrying_forever(tracker):
    client, store = tracker()
    store.add(
        body=render_body("parse-headers", goal="Headers parse", files=["src/swarm/h.py"], verify=VERIFY),
        labels=[READY],
    )
    boom = dict(reason="TypeError: cannot parse header", output="1 failed in 0.4s")

    # Three cycles: the task fails, is retried, and fails identically.
    cycles = [
        (ledger(entry(1, task_id="parse-headers", attempt=1)), {ref(1): record(1, **boom)}),
        (
            ledger(entry(1, task_id="parse-headers", attempt=2)),
            {ref(1): record(1, attempt=1, **boom)},
        ),
        (
            ledger(entry(1, task_id="parse-headers", attempt=3)),
            {ref(1): record(1, attempt=2, **boom)},
        ),
    ]

    # The first cycle has nothing to compare against, so it is the one place a
    # model is worth asking; the two after it are settled by arithmetic.
    oracle = Answers(
        ProgressJudgement(
            request_satisfied=False,
            progress_being_made=True,
            in_loop=False,
            reason="the first attempt has been made",
        )
    )

    verdict = None
    previous: Observation | None = None
    stalls = 0
    for current, results in cycles:
        observation = observe(current, results=results)
        verdict = judge(observation, previous, objective=OBJECTIVE, stalls=stalls, oracle=oracle)
        previous, stalls = observation, verdict.stalls

    assert len(oracle.asked) == 1
    assert verdict is not None and verdict.judgement.in_loop and verdict.stalls == 2
    assert verdict.should_replan(max_stalls=2)

    report = _replan_(
        client,
        load_ledger(client),
        OBJECTIVE,
        verdict,
        proposer=Proposal(
            Plan(
                tasks=[
                    task("parse-headers", goal="Headers parse, character by character"),
                    task("header-fixtures", files=["tests/test_headers.py"]),
                ]
            )
        ),
    )

    assert report.replanned
    after = load_ledger(client)
    assert set(after.entries) == {"parse-headers", "header-fixtures"}
    # The attempt counter survives the rewrite: a replan is not a free retry.
    assert after.entries["parse-headers"].number == 1
    assert len(store.issues) == 2


def test_a_run_whose_pull_requests_are_only_being_rebased_never_replans():
    spy = Spy()
    review = ledger(entry(1, label=REVIEW), entry(2, label=REVIEW))

    stalls = 0
    previous: Observation | None = None
    verdict = None
    for cycle in range(4):
        observation = observe(review, churn={ref(1): cycle, ref(2): cycle})
        verdict = judge(observation, previous, objective=OBJECTIVE, stalls=stalls, oracle=Never())
        previous, stalls = observation, verdict.stalls

    assert verdict is not None and not verdict.stalled and verdict.stalls == 0
    report = _replan_(object(), review, OBJECTIVE, verdict, proposer=Proposal(Plan(tasks=[])), writer=spy)
    assert not report.replanned and report.reason == PROGRESSING
    assert spy.calls == []


def test_a_url_in_the_failure_is_not_a_file_the_task_cannot_edit():
    """pytest prints `-- Docs: https://docs.pytest.org/...capture-warnings.html`
    in its own footer. The path extractor matched everything after the scheme,
    the judge reported the task as failing on a file outside its ## Files, and
    that one diagnosis suppressed the replan which exists for exactly the
    failure at hand. Observed live, on the first greenfield run's issue #1."""
    from swarm.nodes.judge import mentioned_paths

    footer = (
        "1 warning in 0.00s\n"
        "-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n"
        "see also docs.pytest.org/en/stable/reference.html"
    )

    assert mentioned_paths(footer) == ()


def test_repository_paths_are_still_extracted_beside_a_url():
    from swarm.nodes.judge import mentioned_paths

    text = ("FAILED tests/test_main.py::test_total - AssertionError\n"
            "-- Docs: https://docs.pytest.org/en/stable/warnings.html\n"
            "src/pkg/budget.py:12: in get_total")

    assert mentioned_paths(text) == ("tests/test_main.py", "src/pkg/budget.py")
