"""The run recorder: every real run is a replay corpus run (#146).

**Two files converge here.** `orchestrator/observed.py` was split out of
`orchestrator/shadow.py` by #245 so that removing the window would be a file
deletion rather than a dissection; this file held the two tests asserting that
split held. This commit is the deletion, so the recorder's own tests move in from
`test_shadow.py` and this becomes their home.

**#245's two tests are deleted here, and that is the point of them.** They
asserted that `observed.py` does not import `shadow.py` - by an `ast` pass and by
a real import with the window poisoned - so that the deletion stayed simple. With
`shadow.py` gone neither can fail: there is no module left to import, and
poisoning a name nothing imports is a no-op. A test that cannot fail is worse
than no test, because it reads as coverage. They did their job and it was the
deletion; the deletion has landed.

**28 tests went with the window.** They are listed in the pull request rather
than summarised, because a deletion ticket that reports a test count going down
without saying which ones is a deletion nobody can check.

What is left is the recorder, load-bearing for a reason that predates the cutover
and outlives it. `tests/fixtures/runs/README.md` names the missing recorder as the
whole reason #145's replay corpus is *synthesised*, and a synthesised corpus
proves the reducer self-consistent and nothing about reality. Four things this
suite holds down:

1. **A recorded cycle replays.** The recorder writes `observed.jsonl` and a
   manifest, `tests/fixtures/corpus.py` loads the directory with no change of its
   own, and the replay reproduces the line it was recorded from.
2. **It costs no API call.** Asserted by counting the calls a cycle makes,
   because a recorder that quietly doubled the rate-limit spend would be switched
   off by the first person it throttled.
3. **It cannot fail the cycle.** A recorder that raises stops recording and says
   so; the run carries on holding its containers.
4. **A cycle that could not see records nothing.** Blind is not clean, and a
   fixture claiming nothing was in review when the truth is "we could not look"
   is worse than a missing line, because a fixture is trusted.

`swarm show` still reads the shadow's own events, and three tests here hold that
down: the events are gone from *new* runs, and runs recorded before #152 must
still print (#152's seventh acceptance criterion).

The helpers come from `test_reconcile` rather than being rebuilt here. They are
the doubles that drive a real cycle end to end, and a second copy of them would
be a second thing to keep in step with the loop.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

FAILED = "needs-human"

DONE = "landed"

from swarm.artifacts import (
    CORPUS_MANIFEST_NAME,
    OBSERVED_LOG_NAME,
    STATE_DIVERGENCE,
    STATE_OVERRIDE,
    STATE_SHADOW,
    DivergenceTally,
    RunArtifacts,
    read_run,
    show_text,
)
from swarm.containers.manager import CREATED_STATE, RUNNING_STATE, Handle
from swarm.github.readiness import IssueState
from swarm.github.refs import pull_ref
from swarm.github.refs import task_ref as ref
from swarm.orchestrator.checks import PullState
from swarm.github.readiness import READY
from swarm.orchestrator.dispatcher import CLAIMED, REVIEW
from swarm.orchestrator.reconcile import (
    CycleReport,
    ReconcilePlan,
    ReconcileReport,
)
from swarm.orchestrator.authority import INFRASTRUCTURE_CEILING
from swarm.orchestrator.observed import (
    control_labels,
    observation_for,
    observed_line,
    record_cycle,
)
from swarm.run import Run

from test_reconcile import (  # the doubles that drive a real cycle
    REPO,
    RUN_ID,
    TASK_ISSUE,
    TASK_PULL,
    TASK_REF,
    a_lifecycle_run,
    entry,
    green,
    ledger,
    pending,
    reaches_review,
    record,
    recorder,
)
from swarm.store import STORE_DIR_ENV
from swarm.worker.result import (
    EXIT_INFRASTRUCTURE,
    EXIT_TASK_FAILED,
    ResultRecord,
    latest_named,
    load_named,
    record_path,
    write_result,
)
from swarm.orchestrator.derived import LANDED, NEEDS_HUMAN
from swarm.orchestrator.derived import REVIEW as REVIEW_STATE

BLOCKED = "swarm:blocked"


@pytest.fixture(autouse=True)
def store_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`test_reconcile.store_root`'s reason: an autouse fixture is per-module.

    Without it every `reconciler()` built here would open the *operator's*
    store at `.swarm/store` and write test judgments into a real project's
    retry budgets. Nothing would fail; the next real run would simply believe
    something untrue about its own history.
    """
    root = tmp_path / "store"
    monkeypatch.setenv(STORE_DIR_ENV, str(root))
    return root


#: A fixed clock. `record_order` breaks a tie inside one attempt on the moment
#: the record was finished, so two records that share an attempt need distinct
#: ones for "newest" to mean anything.
_BASE = dt.datetime(2026, 8, 14, 12, 0, tzinfo=dt.timezone.utc)


def a_result(issue: int, *, attempt: int, exit_code: int, minute: int) -> ResultRecord:
    """One worker's record, stamped the way `from_worker` stamps every record."""
    return ResultRecord(
        run_id="run-1",
        issue=issue,
        attempt=attempt,
        exit_code=exit_code,
        started_at=_BASE + dt.timedelta(minutes=minute),
        finished_at=_BASE + dt.timedelta(minutes=minute),
    )


# --------------------------------------------------------------------------
# Building a cycle by hand
# --------------------------------------------------------------------------


def report(
    *entries: Any,
    index: int = 0,
    readiness: Any = None,
    dispatched: Any = None,
    checks: Any = None,
) -> CycleReport:
    """A finished `CycleReport` carrying nothing but a ledger.

    The shadow window is a projection of a cycle that has already decided, so
    everything it reads is either on the report or was passed alongside it -
    which is what makes a report this bare a legitimate input rather than a
    stub with holes in it.
    """
    return CycleReport(
        index=index,
        ledger=ledger(*entries),
        result=ReconcileReport(plan=ReconcilePlan()),
        readiness=readiness,
        dispatched=dispatched,
        checks=checks,
    )


def recorded(cycle: CycleReport, **facts: Any) -> tuple[bool, list[Any]]:
    """Record one report and hand back the lines it wrote.

    `pulls={}` by default and never `None`: `None` is "this cycle could not list
    pull requests", which makes the recorder write nothing at all - the
    distinction `checks.read_pulls` exists to keep, and one a test helper must
    not smuggle past.
    """
    facts.setdefault("pulls", {})
    lines: list[Any] = []
    wrote = record_cycle(cycle, record=lines.append, **facts)
    return wrote, lines


def handle(issue: int, *, state: str = RUNNING_STATE) -> Handle:
    return Handle(id=f"{issue:0>64x}", run_id=RUN_ID, issue=issue, state=state)


def pull(number: int, branch: str, *, sha: str = "") -> PullState:
    """A `PullState` the way a listing produces one: the number is a `PullRef`
    since #185, minted through the adapter rather than constructed here."""
    return PullState(number=pull_ref(number), branch=branch, sha=sha or f"{number:0>40x}")


def divergences(seen: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [fields for name, fields in seen if name == STATE_DIVERGENCE]


# --------------------------------------------------------------------------
# 1. A divergence is named
# --------------------------------------------------------------------------










# --------------------------------------------------------------------------
# 2. The labels still decide
# --------------------------------------------------------------------------




def test_recording_adds_no_github_call():
    """#146's fourth criterion, which the recorder inherits.

    `Snapshot` already forced every read the recorder uses, so switching it on
    must not move the number of client calls. Asserted by running the same cycle
    with a recorder and without one, and then checking a line really was written
    - otherwise the counts would match because nothing happened.
    """
    def calls(*, recording: bool) -> tuple[list[str], list[Any]]:
        client, fleet, loop, _ = a_lifecycle_run()
        lines: list[Any] = []
        loop.record = lines.append if recording else None
        loop.cycle()
        return client.log, lines

    on, lines = calls(recording=True)
    off, _ = calls(recording=False)

    assert on == off
    assert len(lines) == 1
    assert lines[0]["cycle"] == 0










# --------------------------------------------------------------------------
# 3. Which control plane, sampled when
# --------------------------------------------------------------------------


def test_the_control_side_is_the_labels_this_cycle_left_not_the_ones_it_read():
    """The module's central decision, made visible.

    Readiness and the dispatcher both write labels the cycle never folds back
    into its ledger. Diffing against the unfolded ledger would report every one
    of those writes as a disagreement - which is the loop's own progress, not
    evidence about anything.
    """
    from swarm.github.readiness import Verdict

    moved = report(
        entry(4, label=BLOCKED),
        readiness=type("Plan", (), {"verdicts": (
            Verdict(ref=ref(4), task_id="task-4", current_label=BLOCKED, label=READY),
        )})(),
    )

    assert control_labels(moved) == {"task-4": READY}


def test_a_task_mergeability_escalated_is_in_the_control_map():
    """The fourth writer of a terminal label, and the one the cycle never folds.

    `lifecycle._landed_or_human` names all four - the reconciler, the recovery
    sweep, mergeability and the check gate - and `Reconciler.cycle` folds three
    of them. A pull request that will not rebase inside its update budget is
    escalated by `apply_mergeability` alone, so a control map built without it
    would report every starved task as a divergence the resolver invented.
    """
    from swarm.orchestrator.mergeability import MergeabilityPlan, MergeabilityReport
    from swarm.orchestrator.reconcile import Transition

    starved = report(
        entry(4, label=REVIEW),
        checks=None,
    )
    starved = replace(
        starved,
        mergeability=MergeabilityReport(
            plan=MergeabilityPlan(),
            applied=(
                Transition(
                    ref=ref(4),
                    task_id="task-4",
                    from_state=REVIEW_STATE,
                    to_state=NEEDS_HUMAN,
                    reason="its branch could not be updated within the round cap",
                ),
            ),
        ),
    )

    assert control_labels(starved) == {"task-4": FAILED}


def test_the_check_gate_wins_over_mergeability_for_the_same_task():
    """The gate runs after mergeability, and step 1 already carries its answer -
    so the earlier writer must not be allowed to overwrite the later one."""
    from swarm.orchestrator.checks import ChecksPlan, ChecksReport
    from swarm.orchestrator.mergeability import MergeabilityPlan, MergeabilityReport
    from swarm.orchestrator.reconcile import Transition

    def moved(to_state: str) -> Any:
        return Transition(
            ref=ref(4),
            task_id="task-4",
            from_state=REVIEW_STATE,
            to_state=to_state,
            reason="moved",
        )

    both = replace(
        report(entry(4, label=DONE), checks=ChecksReport(
            plan=ChecksPlan(), applied=(moved(LANDED),)
        )),
        mergeability=MergeabilityReport(plan=MergeabilityPlan(), applied=(moved(NEEDS_HUMAN),)),
    )

    assert control_labels(both) == {"task-4": DONE}


def test_a_dispatch_that_claimed_and_failed_to_spawn_still_counts_as_claimed():
    """`DispatchFailure.claimed` is exactly "the label was written and no
    container is running under it" - a claim the control plane is holding
    whether or not the spawn worked, and the case #35's sweep exists for."""
    from swarm.orchestrator.dispatcher import DispatchFailure, DispatchPlan, DispatchReport

    failed = report(
        entry(4, label=READY),
        dispatched=DispatchReport(
            plan=DispatchPlan(),
            failed=(DispatchFailure(number=4, reason="docker is not there", claimed=True),),
        ),
    )

    assert control_labels(failed) == {"task-4": CLAIMED}


def test_a_cycle_that_could_not_list_pull_requests_records_nothing(tmp_path):
    """`None` is not `{}`, and the distinction decides what a fixture claims.

    An empty mapping read as the answer records every task in review as having no
    pull request - a corpus line that replays to a world which never existed.
    That is worse than a missing line, because a fixture is trusted: the next
    reader takes it as evidence. So a blind cycle writes nothing and says so.
    """
    lines: list[object] = []

    wrote = record_cycle(report(entry(4, label=REVIEW)), record=lines.append, pulls=None)

    assert wrote is False
    assert lines == []


def test_a_dry_run_neither_shadows_nor_records(tmp_path):
    """Two reasons, and the second is the serious one.

    `apply_plan` returns before writing on a dry run, so nothing is folded and
    the control map would be *last* cycle's labels - the lagging-cache
    comparison this module exists to avoid. And `RunArtifacts.observed` stamps
    the directory `origin: "recorded"`, so a dry run would enter the replay
    corpus wearing the one label that means "this happened for real".
    """
    run = Run.start(REPO, "a dry run", run_id=RUN_ID)
    artifacts = RunArtifacts.open(run, root=tmp_path)

    client, fleet, loop, seen = a_lifecycle_run(label=CLAIMED)
    loop.dry_run = True
    loop.record = artifacts.observed
    loop.cycle()

    # The shadow window and the recorder, which are what the two reasons above
    # are about. **Not** `state.override` (#147): that one is sampled before
    # anything is folded and writes nothing to the corpus, so neither reason
    # applies to it - and a dry run that could not say "the label says claimed,
    # the world says eligible, and a real cycle would have acted on eligible"
    # would be answering a different question than the one it was asked.
    assert [one for one in seen if one[0] in {STATE_SHADOW, STATE_DIVERGENCE}] == []
    assert [one[0] for one in seen if one[0] == STATE_OVERRIDE] == [STATE_OVERRIDE]
    assert not (artifacts.path / OBSERVED_LOG_NAME).exists()
    assert not (artifacts.path / CORPUS_MANIFEST_NAME).exists()


def test_two_containers_under_one_task_reach_the_recording(tmp_path):
    """The raw listing is passed, not `Reconciler._handles`' first-wins map.

    That collapse made an exited container listed ahead of a running one read as
    not-claimed, so a genuine double-spawn - which `dispatcher.release` is written
    about - was structurally invisible. A recorded cycle has to keep the fact, or
    the corpus cannot replay the case it was recorded for.
    """
    wrote, lines = recorded(
        report(entry(4, label=CLAIMED)),
        containers=[handle(4, state="exited"), handle(4, state=RUNNING_STATE)],
    )

    assert wrote is True
    assert [one["running"] for one in lines[0]["containers"]] == [False, True]




def test_a_revival_is_in_the_control_map():
    """The sixth writer, and the one that is not in `cycle` at all.

    `planner.revive` runs from `_judge` - before this window - and moves a task
    `swarm:failed -> swarm:ready` on GitHub with nothing folding it back.
    """
    from swarm.nodes.planner import IssueAction, PlanReport
    from swarm.orchestrator.replan import ReplanReport

    revived = replace(
        report(entry(4, label=FAILED)),
        replanned=ReplanReport(
            repo=REPO,
            replanned=True,
            plan=PlanReport(
                repo=REPO, actions=(IssueAction("revived", "task-4", 4, reason="streak 1 of 3"),)
            ),
        ),
    )

    assert control_labels(revived) == {"task-4": READY}


# --------------------------------------------------------------------------
# 4. Expected against real
# --------------------------------------------------------------------------


















# --- the negative shapes: an account must not absorb a wrong derived answer ---
#
# Each of these is a divergence whose *control* side and evidence match an
# expected kind while its *derived* side contradicts that kind's own argument.
# Before the classifier was two-sided every one of them was filed as expected
# and dropped out of `unexplained` - the number #147's gate reads.














# --------------------------------------------------------------------------
# 5. `swarm show`
# --------------------------------------------------------------------------


def test_show_reports_a_divergence_count_without_anybody_reading_the_jsonl(tmp_path):
    """#146's third criterion, now a statement about **recorded** runs.

    Nothing emits these events since #152 removed the window. The reader stays,
    because `events.jsonl` is append-only and read back: a run recorded before
    this ticket must still print (#152's seventh acceptance criterion), and a
    reader deleted alongside its writer would turn every archived run into a
    traceback.
    """
    events = [
        {
            "event": STATE_SHADOW,
            "cycle": 0,
            "tasks": 2,
            "independent": 1,
            "divergences": 2,
            "unexplained": 1,
        },
        {"event": STATE_DIVERGENCE, "task": "a", "kind": INFRASTRUCTURE_CEILING},
        {"event": STATE_DIVERGENCE, "task": "b", "kind": ""},
    ]
    tally = DivergenceTally.from_events(events)

    assert (tally.ran, tally.cycles, tally.total, tally.unexplained) == (True, 1, 2, 1)
    assert (tally.compared, tally.independent) == (2, 1)
    assert tally.by_kind == ((INFRASTRUCTURE_CEILING, 1),)
    assert tally.unexplained_tasks == ("b",)
    assert "2 task-cycle(s) compared" in tally.text()
    assert "1 of them independent" in tally.text()
    assert "1 unexplained" in tally.text()
    assert "unexplained on: b" in tally.text()


def test_show_says_not_run_rather_than_implying_a_clean_run():
    """The distinction the whole tally exists for: a gate that reads an
    unmeasured run as a clean one passes on nothing at all."""
    assert DivergenceTally.from_events([]).ran is False
    assert "not run" in DivergenceTally.from_events([]).text()


def test_show_prints_the_tally_for_a_run_recorded_before_this_ticket(tmp_path):
    """The events are written by hand because nothing writes them any more.

    That is the point rather than a compromise: this is the archived shape, and
    `swarm show` has to keep rendering it.
    """
    run = Run.start(REPO, "a run from before #152", run_id=RUN_ID)
    artifacts = RunArtifacts.open(run, root=tmp_path)
    artifacts.event(
        STATE_SHADOW, cycle=0, tasks=1, independent=1, divergences=1, unexplained=0
    )
    artifacts.event(STATE_DIVERGENCE, cycle=0, task="task-4", kind=INFRASTRUCTURE_CEILING)

    text = show_text(read_run(artifacts.path))

    assert "derived shadow: 1 task-cycle(s) compared over 1 cycle(s)" in text
    assert "1 divergence(s), 0 unexplained" in text
    assert f"{INFRASTRUCTURE_CEILING} 1" in text


# --------------------------------------------------------------------------
# 6. The recorder, and the round trip that makes a run into corpus
# --------------------------------------------------------------------------


def test_a_shadowed_cycle_records_a_line_the_corpus_loader_accepts(tmp_path):
    """The whole reason the recorder is in this ticket.

    `tests/fixtures/runs/README.md`: four of the five files a corpus run needs
    are already written verbatim by a live run, and `observed.jsonl` is the
    fifth. Once a cycle emits it, dropping a real run into `runs/` is a `cp -r`
    - which retires the largest risk in epic #140, that every run proving the
    resolver was made up.
    """
    from fixtures.corpus import load_corpus

    run = Run.start(REPO, "record a cycle", run_id=RUN_ID)
    artifacts = RunArtifacts.open(run, root=tmp_path)
    write_result(record(TASK_ISSUE, 1, attempt=0), artifacts.results_dir)

    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = artifacts.results_dir
    loop.record = artifacts.observed
    loop.events = artifacts.event
    loop.cycle()
    loop.cycle()

    loaded = load_corpus(artifacts.path)

    assert loaded.origin == "recorded"
    assert [cycle.index for cycle in loaded.cycles] == [0, 1]
    # The manifest declares nothing, on purpose: the harness fails on an
    # undeclared divergence, so a recorded run refuses to pass until a human
    # has written the argument for each one.
    assert loaded.expected == ()


def test_a_recorded_cycle_replays_to_the_line_it_was_recorded_from(tmp_path):
    """The round trip, and the only assertion that proves the recorder wrote the
    observation the resolver actually saw rather than a plausible one.

    **The comparison changed with the window** (#152). It used to be "the replay
    diverges the same way the live cycle reported", against `ShadowWindow.last`.
    There is no live comparison any more, and re-resolving here would only prove
    `resolve` is a function. So the assertion is the one that was always the real
    risk: **recorder and loader agree on the format.** Load the directory, project
    the loaded observation back through the recorder, and require the same line.
    A field the recorder writes and the loader drops - or reads into the wrong
    place - fails here, and that is the failure that would otherwise surface as a
    corpus run quietly exercising a world nobody recorded.

    The fixture carries a container, a pull request *and* a result record on the
    recorded cycle, because a thin observation would round-trip trivially: an
    earlier version of this test passed only because the fixture happened to have
    none of them at that moment.
    """
    from fixtures.corpus import load_corpus
    from swarm.github.branches import task_branch

    run = Run.start(REPO, "replay a cycle", run_id=RUN_ID)
    artifacts = RunArtifacts.open(run, root=tmp_path)

    client, fleet, loop, _ = a_lifecycle_run()
    loop.artifacts = artifacts.results_dir
    loop.record = artifacts.observed

    loop.cycle()  # dispatch: a running container for this task
    # …and now a pull request and a result record, without clearing the
    # container, so the recorded cycle carries all three.
    client.open_pulls = ((TASK_PULL, task_branch(ref(TASK_ISSUE), 0)),)
    client.check_runs = {client.head_of(TASK_PULL): pending()}
    write_result(record(TASK_ISSUE, 0, attempt=0, reason="verified"), artifacts.results_dir)
    loop.cycle()

    written = [
        json.loads(line)
        for line in (artifacts.path / OBSERVED_LOG_NAME).read_text().strip().splitlines()
    ]
    replayed = load_corpus(artifacts.path).cycles[-1]

    # The observation really did carry the three things a thin one would not.
    assert replayed.observation.containers and replayed.observation.pulls
    assert replayed.observation.results
    assert replayed.index == written[-1]["cycle"]

    # Everything but `control`, which is deliberately *not* symmetric: the
    # recorder writes the label and `load_corpus` translates it to the internal
    # state on the way in, because - as `observed_line` says - the day the labels
    # go it is the translation that gets deleted and not the recorded data. So
    # the two vocabularies are asserted separately rather than papered over with
    # a round trip that would have to translate one of them back.
    again = observed_line(replayed.observation, {})
    assert {k: v for k, v in again.items() if k != "control"} == {
        k: v for k, v in written[-1].items() if k != "control"
    }
    assert written[-1]["control"] == {"task-4242": REVIEW}
    assert replayed.control == {"task-4242": REVIEW_STATE}


def test_a_recorded_line_names_the_file_the_cycle_read(tmp_path):
    """#230's recorder half. The name is carried, not rebuilt.

    Two records for one issue at one attempt - what an infrastructure failure
    followed by a task failure leaves behind, since exit 2 consumes no attempt -
    and `write_result` files the second under the next free name rather than
    over the first. The cycle was handed the second; `record_path(issue,
    attempt)` names the first.

    The two disagree on their exit code, which is `AttemptFact.spends_budget`
    and so the attempt count. A line naming the wrong one replays as a
    different run from the one it was recorded from.
    """
    
    write_result(a_result(4, attempt=0, exit_code=EXIT_INFRASTRUCTURE, minute=0), tmp_path)
    read = a_result(4, attempt=0, exit_code=EXIT_TASK_FAILED, minute=5)
    write_result(read, tmp_path)

    latest = latest_named(load_named(tmp_path))
    cycle = report(entry(4, label=READY))
    line = observed_line(
        observation_for(cycle, results={ref(4): latest[4][1]}),
        control_labels(cycle),
        result_names={ref(4): latest[4][0]},
    )

    assert line["results"] == ["issue-4-attempt-1.json"]
    # The name the field would have rebuilt, which holds the other record.
    assert record_path("", 4, read.attempt).name == "issue-4-attempt-0.json"


def test_a_recorder_with_no_names_still_writes_the_rebuilt_one(tmp_path):
    """The fallback, for a caller holding an `Observation` and no directory.

    Exactly right whenever no two records share an attempt, which is every run
    that never hit infrastructure trouble - and every hand-written corpus line
    that predates the argument.
    """
    
    cycle = report(entry(4, label=READY))
    line = observed_line(
        observation_for(cycle, results={ref(4): a_result(4, attempt=2, exit_code=1, minute=0)}),
        control_labels(cycle),
    )

    assert line["results"] == ["issue-4-attempt-2.json"]


def test_the_recorded_line_replays_as_the_record_the_cycle_was_handed(tmp_path):
    """The seam, closed end to end: recorder out, loader in.

    `observed_line` writes the name and `fixtures/corpus._cycle` reads it back,
    over a directory where the name is the only thing telling two records
    apart. The `AttemptFact` the replay builds has to be the record this cycle
    resolved on - the exit code included, because that is what decides whether
    the attempt spent budget.
    """
    from fixtures.corpus import _cycle, _load_results
    
    write_result(a_result(4, attempt=0, exit_code=EXIT_INFRASTRUCTURE, minute=0), tmp_path)
    write_result(a_result(4, attempt=0, exit_code=EXIT_TASK_FAILED, minute=5), tmp_path)

    latest = latest_named(load_named(tmp_path))
    cycle = report(entry(4, label=READY))
    line = observed_line(
        observation_for(cycle, results={ref(4): latest[4][1]}),
        control_labels(cycle),
        result_names={ref(4): latest[4][0]},
    )

    replayed = _cycle(
        {**line, "cycle": 1},
        results=_load_results(tmp_path),
        default_run_id="run-1",
        where="inline",
    )

    assert [(one.attempt, one.exit_code) for one in replayed.observation.results] == [
        (0, EXIT_TASK_FAILED)
    ]


def test_the_recorder_writes_a_line_per_cycle_and_the_manifest_once(tmp_path):
    run = Run.start(REPO, "record", run_id=RUN_ID)
    artifacts = RunArtifacts.open(run, root=tmp_path)

    artifacts.observed({"cycle": 0})
    artifacts.observed({"cycle": 1})

    lines = (artifacts.path / OBSERVED_LOG_NAME).read_text().strip().splitlines()
    manifest = json.loads((artifacts.path / CORPUS_MANIFEST_NAME).read_text())

    assert [json.loads(line)["cycle"] for line in lines] == [0, 1]
    assert manifest["origin"] == "recorded"
    assert manifest["expected_divergences"] == []


def test_a_recorded_line_carries_the_label_not_the_internal_state():
    """The corpus's own decision, and its reason is good: the day epic #140
    removes the labels it is the translation that gets deleted, not the data."""
    cycle = report(entry(4, label=REVIEW))
    line = observed_line(observation_for(cycle), control_labels(cycle))

    assert line["control"] == {"task-4": REVIEW}
    assert line["tasks"] == [
        {
            "ref": "#4",
            "task_id": "task-4",
            "depends_on": [],
            "closed": False,
            "state_reason": None,
        }
    ]


def test_a_label_the_loader_could_not_translate_is_dropped_rather_than_written():
    """`load_corpus` refuses a label it cannot translate, correctly. A recorder
    able to emit one would produce corpus runs that fail at the door for a
    reason that has nothing to do with the resolver."""
    cycle = report(entry(4, label="swarm:something-else"))
    line = observed_line(observation_for(cycle), control_labels(cycle))

    assert line["control"] == {}


def test_the_recorder_never_raises_into_the_cycle(tmp_path, capsys):
    """A disk that will not take the line must cost the run its corpus, not its
    containers.

    The guard moved to `Reconciler._record_observed` when the window went: "have
    I already given up" is a question about this run, and a flag on a
    module-level function would be state two reconcilers shared. So this drives a
    real cycle rather than calling `record_cycle` directly - the guard is only
    worth having where it actually sits.
    """
    def explode(payload: Any) -> None:
        raise OSError("no space left on device")

    client, fleet, loop, _ = a_lifecycle_run()
    loop.record = explode

    loop.cycle()
    loop.cycle()

    err = capsys.readouterr().err
    assert "the run recorder failed" in err
    # Once per run, not once per cycle: a traceback every fifteen seconds buries
    # the one line that mattered.
    assert err.count("the run recorder failed") == 1
    assert loop._recorder_broken is True


# --------------------------------------------------------------------------
# 7. End to end, through the loop
# --------------------------------------------------------------------------








