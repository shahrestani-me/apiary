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
from pathlib import Path
from typing import Any

import pytest

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
from swarm.orchestrator.derived import REVIEW as REVIEW_STATE

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
    belief: Any = None,
    readiness: Any = None,
    dispatched: Any = None,
    checks: Any = None,
) -> CycleReport:
    """A finished `CycleReport` carrying nothing but a ledger.

    The recorder is a projection of a cycle that has already decided, so
    everything it reads is either on the report or was passed alongside it -
    which is what makes a report this bare a legitimate input rather than a
    stub with holes in it.

    `belief` is the one field #152 added and the reason a bare report is now
    *emptier* than it looks: the states used to be readable off
    `LedgerEntry.state_label`, so a projection could recover them from the ledger
    it already had. There is no label, so a report without a belief is a report
    nothing downstream can say what happened in - which is exactly what
    `control_labels` answers with below.
    """
    return CycleReport(
        index=index,
        ledger=ledger(*entries),
        result=ReconcileReport(plan=ReconcilePlan()),
        belief=belief,
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


# --------------------------------------------------------------------------
# 1. It costs no API call
# --------------------------------------------------------------------------
#
# Two banners stood here - "A divergence is named" and "The labels still decide"
# - over the tests that ran the resolver beside the label control plane and
# named what the two disagreed about. They went with the window (#245's split,
# #152's deletion), and the headings are collapsed into this note rather than
# left standing empty over a test about something else.


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
# 3. What fills the control slot now
# --------------------------------------------------------------------------
#
# **Four tests stood here and are deleted (#152).** `control_labels` assembled
# the `swarm:*` label each task wore when the cycle finished, walking five
# writers in the order the cycle wrote them, and each of those tests pinned one
# writer the cycle never folds back into its ledger:
#
# - `test_a_task_mergeability_escalated_is_in_the_control_map` - the fourth
#   writer of a terminal label. A pull request that will not rebase inside its
#   update budget is escalated by `apply_mergeability` alone.
# - `test_the_check_gate_wins_over_mergeability_for_the_same_task` - the gate
#   runs after mergeability, so the earlier writer must not overwrite the later.
# - `test_a_dispatch_that_claimed_and_failed_to_spawn_still_counts_as_claimed` -
#   `DispatchFailure.claimed` is "the label was written and no container is
#   running under it", the case #35's sweep exists for.
# - `test_a_revival_is_in_the_control_map` - the sixth writer, and the one that
#   is not in `cycle` at all: `planner.revive` runs from `_judge`.
#
# None of them can fail. `control_labels` reads `CycleReport.belief` and nothing
# else - there is no label plane to assemble, and assembling one from the
# writers would be inventing a second opinion for a comparison that no longer
# has two sides. The enumeration itself is not lost: `Reconciler._carry_forward`
# still walks the same writers, because a task the dispatcher claimed this cycle
# whose worker exits before the next one must not be remembered as never having
# run, and `tests/test_reconcile.py` is where that is now pinned.


def test_the_control_side_is_this_cycles_own_belief():
    """The module's central decision, with its subject replaced rather than lost.

    What a reader of a *new* recording wants from that slot is "what did the
    orchestrator think", and the belief is exactly that: honest as a record and
    worthless as a comparison. `docs/recording-runs.md` §2 says the second half
    in as many words - a run recorded from here on carries an empty `control` and
    can never be part of #147's divergence gate - and the field is kept rather
    than dropped because `observed.jsonl` is append-only and runs recorded
    *before* this ticket still hold a real control plane in it.
    """
    from swarm.orchestrator.authority import Belief

    believed = report(
        entry(4, label=REVIEW), belief=Belief(states={"task-4": REVIEW_STATE})
    )
    assert control_labels(believed) == {"task-4": REVIEW_STATE}

    # A task the cycle believed nothing about is left out rather than written as
    # `""`. `Belief.state` answers the empty string for a task nothing has an
    # opinion about, deliberately, and a recorder that wrote it would put a
    # state into the corpus that means "no state".
    blank = report(entry(4, label=REVIEW), belief=Belief(states={"task-4": ""}))
    assert control_labels(blank) == {}

    # And a report with no belief at all says nothing rather than guessing from
    # the ledger, because there is no `state_label` on it left to guess from.
    assert control_labels(report(entry(4, label=REVIEW))) == {}


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




# --------------------------------------------------------------------------
# 4. Expected against real - deleted with the classifier (#152)
# --------------------------------------------------------------------------
#
# This section held the two-sided classifier's tests, including the negative
# shapes: a divergence whose *control* side and evidence matched an expected kind
# while its *derived* side contradicted that kind's own argument. They graded a
# comparison between two sources of state and there is one, so the classifier and
# every test of it went with the window. `swarm show` still reads what those runs
# recorded, which is section 5.


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

    # Everything but `control`, which is asserted separately for a reason that
    # changed with #152 rather than going away. It used to be an asymmetry of
    # *vocabulary*: the recorder wrote the `swarm:*` label and `load_corpus`
    # translated it to the internal state on the way in, so a round trip would
    # have had to translate one of them back. It is now an asymmetry of
    # *content* - `observed_line` keeps only values that are labels, and the
    # cycle's belief holds none, so a run recorded from here on carries an empty
    # control plane. `docs/recording-runs.md` §2 says exactly that and draws the
    # consequence: such a run can never be part of #147's divergence gate.
    #
    # The key is still written, and the round trip below is why it has to be:
    # runs recorded *before* this ticket hold a real control plane in it, and a
    # loader that stopped parsing the field would make the archive unreadable.
    again = observed_line(replayed.observation, {})
    assert {k: v for k, v in again.items() if k != "control"} == {
        k: v for k, v in written[-1].items() if k != "control"
    }
    assert written[-1]["control"] == {}
    assert replayed.control == {}


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


def test_a_recorded_line_carries_a_world_and_an_empty_control_plane():
    """What a new recording is, stated once. **The day epic #140 removes the
    labels has arrived**, and this is the test that used to describe the other
    side of it.

    It was `test_a_recorded_line_carries_the_label_not_the_internal_state`, and
    the corpus's decision it pinned was a good one for as long as it had a
    subject: the recorder wrote `swarm:review`, `load_corpus` translated it, and
    so the day the labels went it was the translation that got deleted rather
    than the recorded data. `observed_line` keeps only values that are labels and
    the belief holds none, so the slot is empty now - which
    `docs/recording-runs.md` §2 documents as the price of the removal and not as
    a defect.

    `test_a_label_the_loader_could_not_translate_is_dropped_rather_than_written`
    stood beside it and is deleted rather than kept: it wrote `swarm:something-else`
    onto an entry and required the recorder to drop it, so that `load_corpus`
    could not be handed a label it refuses. Every value is dropped now, so the
    test passes without exercising the filter it was written for - and a test
    that cannot fail reads as coverage. The filter itself is what the first
    assertion below is about.

    The rest of the line is the half that carries all of the meaning now, and it
    is asserted here rather than taken on trust from the round trip: the world
    the cycle saw.
    """
    cycle = report(entry(4, label=REVIEW))
    line = observed_line(observation_for(cycle), control_labels(cycle))

    assert line["control"] == {}
    assert line["tasks"] == [
        {
            "ref": "#4",
            "task_id": "task-4",
            "depends_on": [],
            "closed": False,
            "state_reason": None,
        }
    ]


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








