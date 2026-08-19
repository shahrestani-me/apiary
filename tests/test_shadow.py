"""The derived-state shadow window (#146).

Five things this suite exists to hold down, in the order they would hurt:

1. **A divergence is named, not counted.** Both states, the task and the cycle,
   in every emitted event - #145's acceptance criteria refuse a percentage and
   the reason is in `derived.Divergence`'s docstring.
2. **The labels still decide.** A cycle whose derived state disagrees produces
   byte-identical decisions to one where the shadow never ran. This is the
   assertion #147 is not allowed to start without.
3. **It costs no API call.** Asserted by counting the calls a cycle makes with
   the shadow on and with it off, because a shadow that quietly doubled the
   rate-limit spend would be switched off by the first person it throttled.
4. **The expected divergences are told apart from the real ones.** One test per
   kind. A classifier that fired on the wrong shape would hide a real
   divergence inside an expected one, which is the worst thing this module can
   do and the only one that fails silently.
5. **A recorded cycle replays.** The recorder writes `observed.jsonl` and a
   manifest, `tests/fixtures/corpus.py` loads the directory with no change of
   its own, and the replay reproduces the divergences the live cycle reported.
   That round trip is what turns every real run into corpus, which is the
   cheapest retirement of the largest risk in epic #140.

The helpers come from `test_reconcile` rather than being rebuilt here. They are
the doubles that drive a real cycle end to end, and a second copy of them would
be a second thing to keep in step with the loop.
"""

from __future__ import annotations

import json
from dataclasses import replace
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
from swarm.orchestrator.reconcile import (
    CLAIMED,
    DONE,
    FAILED,
    READY,
    REVIEW,
    CycleReport,
    ReconcilePlan,
    ReconcileReport,
)
from swarm.orchestrator.shadow import (
    BUDGET_RENEWED,
    CLOSED_NOT_PLANNED,
    CONTAINER_CREATED,
    DISPATCHED_THIS_CYCLE,
    EXPECTED_KINDS,
    INFRASTRUCTURE_CEILING,
    MERGED_THIS_CYCLE,
    REVIVED,
    SHADOW_ENV,
    ShadowWindow,
    control_labels,
    observed_line,
    shadow_enabled,
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
from swarm.worker.result import write_result

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


def shadow(
    cycle: CycleReport, **facts: Any
) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    """Run the window over one report and hand back what it announced.

    `pulls={}` by default and never `None`: `None` is "this cycle could not
    list pull requests" and makes the window announce a blind cycle, which is
    the distinction `checks.read_pulls` exists to keep and which a test helper
    must not smuggle past.
    """
    facts.setdefault("pulls", {})
    seen, emit = recorder()
    window = ShadowWindow()
    return window.run(cycle, emit=emit, **facts), seen


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


def test_an_injected_divergence_names_both_states_the_task_and_the_cycle():
    """#146's headline. The label says claimed; nothing in the world agrees.

    The one test the ticket says matters: construct a cycle where derived and
    label state disagree and assert the event carries **both** values. A count
    would pass a run where every disagreement was on `needs-human`, which is
    the one state ADR 0001 reports outbound.
    """
    found, seen = shadow(report(entry(4, label=CLAIMED), index=7))

    assert [one.divergence.derived for one in found.explained] == ["eligible"]
    assert divergences(seen) == [
        {
            "cycle": 7,
            "task": "task-4",
            "derived": "eligible",
            "control": "claimed",
            "expected": False,
            "kind": "",
            "because": (
                "every dependency has landed, no container carries this task "
                "and no pull request is open for it"
            ),
            "why": "",
        }
    ]


def test_the_control_side_of_an_event_is_the_internal_state_never_the_label():
    """#141's rule, which this module inherits: `events.jsonl` is append-only
    and read back, so a payload carrying `swarm:claimed` would be invalidated
    the day #152 deletes the labels."""
    _, seen = shadow(report(entry(4, label=FAILED)))

    assert divergences(seen)[0]["control"] == "needs-human"
    assert "swarm:" not in json.dumps(seen)


def test_a_clean_cycle_still_says_it_ran():
    """The ambiguity `state.shadow` exists to remove.

    An event log with no `state.divergence` in it is either a clean shadowed
    run or a run with the flag off, and "ten consecutive runs with zero
    unexplained divergences" is not a gate that can tell those apart on its
    own.
    """
    found, seen = shadow(report(entry(4, label=READY)))

    assert found.explained == ()
    assert seen == [
        (
            STATE_SHADOW,
            {
                "cycle": 0,
                "tasks": 1,
                "independent": 1,
                "divergences": 0,
                "unexplained": 0,
                "blind": False,
            },
        )
    ]


def test_a_task_the_resolver_never_saw_is_not_a_divergence():
    """`derived.diverge`'s rule, carried through: the control plane holds work
    items this resolver was never shown, and calling those divergences would
    drown the ones that mean something."""
    found, _ = shadow(report(entry(4, label=READY)))

    assert found.tasks == 1
    assert found.control == {"task-4": "eligible"}


# --------------------------------------------------------------------------
# 2. The labels still decide
# --------------------------------------------------------------------------


def test_a_disagreeing_shadow_changes_no_decision(tmp_path):
    """The assertion #147 is not allowed to start without.

    Two runs of the same cycle over the same doubles, one shadowed and one not.
    Every write the cycle made, every transition it planned and every container
    it spawned must be identical - and the shadowed one must actually have
    disagreed, or this asserts nothing.
    """
    def once(*, enabled: bool) -> tuple[Any, list[str], list[int]]:
        client, fleet, loop, seen = a_lifecycle_run(label=CLAIMED)
        loop.artifacts = tmp_path
        loop._shadow = ShadowWindow(enabled=enabled)
        cycle = loop.cycle()
        return cycle, list(client.log), list(fleet.spawned)

    shadowed, shadowed_log, shadowed_spawns = once(enabled=True)
    plain, plain_log, plain_spawns = once(enabled=False)

    assert shadowed.summary() == plain.summary()
    assert shadowed_log == plain_log
    assert shadowed_spawns == plain_spawns
    # …and the shadow really did have something to say about this cycle.
    assert ShadowWindow().run(shadowed, pulls={}).explained


def test_shadowing_adds_no_github_call():
    """#146's fourth criterion. `Snapshot` already forced every read the window
    uses, so the number of client calls must not move."""
    def calls(*, enabled: bool) -> tuple[list[str], ShadowWindow, list]:
        client, fleet, loop, seen = a_lifecycle_run()
        loop._shadow = ShadowWindow(enabled=enabled)
        loop.cycle()
        return client.log, loop._shadow, seen

    on, window, seen = calls(enabled=True)
    off, _, _ = calls(enabled=False)

    assert on == off
    # …and the window really ran, rather than having broken and done nothing,
    # which would make the counts match for the wrong reason.
    assert window.broken is False
    assert [one for one in seen if one[0] == STATE_SHADOW]


def test_the_window_cannot_fail_the_cycle(capsys):
    """A shadow that raises is worse than no shadow, and this module says so in
    its own docstring - so the promise is tested rather than asserted in prose.

    A report with no ledger at all is the shape a rename five modules away
    would produce, and the correct outcome is a shadow that stops reporting and
    a cycle that never noticed.
    """
    window = ShadowWindow()
    broken = object()

    assert window.run(broken, pulls={}) is None  # type: ignore[arg-type]
    assert window.broken is True
    assert "derived-state shadow failed" in capsys.readouterr().err

    # And it stays off rather than printing a traceback every fifteen seconds.
    assert window.run(report(entry(4, label=CLAIMED)), pulls={}) is None
    assert capsys.readouterr().err == ""


def test_the_warning_is_once_per_run(capsys):
    """A standing divergence repeats every cycle until something moves, and a
    warning per cycle trains an operator to ignore the one line that matters."""
    window = ShadowWindow()
    window.run(report(entry(4, label=CLAIMED), index=0), pulls={})
    first = capsys.readouterr().err
    window.run(report(entry(4, label=CLAIMED), index=1), pulls={})

    assert "derived state disagrees" in first
    assert capsys.readouterr().err == ""


def test_the_flag_defaults_on_and_garbage_reads_as_the_default(monkeypatch):
    """Deliberately *not* `checks._env_flag`'s loud-on-garbage behaviour: a
    `ValueError` out of a mistyped variable would be an observer taking down a
    cycle, which is the one thing this module promises it cannot do."""
    assert shadow_enabled({}) is True
    assert shadow_enabled({SHADOW_ENV: "0"}) is False
    assert shadow_enabled({SHADOW_ENV: "off"}) is False
    assert shadow_enabled({SHADOW_ENV: "yse"}) is True

    monkeypatch.setenv(SHADOW_ENV, "no")
    assert shadow_enabled() is False


def test_a_window_that_is_off_announces_nothing():
    found, seen = shadow(report(entry(4, label=CLAIMED)))
    assert found is not None and seen

    seen_off, emit = recorder()
    off = ShadowWindow(enabled=False)
    assert off.run(report(entry(4, label=CLAIMED)), pulls={}, emit=emit) is None
    assert seen_off == []


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
                    from_label=REVIEW,
                    to_label=FAILED,
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

    def moved(to_label: str) -> Any:
        return Transition(
            ref=ref(4),
            task_id="task-4",
            from_label=REVIEW,
            to_label=to_label,
            reason="moved",
        )

    both = replace(
        report(entry(4, label=DONE), checks=ChecksReport(
            plan=ChecksPlan(), applied=(moved(DONE),)
        )),
        mergeability=MergeabilityReport(plan=MergeabilityPlan(), applied=(moved(FAILED),)),
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


def test_a_cycle_that_could_not_list_pull_requests_is_blind_not_clean(tmp_path):
    """`None` is not `{}`, and the distinction decides a whole review queue.

    An empty mapping read as the answer resolves every task in review to
    `eligible` and emits one manufactured unexplained divergence per review
    task, straight into the number the epic's go/no-go reads. Announced as
    blind instead: unmeasured, which is true.
    """
    seen, emit = recorder()
    window = ShadowWindow()

    assert window.run(report(entry(4, label=REVIEW)), pulls=None, emit=emit) is None
    assert [name for name, _ in seen] == [STATE_SHADOW]
    assert seen[0][1]["blind"] is True
    assert DivergenceTally.from_events(
        [{"event": name, **fields} for name, fields in seen]
    ).compared == 0


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


def test_two_containers_under_one_task_reach_the_resolver(tmp_path):
    """The collapse `Reconciler._handles` performs is first-wins, so an exited
    container listed ahead of a running one made the resolver read not-claimed
    - and a genuine double-spawn, which `dispatcher.release` is written about
    and #146 names as a reason to shadow, was structurally invisible."""
    found, _ = shadow(
        report(entry(4, label=CLAIMED)),
        containers=[handle(4, state="exited"), handle(4, state=RUNNING_STATE)],
    )

    assert found.explained == ()
    assert len(found.resolution.verdicts) == 1


def test_the_env_flag_reaches_a_reconciler(monkeypatch):
    """`shadow_enabled` is unit-tested; this is the wiring that reads it - a
    `default_factory` on the dataclass, which is the part that could silently
    stop being called."""
    monkeypatch.setenv(SHADOW_ENV, "0")
    _, _, off, seen = a_lifecycle_run()
    off.cycle()

    assert off._shadow.enabled is False
    assert [one for one in seen if one[0].startswith("state.")] == []

    monkeypatch.setenv(SHADOW_ENV, "1")
    _, _, on, seen_on = a_lifecycle_run()
    on.cycle()

    assert on._shadow.enabled is True
    assert [one for one in seen_on if one[0] == STATE_SHADOW]


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


def test_the_infrastructure_ceiling_is_expected_and_says_why():
    """ADR 0001's first non-derivable state. `infrastructure_streaks` counts
    transitions and exit 2 does not bump the attempt, so N mechanical failures
    write one result filename and the artifacts cannot tell one from three."""
    found, seen = shadow(
        report(entry(4, label=FAILED)),
        infrastructure={ref(4): 3},
        infrastructure_cap=3,
    )

    assert [one.kind for one in found.explained] == [INFRASTRUCTURE_CEILING]
    assert found.unexplained == ()
    assert divergences(seen)[0]["expected"] is True
    assert "exit 2 does not bump the attempt" in divergences(seen)[0]["why"]


def test_a_streak_below_the_cap_is_not_the_ceiling():
    """The classifier tests for the *evidence* of its account, never for the
    pair of states that account would produce. A task at one mechanical failure
    is failed for some other reason, and that reason is news."""
    found, _ = shadow(
        report(entry(4, label=FAILED)), infrastructure={ref(4): 1}, infrastructure_cap=3
    )

    assert [one.kind for one in found.explained] == [""]


def test_a_renewed_budget_is_expected_and_reads_the_store_not_the_states():
    """ADR 0001's second. `_retry_or_give_up` gives up on `streak`, not
    `attempt`, and the renewal is an ADR 0002 store judgment that no branch,
    container or result can see."""
    found, _ = shadow(
        report(entry(4, label=READY, streak=1)),
        pulls={"apiary/%234-attempt-3": pull(11, "apiary/%234-attempt-3")},
        max_attempts=3,
    )

    assert [(one.divergence.derived, one.kind) for one in found.explained] == [
        ("needs-human", BUDGET_RENEWED)
    ]


def test_a_revival_is_expected_and_is_told_apart_from_a_renewal():
    """ADR 0001's third. `planner.revive` "deliberately resets nothing", so the
    counter reads spent while the label reads ready - and the store shows no
    renewal, which is what distinguishes it from the row above."""
    found, _ = shadow(
        report(entry(4, label=READY)),
        pulls={"apiary/%234-attempt-3": pull(11, "apiary/%234-attempt-3")},
        max_attempts=3,
    )

    assert [(one.divergence.derived, one.kind) for one in found.explained] == [
        ("needs-human", REVIVED)
    ]


def test_a_merge_this_cycle_is_the_cycle_outrunning_its_own_read():
    """The fourth kind, and the one the sampling decision creates.

    The merge gate merged a pull request that was open when the world was
    sampled. Not a disagreement about a fact - it converges on the next cycle,
    when the issue reads closed.
    """
    from swarm.orchestrator.checks import ChecksPlan, ChecksReport

    merged = report(
        entry(4, label=DONE),
        checks=ChecksReport(plan=ChecksPlan(), merged=(4,)),
    )
    found, _ = shadow(
        merged, pulls={"apiary/%234-attempt-0": pull(11, "apiary/%234-attempt-0")}
    )

    assert [(one.divergence.derived, one.kind) for one in found.explained] == [
        ("review", MERGED_THIS_CYCLE)
    ]


def test_a_claim_written_after_the_container_listing_is_expected():
    """The dispatcher claims and spawns at the *end* of a cycle, and the
    container listing was taken at the top of it. Converges next cycle."""
    from swarm.orchestrator.dispatcher import DispatchPlan, DispatchReport, Dispatched

    spawned = report(
        entry(4, label=READY),
        dispatched=DispatchReport(
            plan=DispatchPlan(),
            dispatched=(Dispatched(entry=entry(4, label=CLAIMED), handle=handle(4)),),
        ),
    )
    found, _ = shadow(spawned)

    assert [one.kind for one in found.explained] == [DISPATCHED_THIS_CYCLE]


def test_a_container_between_create_and_start_is_expected_and_named():
    """The window #187 did not close, decided explicitly (see the module
    docstring): this reads liveness, not existence, so `created` is not a claim
    - and that reading is reported as a kind rather than hidden."""
    found, _ = shadow(
        report(entry(4, label=CLAIMED)),
        containers=[handle(4, state=CREATED_STATE)],
    )

    assert [one.kind for one in found.explained] == [CONTAINER_CREATED]


def test_a_running_container_is_a_claim_and_an_exited_one_is_not():
    """#187's fact, which is the reason a shadow window could be run at all.

    Before `Handle.state`, `docker ps --all` held every task in `claimed` from
    the moment its worker exited until the reaper arrived - a spurious
    divergence on every task in every run.
    """
    live, _ = shadow(
        report(entry(4, label=CLAIMED)), containers=[handle(4, state=RUNNING_STATE)]
    )
    dead, _ = shadow(
        report(entry(4, label=CLAIMED)), containers=[handle(4, state="exited")]
    )

    assert live.explained == ()
    # An exited container with the label still on `claimed` is a real
    # divergence: it is runs 03 and 04 of the corpus, where derived state is
    # right and the label is the stale one.
    assert [one.kind for one in dead.explained] == [""]


# --- the negative shapes: an account must not absorb a wrong derived answer ---
#
# Each of these is a divergence whose *control* side and evidence match an
# expected kind while its *derived* side contradicts that kind's own argument.
# Before the classifier was two-sided every one of them was filed as expected
# and dropped out of `unexplained` - the number #147's gate reads.


def test_a_ceiling_over_a_derived_landed_is_not_the_ceiling():
    """The worst of them. The resolver says the work item is closed as
    completed while the control plane escalated it: a contradiction, not a
    ceiling the resolver cannot see."""
    done = replace(entry(4, label=FAILED), closed=True)
    found, _ = shadow(
        report(done),
        states={ref(4): IssueState(ref=ref(4), state="closed", state_reason="completed")},
        infrastructure={ref(4): 3},
        infrastructure_cap=3,
    )

    assert [(one.divergence.derived, one.kind) for one in found.explained] == [("landed", "")]


def test_a_dispatch_over_a_spent_counter_is_not_the_dispatch_window():
    """`needs-human` outranks `claimed` in `derived._verdict`, so the container
    the dispatcher spawned would not have changed the answer - the account is
    false and the divergence does not converge next cycle."""
    from swarm.orchestrator.dispatcher import DispatchPlan, DispatchReport, Dispatched

    spent = report(
        entry(4, label=READY),
        dispatched=DispatchReport(
            plan=DispatchPlan(),
            dispatched=(Dispatched(entry=entry(4, label=CLAIMED), handle=handle(4)),),
        ),
    )
    found, _ = shadow(
        replace(spent, ledger=ledger(entry(4, label=CLAIMED))),
        pulls={"apiary/%234-attempt-3": pull(11, "apiary/%234-attempt-3")},
        max_attempts=3,
    )

    # Unexplained, because the store records no renewal either: the account
    # for a spent counter against a live label is `budget-renewed` when the
    # store says so and nothing at all when it does not.
    assert [(one.divergence.derived, one.kind) for one in found.explained] == [
        ("needs-human", "")
    ]


def test_a_created_container_over_a_spent_counter_is_not_the_create_window():
    """Same shape, same reason: the container is irrelevant once the counter
    outranks it, so `container-created` would be an account of nothing."""
    found, _ = shadow(
        report(entry(4, label=CLAIMED)),
        containers=[handle(4, state=CREATED_STATE)],
        pulls={"apiary/%234-attempt-3": pull(11, "apiary/%234-attempt-3")},
        max_attempts=3,
    )

    assert [one.kind for one in found.explained] != [CONTAINER_CREATED]
    assert found.explained[0].divergence.derived == "needs-human"


def test_a_merge_over_a_derived_blocked_is_not_the_merge_window():
    """A task the merge gate merged whose resolver said `blocked` did not see
    the pull request that merged. That is news about the resolver."""
    from swarm.orchestrator.checks import ChecksPlan, ChecksReport

    merged = report(
        replace(entry(4, label=DONE), blocked_by=(ref(9),)),
        entry(9, label=READY),
        checks=ChecksReport(plan=ChecksPlan(), merged=(4,)),
    )
    found, _ = shadow(merged)

    kinds = {one.divergence.task_id: one.kind for one in found.explained}

    assert kinds["task-4"] == ""
    assert found.explained[0].divergence.derived == "blocked"


def test_a_work_item_closed_as_not_planned_no_longer_diverges_at_all():
    """The one classified kind that was a **gap in the resolver** rather than a
    limit of derivation, and #147 closed it.

    `TaskFact.state_reason` carried the fact all along and `derived.py` had no
    rule reading it; it does now (`TaskFact.abandoned`), so the two sides agree
    and there is nothing left to classify. Asserted as *no divergence* rather
    than as a different kind, because a rule that still fired would mean the
    resolver had merely learned a second way of being wrong.
    """
    # `closed=True` on the entry as well as in `states`, which is what a real
    # cycle hands both sides: the ledger reads `state="all"` and the snapshot is
    # the same listing. `TaskFact.closed` comes from the entry and
    # `TaskFact.state_reason` from the states map, so a fixture that set only
    # one of them was testing a world no cycle produces.
    found, seen = shadow(
        report(replace(entry(4, label=FAILED), closed=True)),
        states={ref(4): IssueState(ref=ref(4), state="closed", state_reason="not_planned")},
    )

    assert found.explained == ()
    assert divergences(seen) == []
    # And the classification is still reachable code: if `state_reasons` ever
    # stops reaching the observation the divergence comes back, and it should
    # come back *named* rather than as an unexplained one in the go/no-go count.
    assert CLOSED_NOT_PLANNED in EXPECTED_KINDS


def test_a_closed_completed_item_reads_landed_rather_than_diverging():
    """The other half of the same fact, and the reason `state_reasons` was
    added to `derived.observe`: without it every hand-finished dependency would
    have read `landed` and every abandoned one would have read `landed` too."""
    done = replace(entry(4, label=DONE), closed=True)
    completed, _ = shadow(
        report(done),
        states={ref(4): IssueState(ref=ref(4), state="closed", state_reason="completed")},
    )
    abandoned, _ = shadow(
        report(done),
        states={ref(4): IssueState(ref=ref(4), state="closed", state_reason="not_planned")},
    )

    assert completed.explained == ()
    # The same closed issue, closed differently, is a different lifecycle
    # state - which is the whole reason the reason is carried.
    # `needs-human` rather than `eligible` since #147 taught the resolver
    # `state_reason`; the label on this fixture is `swarm:done`, so the two
    # still disagree - a work item somebody closed as not planned is not a
    # completed one however it is labelled.
    assert [one.divergence.derived for one in abandoned.explained] == ["needs-human"]


# --------------------------------------------------------------------------
# 5. `swarm show`
# --------------------------------------------------------------------------


def test_show_reports_a_divergence_count_without_anybody_reading_the_jsonl(tmp_path):
    """#146's third criterion."""
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


def test_show_prints_the_tally_for_a_real_run(tmp_path):
    run = Run.start(REPO, "shadow a run", run_id=RUN_ID)
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


def test_a_recorded_cycle_replays_to_the_same_divergences(tmp_path):
    """The round trip, and the only assertion that proves the recorder records
    the observation the resolver actually saw rather than a plausible one.

    Compared against `ShadowWindow.last` - the report from the cycle that did
    the recording - rather than against a second `run()` afterwards. A second
    run has to be handed the facts again, and a version of this test that
    handed it none passed only because the fixture happened to have no
    containers and no pull requests at that moment: it compared two different
    observations while looking like it compared one. The fixture below now has
    both, so an empty one would fail.
    """
    from fixtures.corpus import load_corpus
    from swarm.github.branches import task_branch
    from swarm.orchestrator.derived import diverge, resolve

    run = Run.start(REPO, "replay a cycle", run_id=RUN_ID)
    artifacts = RunArtifacts.open(run, root=tmp_path)

    client, fleet, loop, _ = a_lifecycle_run()
    loop.artifacts = artifacts.results_dir
    loop.record = artifacts.observed
    live = ShadowWindow()
    loop._shadow = live

    loop.cycle()  # dispatch: a running container for this task
    # …and now a pull request and a result record, without clearing the
    # container, so the recorded cycle carries all three.
    client.open_pulls = ((TASK_PULL, task_branch(ref(TASK_ISSUE), 0)),)
    client.check_runs = {client.head_of(TASK_PULL): pending()}
    write_result(record(TASK_ISSUE, 0, attempt=0, reason="verified"), artifacts.results_dir)
    loop.cycle()
    reported = live.last

    replayed = load_corpus(artifacts.path).cycles[-1]
    again = diverge(resolve(replayed.observation), replayed.control)

    # The observation really did carry the three things a thin one would not.
    assert replayed.observation.containers and replayed.observation.pulls
    assert replayed.observation.results
    assert reported is not None and reported.cycle == replayed.index
    assert {one.key for one in again} == {one.divergence.key for one in reported.explained}


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
    from swarm.orchestrator.shadow import observation_for

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
    from swarm.orchestrator.shadow import observation_for

    cycle = report(entry(4, label="swarm:something-else"))
    line = observed_line(observation_for(cycle), control_labels(cycle))

    assert line["control"] == {}


def test_the_recorder_never_raises_into_the_cycle(tmp_path, capsys):
    """A disk that will not take the line must cost the run its corpus, not its
    containers."""
    def explode(payload: Any) -> None:
        raise OSError("no space left on device")

    window = ShadowWindow()
    assert window.run(report(entry(4, label=READY)), pulls={}, record=explode) is None
    assert "derived-state shadow failed" in capsys.readouterr().err


# --------------------------------------------------------------------------
# 7. End to end, through the loop
# --------------------------------------------------------------------------


def test_a_task_that_reaches_review_agrees_with_the_control_plane(tmp_path):
    """The shape a greenfield run is made of, and the shape the go/no-go
    counts. A worker finished, the label moved, and the two sides agree."""
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path
    loop.cycle()
    reaches_review(client, fleet, pending())
    write_result(record(TASK_ISSUE, 0, attempt=0, reason="verified"), tmp_path)
    loop.cycle()

    unexplained = [one for one in divergences(seen) if not one["kind"]]

    assert [one for one in seen if one[0] == STATE_SHADOW]
    assert unexplained == []


def test_a_whole_lifecycle_never_breaks_the_window(tmp_path):
    """The guard the `except Exception` makes necessary.

    Swallowing is the right behaviour and it has a cost: a genuine defect in
    this module reads as a shadow that quietly stopped reporting, which is
    silence in the direction of the bug. So the suite asserts `broken` rather
    than trusting the absence of a failure - and it asserts it after a cycle
    that has been through dispatch, a result, review, a check set and a merge,
    which is where the five other modules' records are actually touched.

    This is not hypothetical: the first version of this ticket read
    `PullState.number` as an `int`, #185 had just made it a `PullRef`, and the
    only symptom was a line on stderr.
    """
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path
    loop.cycle()
    reaches_review(client, fleet, pending())
    write_result(record(TASK_ISSUE, 0, attempt=0, reason="verified"), tmp_path)
    loop.cycle()
    client.check_runs = {client.head_of(TASK_PULL): green()}
    loop.cycle()

    assert loop._shadow.broken is False
    assert [one for one in seen if one[0] == STATE_SHADOW]


def test_a_failing_task_walked_to_escalation_never_breaks_the_window(tmp_path):
    """The other half of the same guard: the paths where a counter moves, a
    comment is refused and a terminal label is written."""
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path
    loop.max_attempts = 2
    for attempt in range(3):
        loop.cycle()
        write_result(record(TASK_ISSUE, 1, attempt=attempt), tmp_path)
        fleet.handles.clear()
        loop.cycle()

    assert loop._shadow.broken is False


def test_the_merge_cycle_reports_the_one_expected_divergence(tmp_path):
    """A task landing is the cycle acting after its own read, and the log says
    which kind that is rather than leaving a reader to work it out."""
    client, fleet, loop, seen = a_lifecycle_run()
    loop.artifacts = tmp_path
    loop.cycle()
    reaches_review(client, fleet, green())
    write_result(record(TASK_ISSUE, 0, attempt=0, reason="verified"), tmp_path)
    loop.cycle()

    # Two, and both are the cycle acting after its own read: the first cycle
    # claimed and spawned after the container listing, the second merged after
    # the pull-request listing. Neither is a disagreement about a fact, and
    # neither is unexplained.
    assert [one["kind"] for one in divergences(seen)] == [
        DISPATCHED_THIS_CYCLE,
        MERGED_THIS_CYCLE,
    ]
    assert {one["task"] for one in divergences(seen)} == {TASK_REF}
    assert all(one["expected"] for one in divergences(seen))
