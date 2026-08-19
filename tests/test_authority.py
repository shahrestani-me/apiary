"""The cutover (#147): the resolver decides, and the flag takes it back.

Seven things this suite exists to hold down, in the order they would hurt.

1. **A label edited mid-run changes nothing.** The observable proof the cutover
   happened, and #147's own acceptance criterion. Driven through
   `Reconciler.cycle` rather than asserted on a plan, because "what the
   orchestrator does" is containers and merges, not a dataclass.
2. **`APIARY_STATE_SOURCE=labels` restores the old behaviour completely.** The
   load-bearing criterion, and the reason this may ship before #146's
   ten-clean-run gate was ever met. Asserted as the *inverse* of (1) over five
   of the six labels, so a decision path that quietly stopped consulting the
   authority fails here even if it looks right under the default. The merge gate
   has its own arm, because "for the scheduler but not the merge gate" is the
   shape an incomplete escape hatch takes.
3. **The two edge-triggered rules survive.** A failed worker's result is still
   observed and a rejected pull request still costs an attempt. Both break
   silently under a naive flip - the resolver reads `eligible` in each case -
   and both would have taken the retry engine with them.
4. **ADR 0001's three non-derivable states still work.** The infrastructure
   ceiling, a renewed budget and a revival: each is a case where the resolver is
   *wrong* and apiary's own record is right, and each has one test.
5. **A cycle that cannot see does not act on a guess.** `read_pulls` answering
   `None` falls back to the labels wholesale, because believing an empty pull
   request listing would dispatch a second worker over an open one's files.
6. **A container that exists still blocks a dispatch.** `derived.py` reads
   liveness and is right to; a dispatcher deciding whether to *act* has to read
   existence, which is the one thing the label carried for free.
7. **The criterion is larger than #147's file list.** The stale-claim sweep, the
   goal gate and the replan brief were left reading `entry.state_label` because
   that list did not name them - and the first of the three answered a
   hand-edited label by *spending a retry*, which is the one thing #147's
   acceptance criterion says in as many words must not happen.

The doubles come from `test_reconcile`, for `test_shadow`'s reason: they are
what drive a real cycle end to end, and a second copy would be a second thing to
keep in step with the loop.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from swarm.artifacts import STATE_OVERRIDE, DivergenceTally
from swarm.github.ledger import load_ledger
from swarm.github.readiness import BLOCKED, READY, compute_readiness
from swarm.github.refs import pull_ref, task_ref as ref
from swarm.orchestrator.authority import (
    BUDGET_RENEWED,
    BUDGET_SPENT,
    DERIVED,
    INFRASTRUCTURE_CEILING,
    LABELS,
    LANDED_STANDS,
    REVIVED,
    STATE_SOURCE_ENV,
    UNRESOLVED,
    Belief,
    Grant,
    Remembered,
    believe,
    state_source,
)
from swarm.orchestrator.derived import (
    CLAIMED as CLAIMED_STATE,
)
from swarm.orchestrator.derived import (
    ELIGIBLE,
    LANDED,
    NEEDS_HUMAN,
    ContainerFact,
    PullFact,
    observe,
)
from swarm.orchestrator.derived import (
    REVIEW as REVIEW_STATE,
)
from swarm.orchestrator.derived import PullFact
from swarm.orchestrator.dispatcher import CLAIMED, REVIEW, Capacity, plan_dispatch
from swarm.orchestrator.goal import FAILED as GAVE_UP
from swarm.orchestrator.goal import IN_FLIGHT, abandoned, assess, live, shipped
from swarm.orchestrator.reconcile import DONE, FAILED, plan_reconcile
from swarm.orchestrator.recovery import plan_recovery
from swarm.orchestrator.replan import brief
from swarm.store import STORE_DIR_ENV
from swarm.worker.result import write_result

from test_goal import Says, met  # the goal gate's scripted oracle
from test_reconcile import (  # the doubles that drive a real cycle
    TASK_ISSUE,
    TASK_PULL,
    a_lifecycle_run,
    entry,
    ledger,
    record,
)
from test_replan import stalled  # a verdict that has already refused to be one


@pytest.fixture(autouse=True)
def store_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`test_reconcile.store_root`'s reason: an autouse fixture is per-module.

    Without it every `Reconciler` built here opens the *operator's* store at
    `.swarm/store` and writes test judgments into a real project's retry
    budgets. Nothing fails; the next real run simply believes something untrue
    about its own history.
    """
    root = tmp_path / "store"
    monkeypatch.setenv(STORE_DIR_ENV, str(root))
    return root


# --------------------------------------------------------------------------
# Building a belief by hand
# --------------------------------------------------------------------------


def world(*entries: Any, **facts: Any) -> Any:
    """One observation over these ledger entries and nothing else."""
    facts.setdefault("cycle", 0)
    return observe(entries=entries, **facts)


def belief(*entries: Any, observation: Any = None, **kwargs: Any) -> Belief:
    book = ledger(*entries)
    return believe(
        book,
        world(*entries) if observation is None else observation,
        **kwargs,
    )


def outcome(client: Any, fleet: Any) -> tuple[Any, ...]:
    """What the orchestrator *did*, as a value two runs can be compared on.

    Deliberately not the label-write log. The writes differ between two runs
    that decided identically, because a hand-edited issue is relabelled *from*
    the label it is wearing - which is the point of `Transition.from_label`
    staying the real label. What has to match is the containers, the merges and
    the state the issue ends the cycle in.
    """
    return (
        sorted(client.labels_on(TASK_ISSUE)),
        list(fleet.spawned),
        list(fleet.disposed),
        list(client.merges),
    )


def a_run(label: str, source: str, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any]:
    """One cycle the orchestrator has already seen, then a human edits a label.

    Two cycles, and the first one matters: `authority.believe`'s `remembered` falls back to
    the label for a task this process has never seen, so an edit made before the
    first cycle would be seeding rather than overriding. #147's criterion is
    about a label edited **mid-run**, and this is what mid-run means.
    """
    monkeypatch.setenv(STATE_SOURCE_ENV, source)
    client, fleet, loop, seen = a_lifecycle_run(label=READY)
    # No fleet for the first cycle, so it settles the task without dispatching
    # it and the comparison below is between two second cycles.
    loop.fleet = None
    loop.cycle()

    client.issues[TASK_ISSUE]["labels"] = [{"name": label}]
    loop.fleet = fleet
    loop.cycle()
    return client, fleet, seen


#: Every label that disagrees with a world saying `eligible`. `swarm:blocked` is
#: in here for (1) and out of (2), and the asymmetry is a finding rather than an
#: exemption: it is the one wrong label the label machine already repaired
#: itself, because readiness owns both waiting states and recomputes them from
#: the dependency graph every cycle. The other four it obeyed.
WRONG_LABELS = (BLOCKED, CLAIMED, REVIEW, DONE, FAILED)
#: `swarm:review` leaves this list for `swarm:blocked`'s reason - it is a fact
#: about the old behaviour, not a hole in the flag. Pre-#147 rule 4 read
#: `entry.state_label` directly and charged an attempt for a `swarm:review`
#: whose branch was not in the open listing ("its pull request was closed
#: without merging"). So the old machine did not *hold* a review label either;
#: it read one with no pull request as a rejection and recycled the task. It
#: cannot discriminate here, and the behaviour it does have is pinned by the
#: dedicated test below.
OBEYED_LABELS = (CLAIMED, DONE, FAILED)


# --------------------------------------------------------------------------
# 1. A label edited mid-run changes nothing
# --------------------------------------------------------------------------


def test_a_hand_edited_label_does_not_change_what_the_orchestrator_does(monkeypatch):
    """#147's headline, and the one a user notices.

    A human moves a ready task to `swarm:claimed` while the run is going. Under
    the labels that stranded it: the dispatcher read `claimed`, called it in
    flight, and reserved its files until somebody moved the label back. Under
    the resolver nothing in the world says claimed - no container carries the
    ref and no pull request is open - so the task is dispatched exactly as it
    would have been, and the label is repaired on the way past.
    """
    client, fleet, _ = a_run(CLAIMED, DERIVED, monkeypatch)

    assert fleet.spawned == [TASK_ISSUE]
    assert client.labels_on(TASK_ISSUE) == {CLAIMED}


def test_a_hand_edited_label_is_reported_even_when_the_cycle_repairs_it(monkeypatch):
    """"...and the divergence is reported", which is the other half of the
    criterion and the half that needs its own event.

    The shadow window compares at the *end* of a cycle, and by then readiness
    has relabelled this issue - so a run whose labels a human edited would show
    zero `state.divergence` lines and look exactly like a run where nothing
    happened. `state.override` is sampled before anything is decided, which is
    the only sampling point at which "the label said claimed and we acted on
    eligible" is still true.
    """
    _, _, seen = a_run(CLAIMED, DERIVED, monkeypatch)

    overrides = [fields for name, fields in seen if name == STATE_OVERRIDE]
    assert overrides == [
        {
            "cycle": 1,
            "task": f"task-{TASK_ISSUE}",
            "believed": ELIGIBLE,
            "stored": CLAIMED_STATE,
            "derived": ELIGIBLE,
            "kind": "",
            "why": (
                "every dependency has landed, no container carries this task "
                "and no pull request is open for it"
            ),
        }
    ]
    # And `swarm show` counts it, so a run's own artifacts say how often the
    # labels were wrong without anybody reading the jsonl.
    tally = DivergenceTally.from_events(
        [{"event": name, **fields} for name, fields in seen]
    )
    assert tally.overrides == 1
    assert tally.override_tasks == (f"task-{TASK_ISSUE}",)
    assert "state override" in tally.text()


@pytest.mark.parametrize("label", WRONG_LABELS)
def test_no_wrong_label_changes_a_decision_under_the_resolver(label, monkeypatch):
    """The completeness test, and the one that fails if a path forgets.

    Five of the six labels, each edited onto a task whose world plainly says
    `eligible`, each asserted to produce the decisions of a run whose label was
    never touched. Readiness, dispatch and reconcile all run in every arm, so a
    decision path that still read `entry.state_label` would have to agree with
    the resolver on all five to hide here - and `swarm:done` and `swarm:review`
    disagree with `eligible` in different modules.
    """
    baseline = a_run(READY, DERIVED, monkeypatch)
    edited = a_run(label, DERIVED, monkeypatch)

    assert outcome(*edited[:2]) == outcome(*baseline[:2])


# --------------------------------------------------------------------------
# 2. The flag restores the previous behaviour completely
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label", OBEYED_LABELS)
def test_the_flag_puts_every_one_of_those_decisions_back(label, monkeypatch):
    """The inverse of the test above, and the criterion #147 calls load-bearing.

    apiary develops itself on this control plane, so a cutover with no way back
    blocks its own repair. "Restores the previous behaviour" is only worth
    anything if it is *complete*, so this asserts the label is obeyed again in
    every arm rather than in one - if `APIARY_STATE_SOURCE=labels` reached the
    scheduler but not the merge gate, one of these would still look like the
    resolver's answer.

    `swarm:blocked` is not in the list, and that is a fact about the old
    behaviour rather than a hole in the new one: readiness owns both waiting
    states and recomputed them from the dependency graph every cycle, so the
    label machine already repaired that one by itself.
    """
    baseline = a_run(READY, LABELS, monkeypatch)
    edited = a_run(label, LABELS, monkeypatch)

    assert outcome(*edited[:2]) != outcome(*baseline[:2])
    # And the specific old behaviour, not merely "something differs": the label
    # says this task is in flight or finished, so nothing spawns.
    assert edited[1].spawned == []
    assert edited[0].labels_on(TASK_ISSUE) == {label}


def test_under_labels_a_review_label_with_no_pull_request_is_charged(monkeypatch):
    """`swarm:review`'s arm of the criterion above, which is not "nothing happens".

    Pre-#147, `plan_reconcile` rule 4 read `entry.state_label` on every cycle:

        if entry.state_label != REVIEW: continue
        if open_branches is not None and entry.branch not in open_branches:
            _retry_or_give_up(entry, "its pull request was closed without merging")

    So a human writing `swarm:review` onto a task with no open pull request was
    told its pull request had been closed, and paid an attempt for it. That is
    the behaviour this flag restores - obeying the label, not holding it.

    Carrying the derived path's `remembered` overlay onto the labels path hid
    this: `was` came from last cycle's belief rather than from the label, rule 4
    never matched, and the task sat in `swarm:review` doing nothing. That looked
    more like "the label is obeyed", and was in fact the cutover leaking through
    the hatch meant to switch it off.
    """
    client, fleet, _ = a_run(REVIEW, LABELS, monkeypatch)

    # Charged and recycled, exactly as before #147 - not held, not ignored.
    assert client.labels_on(TASK_ISSUE) != {REVIEW}
    assert fleet.spawned == [TASK_ISSUE]


def test_the_flag_silences_the_override_event_too(monkeypatch):
    """Nothing is overridden when nothing is believed but the label, and the
    log says so by having no line rather than by having a line saying zero."""
    _, _, seen = a_run(CLAIMED, LABELS, monkeypatch)

    assert [name for name, _ in seen if name == STATE_OVERRIDE] == []


def test_a_mistyped_state_source_stops_the_run(monkeypatch):
    """Loud on garbage, unlike `APIARY_DERIVED_SHADOW`.

    That flag reads a typo as its default and argues the case: it decides
    whether an *observer* runs. This one decides who the orchestrator believes,
    and an operator who typed it to get back to the old behaviour after a bad
    cutover must not silently get the new one.
    """
    monkeypatch.setenv(STATE_SOURCE_ENV, "lables")
    with pytest.raises(ValueError, match="APIARY_STATE_SOURCE"):
        state_source()

    monkeypatch.setenv(STATE_SOURCE_ENV, "")
    assert state_source() == DERIVED


@pytest.mark.parametrize("source,merged", [(DERIVED, [900]), (LABELS, [])])
def test_the_merge_gate_follows_the_authority_too(source, merged, monkeypatch):
    """"Not for the scheduler but not the merge gate" is the failure mode.

    #147 names three files, and the acceptance criterion it is really about is
    larger than the file list: a label a human edits mid-run must not change
    *what the orchestrator does*, and merging is the most consequential thing it
    does. `plan_checks` and `run_mergeability` selected on `swarm:review`, so a
    task somebody relabelled `swarm:ready` while its green pull request sat open
    was silently un-mergeable. They ask `authority.in_review` now.

    Both directions are asserted, because a gate that merged under the flag as
    well would mean the escape hatch does not reach it either.
    """
    monkeypatch.setenv(STATE_SOURCE_ENV, source)
    from swarm.github.branches import task_branch

    from test_reconcile import green, pending

    client, fleet, loop, _ = a_lifecycle_run(label=READY)
    loop.cycle()

    # The worker published, and its check run is still going - so the task is
    # genuinely in review and the gate has nothing to decide yet.
    client.issues[TASK_ISSUE]["labels"] = [{"name": REVIEW}]
    client.open_pulls = ((900, task_branch(ref(TASK_ISSUE), 0)),)
    client.check_runs = {client.head_of(900): pending()}
    fleet.handles.clear()
    loop.cycle()
    assert client.merges == []

    # The checks go green, and a human relabels the issue in the same interval.
    # The pull request is still open, so the world still plainly says review.
    client.check_runs = {client.head_of(900): green()}
    client.issues[TASK_ISSUE]["labels"] = [{"name": READY}]
    loop.cycle()

    assert client.merges == merged


# --------------------------------------------------------------------------
# 3. The two edge-triggered rules
# --------------------------------------------------------------------------


def test_a_failed_worker_is_still_observed_when_its_container_has_gone(tmp_path, monkeypatch):
    """The rule a naive cutover kills, and it kills it silently.

    A worker that fails leaves an exited container and no pull request, so the
    resolver says `eligible` - correctly. A reconciler that waited for `claimed`
    would never observe a failed attempt again: no retry comment, no counter, no
    give-up, and a task re-dispatched from scratch every cycle with nothing
    counting. `plan_reconcile` asks what the task *was* instead, which is the
    job the label was quietly doing.
    """
    monkeypatch.setenv(STATE_SOURCE_ENV, DERIVED)
    client, fleet, loop, _ = a_lifecycle_run(label=READY)
    loop.artifacts = tmp_path

    loop.cycle()
    # The worker ran and failed. Its container is gone, which is what makes the
    # resolver read `eligible` and what used to make this rule unreachable.
    fleet.handles.clear()
    write_result(record(TASK_ISSUE, 1, attempt=0, reason="the tests failed"), tmp_path)
    loop.cycle()

    # The counter moved, which is the whole of the retry engine: without this
    # the task is re-dispatched forever and `max_attempts` bounds nothing. The
    # label ends the cycle at `swarm:claimed` rather than `swarm:ready` because
    # the same cycle re-dispatched the retry, which is what it did before #147
    # too.
    assert "attempt=1" in client.issues[TASK_ISSUE]["body"]
    assert fleet.spawned == [TASK_ISSUE, TASK_ISSUE]


def test_a_pull_request_closed_unmerged_still_costs_an_attempt(monkeypatch):
    """The second edge, and the reason the first fix is not enough.

    `Snapshot` lists open pull requests only, so a closed one is invisible and
    the resolver says `eligible` here too. A reconciler that waited for `review`
    would forgive every rejected pull request - "a retry that costs nothing can
    be rejected forever", which is what that rule exists to prevent.
    """
    monkeypatch.setenv(STATE_SOURCE_ENV, DERIVED)
    client, fleet, loop, _ = a_lifecycle_run(label=READY)

    loop.cycle()
    # The worker published, so the task is genuinely in review...
    from swarm.github.branches import task_branch

    # The merge gate is off, so the pull request is left where it is rather
    # than escalated for having no check run: this test's subject is the rule
    # that reads a pull request *disappearing*, not the one that gates it.
    loop.merge_gate = False
    client.issues[TASK_ISSUE]["labels"] = [{"name": REVIEW}]
    client.open_pulls = ((900, task_branch(ref(TASK_ISSUE), 0)),)
    fleet.handles.clear()
    loop.cycle()

    assert "attempt=0" in client.issues[TASK_ISSUE]["body"]

    # ...and then a human closed the pull request without merging it. Nothing in
    # the world records that this ever happened - `Snapshot` lists open pull
    # requests only - so the whole of the evidence is what the orchestrator
    # believed last cycle.
    client.open_pulls = ()
    loop.cycle()

    assert "attempt=1" in client.issues[TASK_ISSUE]["body"]


@pytest.mark.parametrize("source", [DERIVED, LABELS])
def test_the_infrastructure_streak_counts_the_same_under_both_sources(
    source, tmp_path, monkeypatch
):
    """The self-clearing half, which is why `previous` rather than a container.

    An infrastructure verdict deliberately does not bump the counter, so the
    same result file stays "unaccounted for" by the arithmetic and the rule that
    reads it has to be edge-triggered or it fires every cycle. The label stopped
    it by moving to `swarm:ready`; the carried belief does the same job, and a
    rule keyed on a leftover container instead would have escalated a task whose
    disposal the daemon happened to refuse.

    Asserted as *the same number under both sources* rather than as a number,
    because the number itself is `_observe`'s business and this test's subject is
    only that the cutover did not change how often it is reached.
    """
    monkeypatch.setenv(STATE_SOURCE_ENV, source)
    client, fleet, loop, _ = a_lifecycle_run(label=READY)
    loop.artifacts = tmp_path

    loop.cycle()
    write_result(record(TASK_ISSUE, 2, attempt=0, reason="docker: no such image"), tmp_path)
    loop.cycle()

    assert loop._infrastructure.get(ref(TASK_ISSUE), 0) == 1
    assert FAILED not in client.labels_on(TASK_ISSUE)
    assert "attempt=0" in client.issues[TASK_ISSUE]["body"]


# --------------------------------------------------------------------------
# 4. ADR 0001's three, which stayed non-derivable
# --------------------------------------------------------------------------


def test_the_infrastructure_ceiling_still_escalates_though_nothing_can_see_it():
    """ADR 0001's first non-derivable state, and the resolver reads `eligible`.

    Exit 2 does not bump the attempt, so N mechanical failures write one result
    filename and the artifacts cannot tell one from three. The orchestrator
    keeps believing its own counter; making the resolver authoritative did not
    make this derivable, and a cutover that pretended otherwise would retry a
    broken host forever.
    """
    task = entry(4)
    held = belief(task, infrastructure={ref(4): 3}, infrastructure_cap=3)

    assert held.state("task-4") == NEEDS_HUMAN
    assert [one.kind for one in held.overrides] == [INFRASTRUCTURE_CEILING]
    assert "3 consecutive infrastructure verdict(s)" in held.overrides[0].why
    # And it does not outrank a merge: a task whose pull request landed is done
    # with, whatever the host was doing on the way.
    landed = belief(
        replace(task, closed=True),
        infrastructure={ref(4): 3},
        infrastructure_cap=3,
    )
    assert landed.state("task-4") == LANDED


def test_a_renewed_budget_is_read_from_the_store_and_not_from_the_branches():
    """ADR 0001's second, and ADR 0002's whole subject.

    `_retry_or_give_up` gives up on `streak`, not on `attempt`, and a renewal is
    a store judgment no branch, container or result can see. So the resolver's
    `needs-human` - arithmetic over code-host evidence - is advisory, and the
    store decides. Without this a renewed task is escalated on the very
    arithmetic the renewal exists to overrule.
    """
    task = entry(4, attempt=4, streak=1)
    spent = replace(task, streak=3)
    observation = world(task, pulls=(PullFact(number=pull_ref(900), ref=ref(4), attempt=4),))

    renewed = believe(ledger(task), observation, max_attempts=3)
    assert renewed.state("task-4") == REVIEW_STATE
    assert [one.kind for one in renewed.overrides] == [BUDGET_RENEWED]

    # The same world with the store saying the streak really did reach the cap.
    given_up = believe(
        ledger(spent),
        world(spent, pulls=(PullFact(number=pull_ref(900), ref=ref(4), attempt=4),)),
        max_attempts=3,
    )
    assert given_up.state("task-4") == NEEDS_HUMAN


def test_a_task_apiary_gave_up_on_is_not_resurrected_by_a_restart():
    """The other direction of the same rule, and the one that is not obvious.

    A failed task leaves **no code-host evidence at all** once its process is
    gone: results live in the run directory and a run directory is per run, and
    `build_observation` takes branch names off *open* pull requests because a
    remote branch listing is a call no cycle makes. So the resolver reads
    `eligible` from scratch, and under the labels `swarm:failed` was what carried
    the verdict across the restart. The store carries it now.
    """
    task = entry(4, label=FAILED, attempt=3, streak=3)
    held = belief(task, max_attempts=3)

    assert held.state("task-4") == NEEDS_HUMAN
    # Recorded even though it agrees with the label it restores, because "the
    # store said so" and "the world said so" are different claims and this is
    # the overlay whose absence would be silent.
    assert [one.kind for one in held.overrides] == [BUDGET_SPENT]
    assert held.overrides[0].derived == ELIGIBLE

    # The counterfactual, so the assertion above is not passing for some other
    # reason: with nothing in the store the same world resurrects the task.
    forgotten = belief(entry(4, label=FAILED), max_attempts=3)
    assert forgotten.state("task-4") == ELIGIBLE


def test_a_total_cap_give_up_also_survives_a_restart():
    """The branch the test above does not reach, and the one a restart takes.

    `_retry_or_give_up` gives up two ways: the streak reaching `max_attempts`,
    and the counter reaching `max_total_attempts`. The test above covers the
    first, and it survives because `entry.streak` is stored.

    The second is the one renewals produce, and it looked identical to a healthy
    task: every new failure *signature* resets the streak to 1, so a task that
    failed nine times with nine different blockers ends at `attempt=9,
    streak=1`. Testing only the code host's count - which is zero in a fresh
    process, because results are per-run and branch names come off open pull
    requests - left that task reading `eligible`, relabelled `swarm:ready`, and
    dispatched with a whole new budget over work apiary had already abandoned.
    """
    exhausted = entry(4, label=FAILED, attempt=9, streak=1)
    held = believe(ledger(exhausted), world(exhausted), max_attempts=3, max_total_attempts=9)

    assert held.state("task-4") == NEEDS_HUMAN
    assert [one.kind for one in held.overrides] == [BUDGET_SPENT]
    # The resolver really did read it as runnable - this is the overlay working,
    # not the world happening to agree.
    assert held.overrides[0].derived == ELIGIBLE

    # The counterfactual: one attempt below the cap, same low streak, and the
    # task is genuinely runnable. Without it this test would pass against a rule
    # that simply never resurrects anything.
    below = entry(4, label=FAILED, attempt=8, streak=1)
    assert believe(
        ledger(below), world(below), max_attempts=3, max_total_attempts=9
    ).state("task-4") == ELIGIBLE


def test_under_labels_this_cycles_label_beats_last_cycles_belief():
    """`APIARY_STATE_SOURCE=labels` restores the *label read*, not a memory of it.

    Before #147, `plan_reconcile`'s rules read `entry.state_label` directly on
    every cycle. Carrying the derived path's `remembered` overlay onto the
    labels path made last cycle's belief win, and the one event that tells the
    two apart is a human editing a label mid-run - which is the single case this
    flag exists for, and the action apiary's own give-up comment instructs.

    Left in, rule 4 fires on the remembered `review` for a task a human has just
    moved back to `swarm:ready`, consuming an attempt and posting a failure for
    work that was rescheduled rather than failed.
    """
    moved = entry(4, label=READY)
    held = believe(
        ledger(moved),
        None,
        source=LABELS,
        remembered={"task-4": REVIEW_STATE},
    )

    assert held.source == LABELS
    assert held.previous["task-4"] == ELIGIBLE, "last cycle's belief overrode the label"
    assert held.state("task-4") == ELIGIBLE


def test_a_store_that_has_never_judged_a_task_gives_up_sooner_not_later():
    """ADR 0002 names simplifying this fallback to `0` as the change that would
    open the hole it thought it had closed, so it is pinned here too.

    A missing judgment falls back to the attempt counter, which is the largest
    streak consistent with it - absence escalates, it never grants a budget.
    """
    never_judged = entry(4, label=FAILED, attempt=3)  # streak is None
    assert belief(never_judged, max_attempts=3).state("task-4") == NEEDS_HUMAN

    # And a task the counter says is young is left alone, so the fallback is a
    # fallback rather than a blanket escalation.
    young = entry(4, attempt=1)
    assert belief(young, max_attempts=3).state("task-4") == ELIGIBLE


def test_a_revival_grants_exactly_one_attempt_and_then_lapses():
    """ADR 0001's third. `planner.revive` "deliberately resets nothing", so a
    revived task reads spent from every code-host source there is.

    Under the labels it was `swarm:ready` and the dispatcher believed that.
    Under the resolver it would be re-escalated the instant it was revived, and
    the goal gate could never unstick a run - so the grant is recorded as what
    it is, and it lapses the moment the attempt it granted is spent.
    """
    # `swarm:failed` on the entry and three attempts the code host can see, so
    # the resolver says `needs-human` on its own arithmetic - which is exactly
    # what a revived task looks like from the outside.
    task = entry(4, label=FAILED, attempt=3, streak=3)
    spent_world = world(task, results=(record_fact(4, attempt=2),))

    escalated = believe(ledger(task), spent_world, max_attempts=3)
    assert escalated.state("task-4") == NEEDS_HUMAN

    # `Grant(attempt=3)` was a bare `3` before #200, which is the same grant in
    # the shape that predates the lapse. Only the call changes here; what this
    # test asserts about a revival that produces a result does not.
    granted = believe(
        ledger(task), spent_world, max_attempts=3, revived={ref(4): Grant(attempt=3)}
    )
    assert granted.state("task-4") == ELIGIBLE
    assert [one.kind for one in granted.overrides] == [REVIVED]

    # The granted attempt runs and fails: the result carries attempt 3, so the
    # counter the code host can account for has moved past the revival and the
    # streak `planner.revive` deliberately did not reset caps the task again.
    lapsed = believe(
        ledger(task),
        world(task, results=(record_fact(4, attempt=3),)),
        max_attempts=3,
        revived={ref(4): Grant(attempt=3)},
    )
    assert lapsed.state("task-4") == NEEDS_HUMAN


def test_a_revival_whose_attempt_left_no_result_lapses_on_the_dispatch():
    """#200. The grant's other ending, and the one nothing on the code host says.

    A granted attempt killed at `SWARM_WORKER_TIMEOUT`, or whose container was
    reaped mid-cycle, writes **no result record and opens no pull request**. So
    `attempts_spent` sits exactly where it was when the grant was made, the test
    above ("the result carried the granted attempt") never fires, and the grant
    suppressed the give-up for the rest of the run - `needs-human` rewritten
    into a lenient state every cycle, and the task dispatched every cycle,
    indefinitely.

    Every other bound is inert on the same input, which is what makes it a trap
    rather than a leak: `entry.attempt` moves only through `_retry_or_give_up`
    and the infrastructure streak only through a transition, and both need the
    artifact this failure is defined by not producing.

    So the lapse is keyed on the dispatch. Same entry, same world, same code-host
    count in both halves below - the *only* difference is apiary's own record of
    having put a worker on it.
    """
    task = entry(4, label=FAILED, attempt=3, streak=3)

    # Two worlds, because a revival reaches this module from two directions and
    # the overlay that fires differs. Both are the *same* code-host evidence
    # before and after the killed attempt: it wrote nothing, so nothing moved.
    #
    # (a) This run remembers the failures that got the task here - the result
    #     from attempt 2 is still in the run directory, so the resolver reads
    #     `needs-human` on its own arithmetic and the grant is what overrides it.
    remembers = world(task, results=(record_fact(4, attempt=2),))
    # (b) The goal gate revives across a restart: results are per run, so the
    #     new process sees no attempt at all and the resolver reads `eligible`.
    #     Here the grant is invisible and `budget-spent` is what has to bite.
    resumed = world(task)

    for unmoved in (remembers, resumed):
        outstanding = believe(
            ledger(task), unmoved, max_attempts=3, revived={ref(4): Grant(attempt=3)}
        )
        assert outstanding.state("task-4") == ELIGIBLE, "the grant is still buying its attempt"

        spent = believe(
            ledger(task),
            unmoved,
            max_attempts=3,
            revived={ref(4): Grant(attempt=3, dispatched=True)},
        )
        assert spent.state("task-4") == NEEDS_HUMAN

    # And where the resolver would have said `eligible`, the event log says
    # *which* of the grant's two endings this was - "the code host accounts for
    # 0 attempt(s)" on its own reads like the bug rather than the fix.
    lapsed = believe(
        ledger(task), resumed, max_attempts=3, revived={ref(4): Grant(attempt=3, dispatched=True)}
    )
    assert [one.kind for one in lapsed.overrides] == [BUDGET_SPENT]
    assert "has lapsed" in lapsed.overrides[0].why

    # The grant is spent once and stays spent: `Grant.spend` is idempotent, so a
    # cycle that re-reports the same dispatch cannot un-lapse it.
    assert Grant(attempt=3).spend().spend() == Grant(attempt=3, dispatched=True)


def record_fact(issue: int, *, attempt: int, exit_code: int = 1) -> Any:
    from swarm.orchestrator.derived import AttemptFact

    return AttemptFact(ref=ref(issue), attempt=attempt, exit_code=exit_code)


def test_a_task_this_run_has_seen_land_is_never_dispatched_again():
    """`landed` is terminal within a run, and the world stops showing it.

    A merged pull request is not in `Snapshot`'s open listing, so once the merge
    has happened the only remaining evidence is the issue being closed as
    completed - and there are two ordinary ways for that evidence not to be
    there. `checks._decide_passed` writes `swarm:done` *before* GitHub has
    honoured `Closes #<n>`, so on the landing cycle the pull request has already
    left the listing while the issue still reads open; and a human who reopens a
    finished issue takes the evidence away for good.

    Under the labels `swarm:done` was terminal and neither mattered. Under the
    resolver alone both read `eligible`, and the dispatcher puts a worker back
    on work that is already on the default branch.
    """
    merged = entry(4, label=DONE)
    world_after_the_merge = world(merged)  # no open pull request, issue not closed

    # Both routes to the same answer, because they are different failures. The
    # first is this run remembering its own merge; the second is a fresh process
    # with nothing but the label, which is the restart case and the reason the
    # seed exists at all.
    #
    # The remembered value is a `Remembered` rather than a bare `"landed"`, and
    # that is not ceremony: since #201 only a state this process actually
    # *believed*, about this very work item, seeds the ratchet, and a real cycle
    # carries exactly this - `believe` mints them and `Belief.fold`/`hold` carry
    # them through `Reconciler._carry_forward`. The bare string is what the
    # `UNRESOLVED` label fallback leaves behind, and it deliberately no longer
    # counts; `test_a_label_the_resolver_had_no_verdict_for_does_not_pin_a_task`
    # is that half.
    held = believe(
        ledger(merged),
        world_after_the_merge,
        remembered={"task-4": Remembered(LANDED, ref(4))},
    )
    resumed = believe(ledger(merged), world_after_the_merge)

    assert held.state("task-4") == LANDED
    assert resumed.state("task-4") == LANDED
    assert [one.kind for one in held.overrides] == [LANDED_STANDS]
    # The resolver's own verdict is recorded on the override, which is what
    # makes this assertion about a *rescue* rather than about agreement.
    assert held.overrides[0].derived == ELIGIBLE

    plan = plan_dispatch(
        ledger(merged),
        capacity=Capacity(slots=3, configured=2),
        ready=(ref(4),),
        believed=held,
    )
    assert plan.numbers == ()


def test_the_previous_belief_is_seeded_from_the_label_only_for_a_task_never_seen():
    """The seed is the one place a label still reaches a decision, and it is
    bounded to tasks this process has no memory of.

    That bound is what makes #147's criterion hold: a label edited mid-run
    belongs to a task the orchestrator has already seen, so it is never seeded
    and the edit changes nothing.
    """
    task = entry(4, label=REVIEW)
    fresh = believe(ledger(task), world(task))
    assert fresh.previous == {"task-4": REVIEW_STATE}

    remembered = believe(ledger(task), world(task), remembered={"task-4": CLAIMED_STATE})
    assert remembered.previous == {"task-4": CLAIMED_STATE}


# --------------------------------------------------------------------------
# 5. A cycle that cannot see
# --------------------------------------------------------------------------


def test_a_cycle_that_could_not_list_pull_requests_falls_back_to_the_labels():
    """`None` is "could not look" and it is not `{}`.

    `checks.read_pulls` and `shadow.ShadowWindow.run` both keep the two apart,
    and the cost of conflating them is one level worse here than it is in the
    shadow: an empty listing read as the answer resolves every task in review to
    `eligible`, and the dispatcher then spawns a second worker over an open pull
    request's file set.
    """
    task = entry(4, label=REVIEW)
    blind = believe(ledger(task), None)

    assert blind.source == LABELS
    assert blind.state("task-4") == REVIEW_STATE
    assert blind.overrides == ()


def test_a_task_the_resolver_never_saw_keeps_its_label_and_is_counted():
    """A fallback nobody can see is a cutover that looks clean by not happening.

    `Resolution.state` answers `""` for a task it has no verdict for, which is
    "nothing was said" rather than a state. The label stands, and the fallback
    is reported as an override of its own so that a run where the resolver saw
    nothing does not read as a run where it agreed with everything.
    """
    task = entry(4)
    held = believe(ledger(task), world())  # an observation with no tasks in it

    assert held.state("task-4") == ELIGIBLE
    assert [one.kind for one in held.overrides] == [UNRESOLVED]


# --------------------------------------------------------------------------
# 6. Liveness decides nothing; existence blocks
# --------------------------------------------------------------------------


def test_a_container_that_exists_but_has_exited_still_blocks_a_dispatch():
    """`shadow.py` argues the distinction and hands this ticket the bill.

    ADR 0001's `claimed` is a *live* worker, so a container that exited without
    writing a result reads as no claim at all - true, and safe only while
    nothing decided on it. `dispatcher.release` takes the opposite reading
    because it is deciding whether to *act*, and "existence blocks" is the only
    reading that cannot produce two workers over one file set. Dispatch is the
    other half of that decision.
    """
    task = entry(4)
    held = belief(task)
    assert held.state("task-4") == ELIGIBLE

    plan = plan_dispatch(
        ledger(task),
        capacity=Capacity(slots=3, configured=2),
        ready=(ref(4),),
        believed=held,
        holding=(ref(4),),
    )

    assert plan.numbers == ()
    assert [one.number for one in plan.in_flight] == [4]


def test_a_live_container_is_a_claim_and_reserves_its_files():
    """The ordinary path, so the test above is not the only shape asserted."""
    task = entry(4)
    held = belief(
        task,
        observation=world(
            task,
            containers=(ContainerFact(id="c", run_id="", ref=ref(4), running=True),),
        ),
    )

    assert held.state("task-4") == CLAIMED_STATE
    plan = plan_dispatch(
        ledger(task), capacity=Capacity(slots=3, configured=2), believed=held
    )
    assert plan.numbers == ()
    assert [one.number for one in plan.in_flight] == [4]


# --------------------------------------------------------------------------
# The readers, one at a time
# --------------------------------------------------------------------------


def test_readiness_speaks_about_the_entries_the_authority_says_are_waiting():
    """The one thing readiness used a label for. Everything else it decides is a
    question about the code host and is untouched."""
    task = entry(4, label=CLAIMED)  # the world says eligible; a human said claimed

    labelled = compute_readiness(ledger(task), {})
    derived = compute_readiness(ledger(task), {}, transitionable={"task-4"})

    assert labelled.verdicts == ()
    assert [verdict.label for verdict in derived.verdicts] == [READY]
    # `current_label` is still the label really on the issue, because it is the
    # one `_relabel` has to remove - a verdict built on the believed state would
    # leave the issue carrying two.
    assert derived.verdicts[0].current_label == CLAIMED


def test_reconcile_reads_terminal_from_the_authority_and_from_label():
    """A human marks a running task done. The container's fate is the decision.

    Under the labels `swarm:done` is terminal and the worker is disposed
    mid-attempt. Under the resolver a live container is a claim whatever the
    label says, so the worker is left alone - which is the same rule from the
    other side: GitHub wins when a human *closes* an issue, and a label is not a
    closure.
    """
    task = entry(4, label=DONE)
    live = world(task, containers=(ContainerFact(id="c", run_id="", ref=ref(4), running=True),))

    labelled = plan_reconcile(ledger(task), running=[ref(4)])
    assert [one.ref for one in labelled.disposals] == [ref(4)]

    # `remembered`, because this is the mid-run case: the orchestrator has seen
    # this task claimed and a human has since typed `swarm:done` on it. Without
    # a memory the label is the only record there is and the seed reads it -
    # which is the restart case and a different question (see the `landed`
    # ratchet above).
    held = believe(ledger(task), live, remembered={"task-4": CLAIMED_STATE})
    assert held.state("task-4") == CLAIMED_STATE
    assert plan_reconcile(ledger(task), running=[ref(4)], believed=held).disposals == ()


def test_a_belief_advances_by_the_writes_that_landed_and_by_nothing_else():
    """`fold`'s rule, one level along. A belief advanced by a planned write
    GitHub refused would hand the dispatcher a view of a world that never
    existed - which is the failure `fold` exists to prevent, doubled."""
    task = entry(4)
    held = belief(task)
    assert held.state("task-4") == ELIGIBLE

    class Applied:
        task_id = "task-4"
        to_label = CLAIMED

    assert held.fold([Applied()]).state("task-4") == CLAIMED_STATE
    assert held.fold([]).state("task-4") == ELIGIBLE
    # And `waiting()` follows it, so the next stage of the cycle asks readiness
    # about the entries this one left waiting rather than the ones it found.
    assert held.waiting() == {"task-4"}
    assert held.fold([Applied()]).waiting() == frozenset()


# --------------------------------------------------------------------------
# 7. The three modules #147's file list did not name
# --------------------------------------------------------------------------
#
# #147 enumerated three files and a merge gate, and the criterion it wrote down
# is larger than that list: *a label a human edits mid-run must not change what
# the orchestrator does*. Three modules were left reading `entry.state_label`
# because the list did not name them, and each of them decides something the
# criterion is plainly about - a retry budget, a run's exit code, and the words
# a model is shown. Each gets the same pair below: the edit changes nothing
# under `derived`, and changes the answer under `labels`.


def a_hand_edited(label: str, was: str, **facts: Any) -> tuple[Any, Belief, Belief]:
    """One task wearing `label`, a world that disagrees, and both beliefs.

    The two beliefs come from **one** ledger and **one** observation, which is
    what makes each pair below a comparison of who is believed rather than of
    two different runs - `shadow.py`'s rule, and `believe`'s own.

    `was` is what this process believed last cycle, and it is not decoration:
    `believe` seeds `Belief.previous` from the label for a task it has never
    seen, so an edit made without it is *seeding* rather than overriding. #147's
    criterion is about a label edited **mid-run**, and this is what mid-run
    means - the same point `a_run` makes one cycle at a time for the tests that
    drive a whole loop. The `labels` arm ignores it deliberately (see
    `believe`), which is the behaviour the hatch is restoring.
    """
    task = entry(4, label=label)
    book = ledger(task)
    seen = world(task, **facts)
    return (
        book,
        believe(book, seen, remembered={task.task_id: was}),
        believe(book, seen, source=LABELS),
    )


def test_a_claimed_label_typed_onto_a_ready_task_no_longer_burns_an_attempt():
    """`recovery.py`, and the sharpest form of the criterion in this package.

    Releasing a stale claim **consumes an attempt** (`recovery._release`), so
    before this the sweep answered a label a human typed by spending a retry off
    a task that had never run - and, at the cap, by escalating it to a human. It
    is the one place a mid-run edit cost budget rather than a cycle.

    Under the resolver a claim is a *running container*, there is none, and the
    entry is not the sweep's to speak about at all.
    """
    book, derived, labels = a_hand_edited(CLAIMED, ELIGIBLE)

    swept = plan_recovery(book, containers=(), believed=derived)
    assert swept.transitions == ()
    # Not silently ignored either: nothing here holds it, because nothing here
    # selected it. The task is `eligible`, and the dispatcher picks it up.
    assert swept.held == ()
    assert derived.state("task-4") == ELIGIBLE

    obeyed = plan_recovery(book, containers=(), believed=labels)
    assert [str(one) for one in obeyed.transitions] == [
        "#4: swarm:claimed -> swarm:ready, attempt 1 "
        "(claimed with no live container behind it)"
    ]
    # The budget, which is the half a transition's `str` does not show and the
    # only half a user feels.
    assert [one.attempt for one in obeyed.transitions] == [1]

    # And `believed=None` - `Recovery.startup`, the `__main__` dry run - is the
    # labels arm exactly, which is what "every existing caller is unchanged"
    # has to mean for a module whose other entry point runs before any belief
    # exists.
    assert plan_recovery(book, containers=()).transitions == obeyed.transitions


def test_a_failed_label_typed_onto_merged_work_no_longer_resigns_the_run():
    """`goal.py`: the gate partitions the ledger into done / failed / live.

    A `swarm:failed` typed onto a task whose pull request merged put it on the
    wrong side of that partition, and the wrong side is not cosmetic here: the
    abandoned arithmetic is a *refusal*, so the run ended asking a human about a
    task that had landed, with the objective never assessed at all.
    """
    book, derived, labels = a_hand_edited(
        FAILED,
        REVIEW_STATE,
        pulls=(PullFact(number=pull_ref(TASK_PULL), ref=ref(4), merged=True),),
    )
    assert derived.state("task-4") == LANDED

    assert [one.task_id for one in shipped(book, derived)] == ["task-4"]
    assert abandoned(book, derived) == ()
    assert live(book, derived) == ()

    oracle = Says(met())
    assert assess("ship it", book, oracle=oracle, believed=derived).met
    # The model was shown the merged task as shipped, which is the whole of what
    # the assessment is computed from.
    assert "task-4" in oracle.asked[0][1][1]

    resigned = assess("ship it", book, oracle=Says(met()), believed=labels)
    assert not resigned.met
    assert resigned.reason == GAVE_UP
    assert resigned.abandoned == (ref(4),)


def test_a_done_label_typed_onto_an_open_pull_request_no_longer_assesses_early():
    """`goal.py`'s other side: `live` is what stops the gate judging mid-run.

    The refusal exists because "an objective assessed against a half-landed run"
    is an answer nobody can act on. A `swarm:done` a human typed made the ledger
    read exhausted, and the gate assessed - and could declare the run met - over
    a task whose pull request was still open.
    """
    book, derived, labels = a_hand_edited(
        DONE, CLAIMED_STATE, pulls=(PullFact(number=pull_ref(TASK_PULL), ref=ref(4)),)
    )
    assert derived.state("task-4") == REVIEW_STATE

    assert [one.task_id for one in live(book, derived)] == ["task-4"]
    assert shipped(book, derived) == ()

    held = assess("ship it", book, oracle=Says(met()), believed=derived)
    assert not held.met
    assert held.reason.startswith(IN_FLIGHT)
    assert not held.consulted, "no model is worth swapping in for a half-landed run"

    assert assess("ship it", book, oracle=Says(met()), believed=labels).met


def test_the_replan_brief_names_a_task_in_the_runs_own_vocabulary():
    """`replan.py`, and the one call site here that is not a decision.

    Nothing branches on this string; it is a word in a prompt. It is in scope
    for the same epic all the same - `swarm:*` is a storage detail #141 and #140
    are deleting, and a prompt is the worst place to leave one, because the
    model reads the vocabulary as the run's own and re-emits it.

    The mid-run edit is here too, for the same reason it is everywhere else in
    this file: a `swarm:failed` typed onto merged work described that task to
    the planner as abandoned, and `REPLAN_SUFFIX` asks the model to re-emit
    every id it is shown.
    """
    book, derived, labels = a_hand_edited(
        FAILED,
        REVIEW_STATE,
        pulls=(PullFact(number=pull_ref(TASK_PULL), ref=ref(4), merged=True),),
    )
    verdict = stalled()

    _, tracked = brief(book, verdict, derived)
    assert "task-4 (landed):" in tracked
    assert "swarm:" not in tracked

    _, under_the_labels = brief(book, verdict, labels)
    assert "task-4 (needs-human):" in under_the_labels

    # `believed=None` prints the label verbatim rather than translating it: this
    # is a string in a prompt, so "unchanged" has to mean the same characters.
    _, unchanged = brief(book, verdict)
    assert "task-4 (swarm:failed):" in unchanged


# --------------------------------------------------------------------------
# 8. The ratchet has no lapse, so its entry points are its whole safety (#201)
# --------------------------------------------------------------------------
#
# `landed-stands` is the only overlay above that re-tests nothing: its input is
# its own previous output, carried through `Reconciler._carry_forward` and read
# back as `remembered`. That is defensible on "a merge cannot be taken back",
# and it makes every *entry* into `landed` a permanent, silent decision - so
# every test here drives the **silent** direction. A task wrongly pinned
# `landed` produces no transition, no comment and no event; it is simply never
# dispatched and never escalated, and a run ends reporting work as shipped that
# nothing ever ran. Asserting that an ordinary merge still ratchets proves none
# of that, which is why each test below ends on a dispatch plan.


def dispatchable(book: Any, held: Belief, *refs: Any) -> tuple[int, ...]:
    """Which of these tasks the dispatcher would actually put on the fleet.

    The assertion every test in this section ends on. `Belief.state` is what the
    bug is *about*, but a state nobody acts on is not a defect - the failure
    being pinned down here is a worker spawned over merged code, or a task that
    silently never runs again, and only the plan says which happened.
    """
    return plan_dispatch(
        book,
        capacity=Capacity(slots=3, configured=3),
        ready=refs,
        believed=held,
    ).numbers


def test_a_label_the_resolver_had_no_verdict_for_never_pins_a_task():
    """The `UNRESOLVED` arm may decide a cycle; it may not decide the run.

    That arm writes `entry.state_label` into the belief because there is nothing
    else to write - "the resolver said nothing" is not an opinion, and the label
    is the only record left. Harmless for one cycle. The defect was that the
    value was then carried forward like any other and could not be told from a
    verdict, so a `swarm:done` on a task the resolver has no opinion about
    entered the ratchet and pinned it for the life of the process: **a label
    reaching a decision nothing can undo**, which is the exact seam #147 closed
    everywhere else.

    Driven the way it actually happens rather than by handing `believe` a map,
    and **mid-run**, which is what makes it a different fact from the first-sight
    seed below: three cycles, a human typing `swarm:done` onto a live task in the
    second, and a resolver that has no verdict for it on the cycle that matters.
    """
    task = entry(4, label=READY)
    book = ledger(task)

    # Cycle one: an ordinary task this process has now seen.
    seen = believe(book, world(task))
    assert seen.state("task-4") == ELIGIBLE

    # Cycle two: a human types `swarm:done` onto it, and this is one of the
    # cycles the observation has nothing to say about the task - a stack with no
    # image, a listing that came back without it. The label stands in, which is
    # the documented fallback and is harmless for one cycle.
    typed = ledger(entry(4, label=DONE))
    silent = believe(typed, world(), remembered=dict(seen.states))
    assert silent.state("task-4") == LANDED
    assert [one.kind for one in silent.overrides] == [UNRESOLVED]

    # Cycle three: the resolver has an opinion again, and it is `eligible`. The
    # label never became a belief, so there is nothing for the ratchet to stand
    # on and the task runs.
    after = believe(typed, world(task), remembered=dict(silent.states))

    assert after.state("task-4") == ELIGIBLE
    # One override, and it is the plain kind: the label is stale, which is
    # ordinary and is reported. What must not be here is `landed-stands`.
    assert [one.kind for one in after.overrides] == [""]
    assert dispatchable(typed, after, ref(4)) == (4,)


def test_a_cycle_with_no_verdict_does_not_drop_a_ratchet_it_did_not_test():
    """The other direction of the same arm, and the one that would be a
    regression rather than a fix.

    "The resolver returned nothing this cycle" is not evidence that a merge was
    undone. If a silent cycle overwrote the belief with the label, the ratchet
    would be at the mercy of whatever `swarm:*` string the issue happens to be
    wearing on the next cycle the resolver *does* answer - which is the label
    deciding again, one step further along.
    """
    merged = entry(4, label=DONE)
    book = ledger(merged)

    landed = believe(book, world(merged), remembered={"task-4": Remembered(LANDED, ref(4))})
    assert landed.state("task-4") == LANDED

    silent = believe(book, world(), remembered=dict(landed.states))
    resumed = believe(book, world(merged), remembered=dict(silent.states))

    assert silent.state("task-4") == LANDED
    assert resumed.state("task-4") == LANDED
    assert dispatchable(book, resumed, ref(4)) == ()


def adopted_issue(number: int, *, title: str, label: str = READY) -> dict[str, Any]:
    """A hand-written issue: a legal contract body with **no identity marker**.

    The marker is what makes an id stable, so an issue without one is the only
    kind whose id `ledger._adopted_id` has to invent - and inventing it is where
    the reuse below comes from.
    """
    return {
        "number": number,
        "title": title,
        "state": "open",
        "labels": [{"name": label}],
        "body": (
            "## Goal\nDo the thing.\n\n"
            f"## Files\n- src/mod{number}.py\n\n"
            "## Verify\npython -m pytest -q\n\n"
            "## Blocked by\n_none._\n"
        ),
    }


class Tracker:
    """The one call `load_ledger` makes when it is not allowed to write."""

    def __init__(self, *issues: dict[str, Any]) -> None:
        self.issues = list(issues)

    def list_issues(self, *, state: str = "open", **_: Any) -> list[dict[str, Any]]:
        return [dict(issue) for issue in self.issues]


def test_an_adopted_id_minted_for_another_issue_does_not_inherit_landed():
    """`remembered` is keyed by task id, and a task id is not a durable name.

    `ledger._adopted_id` derives a hand-written issue's id from its title and
    disambiguates collisions by *order*: the first taker gets the bare slug and
    later ones get `<slug>-<number>`. So the bare slug is not owned - it is
    leased, and it moves to the next issue in line the moment the holder leaves
    `Ledger.entries`. This drives that for real rather than asserting on a
    hand-built map: one cycle where `a-task` is #7, then a human breaks #7's
    contract, and the very same string now names #9.

    Without the ref carried alongside, #9 inherits #7's `landed` and is **never
    dispatched and never escalated** - the quietest failure in this module,
    because `landed` emits no transition and, once the ratchet is standing, no
    event either.
    """
    first = adopted_issue(7, title="Add a retry", label=DONE)
    second = adopted_issue(9, title="Add a retry")

    before = load_ledger(Tracker(first, second), adopt=False)
    assert sorted(before.entries) == ["add-a-retry", "add-a-retry-9"]
    assert before.entries["add-a-retry"].ref == ref(7)

    # A human edits #7 and drops a required section. §1.4's policy is that the
    # issue lands in `Ledger.errors` and never in `entries` - so the lease on
    # the bare slug is up, and #9 takes it.
    broken = dict(first, body="## Goal\nDo the thing.\n")
    after = load_ledger(Tracker(broken, second), adopt=False)
    assert after.entries["add-a-retry"].ref == ref(9)

    # What the previous cycle carried: #7, believed landed, under the id it held
    # at the time.
    carried = {"add-a-retry": Remembered(LANDED, ref(7))}
    now = believe(after, world(*after.entries.values()), remembered=carried)

    assert now.state("add-a-retry") == ELIGIBLE
    assert [one.kind for one in now.overrides] == []
    assert dispatchable(after, now, ref(9)) == (9,)

    # And the same memory still pins the task it is actually about, which is
    # what makes the assertion above about identity rather than about the
    # ratchet having been weakened.
    still = believe(before, world(*before.entries.values()), remembered=carried)
    assert still.state("add-a-retry") == LANDED
    assert dispatchable(before, still, ref(7), ref(9)) == (9,)


def test_a_revival_cannot_clear_a_merge():
    """`Reconciler._carry_forward` applies its revival overlay unconditionally.

    `{task: ELIGIBLE for task in revived_tasks(report)}` goes through
    `Belief.hold` after the fold, and before #201 it could silently clear the one
    fact the ratchet exists to make unundoable - after which the next cycle's
    resolver, which cannot see a merged pull request, reads `eligible` and a
    worker goes onto code that is already on the default branch.

    **Reachability, which #201 left open.** The goal gate is not a route: since
    #205 `goal._revive_abandoned` selects `abandoned(ledger, believed)`, which is
    `needs-human`, and `landed` never reads as that. The replan is: `_update`
    still selects its revival on `entry.state_label == FAILED` and takes no
    belief at all, so a `swarm:done` issue a human relabels `swarm:failed`
    mid-run - which the ratchet keeps believing landed, correctly - is revived by
    any replan that keeps the task. That selector is pinned below, because the
    guard here is only worth what the argument for needing it is worth.
    """
    from swarm.nodes.planner import _update

    assert "believed" not in inspect.signature(_update).parameters

    merged = entry(4, label=DONE)
    book = ledger(merged)
    held = believe(book, world(merged), remembered={"task-4": Remembered(LANDED, ref(4))})
    assert held.state("task-4") == LANDED

    revived = held.hold({"task-4": ELIGIBLE})

    assert revived.state("task-4") == LANDED
    # The label really did move - `planner.revive` writes `swarm:ready` - and the
    # belief saying otherwise is the disagreement, not a lost write.
    assert revived.stored["task-4"] == ELIGIBLE
    assert dispatchable(book, revived, ref(4)) == ()

    # And it survives the carry, which is the half that matters: a guard that
    # held for one cycle and was forgotten by the next would only move the
    # dispatch one cycle later.
    later = believe(book, world(merged), remembered=dict(revived.states))
    assert later.state("task-4") == LANDED
    assert dispatchable(book, later, ref(4)) == ()


def test_a_ratchet_that_keeps_standing_stops_announcing_itself():
    """One `state.override` for the fact, not one per cycle for the run's length.

    Every other kind here re-tests its input each cycle, so a repeat of one of
    them is news: the store was read again, the streak was counted again, a
    dispatch spent a grant. `landed-stands` re-tests nothing, so left alone it
    contributes one event per landed task per remaining cycle and
    `DivergenceTally.overrides` becomes a measure of how long the run lasted.
    That number is exactly what an operator reads as "the cutover is
    misbehaving", so the one signal that says the cutover is *working* would be
    indistinguishable from the alarm.
    """
    merged = entry(4, label=DONE)
    book = ledger(merged)

    cycles: list[Belief] = []
    carried: dict[str, str] = {"task-4": Remembered(LANDED, ref(4))}
    for _ in range(4):
        held = believe(book, world(merged), remembered=carried)
        cycles.append(held)
        carried = dict(held.states)

    assert [one.state("task-4") for one in cycles] == [LANDED] * 4
    # Announced once, on the cycle it started standing.
    assert [[one.kind for one in held.overrides] for held in cycles] == [
        [LANDED_STANDS],
        [],
        [],
        [],
    ]

    tally = DivergenceTally.from_events(
        [
            {"event": STATE_OVERRIDE, "task": one.task_id, "kind": one.kind}
            for held in cycles
            for one in held.overrides
        ]
    )
    assert tally.overrides == 1
    assert tally.override_kinds == ((LANDED_STANDS, 1),)


def test_the_first_sight_seed_is_the_label_and_the_contract_says_so():
    """The one entry point #201 deliberately **kept**, and its price, written down.

    A task this process has never carried is decided by the label, because on a
    fresh process the label is the only record of the last belief this system
    held. `swarm:done` on an open issue is produced by three different things and
    a restarted orchestrator cannot tell them apart - the window in which
    `checks._decide_passed` has written the label before GitHub honoured
    `Closes #<n>`, a pull request merged without the keyword at all, and a human
    reopening finished work. Two of the three must not be dispatched, so a seed
    that honoured the third would put a worker back on merged code in the other
    two.

    The cost falls on a human: an issue reopened *mid-run* to have the work
    redone stays pinned until the process restarts. That is the same fact as the
    relabel #147 already ignores, in a different field - but it is a fact
    somebody has to be told, so it is in `docs/issue-contract.md` §4 beside the
    human rows rather than only in a docstring. Asserted rather than promised,
    for `test_doctor`'s reason.
    """
    reopened = entry(4, label=DONE)
    book = ledger(reopened)

    fresh = believe(book, world(reopened))
    assert fresh.state("task-4") == LANDED
    assert [one.kind for one in fresh.overrides] == [LANDED_STANDS]
    assert dispatchable(book, fresh, ref(4)) == ()

    # The documented route back, and the reason the pin is not a dead end: the
    # first cycle is the one place the label still seeds, so a restart with the
    # issue already moved off `swarm:done` is honoured.
    moved = ledger(entry(4, label=READY))
    restarted = believe(moved, world(*moved.entries.values()))
    assert restarted.state("task-4") == ELIGIBLE
    assert dispatchable(moved, restarted, ref(4)) == (4,)

    contract = (Path(__file__).resolve().parents[1] / "docs" / "issue-contract.md").read_text()
    section = contract.split("## 4. The label state machine")[1].split("\n## 5.")[0]
    # Unwrapped, because a sentence's line breaks are a formatter's business and
    # a test that pinned them would fail on a reflow that changed nothing.
    prose = " ".join(section.split())
    assert "reopening the issue does not take it back" in prose
    assert "stays pinned until the process restarts" in prose
