"""The cutover (#147): the resolver decides, and the flag takes it back.

Seven things this suite exists to hold down, in the order they would hurt.

1. **A label a human types mid-run changes nothing.** The observable proof the
   cutover happened, and #147's own acceptance criterion. Since #152 it is
   structural rather than obeyed-and-overruled: no `swarm:*` label is written
   and none is read, so the edit never reaches a decision at all. Driven through
   `Reconciler.cycle` rather than asserted on a plan, because "what the
   orchestrator does" is containers and merges, not a dataclass.
2. **The merge gate reads the same authority.** "For the scheduler but not the
   merge gate" is the shape an incomplete cutover takes, and merging is the most
   consequential thing the loop does - so the gate gets an arm of its own rather
   than being taken on trust from (1).
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

**The cutover pair is gone (#152).** Most of what follows used to be written
twice - once under the resolver and once under `APIARY_STATE_SOURCE=labels` -
because the flag was the way back from #147 and the pair was what proved the
cutover complete. #152 deleted the flag, the label writes and
`LedgerEntry.state_label` together, so the second arm has nothing left to read:
`state_of` raises without a belief rather than falling back. Tests that were
*only* that comparison are deleted, each with a note in place saying what stood
there and why it can no longer fail. Tests that assert the resolver's own
behaviour keep the one arm that still exists.

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

#: ADR 0001's internal vocabulary, in the two spellings this file's fixtures
#: still name states by. They were `swarm:done` and `swarm:failed` until #152
#: turned a fixture's declared *label* into a declared *state*.
DONE = "landed"
FAILED = "needs-human"

from swarm.artifacts import STATE_OVERRIDE, DivergenceTally
from swarm.github.readiness import BLOCKED, READY, compute_readiness
from swarm.github.refs import pull_ref, task_ref as ref
from swarm.orchestrator.authority import (
    BUDGET_RENEWED,
    BUDGET_SPENT,
    INFRASTRUCTURE_CEILING,
    LANDED_STANDS,
    REVIVED,
    UNRESOLVED,
    Belief,
    Grant,
    believe,
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
from swarm.orchestrator.goal import IN_FLIGHT, abandoned, assess, live, shipped
from swarm.orchestrator.reconcile import Transition, plan_reconcile
from swarm.orchestrator.recovery import plan_recovery
from swarm.orchestrator.replan import brief
from swarm.store import STORE_DIR_ENV
from swarm.worker.result import write_result

from test_goal import Says, met  # the goal gate's scripted oracle
from test_reconcile import (  # the doubles that drive a real cycle
    OTHER_ISSUE,
    OTHER_PULL,
    TASK_ISSUE,
    TASK_PULL,
    a_lifecycle_run,
    entry,
    ledger,
    record,
)
from test_replan import stalled  # a verdict that has already refused to be one
from fixtures.belief import fixture_belief


# The cycle's belief, supplied from what each fixture declares (see
# `fixtures.belief`). It was read off `LedgerEntry.state_label` until #152.
def _plan_recovery_(book, *args, **kwargs):
    kwargs.setdefault("believed", fixture_belief(book))
    return plan_recovery(book, *args, **kwargs)

def _plan_dispatch_(book, *args, **kwargs):
    kwargs.setdefault("believed", fixture_belief(book))
    return plan_dispatch(book, *args, **kwargs)

# `compute_readiness` has no `believed` of its own: it is handed the task ids the
# authority says are waiting and what it believes them to be waiting in, which is
# the join `Belief` already did. There is no wrapper for it here for that reason
# - the two arguments are the subject of the test below rather than plumbing.


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

    Containers spawned, containers disposed, pull requests merged - the three
    things a user would notice, and deliberately nothing about labels. **The
    fourth slot went with #152**: it was `sorted(client.labels_on(TASK_ISSUE))`,
    which was the state the issue ended the cycle in, and it was worth comparing
    only while the loop still wrote labels and could therefore repair one. It
    writes none now, so that slot holds whatever the human typed and differs
    between the arms by construction - a comparison that could only ever fail.

    Two of the three are live only when the run has something to dispose and
    something to merge, which is what `a_run(alongside=True)` is for: over a
    one-task run `fleet.disposed` and `client.merges` are `[]` in every arm, and
    a comparison of two dead slots is a comparison of dispatch alone (#202).

    Making them live is necessary and was not sufficient (#228). Two of the
    arms that go red under a `plan_reconcile` regression only reach a disposal
    rule at all because the second task also carries the *result record* its
    worker wrote, so what this compares is `a_run(alongside=True,
    artifacts=...)`'s world rather than any two-task one.
    """
    return (
        list(fleet.spawned),
        list(fleet.disposed),
        list(client.merges),
    )


def a_run(
    label: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    alongside: bool = False,
    artifacts: Path | None = None,
) -> tuple[Any, Any, Any]:
    """One cycle the orchestrator has already seen, then a human edits a label.

    Two cycles, and the first one still matters, for a narrower reason than it
    once did. `believe` no longer seeds anything from a label, so an edit made
    before the first cycle cannot *seed* a belief - but a run that had never seen
    the task would be starting on the edited world rather than carrying on
    through it, and only the second is what #147 means by **mid-run**.

    `alongside` runs the same two cycles over `a_lifecycle_run`'s two-task
    world, and the edit moves **every** issue in the run onto `label` - one
    human relabelling a run rather than one issue. When the second cycle's
    `plan_reconcile` runs, the second task is the only one carrying a container
    - this run's own task is dispatched later in the same cycle - so it is the
    only issue whose hand-edited label a disposal rule has to read.

    `artifacts` is passed straight through to `a_lifecycle_run`, where it is
    the directory the second task's result record is written into. Only the
    two-task callers need it, and it is what makes `swarm:claimed` and
    `swarm:review` reachable rules rather than dead ones - see there.

    **What the loop buys, measured rather than argued.** Two variants, each run
    with `plan_reconcile` reverted to `entry.state_label` and only
    `outcome(*edited) == outcome(*baseline)` evaluated, so that no liveness
    assertion can supply the failure:

        arm            relabel TASK_ISSUE only    relabel every issue
        swarm:blocked  equal                      equal
        swarm:claimed  equal                      differs
        swarm:review   equal                      differs
        swarm:done     equal                      differs
        swarm:failed   equal                      differs

    So the loop is load-bearing on four arms and nothing else is: relabelling
    `TASK_ISSUE` alone leaves the disposal rules reading the same label in both
    runs, which is the blindness #202 found. It cannot help `swarm:blocked`,
    and neither can any other fixture - `plan_reconcile` does not branch on
    that state at all, which
    `test_reconcile.test_plan_reconcile_cannot_tell_blocked_from_ready` pins.
    That arm still goes red under the regression, but on the liveness
    assertion, and it is worth knowing which. Under the resolver the edit must
    change nothing in every arm.
    """
    client, fleet, loop, seen = a_lifecycle_run(
        label=READY, alongside=alongside, artifacts=artifacts
    )
    # No fleet for the first cycle, so it settles the task without dispatching
    # it and the comparison below is between two second cycles.
    loop.fleet = None
    loop.cycle()

    for number in client.issues:
        client.issues[number]["labels"] = [{"name": label}]
    loop.fleet = fleet
    loop.cycle()
    return client, fleet, seen


#: Every state a human could type onto an issue that disagrees with a world
#: saying `eligible`, named by the internal state the old `swarm:*` label stored.
#:
#: `OBEYED_LABELS` stood beside it and is deleted with the flag (#152). It was
#: the subset the *label* machine obeyed - the inverse arm's parameter list - and
#: with one source of state there is no machine left to obey anything.
WRONG_LABELS = (BLOCKED, CLAIMED, REVIEW, DONE, FAILED)


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
    client, fleet, _ = a_run(CLAIMED, monkeypatch)

    assert fleet.spawned == [TASK_ISSUE]
    assert client.labels_on(TASK_ISSUE) == {CLAIMED}


def test_a_hand_edited_label_is_not_even_an_event_any_more(monkeypatch):
    """"...and the divergence is reported" - and #152 changes which half of the
    criterion this test is.

    Under the flag the edit was *seen* and overruled, so the second cycle
    emitted a `state.override` saying "the label stores claimed, we acted on
    eligible". There is no label plane left to disagree with: the second cycle
    below runs against an issue a human moved to `swarm:claimed` and has nothing
    at all to say about it, which is the strongest form the criterion can take
    and the reason the assertion is now about cycle 0.

    What is still worth an event is the run meeting a task it has no memory of.
    `Belief.stored` is what this process believed **last** cycle, so on the
    first cycle it is `""` rather than a label, and the override says the
    authority moved rather than that the labels were wrong. `swarm show` counts
    it either way, which is what keeps a run's own artifacts readable without
    anybody opening the jsonl.
    """
    _, _, seen = a_run(CLAIMED, monkeypatch)

    overrides = [fields for name, fields in seen if name == STATE_OVERRIDE]
    assert overrides == [
        {
            "cycle": 0,
            "task": f"task-{TASK_ISSUE}",
            "believed": ELIGIBLE,
            "stored": "",
            "derived": ELIGIBLE,
            "kind": "",
            "why": (
                "every dependency has landed, no container carries this task "
                "and no pull request is open for it"
            ),
        }
    ]
    # And `swarm show` counts it, so a run's own artifacts say how often the
    # authority moved without anybody reading the jsonl.
    tally = DivergenceTally.from_events(
        [{"event": name, **fields} for name, fields in seen]
    )
    assert tally.overrides == 1
    assert tally.override_tasks == (f"task-{TASK_ISSUE}",)
    assert "state override" in tally.text()


@pytest.mark.parametrize("label", WRONG_LABELS)
def test_no_wrong_label_changes_a_decision_under_the_resolver(
    label, monkeypatch, tmp_path
):
    """The completeness test, and the one that fails if a path forgets.

    Five of the six labels, each written over a run whose world plainly
    disagrees with it, each asserted to produce the decisions of a run whose
    labels were never touched. Readiness, dispatch and reconcile all run in
    every arm, so a decision path that still read `entry.state_label` would
    have to agree with the resolver on all five to hide here - and `swarm:done`
    and `swarm:review` disagree with the world in different modules.

    The run has two tasks for #202's reason. `outcome()` compares four things,
    and over a one-task run two of them - the containers disposed and the pull
    requests merged - are `[]` in every arm whatever was decided, so the
    equality was evidence about dispatch and readiness and nothing else. The
    second task carries a container, an open pull request and the result record
    its worker wrote on the way out, which puts both back in play.

    **How far the equality itself reaches, measured** (#228). With
    `plan_reconcile` reverted to `entry.state_label` and the liveness
    assertions below switched off, so that only `outcome()` can fail:

        swarm:blocked  equal    <- blind
        swarm:claimed  differs
        swarm:review   differs
        swarm:done     differs
        swarm:failed   differs

    Four of five, where #202 left two. `swarm:claimed` and `swarm:review` were
    a gap in the fixture rather than in the argument: the second task's worker
    had published a pull request but left no result record, and both of §4's
    label-readable rows are gated on one. `a_lifecycle_run(artifacts=...)`
    writes it, and those two arms now fail on the comparison that states #147's
    criterion rather than on a fixture guard.

    `swarm:blocked` is not a gap and cannot be closed here.
    `plan_reconcile` branches on a closed issue, on terminal, on `claimed` and
    on `review`, and on nothing else - so `blocked` and `eligible` take the
    same path through every rule at every combination of the facts, which
    `test_reconcile.test_plan_reconcile_cannot_tell_blocked_from_ready` asserts
    over all 72 of them. No fixture makes an arm sensitive to a distinction the
    function does not draw. What that costs the completeness claim is exact and
    small: this arm is evidence about readiness and dispatch and none about
    reconcile, and it stays in `WRONG_LABELS` because those two paths are real
    and readiness is where `swarm:blocked` was always decided.
    """
    baseline = a_run(
        READY, monkeypatch, alongside=True, artifacts=tmp_path / "baseline"
    )
    edited = a_run(
        label, monkeypatch, alongside=True, artifacts=tmp_path / "edited"
    )

    assert outcome(*edited[:2]) == outcome(*baseline[:2])
    # And the three things that equality needed to mean anything, because each
    # of them is a line somebody could take out without a test going red and
    # leave #202 exactly where it started. Two live slots: a container was
    # disposed and a pull request was merged, so the comparison ran over three
    # components rather than one. And the edit reached the task carrying that
    # container - relabelling `TASK_ISSUE` alone leaves the disposal rule
    # reading the same label in both arms, which is what made this blind.
    #
    # A fixture guard, deliberately kept as one: on `swarm:blocked` it is still
    # the only assertion here a `plan_reconcile` regression can fail, and on the
    # other four it is what says the equality ran over a world with something
    # in it rather than over four empty slots.
    client, fleet, _ = baseline
    assert fleet.disposed == [OTHER_ISSUE]
    assert client.merges == [OTHER_PULL]
    assert edited[0].labels_on(OTHER_ISSUE) == {label}


# --------------------------------------------------------------------------
# 2. The flag restores the previous behaviour completely
# --------------------------------------------------------------------------


def test_the_merge_gate_follows_the_authority_too(monkeypatch):
    """"Not for the scheduler but not the merge gate" is the failure mode.

    #147 names three files, and the acceptance criterion it is really about is
    larger than the file list: a label a human edits mid-run must not change
    *what the orchestrator does*, and merging is the most consequential thing it
    does. `plan_checks` and `run_mergeability` selected on `swarm:review`, so a
    task somebody relabelled `swarm:ready` while its green pull request sat open
    was silently un-mergeable. They ask `authority.in_review` now.

    The flag's arm went with #152 - it asserted the gate *refusing* to merge
    when `APIARY_STATE_SOURCE=labels` made `swarm:ready` the answer, and there is
    no second source to make it say that. What is left is the direction that
    still has a subject: the pull request is open, so the world says review, and
    the relabel in the last interval does not stop the merge.
    """
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

    assert client.merges == [TASK_PULL]


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
    client, fleet, loop, _ = a_lifecycle_run(label=READY)
    loop.artifacts = tmp_path

    loop.cycle()
    # The worker ran and failed. Its container is gone, which is what makes the
    # resolver read `eligible` and what used to make this rule unreachable.
    fleet.handles.clear()
    write_result(record(TASK_ISSUE, 1, attempt=0, reason="the tests failed"), tmp_path)
    loop.cycle()

    # The counter moved, which is the whole of the retry engine: without this
    # the task is re-dispatched forever and `max_attempts` bounds nothing. It
    # moved in apiary's own store rather than in the issue body since ADR 0005;
    # the property being tested is the same one. The label ends the cycle at
    # `swarm:claimed` rather than `swarm:ready` because the same cycle
    # re-dispatched the retry, which is what it did before #147 too.
    assert loop.store.read()[ref(TASK_ISSUE)].attempt == 1
    assert fleet.spawned == [TASK_ISSUE, TASK_ISSUE]


def test_a_pull_request_closed_unmerged_still_costs_an_attempt(monkeypatch):
    """The second edge, and the reason the first fix is not enough.

    `Snapshot` lists open pull requests only, so a closed one is invisible and
    the resolver says `eligible` here too. A reconciler that waited for `review`
    would forgive every rejected pull request - "a retry that costs nothing can
    be rejected forever", which is what that rule exists to prevent.
    """
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

    assert ref(TASK_ISSUE) not in loop.store.read()

    # ...and then a human closed the pull request without merging it. Nothing in
    # the world records that this ever happened - `Snapshot` lists open pull
    # requests only - so the whole of the evidence is what the orchestrator
    # believed last cycle.
    client.open_pulls = ()
    loop.cycle()

    assert loop.store.read()[ref(TASK_ISSUE)].attempt == 1


def test_the_infrastructure_streak_is_counted_once_and_costs_no_attempt(
    tmp_path, monkeypatch
):
    """The self-clearing half, which is why `previous` rather than a container.

    An infrastructure verdict deliberately does not bump the counter, so the
    same result file stays "unaccounted for" by the arithmetic and the rule that
    reads it has to be edge-triggered or it fires every cycle. The label stopped
    it by moving to `swarm:ready`; the carried belief does the same job, and a
    rule keyed on a leftover container instead would have escalated a task whose
    disposal the daemon happened to refuse.

    **This was a `source` pair and is now a number** (#152). It asserted the
    streak reaching *the same value under both sources* rather than any
    particular value, precisely so that it said nothing about `_observe`'s own
    arithmetic - which was the right shape while there were two sources to
    compare and is unwritable with one. What is left is the property the
    comparison was standing in for: the rule is edge-triggered, so one result
    file counts once and the attempt it did not spend stays unspent.
    """
    client, fleet, loop, _ = a_lifecycle_run(label=READY)
    loop.artifacts = tmp_path

    loop.cycle()
    write_result(record(TASK_ISSUE, 2, attempt=0, reason="docker: no such image"), tmp_path)
    loop.cycle()
    # A third cycle over the *same* record, which is what "edge-triggered" has to
    # mean here: the arithmetic still cannot account for the attempt, so a rule
    # reading the world alone would count it again.
    loop.cycle()

    assert loop._infrastructure.get(ref(TASK_ISSUE), 0) == 1
    # And the attempt an infrastructure failure does not consume is still
    # unconsumed, which is the half a user feels.
    assert loop.store.read().get(ref(TASK_ISSUE)) is None


# --------------------------------------------------------------------------
# 4. ADR 0001's three, which stayed non-derivable
# --------------------------------------------------------------------------


def test_the_infrastructure_ceiling_still_escalates_though_nothing_can_see_it():
    """ADR 0001's first non-derivable state, and the resolver reads `eligible`.

    The artifacts reach the cycle as one record per task, so N mechanical
    failures displace one another there and nothing can count them. The
    orchestrator keeps believing its own counter; making the resolver
    authoritative did not make this derivable, and a cutover that pretended
    otherwise would retry a broken host forever.
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


    # **One route now, and the second is deleted rather than moved.** It was a
    # fresh process with nothing but the label - `believe(ledger(merged),
    # world_after_the_merge)` reading `swarm:done` off the entry and pinning the
    # task on that alone - and #152 removed both the field and the seed. A
    # restarted orchestrator has no memory of a merge it made in a previous
    # process, which is the documented consequence rather than an oversight, and
    # `test_the_first_sight_seed_is_the_label_and_the_contract_says_so` went the
    # same way.
    #
    # What is left is the run remembering its own merge, which is what every
    # cycle after a landing actually does. The memory is two arguments and not
    # one: `remembered` is what the previous cycle believed about this work item
    # and `landed` is the ratchet's own set of refs, which since #214 only a
    # state this process genuinely believed can enter. `believe` fills both and
    # `Reconciler._carry_forward` hands them on - `carry()` below is that pair.
    # What the `UNRESOLVED` fallback leaves behind is a *state* and nothing else,
    # which deliberately does not count;
    # `test_a_label_the_resolver_had_no_verdict_for_never_pins_a_task` is that
    # half.
    held = believe(
        ledger(merged),
        world_after_the_merge,
        remembered={ref(4): LANDED},
        landed=frozenset({ref(4)}),
    )

    assert held.state("task-4") == LANDED
    assert [one.kind for one in held.overrides] == [LANDED_STANDS]
    # The resolver's own verdict is recorded on the override, which is what
    # makes this assertion about a *rescue* rather than about agreement.
    assert held.overrides[0].derived == ELIGIBLE

    plan = _plan_dispatch_(
        ledger(merged),
        capacity=Capacity(slots=3, configured=2),
        ready=(ref(4),),
        believed=held,
    )
    assert plan.numbers == ()


def test_the_previous_belief_is_the_remembered_overlay_alone():
    """The last place a label reached a decision, and #152 closed it.

    `previous` used to be `{**by_label, **seen}`: a task this process had never
    carried was seeded from the `swarm:*` label its issue wore, so that
    `plan_reconcile`'s two edge-triggered rules had a prior state on the first
    cycle of a resumed run. There is no label to seed from, so a task apiary has
    not seen this process is simply unknown to it - which is the safe direction
    and ADR 0001's: an edge fires on a *change*, and an empty `previous` means no
    edge fires on a first sight at all. The cycle after acts on a transition this
    process watched rather than on one it inferred from something a human typed.
    """
    task = entry(4, label=REVIEW)
    fresh = believe(ledger(task), world(task))
    assert fresh.previous == {}, "a label is not a memory"

    remembered = believe(ledger(task), world(task), remembered={ref(4): CLAIMED_STATE})
    assert remembered.previous == {"task-4": CLAIMED_STATE}


# --------------------------------------------------------------------------
# 5. A cycle that cannot see
# --------------------------------------------------------------------------


def test_a_task_the_resolver_never_saw_keeps_what_was_remembered_and_is_counted():
    """A fallback nobody can see is a cutover that looks clean by not happening.

    `Resolution.state` answers `""` for a task it has no verdict for, which is
    "nothing was said" rather than a state. What stands in its place used to be
    the label; since #152 it is what this process believed last cycle, which is
    the only record left. Either way the fallback is reported as an override of
    its own, so that a run where the resolver saw nothing does not read as a run
    where it agreed with everything.
    """
    task = entry(4)
    # An observation with no tasks in it, over a task this cycle *has* carried.
    held = believe(ledger(task), world(), remembered={ref(4): CLAIMED_STATE})

    assert held.state("task-4") == CLAIMED_STATE
    assert [one.kind for one in held.overrides] == [UNRESOLVED]

    # And with nothing remembered either there is genuinely nothing to say, which
    # is `Belief.state`'s own rule: a task nothing has an opinion about must not
    # be handed one, and `eligible` would be an opinion.
    blind = believe(ledger(task), world())
    assert blind.state("task-4") == ""
    assert [one.kind for one in blind.overrides] == [UNRESOLVED]


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

    plan = _plan_dispatch_(
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
    plan = _plan_dispatch_(
        ledger(task), capacity=Capacity(slots=3, configured=2), believed=held
    )
    assert plan.numbers == ()
    assert [one.number for one in plan.in_flight] == [4]


# --------------------------------------------------------------------------
# The readers, one at a time
# --------------------------------------------------------------------------


def test_readiness_speaks_about_the_entries_the_authority_says_are_waiting():
    """The one thing readiness used a label for. Everything else it decides is a
    question about the code host and is untouched.

    Both arguments come from the cycle's belief and neither has a fallback since
    #152. `transitionable=None` speaks about **nothing** rather than about every
    entry: it used to fall back to the issue's label, and a pass that fell back
    to a label nobody writes would recompute readiness for tasks the authority
    says are claimed, in review or landed.
    """
    task = entry(4, label=CLAIMED)  # the world says eligible; the belief says claimed

    silent = compute_readiness(ledger(task), {})
    spoken = compute_readiness(
        ledger(task), {}, transitionable={"task-4"}, current={"task-4": CLAIMED}
    )

    assert silent.verdicts == ()
    assert [verdict.state for verdict in spoken.verdicts] == [READY]
    # `current_state` is what the belief holds rather than what an issue carries
    # - there is nothing on an issue to carry it any more - so `changed` is "this
    # pass moved the task" rather than "the label on GitHub is wrong".
    assert spoken.verdicts[0].current_state == CLAIMED
    assert spoken.verdicts[0].changed


def test_reconcile_reads_terminal_from_the_authority_and_from_state():
    """A human marks a running task done. The container's fate is the decision.

    Under the labels `swarm:done` is terminal and the worker is disposed
    mid-attempt. Under the resolver a live container is a claim whatever the
    label says, so the worker is left alone - which is the same rule from the
    other side: GitHub wins when a human *closes* an issue, and a label is not a
    closure.
    """
    task = entry(4, label=DONE)
    live = world(task, containers=(ContainerFact(id="c", run_id="", ref=ref(4), running=True),))

    # A cycle that really does believe the task landed still takes its worker
    # away. This was the `labels` arm - `plan_reconcile(book, running=[...])`
    # with no belief at all, reading `swarm:done` off the entry - and it is
    # rewritten rather than deleted because the rule it exercises is the
    # reconciler's own and still exists. `state_of` raises without a belief now,
    # so the fixture's declared state is supplied as one.
    believed_landed = fixture_belief(ledger(task))
    disposed = plan_reconcile(ledger(task), running=[ref(4)], believed=believed_landed)
    assert [one.ref for one in disposed.disposals] == [ref(4)]

    # `remembered`, because this is the mid-run case: the orchestrator has seen
    # this task claimed and a human has since typed `swarm:done` on it. The
    # label reaches nothing, the live container is a claim, and the worker is
    # left alone.
    held = believe(ledger(task), live, remembered={ref(4): CLAIMED_STATE})
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
        to_state = CLAIMED_STATE

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
# a model is shown. Each used to get a pair here: the edit changes nothing under
# `derived`, and changes the answer under `labels`. The second half is deleted
# with the flag (#152) - `state_of` raises without a belief rather than reading a
# label, so there is no arm left to compare against. What each test keeps is the
# arm that still runs, and where the pair was carrying the only assertion about a
# consequence a user feels - `recovery` spending an attempt - that assertion is
# rewritten onto a belief rather than dropped.


def a_hand_edited(label: str, was: str, **facts: Any) -> tuple[Any, Belief]:
    """One task wearing `label`, a world that disagrees, and both beliefs.

    The two beliefs come from **one** ledger and **one** observation, which is
    what makes each pair below a comparison of who is believed rather than of
    two different runs - `shadow.py`'s rule, and `believe`'s own.

    `was` is what this process believed last cycle, and it is not decoration:
    an edit made without it belongs to a task the run has never carried, and the
    two edge-triggered rules that read `Belief.previous` would then be reading a
    memory that does not exist. #147's criterion is about a label edited
    **mid-run**, and this is what mid-run means - the same point `a_run` makes
    one cycle at a time for the tests that drive a whole loop.
    """
    task = entry(4, label=label)
    book = ledger(task)
    seen = world(task, **facts)
    # One belief, not a pair. The second was the `labels` arm and it is gone
    # with the flag (#152); what is left is the property the pair existed to
    # demonstrate - that a belief carried from last cycle is overridden by what
    # the world says this cycle.
    return book, believe(book, seen, remembered={task.ref: was})


def test_a_claimed_label_typed_onto_a_ready_task_no_longer_burns_an_attempt():
    """`recovery.py`, and the sharpest form of the criterion in this package.

    Releasing a stale claim **consumes an attempt** (`recovery._release`), so
    before this the sweep answered a label a human typed by spending a retry off
    a task that had never run - and, at the cap, by escalating it to a human. It
    is the one place a mid-run edit cost budget rather than a cycle.

    Under the resolver a claim is a *running container*, there is none, and the
    entry is not the sweep's to speak about at all.
    """
    book, derived = a_hand_edited(CLAIMED, ELIGIBLE)

    swept = _plan_recovery_(book, containers=(), believed=derived)
    assert swept.transitions == ()
    # Not silently ignored either: nothing here holds it, because nothing here
    # selected it. The task is `eligible`, and the dispatcher picks it up.
    assert swept.held == ()
    assert derived.state("task-4") == ELIGIBLE

    # The counterfactual, and what the `labels` arm was carrying that no other
    # test does: the sweep really can spend an attempt, so the equality above is
    # about *this* task not being selected rather than about a sweep that never
    # transitions anything. It runs on a belief now - the fixture's declared
    # `claimed` - rather than on the label the arm used to read.
    #
    # The third assertion the pair had is gone with it: `believed=None` was
    # "`Recovery.startup` and the `__main__` dry run are the labels arm exactly",
    # and `state_of` raises without a belief since #152, so every caller of this
    # module is now on the one arm below.
    claimed = _plan_recovery_(book, containers=())
    assert [str(one) for one in claimed.transitions] == [
        "#4: claimed -> eligible, attempt 1 "
        "(claimed with no live container behind it)"
    ]
    # The budget, which is the half a transition's `str` does not show and the
    # only half a user feels.
    assert [one.attempt for one in claimed.transitions] == [1]


def test_a_failed_label_typed_onto_merged_work_no_longer_resigns_the_run():
    """`goal.py`: the gate partitions the ledger into done / failed / live.

    A `swarm:failed` typed onto a task whose pull request merged put it on the
    wrong side of that partition, and the wrong side is not cosmetic here: the
    abandoned arithmetic is a *refusal*, so the run ended asking a human about a
    task that had landed, with the objective never assessed at all.
    """
    book, derived = a_hand_edited(
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

    # The `labels` arm stood here: the same gate over a belief built from
    # `swarm:failed`, refusing with `GAVE_UP` and `abandoned == (ref(4),)`. It is
    # deleted rather than rewritten because it was the *pair* - a demonstration
    # that the flag put the task on the other side of the partition - and the
    # partition itself is pinned on the one remaining arm above and in
    # `tests/test_goal.py`.


def test_a_done_label_typed_onto_an_open_pull_request_no_longer_assesses_early():
    """`goal.py`'s other side: `live` is what stops the gate judging mid-run.

    The refusal exists because "an objective assessed against a half-landed run"
    is an answer nobody can act on. A `swarm:done` a human typed made the ledger
    read exhausted, and the gate assessed - and could declare the run met - over
    a task whose pull request was still open.
    """
    book, derived = a_hand_edited(
        DONE, CLAIMED_STATE, pulls=(PullFact(number=pull_ref(TASK_PULL), ref=ref(4)),)
    )
    assert derived.state("task-4") == REVIEW_STATE

    assert [one.task_id for one in live(book, derived)] == ["task-4"]
    assert shipped(book, derived) == ()

    held = assess("ship it", book, oracle=Says(met()), believed=derived)
    assert not held.met
    assert held.reason.startswith(IN_FLIGHT)
    assert not held.consulted, "no model is worth swapping in for a half-landed run"

    # The `labels` arm stood here - the same gate under `swarm:done`, assessing
    # the objective as met over a task whose pull request was still open. Deleted
    # with the flag: there is no second belief to build.


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
    book, derived = a_hand_edited(
        FAILED,
        REVIEW_STATE,
        pulls=(PullFact(number=pull_ref(TASK_PULL), ref=ref(4), merged=True),),
    )
    verdict = stalled()

    _, tracked = brief(book, verdict, derived)
    assert "task-4 (landed):" in tracked
    # The whole of the other half of epic #140: no `swarm:*` string reaches a
    # prompt, because a model reads the vocabulary it is shown as the run's own
    # and re-emits it.
    assert "swarm:" not in tracked

    # Two arms stood here and both are deleted. The first was the same brief
    # under `labels`, naming the merged task `needs-human`. The second was
    # `believed=None` printing the label *verbatim* - `task-4 (swarm:failed)` -
    # on the argument that a string in a prompt has to stay byte-identical for
    # the callers outside `Reconciler.cycle`. Neither has a subject: there is no
    # second source, and `state_of` raises without a belief rather than reaching
    # for a label, so every caller of `brief` is on the arm above.


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


def carry(held: Belief) -> dict[str, Any]:
    """What `Reconciler._carry_forward` hands the next cycle, as kwargs.

    Three things rather than one since #214, and the split is the subject of
    this section: `remembered` is keyed by work item because a task id is only
    leased to one, `landed` is the ratchet's own set because a state a label
    reached must not be able to enter it, and `announced` is the event log's
    memory because it decides nothing. Spelled once here so a test says which
    of the three it is exercising rather than restating the plumbing.
    """
    return {
        "remembered": held.carried(),
        "landed": held.landed,
        "announced": dict(held.announced),
    }


def first_seen(state: str, *refs: Any) -> dict[str, Any]:
    """The same three, for the **first** cycle a test drives by hand.

    `believe` consults the ratchet's set only for a work item it is already
    carrying a memory of - `entry.ref in held_by_ref` - so a hand-built `landed`
    with no `remembered` beside it is silently inert. That pairing used to be
    invisible here, because the label seeded `previous` for every task and the
    membership test therefore never took its other branch. #152 removed the seed,
    so a test that wants to start from "this run has already believed this"
    supplies both halves, which is exactly what `carry` hands the cycle after.
    """
    return {
        "remembered": {ref: state for ref in refs},
        "landed": frozenset(refs) if state == LANDED else frozenset(),
    }


def test_the_merge_gates_own_transition_seeds_the_ratchet():
    """The ordinary route in, and the one every other test in this section skips.

    Each of its neighbours hands `believe` a hand-built `landed` set. None of
    them drives the path a real cycle takes: the merge gate applies a
    `swarm:done` transition, `Belief.fold` records it, `_carry_forward` hands
    the result to the next cycle. That path runs through `fold`'s own write to
    `Belief.landed`, and deleting that write leaves the **entire suite green**
    apart from this test while breaking exactly the hole `LANDED_STANDS` exists
    for - the landing cycle's pull request is already out of the open listing,
    the issue is not yet closed, the resolver reads `eligible`, and a worker
    goes back onto merged code.

    So this test folds a real transition rather than asserting one was folded.
    """
    landing = entry(4, label=REVIEW)
    book = ledger(landing)

    # Cycle 1: the merge gate lands it. `fold` is what a real cycle calls.
    believed = believe(book, world(landing))
    after_merge = believed.fold(
        [
            Transition(
                ref=ref(4),
                from_state=REVIEW_STATE,
                to_state=LANDED,
                reason="its pull request merged",
                task_id="task-4",
            )
        ]
    )
    assert after_merge.state("task-4") == LANDED
    assert after_merge.landed == frozenset({ref(4)})

    # Cycle 2: the evidence is gone - no open pull request, issue still open -
    # and the belief carried forward is the only thing standing between the
    # resolver and a second worker.
    merged = entry(4, label=DONE)
    next_cycle = believe(ledger(merged), world(merged), **carry(after_merge))

    assert next_cycle.state("task-4") == LANDED
    assert dispatchable(ledger(merged), next_cycle, ref(4)) == ()


def test_a_reminted_id_does_not_inherit_the_departed_issues_edge():
    """#201's AC 2 in the place it was only half applied.

    #201 gated the *ratchet* on the ref, but `Belief.previous` carried the
    remembered value under the task id - and `previous` is what
    `plan_reconcile`'s edge-triggered rules read through `_was`. So a re-minted
    id inherited the departed issue's `review`, rule 4 found a pull request that
    was never its own missing from the open listing, and `_retry_or_give_up`
    charged an attempt to a brand-new issue - escalating it to `swarm:failed` at
    the cap. Since #214 the memory is keyed by work item, so there is no lookup
    that could find it.

    The fallback is deliberate rather than inherited: a re-minted id is a first
    sight, and since #152 a first sight has **no** prior state at all. It dropped
    to the label seed until this ticket; the second assertion below is what
    changed, and it changed in the safe direction - an edge fires on a change, so
    no memory means no edge rather than an edge inferred from something a human
    typed.
    """
    fresh = entry(9, label=READY)
    carried = {ref(7): REVIEW_STATE}  # believed about #7, whose id #9 has taken

    held = believe(ledger(fresh), world(fresh), remembered=carried)

    assert held.previous.get("task-9") != REVIEW_STATE, "inherited another issue's edge"
    assert "task-9" not in held.previous, "a first sight has no prior state to read"


# `test_a_standing_ratchet_speaks_again_when_the_label_changes` stood here (#215).
# It drove three cycles over a task the ratchet was holding, moved the issue from
# `swarm:done` to `swarm:failed` in the third, and required `landed-stands` to be
# announced a second time: quiet about the *same* disagreement, not quiet forever.
#
# It cannot fail any more, and the reason is structural rather than a fixture
# problem. The mute compares `Belief.announced[ref]` against `was_stored`, and
# `was_stored` is `previous` - which since #152 is what the *previous cycle
# believed*, carried by `Belief.carried()`, rather than the label the issue is
# wearing. Once the ratchet stands, `states[task]` is `landed` on every following
# cycle by construction, so `was_stored` is `landed` on every following cycle
# too. There is no longer any input that can move it while the ratchet holds, so
# the second announcement this test demanded is unreachable and the assertion
# would have had to be inverted to something that says nothing. Deleted rather
# than inverted. What survives is the mute itself, pinned by
# `test_a_ratchet_that_keeps_standing_stops_announcing_itself` below.


def dispatchable(book: Any, held: Belief, *refs: Any) -> tuple[int, ...]:
    """Which of these tasks the dispatcher would actually put on the fleet.

    The assertion every test in this section ends on. `Belief.state` is what the
    bug is *about*, but a state nobody acts on is not a defect - the failure
    being pinned down here is a worker spawned over merged code, or a task that
    silently never runs again, and only the plan says which happened.
    """
    return _plan_dispatch_(
        book,
        capacity=Capacity(slots=3, configured=3),
        ready=refs,
        believed=held,
    ).numbers


def test_a_label_the_resolver_had_no_verdict_for_never_pins_a_task():
    """The `UNRESOLVED` arm may decide a cycle; it may not decide the run.

    That arm writes the remembered state into the belief because there is
    nothing else to write - "the resolver said nothing" is not an opinion, and
    the memory is the only record left. It was `entry.state_label` until #152,
    which is what made it dangerous: a `swarm:done` a human typed onto a task the
    resolver had no opinion about entered the ratchet and pinned it for the life
    of the process, **a label reaching a decision nothing can undo**.

    **The way in is narrower now, and the guarantee is the same one.** The arm
    still has no write to `Belief.landed`, so a `landed` that arrives through it
    decides that cycle and never the run. The input is built by hand rather than
    typed onto an issue - a memory saying `landed` about a work item the ratchet
    was never given - because with the label gone that is the only remaining
    shape of "a state that was never a decision", and it is exactly the pairing
    `believe` distinguishes: `previous` is read for the edge rules, `landed` is
    read for the ratchet, and only the second is a set of things this process
    decided.
    """
    task = entry(4, label=READY)
    book = ledger(task)

    # Cycle one: an ordinary task this process has now seen.
    seen = believe(book, world(task))
    assert seen.state("task-4") == ELIGIBLE

    # Cycle two: the observation has nothing to say about the task - a stack with
    # no image, a listing that came back without it - and what stands in says
    # `landed` while the ratchet's own set does not carry the work item. That is
    # the documented fallback, and it is harmless for one cycle.
    silent = believe(
        book, world(), remembered={ref(4): LANDED}, landed=frozenset()
    )
    assert silent.state("task-4") == LANDED
    assert [one.kind for one in silent.overrides] == [UNRESOLVED]
    # The state decided that cycle; the *set* the ratchet reads is untouched,
    # which is the structural half of #214 - the arm has no write to it.
    assert silent.landed == frozenset()

    # Cycle three: the resolver has an opinion again, and it is `eligible`. The
    # fallback never became a decision, so there is nothing for the ratchet to
    # stand on and the task runs.
    after = believe(book, world(task), **carry(silent))

    assert after.state("task-4") == ELIGIBLE
    # One override, and it is the plain kind: the belief moved, which is ordinary
    # and is reported. What must not be here is `landed-stands`.
    assert [one.kind for one in after.overrides] == [""]
    assert dispatchable(book, after, ref(4)) == (4,)


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

    landed = believe(book, world(merged), **first_seen(LANDED, ref(4)))
    assert landed.state("task-4") == LANDED

    silent = believe(book, world(), **carry(landed))
    resumed = believe(book, world(merged), **carry(silent))

    assert silent.state("task-4") == LANDED
    assert resumed.state("task-4") == LANDED
    assert dispatchable(book, resumed, ref(4)) == ()


def test_an_id_that_now_names_another_issue_does_not_inherit_landed():
    """`remembered` is keyed by task id, and a task id is not a durable name.

    **This drove the id reuse through `ledger._adopted_id` until #152.** A
    hand-written issue joined the ledger by being labelled `swarm:ready`, its id
    was derived from its title, and collisions were disambiguated by *order* - so
    the bare slug was leased rather than owned and moved to the next issue in
    line the moment the holder fell out of `Ledger.entries`. There is no label to
    adopt an issue with any more: membership is the identity marker, an unmarked
    issue is never in the ledger, and that route to a re-minted id is closed.

    The property it was driving is not, and it is #214's whole subject rather
    than adoption's, so the two ledgers are built by hand instead: one where
    `add-a-retry` names #7 and one where the same string names #9. Keyed by task
    id, #9 inherits #7's `landed` and is **never dispatched and never
    escalated** - the quietest failure in this module, because `landed` emits no
    transition and, once the ratchet is standing, no event either. Keyed by
    `TaskRef` the lookup cannot land on somebody else's memory: #9 asks about #9.
    """
    before = ledger(replace(entry(7, label=DONE), task_id="add-a-retry"))
    after = ledger(replace(entry(9, label=READY), task_id="add-a-retry"))

    assert before.entries["add-a-retry"].ref == ref(7)
    assert after.entries["add-a-retry"].ref == ref(9)

    # What the previous cycle carried: #7, believed landed, under the id it held
    # at the time.
    carried = {"remembered": {ref(7): LANDED}, "landed": frozenset({ref(7)})}
    now = believe(after, world(*after.entries.values()), **carried)

    assert now.state("add-a-retry") == ELIGIBLE
    # The plain kind - #9 is a task this run has no memory of, which is reported
    # and is ordinary. What must not be here is `landed-stands`, which is the
    # silent one: it emits no transition and, once standing, no event either.
    assert [one.kind for one in now.overrides] == [""]
    assert dispatchable(after, now, ref(9)) == (9,)

    # And the same memory still pins the task it is actually about, which is
    # what makes the assertion above about identity rather than about the
    # ratchet having been weakened.
    still = believe(before, world(*before.entries.values()), **carried)
    assert still.state("add-a-retry") == LANDED
    assert dispatchable(before, still, ref(7)) == ()


def test_a_revival_cannot_clear_a_merge():
    """`Reconciler._carry_forward` applies its revival overlay unconditionally.

    `{task: ELIGIBLE for task in revived_tasks(report)}` goes through
    `Belief.hold` after the fold, and before #201 it could silently clear the one
    fact the ratchet exists to make unundoable - after which the next cycle's
    resolver, which cannot see a merged pull request, reads `eligible` and a
    worker goes onto code that is already on the default branch.

    **Reachability, which #201 left open and #212 closed.** The goal gate was
    never a route: since #205 `goal._revive_abandoned` selects
    `abandoned(ledger, believed)`, which is `needs-human`, and `landed` never
    reads as that. The replan was: `_update` selected its revival on
    `entry.state_label == FAILED` and took no belief at all, so a `swarm:done`
    issue a human relabels `swarm:failed` mid-run - which the ratchet keeps
    believing landed, correctly - was revived by any replan that kept the task.
    The assertion below is therefore the **inverse** of the one this test shipped
    with: the planner takes the cycle's belief, and the relabel it used to revive
    is pinned in `test_planner_issues.py` on the selection itself.

    The guard is kept all the same, and the reason is not habit. The overlay is
    still an unconditional dict comprehension over `revived_tasks(report)`, so
    what stands between it and a lost merge is this refusal rather than the
    current shape of one caller - and everything below still holds with the
    planner converted.
    """
    from swarm.nodes.planner import _update

    assert "believed" in inspect.signature(_update).parameters

    merged = entry(4, label=DONE)
    book = ledger(merged)
    held = believe(book, world(merged), **first_seen(LANDED, ref(4)))
    assert held.state("task-4") == LANDED

    revived = held.hold({"task-4": ELIGIBLE})

    assert revived.state("task-4") == LANDED
    # And the set the next cycle reads is untouched: `hold` has no write to it
    # at all, which is why the guard cannot be half-applied.
    assert revived.landed == frozenset({ref(4)})
    # The label really did move - `planner.revive` writes `swarm:ready` - and the
    # belief saying otherwise is the disagreement, not a lost write.
    assert revived.stored["task-4"] == ELIGIBLE
    assert dispatchable(book, revived, ref(4)) == ()

    # And it survives the carry, which is the half that matters: a guard that
    # held for one cycle and was forgotten by the next would only move the
    # dispatch one cycle later.
    later = believe(book, world(merged), **carry(revived))
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
    carried: dict[str, Any] = first_seen(LANDED, ref(4))
    for _ in range(4):
        held = believe(book, world(merged), **carried)
        cycles.append(held)
        carried = carry(held)

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


# `test_the_first_sight_seed_is_the_label_and_the_contract_says_so` stood here,
# and #152 is the ticket that removed its subject. It asserted the one entry into
# the ratchet #201 deliberately kept: a task this process had never carried was
# decided by the `swarm:*` label its issue wore, so a restarted orchestrator that
# found `swarm:done` on an open issue pinned it rather than dispatching a worker
# over merged code. It also asserted the price of that seed against
# `docs/issue-contract.md` §4 - a `swarm:done` issue reopened mid-run stays pinned
# until the process restarts - because a cost a human pays belongs in the contract
# and not only in a docstring.
#
# There is no label to seed from. `previous` is the remembered overlay alone, so a
# task apiary has not seen *this process* is unknown to it, and the ratchet cannot
# be entered by anything but a merge this run watched. Both halves of the test go
# with it: the behaviour it pinned is inverted, and the contract prose it quoted
# describes a machine that no longer exists.
#
# What that costs is worth naming rather than leaving in the diff: the ratchet no
# longer survives a restart at all, so the three facts §4 says a fresh process
# cannot tell apart - a reopened `swarm:done` issue, a merge whose pull request
# omitted the closing keyword, and the gate's own write-before-close window - now
# all read `eligible` on the first cycle of a new process. That is the documented
# consequence of removing the control plane and not a hole this suite can close;
# the within-run ratchet is pinned by everything above.
