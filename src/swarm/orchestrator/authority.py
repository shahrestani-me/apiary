"""Who the orchestrator believes about a task's state (#147). The cutover.

#145 built the resolver, #146 ran it beside the labels and believed nothing it
said. This is where the derived value starts being the one that decides:
`github/readiness.py`, `orchestrator/dispatcher.py` and
`orchestrator/reconcile.py` take their state from here rather than from
`LedgerEntry.state_label`.

The merge gate is a fourth, and it is not scope creep. #147 names three files
and the criterion it is really about is larger than the file list - a label a
human edits mid-run must not change what the orchestrator *does*, and merging is
the most consequential thing it does. `checks.plan_checks` and
`mergeability.run_mergeability` selected on `swarm:review`, so a task somebody
relabelled while its green pull request sat open was silently un-mergeable. They
ask `in_review` below, with the same `None`-reads-the-label default everything
else here has.

The stale-claim sweep, the goal gate and the replan brief are the rest of it,
and they are the same argument a second time. `recovery.plan_recovery` selected
`swarm:claimed` and a release **consumes an attempt**, so a label somebody typed
mid-run did not merely mislead the orchestrator - it spent the task's budget,
and at the cap escalated it to a human. `goal.py` partitioned the ledger into
done / failed / live by label, which is the partition a run's exit code is
computed from: a `swarm:failed` on merged work ended the run asking about a task
that had landed. `replan.brief` decides nothing at all and is here for the other
half of epic #140 - it put a `swarm:*` string into a prompt, and a model reads
the vocabulary it is shown as the run's own. All three take `believed`, all
three read the label when it is `None`, and `tests/test_authority.py` §7 is the
pair each of them is held to.

Labels keep being written and keep being compared. #152 removes the writes;
this ticket changes only who is *believed*, which is why the observable proof
that it happened is a task whose label a human edits mid-run: the orchestrator
does not change what it does, and the disagreement is reported.

## The escape hatch, and why it is the load-bearing part

`APIARY_STATE_SOURCE=labels` restores the previous behaviour **completely**.
Not "for the scheduler but not the merge gate": there is exactly one function
in this package that answers "what state is this task in", every decision path
calls it, and with `labels` it returns `lifecycle.internal_state` of the label
and nothing else. `tests/test_authority.py` proves that by running the same
cycle under both settings against hand-edited labels and asserting the
decisions swap.

That is not ceremony. apiary develops itself on this control plane, so a
misbehaving cutover blocks its own repair, and a cutover with no way back
cannot be repaired by the system it broke. The flag is what makes shipping
this before #146's ten-clean-run gate has been met a decision rather than a
gamble - and the gate has *not* been met, because no credential in this
environment can run a greenfield swarm (#145's corpus README, #146, and
`docs/demo-run.md` all say so).

## The resolver is not the whole authority. The store is the other half

ADR 0001 names five lifecycle states and then, in "Three of these are not
derivable from the code host alone", takes three of them back. ADR 0002 says
what the missing half is: *"the five lifecycle states are derived from the code
host, the containers, the run artifacts, **and apiary's own store**."*

So the authority is the resolver **plus apiary's own judgments**, and this
module is the join. `derived.py` answers what the world says; the store and the
run-scoped counters answer what apiary decided about its own execution. Neither
alone is enough, and reading the label instead of either was what made it look
as though one was.

Each of ADR 0001's three, and what the orchestrator now does about it:

**1. The infrastructure ceiling.** Not derivable at all: exit 2 does not bump
the attempt, so N mechanical failures write one result filename and the
artifacts cannot tell one from three (`reconcile.infrastructure_streaks`
counts *transitions*, which are writes). The orchestrator therefore keeps
believing its own counter: a task whose streak has reached
`APIARY_MAX_INFRASTRUCTURE` is `needs-human` whatever the resolver says, and
the override is reported. The counter is run-scoped, exactly as it was before
this ticket - what the label added was that its *consequence* survived a
restart. It no longer does: a resumed run gives an infrastructure-capped task
another `cap` mechanical failures before escalating again. Those attempts are
free by construction (§4 does not consume one for exit 2), so the cost is
cycles rather than budget, and the escalation still arrives.

**2. A renewed per-blocker budget.** `_retry_or_give_up` gives up on `streak`,
not on `attempt`, and a renewal is an ADR 0002 store judgment no branch,
container or result can see. So the resolver's `needs-human` - which is
arithmetic over code-host evidence - is **advisory**, and the store decides.
`budget_spent` below is `_retry_or_give_up`'s own test, run against the same
two numbers, with ADR 0002's own fallback for a task the store has never
judged: absence reads as the attempt counter, which is the largest streak
consistent with it, so a miss gives up sooner and never later.

The rule runs in both directions and the second one was not obvious. A task
apiary has given up on leaves **no code-host evidence at all** once the process
that ran it is gone: results live in the run directory and a run directory is
per run, and `build_observation` takes branch names off *open* pull requests
because a remote branch listing is a call no cycle makes. So a task that failed
three times, with its pull requests closed and its run over, resolves to
`eligible` from scratch on the next process - and under the labels
`swarm:failed` was what carried the verdict across the restart. The store
carries it now (`budget-spent`), which is why the escape hatch is not the only
thing standing between this cutover and a run that resurrects abandoned work.

**3. A revival.** `planner.revive` "deliberately resets nothing", so a revived
task reads spent from every code-host source there is. Under the labels it was
`swarm:ready` and the dispatcher believed that; under the resolver it would be
re-escalated the instant it was revived, and the goal gate could never unstick
a run. So a revival is recorded here as what it actually is - apiary granting
one more attempt - and it lapses the moment that attempt is spent, which is
where `_retry_or_give_up`'s arithmetic takes over again and caps the task on
the streak it never reset. Run-scoped, for the same reason `_infrastructure`
and `update_budget` are, and failing the same way: a restart forgets a pending
revival and the task waits for a human. That is the safe direction - a
forgotten revival escalates, it never grants a budget - and it is the opposite
of what the label did, where a restart preserved the granted retry.

*What "spent" is allowed to mean, and why it is not a result (#200).* The
grant used to lapse on one thing only: the code host accounting for an attempt
past the one it was granted at, which needs a result record or an
attempt-numbered branch on an **open** pull request. A granted attempt that
dies without writing either - killed at `SWARM_WORKER_TIMEOUT`, a spawn that
failed, a container reaped mid-cycle - produces neither, so the grant
suppressed the give-up for the rest of the run and the task was re-dispatched
every cycle, indefinitely.

**Nothing else bounds it, and each of the three reasons is worth naming**,
because a reader who assumes one of them will read this as a slow leak rather
than a loop with no end.

- `_retry_or_give_up` is the only thing that moves `entry.attempt` here, and it
  runs off a result record.
- `infrastructure_streaks(..., result.applied)` counts *transitions*, and this
  input produces none.
- `recovery.plan_recovery` does consume an attempt for a claim with nothing
  behind it - but since #205 it selects on `state_of(entry, believed)` rather
  than on the `swarm:claimed` label, and a revived task holding a grant with no
  live container is believed **eligible**. It is not the sweep's to speak about,
  so it does not speak. Before #205 the sweep released it and bounded the loop
  at `max_total_attempts`; the cutover removed that accident, and this is the
  rule that replaces it deliberately.

So the grant lapses on the **dispatch**, not on what the dispatch produced.
That is the intent stated exactly - the grant buys one attempt, and putting the
task on the fleet is what spends it - and it is the only signal in reach no
worker behaviour can starve, because the orchestrator is the thing that emits
it. A third piece of code-host evidence would have failed on the same input as
the first two. `Grant.dispatched` is that fact, `Reconciler` records it off
`DispatchReport`, and a lapsed grant simply stops suppressing: the streak
`planner.revive` never reset caps the task through the ordinary arithmetic
below, which is exactly where a revival that *did* produce a result has always
ended up.

*The option that looks cheaper and is not.* ADR 0002's other half -
`TaskJudgement.matches`, "a moved counter means it is no longer that attempt" -
would lapse the grant on `grant.attempt != entry.attempt`, with no new field at
all. It is rejected on the same fact the three bullets establish: on this input
`entry.attempt` does not move, so the stamp never goes stale and the comparison
never fires. It would have worked against the pre-#205 sweep, which is exactly
the kind of dependence on a belief-sensitive component this rule must not have.

The lapse cannot escalate a task mid-attempt: the overlay that reads it is
bounded to the *waiting* states, so a live container reads `claimed` and an open
pull request reads `review`, both of which outrank a budget row. It *does*
charge for an attempt the fleet refused to start, and
`reconcile._dispatch_attempted` argues why: a rule that has to classify why an
attempt did not run is a rule with an arm somebody forgot.

None of the three is made derivable by making the resolver authoritative, and
pretending otherwise is how a cutover ships a run that cannot stop retrying.

## What the resolver is *not* asked to decide, and why

**`eligible` versus `blocked`.** The resolver has a rule for it - every
dependency landed - and `github/readiness.py` has a better one: it sees a ring
in the graph and refuses to plan at all, it tells an unresolvable `#404` from
an open issue, and it resolves the state of dependencies that are *not tasks in
the plan*. That last one is not a nicety. `derived._landed` builds from the
observation's own tasks, so a hand-written issue somebody closed by hand - "half
of what a real plan waits on", by its own docstring - is not in it, and a
resolver believed on this split would hold every dependant `blocked` forever.

So the division is: the resolver decides whether a task is *waiting* at all -
landed, needs-human, claimed, review, or none of those - and readiness decides
which of the two waiting states it is in. Readiness reads no label to do it;
what it reads from here is which entries are its to speak about, which is the
one thing it used the label for.

## What no absolute reading of the world can replace

`plan_reconcile` is **incremental** and the resolver is **absolute**, and the
label was quietly doing double duty as the previous state. Two of reconcile's
rules are edge-triggered and neither survives a naive flip:

- *"a claimed task whose worker finished"* - the worker exits, its container
  stops, no pull request exists, and the resolver correctly says `eligible`. A
  reconciler that waited for `claimed` would never observe a failed attempt
  again: no retry comment, no counter, no give-up, and a task re-dispatched
  from scratch every cycle with nothing counting.
- *"a task in review whose pull request was closed unmerged"* - `Snapshot`
  lists open pull requests only, so a closed one is invisible and the resolver
  says `eligible`. A reconciler that waited for `review` would forgive every
  rejected pull request, which is the "a retry that costs nothing can be
  rejected forever" that rule exists to prevent.

Both need the *previous* belief, so `Reconciler` carries one - `_believed`,
what this process last held, in exactly the register `_infrastructure`,
`_stalls` and `update_budget` already are. It self-clears the way the label
did: once the reconciler has acted, the belief it carries forward is the state
it moved the task to, so the rule cannot fire twice on one attempt.

A task this process has never seen has no previous belief, and there the label
**is** seeded - see `Belief.previous`. It is the only durable record of the last
belief this system held, epic #140 has not removed it yet, and the alternative
is that every restart forgives one rejected pull request, one unaccounted
result, and - the worst of the three - every merge whose issue is no longer
closed. A task already in the map is never re-seeded, which is precisely why a
label edited *mid-run* changes nothing: #147's criterion is about a run in
progress, and a label edited before the process started is the only record
there is. When #152 deletes the labels this seam becomes the store's, and the
docstring on `Belief.previous` says so.

**`landed` is the third thing `previous` decides, and it is a ratchet rather
than a trigger.** A merge is terminal within a run (`docs/issue-contract.md`
§4) and the world stops showing it: a merged pull request leaves the open
listing, so the only remaining evidence is the work item being closed as
completed. Two ordinary things take that evidence away - `checks._decide_passed`
writes `swarm:done` *before* GitHub has honoured `Closes #<n>`, and a human can
reopen a finished issue - and in both the resolver reads `eligible` and the
dispatcher puts a worker back on code that is already on the default branch.
So once this run has believed a task landed, it stays landed.

*It has no lapse, so its entry points are the whole of its safety* (#201). Every
other overlay here re-tests its own input every cycle: `revived` lapses on a
dispatch, `budget-spent` re-reads the store, `infrastructure-ceiling` re-reads
the run-scoped streak. This one's input is its own previous output - the set of
work items it decided about, carried forward and read back as next cycle's
`landed` - which is defensible on "a merge cannot be taken back" and makes every
*entry* into `landed` a permanent, silent decision. Three of the four ways it
could be wrong were wider than the paragraph above claims - two entry points and
one way back out - and each is now closed, the first two **structurally** (#214)
rather than by a convention a writer has to keep:

- **A state that was never a belief.** The `UNRESOLVED` arm writes the label
  verbatim, because there is nothing else to write, and that value used to be
  indistinguishable from a verdict once it was carried forward. So a `swarm:done`
  typed onto a task the resolver has no opinion about pinned it for the life of
  the process - a label reaching a decision nothing can undo, outside the seam
  this docstring bounds. The ratchet no longer reads the carried *states* at
  all: `Belief.landed` is a separate set, and only `believe`'s own `LANDED`
  decisions and `fold`'s transitions write to it. The arm that falls back to a
  label has no write site to it, so there is nothing to remember to opt out of.
- **A state carried under an id that has moved.** A task id is not a durable
  name: `ledger._adopted_id` derives a hand-written issue's id from its title
  and disambiguates by order - so when the first taker leaves the ledger the
  next one inherits the bare slug. An id a landed task used, re-minted for a
  different issue, inherited `landed`: never dispatched, never escalated, and
  silent, because `landed` produces no transition. `Belief.landed` is keyed by
  `TaskRef`, so the re-minted id cannot reach the set - it asks about its own
  work item and is answered about that one. `remembered` is keyed by ref for the
  same reason, one rule further out: `previous` is what `plan_reconcile`'s
  edge-triggered rules read, and a re-minted id inheriting the departed issue's
  `review` charged an attempt to a brand-new issue.
- **The revival overlay.** `Reconciler._carry_forward` applies
  `{task: ELIGIBLE for task in revived_tasks(report)}` unconditionally, which
  could clear the one fact this exists to make unundoable. It *was* reachable:
  `nodes/planner._update` selected its revival on `entry.state_label == FAILED`
  rather than on the belief, so a `swarm:done` issue a human relabelled
  `swarm:failed` mid-run was revived by any replan that kept the task. (The goal
  gate was never a route: since #205 `goal._revive_abandoned` selects
  `abandoned(ledger, believed)`, which is `needs-human`, and `landed` never
  reads as that.) #212 converted that planner, so no revival in the tree now
  reaches a task this run believes landed. `Belief.hold` keeps refusing to move
  one out of `landed` all the same, and the reasoning for keeping a guard whose
  caller is gone is on the guard itself.

The fourth route in - the label on a task this process has never carried - is
**kept**, and the reasoning is in `docs/issue-contract.md` §4 beside the human
rows rather than only here, because its cost falls on a human. `swarm:done` on
an open issue is produced by three different things and a fresh process cannot
tell them apart: `checks._decide_passed`'s window, a pull request merged without
the keyword, and a human reopening finished work. Two of the three must not be
dispatched. Honouring the third therefore means re-dispatching over merged code
in the other two, and it is not a new irreparable state either - a mid-run
relabel off `swarm:done` is already ignored by #147, and a mid-run reopen is the
same edit in a different field. §4 has said what to do instead since it was
written: "a reopened issue is new work with a new id".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

from ..github.ledger import Ledger, LedgerEntry
from ..taskref import TaskRef
from .derived import (
    BLOCKED,
    CLAIMED,
    ELIGIBLE,
    LANDED,
    NEEDS_HUMAN,
    REVIEW,
    Budget,
    Observation,
    Resolution,
    resolve,
)

__all__ = [
    "BUDGET_RENEWED",
    "BUDGET_SPENT",
    "DERIVED",
    "INFRASTRUCTURE_CEILING",
    "LABELS",
    "LANDED_STANDS",
    "REVIVED",
    "SOURCES",
    "STATE_SOURCE_ENV",
    "UNRESOLVED",
    "WAITING",
    "Belief",
    "Grant",
    "Override",
    "believe",
    "budget_spent",
    "in_review",
    "label_state",
    "revived_tasks",
    "source_summary",
    "state_of",
    "state_source",
]

#: The switch, and the reason the rest of this module is safe to ship. Defaults
#: to `derived`, which is the cutover; `labels` restores what #146 shipped.
STATE_SOURCE_ENV = "APIARY_STATE_SOURCE"

DERIVED = "derived"
LABELS = "labels"
SOURCES: tuple[str, ...] = (DERIVED, LABELS)

#: The two states a task that is neither in flight nor finished can be in, and
#: the set `github/readiness.py` owns the split of. See the module docstring.
WAITING: frozenset[str] = frozenset({ELIGIBLE, BLOCKED})

#: Why a belief is not simply the resolver's verdict. The first three are ADR
#: 0001's three non-derivable states, spelled the same as
#: `orchestrator/shadow.py`'s classification kinds **and imported from here by
#: it**, so a divergence the shadow explains and an override this module
#: applies are recognisably the same phenomenon in `events.jsonl`.
INFRASTRUCTURE_CEILING = "infrastructure-ceiling"
BUDGET_RENEWED = "budget-renewed"
REVIVED = "revived"
#: The other direction of the same rule, and it has no entry in `shadow.py`
#: because the shadow window never had to decide anything. A task apiary gave up
#: on leaves **no code-host evidence at all** once its run directory is gone -
#: results are per run (`Reconciler._results`) and `build_observation` reads
#: branch names off *open* pull requests, so a task that failed three times with
#: nothing open reads `eligible` from scratch on the next process. Under the
#: labels `swarm:failed` carried that verdict across the restart; under the
#: resolver the store has to, which is the half of ADR 0002 that is not about
#: renewals.
BUDGET_SPENT = "budget-spent"
#: `landed` is terminal **within a run** - `docs/issue-contract.md` §4 says so in
#: as many words, and a merge is the one thing on the code host that cannot be
#: taken back. It is a kind rather than an assumption because the world stops
#: showing it: a merged pull request is not in `Snapshot`'s open listing, so once
#: `Closes #<n>` has been honoured the *only* evidence left is the closed issue -
#: and a human who reopens that issue, or a pull request merged without the
#: keyword, takes that evidence away. The resolver then reads `eligible` and the
#: dispatcher spawns a worker over work that is already on main.
#:
#: **The only kind here with no lapse**, which is why its entry points are
#: enumerated in the module docstring and why it is the only one that does not
#: repeat itself in `events.jsonl`: it re-tests nothing, so every cycle after the
#: first would emit the same sentence about the same task and turn
#: `DivergenceTally.overrides` into a measure of run length. See
#: `Belief.announced`, which is how the next cycle knows it already spoke.
LANDED_STANDS = "landed-stands"

#: The last is not one of ADR 0001's: it is a task the resolver had no verdict
#: for at all, which `Resolution.state` reports as `""` rather than as a state
#: because "nothing was said" is not an opinion. The label stands, and the
#: fallback is counted rather than silent - a cutover that silently fell back to
#: the labels for every task would look exactly like a clean one.
UNRESOLVED = "unresolved"

#: Suppressing the budget rule means resolving against a cap nothing can reach,
#: rather than teaching `derived.resolve` a second mode. The resolver stays a
#: pure function of an observation, the suppression is expressed in the
#: observation, and the two answers are directly comparable because they came
#: from one input.
_UNBOUNDED = 1_000_000_000


def state_source(env: Mapping[str, str] | None = None) -> str:
    """Which control plane decides. **Loud on garbage**, unlike the shadow flag.

    `shadow.shadow_enabled` reads a mistyped value as its default and argues the
    case: that flag decides whether an *observer* runs, and a `ValueError` out of
    it would be a shadow taking down a cycle. This one decides who the
    orchestrator believes, so it belongs with `checks._env_flag` and
    `dispatcher._env_int` instead - an operator who typed `APIARY_STATE_SOURCE=lables`
    to get back to the old behaviour after a bad cutover must not silently get
    the new one.
    """
    raw = (env or os.environ).get(STATE_SOURCE_ENV)
    if raw is None or not raw.strip():
        return DERIVED
    value = raw.strip().lower()
    if value not in SOURCES:
        raise ValueError(
            f"{STATE_SOURCE_ENV}={raw!r} is not one of {', '.join(SOURCES)}"
        )
    return value


def state_of(entry: Any, believed: Belief | None = None) -> str:
    """What state is this task in? **The** function this module's docstring means.

    "There is exactly one function in this package that answers 'what state is
    this task in', every decision path calls it, and with `labels` it returns
    `lifecycle.internal_state` of the label and nothing else" - so it is spelled
    once, here, and every module that decides on a state asks it rather than
    reimplementing the two-line branch. `in_review` below is the first caller
    and is now a comparison against this rather than a second copy of it.

    `believed=None` reads the label, which is every caller outside
    `Reconciler.cycle`: the `__main__` dry runs, and `APIARY_STATE_SOURCE=labels`
    by way of a `Belief` whose states are the labels anyway.

    The answer is always in the **internal** vocabulary, both sides, which is
    what makes a caller's comparison one comparison rather than one per source.
    That is safe because `INTERNAL_STATE` is one-to-one over the six `swarm:*`
    labels and `internal_state` falls back to the suffix, so no two labels a
    caller could be distinguishing collapse into one state here.
    """
    if believed is None:
        return _internal(entry.state_label)
    return believed.state(entry.task_id)


def in_review(entry: Any, believed: Belief | None = None) -> bool:
    """Is this task in review - as the cycle's authority has it, or as its label?

    The merge gate's own predicate, and the reason it lives here rather than
    being spelled a third and fourth time in `checks.py` and `mergeability.py`:
    #147's criterion is that a label a human edits mid-run does not change what
    the orchestrator does, and a gate that still read the label would merge, or
    refuse to merge, on that edit. `believed=None` reads the label, which is
    every existing caller, both `__main__` dry runs and
    `APIARY_STATE_SOURCE=labels`.

    Both sides say the same thing in the ordinary case, and it is worth naming
    which fact each is: the label is a record of `worker/pr.py` having opened a
    pull request, and the derived state is that pull request being open now.
    """
    return state_of(entry, believed) == REVIEW


def source_summary(source: str | None = None) -> str:
    """The line worth printing at startup, because it names what is believed.

    `checks.MergePolicy.summary`'s job, and its rule: the line that reports a
    setting is read at the same call site that chose it, so a run's transcript
    cannot claim one thing while the loop does another.
    """
    chosen = state_source() if source is None else source
    if chosen == LABELS:
        return (
            "state source: the `swarm:*` labels ("
            f"{STATE_SOURCE_ENV}={LABELS}) - the pre-#147 behaviour, restored"
        )
    return (
        "state source: derived from the code host, the containers and apiary's own "
        f"store; the labels are written and compared but not believed "
        f"({STATE_SOURCE_ENV}={LABELS} to go back)"
    )


# --------------------------------------------------------------------------
# The belief
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Grant:
    """One revival, and whether the attempt it bought has been spent (#200).

    `attempt` is the counter the task was on when `planner.revive` returned it
    to the run - a *stamp* of when the grant was made, never a second copy of
    the counter, which is ADR 0002's "As built" rule and the reason this is not
    a number the store keeps. It is compared against the code host's own count,
    so a grant that produced a result lapses on the arithmetic it always did.

    `dispatched` is the half a result cannot suppress, and the whole of #200: a
    granted attempt that dies without writing a result record moves no counter
    at all, so `attempt` alone suppressed the give-up forever. The reconciler
    sets it from `DispatchReport` - the orchestrator's own record of having put
    a worker on the task - which is why it is here rather than derived from
    anything the worker leaves behind.

    Frozen because every value in this module is, and for the same reason: a
    belief, an override and a grant are all things one cycle *decided*, and a
    decision a later stage can edit in place is one two stages can disagree
    about. The map holding them is not frozen - `Reconciler` replaces entries in
    it as the run goes on - so the guarantee is per grant, not per map.
    """

    #: The attempt counter at the moment of the grant.
    attempt: int
    #: Has a dispatch since the grant spent the attempt it bought?
    dispatched: bool = False

    def spend(self) -> Grant:
        """The same grant, spent. Returns `self` unchanged when it already was."""
        return self if self.dispatched else replace(self, dispatched=True)


@dataclass(frozen=True)
class Override:
    """One task the orchestrator believes something other than its label.

    The observable record of the cutover, and the thing #147's acceptance
    criteria are really about: a hand-edited label produces one of these, the
    orchestrator carries on, and the disagreement is announced.

    `kind` is empty for a plain disagreement - the resolver simply read the
    world differently from what the label stored - and carries one of the five
    constants above when this module *itself* moved the answer off the
    resolver's verdict. Both are worth a line in `events.jsonl` and they are
    not the same event: the first is the label being stale, the second is
    apiary knowing something the code host cannot show. An accounted override
    is recorded even when it agrees with the label, because `budget-spent`
    usually does and it is the overlay a reader most needs to see.
    """

    task_id: str
    ref: TaskRef
    believed: str
    stored: str
    #: What the resolver said, before this module's overlays. Equal to
    #: `believed` unless `kind` is set.
    derived: str = ""
    kind: str = ""
    why: str = ""

    def __str__(self) -> str:
        head = f"{self.task_id} ({self.ref}): believed {self.believed}, label stores {self.stored}"
        return f"{head} [{self.kind}: {self.why}]" if self.kind else head


@dataclass(frozen=True)
class Belief:
    """What the orchestrator holds true about every task in one ledger.

    Frozen and folded rather than mutated, for `reconcile.fold`'s reason: a
    cycle writes labels part-way through and every stage after that write has
    to see the new state, so the belief advances by the transitions that
    actually **landed** and by nothing else. A belief advanced by a planned
    write GitHub refused would hand the dispatcher a view of a world that never
    existed, which is the one failure `fold` exists to prevent.
    """

    source: str = DERIVED
    #: task id -> the internal state this orchestrator acts on.
    states: Mapping[str, str] = field(default_factory=dict)
    #: task id -> the internal state the `swarm:*` label is storing. Kept so a
    #: divergence can be reported at the point of belief rather than only at
    #: the end of the cycle, where the shadow window takes it.
    stored: Mapping[str, str] = field(default_factory=dict)
    overrides: tuple[Override, ...] = ()
    #: task id -> what the orchestrator believed **last** cycle, with the label
    #: standing in for a task this process has never seen. `plan_reconcile`'s
    #: two edge-triggered rules read it; see this module's docstring for why no
    #: absolute reading of the world can replace it, and why seeding from the
    #: label is the honest answer until #152 moves the seam to the store.
    previous: Mapping[str, str] = field(default_factory=dict)
    #: The resolution the belief was built from, or `None` under `labels`.
    #: Retained for a caller that wants the verdict's own sentence; nothing
    #: here decides on it a second time.
    resolution: Resolution | None = None
    #: task id -> the work item that id named in the ledger this belief was
    #: built from. Not a second copy of the ledger: it is what `carried` keys
    #: this belief by, so what one cycle hands the next is filed under the work
    #: item rather than under a slug `ledger._adopted_id` only *leased* to it.
    refs: Mapping[str, TaskRef] = field(default_factory=dict)
    #: **The ratchet's whole memory** (#201, #214): the work items this run has
    #: believed landed. Keyed by `TaskRef` rather than by task id, which is what
    #: makes an id re-mint unable to reach it - the two are not the same name,
    #: and only one of them is durable. Written in exactly two places, both of
    #: them decisions rather than readings: `believe`'s own `LANDED` beliefs and
    #: `fold`'s transitions. The `UNRESOLVED` arm, which falls back to a label
    #: the resolver had no verdict for, has no write site here at all - so the
    #: rule "a label may decide a cycle, never the run" is a property of the
    #: type rather than a convention every future arm has to remember.
    landed: frozenset[TaskRef] = frozenset()
    #: The ratchet's *footprint in the log*, and deliberately not part of the
    #: decision above: work item -> the control-plane state `landed-stands` was
    #: last announced against. `landed-stands` re-tests nothing, so once it
    #: starts it fires for the life of the process, and an `Override` on every
    #: one of those cycles makes `artifacts.DivergenceTally.overrides` a measure
    #: of run length rather than of what the orchestrator did. It is quiet about
    #: the *same* disagreement rather than quiet forever (#215): a task the
    #: ratchet is holding whose label a human then moves is news, and it is the
    #: very case `hold`'s guard exists for. Separate from `landed` so that a
    #: question about the event log can never perturb the question about the
    #: merge; a key here is always a member of `landed`, never the reverse.
    announced: Mapping[TaskRef, str] = field(default_factory=dict)

    @property
    def derived(self) -> bool:
        """Is the resolver being believed at all? The one branch that differs."""
        return self.source == DERIVED

    def state(self, task_id: str) -> str:
        """This task's believed state, or `""` for a task not in the ledger.

        The empty string rather than a default, `derived.Resolution.state`'s
        rule: a task nothing here has an opinion about must not be handed one,
        and `blocked` would be an opinion.
        """
        return self.states.get(task_id, "")

    def holds(self, task_id: str, *states: str) -> bool:
        """Is this task believed to be in any of `states`? The readers' idiom."""
        return self.states.get(task_id, "") in states

    def fold(self, transitions: Iterable[Any]) -> Belief:
        """Advance the belief by the label writes that landed. `fold`'s rule.

        **Unguarded against `landed`, unlike `hold` ten lines below.** A
        `Transition` is apiary's own write *this* cycle, so it is the one input
        allowed to move a task the ratchet is holding. `hold`'s guard exists
        because the overlay reaching it is applied unconditionally, from a
        selection some earlier stage made - and since #212 that selection reads
        the belief, which is a better argument for the guard than the stale
        selector it replaced rather than a reason to drop it. Neither half of
        that argument reaches here.

        `Transition.to_label` is apiary's own decision about a task, translated
        through the same table `lifecycle.py` announces with - not a label read
        back off the code host. The distinction matters: this is the cycle
        learning what it just did, which is the only thing the label vocabulary
        is still used for here.

        **This is the ordinary way into the ratchet**, and the one every test in
        §8 of `tests/test_authority.py` but one skips: the merge gate applies a
        `swarm:done` transition, this records it, and `Reconciler._carry_forward`
        hands the result to the next cycle - which is the cycle where the merged
        pull request has already left the open listing and the issue is not yet
        closed. A `Transition` carries its own `ref`, so the set is filled from
        the work item apiary just wrote to rather than from a lookup.
        """
        applied: dict[str, str] = {}
        landed = set(self.landed)
        for transition in transitions:
            task_id = getattr(transition, "task_id", "")
            if not task_id:
                continue
            state = _internal(transition.to_label)
            applied[task_id] = state
            ref = getattr(transition, "ref", None)
            if ref is None:
                continue
            # Symmetric on purpose, and the asymmetry would be the bug: `fold`
            # is the one input allowed to move a task the ratchet is holding,
            # so a transition off `landed` has to take the memory with it or
            # the two would disagree about the same task.
            if state == LANDED:
                landed.add(ref)
            else:
                landed.discard(ref)
        if not applied:
            return self
        return replace(
            self,
            states={**self.states, **applied},
            stored={**self.stored, **applied},
            landed=frozenset(landed),
            announced={
                ref: state for ref, state in self.announced.items() if ref in landed
            },
        )

    def waiting(self) -> frozenset[str]:
        """The task ids `github/readiness.py` may speak about. See its docstring.

        Under `labels` this is exactly `readiness.TRANSITIONABLE` translated,
        which is what that module was computing for itself before #147 - the
        flag restores the behaviour rather than approximating it.
        """
        return frozenset(task for task, state in self.states.items() if state in WAITING)

    def hold(self, states: Mapping[str, str]) -> Belief:
        """Overlay states this cycle produced through something other than a
        transition. The three writers `fold` cannot see.

        `shadow.control_labels` enumerates them and this is the same list from
        the other side: the readiness pass's verdicts, the dispatcher's claims
        and `planner.revive`, none of which is a `Transition` and none of which
        is folded back into the ledger. They matter here for one reason -
        `Reconciler` carries the belief forward as next cycle's *previous*, and
        a task the dispatcher claimed this cycle whose worker exits before the
        next one would otherwise be remembered as never having run.
        """
        if not states:
            return self
        # **A merge is the one fact in this map nothing may take back** (#201).
        # `Reconciler._carry_forward` builds its overlay as
        # `{task: ELIGIBLE for task in revived_tasks(report)}` and applies it
        # unconditionally, so a revival that reached a task this run believes
        # landed would silently clear the ratchet the module docstring exists to
        # justify - and the next cycle's resolver, which cannot see a merged pull
        # request, would read `eligible` and put a worker on work already on the
        # default branch.
        #
        # **No revival reaches here any more** (#212): `nodes/planner._update`
        # selected on `entry.state_label == FAILED`, which was the whole of how a
        # `swarm:done` issue relabelled `swarm:failed` mid-run got revived, and
        # it now selects on `authority.state_of` like every other decision path.
        # The guard stays regardless, and not out of habit - it is the only thing
        # standing between an overlay written as an unconditional dict
        # comprehension and a fact this system treats as unundoable. A guard
        # whose argument for existing is "a caller might" is worth less than one
        # whose argument is "this map may not lose a merge", which is what it now
        # has.
        #
        # The guard is on `states` alone, not on `stored`. `planner.revive` does
        # write `swarm:ready` to the issue, and that write is real: the label
        # moved, the belief did not, and reporting both is what makes the next
        # cycle's `state.override` say something true.
        held = {
            task: state
            for task, state in states.items()
            if self.states.get(task) != LANDED
        }
        # `landed` and `announced` ride through untouched, which is the whole of
        # what this method does about the ratchet: none of the three writers
        # here decides that a task merged - readiness verdicts are `ready` or
        # `blocked`, a claim is a claim, and a revival is the thing the guard
        # above exists to refuse - so there is no arm that could seed it, and
        # the guard keeps `states` and the set saying the same thing.
        return replace(
            self,
            states={**self.states, **held},
            stored={**self.stored, **states},
        )

    def carried(self) -> dict[TaskRef, str]:
        """This belief keyed by work item, for the next cycle's `remembered`.

        Keyed by ref rather than by task id because that is the only durable
        name a task has: `ledger._adopted_id` leases the bare slug to the first
        taker in issue order, so an id can name a different issue next cycle -
        and `previous` is what `plan_reconcile`'s two edge-triggered rules read.
        Left keyed by id, a re-minted slug inherited the departed issue's
        `review`, rule 4 found a pull request that was never its own missing
        from the open listing, and `_retry_or_give_up` charged an attempt to a
        brand-new issue. Re-keying it here means `believe` cannot look the wrong
        memory up rather than having to notice that it did.

        A task in `states` but not in `refs` is an overlay against a ledger this
        belief was not built from - no caller in the loop produces one - and it
        is dropped rather than given a ref that would be a guess.
        """
        return {
            ref: self.states[task_id]
            for task_id, ref in self.refs.items()
            if task_id in self.states
        }

    def summary(self) -> str:
        if not self.derived:
            return f"state source: {self.source} ({STATE_SOURCE_ENV}={LABELS})"
        explained = sum(1 for one in self.overrides if one.kind)
        return (
            f"state source: the resolver over {len(self.states)} task(s), "
            f"{len(self.overrides)} disagreeing with the label "
            f"({explained} accounted for)"
        )


# --------------------------------------------------------------------------
# Believing
# --------------------------------------------------------------------------


def believe(
    ledger: Ledger,
    observation: Observation | None,
    *,
    source: str = DERIVED,
    infrastructure: Mapping[TaskRef, int] | None = None,
    infrastructure_cap: int = 3,
    revived: Mapping[TaskRef, Grant] | None = None,
    remembered: Mapping[TaskRef, str] | None = None,
    landed: frozenset[TaskRef] | None = None,
    announced: Mapping[TaskRef, str] | None = None,
    max_attempts: int = 3,
    max_total_attempts: int = 9,
) -> Belief:
    """One cycle's belief about every entry in the ledger. Pure.

    `observation=None` means the resolver was not asked - a cycle that could
    not list pull requests, or `source=labels`. It is **not** an empty
    observation: `checks.read_pulls` and `shadow.ShadowWindow.run` both go to
    lengths to keep "could not look" apart from "nothing there", and the cost of
    conflating them is one level worse here than it is in the shadow. A cycle
    that read no pull requests and believed the resolver anyway would resolve
    every task in review to `eligible` and **dispatch a second worker over an
    open pull request's file set**. So a blind cycle falls back to the labels
    wholesale and says so, which is the same answer `plan_reconcile` already
    gives its own blind rules: they did not run; they did not fail.

    `remembered` is what this process believed last cycle, keyed by the work
    item rather than by the task id it happened to be filed under - see
    `Belief.carried`. It becomes `Belief.previous`, seeded from the labels for a
    task never seen, which is the one place a label still reaches a decision and
    is argued in the module docstring.

    `landed` is a **separate** input, and the separation is the point (#214).
    The ratchet decides once and forever, so it may read only what this process
    genuinely decided; `previous` is what `plan_reconcile`'s two edge-triggered
    rules ask, and both of those re-decide every cycle, so a label standing in
    for a missing memory costs at most one wrong edge there. Carrying the
    ratchet's memory as its own set of refs is what makes that difference
    structural: the states a label fell into cannot be mistaken for a decision,
    because they are not in the set - the `UNRESOLVED` arm below never writes to
    it. The one exception is the first cycle a task is seen at all, where the
    label is the only record that exists and `docs/issue-contract.md` §4 records
    what that costs.

    `announced` is the ratchet's log memory and decides nothing; see
    `Belief.announced`.

    `revived` is the run's grants (#200): the attempt each revival was made at,
    and whether the one attempt it bought has been spent. A bare attempt number
    was the shape before #200 and is deliberately **not** still accepted - it
    means "granted, and no dispatch can ever spend it", which is the defect, and
    leaving it legal would let a later caller reopen it with no type complaining.
    """
    infrastructure = infrastructure or {}
    grants = dict(revived or {})
    entries = sorted(ledger.entries.values(), key=lambda entry: entry.ref)

    by_label = {entry.task_id: _internal(entry.state_label) for entry in entries}
    refs = {entry.task_id: entry.ref for entry in entries}
    held_by_ref = dict(remembered or {})
    was_landed = frozenset(landed or ())
    spoken = dict(announced or {})
    # A memory is looked up by the work item it was about, never by the id it
    # was filed under (#201's AC 2, applied in full; #214 makes it structural).
    # `ledger._adopted_id` leases the bare slug to the first taker in issue
    # order, so an id can be re-minted for a *different* issue - and `previous`
    # is what `plan_reconcile`'s edge-triggered rules read through `_was`. Keyed
    # by id, a re-minted slug inherited the departed issue's `review`, rule 4
    # saw a pull request that was never its own missing from the open listing,
    # and `_retry_or_give_up` charged an attempt to a brand-new issue -
    # escalating it to `swarm:failed` at the cap. Keyed by ref there is nothing
    # to drop: the new issue asks about itself and is answered about itself.
    #
    # The consequence is chosen rather than inherited: a re-minted id falls back
    # to the cycle-0 label seed, because a re-minted id **is** a first sight.
    # That is the same answer `docs/issue-contract.md` §4 already gives for the
    # human case - "a reopened issue is new work with a new id".
    seen = {
        entry.task_id: held_by_ref[entry.ref]
        for entry in entries
        if entry.ref in held_by_ref
    }
    previous = {**by_label, **seen}

    if source != DERIVED or observation is None:
        # `previous` is the labels alone here, deliberately, and not the
        # `remembered` overlay the derived path uses.
        #
        # `APIARY_STATE_SOURCE=labels` has one job: restore what the
        # orchestrator did before #147. Before #147 `plan_reconcile`'s rules 3
        # and 4 read `entry.state_label` directly, every cycle. Carrying last
        # cycle's belief over this cycle's label makes the remembered value win,
        # and the one event that distinguishes them is a human editing a label
        # mid-run - which is the single case the hatch exists for, and the
        # action apiary's own give-up comment instructs ("move this back to
        # `swarm:ready`"). Rule 4 would then fire on the remembered `review`,
        # consume an attempt and post a failure for work a human had just
        # rescheduled.
        #
        # `landed` and `announced` ride through unchanged rather than being
        # dropped, and that is a deliberate strengthening (#214). Under
        # `source=labels` nothing below reads them, so it is inert; the case it
        # is not inert for is the *blind* cycle, where a run that could not list
        # pull requests once used to lose every merge it had ratcheted for the
        # rest of the process. "This cycle could not look" is not evidence that
        # a merge was undone - the same sentence the no-verdict arm below is
        # already decided by.
        return Belief(
            source=LABELS,
            states=dict(by_label),
            stored=by_label,
            previous=dict(by_label),
            refs=refs,
            landed=was_landed,
            announced=spoken,
        )

    resolution = resolve(observation)
    # The same observation with the budget rule suppressed, so a task whose
    # budget apiary has not in fact spent can be given the state it would
    # otherwise have had - `claimed` while its revived attempt runs, `review`
    # while its pull request is open - rather than a bare fallback to the
    # label. Two resolutions over one input, which is the only comparison that
    # says anything about either (`shadow.py`'s rule, one level along).
    lenient = resolve(
        replace(
            observation,
            budget=Budget(max_attempts=_UNBOUNDED, max_total_attempts=_UNBOUNDED),
        )
    ).by_task
    verdicts = resolution.by_task

    states: dict[str, str] = {}
    stored: dict[str, str] = {}
    # The next cycle's ratchet, and **the only two places it is written** are
    # the arms below that decide `landed` - not this cycle's `states`, which a
    # label can reach. Deriving it at the end from `states == LANDED` would put
    # the `UNRESOLVED` fallback straight back into it.
    still_landed: set[TaskRef] = set()
    spoke: dict[TaskRef, str] = {}
    overrides: list[Override] = []
    cap = max(int(infrastructure_cap), 0)

    for entry in entries:
        # Shadowing the module-level `label_state` here would be a name a reader
        # has to disambiguate in the one function that holds both sides.
        was_stored = _internal(entry.state_label)
        stored[entry.task_id] = was_stored
        # **The ratchet's one input**, and the two problems #201 found in it are
        # both in this expression rather than in the arm below.
        #
        # A task this process has never carried is decided by the label, and
        # that is a policy choice rather than an oversight - `docs/issue-contract.md`
        # §4 records it beside the two `any -> ...` human rows. `swarm:done` on
        # an open issue is produced by three different things and a fresh
        # process cannot tell them apart: the window in which
        # `checks._decide_passed` has written the label before GitHub honoured
        # `Closes #<n>`, a pull request merged without the keyword at all, and a
        # human reopening finished work. Two of the three must not be
        # dispatched, so a seed that honoured the third would put a worker back
        # on merged code in the other two - which is the hole the ratchet was
        # written for, and it was reproduced. §4 already answers the human:
        # "a reopened issue is new work with a new id".
        #
        # A task it *has* carried asks the set, which is narrower than
        # `previous` in the two ways that matter and is narrower by construction
        # rather than by inspection: a state that was never a belief is not in
        # it (the `UNRESOLVED` arm's label fallback, just below, writes only to
        # `states`), and a state carried under an id that now names a different
        # work item cannot be found by it (`entry.ref`, not `entry.task_id`).
        landed_before = (
            entry.ref in was_landed
            if entry.ref in held_by_ref
            else was_stored == LANDED
        )
        # Has the ratchet already said its piece about this work item, and about
        # this same label? It re-tests nothing, so it fires every cycle for the
        # rest of the process once it starts - and an `Override` per cycle per
        # task is what makes `DivergenceTally.overrides` climb monotonically for
        # a run that is behaving exactly as designed. See `Belief.announced`.
        spoken_against = spoken.get(entry.ref)
        verdict = verdicts.get(entry.task_id)
        if verdict is None:
            if landed_before:
                # The ratchet outranks the fallback for the same reason it
                # outranks every overlay below: "the resolver said nothing this
                # cycle" is not evidence that a merge was undone, and letting a
                # silent cycle drop the belief would put the ratchet back at the
                # mercy of whatever the label happens to read next.
                states[entry.task_id] = LANDED
                still_landed.add(entry.ref)
                spoke[entry.ref] = was_stored
                if spoken_against != was_stored:
                    overrides.append(
                        Override(
                            task_id=entry.task_id,
                            ref=entry.ref,
                            believed=LANDED,
                            stored=was_stored,
                            kind=LANDED_STANDS,
                            why=(
                                "this run has already seen this task land, and a merge "
                                "is terminal within a run (`docs/issue-contract.md` §4). "
                                "The resolver returned no verdict this cycle, which is "
                                "not evidence that the merge was undone."
                            ),
                        )
                    )
                continue
            # **The arm with no write to `still_landed`, and that is the whole
            # of the guarantee.** This is the label standing in for a verdict
            # that does not exist, and the ratchet reads only decisions. Were
            # this state able to seed it, a `swarm:done` a human typed onto a
            # task the resolver has no opinion about would enter next cycle's
            # ratchet as `landed` and pin the task for the life of the process -
            # a label reaching a permanent decision outside the seam this
            # module's docstring bounds, which is the one thing #147 says must
            # not happen. Before #214 the exception was a convention - the arm
            # wrote a bare `str` where its neighbours minted a `Remembered`, and
            # `_unbelieved` existed to make the omission greppable. A fifth arm
            # would have had to remember to opt out; there is now nothing to opt
            # out of.
            states[entry.task_id] = was_stored
            overrides.append(
                Override(
                    task_id=entry.task_id,
                    ref=entry.ref,
                    believed=was_stored,
                    stored=was_stored,
                    kind=UNRESOLVED,
                    why=(
                        "the resolver returned no verdict for this task, so there is "
                        "nothing to believe instead of the label. Counted rather than "
                        "silent: a cutover that fell back for every task would look "
                        "exactly like a clean one."
                    ),
                )
            )
            continue

        believed, kind, why = verdict.state, "", ""
        grant = grants.get(entry.ref)
        spent = budget_spent(
            entry,
            verdict.attempts_spent,
            grant,
            max_attempts=max_attempts,
            max_total_attempts=max_total_attempts,
        )

        if believed != LANDED and landed_before:
            # Ahead of every other overlay, because it is the only one that is
            # about a fact nothing can undo. It also removes a transient the
            # merge gate would otherwise produce every time it lands a task:
            # `swarm:done` is written *before* GitHub has honoured `Closes #<n>`
            # (`checks._decide_passed` says why), so the merged pull request has
            # already left the open listing while the issue still reads open.
            believed, kind = LANDED, LANDED_STANDS
            why = (
                "this run has already seen this task land, and a merge is terminal "
                "within a run (`docs/issue-contract.md` §4). The merged pull request "
                "is not in the open listing and the work item is not closed as "
                f"completed, so the resolver reads {verdict.state} - which would put a "
                "worker back on work that is already on the default branch."
            )
        elif cap and believed != LANDED and infrastructure.get(entry.ref, 0) >= cap:
            streak = infrastructure.get(entry.ref, 0)
            believed, kind = NEEDS_HUMAN, INFRASTRUCTURE_CEILING
            why = (
                f"{streak} consecutive infrastructure verdict(s) against a cap of "
                f"{cap}. Exit 2 does not bump the attempt, so N mechanical failures "
                "write one result filename and no artifact can count them - ADR 0001's "
                f"first non-derivable state. The resolver reads {verdict.state}."
            )
        elif believed in WAITING and spent:
            # The store decides `needs-human`, and the resolver's arithmetic is
            # a lower bound it falls back on. Bounded to the *waiting* states on
            # purpose: a live container or an open pull request is stronger
            # evidence about now than a budget row is, `landed` outranks
            # everything, and apiary's own judgment is only ever being asked to
            # choose between "wait" and "a human, please".
            believed, kind = NEEDS_HUMAN, BUDGET_SPENT
            why = (
                f"apiary gave up on this task (streak={entry.streak}, "
                f"attempt={entry.attempt}) against a cap of {max_attempts}, and the "
                f"code host accounts for only {verdict.attempts_spent} attempt(s) - "
                "a failed task whose run directory is gone and whose pull requests "
                f"are closed leaves none. The resolver reads {verdict.state}."
            )
            if grant is not None and grant.dispatched:
                # The one line that tells a reader which of the two ways a
                # revival ends they are looking at, and the only visible trace
                # of #200's lapse: the code-host count is *still* sitting where
                # the grant was made, and the task is escalating anyway.
                why += (
                    f" The revival granted at attempt {grant.attempt} has lapsed: a "
                    "worker was dispatched on it, and a dispatch spends the one "
                    "attempt a revival buys whether or not it leaves a result."
                )
        elif believed == NEEDS_HUMAN and not spent:
            lenient_verdict = lenient.get(entry.task_id)
            believed = lenient_verdict.state if lenient_verdict is not None else ELIGIBLE
            # `not grant.dispatched`, because a spent grant is a tombstone
            # rather than a live one: a task whose budget is renewed later by a
            # new blocker signature is a `budget-renewed`, and explaining it with
            # a revival that lapsed cycles ago would name the wrong mechanism on
            # the one log that exists to tell the two apart.
            outstanding = grant if grant is not None and not grant.dispatched else None
            kind = REVIVED if outstanding is not None else BUDGET_RENEWED
            why = (
                f"the code host accounts for {verdict.attempts_spent} attempt(s) "
                f"against a cap of {max_attempts}, but apiary's own record says the "
                f"budget is not spent (streak={entry.streak}, attempt={entry.attempt}"
                + (f", revived at attempt {outstanding.attempt}" if outstanding else "")
                + f"). `_retry_or_give_up` gives up on the streak, so {believed} stands."
            )

        states[entry.task_id] = believed
        # The other of the two write sites, and it is a *decision* about this
        # work item rather than a reading of its label: either the resolver said
        # `landed` or the ratchet above did. Both are things this process
        # believed, which is exactly what the next cycle's ratchet is allowed to
        # stand on.
        if believed == LANDED:
            still_landed.add(entry.ref)
        if kind == LANDED_STANDS:
            spoke[entry.ref] = was_stored
        # Recorded when the belief differs from the label **or** from the
        # resolver, and `kind` is what tells a reader which. The second half is
        # not decoration: `budget-spent` usually restores exactly what the label
        # already said, and an event log that only reported disagreements with
        # the label would show nothing at all for the one overlay that keeps a
        # resumed run from resurrecting abandoned work.
        #
        # The one exception is the ratchet once it is already standing, and it
        # is an exception about *repetition* rather than about importance.
        # `landed-stands` re-tests nothing, so it reports the same sentence
        # about the same task on every remaining cycle of the process - and
        # `artifacts.DivergenceTally.overrides` counts events rather than tasks.
        # Announced on the cycle it starts standing, then quiet.
        #
        # It is the worst case rather than the only one: `budget-spent` and
        # `infrastructure-ceiling` re-read inputs that, for a task which stays
        # given up, cannot change - so they repeat too. #201 scoped only the
        # ratchet and this comment does not claim more than it fixed.
        # Quiet about the *same* disagreement, not quiet forever: a task the
        # ratchet is holding whose label a human then changes is news, and it is
        # exactly the case `hold`'s guard exists for.
        muted = kind == LANDED_STANDS and spoken_against == was_stored
        if (believed != was_stored or kind) and not muted:
            overrides.append(
                Override(
                    task_id=entry.task_id,
                    ref=entry.ref,
                    believed=believed,
                    stored=was_stored,
                    derived=verdict.state,
                    kind=kind,
                    why=why or verdict.because,
                )
            )

    return Belief(
        source=DERIVED,
        states=states,
        stored=stored,
        previous=previous,
        overrides=tuple(overrides),
        resolution=resolution,
        refs=refs,
        landed=frozenset(still_landed),
        announced=spoke,
    )


def budget_spent(
    entry: LedgerEntry,
    attempts_spent: int,
    grant: Grant | None,
    *,
    max_attempts: int,
    max_total_attempts: int,
) -> bool:
    """Has apiary in fact spent this task's retry budget? `_retry_or_give_up`'s test.

    Deliberately the same two comparisons that function makes, against the same
    two numbers, rather than a second opinion about them: a give-up rule and the
    state that reports it disagreeing would be a task escalated by one and
    dispatched by the other, forever.

    **Public because the console board is the second caller** (#158's review).
    A board projecting the resolver alone reported `needs-human` on a task whose
    per-blocker budget apiary had *renewed* - a verdict the machine had already
    withdrawn, on the one element that ticket says must never hide. The renewal
    is in the store, so the board can read it; what the board must not do is
    make its own version of this test, which is the same argument the paragraph
    above makes one caller earlier. `grant=None` is the honest answer from a
    caller with no run memory: a `Grant` is `Reconciler` state that lapses with
    the process, so a board cannot know a revival happened and reads the task as
    capped - escalating rather than granting budget, which is the direction a
    projection should fail in.

    The hard total cap is checked against the **code host's** count, because
    `max_total_attempts` bounds the task rather than one blocker and the
    counter it runs on is monotonic - a lower bound from a branch or a result is
    a lower bound on it too, and over-counting there gives up sooner.

    A revival grants exactly one attempt and lapses when it is spent, which is
    the whole of what `planner.revive` does: it "resets nothing", so the moment
    the granted attempt produces a result the streak it never reset caps the
    task again.

    **Spent has two readings and only one of them is the code host's** (#200).
    A result carries the granted attempt and `attempts_spent` moves past it,
    which is the reading above. `Grant.dispatched` is the other, for the attempt
    that writes no result at all; the module docstring argues why it is a
    dispatch rather than a third reading of the code host, and why no other
    counter bounds that input. A dispatched grant stops suppressing and falls
    through to the streak test, which is the same place the first reading lands -
    so a revival that produced a result is decided by exactly the arithmetic it
    always was.

    The fallback for a task the store has never judged is ADR 0002's own -
    `previous_streak = entry.attempt if entry.streak is None else entry.streak`,
    quoted because that document names simplifying it to `0` as the change that
    would open the hole it thought it had closed.
    """
    cap = max(int(max_attempts), 1)
    total_cap = max(int(max_total_attempts), cap)
    # Two counters, because they are not the same number after a restart and
    # only one of them survives one.
    #
    # `attempts_spent` is the **code host's** count, and this module's own
    # docstring says why it resets: results are per-run and the observation
    # takes branch names off *open* pull requests, so a task apiary gave up on
    # resolves to `eligible` from scratch in the next process. `entry.attempt`
    # is apiary's own monotonic counter, carried in the issue marker and the
    # store, and it is the one `_retry_or_give_up` actually gives up on.
    #
    # Testing only the code-host count left one of that function's two branches
    # uncovered, and precisely the branch a restart reaches. A *streak* give-up
    # survives, because `entry.streak` is stored. A *total-cap* give-up did not:
    # reaching `max_total_attempts` without the streak reaching `max_attempts`
    # is exactly what renewals produce - every new failure signature resets the
    # streak to 1 - so `streak >= cap` was False, nothing said the budget was
    # spent, and the task was relabelled `swarm:ready` and dispatched with a
    # fresh budget over work apiary had already abandoned.
    if attempts_spent >= total_cap or int(entry.attempt) >= total_cap:
        return True
    if grant is not None and not grant.dispatched and attempts_spent <= grant.attempt:
        return False
    streak = entry.attempt if entry.streak is None else entry.streak
    return int(streak) >= cap


def label_state(label: str) -> str:
    """The internal state a `swarm:*` label stores. `lifecycle.internal_state`.

    Public because two modules need it *without* being able to import
    `lifecycle` - `reconcile`, which `lifecycle` imports, and this one. It is the
    translation, not a reading: a caller reaching for it is either advancing a
    belief by a write apiary just made or falling back to the labels on purpose.
    """
    return _internal(label)


def _internal(label: str) -> str:
    """`lifecycle.internal_state`, reached through a local import.

    `lifecycle` imports `reconcile` and `reconcile` imports this module, so a
    top-level import here would close the ring. The same reason `reconcile`
    reaches `checks` and `mergeability` from inside `cycle`.
    """
    from .lifecycle import internal_state

    return internal_state(label)


def revived_tasks(report: Any) -> frozenset[str]:
    """Tasks `planner.revive` returned to `swarm:ready` during one cycle.

    Two callers reach `revive` and both run inside `Reconciler._judge`: the
    replanner through `_update`, and the goal gate through
    `_revive_abandoned`. Neither result is folded into the ledger, so this is
    the only account of them.

    Read by two modules for two purposes and therefore living below both:
    `orchestrator/shadow.py` needs it to build the control map a cycle actually
    left behind, and `Reconciler` needs it to record the one attempt a revival
    grants. Typed loosely because a `CycleReport` lives in `reconcile`, which
    imports this module.
    """
    found: set[str] = set()
    for source in (getattr(getattr(report, "replanned", None), "plan", None), getattr(report, "goal", None)):
        for action in getattr(source, "revived", ()) or ():
            task = getattr(action, "task_id", "")
            if task:
                found.add(str(task))
    return frozenset(found)
