"""The derived resolver, run beside the labels and compared with them (#146).

**Since #147 the comparison runs in reverse.** This module's title used to end
"and believed by nothing"; `orchestrator/authority.py` now takes the resolver's
answer and decides on it, and the labels are the side being checked. Nothing in
*this* file changed direction - it still reads only what the cycle already read,
still cannot fail a cycle, and still decides nothing - but what a divergence
here *means* did: it used to say "the derived value is not ready to be
believed", and it now says "the label has drifted from what the orchestrator
acted on". Both are worth an event and they are not the same claim, which is
why `authority.py` emits its own at the point of belief rather than reusing
this one.

`orchestrator/derived.py` computes ADR 0001's five lifecycle states from the
code host, the containers and the run artifacts. #145 proved that reducer
self-consistent against a replay corpus - **and every run in that corpus was
synthesised**, because no credential existed to record a real one
(`tests/fixtures/runs/README.md` says so at length). A green replay proved the
reducer agrees with itself. It proved nothing about whether the reducer's model
of reality matches reality.

This module is where that question is asked. Every cycle, it builds the
`Observation` out of facts the cycle **has already read**, resolves it, diffs
the answer against what the control plane ended the cycle holding, and writes
the disagreements into `events.jsonl`. Nothing here is read by any decision -
that is `authority.py`'s job since #147, and this window is what a reader
checks it against.

**#147 shipped before this window's own gate was met, deliberately.** The
acceptance criterion was ten consecutive greenfield runs with zero unexplained
divergences, and no credential in this environment can run one - the same wall
#145's corpus README, #146 and `docs/demo-run.md` all hit. So the cutover is
guarded by `APIARY_STATE_SOURCE=labels` instead, which restores this module's
world in full. That is a weaker guarantee than a clean window and it is written
down here rather than left for whoever reads the first real run.

## Three properties, and each one is a way this could have been worthless

**It never affects the cycle.** `ShadowWindow.run` catches every exception,
including the ones that mean this module is broken, and reports the failure on
stderr rather than through the loop. A shadow that raised would turn an
observation into an outage; a shadow that retried or slept would change the
pacing it is meant to be measuring. It observes.

That swallow has a cost worth naming: a genuine defect here reads as a shadow
that quietly stopped reporting, which is silence in the direction of the bug.
So `ShadowWindow.broken` is a field rather than a local, and
`tests/test_shadow.py` asserts it is false after a cycle that has been through
dispatch, a result, review, a check set and a merge. It is not hypothetical -
the first version of this module read `PullState.number` as an `int`, #185 had
just made it a `PullRef`, and the only symptom was a line on stderr.

**It costs no API call.** Everything it reads is passed in by
`Reconciler.cycle` from that cycle's own reads - the handles from one
`docker ps`, the pull requests from the listing `Snapshot` already forced, the
result records from the directory the cycle already globbed, the ledger from
the one issue listing. Nothing here calls GitHub. `tests/test_shadow.py`
asserts the call count is unchanged, because a shadow that silently doubled the
rate-limit spend would be switched off by the first person it throttled.

**It diffs two answers to one observation, not two samples of one clock.** See
"Which control plane, sampled when" below; this is the part a reader of the log
has to understand and the part the log cannot tell them. "What a clean window is,
and is not, evidence of" is the other part, and it is why every cycle reports how
many of its comparisons were independent of the cycle's own writes.

## Which control plane, and sampled when

The obvious comparison is wrong. The labels *as the cycle read them* are last
cycle's answer to last cycle's world: a worker that exited during the interval
has written its record and opened its pull request, and the label still says
`claimed` because nothing has looked yet. Diffing against that reports "the
label is a lagging cache" - which is true, is ADR 0001's whole thesis, and
would fire for **every task in every run**, drowning the disagreements that
mean something in the one that means nothing.

So the diff is against the control plane **as the cycle left it**: the ledger
folded with what `apply_plan`, the recovery sweep and the check gate actually
applied, plus the readiness pass's verdicts and the dispatcher's claims. That
is the label machine's answer to the same observation this resolver was handed,
and comparing two reducers over one input is the only comparison that says
anything about the reducers.

**One consequence, and it is the fourth expected divergence.** A cycle acts
after it reads. The merge gate merges a pull request that was open when the
world was sampled, so on the cycle a task lands the control plane says `landed`
and the derived side - reading the pre-merge listing - says `review`. The same
goes for a task the dispatcher claims after the container listing was taken.
Neither is a disagreement about a fact; both are the cycle's own action
outrunning its own read. They are classified and named rather than hidden,
because the alternative - feeding "what this cycle wrote" back into the
observation - is exactly the sourcing violation `derived.py` exists to prevent.

## What a clean window is, and is not, evidence of

This is the question #152 will be answered with, so it is written down rather
than left for whoever reads the first clean run.

`plan_reconcile` computes this cycle's label writes from `snapshot.states()`,
`snapshot.open_branches()`, the results directory and the container listing.
The resolver reads the same four. **So for a task this cycle relabelled, the two
sides were fed the same observation**, and their agreeing shows that two
reducers implement the same rules over one input. That is not nothing - it is
exactly the statement "the label is a redundant cache", which is what #152 needs
- but it is not the resolver tracking a world nobody told it about, and a reader
who counted those as independent confirmations would be over-reading a clean
window.

The independent comparisons are the tasks the cycle did **not** write. There the
label is the accumulation of every earlier cycle's decisions over earlier
observations, and the derived state is one absolute reading of now; nothing
shared produced them. `plan_reconcile` is incremental (current label plus facts
-> a transition) and the resolver is absolute (facts -> a state), so agreement on
an untouched task is the accumulated state matching a recomputation from
scratch, which is the property that makes the cache deletable.

The share is structural rather than a matter of luck: a cycle writes labels for
at most the dispatch cap, plus `APIARY_MERGES_PER_CYCLE`, plus whatever
reconcile and the gates moved - a small constant - out of a ledger of N tasks.
So the independent share tends to `(N - O(1)) / N` and rises with plan size. It
is not assumed, though: `ShadowReport.independent` counts it per cycle, every
`state.shadow` event carries it, and `swarm show` prints the run's total beside
the compared total. **A clean window whose independent count is small is a weak
result and now says so on its own face.**

## Expected divergences, and why they are classified rather than suppressed

ADR 0001's section "Three of these are not derivable from the code host alone"
names three states the resolver cannot reach, and `derived.py` argues each one.
A shadow window that reported them as failures would be unreadable within one
run; a shadow window that filtered them out would be a harness tuned until it
agreed, which is the failure #145's corpus went to some trouble to avoid.

So every divergence is *classified*, and the classification carries the
argument. `Explained.kind` is empty for the ones nobody has an account of, and
**the unexplained count is the number the go/no-go reads.** The seven kinds:

| Kind | The account |
|---|---|
| `infrastructure-ceiling` | `infrastructure_streaks` counts transitions and exit 2 does not bump the attempt, so N mechanical failures write one result filename. Not in the artifacts at all. |
| `budget-renewed` | `_retry_or_give_up` gives up on `streak`, not `attempt`, and a renewal is an ADR 0002 store judgment. |
| `revived` | `planner.revive` "resets nothing", so the counter reads spent while the label reads ready. Converges on merge. |
| `merged-this-cycle` | The merge gate landed it after the world was read. |
| `dispatched-this-cycle` | The dispatcher claimed and spawned after the container listing was taken. |
| `container-created` | A container between `docker create` and `docker start`. See below. |
| `closed-not-planned` | **Closed by #147.** A human closed the work item as not planned; `reconcile._closed_verdict` escalated and `derived.py` had no rule for it. `TaskFact.abandoned` now derives it, so this should never fire again - the rule is kept because a divergence that reappears here means `state_reasons` stopped reaching the observation, which is worth a named line rather than an unexplained one. |

The last one was the only entry here that was a **gap rather than a finding**:
it is derivable - `TaskFact.state_reason` carries the fact - and `derived.py`
simply had no rule reading it. #147 added the rule (`TaskFact.abandoned`), so
the two sides now agree and no divergence is raised at all. The classification
stays, unreachable, for the reason a `container-created` line would be news: if
one ever appears, something stopped passing `state_reasons` into the
observation, and that is a defect worth naming rather than one to discover in
the unexplained count.

### The create-to-start window, decided explicitly

`Handle.running` is `state == RUNNING_STATE`, and a container between
`docker create` returning and `docker start` taking effect reads `created`, so
`running` is false. `recovery.py` notes that `docker ps --all` lists a
container from the instant `docker create` returns, so the window is real.

**This module reads liveness, not existence**, and that is a decision rather
than an inheritance. `dispatcher.release` takes the opposite position - any
container at all blocks a release - and is right to: it is deciding whether to
*act*, a `docker start` whose result this process failed to read may have
started a container that already pushed, and "existence blocks" is the only
reading that cannot produce two workers over one file set. This module decides
nothing, so it takes the reading that is *true* rather than the one that is
safe: ADR 0001's `claimed` is "a live worker container … falsifiable with
`docker ps`", and #187 was merged precisely because reading existence held
every task in `claimed` from the moment its worker exited until the reaper
arrived - a spurious divergence on every task in every run.

The cost is that the create-to-start gap reads as not-claimed, so it is the
sixth kind above. In practice it should never fire: `ContainerManager.spawn`
calls `create` and `start` in one breath and the cycle's container listing is
taken at the *top* of a cycle, so a container created by this cycle's dispatch
is `dispatched-this-cycle` instead. If `container-created` ever appears in a
log, something is spawning outside `spawn`, and that is news rather than noise.

## Why a `state.shadow` event exists as well

An events log with no `state.divergence` line in it is ambiguous in the worst
possible direction: it is either a clean shadowed run or a run where the flag
was off. #147's gate is "ten consecutive runs with zero unexplained
divergences", and a gate that cannot tell zero from unmeasured is a gate that
passes on nothing. So every shadowed cycle emits `state.shadow` carrying the
task count it compared, and `swarm show` reports "not run" when there are none.
That count is also the coverage number `derived.diverge`'s docstring declines
to compute, for its stated reason: a task on one side and not the other is not
a divergence, and the honest place for "how many were even compared" is here.

## Recording, which is the cheapest thing in this file and the most valuable

`tests/fixtures/runs/README.md`: four of the five files a corpus run needs are
"produced verbatim by a live run today", and `observed.jsonl` is the fifth. It
is not new state - every field in it is something the cycle already read - so
the recorder is a projection of an `Observation` this module has already built,
plus the labels it has already collected. Thirty lines.

Thirty lines that turn every shadowed run into a replay corpus run. The largest
risk in epic #140 is that the corpus proving the resolver is entirely
synthesised; dropping a run directory into `tests/fixtures/runs/` retires it,
and this is what makes that a `cp -r` rather than a project. The manifest is
written with `origin: "recorded"` and **no declared divergences on purpose**:
the corpus harness fails on an undeclared divergence, so a recorded run refuses
to pass until a human has written the argument for each one, which is the rule
the README states and the only part of this a machine must not do for them.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..containers.manager import CREATED_STATE, Handle
from ..github.branches import parse_task_branch, task_branch
from ..github.readiness import READY as READY_LABEL
from ..github.readiness import SATISFYING_STATE_REASONS, IssueState
from ..github.refs import issue_number, pull_number, task_ref
from ..taskref import TaskRef
from ..worker.result import ResultRecord, record_path
from .authority import BUDGET_RENEWED, INFRASTRUCTURE_CEILING, REVIVED, revived_tasks
from .checks import PullState
# **Bare `CLAIMED` and `REVIEW` in this module are ADR 0001's internal states**,
# and the two `swarm:*` labels that store them are imported under names that say
# so. Not decoration: the first version of the classifier tested
# `divergence.control == CLAIMED` against the dispatcher's `"swarm:claimed"`,
# which never matched, and every `dispatched-this-cycle` divergence silently
# became unexplained. The suffix is what makes that a type of mistake a reader
# sees rather than one the tests have to find.
from .derived import (
    BLOCKED,
    CLAIMED,
    ELIGIBLE,
    LANDED,
    NEEDS_HUMAN,
    REVIEW,
    AttemptFact,
    Budget,
    ContainerFact,
    Divergence,
    Observation,
    PullFact,
    Resolution,
    diverge,
    observe,
    report as render,
    resolve,
)
from .dispatcher import CLAIMED as CLAIMED_LABEL
from .lifecycle import INTERNAL_STATE, internal_state
from .reconcile import CycleReport

__all__ = [
    "BUDGET_RENEWED",
    "CLOSED_NOT_PLANNED",
    "CONTAINER_CREATED",
    "DISPATCHED_THIS_CYCLE",
    "EXPECTED_KINDS",
    "INFRASTRUCTURE_CEILING",
    "MERGED_THIS_CYCLE",
    "REVIVED",
    "SHADOW_ENV",
    "Explained",
    "Judgment",
    "ShadowReport",
    "ShadowWindow",
    "build_observation",
    "classify",
    "control_labels",
    "observation_for",
    "observed_line",
    "revived_tasks",
    "shadow_enabled",
    "written_this_cycle",
]

#: The switch. **Defaulting on**, which is the ticket's word and the right
#: default for a read-only observer: a window nobody turned on measures
#: nothing, and the go/no-go for the rest of epic #140 is exactly "what did ten
#: real runs show". `APIARY_DERIVED_SHADOW=0` turns it off.
SHADOW_ENV = "APIARY_DERIVED_SHADOW"

#: The seven accounts a divergence can have. Empty string means none, and the
#: count of those is what #147's gate reads - see the module docstring.
#:
#: The first three are **imported from `orchestrator/authority.py`**, which is
#: where the same three names became decisions rather than explanations: a
#: divergence this window classifies `infrastructure-ceiling` and an override
#: that module applies for the same reason are one phenomenon, and two spellings
#: of it in `events.jsonl` would be two things to join.
MERGED_THIS_CYCLE = "merged-this-cycle"
DISPATCHED_THIS_CYCLE = "dispatched-this-cycle"
CONTAINER_CREATED = "container-created"
CLOSED_NOT_PLANNED = "closed-not-planned"

EXPECTED_KINDS: tuple[str, ...] = (
    INFRASTRUCTURE_CEILING,
    BUDGET_RENEWED,
    REVIVED,
    MERGED_THIS_CYCLE,
    DISPATCHED_THIS_CYCLE,
    CONTAINER_CREATED,
    CLOSED_NOT_PLANNED,
)


def shadow_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Is the shadow window on? Default yes; garbage reads as the default.

    Deliberately *not* `checks._env_flag`'s loud-on-garbage behaviour, and the
    difference is the whole design of this module. That flag decides whether a
    repository merges without review, so a typo there must stop the run. This
    one decides whether an observer runs, and a `ValueError` out of a mistyped
    `APIARY_DERIVED_SHADOW=yse` would be a shadow taking down a cycle - the one
    thing this file promises it cannot do.
    """
    raw = (env or os.environ).get(SHADOW_ENV)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# --------------------------------------------------------------------------
# The two sides
# --------------------------------------------------------------------------


def control_labels(report: CycleReport) -> dict[str, str]:
    """The `swarm:*` label each task wore when this cycle finished. By task id.

    Five sources, in the order the cycle writes them, so the last word wins:

    1. `report.ledger`, which `cycle` has already folded with what `apply_plan`,
       the recovery sweep and the check gate **applied** - `fold`'s rule, and
       for `fold`'s reason: a label write GitHub refused left the task where it
       was, and crediting it would diff against a state the control plane never
       reached.
    2. mergeability, which is the fourth writer of a terminal label
       (`lifecycle._landed_or_human` names all four) and the one the cycle does
       **not** fold back: a pull request that will not rebase inside its update
       budget is escalated here, and a control map built without it would report
       every starved task as a divergence the resolver invented. Skipped for a
       task the check gate also wrote, because the gate runs after this one and
       step 1 already carries its answer.
    3. the readiness pass, which writes `swarm:ready` / `swarm:blocked` and
       whose result the cycle does not fold back into the ledger either. Its
       verdicts cover exactly the transitionable entries, so nothing here can
       overwrite a `claimed`, `review`, `done` or `failed` that step 1
       established.
    4. the dispatcher's claims, which are written after readiness and are also
       not folded.
    5. a dispatch that claimed and then failed to spawn. `DispatchFailure.claimed`
       is precisely "the label was written and no container is running under
       it", which is a claim the control plane is holding whether or not the
       spawn worked - and the case #35's recovery sweep exists for.

    And a sixth that is not a label writer in `cycle` at all: `planner.revive`,
    reached from step 5 through `replan` and from the goal gate, moves a task
    `swarm:failed -> swarm:ready` on GitHub. `_judge` runs *before* this window
    and nothing folds it either, so a map without it reports `needs-human` for a
    task the cycle left `ready`. Overlaid before readiness, because that is
    where it happens in the cycle and because a revived task is exactly the one
    readiness may then move again.

    Labels rather than internal states, because that is what the replay corpus
    records (`tests/fixtures/runs/README.md`: "the corpus records what the
    control plane actually held, so the day epic #140 removes the labels it is
    the translation that gets deleted and not the data"). The translation
    happens once, on the way into `diverge`.
    """
    labels: dict[str, str] = {}
    for entry in report.ledger.entries.values():
        if entry.task_id:
            labels[entry.task_id] = entry.state_label
    if report.mergeability is not None:
        gated = {
            transition.task_id
            for transition in getattr(report.checks, "applied", ()) or ()
            if transition.task_id
        }
        for transition in report.mergeability.applied:
            if transition.task_id and transition.task_id not in gated:
                labels[transition.task_id] = transition.to_label
    for task in revived_tasks(report):
        labels[task] = READY_LABEL
    if report.readiness is not None:
        for verdict in report.readiness.verdicts:
            if verdict.task_id:
                labels[verdict.task_id] = verdict.label
    if report.dispatched is not None:
        slugs = {entry.ref: entry.task_id for entry in report.ledger.entries.values()}
        for item in report.dispatched.dispatched:
            if item.entry.task_id:
                labels[item.entry.task_id] = CLAIMED_LABEL
        for failure in report.dispatched.failed:
            # `DispatchFailure` carries the issue number rather than the entry,
            # so the ref is re-minted through the adapter and joined back to the
            # slug the ledger holds - `mergeability.py`'s rule, a task is a ref
            # and an API address is a number.
            task = slugs.get(task_ref(int(failure.number)), "")
            if failure.claimed and task:
                labels[task] = CLAIMED_LABEL
    return labels


def build_observation(
    *,
    cycle: int,
    entries: Iterable[Any],
    containers: Iterable[Handle] = (),
    pulls: Mapping[str, PullState] | None = None,
    results: Mapping[TaskRef, ResultRecord] | None = None,
    states: Mapping[TaskRef, IssueState] | None = None,
    budget: Budget | None = None,
    live_run_ids: Iterable[str] = (),
) -> Observation:
    """One `Observation` from raw cycle inputs. **Reads nothing.**

    Every argument is something `Reconciler.cycle` computed for its own reasons,
    which is what makes "shadowing adds no API call" a structural claim rather
    than a promise: there is no client here to call one with.

    **Raw inputs rather than a `CycleReport`**, with `observation_for` below as
    the adapter. All of these are read at the *top* of a cycle, so an
    observation can be built before the cycle decides anything - which is the
    shape #147 needs when the derived state becomes the thing decided *on*, and
    is a free property to keep now rather than a reshape later.

    `containers` is the **raw listing, one entry per container**, not
    `Reconciler._handles`' first-wins map. Two containers under one task is the
    double-spawn `dispatcher.release` is written about and one of the cases #146
    gives as a reason to shadow at all; a collapsed map cannot express it, and
    an exited container listed ahead of a running one would additionally make
    the resolver read not-claimed. `ContainerFact` is per-container by design.

    Two inputs are narrower than `derived.py` would like, and saying so is more
    useful than pretending otherwise:

    - **Branch names are the head refs of open pull requests, and nothing more.**
      A remote branch listing is not a call this cycle makes, and #146's
      acceptance criteria forbid adding one. `attempts_spent` takes the maximum
      of three lower bounds, so a missing source lowers the bound rather than
      corrupting it - and the pull request carries the same attempt the branch
      does, which is `derived.py`'s own argument for keeping them separate.
    - **Merged pull requests are absent.** `Snapshot.pull_requests` lists open
      ones only, deliberately, because a merge closes its issue through
      `Closes #<n>` and the issue listing already carries it. So `landed`
      arrives here through `TaskFact.closed_as_completed`, which is the second
      of the two paths `derived._landed` documents - and the reason
      `state_reason` is passed at all.
    """
    pulls = pulls or {}
    results = results or {}
    states = states or {}

    facts: list[PullFact] = []
    for branch, pull in pulls.items():
        parsed = parse_task_branch(str(branch))
        if parsed is None:
            # A pull request whose head this system did not mint - a human's,
            # against the same repository. Dropped exactly as
            # `lifecycle.lifecycle_events` and the corpus loader drop it.
            continue
        facts.append(
            PullFact(
                # Both sides are `PullRef` since #208, so the un-mint that used
                # to stand here (`pull_number(pull.number)`) is gone rather than
                # merely justified: this is orchestrator-to-orchestrator, not an
                # API boundary, and the only reason it un-minted was that
                # `PullFact.number` could not hold the type. Nothing left in
                # this hop can put a pull request's number where an issue's
                # belongs, which is what #185 was for.
                number=pull.number,
                ref=parsed.ref,
                attempt=parsed.attempt,
                draft=bool(pull.draft),
                head_sha=str(pull.sha or ""),
            )
        )

    return observe(
        cycle=cycle,
        entries=entries,
        branch_names=[str(name) for name in pulls],
        containers=[
            ContainerFact(
                id=handle.id,
                run_id=handle.run_id,
                ref=task_ref(int(handle.issue)),
                running=handle.running,
            )
            for handle in containers
            if handle.issue is not None
        ],
        pulls=facts,
        results=[
            AttemptFact(ref=ref, attempt=record.attempt, exit_code=record.exit_code)
            for ref, record in results.items()
        ],
        budget=budget,
        live_run_ids=live_run_ids,
        state_reasons={ref: state.state_reason for ref, state in states.items()},
    )


def observation_for(report: CycleReport, **facts: Any) -> Observation:
    """`build_observation` over a finished cycle. The two lines a report adds.

    A thin adapter and nothing more, so that the builder above stays usable
    before a cycle has decided anything: the ledger and the cycle index are the
    only things a `CycleReport` contributes, and both are read at the top of the
    cycle rather than produced by it.
    """
    return build_observation(
        cycle=report.index, entries=report.ledger.entries.values(), **facts
    )


# --------------------------------------------------------------------------
# The classification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Explained:
    """One divergence and the account of it, or the absence of one.

    `kind` empty is the interesting case and is spelled as an empty string
    rather than `None` so that `sum(1 for one in explained if not one.kind)` is
    the go/no-go number without a second concept. `why` is the argument, in the
    same register `tests/fixtures/corpus.py`'s `ExpectedDivergence.why`
    demands: a sentence a human can disagree with, not a label.
    """

    divergence: Divergence
    kind: str = ""
    why: str = ""

    @property
    def expected(self) -> bool:
        return bool(self.kind)

    def __str__(self) -> str:
        head = str(self.divergence)
        return f"{head} [{self.kind}: {self.why}]" if self.kind else f"{head} [UNEXPLAINED]"


#: What each account **predicts the derived side will say**, and the reason the
#: sets exist at all.
#:
#: A rule that tested only `control` plus its external evidence would file a
#: divergence as expected on the strength of half the pair - so a task the merge
#: gate merged whose resolver had said `blocked` would be classified
#: `merged-this-cycle`, drop out of `unexplained`, and #147's gate would read
#: clean over a resolver that was simply wrong. That is the one failure this
#: module exists to prevent, and it is silent. So every account below is
#: two-sided: it names the states its own argument predicts, and a divergence
#: outside them falls through to the next rule and, in the end, to unexplained.
#: The failure direction becomes noise rather than silence, which is the
#: direction this module takes everywhere else.

#: A merge landed after the world was read, so the pre-merge reading is an open
#: pull request - `review`, or `claimed` if a container of the merged attempt is
#: still listed as running. Anything else (`blocked`, `eligible`, `needs-human`)
#: means the resolver did not see the pull request this cycle merged, which is
#: news about the resolver rather than about the sampling.
_PRE_MERGE = frozenset({REVIEW, CLAIMED})

#: A claim written after the container listing was taken. The pre-claim reading
#: is whatever the task was before: `eligible` for a first dispatch, `review`
#: for a retry (`worker/pr.py` reuses one pull request across attempts), and
#: `blocked` only if readiness and this resolver disagree about an edge - which
#: is itself a divergence worth keeping, so it is in the set for the claim
#: classes and reported through them rather than hidden. **`needs-human` is
#: deliberately out**: a spent counter outranks `claimed` in `derived._verdict`,
#: so a container would not have changed the answer and the account is false.
_PRE_CLAIM = frozenset({ELIGIBLE, BLOCKED, REVIEW})

#: The infrastructure ceiling, whose own argument in `derived.py` is that an
#: escalation raised on it "reads here as whatever the task would otherwise be -
#: `eligible`, usually". `landed` is not one of those: a task whose work item
#: reads closed-as-completed while the control plane escalated it is a genuine
#: contradiction, not a ceiling. Nor is `claimed` - the escalation is raised on
#: observing a result record, which means the worker exited.
_PRE_CEILING = frozenset({ELIGIBLE, BLOCKED, REVIEW})

#: A work item closed as not planned. The weakest of the four sets and
#: deliberately so: a human can close an issue at any point in a task's life, so
#: every non-terminal reading is plausible. What it still rules out is the one
#: reading that would be a contradiction - `landed` - because
#: `TaskFact.state_reason` is exactly what stops a not-planned closure reading
#: as a completed one, and a regression that stopped passing it would otherwise
#: be absorbed here as an expected divergence.
_NOT_TERMINAL = frozenset({ELIGIBLE, BLOCKED, REVIEW, CLAIMED})


@dataclass(frozen=True)
class Judgment:
    """The two numbers from apiary's own store that a classification reads.

    **Not the `LedgerEntry` they came from**, and that is the point: an entry
    carries `state_label`, and this is the module whose whole discipline is that
    a label cannot reach a resolver. Holding an entry here would put one an
    attribute access away from a classification rule, in the one function that
    is allowed to look at both sides. Two ints cannot be misused that way.

    `streak` is `None` for a task the store has never judged (`LedgerEntry`
    says so), which is not the same as zero and is why it stays optional.
    """

    streak: int | None = None
    renewals: int = 0

    @classmethod
    def of(cls, entry: Any) -> Judgment:
        return cls(streak=getattr(entry, "streak", None), renewals=int(getattr(entry, "renewals", 0) or 0))


def classify(
    divergences: Iterable[Divergence],
    *,
    resolution: Resolution,
    judgments: Mapping[str, Judgment] | None = None,
    containers: Iterable[Handle] = (),
    states: Mapping[TaskRef, IssueState] | None = None,
    infrastructure: Mapping[TaskRef, int] | None = None,
    infrastructure_cap: int = 3,
    max_attempts: int = 3,
    merged: Iterable[TaskRef] = (),
    dispatched: Iterable[TaskRef] = (),
) -> tuple[Explained, ...]:
    """Attach an account to each divergence, or leave it unexplained.

    Ordered most-specific first. **Every rule is two-sided**: it tests the
    control state, the derived state its own argument predicts, and the
    evidence of its account - the streak counter, the store's renewal count, the
    container's own `created`, the merge this cycle performed. Never the pair of
    states alone, and never the control state alone.

    That is not fastidiousness. A one-sided rule absorbs a *wrong derived
    answer* into an expected divergence and removes it from `unexplained`, which
    is the number #147's gate reads and #152 acts on. The sets above name what
    each account predicts and the reasoning for each; a divergence outside them
    falls through to the next rule and, in the end, to unexplained.
    """
    judgments = judgments or {}
    states = states or {}
    infrastructure = infrastructure or {}
    merged_refs = frozenset(merged)
    dispatched_refs = frozenset(dispatched)
    created: dict[TaskRef, Handle] = {}
    for handle in containers:
        if handle.issue is not None and handle.state == CREATED_STATE:
            created.setdefault(task_ref(int(handle.issue)), handle)
    verdicts = resolution.by_task

    out: list[Explained] = []
    for one in divergences:
        judgment = judgments.get(one.task_id, Judgment())
        verdict = verdicts.get(one.task_id)
        spent = verdict.attempts_spent if verdict is not None else 0
        streak = infrastructure.get(one.ref, 0)
        kind, why = "", ""

        if one.control == LANDED and one.derived in _PRE_MERGE and one.ref in merged_refs:
            kind = MERGED_THIS_CYCLE
            why = (
                "this cycle's merge gate merged the pull request after the world was "
                "read, so the derived side is resolving a listing that predates the "
                "merge. It converges on the next cycle, when the issue reads closed."
            )
        elif (
            one.control == CLAIMED
            and one.derived in _PRE_CLAIM
            and one.ref in dispatched_refs
        ):
            kind = DISPATCHED_THIS_CYCLE
            why = (
                "the dispatcher claimed and spawned for this task after the container "
                "listing was taken, so no container could have been in the observation "
                f"- and {one.derived} is what the task read before the claim. It "
                "converges on the next cycle, when the listing includes it."
            )
        elif (
            one.control == CLAIMED
            and one.derived in _PRE_CLAIM
            and one.ref in created
        ):
            kind = CONTAINER_CREATED
            why = (
                f"container {created[one.ref].short_id} exists but the daemon reports "
                f"it as {CREATED_STATE!r} rather than running, so `Handle.running` is "
                "false and the resolver reads liveness rather than existence. See this "
                "module's docstring: the reading is deliberate, and this window should "
                "be unreachable through `spawn`."
            )
        elif (
            one.control == NEEDS_HUMAN
            and one.derived in _PRE_CEILING
            and streak >= max(int(infrastructure_cap), 1)
        ):
            kind = INFRASTRUCTURE_CEILING
            why = (
                f"this task has {streak} consecutive infrastructure verdicts against a "
                f"cap of {infrastructure_cap}. ADR 0001: the ceiling is counted from "
                "transitions and exit 2 does not bump the attempt, so N mechanical "
                "failures write one result filename and the artifacts cannot tell one "
                f"from three. Not derivable at all, and {one.derived} is what the task "
                "reads without it."
            )
        elif (
            one.control == NEEDS_HUMAN
            and one.derived in _NOT_TERMINAL
            and _closed_not_planned(states.get(one.ref))
        ):
            kind = CLOSED_NOT_PLANNED
            why = (
                "a human closed the work item as not planned, which "
                "`reconcile._closed_verdict` escalates. #147 taught the resolver the "
                "same rule (`TaskFact.abandoned`), so reaching this line means the "
                "observation was built without `state_reasons` - a defect in the "
                "wiring rather than a limit of derivation."
            )
        elif one.derived == NEEDS_HUMAN and _renewed(judgment, spent, max_attempts):
            kind = BUDGET_RENEWED
            why = (
                f"the code host accounts for {spent} attempt(s) against a cap of "
                f"{max_attempts}, but `_retry_or_give_up` gives up on the streak and "
                f"the store records streak={judgment.streak}, "
                f"renewals={judgment.renewals}. The renewal is an ADR 0002 store "
                "judgment and no branch, container or result can see it."
            )
        elif one.derived == NEEDS_HUMAN and one.control in {ELIGIBLE, BLOCKED}:
            kind = REVIVED
            why = (
                f"the counter reads {spent} spent against a cap of {max_attempts} while "
                "the control plane holds this task live and the store records no "
                "renewal. That is what `planner.revive` leaves behind - it "
                "'deliberately resets nothing' - and it converges on merge, which is "
                "what `derived.py`'s `landed > needs-human` precedence is for."
            )

        out.append(Explained(divergence=one, kind=kind, why=why))
    return tuple(out)


def _closed_not_planned(state: IssueState | None) -> bool:
    """Closed, and closed in a way that discharges nothing.

    `github.readiness.SATISFYING_STATE_REASONS` rather than a literal, because
    the question "does this closure count" is one judgement and three modules
    already share it.
    """
    return state is not None and state.closed and state.state_reason not in SATISFYING_STATE_REASONS


def _renewed(judgment: Judgment, spent: int, max_attempts: int) -> bool:
    """Did apiary's own store renew this task's per-blocker budget?

    Two spellings of the same evidence, both from the store-backed fields
    `LedgerEntry` carries since #159. `renewals` is the count `TaskJudgement`
    keeps. A `streak` below the attempt bound is the same fact seen from the
    other side: the counter moved and the streak did not, which is precisely
    what a changed blocker signature does.
    """
    if judgment.renewals > 0:
        return True
    streak = judgment.streak
    return streak is not None and int(streak) < min(spent, max(int(max_attempts), 1))


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowReport:
    """One cycle, resolved and diffed. Read by a human and by nothing else."""

    cycle: int
    resolution: Resolution
    control: Mapping[str, str]
    explained: tuple[Explained, ...] = ()
    #: Task ids whose label **this cycle wrote**. See `independent` below; this
    #: is the field that answers what a clean shadow window is evidence of.
    written: frozenset[str] = frozenset()

    @property
    def divergences(self) -> tuple[Divergence, ...]:
        return tuple(one.divergence for one in self.explained)

    @property
    def unexplained(self) -> tuple[Explained, ...]:
        """The ones nobody has an account of. **The go/no-go number.**"""
        return tuple(one for one in self.explained if not one.kind)

    @property
    def tasks(self) -> int:
        """How many tasks were actually compared - the coverage number.

        `derived.diverge` deliberately does not compute one, because a task on
        one side and not the other is not a divergence. It is computed here
        because "zero divergences over zero tasks" and "zero divergences over
        eleven tasks" are different claims and the second is the only one that
        is evidence.
        """
        return sum(1 for verdict in self.resolution.verdicts if verdict.task_id in self.control)

    @property
    def independent(self) -> int:
        """Compared tasks whose label this cycle did **not** write.

        The honest strength of the comparison, and the module docstring argues
        it at length. For a task the cycle relabelled, `plan_reconcile` computed
        that label from the same issue listing, the same results directory and
        the same container listing this resolver read - so agreement shows the
        two reducers implement the same rules over one input. That is worth
        something (it is exactly "the cache is redundant"), but it is not the
        resolver tracking a world nobody told it about.

        For a task the cycle left alone, the label is the accumulation of every
        earlier cycle's decisions over earlier observations, and the derived
        state is one absolute reading of now. Those are the independent
        comparisons, and this counts them so that a reader of a clean run can
        tell how many there were.
        """
        return sum(
            1
            for verdict in self.resolution.verdicts
            if verdict.task_id in self.control and verdict.task_id not in self.written
        )

    def text(self) -> str:
        """`derived.report`, plus the account of each divergence."""
        lines = [render(self.resolution, self.divergences)]
        for one in self.explained:
            lines.append(f"    {'~' if one.kind else '!'} {one}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The recorder
# --------------------------------------------------------------------------


def observed_line(
    observation: Observation, control: Mapping[str, str], *, results_dir: Any = None
) -> dict[str, Any]:
    """One `observed.jsonl` line, in the shape `tests/fixtures/corpus.py` loads.

    A projection of an `Observation` that has already been built, so nothing
    here reads anything and nothing here can disagree with what was resolved.
    The one asymmetry is `results`: the corpus records the *file names* a cycle
    could see rather than the records themselves, so the name is rebuilt
    through `worker.result.record_path` - the one spelling of that name in the
    codebase, which is what keeps the recorder and the loader from drifting.

    Labels rather than internal states in `control`, because that is the
    corpus's own decision and its reason is good: the day the labels go away it
    is the translation that gets deleted, not the recorded data.

    Tasks whose label is not one of the six are dropped rather than written.
    `load_corpus` refuses a label it cannot translate - correctly - and a
    recorder that could emit an unloadable line would produce corpus runs that
    fail at the door for a reason that has nothing to do with the resolver.
    """
    return {
        "cycle": observation.cycle,
        "tasks": [
            {
                "ref": str(fact.ref),
                "task_id": fact.task_id,
                "depends_on": [str(dep) for dep in fact.depends_on],
                "closed": fact.closed,
                "state_reason": fact.state_reason,
            }
            for fact in observation.tasks
        ],
        "branches": [task_branch(branch.ref, branch.attempt) for branch in observation.branches],
        "containers": [
            {
                "id": container.id,
                "run_id": container.run_id,
                "ref": None if container.ref is None else str(container.ref),
                "running": container.running,
            }
            for container in observation.containers
        ],
        "pulls": [
            {
                # Un-minted, because this one *is* a boundary: the corpus format
                # `tests/fixtures/runs/README.md` documents holds a JSON number,
                # every existing fixture carries one, and a `PullRef` would not
                # serialise. The loader mints it back (`fixtures/corpus.py`), so
                # the type lives inside the process and the file stays the file.
                "number": pull_number(pull.number),
                "head": task_branch(pull.ref, pull.attempt),
                "merged": pull.merged,
                "closed": pull.closed,
                "draft": pull.draft,
                "sha": pull.head_sha,
            }
            for pull in observation.pulls
        ],
        "results": [
            record_path("", _issue_of(record.ref), record.attempt).name
            for record in observation.results
        ],
        "budget": {
            "max_attempts": observation.budget.max_attempts,
            "max_total_attempts": observation.budget.max_total_attempts,
        },
        "live_run_ids": sorted(observation.live_run_ids),
        "control": {
            task: label
            for task, label in sorted(control.items())
            if label in INTERNAL_STATE
        },
    }


def _issue_of(ref: TaskRef) -> int:
    """The issue number a result file is named after. Through the adapter.

    `record_path` takes an `int` because the worker files its record under the
    number the code host gave the issue, and `github.refs.issue_number` is the
    only module allowed to turn a ref back into one (#142/#166). A ref this
    adapter did not mint has no result file under any name, and `0` produces a
    corpus line the loader rejects loudly rather than one that quietly points
    at the wrong record.
    """
    try:
        return issue_number(ref)
    except ValueError:
        return 0


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------


@dataclass
class ShadowWindow:
    """Runs the resolver beside one cycle. **Cannot fail the cycle.**

    Run-scoped, for `LifecycleLog`'s reason: "have I warned about this yet" is
    a question about a sequence, and a restart warning once more is the
    harmless direction.
    """

    enabled: bool = True
    #: Warn on stderr the first time this run sees something nobody has an
    #: account of, and then stop. The ticket asks for once per run because an
    #: unexplained divergence is usually *standing* - it repeats every cycle
    #: until something moves - and a warning per cycle would train an operator
    #: to ignore the one line in the run that matters.
    warned: bool = field(default=False, repr=False)
    #: Set once if this module itself raised. A broken shadow says so and then
    #: stays quiet, rather than printing a traceback every fifteen seconds for
    #: the rest of the run.
    broken: bool = field(default=False, repr=False)
    #: The last cycle this window actually resolved. Retained for one caller and
    #: one reason: a test asserting that what the recorder wrote replays to what
    #: the live cycle concluded has to compare against *that* cycle's report,
    #: and re-running the window afterwards with different facts compares two
    #: different observations while looking like it compares one.
    last: ShadowReport | None = field(default=None, repr=False)

    def _blind(self, report: CycleReport, emit: Callable[..., Any] | None) -> None:
        """Announce a cycle that could not see. See `run`'s docstring."""
        if emit is None:
            return
        from ..artifacts import STATE_SHADOW

        emit(
            STATE_SHADOW,
            cycle=report.index,
            tasks=0,
            independent=0,
            divergences=0,
            unexplained=0,
            blind=True,
        )

    def run(
        self,
        report: CycleReport,
        *,
        containers: Iterable[Handle] = (),
        pulls: Mapping[str, PullState] | None = None,
        results: Mapping[TaskRef, ResultRecord] | None = None,
        states: Mapping[TaskRef, IssueState] | None = None,
        infrastructure: Mapping[TaskRef, int] | None = None,
        infrastructure_cap: int = 3,
        max_attempts: int = 3,
        max_total_attempts: int = 9,
        live_run_ids: Iterable[str] = (),
        emit: Callable[..., Any] | None = None,
        record: Callable[..., Any] | None = None,
    ) -> ShadowReport | None:
        """Resolve, diff, classify, announce. `None` when off or when it broke.

        The whole body is inside one `except Exception`, which is normally a
        smell and is the entire point here. This module reads a dozen
        attributes off objects five other modules own; the day one of them is
        renamed, the correct outcome is a shadow that stops reporting and says
        so, not a run that dies holding containers. `#146`'s own ticket puts it
        plainly: a shadow that raises is worse than no shadow.

        **`pulls=None` is "this cycle could not look", and it is not `{}`.**
        `checks.read_pulls` and `Snapshot.open_branches` both go to lengths to
        keep the two apart because conflating them relabels the whole review
        queue; here the cost is the same shape one level along. An empty
        mapping read as the answer would resolve every task in review to
        `eligible` and emit one manufactured unexplained divergence per review
        task - straight into the number the epic's go/no-go reads. So the cycle
        is announced as blind and nothing is compared: unmeasured, which is
        true, rather than dirty, which is not.
        """
        if not self.enabled or self.broken:
            return None
        try:
            if pulls is None:
                self._blind(report, emit)
                return None
            self.last = self._run(
                report,
                containers=list(containers),
                pulls=pulls,
                results=results,
                states=states,
                infrastructure=infrastructure,
                infrastructure_cap=infrastructure_cap,
                max_attempts=max_attempts,
                max_total_attempts=max_total_attempts,
                live_run_ids=live_run_ids,
                emit=emit,
                record=record,
            )
            return self.last
        except Exception as exc:  # noqa: BLE001 - see the docstring
            self.broken = True
            print(
                f"! the derived-state shadow failed and is now off for this run: "
                f"{exc!r}. The cycle is unaffected; nothing reads it "
                f"({SHADOW_ENV}=0 to silence).",
                file=sys.stderr,
            )
            return None

    # --- the body, which may raise ---------------------------------------

    def _run(
        self,
        report: CycleReport,
        *,
        containers: Sequence[Handle],
        pulls: Mapping[str, PullState] | None,
        results: Mapping[TaskRef, ResultRecord] | None,
        states: Mapping[TaskRef, IssueState] | None,
        infrastructure: Mapping[TaskRef, int] | None,
        infrastructure_cap: int,
        max_attempts: int,
        max_total_attempts: int,
        live_run_ids: Iterable[str],
        emit: Callable[..., Any] | None,
        record: Callable[..., Any] | None,
    ) -> ShadowReport:
        labels = control_labels(report)
        observation = observation_for(
            report,
            containers=containers,
            pulls=pulls,
            results=results,
            states=states,
            budget=Budget(max_attempts=max_attempts, max_total_attempts=max_total_attempts),
            live_run_ids=live_run_ids,
        )
        resolution = resolve(observation)
        control = {
            task: internal_state(label)
            for task, label in labels.items()
            if label in INTERNAL_STATE
        }
        explained = classify(
            diverge(resolution, control),
            resolution=resolution,
            # Two numbers, never the `LedgerEntry` they came from: an entry
            # carries `state_label`, and this is the one function allowed to
            # look at both sides. See `Judgment`.
            judgments={
                entry.task_id: Judgment.of(entry)
                for entry in report.ledger.entries.values()
                if entry.task_id
            },
            containers=containers,
            states=states,
            infrastructure=infrastructure,
            infrastructure_cap=infrastructure_cap,
            max_attempts=max_attempts,
            merged=_merged_refs(report),
            dispatched=_dispatched_refs(report),
        )
        shadow = ShadowReport(
            cycle=report.index,
            resolution=resolution,
            control=control,
            explained=explained,
            written=written_this_cycle(report),
        )

        if record is not None:
            record(observed_line(observation, labels))
        if emit is not None:
            self._announce(shadow, emit)
        if shadow.unexplained and not self.warned:
            self.warned = True
            first = shadow.unexplained[0]
            print(
                f"! derived state disagrees with the control plane and nothing here "
                f"accounts for it: {first.divergence}. Labels remain authoritative "
                f"and this changed no decision; see {len(shadow.unexplained)} "
                f"state.divergence event(s) this cycle.",
                file=sys.stderr,
            )
        return shadow

    def _announce(self, shadow: ShadowReport, emit: Callable[..., Any]) -> None:
        """One `state.shadow` per cycle, one `state.divergence` per disagreement.

        Keyed by the task id and speaking ADR 0001's vocabulary on both sides,
        which is #141's rule for everything in `events.jsonl`: the log is
        append-only and read back, so a payload carrying a `swarm:*` label
        would be invalidated the day #152 removes them. `control` here is the
        internal state the label was *storing*, translated on the way in.

        Redaction is the emitter's - `RunArtifacts.event` runs every string
        through the run's redactor - which is why this hands over fields rather
        than a formatted line.
        """
        from ..artifacts import STATE_DIVERGENCE, STATE_SHADOW

        emit(
            STATE_SHADOW,
            cycle=shadow.cycle,
            tasks=shadow.tasks,
            # The half of the coverage number that is evidence rather than
            # arithmetic - see `ShadowReport.independent`. Carried per cycle
            # because a run's total is the only way a reader of a clean window
            # can tell how much of it the cycle's own writes accounted for.
            independent=shadow.independent,
            divergences=len(shadow.explained),
            unexplained=len(shadow.unexplained),
            blind=False,
        )
        for one in shadow.explained:
            emit(
                STATE_DIVERGENCE,
                cycle=one.divergence.cycle,
                task=one.divergence.task_id,
                derived=one.divergence.derived,
                control=one.divergence.control,
                expected=one.expected,
                kind=one.kind,
                because=one.divergence.because,
                why=one.why,
            )


# `revived_tasks` moved to `orchestrator/authority.py` in #147 and is re-exported
# from here. Two callers want it now - this window, to build the control map a
# cycle actually left behind, and `Reconciler`, to record the one attempt a
# revival grants - and the module that *decides* on it is the one that should
# own it. Nothing about the answer changed; `control_labels` still overlays it
# before readiness, for the reason its own docstring gives.


def written_this_cycle(report: CycleReport) -> frozenset[str]:
    """Tasks whose label **this cycle wrote**. See `ShadowReport.independent`.

    Every writer `control_labels` walks, narrowed to the ones that actually
    moved something: readiness contributes `transitions` rather than every
    verdict, because a verdict that agreed with the label wrote nothing and the
    task is therefore still an independent comparison.
    """
    written: set[str] = set()
    for transitions in (
        report.result.applied,
        getattr(getattr(report.recovered, "result", None), "applied", ()) or (),
        getattr(report.mergeability, "applied", ()) or (),
        getattr(report.checks, "applied", ()) or (),
    ):
        written.update(one.task_id for one in transitions if one.task_id)
    if report.readiness is not None:
        written.update(one.task_id for one in report.readiness.transitions if one.task_id)
    if report.dispatched is not None:
        written.update(
            item.entry.task_id for item in report.dispatched.dispatched if item.entry.task_id
        )
        slugs = {entry.ref: entry.task_id for entry in report.ledger.entries.values()}
        written.update(
            slugs.get(task_ref(int(failure.number)), "")
            for failure in report.dispatched.failed
            if failure.claimed
        )
    written |= revived_tasks(report)
    return frozenset(task for task in written if task)


def _merged_refs(report: CycleReport) -> tuple[TaskRef, ...]:
    """Tasks this cycle's merge gate actually merged. See `MERGED_THIS_CYCLE`.

    `ChecksReport.merged` is issue numbers, because that is what the merge API
    answers about. Minted into refs through the adapter rather than compared as
    numbers, for `github/refs.py`'s reason (#142/#166).
    """
    checks = report.checks
    if checks is None:
        return ()
    return tuple(task_ref(int(number)) for number in getattr(checks, "merged", ()) or ())


def _dispatched_refs(report: CycleReport) -> tuple[TaskRef, ...]:
    """Tasks this cycle claimed, whether or not the spawn then worked.

    Both halves, because both leave a claim on the control plane:
    `DispatchFailure.claimed` is documented as "the label was written and no
    container is running under it", which is the same divergence with a
    different cause and the same account.
    """
    dispatched = report.dispatched
    if dispatched is None:
        return ()
    refs: list[TaskRef] = [item.entry.ref for item in dispatched.dispatched]
    refs += [
        task_ref(int(failure.number)) for failure in dispatched.failed if failure.claimed
    ]
    return tuple(refs)
