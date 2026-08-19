"""The five lifecycle states, computed from the world instead of read off a label.

`docs/adr/0001-task-systems-are-integrations.md` makes one claim that the whole
of epic #140 rests on: apiary's internal workflow is **derived, not stored**.
`eligible` from dependencies discharged, `claimed` from a live container,
`review` from an open pull request, `landed` from that pull request merging,
`needs-human` from the attempt budget being spent. If that is true the six
`swarm:*` labels are a cache with no reader, and #152 can delete them. If it is
false, the ADR is wrong and the labels are load-bearing.

This module is the half of the answer that computes. It reads the code host,
the container layer and the run artifacts, and it says what each task's state
is. **Nothing here writes.** #146 wired it into the live cycle through
`orchestrator/shadow.py`, which resolves every cycle beside the labels and
records where the two disagree; #147 made the answer authoritative through
`orchestrator/authority.py`, which is the module that decides how much of it to
believe. Nothing here reads a label to compute it, which is still the invariant
the rest of this docstring is about.

**Authoritative is not the same as sufficient.** `authority.py` joins this
resolver to apiary's own store and its own run-scoped counters, because the
three states named below stayed non-derivable after the cutover - making a
value decisive does not make it complete. What #147 did fix here is the fourth
item, the one #146 called a gap rather than a limit: see `TaskFact.abandoned`.

## The sourcing invariant, which is the entire point

`lifecycle.py` states it for the event log and it is repeated here because this
module is where breaking it would be invisible:

> Derived facts and applied writes are sourced differently. `eligible`,
> `result`, `pr.opened` and `pr.checks` are facts about the world;
> `claimed`, `pr.merged`, `landed` and `needs_human` are consequences of
> writes.

A resolver that reached for the applied half would be reading the control plane
back to itself. It would agree with the labels perfectly, prove nothing, and
the agreement would be an artefact of the wiring rather than evidence about the
ADR. So `Observation` - the only input `resolve` takes - **has no field that can
hold a label, and no field that can hold a state**. Not a convention: there is
physically nowhere to put one. A future contributor who needs a label in here
has to add a field, and adding it is the review conversation.

The same rule is why `task.landed` and `task.needs_human` from `events.jsonl`
are not inputs either, though they are the two events that would make this file
much shorter. They are announcements of writes that landed. The corpus format
under `tests/fixtures/runs/` keeps them on the other side of the line, as the
*control-plane state to be diffed against*, never as evidence.

## What is derivable, and the three things that turned out not to be

Four of the five come out clean, each from exactly the fact ADR 0001 named:

| State | The fact |
|---|---|
| `landed` | a merged pull request whose head branch carries this ref |
| `claimed` | a running container labelled with this run and this ref |
| `review` | an open pull request whose head branch carries this ref |
| `eligible` | every dependency landed, and this task has not |

A fifth row was missing until #147 and belongs with these four rather than with
the three below, because it really is a fact about the code host: a work item a
human closed **as not planned** is `needs-human`, and `TaskFact.state_reason`
carries it. `reconcile._closed_verdict` had escalated on that fact since #22
and this module had no rule reading it, so #146's shadow window reported the
disagreement on every abandoned issue and classified it `closed-not-planned` -
"a gap rather than a limit", in ADR 0001's words. The classification should now
never fire.

The remaining `needs-human` does not come out clean, and the reasons are worth
naming because they are findings about the ADR rather than gaps in this file.

**The infrastructure ceiling is not in the artifacts.** ADR 0001 says
`needs-human` is "attempts exhausted, **or the infrastructure cap hit**", and
`reconcile.infrastructure_streaks` explains in its own docstring why the second
half cannot be recomputed here: exit 2 does not bump the attempt, so two
consecutive mechanical failures write the *same* result filename and the
directory cannot tell one from two. The streak is counted from transitions,
which are writes. So this module derives the attempt half of `needs-human` and
nothing else, and an escalation raised on the infrastructure ceiling reads here
as whatever the task would otherwise be - `eligible`, usually.

**The retry budget renews, and the renewal is a stored judgment.**
`docs/adr/0002-apiary-owns-a-thin-task-store.md` exists because `blocker` and
`streak` are apiary's opinions about its own execution rather than facts about
anything external, and `reconcile._retry_or_give_up` gives up on `streak`, not
on `attempt`. With no renewal the two are equal - which is why
`Budget.max_attempts` is the right cap to compare against and why the common
path agrees - but a renewed budget lets a task run past it, and no amount of
looking at branches, containers or results can see that. It is in the store, by
design.

**A revival is invisible from the code host.** `nodes/planner.revive` takes a
task out of `swarm:failed` and "deliberately resets **nothing**": the counter
stays where it was. So a revived task is eligible while its spent counter still
says `needs-human`, and this module says `needs-human`. That divergence is not a
bug here; it is the mechanism working exactly as `planner.py` documents it,
seen from the side that cannot see the decision.

All three are recorded as *expected divergences* in the replay corpus rather
than papered over, because the point of the exercise is to find out where
derived state stops reproducing label state, and a harness tuned until it agrees
would have told us nothing.

**And one of the four that do come out clean needs a fact the container layer
did not expose.** `claimed` is "a live worker container", and
`containers.find_containers` ran `docker ps --all` with a format string asking
for id, name, image and two labels - not the container's state. So the listing
returned the *exited* container whose worker finished this cycle, and a
resolver reading it as it stood would hold a task in `claimed` from the moment
its worker exited until the reaper got to it. That is precisely the cycle in
which `claimed` and `review` disagree, so it is not a rare window. `#187` added
`{{.State}}` to `_PS_FORMAT`, so `containers.manager.Handle` now carries the
state and answers `running`; `ContainerFact.running` is that fact, and #146
wired the two together, since a shadow window cannot run without it. The other
edge of the same field is the create-to-start gap, which `shadow.py` reports as
an expected divergence rather than as a claim - see its docstring for why
liveness is the right reading here and existence is the right one in
`dispatcher.release`.

## Where the attempt counter is read from, and why not the issue body

ADR 0002's "As built" section says the counter stays in the issue marker,
because the worker is a container whose only picture of the task is the body it
fetched. That is a fine home for it and a bad *source* for this module: the body
is the customer's tracker, which decision 2 of ADR 0001 puts out of bounds for
exactly the reason labels are.

The code host holds the same number in a form #144 put there on purpose. A
branch named `apiary/<ref>-attempt-<n>` says which task and which attempt, so
the highest attempt on the remote is a lower bound on the counter that survives
an orchestrator crash - and the head branch of an open pull request carries the
same number without a second API call. `worker/result.py`'s records are the
third source: a record whose failure moved the counter accounts for the attempt
after it.

None is authoritative alone. A worker that died before pushing left no branch
and no record; a release by `orchestrator/recovery.py` consumes an attempt and
writes neither; a pull request listing is one call every cycle already makes
while a branch listing is not. So `attempts_spent` takes the **maximum** of the
three and never their sum - each is a partial view of one counter, and adding
them would charge a task twice for the attempt that both pushed a branch and
wrote a record. The maximum over-counts rather than under-counts, which is the
direction `reconcile._retry_or_give_up` also chose: a counter that can fail to
bound retries is worse than one that gives up a cycle early.

## Precedence, and why it is not the order of the table

A task can satisfy several of these at once - a merged pull request and a
container the reaper has not reached yet - and the label machine stores exactly
one state, so the resolver has to pick one too. The order is:

    landed > needs-human > claimed > review > eligible > blocked

`landed` first because it is the only terminal state that is also *good*, and
because a merge is the strongest fact on the code host. `needs-human` second
because a spent budget is the thing a person is waiting to be told, and hiding
it behind a container that is on its way out is how an escalation gets lost.
`claimed` above `review` because a running container is a claim about *now*
while an open pull request is a claim about work that may be several attempts
old - and `worker/pr.py` reuses one pull request across retries, so a re-claimed
task still has one open.

`blocked` is the sixth, and it is here for `lifecycle.INTERNAL_STATE`'s reason:
ADR 0001's table names five because it describes the states a *running* task
passes through, but readiness holds tasks outside them and the control plane
spells that `swarm:blocked`. A resolver that folded it into `eligible` would
report a divergence on every task in every plan that has an edge.

There is no `__main__` here, unlike `mergeability.py`. A dry run needs a run
directory to read, and reading one is the corpus loader's job
(`tests/fixtures/corpus.py`) - putting a second reader of the same format in
the shipped package would give the format two implementations to drift apart.
The shipped package does now *write* that format: `RunArtifacts.observed`
records one line per shadowed cycle (#146), so a real run replays through the
same loader a synthesised one does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol, Sequence

from ..github.branches import TaskBranch, parse_task_branch
from ..taskref import PullRef, TaskRef
from ..worker.result import EXIT_INFRASTRUCTURE

__all__ = [
    "BLOCKED",
    "CLAIMED",
    "ELIGIBLE",
    "LANDED",
    "NEEDS_HUMAN",
    "REVIEW",
    "STATES",
    "AttemptFact",
    "Budget",
    "ContainerFact",
    "Divergence",
    "Observation",
    "PullFact",
    "Resolution",
    "TaskFact",
    "Verdict",
    "diverge",
    "observe",
    "report",
    "resolve",
]

#: ADR 0001's internal vocabulary, spelled here and nowhere else in this module.
#: The same six strings `lifecycle.INTERNAL_STATE` maps the labels onto - which
#: is what makes a diff between this resolver and the control plane a comparison
#: of two states rather than a translation exercise. `derived.py` deliberately
#: does not import that mapping: it must not be able to see a label at all, and
#: a module that imports the translation table is one import away from using it.
ELIGIBLE = "eligible"
BLOCKED = "blocked"
CLAIMED = "claimed"
REVIEW = "review"
LANDED = "landed"
NEEDS_HUMAN = "needs-human"

#: In precedence order, highest first. Written down rather than left implicit in
#: a chain of `if`s, because the ordering *is* the design decision the module
#: docstring argues for, and a reader checking it should not have to reconstruct
#: it from control flow.
STATES: tuple[str, ...] = (LANDED, NEEDS_HUMAN, CLAIMED, REVIEW, ELIGIBLE, BLOCKED)

#: GitHub's `state_reason` values that discharge a dependency. The same set
#: `github/readiness.SATISFYING_STATE_REASONS` uses, restated rather than
#: imported for the reason the constants above are: this module answers to the
#: code host, and readiness answers to a label plan. Sharing the constant would
#: be harmless today and would be the seam somebody later routes a label
#: through.
SATISFYING_STATE_REASONS = frozenset({"completed", None})


# --------------------------------------------------------------------------
# The world, as one cycle read it
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskFact:
    """One work item's identity and its declared dependencies. **No state.**

    Everything here is something the tracker was asked and answered: which ref,
    which slug the marker carries, what it declared it was blocked by, and
    whether the item is closed. `state_label` is conspicuously absent and its
    absence is the module docstring's invariant made structural - there is
    nowhere in an `Observation` to put a label, so `resolve` cannot read one
    even by accident.

    `closed` is here and is not a state either. It is the fact
    `github/readiness.IssueState.satisfied` already runs on: a dependency is
    discharged when the item it names is closed as completed. `state_reason`
    rides along because "closed as not planned" does not discharge anything -
    dropping it would silently unblock every task waiting on work somebody
    abandoned.
    """

    ref: TaskRef
    task_id: str
    depends_on: tuple[TaskRef, ...] = ()
    closed: bool = False
    state_reason: str | None = None

    @property
    def closed_as_completed(self) -> bool:
        return self.closed and self.state_reason in SATISFYING_STATE_REASONS

    @property
    def abandoned(self) -> bool:
        """Closed, and closed in a way that discharges nothing. **`needs-human`.**

        The other edge of the field above, and the rule #146 found missing.
        ADR 0001 calls it "one finding that is a gap rather than a limit": a
        human closing a work item as *not planned* is escalated by
        `reconcile._closed_verdict`, `state_reason` carries the fact, and this
        module simply had no rule reading it - so the shadow window classified
        the divergence as `closed-not-planned` and named #147 as the fix.
        Unlike the three states ADR 0001 takes back, this one is derivable, and
        as of #147 it is derived.
        """
        return self.closed and self.state_reason not in SATISFYING_STATE_REASONS


@dataclass(frozen=True)
class ContainerFact:
    """One container the daemon listed, reduced to the three fields that decide.

    `running` is not decoration and it is the field `containers.manager.Handle`
    lacked until #187. `find_containers` runs `docker ps --all`, so an *exited*
    container is still listed - and a worker that finished thirty seconds ago
    is exactly the case where `claimed` and `review` disagree. A resolver
    reading that listing without the state would hold a task in `claimed` from
    the moment its worker exited until the reaper got to it. So the liveness
    the ADR's table means by "a live worker container" is read out of
    `docker ps --format {{.State}}`, which the format string now asks for and
    `Handle.running` answers from.

    `run_id` is kept rather than filtered at the edge because
    `orchestrator/recovery.py` makes liveness a question about *whose* run:
    a container labelled with a run id this process does not answer to belongs
    to a process that is gone, and holds nothing. `Observation.live_run_ids` is
    the same escape hatch `recovery.holders` takes.
    """

    id: str
    run_id: str
    ref: TaskRef | None = None
    running: bool = False


@dataclass(frozen=True)
class PullFact:
    """One pull request, joined to a task the way #144 says to join one.

    **By the ref inside the head branch, never by `Closes #<n>` and never by
    `LedgerEntry.branch`.** `number` is a `PullRef` and `ref` is a `TaskRef`
    because they are not the same fact in two spellings: the number addresses
    the API and the ref identifies the work. #185 retyped the orchestrator's
    pull-request numbers and stopped at the resolver's edge, leaving this field
    an `int` because retyping it needed `PullRef` to sort and it did not; #208
    gave it an ordering and closed the gap, so `mergeability.py`'s rule - a task
    is a ref, an API address is a number - now holds through the resolver too.
    Comparing against a rebuilt `entry.branch` is right until the counter moves,
    and the cycle where it moves is the cycle a crash happened, which is the only
    cycle any of this matters in.

    `attempt` comes out of the same name and is the second half of what #144 put
    there. It is what lets `attempts_spent` survive an orchestrator that lost
    every scrap of local memory.
    """

    number: PullRef
    ref: TaskRef
    attempt: int = 0
    merged: bool = False
    closed: bool = False
    draft: bool = False
    head_sha: str = ""

    @property
    def open(self) -> bool:
        """Open means open. A merged pull request is closed, and GitHub says so
        with two fields rather than one, so both are checked - a payload that
        carried `merged` without `closed` would otherwise read as a task in
        review forever."""
        return not self.merged and not self.closed


@dataclass(frozen=True)
class AttemptFact:
    """One worker's testimony, reduced to what the budget arithmetic needs.

    A projection of `worker.result.ResultRecord` rather than the record itself,
    and the narrowing is deliberate: the record carries a verify command, a
    tail of output and a list of written files, none of which decides a
    lifecycle state, and a resolver holding them would invite a rule that reads
    one. `consumes_attempt` is restated here from the exit code for the same
    reason the docstring gives for `SATISFYING_STATE_REASONS`.
    """

    ref: TaskRef
    attempt: int
    exit_code: int

    @property
    def consumes_attempt(self) -> bool:
        """`docs/issue-contract.md` §4: only a clean infrastructure failure is free.

        The imported constant rather than a literal `2`, and the same test
        `ResultRecord.consumes_attempt` makes. Restating the rule with a
        magic number here would be a second copy of §4 to drift from the first,
        and this is the one rule in the module where the two sides disagreeing
        would silently unbound a retry budget rather than merely misreport a
        state.
        """
        return self.exit_code != EXIT_INFRASTRUCTURE

    @property
    def spends_budget(self) -> bool:
        """Whether this record moved the counter. **Not the same as §4's rule.**

        `consumes_attempt` answers "may this failure be charged for", and
        `ResultRecord` is right to stop there - it is the worker's testimony and
        the worker does not know what the orchestrator did with it. The counter
        moves somewhere else: `reconcile._observe` calls `_retry_or_give_up`
        only on a path that has already decided the attempt *failed*, and an
        exit 0 "moves no label and writes no counter" because the worker's own
        `claimed -> review` is the transition it caused.

        Reading `consumes_attempt` here instead was the first version of this
        module and it was wrong in the direction that matters: a task that
        succeeded on its third attempt would have read `spent == 3` against a
        cap of 3 and been reported `needs-human` while its pull request sat
        open in review. Precedence hides it once the pull request merges, which
        is exactly how long it would have taken anyone to notice.
        """
        return self.exit_code != 0 and self.consumes_attempt


@dataclass(frozen=True)
class Budget:
    """The caps a task is judged against, as this process was configured.

    Not a fact about the world, and the one input here that is neither code
    host nor container nor artifact. It is the operator's setting, read from
    the environment at the edge (`config.SETTINGS`) - so it is passed in rather
    than reached for, which keeps `resolve` pure and keeps a test from having to
    set an environment variable to describe a run that used a different cap.

    **`max_attempts`, not `max_total_attempts`, is what this module compares
    against**, and the reasoning is in the module docstring: with no renewal
    `reconcile._retry_or_give_up` gives up when `streak` reaches `max_attempts`,
    and `streak` equals `attempt` until a renewal moves them apart. The total
    cap is carried anyway because it is the number a reader wants when the two
    disagree, and because a renewed task really is judged against it.
    """

    max_attempts: int = 3
    max_total_attempts: int = 9


@dataclass(frozen=True)
class Observation:
    """Everything one cycle saw, and nothing it decided.

    The complete input to `resolve`. Frozen, self-contained and serialisable,
    which is what makes a replay corpus possible at all: a recorded cycle is
    this object written down, and a synthesised one is this object written by
    hand. Neither the resolver nor the harness can tell which it was handed,
    which is the property `tests/fixtures/runs/README.md` calls the deliverable.

    Check sets are not here. `orchestrator/checks.py` decides whether a pull
    request may merge, and the answer moves a task between `review` and
    `review` - a pending gate, a failing gate and a passing gate are all
    `review` until something merges or the attempt is consumed. Carrying them
    would put a fact in the input that no rule reads, and the next rule to read
    one would be reading a gate verdict as a lifecycle state.
    """

    cycle: int
    tasks: tuple[TaskFact, ...] = ()
    branches: tuple[TaskBranch, ...] = ()
    containers: tuple[ContainerFact, ...] = ()
    pulls: tuple[PullFact, ...] = ()
    results: tuple[AttemptFact, ...] = ()
    budget: Budget = field(default_factory=Budget)
    #: This process's own run id, plus any sibling the operator declared.
    #: `recovery.py`'s rule, not a second one: a container speaks for a claim
    #: when its `apiary.run` names a run that is live. Empty means "believe every
    #: container", which is the right reading for a corpus that records one run.
    live_run_ids: frozenset[str] = frozenset()


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """One task's derived state, and the fact that decided it.

    `because` is not a log line. It is the thing that makes a divergence
    actionable: "derived said eligible, the label said claimed" is a puzzle,
    and "derived said eligible because no container carries this ref and no
    pull request is open for it, the label said claimed" is a diagnosis. #146
    renders these into a shadow report and #147 puts them in front of an
    operator, so the sentence is written for a human from the start rather than
    retrofitted when somebody has to read one.
    """

    ref: TaskRef
    task_id: str
    state: str
    because: str
    attempts_spent: int = 0
    #: A `PullRef`, not an `int` (#208). It is copied straight off `PullFact`, so
    #: an `int` here would put a bare number back beside `ref` on the one record
    #: the decision path reads - which is the shape #185 removed. Nothing
    #: un-mints it: a verdict is read by a human or a board, never by an
    #: endpoint, and `str(pull)` already renders `#101`.
    pull: PullRef | None = None
    container: str = ""

    def __str__(self) -> str:
        return f"{self.ref} {self.task_id}: {self.state} ({self.because})"


@dataclass(frozen=True)
class Resolution:
    """One cycle's verdicts, keyed by ref.

    Keyed by `TaskRef` because everything the resolver read is (#142), and
    exposing `by_task` alongside because the control plane's own record is
    keyed by the marker slug - `Ledger.entries` is, and so is every payload in
    `events.jsonl`. The join between the two happens here, once, rather than at
    each of the three places that would otherwise do it: `mergeability.py`'s
    docstring is about exactly this class of bug, and #174 is what happens when
    a join like it misses silently.
    """

    cycle: int
    verdicts: tuple[Verdict, ...] = ()

    @property
    def by_ref(self) -> dict[TaskRef, Verdict]:
        return {verdict.ref: verdict for verdict in self.verdicts}

    @property
    def by_task(self) -> dict[str, Verdict]:
        return {verdict.task_id: verdict for verdict in self.verdicts if verdict.task_id}

    def state(self, task_id: str) -> str:
        """This task's derived state, or `""` when the resolver never saw it.

        The empty string rather than a raise, and rather than a default state:
        a task the observation did not carry is a task nothing here has an
        opinion about, and `blocked` would be an opinion.
        """
        verdict = self.by_task.get(task_id)
        return verdict.state if verdict is not None else ""


@dataclass(frozen=True)
class Divergence:
    """One task, one cycle, two states that disagree. **Never a boolean.**

    #145's acceptance criteria say so in as many words, and the reason is the
    shape of the thing being measured. "The resolver agreed 94% of the time" is
    compatible with every disagreement being on `needs-human`, which is the one
    state ADR 0001 reports outbound and the one a customer's tracker cannot
    infer. A count would pass that run. This names the task, the cycle and both
    states, so the interesting case cannot hide inside the uninteresting ones.
    """

    cycle: int
    ref: TaskRef
    task_id: str
    derived: str
    control: str
    because: str = ""

    def __str__(self) -> str:
        tail = f" - derived {self.because}" if self.because else ""
        return (
            f"cycle {self.cycle}: {self.task_id} ({self.ref}) "
            f"derived {self.derived}, control plane {self.control}{tail}"
        )

    @property
    def key(self) -> tuple[int, str, str, str]:
        """What makes two divergences the same one, for set comparison.

        `because` is left out on purpose. It is prose written for a human and
        it will be reworded; a corpus that had to declare it verbatim would
        break on every improvement to a sentence, and a harness people are
        afraid to touch stops being run.
        """
        return (self.cycle, self.task_id, self.derived, self.control)


# --------------------------------------------------------------------------
# The reducer. Pure - no Docker, no network, no clock.
# --------------------------------------------------------------------------


def resolve(observation: Observation) -> Resolution:
    """Every task's derived state for one cycle. Pure.

    Two passes, and the split is forced rather than stylistic: `eligible` is
    defined in terms of the *dependencies'* states, so nothing can be eligible
    until every task's terminal state is known. Doing it in one pass would mean
    resolving a dependency's state twice - once for itself and once for each
    dependent - and a task that resolved differently in the two calls would be
    a bug nobody could see from either call site.
    """
    tasks = tuple(sorted(observation.tasks, key=lambda fact: fact.ref))
    landed = _landed(observation, tasks)
    verdicts = [_verdict(fact, observation, landed) for fact in tasks]
    return Resolution(cycle=observation.cycle, verdicts=tuple(verdicts))


def _landed(observation: Observation, tasks: Sequence[TaskFact]) -> dict[TaskRef, str]:
    """The refs that are done with, and the fact that says so.

    Two ways in, and the second one is not a concession. A merged pull request
    is the code-host fact ADR 0001 names, and it is the one that arrives for
    work apiary did. The other is a work item somebody closed as completed by
    hand - the planner does it when a task is superseded, and half of what a
    real plan waits on is a hand-written issue that no worker will ever touch.
    `github/readiness.IssueState.satisfied` already runs on exactly that rule,
    so a resolver that recognised only merges would report every task waiting
    on hand-finished work as `blocked` while the control plane had long since
    moved on.
    """
    merged: dict[TaskRef, str] = {}
    # Sorted so that a task with two merged pull requests - the orphan
    # `recovery.py` documents, merged by a human rather than closed - names the
    # same one every cycle instead of whichever GitHub's listing happened to
    # return first. The key was an `int`; `PullRef` sorts since #208.
    for pull in sorted(observation.pulls, key=lambda one: one.number):
        if pull.merged:
            # `{pull.number}` renders `#101` on its own - the ref carries the
            # `#` - so there is no second one in the format string. The sentence
            # a human reads is byte-for-byte the one the `int` produced.
            merged.setdefault(pull.ref, f"pull request {pull.number} merged")
    for fact in tasks:
        if fact.ref not in merged and fact.closed_as_completed:
            merged[fact.ref] = "the work item is closed as completed"
    return merged


def _verdict(
    fact: TaskFact, observation: Observation, landed: Mapping[TaskRef, str]
) -> Verdict:
    """One task's state, in `STATES` order. See the module docstring for why that order."""
    spent = _attempts_spent(fact.ref, observation)
    container = _claiming_container(fact.ref, observation)
    pull = _open_pull(fact.ref, observation)

    if fact.ref in landed:
        return Verdict(
            ref=fact.ref,
            task_id=fact.task_id,
            state=LANDED,
            because=landed[fact.ref],
            attempts_spent=spent,
            pull=_merged_pull(fact.ref, observation),
        )

    # Before the budget, because a human's closure outranks apiary's arithmetic
    # and because the sentence a reader wants is the one that names the human.
    # After `landed`, because `closed_as_completed` and `abandoned` are the two
    # halves of one field and a work item cannot be both.
    if fact.abandoned:
        reason = (fact.state_reason or "unknown").replace("_", " ")
        return Verdict(
            ref=fact.ref,
            task_id=fact.task_id,
            state=NEEDS_HUMAN,
            because=f"the work item is closed as {reason}",
            attempts_spent=spent,
            pull=pull.number if pull is not None else None,
        )

    cap = max(int(observation.budget.max_attempts), 1)
    if spent >= cap:
        return Verdict(
            ref=fact.ref,
            task_id=fact.task_id,
            state=NEEDS_HUMAN,
            because=f"{spent} attempt(s) spent against a cap of {cap}",
            attempts_spent=spent,
            pull=pull.number if pull is not None else None,
        )

    if container is not None:
        return Verdict(
            ref=fact.ref,
            task_id=fact.task_id,
            state=CLAIMED,
            because=f"container {container.id[:12]} is running for this task",
            attempts_spent=spent,
            container=container.id[:12],
            pull=pull.number if pull is not None else None,
        )

    if pull is not None:
        return Verdict(
            ref=fact.ref,
            task_id=fact.task_id,
            state=REVIEW,
            because=f"pull request {pull.number} is open for this task",
            attempts_spent=spent,
            pull=pull.number,
        )

    unmet = tuple(ref for ref in fact.depends_on if ref not in landed)
    if unmet:
        names = ", ".join(str(ref) for ref in unmet)
        return Verdict(
            ref=fact.ref,
            task_id=fact.task_id,
            state=BLOCKED,
            because=f"{len(unmet)} dependency/dependencies not landed: {names}",
            attempts_spent=spent,
        )

    return Verdict(
        ref=fact.ref,
        task_id=fact.task_id,
        state=ELIGIBLE,
        because=(
            "every dependency has landed, no container carries this task "
            "and no pull request is open for it"
        ),
        attempts_spent=spent,
    )


def _attempts_spent(ref: TaskRef, observation: Observation) -> int:
    """How many attempts the code host and the artifacts can account for.

    The maximum of three lower bounds, never their sum - each is a different
    partial view of one counter, and adding them would charge a task twice for
    the attempt that both pushed a branch and wrote a record.

    - A **result that spent budget** means the counter moved past that number,
      so it accounts for `attempt + 1`. A result that did not - exit 2
      (`docs/issue-contract.md` §4), or an exit 0 whose transition writes no
      counter - accounts for nothing. `AttemptFact.spends_budget` is where the
      difference between those two exemptions is argued.
    - A **branch** named `apiary/<ref>-attempt-<n>` was pushed by a worker that
      was dispatched when the counter read `n`, so it accounts for `n`. It is
      the only one of the three that survives a crash with no local memory,
      which is what #144 put it there for.
    - An **open or merged pull request** carries the same number in its head
      branch, and is kept separately because a branch listing is one API call a
      caller may not have made while the pull request listing is one every cycle
      makes anyway.

    Taking the maximum is the same direction `_retry_or_give_up` takes when it
    persists the increment before moving the label: an over-count gives up
    sooner, and a counter that can fail to bound retries is worse than one that
    over-counts.
    """
    spent = 0
    for record in observation.results:
        if record.ref == ref and record.spends_budget:
            spent = max(spent, record.attempt + 1)
    for branch in observation.branches:
        if branch.ref == ref:
            spent = max(spent, branch.attempt)
    for pull in observation.pulls:
        if pull.ref == ref:
            spent = max(spent, pull.attempt)
    return spent


def _claiming_container(ref: TaskRef, observation: Observation) -> ContainerFact | None:
    """The running container that speaks for this task's claim, if any.

    Three filters, and each one is a way a claim is wrongly kept or wrongly
    released. The ref has to match, because a container is labelled with the
    task it was spawned for. It has to be *running*, because `docker ps --all`
    lists the exited one whose worker finished this cycle and whose pull request
    is already open - see `ContainerFact.running`. And its run has to be live,
    because `orchestrator/recovery.py` makes a container of a dead run an orphan
    that holds nothing: run ids are never reused, so a container wearing another
    id belongs to a process that is gone.

    Empty `live_run_ids` believes every container. That is the right default for
    a corpus - one run, recorded from inside it - and the wrong one for a
    machine shared with a sibling orchestrator, which is why the field exists.
    """
    live = observation.live_run_ids
    for container in observation.containers:
        if container.ref != ref or not container.running:
            continue
        if live and container.run_id not in live:
            continue
        return container
    return None


def _open_pull(ref: TaskRef, observation: Observation) -> PullFact | None:
    """The open pull request for this task - the newest attempt's, if several.

    Several is not hypothetical. `orchestrator/recovery.py` says so: since #144
    every attempt has a branch of its own, so a retry dispatched over a worker
    that had in fact published opens a *second* pull request rather than
    updating the first, and a human has to close the one the crash orphaned.
    Until they do, two are open for one task. The highest attempt is the one the
    run is actually waiting on; the other is a leftover, and reporting the
    leftover's number would send a reader to the wrong diff.

    The number is the tiebreak when both are on the same attempt, and it is the
    third site that needed `PullRef` to sort (#208) - the `>` below is a tuple
    comparison, so it reaches the second element only when the attempts are
    equal, and an unordered second element would have made that comparison a
    `TypeError` rather than a wrong answer. Two open pull requests on one
    attempt means a worker published twice for one dispatch; the higher number
    is the newer one.
    """
    best: PullFact | None = None
    for pull in observation.pulls:
        if pull.ref != ref or not pull.open:
            continue
        if best is None or (pull.attempt, pull.number) > (best.attempt, best.number):
            best = pull
    return best


def _merged_pull(ref: TaskRef, observation: Observation) -> PullRef | None:
    """The merged pull request this task landed through, lowest number first.

    The same key as `_landed`, and it has to be: that function writes the
    sentence this number is rendered beside, so a reader told "pull request #101
    merged" and linked to #104 would be sent to a diff that explains nothing.
    What the ordering buys is that neither answer depends on the order GitHub
    listed the pull requests in - which is not stable between cycles, and is not
    the same in a replay as it was in the run being replayed.
    """
    for pull in sorted(observation.pulls, key=lambda one: one.number):
        if pull.ref == ref and pull.merged:
            return pull.number
    return None


# --------------------------------------------------------------------------
# The diff against the control plane
# --------------------------------------------------------------------------


def diverge(
    resolution: Resolution, control: Mapping[str, str], *, cycle: int | None = None
) -> tuple[Divergence, ...]:
    """Where this resolution and the control plane disagree, named one by one.

    `control` is keyed by task id and its values are already in the internal
    vocabulary - `lifecycle.INTERNAL_STATE` is the translation and it belongs to
    whoever *reads* a label, not here. This module has never seen a `swarm:*`
    string and the module docstring explains at length why that is worth the
    small awkwardness of making the caller translate.

    A task in one side and not the other is not reported. The control plane
    holds work items this resolver was never shown - a malformed issue that
    never entered the ledger (`docs/issue-contract.md` §1.4), a task from
    another run - and calling those divergences would drown the ones that mean
    something in ones that mean "these are two different sets". #146's shadow
    report is where a coverage number belongs.
    """
    at = resolution.cycle if cycle is None else cycle
    found: list[Divergence] = []
    for verdict in resolution.verdicts:
        expected = control.get(verdict.task_id)
        if expected is None or expected == verdict.state:
            continue
        found.append(
            Divergence(
                cycle=at,
                ref=verdict.ref,
                task_id=verdict.task_id,
                derived=verdict.state,
                control=expected,
                because=verdict.because,
            )
        )
    return tuple(found)


# --------------------------------------------------------------------------
# The edges. Everything above this line is pure.
# --------------------------------------------------------------------------


class _Entry(Protocol):
    """The slice of `github.ledger.LedgerEntry` an observation is built from.

    A protocol rather than the class, so `observe` does not drag the ledger -
    and through it the GitHub client - into anything that wants to build an
    observation. It is also the honest statement of how little of an entry this
    needs: a ref, a slug, its declared dependencies and whether it is closed.
    Notably **not** `state_label`, which the real entry carries and which
    nothing here may look at.
    """

    @property
    def ref(self) -> TaskRef: ...

    @property
    def task_id(self) -> str: ...

    @property
    def blocked_by(self) -> tuple[TaskRef, ...]: ...

    @property
    def closed(self) -> bool: ...


def observe(
    *,
    cycle: int,
    entries: Iterable[_Entry],
    branch_names: Iterable[str] = (),
    containers: Iterable[ContainerFact] = (),
    pulls: Iterable[PullFact] = (),
    results: Iterable[AttemptFact] = (),
    budget: Budget | None = None,
    live_run_ids: Iterable[str] = (),
    state_reasons: Mapping[TaskRef, str | None] | None = None,
) -> Observation:
    """Assemble one `Observation` from what a cycle already holds. Read-only.

    The whole of this module's I/O, and there is deliberately none of it here
    either: the caller has already listed the containers, read the pulls and
    loaded the results, because a cycle does all three for its own reasons and a
    resolver that repeated them would double the API budget to answer a question
    nobody has wired up yet. `orchestrator/shadow.py` is the caller (#146), and
    it passes only facts the cycle already holds.

    Branch names come in as strings and are parsed here, because
    `github/branches.parse_task_branch` answers `None` for everything apiary did
    not mint - `main`, a human's `fix/typo`, every `swarm/issue-<n>` from before
    #144 - and the caller counting those rather than acting on them is the
    discipline that module's docstring asks for. Dropping them silently is
    correct: a branch this system did not create says nothing about a task.

    `state_reasons` is the one fact an `_Entry` does not carry and `_landed`
    needs. A closed work item discharges a dependency only when it was closed
    **as completed** - `github/readiness.IssueState.satisfied`'s rule, and
    `reconcile._closed_verdict`'s - so without it every issue somebody closed as
    not planned would read `landed` here while the control plane escalated it.
    Keyed by ref and optional, because the caller that has a snapshot of issue
    states has it for free (#146) and a caller that does not should not have to
    invent one. It is a code-host fact and not a state: `TaskFact` has carried
    the field since #145 and the sourcing invariant is untouched.
    """
    parsed: list[TaskBranch] = []
    for name in branch_names:
        branch = parse_task_branch(name)
        if branch is not None:
            parsed.append(branch)
    reasons: Mapping[TaskRef, str | None] = state_reasons or {}
    return Observation(
        cycle=cycle,
        tasks=tuple(
            TaskFact(
                ref=entry.ref,
                task_id=entry.task_id,
                depends_on=tuple(entry.blocked_by),
                closed=entry.closed,
                # A closed work item the caller recorded no reason for reads as
                # completed, matching `readiness.SATISFYING_STATE_REASONS`'
                # inclusion of `None`: GitHub omits the field on issues closed
                # before it existed, and treating those as "not planned" would
                # strand every plan that depends on old work.
                state_reason=reasons.get(entry.ref),
            )
            for entry in entries
        ),
        branches=tuple(parsed),
        containers=tuple(containers),
        pulls=tuple(pulls),
        results=tuple(results),
        budget=budget or Budget(),
        live_run_ids=frozenset(live_run_ids),
    )


def report(resolution: Resolution, divergences: Iterable[Divergence] = ()) -> str:
    """One cycle, rendered for a human. The text #146's shadow window prints.

    Here rather than in the caller because the two halves have to be readable
    *together*: a divergence names two states and the verdict names the fact
    that produced one of them, and a reader who has to correlate two separate
    reports by task id will stop reading the second one. Everything this prints
    is already on the objects; nothing is recomputed, so the report cannot
    disagree with the resolution it describes.
    """
    lines = [f"cycle {resolution.cycle}: {len(resolution.verdicts)} task(s)"]
    lines += [f"  {verdict}" for verdict in resolution.verdicts]
    found = tuple(divergences)
    if not found:
        lines.append("  no divergence from the control plane")
        return "\n".join(lines)
    lines.append(f"  {len(found)} divergence(s):")
    lines += [f"    ! {divergence}" for divergence in found]
    return "\n".join(lines)
