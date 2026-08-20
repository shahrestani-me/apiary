"""Claims with nothing behind them, and the issues they make unreachable.

[#20](https://github.com/shahrestani-me/apiary/issues/20) reaps the orphaned
*containers*; this module reaps the orphaned *labels*, and the two halves are
the same fault seen from the two sides of `docs/issue-contract.md` §4. The
dispatcher (#21) writes `swarm:claimed` **before** it spawns, which is the
correct order - the reverse loses issues to crashes outright - and the price is
a window in which a claim exists with no container behind it. GitHub is
authoritative and this process holds no memory across restarts, so nothing else
in the system will ever take that label off. The issue is then permanently
undispatchable while looking perfectly healthy, which is the worst shape a
failure takes here.

Two rows of §4 belong to this module and to nothing else:

    claimed -> ready   stale claim, no live container - attempt consumed
    claimed -> review  stale claim but an open PR exists (the worker died
                       after pushing, before it could relabel)

**Liveness is #20's rule, not a second one.** A container speaks for a claim
when its `apiary.run` names a run that is live: this process's, or one the
caller declared. Ids are never reused ([`run.py`](../run.py)), so a container
labelled with any other id belongs to a process that is gone and holds nothing
- it is an orphan the reaper is already on its way to removing. Inventing a
separate lease, heartbeat or pid file for the label half would put durable
state back on the local disk, which is the thing `docs/architecture-v2.md`
spent the whole control plane removing, and it would leave two liveness rules
in one system to disagree with each other.

The assumption that rule rests on is worth restating rather than hiding:
**concurrent orchestrators against one repository are unsupported by default**,
exactly as [`reaper.py`](../containers/reaper.py) says for containers. The
escape hatch is the same one and has the same name: `live_run_ids` spares every
claim a sibling's containers are holding. What it cannot cover is a sibling's
own claim-to-spawn window, which is invisible from here by construction - there
is no container to see - and is the same sub-second window #20 leaves open.

**Nothing is stolen silently.** A claim that is left standing is reported with
the reason, because "recovery ran and released nothing" and "recovery ran and
decided four claims were somebody else's" are different facts, and only the
second is reassuring to a human wondering why an issue is stuck.

## Beyond the ticket: startup is not the only moment

#35 scopes recovery to startup, and the window that actually opened during
development was mid-cycle: three issues were labelled `swarm:claimed`, the
first was spawned, and the process was interrupted before the other two ever
reached `manager.spawn`. For about a minute two issues were claimed with
nothing running. A startup-only sweep does not run again until somebody
restarts the orchestrator, so had the process died a little later those two
would have sat there until a human noticed - the exact failure the ticket is
about, reached by a route the ticket does not cover.

So the sweep is a pure pass over facts a cycle has already read, cheap enough
to run every cycle, and it holds no state between passes - which is precisely
what makes "at startup" and "every cycle" the same code with no mode flag. Two
properties make the mid-cycle pass safe rather than merely convenient:

- The claim and the spawn are one call in one single-threaded loop
  (`dispatcher.dispatch`), so there is no moment at which this module can
  observe a claim whose spawn is still in flight.
- `docker ps --all` reports a container from the instant `docker create`
  returns, and `ContainerManager.spawn` disposes what it created before
  re-raising. A claim with no container is therefore either one whose spawn
  never happened or one whose container is gone - and both are claims this
  module is *supposed* to release. `dispatcher.dispatch` says so in as many
  words: a failed spawn keeps its claim, because #35 resolves the ambiguity by
  looking for a live container before it releases anything.

## Two decisions that are not the reconciler's

**The counter is bumped on release and not on publication** (§5). A claim
returned to the pool has cost an attempt: an issue that crashes the
orchestrator is precisely the issue that should reach `swarm:failed` rather
than loop, and §5 makes the counter an upper bound on attempts made for exactly
this reason. The `review` path bumps nothing - that worker finished its work,
and charging it an attempt for a label that did not stick would give up on a
task that succeeded.

**An unreadable pull-request list does not stop the sweep.** #22 refuses to act
while blind, and is right to: a `swarm:review` issue is in a state that is
already correct, and moving it on a guess can only make things worse. Here the
subject is a `swarm:claimed` issue already known to have nothing behind it, so
the states are not symmetrical. Holding it would leave the ticket's own
acceptance criterion unmet with today's client, which has no
`list_pull_requests` at all.

What releasing costs changed with #144 and is worth restating rather than
leaving as a stale claim. It used to cost one redundant attempt and nothing
else: one branch served every attempt, so `POST /pulls` for a head that already
had an open PR was a 422 and the retry updated the pull request that existed.
An attempt now has a branch of its own, so a retry dispatched over a worker that
had in fact published opens a *second* pull request rather than updating the
first, and a human has to close the one the crash orphaned.

That is still the better side of the trade, and narrowly so rather than by a
wide margin. The path needs a client that cannot list pull requests at all -
today's can, so the readable path above catches this case and moves the label
forward without spending anything. Holding instead does not converge at all: the
claim stays, nothing is running behind it, and no later sweep has any more
information than this one did. And a second pull request that overlaps a
finished one is at least *visible*, where the alternative - updating the older
pull request's title and body while its head still points at the previous
attempt's commit - would put a description in front of a reviewer that does not
describe the code underneath it.

## The branch name is the primary source; the label is the fallback

ADR 0001 makes agent execution state derived rather than stored, and #144 put
the two facts a crash destroys into the one artefact that survives it: a branch
named `apiary/<ref>-attempt-<n>` says which task it is for and how much budget
it has spent. So the row above that decides "the worker published and died"
now matches on the **ref parsed out of an open pull request's head branch**
(`github/branches.parse_task_branch`), not on `LedgerEntry.branch`.

The difference is the whole reason the name carries a pair. `entry.branch`
names the ticket's *current* attempt, rebuilt from a counter this very sweep may
be about to move; the head branch on the remote names the attempt that actually
pushed. Comparing the two as strings is right until they disagree, and the case
where they disagree is a crash - which is the only case this module runs for.

`swarm:claimed` is still what selects the entries this pass considers, and that
is deliberate rather than unfinished: epic #140 removes the label store in T6,
and until it does the label is the fallback that keeps this sweep working on a
repository whose branches predate #144. Those branches do not parse and are not
guessed at - they are counted onto `RecoveryPlan.unrecognised` and reported,
for the reason every other refusal in this module is reported.

This module removes no container, on any path. If a claim is stale because its
run died, the container that run left behind is #20's to sweep, and doing it
from here would put two modules on one `docker rm`.

Manual dry run against a real repo - reads only, writes nothing, removes
nothing:

    GITHUB_TOKEN=... python -m swarm.orchestrator.recovery shahrestani-me/apiary
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Collection, Iterable, Mapping

from ..config import SETTINGS
from ..containers.manager import DockerCLI, Handle, Redactor, find_containers
from ..github.branches import TaskBranch, parse_task_branch
from ..github.client import GitHubClient
from ..github.ledger import Ledger, LedgerEntry, load_ledger
from ..github.readiness import IssueState
from ..github.refs import task_ref
from ..run import Run, RunError, validate_run_id
from ..taskref import TaskRef
from .authority import Belief, state_of
from .derived import CLAIMED as CLAIMED_STATE
from .derived import ELIGIBLE, NEEDS_HUMAN
from .derived import REVIEW as REVIEW_STATE
from .dispatcher import CLAIMED, REVIEW
from ..store import TaskStore
from .reconcile import ReconcilePlan, ReconcileReport, Snapshot, Transition, apply_plan


# --------------------------------------------------------------------------
# Liveness
# --------------------------------------------------------------------------


def live_runs(run: Run | None = None, extra: Collection[str] = ()) -> frozenset[str]:
    """The run ids whose containers still speak for a claim.

    Caller-declared ids are taken verbatim and never validated, for
    `Reaper._live_ids`' reason: declaring something live is the safe direction,
    and refusing to spare a sibling's claim because its id looked wrong is the
    one failure mode worth designing out.
    """
    live = set(extra)
    if run is not None:
        live.add(run.id)
    return frozenset(live)


def holders(containers: Iterable[Handle], live: Collection[str]) -> dict[TaskRef, Handle]:
    """Task ref -> a container that could plausibly be running it.

    Two kinds of container answer for an issue, and both of them are "not ours
    to release":

    - one whose run is live, in any state. A container that has *exited* still
      counts, because reading its exit code and moving its label is #22's row
      and not this one's; recovery deciding the same issue from a weaker signal
      is how two modules end up writing one label.
    - one wearing `apiary.run` with a value this system could not have minted.
      The reaper spares those because a container it did not create is not its
      to remove; the same container is not this module's to release a claim
      over either, and the alternative is stealing a claim from whatever else
      is wearing our labels.

    A container of a *dead* run is what is left out, and that is the whole
    signal: its orchestrator is gone, nobody will ever read its exit code, and
    #20 is on its way to removing it.
    """
    found: dict[TaskRef, Handle] = {}
    for handle in containers:
        if handle.issue is None:
            continue
        if handle.run_id in live or not _is_run_id(handle.run_id):
            # The docker label is an issue number (`containers/manager.py`);
            # minted here so this map keys on the same identity the ledger does.
            found.setdefault(task_ref(int(handle.issue)), handle)
    return found


def in_flight(branches: Iterable[str]) -> dict[TaskRef, TaskBranch]:
    """Task ref -> the furthest attempt any of `branches` was pushed for.

    The derivation ADR 0001 promises: hand it the head refs of a repository's
    open pull requests and it reconstructs which tasks got as far as a worker,
    and how many attempts each of them cost, with no ledger, no label and no
    local memory involved.

    Furthest rather than first because a ref can legitimately own more than one
    branch - one per attempt - and only the newest speaks for where the task is
    now. An older attempt's branch outliving its pull request is not a
    contradiction to resolve, it is history.
    """
    found: dict[TaskRef, TaskBranch] = {}
    for name in branches:
        parsed = parse_task_branch(name)
        if parsed is None:
            continue
        seen = found.get(parsed.ref)
        if seen is None or parsed.attempt > seen.attempt:
            found[parsed.ref] = parsed
    return found


def unrecognised(branches: Iterable[str]) -> tuple[str, ...]:
    """The names this parser could not read, sorted, for reporting only.

    Branches from before #144, and any a human pushed. Named rather than
    dropped: a sweep that silently ignored a branch it did not understand is
    indistinguishable from one that found nothing, and the first time that
    matters is a migration - a repository mid-run when the naming changed,
    where every claim looks abandoned and every one of them is not.
    """
    return tuple(sorted(name for name in branches if parse_task_branch(name) is None))


def _is_run_id(value: str) -> bool:
    try:
        validate_run_id(value)
    except RunError:
        return False
    return True


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Held:
    """A claim this pass deliberately left standing, and why.

    Carried rather than discarded: an issue that stays `swarm:claimed` through
    a sweep is either healthy or stuck, the two look identical from the
    tracker, and this sentence is the only thing that tells them apart.
    """

    ref: TaskRef
    reason: str

    def __str__(self) -> str:
        return f"{self.ref}: held ({self.reason})"


@dataclass(frozen=True)
class RecoveryPlan:
    """Every claim this pass would move, and every one it would not.

    Pure, so the interesting cases - a sibling's container, an orphan's, a
    worker that published and died, a claim at the attempt cap - are testable
    as data rather than against a daemon.
    """

    transitions: tuple[Transition, ...] = ()
    held: tuple[Held, ...] = ()
    #: True when pull-request state could not be read at all. Unlike #22 this
    #: does not suppress a rule - see the module docstring - but a released
    #: claim that might have had a PR behind it is worth saying out loud.
    blind: bool = False
    #: Head branches this pass could not read a `(ref, attempt)` out of. Every
    #: one of them is a pull request no claim can ever be matched against, so
    #: they are the difference between "nothing was in flight" and "something
    #: was, on a name from before #144".
    unrecognised: tuple[str, ...] = ()

    @property
    def released(self) -> tuple[Transition, ...]:
        """Claims returned to the pool, or handed to a human at the cap."""
        return tuple(item for item in self.transitions if item.to_state != REVIEW_STATE)

    @property
    def published(self) -> tuple[Transition, ...]:
        """Claims whose worker finished and died before it could relabel."""
        return tuple(item for item in self.transitions if item.to_state == REVIEW_STATE)

    @property
    def changed(self) -> bool:
        return bool(self.transitions)

    @property
    def refs(self) -> tuple[TaskRef, ...]:
        return tuple(transition.ref for transition in self.transitions)

    def summary(self) -> str:
        parts = [
            f"released {len(self.released)} stale claim(s)",
            f"{len(self.published)} to review",
            f"{len(self.held)} held",
        ]
        if self.blind:
            parts.append("pull request state unreadable")
        if self.unrecognised:
            parts.append(f"{len(self.unrecognised)} branch name(s) not apiary's")
        return ", ".join(parts)


def _release(entry: LedgerEntry, max_attempts: int,
    *,
    believed: Belief | None = None,
) -> Transition:
    """Consume an attempt for a claim nothing was running, and decide the label.

    The increment rides on the transition so `apply_plan` persists it *before*
    the label goes back to `swarm:ready` (§5): a crash between the two costs an
    attempt rather than granting a free one. An orchestrator that dies in the
    claim-to-spawn window is exactly the run that dies there again, and a
    counter that fails to bound that loop is worse than one that over-counts.
    """
    attempt = entry.attempt + 1
    cap = max(int(max_attempts), 1)
    reason = "claimed with no live container behind it"
    if attempt >= cap:
        return Transition(
            ref=entry.ref,
            from_state=state_of(entry, believed),
            to_state=NEEDS_HUMAN,
            reason=f"{reason}; {attempt} attempt(s) made against a cap of {cap}",
            task_id=entry.task_id,
            attempt=attempt,
            comment=(
                f"apiary: recovered a stale `{CLAIMED}` label after {attempt} attempt(s), "
                f"which is the cap. No container was running this issue and no pull request "
                f"was open for it."
            ),
        )
    return Transition(
        ref=entry.ref,
        from_state=state_of(entry, believed),
        to_state=ELIGIBLE,
        reason=reason,
        task_id=entry.task_id,
        attempt=attempt,
    )


def plan_recovery(
    ledger: Ledger,
    *,
    containers: Iterable[Handle] = (),
    live_run_ids: Collection[str] = (),
    states: Mapping[TaskRef, IssueState] | None = None,
    open_branches: Collection[str] | None = None,
    max_attempts: int = SETTINGS.max_attempts_per_task,
    believed: Belief | None = None,
) -> RecoveryPlan:
    """Decide what each claimed issue's claim is worth. Pure.

    `containers` is **every** apiary container on the daemon, not one run's:
    the question each claim asks is whether *anybody* is running it, and a
    listing narrowed to this run answers a different one and would release a
    sibling's work. `live_run_ids` then decides which of those count.

    `believed` is the cycle's authority on state (#147), and this loop is one of
    the places that criterion is about rather than a nicety: a released claim
    **consumes an attempt** (`_release`), so a `swarm:claimed` somebody typed
    onto a ready issue mid-run burned a retry off a task that had never run.
    The resolver reads a claim off a running container, so a label with no
    worker behind it selects nothing here and nothing is spent.

    **Under the resolver this sweep stops writing in the live cycle, and that
    is the finding rather than a regression.** `derived._claiming_container`
    says `claimed` only for a *running* container of a live run, and `holders`
    above answers for that same container and more - it keeps the exited one
    too - so every entry the authority calls claimed is one rule 2 holds. The
    three cases the label used to bring here are each answered better
    elsewhere, and none of them costs an attempt any more:

    - **the claim-then-spawn gap**, no container ever started. Under the labels
      this was undispatchable forever, because the *dispatcher* believed the
      label too. It now reads the same authority this does, sees `eligible`,
      and simply dispatches the task again - `plan_reconcile`'s rule 3 does not
      fire, because no worker wrote a result.
    - **a worker that died after publishing.** The resolver reads `review` off
      the open pull request, and the merge gate asks `authority.in_review`, so
      the label being behind stops mattering; moving it was all rule 3 here did.
    - **a worker that finished.** `plan_reconcile`'s rule 3 observes the result
      it wrote, which is the accounting `_release` was approximating from a
      missing container.

    `Recovery.startup` keeps every one of them, and deliberately: it runs before
    there is a cycle, on labels a *dead* orchestrator left behind, where the
    label is the only record of what was claimed. That is the same seam
    `authority.Belief.previous` argues for, one process earlier.

    `believed=None` reads the label, which is every caller outside
    `Reconciler.cycle`: `Recovery.startup`, the `__main__` dry run, and
    `APIARY_STATE_SOURCE=labels`.

    Nothing here is written, no API is called and no daemon is touched, which
    is what lets the rules be asserted as data.
    """
    states = states or {}
    live = holders(containers, live_run_ids)
    # The derived half (#144). `open_branches` is a set of head refs, and every
    # one that parses names a task and an attempt - which is what lets a claim
    # be matched to the pull request behind it without either of them having to
    # agree with a counter this pass may be about to move.
    published = in_flight(open_branches or ())

    transitions: list[Transition] = []
    held: list[Held] = []

    for entry in sorted(ledger.entries.values(), key=lambda entry: entry.ref):
        if state_of(entry, believed) != CLAIMED_STATE:
            continue

        # 1. GitHub wins, and a closed issue is out of the run whatever its
        #    labels say. What it becomes - `done` or `failed` - is #22's rule
        #    and reading it from here would be a second opinion on one fact.
        state = states.get(entry.ref)
        if state is not None and state.closed:
            held.append(
                Held(entry.ref, "closed on GitHub; the reconciler decides what it becomes")
            )
            continue

        # 2. Somebody is running it. The claim is doing its job.
        handle = live.get(entry.ref)
        if handle is not None:
            held.append(Held(entry.ref, f"a container of run {handle.run_id!r} is holding it"))
            continue

        # 3. The worker got as far as a pull request and no cycle survived to
        #    observe its result - the orchestrator died with the claim
        #    outstanding, which is what `Recovery.startup` runs on. (Before #148
        #    the worker wrote `swarm:review` itself and this row covered a worker
        #    dying between the push and that write; the write now belongs to
        #    `reconcile._verdict`, so the case is the orchestrator's death rather
        #    than the container's.) The work exists; moving the label forward is
        #    all that is left.
        #
        #    Matched on the ref inside the head branch rather than on
        #    `entry.branch`, for the module docstring's reason: the name on the
        #    remote is the attempt that actually pushed, and the entry's is
        #    whatever the counter currently says. A string comparison agrees
        #    with that right up until a crash, and a crash is when this runs.
        branch = published.get(entry.ref) if open_branches is not None else None
        if branch is not None:
            transitions.append(
                Transition(
                    ref=entry.ref,
                    from_state=state_of(entry, believed),
                    to_state=REVIEW_STATE,
                    reason=(
                        f"{branch} has an open pull request; "
                        f"no cycle survived to observe the result"
                    ),
                    task_id=entry.task_id,
                )
            )
            continue

        # 4. A claim with nothing behind it at all. This is the ticket.
        transitions.append(_release(entry, max_attempts, believed=believed))

    return RecoveryPlan(
        transitions=tuple(transitions),
        held=tuple(held),
        blind=open_branches is None,
        unrecognised=unrecognised(open_branches or ()),
    )


# --------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryReport:
    """What one sweep planned and what its writes achieved.

    Both halves are kept because they fail independently: GitHub refusing one
    label leaves that claim exactly where it was, and `ReconcileReport` is
    where that lands (it collects rather than raises, for the reason every
    other sweep in this codebase does).
    """

    plan: RecoveryPlan
    result: ReconcileReport

    @property
    def applied(self) -> tuple[Transition, ...]:
        """The transitions that actually landed - what a caller may `fold`."""
        return self.result.applied

    @property
    def refs(self) -> tuple[TaskRef, ...]:
        return tuple(transition.ref for transition in self.applied)

    @property
    def ok(self) -> bool:
        return self.result.ok

    def summary(self) -> str:
        return f"{self.plan.summary()}; {self.result.summary()}"


@dataclass
class Recovery:
    """The sweep, at startup and at the top of every cycle after it.

    Holds a client, optionally this process's `Run`, and no state that survives
    a pass. Two `Recovery` objects over one repository reach the same verdicts
    because both of them ask GitHub and the daemon rather than each other -
    which is the property that lets the same object be used for the startup
    sweep and the mid-cycle one.
    """

    #: Anything with the label calls and `get_issue`/`update_issue`. A
    #: `Snapshot` satisfies it too, which is how a mid-cycle sweep costs no
    #: extra issue listing.
    client: Any

    #: This process's run. Its containers are live; without one, every
    #: container on the daemon is somebody else's and `live_run_ids` is the
    #: only thing sparing anything.
    run: Run | None = None

    #: Runs the caller knows are alive, verbatim from `Reaper.live_run_ids`.
    #: The named escape from "one orchestrator per repository".
    live_run_ids: Collection[str] = ()

    #: The daemon. Given one rather than built one when the caller already has
    #: a run's `DockerCLI`, so one `docker ps` can serve both halves of the
    #: sweep.
    docker: DockerCLI | None = None

    max_attempts: int = SETTINGS.max_attempts_per_task

    #: Where apiary's own judgments live (#159). A released claim consumes an
    #: attempt through a channel that has no verify output to sign, so the
    #: judgment written here carries no signature - which is the deliberate
    #: "clear the record" direction §5 describes: a stale signature could renew
    #: a budget the blocker never released, and the counter must over-bound
    #: rather than under-bound. `None` leaves the store alone, and the outcome
    #: is the same either way, because a judgment stamped at an attempt the
    #: counter has since passed is read as stale and dropped
    #: (`store.TaskJudgement.matches`). It is passed all the same, so the
    #: store never sits holding a belief that has already been superseded -
    #: a run somebody is debugging should not have to know the staleness rule
    #: to read it.
    store: TaskStore | None = None

    #: Plans and prints, writes nothing - including no marker adoption, which
    #: is a body `PATCH` a dry run promised not to make.
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.docker is None:
            # A bare `DockerCLI` redacts nothing it was told about, but
            # `SECRET_PATTERNS` still catches a token by shape - and the only
            # text this module takes from the daemon is a `docker ps` table.
            self.docker = DockerCLI(redact=Redactor())

    # --- what the sweep reads --------------------------------------------

    def containers(self) -> list[Handle]:
        """Every apiary container on the daemon. One `docker ps`, whatever the count.

        Deliberately not narrowed to this run: a claim asks whether anybody is
        running it, and `find_containers` already filters on `apiary.run` so
        the human's own containers are never in the list.
        """
        return find_containers(self._cli)

    # --- one pass ---------------------------------------------------------

    def plan(
        self,
        ledger: Ledger,
        *,
        containers: Iterable[Handle] | None = None,
        states: Mapping[TaskRef, IssueState] | None = None,
        open_branches: Collection[str] | None = None,
        believed: Belief | None = None,
    ) -> RecoveryPlan:
        """What this pass would do, without doing any of it."""
        return plan_recovery(
            ledger,
            containers=self.containers() if containers is None else containers,
            live_run_ids=live_runs(self.run, self.live_run_ids),
            states=states,
            open_branches=open_branches,
            max_attempts=self.max_attempts,
            believed=believed,
        )

    def sweep(
        self,
        ledger: Ledger,
        *,
        containers: Iterable[Handle] | None = None,
        states: Mapping[TaskRef, IssueState] | None = None,
        open_branches: Collection[str] | None = None,
        believed: Belief | None = None,
    ) -> RecoveryReport:
        """Plan and write. The mid-cycle entry point.

        Every fact is a parameter because a cycle has already read all of them
        (`reconcile.Snapshot`), and re-reading them here would spend the
        polling budget to learn what the caller is holding. `believed` is one
        more of them, and the one that decides which claims are this pass's to
        speak about - see `plan_recovery`. The writes go through
        `reconcile.apply_plan`, which owns the ordering §5 requires - counter,
        then add, then remove - so there is one implementation of it rather
        than two that can drift.
        """
        plan = self.plan(
            ledger,
            containers=containers,
            states=states,
            open_branches=open_branches,
            believed=believed,
        )
        result = apply_plan(
            self.client,
            ReconcilePlan(transitions=plan.transitions),
            store=self.store,
            dry_run=self.dry_run,
        )
        return RecoveryReport(plan=plan, result=result)

    def startup(self) -> RecoveryReport:
        """The ticket's row: sweep the claims a dead orchestrator left behind.

        Reads the ledger itself, because a startup sweep runs before there is a
        cycle to share a read with. One issue listing serves the ledger, each
        issue's open/closed state and the pull-request probe, exactly as it
        does inside a cycle.

        **It resolves a belief of its own, and #152 is why it has to.** The
        sweep decides on a task's state, and until this ticket that state was
        the `swarm:claimed` label a dead process had left on the issue - which
        is precisely what "sweep what a dead orchestrator left labelled" meant.
        There is no such label now, so a startup sweep with no belief asks
        `state_of` a question it can only raise on, and the method was
        unreachable in practice.

        The observation is built from the same one listing plus the daemon's
        container list, which is what a cycle's first pass does. `believe` with
        no `remembered` is the honest shape here: a startup sweep is by
        definition the first sight of this repository by this process, so there
        is nothing carried forward for it to consult.
        """
        from .derived import build_observation
        from .authority import believe

        snapshot = Snapshot(self.client)
        ledger = load_ledger(
            snapshot,  # type: ignore[arg-type]
            store=self.store,
        )
        states = snapshot.states()
        open_branches = snapshot.open_branches()
        containers = self.containers()
        observation = build_observation(
            cycle=0,
            entries=ledger.entries.values(),
            branch_names=open_branches or (),
            containers=containers,
            state_reasons={ref: state.state_reason for ref, state in states.items()},
        )
        return self.sweep(
            ledger,
            containers=containers,
            states=states,
            open_branches=open_branches,
            believed=believe(ledger, observation),
        )

    # --- plumbing ---------------------------------------------------------

    @property
    def _cli(self) -> DockerCLI:
        assert self.docker is not None  # set in __post_init__
        return self.docker


__all__ = [
    "Held",
    "Recovery",
    "RecoveryPlan",
    "RecoveryReport",
    "holders",
    "in_flight",
    "live_runs",
    "plan_recovery",
    "unrecognised",
]


if __name__ == "__main__":  # pragma: no cover - manual dry run, see module docstring
    import os

    repo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_REPOSITORY", "")
    # Read-only on every path: no label, no comment, no adoption, no removal.
    dry = Recovery(client=GitHubClient.from_env(repo), dry_run=True).startup()
    for planned in dry.plan.transitions:
        print(f"would {planned}")
    for standing in dry.plan.held:
        print(f"would leave {standing}")
    print(dry.plan.summary())
