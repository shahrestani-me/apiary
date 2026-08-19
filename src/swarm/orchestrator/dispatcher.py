"""Ready issues in, running containers out - and nothing else at the same time.

Step 3 of `docs/architecture-v2.md`'s orchestration loop, and the row of
`docs/issue-contract.md` §4 that reads `ready -> claimed, picked for dispatch,
**before** the container is spawned, dispatcher (#21)`. Readiness (#11) decides
*which* issues may run; this module decides *how many* and *which of those at
once*, and it is the only thing in the system that spawns a worker.

**Claim, then spawn. Never the other way round.** Both orders have a crash
window and they are not equally bad. Claiming first can strand a
`swarm:claimed` issue with no container, which #35 recovers by looking for a
live container and finding none. Spawning first can leave a container running
against an issue still labelled `swarm:ready`, which the next cycle dispatches
again - two containers editing one file set, and the second one's push destroys
the first one's. So the claim is written first, one issue at a time,
immediately before its spawn, and a spawn that fails leaves the claim standing
rather than being tidied away (see `dispatch`).

**Overlapping `## Files` sets serialize.** This is v1's rule and it survives
unchanged: two workers editing one file produce two branches that cannot both
merge, and the wrong place to discover that is inside a running container. The
planner is *supposed* to emit disjoint sets, and `ledger.parse_contract`
refuses globs precisely so the sets can be intersected here - but "the planner
promised" is not a mechanism. A human retitling and re-scoping an issue
mid-run, which architecture-v2 calls a feature, is enough to break the
promise. So the intersection is computed every cycle against everything already
in flight, case-insensitively, because the development host's filesystem is
(`docs/issue-contract.md` §1.3).

**The cap is not a worker count, it is an inference budget.** Containers buy
isolation, not parallelism (architecture-v2, second constraint):
`OLLAMA_NUM_PARALLEL` is a property of the host server, so N containers queue
against the same slots and the N+1th buys queueing. Two things the naive cap
misses, and `Capacity` exists for both:

- **The orchestrator competes with its own workers.** The planner and the judge
  call the dense orchestrator model while workers call the MoE one, and
  `config.py` measures 19 GB + 17 GB against a ~27 GB GPU budget - so
  `OLLAMA_MAX_LOADED_MODELS=1` and every orchestrator call is a ~6.7 s model
  swap, not a queued request. A cap derived from the slot count alone hands
  every slot to the workers and leaves the process that has to judge them
  waiting behind a swap. One slot is reserved.
- **Host RAM is not what a container can have.** On Docker Desktop the Linux VM
  has a fixed allocation - 7.6 GiB against 36 GiB of host on the machine this
  was written for - already carved out before Ollama sees any of it. #19's
  `HostBudget` found that, so the memory bound is asked of it rather than
  recomputed here, and a caller who has paid for `daemon_memory_gb` passes it
  in.

**This module makes no model call, on any path.** Every decision it takes -
oldest first, cap arithmetic, set intersection - is arithmetic over what the
ledger already said, so a cycle that only dispatches costs no model load. That
is the deterministic short-circuit #24 keeps, and it is why
`DispatchReport.needs_judgement` exists: the loop asks the judge on the cycles
that learned nothing, so a run's model-swap count is bounded by the cycles that
needed a judgement rather than by the number of cycles it ran.

Manual dry run against a real repo - reads only, spawns nothing:

    GITHUB_TOKEN=... python -m swarm.orchestrator.dispatcher shahrestani-me/apiary
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Protocol

from ..config import SETTINGS
from ..containers.limits import Budget, HostBudget, LimitError
from ..containers.manager import (
    ContainerError,
    Handle,
    StackImages,
    build_hint,
    missing_image,
)
from ..github.client import GitHubClient, GitHubError
from ..github.ledger import Ledger, LedgerEntry, load_ledger
from ..github.readiness import READY
from ..run import TERMINAL_LABELS
from ..taskref import TaskRef

#: The label this module writes, and the one it writes it over. `READY` is
#: imported rather than respelled because readiness (#11) owns that string; no
#: module owns the other two, and `ledger.LABEL_PRECEDENCE` has the full six.
CLAIMED = "swarm:claimed"

#: The daemon is not there at all, as opposed to this one spawn going wrong.
#:
#: Matching the message is unpleasant, and it is the same trade `is_missing`
#: already makes for the same reason: the CLI gives no other signal, and both
#: cases arrive as `ContainerError`. The set is kept **narrow on purpose**.
#: Everything unrecognised is treated as this issue's problem and the cycle
#: continues, because that is the cheaper way to be wrong: a fleet-wide outage
#: misread as per-issue costs one ready→claimed→ready label round-trip per
#: issue and no attempts, whereas one missing image misread as fleet-wide halts
#: every cycle - which is the bug this ticket exists to fix.
#:
#: `is not on PATH` is here because `DockerCLI._run` raises a bare
#: `ContainerError` for it rather than a `DockerError`: no binary means no
#: daemon for any issue, and it is the failure the orchestrator image actually
#: shipped with.
DAEMON_DOWN_RE = re.compile(
    r"cannot connect to the docker daemon"
    r"|is the docker daemon running"
    r"|error during connect"
    r"|is not on PATH",
    re.I,
)


def daemon_is_down(error: ContainerError) -> bool:
    """Is this failure about the daemon rather than about one issue?

    `DockerError.__str__` folds the argv, the exit code and the captured output
    into one string, so searching the rendered error covers both it and the
    plain `ContainerError` that a missing binary raises.
    """
    return bool(DAEMON_DOWN_RE.search(str(error)))
REVIEW = "swarm:review"

#: Issues whose `## Files` are spoken for. `swarm:claimed` is the obvious half -
#: a container is editing them right now. `swarm:review` is the half worth
#: arguing about: that work is not running any more, but it sits in an open PR
#: against the same base, so a second task editing those files produces the
#: merge conflict this rule exists to prevent. The hold is temporary either way
#: - the PR merges to `swarm:done` or falls back to `swarm:ready` (§4) - so
#: reserving cannot deadlock, it can only defer.
RESERVING_LABELS = frozenset({CLAIMED, REVIEW})

#: The host property the cap is really about. Read from the environment rather
#: than from `Settings` because it belongs to the Ollama server, not to us;
#: `config.py`'s `SWARM_MAX_PARALLEL` is the fallback because its default of 2
#: is that comment's record of what this host's server is set to.
ENV_SLOTS = "OLLAMA_NUM_PARALLEL"

#: Slots the orchestrator itself holds. Not a fudge factor: with
#: `OLLAMA_MAX_LOADED_MODELS=1` a planner or judge call evicts the worker model
#: outright, so the orchestrator is not one more caller queueing politely for a
#: slot - it is the caller that stops every other slot for the duration of a
#: swap. Reserving one is the cheapest honest approximation of that.
ORCHESTRATOR_SLOTS = 1

#: The floor under *derived* concurrency. Slot arithmetic on a one-slot host
#: would otherwise produce a cap of zero, and a run that dispatches nothing
#: forever looks exactly like a run with nothing to do. An explicit
#: `SWARM_MAX_PARALLEL=0` is honoured as written - "dispatch nothing" is a
#: legitimate thing to ask for while debugging - and only the arithmetic is
#: floored.
MIN_WORKERS = 1


# --------------------------------------------------------------------------
# Capacity
# --------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    """Loud on garbage, like `limits.py`'s reader and for the same reason.

    A mistyped `OLLAMA_NUM_PARALLEL` that quietly fell back to the default
    would produce a concurrency nobody chose, which is the one property this
    whole type exists to make deliberate.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LimitError(f"{name}={raw!r} is not an integer") from exc


def container_memory_cap(host: HostBudget, *, ceiling: int) -> int:
    """How many workers fit in what the host had left, at most `ceiling`.

    Asked of #19's `Budget` rather than recomputed: it already subtracts
    Ollama's reservation, honours the Docker daemon's own total where a caller
    supplied one, and clamps a worker's share at `MIN_WORKER_MEMORY_GB`. That
    clamp is what makes this a real bound - past a certain count the shares stop
    shrinking and `Budget.fits` starts saying no.

    `ceiling` is the cap the other constraints already imply, so the search
    never asks about a concurrency that could not be used anyway.
    """
    fitting = [n for n in range(1, max(ceiling, 0) + 1) if Budget.for_host(host, workers=n).fits]
    # No n fits at all: the host is overcommitted before the first worker. One
    # is still dispatched - the limits are enforced per container either way,
    # and `Budget.fits` is where that is reported (#19), not here.
    return max(fitting) if fitting else MIN_WORKERS


@dataclass(frozen=True)
class Capacity:
    """How many workers may run at once, and which constraint decided it.

    Three bounds, kept as separate fields rather than pre-multiplied into one
    number, because the interesting question when a run feels slow is *which*
    of them is binding - and the answer decides whether to change a host
    setting, an environment variable, or the machine.
    """

    slots: int
    orchestrator_slots: int = ORCHESTRATOR_SLOTS
    configured: int = field(default_factory=lambda: SETTINGS.max_workers_parallel)
    #: `None` means the memory bound was not consulted - see `detect`.
    memory: int | None = None

    @classmethod
    def detect(
        cls,
        *,
        host: HostBudget | None = None,
        daemon_gb: float | None = None,
    ) -> Capacity:
        """This machine's cap. `daemon_gb` is the Docker VM's own total.

        Not fetched automatically: `limits.daemon_memory_gb` costs a subprocess
        and a running daemon, and constructing a capacity must work in a unit
        test and before the daemon is up. On Docker Desktop it is nonetheless
        the number that actually binds, so a caller that has a `DockerCLI` in
        hand should pass it.
        """
        inference = cls(slots=_env_int(ENV_SLOTS, SETTINGS.max_workers_parallel))
        machine = host if host is not None else HostBudget.detect(daemon_gb=daemon_gb)
        # The memory search is bounded by the cap the other two already imply:
        # asking whether 8 workers fit is wasted when 2 slots exist.
        return replace(inference, memory=container_memory_cap(machine, ceiling=inference.workers))

    @property
    def inference(self) -> int:
        """Slots left once the orchestrator's is reserved. Floored, never zero."""
        return max(self.slots - self.orchestrator_slots, MIN_WORKERS)

    @property
    def bounds(self) -> dict[str, int]:
        """Every bound in play, by the name a human would act on."""
        bounds = {"inference slots": self.inference, "SWARM_MAX_PARALLEL": self.configured}
        if self.memory is not None:
            bounds["container memory"] = self.memory
        return bounds

    @property
    def workers(self) -> int:
        """The cap. Never negative; zero only if it was configured as zero."""
        return max(min(self.bounds.values()), 0)

    @property
    def bound_by(self) -> str:
        """Which constraint decided the cap, for the line worth logging."""
        return min(self.bounds, key=lambda name: self.bounds[name])

    def summary(self) -> str:
        detail = ", ".join(f"{name} {value}" for name, value in self.bounds.items())
        return f"cap {self.workers} worker(s), bound by {self.bound_by} ({detail})"


# --------------------------------------------------------------------------
# Collaborators
# --------------------------------------------------------------------------


class Labeller(Protocol):
    """The two label calls a claim is made of. `GitHubClient` satisfies it."""

    def add_labels(self, number: int, labels: Iterable[str]) -> Any: ...

    def remove_label(self, number: int, label: str) -> bool: ...


class Spawner(Protocol):
    """One worker container. `ContainerManager` satisfies it.

    A `Protocol` rather than the class, for the reason every other seam in this
    codebase is one: a dispatcher test that needs a Docker daemon to reach the
    ordering rule above is a test that does not run.
    """

    def spawn(
        self,
        task: TaskRef,
        base_commit: str,
        *,
        issue: int | None = None,
        image: str | None = None,
    ) -> Handle: ...

    #: The safe-release probe. A spawn that raised may still have left a
    #: container running - `docker start` can fail this process's read after
    #: the daemon acted - and releasing a claim on that reading buys the issue
    #: a second worker next cycle. So the claim is only given back when the
    #: daemon says there is nothing there. This is the same question #35's
    #: recovery asks before it releases anything, asked one cycle earlier.
    def find(self, *, ref: TaskRef | None = None) -> list[Handle]: ...


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


def normalise(path: str) -> str:
    """One file's identity for the overlap test.

    Case-insensitive because the development host's filesystem is
    (`docs/issue-contract.md` §1.3): `src/Thing.py` and `src/thing.py` are one
    file there, and a case-sensitive intersection would report two tasks as
    disjoint and then hand them the same file.
    """
    return path.strip().casefold()


def held_files(entries: Iterable[LedgerEntry]) -> dict[str, int]:
    """Normalised path -> the number of the issue holding it.

    First holder wins, so a path claimed by two in-flight issues - which should
    not happen, but is what a human relabelling two issues `swarm:claimed`
    produces - reports the older one rather than raising. Nothing is dispatched
    over it either way, which is the decision that matters.
    """
    held: dict[str, int] = {}
    for entry in sorted(entries, key=lambda entry: entry.ref):
        for path in entry.files:
            held.setdefault(normalise(path), entry.number)
    return held


@dataclass(frozen=True)
class Deferred:
    """A ready issue this cycle did not dispatch, and why not.

    Deferral is not failure and is never written to the tracker: the issue stays
    `swarm:ready` and the next cycle reconsiders it. The reason exists to be
    logged, because "the swarm is doing nothing" and "the swarm is at its cap"
    are indistinguishable from the outside otherwise.
    """

    number: int
    reason: str

    def __str__(self) -> str:
        return f"#{self.number}: {self.reason}"


@dataclass(frozen=True)
class DispatchPlan:
    """What one cycle would dispatch, what it held back, and what is already up.

    Computed without touching GitHub or Docker, so every interesting case - more
    ready issues than the cap, two tasks over one file, a cap already filled by
    a previous cycle - is testable as data.
    """

    dispatch: tuple[LedgerEntry, ...] = ()
    deferred: tuple[Deferred, ...] = ()
    in_flight: tuple[LedgerEntry, ...] = ()
    capacity: Capacity | None = None

    @property
    def cap(self) -> int:
        return self.capacity.workers if self.capacity is not None else len(self.dispatch)

    @property
    def numbers(self) -> tuple[int, ...]:
        """The issues this cycle would dispatch, as numbers.

        Numbers rather than `TaskRef`s, unlike `ReconcilePlan.refs` and
        `RecoveryPlan.refs`: what `ready` *filters* on is identity and was
        retyped (#142), but this plan's own reporting rows - here, `Deferred`,
        `DispatchFailure` - were left in the older vocabulary along with
        `checks` and `mergeability`, so the retype stayed the ticket's size.
        Nothing keys a `TaskRef` map on these.
        """
        return tuple(entry.number for entry in self.dispatch)

    def summary(self) -> str:
        return (
            f"{len(self.dispatch)} to dispatch, {len(self.in_flight)} in flight, "
            f"{len(self.deferred)} deferred, cap {self.cap}"
        )


def plan_dispatch(
    ledger: Ledger,
    *,
    capacity: Capacity | None = None,
    ready: Iterable[TaskRef] | None = None,
) -> DispatchPlan:
    """Choose this cycle's issues. Pure - no API call, no daemon, no model.

    Oldest first, by issue number: GitHub numbers ascend with creation, so it is
    the arrival order without a second field to trust, and it is stable, which
    matters more than it sounds. A cap that picked differently on identical
    input would make "why did that one not run" unanswerable.

    `ready` is the readiness pass's own verdict (`ReadinessPlan.ready`), passed
    when the two run in one cycle so the labels it just wrote need not be read
    back. Without it the labels on the ledger decide, which is the resumption
    case: a fresh process reads `swarm:ready` from GitHub, exactly as
    architecture-v2 intends.
    """
    limit = capacity if capacity is not None else Capacity.detect()
    allowed = None if ready is None else set(ready)

    in_flight: list[LedgerEntry] = []
    candidates: list[LedgerEntry] = []
    for entry in sorted(ledger.entries.values(), key=lambda entry: entry.ref):
        if entry.state_label in RESERVING_LABELS:
            in_flight.append(entry)
            continue
        if entry.state_label in TERMINAL_LABELS:
            continue
        # A caller's list is a filter, never a promotion: an issue readiness
        # called ready but whose label says otherwise is still not dispatched,
        # because §3 makes the label set the authority.
        if allowed is None:
            if entry.state_label != READY:
                continue
        elif entry.ref not in allowed or entry.state_label != READY:
            continue
        candidates.append(entry)

    held = held_files(in_flight)
    # In-flight containers spend the same cap. A previous cycle's workers are
    # queued against the same inference slots as this cycle's would be, and a
    # cap counted per cycle rather than per run is not a cap at all.
    free = max(limit.workers - len(in_flight), 0)

    dispatch: list[LedgerEntry] = []
    deferred: list[Deferred] = []
    for entry in candidates:
        if len(dispatch) >= free:
            deferred.append(Deferred(entry.number, f"at the cap of {limit.workers}"))
            continue
        clash = next((path for path in entry.files if normalise(path) in held), None)
        if clash is not None:
            deferred.append(
                Deferred(entry.number, f"{clash} is held by #{held[normalise(clash)]}")
            )
            continue
        dispatch.append(entry)
        # Registered before the next candidate is considered, so two *ready*
        # issues over one file serialize against each other and not only
        # against work that was already running.
        for path in entry.files:
            held.setdefault(normalise(path), entry.number)

    return DispatchPlan(
        dispatch=tuple(dispatch),
        deferred=tuple(deferred),
        in_flight=tuple(in_flight),
        capacity=limit,
    )


# --------------------------------------------------------------------------
# Dispatching
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Dispatched:
    """One issue that is now a running container."""

    entry: LedgerEntry
    handle: Handle

    @property
    def number(self) -> int:
        return self.entry.number

    def __str__(self) -> str:
        return f"#{self.number} {self.entry.task_id} -> {self.handle}"


@dataclass(frozen=True)
class DispatchFailure:
    """An issue that could not be dispatched, and what state it was left in.

    `claimed` is the field that matters to #35. True means the label was written
    and no container is running under it, which is precisely the stale claim
    that recovery sweeps; false means nothing was written and the issue is
    untouched.
    """

    number: int
    reason: str
    claimed: bool = False
    #: Did this failure end the cycle? True for a daemon that is not there at
    #: all, false for one issue's spawn going wrong. The distinction is the
    #: whole point of #94 and it has to survive into the log, because the two
    #: read identically at a glance and want opposite responses: restart Docker,
    #: versus look at what is wrong with that one issue.
    fatal: bool = False

    def __str__(self) -> str:
        state = "claimed, no container" if self.claimed else "not claimed"
        scope = "daemon-level, cycle stopped" if self.fatal else "this issue only"
        return f"#{self.number}: {self.reason} ({state}; {scope})"


@dataclass(frozen=True)
class DispatchReport:
    """What one cycle actually did. The plan, plus what happened to it."""

    plan: DispatchPlan
    dispatched: tuple[Dispatched, ...] = ()
    failed: tuple[DispatchFailure, ...] = ()

    @property
    def handles(self) -> tuple[Handle, ...]:
        return tuple(item.handle for item in self.dispatched)

    @property
    def needs_judgement(self) -> bool:
        """Is this cycle worth paying a model swap to judge?

        No, when something was dispatched or something is still running: the
        cycle has an answer already and asking the judge would buy a ~6.7 s
        model swap and a re-load of the worker model afterwards, per cycle, for
        news the arithmetic already had. Yes, when a cycle changed nothing and
        nothing is in flight - that is a stall, which is the thing the judge
        exists to diagnose (`docs/architecture-v2.md`, step 5).

        A failed dispatch is not a stall. Docker or GitHub being unreachable is
        an infrastructure fact, and no model has an opinion about it.
        """
        return not self.dispatched and not self.plan.in_flight and not self.failed

    def summary(self) -> str:
        head = f"dispatched {len(self.dispatched)}; {self.plan.summary()}"
        return f"{head}; {len(self.failed)} failed" if self.failed else head


def claim(client: Labeller, entry: LedgerEntry) -> None:
    """Move one issue `ready -> claimed`, adding the new label first.

    The order is `readiness._relabel`'s and is load-bearing for the same reason:
    GitHub has no transaction across two label calls, and a crash between them
    leaves either two state labels or none. Two is repairable - §3's precedence
    puts `claimed` above `ready`, so the conservative reading wins. None puts
    the issue outside the ledger, where nothing looks at it again.
    """
    client.add_labels(entry.number, [CLAIMED])
    client.remove_label(entry.number, READY)


def release(client: Labeller, manager: Spawner, entry: LedgerEntry) -> bool:
    """Undo one claim, `claimed -> ready`, and say whether it was undone.

    **Only when the daemon says no container *exists* under that issue.**
    Exists, not "is running": `find` here is `docker ps --all`, so a container
    that has already exited still blocks the release, and that width is
    deliberate rather than left over. This runs after a failed spawn, and a
    `docker start` that failed *this process's* read may still have started a
    container - one that ran, pushed and exited in the same breath is
    indistinguishable from one that never started at all to a probe that asks
    only about liveness. An issue put back to `swarm:ready` on that reading
    gets a second container next cycle - two workers, one file set, one of the
    two pushes lost. "Any container blocks the release" is the only reading
    that cannot produce that, which is why #187 narrowed the callers that
    genuinely mean liveness and deliberately left this one alone. That risk is
    what kept the claim in the first place; `find` is what removes it, by
    asking the question #35's recovery already asks instead of assuming the
    answer.

    Label order is `claim`'s, inverted, and load-bearing for the same reason:
    §3's precedence ranks `claimed` above `ready`, so a crash between the two
    calls leaves the conservative reading in place and recovery sweeps it.

    Every failure here answers False, and False is safe: the caller records the
    issue as still claimed, which is exactly what #35 was built to sweep.
    """
    try:
        if manager.find(ref=entry.ref):
            return False
    except ContainerError:
        # The probe itself could not answer. "Do not release" is the reading
        # that cannot produce two workers.
        return False
    try:
        client.add_labels(entry.number, [READY])
        client.remove_label(entry.number, CLAIMED)
    except GitHubError:
        return False
    return True


def dispatch(
    client: Labeller,
    manager: Spawner,
    ledger: Ledger,
    base_commit: str,
    *,
    capacity: Capacity | None = None,
    ready: Iterable[TaskRef] | None = None,
    dry_run: bool = False,
    images: StackImages | None = None,
) -> DispatchReport:
    """Claim and spawn this cycle's issues, one at a time, in that order.

    One at a time is not an accident of the loop shape. The claim and the spawn
    are interleaved per issue so that an outage strands one issue rather than
    the whole cycle: everything not yet reached is still `swarm:ready` and the
    next cycle picks it up untouched.

    **A failed spawn defers its own issue; the cycle continues.** This used to
    `break`, and the argument for it was that a spawn failure means the daemon
    is down, so the second spawn would fail exactly like the first. That held
    while there was one image. It stops holding the moment the image is chosen
    per task (#99): a single missing image would halt every cycle and burn
    attempts across the whole ledger, on issues with nothing wrong with them.

    So the two cases are now told apart. `daemon_is_down` recognises a daemon
    that is not there, and only that still stops the cycle - it is the case the
    `break` was right about, and retrying eighteen issues against a dead socket
    helps nobody. Everything else defers one issue and moves on. The recognised
    set is deliberately narrow and the unrecognised default is "keep going",
    because being wrong that way costs label churn and being wrong the other way
    is the bug being fixed.

    **A deferred issue gives its claim back, when that is provably safe.**
    `release` asks the daemon whether a container exists for that issue before
    it writes anything, so the "the start failed my read but succeeded" case
    keeps its claim and goes to #35's recovery, as before. What changes is that
    the ordinary case - `docker create` refused, nothing exists - no longer
    leaves a claim for recovery to sweep and an attempt for it to burn.

    A **claim** that fails still stops the cycle. It is a GitHub write, and the
    things that break GitHub writes for one issue - the token, the rate limit,
    the network - break them for all of them. That one is unchanged and out of
    this ticket's scope.

    `dry_run=True` returns the plan and writes nothing at all.
    """
    plan = plan_dispatch(ledger, capacity=capacity, ready=ready)
    if dry_run:
        return DispatchReport(plan=plan)

    images = images if images is not None else StackImages()
    dispatched: list[Dispatched] = []
    failed: list[DispatchFailure] = []
    for entry in plan.dispatch:
        # Resolved *before* the claim, on purpose. A task whose stack this host
        # has no image for cannot be run whatever happens next, and a claim
        # spent on it is a claim #35 has to sweep - so the refusal costs
        # nothing and names something to do about it, where a `docker create`
        # failure names a tag.
        try:
            image = images.for_stack(entry.stack)
        except ContainerError as exc:
            failed.append(DispatchFailure(entry.number, f"no image: {exc}"))
            continue
        try:
            claim(client, entry)
        except GitHubError as exc:
            # Nothing was spawned and the label may or may not have landed; the
            # issue is either still ready or carries two labels, and §3's
            # precedence repairs the second case. Fatal because whatever breaks
            # one issue's label write breaks the next one's too.
            failed.append(DispatchFailure(entry.number, f"claim failed: {exc}", fatal=True))
            break
        try:
            # Both, and they are not the same fact: the container is labelled
            # and named by task, and the worker is handed the issue number it
            # needs to read a contract and open a pull request against it.
            handle = manager.spawn(
                entry.ref, base_commit, issue=entry.number, image=image
            )
        except ContainerError as exc:
            fatal = daemon_is_down(exc)
            if missing_image(exc):
                # The one failure whose fix is a command rather than an
                # investigation. The orchestrator cannot build or pull, so a
                # human has to, and the message should say which line to run.
                exc = ContainerError(f"{exc}; {build_hint(image)}")
            # Only worth asking when the cycle is going to continue: if the
            # daemon is unreachable then `find` cannot answer either, and the
            # claim stays for #35 exactly as it did before.
            claimed = True if fatal else not release(client, manager, entry)
            failed.append(
                DispatchFailure(
                    entry.number, f"spawn failed: {exc}", claimed=claimed, fatal=fatal
                )
            )
            if fatal:
                break
            continue
        dispatched.append(Dispatched(entry=entry, handle=handle))

    return DispatchReport(plan=plan, dispatched=tuple(dispatched), failed=tuple(failed))


if __name__ == "__main__":  # pragma: no cover - manual dry run, see module docstring
    import sys

    repo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_REPOSITORY", "")
    # Read-only on every path: no adoption write, no label, no container.
    report = plan_dispatch(load_ledger(GitHubClient.from_env(repo), adopt=False))
    print(Capacity.detect().summary())
    for chosen in report.dispatch:
        print(f"dispatch #{chosen.number} {chosen.task_id}: {', '.join(chosen.files)}")
    for held_back in report.deferred:
        print(f"defer    {held_back}")
    print(report.summary())
