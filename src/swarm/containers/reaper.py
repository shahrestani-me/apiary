"""Sweep containers whose orchestrator is gone, and take their logs with them.

`docs/architecture-v2.md`: "An orphan reaper sweeps containers matching
`apiary.run=<id>` at startup and shutdown. Crashed orchestrators otherwise leak
containers that hold clones and disk." This module is that sweep, and it is the
one place in the system that removes a container it did not spawn.

**Every decision here is made from one label and nothing else.** A container is
reaped because `apiary.run` names a run that is not live - never because its
image is `apiary-worker`, never because its name starts with `apiary-`. Image
and name are conventions the human can also adopt by accident; the run label is
minted by `swarm.run` (#33) and means exactly one orchestrator process. Reaping
by pattern is how a sweep eats the developer's own build container, and there
is no undo for `docker rm`.

**Liveness: a run id that is not this process's is dead.** Ids are never reused
([`run.py`](../run.py)), so the containers a killed orchestrator left behind
carry an id no live process answers to, and that is the whole signal. It is a
real assumption and it is worth stating plainly: **concurrent orchestrators on
one host are unsupported by default**, exactly as
[#35](https://github.com/shahrestani-me/apiary/issues/35) permits for the
label half of the same problem. The alternative - inventing a lease, a pid file
or a heartbeat - puts durable state back on the local disk, which is the thing
`docs/architecture-v2.md` spent the whole control plane removing.

Two escape hatches make the assumption survivable rather than merely stated:

- `live_run_ids` spares runs the caller knows are alive. A sibling orchestrator,
  or a scheduler driving several, names its runs here and keeps them.
- `scope` bounds the sweep to named runs, so a reaper can be pointed at one dead
  run instead of at every apiary container on the machine. `swarm reap <run-id>`
  is that, and so is every test in this repository that touches a real daemon.

A container whose `apiary.run` value is not a well-formed run id is spared too.
This system validates ids at mint, so an unparseable one cannot have come from
here, and something else wearing our label is somebody else's container.

**Capture precedes removal, and it matters most here.** `dispose_container`
returns what it salvaged, so the ordering is structural rather than remembered.
An orphan is a container the orchestrator never saw finish - nobody read its
exit code, nobody read its output, and the reason the run went wrong is in
there. Those logs are the most valuable in the run and they are gone the
instant the container is. `sink` is where #29 receives them.

**Ctrl-C is a shutdown.** `guard()` sweeps on entry, installs `SIGINT` and
`SIGTERM` handlers, sweeps on exit, and chains to whatever handler it displaced
so the process still dies when the human asked it to. Without the handlers the
common case - a human stopping a run - is precisely the case that leaks, since
`KeyboardInterrupt` inside a `docker wait` unwinds past every `finally` that had
not started yet.
"""

from __future__ import annotations

import contextlib
import signal
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Callable, Collection, Iterator, Sequence

from ..run import Run, RunError, validate_run_id
from .manager import (
    ContainerError,
    DockerCLI,
    Handle,
    Redactor,
    dispose_container,
    find_containers,
)

#: What `guard` installs handlers for. `SIGINT` is Ctrl-C, `SIGTERM` is
#: `docker stop` on the orchestrator's own container and what a supervisor
#: sends first. `SIGKILL` is deliberately absent: it cannot be caught, and the
#: startup sweep of the *next* run is the only thing that ever cleans up after
#: it - which is why that sweep exists rather than trusting shutdown alone.
REAPED_SIGNALS: tuple[int, ...] = (signal.SIGINT, signal.SIGTERM)


# --------------------------------------------------------------------------
# What a sweep did
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sweep:
    """The outcome of one pass, in enough detail to print and to assert on.

    `spared` is carried rather than discarded because "the reaper ran and
    removed nothing" and "the reaper ran and decided six containers were
    somebody else's" are different facts, and only the second one is reassuring
    when a human is wondering why a container is still up.
    """

    reaped: tuple[Handle, ...] = ()
    spared: tuple[Handle, ...] = ()
    failures: tuple[tuple[Handle, Exception], ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def run_ids(self) -> tuple[str, ...]:
        """The dead runs this sweep emptied, in order."""
        seen: list[str] = []
        for handle in self.reaped:
            if handle.run_id not in seen:
                seen.append(handle.run_id)
        return tuple(seen)

    def summary(self) -> str:
        parts = [f"reaped {len(self.reaped)} container(s) from {len(self.run_ids)} dead run(s)"]
        if self.spared:
            parts.append(f"spared {len(self.spared)}")
        if self.failures:
            detail = "; ".join(f"{handle}: {exc}" for handle, exc in self.failures)
            parts.append(f"{len(self.failures)} problem(s): {detail}")
        return ", ".join(parts)


# --------------------------------------------------------------------------
# The reaper
# --------------------------------------------------------------------------


@dataclass
class Reaper:
    """Removes the containers of runs that are no longer live.

    Holds no `Run` of its own by necessity: the containers it exists for belong
    to processes that are gone, so the module functions in `manager` - not
    `ContainerManager` - are what it reaches the daemon through.
    """

    #: The daemon. Given one rather than built one when the caller already has
    #: a run's `DockerCLI`, so an orphan's logs pass through the same redactor
    #: as a live worker's.
    docker: DockerCLI | None = None

    #: This process's run, if there is one. Its containers are live and are
    #: spared by every sweep except the shutdown one, where by definition they
    #: are not live any more.
    run: Run | None = None

    #: Runs the caller knows are alive. The named escape from "one orchestrator
    #: per host": a scheduler that drives several lists them here and keeps
    #: every one of their workers.
    live_run_ids: Collection[str] = ()

    #: The runs this reaper may touch at all. `None` means every apiary run on
    #: the daemon. A set narrows the `docker ps` filter itself, so a sweep
    #: cannot even see a container outside it.
    scope: Collection[str] | None = None

    #: Where the salvaged logs go - #29's artifact writer. Called once per
    #: reaped container, after removal, with the handle that holds `captured`.
    sink: Callable[[Handle], None] | None = None

    _previous: dict[int, Any] = field(default_factory=dict, repr=False)
    _reaping: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.docker is None:
            # A bare `DockerCLI` redacts nothing. The literals a dead run held
            # are unrecoverable, but `SECRET_PATTERNS` still catches a token by
            # shape, and an orphan's logs are the ones most likely to be pasted
            # into an issue.
            self.docker = DockerCLI(redact=Redactor())

    # --- selection ------------------------------------------------------

    def orphans(self, *, include_own: bool = False) -> list[Handle]:
        """The containers a sweep would remove. The dry run, and the whole rule.

        Public because the decision to destroy something should be inspectable
        without performing it: `swarm reap --dry-run` is this, and so is every
        assertion about the policy that does not want a daemon.
        """
        return self._partition(include_own=include_own)[0]

    def _partition(self, *, include_own: bool) -> tuple[list[Handle], list[Handle]]:
        live = self._live_ids(include_own=include_own)
        doomed: list[Handle] = []
        spared: list[Handle] = []
        for handle in self._candidates():
            (spared if self._is_spared(handle, live) else doomed).append(handle)
        return doomed, spared

    def _live_ids(self, *, include_own: bool) -> frozenset[str]:
        """Run ids this sweep will not touch.

        Caller-declared ids are taken verbatim and not validated: sparing is the
        safe direction, and refusing to spare something because its id looked
        wrong is the one failure mode worth designing out.
        """
        live = set(self.live_run_ids)
        if self.run is not None and not include_own:
            live.add(self.run.id)
        return frozenset(live)

    def _candidates(self) -> list[Handle]:
        """Every container in scope, always filtered on `apiary.run`.

        With a `scope` the filter carries the run id, so the daemon never even
        reports a container from another run - which is what makes a sweep on a
        shared development machine safe to run at all.
        """
        if self.scope is None:
            return find_containers(self._cli)
        handles: list[Handle] = []
        for run_id in sorted(set(self.scope)):
            handles += find_containers(self._cli, run_id=run_id)
        return handles

    def _is_spared(self, handle: Handle, live: frozenset[str]) -> bool:
        if handle.run_id in live:
            return True
        # Wearing the label without a well-formed value means it was not minted
        # here, and a container this system did not create is not its to remove.
        return not _is_run_id(handle.run_id)

    # --- sweeping -------------------------------------------------------

    def sweep(self, *, include_own: bool = False) -> Sweep:
        """Capture and remove every orphan in scope. Never raises for one container.

        One unremovable container must not leave the other five running: the
        failure is recorded against its handle and the sweep carries on. The
        caller gets `Sweep.ok` and a summary naming what went wrong, which is
        strictly more than an exception thrown from the middle of the loop
        would have left it with.
        """
        doomed, spared = self._partition(include_own=include_own)
        reaped: list[Handle] = []
        failures: list[tuple[Handle, Exception]] = []

        for handle in doomed:
            try:
                dispose_container(handle, self._cli)
            except ContainerError as exc:
                failures.append((handle, exc))
                continue
            reaped.append(handle)
            # After removal, deliberately. The logs are already on the handle by
            # then, so a writer that fails costs an artifact rather than leaking
            # the container the artifact was about.
            if self.sink is not None:
                try:
                    self.sink(handle)
                except Exception as exc:  # noqa: BLE001 - a sink is somebody else's code
                    failures.append((handle, exc))

        return Sweep(tuple(reaped), tuple(spared), tuple(failures))

    def startup(self) -> Sweep:
        """Sweep what a previous, dead orchestrator left behind.

        This is the sweep that satisfies the ticket. A `SIGKILL`ed process ran
        no shutdown, and its run id is never minted again, so the next startup
        is the only thing left that can find its containers.
        """
        return self.sweep()

    def shutdown(self) -> Sweep:
        """Sweep this run's containers too - it is ending, so they are orphans."""
        return self.sweep(include_own=True)

    # --- signals --------------------------------------------------------

    @contextlib.contextmanager
    def guard(self, *, signals: Sequence[int] = REAPED_SIGNALS) -> Iterator[Reaper]:
        """Sweep on entry, on exit, and on the signal that skips the exit.

            with Reaper(run=run).guard() as reaper:
                orchestrate(...)

        The `finally` sweep and the handler's sweep can both happen for one
        Ctrl-C - the handler reaps, chains, and the resulting `KeyboardInterrupt`
        unwinds through here. That is not a bug to fix but the reason disposal
        was made idempotent: the second sweep finds nothing and costs one
        `docker ps`.
        """
        self.startup()
        self.install(signals=signals)
        try:
            yield self
        finally:
            self.restore()
            self.shutdown()

    def install(self, *, signals: Sequence[int] = REAPED_SIGNALS) -> None:
        """Take over `SIGINT`/`SIGTERM`, remembering what was there before.

        Only from the main thread - `signal.signal` says so - and that is where
        the orchestrator's loop runs.
        """
        for signum in signals:
            if signum in self._previous:
                continue
            # `getsignal` answers `None` for a handler installed from C, which
            # `signal.signal` then refuses to take back. The default
            # disposition is the honest approximation and the only one that
            # restores.
            self._previous[signum] = signal.getsignal(signum) or signal.SIG_DFL
            signal.signal(signum, self._on_signal)

    def restore(self) -> None:
        """Put back every handler `install` displaced. Idempotent."""
        while self._previous:
            signum, handler = self._previous.popitem()
            signal.signal(signum, handler)

    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
        """Reap, then let the signal do what it was going to do anyway.

        Chaining is not politeness. Swallowing `SIGINT` here would mean a
        `KeyboardInterrupt` that never arrives and a Ctrl-C that appears to do
        nothing, and swallowing `SIGTERM` would make the orchestrator container
        ignore `docker stop` until the daemon escalated to `SIGKILL` - which is
        the one signal this module cannot clean up after.
        """
        previous = self._previous.get(signum, signal.SIG_DFL)

        if self._reaping:
            # A second Ctrl-C during the first sweep means the human is done
            # waiting. Stop reaping and take the original disposition now; some
            # containers may survive, and that was their choice.
            self.restore()
            _chain(previous, signum, frame)
            return

        self._reaping = True
        try:
            self.shutdown()
        finally:
            self._reaping = False
            self.restore()
        _chain(previous, signum, frame)

    # --- plumbing -------------------------------------------------------

    @property
    def _cli(self) -> DockerCLI:
        assert self.docker is not None  # set in __post_init__
        return self.docker


def _chain(previous: Any, signum: int, frame: FrameType | None) -> None:
    """Hand the signal to the handler we displaced, or to the default one."""
    if callable(previous):
        previous(signum, frame)
        return
    if previous == signal.SIG_IGN:
        return
    # `SIG_DFL`, and the handlers are already restored, so this is the plain
    # default disposition: terminate with the status a shell reports as
    # 128+signum, rather than exiting 0 and claiming the run finished.
    signal.raise_signal(signum)


def _is_run_id(value: str) -> bool:
    try:
        validate_run_id(value)
    except RunError:
        return False
    return True


__all__ = ["REAPED_SIGNALS", "Reaper", "Sweep"]
