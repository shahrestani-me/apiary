"""How much of this machine a worker may hold, and for how long.

`manager.py` already owns the mechanism: every container is created with
explicit `--cpus`, `--memory` and `--pids-limit`, and `wait` raises
`ContainerTimeout` when the wall clock beats the worker. This module owns the
**policy** - which numbers those flags get, where the numbers come from, and
what the orchestrator is supposed to believe about a task whose container it
had to kill. There is deliberately no second spelling of the mechanism here;
`WorkerLimits` subclasses `Limits` and `wait_within` calls `wait`.

**The budget starts with Ollama, not with the worker.** The host runs 17-19 GB
of weights (`config.py`), and inference on the host is the thing every worker
is waiting on - see architecture-v2's first constraint. Memory handed to a
container is memory the model cannot have, so a limit chosen by asking "what
does a test suite need?" makes the swarm compete with the reason it exists.
The arithmetic here therefore runs the other way: subtract Ollama's resident
set and the system's own needs from host RAM first, and divide *what is left*
by the concurrency cap. On the 36 GB machine this was written for, that yields
exactly the 2 CPUs / 4 GiB / 512 pids that `DEFAULT_LIMITS` states as its
conservative default - which is the point. The default was never arbitrary,
and now it is derived.

**Docker Desktop's VM is a second, tighter ceiling.** Host RAM is not what a
container can allocate on macOS: the Linux VM has its own fixed allocation
(7.6 GiB on this machine, against 36 GiB of host), already carved out of the
host by the time Ollama sees it. Whichever of the two budgets is smaller wins,
so `HostBudget.daemon_gb` is honoured when a caller has asked the daemon. It
is not asked for automatically: importing this module must not require a
running Docker.

**Clamped at both ends, and the clamps matter more than the arithmetic.** The
floor keeps the numbers legal on a machine that has no headroom at all - the
honest answer there is "this host cannot run the swarm", and `Budget.fits`
says so, but a container created with `--memory 0` is a container with no
limit, which is the failure this ticket exists to prevent. The ceiling stops a
128 GB workstation from handing 40 GiB to code no human has read.

**A killed worker reports nothing, so somebody has to report for it.** #18
gives the worker three exit codes and a result file it writes on its way out.
A worker stopped by the wall clock writes neither: it is SIGKILLed mid-edit and
the artifacts directory keeps whatever it had flushed. `wait_within` is that
gap closed - it turns `ContainerTimeout` into an `Outcome` carrying
`TIMEOUT_EXIT_CODE`, so the dispatcher gets an exit code on every path and #18's
writer has a record to persist. `wait` stopped the container rather than
removing it, so the logs are still there to capture and the reason string is
still true when a human reads it.

The knobs below are read from the environment at call time rather than from
`Settings`, which is where they belong: `config.py` is not this ticket's file.
Anything already in `Settings` - the concurrency cap, the wall-clock budget -
is read from there rather than re-invented.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

from ..config import SETTINGS
from .manager import ContainerTimeout, DockerCLI, Handle, Limits

GIB = float(1 << 30)

#: What the host keeps for Ollama. `config.py` measures the dense orchestrator
#: model at 19 GB and the MoE worker model at 17 GB, with
#: `OLLAMA_MAX_LOADED_MODELS=1` so only one is resident - plus a KV cache sized
#: by `num_ctx` x `OLLAMA_NUM_PARALLEL`, and the swap that happens when the two
#: roles alternate. 22 covers the larger model and its cache; it is a
#: reservation, not a measurement, and lowering it is how a worker gets bigger.
OLLAMA_RESERVE_GB = 22.0

#: macOS, the orchestrator process, the Docker daemon and whatever the human is
#: editing while the swarm runs. A host with no slack for this does not become
#: safe by being told it has slack.
SYSTEM_RESERVE_GB = 6.0

#: Held back from what the Docker daemon reports as its own total, for the VM's
#: kernel and the daemon itself. Containers cannot have all of it.
DAEMON_RESERVE_GB = 1.0

#: Assumed host RAM when the platform will not say. Deliberately small: the
#: clamp then lands on the floor, and the floor is the safe direction to be
#: wrong in.
FALLBACK_TOTAL_GB = 16.0

#: A worker clones a repo and runs a test suite. The floor is what makes the
#: flag legal on a host with no headroom; the ceiling is the real policy - past
#: a few GiB a worker is not doing more work, it is leaking.
MIN_WORKER_MEMORY_GB = 1.0
MAX_WORKER_MEMORY_GB = 4.0

#: Cores kept off the workers, for the orchestrator and for Ollama's own
#: prompt processing - which is CPU work even when generation is on the GPU.
RESERVED_CPUS = 2.0

#: Containers do not buy parallelism (architecture-v2, second constraint): a
#: worker's wall clock is dominated by inference it is queued for on the host,
#: not by its own CPU. So the ceiling is low on purpose. Spending cores on a
#: worker buys queueing, and takes them from the thing being queued for.
MIN_WORKER_CPUS = 1.0
MAX_WORKER_CPUS = 2.0

#: A fork bomb's ceiling. High enough that a normal build - a test runner
#: spawning compilers, `git` spawning `ssh` - never notices, low enough that
#: the container hits it long before the VM's own process table does.
WORKER_PIDS = 512

#: What a timed-out attempt is recorded as. `docs/issue-contract.md` §4: exit 1
#: is "task failed, attempt consumed", exit 2 is "infrastructure, attempt not
#: consumed". A worker that ran for its whole budget and produced nothing is
#: the first: the model looping is exactly the failure the attempt cap exists
#: to bound, and recording it as infrastructure would let one hanging task
#: retry until the run's round budget ran out instead.
TIMEOUT_EXIT_CODE = 1

#: Overrides. Named like `config.py`'s, because they will move there.
ENV_CPUS = "SWARM_WORKER_CPUS"
ENV_MEMORY = "SWARM_WORKER_MEMORY"
ENV_PIDS = "SWARM_WORKER_PIDS"
ENV_OLLAMA_RESERVE = "SWARM_OLLAMA_RESERVE_GB"
ENV_SYSTEM_RESERVE = "SWARM_SYSTEM_RESERVE_GB"

#: Docker's size grammar: a number and an optional unit, `b`/`k`/`m`/`g`, with
#: an optional trailing `b` (`4g` and `4gb` are the same size). A bare number
#: is bytes.
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([bkmg])?b?\s*$", re.I)
_UNIT_GIB = {"b": 1.0 / GIB, "k": 1.0 / (1 << 20), "m": 1.0 / (1 << 10), "g": 1.0}


class LimitError(ValueError):
    """A configured limit could not be understood, so no limit was applied."""


# --------------------------------------------------------------------------
# Plumbing
#
# Above everything else rather than below it, because `DEFAULT_BUDGET` is
# derived at import and would otherwise reach for a helper this file has not
# defined yet.
# --------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise LimitError(f"{name}={raw!r} is not a number") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LimitError(f"{name}={raw!r} is not an integer") from exc


# --------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------


def parse_size(text: str) -> float:
    """Docker's size grammar to GiB. Raises `LimitError` on anything else.

    Loud rather than lenient: a mistyped `SWARM_WORKER_MEMORY` that parsed to a
    default would produce a container running with a limit nobody chose, and
    the whole value of this module is that the number is deliberate.
    """
    match = _SIZE_RE.match(text or "")
    if not match:
        raise LimitError(f"{text!r} is not a docker size like '512m' or '4g'")
    amount, unit = match.groups()
    return float(amount) * _UNIT_GIB[(unit or "b").lower()]


def format_size(gib: float) -> str:
    """GiB to the flag value, always in whole MiB.

    MiB rather than the prettier `4g`, because the arithmetic above produces
    thirds of a gigabyte the moment the concurrency cap is not a power of two,
    and rounding those up to `4g` would hand out memory the budget does not
    have.
    """
    return f"{max(int(round(gib * 1024)), 1)}m"


# --------------------------------------------------------------------------
# The host
# --------------------------------------------------------------------------


def host_memory_gb() -> float:
    """Physical RAM, or `FALLBACK_TOTAL_GB` where the platform will not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB
    except (ValueError, OSError, AttributeError):
        return FALLBACK_TOTAL_GB


def daemon_memory_gb(docker: DockerCLI | None = None) -> float:
    """What the Docker daemon says it has, in GiB. Needs a reachable daemon.

    Separate from `HostBudget.detect` and never called by it: on Docker Desktop
    this is the number that actually binds, but asking for it costs a
    subprocess and a running daemon, and constructing a budget must work in a
    unit test and on a machine where Docker is not up yet.
    """
    cli = docker or DockerCLI()
    return float(cli("info", "--format", "{{.MemTotal}}").strip()) / GIB


@dataclass(frozen=True)
class HostBudget:
    """What this machine can lend, after the things it cannot lend.

    `daemon_gb` is the Docker daemon's own total when a caller has bothered to
    ask (`daemon_memory_gb`). On Linux it is the same RAM counted twice and
    changes nothing; on Docker Desktop it is a VM allocation far below host RAM
    and is the ceiling that actually applies.
    """

    total_gb: float
    cpus: int
    ollama_gb: float = OLLAMA_RESERVE_GB
    system_gb: float = SYSTEM_RESERVE_GB
    daemon_gb: float | None = None

    @classmethod
    def detect(cls, *, daemon_gb: float | None = None) -> HostBudget:
        """This machine, with the reserves the environment asks for."""
        return cls(
            total_gb=host_memory_gb(),
            cpus=os.cpu_count() or 1,
            ollama_gb=_env_float(ENV_OLLAMA_RESERVE, OLLAMA_RESERVE_GB),
            system_gb=_env_float(ENV_SYSTEM_RESERVE, SYSTEM_RESERVE_GB),
            daemon_gb=daemon_gb,
        )

    @property
    def workers_gb(self) -> float:
        """Memory all workers together may hold. Never negative.

        Zero is a meaningful answer and is not rounded up to something
        comfortable: it means every byte of this host is already spoken for,
        and `Budget.fits` is where that gets reported rather than hidden.
        """
        left = self.total_gb - self.ollama_gb - self.system_gb
        if self.daemon_gb is not None:
            left = min(left, self.daemon_gb - DAEMON_RESERVE_GB)
        return max(left, 0.0)

    @property
    def workers_cpus(self) -> float:
        """Cores all workers together may use. Never negative."""
        return max(float(self.cpus) - RESERVED_CPUS, 0.0)

    def summary(self) -> str:
        seen = f", docker {self.daemon_gb:.1f}" if self.daemon_gb is not None else ""
        return (
            f"{self.workers_gb:.1f} GiB and {self.workers_cpus:g} cores for workers "
            f"(host {self.total_gb:.1f} GiB{seen}, "
            f"reserved {self.ollama_gb:g} for Ollama + {self.system_gb:g} for the system)"
        )


# --------------------------------------------------------------------------
# The limits
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerLimits(Limits):
    """`Limits`, plus the one flag the policy adds: no swap.

    Docker's default `--memory-swap` is twice `--memory`, so a container that
    exhausts its limit does not die - it swaps, at the VM's disk, for as long
    as it takes to fill twice the allowance. That is the "the Mac becomes
    unusable" failure with an extra step. Pinning swap to the memory limit
    makes the ceiling the ceiling: the allocation fails, the kernel OOM-kills
    the process, and one task fails.

    Subclassed rather than folded into `Limits` because `manager.py` is not
    this ticket's file. `flags()` stays the only place a flag is spelled.
    """

    def flags(self) -> list[str]:
        return [*super().flags(), "--memory-swap", self.memory]


def _worker_limits(host: HostBudget, workers: int) -> WorkerLimits:
    """Divide the budget, clamp it, then let the environment overrule it."""
    share_gb = host.workers_gb / max(workers, 1)
    share_cpus = host.workers_cpus / max(workers, 1)
    return WorkerLimits(
        cpus=_env_float(ENV_CPUS, _clamp(share_cpus, MIN_WORKER_CPUS, MAX_WORKER_CPUS)),
        memory=os.environ.get(ENV_MEMORY)
        or format_size(_clamp(share_gb, MIN_WORKER_MEMORY_GB, MAX_WORKER_MEMORY_GB)),
        pids=_env_int(ENV_PIDS, WORKER_PIDS),
    )


@dataclass(frozen=True)
class Budget:
    """One worker's whole allowance: what it may hold, and how long it may hold it.

    The pair travels together because they fail together. Memory and pids bound
    a worker that is doing something violent; the wall clock bounds one that is
    doing nothing at all, which is the more common shape of a stuck model and
    the one no cgroup notices.
    """

    limits: WorkerLimits
    timeout_s: float
    workers: int
    host: HostBudget

    @classmethod
    def for_host(
        cls,
        host: HostBudget | None = None,
        *,
        workers: int | None = None,
        timeout_s: float | None = None,
    ) -> Budget:
        """Derive an allowance for `workers` concurrent containers on `host`.

        Defaults come from `Settings`: the concurrency cap is already tuned
        against Ollama's parallel slots there, and the wall clock is already
        `SWARM_WORKER_TIMEOUT`. A second copy of either would drift.
        """
        machine = host if host is not None else HostBudget.detect()
        count = max(int(workers if workers is not None else SETTINGS.max_workers_parallel), 1)
        return cls(
            limits=_worker_limits(machine, count),
            timeout_s=float(timeout_s if timeout_s is not None else SETTINGS.worker_timeout_s),
            workers=count,
            host=machine,
        )

    @property
    def memory_gb(self) -> float:
        """One worker's memory limit, as a number rather than a flag value."""
        return parse_size(self.limits.memory)

    @property
    def total_gb(self) -> float:
        """What every worker together would hold if all of them filled up."""
        return self.memory_gb * self.workers

    @property
    def fits(self) -> bool:
        """Does the whole swarm fit in what the host had left?

        False means the limits are still enforced - a worker will be killed at
        its own ceiling either way - but the ceilings add up to more than the
        host can spare, so Ollama, the orchestrator, or the human's editor pays
        for it. Either the concurrency cap is too high for this machine or the
        reserves are wrong. A caller logs this; nothing here refuses to run,
        because refusing would leave the host with no limits at all.
        """
        # A hair of tolerance: the memory limit is whole MiB and the budget is
        # not, so an exact division comes back a rounding error over.
        return self.total_gb <= self.host.workers_gb + 1 / 1024

    def summary(self) -> str:
        """The line worth logging when a run starts."""
        head = (
            f"{self.workers} worker(s) x "
            f"{self.limits.cpus:g} cpus / {self.limits.memory} / {self.limits.pids} pids, "
            f"{self.timeout_s:g}s each; {self.host.summary()}"
        )
        if self.fits:
            return head
        return f"{head}; OVERCOMMITTED by {self.total_gb - self.host.workers_gb:.1f} GiB"


#: The allowance a dispatcher gets by not asking. Derived at import from this
#: machine and `Settings`, like `SETTINGS` itself and for the same reason: the
#: numbers are read in enough places that computing them per caller would let
#: two callers disagree. The daemon's own ceiling is absent here by design -
#: see `daemon_memory_gb`.
DEFAULT_BUDGET = Budget.for_host()


# --------------------------------------------------------------------------
# The wall clock
# --------------------------------------------------------------------------


class Waiter(Protocol):
    """The one thing `wait_within` needs. `ContainerManager` satisfies it."""

    def wait(self, handle: Handle, *, timeout_s: float | None = None) -> int: ...


@dataclass(frozen=True)
class Outcome:
    """How a container finished, including when it did not finish itself.

    `exit_code` is the worker protocol (`docs/issue-contract.md`) whether the
    worker chose it or the wall clock chose it for it, so the reconciler has
    one thing to read. `timed_out` is what distinguishes the two, and #18's
    result writer needs it: a synthesised record must be legible as
    synthesised, or a run summary reads as though a killed worker ran its
    verify command and disagreed with it.
    """

    exit_code: int
    timed_out: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def wait_within(
    waiter: Waiter,
    handle: Handle,
    *,
    timeout_s: float | None = None,
) -> Outcome:
    """Wait for one worker and always come back with an exit code.

    `ContainerManager.wait` raises when the budget is gone, which is right: a
    container that never exited has no exit code, and inventing one inside the
    mechanism would make "the worker said 1" indistinguishable from "we gave
    up". Inventing one is a policy decision, so it is made here, once, where
    `TIMEOUT_EXIT_CODE` is written down with its reasoning next to it.

    Passing `timeout_s` through rather than defaulting it keeps one number in
    play: the manager was built with a budget, and a second default here would
    be the one that silently won.

    The container is stopped but not removed when this returns a timed-out
    outcome - `wait` guarantees that - so the caller still disposes it, and the
    logs of whatever it was stuck on are still capturable. That capture is all
    the evidence there will be, since the worker never wrote its result file.
    """
    try:
        return Outcome(exit_code=waiter.wait(handle, timeout_s=timeout_s))
    except ContainerTimeout as exc:
        return Outcome(exit_code=TIMEOUT_EXIT_CODE, timed_out=True, reason=str(exc))
