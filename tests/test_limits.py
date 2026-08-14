"""Tests for the worker resource policy.

Three properties, and the first two are about the machine rather than about the
container.

**Ollama is paid first.** The budget is host RAM minus the weights minus the
system, divided by the concurrency cap - not "what does a test suite need?".
The tests pin the arithmetic to the 36 GiB machine the defaults were chosen
for, because a change that quietly hands a worker Ollama's memory looks exactly
like a change that makes workers faster.

**A limit that is not on the container is not a limit.** The derived numbers
are asserted where they actually matter: on the argv of `docker create`. A
`Budget` whose flags never reach the daemon would pass every arithmetic test in
this file.

**The wall clock produces an exit code like everything else.** A container that
never exits has none, and #18's result file is written by a worker that no
longer exists. The synthesised outcome is the whole coordination between the
two tickets, so it is tested at the seam (`ContainerTimeout`) and again against
a real container that genuinely does not stop.

The runaway tests carry the `docker` marker and are deselected by default. They
are deliberately *small* runaways: an exponential memory bomb under a 64 MiB
ceiling dies in a fraction of a second having touched 0.2% of this host's RAM,
and a fork bomb under a 24-process ceiling never reaches the VM's process
table. The host running this suite is also running 17-19 GB of Ollama weights,
so a test that proved enforcement by genuinely exhausting the machine would be
a worse outage than the bug it guards against. Every one of them filters to the
run id it minted and disposes what it spawned - this machine runs other
people's containers, and a `docker ps -a` here is somebody else's afternoon.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Sequence

import pytest

from swarm.config import SETTINGS
from swarm.containers.limits import (
    DEFAULT_BUDGET,
    MAX_WORKER_CPUS,
    MAX_WORKER_MEMORY_GB,
    MIN_WORKER_MEMORY_GB,
    TIMEOUT_EXIT_CODE,
    WORKER_PIDS,
    Budget,
    HostBudget,
    LimitError,
    Outcome,
    WorkerLimits,
    daemon_memory_gb,
    format_size,
    parse_size,
    wait_within,
)
from swarm.containers.manager import (
    ContainerError,
    ContainerManager,
    ContainerTimeout,
    DockerCLI,
    Handle,
)
from swarm.run import Run

REPO = "shahrestani-me/apiary"
OBJECTIVE = "add retry with exponential backoff to the http client"
BASE_COMMIT = "9f2c1ab3d4e5f60718293a4b5c6d7e8f90a1b2c3"

CONTAINER_ID = "c0ffee" + "0" * 58

#: The machine `config.py`'s model choices and `DEFAULT_LIMITS` were written
#: for: Mac Studio M4 Max, 36 GiB unified memory, 14 cores.
MAC_STUDIO = HostBudget(total_gb=36.0, cpus=14)

#: What Docker Desktop reports for its own VM on that same machine. A fifth of
#: host RAM, and the number that actually binds a container there.
DESKTOP_VM_GB = 7.65

ENV_NAMES = (
    "SWARM_WORKER_CPUS",
    "SWARM_WORKER_MEMORY",
    "SWARM_WORKER_PIDS",
    "SWARM_OLLAMA_RESERVE_GB",
    "SWARM_SYSTEM_RESERVE_GB",
)


@pytest.fixture(autouse=True)
def _no_ambient_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Derived numbers are only assertable if the developer's shell is not one.

    Every knob is read from the environment, and a laptop that exports one
    would otherwise turn the arithmetic tests below into tests of that export.
    """
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


@dataclass
class RecordingRunner:
    """A `Runner` that succeeds at everything and remembers the argv.

    Enough for `spawn`, which is all this file needs from the mechanism: the
    lifecycle itself is `tests/test_container_manager.py`'s subject.
    """

    calls: list[list[str]] = field(default_factory=list)

    def __call__(
        self, argv: Sequence[str], *, timeout_s: float | None, merge: bool
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        stdout = f"{CONTAINER_ID}\n" if argv[1] == "create" else ""
        return subprocess.CompletedProcess(list(argv), 0, stdout, "")

    def argv(self, subcommand: str) -> list[str]:
        for call in self.calls:
            if call[1] == subcommand:
                return call
        raise AssertionError(f"no `docker {subcommand}` was issued; got {self.calls}")

    def flag(self, subcommand: str, name: str) -> str:
        argv = self.argv(subcommand)
        return argv[argv.index(name) + 1]


@dataclass
class StuckWaiter:
    """A `Waiter` that never finishes, exactly as `ContainerManager` reports it."""

    limit: float = 600.0
    asked: list[float | None] = field(default_factory=list)

    def wait(self, handle: Handle, *, timeout_s: float | None = None) -> int:
        self.asked.append(timeout_s)
        raise ContainerTimeout(f"{handle} exceeded {self.limit}s and was stopped")


@dataclass
class FinishedWaiter:
    """A `Waiter` whose container exited on its own."""

    code: int = 0
    asked: list[float | None] = field(default_factory=list)

    def wait(self, handle: Handle, *, timeout_s: float | None = None) -> int:
        self.asked.append(timeout_s)
        return self.code


def make_run() -> Run:
    return Run.start(REPO, OBJECTIVE)


def a_handle() -> Handle:
    return Handle(id=CONTAINER_ID, run_id=make_run().id, issue=7, name="apiary-probe")


# --------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, gib",
    [
        ("4g", 4.0),
        ("4G", 4.0),
        ("4gb", 4.0),
        ("512m", 0.5),
        ("2048k", 1.0 / 512),
        ("1073741824", 1.0),
        ("  2.5g  ", 2.5),
    ],
)
def test_docker_sizes_are_understood_the_way_docker_understands_them(text: str, gib: float):
    assert parse_size(text) == pytest.approx(gib)


@pytest.mark.parametrize("text", ["", "lots", "4 gigs", "-4g", "g", "4z"])
def test_a_size_that_cannot_be_parsed_is_an_error_not_a_default(text: str):
    """A mistyped override must never become a container with a limit nobody chose."""
    with pytest.raises(LimitError):
        parse_size(text)


def test_sizes_round_trip_through_whole_mebibytes():
    assert format_size(4.0) == "4096m"
    assert parse_size(format_size(2.5)) == pytest.approx(2.5)
    # Thirds of a gigabyte are what three workers on one budget produce, and
    # they must not round up into memory the host does not have.
    assert parse_size(format_size(8.0 / 3)) <= 8.0 / 3 + 1 / 1024


# --------------------------------------------------------------------------
# The host budget
# --------------------------------------------------------------------------


def test_ollama_is_paid_before_the_workers_are():
    """36 GiB, minus 22 for the weights, minus 6 for the system, leaves 8."""
    assert MAC_STUDIO.workers_gb == pytest.approx(8.0)
    assert MAC_STUDIO.workers_cpus == pytest.approx(12.0)

    # And the reserve is the lever: a bigger model leaves less, immediately.
    hungrier = HostBudget(total_gb=36.0, cpus=14, ollama_gb=28.0)
    assert hungrier.workers_gb == pytest.approx(2.0)


def test_the_docker_vm_is_the_tighter_ceiling_when_it_is_known():
    """On Docker Desktop, host RAM is not what a container can allocate."""
    desktop = HostBudget(total_gb=36.0, cpus=14, daemon_gb=DESKTOP_VM_GB)

    assert desktop.workers_gb == pytest.approx(DESKTOP_VM_GB - 1.0)
    assert desktop.workers_gb < MAC_STUDIO.workers_gb

    # A Linux host counts the same RAM twice, and nothing changes.
    native = HostBudget(total_gb=36.0, cpus=14, daemon_gb=36.0)
    assert native.workers_gb == MAC_STUDIO.workers_gb


def test_a_host_with_no_headroom_reports_zero_rather_than_a_negative_budget():
    tiny = HostBudget(total_gb=8.0, cpus=4)
    assert tiny.workers_gb == 0.0
    assert tiny.workers_cpus == pytest.approx(2.0)


# --------------------------------------------------------------------------
# The derived limits
# --------------------------------------------------------------------------


def test_the_documented_host_derives_the_documented_defaults():
    """2 cpus / 4 GiB / 512 pids on the machine those numbers were chosen for.

    `DEFAULT_LIMITS` in `manager.py` states them as a conservative default.
    This is the same triple arrived at by subtracting Ollama, and that
    agreement is the point: the mechanism's default is now derived rather than
    asserted.
    """
    budget = Budget.for_host(MAC_STUDIO, workers=2, timeout_s=600)

    assert budget.limits.cpus == pytest.approx(2.0)
    assert budget.memory_gb == pytest.approx(4.0)
    assert budget.limits.pids == WORKER_PIDS
    assert budget.fits


def test_more_workers_divide_the_same_budget_rather_than_multiplying_it():
    two = Budget.for_host(MAC_STUDIO, workers=2)
    four = Budget.for_host(MAC_STUDIO, workers=4)

    assert four.memory_gb == pytest.approx(two.memory_gb / 2)
    assert four.total_gb == pytest.approx(two.total_gb)
    assert four.fits


def test_a_worker_never_gets_an_unlimited_or_illegal_limit_on_a_starved_host():
    """The floor exists because `--memory 0` means "no limit", not "no memory"."""
    budget = Budget.for_host(HostBudget(total_gb=8.0, cpus=4), workers=2)

    assert budget.memory_gb == pytest.approx(MIN_WORKER_MEMORY_GB)
    assert budget.limits.cpus >= 1.0
    assert budget.limits.pids == WORKER_PIDS
    # Honest about it: the limits still bind, but they add up to more than this
    # machine had spare, and something is going to pay for that.
    assert not budget.fits
    assert "OVERCOMMITTED" in budget.summary()


def test_a_large_host_is_capped_rather_than_generous():
    """A workstation does not hand 40 GiB and 30 cores to unreviewed code."""
    budget = Budget.for_host(HostBudget(total_gb=128.0, cpus=64), workers=2)

    assert budget.memory_gb == pytest.approx(MAX_WORKER_MEMORY_GB)
    assert budget.limits.cpus == pytest.approx(MAX_WORKER_CPUS)
    assert budget.fits


def test_the_environment_overrules_the_arithmetic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SWARM_WORKER_CPUS", "0.5")
    monkeypatch.setenv("SWARM_WORKER_MEMORY", "256m")
    monkeypatch.setenv("SWARM_WORKER_PIDS", "64")
    monkeypatch.setenv("SWARM_OLLAMA_RESERVE_GB", "10")

    budget = Budget.for_host(HostBudget.detect(), workers=2)

    assert budget.limits.cpus == pytest.approx(0.5)
    assert budget.limits.memory == "256m"
    assert budget.limits.pids == 64
    assert budget.host.ollama_gb == pytest.approx(10.0)


def test_an_unparseable_override_is_refused_rather_than_ignored(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SWARM_WORKER_PIDS", "many")
    with pytest.raises(LimitError):
        Budget.for_host(MAC_STUDIO, workers=2)


def test_the_default_budget_follows_settings():
    """One source for the concurrency cap and the wall clock, not two."""
    assert DEFAULT_BUDGET.workers == max(SETTINGS.max_workers_parallel, 1)
    assert DEFAULT_BUDGET.timeout_s == float(SETTINGS.worker_timeout_s)
    assert DEFAULT_BUDGET.limits.pids >= 1
    assert parse_size(DEFAULT_BUDGET.limits.memory) >= MIN_WORKER_MEMORY_GB


def test_the_summary_names_the_reserve_it_took():
    summary = Budget.for_host(MAC_STUDIO, workers=2, timeout_s=600).summary()

    assert "22" in summary and "Ollama" in summary
    assert "600s" in summary
    assert "OVERCOMMITTED" not in summary


# --------------------------------------------------------------------------
# The flags on the container
# --------------------------------------------------------------------------


def test_swap_is_pinned_to_the_memory_limit():
    """Docker's default lets a container have twice its limit before it dies.

    Two gigabytes of swap on the VM's disk is the "the Mac becomes unusable"
    failure with an extra step in front of it.
    """
    flags = WorkerLimits(cpus=1.5, memory="512m", pids=64).flags()

    assert flags == [
        "--cpus", "1.5",
        "--memory", "512m",
        "--pids-limit", "64",
        "--memory-swap", "512m",
    ]


def test_every_derived_limit_reaches_the_docker_command():
    """The test that would fail if this module were arithmetic and nothing else."""
    budget = Budget.for_host(MAC_STUDIO, workers=2, timeout_s=600)
    runner = RecordingRunner()
    manager = ContainerManager(run=make_run(), limits=budget.limits, env={}, runner=runner)

    manager.spawn(19, BASE_COMMIT)

    assert runner.flag("create", "--cpus") == "2"
    assert runner.flag("create", "--memory") == "4096m"
    assert runner.flag("create", "--memory-swap") == "4096m"
    assert runner.flag("create", "--pids-limit") == str(WORKER_PIDS)


# --------------------------------------------------------------------------
# The wall clock
# --------------------------------------------------------------------------


def test_a_worker_that_finished_reports_its_own_exit_code():
    waiter = FinishedWaiter(code=1)

    outcome = wait_within(waiter, a_handle(), timeout_s=42.0)

    assert outcome == Outcome(exit_code=1)
    assert not outcome.timed_out
    assert not outcome.ok
    # Forwarded, not re-defaulted: the budget the manager was built with is the
    # one that applies, and a second default here would be the one that won.
    assert waiter.asked == [42.0]


def test_a_killed_worker_gets_the_result_it_could_not_write():
    """#18's coordination point. The worker is SIGKILLed; it reports nothing.

    Recorded as exit 1 - `docs/issue-contract.md` §4's "task failed, attempt
    consumed" - because a model looping is the failure the attempt cap exists
    to bound. Exit 2 would make a permanently hanging task retry for free.
    """
    outcome = wait_within(StuckWaiter(limit=600.0), a_handle())

    assert outcome.timed_out
    assert outcome.exit_code == TIMEOUT_EXIT_CODE
    assert not outcome.ok
    # The reason survives into the run summary, so it has to say what happened.
    assert "600.0s" in outcome.reason and "stopped" in outcome.reason


def test_no_timeout_is_swallowed_into_a_zero():
    """The one outcome that must be impossible: a killed worker reading as a pass."""
    outcome = wait_within(StuckWaiter(), a_handle())

    assert outcome.exit_code != 0
    assert not outcome.ok


# --------------------------------------------------------------------------
# Against a real daemon
# --------------------------------------------------------------------------

#: Small enough that the bomb below dies inside a second and touches 0.2% of a
#: 36 GiB host - see the module docstring on why these are not real runaways.
PROBE_MEMORY = "64m"
PROBE_MEMORY_BYTES = 64 * 1024 * 1024
PROBE_PIDS = 24
PROBE_CPUS = 0.5

#: The wall clock under test. Seconds, because the thing being proved is that
#: something kills the container, not how patient the real budget is.
PROBE_TIMEOUT_S = 3.0

FORK_ATTEMPTS = 200

CANDIDATE_IMAGES = ("apiary-worker", "busybox", "alpine", "python:3.12-slim")


@pytest.fixture(scope="module")
def trivial_image() -> str:
    """A locally present image with a shell. Nothing is pulled.

    A near-twin of `test_container_manager.py`'s fixture; the shared home for
    it is `tests/conftest.py`, which is not this ticket's file. Pulling would
    make a `docker`-marked test quietly need the network too.
    """
    docker = DockerCLI()
    for name in CANDIDATE_IMAGES:
        try:
            docker("image", "inspect", "--format", "{{.Id}}", name)
        except ContainerError:
            continue
        return name
    pytest.skip(
        "no local image to spawn a probe container from; build one with "
        "`docker build -f Dockerfile.worker -t apiary-worker .`"
    )


@pytest.fixture()
def runaway(trivial_image: str) -> ContainerManager:
    """A manager whose workers are held on a very short leash.

    `env={}` deliberately: nothing below deserves a token, and the point of
    this fixture is that what runs in it is assumed hostile.
    """
    return ContainerManager(
        run=make_run(),
        image=trivial_image,
        env={},
        limits=WorkerLimits(cpus=PROBE_CPUS, memory=PROBE_MEMORY, pids=PROBE_PIDS),
        timeout_s=PROBE_TIMEOUT_S,
    )


def shell(manager: ContainerManager, script: str, issue: int = 19) -> Handle:
    return manager.spawn(issue, BASE_COMMIT, entrypoint="/bin/sh", command=["-c", script])


def inspected(manager: ContainerManager, handle: Handle, template: str) -> str:
    """One field off this container. Addressed by id, never by a listing."""
    docker: Any = manager.docker
    return docker("inspect", "--format", template, handle.id).strip()


@pytest.mark.docker
def test_the_daemon_reports_a_ceiling_this_module_can_use():
    """`docker info` is one number away from a budget, and it has to parse.

    Marked rather than mocked: the format string is a contract with the CLI,
    and a double asserting that `"{{.MemTotal}}"` returns bytes would only ever
    test the double.
    """
    ceiling = daemon_memory_gb()

    assert ceiling > 0
    budget = Budget.for_host(HostBudget(total_gb=36.0, cpus=14, daemon_gb=ceiling), workers=2)
    assert budget.memory_gb <= MAX_WORKER_MEMORY_GB


@pytest.mark.docker
def test_the_limits_are_on_the_container_the_daemon_actually_made(runaway: ContainerManager):
    """Before proving anything is enforced, prove the flags were accepted."""
    handle = shell(runaway, "sleep 30")
    try:
        assert inspected(runaway, handle, "{{.HostConfig.Memory}}") == str(PROBE_MEMORY_BYTES)
        # Equal to `Memory`, so the container gets no swap to die slowly in.
        assert inspected(runaway, handle, "{{.HostConfig.MemorySwap}}") == str(PROBE_MEMORY_BYTES)
        assert inspected(runaway, handle, "{{.HostConfig.PidsLimit}}") == str(PROBE_PIDS)
        assert inspected(runaway, handle, "{{.HostConfig.NanoCpus}}") == str(int(PROBE_CPUS * 1e9))
    finally:
        runaway.dispose(handle)


@pytest.mark.docker
def test_a_memory_bomb_kills_itself_rather_than_the_host(runaway: ContainerManager):
    """#19's acceptance criterion for the memory half.

    The workload doubles a shell variable forever - unbounded by construction,
    and bounded in practice by the 64 MiB ceiling it reaches in about a
    quarter of a second. Under Docker's default swap allowance it would instead
    grind through 128 MiB of the VM's disk first, which is why `WorkerLimits`
    pins swap.
    """
    handle = shell(runaway, 's=x; while :; do s="$s$s"; done')
    try:
        outcome = wait_within(runaway, handle, timeout_s=PROBE_TIMEOUT_S)

        # The cgroup got there first; the wall clock was never needed.
        assert not outcome.timed_out
        assert outcome.exit_code != 0
        assert inspected(runaway, handle, "{{.State.OOMKilled}}") == "true"
    finally:
        runaway.dispose(handle)
    assert runaway.find() == []


@pytest.mark.docker
def test_a_fork_bomb_hits_its_own_process_ceiling(runaway: ContainerManager):
    """#19's acceptance criterion for the pids half.

    200 attempted forks against a ceiling of 24. The container dies on its own
    process table rather than the VM's, and says so.
    """
    handle = shell(
        runaway,
        f"i=0; while [ $i -lt {FORK_ATTEMPTS} ]; do sleep 30 & i=$((i+1)); done; "
        'echo "spawned all"',
    )
    try:
        outcome = wait_within(runaway, handle, timeout_s=PROBE_TIMEOUT_S)

        assert not outcome.timed_out
        assert outcome.exit_code != 0
        logs = runaway.logs(handle)
        # dash says "Cannot fork", busybox says "can't fork"; both name it.
        assert "fork" in logs.lower()
        assert "spawned all" not in logs
    finally:
        runaway.dispose(handle)
    assert runaway.find() == []


@pytest.mark.docker
def test_an_infinite_loop_is_killed_by_the_wall_clock_and_still_leaves_evidence(
    runaway: ContainerManager,
):
    """#19's acceptance criterion for the half no cgroup notices.

    An infinite loop breaks no limit: it holds a few kilobytes, forks nothing,
    and is capped at half a core. Only the wall clock ends it - and then #18's
    result file was never written, so `wait_within`'s synthesised outcome and
    the logs `dispose` salvages are the entire record of the attempt.
    """
    handle = shell(runaway, 'echo "looping"; while :; do :; done')
    try:
        outcome = wait_within(runaway, handle, timeout_s=PROBE_TIMEOUT_S)

        assert outcome.timed_out
        assert outcome.exit_code == TIMEOUT_EXIT_CODE
        assert str(handle.issue) in outcome.reason

        # Stopped, not removed: the run continues, this task is failed, and the
        # evidence is still readable.
        assert [h.id for h in runaway.find(issue=19)] == [handle.id]
        assert "looping" in runaway.dispose(handle)
    finally:
        runaway.dispose(handle)
    assert runaway.find() == []
