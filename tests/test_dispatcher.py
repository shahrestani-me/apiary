"""Tests for the dispatcher.

Four properties carry this file, and each is a way the swarm loses work rather
than a way it computes something wrong.

**The container is the claim.** It used to be a `swarm:claimed` label written
immediately before the spawn, and the ordering assertions here were about which
of two crash windows that write chose. #152 removed the write: the thing that
says a task is claimed is the running container, which the resolver reads
directly. So what is asserted now is that a dispatch writes nothing to the
tracker at all, and that a spawn which failed gives its claim back only when the
daemon says there is no container to give it back to - releasing on any other
reading buys the issue a second worker next cycle, and one of the two pushes is
lost.

**Overlapping file sets never run at once.** Against work already in flight and
against each other, case-insensitively, whatever the planner promised.

**The cap is an inference budget, not a worker count.** The orchestrator's own
planner and judge calls hold a slot, and on Docker Desktop the Linux VM's
memory - not the host's - is what a container can have.

**A cycle that only dispatched has nothing to ask a model about.** The
dispatcher reaches for no model on any path, and says so in the one bit the
reconcile loop needs to decide whether this cycle is worth a swap.

Entirely hermetic. `Spawner` is a `Protocol` and the double below records into a
log, so every ordering assertion is data; `Labeller` is the empty protocol #152
left behind and the same double stands in for it, so that the two-argument
signature the dispatcher still has is exercised by these tests as well.

Each fixture says what state its task is in by passing `entry(state=...)`, which
is stashed on `LedgerEntry.labels` and turned into the cycle's `Belief` by
`fixtures.belief`. That is the same declaration the tests always made - it was
read back off `LedgerEntry.state_label` until #152 - moved to where production
now reads it from.
"""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from swarm.containers.limits import HostBudget, LimitError
from swarm.containers.manager import (
    STACK_IMAGES_ENV,
    ContainerError,
    DockerError,
    Handle,
    StackImages,
)
from swarm.github.ledger import Ledger, LedgerEntry
from swarm.github.refs import issue_number, task_ref
from swarm.orchestrator import dispatcher
from swarm.orchestrator.derived import BLOCKED, ELIGIBLE, LANDED, NEEDS_HUMAN
from swarm.orchestrator.dispatcher import (
    CLAIMED,
    ORCHESTRATOR_SLOTS,
    REVIEW,
    Capacity,
    container_memory_cap,
    dispatch,
    held_files,
    plan_dispatch,
)
from swarm.taskref import TaskRef
from fixtures.belief import fixture_belief


# The cycle's belief, supplied from what each fixture declares (see
# `fixtures.belief`). It was read off `LedgerEntry.state_label` until #152.
def _plan_dispatch_(book, *args, **kwargs):
    kwargs.setdefault("believed", fixture_belief(book))
    return plan_dispatch(book, *args, **kwargs)


def _dispatch_(client, manager, book, *args, **kwargs):
    """`dispatch`, with the same belief `_plan_dispatch_` supplies.

    `dispatch` plans before it spawns, so a cycle with no belief believes every
    task to be in no state at all and spawns nothing - which would make most of
    the tests below pass for the wrong reason.
    """
    kwargs.setdefault("believed", fixture_belief(book))
    return dispatch(client, manager, book, *args, **kwargs)


BASE_COMMIT = "9f2c1ab3d4e5f60718293a4b5c6d7e8f90a1b2c3"

#: The machine `config.py` and `limits.py` were written against: 36 GiB of host
#: RAM, and a Docker Desktop VM holding 7.6 of it.
HOST_GB = 36.0
DAEMON_GB = 7.6


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def entry(number: int, *files: str, state: str = ELIGIBLE, attempt: int = 0) -> LedgerEntry:
    """One ledger entry, and the state this fixture says its task is in.

    The state goes on `labels` because that is where `fixtures.belief` reads a
    fixture's declaration from. Nothing in production reads it back from there:
    the belief built out of it is what the dispatcher is given.
    """
    return LedgerEntry(
        number=number,
        title=f"issue {number}",
        task_id=f"task-{number}",
        attempt=attempt,
        goal="do the thing",
        files=files or (f"src/mod{number}.py",),
        verify="python -m pytest -q",
        blocked_by=(),
        labels=frozenset({state}),
    )


def ledger(*entries: LedgerEntry) -> Ledger:
    return Ledger(entries={item.task_id: item for item in entries})


def capacity(workers: int) -> Capacity:
    """A cap of exactly `workers`, with the memory bound left unconsulted.

    Explicit rather than detected, so the planning tests say what they mean and
    do not change answer on a machine with different RAM.
    """
    return Capacity(slots=workers + ORCHESTRATOR_SLOTS, configured=workers)


@dataclass
class FakeSwarm:
    """`Labeller` and `Spawner` in one object, recording into one log.

    The log outlived the reason it was one log. It existed so that a label call
    and a spawn could be ordered against each other; `Labeller` is empty since
    #152 and this double satisfies it by having no methods it needs, so the only
    things in the log now are the daemon calls - which is itself the assertion
    several tests below make.
    """

    log: list[str] = field(default_factory=list)
    spawn_error: Exception | None = None
    #: Per-issue spawn failures, which is what #94 is about: one image is
    #: missing and seventeen other issues are fine. `spawn_error` stays for the
    #: fleet-wide case and the two compose, this one winning.
    spawn_errors: dict[int, Exception] = field(default_factory=dict)
    handles: dict[int, Handle] = field(default_factory=dict)
    #: issue -> the attempt the dispatcher told that container it is. See `spawn`.
    attempts: dict[int, int | None] = field(default_factory=dict)
    #: What `find` reports, per issue. Empty means "the daemon says nothing is
    #: running under that issue", which is the only reading that releases a
    #: claim.
    running: dict[int, list[Handle]] = field(default_factory=dict)
    find_error: Exception | None = None

    # --- Labeller -------------------------------------------------------
    #
    # `add_labels` and `remove_label` stood here and recorded the two calls a
    # claim was made of. #152 deleted both calls, and the protocol they
    # satisfied is empty now, so a double with nothing here satisfies it.

    # --- Spawner --------------------------------------------------------

    def spawn(
        self,
        task: TaskRef,
        base_commit: str,
        *,
        issue: int | None = None,
        image: str | None = None,
        attempt: int | None = None,
    ) -> Handle:
        # The fake keeps docker's int-keyed bookkeeping; only the seam changed.
        # `issue` is asserted rather than ignored: the dispatcher has to pass
        # both, and a caller that stopped would leave the worker without the
        # number it needs to open a pull request.
        assert issue is not None and task.label_value == str(issue)
        # Kept per issue rather than logged, because the assertion about it is
        # "which number did *this* task get" and a log line would make that a
        # string-parsing exercise.
        self.attempts[issue] = attempt
        # The image is in the log, because #99's whole question is which one a
        # task got and the ordering assertions read this log.
        self.log.append(f"spawn #{issue}" + (f" [{image}]" if image else ""))
        error = self.spawn_errors.get(issue, self.spawn_error)
        if error is not None:
            raise error
        handle = Handle(id=f"{issue:0>64x}", run_id="apiary-test", issue=issue)
        self.handles[issue] = handle
        return handle

    def find(self, *, ref: TaskRef | None = None) -> list[Handle]:
        # The fake keeps its int-keyed bookkeeping; the seam is what changed.
        issue = None if ref is None else issue_number(ref)
        self.log.append(f"find #{issue}")
        if self.find_error is not None:
            raise self.find_error
        return list(self.running.get(issue, []))

    # --- what the assertions read ---------------------------------------

    @property
    def spawned(self) -> list[int]:
        return [
            int(line.split("#")[1].split()[0]) for line in self.log if line.startswith("spawn")
        ]

    @property
    def images(self) -> list[str]:
        """The image each spawn was asked for, in order."""
        return [
            line.split("[", 1)[1].rstrip("]")
            for line in self.log
            if line.startswith("spawn") and "[" in line
        ]

    @property
    def probed(self) -> list[int]:
        """Issues the daemon was asked about before a claim was given back.

        This replaces `claimed` and `released`, which read the two label calls
        out of the log. There is no such call to read: what a release does now
        is ask `find` and answer, so the question this double can still answer
        is which issues were asked about, and `DispatchFailure.claimed` carries
        the answer.
        """
        return [int(line.split("#")[1]) for line in self.log if line.startswith("find")]


def daemon_down() -> DockerError:
    return DockerError(["docker", "create"], 125, "Cannot connect to the Docker daemon")


# --------------------------------------------------------------------------
# Capacity
# --------------------------------------------------------------------------


def test_the_orchestrator_holds_one_of_the_hosts_inference_slots():
    # OLLAMA_MAX_LOADED_MODELS=1 and two models that do not fit together, so a
    # planner or judge call is a ~6.7 s swap rather than a queued request. A cap
    # that handed both slots to workers would leave the process that has to
    # judge them waiting behind one.
    assert Capacity(slots=2, configured=8).workers == 1
    assert Capacity(slots=4, configured=8).workers == 3


def test_a_single_slot_host_still_dispatches_one_worker():
    # The arithmetic wants zero here, and a run that dispatches nothing forever
    # is indistinguishable from a run with nothing left to do.
    assert Capacity(slots=1, configured=8).workers == 1


def test_the_configured_cap_binds_when_it_is_lower_than_the_slots():
    cap = Capacity(slots=8, configured=2)

    assert cap.workers == 2
    assert cap.bound_by == "SWARM_MAX_PARALLEL"


def test_a_cap_configured_at_zero_is_honoured_as_written():
    # The floor exists for derived arithmetic. Somebody who typed 0 meant it.
    assert Capacity(slots=8, configured=0).workers == 0


def test_the_docker_vms_memory_binds_long_before_the_hosts_does(monkeypatch):
    for name in ("SWARM_WORKER_CPUS", "SWARM_WORKER_MEMORY", "SWARM_WORKER_PIDS"):
        monkeypatch.delenv(name, raising=False)
    host = HostBudget(total_gb=HOST_GB, cpus=14)
    vm = HostBudget(total_gb=HOST_GB, cpus=14, daemon_gb=DAEMON_GB)

    # Docker Desktop runs a Linux VM with a fixed allocation, already carved out
    # of the host before Ollama sees any of it. Believing the host's 36 GiB
    # overcommits the VM by more than it has.
    assert container_memory_cap(host, ceiling=12) == 8
    assert container_memory_cap(vm, ceiling=12) == 6


def test_the_memory_bound_is_never_asked_about_a_concurrency_nobody_could_use(monkeypatch):
    for name in ("SWARM_WORKER_CPUS", "SWARM_WORKER_MEMORY", "SWARM_WORKER_PIDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "2")

    cap = Capacity.detect(host=HostBudget(total_gb=HOST_GB, cpus=14))

    # Two slots less the orchestrator's is one worker, so the search stops
    # there rather than pricing eight of them.
    assert cap.memory == 1
    assert cap.workers == 1
    assert cap.bound_by in cap.bounds


def test_the_slot_count_comes_from_the_ollama_server_not_from_us(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "6")

    # It is the host server's property: raising SWARM_MAX_PARALLEL cannot make
    # the machine answer more requests at once.
    assert Capacity.detect(host=HostBudget(total_gb=HOST_GB, cpus=14)).slots == 6


def test_an_unreadable_slot_count_is_refused_rather_than_defaulted(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "two")

    # Falling back would run a concurrency nobody chose, which is the one thing
    # this type exists to prevent.
    with pytest.raises(LimitError):
        Capacity.detect(host=HostBudget(total_gb=HOST_GB, cpus=14))


def test_the_capacity_summary_names_the_constraint_a_human_would_change():
    assert "SWARM_MAX_PARALLEL" in Capacity(slots=8, configured=2, memory=6).summary()


# --------------------------------------------------------------------------
# Planning: the cap
# --------------------------------------------------------------------------


def test_more_ready_issues_than_the_cap_dispatches_exactly_the_cap():
    plan = _plan_dispatch_(ledger(entry(4), entry(5), entry(6), entry(7)), capacity=capacity(2))

    # #21's acceptance criterion, and the oldest-first rule with it: issue
    # numbers ascend with creation, so this is arrival order.
    assert plan.numbers == (4, 5)
    assert [held.number for held in plan.deferred] == [6, 7]
    assert all("at the cap of 2" in held.reason for held in plan.deferred)


def test_containers_already_running_spend_the_same_cap():
    plan = _plan_dispatch_(
        ledger(entry(4, state=CLAIMED), entry(5), entry(6)),
        capacity=capacity(2),
    )

    # A previous cycle's worker queues against the same inference slots as this
    # cycle's would. A cap counted per cycle is not a cap.
    assert plan.numbers == (5,)
    assert [item.number for item in plan.in_flight] == [4]


def test_a_cap_already_full_dispatches_nothing_and_says_why():
    plan = _plan_dispatch_(
        ledger(entry(4, state=CLAIMED), entry(5, state=CLAIMED), entry(6)),
        capacity=capacity(2),
    )

    assert plan.numbers == ()
    assert [held.number for held in plan.deferred] == [6]


def test_only_ready_issues_are_candidates():
    plan = _plan_dispatch_(
        ledger(
            entry(1, state=BLOCKED),
            entry(2, state=LANDED),
            entry(3, state=NEEDS_HUMAN),
            entry(4),
        ),
        capacity=capacity(4),
    )

    # Blocked work has an unmet dependency, done and failed are terminal (§4).
    # None of the three is deferred either - they were never candidates.
    assert plan.numbers == (4,)
    assert plan.deferred == ()


def test_a_readiness_verdict_narrows_the_candidates_but_never_promotes_one():
    entries = ledger(entry(4), entry(5), entry(6, state=LANDED))

    plan = _plan_dispatch_(entries, capacity=capacity(4), ready=[task_ref(5), task_ref(6)])

    # The readiness pass ran this cycle, so passing its verdict saves resolving
    # the same question twice. It is a filter, not an authority: #4 is believed
    # eligible and is not on the list, #6 is on the list and is believed done
    # with, and the dispatch is the intersection - both have to agree.
    assert plan.numbers == (5,)


def test_a_blocked_task_the_readiness_pass_called_ready_is_dispatched():
    """The one state a verdict *does* admit, and it is not a promotion.

    Readiness sees things the resolver cannot - dependency rings, unresolvable
    refs, dependencies that are not tasks in the plan - so between `eligible`
    and `blocked` its verdict is the better one. The resolver's own half would
    hold every task waiting on a hand-written issue blocked forever.
    """
    entries = ledger(entry(4, state=BLOCKED), entry(5, state=CLAIMED))

    plan = _plan_dispatch_(entries, capacity=capacity(4), ready=[task_ref(4), task_ref(5)])

    # And it stops at waiting: #5 is on the same list and a container is
    # editing its files, which no verdict overrules.
    assert plan.numbers == (4,)
    assert [item.number for item in plan.in_flight] == [5]


# --------------------------------------------------------------------------
# Planning: overlapping files
# --------------------------------------------------------------------------


def test_two_ready_issues_over_one_file_serialize():
    plan = _plan_dispatch_(
        ledger(
            entry(4, "src/swarm/graph.py", "tests/test_graph.py"),
            entry(5, "src/swarm/graph.py", "docs/graph.md"),
        ),
        capacity=capacity(4),
    )

    # Room for both, and still only one runs: two branches editing one file
    # cannot both merge, and the wrong place to find that out is inside a
    # running container.
    assert plan.numbers == (4,)
    assert str(plan.deferred[0]) == "#5: src/swarm/graph.py is held by #4"


def test_a_running_container_holds_its_files_against_the_whole_backlog():
    plan = _plan_dispatch_(
        ledger(
            entry(4, "src/swarm/graph.py", state=CLAIMED),
            entry(5, "src/swarm/graph.py"),
            entry(6, "src/swarm/state.py"),
        ),
        capacity=capacity(4),
    )

    assert plan.numbers == (6,)
    assert "held by #4" in plan.deferred[0].reason


def test_an_open_pr_holds_its_files_too():
    plan = _plan_dispatch_(
        ledger(entry(4, "src/swarm/graph.py", state=REVIEW), entry(5, "src/swarm/graph.py")),
        capacity=capacity(4),
    )

    # The container is gone but the branch is not: that work is an unmerged PR
    # against the same base, and a second task over the same file is the merge
    # conflict this rule exists to prevent. The hold lifts when the PR does.
    assert plan.numbers == ()
    assert "held by #4" in plan.deferred[0].reason


def test_overlap_is_case_insensitive_because_this_filesystem_is():
    plan = _plan_dispatch_(
        ledger(entry(4, "src/Thing.py"), entry(5, "src/thing.py")),
        capacity=capacity(4),
    )

    # `docs/issue-contract.md` §1.3. On the development host these are one file,
    # and a case-sensitive intersection would call the two tasks disjoint and
    # then hand them the same file.
    assert plan.numbers == (4,)


def test_disjoint_issues_run_together_up_to_the_cap():
    plan = _plan_dispatch_(
        ledger(entry(4, "src/a.py"), entry(5, "src/b.py"), entry(6, "src/c.py")),
        capacity=capacity(3),
    )

    assert plan.numbers == (4, 5, 6)
    assert plan.deferred == ()


def test_a_file_claimed_by_two_in_flight_issues_reports_the_older_one():
    # Not a state the system creates, but two live containers over one file set
    # - however they came to exist - do. Refusing to dispatch is the decision
    # that matters; which number is named is only diagnostics, and it must be
    # stable.
    assert held_files([entry(9, "src/a.py"), entry(4, "src/a.py")]) == {"src/a.py": 4}


# --------------------------------------------------------------------------
# Dispatching
# --------------------------------------------------------------------------


def test_dispatching_an_issue_writes_nothing_and_spawns_a_container():
    swarm = FakeSwarm()

    _dispatch_(swarm, swarm, ledger(entry(7)), BASE_COMMIT, capacity=capacity(1))

    # This asserted `+swarm:claimed`, `-swarm:ready`, spawn - in that order,
    # because the label was the claim and the order chose which crash window the
    # cycle lived with. The container is the claim now, so the whole of what a
    # successful dispatch does is the last line, and a human watching the
    # repository sees the swarm start work without the issue changing.
    assert swarm.log == ["spawn #7 [apiary-worker]"]


def test_a_spawned_worker_is_told_which_attempt_it_is():
    """The counter reaches the container from the entry, not from the issue body.

    The worker names its branch `apiary/<ref>-attempt-N`, and that name is what
    derived state is read back from (#144, #145). Before this it learned N by
    fetching the issue and parsing the marker - which made the customer's issue
    body the transport for one of apiary's own numbers, and is the write #152
    removes.
    """
    swarm = FakeSwarm()

    _dispatch_(swarm, swarm, ledger(entry(7, attempt=2)), BASE_COMMIT, capacity=capacity(1))

    assert swarm.attempts == {7: 2}


def test_each_worker_is_told_its_own_attempt_and_not_the_fleets():
    """Two tasks at different attempts in one cycle is the ordinary case.

    A retry and a first attempt dispatch together all the time, and one number
    for the whole cycle would name one of the two branches wrongly - silently,
    because the push succeeds either way.
    """
    swarm = FakeSwarm()

    _dispatch_(
        swarm, swarm, ledger(entry(4, attempt=0), entry(5, attempt=3)),
        BASE_COMMIT, capacity=capacity(2),
    )

    assert swarm.attempts == {4: 0, 5: 3}


# `test_the_new_label_is_added_before_the_old_one_is_removed` stood here. It
# asserted that a claim added `swarm:claimed` before it removed `swarm:ready`,
# so that a crash between the two calls left an issue with two state labels
# (which §3's precedence repairs) rather than none (which puts it outside the
# ledger). #152 removed both calls: there is no add, no remove, and no gap
# between them for a crash to land in, so nothing here can fail.
#
# `test_each_issue_is_claimed_immediately_before_its_own_spawn` stood here too,
# asserting that the claim/spawn pairs were interleaved per issue rather than
# claimed in one phase and spawned in another. With the claim a no-op there is
# one call per issue left and no second phase to interleave it with; that a
# failed spawn strands only its own issue is still asserted, by
# `test_one_failed_spawn_does_not_stop_the_other_issues_in_the_cycle` and
# `test_a_daemon_that_is_not_there_stops_the_cycle_rather_than_burning_every_claim`.


def test_a_dispatched_issue_reports_the_container_holding_it():
    swarm = FakeSwarm()

    report = _dispatch_(swarm, swarm, ledger(entry(7)), BASE_COMMIT, capacity=capacity(1))

    assert [item.number for item in report.dispatched] == [7]
    assert report.handles[0].issue == 7
    assert report.failed == ()


def test_a_spawn_that_failed_on_a_dead_daemon_keeps_its_claim():
    swarm = FakeSwarm(spawn_error=daemon_down())

    report = _dispatch_(swarm, swarm, ledger(entry(7)), BASE_COMMIT, capacity=capacity(1))

    # Releasing it looks tidier and is wrong: a `docker start` whose reply this
    # process never read may still have started a container, and treating the
    # issue as unclaimed would give it a second one next cycle. #35 resolves
    # that by looking for a live container first - and with the daemon down,
    # `find` could not answer that question either, so nothing is asked.
    assert report.failed[0].claimed is True
    assert swarm.probed == []


def test_a_daemon_that_is_not_there_stops_the_cycle_rather_than_burning_every_claim():
    swarm = FakeSwarm(spawn_error=daemon_down())

    report = _dispatch_(swarm, swarm, ledger(entry(4), entry(5)), BASE_COMMIT, capacity=capacity(2))

    # If the daemon is down the second spawn fails exactly like the first, and
    # each attempt costs one more stuck claim. #5 is never reached and stays
    # eligible. This is the case the `break` was always right about, and it is
    # the only one left that still halts a cycle.
    assert swarm.spawned == [4]
    assert [failure.number for failure in report.failed] == [4]
    assert report.failed[0].fatal is True


# --------------------------------------------------------------------------
# One failed spawn must not halt the whole cycle (#94)
# --------------------------------------------------------------------------
#
# The `break` above was defensible while one image existed: a spawn failure
# meant the daemon was down, so the second spawn would fail exactly like the
# first. #99 chooses the image per task, and from then on a single missing
# image is a fact about one issue - one that would otherwise halt every cycle
# and burn attempts right across the ledger, on issues with nothing wrong with
# them.


def missing_image() -> DockerError:
    """The failure #99 makes reachable: this task's stack has no image here."""
    return DockerError(
        ["docker", "create"], 125, "Unable to find image 'apiary-worker-node:latest' locally"
    )


def test_one_failed_spawn_does_not_stop_the_other_issues_in_the_cycle():
    """The ticket's first criterion, and the reason it exists."""
    swarm = FakeSwarm(spawn_errors={4: missing_image()})

    report = _dispatch_(
        swarm, swarm, ledger(entry(4), entry(5), entry(6)), BASE_COMMIT, capacity=capacity(3)
    )

    assert swarm.spawned == [4, 5, 6]
    assert [item.number for item in report.dispatched] == [5, 6]
    assert [failure.number for failure in report.failed] == [4]


def test_a_deferred_issue_does_not_stay_claimed_with_no_container():
    """The second criterion. Left claimed, it waits for #35 to sweep it and
    costs an attempt on the way through; released, the next cycle just retries
    it."""
    swarm = FakeSwarm(spawn_errors={7: missing_image()})

    report = _dispatch_(swarm, swarm, ledger(entry(7)), BASE_COMMIT, capacity=capacity(1))

    assert report.failed[0].claimed is False
    # The release is the daemon's answer and nothing else: asked after the spawn
    # failed, answered "no container", and with no label pair to write behind it
    # the whole of the cycle's dealings with #7 is these two lines.
    assert swarm.log == ["spawn #7 [apiary-worker]", "find #7"]


def test_a_deferred_issue_keeps_its_claim_when_a_container_did_start():
    """The ambiguous case the old comment was right about, now decided by
    asking rather than by assuming. `docker start` failed this process's read;
    the daemon says a container exists; releasing would buy a second worker."""
    swarm = FakeSwarm(
        spawn_errors={7: missing_image()},
        running={7: [Handle(id="f" * 64, run_id="apiary-test", issue=7)]},
    )

    report = _dispatch_(swarm, swarm, ledger(entry(7)), BASE_COMMIT, capacity=capacity(1))

    # Asked, and the answer is what keeps the claim standing for #35 to resolve.
    assert swarm.probed == [7]
    assert report.failed[0].claimed is True


def test_a_probe_that_cannot_answer_keeps_the_claim():
    """False is the safe answer: a claim #35 sweeps beats two containers."""
    swarm = FakeSwarm(spawn_errors={7: missing_image()}, find_error=daemon_down())

    report = _dispatch_(swarm, swarm, ledger(entry(7)), BASE_COMMIT, capacity=capacity(1))

    assert swarm.probed == [7]
    assert report.failed[0].claimed is True


def test_an_unrecognised_spawn_failure_defers_rather_than_halting():
    """`DAEMON_DOWN_RE` is narrow and the default is to keep going.

    Being wrong that way costs one wasted spawn attempt per issue and no
    attempt counters. Being wrong the other way is the bug this ticket fixes.
    """
    swarm = FakeSwarm(
        spawn_errors={4: DockerError(["docker", "create"], 125, "something nobody has seen")}
    )

    report = _dispatch_(swarm, swarm, ledger(entry(4), entry(5)), BASE_COMMIT, capacity=capacity(2))

    assert swarm.spawned == [4, 5]
    assert report.failed[0].fatal is False


@pytest.mark.parametrize(
    "output",
    [
        "Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
        "Is the docker daemon running?",
        "error during connect: Get http://docker/v1.45/containers/create",
    ],
)
def test_every_daemon_signature_stops_the_cycle(output):
    swarm = FakeSwarm(spawn_error=DockerError(["docker", "create"], 125, output))

    report = _dispatch_(swarm, swarm, ledger(entry(4), entry(5)), BASE_COMMIT, capacity=capacity(2))

    assert swarm.spawned == [4]
    assert report.failed[0].fatal is True


def test_a_docker_binary_that_is_not_on_path_is_a_daemon_level_failure():
    """`DockerCLI._run` raises a bare `ContainerError` for this one, not a
    `DockerError` - and no binary means no daemon for any issue. It is also the
    failure the orchestrator image actually shipped with."""
    swarm = FakeSwarm(
        spawn_error=ContainerError("'docker' is not on PATH; the orchestrator reaches the daemon")
    )

    report = _dispatch_(swarm, swarm, ledger(entry(4), entry(5)), BASE_COMMIT, capacity=capacity(2))

    assert swarm.spawned == [4]
    assert report.failed[0].fatal is True


def test_the_two_kinds_of_failure_are_distinguishable_in_the_log():
    """The last criterion. They read identically at a glance and want opposite
    responses: restart Docker, versus look at that one issue."""
    halted = FakeSwarm(spawn_error=daemon_down())
    deferred = FakeSwarm(spawn_errors={7: missing_image()})

    fatal = _dispatch_(halted, halted, ledger(entry(7)), BASE_COMMIT, capacity=capacity(1)).failed[0]
    local = _dispatch_(
        deferred, deferred, ledger(entry(7)), BASE_COMMIT, capacity=capacity(1)
    ).failed[0]

    assert "daemon-level, cycle stopped" in str(fatal)
    assert "this issue only" in str(local)
    assert str(fatal) != str(local)


# `test_a_claim_that_could_not_be_written_spawns_nothing` stood here, with a
# `rate_limited()` helper that made the fake's `add_labels` raise a 503. It
# asserted that a claim GitHub refused stopped the cycle before any spawn, so
# that a rate limit could not produce a container running against an issue still
# reading `swarm:ready`. `claim` writes nothing to GitHub now, so there is no
# request left to rate-limit and no `claim failed` branch to reach: the label
# write it was about is the one #152 removed. `DispatchFailure.claimed` and
# `fatal` are still asserted, by the spawn-failure tests above.


def test_a_dry_run_writes_nothing_at_all():
    swarm = FakeSwarm()

    report = _dispatch_(
        swarm, swarm, ledger(entry(4), entry(5)), BASE_COMMIT, capacity=capacity(2), dry_run=True
    )

    assert swarm.log == []
    assert report.dispatched == ()
    # ... and still answers "what would this cycle do".
    assert report.plan.numbers == (4, 5)


def test_nothing_is_dispatched_when_the_ledger_has_nothing_ready():
    swarm = FakeSwarm()

    report = _dispatch_(swarm, swarm, ledger(entry(4, state=LANDED)), BASE_COMMIT, capacity=capacity(2))

    assert swarm.log == []
    assert report.dispatched == ()


# --------------------------------------------------------------------------
# What a cycle costs
# --------------------------------------------------------------------------


def test_a_cycle_that_dispatched_does_not_ask_the_judge():
    swarm = FakeSwarm()

    report = _dispatch_(swarm, swarm, ledger(entry(7)), BASE_COMMIT, capacity=capacity(1))

    # The judge call is a dense-model call, which under OLLAMA_MAX_LOADED_MODELS=1
    # evicts the worker model: ~6.7 s of swap, then another to load it back.
    # Paying that per cycle to be told what the arithmetic already knew is how a
    # run spends more time swapping than working.
    assert report.needs_judgement is False


def test_a_cycle_at_the_cap_does_not_ask_the_judge_either():
    swarm = FakeSwarm()

    report = _dispatch_(
        swarm, swarm, ledger(entry(4, state=CLAIMED), entry(5)), BASE_COMMIT, capacity=capacity(1)
    )

    # Nothing new started, but a container is running: waiting for it is not a
    # stall, and no model has anything to add about it.
    assert report.dispatched == ()
    assert report.needs_judgement is False


def test_a_cycle_that_could_do_nothing_at_all_is_worth_a_judgement():
    swarm = FakeSwarm()

    report = _dispatch_(swarm, swarm, ledger(entry(4, state=NEEDS_HUMAN)), BASE_COMMIT, capacity=capacity(2))

    # Nothing dispatched and nothing running is a stall, which is exactly what
    # the judge exists to diagnose (architecture-v2, step 5).
    assert report.needs_judgement is True


def test_an_infrastructure_failure_is_not_a_stall_to_judge():
    swarm = FakeSwarm(spawn_error=daemon_down())

    report = _dispatch_(swarm, swarm, ledger(entry(7)), BASE_COMMIT, capacity=capacity(1))

    # Docker being unreachable is a fact, not a question, and replanning around
    # it would rewrite a backlog that was never the problem.
    assert report.needs_judgement is False


def test_dispatching_reaches_for_no_model_on_any_path():
    """The deterministic short-circuit #24 keeps, enforced structurally."""
    tree = ast.parse(Path(dispatcher.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    # Every decision here is arithmetic over what the ledger already said, so a
    # cycle that only dispatches costs no model load. An import of `llm`, of a
    # node that calls one, or of a client library would quietly make the cheap
    # cycle expensive again, and the cost is invisible until a run is slow.
    assert not [name for name in imported if "llm" in name or "ollama" in name.lower()]
    assert not [name for name in imported if name.startswith("nodes")]


# --------------------------------------------------------------------------
# One image per task (#99)
# --------------------------------------------------------------------------


def test_a_task_is_spawned_in_its_own_stacks_image():
    swarm = FakeSwarm()
    node = entry(4)
    node = dataclasses.replace(node, stack="node")

    _dispatch_(swarm, swarm, ledger(node, entry(5)), BASE_COMMIT, capacity=capacity(2))

    assert swarm.images == ["apiary-worker-node", "apiary-worker"]


def test_a_stack_with_no_image_is_refused_before_it_is_claimed():
    """Never a mid-run `docker create` failure. A claim spent on a task this
    host cannot run is a claim #35 has to sweep, and the message a create
    failure produces names a tag rather than a thing to do about it."""
    swarm = FakeSwarm()
    unrunnable = dataclasses.replace(entry(4), stack="rust")

    report = _dispatch_(swarm, swarm, ledger(unrunnable), BASE_COMMIT, capacity=capacity(1))

    assert swarm.log == []  # not claimed, not spawned
    assert report.failed[0].claimed is False
    assert "rust" in report.failed[0].reason


def test_one_unrunnable_stack_does_not_stop_the_others():
    """The #94 property, restated for the failure #99 makes possible."""
    swarm = FakeSwarm()
    unrunnable = dataclasses.replace(entry(4), stack="rust")

    report = _dispatch_(
        swarm, swarm, ledger(unrunnable, entry(5)), BASE_COMMIT, capacity=capacity(2)
    )

    assert swarm.spawned == [5]
    assert [item.number for item in report.dispatched] == [5]


def test_an_image_that_was_never_built_says_which_command_builds_it():
    """The one failure whose fix is a command rather than an investigation: the
    orchestrator can neither build nor pull, so a human has to."""
    swarm = FakeSwarm(
        spawn_errors={
            4: DockerError(
                ["docker", "create"], 125, "Unable to find image 'apiary-worker-node' locally"
            )
        }
    )
    node = dataclasses.replace(entry(4), stack="node")

    report = _dispatch_(swarm, swarm, ledger(node), BASE_COMMIT, capacity=capacity(1))

    assert "docker build -f Dockerfile.worker.node" in report.failed[0].reason
    # And it is still a per-issue defer, not a halted cycle.
    assert report.failed[0].fatal is False


def test_an_override_reaches_the_spawn():
    swarm = FakeSwarm()
    node = dataclasses.replace(entry(4), stack="node")

    _dispatch_(
        swarm,
        swarm,
        ledger(node),
        BASE_COMMIT,
        capacity=capacity(1),
        images=StackImages.from_env({STACK_IMAGES_ENV: "node=my-node:dev"}),
    )

    assert swarm.images == ["my-node:dev"]
