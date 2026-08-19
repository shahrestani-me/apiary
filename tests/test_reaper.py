"""Tests for the orphan reaper.

The reaper is the only code in the swarm that destroys a container it did not
create, so these tests are mostly about restraint. Three properties:

**It removes exactly the orphans.** A container of a live run, a container of a
run the caller declared live, a container wearing `apiary.run` with a value this
system could not have minted - all survive. And a container is never selected
for its image or its name, only for the run label, because "starts with apiary-"
also describes things the human built by hand.

**It never lists unfiltered.** `Daemon` below refuses a `docker ps` that carries
no label filter rather than answering it, so a regression to a bare listing
fails every test in the file instead of one. That matters more here than
anywhere else: the development machine this runs on has other people's
containers on it, and the caller of a listing here is a function that removes
what it is handed.

**The logs outlive the container.** An orphan is a container nobody watched
finish - no exit code was read and no output was, and the reason the run went
wrong is in there. Capture happens before removal and the capture is handed on.

Most of it is hermetic against `Daemon`, a stand-in that really parses the
`--filter` arguments, so the selection logic is under test rather than mocked.
The handful that need a daemon carry the `docker` marker (tests/conftest.py) and
are deselected by default; every one of them is scoped to run ids it minted
itself, so it cannot see - let alone remove - anything else on the machine.
"""

from __future__ import annotations

import signal
from typing import Any, Iterator, Sequence

import pytest

from swarm.containers.manager import (
    ContainerError,
    ContainerManager,
    DockerCLI,
    Handle,
    Limits,
    find_containers,
)
from swarm.containers.reaper import REAPED_SIGNALS, Reaper, Sweep
from swarm.github.refs import task_ref
from swarm.run import RUN_LABEL, Run

from fixtures.docker import Container, Daemon

REPO = "shahrestani-me/apiary"
OBJECTIVE = "reap the containers a crashed orchestrator left holding clones"
BASE_COMMIT = "9f2c1ab3d4e5f60718293a4b5c6d7e8f90a1b2c3"

#: A run that ended when its orchestrator did, and whose id will never be
#: minted again - which is the entire signal the reaper works from.
DEAD_RUN = "apiary-20260814-142530-k3f9qz"
OTHER_DEAD_RUN = "apiary-20260814-150000-mn4p7t"
SIBLING_RUN = "apiary-20260814-151500-qq8w2r"


def make_run(run_id: str | None = None) -> Run:
    return Run.start(REPO, OBJECTIVE, run_id=run_id)


def make_reaper(daemon: Daemon, **kwargs: Any) -> Reaper:
    return Reaper(docker=DockerCLI(runner=daemon), **kwargs)


@pytest.fixture(autouse=True)
def restore_signal_handlers() -> Iterator[None]:
    """No test may leave this process's `SIGINT` pointing at a dead reaper."""
    before = {signum: signal.getsignal(signum) for signum in REAPED_SIGNALS}
    yield
    for signum, handler in before.items():
        signal.signal(signum, handler)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_a_container_whose_run_is_gone_is_reaped():
    run = make_run()
    daemon = Daemon([Container("aaa", run_id=DEAD_RUN, issue=7, logs="clone failed\n")])

    swept = make_reaper(daemon, run=run).startup()

    # The whole ticket: ids are never reused, so a label naming a run this
    # process is not is a label naming a process that is gone.
    assert [handle.id for handle in swept.reaped] == ["aaa"]
    assert daemon.ids == []
    assert swept.ok


def test_an_exited_container_is_reaped_and_its_logs_come_with_it():
    """The one caller that must keep seeing what has already stopped (#187).

    `find_containers` grew a `running=` filter so a caller asking about
    liveness stops reading a corpse as a live worker. This sweep must never use
    it: an exited container still holds the clone it made and the only account
    of why the run went wrong, and "nobody read its exit code" is the
    definition of the orphan this module exists for.
    """
    run = make_run()
    daemon = Daemon(
        [Container("stopped", run_id=DEAD_RUN, issue=7, state="exited", logs="clone failed\n")]
    )
    written: list[Handle] = []

    swept = make_reaper(daemon, run=run, sink=written.append).startup()

    assert [handle.id for handle in swept.reaped] == ["stopped"]
    assert daemon.ids == []
    # Carried, not selected on: the summary can say what state it removed.
    assert [handle.state for handle in swept.reaped] == ["exited"]
    assert [handle.captured for handle in written] == ["clone failed\n"]


def test_no_sweep_ever_narrows_its_listing_to_running_containers():
    """Pinned, because narrowing here is a one-word change with no failing test.

    A sweep that asked the daemon for `status=running` would leave precisely
    the containers that need reaping most, and every existing assertion here
    would still pass - the orphans in them are removed while running.
    """
    run = make_run()
    daemon = Daemon(
        [
            Container("stopped", run_id=DEAD_RUN, issue=7, state="exited"),
            Container("working", run_id=DEAD_RUN, issue=9, state="running"),
        ]
    )

    swept = make_reaper(daemon, run=run).startup()

    assert sorted(handle.id for handle in swept.reaped) == ["stopped", "working"]
    for argv in daemon.argvs_for("ps"):
        assert not [part for part in argv if part.startswith("status=")]
        assert "--all" in argv


def test_this_runs_own_containers_survive_a_startup_sweep():
    run = make_run()
    daemon = Daemon(
        [
            Container("mine", run_id=run.id, issue=7),
            Container("theirs", run_id=DEAD_RUN, issue=9),
        ]
    )

    swept = make_reaper(daemon, run=run).startup()

    # Reaping its own live workers is the failure `run.py` mints a fresh id on
    # every invocation to make impossible; this is that guarantee, asserted.
    assert daemon.ids == ["mine"]
    assert [handle.id for handle in swept.spared] == ["mine"]


def test_shutdown_reaps_this_runs_containers_too():
    run = make_run()
    daemon = Daemon([Container("mine", run_id=run.id, issue=7)])

    swept = make_reaper(daemon, run=run).shutdown()

    # The run is ending, so its own containers are exactly as orphaned as
    # anybody else's.
    assert [handle.id for handle in swept.reaped] == ["mine"]
    assert daemon.ids == []


def test_a_run_the_caller_declared_live_is_never_touched():
    run = make_run()
    daemon = Daemon(
        [
            Container("sibling", run_id=SIBLING_RUN, issue=3),
            Container("dead", run_id=DEAD_RUN, issue=4),
        ]
    )

    swept = make_reaper(daemon, run=run, live_run_ids={SIBLING_RUN}).shutdown()

    # "One orchestrator per host" is the default, not a law. A scheduler driving
    # several names them here, and #35's "do not silently steal another
    # process's claim" holds for containers as well as labels.
    assert daemon.ids == ["sibling"]
    assert [handle.id for handle in swept.reaped] == ["dead"]


def test_a_container_is_never_reaped_for_its_image_or_its_name():
    run = make_run()
    daemon = Daemon(
        [
            # Looks exactly like a worker, belongs to a live run.
            Container("live", run_id=run.id, name="apiary-worker-probe", image="apiary-worker"),
            # Wears the label with a value this system could not have minted.
            Container("foreign", run_id="Not A Run Id", name="apiary-build", image="apiary-worker"),
            # Wears the label with no value at all.
            Container("blank", run_id="", name="apiary-cache", image="apiary-worker"),
        ]
    )

    swept = make_reaper(daemon, run=run).shutdown()

    # Only `apiary.run` decides, and only when it holds an id `run.py` would
    # have produced. Everything else is a naming convention a human can adopt
    # by accident, and `docker rm` has no undo.
    assert daemon.ids == ["foreign", "blank"]
    assert [handle.id for handle in swept.reaped] == ["live"]


def test_a_container_without_the_run_label_is_never_even_listed():
    run = make_run()
    daemon = Daemon(
        [
            Container("postgres", run_id=None, name="pg", image="postgres:16"),
            Container("dead", run_id=DEAD_RUN, issue=1),
        ]
    )

    make_reaper(daemon, run=run).startup()

    assert daemon.ids == ["postgres"]
    # Filtered daemon-side, not filtered after the fact: the containers of the
    # human's afternoon are never in a list this module iterates over.
    for argv in daemon.argvs_for("ps"):
        assert f"label={RUN_LABEL}" in argv
        assert "--all" in argv


def test_orphans_previews_the_sweep_without_performing_it():
    run = make_run()
    daemon = Daemon([Container("dead", run_id=DEAD_RUN, issue=2)])

    doomed = make_reaper(daemon, run=run).orphans()

    # The decision to destroy something should be inspectable without doing it.
    assert [handle.id for handle in doomed] == ["dead"]
    assert daemon.ids == ["dead"]
    assert "rm" not in daemon.commands


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------


def test_scope_narrows_the_filter_the_daemon_itself_applies():
    daemon = Daemon(
        [
            Container("in", run_id=DEAD_RUN, issue=5),
            Container("out", run_id=OTHER_DEAD_RUN, issue=6),
        ]
    )

    swept = make_reaper(daemon, scope={DEAD_RUN}).sweep()

    # Both runs are dead. Only one was in scope, and the other was never
    # reported by the daemon at all - which is what makes a sweep on a shared
    # machine safe to run rather than merely safe to reason about.
    assert daemon.ids == ["out"]
    assert [handle.id for handle in swept.reaped] == ["in"]
    assert [f"label={RUN_LABEL}={DEAD_RUN}" in argv for argv in daemon.argvs_for("ps")] == [True]


def test_a_scope_of_several_runs_lists_each_one_separately():
    daemon = Daemon(
        [
            Container("a", run_id=DEAD_RUN, issue=1),
            Container("b", run_id=OTHER_DEAD_RUN, issue=2),
            Container("c", run_id=SIBLING_RUN, issue=3),
        ]
    )

    swept = make_reaper(daemon, scope=[DEAD_RUN, OTHER_DEAD_RUN]).sweep()

    assert sorted(handle.id for handle in swept.reaped) == ["a", "b"]
    assert daemon.ids == ["c"]


def test_scope_and_a_live_declaration_compose():
    daemon = Daemon(
        [
            Container("dead", run_id=DEAD_RUN, issue=1),
            Container("sibling", run_id=SIBLING_RUN, issue=2),
        ]
    )

    reaper = make_reaper(daemon, scope=[DEAD_RUN, SIBLING_RUN], live_run_ids={SIBLING_RUN})

    assert [handle.id for handle in reaper.sweep().reaped] == ["dead"]
    assert daemon.ids == ["sibling"]


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------


def test_the_logs_are_captured_before_the_container_is_removed():
    daemon = Daemon([Container("dead", run_id=DEAD_RUN, issue=7, logs="ollama refused\n")])

    swept = make_reaper(daemon, scope={DEAD_RUN}).sweep()

    assert daemon.commands == ["ps", "logs", "rm"]
    # An orphan is a container nobody watched finish: no exit code was read and
    # no output was, so this string is the only account of what went wrong.
    assert swept.reaped[0].captured == "ollama refused\n"


def test_the_sink_receives_a_container_the_orchestrator_never_saw_finish():
    received: list[Handle] = []
    daemon = Daemon([Container("dead", run_id=DEAD_RUN, issue=7, logs="still cloning\n")])

    make_reaper(daemon, scope={DEAD_RUN}, sink=received.append).sweep()

    # #29 writes these to the run's artifact directory, which is where a human
    # reads them the morning after.
    assert [handle.captured for handle in received] == ["still cloning\n"]
    assert received[0].removed is True


def test_a_sink_that_fails_costs_an_artifact_not_a_leaked_container():
    def explode(handle: Handle) -> None:
        raise OSError("artifacts directory is read-only")

    daemon = Daemon([Container("dead", run_id=DEAD_RUN, issue=7)])

    swept = make_reaper(daemon, scope={DEAD_RUN}, sink=explode).sweep()

    # The handoff happens after removal precisely so this ordering holds: the
    # container is gone, and the writer's problem is reported rather than
    # rolled back into a leak.
    assert daemon.ids == []
    assert [handle.id for handle in swept.reaped] == ["dead"]
    assert not swept.ok
    assert "read-only" in swept.summary()


# --------------------------------------------------------------------------
# Failure
# --------------------------------------------------------------------------


def test_one_unremovable_container_does_not_abandon_the_others():
    daemon = Daemon(
        [
            Container("stuck", run_id=DEAD_RUN, issue=1),
            Container("removable", run_id=DEAD_RUN, issue=2),
        ],
        rm_fails={"stuck": "Error response from daemon: permission denied\n"},
    )

    swept = make_reaper(daemon, scope={DEAD_RUN}).sweep()

    # Raising from the middle of the loop would leave the remaining orphans up
    # *and* leave the caller with less information than it has here.
    assert daemon.ids == ["stuck"]
    assert [handle.id for handle in swept.reaped] == ["removable"]
    assert [handle.id for handle, _ in swept.failures] == ["stuck"]
    assert not swept.ok


def test_a_container_that_vanished_mid_sweep_is_not_a_failure():
    daemon = Daemon(
        [Container("ghost", run_id=DEAD_RUN, issue=1)],
        rm_fails={"ghost": "Error: No such container: ghost\n"},
    )

    swept = make_reaper(daemon, scope={DEAD_RUN}).sweep()

    # A human running `docker rm` at the same moment wanted the same end state.
    assert swept.ok
    assert [handle.id for handle in swept.reaped] == ["ghost"]


def test_a_sweep_that_finds_nothing_says_so():
    daemon = Daemon([])

    swept = make_reaper(daemon, run=make_run()).startup()

    assert swept == Sweep()
    assert swept.ok
    assert "reaped 0 container(s)" in swept.summary()


def test_the_summary_counts_dead_runs_not_just_containers():
    daemon = Daemon(
        [
            Container("a", run_id=DEAD_RUN, issue=1),
            Container("b", run_id=DEAD_RUN, issue=2),
            Container("c", run_id=OTHER_DEAD_RUN, issue=3),
            Container("mine", run_id=SIBLING_RUN, issue=4),
        ]
    )

    swept = make_reaper(daemon, live_run_ids={SIBLING_RUN}).sweep()

    assert swept.run_ids == (DEAD_RUN, OTHER_DEAD_RUN)
    assert swept.summary() == "reaped 3 container(s) from 2 dead run(s), spared 1"


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


def test_ctrl_c_reaps_before_the_interrupt_propagates():
    run = make_run()
    daemon = Daemon([Container("mine", run_id=run.id, issue=7)])
    reaper = make_reaper(daemon, run=run)
    reaper.install()

    with pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGINT)

    # Without this the common case - a human stopping a run - is the case that
    # leaks: `KeyboardInterrupt` inside a `docker wait` unwinds past every
    # `finally` that had not started yet.
    assert daemon.ids == []
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_sigterm_reaps_and_then_chains_to_the_handler_it_displaced():
    run = make_run()
    seen: list[int] = []
    signal.signal(signal.SIGTERM, lambda signum, frame: seen.append(signum))
    daemon = Daemon([Container("mine", run_id=run.id, issue=7)])
    reaper = make_reaper(daemon, run=run)
    reaper.install()

    signal.raise_signal(signal.SIGTERM)

    # Chaining is not politeness: an orchestrator that swallowed `SIGTERM` would
    # ignore `docker stop` until the daemon escalated to `SIGKILL`, which is the
    # one signal this module cannot clean up after.
    assert seen == [signal.SIGTERM]
    assert daemon.ids == []


def test_a_default_disposition_is_re_raised_after_the_handlers_are_restored(monkeypatch):
    raised: list[int] = []
    monkeypatch.setattr(signal, "raise_signal", lambda signum: raised.append(signum))
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    run = make_run()
    daemon = Daemon([Container("mine", run_id=run.id, issue=7)])
    reaper = make_reaper(daemon, run=run)
    reaper.install()

    reaper._on_signal(signal.SIGTERM, None)

    # Exiting 0 here would claim the run finished. The handlers are back first,
    # so the re-raise gets the plain default and not this reaper again.
    assert raised == [signal.SIGTERM]
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    assert daemon.ids == []


def test_an_ignored_signal_stays_ignored():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    run = make_run()
    daemon = Daemon([Container("mine", run_id=run.id, issue=7)])
    reaper = make_reaper(daemon, run=run)
    reaper.install()

    signal.raise_signal(signal.SIGTERM)

    # Still swept - the process was told to stop even if it chose not to die.
    assert daemon.ids == []
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN


def test_a_second_ctrl_c_during_a_slow_sweep_stops_waiting():
    run = make_run()
    reaper: Reaper

    def impatient(argv: Sequence[str], *, timeout_s: float | None, merge: bool):
        if argv[1] == "rm":
            # The human pressing Ctrl-C again while the first sweep is still
            # removing containers.
            reaper._on_signal(signal.SIGINT, None)
        return daemon(argv, timeout_s=timeout_s, merge=merge)

    daemon = Daemon([Container("mine", run_id=run.id, issue=7)])
    reaper = Reaper(docker=DockerCLI(runner=impatient), run=run)
    reaper.install()

    with pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGINT)

    # The second signal takes the original disposition immediately instead of
    # queueing a second sweep behind the first.
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_guard_sweeps_on_entry_and_on_exit_and_gives_the_handlers_back():
    run = make_run()
    daemon = Daemon([Container("stale", run_id=DEAD_RUN, issue=1)])
    reaper = make_reaper(daemon, run=run)
    before = signal.getsignal(signal.SIGINT)

    with reaper.guard():
        assert daemon.ids == []  # the entry sweep took the previous run's leftovers
        assert signal.getsignal(signal.SIGINT) is not before
        daemon.containers.append(Container("mine", run_id=run.id, issue=7))

    assert daemon.ids == []
    assert signal.getsignal(signal.SIGINT) is before


def test_guard_reaps_and_restores_even_when_the_body_raises():
    run = make_run()
    daemon = Daemon([])
    reaper = make_reaper(daemon, run=run)
    before = signal.getsignal(signal.SIGTERM)

    with pytest.raises(RuntimeError):
        with reaper.guard():
            daemon.containers.append(Container("mine", run_id=run.id, issue=7))
            raise RuntimeError("the planner exploded")

    # An orchestrator that dies of its own exception leaks exactly as much as
    # one that is killed.
    assert daemon.ids == []
    assert signal.getsignal(signal.SIGTERM) is before


# --------------------------------------------------------------------------
# Against a real daemon
# --------------------------------------------------------------------------


CANDIDATE_IMAGES = ("apiary-worker", "busybox", "alpine", "python:3.12-slim")


@pytest.fixture(scope="module")
def trivial_image() -> str:
    """A locally present image with a shell. Nothing is pulled."""
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


def make_manager(image: str, run: Run) -> ContainerManager:
    return ContainerManager(
        run=run,
        image=image,
        env={},
        limits=Limits(cpus=1.0, memory="256m", pids=64),
        timeout_s=60,
    )


def probe(manager: ContainerManager, script: str, issue: int) -> Handle:
    return manager.spawn(
        task_ref(issue), BASE_COMMIT, issue=issue, entrypoint="/bin/sh", command=["-c", script]
    )


@pytest.fixture()
def two_runs(trivial_image: str) -> Iterator[tuple[ContainerManager, ContainerManager]]:
    """Two managers under two freshly minted run ids, cleaned up either way.

    Freshly minted, and every listing below is scoped to these two ids. This
    machine runs other people's containers - including, quite possibly, another
    apiary run - and none of them are in reach of anything in this file.
    """
    doomed = make_manager(trivial_image, make_run())
    sibling = make_manager(trivial_image, make_run())
    try:
        yield doomed, sibling
    finally:
        for manager in (doomed, sibling):
            for handle in manager.find():
                manager.dispose(handle)


@pytest.mark.docker
def test_the_containers_of_a_dead_run_are_reaped_and_a_live_ones_are_not(two_runs):
    """#20's acceptance criterion: nothing of the dead run outlives the sweep."""
    doomed, sibling = two_runs
    probe(doomed, 'echo "cloning"; sleep 300', issue=7)
    probe(doomed, 'echo "verifying"; sleep 300', issue=8)
    probe(sibling, 'echo "still working"; sleep 300', issue=9)

    reaper = Reaper(
        # Scoped to the two ids this test minted, and to nothing else on this
        # machine. The production default sweeps every apiary run; a test may
        # not, because a real orchestrator may be running beside it.
        scope={doomed.run.id, sibling.run.id},
        live_run_ids={sibling.run.id},
    )
    swept = reaper.sweep()

    assert swept.ok, swept.summary()
    assert sorted(handle.issue for handle in swept.reaped) == [7, 8]
    assert doomed.find() == []
    # The live run kept every one of its workers, which is the half of this that
    # a `docker system prune` gets wrong.
    assert [handle.issue for handle in sibling.find()] == [9]


@pytest.mark.docker
def test_the_logs_of_a_container_nobody_watched_finish_survive_it(two_runs):
    doomed, _ = two_runs
    probe(doomed, 'echo "cloned at 02:14"; sleep 300', issue=11)

    received: list[Handle] = []
    swept = Reaper(scope={doomed.run.id}, sink=received.append).sweep()

    # It was still running: no exit code was ever read and nothing else in the
    # system holds an account of what it did.
    assert "cloned at 02:14" in swept.reaped[0].captured
    assert [handle.captured for handle in received] == [swept.reaped[0].captured]
    assert doomed.find() == []


@pytest.mark.docker
def test_a_container_that_already_exited_is_still_found_and_still_disposed(two_runs):
    """Against a real daemon, the state #187 added must not narrow this sweep.

    An exited orphan is the common one - the orchestrator died, so nothing ever
    read the exit code - and it holds its clone and its logs until something
    removes it.
    """
    doomed, _ = two_runs
    handle = probe(doomed, 'echo "clone failed at 02:14"; exit 1', issue=12)
    assert doomed.wait(handle) == 1

    listed = doomed.find()
    assert [h.state for h in listed] == ["exited"]

    swept = Reaper(scope={doomed.run.id}).sweep()

    assert swept.ok, swept.summary()
    assert [h.id for h in swept.reaped] == [handle.id]
    assert "clone failed at 02:14" in swept.reaped[0].captured
    assert doomed.find() == []


@pytest.mark.docker
def test_ctrl_c_leaves_no_container_behind(two_runs):
    """The done-when, with the signal a human actually sends."""
    doomed, _ = two_runs
    probe(doomed, "sleep 300", issue=12)
    reaper = Reaper(run=doomed.run, scope={doomed.run.id})
    reaper.install()

    try:
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
    finally:
        reaper.restore()

    assert doomed.find() == []


@pytest.mark.docker
def test_a_second_sweep_of_the_same_run_is_quiet(two_runs):
    doomed, _ = two_runs
    probe(doomed, "sleep 300", issue=13)
    reaper = Reaper(scope={doomed.run.id})

    assert len(reaper.sweep().reaped) == 1
    again = reaper.sweep()

    # Two things race to remove a container and "already gone" is the outcome
    # both of them wanted.
    assert again == Sweep()
    assert again.ok


@pytest.mark.docker
def test_a_scoped_sweep_reads_back_only_the_run_it_was_given(two_runs):
    doomed, sibling = two_runs
    probe(sibling, "sleep 300", issue=14)

    assert Reaper(scope={doomed.run.id}).orphans() == []
    # Belt and braces on the guarantee the whole file rests on: a listing for
    # one run id cannot see another run's container, let alone the machine's.
    assert find_containers(DockerCLI(), run_id=doomed.run.id) == []
    assert [handle.issue for handle in find_containers(DockerCLI(), run_id=sibling.run.id)] == [14]
