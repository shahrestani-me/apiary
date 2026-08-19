"""Tests for the container lifecycle manager.

Three properties carry this file, and all three are about what is left behind
rather than about what runs.

**Nothing outlives the run.** A container that failed to start, a container
that was disposed twice, a container someone else removed first - every one of
those paths ends with no container and no exception the caller has to know
about. Disposal is the operation that must never be conditional.

**The logs exist after the container does not.** `dispose` captures before it
removes and hands the capture back, so #29 has something to write and a human
has something to read about a run that went wrong at 02:00.

**No credential reaches a captured string.** Not through the log stream, and
not through the message of a failed command either - `docker create` carries
the worker's environment on its argv, so a create that fails is exactly as
dangerous as a log that prints a token, and both go to the same artifact file.

Most of it is hermetic: `ScriptedRunner` answers where a `docker` subprocess
would, so the real `DockerCLI` - redaction, error interpretation, timeout
handling - is under test rather than mocked out. The handful of tests that need
a daemon carry the `docker` marker (tests/conftest.py) and are deselected by
default; they filter every listing to the run id they minted, because this
machine runs other people's containers.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest

from swarm.containers.manager import (
    DEFAULT_STACK_IMAGES,
    IMAGE_ENV,
    ISSUE_LABEL,
    MAX_LOG_CHARS,
    PLACEHOLDER,
    ContainerError,
    ContainerManager,
    ContainerTimeout,
    DockerCLI,
    DockerError,
    Handle,
    Limits,
    Redactor,
    STACK_IMAGES_ENV,
    StackImages,
    WORKER_IMAGE,
    build_hint,
    dispose_container,
    find_containers,
    missing_image,
)
from swarm.containers import manager as manager_module
from swarm.github.refs import task_ref
from swarm.run import RUN_LABEL, Run

#: The repository root, for the one test that asserts a generated command names
#: a file that exists.
REPO_ROOT = Path(__file__).resolve().parent.parent

REPO = "shahrestani-me/apiary"
OBJECTIVE = "add retry with exponential backoff to the http client"
BASE_COMMIT = "9f2c1ab3d4e5f60718293a4b5c6d7e8f90a1b2c3"

#: Deliberately *not* GitHub-shaped. A token matching `ghp_...` would be
#: redacted by pattern even if nothing had registered it, and then these tests
#: would pass without the enrolment path working at all.
TOKEN = "s3cr3t-push-credential-9f2c1ab3"

CONTAINER_ID = "c0ffee" + "0" * 58


# --------------------------------------------------------------------------
# The daemon double
# --------------------------------------------------------------------------


@dataclass
class Reply:
    """What the `docker` binary would have written, and how it would have exited."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


@dataclass
class ScriptedRunner:
    """A `Runner` keyed by docker subcommand. Records every argv it was given.

    A reply may also be an exception instance, which is raised instead - that
    is how `subprocess.TimeoutExpired` and `FileNotFoundError` get exercised
    against the real `DockerCLI` translation rather than against a double of it.
    """

    replies: dict[str, Any] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)

    #: Enough of a happy path that a test only scripts what it is about.
    DEFAULTS = {
        "create": Reply(stdout=CONTAINER_ID + "\n"),
        "start": Reply(stdout=CONTAINER_ID + "\n"),
        "wait": Reply(stdout="0\n"),
        "logs": Reply(stdout=""),
        "rm": Reply(stdout=CONTAINER_ID + "\n"),
        "stop": Reply(stdout=CONTAINER_ID + "\n"),
        "ps": Reply(stdout=""),
    }

    def __call__(
        self, argv: Sequence[str], *, timeout_s: float | None, merge: bool
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        reply = self.replies.get(argv[1], self.DEFAULTS.get(argv[1], Reply()))
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            reply = reply(list(argv))
        stdout = reply.stdout + reply.stderr if merge else reply.stdout
        stderr = "" if merge else reply.stderr
        return subprocess.CompletedProcess(list(argv), reply.returncode, stdout, stderr)

    # --- what the assertions read --------------------------------------

    @property
    def commands(self) -> list[str]:
        """The docker subcommands, in the order they were issued."""
        return [call[1] for call in self.calls]

    def argvs_for(self, subcommand: str) -> list[list[str]]:
        return [call for call in self.calls if call[1] == subcommand]

    def argv_for(self, subcommand: str) -> list[str]:
        found = self.argvs_for(subcommand)
        if not found:
            raise AssertionError(f"no {subcommand!r} command was issued; got {self.commands}")
        return found[0]

    def flag(self, subcommand: str, flag: str) -> str:
        """The value following `flag`, e.g. `flag("create", "--memory")`."""
        argv = self.argv_for(subcommand)
        return argv[argv.index(flag) + 1]

    def flags(self, subcommand: str, flag: str) -> list[str]:
        argv = self.argv_for(subcommand)
        return [argv[i + 1] for i, part in enumerate(argv) if part == flag]


def make_run() -> Run:
    return Run.start(REPO, OBJECTIVE)


def make_manager(**kwargs: Any) -> tuple[ContainerManager, ScriptedRunner]:
    runner = kwargs.pop("runner", None) or ScriptedRunner()
    kwargs.setdefault("env", {"GITHUB_TOKEN": TOKEN})
    manager = ContainerManager(run=make_run(), runner=runner, **kwargs)
    return manager, runner


def spawned(manager: ContainerManager, issue: int = 7) -> Handle:
    return manager.spawn(task_ref(issue), BASE_COMMIT, issue=issue)


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------


def test_a_registered_secret_never_survives_a_capture():
    redact = Redactor([TOKEN])

    assert redact(f"remote: rejected, token {TOKEN} lacks push") == (
        f"remote: rejected, token {PLACEHOLDER} lacks push"
    )


def test_a_secret_is_redacted_in_its_percent_encoded_form_too():
    secret = "p@ss/word-with-symbols"
    redact = Redactor([secret])

    # This is the form a credential takes inside a URL, which is the form git
    # prints back when a push fails - the one code path most likely to leak it.
    assert secret not in redact(f"https://user:p%40ss%2Fword-with-symbols@github.com/o/r")


def test_credentials_embedded_in_a_remote_url_are_redacted_by_shape():
    # Nothing registered this one: it is the URL shape itself that is a secret.
    redacted = Redactor()("fatal: could not read from https://x-access-token:abc123def456@github.com/o/r.git")

    assert "abc123def456" not in redacted
    # The line still has to be diagnostic afterwards, or redaction has simply
    # destroyed the artifact instead of sanitising it.
    assert "github.com/o/r.git" in redacted


def test_a_token_this_process_never_saw_is_still_redacted():
    minted = "ghp_" + "A1b2C3d4E5f6G7h8I9j0"

    assert minted not in Redactor()(f"::add-mask::{minted}")


def test_a_short_environment_value_is_not_treated_as_a_secret():
    redact = Redactor()
    redact.add_env({"API_KEY": "1", "GITHUB_TOKEN": TOKEN})

    # Registering "1" would turn every number in every log into asterisks and
    # destroy the artifact this module exists to preserve.
    assert redact("attempt 1 of 3") == "attempt 1 of 3"
    assert TOKEN not in redact(TOKEN)


def test_the_secrets_a_container_is_handed_are_enrolled_without_being_named():
    # The manager was told nothing about redaction; it was told to pass
    # GITHUB_TOKEN to the container. Those are the same act.
    manager, runner = make_manager()
    runner.replies["logs"] = Reply(stderr=f"fatal: authentication failed for token {TOKEN}\n")

    logs = manager.logs(spawned(manager))

    assert TOKEN not in logs
    assert "authentication failed" in logs


def test_the_message_of_a_failed_command_is_redacted_too():
    """The argv of a `create` carries the worker's whole environment."""
    manager, runner = make_manager()
    runner.replies["create"] = Reply(stderr=f"invalid env GITHUB_TOKEN={TOKEN}\n", returncode=125)

    with pytest.raises(DockerError) as excinfo:
        spawned(manager)

    assert TOKEN not in str(excinfo.value)
    assert PLACEHOLDER in str(excinfo.value)


# --------------------------------------------------------------------------
# Spawning
# --------------------------------------------------------------------------


def test_a_worker_container_carries_the_run_and_issue_labels():
    manager, runner = make_manager()

    handle = manager.spawn(task_ref(11), BASE_COMMIT, issue=11)

    # #20 reaps on the run label and #29 names an artifacts directory after it;
    # the issue label is how either of them says which task the logs belong to.
    assert set(runner.flags("create", "--label")) == {
        f"{RUN_LABEL}={manager.run.id}",
        f"{ISSUE_LABEL}=11",
    }
    assert handle.id == CONTAINER_ID
    assert handle.issue == 11
    assert handle.run_id == manager.run.id


def test_a_worker_container_is_created_with_explicit_resource_limits():
    manager, runner = make_manager(limits=Limits(cpus=1.5, memory="2g", pids=256))

    spawned(manager)

    # Unlimited is not a default anyone chose: an LLM that writes a fork bomb
    # makes the host unusable rather than failing one task.
    assert runner.flag("create", "--cpus") == "1.5"
    assert runner.flag("create", "--memory") == "2g"
    assert runner.flag("create", "--pids-limit") == "256"
    assert "--init" in runner.argv_for("create")


def test_the_worker_gets_the_three_arguments_its_image_documents():
    manager, runner = make_manager()

    spawned(manager, issue=7)

    assert runner.flag("create", "--repo") == f"https://github.com/{REPO}.git"
    assert runner.flag("create", "--issue") == "7"
    assert runner.flag("create", "--base-commit") == BASE_COMMIT


def test_the_clone_url_carries_no_credential():
    manager, runner = make_manager()

    spawned(manager)

    # The token travels as an environment variable (#28). In the URL it would
    # land in `docker inspect`'s Args and in every git error the worker prints.
    assert TOKEN not in runner.flag("create", "--repo")
    assert f"GITHUB_TOKEN={TOKEN}" in runner.flags("create", "--env")


def test_a_token_already_in_the_environment_is_passed_by_name_not_by_value(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    manager, runner = make_manager(env=None)

    spawned(manager)

    # `--env GITHUB_TOKEN` makes the docker CLI read the value from this
    # process, so the token is on no command line at all - which is the state
    # the redactor exists to cope with rather than to rely on.
    assert "GITHUB_TOKEN" in runner.flags("create", "--env")
    assert TOKEN not in " ".join(runner.argv_for("create"))


def test_the_worker_tuning_knobs_cross_the_container_boundary(monkeypatch):
    """`SWARM_WORKER_CTX`/`SWARM_WORKER_MODEL` are read by `Settings` INSIDE
    the container, so an operator's export that stopped at the boundary would
    tune nothing while the docs said otherwise - the observed shape of that
    lie is a 16K truncation on a host tuned past it."""
    assert "SWARM_WORKER_CTX" in manager_module.INHERITED_ENV
    assert "SWARM_WORKER_MODEL" in manager_module.INHERITED_ENV

    monkeypatch.setenv("SWARM_WORKER_CTX", "32768")
    monkeypatch.setenv("SWARM_WORKER_MODEL", "gemma4:26b")
    manager, runner = make_manager(env=None)

    spawned(manager)

    # Bare `--env NAME`: the values are already this process's own, so the CLI
    # reads them from here - same form the token takes, harmless for these
    # since neither is a secret.
    flags = runner.flags("create", "--env")
    assert "SWARM_WORKER_CTX" in flags
    assert "SWARM_WORKER_MODEL" in flags


def test_the_worker_reaches_the_hosts_ollama_rather_than_its_own():
    manager, runner = make_manager()

    spawned(manager)

    # architecture-v2, "Three constraints": inference stays on the host, and a
    # Linux host does not resolve this name without being told to.
    assert runner.flag("create", "--add-host") == "host.docker.internal:host-gateway"


def test_two_containers_for_one_issue_do_not_collide_on_a_name():
    manager, runner = make_manager()

    spawned(manager)
    spawned(manager)

    names = [argv[argv.index("--name") + 1] for argv in runner.argvs_for("create")]
    # A retry of one issue inside one run would otherwise collide with the
    # corpse of the first attempt if that removal had failed.
    assert len(set(names)) == 2
    assert all(name.startswith(f"apiary-{manager.run.id}-issue-7-") for name in names)


def test_a_container_that_cannot_be_started_is_removed_rather_than_leaked():
    manager, runner = make_manager()
    runner.replies["start"] = Reply(stderr="error: cgroup limits are unavailable\n", returncode=1)

    with pytest.raises(DockerError):
        spawned(manager)

    # Created-but-never-started is still a container holding a name and a
    # writable layer. Creating and starting are two commands precisely so this
    # path has a handle to dispose.
    assert runner.commands == ["create", "start", "logs", "rm"]
    assert runner.argv_for("rm")[-1] == CONTAINER_ID


def test_a_create_that_returns_no_container_id_fails_loudly():
    manager, runner = make_manager()
    runner.replies["create"] = Reply(stdout="\n")

    with pytest.raises(ContainerError):
        spawned(manager)


def test_a_missing_docker_binary_says_which_daemon_it_was_looking_for():
    manager, runner = make_manager()
    runner.replies["create"] = FileNotFoundError(2, "No such file or directory", "docker")

    with pytest.raises(ContainerError) as excinfo:
        spawned(manager)

    assert "DOCKER_HOST" in str(excinfo.value)


# --------------------------------------------------------------------------
# Waiting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("code", [0, 1, 2])
def test_wait_reports_the_exit_code_without_interpreting_it(code: int):
    manager, runner = make_manager()
    runner.replies["wait"] = Reply(stdout=f"{code}\n")

    # 0 = PR open, 1 = task failed, 2 = infrastructure error. Which label that
    # becomes is the reconciler's decision (#22), not this module's.
    assert manager.wait(spawned(manager)) == code


def test_a_worker_that_never_finishes_is_stopped_but_not_removed():
    manager, runner = make_manager(timeout_s=0.01)
    runner.replies["wait"] = subprocess.TimeoutExpired(cmd="docker wait", timeout=0.01)

    handle = spawned(manager)
    with pytest.raises(ContainerTimeout):
        manager.wait(handle)

    # Stopped, so it stops burning an inference slot; kept, so the logs of
    # whatever it was stuck on are still readable.
    assert "stop" in runner.commands
    assert "rm" not in runner.commands


def test_wait_refuses_to_invent_an_exit_code():
    manager, runner = make_manager()
    runner.replies["wait"] = Reply(stdout="")

    # Returning 0 here would mark a task done on the strength of a parse
    # failure, and 1 would burn an attempt for the same reason.
    with pytest.raises(ContainerError):
        manager.wait(spawned(manager))


# --------------------------------------------------------------------------
# Logs and disposal
# --------------------------------------------------------------------------


def test_the_logs_are_captured_before_the_container_is_removed():
    manager, runner = make_manager()
    runner.replies["logs"] = Reply(stdout="cloning...\n", stderr="verify failed\n")

    captured = manager.dispose(spawned(manager))

    assert runner.commands == ["create", "start", "logs", "rm"]
    # One stream, in order: a traceback on stderr is not diagnostic without the
    # line printed just before it on stdout.
    assert captured == "cloning...\nverify failed\n"


def test_a_disposed_container_can_still_be_read_from():
    manager, runner = make_manager()
    runner.replies["logs"] = Reply(stdout="the last thing it said\n")

    handle = spawned(manager)
    manager.dispose(handle)
    before = list(runner.commands)

    # This is the reason capture happens at all: after removal the handle is
    # the only evidence left, and #29 reads it long after the daemon forgot.
    assert manager.logs(handle) == "the last thing it said\n"
    assert runner.commands == before


def test_disposal_is_idempotent():
    manager, runner = make_manager()

    handle = spawned(manager)
    manager.dispose(handle)
    manager.dispose(handle)

    # The dispatcher disposing what it spawned and the reaper (#20) sweeping
    # the run both want the same end state; whichever loses the race got it.
    assert runner.commands.count("rm") == 1


def test_disposing_a_container_that_is_already_gone_does_not_raise():
    manager, runner = make_manager()
    gone = Reply(stderr="Error: No such container: c0ffee\n", returncode=1)
    runner.replies["logs"] = gone
    runner.replies["rm"] = gone

    handle = spawned(manager)

    assert manager.dispose(handle) == ""
    assert handle.removed is True


def test_an_unreadable_log_does_not_prevent_the_removal():
    manager, runner = make_manager()
    runner.replies["logs"] = Reply(stderr="Cannot connect to the Docker daemon\n", returncode=1)

    handle = spawned(manager)
    captured = manager.dispose(handle)

    # Leaking a container that holds a clone is the worse failure, and it is
    # the one that makes the next run's concurrency accounting wrong.
    assert "rm" in runner.commands
    assert "could not be captured" in captured


def test_a_removal_failure_that_is_not_a_missing_container_surfaces():
    manager, runner = make_manager()
    runner.replies["rm"] = Reply(stderr="Error: permission denied\n", returncode=1)

    # "Already gone" is the outcome disposal wanted. "You may not" is not, and
    # swallowing it would leave the caller believing in a removal that never
    # happened.
    with pytest.raises(DockerError):
        manager.dispose(spawned(manager))


def test_a_runaway_log_is_bounded_at_both_ends():
    manager, runner = make_manager()
    runner.replies["logs"] = Reply(
        stdout="FIRST\n" + "spinning\n" * 200_000 + "LAST\n"
    )

    captured = manager.logs(spawned(manager))

    assert len(captured) < MAX_LOG_CHARS + 100
    # The head holds the setup that went wrong and the tail holds the failure;
    # the middle of a loop is the same line two hundred thousand times.
    assert captured.startswith("FIRST\n")
    assert captured.endswith("LAST\n")
    assert "characters elided" in captured


# --------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------


def test_a_listing_is_never_an_unfiltered_docker_ps():
    manager, runner = make_manager()

    manager.find()

    argv = runner.argv_for("ps")
    # The reaper removes what a listing returns, and this daemon is shared with
    # everything else the developer is running.
    assert f"label={RUN_LABEL}={manager.run.id}" in argv
    assert "--all" in argv


def test_a_listing_reports_the_issue_each_container_belongs_to():
    manager, runner = make_manager()
    runner.replies["ps"] = Reply(
        stdout=(
            f"{CONTAINER_ID}\tapiary-run-issue-7-ab2c\tapiary-worker\t{make_run().id}\t7\trunning\n"
            f"{'d' * 64}\tapiary-run-issue-9-zz9y\tapiary-worker\tsome-other-run\tnonsense\texited\n"
        )
    )

    handles = manager.find()

    assert [handle.issue for handle in handles] == [7, None]
    assert handles[0].name == "apiary-run-issue-7-ab2c"
    assert handles[1].run_id == "some-other-run"


def test_a_listing_can_be_narrowed_to_one_issue():
    manager, runner = make_manager()

    manager.find(ref=task_ref(7))

    assert f"label={ISSUE_LABEL}=7" in runner.argv_for("ps")


def test_containers_of_every_run_are_findable_without_a_run_object():
    """#20 sweeps runs whose orchestrator is dead and whose `Run` is gone."""
    runner = ScriptedRunner()
    runner.replies["ps"] = Reply(
        stdout=f"{CONTAINER_ID}\tapiary-x\tapiary-worker\tdead-run-1\t4\texited\n"
    )
    docker = DockerCLI(runner=runner)

    handles = find_containers(docker)

    assert [handle.run_id for handle in handles] == ["dead-run-1"]
    # Still filtered - on the label's presence rather than on its value.
    assert f"label={RUN_LABEL}" in runner.argv_for("ps")
    # And the same disposal, so a foreign container's logs are captured too.
    assert dispose_container(handles[0], docker) == ""
    assert runner.commands == ["ps", "logs", "rm"]


# --------------------------------------------------------------------------
# Liveness (#187)
#
# `docker ps --all` lists a container that has exited, so a listing without the
# state cannot tell a worker that is still working from one that finished this
# cycle - and that window is precisely where the derived resolver's `claimed`
# and `review` disagree.
# --------------------------------------------------------------------------


def test_the_listing_asks_the_daemon_for_the_container_state():
    """The format string, pinned. Without this column there is nothing to read."""
    manager, runner = make_manager()

    manager.find()

    fmt = runner.flag("ps", "--format")
    assert "{{.State}}" in fmt
    # One word from a closed set, not the human sentence that would need parsing.
    assert "{{.Status}}" not in fmt


def test_a_listing_reads_each_container_state_off_the_daemon():
    manager, runner = make_manager()
    runner.replies["ps"] = Reply(
        stdout=(
            f"{CONTAINER_ID}\tapiary-a\tapiary-worker\t{manager.run.id}\t7\trunning\n"
            f"{'d' * 64}\tapiary-b\tapiary-worker\t{manager.run.id}\t9\texited\n"
        )
    )

    handles = manager.find()

    assert [handle.state for handle in handles] == ["running", "exited"]
    # The field the resolver actually asks for, and the one `Handle` lacked.
    assert [handle.running for handle in handles] == [True, False]


def test_a_handle_that_came_from_no_listing_does_not_claim_to_be_running():
    """`spawn` infers nothing: an unread state is not a live worker.

    A container can be started and be gone in the same second - that is the
    ordinary shape of a worker - so a handle asserting liveness the daemon
    never confirmed would be wrong in exactly the window this field exists for.
    """
    manager, _ = make_manager()

    handle = spawned(manager)

    assert handle.state == ""
    assert handle.running is False


def test_a_caller_asking_for_running_containers_says_so_to_the_daemon():
    """The exited container is never listed, rather than listed and discarded."""
    manager, runner = make_manager()

    manager.find(running=True)

    assert "status=running" in runner.flags("ps", "--filter")


def test_a_listing_asks_for_every_state_by_default():
    """The reaper and `recovery.holders` both need what has already stopped."""
    manager, runner = make_manager()

    manager.find()

    assert not [f for f in runner.flags("ps", "--filter") if f.startswith("status=")]
    assert "--all" in runner.argv_for("ps")


# --------------------------------------------------------------------------
# Against a real daemon
# --------------------------------------------------------------------------


CANDIDATE_IMAGES = ("apiary-worker", "busybox", "alpine", "python:3.12-slim")


@pytest.fixture(scope="module")
def trivial_image() -> str:
    """A locally present image with a shell. Nothing is pulled.

    `apiary-worker` first, because #14's verify command builds it and it is the
    image these containers really run; the rest are the ones a developer
    machine tends to have already. Pulling would make a `docker`-marked test
    quietly need the network too.
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
def live(trivial_image: str) -> ContainerManager:
    """A manager on a real daemon, with a real token-shaped secret in its env."""
    return ContainerManager(
        run=make_run(),
        image=trivial_image,
        env={"GITHUB_TOKEN": TOKEN},
        limits=Limits(cpus=1.0, memory="256m", pids=64),
        timeout_s=60,
    )


def shell(manager: ContainerManager, script: str, issue: int = 7) -> Handle:
    """Spawn a probe container running `script`, under this run's labels."""
    return manager.spawn(
        task_ref(issue), BASE_COMMIT, issue=issue, entrypoint="/bin/sh", command=["-c", script]
    )


@pytest.mark.docker
def test_a_container_runs_is_read_disposed_and_leaves_nothing(live: ContainerManager):
    """#15's acceptance criterion, end to end."""
    handle = shell(
        live,
        'echo "cloning"; echo "remote: https://x-access-token:$GITHUB_TOKEN@github.com/o/r" >&2;'
        ' echo "token=$GITHUB_TOKEN"',
    )

    assert live.wait(handle) == 0
    logs = live.logs(handle)
    assert "cloning" in logs
    # Both leak shapes: inside a remote URL, and printed bare by a shell trace.
    assert TOKEN not in logs
    assert logs.count(PLACEHOLDER) >= 2
    # ... and the line is still diagnostic afterwards.
    assert "github.com/o/r" in logs

    assert live.dispose(handle) == logs
    # `docker ps -a`, filtered to this run and nothing else on this machine.
    assert live.find() == []


@pytest.mark.docker
def test_the_logs_of_a_failed_container_survive_its_removal(live: ContainerManager):
    handle = shell(live, 'echo "verify failed" >&2; exit 3')

    assert live.wait(handle) == 3
    captured = live.dispose(handle)

    # The whole reason capture precedes removal: this is the diagnosis, and the
    # container that held it no longer exists.
    assert "verify failed" in captured
    assert live.logs(handle) == captured
    assert live.find() == []


@pytest.mark.docker
def test_a_still_running_container_is_disposable_and_disposal_repeats(live: ContainerManager):
    handle = shell(live, 'echo "working"; sleep 300')

    assert [h.id for h in live.find(ref=task_ref(7))] == [handle.id]
    captured = live.dispose(handle)
    live.dispose(handle)

    # Cattle: a live worker is removed without being asked nicely first, and a
    # second disposal is somebody else's race, not an error.
    assert "working" in captured
    assert live.find() == []


@pytest.mark.docker
def test_labels_are_readable_back_off_a_live_container(live: ContainerManager):
    handle = shell(live, "sleep 300", issue=42)
    try:
        found = live.find()

        # #20 and #29 both start from exactly this listing.
        assert [(h.run_id, h.issue) for h in found] == [(live.run.id, 42)]
        assert found[0].image == live.image
    finally:
        live.dispose(handle)


@pytest.mark.docker
def test_a_container_that_exited_reads_back_as_exited_not_as_absent(live: ContainerManager):
    """The round trip #187 is about, against a real daemon.

    The whole bug is that this container is still *listed* - `docker ps --all`
    is what makes the reaper possible - while saying nothing about having
    stopped. A resolver reading the listing held its task in `claimed` from
    here until the reaper arrived.
    """
    handle = shell(live, 'echo "done"', issue=7)
    try:
        assert live.wait(handle) == 0

        listed = live.find()

        # Still there, which is the reaper's whole premise, and #29's.
        assert [h.id for h in listed] == [handle.id]
        assert listed[0].state == "exited"
        assert listed[0].running is False
    finally:
        live.dispose(handle)


@pytest.mark.docker
def test_a_caller_that_asked_for_running_containers_gets_only_those(live: ContainerManager):
    working = shell(live, "sleep 300", issue=42)
    finished = shell(live, "true", issue=7)
    try:
        assert live.wait(finished) == 0

        assert {h.id for h in live.find()} == {working.id, finished.id}
        assert [h.id for h in live.find(running=True)] == [working.id]
        assert [h.state for h in live.find(running=True)] == ["running"]
        # And the narrowing composes with the per-task filter.
        assert live.find(ref=task_ref(7), running=True) == []
        assert [h.id for h in live.find(ref=task_ref(7))] == [finished.id]
    finally:
        live.dispose(working)
        live.dispose(finished)


def test_a_manager_refuses_to_carry_the_boot_key_into_a_container(monkeypatch):
    """The separation is enforced where containers are made, not only in prose.

    `ContainerManager.__post_init__` is the one place a worker's environment is
    decided, so it is the only place the check is worth putting. A refusal to
    start beats a container that holds `administration` and `workflows` while
    running whatever the model just wrote.
    """
    from swarm.security import PROVISION_TOKEN_ENV, PolicyError

    monkeypatch.delenv(PROVISION_TOKEN_ENV, raising=False)
    with pytest.raises(PolicyError):
        ContainerManager(run=make_run(), env={PROVISION_TOKEN_ENV: "github_pat_" + "a" * 40})


def test_the_boot_key_is_redacted_even_though_it_is_never_passed(monkeypatch):
    """Belt and braces: enrolled so an unforeseen route cannot leak it."""
    from swarm.security import PROVISION_TOKEN_ENV

    secret = "github_pat_" + "z" * 40
    monkeypatch.setenv(PROVISION_TOKEN_ENV, secret)
    manager, _ = make_manager()
    assert secret not in manager.redactor(f"leaked {secret} somehow")


# --------------------------------------------------------------------------
# One image per stack (#99)
# --------------------------------------------------------------------------
#
# `Dockerfile.worker` argued for baking in no toolchain, on the grounds that a
# baked-in stack "would quietly narrow the swarm to repos that happen to use
# that stack". The intent was agnosticism; the effect was the opposite, because
# the one toolchain every image did carry was the Python the package needs.
# Agnosticism is bought here instead: several images, selected per task.


def test_the_default_mapping_covers_every_declarable_stack():
    """A stack an issue may declare but no image can run is a task that parses
    and then cannot be dispatched - the drift `KNOWN_STACKS` was written to
    make visible, checked from the other side."""
    from swarm.github.ledger import KNOWN_STACKS

    assert set(DEFAULT_STACK_IMAGES) == KNOWN_STACKS


def test_react_does_not_share_the_node_image():
    """It did, on the grounds that "React web needs Node and nothing else at
    the toolchain level", and that was wrong in the one way that mattered:
    `node --test` has no JSX transform, so the shared image could read a React
    project's files only as syntax errors. #106 gives React its own image;
    `test_the_build_hint_names_the_dockerfile_that_exists` is what pins the
    tag to a file in this repository."""
    images = StackImages()

    assert images.for_stack("react") == "apiary-worker-react"
    assert images.for_stack("react") != images.for_stack("node")


def test_a_stack_is_resolved_case_insensitively():
    assert StackImages().for_stack("Node") == "apiary-worker-node"


def test_a_stack_with_no_image_is_refused_with_something_to_do_about_it():
    """Never a mid-run `docker create` failure: the message a create failure
    produces names a tag, and this one names a thing to run."""
    with pytest.raises(ContainerError) as raised:
        StackImages().for_stack("rust")

    assert "rust" in str(raised.value)
    assert STACK_IMAGES_ENV in str(raised.value)
    assert "SETUP.md" in str(raised.value)


def test_an_override_merges_rather_than_replaces():
    """Overriding one stack must not silently un-configure the others, which is
    the failure mode of a whole-mapping override somebody edits in a hurry."""
    images = StackImages.from_env({STACK_IMAGES_ENV: "node=my-node:dev"})

    assert images.for_stack("node") == "my-node:dev"
    assert images.for_stack("python") == WORKER_IMAGE


def test_an_override_can_add_a_stack_the_defaults_do_not_have():
    images = StackImages.from_env({STACK_IMAGES_ENV: "go=apiary-worker-go"})

    assert images.for_stack("go") == "apiary-worker-go"


@pytest.mark.parametrize("raw", ["node", "node=", "=my-node", "node=a,,,broken"])
def test_a_malformed_override_is_loud(raw):
    """Loud on garbage, like `_env_int` and `_env_flag`. A mistyped pair that
    silently fell back would run the whole ledger on the Python image while
    somebody believed otherwise."""
    with pytest.raises(ContainerError):
        StackImages.from_env({STACK_IMAGES_ENV: raw})


def test_an_absent_override_is_the_defaults():
    assert StackImages.from_env({}).images == dict(DEFAULT_STACK_IMAGES)


def test_the_build_hint_names_the_dockerfile_that_exists():
    """One place, so `doctor`'s fix hint, the dispatcher's refusal and SETUP.md
    cannot drift - and asserted against the repository, so a renamed Dockerfile
    fails here rather than in somebody's terminal."""
    for image in DEFAULT_STACK_IMAGES.values():
        hint = build_hint(image)
        dockerfile = hint.split("-f ")[1].split()[0]
        assert (REPO_ROOT / dockerfile).is_file(), hint


def test_a_missing_image_is_told_from_every_other_docker_error():
    assert missing_image(
        DockerError(["docker", "create"], 125, "Unable to find image 'apiary-worker-node' locally")
    )
    assert not missing_image(
        DockerError(["docker", "create"], 125, "Cannot connect to the Docker daemon")
    )


def test_the_image_variable_matches_the_workers_own():
    """The spawner writes it, the worker reads it, and neither imports the
    other: `worker/entrypoint.py` is what runs *inside* the container and
    depends on nothing in `containers/`. Two spellings of one name, pinned."""
    from swarm.worker.entrypoint import IMAGE_ENV as worker_side

    assert IMAGE_ENV == worker_side


def test_a_spawn_uses_the_image_it_was_given_not_the_managers():
    manager, runner = make_manager()

    handle = manager.spawn(task_ref(7), BASE_COMMIT, issue=7, image="apiary-worker-node")

    assert handle.image == "apiary-worker-node"
    assert "apiary-worker-node" in runner.argv_for("create")


def test_a_spawn_with_no_image_still_uses_the_managers():
    """A run whose tasks never declare anything must keep working unchanged."""
    manager, runner = make_manager()

    handle = manager.spawn(task_ref(7), BASE_COMMIT, issue=7)

    assert handle.image == WORKER_IMAGE
    assert WORKER_IMAGE in runner.argv_for("create")


def test_the_worker_is_told_which_image_it_is_running_in():
    """It cannot ask: no socket, no `docker` binary. That is the containment
    working, so #97's result record can only name the image if it is told."""
    manager, runner = make_manager()

    manager.spawn(task_ref(7), BASE_COMMIT, issue=7, image="apiary-worker-node")

    assert f"{IMAGE_ENV}=apiary-worker-node" in runner.argv_for("create")


@pytest.mark.docker
def test_the_node_image_carries_node_and_a_writable_npm_cache():
    """The one live check on the second worker image.

    Two things, and the second is the one that would otherwise be found by a
    task rather than by a test: npm writes a cache on **every** invocation,
    including ones that install nothing, and an unwritable cache directory
    fails the command rather than skipping the cache. `/opt/venv` is owned by
    uid 10001 in `Dockerfile.worker` for the same reason.

    Skipped rather than failed when the image is not built: it is a manual
    `docker build` on the host by design (the orchestrator has `BUILD=0`), so
    "not built here" is a fact about the machine, not a regression.
    """
    image = StackImages().for_stack("node")
    manager = ContainerManager(
        run=make_run(),
        image=image,
        env={},
        limits=Limits(cpus=1.0, memory="512m", pids=128),
        timeout_s=120,
    )
    try:
        handle = shell(
            manager,
            "node --version && id -u && "
            # Write into the cache the way npm itself would, as uid 10001.
            'test -w "$NPM_CONFIG_CACHE" && echo "cache writable"',
        )
    except ContainerError as exc:
        if not missing_image(exc):
            raise
        pytest.skip(f"{image} is not built on this host: {build_hint(image)}")

    assert manager.wait(handle) == 0
    logs = manager.logs(handle)
    manager.dispose(handle)

    assert "v22." in logs
    assert "10001" in logs
    assert "cache writable" in logs


def test_the_container_layer_never_imports_the_tracker_adapter():
    """ADR 0001's line, asserted rather than described (#142).

    `swarm/containers/` is the execution plane. It labels, names and finds a
    container by `TaskRef.label_value`, which is the ref's own Docker-safe
    form - so a run against a tracker whose ids are `ENG-123` needs nothing
    here to change. An import of `swarm.github` would put that back silently,
    and the failure would not surface until somebody wired up a second tracker,
    which is far too late to find out.

    Deliberately a source scan rather than a check of `sys.modules`: the
    coupling that matters is the one written down, and an import inside a
    function would pass a runtime check on any path that did not take it.
    """
    package = Path(manager_module.__file__).parent
    offenders = {
        source.name: [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith(("import ", "from "))
            and ("swarm.github" in line or "..github" in line or ".github " in line)
        ]
        for source in sorted(package.glob("*.py"))
    }
    assert {name: lines for name, lines in offenders.items() if lines} == {}
