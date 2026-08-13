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
from typing import Any, Callable, Sequence

import pytest

from swarm.containers.manager import (
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
    dispose_container,
    find_containers,
)
from swarm.run import RUN_LABEL, Run

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
    return manager.spawn(issue, BASE_COMMIT)


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

    handle = manager.spawn(11, BASE_COMMIT)

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
            f"{CONTAINER_ID}\tapiary-run-issue-7-ab2c\tapiary-worker\t{make_run().id}\t7\n"
            f"{'d' * 64}\tapiary-run-issue-9-zz9y\tapiary-worker\tsome-other-run\tnonsense\n"
        )
    )

    handles = manager.find()

    assert [handle.issue for handle in handles] == [7, None]
    assert handles[0].name == "apiary-run-issue-7-ab2c"
    assert handles[1].run_id == "some-other-run"


def test_a_listing_can_be_narrowed_to_one_issue():
    manager, runner = make_manager()

    manager.find(issue=7)

    assert f"label={ISSUE_LABEL}=7" in runner.argv_for("ps")


def test_containers_of_every_run_are_findable_without_a_run_object():
    """#20 sweeps runs whose orchestrator is dead and whose `Run` is gone."""
    runner = ScriptedRunner()
    runner.replies["ps"] = Reply(stdout=f"{CONTAINER_ID}\tapiary-x\tapiary-worker\tdead-run-1\t4\n")
    docker = DockerCLI(runner=runner)

    handles = find_containers(docker)

    assert [handle.run_id for handle in handles] == ["dead-run-1"]
    # Still filtered - on the label's presence rather than on its value.
    assert f"label={RUN_LABEL}" in runner.argv_for("ps")
    # And the same disposal, so a foreign container's logs are captured too.
    assert dispose_container(handles[0], docker) == ""
    assert runner.commands == ["ps", "logs", "rm"]


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
    return manager.spawn(issue, BASE_COMMIT, entrypoint="/bin/sh", command=["-c", script])


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

    assert [h.id for h in live.find(issue=7)] == [handle.id]
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
