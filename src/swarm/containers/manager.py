"""Spawn a worker container, wait on it, read what it said, destroy it.

`docs/architecture-v2.md` gives the execution plane one paragraph of contract -
"containers are cattle", "disposal is unconditional", "logs are captured to the
run artifact directory *before* removal" - and this module is all of it. The
dispatcher (#21) spawns through here, the reaper (#20) finds and disposes
through here, and #29 writes what `logs` returned to disk.

**The Docker CLI, not the SDK.** `pyproject.toml` grows no `docker` dependency
for this: the CLI is already the thing the human debugs with, so every call
below is a command a reader can paste into a terminal, and `DOCKER_HOST` -
which is how #28 puts a socket proxy in front of the daemon - is honoured by
the binary without this module knowing the proxy exists. The cost is parsing
text, and the parsing is confined to `DockerCLI`.

**Redaction happens at the capture boundary, and the boundary is one class.**
A worker holds a token that can push. Git puts a failing remote URL into its
error output, a shell trace puts the environment there, and #29 then writes
that stream to disk where it is the single most likely thing to be pasted into
an issue. Redacting later means redacting in every consumer and getting it
right in all of them forever. So `DockerCLI` owns a `Redactor` and applies it
to **everything** the daemon hands back - container logs, yes, but also the
text of every failure, because `docker create ... --env GITHUB_TOKEN=ghp_...`
failing would otherwise put the token into an exception message, and exception
messages get logged too.

**Capture precedes removal, structurally.** `dispose_container` returns the
logs it captured, so there is no ordering to remember and no way to write the
removal without the capture. The logs of a container that failed are exactly
the ones worth having and they are gone the instant it is removed.

**Disposal is idempotent and forgiving.** Two things race to remove a
container - the dispatcher that spawned it and the reaper that sweeps the run -
and "already gone" is the outcome both of them wanted. It is not an error.

Labels are the join key for everything downstream: `apiary.run=<id>` from
`swarm.run` (#33), which #20 reaps on and #29 names a directory after, plus
`apiary.issue=<n>`. Every listing in this module filters on the run label. A
bare `docker ps -a` on a development machine returns the human's databases and
editors as well, and the caller of a listing here is a function that removes
what it is handed.
"""

from __future__ import annotations

import os
import re
import secrets as _secrets
import subprocess
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from ..config import SETTINGS
from ..run import RUN_ID_ENV, RUN_LABEL, SUFFIX_ALPHABET, Run, validate_run_id

#: The image #14 builds. Overridable per manager, because a locally built tag
#: is how anyone tests a worker change before it is published anywhere.
WORKER_IMAGE = "apiary-worker"

#: The per-container half of the label pair. `docs/issue-contract.md` §2: the
#: issue *number* addresses a task, the marker id identifies it, and a label
#: read back off a container is an address.
ISSUE_LABEL = "apiary.issue"

#: What a redacted secret is replaced by. Deliberately not a hint at the
#: length or shape of what it replaced.
PLACEHOLDER = "***"

#: A shorter value is not redacted at all. `--env DEBUG=1` would otherwise
#: register "1" as a secret and turn every timestamp in the logs into asterisks,
#: which destroys the artifact this whole module exists to preserve.
MIN_SECRET_LENGTH = 8

#: An environment variable whose *name* matches this has its value redacted
#: from everything the daemon says, without anyone having to remember to
#: register it. Handing a container a secret is what enrols it.
SECRET_NAME_RE = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|APIKEY|_KEY\b", re.I)

#: Shapes that are secrets regardless of whether this process was told about
#: them. A worker can print a token it minted itself, or one that arrived in a
#: file, and the literal list would not know.
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # GitHub's own prefixes: ghp_ (classic PAT), gho_/ghu_ (OAuth), ghs_
    # (server-to-server, which is what a fine-grained app token is), ghr_.
    re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{16,}"),
    # `https://user:password@host/...`. This is the exact shape git prints back
    # when a push to an authenticated remote fails, and the one #28's clone URL
    # would take if it ever carried credentials.
    re.compile(r"(?<=://)[^\s/:@]+:[^\s/@]+(?=@)"),
)

#: Logs are captured whole and then bounded. A runaway `while true: print()`
#: inside a worker is not hypothetical, and #29 writes this string to disk.
#: Both ends are kept: the head holds the setup that went wrong, the tail holds
#: the failure, and the middle of a loop is the same line 400,000 times.
MAX_LOG_CHARS = 256_000
HEAD_SHARE = 0.25

#: `docker rm` on something already removed. Matching the message is unpleasant
#: but it is the only signal the CLI gives, and the alternative - inspect first,
#: then remove - is a race with the reaper rather than a fix for one.
MISSING_RE = re.compile(r"no such container|is already in progress", re.I)

#: `docker ps` columns, tab-separated because a container name cannot contain a
#: tab and an image reference cannot either.
_PS_FORMAT = (
    '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Label "' + RUN_LABEL + '"}}\t{{.Label "' + ISSUE_LABEL + '"}}'
)

#: Container names carry a random tail for the same reason run ids do: a second
#: attempt at one issue inside one run must not collide with the corpse of the
#: first if that one's removal failed.
NAME_SUFFIX_LENGTH = 4

#: Grace period for `docker stop` before it escalates to SIGKILL. A worker that
#: has already blown its timeout has forfeited a long goodbye.
STOP_GRACE_S = 10


class ContainerError(RuntimeError):
    """Base for everything this module raises."""


class DockerError(ContainerError):
    """The daemon refused a command. Message and argv are already redacted."""

    def __init__(self, argv: Sequence[str], returncode: int, output: str) -> None:
        self.argv = list(argv)
        self.returncode = returncode
        self.output = output
        super().__init__(f"{' '.join(self.argv)} -> exit {returncode}: {output.strip()}")


class ContainerTimeout(ContainerError):
    """A container outlived its budget and was stopped.

    Distinct from `DockerError` because the reconciler (#22) has to tell "the
    task ran and failed" from "the task never finished", and only the second
    one means the container is still there holding a clone.
    """


def is_missing(error: ContainerError) -> bool:
    """Was this failure just "that container is already gone"?

    Public because #20 makes the same judgement about containers it did not
    spawn, and a second copy of this predicate would eventually disagree with
    this one.
    """
    return isinstance(error, DockerError) and bool(MISSING_RE.search(error.output))


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------


class Redactor:
    """Removes known secrets, and secret-shaped strings, from captured text.

    Two sources, because neither alone is enough. Registered literals catch the
    token this process handed the container even though it looks like nothing
    in particular; the patterns catch a credential this process never saw. A
    redactor that only knew its own secrets would pass through a token the
    worker fetched itself, and one that only knew shapes would pass through the
    first credential format GitHub invents next.
    """

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._literals: list[str] = []
        for secret in secrets:
            self.add(secret)

    def add(self, secret: str | None) -> None:
        """Register one value. Short and empty values are ignored - see `MIN_SECRET_LENGTH`."""
        if not secret or len(secret) < MIN_SECRET_LENGTH:
            return
        # A token inside a URL arrives percent-encoded, and `replace` is
        # literal. Registering both forms costs nothing and the alternative is
        # a secret that survives exactly the code path most likely to print it.
        for form in (secret, urllib.parse.quote(secret, safe="")):
            if form not in self._literals:
                self._literals.append(form)

    def add_env(self, env: Mapping[str, str]) -> None:
        """Register every value whose variable name looks like a credential.

        This is what makes redaction automatic rather than remembered: handing
        a container `GITHUB_TOKEN` is the same act as enrolling its value here,
        so a future variable cannot be added to a spawn without being covered.
        """
        for name, value in env.items():
            if SECRET_NAME_RE.search(name):
                self.add(value)

    def __call__(self, text: str) -> str:
        if not text:
            return text
        # Longest first: a short literal that happens to be a prefix of a long
        # one would otherwise cut the long one in half and leave the tail.
        for literal in sorted(self._literals, key=len, reverse=True):
            text = text.replace(literal, PLACEHOLDER)
        for pattern in SECRET_PATTERNS:
            text = pattern.sub(PLACEHOLDER, text)
        return text


def _identity(text: str) -> str:
    return text


# --------------------------------------------------------------------------
# The daemon
# --------------------------------------------------------------------------


class Runner(Protocol):
    """Starts one process and waits for it. The seam `subprocess` sits behind.

    Same shape as `swarm.github.client.Transport`, and for the same reason: the
    interesting logic here is redaction and error interpretation, and a test
    that has to have a Docker daemon to reach it is a test that does not run.
    """

    def __call__(
        self, argv: Sequence[str], *, timeout_s: float | None, merge: bool
    ) -> subprocess.CompletedProcess: ...


def subprocess_runner(
    argv: Sequence[str], *, timeout_s: float | None, merge: bool
) -> subprocess.CompletedProcess:
    """The real one. `merge` sends both channels down a single pipe.

    Merging is what preserves interleaving, which is the whole value of a
    container's log: a traceback on stderr means little without the line
    printed on stdout just before it. Separate pipes are what lets a *failure*
    be quoted without the command's own output wrapped around it.
    """
    streams = (
        {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
        if merge
        else {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    )
    return subprocess.run(list(argv), text=True, timeout=timeout_s, **streams)


class DockerCLI:
    """Every call this module makes to Docker, and the only place text is kept.

    Both output channels pass through `redact` before anything else can see
    them, which is what makes "redact at the capture boundary" a property of
    the code rather than a rule contributors have to know.
    """

    def __init__(
        self,
        binary: str = "docker",
        redact: Callable[[str], str] | None = None,
        runner: Runner | None = None,
    ) -> None:
        self.binary = binary
        self.redact = redact or _identity
        self.runner = runner or subprocess_runner

    def __call__(self, *args: str, timeout_s: float | None = None) -> str:
        """Run a command, return its stdout. Raises `DockerError` on failure."""
        done = self._run(args, timeout_s=timeout_s, merge=False)
        if done.returncode != 0:
            raise DockerError(self._argv(args), done.returncode, self.redact(done.stderr or done.stdout))
        return self.redact(done.stdout)

    def combined(self, *args: str, timeout_s: float | None = None) -> str:
        """Run a command, return stdout and stderr as one interleaved stream.

        For `docker logs`, where the two streams are one story - see
        `subprocess_runner`. Everything else uses `__call__`.
        """
        done = self._run(args, timeout_s=timeout_s, merge=True)
        output = self.redact(done.stdout)
        if done.returncode != 0:
            raise DockerError(self._argv(args), done.returncode, output)
        return output

    # --- plumbing -------------------------------------------------------

    def _run(self, args: Sequence[str], *, timeout_s: float | None, merge: bool) -> subprocess.CompletedProcess:
        argv = [self.binary, *args]
        try:
            return self.runner(argv, timeout_s=timeout_s, merge=merge)
        except FileNotFoundError as exc:
            raise ContainerError(
                f"{self.binary!r} is not on PATH; the orchestrator reaches the daemon "
                f"through the CLI, and DOCKER_HOST decides which daemon"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            # Redacted, because the argv of a `create` carries the environment.
            raise ContainerTimeout(
                f"{' '.join(self._argv(args))} did not return within {timeout_s}s"
            ) from exc

    def _argv(self, args: Sequence[str]) -> list[str]:
        return [self.redact(part) for part in (self.binary, *args)]


# --------------------------------------------------------------------------
# Handles
# --------------------------------------------------------------------------


@dataclass
class Handle:
    """One container, plus whatever was salvaged from it before removal.

    Mutable on purpose. `captured` is the whole point of the type: after
    `dispose_container` the container does not exist and this object is the only
    remaining evidence of what it did, which is what #29 persists.
    """

    id: str
    run_id: str
    issue: int | None = None
    name: str = ""
    image: str = ""
    captured: str | None = None
    removed: bool = False

    @property
    def short_id(self) -> str:
        return self.id[:12]

    def __str__(self) -> str:
        label = self.name or self.short_id
        return f"{label} (issue #{self.issue})" if self.issue is not None else label


def find_containers(
    docker: DockerCLI,
    *,
    run_id: str | None = None,
    issue: int | None = None,
) -> list[Handle]:
    """Containers this system created, newest first, optionally one run's only.

    Always filtered on `apiary.run`. The reaper (#20) removes what this returns,
    and the daemon on a development machine is shared with everything else the
    human has running - an unfiltered listing here is a `docker rm` sweep across
    somebody's afternoon.
    """
    label = f"label={RUN_LABEL}" + (f"={validate_run_id(run_id)}" if run_id else "")
    args = ["ps", "--all", "--no-trunc", "--filter", label]
    if issue is not None:
        args += ["--filter", f"label={ISSUE_LABEL}={int(issue)}"]
    args += ["--format", _PS_FORMAT]

    handles: list[Handle] = []
    for line in docker(*args).splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 5:
            continue
        container_id, name, image, run_label, issue_label = (part.strip() for part in fields[:5])
        handles.append(
            Handle(
                id=container_id,
                run_id=run_label,
                issue=int(issue_label) if issue_label.isdigit() else None,
                name=name,
                image=image,
            )
        )
    return handles


def capture_logs(handle: Handle, docker: DockerCLI, *, max_chars: int = MAX_LOG_CHARS) -> str:
    """This container's output, redacted and bounded.

    A disposed handle answers from `captured` without touching the daemon,
    which is what makes `logs` meaningful after the container is gone.
    """
    if handle.captured is not None:
        return handle.captured
    try:
        stream = docker.combined("logs", handle.id)
    except ContainerError as exc:
        if is_missing(exc):
            # Removed under us - by the reaper, by `docker system prune`, by a
            # human. There is nothing left to read and nothing to fix.
            handle.captured = ""
            return handle.captured
        raise
    # Truncation happens *after* redaction, never before: a secret split across
    # the elision would otherwise survive as two halves that a reader can still
    # recognise, and neither half would match a literal.
    return _bounded(stream, max_chars)


def dispose_container(handle: Handle, docker: DockerCLI) -> str:
    """Capture the logs, then destroy the container. Returns what was captured.

    Idempotent, and silent about a container that is already gone: the
    dispatcher disposing what it spawned and the reaper sweeping the run both
    want the same end state, and whichever loses the race got it.

    The return value is the ordering guarantee made structural - there is no
    way to spell the removal without having taken the logs first.
    """
    if handle.removed:
        return handle.captured or ""

    try:
        handle.captured = capture_logs(handle, docker)
    except ContainerError as exc:
        # An unreadable log must never prevent a removal. Leaking a container
        # that holds a clone is the worse failure, and it is the one that makes
        # the next run's concurrency accounting wrong.
        handle.captured = f"[apiary] logs could not be captured: {exc}"

    try:
        docker("rm", "--force", "--volumes", handle.id)
    except ContainerError as exc:
        if not is_missing(exc):
            raise
    handle.removed = True
    return handle.captured


def _bounded(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * HEAD_SHARE)
    tail = max_chars - head
    elided = len(text) - max_chars
    return f"{text[:head]}\n[apiary] {elided} characters elided\n{text[-tail:]}"


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Limits:
    """Explicit `--cpus`, `--memory` and `--pids-limit`, per architecture-v2.

    Not tuning knobs. A worker runs LLM-written code, so an infinite loop or a
    fork bomb is an ordinary Tuesday, and the failure mode without these is
    "the Mac becomes unusable" rather than "one task fails". Defaults leave
    room for the host's Ollama, which is where the machine's real work happens.
    """

    cpus: float = 2.0
    memory: str = "4g"
    pids: int = 512

    def flags(self) -> list[str]:
        return [
            "--cpus", f"{self.cpus:g}",
            "--memory", self.memory,
            "--pids-limit", str(self.pids),
        ]


DEFAULT_LIMITS = Limits()

#: Variables inherited from the orchestrator's own environment when a caller
#: does not name one explicitly. The token is the reason the class below has a
#: redactor at all; `OLLAMA_HOST` is inherited rather than defaulted because
#: `Dockerfile.worker` already sets a sane one and overriding it blindly would
#: undo that.
INHERITED_ENV = ("GITHUB_TOKEN", "OLLAMA_HOST")


# --------------------------------------------------------------------------
# The manager
# --------------------------------------------------------------------------


@dataclass
class ContainerManager:
    """One run's worker containers: spawn, wait, read, dispose.

    Bound to a `Run` because every container it creates carries that run's
    label (#33) and because the repository to clone is the run's. Anything that
    needs to act on containers *without* a `Run` - the reaper, which sweeps
    runs that belong to dead processes - uses the module functions above, which
    is why they are module functions.
    """

    run: Run
    image: str = WORKER_IMAGE
    env: Mapping[str, str] | None = None
    limits: Limits = DEFAULT_LIMITS
    timeout_s: float = float(SETTINGS.worker_timeout_s)
    clone_url: str | None = None
    #: Extra `docker create` flags, appended before the image name. This is how
    #: a worker gets somewhere to write (`RunArtifacts.mount_flags`) and how it
    #: gets confined (`security.worker_create_flags`); both produce argv, and
    #: neither module owns this call site. Everything passed here goes through
    #: `assert_unprivileged` with the rest of the argv, so a flag that would
    #: create a privileged container is refused wherever it came from.
    extra_flags: Sequence[str] = ()
    docker: DockerCLI | None = None
    runner: Runner | None = None
    redactor: Redactor = field(default_factory=Redactor)

    def __post_init__(self) -> None:
        self.env = dict(self.env) if self.env is not None else _inherited_env()
        # The worker names its own result file after the run and the attempt.
        # It learns the attempt from the contract it already reads; the run id
        # only exists out here, so it travels as an environment variable rather
        # than as a fourth positional argument nobody else needs.
        self.env.setdefault(RUN_ID_ENV, self.run.id)
        # Enrolment, not a copy: everything handed to a container as a
        # credential is registered before a single command runs, so no code
        # path exists in which a secret reaches the daemon before the redactor
        # knows about it.
        # The boot key (`administration` + `workflows`) must never reach a
        # worker. Checked here because this is the one place a container's
        # environment is decided, and a refusal to start beats a container that
        # quietly holds the permissions that would let model output rewrite the
        # CI judging it. Imported locally: `security` imports this module.
        from ..security import PROVISION_TOKEN_ENV, assert_no_provision_token

        assert_no_provision_token(self.env)
        # Enrolled even though it is not passed: if it ever reaches a log by a
        # route nobody predicted, the redactor has already seen it.
        boot_key = os.environ.get(PROVISION_TOKEN_ENV)
        if boot_key:
            self.redactor.add(boot_key)

        self.redactor.add_env(self.env)
        if self.docker is None:
            # `runner` is the short spelling of "same manager, different
            # process starter"; handing it a whole `DockerCLI` would mean
            # remembering to wire this run's redactor into it by hand, and the
            # one that got forgotten would log a token.
            self.docker = DockerCLI(redact=self.redactor, runner=self.runner)

    # --- lifecycle ------------------------------------------------------

    def spawn(
        self,
        issue: int,
        base_commit: str,
        *,
        entrypoint: str | None = None,
        command: Sequence[str] | None = None,
    ) -> Handle:
        """Create and start one worker container for `issue`, at `base_commit`.

        Created and started as two steps rather than one `docker run -d`: a
        container that was created and then failed to start is still a
        container, and this way there is a handle to dispose it with instead of
        an id that only exists inside a failed command's output.

        `entrypoint` and `command` override what the image runs. The dispatcher
        never passes either - they exist so a caller can put a probe container
        under this run's labels, limits and redaction, which is what the
        integration tests do and what a `swarm doctor` check would want.
        """
        name = self._container_name(issue)
        args = [
            "create",
            "--name", name,
            # PID 1 in the worker is the entrypoint, which spawns git and the
            # verify command; without an init it reaps none of them and a
            # finished task can sit on zombies until the pids limit bites.
            "--init",
            "--label", f"{RUN_LABEL}={self.run.id}",
            "--label", f"{ISSUE_LABEL}={int(issue)}",
            *self.limits.flags(),
            # Docker Desktop resolves this itself; a Linux host does not, and
            # the host's Ollama is the one piece of infrastructure every worker
            # needs (architecture-v2, "Three constraints").
            "--add-host", "host.docker.internal:host-gateway",
            *self._env_flags(),
            *self.extra_flags,
        ]
        if entrypoint is not None:
            args += ["--entrypoint", entrypoint]
        args.append(self.image)
        args += list(command) if command is not None else self._worker_args(issue, base_commit)

        created = self._cli(*args).strip().splitlines()
        if not created or not created[-1]:
            raise ContainerError(f"docker create returned no container id for issue #{issue}")

        handle = Handle(
            id=created[-1].strip(),
            run_id=self.run.id,
            issue=int(issue),
            name=name,
            image=self.image,
        )
        try:
            self._cli("start", handle.id)
        except ContainerError:
            # Leave nothing behind, including on the path where nothing ran.
            dispose_container(handle, self._cli)
            raise
        return handle

    def wait(self, handle: Handle, *, timeout_s: float | None = None) -> int:
        """Block until the container exits; return its exit code.

        The exit code is the worker protocol (`docs/issue-contract.md`): 0 = PR
        open, 1 = task failed, 2 = infrastructure error. Nothing here
        interprets it - that is the reconciler's decision (#22) - but a
        container that never finishes has no exit code to interpret, so a
        timeout stops it and raises rather than returning a number nobody
        computed. The container is stopped, not removed, so the caller can
        still read the logs of whatever it was stuck on.
        """
        limit = self.timeout_s if timeout_s is None else timeout_s
        try:
            output = self._cli("wait", handle.id, timeout_s=limit)
        except ContainerTimeout as exc:
            self.stop(handle)
            raise ContainerTimeout(f"{handle} exceeded {limit}s and was stopped") from exc

        for line in reversed(output.strip().splitlines()):
            if line.strip().lstrip("-").isdigit():
                return int(line.strip())
        raise ContainerError(f"docker wait said {output.strip()!r} for {handle}, not an exit code")

    def stop(self, handle: Handle) -> None:
        """SIGTERM, then SIGKILL after a grace period. Tolerates a gone container."""
        try:
            self._cli(
                "stop", "--timeout", str(STOP_GRACE_S), handle.id,
                timeout_s=STOP_GRACE_S + 30,
            )
        except ContainerError as exc:
            if not is_missing(exc):
                raise

    def logs(self, handle: Handle) -> str:
        """Redacted output, from the daemon or from what disposal salvaged."""
        return capture_logs(handle, self._cli)

    def dispose(self, handle: Handle) -> str:
        """Capture, then destroy. Idempotent; see `dispose_container`."""
        return dispose_container(handle, self._cli)

    def find(self, *, issue: int | None = None) -> list[Handle]:
        """This run's containers, running or not."""
        return find_containers(self._cli, run_id=self.run.id, issue=issue)

    # --- plumbing -------------------------------------------------------

    @property
    def _cli(self) -> DockerCLI:
        assert self.docker is not None  # set in __post_init__
        return self.docker

    def _worker_args(self, issue: int, base_commit: str) -> list[str]:
        """The three arguments `Dockerfile.worker`'s entrypoint documents.

        The clone URL carries no credential. The token travels as an
        environment variable (#28), which keeps it out of the container's own
        `Args` - where `docker inspect` prints it, where the worker's every git
        error would quote it back, and where it would survive in the image's
        history of a `docker commit`.
        """
        return [
            "--repo", self.clone_url or f"https://github.com/{self.run.repo}.git",
            "--issue", str(int(issue)),
            "--base-commit", base_commit,
        ]

    def _env_flags(self) -> list[str]:
        """`--env NAME` where the value is already ours, `--env NAME=VALUE` otherwise.

        The bare form tells the docker CLI to read the value out of *this*
        process's environment, which is how the token arrives from compose. It
        then never appears on a command line at all - not in the process table,
        and not in the text of a failed `create`. The redactor covers the other
        form; this removes the need for it.
        """
        env = self.env or {}
        flags: list[str] = []
        for name in sorted(env):
            value = env[name]
            flags += ["--env", name if os.environ.get(name) == value else f"{name}={value}"]
        return flags

    def _container_name(self, issue: int) -> str:
        tail = "".join(_secrets.choice(SUFFIX_ALPHABET) for _ in range(NAME_SUFFIX_LENGTH))
        return f"apiary-{self.run.id}-issue-{int(issue)}-{tail}"


def _inherited_env() -> dict[str, str]:
    return {name: os.environ[name] for name in INHERITED_ENV if os.environ.get(name)}
