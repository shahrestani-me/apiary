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
`apiary.issue=<task>`. Every listing in this module filters on the run label. A
bare `docker ps -a` on a development machine returns the human's databases and
editors as well, and the caller of a listing here is a function that removes
what it is handed.

**Nothing here knows which tracker a run reads.** A container is labelled and
named with `TaskRef.label_value` - the ref's own Docker-safe form - so this
module never converts a task id, and `docs/adr/0001-task-systems-are-integrations.md`
does not acquire an execution plane that only works for GitHub. The one piece of
the code host that does live here is the clone URL and the worker's `--issue`
argument, and both are deliberate: ADR 0001 keeps the *code host* integration
and narrows only the tracker, and the worker is itself a GitHub client.
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
from ..taskref import TaskRef

#: The image #14 builds. Overridable per manager, because a locally built tag
#: is how anyone tests a worker change before it is published anywhere.
WORKER_IMAGE = "apiary-worker"

#: Which image carries which stack's toolchain.
#:
#: `Dockerfile.worker` deliberately baked in no toolchain, and the stated
#: purpose of that was stack-agnosticism - "baking a stack in would quietly
#: narrow the swarm to repos that happen to use that stack". Baking *none*
#: narrowed it to Python instead, because the one thing every image did have
#: was the Python the package itself needs. Agnosticism is bought here instead:
#: several images, one per stack, selected per task.
#:
#: Node and React used to share an image, on the grounds that "React web needs
#: Node and nothing else at the toolchain level". That was wrong in the one way
#: that mattered: `node --test` has no JSX transform, so the shared image could
#: run a React project's files only as syntax errors. #106 gives React its own
#: image carrying react, react-dom, vitest and a DOM - installed at build time,
#: where the network is allowed, so the gate still needs no registry at run
#: time. `Dockerfile.worker.react` says why that is the whole design.
DEFAULT_STACK_IMAGES: Mapping[str, str] = {
    "python": WORKER_IMAGE,
    "node": "apiary-worker-node",
    "react": "apiary-worker-react",
}

#: Override, as `stack=image` pairs: `APIARY_WORKER_IMAGES=node=my-node:dev`.
#: Merged over the defaults rather than replacing them, so overriding one stack
#: does not silently un-configure the others - the same call this codebase makes
#: everywhere a mapping is env-overridable.
STACK_IMAGES_ENV = "APIARY_WORKER_IMAGES"

#: How a worker learns which image it is running in, so its own result record
#: can say. Spelled here rather than imported from `worker/entrypoint.py`: that
#: module imports nothing from this package by design - it is what runs *inside*
#: the container - and `test_the_image_variable_matches_the_workers_own` pins
#: the two spellings together.
IMAGE_ENV = "APIARY_WORKER_IMAGE"

#: The label a stack image carries so it can be recognised without being run.
#: `docker image inspect` is denied to a containerised orchestrator
#: (`SOCKET_PROXY_ENV` sets `IMAGES=0`), so this is for a human and for #100's
#: doctor check running on the host, not for the dispatch path.
STACK_LABEL = "org.apiary.stack"

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
#:
#: `{{.State}}` is the last column and it is load-bearing rather than
#: decorative: this listing is `docker ps --all`, so an *exited* container is
#: still in it, and without the state a reader cannot tell the worker that is
#: still working from the one that finished thirty seconds ago. `{{.State}}`
#: rather than `{{.Status}}`, which is the human sentence ("Exited (0) 3
#: minutes ago") and would have to be parsed; the state is one word from a
#: closed set.
_PS_FORMAT = (
    '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Label "'
    + RUN_LABEL
    + '"}}\t{{.Label "'
    + ISSUE_LABEL
    + '"}}\t{{.State}}'
)

#: How many columns `_PS_FORMAT` asks for. A row with fewer came from a daemon
#: that answered something other than what was asked, and is skipped rather
#: than guessed at.
_PS_FIELDS = 6

#: The `docker ps` filter that means "still working". Applied by the daemon
#: rather than by this module, exactly as the label filter is: the listing a
#: caller asked for is the listing it gets, and the argv still reads as
#: something a human can paste.
_RUNNING_FILTER = "status=running"

#: The one value of `{{.State}}` that means a worker is still working. The rest
#: - `created`, `restarting`, `paused`, `removing`, `exited`, `dead` - are all
#: "not running now", and a caller that meant liveness must read none of them
#: as a live worker.
RUNNING_STATE = "running"

#: The value of `{{.State}}` for a container `docker create` has returned for and
#: `docker start` has not yet taken effect on. `recovery.py` notes that
#: `docker ps --all` lists a container from the instant `create` returns, so this
#: is a real window in which a container exists, is not running, and a task is
#: legitimately claimed. `spawn` closes it by calling `start` in the same breath,
#: but the value is named here rather than spelled at the one site that reads it
#: (`orchestrator/shadow.py`, which reports the window as an expected divergence
#: rather than a finding): a second spelling of a state string is how two
#: readings of the same daemon field drift apart.
CREATED_STATE = "created"

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


@dataclass(frozen=True)
class StackImages:
    """Which image to spawn for a task's declared stack.

    House convention, matching `MergePolicy` and `InfrastructurePolicy`: a
    documented default, an `APIARY_*` override, a `from_env()` read once at the
    call site, and a `summary()` worth logging at startup.

    **The supply side is the hard half, and it is not solved here.** The
    orchestrator can neither pull (`IMAGES=0`) nor build (`BUILD=0`) - that is
    the socket-proxy narrowing working as designed - so every stack image is a
    manual `docker build` on the host. Nothing in the repository documented that
    command before this ticket; it appeared only in a `doctor` fix hint. So the
    refusal below names the build line, and `SETUP.md` gains a step.
    """

    images: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_STACK_IMAGES))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> StackImages:
        """`APIARY_WORKER_IMAGES=node=my-node:dev,python=my-py:dev`.

        Merged over the defaults rather than replacing them: overriding one
        stack must not silently un-configure the others, which is the failure
        mode of a whole-mapping override that somebody edits in a hurry.
        """
        source = os.environ if env is None else env
        raw = (source.get(STACK_IMAGES_ENV) or "").strip()
        images = dict(DEFAULT_STACK_IMAGES)
        for pair in raw.split(","):
            pair = pair.strip()
            if not pair:
                continue
            stack, sep, image = pair.partition("=")
            if not sep or not stack.strip() or not image.strip():
                # Loud on garbage, like `_env_int` and `_env_flag`. A mistyped
                # pair that silently fell back would run the whole ledger on
                # the Python image while somebody believed otherwise.
                raise ContainerError(
                    f"{STACK_IMAGES_ENV}={raw!r}: expected comma-separated stack=image pairs, "
                    f"got {pair!r}"
                )
            images[stack.strip().casefold()] = image.strip()
        return cls(images=images)

    def for_stack(self, stack: str) -> str:
        """The image for one stack, or `ContainerError` naming what is missing.

        Refused here, before anything is claimed, rather than as a `docker
        create` failure mid-cycle: a claim spent on a task this host cannot run
        is a claim #35 has to sweep, and the message a create failure produces
        names a tag rather than a thing to do about it.
        """
        image = self.images.get((stack or "").casefold())
        if image:
            return image
        known = ", ".join(sorted(self.images)) or "(none)"
        raise ContainerError(
            f"no worker image is configured for stack {stack!r}; this host knows {known}. "
            f"Add one with {STACK_IMAGES_ENV}=<stack>=<image>, and build it - see SETUP.md"
        )

    def summary(self) -> str:
        pairs = ", ".join(f"{stack}={image}" for stack, image in sorted(self.images.items()))
        return f"worker images: {pairs} ({STACK_IMAGES_ENV} to override)"


#: `docker create` on an image that was never built. Matching the message is
#: the same trade `is_missing` makes, for the same reason: the CLI gives no
#: other signal. Used only to *improve* an error, never to decide anything.
_NO_IMAGE_RE = re.compile(r"unable to find image|no such image|manifest unknown", re.I)


def missing_image(error: ContainerError) -> bool:
    """Did this fail because the image is not on this host?

    The one failure whose fix is a command rather than an investigation, which
    is why it is worth telling apart: the orchestrator cannot build or pull, so
    a human has to, and the message should say so.
    """
    return isinstance(error, DockerError) and bool(_NO_IMAGE_RE.search(error.output))


def build_hint(image: str) -> str:
    """The command a human runs to fix `missing_image`.

    One place, so the fix hint in `doctor`, the dispatcher's refusal and
    `SETUP.md` cannot drift. The orchestrator can neither build nor pull -
    `SOCKET_PROXY_ENV` sets `BUILD=0` and `IMAGES=0` - so this really is the
    whole remedy, and it is a human's to run.

    `apiary-worker-node:tag` -> `Dockerfile.worker.node`. A tag this convention
    does not cover falls back to naming the image, which is still more use than
    "no such image".
    """
    tag = image.split(":")[0]
    if not tag.startswith("apiary-"):
        return f"build or pull {image}"
    return f"docker build -f Dockerfile.{tag.removeprefix('apiary-').replace('-', '.')} -t {image} ."


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
    #: The daemon's own word for what this container is doing, as `docker ps`
    #: printed it - `running`, `exited`, `created`, and the rest of the closed
    #: set. Read, never inferred: `spawn` leaves it empty even after a
    #: successful `docker start`, because a container that started and exited
    #: in the same breath is the ordinary case for a worker and a handle that
    #: claimed otherwise would be lying about the exact window this field
    #: exists for. Empty means "no listing said", which `running` reads as
    #: false - the direction that cannot mistake a corpse for a live worker.
    state: str = ""
    captured: str | None = None
    removed: bool = False

    @property
    def short_id(self) -> str:
        return self.id[:12]

    @property
    def running(self) -> bool:
        """Was this container still working when the listing was taken?

        The question `docker ps --all` does not answer on its own, and the one
        the derived-state resolver (#146) asks: a task is `claimed` while a
        *live* worker holds it, and a container that exited this cycle has
        already opened its pull request. Reading the listing without this held
        every task in `claimed` from the moment its worker exited until the
        reaper arrived.
        """
        return self.state == RUNNING_STATE

    def __str__(self) -> str:
        label = self.name or self.short_id
        return f"{label} (issue #{self.issue})" if self.issue is not None else label


def find_containers(
    docker: DockerCLI,
    *,
    run_id: str | None = None,
    task: str | None = None,
    running: bool = False,
) -> list[Handle]:
    """Containers this system created, newest first, optionally one run's only.

    Always filtered on `apiary.run`. The reaper (#20) removes what this returns,
    and the daemon on a development machine is shared with everything else the
    human has running - an unfiltered listing here is a `docker rm` sweep across
    somebody's afternoon.

    `task` is a label value, not a task id: `TaskRef.label_value`, which the
    caller derives. Taking the token rather than the ref is what keeps this
    function - and the reaper that calls it - free of any opinion about what a
    task id looks like.

    **`running` is off by default and that default is deliberate.** The listing
    is `docker ps --all` because the two callers that matter most need what has
    already stopped: the reaper disposes exited containers - they still hold a
    clone and their logs are the run's most valuable - and `recovery.holders`
    counts an exited container of a live run as a claim somebody is honouring.
    Narrowing the default would break both. `running=True` is for the caller
    that means liveness and had, until now, no way to say so; the daemon
    applies it, so an exited container is not fetched and discarded, it is
    never listed. Every handle carries `state` either way, so a caller taking
    the whole listing can still tell the two apart.
    """
    label = f"label={RUN_LABEL}" + (f"={validate_run_id(run_id)}" if run_id else "")
    args = ["ps", "--all", "--no-trunc", "--filter", label]
    if task is not None:
        args += ["--filter", f"label={ISSUE_LABEL}={task}"]
    if running:
        args += ["--filter", _RUNNING_FILTER]
    args += ["--format", _PS_FORMAT]

    handles: list[Handle] = []
    for line in docker(*args).splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < _PS_FIELDS:
            continue
        container_id, name, image, run_label, issue_label, state = (
            part.strip() for part in fields[:_PS_FIELDS]
        )
        handles.append(
            Handle(
                id=container_id,
                run_id=run_label,
                issue=int(issue_label) if issue_label.isdigit() else None,
                name=name,
                image=image,
                state=state,
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
#:
#: The two `SWARM_WORKER_*` knobs are here because the worker reads `Settings`
#: *inside* the container: an operator's `export SWARM_WORKER_CTX=32768` that
#: silently stopped at the container boundary would leave the tuning tables in
#: README/SETUP.md describing a knob that turns nothing - the worker would run
#: at the baked-in default while the host believed otherwise (observed live:
#: an over-long prompt truncated at 16K on a host tuned past it). Both values
#: are non-secret, so either `--env` form `_env_flags` emits is fine.
INHERITED_ENV = ("GITHUB_TOKEN", "OLLAMA_HOST", "SWARM_WORKER_CTX", "SWARM_WORKER_MODEL")


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
        task: TaskRef,
        base_commit: str,
        *,
        issue: int | None = None,
        image: str | None = None,
        entrypoint: str | None = None,
        command: Sequence[str] | None = None,
    ) -> Handle:
        """Create and start one worker container for `task`, at `base_commit`.

        Created and started as two steps rather than one `docker run -d`: a
        container that was created and then failed to start is still a
        container, and this way there is a handle to dispose it with instead of
        an id that only exists inside a failed command's output.

        **`task` and `issue` are two different things and both are needed.**
        `task` is what the *container* is: it labels and names it, and `find`
        matches on it, all through `TaskRef.label_value` so this module never
        learns a tracker's id format. `issue` is what the *worker* is told, on a
        command line that reaches a process which opens pull requests and closes
        issues - the worker is a GitHub client, and ADR 0001 keeps that. Passing
        one and deriving the other is what would put the coupling back, silently.

        `issue` is therefore required exactly when the default worker command is
        used; a caller supplying its own `command` - a probe - needs neither.

        `image` is per *task*, not per manager: #98 lets an issue declare its
        stack and the toolchain that stack needs is not in one image. The
        manager's own `image` stays the default, because a run whose tasks
        never declare anything must keep working unchanged.

        `entrypoint` and `command` override what the image runs. The dispatcher
        never passes either - they exist so a caller can put a probe container
        under this run's labels, limits and redaction, which is what the
        integration tests do and what a `swarm doctor` check would want.
        """
        image = image or self.image
        token = task.label_value
        name = self._container_name(token)
        args = [
            "create",
            "--name", name,
            # The worker cannot ask the daemon which image it is running in -
            # it has no socket and no `docker` binary, which is the containment
            # working - so #97's result record can only name it if it is told.
            "--env", f"{IMAGE_ENV}={image}",
            # PID 1 in the worker is the entrypoint, which spawns git and the
            # verify command; without an init it reaps none of them and a
            # finished task can sit on zombies until the pids limit bites.
            "--init",
            "--label", f"{RUN_LABEL}={self.run.id}",
            "--label", f"{ISSUE_LABEL}={token}",
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
        args.append(image)
        args += list(command) if command is not None else self._worker_args(issue, base_commit)

        created = self._cli(*args).strip().splitlines()
        if not created or not created[-1]:
            raise ContainerError(f"docker create returned no container id for {task}")

        handle = Handle(
            id=created[-1].strip(),
            run_id=self.run.id,
            # `Handle.issue` is the artifact layer's field, not the container's
            # identity - `worker/result.py` files a record under it and
            # `artifacts.py` names a log after it, both of which are the code
            # host's own numbering. It is the number that was passed, never one
            # read back out of the ref.
            issue=None if issue is None else int(issue),
            name=name,
            image=image,
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

    def find(self, *, ref: TaskRef | None = None, running: bool = False) -> list[Handle]:
        """This run's containers, running or not, optionally one task's only.

        The filter is `ref.label_value`, the same token `spawn` labelled the
        container with. The ref is not taken apart and no adapter is consulted,
        which is the property that lets a run against a tracker whose ids are
        `ENG-123` find its own containers.

        `running=True` narrows the answer to live workers - see
        `find_containers`, which explains why that is not the default. Whatever
        comes back, each handle carries the `state` the daemon reported.
        """
        return find_containers(
            self._cli,
            run_id=self.run.id,
            task=None if ref is None else ref.label_value,
            running=running,
        )

    # --- plumbing -------------------------------------------------------

    @property
    def _cli(self) -> DockerCLI:
        assert self.docker is not None  # set in __post_init__
        return self.docker

    def _worker_args(self, issue: int | None, base_commit: str) -> list[str]:
        """The three arguments `Dockerfile.worker`'s entrypoint documents.

        The clone URL carries no credential. The token travels as an
        environment variable (#28), which keeps it out of the container's own
        `Args` - where `docker inspect` prints it, where the worker's every git
        error would quote it back, and where it would survive in the image's
        history of a `docker commit`.
        """
        if issue is None:
            # Reachable only by a caller that wanted the default worker command
            # without saying which issue it is for, which the worker cannot run:
            # `--issue` is how it finds the contract to satisfy.
            raise ContainerError("spawning a worker needs an issue number; pass issue=")
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

    def _container_name(self, token: str) -> str:
        """`apiary-<run>-issue-<token>-<suffix>`; `token` is Docker-safe already.

        Docker names match `[a-zA-Z0-9][a-zA-Z0-9_.-]*`, which is exactly the
        set `TaskRef.label_value` reduces a ref to - so a name is a `#`-free
        rendering of a ref without this function knowing what a `#` means.
        """
        tail = "".join(_secrets.choice(SUFFIX_ALPHABET) for _ in range(NAME_SUFFIX_LENGTH))
        return f"apiary-{self.run.id}-issue-{token}-{tail}"


def _inherited_env() -> dict[str, str]:
    return {name: os.environ[name] for name in INHERITED_ENV if os.environ.get(name)}
