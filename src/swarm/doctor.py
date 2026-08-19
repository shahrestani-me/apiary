"""Every environmental precondition, checked before anything expensive runs.

The failures this module exists to catch all share one property: **none of them
looks like what it is.** An absent model presents as a planner that returns
nothing useful. A token missing a scope presents as a permissions bug in
whichever module happened to make the first write. A target repo with no CI
presents, an hour later, as issues that reach `swarm:review` and stay there
forever. Every one of them is cheap to detect up front and expensive to
diagnose from the symptom, which is the entire argument for a preflight.

So each check answers with a verdict *and the command that fixes it*. A check
that reports "ollama.models: failed" has moved the hour of confusion rather
than removed it; `Check.__post_init__` therefore refuses to construct a failing
check without a fix, which makes "every failure names its remedy" a property of
the type rather than a habit.

**Doctor is read-only, and that is load-bearing.** It runs against a live
tracker, a live daemon and a live model server, at the moment when the operator
has the least confidence that any of them is configured correctly - which is
the worst possible moment to also be creating things. Concretely:

- `github/labels.py` can create the missing `swarm:*` labels, and this module
  deliberately does not call it. It imports `SWARM_LABELS` and
  `list_label_names` - the name list and the read - and reports
  `python -m swarm.github.labels <repo>` as the fix. Provisioning is a decision
  someone makes; a diagnostic that quietly repaired the thing it was asked to
  measure would report a repo that was never in the state it just described.
- Write permission on the repository is *not* probed, for the same reason: the
  only honest probe is a write. `check_token` asserts the token's shape through
  `security.assert_scoped_token` and names the four permissions it must carry;
  what it cannot do is prove they are granted without spending one.
- The Docker checks run `version` and `image inspect` and nothing else. No
  container is created, started or removed.

`tests/test_doctor.py` asserts both halves of that: every request the GitHub
checks make is a `GET`, and every `docker` argv is a read.

**Three checks are here that the ticket did not ask for**, each because a run
tripped over it:

- `docker.cli` - the orchestrator image installs `git` and no `docker` binary,
  and `containers/manager.py` reaches the daemon by shelling out to that
  binary. `DOCKER_HOST` is honoured by the CLI, so with no CLI the socket proxy
  is configured, reachable, and dialled by nobody. The daemon check cannot see
  this: it fails identically to a stopped daemon, and the fix is completely
  different.
- `github.ci` - `docs/architecture-v2.md` makes CI the gate ("PRs are the
  integration mechanism"), so against a repo with no checks nothing can ever
  leave `swarm:review`. That is a repo the swarm should refuse to start on, not
  one it should discover after eight issues are stuck.
- `ollama.target` - Ollama spells a server *bind* address and a client *target*
  with the same `OLLAMA_HOST`. `config.py` reads it as the target, and the
  value that lets containers reach the host server is `0.0.0.0:11434`, which as
  a target points at nothing. `compose.yaml` sidesteps it with
  `APIARY_OLLAMA_HOST`; a process started from a shell that exported the bind
  address does not, and gets a connection error that reads like a dead server.

**The tracker checks are the same argument, one integration further out.**
ADR 0001 reaches a customer's task system through *their* MCP server, named in
a capability contract (`mcp/contract.py`, #150). Everything that can be wrong
with that arrangement fails late and reads as something else: a tool name the
server does not have surfaces on the first cycle that needs it, a credential
that expired surfaces as an orchestrator that has gone quiet, and a host
missing from the egress allowlist is answered `403 Filtered` by the proxy,
which reads exactly like the server refusing the request. So `tracker.*` asks
all four questions up front - the block is valid, the server answers, the
credential is accepted, every named tool exists - and each answer names its own
remedy, including the *per-server* command that mints a credential, because
those differ and "401" on its own is not actionable.

Three notes on how those four are cut, each of them a distinction that would
otherwise be lost:

- **A 401 proves reachability.** The probe is built without a credential when
  none is exported, so a server that refuses it has still answered - and
  "unreachable" and "unauthorized" stay separate verdicts with separate fixes
  rather than one confusing one.
- **No tracker configured is not a failure.** apiary runs on the label control
  plane until #152, so an installation with no contract is a normal one; it
  gets a single skip that says so, and nothing that depends on it is reported
  as broken.
- **Still read-only, and now against somebody else's system.** The tracker
  probe makes exactly two calls, `initialize` and `tools/list`. It never calls
  a tool: `tools/call` on a tracker is a comment somebody receives or a ticket
  somebody triages, which is the most expensive way this module could break its
  own rule. `tests/test_doctor.py` asserts the recorded call list.

Manual run against a real repo - reads only, writes nothing:

    GITHUB_TOKEN=... python -m swarm.doctor shahrestani-me/apiary

`swarm doctor` itself is one `sub.add_parser` away, in `cli.py`, which is
outside this issue's `## Files`; `main` below is already the shape that
subcommand needs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping, Protocol, Sequence

from pydantic import BaseModel

from .config import SETTINGS, TRACKER_CONFIG_ENV, Settings
from .containers.manager import (
    DEFAULT_STACK_IMAGES,
    STACK_LABEL,
    WORKER_IMAGE,
    ContainerError,
    DockerCLI,
    Redactor,
    StackImages,
    build_hint,
)
from .github.client import GitHubClient, GitHubError, GitHubHTTPError
from .github.labels import SWARM_LABELS, list_label_names
from .llm import orchestrator_llm, structured, worker_llm
from .mcp.client import McpAuthError, McpEgressBlocked, McpError, ServerInfo, ToolSpec
from .mcp.contract import (
    CAPABILITIES,
    ContractError,
    TrackerContract,
    client_for,
    load_tracker,
)
from .security import (
    PROVISION_PERMISSIONS,
    PROVISION_TOKEN_ENV,
    REQUIRED_PERMISSIONS,
    CredentialError,
    assert_provision_token,
    assert_scoped_token,
)

__all__ = [
    "DoctorError",
    "Check",
    "Diagnosis",
    "Doctor",
    "HostInference",
    "Inference",
    "TrackerProbe",
    "OK",
    "FAIL",
    "SKIP",
    "main",
    "preflight",
    "stack_check",
]


class DoctorError(RuntimeError):
    """A probe could not answer. Never a verdict - verdicts are `Check`s."""


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

OK = "ok"
FAIL = "fail"
SKIP = "skip"

#: Check names. Constants because a test addresses a check by name and an
#: operator greps for one; a renamed literal in three places is a rename that
#: only two of them get.
CHECK_OLLAMA_TARGET = "ollama.target"
CHECK_OLLAMA_REACHABLE = "ollama.reachable"
CHECK_OLLAMA_MODELS = "ollama.models"
CHECK_OLLAMA_SCHEMA = "ollama.schema"
CHECK_TOKEN = "github.token"
CHECK_BOOT_TOKEN = "github.boot-token"
CHECK_REPO = "github.repo"
CHECK_LABELS = "github.labels"
CHECK_CI = "github.ci"
CHECK_TIMEOUTS = "config.timeouts"
CHECK_DOCKER_CLI = "docker.cli"
CHECK_DOCKER_DAEMON = "docker.daemon"
#: The capability contract (#150), in the four ways it can be wrong. Four
#: rather than one because their remedies have nothing in common: edit a file,
#: open an egress hole, mint a credential, rename a tool.
CHECK_TRACKER_CONFIG = "tracker.config"
CHECK_TRACKER_REACHABLE = "tracker.reachable"
CHECK_TRACKER_AUTH = "tracker.auth"
CHECK_TRACKER_TOOLS = "tracker.tools"
#: The prefix a per-stack image check reports under: `docker.image.node`. One
#: per stack rather than one `docker.image`, because #99 chooses the image per
#: task - "the worker image is present" stopped being a single fact about a
#: host the moment a plan could reference two stacks.
CHECK_WORKER_IMAGE = "docker.image"


def stack_check(stack: str) -> str:
    """`docker.image.node`. A function rather than a table, because the stacks
    a run needs come from its plan, not from a list this module holds."""
    return f"{CHECK_WORKER_IMAGE}.{stack}"


_MARK = {OK: "ok  ", FAIL: "FAIL", SKIP: "skip"}
#: Wide enough for the longest fixed name and for a per-stack one. Computed
#: from the stacks that exist rather than a magic number, so adding a stack
#: with a long id cannot silently ragged-edge the report.
_NAME_WIDTH = max(
    len(name)
    for name in (
        CHECK_OLLAMA_TARGET, CHECK_OLLAMA_REACHABLE, CHECK_OLLAMA_MODELS, CHECK_OLLAMA_SCHEMA,
        CHECK_TOKEN, CHECK_BOOT_TOKEN, CHECK_REPO, CHECK_LABELS, CHECK_CI, CHECK_TIMEOUTS,
        CHECK_DOCKER_CLI, CHECK_DOCKER_DAEMON,
        CHECK_TRACKER_CONFIG, CHECK_TRACKER_REACHABLE, CHECK_TRACKER_AUTH, CHECK_TRACKER_TOOLS,
        *(stack_check(stack) for stack in DEFAULT_STACK_IMAGES),
    )
)


@dataclass(frozen=True)
class Check:
    """One precondition, its verdict, and - when it failed - its remedy.

    The constructor enforces the "done when" of this module's ticket: a failing
    check without a `fix` cannot be built. Every failure below therefore names
    a command, a variable to export, or a file to add, and a future check that
    forgets to fails the suite at the point it is written rather than at the
    point somebody needed it.

    `skip` is a third state on purpose. "The worker image is not on this
    daemon" and "there is no daemon to ask" are different sentences, and
    collapsing the second into a failure of the first sends the reader after
    the wrong thing - which is precisely the confusion this module exists to
    end.
    """

    name: str
    status: str
    detail: str
    fix: str = ""

    def __post_init__(self) -> None:
        if self.status not in (OK, FAIL, SKIP):
            raise ValueError(f"{self.name}: unknown status {self.status!r}")
        if self.status == FAIL and not self.fix.strip():
            raise ValueError(f"{self.name}: a failing check must name the fix for it")

    @classmethod
    def passed(cls, name: str, detail: str) -> Check:
        return cls(name, OK, detail)

    @classmethod
    def failed(cls, name: str, detail: str, *, fix: str) -> Check:
        return cls(name, FAIL, detail, fix)

    @classmethod
    def skipped(cls, name: str, detail: str) -> Check:
        return cls(name, SKIP, detail)

    @property
    def ok(self) -> bool:
        return self.status == OK

    def lines(self) -> list[str]:
        """The verdict, and its remedy indented under the same column.

        A fix may be more than one line - the useful shape for one is a
        sentence and then the command on its own, which is how `mcp/contract.py`
        writes every refusal it hands over. Continuation lines are padded to the
        same column rather than printed flush left, because the report is read
        as a table and a stray line at column zero reads as a new check.
        """
        head = f"{_MARK[self.status]}  {self.name:<{_NAME_WIDTH}}  {self.detail}"
        if not self.fix:
            return [head]
        pad = f"{'':6}{'':<{_NAME_WIDTH}}  "
        first, *rest = self.fix.splitlines()
        return [head, f"{pad}fix: {first}", *(f"{pad}     {line.strip()}" for line in rest)]

    def __str__(self) -> str:
        return "\n".join(self.lines())


@dataclass(frozen=True)
class Diagnosis:
    """Every check of one run, and the single question the caller asked.

    Skips do not fail a run. A skipped check is one this machine could not
    answer - the daemon it needed is down, and the check that says so has
    already failed - so counting it again would report two problems where the
    operator has one.
    """

    checks: tuple[Check, ...] = ()

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if check.status == FAIL)

    @property
    def skipped(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if check.status == SKIP)

    @property
    def ok(self) -> bool:
        return not self.failures

    def by_name(self, name: str) -> Check:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)

    def summary(self) -> str:
        if self.ok and not self.skipped:
            return f"all {len(self.checks)} preconditions met"
        parts = [f"{len(self.checks) - len(self.failures) - len(self.skipped)} ok"]
        if self.failures:
            parts.append(f"{len(self.failures)} failed")
        if self.skipped:
            parts.append(f"{len(self.skipped)} not attempted")
        return f"{len(self.checks)} checks: " + ", ".join(parts)

    def report(self) -> str:
        lines: list[str] = []
        for check in self.checks:
            lines += check.lines()
        lines += ["", f"» {self.summary()}"]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The model server
# --------------------------------------------------------------------------


class Ping(BaseModel):
    """The trivial schema the format probe forces a model to fill.

    Two fields of different types, because the failure being tested for is a
    model that emits *plausible JSON of its own shape* rather than the one it
    was constrained to. A single boolean is guessable; `{"ok": "yes"}` is the
    exact answer a model that ignored the constraint produces, and pydantic
    rejects it.
    """

    ok: bool
    answer: int


class Inference(Protocol):
    """What doctor asks of Ollama. The seam that keeps the suite hermetic.

    Same shape and same reason as `github.client.Transport` and
    `containers.manager.Runner`: the interesting logic here is the wording of
    the verdicts, and a test that needs a running model server to reach it is a
    test that runs on one machine.
    """

    def version(self) -> str: ...

    def installed(self) -> list[str]: ...

    def schema_probe(self, role: str) -> str: ...


#: The two roles `config.py` defines: the settings field naming the model, and
#: the `llm.py` factory that builds it. The factories are used rather than a
#: `ChatOllama` constructed here, so the probe exercises the client the
#: orchestrator and the workers actually get - same base URL, same `num_ctx`,
#: same temperature. A probe against a lookalike proves nothing about the one
#: that runs.
ROLES: dict[str, tuple[str, Callable[[], Any]]] = {
    "orchestrator": ("orchestrator_model", orchestrator_llm),
    "worker": ("worker_model", worker_llm),
}

PROBE_PROMPT = "Return ok=true and answer=7."


@dataclass
class HostInference:
    """The real one: HTTP to the configured Ollama, and one call per model.

    `/api/version` and `/api/tags` are stdlib HTTP because they are two GETs
    and because they must work when the *reason* nothing works is that the
    model client cannot be constructed at all.
    """

    settings: Settings = SETTINGS
    timeout_s: float = 5.0
    fetch: Callable[[str, float], bytes] | None = None

    def __post_init__(self) -> None:
        if self.fetch is None:
            self.fetch = _fetch

    @property
    def base_url(self) -> str:
        return self.settings.ollama_base_url.rstrip("/")

    def version(self) -> str:
        return str(self._get("/api/version").get("version") or "unknown")

    def installed(self) -> list[str]:
        models = self._get("/api/tags").get("models") or []
        return [str(entry.get("model") or entry.get("name") or "") for entry in models]

    def schema_probe(self, role: str) -> str:
        """Force `Ping` out of one role's model. Raises `DoctorError` if it will not.

        Broad `except` on purpose, and the only one in this module: everything
        between here and the socket - langchain, pydantic, the HTTP client -
        signals a refusal its own way, and a probe that let one of those escape
        would abort the remaining checks over the very condition it exists to
        report.
        """
        _, factory = ROLES[role]
        try:
            answer = structured(factory(), Ping).invoke(PROBE_PROMPT)
        except Exception as exc:  # noqa: BLE001 - see the docstring
            raise DoctorError(f"{type(exc).__name__}: {exc}") from exc
        if not isinstance(answer, Ping):
            # The value is not asserted, only the shape. Whether the model can
            # count to seven is not this project's problem; whether Ollama's
            # `format` constrained its decoding is.
            raise DoctorError(f"returned {type(answer).__name__}, not the requested schema")
        return f"{answer.__class__.__name__} parsed"

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        fetch = self.fetch or _fetch
        try:
            payload = json.loads(fetch(url, self.timeout_s).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise DoctorError(f"GET {url} -> {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise DoctorError(f"GET {url}: {getattr(exc, 'reason', exc)}") from exc
        except ValueError as exc:
            # Two arrive here: a body that is not JSON, and a base URL with no
            # scheme, which `urlopen` rejects before it opens anything. The
            # second is what `ollama.target` is about.
            raise DoctorError(f"GET {url}: {exc}") from exc
        return payload if isinstance(payload, dict) else {}


def _fetch(url: str, timeout_s: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "apiary-swarm"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------

#: Hosts that are a bind address rather than somewhere to dial. `0.0.0.0` is
#: what SETUP.md tells the operator to give the *server* so containers can
#: reach it, and handing it to a client points that client at itself.
WILDCARD_HOSTS: tuple[str, ...] = ("0.0.0.0", "::", "[::]", "")

#: Where the host's Ollama is, seen from inside a container. The one
#: `Dockerfile` and `Dockerfile.worker` both bake in.
CONTAINER_OLLAMA_URL = "http://host.docker.internal:11434"

LOOPBACK_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1", "[::1]")

#: The remedy for both of the wrong-target shapes, and the paragraph this whole
#: check exists for. It names three things because the mistake has three parts:
#: which variable this process reads, where the *server's* bind address belongs
#: instead, and why compose spells its own override differently.
_TARGET_FIX = (
    "Ollama spells the server's bind address and the client's target with the same "
    "OLLAMA_HOST, and config.py reads it as the target. Give this process a URL - "
    "`export OLLAMA_HOST=http://localhost:11434` - and give the server its bind address "
    "through the server's own environment (`launchctl setenv OLLAMA_HOST 0.0.0.0:11434` "
    "on macOS, where the app never reads your shell). compose.yaml reads "
    "APIARY_OLLAMA_HOST for the container's copy, for exactly this reason"
)

#: Marks a filesystem as a container's. Cheap, and wrong only in the direction
#: that matters least: a false negative reports one fewer check.
DOCKERENV = "/.dockerenv"

DOCKER_BINARY = "docker"

#: The ref whose check runs answer "does anything gate a PR here". A branch
#: name is a valid ref for `GET /commits/{ref}/check-runs`.
DEFAULT_CI_REF = "main"

#: `security.SOCKET_PROXY_ENV` sets `IMAGES=0`, so `docker image inspect`
#: through the proxy is a 403 by design rather than a missing image.
_DENIED_MARKERS: tuple[str, ...] = ("403", "forbidden")


class TrackerProbe(Protocol):
    """What doctor asks of an MCP server, and the whole of it.

    Two reads and a teardown. `McpClient` satisfies this structurally, and the
    narrowness is the point rather than an accident of what the tests needed:
    the type an operator can see makes it obvious that a preflight against
    somebody's live Jira cannot file a ticket in it, because there is no method
    here that would.
    """

    def connect(self) -> ServerInfo: ...

    def list_tools(self) -> list[ToolSpec]: ...

    def close(self) -> None: ...


@dataclass
class Doctor:
    """Every check, and the collaborators each of them needs.

    Constructed with its collaborators rather than reaching for globals, so a
    test provokes a failure by handing over a double instead of by breaking the
    machine it runs on. `from_env` is the production wiring.
    """

    repo: str | None = None
    settings: Settings = SETTINGS
    github: GitHubClient | None = None
    docker: DockerCLI | None = None
    inference: Inference | None = None
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    which: Callable[[str], str | None] = shutil.which
    in_container: bool | None = None
    image: str = WORKER_IMAGE
    #: Which image carries which stack, and which stacks this run needs. The
    #: second is a *plan* fact, so it is passed in rather than assumed: a
    #: Python-only backlog must not be told to build a Node image it will
    #: never spawn. Defaulting to every known stack is the honest answer for a
    #: bare `swarm doctor`, which has no plan to read.
    images: StackImages = field(default_factory=StackImages)
    stacks: Sequence[str] = tuple(sorted(DEFAULT_STACK_IMAGES))
    ci_ref: str = DEFAULT_CI_REF
    probe_schema: bool = True
    #: The capability contract, or None when this installation configures no
    #: tracker - which is still the normal case until #152 removes the label
    #: control plane, and is reported as a skip rather than as a failure.
    tracker: TrackerContract | None = None
    #: Why there is no contract, when a file said there should be one. Carried
    #: as a string because `from_env` must not raise: an unparseable block is
    #: precisely what this module exists to report.
    tracker_error: str = ""
    #: The seam, same shape and same reason as `inference` and `github`. Left
    #: unset in production, where `check_tracker_*` builds an `McpClient` from
    #: the contract itself.
    mcp: TrackerProbe | None = None
    _probe: tuple[Any, Exception | None] | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.inference is None:
            self.inference = HostInference(self.settings)
        if self.docker is None:
            # Redacting even here. A `docker` failure is quoted verbatim into a
            # report an operator pastes into an issue, and the orchestrator's
            # environment is where the token lives.
            redactor = Redactor()
            redactor.add_env(self.env)
            self.docker = DockerCLI(redact=redactor)
        if self.in_container is None:
            self.in_container = os.path.exists(DOCKERENV)

    @classmethod
    def from_env(cls, repo: str | None = None, **kwargs: Any) -> Doctor:
        """Wire the real probes up from `GITHUB_TOKEN` and `GITHUB_REPOSITORY`.

        A missing or malformed target leaves `github` unset rather than
        raising: "there is no usable token" is a verdict this module reports,
        not an error it dies of. The tracker contract is loaded the same way
        and for the same reason - a block that will not parse is the single
        most likely thing to be wrong on the run where somebody types `swarm
        doctor`, and dying of it would report nothing else at all.
        """
        env: Mapping[str, str] = kwargs.pop("env", None) or dict(os.environ)
        repo = repo or env.get("GITHUB_REPOSITORY") or None
        token = env.get("GITHUB_TOKEN")
        github = kwargs.pop("github", None)
        if github is None and token and repo and _is_repo(repo):
            github = GitHubClient(repo, token)

        tracker = kwargs.pop("tracker", None)
        tracker_error = kwargs.pop("tracker_error", "")
        if tracker is None and not tracker_error:
            try:
                tracker = load_tracker(env=env)
            except ContractError as exc:
                tracker_error = str(exc)
        return cls(
            repo=repo,
            github=github,
            env=env,
            tracker=tracker,
            tracker_error=tracker_error,
            **kwargs,
        )

    # --- the run --------------------------------------------------------

    def run(self) -> Diagnosis:
        """Every check, in dependency order, and never fewer than all of them.

        Nothing short-circuits the *run*: a broken Ollama must not hide a
        broken token, because the operator would then fix one thing, re-run,
        and be told about the next. Only checks whose prerequisite failed are
        skipped, and they say which prerequisite.
        """
        target = self.check_ollama_target()
        # Reachability chains off the target rather than being asked anyway: a
        # bind address produces a connection error that names `ollama serve`,
        # and following that advice fixes nothing. One problem, one verdict.
        reachable = self._after(target, CHECK_OLLAMA_REACHABLE, self.check_ollama_reachable)
        models = self._after(reachable, CHECK_OLLAMA_MODELS, self.check_models)
        checks: list[Check] = [
            target,
            reachable,
            models,
            self._after(models, CHECK_OLLAMA_SCHEMA, self.check_schema),
        ]

        token = self.check_token()
        access = self._after(token, CHECK_REPO, self.check_repo_access)
        checks += [
            token,
            self.check_boot_token(),
            access,
            self._after(access, CHECK_LABELS, self.check_labels),
            self._after(access, CHECK_CI, self.check_ci),
        ]

        cli = self.check_docker_cli()
        daemon = self._after(cli, CHECK_DOCKER_DAEMON, self.check_docker_daemon)
        checks += [self.check_timeouts(), cli, daemon]
        # One per stack rather than one for `apiary-worker`: #99 chooses the
        # image per task, so "the worker image is present" stopped being a
        # single fact about a host the moment a plan could reference two.
        for stack in self.stacks:
            checks.append(
                self._after(daemon, stack_check(stack), partial(self.check_stack_image, stack))
            )

        checks += self.tracker_checks()
        return Diagnosis(tuple(checks))

    def tracker_checks(self) -> list[Check]:
        """The capability contract, in the four ways it can be wrong.

        One check rather than four when nothing is configured, and that is a
        judgement rather than tidiness: three further lines reading "not
        attempted: tracker.config did not pass" describe an installation that
        has nothing wrong with it, and a preflight that reports three
        non-problems on every run is one people stop reading.

        The probe is closed at the end whatever the verdicts were. A stdio
        contract's client is a *subprocess*, and a diagnostic that left one
        running per invocation would be a leak in the tool an operator reaches
        for when they already suspect their machine.
        """
        config = self.check_tracker_config()
        if self.tracker is None:
            return [config]
        try:
            reachable = self._after(config, CHECK_TRACKER_REACHABLE, self.check_tracker_reachable)
            auth = self._after(reachable, CHECK_TRACKER_AUTH, self.check_tracker_auth)
            tools = self._after(auth, CHECK_TRACKER_TOOLS, self.check_tracker_tools)
            return [config, reachable, auth, tools]
        finally:
            self._close_probe()

    @staticmethod
    def _after(prior: Check, name: str, check: Callable[[], Check]) -> Check:
        """Run `check`, or explain which earlier verdict makes it unanswerable."""
        if prior.ok:
            return check()
        return Check.skipped(name, f"not attempted: {prior.name} did not pass")

    # --- ollama ---------------------------------------------------------

    def check_ollama_target(self) -> Check:
        """Is the configured URL something a *client* can dial?

        Costs no I/O and runs before the reachability check, because when this
        one is wrong the reachability failure is a connection error that reads
        like a stopped server and sends the reader to `ollama serve`.
        """
        url = self.settings.ollama_base_url
        host = _hostname(url)

        if "://" not in url:
            return Check.failed(
                CHECK_OLLAMA_TARGET,
                f"OLLAMA_HOST={url!r} has no scheme, which is the shape of a bind address",
                fix=_TARGET_FIX,
            )
        if host in WILDCARD_HOSTS:
            return Check.failed(
                CHECK_OLLAMA_TARGET,
                f"OLLAMA_HOST={url!r} names a wildcard bind address; a client cannot dial it",
                fix=_TARGET_FIX,
            )
        if self.in_container and host in LOOPBACK_HOSTS:
            return Check.failed(
                CHECK_OLLAMA_TARGET,
                f"OLLAMA_HOST={url!r} is this container's own loopback, and Ollama "
                f"runs on the host (docs/architecture-v2.md, first constraint)",
                fix=(
                    f"export APIARY_OLLAMA_HOST={CONTAINER_OLLAMA_URL} before "
                    f"`docker compose run orchestrator ...`; compose passes it through as "
                    f"the container's OLLAMA_HOST"
                ),
            )
        return Check.passed(CHECK_OLLAMA_TARGET, f"client target is {url}")

    def check_ollama_reachable(self) -> Check:
        assert self.inference is not None  # set in __post_init__
        try:
            version = self.inference.version()
        except DoctorError as exc:
            return Check.failed(
                CHECK_OLLAMA_REACHABLE,
                f"no answer from {self.settings.ollama_base_url}: {exc}",
                fix=(
                    "start the server - `ollama serve`, or launch the Ollama app - and "
                    f"confirm with `curl {self.settings.ollama_base_url}/api/version`. "
                    "It runs on the HOST, never in a container: Docker Desktop has no "
                    "Metal passthrough (docs/architecture-v2.md)"
                ),
            )
        return Check.passed(
            CHECK_OLLAMA_REACHABLE, f"ollama {version} at {self.settings.ollama_base_url}"
        )

    def check_models(self) -> Check:
        assert self.inference is not None
        try:
            installed = self.inference.installed()
        except DoctorError as exc:
            return Check.failed(
                CHECK_OLLAMA_MODELS,
                f"could not list models: {exc}",
                fix=f"`ollama list` against {self.settings.ollama_base_url} reproduces this",
            )

        wanted = self._wanted_models()
        missing = [(role, model) for role, model in wanted if not _has_model(installed, model)]
        if missing:
            return Check.failed(
                CHECK_OLLAMA_MODELS,
                ", ".join(f"{model} ({role}) is not pulled" for role, model in missing),
                fix=(
                    "; ".join(f"ollama pull {model}" for _, model in missing)
                    + "  (or point the role elsewhere: "
                    + ", ".join(f"SWARM_{role.upper()}_MODEL" for role, _ in missing)
                    + ")"
                ),
            )
        return Check.passed(
            CHECK_OLLAMA_MODELS,
            ", ".join(f"{model} ({role})" for role, model in wanted),
        )

    def check_schema(self) -> Check:
        """Does schema-forced JSON actually constrain these models?

        The whole orchestrator is `structured()` calls. A model that accepts
        the `format` parameter and then ignores it produces empty plans and
        unparseable judgements, which is indistinguishable from a planner bug
        and is where an afternoon goes.

        This is the one check that costs real inference - two model loads, and
        with `OLLAMA_MAX_LOADED_MODELS=1` a swap between them - so it is the
        one with an off switch.
        """
        assert self.inference is not None
        if not self.probe_schema:
            return Check.skipped(CHECK_OLLAMA_SCHEMA, "not attempted: --skip-schema")

        results: list[str] = []
        for role, model in self._wanted_models():
            try:
                results.append(f"{model} ({role}) {self.inference.schema_probe(role)}")
            except DoctorError as exc:
                return Check.failed(
                    CHECK_OLLAMA_SCHEMA,
                    f"{model} ({role}) did not return the requested schema: {exc}",
                    fix=(
                        f"`ollama show {model}` - a model without structured-output support "
                        f"cannot orchestrate; set SWARM_{role.upper()}_MODEL to one that has "
                        f"it, and check `ollama --version` (format-constrained decoding needs "
                        f"a recent server)"
                    ),
                )
        return Check.passed(CHECK_OLLAMA_SCHEMA, "; ".join(results))

    def _wanted_models(self) -> list[tuple[str, str]]:
        return [(role, getattr(self.settings, attr)) for role, (attr, _) in ROLES.items()]

    # --- github ---------------------------------------------------------

    def check_token(self) -> Check:
        """The shape of the credential, before anything spends it.

        `security.assert_scoped_token` is the whole check and this is its first
        production caller: a classic or OAuth token works perfectly against the
        target repo and also reaches every other repo the account can, so the
        failure it prevents is not one the remaining checks could ever see.
        """
        try:
            kind = assert_scoped_token(self.env.get("GITHUB_TOKEN"))
        except CredentialError as exc:
            return Check.failed(CHECK_TOKEN, str(exc), fix=self._token_fix())
        return Check.passed(CHECK_TOKEN, f"a {kind} token is set")

    def check_boot_token(self) -> Check:
        """Is a boot key present, and is it a shape that can be repo-scoped?

        Reported rather than required, because it is only needed to *create* a
        repository: a run against a repo that already exists never touches it.
        Skipping loudly beats failing, since most runs legitimately have no
        boot key - but staying silent would leave `--new` failing with a 403
        from GitHub, three steps into something that has already created
        nothing.
        """
        token = self.env.get(PROVISION_TOKEN_ENV)
        if not token:
            return Check.skipped(
                CHECK_BOOT_TOKEN,
                f"{PROVISION_TOKEN_ENV} is not set; `swarm run --new` cannot create a "
                f"repository, and every other command is unaffected",
            )
        try:
            kind = assert_provision_token(token)
        except CredentialError as exc:
            return Check.failed(
                CHECK_BOOT_TOKEN,
                str(exc),
                fix=(
                    f"mint a second fine-grained token with "
                    f"{', '.join(f'{k}:{v}' for k, v in sorted(PROVISION_PERMISSIONS.items()))} "
                    f"and export it as {PROVISION_TOKEN_ENV} - see docs/security.md"
                ),
            )
        if token == self.env.get("GITHUB_TOKEN"):
            return Check.failed(
                CHECK_BOOT_TOKEN,
                "the boot key and the work key are the same token, so workers would "
                "hold `administration` and `workflows` - and a worker that can edit "
                "`.github/workflows` can rewrite the CI that judges its own work",
                fix=(
                    f"mint a separate token for {PROVISION_TOKEN_ENV}; the two are "
                    f"deliberately different credentials, not one used twice"
                ),
            )
        return Check.passed(CHECK_BOOT_TOKEN, f"a {kind} boot key is set, distinct from the work key")

    def check_repo_access(self) -> Check:
        """Can this token read this repository's ledger?

        Reads the issues, because that is what the orchestrator does on every
        cycle - a check against a cheaper endpoint would prove less about the
        call that matters. Write access is deliberately unprobed: the only
        proof is a write, and this module makes none.
        """
        if not self.repo:
            return Check.failed(
                CHECK_REPO,
                "no target repository",
                fix="pass one - `python -m swarm.doctor owner/name` - or export GITHUB_REPOSITORY",
            )
        if not _is_repo(self.repo):
            return Check.failed(
                CHECK_REPO,
                f"{self.repo!r} is not a repository reference",
                fix="spell it owner/name, as in shahrestani-me/apiary",
            )
        if self.github is None:
            return Check.skipped(CHECK_REPO, f"not attempted: no client (see {CHECK_TOKEN})")

        try:
            issues = self.github.list_issues(state="open")
        except GitHubHTTPError as exc:
            return Check.failed(CHECK_REPO, f"{self.repo}: {exc}", fix=self._access_fix(exc.status))
        except GitHubError as exc:
            return Check.failed(
                CHECK_REPO,
                f"{self.repo}: {exc}",
                fix=(
                    "the API was not reachable, not refused: check the network, and - inside "
                    "compose - that api.github.com is in the egress proxy's FilterURL block "
                    "in compose.yaml. (APIARY_EGRESS_ALLOW is documented in several places "
                    "and read by none; the enforced list is the static one in compose.)"
                ),
            )
        return Check.passed(
            CHECK_REPO,
            f"{self.repo} readable, {len(issues)} open issues (write access is asserted "
            f"by token shape, never probed - doctor writes nothing)",
        )

    def check_labels(self) -> Check:
        """Are the six `swarm:*` labels there? Reported, never created.

        `POST /issues/{n}/labels` with an unknown name *invents* the label, so
        a missing one is not a crash: the run proceeds and the ledger fills
        with labels nobody chose, in colours nobody picked. That is why this is
        a preflight and not an exception handler.
        """
        assert self.github is not None  # guarded by `_after(access, ...)`
        try:
            present = {name.casefold() for name in list_label_names(self.github)}
        except GitHubError as exc:
            return Check.failed(
                CHECK_LABELS,
                f"could not list the labels of {self.repo}: {exc}",
                fix=self._access_fix(getattr(exc, "status", 0)),
            )

        missing = [spec.name for spec in SWARM_LABELS if spec.name.casefold() not in present]
        if missing:
            return Check.failed(
                CHECK_LABELS,
                f"{self.repo} is missing {', '.join(missing)}",
                fix=(
                    f"python -m swarm.github.labels {self.repo}  (creates only what is "
                    f"missing and touches nothing that exists; doctor reports rather than "
                    f"provisions, so that what it measured is what was there)"
                ),
            )
        return Check.passed(CHECK_LABELS, f"all {len(SWARM_LABELS)} swarm:* labels present")

    def check_ci(self) -> Check:
        """Does anything gate a PR on this repo?

        `docs/architecture-v2.md` makes CI the integration gate, and #23 waits
        on check runs before merging. Against a repo with none, every issue
        reaches `swarm:review` and stops - a swarm that looks busy and finishes
        nothing. Refusing to start is the cheaper outcome.

        Asked the same way the merge gate asks - check runs first, workflow
        runs when a fine-grained PAT is refused them - so an answer here is the
        answer #23 will get. A doctor that probed an endpoint the run never
        uses would pass while the gate stayed blind, or fail while it worked.
        """
        assert self.github is not None
        try:
            runs = self.github.list_check_runs(self.ci_ref)
        except GitHubHTTPError as exc:
            if exc.status == 404:
                return Check.failed(
                    CHECK_CI,
                    f"{self.repo} has no commit at {self.ci_ref!r}",
                    fix=f"name the default branch: --ci-ref <branch> (this run used {self.ci_ref!r})",
                )
            if exc.status != 403:
                return Check.failed(CHECK_CI, f"{self.repo}: {exc}", fix=self._access_fix(exc.status))
            # Expected for every least-privilege token: `checks` cannot be
            # granted to a fine-grained PAT. Fall through to the endpoint that
            # can, exactly as `checks.read_checks` does.
            try:
                runs = self.github.list_workflow_runs(self.ci_ref)
            except GitHubHTTPError as actions_exc:
                return Check.failed(
                    CHECK_CI,
                    f"{self.repo}: check runs and workflow runs are both unreadable "
                    f"({actions_exc})",
                    fix=self._access_fix(actions_exc.status),
                )

        if not runs:
            return Check.failed(
                CHECK_CI,
                f"no check runs on {self.ci_ref} of {self.repo}: nothing would gate a PR, "
                f"so no issue could ever leave swarm:review",
                fix=(
                    "add a workflow that runs the ## Verify command - .github/workflows/ci.yml, "
                    "`on: [push, pull_request]`. If yours triggers on pull_request only, point "
                    "this check at a recent PR head instead: --ci-ref <sha>"
                ),
            )
        names = sorted({str(run.get("name") or "?") for run in runs})
        return Check.passed(
            CHECK_CI,
            f"{len(runs)} check runs on {self.ci_ref}: {', '.join(names[:4])}",
        )

    def _token_fix(self) -> str:
        permissions = ", ".join(f"{name}:{level}" for name, level in REQUIRED_PERMISSIONS.items())
        target = self.repo or "the single target repo"
        return (
            f"mint a fine-grained PAT on {target} with {permissions} and nothing else "
            f"(https://github.com/settings/personal-access-tokens/new), then "
            f"`export GITHUB_TOKEN=github_pat_...` - see docs/security.md"
        )

    def _access_fix(self, status: int) -> str:
        """The remedy for a refusal, which differs entirely by status code."""
        if status == 401:
            return f"the token was rejected outright - it is expired or mistyped; {self._token_fix()}"
        if status == 403:
            return (
                "the token authenticates but is not permitted here: grant it "
                + ", ".join(f"{name}:{level}" for name, level in REQUIRED_PERMISSIONS.items())
                + f" on {self.repo}, or wait out a rate limit if that is what this is"
            )
        if status == 404:
            return (
                f"a fine-grained PAT sees only the repositories it was granted, and a 404 is "
                f"how it reports one it was not - never a 403: add {self.repo} under the "
                f"token's Repository access at "
                f"https://github.com/settings/personal-access-tokens, or check the spelling "
                f"of the repo itself"
            )
        return self._token_fix()

    # --- config ---------------------------------------------------------

    def check_timeouts(self) -> Check:
        """Can the inner clock ever be reached before the outer one fires?

        Costs no I/O and probes nothing - it is arithmetic over two environment
        variables - and it earns its place here for the reason every check here
        does: it fails as something else entirely. A container killed at the
        outer cap is recorded as a consumed attempt whose reason names the
        *container*, so an operator raising `SWARM_VERIFY_TIMEOUT` in response
        buys literally nothing and sees the same failure again with the same
        wording. The pair is only meaningful together, so it is checked
        together.

        `Settings.clock_conflict` owns the sentence, because the numbers and
        the reason they relate belong beside the defaults rather than beside
        the report that prints them.
        """
        conflict = self.settings.clock_conflict()
        if not conflict:
            return Check.passed(
                CHECK_TIMEOUTS,
                f"verify {self.settings.verify_timeout_s}s inside "
                f"worker {self.settings.worker_timeout_s}s",
            )
        return Check.failed(
            CHECK_TIMEOUTS,
            conflict,
            fix=(
                f"export SWARM_WORKER_TIMEOUT={max(self.settings.verify_timeout_s * 4, 1200)}  "
                f"# must exceed SWARM_VERIFY_TIMEOUT={self.settings.verify_timeout_s} "
                "with room for the clone, the inference call and the push"
            ),
        )

    # --- docker ---------------------------------------------------------

    def check_docker_cli(self) -> Check:
        """Is there a `docker` binary at all?

        Not a formality. `containers/manager.py` reaches the daemon by shelling
        out - deliberately, so that every call is one a human can paste - and
        `DOCKER_HOST` is honoured by *the binary*. The orchestrator image
        installs `git` and no docker CLI, so inside it the socket proxy is
        configured, running, reachable and dialled by nothing, and the failure
        arrives as a `ContainerError` from the first dispatch.
        """
        path = self.which(DOCKER_BINARY)
        if not path:
            host = self.env.get("DOCKER_HOST", "(unset)")
            return Check.failed(
                CHECK_DOCKER_CLI,
                f"no {DOCKER_BINARY!r} on PATH, so DOCKER_HOST={host} cannot be honoured",
                fix=(
                    "install the client in the image that runs the orchestrator - "
                    "`COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker` in "
                    "Dockerfile - or run the orchestrator on the host, where it is already "
                    "on PATH"
                ),
            )
        return Check.passed(CHECK_DOCKER_CLI, f"{DOCKER_BINARY} at {path}")

    def check_docker_daemon(self) -> Check:
        assert self.docker is not None  # set in __post_init__
        try:
            version = self.docker("version", "--format", "{{.Server.Version}}").strip()
        except ContainerError as exc:
            host = self.env.get("DOCKER_HOST", "(unset)")
            return Check.failed(
                CHECK_DOCKER_DAEMON,
                f"the daemon did not answer: {exc}",
                fix=(
                    "start Docker Desktop (or `colima start`), then `docker version`. Inside "
                    f"compose the daemon is the socket proxy, so check DOCKER_HOST={host} and "
                    "that the docker-socket-proxy service is up"
                ),
            )
        return Check.passed(CHECK_DOCKER_DAEMON, f"daemon {version or 'reachable'}")

    def check_stack_image(self, stack: str) -> Check:
        """Can this host run one stack's worker image?

        A missing image is not a runtime inconvenience. `SOCKET_PROXY_ENV` sets
        `IMAGES=0` and `BUILD=0`, so the orchestrator can neither pull nor build
        one - which means a missing image is a guaranteed all-infrastructure run,
        discovered mid-cycle, after a real repository already exists.

        Three answers, and the third is the reason this check is trustworthy at
        all: through the socket proxy the probe is *denied*, so it reports
        unanswered rather than reporting a missing image. Doctor is read-only
        and its own inability to look is not evidence about the host.

        **What it does not check is that the image can run the gate.** That
        means running it, and this module writes nothing and starts nothing -
        the property that makes its report worth reading. #102's falsification
        already creates a container and is the honest home for that probe.
        """
        name = stack_check(stack)
        image = self.images.for_stack(stack)
        assert self.docker is not None
        try:
            # `{{json .Config}}` rather than `{{index .Config.Labels "..."}}`:
            # Go templates raise on a *missing map key*, and an image built
            # before this label existed has no `Labels` key at all - so the
            # obvious form turns "present but unlabelled", the case this check
            # exists to report, into a template parsing error that reads like a
            # bug in doctor. Found by running it against a real daemon.
            inspected = self.docker(
                "image", "inspect", image, "--format", "{{.Id}}|{{json .Config}}"
            ).strip()
        except ContainerError as exc:
            if any(marker in str(exc).lower() for marker in _DENIED_MARKERS):
                return Check.skipped(
                    name,
                    "not attempted: the socket proxy denies /images by design "
                    "(IMAGES=0, src/swarm/security.py). Run doctor on the host to check it",
                )
            return Check.failed(
                name,
                f"{image} is not on this daemon: {exc}",
                fix=(
                    f"{build_hint(image)}  (it cannot be pulled: the socket proxy sets "
                    f"IMAGES=0 and BUILD=0, so a worker image must already be built on "
                    f"the host - see SETUP.md step 4)"
                ),
            )
        image_id, _, config = inspected.partition("|")
        try:
            labels = json.loads(config or "{}").get("Labels") or {}
        except json.JSONDecodeError:
            labels = {}
        labelled = str(labels.get(STACK_LABEL) or "").strip()
        if not labelled:
            # Not fatal-looking but worth failing on: an image under the right
            # tag with no stack label is indistinguishable from a stale build
            # of a different Dockerfile that happened to be tagged this way,
            # and the failure that produces lands inside a worker.
            return Check.failed(
                name,
                f"{image} carries no {STACK_LABEL} label, so it is indistinguishable "
                f"from a stale build of another image tagged {image}",
                fix=f"{build_hint(image)}  (rebuild it, so the tag and the contents agree)",
            )
        return Check.passed(name, f"{image} present for {labelled} ({image_id[:19]})")

    # --- the tracker ------------------------------------------------------

    def check_tracker_config(self) -> Check:
        """Does a capability contract exist, and does it validate?

        Costs no I/O and runs before everything else that touches the tracker,
        because a block that names the wrong tool is not a network problem and
        the three checks after this one would all be describing it as one.
        """
        if self.tracker_error:
            detail, fix = _contract_verdict(self.tracker_error, self.settings.tracker_config)
            return Check.failed(CHECK_TRACKER_CONFIG, detail, fix=fix)
        if self.tracker is None:
            return Check.skipped(
                CHECK_TRACKER_CONFIG,
                f"no tracker configured: {self.settings.tracker_config} does not exist and "
                f"{TRACKER_CONFIG_ENV} is unset, which is still a normal installation - "
                f"apiary runs on the label control plane until the tracker path lands "
                f"(ADR 0001)",
            )
        contract = self.tracker
        return Check.passed(
            CHECK_TRACKER_CONFIG,
            f"{contract.source}: {contract.mcp} at {contract.endpoint}, "
            f"{'/'.join(contract.capability(name).tool for name in CAPABILITIES)}",
        )

    def check_tracker_reachable(self) -> Check:
        """Does the configured server answer at all?

        A refused *credential* counts as reachable, and separating the two is
        the whole reason this is not one check: "nothing is listening there"
        and "your token expired" have nothing in common except that both stop
        the run, and an operator told the wrong one of the two goes looking in
        entirely the wrong place. The probe is built without a credential when
        none is exported precisely so that this question can still be asked.
        """
        contract = self.tracker
        assert contract is not None  # guarded by `tracker_checks`
        info, error = self._tracker_probe()

        if isinstance(error, McpAuthError):
            return Check.passed(
                CHECK_TRACKER_REACHABLE,
                f"{contract.endpoint} answered, and refused the credential "
                f"(see {CHECK_TRACKER_AUTH})",
            )
        if isinstance(error, McpEgressBlocked):
            return Check.failed(
                CHECK_TRACKER_REACHABLE,
                f"{contract.endpoint}: {error}",
                fix=(
                    f"add the host to security.MCP_HOSTS and to the FilterURL block in "
                    f"compose.yaml - tests/test_security.py asserts the two agree. "
                    f"(APIARY_EGRESS_ALLOW is documented in four places and read by none; "
                    f"the enforced list is generated from the tuple in "
                    f"src/swarm/security.py.)"
                ),
            )
        if error is not None:
            return Check.failed(
                CHECK_TRACKER_REACHABLE,
                f"{contract.endpoint} did not answer: {error}",
                fix=self._unreachable_fix(contract),
            )
        assert info is not None
        return Check.passed(
            CHECK_TRACKER_REACHABLE,
            f"{contract.endpoint}: {info.name} {info.version}, MCP {info.protocol_version}",
        )

    def check_tracker_auth(self) -> Check:
        """Is there a credential, and does the server accept it?

        Asked before a cycle needs it because #143 settled that apiary drives
        no OAuth flow and holds no refresh token: the credential is pre-minted
        and static, which makes expiry a routine event rather than an
        exceptional one. A 401 is never retried (`mcp/client.py`), so an
        expired token is a run that stops, and the only useful thing to say
        about it is the command that mints a new one - which differs per
        server, and which the contract therefore carries.
        """
        contract = self.tracker
        assert contract is not None
        auth = contract.auth
        if not auth.credential(self.env):
            return Check.failed(
                CHECK_TRACKER_AUTH,
                f"{auth.value_env} is not set, and it is where {contract.mcp}'s credential "
                f"is read from",
                fix=auth.absent_fix(),
            )

        info, error = self._tracker_probe()
        if isinstance(error, McpAuthError):
            return Check.failed(
                CHECK_TRACKER_AUTH,
                f"{contract.endpoint} rejected the credential in {auth.value_env} "
                f"({error.status})",
                fix=(
                    f"the token is expired or revoked, which is not transient and is not "
                    f"retried - mint a new one and re-export it:\n    {auth.absent_fix()}"
                ),
            )
        if error is not None:  # pragma: no cover - `reachable` already failed
            return Check.skipped(
                CHECK_TRACKER_AUTH, f"not attempted: {CHECK_TRACKER_REACHABLE} did not pass"
            )
        assert info is not None
        delivery = (
            f"the server reads it from {auth.server_env}"
            if contract.is_stdio
            else f"{auth.header}: {auth.scheme}"
        )
        return Check.passed(
            CHECK_TRACKER_AUTH, f"{auth.value_env} accepted by {info.name} ({delivery})"
        )

    def check_tracker_tools(self) -> Check:
        """Does every tool the contract names exist on the server?

        The failure this is for is the one #150 is written around: a tool name
        that is a typo, or that a server renamed between versions, is not
        discovered until the first cycle that needs the capability - which for
        `create` may be an hour into a run, and for `comment` is the moment a
        pull request has already been opened and nobody is told about it.

        `tools/list` and nothing else. Proving a tool *works* means calling it,
        and calling a tracker's tools means writing to somebody's tracker.
        """
        contract = self.tracker
        assert contract is not None
        info, _ = self._tracker_probe()
        assert info is not None  # guarded by `_after(auth, ...)`

        if not info.supports_tools:
            return Check.failed(
                CHECK_TRACKER_TOOLS,
                f"{info.name} advertises no `tools` capability, so it offers nothing to call",
                fix=(
                    f"check {contract.endpoint} is the MCP endpoint rather than the "
                    f"vendor's API root, and re-check the block with:\n"
                    f"    python -m swarm.mcp.contract {contract.source}"
                ),
            )
        try:
            offered = {spec.name for spec in self._tracker_client().list_tools()}
        except McpError as exc:
            return Check.failed(
                CHECK_TRACKER_TOOLS,
                f"{contract.endpoint} would not list its tools: {exc}",
                fix=self._unreachable_fix(contract),
            )

        missing = [name for name in contract.tools if name not in offered]
        if missing:
            named_by = {
                name: [
                    capability
                    for capability in CAPABILITIES
                    if contract.capability(capability).tool == name
                ]
                for name in missing
            }
            return Check.failed(
                CHECK_TRACKER_TOOLS,
                f"{info.name} has no "
                + ", ".join(f"{name} ({'/'.join(named_by[name])})" for name in missing),
                fix=(
                    f"it offers {', '.join(sorted(offered)) or 'nothing'}. Name one of those "
                    f"in {contract.source}, then re-check with:\n"
                    f"    python -m swarm.mcp.contract {contract.source}"
                ),
            )
        return Check.passed(
            CHECK_TRACKER_TOOLS,
            f"{info.name} offers all {len(contract.tools)} named tools "
            f"({', '.join(contract.tools)}); none was called",
        )

    # --- the tracker probe ------------------------------------------------

    def _tracker_client(self) -> TrackerProbe:
        """The injected probe, or one built from the contract.

        `require_credential=False`: an absent credential is a verdict
        `tracker.auth` reports, not a reason to be unable to ask whether the
        server exists.
        """
        contract = self.tracker
        assert contract is not None
        if self.mcp is None:
            self.mcp = client_for(contract, env=self.env, require_credential=False)
        return self.mcp

    def _tracker_probe(self) -> tuple[Any, Exception | None]:
        """`initialize`, once, and whatever it answered - a `ServerInfo` or a refusal.

        Memoized because three checks read one handshake, and because a
        preflight that connected three times to somebody's tracker would be
        three chances to trip a rate limit while reporting that nothing is
        wrong.
        """
        if self._probe is None:
            try:
                self._probe = (self._tracker_client().connect(), None)
            except (McpError, ContractError, OSError) as exc:
                # The three families a connect can refuse through: the client's
                # own classification, a contract that could not be turned into
                # a client, and a socket. Anything else is a bug in this
                # module rather than a verdict about the operator's machine,
                # and hiding it as a check result would be the wrong trade.
                self._probe = (None, exc)
        return self._probe

    def _close_probe(self) -> None:
        """End the session, and the subprocess if there was one. Best effort."""
        probe, self.mcp, self._probe = self.mcp, None, None
        if probe is None:
            return
        try:
            probe.close()
        except Exception:  # noqa: BLE001 - a teardown must not become the verdict
            pass

    @staticmethod
    def _unreachable_fix(contract: TrackerContract) -> str:
        """Two completely different problems wearing the same exception.

        A remote server that does not answer is a URL, a network or a proxy. A
        local one is a binary that is not on this PATH - and `command:` naming
        something uninstalled is the likelier of the two, because the GitHub
        profile's server is a separate download that nothing in this repository
        installs.
        """
        if contract.is_stdio:
            binary = contract.command[0] if contract.command else contract.endpoint
            return (
                f"`which {binary}` - a stdio tracker is a local binary, and the GitHub "
                f"profile's is a separate install "
                f"(https://github.com/github/github-mcp-server). Correct `command:` in "
                f"{contract.source} if it is installed under another name:\n"
                f"    python -m swarm.mcp.contract {contract.source}"
            )
        return (
            f"check the endpoint with `curl -i {contract.endpoint}` - a 4xx means the URL "
            f"is wrong and a timeout means the network is. Inside compose, confirm the "
            f"host is in security.MCP_HOSTS and in compose.yaml's FilterURL block:\n"
            f"    python -m swarm.mcp.contract {contract.source}"
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _contract_verdict(message: str, path: str) -> tuple[str, str]:
    """One `ContractError` as `(detail, fix)`.

    Every refusal in `mcp/contract.py` is written as a sentence naming the
    field, then the example that fixes it on the following lines - which is the
    same two halves a `Check` carries, so the split is a split rather than a
    rewrite. A message with no second half still gets a fix, because
    `Check.__post_init__` is right to insist on one.
    """
    head, _, tail = message.partition("\n")
    remedy = " ".join(part.strip() for part in tail.splitlines() if part.strip())
    validate = f"python -m swarm.mcp.contract {path}"
    return head.strip(), f"{remedy}  ({validate})" if remedy else validate


def _is_repo(value: str) -> bool:
    return value.count("/") == 1 and all(part for part in value.split("/"))


def _hostname(url: str) -> str:
    """`http://0.0.0.0:11434/` -> `0.0.0.0`. Scheme optional, lowercased."""
    value = url.strip().lower()
    if "://" in value:
        value = value.split("://", 1)[1]
    value = value.split("/", 1)[0].split("@")[-1]
    if value.startswith("["):
        return value.partition("]")[0] + "]"
    return value.split(":", 1)[0]


def _has_model(installed: Sequence[str], wanted: str) -> bool:
    """Is `wanted` among `installed`, allowing for the implicit `:latest` tag?

    Ollama reports `gemma4:31b` for a pull of `gemma4:31b` but `llama3:latest`
    for a pull of `llama3`, and a config that names the untagged form is not
    wrong - it is the form `ollama run` accepts.
    """
    candidates = {wanted, f"{wanted}:latest"} if ":" not in wanted else {wanted}
    return any(name in candidates or name.removesuffix(":latest") == wanted for name in installed)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swarm doctor",
        description="check every environmental precondition. Reads only; writes nothing.",
    )
    parser.add_argument(
        "repo",
        nargs="?",
        default=None,
        help="target repository as owner/name (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--ci-ref",
        default=DEFAULT_CI_REF,
        help=f"ref whose check runs prove CI exists (default: {DEFAULT_CI_REF})",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="do not invoke the models; skips the only check that costs inference",
    )
    return parser


def preflight(stacks: Sequence[str], *, doctor: Doctor | None = None) -> Diagnosis:
    """The image checks alone, for a caller about to start a run.

    `swarm run` calls this rather than the whole preflight, because most of
    what `doctor` asks is expensive, is answered elsewhere, or is a judgement a
    human should be making before they type the command - and because a preflight
    that refused a run over an unrelated `github.ci` verdict would be turned off
    within a week.

    A missing worker image is different in kind: `IMAGES=0` and `BUILD=0` mean
    the orchestrator can neither pull nor build one, so the run is *guaranteed*
    to be all-infrastructure, and it would discover that mid-cycle after a real
    repository already exists. That is worth stopping for.

    A skip is not a failure. Through the socket proxy the probe is denied, and
    doctor's inability to look is not evidence about the host.
    """
    subject = doctor or Doctor.from_env(stacks=tuple(stacks))
    daemon = subject.check_docker_daemon()
    checks = [
        Doctor._after(daemon, stack_check(stack), partial(subject.check_stack_image, stack))
        for stack in subject.stacks
    ]
    return Diagnosis(tuple(checks))


def main(argv: Sequence[str] | None = None, *, doctor: Doctor | None = None) -> int:
    """Print the report; exit non-zero if any check failed.

    `doctor` is the test seam, the same one `cli.main` uses for its client.
    """
    args = build_parser().parse_args(argv)
    if doctor is None:
        doctor = Doctor.from_env(args.repo, ci_ref=args.ci_ref, probe_schema=not args.skip_schema)

    diagnosis = doctor.run()
    print(diagnosis.report())
    if not diagnosis.ok:
        print("! start nothing until these are fixed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
