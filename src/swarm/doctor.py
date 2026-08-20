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

- the `swarm:*` labels are gone (#152), so nothing here checks for them. The
  check that did was the clearest example of this module's rule: it could see
  a missing label and could have created it, and deliberately did not.
  Provisioning is a decision someone makes; a diagnostic that quietly repaired
  the thing it was asked to measure would report a repo that was never in the
  state it just described.
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

from .config import SETTINGS, TRACKER_CONFIG_ENV, ConfigError, Settings
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
from .llm import BEDROCK, OLLAMA, OPENAI, orchestrator_llm, structured, worker_llm
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


class MissingCredential(DoctorError):
    """No credential at all was found for a provider that needs one.

    Its own type because the remedy is different from every other reachability
    failure: there is nothing to check, expire or re-scope - something has to
    be exported or configured. Against a local Ollama this state cannot exist,
    which is why it arrived with the remote providers rather than before them.
    """


class InvalidCredential(DoctorError):
    """A credential was found and the provider rejected it.

    Distinct from `MissingCredential` because the two read identically in a log
    and share no remedy: one is "set it", the other is "the one you set is
    expired, mistyped, or lacks the permission this call needs". Both are
    common, and telling an operator to export a variable they have already
    exported is exactly the moved-hour-of-confusion this module exists to end.
    """


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

OK = "ok"
FAIL = "fail"
SKIP = "skip"

#: Check names. Constants because a test addresses a check by name and an
#: operator greps for one; a renamed literal in three places is a rename that
#: only two of them get.
#: The four model checks. Named for the *question* rather than for Ollama,
#: because since ADR 0006 the configured provider may not be Ollama - and this
#: module's own docstring says a check "that reports `ollama.models: failed`
#: has moved the hour of confusion rather than removing it". A doctor
#: reporting on a provider you are not using does exactly that.
#:
#: The meanings are stable across providers and the Ollama implementations
#: moved rather than changed:
#:
#:   target     - where will this dial, and which source decided
#:   reachable  - can it be reached, credentials included
#:   available  - are the configured models actually available to this account
#:   schema     - does schema-forced output actually constrain these models
CHECK_MODEL_TARGET = "model.target"
CHECK_MODEL_REACHABLE = "model.reachable"
CHECK_MODEL_AVAILABLE = "model.available"
CHECK_MODEL_SCHEMA = "model.schema"
CHECK_TOKEN = "github.token"
CHECK_BOOT_TOKEN = "github.boot-token"
CHECK_REPO = "github.repo"
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
        CHECK_MODEL_TARGET, CHECK_MODEL_REACHABLE, CHECK_MODEL_AVAILABLE, CHECK_MODEL_SCHEMA,
        CHECK_TOKEN, CHECK_BOOT_TOKEN, CHECK_REPO, CHECK_CI, CHECK_TIMEOUTS,
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


class Unlistable(DoctorError):
    """This provider will not enumerate its catalogue for this account.

    A skip, never a failure. Listing is a *different permission* from calling
    on every remote provider - an account routinely holds the second without
    the first - so a denied listing says nothing about whether the configured
    model works, and the schema probe answers that with a real call anyway.
    Reporting it as a failure would send an operator to fix an entitlement they
    do not need.
    """


@dataclass
class RemoteInference:
    """The probe for a provider that is somewhere else.

    Deliberately thin. It answers the two questions the local probe answers -
    can this be reached, and is the model there - in terms a remote provider
    can actually answer, and it delegates the third to exactly the same code
    path `HostInference` uses, because the schema probe must exercise the
    client the orchestrator and the workers really get.

    **It reads a credential's presence, never its value.** `spec.credential` is
    a sentence about where the credential comes from, and that is all this
    module ever holds - which is what keeps a doctor report safe to paste into
    an issue.
    """

    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))

    def _specs(self) -> list[Any]:
        from .models import ROLES as MODEL_ROLES, resolve  # noqa: PLC0415 - lazy

        return [resolve(role, env=self.env).spec for role in MODEL_ROLES]

    def version(self) -> str:
        """Can this be dialled at all - credentials included?

        Constructing the client is the probe, and on these providers that is
        not a formality: the OpenAI client refuses to exist without a key, and
        `langchain-aws` resolves the AWS credential chain in a field validator.
        So a client that builds is a credential that was found, and building it
        costs no inference and no request.

        Telling apart *missing* from *rejected* is the point. `llm.py` raises
        `ConfigError` for both, so the distinction is made on the text - which
        is honest about being a heuristic, and errs toward `MissingCredential`
        only when the message says something is not set.
        """
        from .llm import PROVIDERS  # noqa: PLC0415 - lazy, like every provider import

        answers: list[str] = []
        for spec in self._specs():
            if spec.provider == OLLAMA:
                continue
            try:
                PROVIDERS[spec.provider].build(spec, None)
            except ConfigError as exc:
                message = str(exc)
                if "is not set" in message or "needs a region" in message:
                    raise MissingCredential(message) from exc
                raise InvalidCredential(message) from exc
            except Exception as exc:  # noqa: BLE001 - same reason as `schema_probe`
                raise DoctorError(f"{type(exc).__name__}: {exc}") from exc
            answers.append(f"reachable as {spec.credential}")
        return "; ".join(answers) or "nothing remote configured"

    def installed(self) -> list[str]:
        """Refused, always, and that is the honest answer rather than a stub.

        Enumerating a remote catalogue costs a second permission, returns
        hundreds of entries on one provider and needs a separate SDK client on
        the other - and it would still not answer the question anyone asks,
        which is "does *my* model work". `check_model_schema` answers that with
        one real call, and this stays a skip with a reason attached.
        """
        raise Unlistable(
            "listing a remote catalogue is a separate permission from calling a model; "
            "the schema check below proves the configured model with one real call"
        )

    def schema_probe(self, role: str) -> str:
        """Identical to the local one, deliberately - see `HostInference`."""
        return HostInference.schema_probe(self, role)  # type: ignore[arg-type]


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
def _endpoint(spec: Any) -> str:
    """Where a spec will dial, in one phrase.

    Per provider, because the answer is a different kind of thing each time - a
    URL an operator can curl, a hosted API with a well-known name, a regional
    service. "Where will this dial" is the question `model.target` exists to
    answer, and answering it as "the model server" for all three would be the
    Ollama-shaped report this ticket removes.
    """
    if spec.provider == OLLAMA:
        return spec.option("base_url") or SETTINGS.ollama_base_url
    if spec.provider == BEDROCK:
        region = spec.option("region") or "no region"
        return f"bedrock-runtime.{region}.amazonaws.com"
    return spec.option("base_url") or "api.openai.com"


def _credential_fix(spec: Any) -> str:
    """The remedy for "there is no credential", per provider.

    Three shapes, and only one of them is "export a variable" - which is why
    the fix is built here rather than written once with the variable name
    interpolated into it.
    """
    if spec.provider == BEDROCK:
        return (
            "configure AWS for this shell - `aws sso login --profile <name>` or an instance "
            "role - and name it with SWARM_<ROLE>_MODEL_OPTIONS=\"profile=<name>,region=<region>\", "
            "or export AWS_PROFILE / AWS_REGION. `pip install -e \".[bedrock]\"` if the SDK "
            "is missing"
        )
    if spec.provider == OPENAI:
        return (
            "export OPENAI_API_KEY (or name another variable with "
            "SWARM_<ROLE>_MODEL_OPTIONS=\"api_key_env=<NAME>\"), and "
            "`pip install -e \".[openai]\"` if the SDK is missing"
        )
    return _TARGET_FIX


def _unreachable_fix(spec: Any, settings: Settings) -> str:
    """The remedy for "it did not answer", per provider."""
    if spec.provider == OLLAMA:
        return (
            "start the server - `ollama serve`, or launch the Ollama app - and "
            f"confirm with `curl {settings.ollama_base_url}/api/version`. "
            "It runs on the HOST, never in a container: Docker Desktop has no "
            "Metal passthrough (docs/architecture-v2.md)"
        )
    return (
        f"check network egress to {_endpoint(spec)} and that the provider is not in an "
        f"outage; the credential itself was accepted, so this is the path rather than "
        f"the permission"
    )


def _missing_model_fix(role: str, spec: Any) -> str:
    """The third credential verdict: valid credential, no access to this model.

    On Ollama the remedy is a download. On a remote provider it is an
    entitlement somebody grants in a console, and telling an operator to pull
    something would be advice they cannot act on.
    """
    variable = f"SWARM_{role.upper()}_MODEL"
    if spec.provider == OLLAMA:
        return f"ollama pull {spec.model}  (or point the role elsewhere: {variable})"
    if spec.provider == BEDROCK:
        return (
            f"request access to {spec.model} for this account in the Bedrock console - model "
            f"access is granted per model and per region, and {spec.credential} has a valid "
            f"credential without it. Or point the role elsewhere: {variable}"
        )
    return (
        f"this account has no access to {spec.model}; check the model name and the "
        f"organisation's allowed models, or point the role elsewhere: {variable}"
    )


def _schema_fix(role: str, spec: Any) -> str:
    """Why schema-forced output failed, per provider - three different causes.

    Ollama's is a model or a server too old for format-constrained decoding.
    Bedrock's is most often a model that does not serve `json_schema` at all,
    which is why `llm.py` leaves `method` an option. OpenAI's is a model
    predating the Structured Outputs API.
    """
    variable = f"SWARM_{role.upper()}_MODEL"
    if spec.provider == OLLAMA:
        return (
            f"`ollama show {spec.model}` - a model without structured-output support "
            f"cannot orchestrate; set {variable} to one that has it, and check "
            f"`ollama --version` (format-constrained decoding needs a recent server)"
        )
    if spec.provider == BEDROCK:
        return (
            f"not every model on Bedrock serves json_schema. Try "
            f"SWARM_{role.upper()}_MODEL_OPTIONS=\"method=function_calling\", or set "
            f"{variable} to a model that does - every orchestrator call is a structured() "
            f"call, so a model that cannot be constrained cannot hold this role"
        )
    return (
        f"{spec.model} may predate the Structured Outputs API; set {variable} to a model "
        f"that supports strict json_schema - every orchestrator call is a structured() call"
    )


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
    #: The probe for every provider that is not Ollama. Same shape and same
    #: reason as `inference`, and left unset in production - a test that needed
    #: an AWS account or an OpenAI key to reach a verdict is a test that runs
    #: nowhere. Built on first use rather than in `__post_init__`, so a fully
    #: local installation never constructs one.
    remote: Inference | None = None
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
        target = self.check_model_target()
        # Reachability chains off the target rather than being asked anyway: a
        # bind address produces a connection error that names `ollama serve`,
        # and following that advice fixes nothing. One problem, one verdict.
        reachable = self._after(target, CHECK_MODEL_REACHABLE, self.check_model_reachable)
        models = self._after(reachable, CHECK_MODEL_AVAILABLE, self.check_model_available)
        checks: list[Check] = [
            target,
            reachable,
            models,
            self._after(models, CHECK_MODEL_SCHEMA, self.check_model_schema),
        ]

        token = self.check_token()
        access = self._after(token, CHECK_REPO, self.check_repo_access)
        checks += [
            token,
            self.check_boot_token(),
            access,
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

    # --- the models -----------------------------------------------------
    #
    # Four checks, dispatched by provider. The Ollama implementations moved
    # here rather than changing: `_target_ollama` still carries the whole
    # bind-address-versus-client-target argument, because that trap is Ollama's
    # own and no other provider has it.
    #
    # Two roles, and they may be on two different providers - the epic's whole
    # premise is that the answer is genuinely different for each, since the
    # orchestrator emits a few hundred tokens of schema-constrained JSON and the
    # worker emits whole files. So every check below iterates roles and groups
    # by provider, rather than asking one question about "the model server".

    def _resolved(self) -> list[tuple[str, Any, Any]]:
        """`(role, spec, resolution)` per role, or a raised `ConfigError`.

        Never cached across a run: `swarm doctor` is short-lived, and a value
        memoised before the checks run would be a second place for the four
        checks below to disagree with each other about what is configured.
        """
        from .models import ROLES as MODEL_ROLES, resolve  # noqa: PLC0415 - lazy, see `_probe`

        return [
            (role, (r := resolve(role, env=self.env, settings=self.settings)).spec, r)
            for role in MODEL_ROLES
        ]

    def _inference_for(self, provider: str) -> Inference:
        """The probe for one provider.

        Named `_inference_for` rather than `_probe`, which is already this
        class's tracker-handshake cache - two different probes, and one name
        for both is how a method quietly shadows a field.

        `self.inference` is the Ollama seam and stays exactly what it was, so
        every existing test that hands one over keeps working unchanged.
        Remote providers get `self.remote`, which is the same shape and the
        same reason - the interesting logic here is the wording of the
        verdicts, and a test that needs an AWS account to reach it is a test
        that runs nowhere.
        """
        if provider == OLLAMA:
            assert self.inference is not None  # set in __post_init__
            return self.inference
        if self.remote is None:
            self.remote = RemoteInference(env=self.env)
        return self.remote

    # -- target ----------------------------------------------------------

    def check_model_target(self) -> Check:
        """Where will each role dial, and which source decided?

        Costs no I/O and runs before the reachability check, because when this
        one is wrong the reachability failure reads like something else
        entirely - a stopped server, an expired key - and sends the reader
        after the wrong thing.

        Reporting the *source* is new with ADR 0006 and is the half an operator
        cannot otherwise discover: a model can now come from an argument, an
        environment variable, a file written by a console session last week, or
        a built-in default, and only the first two are visible from a shell.
        """
        try:
            resolved = self._resolved()
        except ConfigError as exc:
            return Check.failed(
                CHECK_MODEL_TARGET,
                f"the configured model cannot be read: {exc}",
                fix="fix the SWARM_*_MODEL variables named above, or unset them to fall "
                    "back to the built-in local defaults",
            )

        problems: list[Check] = []
        lines: list[str] = []
        for role, spec, resolution in resolved:
            if spec.provider == OLLAMA:
                bad = self._target_ollama(spec)
                if bad is not None:
                    problems.append(bad)
                    continue
            lines.append(f"{role} -> {spec.label} via {_endpoint(spec)}, from {resolution.source}"
                         + (f" ({resolution.detail})" if resolution.detail else ""))
        if problems:
            return problems[0]
        return Check.passed(CHECK_MODEL_TARGET, "; ".join(lines))

    def _target_ollama(self, spec: Any) -> Check | None:
        """Ollama's own trap, unchanged, and still the only provider with it.

        `OLLAMA_HOST` answers two different questions with one value: to the
        server it is a bind address, to a client it is a target. Those coincide
        on a laptop running both and diverge on exactly the machines this
        project is built for.
        """
        url = spec.option("base_url") or self.settings.ollama_base_url
        host = _hostname(url)

        if "://" not in url:
            return Check.failed(
                CHECK_MODEL_TARGET,
                f"OLLAMA_HOST={url!r} has no scheme, which is the shape of a bind address",
                fix=_TARGET_FIX,
            )
        if host in WILDCARD_HOSTS:
            return Check.failed(
                CHECK_MODEL_TARGET,
                f"OLLAMA_HOST={url!r} names a wildcard bind address; a client cannot dial it",
                fix=_TARGET_FIX,
            )
        if self.in_container and host in LOOPBACK_HOSTS:
            return Check.failed(
                CHECK_MODEL_TARGET,
                f"OLLAMA_HOST={url!r} is this container's own loopback, and Ollama "
                f"runs on the host (docs/architecture-v2.md, first constraint)",
                fix=(
                    f"export APIARY_OLLAMA_HOST={CONTAINER_OLLAMA_URL} before "
                    f"`docker compose run orchestrator ...`; compose passes it through as "
                    f"the container's OLLAMA_HOST"
                ),
            )
        return None

    # -- reachable -------------------------------------------------------

    def check_model_reachable(self) -> Check:
        """Can each configured provider be reached, credentials included?

        For a remote provider "no key" and "bad key" are different verdicts and
        both are common. Telling an operator to export a variable they have
        already exported is precisely the moved-hour-of-confusion this module
        exists to end, so the two arrive here as different exception types and
        leave as different fixes.
        """
        try:
            providers = {spec.provider: spec for _, spec, _ in self._resolved()}
        except ConfigError:
            return Check.skipped(
                CHECK_MODEL_REACHABLE, f"not attempted: {CHECK_MODEL_TARGET} did not pass"
            )

        answers: list[str] = []
        for provider, spec in providers.items():
            try:
                answers.append(f"{provider} {self._inference_for(provider).version()}")
            except MissingCredential as exc:
                return Check.failed(
                    CHECK_MODEL_REACHABLE,
                    f"{provider} has no credential: {exc}",
                    fix=_credential_fix(spec),
                )
            except InvalidCredential as exc:
                return Check.failed(
                    CHECK_MODEL_REACHABLE,
                    f"{provider} rejected the credential it was given: {exc}",
                    fix=(
                        f"the credential for {spec.credential} is set but not accepted - it is "
                        f"expired, mistyped, or lacks the permission this call needs. It is not "
                        f"missing, so exporting it again will not help"
                    ),
                )
            except DoctorError as exc:
                return Check.failed(
                    CHECK_MODEL_REACHABLE,
                    f"no answer from {_endpoint(spec)}: {exc}",
                    fix=_unreachable_fix(spec, self.settings),
                )
        return Check.passed(CHECK_MODEL_REACHABLE, ", ".join(answers))

    # -- available -------------------------------------------------------

    def check_model_available(self) -> Check:
        """Are the configured models actually available to this account?

        The third of the three credential verdicts: a key can be present, valid
        and still have no access to *this* model - which on a remote provider is
        an entitlement somebody grants in a console, not something an operator
        can fix by pulling anything.
        """
        try:
            wanted = self._resolved()
        except ConfigError:
            return Check.skipped(
                CHECK_MODEL_AVAILABLE, f"not attempted: {CHECK_MODEL_TARGET} did not pass"
            )

        served: list[str] = []
        for role, spec, _ in wanted:
            try:
                installed = self._inference_for(spec.provider).installed()
            except Unlistable as exc:
                # Not a failure. A provider that will not enumerate its catalogue
                # says nothing about whether *this* model works, and the schema
                # probe below answers that question with a real call anyway.
                served.append(f"{spec.label} ({role}) not listable: {exc}")
                continue
            except DoctorError as exc:
                return Check.failed(
                    CHECK_MODEL_AVAILABLE,
                    f"could not list {spec.provider} models: {exc}",
                    fix=_unreachable_fix(spec, self.settings),
                )
            if not _has_model(installed, spec.model):
                return Check.failed(
                    CHECK_MODEL_AVAILABLE,
                    f"{spec.model} ({role}) is not available to this {spec.provider} account",
                    fix=_missing_model_fix(role, spec),
                )
            served.append(f"{spec.label} ({role})")
        return Check.passed(CHECK_MODEL_AVAILABLE, ", ".join(served))

    # -- schema ----------------------------------------------------------

    def check_model_schema(self) -> Check:
        """Does schema-forced output actually constrain these models?

        The one check that matters most, and it is unchanged in purpose: the
        whole orchestrator is `structured()` calls, and a model that accepts the
        constraint and then ignores it produces empty plans and unparseable
        judgements - indistinguishable from a planner bug, and where an
        afternoon goes.

        It now runs against whichever provider is configured, which makes it the
        check that carries the most weight after ADR 0006: Ollama constrains
        *decoding* so the model cannot wander off-format, while a remote strict
        schema is a different mechanism with a different failure mode. Whether
        that mechanism holds is #259's question, and this is where an operator
        asks it about their own account.

        Still the one check that costs real inference, and still the one with an
        off switch.
        """
        if not self.probe_schema:
            return Check.skipped(CHECK_MODEL_SCHEMA, "not attempted: --skip-schema")
        try:
            wanted = self._resolved()
        except ConfigError:
            return Check.skipped(
                CHECK_MODEL_SCHEMA, f"not attempted: {CHECK_MODEL_TARGET} did not pass"
            )

        results: list[str] = []
        for role, spec, _ in wanted:
            try:
                results.append(f"{spec.label} ({role}) {self._inference_for(spec.provider).schema_probe(role)}")
            except DoctorError as exc:
                return Check.failed(
                    CHECK_MODEL_SCHEMA,
                    f"{spec.label} ({role}) did not return the requested schema: {exc}",
                    fix=_schema_fix(role, spec),
                )
        return Check.passed(CHECK_MODEL_SCHEMA, "; ".join(results))

    def _wanted_models(self) -> list[tuple[str, str]]:
        """`(role, model)`, kept for readers that only want the names."""
        return [(role, spec.model) for role, spec, _ in self._resolved()]

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
        """End the session, and the subprocess if there was one. Best effort.

        The client object is kept and only the handshake is forgotten. A second
        `run()` then re-handshakes through the same one, which is what a caller
        who injected a probe means; discarding it would quietly replace their
        double with a live connection to somebody's tracker on the second call.
        """
        self._probe = None
        if self.mcp is None:
            return
        try:
            self.mcp.close()
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
