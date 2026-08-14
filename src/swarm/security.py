"""What a worker may hold, what it may reach, and where its token may not go.

`docs/architecture-v2.md`, "Three constraints", third one: a worker container
executes LLM-generated code *and* holds a token that can push. This module is
the single definition of the three answers to that, and `docs/security.md` is
the prose version of the same thing.

Most of the enforcement already exists and is deliberately not re-implemented
here:

- `containers/manager.py` redacts at the capture boundary, so no captured
  string - log or exception - can carry a credential onward.
- `worker/pr.py` pushes through a `credential.helper` snippet, so the token is
  in neither the remote URL, nor `.git/config`, nor `argv`.
- `Dockerfile.worker` runs as uid 10001 and bakes in no credential at all.

What was missing is the part that is *policy* rather than mechanism: which
token shape is acceptable, which hosts a worker may talk to, which slice of the
Docker API the orchestrator gets, and a way to prove after the fact that
nothing leaked. Those are the four sections below, in that order.

**Why policy lives in a module rather than only in compose.yaml.** The
allowlist is used twice - once as a proxy configuration file and once as a
predicate - and two hand-maintained copies of an allowlist diverge. Here the
proxy's filter lines are *generated* from the same tuple the predicate reads,
and `tests/test_security.py` asserts that what compose.yaml carries is what
this module produces. A widened allowlist that someone edited into the YAML
alone fails the suite instead of quietly shipping.

**What this module does not do.** It refuses nothing on its own. Every function
here is called by something else - the compose file, the dispatcher (#21), a
test - because a check that runs in one place is a check that can be routed
around, and the routing around is what an exfiltration looks like. Where a
caller does not exist yet, `docs/security.md` names the seam and the ticket.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

# The shapes the capture boundary knows are public surface, and importing them
# is the point: a second opinion here about what a secret looks like would
# drift from the one that actually does the redacting, and this module's job is
# to *audit* that one rather than to compete with it.
from .containers.manager import MIN_SECRET_LENGTH, SECRET_NAME_RE, SECRET_PATTERNS

__all__ = [
    "SecurityError",
    "CredentialError",
    "PolicyError",
    "REQUIRED_PERMISSIONS",
    "FORBIDDEN_PERMISSIONS",
    "PROVISION_TOKEN_ENV",
    "PROVISION_PERMISSIONS",
    "assert_provision_token",
    "assert_no_provision_token",
    "classify_token",
    "assert_scoped_token",
    "EgressPolicy",
    "EGRESS_ALLOWLIST",
    "WORKER_NETWORK",
    "SOCKET_PROXY_ENV",
    "SOCKET_PROXY_HOST",
    "DOCKER_HOST_URL",
    "worker_create_flags",
    "assert_unprivileged",
    "Leak",
    "find_secrets",
    "scan_artifacts",
]


class SecurityError(RuntimeError):
    """Base for everything this module raises."""


class CredentialError(SecurityError):
    """The token is the wrong shape, or absent."""


class PolicyError(SecurityError):
    """Something was about to be done that the policy forbids."""


# --------------------------------------------------------------------------
# 1. The token
# --------------------------------------------------------------------------

#: The minimum a fine-grained PAT needs, on the **single** target repository.
#: Nothing here is optional and nothing else is granted:
#:
#:   contents       push the work branch (`worker/pr.py`)
#:   pull_requests  open and update the one PR per issue
#:   issues         read the contract, write `swarm:*` labels and comments
#:   metadata       mandatory on every fine-grained PAT; GitHub adds it anyway
REQUIRED_PERMISSIONS: dict[str, str] = {
    "contents": "write",
    "pull_requests": "write",
    "issues": "write",
    "metadata": "read",
}

#: Permissions that must stay off, listed rather than merely omitted because
#: the GitHub UI offers them one checkbox away from the four above and each one
#: converts "a bad generation wrote a bad PR" into something worse.
#:
#: `workflows` is the sharp one: with it, generated code can rewrite
#: `.github/workflows/*` - and CI is the neutral ground that re-runs the verify
#: command (`docs/architecture-v2.md`, "PRs are the integration mechanism"). A
#: swarm that can edit its own gate has no gate.
FORBIDDEN_PERMISSIONS: tuple[str, ...] = (
    "workflows",
    "actions",
    "administration",
    "secrets",
    "environments",
    "packages",
    "members",
    "organization_administration",
)

# --------------------------------------------------------------------------
# 1b. The boot key
# --------------------------------------------------------------------------
#
# Creating a repository and pushing a CI workflow into its first commit need
# `administration` and `workflows` - the two permissions `FORBIDDEN_PERMISSIONS`
# names first, because a worker holding `workflows` can rewrite the very file
# that independently re-runs its verify command. The model would be editing its
# own exam, and every gate downstream would become decoration.
#
# So the credential that boots a project and the credential workers hold are
# not the same credential. The split is a *lifetime* as much as a scope: the
# boot key is used by the orchestrator before any container exists and is never
# put into one. `assert_no_provision_token` is the check that keeps that true
# even if somebody wires an environment through by accident.

#: Where the boot key lives. Deliberately not `GITHUB_TOKEN`: the work key
#: already owns that name, and a single name for two credentials is how the
#: broad one ends up in a container.
PROVISION_TOKEN_ENV = "APIARY_PROVISION_TOKEN"

#: What the boot key needs, and nothing beyond it. `contents` is here because
#: the initial commit is written through the git data API.
PROVISION_PERMISSIONS: dict[str, str] = {
    "administration": "write",
    "contents": "write",
    "workflows": "write",
    "metadata": "read",
}


#: Fine-grained PATs. The only shape a human should be minting for this.
FINE_GRAINED_PREFIX = "github_pat_"

#: GitHub App installation tokens - also per-repository, and additionally
#: short-lived, which is the better answer where an App is available.
APP_TOKEN_PREFIXES: tuple[str, ...] = ("ghs_",)

#: Classic PATs and OAuth tokens. Every one of these is account-wide: its scope
#: is a verb (`repo`), never a repository, so it reaches every repo the human
#: can reach. `gho_` is what `gh auth token` prints, which is exactly why this
#: check exists - it is the most convenient token on a developer's machine and
#: the worst one to hand a container.
ACCOUNT_WIDE_PREFIXES: tuple[str, ...] = ("ghp_", "gho_", "ghu_", "ghr_")


def classify_token(token: str | None) -> str:
    """`"fine-grained"`, `"app"`, `"account-wide"` or `"unrecognised"`.

    Prefix-only, because that is all a token tells you offline. Asking GitHub
    what a token can do is a network call with its own failure modes, and the
    one distinction that matters - per-repository or not - is already in the
    first eleven characters.
    """
    if not token:
        return "unrecognised"
    if token.startswith(FINE_GRAINED_PREFIX):
        return "fine-grained"
    if token.startswith(APP_TOKEN_PREFIXES):
        return "app"
    if token.startswith(ACCOUNT_WIDE_PREFIXES):
        return "account-wide"
    return "unrecognised"


def assert_provision_token(token: str | None, *, allow_unrecognised: bool = False) -> str:
    """Refuse a boot key that cannot be repo-scoped. Returns its kind.

    Same shape rule as the work key - it is still a credential, and a classic
    PAT here would reach every repository the account owns while holding
    `administration`, which is the worst combination in this file. What differs
    is only what it is *for* and how long it lives.
    """
    if not token:
        raise CredentialError(
            f"{PROVISION_TOKEN_ENV} is not set, and creating a repository needs it. "
            f"The work key deliberately cannot do this: it is held by containers "
            f"running model-written code, and "
            f"{', '.join(sorted(PROVISION_PERMISSIONS))} are permissions that code "
            f"must never have. Mint a second fine-grained token with "
            f"{', '.join(f'{k}:{v}' for k, v in sorted(PROVISION_PERMISSIONS.items()))} "
            f"and export it as {PROVISION_TOKEN_ENV} - see docs/security.md"
        )
    return assert_scoped_token(token, allow_unrecognised=allow_unrecognised)


def assert_no_provision_token(env: Mapping[str, str] | None) -> None:
    """Raise if a container's environment carries the boot key.

    The separation is only worth something if it is enforced somewhere other
    than in prose. `ContainerManager` runs every environment it is about to
    hand a worker through here, so the failure is a refusal to start rather
    than a container that quietly holds `administration` and `workflows` while
    executing whatever the model just wrote.

    Both the name and the value are checked: renaming the variable on the way
    in would defeat a name-only check, and the value is what actually grants
    the permissions.
    """
    if not env:
        return
    secret = os.environ.get(PROVISION_TOKEN_ENV)
    for name, value in env.items():
        if name == PROVISION_TOKEN_ENV:
            raise PolicyError(
                f"{PROVISION_TOKEN_ENV} must never reach a worker: it carries "
                f"{', '.join(sorted(PROVISION_PERMISSIONS))}, and a worker holding "
                f"`workflows` can rewrite the CI that judges its own work"
            )
        if secret and value == secret:
            raise PolicyError(
                f"the value of {PROVISION_TOKEN_ENV} is being passed to a worker as "
                f"{name!r}; renaming it does not narrow what it can do"
            )


def assert_scoped_token(token: str | None, *, allow_unrecognised: bool = False) -> str:
    """Refuse a credential that cannot be repo-scoped. Returns its kind.

    The goal sentence of #28 is that a worker "cannot reach an unrelated repo",
    and no amount of network policy delivers that: `github.com` is one host and
    every repository lives behind it. The token is the only thing that knows
    which repository, so a token that is account-wide makes the rest of this
    module decoration.

    `allow_unrecognised` exists for the enterprise and GHES cases, where the
    prefix is not one of GitHub.com's. It is opt-in rather than the default so
    that a typo'd or truncated token fails here rather than as a 401 inside a
    container, three minutes of inference later.
    """
    kind = classify_token(token)
    if kind in ("fine-grained", "app"):
        return kind
    if kind == "account-wide":
        raise CredentialError(
            "this is a classic or OAuth token, which is scoped to verbs and not to "
            "repositories: it reaches every repository the account can reach. Mint a "
            "fine-grained PAT on the single target repo with "
            + ", ".join(f"{name}:{level}" for name, level in REQUIRED_PERMISSIONS.items())
            + " - see docs/security.md"
        )
    if not token:
        raise CredentialError("no token: GITHUB_TOKEN is empty or unset")
    if allow_unrecognised:
        return kind
    raise CredentialError(
        "unrecognised token prefix; pass allow_unrecognised=True if this is a GHES "
        "or enterprise credential you have scoped by hand"
    )


# --------------------------------------------------------------------------
# 2. Egress
# --------------------------------------------------------------------------

#: The two destinations the goal sentence allows, and nothing else.
#: `codeload` is separate from `github.com` for archive downloads, and
#: `host.docker.internal` is the host's Ollama - the one piece of
#: infrastructure every container needs (`docs/architecture-v2.md`).
GITHUB_HOSTS: tuple[str, ...] = ("github.com", "api.github.com", "codeload.github.com")
OLLAMA_HOST_NAME = "host.docker.internal"
EGRESS_ALLOWLIST: tuple[str, ...] = (*GITHUB_HOSTS, OLLAMA_HOST_NAME)

#: Where an operator widens it. Comma-separated hostnames, and the reason it is
#: an environment variable rather than a constant is that the honest default
#: breaks something: a `## Verify` command that runs `pip install -e .` needs
#: `pypi.org` and `files.pythonhosted.org`, and `Dockerfile.worker` deliberately
#: bakes in no toolchain (see its header), so the install happens at run time.
#:
#: Adding a package index is adding an exfiltration channel - a registry that
#: accepts uploads accepts a token in a package name - so it is a decision an
#: operator makes per target repo, out loud, rather than a default nobody reads.
EGRESS_EXTRA_ENV = "APIARY_EGRESS_ALLOW"

#: Compose service names, which are also the DNS names on the internal network.
EGRESS_PROXY_HOST = "egress-proxy"
EGRESS_PROXY_PORT = 8888

#: The socket proxy's name lives up here beside the other service name because
#: `NO_PROXY_HOSTS` below needs it; the Docker API constants that use it are in
#: section 3.
SOCKET_PROXY_HOST = "docker-socket-proxy"
SOCKET_PROXY_PORT = 2375

#: The network a worker is created on: internal, so it has no default route,
#: and shared with the egress proxy and nothing else. Fixed with `name:` in
#: compose.yaml rather than left to compose's project prefix, because the
#: dispatcher passes this string to `docker create --network`.
WORKER_NETWORK = "apiary-egress"

#: Reached directly rather than through the proxy. The socket proxy is on a
#: different internal network and asking a proxy to reach it would fail
#: confusingly; `localhost` is the container itself.
#:
#: `SOCKET_PROXY_HOST` is load-bearing and was missing until the orchestrator
#: image gained a `docker` client. The CLI honours `HTTP_PROXY` for its own API
#: calls, so `docker version` went to the egress proxy, which answered `403
#: Filtered` - a deny-by-default egress rule refusing a request that never
#: should have left the container. Nothing detected it earlier because there
#: was no client in the image to make the call.
NO_PROXY_HOSTS: tuple[str, ...] = (
    "localhost",
    "127.0.0.1",
    EGRESS_PROXY_HOST,
    SOCKET_PROXY_HOST,
)


@dataclass(frozen=True)
class EgressPolicy:
    """Which hosts a container may reach, as a predicate and as configuration.

    One object with two renderings on purpose. `allows` is what a test and a
    human reason about; `filter_lines` is what tinyproxy enforces; and they
    cannot disagree because the second is generated from the first. The suite
    checks the two against a corpus of hostnames rather than trusting that
    reading.
    """

    hosts: tuple[str, ...] = EGRESS_ALLOWLIST
    proxy_host: str = EGRESS_PROXY_HOST
    proxy_port: int = EGRESS_PROXY_PORT

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> EgressPolicy:
        """The default policy plus whatever `APIARY_EGRESS_ALLOW` names."""
        source = os.environ if env is None else env
        extra = tuple(
            host.strip().lower()
            for host in source.get(EGRESS_EXTRA_ENV, "").split(",")
            if host.strip()
        )
        base = cls()
        return replace(base, hosts=base.hosts + tuple(h for h in extra if h not in base.hosts))

    # --- the predicate --------------------------------------------------

    def allows(self, host: str) -> bool:
        """Is `host` on the allowlist, as itself or as a subdomain of an entry?

        `host` may carry a scheme and a port; both are stripped. Subdomains
        match because `api.` and `codeload.` are not the whole of GitHub -
        `objects.githubusercontent.com` serves release assets and LFS - and an
        allowlist that has to be extended every time GitHub adds a hostname is
        one that gets widened to `.*` by the third person who hits it.
        """
        host = _hostname(host)
        if not host:
            return False
        return any(host == entry or host.endswith("." + entry) for entry in self._entries)

    # --- the configuration ----------------------------------------------

    def filter_lines(self) -> list[str]:
        """The allowlist file tinyproxy reads, one extended regex per host.

        Anchored at both ends: an unanchored `github\\.com` matches
        `github.com.attacker.net`, which is a hostname an attacker owns and a
        filter file is exactly the wrong place to learn that.
        """
        return [rf"^(.*\.)?{re.escape(entry)}$" for entry in self._entries]

    def proxy_url(self) -> str:
        return f"http://{self.proxy_host}:{self.proxy_port}"

    def proxy_env(self) -> dict[str, str]:
        """The environment that routes a container's HTTP client at the proxy.

        Both cases of every name. Python's `urllib` lowercases what it finds,
        git reads the lowercase spelling, and various runtimes read only the
        uppercase one; setting four variables costs nothing and the alternative
        is one client that silently bypasses the proxy - which, on an internal
        network with no default route, presents as a hang rather than as a
        refusal.
        """
        url = self.proxy_url()
        no_proxy = ",".join(NO_PROXY_HOSTS)
        return {
            "HTTP_PROXY": url,
            "HTTPS_PROXY": url,
            "http_proxy": url,
            "https_proxy": url,
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }

    @property
    def _entries(self) -> tuple[str, ...]:
        return tuple(host.lower() for host in self.hosts)


DEFAULT_EGRESS = EgressPolicy()


def _hostname(value: str) -> str:
    """`https://api.github.com:443/x` -> `api.github.com`. Lowercased."""
    value = value.strip().lower()
    if "://" in value:
        value = value.split("://", 1)[1]
    value = value.split("/", 1)[0].split("@")[-1]
    if value.startswith("["):  # bracketed IPv6 literal
        return value.partition("]")[0] + "]"
    return value.split(":", 1)[0]


# --------------------------------------------------------------------------
# 3. The Docker API
# --------------------------------------------------------------------------

#: The address the orchestrator's `docker` CLI dials instead of
#: `/var/run/docker.sock`. The host and port themselves are declared in
#: section 2, beside the other compose service name, because `NO_PROXY_HOSTS`
#: has to name this host to keep the CLI off the egress proxy.
DOCKER_HOST_URL = f"tcp://{SOCKET_PROXY_HOST}:{SOCKET_PROXY_PORT}"

#: The proxy's API surface, as `tecnativa/docker-socket-proxy` spells it. Every
#: key is set explicitly, including the ones whose default is already `0`: the
#: file is the audit, and "this is off" reads very differently from "nobody
#: thought about it" when the next person widens it.
#:
#: What is on is exactly what `containers/manager.py` calls: `create`, `start`,
#: `wait`, `logs`, `stop`, `rm`, `ps` - all `/containers` - plus the `/version`
#: handshake the CLI performs before any of them and `/_ping`.
#:
#: `IMAGES=0` is the quiet one. It means the orchestrator cannot pull, so it
#: can only run images already on the host, and `apiary-worker` has to be built
#: locally. That is a constraint worth having: a compromised orchestrator
#: cannot fetch an image of its choosing to run as its worker.
SOCKET_PROXY_ENV: dict[str, str] = {
    "CONTAINERS": "1",
    "POST": "1",
    "ALLOW_START": "1",
    "ALLOW_STOP": "1",
    "PING": "1",
    "VERSION": "1",
    # Everything else. `EXEC` is the first one an attacker reaches for - it
    # turns any running container into a shell - and `BUILD` is the second,
    # because a build context is arbitrary code with a filesystem.
    "ALLOW_RESTARTS": "0",
    "AUTH": "0",
    "BUILD": "0",
    "COMMIT": "0",
    "CONFIGS": "0",
    "DISTRIBUTION": "0",
    "EXEC": "0",
    "IMAGES": "0",
    "INFO": "0",
    "NETWORKS": "0",
    "NODES": "0",
    "PLUGINS": "0",
    "SECRETS": "0",
    "SERVICES": "0",
    "SESSION": "0",
    "SWARM": "0",
    "SYSTEM": "0",
    "TASKS": "0",
    "VOLUMES": "0",
}

#: Flags that hand a container some part of the host. A `docker create` argv
#: carrying any of these is a privileged container by another name, and the
#: socket proxy cannot see them - it routes on method and path, and all of
#: these travel in the *body* of one `POST /containers/create`.
FORBIDDEN_FLAGS: dict[str, str] = {
    "--privileged": "gives the container the host's full capability set",
    "--cap-add": "adds back a capability the default profile removed",
    "--device": "maps a host device into the container",
    "--device-cgroup-rule": "grants access to a class of host devices",
    "--group-add": "joins a host group, typically to reach a device or socket",
}

#: Flags that are fine except when their value is `host`: sharing a namespace
#: with the host is most of what `--privileged` buys, one namespace at a time.
NAMESPACE_FLAGS: tuple[str, ...] = ("--pid", "--ipc", "--uts", "--userns", "--cgroupns", "--network")

#: Mount flags, checked by value rather than by name - a worker mounts nothing
#: today, but the two that matter are the Docker socket (host root) and the
#: host root itself.
MOUNT_FLAGS: tuple[str, ...] = ("-v", "--volume", "--mount")

#: What a worker is created with, beyond `Limits.flags()`. `no-new-privileges`
#: closes the setuid escalation path, and `cap-drop=ALL` is affordable here
#: because the worker's whole toolchain is git, python and whatever the verify
#: command installs, none of which needs a capability.
HARDENING_FLAGS: tuple[str, ...] = (
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges:true",
)


def worker_create_flags(policy: EgressPolicy | None = None) -> list[str]:
    """The confinement half of a worker's `docker create` argv.

    Separate from `ContainerManager.spawn`'s own flags because that module owns
    the lifecycle and this one owns the policy; the dispatcher (#21) is the
    caller that holds both. `docs/security.md` names that seam - until it
    exists, a worker lands on the default bridge with the daemon's ordinary
    egress, and the token is what stands between it and an unrelated repo.
    """
    del policy  # the network name is fixed by compose; the argument is for symmetry
    return ["--network", WORKER_NETWORK, *HARDENING_FLAGS]


def assert_unprivileged(argv: Sequence[str]) -> None:
    """Raise `PolicyError` if this `docker` argv would create a privileged container.

    The socket proxy narrows *which endpoints* the orchestrator can call. It
    cannot narrow what it says to them, because it routes on method and path
    and `--privileged` is a JSON field in the body of the one POST it must
    allow. This function is the layer that does look, and the only place a
    container is created is `ContainerManager.spawn`, whose argv the suite
    feeds through here.
    """
    reasons = [reason for _, reason in _violations(argv)]
    if reasons:
        raise PolicyError("refusing to create a privileged container: " + "; ".join(reasons))


def _violations(argv: Sequence[str]) -> Iterator[tuple[str, str]]:
    for flag, value in _flags(argv):
        if flag in FORBIDDEN_FLAGS:
            yield flag, f"{flag} {FORBIDDEN_FLAGS[flag]}"
        elif flag in NAMESPACE_FLAGS and (value or "").lower() == "host":
            yield flag, f"{flag}=host shares a host namespace with the container"
        elif flag == "--security-opt" and "unconfined" in (value or "").lower():
            yield flag, f"{flag}={value} disables a mandatory access control profile"
        elif flag in MOUNT_FLAGS and _dangerous_mount(value or ""):
            yield flag, f"{flag} mounts {_dangerous_mount(value or '')} from the host"
        elif flag == "--user" and (value or "").split(":")[0] in ("0", "root"):
            yield flag, "--user runs the container as root"


def _dangerous_mount(value: str) -> str:
    """What is being mounted that must not be, or `""`."""
    lowered = value.lower()
    if "docker.sock" in lowered:
        # The socket in a worker is the whole ticket in reverse: host root
        # inside the container that runs generated code.
        return "the Docker socket"
    source = lowered.split("source=", 1)[1].split(",")[0] if "source=" in lowered else lowered.split(":")[0]
    if source.strip() in ("/", "/etc", "/var/run", "/run"):
        return f"{source.strip()} (the host filesystem)"
    return ""


def _flags(argv: Sequence[str]) -> Iterator[tuple[str, str | None]]:
    """`docker` argv as `(flag, value)` pairs, both spellings of a value.

    Over-reads by design: a boolean flag followed by a positional yields that
    positional as its value. Every caller above matches on the flag first, so
    the only cost is a value nobody looks at.
    """
    tokens = list(argv)
    for index, token in enumerate(tokens):
        if not token.startswith("-"):
            continue
        if "=" in token:
            flag, _, value = token.partition("=")
            yield flag, value
            continue
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        yield token, None if following is None or following.startswith("-") else following


# --------------------------------------------------------------------------
# 4. Artifacts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Leak:
    """One credential found somewhere it was written down.

    Carries a location and the *kind* of match, never the matched text. A leak
    report that quotes the leak is a second copy of it, in a file that gets
    pasted into an issue precisely because something went wrong.
    """

    path: Path
    line: int
    kind: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}"


def find_secrets(text: str, *, literals: Iterable[str] = ()) -> list[tuple[int, str]]:
    """`(line number, what matched)` for every credential in `text`.

    The detection half of `containers.manager.Redactor`, using that module's
    own patterns so the auditor and the redactor cannot disagree about what a
    secret is. Registered literals shorter than `MIN_SECRET_LENGTH` are ignored
    for the reason the redactor ignores them: `DEBUG=1` would otherwise make
    every `1` in a log a finding.
    """
    known = [literal for literal in literals if literal and len(literal) >= MIN_SECRET_LENGTH]
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if any(literal in line for literal in known):
            found.append((number, "a registered credential appears verbatim"))
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                found.append((number, f"matches the secret shape {pattern.pattern}"))
                break
    return found


def scan_artifacts(
    root: Path,
    *,
    literals: Iterable[str] = (),
    env: Mapping[str, str] | None = None,
) -> list[Leak]:
    """Walk everything under `root` and report credentials written to disk.

    The check the "done when" of #28 asks for, and the reason it is a function
    rather than only a test: `.swarm/runs/` is kept forever by design (#29), so
    "no token in a run artifact" is a property to re-check after a bad run, not
    only in CI. Point it at a run directory and believe an empty list.

    `env` enrols credentials by variable name, using the same `SECRET_NAME_RE`
    that enrols them for redaction, so a caller passes `os.environ` rather than
    remembering which of its variables are secret - and a variable added to a
    spawn later is covered here without anyone editing this file.

    Unreadable and binary files are skipped rather than raising: a run
    directory can contain a core dump, and a scanner that stops at the first
    one has scanned nothing.
    """
    known = list(literals)
    if env is not None:
        known += [value for name, value in env.items() if value and SECRET_NAME_RE.search(name)]

    leaks: list[Leak] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        leaks += [Leak(path=path, line=number, kind=kind) for number, kind in find_secrets(text, literals=known)]
    return leaks
