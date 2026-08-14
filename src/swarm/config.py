"""Central configuration.

Two model roles:

  ORCHESTRATOR_MODEL - planning, routing, progress judgement. Short
                       schema-constrained JSON calls, a few hundred tokens a
                       round. Cheap even on a large model, so buy quality here.

  WORKER_MODEL       - the one that writes code. It emits whole files, so this
                       is where the wall-clock time goes. Buy throughput here.

The two are not "big vs small" - see the note on the model fields below.

Override anything via environment variables.

One of those overrides is not like the others, and the block above
`resolve_ollama_target` explains why: Ollama spells a server's *bind address*
and a client's *target* with the same `OLLAMA_HOST`, so this module cannot
simply read it and believe it.
"""

from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

__all__ = [
    "APIARY_OLLAMA_HOST_ENV",
    "CONTAINER_OLLAMA_HOSTNAME",
    "CONTAINER_OLLAMA_URL",
    "HOST_OLLAMA_URL",
    "IN_CONTAINER_ENV",
    "OLLAMA_HOST_ENV",
    "OllamaTarget",
    "SETTINGS",
    "Settings",
    "client_target",
    "container_evidence",
    "explain_unresolvable",
    "in_container",
    "ollama_base_url",
    "resolve_ollama_target",
]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --------------------------------------------------------------------------
# Where Ollama is
# --------------------------------------------------------------------------
#
# `OLLAMA_HOST` answers two different questions with one value. To the server
# it is a bind address; to a client it is a target. Those coincide on a laptop
# running both, and diverge on exactly the machines this project is built for:
# letting containers reach the host's server requires `OLLAMA_HOST=0.0.0.0:11434`
# in the *server's* environment, and an orchestrator that reads that variable as
# its target is pointed at the wildcard address, which as a target means "this
# host" - the container itself, where nothing listens.
#
# Worse, the mistake is nearly silent on the development host: BSD and Linux
# sockets treat a connect() to 0.0.0.0 as a connect to loopback, so on macOS the
# wrong value works by accident and only fails once the same environment is
# inherited by a container. The failure then reads as a dead Ollama.
#
# So this module never treats `OLLAMA_HOST` as authoritative. It takes the first
# value that is *shaped like something a client can dial*, in this order:
#
#   1. APIARY_OLLAMA_HOST - unambiguous by construction; it names no server and
#      is what compose.yaml already passes into containers (#12).
#   2. OLLAMA_HOST - honoured when it is a client target, declined with a
#      reason when it is a bind address or a container's own loopback.
#   3. the detected default - host.docker.internal in a container, localhost
#      otherwise, which is the whole point of #13: neither case needs a human
#      to export anything.
#
# Declining rather than raising is deliberate. `swarm --help` has to work on a
# misconfigured machine, and the operator-facing verdict belongs to `swarm
# doctor`'s `ollama.target` check (#32), which reads the same shapes this module
# rejects and prints the remedy. `OllamaTarget.note` is what it has to quote.

#: The unambiguous override. Set this, not `OLLAMA_HOST`, when a client needs
#: to be told where the server is; nothing reads it as a bind address.
APIARY_OLLAMA_HOST_ENV = "APIARY_OLLAMA_HOST"

#: Ollama's own variable, and the ambiguous one.
OLLAMA_HOST_ENV = "OLLAMA_HOST"

#: Escape hatch for the detection below - `1`/`0`, `true`/`false`, `yes`/`no`.
#: Needed because container detection is evidence, not proof, and an operator
#: who knows the answer should not have to argue with a heuristic.
IN_CONTAINER_ENV = "APIARY_IN_CONTAINER"

DEFAULT_OLLAMA_PORT = 11434

#: Ollama on the host, seen from the host.
HOST_OLLAMA_URL = f"http://localhost:{DEFAULT_OLLAMA_PORT}"

#: Ollama on the host, seen from inside a container. Both Dockerfiles bake this
#: in and `containers/manager.py` passes the `--add-host` that makes the name
#: resolve; the default exists for the container that is started by neither.
CONTAINER_OLLAMA_HOSTNAME = "host.docker.internal"
CONTAINER_OLLAMA_URL = f"http://{CONTAINER_OLLAMA_HOSTNAME}:{DEFAULT_OLLAMA_PORT}"

#: Hosts that are a place to listen rather than a place to dial. The empty
#: string is here because `OLLAMA_HOST=:11434` is a legal bind address.
WILDCARD_HOSTS: tuple[str, ...] = ("0.0.0.0", "::", "*", "")

#: Dialable, but inside a container it is the container.
LOOPBACK_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")

#: Files a container runtime drops in the root filesystem. Present in Docker
#: and Podman containers, and - the case this module has to survive - also
#: present on the occasional host where a script or an image build left one.
CONTAINER_MARKER_FILES: tuple[str, ...] = (".dockerenv", "run/.containerenv")

#: Substrings in `/proc/1/cgroup` that name a container runtime. Only cgroup v1
#: and `cgroupns=host` produce them, which is why they are one signal and not
#: the signal: under Docker Desktop's private cgroup namespace the file reads
#: `0::/` and names nothing at all.
CGROUP_MARKERS: tuple[str, ...] = ("docker", "containerd", "kubepods", "lxc", "buildkit", "podman")

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


@dataclass(frozen=True)
class OllamaTarget:
    """Where this process will dial, which variable decided it, and what it declined.

    `note` is the interesting field. A resolution that quietly ignores an
    environment variable the operator believes in is a resolution nobody can
    debug, so every declined value leaves a sentence behind for `swarm doctor`
    and for the worker's logs.
    """

    url: str
    source: str
    note: str | None = None
    in_container: bool = False

    @property
    def needs_host_gateway(self) -> bool:
        """Does dialling this depend on `host.docker.internal` being injected?

        True for the container default, which Docker Desktop resolves by itself
        and a Linux daemon does not - see `explain_unresolvable`.
        """
        return _hostname(self.url) == CONTAINER_OLLAMA_HOSTNAME


def client_target(value: str | None, *, container: bool = False) -> str | None:
    """Normalise `value` to a URL a client can dial, or `None` if it is not one.

    Accepts what Ollama accepts - `host`, `host:port`, or a full URL - and
    returns the full URL form, because that is what `langchain_ollama` wants for
    `base_url`. A missing port becomes 11434 exactly as it does in Ollama's own
    client, except behind `https://`, where the scheme already carries a default
    and pinning 11434 would be inventing a fact.

    `None` means "this is not a target", for one of three reasons: it is empty,
    it is a wildcard bind address, or it is a container's own loopback while
    `container` is true. The caller decides what to do about it; this function
    is a shape test and does no I/O.
    """
    value = (value or "").strip()
    if not value:
        return None

    try:
        parts = urlsplit(value if "://" in value else f"http://{value}")
        host, port = parts.hostname, parts.port
    except ValueError:
        # An unbracketed IPv6 literal or a non-numeric port. Unparseable is
        # not dialable, and guessing at the intent would guess wrong.
        return None
    host = (host or "").strip("[]").lower()

    if host in WILDCARD_HOSTS:
        return None
    if container and host in LOOPBACK_HOSTS:
        return None

    netloc = parts.netloc
    if port is None and parts.scheme != "https":
        netloc = f"{netloc}:{DEFAULT_OLLAMA_PORT}"
    # Rebuilt rather than string-appended: `OLLAMA_HOST=http://box/ollama` has a
    # path, and gluing a port on the end of that produces `/ollama:11434`.
    return urlunsplit((parts.scheme, netloc, parts.path.rstrip("/"), parts.query, parts.fragment))


def in_container(
    env: Mapping[str, str] | None = None,
    *,
    platform: str = sys.platform,
    root: Path | str = "/",
) -> bool:
    """Is this process inside a container?

    Two rules, both about not misfiring:

    - **A non-Linux process is not in one of our containers.** Every image this
      project builds is Linux, so a macOS or Windows host answers `False`
      without touching the filesystem. This is what keeps a stray `/.dockerenv`
      on the development host - left by a script, restored from a backup,
      whatever - from routing the orchestrator at `host.docker.internal` and
      failing to resolve it.
    - **One signal is not evidence.** On Linux the answer needs two independent
      signals out of the marker file, the cgroup names, and an overlay root
      filesystem. `/.dockerenv` alone is exactly the shape of the quirk above;
      an overlay root alone is a live USB.

    `APIARY_IN_CONTAINER` overrides both, in either direction.
    """
    env = os.environ if env is None else env
    override = _flag(env.get(IN_CONTAINER_ENV))
    if override is not None:
        return override
    if not platform.startswith("linux"):
        return False
    return len(container_evidence(root)) >= 2


def container_evidence(root: Path | str = "/") -> tuple[str, ...]:
    """The names of the container signals visible under `root`.

    Separate from `in_container` so a test can state which signals it planted,
    and so a future doctor check can print them rather than a bare boolean.
    `root` is a parameter for the same reason: the alternative to injecting a
    fake filesystem is a test that only runs inside a container.
    """
    root = Path(root)
    found: list[str] = []

    for marker in CONTAINER_MARKER_FILES:
        if (root / marker).is_file():
            found.append(marker)
            break

    cgroup = _read(root / "proc/1/cgroup")
    if cgroup and any(marker in cgroup for marker in CGROUP_MARKERS):
        found.append("cgroup")

    if _overlay_root(_read(root / "proc/self/mountinfo")):
        found.append("overlay-root")

    return tuple(found)


def resolve_ollama_target(
    env: Mapping[str, str] | None = None,
    *,
    container: bool | None = None,
) -> OllamaTarget:
    """Decide where to dial, from the environment and the machine.

    The precedence and the reasoning are in the block at the top of this
    section. Nothing here does I/O - no DNS, no HTTP - because this runs at
    import time through `Settings`, and a module that cannot be imported on a
    machine with a broken network is a module that cannot print its own
    diagnostics.
    """
    env = os.environ if env is None else env
    if container is None:
        container = in_container(env)

    notes: list[str] = []
    for name in (APIARY_OLLAMA_HOST_ENV, OLLAMA_HOST_ENV):
        raw = (env.get(name) or "").strip()
        if not raw:
            # Unset and empty are the same statement. compose.yaml passes
            # `${VAR:-}` through for variables the caller may not have, and an
            # empty string there means "not configured", not "bind to
            # everything".
            continue
        # The unambiguous variable is only ever read as a target, so the one
        # thing that can disqualify it is being undialable - a container
        # started with `--network host` genuinely does reach Ollama on
        # localhost, and an operator who says so in APIARY_OLLAMA_HOST is not
        # guessing. `OLLAMA_HOST` gets no such benefit of the doubt: loopback
        # there is what a shell that configured the *server* leaves behind.
        loopback_is_local = container and name == OLLAMA_HOST_ENV
        target = client_target(raw, container=loopback_is_local)
        if target is not None:
            return OllamaTarget(target, source=name, note=_note(notes), in_container=container)
        notes.append(
            f"{name}={raw!r} is not a client target ({_why(raw, loopback_is_local)}); ignored"
        )

    url = CONTAINER_OLLAMA_URL if container else HOST_OLLAMA_URL
    where = "in a container" if container else "on the host"
    notes.append(f"defaulted to {url} - detected {where}, and nothing usable was set")
    return OllamaTarget(url, source="default", note=_note(notes), in_container=container)


def ollama_base_url(env: Mapping[str, str] | None = None) -> str:
    """`resolve_ollama_target(...).url`, which is all `Settings` needs."""
    return resolve_ollama_target(env).url


def explain_unresolvable(
    target: OllamaTarget,
    *,
    resolve: Callable[[str], Any] = socket.gethostbyname,
) -> str | None:
    """`None` if the target's hostname resolves here, else the sentence that says why not.

    This is the one function in the module that touches the network, and it is
    never called at import. It exists because the container default fails in a
    genuinely confusing way on a Linux host: `host.docker.internal` is a name
    Docker Desktop injects into every container and a Linux daemon does not, so
    the same image that works on a Mac reports a DNS failure for a name the
    operator never typed. Saying `--add-host` out loud is the whole point.
    """
    host = _hostname(target.url)
    if not host:
        return None
    try:
        resolve(host)
    except OSError as exc:
        if target.needs_host_gateway:
            return (
                f"{host} does not resolve here ({exc}). Docker Desktop injects that name; "
                f"a Linux daemon does not. Start the container with "
                f"`--add-host {CONTAINER_OLLAMA_HOSTNAME}:host-gateway` (compose.yaml spells it "
                f"`extra_hosts:`, and src/swarm/containers/manager.py already passes it for "
                f"workers), or set {APIARY_OLLAMA_HOST_ENV} to the host's address on a network "
                f"the container can reach"
            )
        return (
            f"{host} does not resolve here ({exc}). Set {APIARY_OLLAMA_HOST_ENV} to a name or "
            f"address this process can dial - it is read as a client target and never as a "
            f"server bind address"
        )
    return None


def _hostname(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").strip("[]").lower()
    except ValueError:
        return ""


def _why(raw: str, loopback_is_local: bool) -> str:
    """The half-sentence naming which disqualification `raw` hit."""
    host = _hostname(raw if "://" in raw else f"http://{raw}")
    if host in WILDCARD_HOSTS:
        return "a wildcard bind address, which as a target means this machine"
    if loopback_is_local and host in LOOPBACK_HOSTS:
        return "this container's own loopback, and Ollama runs on the host"
    return "unparseable as a URL"


def _note(notes: list[str]) -> str | None:
    return "; ".join(notes) if notes else None


def _flag(value: str | None) -> bool | None:
    value = (value or "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def _read(path: Path) -> str:
    """The file's text, or `""`. A signal that cannot be read is absent, not fatal."""
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _overlay_root(mountinfo: str) -> bool:
    """Is `/` an overlay mount, per `/proc/self/mountinfo`?

    True in every container built `FROM` an image, whatever the cgroup version,
    and false on an ordinary ext4/btrfs/xfs/APFS root. The fields before the
    ` - ` separator are optional and variable in number, so the filesystem type
    is read as the field after it rather than by index from the left.
    """
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[4] != "/":
            continue
        if "-" in fields and fields.index("-") + 1 < len(fields):
            return fields[fields.index("-") + 1] == "overlay"
    return False


@dataclass(frozen=True)
class Settings:
    # --- Ollama ---------------------------------------------------------
    # Container-aware and bind-address-aware; see the block above
    # `resolve_ollama_target`. Reading `OLLAMA_HOST` directly here is the bug
    # this field used to have, and it broke precisely on the machines whose
    # host server had been configured correctly for containers.
    ollama_base_url: str = field(default_factory=lambda: resolve_ollama_target().url)

    # --- Models ---------------------------------------------------------
    # Defaults tuned for Mac Studio M4 Max / 36 GB unified memory.
    #
    # Split by ARCHITECTURE, not by size. On Apple Silicon, generation speed is
    # roughly bandwidth / bytes-read-per-token, so a dense model reads its whole
    # file per token while an MoE reads only its active experts. Measured here:
    #
    #   gemma4:31b  dense  19 GB  30.7B active   17.4 tok/s
    #   gemma4:26b  MoE    17 GB   3.8B active   81.7 tok/s   (4.7x)
    #
    # The orchestrator emits a few hundred tokens of schema-constrained JSON per
    # round; quality there is worth more than speed, so it gets the dense model.
    # The worker emits whole files - that is where the wall-clock goes - so it
    # gets the MoE.
    #
    # 19 + 17 GB of weights exceeds the ~27 GB GPU budget, so keep
    # OLLAMA_MAX_LOADED_MODELS=1 and let Ollama swap. A swap measured 6.7 s here,
    # roughly 2 swaps per round, against minutes saved on every worker call.
    # Set both to gemma4:26b to eliminate swapping entirely if you prefer.
    orchestrator_model: str = field(default_factory=lambda: _env("SWARM_ORCHESTRATOR_MODEL", "gemma4:31b"))
    worker_model: str = field(default_factory=lambda: _env("SWARM_WORKER_MODEL", "gemma4:26b"))

    # Low temperature: we want obedient structure, not creativity.
    orchestrator_temperature: float = 0.0
    worker_temperature: float = 0.1

    # Context window requested from Ollama.
    #
    # IMPORTANT: gemma4 advertises 256K context, but the KV cache for a 31B
    # model at 256K would need more memory than the weights themselves. Never
    # request the advertised maximum. 16-32K is plenty for one focused task,
    # and curated context beats large context on every model we tested.
    # Ollama allocates a runner per (model, options) combination, so if you set
    # both roles to the SAME model, keep these two values equal - differing
    # num_ctx would spawn a second runner and double your memory use.
    orchestrator_num_ctx: int = int(_env("SWARM_ORCH_CTX", "16384"))
    worker_num_ctx: int = int(_env("SWARM_WORKER_CTX", "16384"))

    # --- Repo / execution ----------------------------------------------
    repo_path: str = field(default_factory=lambda: _env("SWARM_REPO", os.getcwd()))
    worktree_root: str = field(default_factory=lambda: _env("SWARM_WORKTREES", ".swarm/worktrees"))
    verify_command: str = field(default_factory=lambda: _env("SWARM_VERIFY", "python -m pytest -q"))

    # --- Safety rails (the part that stops runaway cost / infinite loops) -
    # 2 on a 36 GB machine: Ollama allocates KV cache as
    # OLLAMA_NUM_PARALLEL x num_ctx, so raising this raises memory linearly.
    max_workers_parallel: int = int(_env("SWARM_MAX_PARALLEL", "2"))
    max_rounds: int = int(_env("SWARM_MAX_ROUNDS", "8"))
    max_stalls: int = int(_env("SWARM_MAX_STALLS", "2"))
    max_attempts_per_task: int = int(_env("SWARM_MAX_ATTEMPTS", "3"))
    worker_timeout_s: int = int(_env("SWARM_WORKER_TIMEOUT", "600"))
    verify_timeout_s: int = int(_env("SWARM_VERIFY_TIMEOUT", "300"))

    # --- Persistence ----------------------------------------------------
    checkpoint_db: str = field(default_factory=lambda: _env("SWARM_CHECKPOINTS", ".swarm/checkpoints.sqlite"))


SETTINGS = Settings()
