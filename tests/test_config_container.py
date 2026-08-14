"""Where this process decides Ollama is, and the trap that decision exists for.

`OLLAMA_HOST` names two different things. To the server it is a **bind
address**; to a client it is a **target**. The value that lets containers reach
the host's Ollama is `OLLAMA_HOST=0.0.0.0:11434` in the server's environment,
and a client that reads the same variable is then pointed at the wildcard
address, which means "this machine" - the container, where nothing listens.

The reason that bug survives review is that it is nearly invisible on the
development host: BSD and Linux sockets turn a connect() to `0.0.0.0` into a
connect to loopback, so on the Mac the wrong value works by accident. It only
fails once the same environment reaches a container, and it fails there as a
connection error that reads like a stopped server. So the tests that matter
most in this file are the ones that assert a *correctly configured host* -
`OLLAMA_HOST=0.0.0.0:11434` exported, exactly as SETUP.md asks for - still
produces a URL something can dial.

Three properties, then:

**A bind address is never a target.** Whichever variable it arrives in, and
whether or not the wrong answer would have worked by accident here.

**Detection is evidence, not a marker file.** `/.dockerenv` is one signal and
this project's own host is a Mac, which cannot be running a Linux container
however many container-shaped files it has lying around. A misfire routes the
orchestrator at `host.docker.internal` and fails to resolve it, which is a
worse outcome than the default it replaced.

**The Linux failure says `--add-host`.** `host.docker.internal` is a name
Docker Desktop injects and a Linux daemon does not, so the same image that
works on a Mac reports DNS trouble for a name nobody typed. `explain_unresolvable`
is the sentence that turns that into a fix.

The last two tests are gated. `test_the_host_default_reaches_a_real_ollama`
needs the server (`--with-ollama`) and
`test_a_worker_container_reaches_the_host_ollama` needs the daemon and the
server (`--with-all`); between them they are this issue's "done when", proven
against the real thing rather than against a double.
"""

from __future__ import annotations

import json
import socket
import urllib.request
from pathlib import Path

import pytest

from swarm.config import (
    APIARY_OLLAMA_HOST_ENV,
    CONTAINER_OLLAMA_HOSTNAME,
    CONTAINER_OLLAMA_URL,
    HOST_OLLAMA_URL,
    IN_CONTAINER_ENV,
    OLLAMA_HOST_ENV,
    SETTINGS,
    OllamaTarget,
    Settings,
    client_target,
    container_evidence,
    explain_unresolvable,
    in_container,
    ollama_base_url,
    resolve_ollama_target,
)
from swarm.containers.manager import ContainerError, ContainerManager, DockerCLI, Handle
from swarm.run import Run

#: What SETUP.md tells the operator to give the *server* so containers can
#: reach it, and what is exported on the machine this project is developed on.
BIND_ADDRESS = "0.0.0.0:11434"

REPO = "shahrestani-me/apiary"
OBJECTIVE = "reach host ollama from inside the container"
BASE_COMMIT = "9f2c1ab3d4e5f60718293a4b5c6d7e8f90a1b2c3"


# --------------------------------------------------------------------------
# Fake filesystems
# --------------------------------------------------------------------------
#
# `in_container` takes a root so these can exist. The alternative is a
# detection test that only runs inside a container, i.e. one that runs nowhere.

#: A container's root really is an overlay mount, whatever the cgroup version.
CONTAINER_MOUNTINFO = (
    "1234 1233 0:161 / / rw,relatime - overlay overlay rw,lowerdir=/l,upperdir=/u\n"
    "1235 1234 0:162 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw\n"
)

#: An ordinary Linux host - including one that runs Docker, whose overlay
#: mounts are all somewhere that is not `/`.
HOST_MOUNTINFO = (
    "25 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n"
    "30 25 0:35 / /var/lib/docker/overlay2/9c1/merged rw,relatime shared:9 - overlay overlay rw\n"
)

#: Docker Desktop's private cgroup namespace names no runtime at all, which is
#: why the cgroup file cannot be the only signal.
CGROUP_V2_PRIVATE = "0::/\n"
CGROUP_V1_DOCKER = "12:pids:/docker/3f2a9b\n11:memory:/docker/3f2a9b\n"
CGROUP_KUBEPODS = "0::/kubepods/besteffort/pod9c1/3f2a9b\n"
CGROUP_HOST = "0::/init.scope\n"


def fs(
    tmp_path: Path,
    *,
    dockerenv: bool = False,
    containerenv: bool = False,
    cgroup: str | None = None,
    mountinfo: str | None = None,
) -> Path:
    """A root directory carrying exactly the signals named."""
    root = tmp_path / "root"
    (root / "proc" / "1").mkdir(parents=True)
    (root / "proc" / "self").mkdir(parents=True)
    if dockerenv:
        (root / ".dockerenv").write_text("")
    if containerenv:
        (root / "run").mkdir()
        (root / "run" / ".containerenv").write_text("")
    if cgroup is not None:
        (root / "proc" / "1" / "cgroup").write_text(cgroup)
    if mountinfo is not None:
        (root / "proc" / "self" / "mountinfo").write_text(mountinfo)
    return root


def container_root(tmp_path: Path) -> Path:
    """What Docker Desktop actually presents: a marker, `0::/`, an overlay root."""
    return fs(tmp_path, dockerenv=True, cgroup=CGROUP_V2_PRIVATE, mountinfo=CONTAINER_MOUNTINFO)


def host_root(tmp_path: Path) -> Path:
    return fs(tmp_path, cgroup=CGROUP_HOST, mountinfo=HOST_MOUNTINFO)


# --------------------------------------------------------------------------
# A bind address is not a target
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exported",
    [
        BIND_ADDRESS,
        "http://0.0.0.0:11434",
        "0.0.0.0",
        ":11434",
        "[::]:11434",
        "http://[::]:11434",
        "*:11434",
    ],
)
def test_a_bind_address_never_becomes_a_client_target(exported: str):
    """Every shape `ollama serve` accepts as "listen everywhere"."""
    assert client_target(exported) is None


def test_the_correctly_configured_host_still_gets_a_url_it_can_dial():
    """The whole ticket, on the host half.

    A machine whose Ollama has been configured for containers exports the bind
    address, and the naive `os.environ["OLLAMA_HOST"]` hands that straight to
    `langchain_ollama` as `base_url`. This is the assertion that fails on the
    machines that did everything right.
    """
    target = resolve_ollama_target({OLLAMA_HOST_ENV: BIND_ADDRESS}, container=False)

    assert target.url == HOST_OLLAMA_URL
    assert target.source == "default"
    # Silently ignoring a variable the operator believes in is undebuggable, so
    # the reason survives for `swarm doctor` (#32) to quote.
    assert target.note is not None
    assert OLLAMA_HOST_ENV in target.note
    assert "bind address" in target.note


def test_the_correctly_configured_host_still_gets_a_url_a_container_can_dial():
    """The same variable, inherited by a container, where 0.0.0.0 is fatal.

    On the host, connecting to `0.0.0.0` lands on loopback and the mistake is
    invisible. In a container it is the container.
    """
    target = resolve_ollama_target({OLLAMA_HOST_ENV: BIND_ADDRESS}, container=True)

    assert target.url == CONTAINER_OLLAMA_URL
    assert target.in_container
    assert target.needs_host_gateway


def test_settings_never_ships_a_bind_address_however_the_shell_is_configured(
    monkeypatch: pytest.MonkeyPatch,
):
    """The seam every other module reads: `SETTINGS.ollama_base_url`."""
    monkeypatch.setenv(OLLAMA_HOST_ENV, BIND_ADDRESS)
    monkeypatch.delenv(APIARY_OLLAMA_HOST_ENV, raising=False)
    # Pinned rather than detected: this suite runs on laptops and in CI, and
    # the property under test is about the variable, not about the machine.
    monkeypatch.setenv(IN_CONTAINER_ENV, "0")

    assert Settings().ollama_base_url == HOST_OLLAMA_URL


def test_the_shipped_settings_object_is_dialable_wherever_this_runs():
    """No environment produces a `SETTINGS` that a client cannot use.

    Cheap, and it is the assertion that would have caught the original default
    on this project's own host.
    """
    assert client_target(SETTINGS.ollama_base_url) == SETTINGS.ollama_base_url
    assert "://" in SETTINGS.ollama_base_url


def test_deciding_where_ollama_is_costs_no_dns(monkeypatch: pytest.MonkeyPatch):
    """`Settings` is built at import time, so resolution must not touch the network.

    A module that cannot be imported on a machine with broken DNS is a module
    that cannot print the diagnostic explaining the broken DNS.
    """

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("resolution did a name lookup")

    monkeypatch.setattr(socket, "gethostbyname", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setenv(IN_CONTAINER_ENV, "1")

    assert resolve_ollama_target({IN_CONTAINER_ENV: "1"}).url == CONTAINER_OLLAMA_URL
    assert Settings().ollama_base_url == CONTAINER_OLLAMA_URL


# --------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------


def test_nothing_set_defaults_by_detection():
    assert resolve_ollama_target({}, container=False).url == HOST_OLLAMA_URL
    assert resolve_ollama_target({}, container=True).url == CONTAINER_OLLAMA_URL


def test_an_explicit_client_target_wins_in_both_places():
    """The ticket's "an explicit override always wins", for values that are targets."""
    env = {APIARY_OLLAMA_HOST_ENV: "http://ollama.lan:11434"}

    for container in (False, True):
        target = resolve_ollama_target(env, container=container)
        assert target.url == "http://ollama.lan:11434"
        assert target.source == APIARY_OLLAMA_HOST_ENV
        assert target.note is None


def test_the_unambiguous_variable_outranks_ollamas_own():
    env = {
        APIARY_OLLAMA_HOST_ENV: "http://ollama.lan:11434",
        OLLAMA_HOST_ENV: "http://elsewhere:11434",
    }

    assert resolve_ollama_target(env, container=True).url == "http://ollama.lan:11434"


def test_ollama_host_is_honoured_when_it_is_actually_a_target():
    """Declining bind addresses is not declining the variable."""
    target = resolve_ollama_target({OLLAMA_HOST_ENV: "http://mac.local:11434"}, container=True)

    assert target.url == "http://mac.local:11434"
    assert target.source == OLLAMA_HOST_ENV
    assert not target.needs_host_gateway


def test_an_undialable_override_falls_through_to_the_next_candidate():
    """A useless APIARY_OLLAMA_HOST must not shadow a usable OLLAMA_HOST."""
    env = {APIARY_OLLAMA_HOST_ENV: BIND_ADDRESS, OLLAMA_HOST_ENV: "http://mac.local:11434"}

    target = resolve_ollama_target(env, container=True)

    assert target.url == "http://mac.local:11434"
    assert target.source == OLLAMA_HOST_ENV
    assert target.note is not None and APIARY_OLLAMA_HOST_ENV in target.note


def test_an_empty_variable_is_unset_rather_than_wrong():
    """compose.yaml passes `${VAR:-}` through, and empty means "not configured"."""
    env = {APIARY_OLLAMA_HOST_ENV: "", OLLAMA_HOST_ENV: "   "}

    target = resolve_ollama_target(env, container=False)

    assert target.url == HOST_OLLAMA_URL
    assert target.note is not None
    assert APIARY_OLLAMA_HOST_ENV not in target.note


def test_loopback_in_ollama_host_is_the_container_itself():
    """The other half of the inherited-shell problem, and the commoner half.

    `OLLAMA_HOST=localhost:11434` is what a developer's shell says, and inside
    a container it names the container. There is nothing there, and the error
    it produces sends the reader to `ollama serve`.
    """
    target = resolve_ollama_target({OLLAMA_HOST_ENV: "localhost:11434"}, container=True)

    assert target.url == CONTAINER_OLLAMA_URL
    assert target.note is not None and "loopback" in target.note

    # ... and on the host it is simply correct.
    assert (
        resolve_ollama_target({OLLAMA_HOST_ENV: "localhost:11434"}, container=False).url
        == HOST_OLLAMA_URL
    )


def test_loopback_in_the_unambiguous_variable_is_taken_at_its_word():
    """`--network host` is a real way to run this, and only the operator knows.

    APIARY_OLLAMA_HOST is read as a target and as nothing else, so a loopback
    there is a statement rather than a leaked server setting.
    """
    target = resolve_ollama_target({APIARY_OLLAMA_HOST_ENV: "localhost:11434"}, container=True)

    assert target.url == "http://localhost:11434"
    assert target.source == APIARY_OLLAMA_HOST_ENV


def test_ollama_base_url_is_the_url_of_the_resolved_target(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(IN_CONTAINER_ENV, "1")
    monkeypatch.delenv(OLLAMA_HOST_ENV, raising=False)
    monkeypatch.delenv(APIARY_OLLAMA_HOST_ENV, raising=False)

    assert ollama_base_url() == CONTAINER_OLLAMA_URL


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exported", "expected"),
    [
        # The forms `ollama` itself accepts, which is what a shell will hold.
        ("localhost:11434", "http://localhost:11434"),
        ("mac.local", "http://mac.local:11434"),
        ("127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("[::1]:11434", "http://[::1]:11434"),
        # Already a URL: kept, minus the trailing slash langchain would double.
        ("http://mac.local:11434/", "http://mac.local:11434"),
        ("http://mac.local", "http://mac.local:11434"),
        # A port glued onto the end would land after the path, not the host.
        ("http://box/ollama", "http://box:11434/ollama"),
        # https carries its own default port; inventing 11434 would break it.
        ("https://ollama.example.com", "https://ollama.example.com"),
        ("  http://mac.local:11434  ", "http://mac.local:11434"),
    ],
)
def test_a_target_is_normalised_to_something_langchain_can_use(exported: str, expected: str):
    assert client_target(exported) == expected


@pytest.mark.parametrize("exported", ["", "   ", None, "http://[::1:11434", "mac.local:port"])
def test_a_value_that_is_not_a_url_is_not_a_target(exported: str | None):
    """Unparseable is declined, never guessed at and never raised at import."""
    assert client_target(exported) is None


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def test_a_mac_with_a_container_shaped_filesystem_is_still_a_mac(tmp_path: Path):
    """The misfire this ticket names, and the one that costs a working setup.

    A stray `/.dockerenv` on the development host - left by a script, restored
    from a backup - would otherwise route the orchestrator at
    `host.docker.internal`, a name that resolves nowhere outside a container.
    """
    root = container_root(tmp_path)

    assert in_container({}, platform="darwin", root=root) is False
    assert resolve_ollama_target({}, container=False).url == HOST_OLLAMA_URL


def test_one_signal_is_not_evidence(tmp_path: Path):
    """A Linux host with a leftover marker file and nothing else about it."""
    root = fs(tmp_path, dockerenv=True, cgroup=CGROUP_HOST, mountinfo=HOST_MOUNTINFO)

    assert container_evidence(root) == (".dockerenv",)
    assert in_container({}, platform="linux", root=root) is False


def test_docker_desktop_is_recognised_from_inside(tmp_path: Path):
    """The real shape: a marker, a cgroup file that names nothing, an overlay root."""
    root = container_root(tmp_path)

    assert in_container({}, platform="linux", root=root) is True
    assert resolve_ollama_target({}, container=True).url == CONTAINER_OLLAMA_URL


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # cgroup v1, or cgroupns=host: the runtime names itself.
        ({"dockerenv": True, "cgroup": CGROUP_V1_DOCKER}, ("cgroup",)),
        # Kubernetes, which drops no marker file.
        ({"cgroup": CGROUP_KUBEPODS, "mountinfo": CONTAINER_MOUNTINFO}, ("cgroup", "overlay-root")),
        # Podman.
        ({"containerenv": True, "mountinfo": CONTAINER_MOUNTINFO}, ("overlay-root",)),
    ],
)
def test_containers_are_recognised_by_more_than_one_runtime(
    tmp_path: Path, kwargs: dict, expected: tuple[str, ...]
):
    root = fs(tmp_path, **kwargs)

    assert in_container({}, platform="linux", root=root) is True
    for signal in expected:
        assert signal in container_evidence(root)


def test_a_docker_host_is_not_a_container_because_it_has_overlay_mounts(tmp_path: Path):
    """`/var/lib/docker/overlay2/.../merged` is not `/`."""
    root = host_root(tmp_path)

    assert container_evidence(root) == ()
    assert in_container({}, platform="linux", root=root) is False


def test_a_root_that_cannot_be_read_reports_nothing_rather_than_crashing(tmp_path: Path):
    """Detection is best-effort; a missing /proc is an answer, not an exception."""
    root = tmp_path / "empty"
    root.mkdir()

    assert container_evidence(root) == ()
    assert in_container({}, platform="linux", root=root) is False


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("true", True), ("yes", True)])
def test_the_operator_can_overrule_the_heuristic(tmp_path: Path, value: str, expected: bool):
    """Detection is evidence; an operator who knows should not have to argue."""
    root = host_root(tmp_path)

    assert in_container({IN_CONTAINER_ENV: value}, platform="darwin", root=root) is expected


@pytest.mark.parametrize("value", ["0", "false", "no"])
def test_the_override_works_in_the_denying_direction_too(tmp_path: Path, value: str):
    root = container_root(tmp_path)

    assert in_container({IN_CONTAINER_ENV: value}, platform="linux", root=root) is False


@pytest.mark.parametrize("value", ["", "  ", "maybe"])
def test_an_unreadable_override_falls_back_to_the_evidence(tmp_path: Path, value: str):
    """A typo must not silently mean "no"."""
    root = container_root(tmp_path)

    assert in_container({IN_CONTAINER_ENV: value}, platform="linux", root=root) is True


# --------------------------------------------------------------------------
# The Linux failure
# --------------------------------------------------------------------------


def refuses(host: str) -> None:
    raise socket.gaierror("[Errno -2] Name or service not known")


def resolves(host: str) -> str:
    return "192.168.65.254"


def test_an_unresolvable_container_default_names_add_host():
    """The Linux case the ticket asks to fail comprehensibly.

    Docker Desktop injects `host.docker.internal`; a Linux daemon does not. The
    naked failure is a DNS error for a name the operator never typed, which is
    the least actionable message in the system.
    """
    target = resolve_ollama_target({}, container=True)

    complaint = explain_unresolvable(target, resolve=refuses)

    assert complaint is not None
    assert f"--add-host {CONTAINER_OLLAMA_HOSTNAME}:host-gateway" in complaint
    assert "extra_hosts" in complaint
    assert APIARY_OLLAMA_HOST_ENV in complaint


def test_a_resolvable_target_has_nothing_to_explain():
    target = resolve_ollama_target({}, container=True)

    assert explain_unresolvable(target, resolve=resolves) is None


def test_an_unresolvable_target_that_is_not_the_gateway_name_says_something_else():
    """`--add-host` would be nonsense advice for a hostname the operator chose."""
    target = resolve_ollama_target({APIARY_OLLAMA_HOST_ENV: "http://ollama.lan:11434"})

    complaint = explain_unresolvable(target, resolve=refuses)

    assert complaint is not None
    assert "ollama.lan" in complaint
    assert "--add-host" not in complaint


def test_the_name_looked_up_is_a_hostname_and_not_a_url():
    """`gethostbyname("http://x:11434")` fails for the wrong reason."""
    asked: list[str] = []

    def record(host: str) -> str:
        asked.append(host)
        return "127.0.0.1"

    explain_unresolvable(OllamaTarget(CONTAINER_OLLAMA_URL, source="default"), resolve=record)

    assert asked == [CONTAINER_OLLAMA_HOSTNAME]


# --------------------------------------------------------------------------
# Against the real thing
# --------------------------------------------------------------------------


def version_probe(url: str) -> str:
    """The smallest request that proves an Ollama answered."""
    with urllib.request.urlopen(f"{url}/api/version", timeout=10) as response:
        return json.loads(response.read())["version"]


@pytest.mark.ollama
def test_the_host_default_reaches_a_real_ollama():
    """Half of "done when", from the host, with nothing set by hand.

    Note what this asserts on the development machine: `OLLAMA_HOST` is
    exported there as `0.0.0.0:11434`, and the URL under test is the one this
    module produced *instead* of it.
    """
    assert version_probe(resolve_ollama_target(container=False).url)


CANDIDATE_IMAGES = ("apiary-worker", "apiary-worker:dev", "python:3.12-slim")


@pytest.fixture(scope="module")
def python_image() -> str:
    """A locally present image with a Python in it. Nothing is pulled.

    Same shape as `test_limits.py`'s `trivial_image`, and a shell is not enough
    here: the probe needs an HTTP client, and neither busybox nor a slim Debian
    is guaranteed one.
    """
    docker = DockerCLI()
    for name in CANDIDATE_IMAGES:
        try:
            docker("image", "inspect", "--format", "{{.Id}}", name)
        except ContainerError:
            continue
        return name
    pytest.skip(
        "no local image with a Python to probe from; build one with "
        "`docker build -f Dockerfile.worker -t apiary-worker .`"
    )


@pytest.mark.docker
@pytest.mark.ollama
def test_a_worker_container_reaches_the_host_ollama(python_image: str):
    """The other half of "done when", and the one no double can stand in for.

    A container spawned the way the dispatcher spawns them - `--add-host`
    included, no `OLLAMA_HOST` chosen by a human - dials the URL this module
    picks for a container and gets a version back. The probe is written out as
    a literal rather than importing `swarm.config` inside the container,
    because the image holds whatever `swarm` was installed when it was built,
    which is not the code under test.
    """
    manager = ContainerManager(run=Run.start(REPO, OBJECTIVE), image=python_image, env={})
    probe = (
        "import urllib.request;"
        f"print(urllib.request.urlopen('{CONTAINER_OLLAMA_URL}/api/version', timeout=15).read())"
    )
    handle: Handle = manager.spawn(13, BASE_COMMIT, entrypoint="python", command=["-c", probe])
    try:
        exit_code = manager.wait(handle, timeout_s=60)
        logs = manager.logs(handle)
    finally:
        manager.dispose(handle)

    assert exit_code == 0, (
        f"a container could not reach {CONTAINER_OLLAMA_URL}: {logs}\n"
        "The host's Ollama must be listening on more than loopback for this to work - "
        "`launchctl setenv OLLAMA_HOST 0.0.0.0:11434` on macOS, where the app never reads "
        "your shell. That value is the SERVER's bind address; nothing in this repository "
        "reads it as a client target, which is what the rest of this file is about."
    )
    assert "version" in logs
    assert manager.find() == []
