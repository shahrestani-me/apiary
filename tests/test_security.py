"""Tests for the credential, egress and Docker-API policy.

Four claims, one per section of `src/swarm/security.py`, and each of them is
about something that must *not* happen - which is the hard kind to test,
because a check that looks at nothing passes every negative assertion
perfectly. So each negative here is paired with the positive control that
proves the check can see at all.

**The token cannot be account-wide.** `gh auth token` prints a `gho_`, which is
the most convenient credential on a developer's machine and reaches every
repository the account can reach. The control is that a fine-grained token is
accepted.

**The allowlist and the proxy configuration cannot drift.** They are two
renderings of one tuple, and the test compiles the generated regexes and checks
them against `EgressPolicy.allows` over a corpus that includes the hostnames an
unanchored pattern would wrongly admit.

**The one path that creates containers creates no privileged one.**
`ContainerManager.spawn`'s real argv goes through `assert_unprivileged`, so this
is a regression test on `containers/manager.py` rather than on a fixture. The
control is a corpus of argvs that must each be rejected, one per rule.

**The token reaches no artifact on disk.** A container that deliberately echoes
its token is run through the real capture path - `DockerCLI`, its `Redactor`,
`dispose_container` - and what comes out is written to a run directory and
scanned. The control writes the raw token to the same directory and requires
the scan to find it.

Hermetic throughout: `EchoingRunner` answers where a `docker` subprocess would,
so the real capture boundary is under test rather than mocked out.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pytest

from swarm.containers.manager import (
    INHERITED_ENV,
    STACK_IMAGES_ENV,
    ContainerManager,
    Handle,
    Redactor,
    dispose_container,
)
from swarm.github.refs import task_ref
from swarm.mcp.client import TRACKER_ENDPOINT_ENV, TRACKER_TOKEN_ENV
from swarm.run import Run
from swarm.security import (
    DOCKER_HOST_URL,
    EGRESS_ALLOWLIST,
    FORBIDDEN_PERMISSIONS,
    MCP_HOSTS,
    REQUIRED_PERMISSIONS,
    PROVISION_PERMISSIONS,
    PROVISION_TOKEN_ENV,
    SOCKET_PROXY_ENV,
    SOCKET_PROXY_HOST,
    WORKER_NETWORK,
    WORKER_PERMISSIONS,
    CredentialError,
    EgressPolicy,
    PolicyError,
    assert_no_provision_token,
    assert_provision_token,
    assert_scoped_token,
    assert_unprivileged,
    classify_token,
    find_secrets,
    scan_artifacts,
    worker_create_flags,
)

REPO = "shahrestani-me/apiary"
BASE_COMMIT = "9f2c1ab3d4e5f60718293a4b5c6d7e8f90a1b2c3"
CONTAINER_ID = "c0ffee" + "0" * 58

#: Deliberately *not* GitHub-shaped, for the reason `test_container_manager`
#: gives: a `ghp_...` string is caught by pattern alone, and these tests would
#: then pass without the enrolment path working at all.
TOKEN = "s3cr3t-push-credential-9f2c1ab3"

COMPOSE = Path(__file__).resolve().parents[1] / "compose.yaml"


@pytest.fixture()
def run() -> Run:
    return Run.start(REPO, "harden the credential path", run_id="apiary-20260814-120000-abcd")


# --------------------------------------------------------------------------
# The daemon double
# --------------------------------------------------------------------------


@dataclass
class EchoingRunner:
    """A `Runner` whose containers print `echo_text` when their logs are read.

    Records every argv, which is what the privilege assertions read. `docker
    logs` is the only command that answers with anything interesting, because
    the leak being tested for is a worker that printed its own environment.
    """

    echo_text: str = ""
    calls: list[list[str]] = field(default_factory=list)

    def __call__(
        self, argv: Sequence[str], *, timeout_s: float | None, merge: bool
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        stdout = self.echo_text if argv[1] == "logs" else CONTAINER_ID + "\n"
        return subprocess.CompletedProcess(list(argv), 0, stdout, "")

    def argv_for(self, subcommand: str) -> list[str]:
        for call in self.calls:
            if call[1] == subcommand:
                return call
        raise AssertionError(f"no {subcommand!r} command was issued")


# --------------------------------------------------------------------------
# 1. The token
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "kind"),
    [
        ("github_pat_11ABCDEFG0abcdefghijklmnop", "fine-grained"),
        ("ghs_16charsminimum0000000000", "app"),
        ("ghp_16charsminimum0000000000", "account-wide"),
        ("gho_16charsminimum0000000000", "account-wide"),
        ("ghu_16charsminimum0000000000", "account-wide"),
        ("ghr_16charsminimum0000000000", "account-wide"),
        ("v1.0123456789abcdef", "unrecognised"),
        ("", "unrecognised"),
        (None, "unrecognised"),
    ],
)
def test_token_kinds_are_told_apart_by_prefix(token: str | None, kind: str) -> None:
    assert classify_token(token) == kind


def test_a_fine_grained_token_is_accepted() -> None:
    """The control: the check has to admit the token the docs tell you to mint."""
    assert assert_scoped_token("github_pat_11ABCDEFG0abcdefghijklmnop") == "fine-grained"


def test_an_app_installation_token_is_accepted() -> None:
    assert assert_scoped_token("ghs_16charsminimum0000000000") == "app"


@pytest.mark.parametrize("prefix", ["ghp_", "gho_", "ghu_", "ghr_"])
def test_an_account_wide_token_is_refused(prefix: str) -> None:
    """A classic or OAuth token is scoped to verbs, so it reaches every repo."""
    with pytest.raises(CredentialError) as caught:
        assert_scoped_token(prefix + "16charsminimum0000000000")
    # The message has to say what to do instead; a refusal with no remedy gets
    # worked around rather than fixed.
    assert "fine-grained" in str(caught.value)
    assert "contents:write" in str(caught.value)


def test_an_absent_token_is_refused_before_anything_spends_time_on_it() -> None:
    with pytest.raises(CredentialError, match="GITHUB_TOKEN"):
        assert_scoped_token(None)


def test_an_unrecognised_prefix_is_refused_unless_asked_for() -> None:
    with pytest.raises(CredentialError, match="allow_unrecognised"):
        assert_scoped_token("some-enterprise-token")
    assert assert_scoped_token("some-enterprise-token", allow_unrecognised=True) == "unrecognised"


def test_the_required_permissions_are_these_and_only_these() -> None:
    """`actions` is read-only on purpose, and it earned its place the hard way.

    The documented permission for check runs is `checks`, and GitHub does not
    offer it when minting a fine-grained PAT - so a least-privilege token gets
    403 on `/commits/{ref}/check-runs` and on the combined-status endpoint no
    matter what it is granted. `actions:read` is grantable and answers 200 for
    the same commit, which is why the merge gate reads workflow runs.

    Write is refused: `actions:write` would let a worker re-run, cancel or
    delete the workflow judging it - the failure `workflows` is forbidden for,
    reached by a different door.
    """
    assert REQUIRED_PERMISSIONS == {
        "contents": "write",
        "pull_requests": "write",
        "issues": "write",
        "actions": "read",
        "metadata": "read",
    }
    assert REQUIRED_PERMISSIONS["actions"] == "read"


def test_a_worker_needs_no_issue_write() -> None:
    """#148, as the permission it removes rather than as the call it deletes.

    `worker/pr.py` used to apply `swarm:review` after opening its pull request,
    which made `issues:write` a requirement of a credential held by a container
    running model-generated code. The orchestrator derives `review` from the
    open pull request and writes the label itself, so the worker's half of the
    work key is strictly smaller - and this is the assertion that keeps it that
    way, because a scope nobody re-checks is a scope that grows back.
    """
    assert WORKER_PERMISSIONS == {
        "contents": "write",
        "pull_requests": "write",
        # A read: `entrypoint` still reads the contract and the retry comments.
        # #151 moves those behind the tracker MCP server.
        "issues": "read",
        "metadata": "read",
    }
    # Strictly a narrowing of the work key, never a different set: a permission
    # here that the orchestrator's token does not carry would be a scope apiary
    # asks nobody to grant, which fails as a 403 three minutes into a run.
    assert set(WORKER_PERMISSIONS) <= set(REQUIRED_PERMISSIONS)
    assert not set(WORKER_PERMISSIONS) & set(FORBIDDEN_PERMISSIONS)


def test_a_refusal_on_the_worker_path_does_not_ask_for_an_issue_write() -> None:
    """The remedy in the message is what a human actually grants.

    A refusal that names `issues:write` for a worker credential is how the
    permission #148 removed comes back: nobody mints two tokens to be told the
    narrower one is wrong.
    """
    with pytest.raises(CredentialError) as caught:
        assert_scoped_token(
            "ghp_16charsminimum0000000000", permissions=WORKER_PERMISSIONS
        )

    message = str(caught.value)
    assert "issues:read" in message
    assert "issues:write" not in message
    # The default is still the orchestrator's list, which does ask for it.
    with pytest.raises(CredentialError) as work_key:
        assert_scoped_token("ghp_16charsminimum0000000000")
    assert "issues:write" in str(work_key.value)


def test_workflows_is_forbidden() -> None:
    """The sharp one: with it, generated code can rewrite the gate that checks it."""
    assert "workflows" in FORBIDDEN_PERMISSIONS
    # `actions` appears in both lists and that is not a contradiction: the work
    # key needs it read-only to see whether CI passed, and
    # `FORBIDDEN_PERMISSIONS` is about write access - re-running or deleting the
    # workflow that judges you is the failure, reading its result is not.
    overlap = set(REQUIRED_PERMISSIONS) & set(FORBIDDEN_PERMISSIONS)
    assert overlap == {"actions"}
    assert REQUIRED_PERMISSIONS["actions"] == "read"


# --------------------------------------------------------------------------
# 2. Egress
# --------------------------------------------------------------------------

#: Hosts the policy must admit, and hosts it must refuse. The refusals are the
#: interesting half: `github.com.attacker.net` and `notgithub.com` are the two
#: shapes a careless allowlist admits, and both are hostnames someone else owns.
ALLOWED_HOSTS = [
    "github.com",
    "api.github.com",
    "codeload.github.com",
    "objects.githubusercontent.com.github.com",
    "host.docker.internal",
    "https://api.github.com/repos/x/y",
    "HOST.DOCKER.INTERNAL:11434",
    "mcp.linear.app",
    "https://mcp.linear.app/mcp",
]
REFUSED_HOSTS = [
    "gitlab.com",
    "pypi.org",
    "notgithub.com",
    "github.com.attacker.net",
    "attacker.net",
    "169.254.169.254",
    "",
    # The remote GitHub MCP server, absent on purpose: it advertises the
    # classic OAuth scopes `assert_scoped_token` refuses, so the GitHub tracker
    # runs as a local stdio server against api.github.com instead (#143).
    "api.githubcopilot.com",
    # Jira is deferred, and a hole for a tracker nobody configured is exactly
    # the quiet widening this file exists to catch.
    "api.atlassian.com",
    "linear.app",
]


@pytest.mark.parametrize("host", ALLOWED_HOSTS)
def test_the_allowlist_admits_github_and_the_hosts_ollama(host: str) -> None:
    assert EgressPolicy().allows(host)


@pytest.mark.parametrize("host", REFUSED_HOSTS)
def test_the_allowlist_refuses_everything_else(host: str) -> None:
    assert not EgressPolicy().allows(host)


def test_the_generated_filter_agrees_with_the_predicate() -> None:
    """The proxy's configuration and the policy cannot mean different things.

    `filter_lines` is what tinyproxy enforces and `allows` is what everything
    else reasons about; they are one tuple rendered twice, and this is the
    assertion that keeps them that way.
    """
    policy = EgressPolicy()
    patterns = [re.compile(line, re.IGNORECASE) for line in policy.filter_lines()]

    for host in ALLOWED_HOSTS + REFUSED_HOSTS:
        bare = host.split("://")[-1].split("/")[0].split(":")[0]
        matched = any(pattern.match(bare) for pattern in patterns)
        assert matched == policy.allows(host), host


def test_the_filter_is_anchored_at_both_ends() -> None:
    for line in EgressPolicy().filter_lines():
        assert line.startswith("^") and line.endswith("$")


def test_the_default_allowlist_is_the_goal_sentence_and_nothing_more() -> None:
    assert EgressPolicy().hosts == EGRESS_ALLOWLIST
    assert set(EGRESS_ALLOWLIST) == {
        "github.com",
        "api.github.com",
        "codeload.github.com",
        "host.docker.internal",
        # The one destination ADR 0001 adds. `MCP_HOSTS` says at length why it
        # is one and not three.
        "mcp.linear.app",
    }
    assert MCP_HOSTS == ("mcp.linear.app",)


def test_the_tracker_credential_never_reaches_a_worker() -> None:
    """What actually makes the MCP endpoint the orchestrator's alone.

    tinyproxy filters on hostname, so `mcp.linear.app` on the allowlist is
    reachable by every container on `apiary-egress` - a worker included. The
    confinement is the credential, exactly as `assert_scoped_token` argues for
    `github.com`: one host serves every customer and only the token knows
    which. `INHERITED_ENV` is the list that decides, so it is the list this
    asserts on, and a worker that reached the endpoint without the credential
    is answered 401.
    """
    assert TRACKER_TOKEN_ENV not in INHERITED_ENV
    assert TRACKER_ENDPOINT_ENV not in INHERITED_ENV
    # The full list, pinned so growing it is a decision made here. The three
    # `SWARM_WORKER_*` knobs are tuning values, not credentials - they exist
    # in the list so an operator's export reaches the `Settings` the worker
    # reads inside the container - and none names a secret.
    #
    # `SWARM_WORKER_MODEL_OPTIONS` is the newest and the only one that could
    # name a host, through a provider's `base_url` option. It still opens no
    # egress: tinyproxy allows `EGRESS_ALLOWLIST` and nothing else, so a worker
    # pointed at an endpoint that is not on the list is refused by the proxy
    # rather than reaching it. What the option carries is where to dial and
    # which variable or profile holds the credential - never the credential.
    # **Whether a worker may hold a model-provider credential at all is #269**,
    # and this list deliberately does not answer it.
    assert set(INHERITED_ENV) == {
        "GITHUB_TOKEN",
        "OLLAMA_HOST",
        "SWARM_WORKER_CTX",
        "SWARM_WORKER_MODEL",
        "SWARM_WORKER_MODEL_OPTIONS",
    }


def test_a_package_index_is_off_until_an_operator_asks_for_it() -> None:
    """The widening knob is opt-in, and opting in is one variable, not an edit."""
    assert not EgressPolicy.from_env({}).allows("pypi.org")

    widened = EgressPolicy.from_env({"APIARY_EGRESS_ALLOW": "pypi.org, files.pythonhosted.org"})
    assert widened.allows("pypi.org")
    assert widened.allows("files.pythonhosted.org")
    # ... and widening never narrows: the defaults are still there.
    assert widened.allows("api.github.com")


def test_the_proxy_environment_is_set_in_both_cases() -> None:
    """One client reading only the spelling nobody set is a silent bypass."""
    env = EgressPolicy().proxy_env()
    assert env["HTTP_PROXY"] == env["http_proxy"] == "http://egress-proxy:8888"
    assert env["HTTPS_PROXY"] == env["https_proxy"] == "http://egress-proxy:8888"
    assert "egress-proxy" in env["NO_PROXY"] == env["no_proxy"]


def test_the_docker_socket_proxy_is_never_reached_through_the_egress_proxy() -> None:
    """The `docker` CLI honours HTTP_PROXY for its own API calls.

    Without the socket proxy in NO_PROXY, `docker version` is sent to the
    egress proxy, which refuses it - `403 Filtered` from tinyproxy, a
    deny-by-default egress rule rejecting traffic that should never have left
    the container. It presents as "the daemon did not answer", which points at
    the socket proxy rather than at the egress one.
    """
    env = EgressPolicy().proxy_env()
    for spelling in ("NO_PROXY", "no_proxy"):
        assert SOCKET_PROXY_HOST in env[spelling].split(",")

    # DOCKER_HOST names the same host, so the two cannot drift apart.
    assert SOCKET_PROXY_HOST in DOCKER_HOST_URL


# --------------------------------------------------------------------------
# 3. The Docker API
# --------------------------------------------------------------------------


def test_the_socket_proxy_grants_containers_and_the_version_handshake() -> None:
    """Exactly what `containers/manager.py` calls, and the handshake before it."""
    assert SOCKET_PROXY_ENV["CONTAINERS"] == "1"
    assert SOCKET_PROXY_ENV["POST"] == "1"
    assert SOCKET_PROXY_ENV["ALLOW_START"] == "1"
    assert SOCKET_PROXY_ENV["ALLOW_STOP"] == "1"
    # Without /version the CLI cannot negotiate an API version and every
    # command fails before it is routed anywhere.
    assert SOCKET_PROXY_ENV["VERSION"] == "1"


@pytest.mark.parametrize("endpoint", ["EXEC", "BUILD", "IMAGES", "VOLUMES", "NETWORKS", "SWARM", "SYSTEM"])
def test_the_socket_proxy_denies_everything_else(endpoint: str) -> None:
    assert SOCKET_PROXY_ENV[endpoint] == "0"


def test_the_socket_proxy_surface_is_stated_in_full() -> None:
    """Every key explicit, including the ones whose default is already off.

    "This is off" and "nobody thought about it" read identically in a file that
    omits them, and the next person to widen the surface reads this one.
    """
    assert set(SOCKET_PROXY_ENV) >= {
        "AUTH", "BUILD", "COMMIT", "CONFIGS", "DISTRIBUTION", "EXEC", "IMAGES",
        "INFO", "NETWORKS", "NODES", "PLUGINS", "SECRETS", "SERVICES", "SESSION",
        "SWARM", "SYSTEM", "TASKS", "VOLUMES",
    }


PRIVILEGED_ARGVS = [
    ["create", "--privileged", "apiary-worker"],
    ["create", "--cap-add", "SYS_ADMIN", "apiary-worker"],
    ["create", "--cap-add=NET_ADMIN", "apiary-worker"],
    ["create", "--device", "/dev/kmsg", "apiary-worker"],
    ["create", "--device-cgroup-rule", "c 1:* rmw", "apiary-worker"],
    ["create", "--group-add", "docker", "apiary-worker"],
    ["create", "--pid", "host", "apiary-worker"],
    ["create", "--pid=host", "apiary-worker"],
    ["create", "--ipc=host", "apiary-worker"],
    ["create", "--userns=host", "apiary-worker"],
    ["create", "--network=host", "apiary-worker"],
    ["create", "--security-opt", "seccomp=unconfined", "apiary-worker"],
    ["create", "--security-opt=apparmor=unconfined", "apiary-worker"],
    ["create", "-v", "/var/run/docker.sock:/var/run/docker.sock", "apiary-worker"],
    ["create", "--volume", "/:/host", "apiary-worker"],
    ["create", "--mount", "type=bind,source=/etc,target=/host-etc", "apiary-worker"],
    ["create", "--user", "0", "apiary-worker"],
    ["create", "--user", "root:root", "apiary-worker"],
]


@pytest.mark.parametrize("argv", PRIVILEGED_ARGVS, ids=lambda argv: argv[1])
def test_a_privileged_create_is_refused(argv: list[str]) -> None:
    """One case per rule, because the proxy cannot see any of them.

    All of these travel in the body of the single `POST /containers/create`
    the socket proxy has to allow, so this function is the only thing that
    looks at them at all.
    """
    with pytest.raises(PolicyError):
        assert_unprivileged(argv)


def test_the_confinement_flags_are_themselves_unprivileged() -> None:
    """The control: what a worker is meant to be created with must pass."""
    flags = worker_create_flags()
    assert flags[:2] == ["--network", WORKER_NETWORK]
    assert "--cap-drop" in flags and "no-new-privileges:true" in flags
    assert_unprivileged(["create", *flags, "apiary-worker"])


def test_the_one_path_that_creates_containers_creates_no_privileged_one(run: Run) -> None:
    """A regression test on `ContainerManager.spawn`, not on a fixture.

    `spawn` builds the only `docker create` this system ever issues. If a
    future change to it adds a mount, a capability or a host namespace, this
    fails here rather than in production, where the socket proxy would route
    it through without looking.
    """
    runner = EchoingRunner()
    manager = ContainerManager(run=run, env={"GITHUB_TOKEN": TOKEN}, runner=runner)
    manager.spawn(task_ref(28), BASE_COMMIT, issue=28)

    assert_unprivileged(runner.argv_for("create"))


def test_the_docker_host_url_names_the_proxy_and_not_the_socket() -> None:
    assert DOCKER_HOST_URL == "tcp://docker-socket-proxy:2375"
    assert "docker.sock" not in DOCKER_HOST_URL


# --------------------------------------------------------------------------
# 4. Artifacts
# --------------------------------------------------------------------------


def test_a_worker_that_echoes_its_token_leaves_it_in_no_artifact(
    run: Run, tmp_path: Path
) -> None:
    """The "done when" clause of #28, end to end through the real capture path.

    The container prints its whole environment, as a shell trace or a debug
    dump would. What `dispose` hands back is what #29 writes to disk, so
    writing exactly that to a run directory and scanning it is the same
    question asked of the same string.
    """
    leaky = (
        "+ git push https://x-access-token:" + TOKEN + "@github.com/o/n.git\n"
        f"GITHUB_TOKEN={TOKEN}\n"
        "fatal: could not read Username for 'https://github.com'\n"
    )
    runner = EchoingRunner(echo_text=leaky)
    manager = ContainerManager(run=run, env={"GITHUB_TOKEN": TOKEN}, runner=runner)

    handle = manager.spawn(task_ref(28), BASE_COMMIT, issue=28)
    captured = manager.dispose(handle)

    artifacts = run.artifacts_dir(tmp_path)
    artifacts.mkdir(parents=True)
    (artifacts / "worker-28.log").write_text(captured, encoding="utf-8")

    assert scan_artifacts(artifacts, literals=[TOKEN]) == []
    assert TOKEN not in captured


def test_the_scan_finds_a_token_that_was_written_raw(run: Run, tmp_path: Path) -> None:
    """The control. A scanner that looks at nothing passes the test above.

    Two findings, because both halves of the detector have to be live: the
    literal this process registered, and the GitHub-shaped string it was never
    told about.
    """
    artifacts = run.artifacts_dir(tmp_path)
    artifacts.mkdir(parents=True)
    (artifacts / "worker-28.log").write_text(
        f"GITHUB_TOKEN={TOKEN}\nnothing to see here\nghp_0123456789abcdefghij\n",
        encoding="utf-8",
    )

    leaks = scan_artifacts(artifacts, literals=[TOKEN])
    assert [leak.line for leak in leaks] == [1, 3]
    # A report that quotes the leak is a second copy of it.
    assert all(TOKEN not in str(leak) for leak in leaks)


def test_the_scan_enrols_credentials_by_variable_name(tmp_path: Path) -> None:
    """`env=os.environ` is the calling convention; nobody lists their secrets."""
    (tmp_path / "run.log").write_text(f"the value was {TOKEN}\n", encoding="utf-8")

    assert scan_artifacts(tmp_path, env={"GITHUB_TOKEN": TOKEN})
    assert scan_artifacts(tmp_path, env={"HARMLESS": TOKEN}) == []


def test_the_scan_survives_a_binary_file(tmp_path: Path) -> None:
    """A run directory can hold a core dump; stopping at it scans nothing."""
    (tmp_path / "core").write_bytes(b"\x00\xff\xfe" * 64)
    (tmp_path / "run.log").write_text(f"{TOKEN}\n", encoding="utf-8")

    assert [leak.path.name for leak in scan_artifacts(tmp_path, literals=[TOKEN])] == ["run.log"]


def test_short_values_are_not_treated_as_secrets() -> None:
    """`DEBUG=1` must not make every `1` in a log a finding."""
    assert find_secrets("level=1\nverbose=1\n", literals=["1"]) == []


def test_the_scanner_and_the_redactor_agree_about_what_a_secret_is() -> None:
    """One definition, in `containers/manager.py`; this module audits it.

    Anything the redactor would remove must be something the scanner reports,
    or an artifact could pass the scan while carrying what redaction missed -
    and the two would have to be kept in step by hand forever.
    """
    redactor = Redactor([TOKEN])
    for line in (
        f"GITHUB_TOKEN={TOKEN}",
        "https://x-access-token:ghp_0123456789abcdefghij@github.com/o/n.git",
        "github_pat_11ABCDEFG0abcdefghijklmnop",
    ):
        assert redactor(line) != line
        assert find_secrets(line, literals=[TOKEN])


# --------------------------------------------------------------------------
# The deployed copy
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compose_text() -> str:
    return COMPOSE.read_text(encoding="utf-8")


def test_compose_carries_the_generated_allowlist(compose_text: str) -> None:
    """A widened allowlist cannot land in the YAML alone."""
    for line in EgressPolicy().filter_lines():
        assert f"      {line}\n" in compose_text, line


def test_compose_carries_the_generated_allowlist_and_nothing_more(compose_text: str) -> None:
    """The other direction, and the one that was missing (#269).

    The test above asserts every generated line is *in* the YAML, which a
    hand-widened `compose.yaml` satisfies for free - an extra entry pasted into
    the block is an extra destination for every worker, and nothing failed.
    Equality is what "the allowlist stays generated from `security.py`" was
    always supposed to mean.

    Read out of the `configs:` block by shape rather than by parsing YAML: the
    lines are anchored regexes, so they are recognisable on their own and the
    suite needs no YAML dependency to find them.
    """
    carried = [
        line.strip()
        for line in compose_text.splitlines()
        if line.startswith("      ^") and line.rstrip().endswith("$")
    ]

    assert carried == EgressPolicy().filter_lines(), (
        "compose.yaml's allowlist and EgressPolicy.filter_lines() disagree; "
        "regenerate the block rather than editing it"
    )


def test_compose_points_the_orchestrator_at_the_socket_proxy(compose_text: str) -> None:
    assert f"DOCKER_HOST: {DOCKER_HOST_URL}" in compose_text


def test_compose_grants_the_socket_proxy_exactly_the_stated_surface(compose_text: str) -> None:
    for name, value in SOCKET_PROXY_ENV.items():
        assert f"      {name}: {value}\n" in compose_text, name


def test_only_the_socket_proxy_sees_the_socket(compose_text: str) -> None:
    """One mount, read-only, and nowhere near the container that runs model output.

    Comment lines are skipped, and the socket is discussed at length in them:
    the file explains why `:ro` is worth having and why it is not what makes
    this safe.
    """
    mounts = [
        line
        for line in compose_text.splitlines()
        if "/var/run/docker.sock" in line and not line.lstrip().startswith("#")
    ]
    assert mounts == ["      - /var/run/docker.sock:/var/run/docker.sock:ro"]


def test_the_worker_network_has_no_route_off_the_host(compose_text: str) -> None:
    """`internal: true` is what makes ignoring the proxy a failure, not a bypass."""
    assert f"  {WORKER_NETWORK}:\n    name: {WORKER_NETWORK}\n    internal: true\n" in compose_text
    assert "  apiary-control:\n    name: apiary-control\n    internal: true\n" in compose_text


def test_no_service_is_declared_privileged(compose_text: str) -> None:
    assert "privileged:" not in compose_text


# --------------------------------------------------------------------------
# 1b. The boot key
# --------------------------------------------------------------------------


def test_the_boot_key_needs_exactly_what_the_work_key_must_never_have():
    """The two permission sets are disjoint where it matters.

    This is the whole argument for a second credential in one assertion: the
    boot key needs two permissions `FORBIDDEN_PERMISSIONS` refuses a worker, so
    no single token can do both jobs safely.

    The overlap the other way is fine and expected - both keys write labels -
    so the claim is about the dangerous permissions specifically, not about the
    two sets being disjoint.
    """
    dangerous = set(PROVISION_PERMISSIONS) & set(FORBIDDEN_PERMISSIONS)
    assert dangerous == {"administration", "workflows"}
    assert not dangerous & set(REQUIRED_PERMISSIONS)

    # Shared, and harmless: nothing about writing a label lets generated code
    # reach the machinery that judges it.
    assert set(PROVISION_PERMISSIONS) & set(REQUIRED_PERMISSIONS) == {
        "issues", "contents", "metadata"
    }


def test_a_missing_boot_key_names_the_variable_and_the_reason():
    with pytest.raises(CredentialError) as caught:
        assert_provision_token(None)
    message = str(caught.value)
    assert PROVISION_TOKEN_ENV in message
    assert "administration" in message and "workflows" in message


def test_a_boot_key_is_held_to_the_same_shape_rule():
    """An account-wide token here is the worst combination in the module."""
    with pytest.raises(CredentialError):
        assert_provision_token("ghp_" + "a" * 36)
    assert assert_provision_token("github_pat_" + "b" * 40) == "fine-grained"


def test_the_boot_key_is_refused_by_name_in_a_worker_environment():
    with pytest.raises(PolicyError) as caught:
        assert_no_provision_token({PROVISION_TOKEN_ENV: "github_pat_" + "c" * 40})
    assert "workflows" in str(caught.value)


def test_renaming_the_boot_key_does_not_smuggle_it_into_a_worker(monkeypatch):
    """A name-only check is defeated by one assignment.

    The value is what grants the permissions, so the value is what is matched.
    """
    secret = "github_pat_" + "d" * 40
    monkeypatch.setenv(PROVISION_TOKEN_ENV, secret)
    with pytest.raises(PolicyError) as caught:
        assert_no_provision_token({"GITHUB_TOKEN": secret})
    assert "renaming it does not narrow" in str(caught.value)


def test_an_ordinary_worker_environment_passes(monkeypatch):
    monkeypatch.setenv(PROVISION_TOKEN_ENV, "github_pat_" + "e" * 40)
    assert_no_provision_token({"GITHUB_TOKEN": "github_pat_" + "f" * 40})
    assert_no_provision_token({})
    assert_no_provision_token(None)


def test_compose_hands_the_boot_key_to_the_orchestrator_and_no_one_else():
    """The split is only real if the key actually arrives where it is needed.

    It was added to `security.py` and to `doctor` before it was added to
    `compose.yaml`, so the containerized `--new` path failed at the credential
    check having already asked the operator to confirm the repository name.
    The check did its job; the wiring had not been done.
    """
    compose = (Path(__file__).resolve().parents[1] / "compose.yaml").read_text()
    orchestrator, _, rest = compose.partition("docker-socket-proxy:")

    assert f"{PROVISION_TOKEN_ENV}: ${{{PROVISION_TOKEN_ENV}:-}}" in orchestrator
    # And nowhere else: the worker network's containers are created by
    # `ContainerManager`, not by compose, but a future service added here must
    # not be handed it either.
    assert PROVISION_TOKEN_ENV not in rest


def test_compose_passes_the_worker_image_override_through(compose_text: str) -> None:
    """`environment:` is an explicit list, not a passthrough.

    An override absent from it never reaches a containerised orchestrator: the
    process reads its default and the operator reads their `.env`, and the two
    disagree silently for the length of a run. This file already polices that
    drift for the token and the proxy variables; #99's mapping joins them.
    """
    assert f"      {STACK_IMAGES_ENV}: ${{{STACK_IMAGES_ENV}:-}}\n" in compose_text


# --------------------------------------------------------------------------
# A model-provider credential in a worker container (#269)
# --------------------------------------------------------------------------
#
# The decision and its reasoning are in docs/security.md §5b. What is asserted
# here is the enforcement, and the assertion that matters most is the first
# one: a fully local run reaches none of this.


OPT_IN = {"APIARY_WORKER_MODEL_CREDENTIAL": "1"}
SESSION = {
    "AWS_ACCESS_KEY_ID": "ASIAABCDEFGHIJKLMNOP",
    "AWS_SECRET_ACCESS_KEY": "secret",
    "AWS_SESSION_TOKEN": "token",
    "AWS_REGION": "eu-west-1",
}


def test_the_local_path_needs_no_credential_and_never_raises() -> None:
    """The property of the local-first design that nobody chose and everybody
    relies on. ADR 0006 made the provider configurable; it must not have made
    the default path pay for it."""
    from swarm.security import worker_model_credentials

    assert worker_model_credentials("ollama", {}) == {}
    assert worker_model_credentials("ollama", OPT_IN) == {}


def test_a_remote_worker_is_refused_before_a_container_exists() -> None:
    """A worker shipped without a credential fails at its first call instead -
    several minutes and one container later, reading like a broken model
    rather than like a policy."""
    from swarm.security import worker_model_credentials

    with pytest.raises(CredentialError) as caught:
        worker_model_credentials("bedrock", SESSION)

    assert "APIARY_WORKER_MODEL_CREDENTIAL" in str(caught.value)
    assert "docs/security.md" in str(caught.value)


def test_the_opt_in_carries_the_session_variables_and_nothing_else() -> None:
    from swarm.security import worker_model_credentials

    carried = worker_model_credentials("bedrock", {**OPT_IN, **SESSION, "GITHUB_TOKEN": "ghp_x"})

    assert set(carried) == set(SESSION)
    assert "GITHUB_TOKEN" not in carried


def test_a_long_lived_aws_key_is_refused_even_under_the_opt_in() -> None:
    """The enforceable half of "prefer short-lived". `AKIA` is an IAM user key
    and does not expire; `ASIA` is a session credential and does. It is also
    what `aws configure` writes, so it is the one reached for by accident."""
    from swarm.security import worker_model_credentials

    long_lived = dict(SESSION, AWS_ACCESS_KEY_ID="AKIAABCDEFGHIJKLMNOP")

    with pytest.raises(CredentialError) as caught:
        worker_model_credentials("bedrock", {**OPT_IN, **long_lived})

    assert "AKIA" in str(caught.value)
    assert "sso login" in str(caught.value)


def test_aws_credentials_with_no_session_token_are_refused() -> None:
    """Nothing would bound how long the worker's credential stays valid."""
    from swarm.security import worker_model_credentials

    without = {k: v for k, v in SESSION.items() if k != "AWS_SESSION_TOKEN"}

    with pytest.raises(CredentialError):
        worker_model_credentials("bedrock", {**OPT_IN, **without})


def test_an_opt_in_with_nothing_to_carry_is_refused_rather_than_empty() -> None:
    from swarm.security import worker_model_credentials

    with pytest.raises(CredentialError) as caught:
        worker_model_credentials("openai", OPT_IN)

    assert "shipped without one" in str(caught.value)


def test_a_profile_name_is_never_carried_into_a_container() -> None:
    """It is useless inside a container with no `~/.aws`, and passing one would
    produce "profile not found" rather than an honest "no credential"."""
    from swarm.security import MODEL_CREDENTIAL_ENV

    assert "AWS_PROFILE" not in MODEL_CREDENTIAL_ENV["bedrock"]


# --- what the credential can reach ---------------------------------------


def test_a_remote_worker_opens_exactly_one_host() -> None:
    from swarm.security import EgressPolicy as Policy

    widened = Policy().with_model("bedrock", "eu-west-1")

    assert widened.allows("bedrock-runtime.eu-west-1.amazonaws.com")
    assert set(widened.hosts) - set(EGRESS_ALLOWLIST) == {
        "bedrock-runtime.eu-west-1.amazonaws.com"
    }


def test_bedrocks_entry_is_regional_because_the_short_one_is_all_of_aws() -> None:
    """`allows` matches subdomains, so `amazonaws.com` would open S3, STS and
    every other AWS service to a container running generated code. That is not
    a small widening."""
    from swarm.security import EgressPolicy as Policy

    widened = Policy().with_model("bedrock", "eu-west-1")

    assert not widened.allows("s3.amazonaws.com")
    assert not widened.allows("sts.amazonaws.com")
    assert not widened.allows("bedrock-runtime.us-east-1.amazonaws.com")


def test_a_provider_host_is_not_in_the_allowlist_every_installation_gets() -> None:
    """A destination no container will ever dial does not belong in the tuple
    the fully local default carries."""
    from swarm.security import EgressPolicy as Policy

    assert not Policy().allows("bedrock-runtime.eu-west-1.amazonaws.com")
    assert not Policy().allows("api.openai.com")


def test_a_bedrock_worker_with_no_region_opens_nothing() -> None:
    """Better than guessing a region: an entry for the wrong one is a hole with
    no matching use."""
    from swarm.security import EgressPolicy as Policy

    assert Policy().with_model("bedrock").hosts == EGRESS_ALLOWLIST


# --- redaction ------------------------------------------------------------


def test_an_aws_key_id_is_redacted_by_name() -> None:
    """It matched none of the existing alternatives - `_KEY\\b` fails because
    `_ID` follows - so the id reached a log unredacted while its secret half
    was covered."""
    from swarm.containers.manager import Redactor

    redactor = Redactor()
    redactor.add_env({"AWS_ACCESS_KEY_ID": "ASIAABCDEFGHIJKLMNOP"})

    assert "ASIAABCDEFGHIJKLMNOP" not in redactor("id=ASIAABCDEFGHIJKLMNOP")


@pytest.mark.parametrize(
    "secret",
    [
        "AKIAABCDEFGHIJKLMNOP",
        "ASIAABCDEFGHIJKLMNOP",
        "sk-abcdefghijklmnopqrstuvwxyz",
        "sk-proj-abcdefghijklmnopqrstuvwxyz",
    ],
)
def test_a_model_credential_is_redacted_by_shape_too(secret: str) -> None:
    """A worker can print a credential that arrived in a file or was minted
    inside the container, and the literal list would not know it. `AKIA` is
    included although it is refused: a refusal that printed the credential in
    its own error would be the joke version of the control."""
    from swarm.containers.manager import Redactor

    assert secret not in Redactor()(f"leaked {secret} here")
