"""Tests for the preflight.

The ticket's "done when" has two clauses, and they are the two things this file
is about.

**Every check can be provoked to fail.** A diagnostic is the one kind of code
whose happy path proves nothing: a check that looks at nothing passes forever.
So each check appears twice - once against an environment where it must pass,
once against the single change that must make it fail - and the pair is what
shows the check can see at all.

**Every failure names the fix.** Asserted structurally (`Check` refuses to be
constructed failing-without-a-fix) *and* over the real corpus: `test_every_
failure_names_a_command` provokes one failure per check and requires each fix to
contain something executable, because "check your configuration" satisfies a
non-empty string and helps nobody.

Two further claims, both from the hard constraint that doctor is read-only:

**It writes nothing to GitHub.** The whole run goes through a scripted
transport and every request it made is required to be a `GET`. That is the
assertion that keeps `ensure_labels` - a *write*, one import away - out of this
module however tempting the "while we're here" refactor becomes.

**It creates no container and pulls no image.** The same run drives a recording
`Runner`, and every `docker` argv is required to be one of the two read
subcommands.

Hermetic by construction: `FakeInference` answers where the model server would,
`fake_github`'s scripted transport answers where the API would, and
`RecordingRunner` answers where `docker` would. The three gated live checks at
the bottom - marked `ollama` and `docker` per `tests/conftest.py` - run the
real probes against this machine and are deselected by default.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pytest

from fixtures.github import page, response
from fixtures.mcp import ENDPOINT, FakeMcpServer
from fixtures.mcp import client as mcp_client
from swarm.config import Settings
from swarm.containers.manager import (
    DEFAULT_STACK_IMAGES,
    STACK_LABEL,
    WORKER_IMAGE,
    DockerCLI,
)
from swarm.security import PROVISION_TOKEN_ENV
from swarm.doctor import (
    CHECK_CI,
    CHECK_DOCKER_CLI,
    CHECK_DOCKER_DAEMON,
    CHECK_LABELS,
    CHECK_OLLAMA_MODELS,
    CHECK_OLLAMA_REACHABLE,
    CHECK_OLLAMA_SCHEMA,
    CHECK_OLLAMA_TARGET,
    CHECK_BOOT_TOKEN,
    CHECK_REPO,
    CHECK_TIMEOUTS,
    CHECK_TOKEN,
    CHECK_TRACKER_AUTH,
    CHECK_TRACKER_CONFIG,
    CHECK_TRACKER_REACHABLE,
    CHECK_TRACKER_TOOLS,
    CHECK_WORKER_IMAGE,
    FAIL,
    OK,
    SKIP,
    Check,
    Diagnosis,
    Doctor,
    DoctorError,
    HostInference,
    main,
    stack_check,
)
from swarm.github.labels import SWARM_LABELS
from swarm.mcp.contract import ContractError, parse_tracker

REPO = "shahrestani-me/apiary"

#: A fine-grained PAT's prefix, which is the only shape `assert_scoped_token`
#: accepts. The tail is not token-shaped enough to be a real one and long
#: enough to be redacted like one.
BOOT_TOKEN = "github_pat_" + "b" * 40
TOKEN = "github_pat_11ABCDEFG0doctorpreflightfixture"

#: What `config.py` produces on a correctly configured machine.
GOOD_URL = "http://localhost:11434"

#: The tracker credential the built-in `linear` profile names, and a contract
#: written the way a customer's is: a profile, plus the one constant only that
#: organization knows.
LINEAR_TOKEN = "lin_api_" + "c" * 32
TRACKER_BLOCK = "mcp: linear\nargs: { teamId: TEAM-1 }\n"
TRACKER = parse_tracker(TRACKER_BLOCK, source=".swarm/tracker.yaml")

#: What it produces on a machine that exported the *server's* bind address,
#: which SETUP.md tells the operator to do so containers can reach Ollama.
BIND_URL = "0.0.0.0:11434"

ORCHESTRATOR_MODEL = "gemma4:31b"
WORKER_MODEL = "gemma4:26b"


def settings(**overrides: Any) -> Settings:
    """`Settings` that ignore this machine's environment.

    `Settings` reads `os.environ` through its field defaults, and the machine
    running these tests may well have `OLLAMA_HOST` exported - it is the very
    misconfiguration `ollama.target` is about. Every field the doctor reads is
    therefore pinned here.
    """
    base = dict(
        ollama_base_url=GOOD_URL,
        orchestrator_model=ORCHESTRATOR_MODEL,
        worker_model=WORKER_MODEL,
        # `config.timeouts` reads these two, and a shell that exported either
        # would decide this file's healthy case. Pinned for the same reason as
        # `OLLAMA_HOST` above.
        worker_timeout_s=1200,
        verify_timeout_s=300,
    )
    return Settings(**{**base, **overrides})


# --------------------------------------------------------------------------
# The doubles
# --------------------------------------------------------------------------


@dataclass
class FakeInference:
    """A model server that is up, has both models, and obeys the schema.

    Each of those three is a field, so a test breaks exactly one of them and
    leaves the rest of the environment healthy - which is what makes the
    resulting failure attributable to the check under test.
    """

    version_answer: str | Exception = "0.32.9"
    models: list[str] = field(default_factory=lambda: [ORCHESTRATOR_MODEL, WORKER_MODEL])
    schema_failures: tuple[str, ...] = ()
    probed: list[str] = field(default_factory=list)

    def version(self) -> str:
        if isinstance(self.version_answer, Exception):
            raise self.version_answer
        return self.version_answer

    def installed(self) -> list[str]:
        return list(self.models)

    def schema_probe(self, role: str) -> str:
        self.probed.append(role)
        if role in self.schema_failures:
            raise DoctorError("returned str, not the requested schema")
        return "Ping parsed"


@dataclass
class RecordingRunner:
    """A `Runner` that answers `docker` and records every argv.

    Answers by subcommand rather than by script: the doctor's two docker calls
    are order-dependent on each other and a positional script would have to be
    rewritten every time a check moves.
    """

    server_version: str = "29.2.1"
    #: image -> the `org.apiary.stack` label it carries. An image mapped to ""
    #: is present but unlabelled, which #100 treats as indistinguishable from a
    #: stale build of something else tagged the same way.
    images: Mapping[str, str] = field(
        default_factory=lambda: {image: stack for stack, image in DEFAULT_STACK_IMAGES.items()}
    )
    denied: bool = False
    calls: list[list[str]] = field(default_factory=list)

    def __call__(
        self, argv: Sequence[str], *, timeout_s: float | None, merge: bool
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        args = list(argv)[1:]
        if args[:1] == ["version"]:
            return self._done(0, self.server_version)
        if args[:2] == ["image", "inspect"]:
            if self.denied:
                return self._done(1, "", "Error response from daemon: 403 Forbidden")
            if args[2] in self.images:
                # The real `--format` is `{{.Id}}|{{json .Config}}`, so the
                # double answers in that shape rather than in one only this
                # file would accept - including the "no Labels key at all"
                # case, which is what an image built before the label existed
                # looks like and what broke the first implementation.
                labels = (
                    {STACK_LABEL: self.images[args[2]]} if self.images[args[2]] else {}
                )
                config = json.dumps({"Entrypoint": ["apiary-worker"], **({"Labels": labels} if labels else {})})
                return self._done(0, f"sha256:{'d0c70' * 12}|{config}")
            return self._done(1, "", f"Error: No such image: {args[2]}")
        raise AssertionError(f"unexpected docker call: {argv}")

    @staticmethod
    def _done(code: int, stdout: str, stderr: str = "") -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess([], code, stdout + "\n", stderr)

    @property
    def subcommands(self) -> list[str]:
        return [call[1] for call in self.calls]


@dataclass
class DeadRunner(RecordingRunner):
    """A daemon that is not there. What a stopped Docker Desktop looks like."""

    def __call__(
        self, argv: Sequence[str], *, timeout_s: float | None, merge: bool
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        return self._done(1, "", "Cannot connect to the Docker daemon at unix:///docker.sock")


@dataclass
class LeakingRunner(RecordingRunner):
    """A daemon that quotes the credential back. Git and docker both do this."""

    def __call__(
        self, argv: Sequence[str], *, timeout_s: float | None, merge: bool
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        return self._done(1, "", f"denied: bad credential {TOKEN}")


def label_page(*, missing: Sequence[str] = ()) -> Any:
    """The repo's labels, minus whichever `swarm:*` names a test removed."""
    absent = {name.casefold() for name in missing}
    names = [spec.name for spec in SWARM_LABELS if spec.name.casefold() not in absent]
    return page([{"name": name} for name in ["bug", "area/ops", *names]])


def check_runs(*names: str) -> Any:
    """A `GET /commits/{ref}/check-runs` body, in its envelope."""
    return response(200, {"total_count": len(names), "check_runs": [{"name": n} for n in names]})


def issue_page(count: int = 3) -> Any:
    return page([{"number": n, "title": f"issue {n}"} for n in range(1, count + 1)])


def github_script(
    *,
    issues: Any = None,
    labels: Any = None,
    ci: Any = None,
) -> list[Any]:
    """The three reads a healthy run makes, in the order `run` makes them."""
    return [
        issues if issues is not None else issue_page(),
        labels if labels is not None else label_page(),
        ci if ci is not None else check_runs("test"),
    ]


@pytest.fixture()
def doctor(fake_github) -> Any:
    """A factory for a `Doctor` whose every collaborator is healthy.

    Returns `(doctor, transport, runner, inference)` so a test can assert on
    what was asked as well as on what was answered. Keyword arguments override
    one collaborator at a time; `script` replaces the GitHub responses.
    """

    def build(
        *,
        script: Sequence[Any] | None = None,
        inference: FakeInference | None = None,
        runner: RecordingRunner | None = None,
        server: FakeMcpServer | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        gh, transport, _ = fake_github(*(script if script is not None else github_script()))
        probe = inference or FakeInference()
        docker_runner = runner or RecordingRunner()
        subject = Doctor(
            repo=kwargs.pop("repo", REPO),
            settings=kwargs.pop("settings", None) or settings(),
            github=gh,
            docker=DockerCLI(runner=docker_runner),
            inference=probe,
            env=(
                {
                    "GITHUB_TOKEN": TOKEN,
                    PROVISION_TOKEN_ENV: BOOT_TOKEN,
                    "APIARY_LINEAR_TOKEN": LINEAR_TOKEN,
                }
                if env is None
                else env
            ),
            # A contract and a server by default, so the healthy case exercises
            # the tracker checks rather than skipping past them. `tracker=None`
            # is how a test says "this installation configures no tracker".
            tracker=kwargs.pop("tracker", TRACKER),
            mcp=kwargs.pop("mcp", mcp_client(server or FakeMcpServer())),
            which=kwargs.pop("which", lambda name: f"/usr/local/bin/{name}"),
            in_container=kwargs.pop("in_container", False),
            **kwargs,
        )
        return subject, transport, docker_runner, probe

    return build


# --------------------------------------------------------------------------
# The verdict type
# --------------------------------------------------------------------------


def test_a_failing_check_cannot_be_built_without_a_fix():
    """The ticket's second clause, made structural rather than remembered."""
    with pytest.raises(ValueError, match="must name the fix"):
        Check("some.check", FAIL, "it is broken")

    # The other two statuses have nothing to remedy.
    assert Check.passed("some.check", "fine").fix == ""
    assert Check.skipped("some.check", "not attempted").fix == ""


def test_a_multi_line_fix_stays_inside_its_column():
    """The report is read as a table.

    A fix whose useful shape is a sentence and then the command on its own line
    - which is how every refusal `mcp/contract.py` hands over is written -
    would otherwise print its second line flush left, where it reads as a new
    check rather than as part of this one.
    """
    lines = Check.failed("some.check", "broken", fix="do this:\n    then-run --this").lines()

    assert len(lines) == 3
    assert lines[1].lstrip().startswith("fix: do this:")
    assert lines[2].startswith(" " * 20) and "then-run --this" in lines[2]


def test_unknown_status_is_refused():
    with pytest.raises(ValueError, match="unknown status"):
        Check("some.check", "probably", "who knows")


def test_skips_do_not_fail_a_diagnosis():
    """A skip is "this machine could not answer", and its cause already failed.

    Counting it again would report two problems where the operator has one.
    """
    diagnosis = Diagnosis((
        Check.passed("a", "fine"),
        Check.skipped("b", "not attempted: a did not pass"),
    ))
    assert diagnosis.ok
    assert diagnosis.summary() == "2 checks: 1 ok, 1 not attempted"

    broken = Diagnosis((Check.failed("a", "broken", fix="do the thing"),))
    assert not broken.ok
    assert "do the thing" in broken.report()


# --------------------------------------------------------------------------
# The healthy machine
# --------------------------------------------------------------------------


def test_a_healthy_environment_passes_every_check(doctor):
    subject, _, _, probe = doctor()
    diagnosis = subject.run()

    assert [check.name for check in diagnosis.checks] == [
        CHECK_OLLAMA_TARGET,
        CHECK_OLLAMA_REACHABLE,
        CHECK_OLLAMA_MODELS,
        CHECK_OLLAMA_SCHEMA,
        CHECK_TOKEN,
        CHECK_BOOT_TOKEN,
        CHECK_REPO,
        CHECK_LABELS,
        CHECK_CI,
        CHECK_TIMEOUTS,
        CHECK_DOCKER_CLI,
        CHECK_DOCKER_DAEMON,
        # One per stack: #99 chooses the image per task, so "the worker image
        # is present" stopped being a single fact about a host.
        stack_check("node"),
        stack_check("python"),
        stack_check("react"),
        # The capability contract (#150). Four rather than one because "there
        # is no block", "nothing answered", "the credential was refused" and
        # "that tool does not exist" have four unrelated remedies.
        CHECK_TRACKER_CONFIG,
        CHECK_TRACKER_REACHABLE,
        CHECK_TRACKER_AUTH,
        CHECK_TRACKER_TOOLS,
    ]
    assert diagnosis.ok, diagnosis.report()
    assert not diagnosis.skipped
    # Both roles were probed, not just the first one that happened to work.
    assert probe.probed == ["orchestrator", "worker"]


def test_the_run_reads_and_never_writes(doctor):
    """The hard constraint, asserted over the whole run rather than per call.

    `github/labels.py` can create what `github.labels` reports missing, and it
    is one import away. This is the test that keeps it out.
    """
    subject, transport, runner, _ = doctor()
    subject.run()

    assert {method for method, _ in transport.calls} == {"GET"}
    # `version` once, then one `image inspect` per stack. Still only the two
    # read subcommands - which is the assertion, not the count.
    assert set(runner.subcommands) == {"version", "image"}
    assert runner.subcommands[0] == "version"
    assert all("create" not in call and "pull" not in call for call in runner.calls)


def test_the_report_is_readable(doctor):
    subject, _, _, _ = doctor()
    report = subject.run().report()

    assert "all 19 preconditions met" in report
    for name in (CHECK_OLLAMA_SCHEMA, CHECK_TOKEN, stack_check("python")):
        assert name in report


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------


def test_a_bind_address_is_caught_before_the_connection_error(doctor):
    """The overloaded `OLLAMA_HOST`, which is two variables wearing one name.

    The reachability check would also fail here, with a connection error that
    reads exactly like a stopped server - so the target check runs first and
    the fix it names is the one that actually works.
    """
    subject, _, _, _ = doctor(settings=settings(ollama_base_url=BIND_URL))
    diagnosis = subject.run()
    target = diagnosis.by_name(CHECK_OLLAMA_TARGET)

    assert target.status == FAIL
    assert "bind address" in target.detail
    assert "APIARY_OLLAMA_HOST" in target.fix
    assert "export OLLAMA_HOST=http://localhost:11434" in target.fix

    # And the connection error that would have followed is not reported at
    # all: `ollama serve` is what it advises, and it would fix nothing.
    assert diagnosis.by_name(CHECK_OLLAMA_REACHABLE).status == SKIP


@pytest.mark.parametrize("url", ["http://0.0.0.0:11434", "http://[::]:11434", "http://:11434"])
def test_every_wildcard_spelling_is_a_bind_address(doctor, url):
    subject, _, _, _ = doctor(settings=settings(ollama_base_url=url))
    assert subject.check_ollama_target().status == FAIL


def test_loopback_is_only_wrong_inside_a_container(doctor):
    """`localhost` from a container is the container, and Ollama is on the host."""
    inside, _, _, _ = doctor(in_container=True)
    verdict = inside.check_ollama_target()
    assert verdict.status == FAIL
    assert "host.docker.internal" in verdict.fix

    outside, _, _, _ = doctor(in_container=False)
    assert outside.check_ollama_target().ok


def test_an_unreachable_server_names_the_command_that_starts_it(doctor):
    dead = FakeInference(version_answer=DoctorError("Connection refused"))
    subject, _, _, _ = doctor(inference=dead)
    diagnosis = subject.run()

    reachable = diagnosis.by_name(CHECK_OLLAMA_REACHABLE)
    assert reachable.status == FAIL
    assert "ollama serve" in reachable.fix

    # And the two checks that need a server are not attempted, rather than
    # reporting an absent model. Each names the verdict directly above it, so
    # the chain leads back to the one thing that is actually wrong.
    assert diagnosis.by_name(CHECK_OLLAMA_MODELS).status == SKIP
    assert CHECK_OLLAMA_REACHABLE in diagnosis.by_name(CHECK_OLLAMA_MODELS).detail
    assert diagnosis.by_name(CHECK_OLLAMA_SCHEMA).status == SKIP
    assert CHECK_OLLAMA_MODELS in diagnosis.by_name(CHECK_OLLAMA_SCHEMA).detail


def test_a_missing_model_is_named_with_its_pull_command(doctor):
    """The failure that otherwise presents as "the planner is broken"."""
    subject, _, _, _ = doctor(inference=FakeInference(models=[WORKER_MODEL]))
    verdict = subject.run().by_name(CHECK_OLLAMA_MODELS)

    assert verdict.status == FAIL
    assert ORCHESTRATOR_MODEL in verdict.detail
    assert f"ollama pull {ORCHESTRATOR_MODEL}" in verdict.fix
    assert WORKER_MODEL not in verdict.fix


@pytest.mark.parametrize(
    "installed, wanted, expected",
    [
        (["gemma4:31b"], "gemma4:31b", True),
        (["llama3:latest"], "llama3", True),
        (["llama3"], "llama3", True),
        (["gemma4:26b"], "gemma4:31b", False),
        (["gemma4:31b-instruct"], "gemma4:31b", False),
    ],
)
def test_the_implicit_latest_tag_is_not_a_missing_model(doctor, installed, wanted, expected):
    subject, _, _, _ = doctor(
        inference=FakeInference(models=installed),
        settings=settings(orchestrator_model=wanted, worker_model=wanted),
    )
    assert subject.check_models().ok is expected


def test_a_model_that_ignores_the_schema_is_a_failure_not_a_planning_bug(doctor):
    subject, _, _, _ = doctor(inference=FakeInference(schema_failures=("worker",)))
    verdict = subject.run().by_name(CHECK_OLLAMA_SCHEMA)

    assert verdict.status == FAIL
    assert WORKER_MODEL in verdict.detail
    assert "SWARM_WORKER_MODEL" in verdict.fix
    assert "ollama show" in verdict.fix


def test_the_schema_probe_can_be_turned_off(doctor):
    """It is the only check that costs a model load, and two on a swapping host."""
    subject, _, _, probe = doctor(probe_schema=False)
    diagnosis = subject.run()

    assert diagnosis.by_name(CHECK_OLLAMA_SCHEMA).status == SKIP
    assert probe.probed == []
    assert diagnosis.ok


# --------------------------------------------------------------------------
# GitHub
# --------------------------------------------------------------------------


def test_an_account_wide_token_is_refused_before_a_container_holds_it(doctor):
    """#28's `assert_scoped_token`, in its first production caller.

    `gh auth token` prints a `gho_`, which works perfectly against the target
    repo and reaches every other repository the account can - so no later check
    could ever notice.
    """
    subject, _, _, _ = doctor(env={"GITHUB_TOKEN": "gho_" + "a" * 36})
    verdict = subject.check_token()

    assert verdict.status == FAIL
    assert "fine-grained" in verdict.fix
    assert "contents:write" in verdict.fix


def test_a_missing_token_stops_the_github_checks_rather_than_the_run(doctor):
    subject, _, _, _ = doctor(env={})
    diagnosis = subject.run()

    assert diagnosis.by_name(CHECK_TOKEN).status == FAIL
    for name in (CHECK_REPO, CHECK_LABELS, CHECK_CI):
        assert diagnosis.by_name(name).status == SKIP
    # The rest of the environment is still reported: an operator who has two
    # problems should learn about both on the first run.
    assert diagnosis.by_name(CHECK_DOCKER_DAEMON).ok
    assert diagnosis.by_name(CHECK_OLLAMA_MODELS).ok


@pytest.mark.parametrize(
    "status, expected",
    [
        (401, "expired or mistyped"),
        (403, "not permitted here"),
        (404, "Repository access"),
    ],
)
def test_each_refusal_gets_the_remedy_that_matches_it(doctor, status, expected):
    """A 404 from a fine-grained PAT means "not granted this repo", not "no repo".

    Three status codes, three completely different fixes, and guessing wrong
    costs the afternoon this module exists to save.
    """
    subject, _, _, _ = doctor(script=github_script(issues=response(status, {"message": "no"})))
    verdict = subject.run().by_name(CHECK_REPO)

    assert verdict.status == FAIL
    assert expected in verdict.fix


def test_a_malformed_repo_is_refused_without_a_request(doctor):
    subject, transport, _, _ = doctor(script=[], repo="not-a-repo")
    verdict = subject.check_repo_access()

    assert verdict.status == FAIL
    assert "owner/name" in verdict.fix
    assert transport.sent == []


def test_missing_labels_are_reported_never_created(doctor):
    subject, transport, _, _ = doctor(
        script=github_script(labels=label_page(missing=["swarm:review", "swarm:done"]))
    )
    verdict = subject.run().by_name(CHECK_LABELS)

    assert verdict.status == FAIL
    assert "swarm:review" in verdict.detail and "swarm:done" in verdict.detail
    assert "python -m swarm.github.labels" in verdict.fix
    assert {method for method, _ in transport.calls} == {"GET"}


def test_a_case_variant_label_counts_as_present(doctor):
    """GitHub's label uniqueness is case-insensitive; so is this comparison.

    A repo carrying `Swarm:Ready` answers a `POST` of `swarm:ready` with a 422,
    so reporting it missing would send the operator to a command that fails.
    """
    labels = page([{"name": spec.name.title()} for spec in SWARM_LABELS])
    subject, _, _, _ = doctor(script=github_script(labels=labels))
    assert subject.run().by_name(CHECK_LABELS).ok


def test_a_repo_with_no_ci_is_refused(doctor):
    """Nothing gates a PR, so nothing ever leaves `swarm:review`."""
    subject, _, _, _ = doctor(script=github_script(ci=check_runs()))
    verdict = subject.run().by_name(CHECK_CI)

    assert verdict.status == FAIL
    assert "swarm:review" in verdict.detail
    assert ".github/workflows" in verdict.fix
    assert "--ci-ref" in verdict.fix


def test_an_absent_ci_ref_is_not_an_absent_workflow(doctor):
    """A repo whose default branch is not `main` must not read as a repo with no CI."""
    subject, _, _, _ = doctor(script=github_script(ci=response(404, {"message": "Not Found"})))
    verdict = subject.run().by_name(CHECK_CI)

    assert verdict.status == FAIL
    assert "--ci-ref" in verdict.fix
    assert ".github/workflows" not in verdict.fix


def test_the_ci_ref_is_the_one_that_was_asked_for(doctor):
    subject, transport, _, _ = doctor(ci_ref="trunk")
    subject.run()

    assert any("/commits/trunk/check-runs" in path for _, path in transport.calls)


# --------------------------------------------------------------------------
# Docker
# --------------------------------------------------------------------------


def test_a_missing_docker_binary_is_its_own_failure(doctor):
    """The one a real run tripped over: DOCKER_HOST honoured by nobody.

    The orchestrator image installs `git` and no docker CLI, and
    `containers/manager.py` reaches the daemon by shelling out to that binary.
    Reported as a daemon failure this sends the reader to Docker Desktop, which
    is already running.
    """
    subject, _, runner, _ = doctor(
        which=lambda name: None,
        env={"GITHUB_TOKEN": TOKEN, "DOCKER_HOST": "tcp://docker-socket-proxy:2375"},
    )
    diagnosis = subject.run()
    verdict = diagnosis.by_name(CHECK_DOCKER_CLI)

    assert verdict.status == FAIL
    assert "tcp://docker-socket-proxy:2375" in verdict.detail
    assert "docker:cli" in verdict.fix
    # And nothing tried to shell out to a binary that is not there.
    assert diagnosis.by_name(CHECK_DOCKER_DAEMON).status == SKIP
    assert runner.calls == []


def test_an_unreachable_daemon_names_docker_host(doctor):
    subject, _, _, _ = doctor(runner=DeadRunner())
    diagnosis = subject.run()
    verdict = diagnosis.by_name(CHECK_DOCKER_DAEMON)

    assert verdict.status == FAIL
    assert "DOCKER_HOST" in verdict.fix
    assert diagnosis.by_name(stack_check("python")).status == SKIP


def test_a_missing_worker_image_names_its_build_command(doctor):
    subject, _, _, _ = doctor(runner=RecordingRunner(images={}))
    verdict = subject.run().by_name(stack_check("python"))

    assert verdict.status == FAIL
    assert f"docker build -f Dockerfile.worker -t {WORKER_IMAGE} ." in verdict.fix
    assert "IMAGES=0" in verdict.fix
    # And BUILD=0, because "pull it then" is the obvious next thought.
    assert "BUILD=0" in verdict.fix


def test_each_stack_is_checked_separately(doctor):
    """A host with the Python image and not the Node one is the normal state of
    a machine that has only ever run Python backlogs, and it must read as
    exactly that rather than as "the worker image is missing"."""
    subject, _, _, _ = doctor(runner=RecordingRunner(images={WORKER_IMAGE: "python"}))
    diagnosis = subject.run()

    assert diagnosis.by_name(stack_check("python")).status == OK
    assert diagnosis.by_name(stack_check("node")).status == FAIL
    assert "Dockerfile.worker.node" in diagnosis.by_name(stack_check("node")).fix


def test_only_the_stacks_a_run_needs_are_checked(doctor):
    """A Python-only backlog must not be told to build a Node image it will
    never spawn."""
    subject, _, _, _ = doctor(
        runner=RecordingRunner(images={WORKER_IMAGE: "python"}), stacks=("python",)
    )
    diagnosis = subject.run()

    assert [check.name for check in diagnosis.checks if check.name.startswith("docker.image")] == [
        stack_check("python")
    ]
    assert diagnosis.ok


def test_an_image_with_no_stack_label_is_not_trusted(doctor):
    """An image under the right tag with no label is indistinguishable from a
    stale build of a different Dockerfile that happened to be tagged this way,
    and the failure that produces lands inside a worker."""
    subject, _, _, _ = doctor(
        runner=RecordingRunner(images={image: "" for image in DEFAULT_STACK_IMAGES.values()})
    )
    verdict = subject.run().by_name(stack_check("node"))

    assert verdict.status == FAIL
    assert "stale build" in verdict.detail
    assert "rebuild" in verdict.fix


def test_the_image_check_reports_the_stack_the_image_says_it_carries(doctor):
    subject, _, _, _ = doctor()

    assert "node" in subject.run().by_name(stack_check("node")).detail


def test_the_socket_proxy_denying_images_is_unanswered_not_missing(doctor):
    """`SOCKET_PROXY_ENV` sets `IMAGES=0`, so the 403 is the design working.

    Reported as a missing image it would send the operator to rebuild an image
    that is already there.
    """
    subject, _, _, _ = doctor(runner=RecordingRunner(denied=True))
    diagnosis = subject.run()

    assert diagnosis.by_name(stack_check("python")).status == SKIP
    assert "IMAGES=0" in diagnosis.by_name(stack_check("python")).detail
    assert diagnosis.ok


def test_a_docker_failure_cannot_print_the_token():
    """The report is the thing an operator pastes into an issue.

    Built through the production wiring rather than through the fixture: the
    redactor is attached in `Doctor.__post_init__` from the environment it was
    handed, and a hand-made `DockerCLI` here would prove only that the double
    behaved.
    """
    subject = Doctor(
        repo=REPO,
        settings=settings(),
        inference=FakeInference(),
        env={"GITHUB_TOKEN": TOKEN},
        which=lambda name: "/usr/local/bin/docker",
        in_container=False,
    )
    assert subject.docker is not None
    subject.docker = DockerCLI(redact=subject.docker.redact, runner=LeakingRunner())

    verdict = subject.check_docker_daemon()
    assert verdict.status == FAIL
    assert TOKEN not in verdict.detail
    assert "***" in verdict.detail


# --------------------------------------------------------------------------
# The tracker
#
# Four checks over one handshake, and the cuts between them are the point:
# "there is no block", "nothing answered", "the credential was refused" and
# "that tool does not exist" have four different remedies, and an operator told
# the wrong one of the four goes looking in the wrong place. The read-only
# constraint gets stricter here than anywhere else in this module - a
# `tools/call` against a tracker is a comment somebody receives.
# --------------------------------------------------------------------------


def test_a_healthy_tracker_passes_all_four_checks(doctor):
    server = FakeMcpServer()
    subject, _, _, _ = doctor(server=server)
    diagnosis = subject.run()

    for name in (
        CHECK_TRACKER_CONFIG,
        CHECK_TRACKER_REACHABLE,
        CHECK_TRACKER_AUTH,
        CHECK_TRACKER_TOOLS,
    ):
        assert diagnosis.by_name(name).status == OK, diagnosis.by_name(name)
    assert "list_issues" in diagnosis.by_name(CHECK_TRACKER_TOOLS).detail


def test_the_tracker_probe_calls_no_tool(doctor):
    """The read-only rule, at the point where breaking it is most expensive.

    `tools/call` on somebody's tracker is a comment they receive or a ticket
    they have to triage, and unlike the GitHub half of this module the write
    would land in a system apiary does not own.
    """
    server = FakeMcpServer()
    subject, _, _, _ = doctor(server=server)
    subject.run()

    assert server.called_tools == []
    assert set(server.methods) <= {
        "initialize",
        "notifications/initialized",
        "tools/list",
        "session/delete",
    }, server.methods


def test_the_probe_is_closed_when_the_run_finishes(doctor):
    """A stdio contract's client is a subprocess, and doctor is a diagnostic.

    One left running per invocation would be a leak in the tool an operator
    reaches for when they already suspect their machine.
    """
    server = FakeMcpServer()
    subject, _, _, _ = doctor(server=server)
    subject.run()

    assert server.closed


def test_a_second_run_reuses_the_injected_probe(doctor):
    """Closing the session must not discard the client.

    A caller who handed over a probe means it for the object's lifetime, and a
    second `run()` that quietly built a live client instead would connect to
    somebody's tracker from a test suite.
    """
    server = FakeMcpServer()
    subject, _, _, _ = doctor(server=server)
    # The tracker half twice rather than the whole run twice: the GitHub double
    # is a script, and exhausting it would be a failure about the wrong thing.
    assert all(check.ok for check in subject.tracker_checks())
    assert all(check.ok for check in subject.tracker_checks())

    assert server.methods.count("initialize") == 2
    assert server.methods.count("tools/list") == 2
    assert server.called_tools == []


def test_one_handshake_serves_every_tracker_check(doctor):
    """Three checks read one `initialize`.

    Connecting per check would be three chances to trip somebody's rate limit
    while reporting that nothing is wrong.
    """
    server = FakeMcpServer()
    subject, _, _, _ = doctor(server=server)
    subject.run()

    assert server.methods.count("initialize") == 1


def test_no_tracker_configured_is_one_skip_and_not_a_failure(doctor):
    """apiary runs on the label control plane until #152.

    An installation with no contract is a normal one. Three further lines
    reading "not attempted" would describe a machine with nothing wrong with
    it, and a preflight that reports non-problems is one people stop reading.
    """
    subject, _, _, _ = doctor(tracker=None, mcp=None)
    diagnosis = subject.run()

    assert diagnosis.ok
    assert diagnosis.by_name(CHECK_TRACKER_CONFIG).status == SKIP
    assert "no tracker configured" in diagnosis.by_name(CHECK_TRACKER_CONFIG).detail
    for name in (CHECK_TRACKER_REACHABLE, CHECK_TRACKER_AUTH, CHECK_TRACKER_TOOLS):
        with pytest.raises(KeyError):
            diagnosis.by_name(name)


def test_a_malformed_block_is_reported_as_a_verdict_not_a_traceback(doctor):
    """`from_env` catches `ContractError`, and this is why.

    An unparseable block is the single most likely thing to be wrong on the run
    where somebody types `swarm doctor`, and dying of it would report nothing
    else at all.
    """
    try:
        parse_tracker("mcp: github\ncomments: { tool: t }", source=".swarm/tracker.yaml")
    except ContractError as exc:
        message = str(exc)

    subject, _, _, _ = doctor(tracker=None, mcp=None, tracker_error=message)
    verdict = subject.run().by_name(CHECK_TRACKER_CONFIG)

    assert verdict.status == FAIL
    assert "comments" in verdict.detail
    assert "python -m swarm.mcp.contract" in verdict.fix


def test_an_unreachable_server_is_not_reported_as_an_unauthorized_one(doctor):
    server = FakeMcpServer(unreachable=OSError("connection refused"))
    subject, _, _, _ = doctor(server=server)
    diagnosis = subject.run()

    reachable = diagnosis.by_name(CHECK_TRACKER_REACHABLE)
    assert reachable.status == FAIL
    assert ENDPOINT in reachable.detail
    assert "curl -i" in reachable.fix
    # And the checks that cannot be answered say which one to fix first.
    assert diagnosis.by_name(CHECK_TRACKER_AUTH).status == SKIP
    assert diagnosis.by_name(CHECK_TRACKER_TOOLS).status == SKIP


def test_a_refused_credential_still_proves_the_server_is_there(doctor):
    """The cut that makes four checks worth having.

    A 401 is an answer. Reporting it as "unreachable" sends the operator to
    their network, and reporting an unreachable host as "unauthorized" sends
    them to mint a token they already have.
    """
    server = FakeMcpServer(unauthorized=True)
    subject, _, _, _ = doctor(server=server)
    diagnosis = subject.run()

    assert diagnosis.by_name(CHECK_TRACKER_REACHABLE).status == OK
    auth = diagnosis.by_name(CHECK_TRACKER_AUTH)
    assert auth.status == FAIL
    assert "APIARY_LINEAR_TOKEN" in auth.detail
    # The per-server minting command, which is the whole content of a useful
    # 401 and which differs per tracker (#143).
    assert "linear.app/settings/api" in auth.fix
    assert "not retried" in auth.fix


def test_an_absent_credential_names_the_variable_before_the_network_is_blamed(doctor):
    subject, _, _, _ = doctor(env={"GITHUB_TOKEN": TOKEN})
    verdict = subject.run().by_name(CHECK_TRACKER_AUTH)

    assert verdict.status == FAIL
    assert "export APIARY_LINEAR_TOKEN=" in verdict.fix


def test_a_tool_the_server_does_not_have_is_caught_here(doctor):
    """The failure #150 exists to move.

    A renamed or mistyped tool is not discovered until the first cycle that
    needs the capability - which for `comment` is the moment a pull request has
    been opened and nobody is told about it.
    """
    server = FakeMcpServer(tools=("list_issues", "create_issue"))
    subject, _, _, _ = doctor(server=server)
    verdict = subject.run().by_name(CHECK_TRACKER_TOOLS)

    assert verdict.status == FAIL
    assert "create_comment" in verdict.detail
    # Which capability named it, and what the server does offer instead.
    assert "comment" in verdict.detail
    assert "list_issues" in verdict.fix


def test_a_server_with_no_tools_capability_says_so_separately(doctor):
    """Pointed at a vendor's API root rather than at its MCP endpoint.

    Reported as a missing tool it would read as a contract to edit, when what
    is wrong is the URL.
    """
    server = FakeMcpServer(capabilities={})
    subject, _, _, _ = doctor(server=server)
    verdict = subject.run().by_name(CHECK_TRACKER_TOOLS)

    assert verdict.status == FAIL
    assert "no `tools` capability" in verdict.detail


def test_a_local_server_is_told_to_check_the_binary_not_the_network(doctor):
    """Two completely different problems wearing the same exception.

    The GitHub profile's server is a separate download that nothing in this
    repository installs, so "not on PATH" is the likelier of the two and
    `curl` is advice that cannot help.
    """
    github = parse_tracker("mcp: github\nargs: { owner: o, repo: r }", source="t.yaml")
    server = FakeMcpServer(unreachable=OSError("no such file or directory"))
    subject, _, _, _ = doctor(tracker=github, mcp=mcp_client(server))
    verdict = subject.run().by_name(CHECK_TRACKER_REACHABLE)

    assert verdict.status == FAIL
    assert "which github-mcp-server" in verdict.fix


# --------------------------------------------------------------------------
# The two clauses of "done when"
# --------------------------------------------------------------------------


#: One `ContractError`, rendered the way `Doctor.from_env` hands it over.
_BAD_BLOCK = "tracker.yaml: tracker.create.tool is missing.\n    create: { tool: issue_write }"


def provoked_failures(doctor) -> list[Check]:
    """One provoked failure per check, using the doubles above.

    Each entry breaks exactly one thing about an otherwise healthy machine,
    which is what makes it a control for that check rather than a general
    catastrophe that would fail everything.
    """
    cases: list[tuple[str, dict[str, Any]]] = [
        (CHECK_OLLAMA_TARGET, {"settings": settings(ollama_base_url=BIND_URL)}),
        (CHECK_OLLAMA_REACHABLE, {"inference": FakeInference(DoctorError("refused"))}),
        (CHECK_OLLAMA_MODELS, {"inference": FakeInference(models=[])}),
        (CHECK_OLLAMA_SCHEMA, {"inference": FakeInference(schema_failures=("orchestrator",))}),
        (CHECK_TOKEN, {"env": {}}),
        (CHECK_REPO, {"script": github_script(issues=response(404, {"message": "Not Found"}))}),
        (CHECK_LABELS, {"script": github_script(labels=label_page(missing=["swarm:ready"]))}),
        (CHECK_CI, {"script": github_script(ci=check_runs())}),
        (CHECK_TIMEOUTS, {"settings": settings(verify_timeout_s=1800)}),
        (CHECK_DOCKER_CLI, {"which": lambda name: None}),
        (CHECK_DOCKER_DAEMON, {"runner": DeadRunner()}),
        (stack_check("python"), {"runner": RecordingRunner(images={})}),
        (CHECK_TRACKER_CONFIG, {"tracker": None, "mcp": None, "tracker_error": _BAD_BLOCK}),
        (CHECK_TRACKER_REACHABLE, {"server": FakeMcpServer(unreachable=OSError("refused"))}),
        (CHECK_TRACKER_AUTH, {"server": FakeMcpServer(unauthorized=True)}),
        (CHECK_TRACKER_TOOLS, {"server": FakeMcpServer(tools=("list_issues",))}),
    ]

    failures: list[Check] = []
    for name, kwargs in cases:
        subject, _, _, _ = doctor(**kwargs)
        verdict = subject.run().by_name(name)
        assert verdict.status == FAIL, f"{name} did not fail when it should have: {verdict}"
        failures.append(verdict)
    return failures


def test_every_check_can_be_provoked_to_fail(doctor):
    """The first clause of the ticket's "done when", for every check."""
    names = [check.name for check in provoked_failures(doctor)]
    assert len(names) == len(set(names)) == 16


def test_every_failure_names_a_command(doctor):
    """The second clause, and the reason this module exists.

    A fix that reads "check your configuration" satisfies "non-empty" and
    helps nobody, so the assertion is that something in it is executable: a
    command, a variable to export, or the file to add.
    """
    executable = ("ollama ", "docker ", "python -m ", "export ", ".github/workflows", "http")
    for check in provoked_failures(doctor):
        assert any(hint in check.fix for hint in executable), f"{check.name}: {check.fix}"
        assert len(check.fix) > 40, f"{check.name}: the fix is too terse to act on"


# --------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------


def test_main_exits_non_zero_when_something_is_wrong(doctor, capsys):
    healthy, _, _, _ = doctor()
    assert main([REPO], doctor=healthy) == 0
    assert "all 19 preconditions met" in capsys.readouterr().out

    broken, _, _, _ = doctor(runner=RecordingRunner(images=()))
    assert main([REPO], doctor=broken) == 1
    captured = capsys.readouterr()
    assert "docker build" in captured.out
    assert "start nothing" in captured.err


# --------------------------------------------------------------------------
# Live probes
#
# The doubles above prove the verdicts; these prove the probes. Both halves are
# needed and neither substitutes for the other: a `FakeInference` cannot show
# that `/api/tags` is the right endpoint, and a live check cannot be provoked
# to fail on demand. Gated per `tests/conftest.py`, so `pytest -q` stays
# hermetic and this file's `## Verify` command runs anywhere.
# --------------------------------------------------------------------------


@pytest.mark.ollama
def test_live_ollama_answers_the_two_endpoints():
    probe = HostInference(settings())
    assert probe.version()
    assert probe.installed()


@pytest.mark.ollama
def test_live_models_honour_the_schema():
    """The expensive one: two model loads, and a swap between them.

    Uses this machine's real `SETTINGS`, not the pinned test ones, because the
    question is whether *this* configuration works.
    """
    subject = Doctor(inference=HostInference())
    reachable = subject.check_ollama_reachable()
    if not reachable.ok:
        pytest.skip(reachable.detail)
    assert subject.check_models().ok, subject.check_models().detail
    assert subject.check_schema().ok, subject.check_schema().detail


@pytest.mark.network
def test_live_linear_answers_the_tracker_probe():
    """The half a double cannot prove: that this is really how a server behaves.

    `mcp.linear.app` challenges an unauthenticated `tools/list` with a 401
    (#143 probed it and quoted the response), so with no credential exported
    the correct verdicts are *reachable* and *unauthorized* - which is the cut
    this file's hermetic tests assert against a fake, and the cut that would be
    worth nothing if a real server did something else. No token is needed to
    run it, and none is sent.
    """
    subject = Doctor(
        env={},
        settings=settings(),
        inference=FakeInference(),
        tracker=parse_tracker(TRACKER_BLOCK, source="<live>"),
    )
    try:
        reachable = subject.check_tracker_reachable()
        if not reachable.ok:
            pytest.skip(reachable.detail)
        assert "refused the credential" in reachable.detail

        auth = subject.check_tracker_auth()
        assert auth.status == FAIL
        assert "APIARY_LINEAR_TOKEN" in auth.fix
    finally:
        subject._close_probe()


@pytest.mark.docker
def test_live_docker_checks_run_against_the_real_daemon():
    subject = Doctor(env={})
    assert subject.check_docker_cli().ok, subject.check_docker_cli().fix

    daemon = subject.check_docker_daemon()
    assert daemon.ok, daemon.fix
    # One per stack since #99 chooses the image per task. Whether any given
    # image is built on this machine is a fact about the machine; what must
    # hold is that each check reaches a verdict and that a failure names the
    # build command, because the orchestrator cannot build or pull one itself.
    for stack in subject.stacks:
        image = subject.check_stack_image(stack)
        assert image.status in (OK, FAIL, SKIP), stack
        if image.status == FAIL:
            assert "docker build" in image.fix, stack


# --------------------------------------------------------------------------
# The two clocks
# --------------------------------------------------------------------------
#
# `verify_timeout_s` runs *inside* `worker_timeout_s`, and the outer one has to
# cover the clone, one whole-file inference call at a measured ~83 tok/s, the
# verify run, the commit, the push and the pull request. At the old 600 the
# inner 300 was not reachable in practice, and the failure that produced named
# the container rather than the gate - so raising the verify budget in response
# bought nothing and looked like a different bug.


def test_the_default_pair_leaves_the_verify_budget_reachable():
    """The outer clock was the binding one, so the outer clock is what moved.

    Measured worst case for a verify run is 59s cold and 6s warm, so 300s has
    ample headroom and raising it would buy nothing.
    """
    defaults = Settings()

    assert defaults.worker_timeout_s == 1200
    assert defaults.verify_timeout_s == 300
    assert defaults.clock_conflict() == ""


@pytest.mark.parametrize(
    "worker, verify",
    [
        (600, 600),  # equal: the inner clock can never fire first
        (600, 900),  # inverted outright
        (300, 300),
    ],
)
def test_an_inverted_pair_is_refused(doctor, worker, verify):
    subject, _, _, _ = doctor(
        settings=settings(worker_timeout_s=worker, verify_timeout_s=verify)
    )

    verdict = subject.run().by_name(CHECK_TIMEOUTS)

    assert verdict.status == FAIL
    # Both variables, because either one of them could be the one that is
    # wrong and the operator set only one of them.
    assert "SWARM_WORKER_TIMEOUT" in verdict.detail
    assert "SWARM_VERIFY_TIMEOUT" in verdict.detail


def test_the_fix_says_why_the_outer_clock_must_be_bigger(doctor):
    """`Check.__post_init__` enforces that a fix exists; this is about it being
    a fix rather than a restatement. "Raise the timeout" would not tell anyone
    which of the two, or by how much, or what the extra budget is for."""
    subject, _, _, _ = doctor(settings=settings(worker_timeout_s=600, verify_timeout_s=900))

    verdict = subject.run().by_name(CHECK_TIMEOUTS)

    assert "export SWARM_WORKER_TIMEOUT=" in verdict.fix
    assert "SWARM_VERIFY_TIMEOUT" in verdict.fix
    for cause in ("clone", "inference", "push"):
        assert cause in verdict.fix


def test_a_healthy_pair_reports_both_numbers(doctor):
    subject, _, _, _ = doctor(settings=settings(worker_timeout_s=1200, verify_timeout_s=300))

    verdict = subject.run().by_name(CHECK_TIMEOUTS)

    assert verdict.status == OK
    assert "300" in verdict.detail and "1200" in verdict.detail


def test_the_timeout_check_probes_nothing(doctor):
    """Arithmetic over two variables. It must not need a daemon, a token or a
    model to answer - it is the cheapest check in the module and it stays that
    way."""
    subject, transport, runner, _ = doctor(settings=settings(verify_timeout_s=9000))
    before = (len(transport.calls), len(runner.calls))

    verdict = subject.check_timeouts()

    assert verdict.status == FAIL
    assert (len(transport.calls), len(runner.calls)) == before
