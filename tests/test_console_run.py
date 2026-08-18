"""The swarm tab: the whole pipeline behind the console, without a subprocess.

Every test here drives `SwarmRuns` through its `spawn` seam with a scripted
fake process, exactly as `provision`'s tests drive it through `target`. What
is pinned: the argv a form becomes, the environment the child inherits, the
refusals and their fixes, the streaming contract (`since`/`next`), the
progress strip parsed from the run's own output, single-flight, and the stop
path being SIGINT - because SIGINT is what lets `cli._loop` dispose the run's
containers on the way out.
"""

from __future__ import annotations

import queue
import signal
import sys
import threading
import time

import pytest

from swarm.console import Console
from swarm.console_runs import (
    MERGE_OVERRIDE_ENV,
    PROVISION_TOKEN_ENV,
    SWARM_SITE,
    WORK_TOKEN_ENV,
    SwarmRunError,
    SwarmRuns,
    build_argv,
    child_env,
)

HOST = {"Host": "127.0.0.1:8117"}


# --------------------------------------------------------------------------
# A process the test scripts
# --------------------------------------------------------------------------


class Script:
    """A stdout whose lines arrive when the test says so."""

    def __init__(self) -> None:
        self._lines: queue.Queue[str] = queue.Queue()

    def feed(self, *lines: str) -> None:
        for line in lines:
            self._lines.put(line + "\n")

    def close(self) -> None:
        self._lines.put("")

    def readline(self) -> str:
        return self._lines.get()


class FakeProc:
    def __init__(self, returncode: int = 0) -> None:
        self.stdout = Script()
        self.returncode = returncode
        self.signals: list[int] = []
        self._done = threading.Event()

    def feed(self, *lines: str) -> None:
        self.stdout.feed(*lines)

    def finish(self, returncode: int | None = None) -> None:
        if returncode is not None:
            self.returncode = returncode
        self.stdout.close()
        self._done.set()

    def wait(self) -> int:
        self._done.wait(timeout=5)
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.finish(130)


def spawner(proc: FakeProc):
    """A `spawn` that records the argv and env it was invoked with."""

    def spawn(argv, **kwargs):
        spawn.argv = argv
        spawn.env = kwargs.get("env")
        return proc

    spawn.argv = None
    spawn.env = None
    return spawn


def settle(job, *, state: str | None = None) -> None:
    """Wait for the watcher thread to publish, bounded rather than flaky."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if state is None and job.state != "running":
            return
        if state is not None and job.state == state:
            return
        time.sleep(0.01)
    raise AssertionError(f"run never settled: state={job.state!r}")


@pytest.fixture
def tokens(monkeypatch):
    monkeypatch.setenv(WORK_TOKEN_ENV, "github_pat_test")
    monkeypatch.setenv(PROVISION_TOKEN_ENV, "github_pat_boot")


# --------------------------------------------------------------------------
# The form becomes the command a terminal would run
# --------------------------------------------------------------------------


def test_a_missing_repository_becomes_the_new_command():
    """The operator names where the work should go; existence picks the mode."""
    argv = build_argv({"objective": "an expense tracker",
                       "repo": "kamyar-finlex/expense-tracker"}, exists=False)

    assert argv[:2] == [sys.executable, "-u"]
    assert argv[2:] == ["-m", "swarm.cli", "run",
                        "--new", "an expense tracker",
                        "--owner", "kamyar-finlex", "--name", "expense-tracker", "--yes"]


def test_the_public_checkbox_reaches_the_new_command():
    """Private + free plan 403s on rulesets after the repo exists, so the
    visibility choice is the page's to state, not a default to discover."""
    public = build_argv({"objective": "x", "repo": "me/thing", "public": "1"}, exists=False)
    private = build_argv({"objective": "x", "repo": "me/thing", "public": ""}, exists=False)
    existing = build_argv({"objective": "x", "repo": "me/thing", "public": "1"}, exists=True)

    assert "--public" in public
    assert "--public" not in private
    assert "--public" not in existing  # visibility of an existing repo is not ours


def test_an_existing_repository_becomes_repo_and_objective():
    argv = build_argv({"objective": "add retries", "repo": "me/thing"}, exists=True)

    assert argv[2:] == ["-m", "swarm.cli", "run", "--repo", "me/thing", "--objective", "add retries"]


def test_the_optional_fields_pass_through():
    argv = build_argv({
        "objective": "a tool", "repo": "me/expense-tracker",
        "verify": "python -m pytest -q", "stack": "python", "max_cycles": "3",
    }, exists=False)

    tail = argv[5:]
    assert tail[:7] == ["--new", "a tool", "--owner", "me", "--name", "expense-tracker", "--yes"]
    for pair in (["--verify", "python -m pytest -q"],
                 ["--stack", "python"], ["--max-cycles", "3"]):
        i = tail.index(pair[0])
        assert tail[i:i + 2] == pair


@pytest.mark.parametrize(
    ("values", "said"),
    [
        ({"repo": "me/thing"}, "objective"),
        ({"objective": "x"}, "needs a repository"),
        ({"objective": "x", "repo": "not-a-slug"}, "owner/name"),
        ({"objective": "x", "repo": "a/b", "stack": "cobol"}, "unknown stack"),
        ({"objective": "x", "repo": "a/b", "max_cycles": "soon"}, "number"),
    ],
)
def test_a_bad_form_is_refused_before_anything_starts(values, said):
    with pytest.raises(SwarmRunError) as caught:
        build_argv(values, exists=True)

    assert said in str(caught.value)
    assert caught.value.fix  # every refusal names its fix


def test_the_module_the_console_execs_actually_executes():
    """Every other test here fakes the process, which is exactly how
    `-m swarm` shipped: the package has no `__main__.py`, so the first real
    click on the page died with 'swarm is a package and cannot be directly
    executed'. This one runs the real interpreter against the real module."""
    import subprocess

    argv = build_argv({"objective": "x", "repo": "a/b"}, exists=True)
    probe = subprocess.run([*argv[:5], "--help"], capture_output=True, text=True, timeout=30)

    assert probe.returncode == 0, probe.stderr
    assert "--repo" in probe.stdout


# --------------------------------------------------------------------------
# The child's environment
# --------------------------------------------------------------------------


def test_the_checkbox_is_the_merge_policy_whatever_the_shell_said():
    base = {WORK_TOKEN_ENV: "t", MERGE_OVERRIDE_ENV: "0"}

    on = child_env({"auto_merge": "1"}, greenfield=False, base=base)
    off = child_env({"auto_merge": ""}, greenfield=False,
                    base={**base, MERGE_OVERRIDE_ENV: "1"})

    assert on[MERGE_OVERRIDE_ENV] == "1"
    assert off[MERGE_OVERRIDE_ENV] == "0"


def test_a_missing_work_token_is_refused_with_the_fix():
    with pytest.raises(SwarmRunError) as caught:
        child_env({}, greenfield=False, base={})

    assert WORK_TOKEN_ENV in str(caught.value)
    assert "source" in caught.value.fix


def test_greenfield_needs_the_boot_token_and_an_existing_repo_does_not():
    base = {WORK_TOKEN_ENV: "t"}

    child_env({}, greenfield=False, base=base)  # fine without the boot key
    with pytest.raises(SwarmRunError) as caught:
        child_env({}, greenfield=True, base=base)

    assert PROVISION_TOKEN_ENV in str(caught.value)


# --------------------------------------------------------------------------
# Local mode: no GitHub at all
# --------------------------------------------------------------------------


def test_a_local_run_becomes_the_local_command():
    from swarm.console_runs import build_local_argv

    argv = build_local_argv({"objective": "a wallet tool", "repo": "~/poc/wallet-local",
                             "verify": "pytest -q", "max_cycles": "6"})

    assert argv[2:] == ["-m", "swarm.cli", "local",
                        "--repo", "~/poc/wallet-local", "--objective", "a wallet tool",
                        "--verify", "pytest -q", "--max-rounds", "6"]


def test_a_local_run_needs_no_tokens_and_asks_github_nothing(monkeypatch):
    """The whole point of local mode is days when GitHub is down."""
    monkeypatch.delenv(WORK_TOKEN_ENV, raising=False)
    monkeypatch.delenv(PROVISION_TOKEN_ENV, raising=False)

    def never(repo: str) -> bool:
        raise AssertionError("a local run must not probe GitHub")

    proc = FakeProc()
    spawn = spawner(proc)
    runs = SwarmRuns(spawn=spawn, exists=never)

    job = runs.start({"objective": "x", "repo": "~/poc/demo", "local": "1"})
    proc.finish(0)
    settle(job)

    assert "local" in spawn.argv
    assert job.command.startswith("swarm local ")


def test_a_local_path_never_becomes_a_github_link(tokens):
    """`Users/Kamyar` is a valid `owner/name`, which is exactly the trap."""
    proc = FakeProc()
    runs = SwarmRuns(spawn=spawner(proc), exists=lambda r: True)

    job = runs.start({"objective": "x", "repo": "/Users/Kamyar/poc/demo", "local": "1"})
    proc.feed("» local run  repo /Users/Kamyar/poc/demo  verify: pytest -q")
    proc.finish(0)
    settle(job)

    assert runs.status(job.id)["progress"]["repo_url"] == ""


# --------------------------------------------------------------------------
# Existence picks the mode
# --------------------------------------------------------------------------


def test_a_repository_that_is_not_there_is_created_not_404d(tokens):
    """The operator types where the work should go; the console asks GitHub
    whether it exists and picks attach or create. Both earlier forms that made
    the operator pick the mode were filled in wrong on their first day."""
    proc = FakeProc()
    spawn = spawner(proc)
    runs = SwarmRuns(spawn=spawn, exists=lambda r: False)

    runs.start({"objective": "a wallet tool", "repo": "kamyar-finlex/wallet-service"})
    proc.finish(0)

    assert "--new" in spawn.argv
    i = spawn.argv.index("--owner")
    assert spawn.argv[i:i + 4] == ["--owner", "kamyar-finlex", "--name", "wallet-service"]


def test_an_unanswerable_existence_probe_refuses_rather_than_guesses(tokens):
    """Against a GitHub outage, 'assume it is missing' would provision a
    duplicate the moment GitHub recovers."""

    def unsure(repo: str) -> bool:
        raise SwarmRunError(f"GitHub could not confirm whether {repo} exists: 503")

    runs = SwarmRuns(spawn=spawner(FakeProc()), exists=unsure)

    with pytest.raises(SwarmRunError) as caught:
        runs.start({"objective": "x", "repo": "a/b"})

    assert "could not confirm" in str(caught.value)
    assert runs.jobs == {}  # nothing half-started


# --------------------------------------------------------------------------
# Streaming, progress, single-flight, stop
# --------------------------------------------------------------------------


def test_the_page_reads_the_lines_the_run_printed(tokens):
    proc = FakeProc()
    runs = SwarmRuns(spawn=spawner(proc), exists=lambda r: True)

    job = runs.start({"objective": "x", "repo": "a/b"})
    proc.feed("» run a-b-20260817-120000-abcdef  repo a/b  objective: x",
              "  · cycle 1: dispatched 1")
    proc.finish(0)
    settle(job)

    status = runs.status(job.id)
    assert status["state"] == "done"
    assert status["lines"][0].startswith("» run")
    later = runs.status(job.id, since=status["next"])
    assert later["lines"] == []  # `since` means the page never re-downloads the log


def test_progress_is_parsed_from_the_run_output(tokens):
    proc = FakeProc()
    runs = SwarmRuns(spawn=spawner(proc), exists=lambda r: True)

    job = runs.start({"objective": "x", "repo": "kamyar-finlex/apiary-sandbox"})
    proc.feed(
        "» run apiary-sandbox-20260817-120000-abcdef  repo kamyar-finlex/apiary-sandbox  objective: x",
        "  · cycle 3: applied 1/1 transition(s); 1 live issue(s)",
        "      #3: passed - checks passed - waiting for a human to merge PR #5",
        "» objective met: the objective is met",
    )
    proc.finish(0)
    settle(job)

    p = runs.status(job.id)["progress"]
    assert p["repo"] == "kamyar-finlex/apiary-sandbox"
    assert p["repo_url"] == "https://github.com/kamyar-finlex/apiary-sandbox"
    assert p["cycle"] == 3
    assert p["issues"] == [3]
    assert p["prs"] == [5]
    assert p["met"] is True


def test_a_second_run_is_refused_not_queued(tokens):
    proc = FakeProc()
    runs = SwarmRuns(spawn=spawner(proc), exists=lambda r: True)
    job = runs.start({"objective": "x", "repo": "a/b"})

    with pytest.raises(SwarmRunError) as caught:
        runs.start({"objective": "y", "repo": "a/b"})

    assert "in flight" in str(caught.value)
    proc.finish(0)
    settle(job)
    runs.start({"objective": "y", "repo": "a/b"})  # and after it ends, allowed


def test_a_failing_run_reports_its_exit_code(tokens):
    proc = FakeProc()
    runs = SwarmRuns(spawn=spawner(proc), exists=lambda r: True)

    job = runs.start({"objective": "x", "repo": "a/b"})
    proc.feed("! planning failed: the model answered nothing")
    proc.finish(1)
    settle(job)

    status = runs.status(job.id)
    assert status["state"] == "failed"
    assert status["returncode"] == 1
    assert "planning failed" in status["progress"]["note"]


def test_stop_is_sigint_so_the_run_disposes_its_containers(tokens):
    proc = FakeProc()
    runs = SwarmRuns(spawn=spawner(proc), exists=lambda r: True)
    job = runs.start({"objective": "x", "repo": "a/b"})

    runs.stop(job.id)
    settle(job, state="stopped")

    assert proc.signals == [signal.SIGINT]
    assert runs.status(job.id)["state"] == "stopped"


def test_the_merge_override_reaches_the_child(tokens):
    proc = FakeProc()
    spawn = spawner(proc)
    runs = SwarmRuns(spawn=spawn, exists=lambda r: True)

    job = runs.start({"objective": "x", "repo": "a/b", "auto_merge": "1"})
    proc.finish(0)
    settle(job)

    assert spawn.env[MERGE_OVERRIDE_ENV] == "1"
    assert spawn.argv[2:5] == ["-m", "swarm.cli", "run"]


# --------------------------------------------------------------------------
# The routes
# --------------------------------------------------------------------------


def test_the_swarm_tab_is_served_beside_the_sites_not_among_them():
    import json

    console = Console()
    body = json.loads(console.render("GET", "/sites", HOST).body)

    assert body["swarm"]["kind"] == "swarm"
    assert all(s.get("kind") != "swarm" for s in body["sites"])
    names = [f["name"] for f in body["swarm"]["fields"]]
    assert "objective" in names and "auto_merge" in names


def test_the_start_route_streams_and_the_status_route_follows(tokens):
    import json

    proc = FakeProc()
    console = Console(runs=SwarmRuns(spawn=spawner(proc), exists=lambda r: True))

    started = console.render(
        "POST", "/swarm/start", HOST,
        json.dumps({"values": {"objective": "x", "repo": "a/b"}}).encode(),
    )
    assert started.status == 202
    job_id = json.loads(started.body)["id"]

    proc.feed("» run a-b-1  repo a/b  objective: x")
    proc.finish(0)
    settle(console.runs.jobs[job_id])

    status = json.loads(console.render("GET", f"/swarm/status?id={job_id}", HOST).body)
    assert status["state"] == "done"
    assert status["progress"]["repo"] == "a/b"


def test_a_bad_form_is_a_400_with_the_fix_attached(tokens):
    import json

    console = Console(runs=SwarmRuns(spawn=spawner(FakeProc()), exists=lambda r: True))
    refused = console.render("POST", "/swarm/start", HOST,
                             json.dumps({"values": {"objective": "x"}}).encode())

    assert refused.status == 400
    body = json.loads(refused.body)
    assert "needs a repository" in body["error"]
    assert body["fix"]


def test_a_second_start_is_a_409(tokens):
    import json

    proc = FakeProc()
    console = Console(runs=SwarmRuns(spawn=spawner(proc), exists=lambda r: True))
    payload = json.dumps({"values": {"objective": "x", "repo": "a/b"}}).encode()
    first = console.render("POST", "/swarm/start", HOST, payload)

    second = console.render("POST", "/swarm/start", HOST, payload)

    assert first.status == 202 and second.status == 409
    proc.finish(0)


def test_the_status_route_refuses_an_unknown_or_traversing_id():
    console = Console()

    assert console.render("GET", "/swarm/status?id=deadbeef", HOST).status == 404
    assert console.render("GET", "/swarm/status?id=../../etc", HOST).status == 400


def test_a_reloaded_page_adopts_the_latest_run(tokens):
    """A run fired before a reload - or by another session entirely - must be
    visible, whole log included, not an empty tab beside a working swarm."""
    import json

    proc = FakeProc()
    console = Console(runs=SwarmRuns(spawn=spawner(proc), exists=lambda r: True))
    assert console.render("GET", "/swarm/latest", HOST).status == 404  # before any run

    started = console.render("POST", "/swarm/start", HOST,
                             json.dumps({"values": {"objective": "x", "repo": "a/b"}}).encode())
    proc.feed("» run a-b-1  repo a/b  objective: x")
    proc.finish(0)
    settle(console.runs.jobs[json.loads(started.body)["id"]])

    latest = json.loads(console.render("GET", "/swarm/latest", HOST).body)
    assert latest["id"] == json.loads(started.body)["id"]
    assert latest["lines"][0].startswith("» run")  # the whole log, from line zero


def test_the_swarm_routes_check_the_host_like_every_other_route():
    console = Console()
    evil = {"Host": "attacker.example"}

    assert console.render("POST", "/swarm/start", evil, b"{}").status == 403
    assert console.render("GET", "/swarm/status?id=deadbeef", evil).status == 403


def test_the_plan_exhausted_checkbox_reaches_the_command():
    """The goal gate is right for autonomy and wrong for a bounded demo - the
    strict local judge can keep planning follow-ups it never names a gap for.
    The operator's off-switch must not require a terminal."""
    on = build_argv({"objective": "x", "repo": "a/b", "no_goal_check": "1"}, exists=True)
    off = build_argv({"objective": "x", "repo": "a/b", "no_goal_check": ""}, exists=True)

    assert "--no-goal-check" in on
    assert "--no-goal-check" not in off
