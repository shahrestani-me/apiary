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

import pathlib
import signal
import sys
import threading
import time
import time
from types import SimpleNamespace

import pytest

from swarm.console import Console
from swarm.console_projects import ProjectStore
from swarm.console_runs import (
    MAX_LINES,
    MERGE_OVERRIDE_ENV,
    PROVISION_TOKEN_ENV,
    SWARM_SITE,
    WORK_TOKEN_ENV,
    RunJob,
    SwarmRunError,
    SwarmRuns,
    build_argv,
    child_env,
)
from fixtures.procs import FakeProc, settle, spawner

HOST = {"Host": "127.0.0.1:8117"}


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


def scratch_store(tmp_path) -> ProjectStore:
    """A projects store that cannot touch the real `.swarm/projects.sqlite`.

    Any console that *starts* a run records a project as a side effect, so
    every test that reaches a successful start hands its console one of these.
    """
    return ProjectStore(path=tmp_path / "projects.sqlite", runs_root=tmp_path / "runs")


def test_the_start_route_streams_and_the_status_route_follows(tokens, tmp_path):
    import json

    proc = FakeProc()
    console = Console(runs=SwarmRuns(spawn=spawner(proc), exists=lambda r: True),
                      projects=scratch_store(tmp_path))

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


def test_a_second_start_is_a_409(tokens, tmp_path):
    import json

    proc = FakeProc()
    console = Console(runs=SwarmRuns(spawn=spawner(proc), exists=lambda r: True),
                      projects=scratch_store(tmp_path))
    payload = json.dumps({"values": {"objective": "x", "repo": "a/b"}}).encode()
    first = console.render("POST", "/swarm/start", HOST, payload)

    second = console.render("POST", "/swarm/start", HOST, payload)

    assert first.status == 202 and second.status == 409
    proc.finish(0)


def test_the_status_route_refuses_an_unknown_or_traversing_id():
    console = Console()

    assert console.render("GET", "/swarm/status?id=deadbeef", HOST).status == 404
    assert console.render("GET", "/swarm/status?id=../../etc", HOST).status == 400


def test_a_reloaded_page_adopts_the_latest_run(tokens, tmp_path):
    """A run fired before a reload - or by another session entirely - must be
    visible, whole log included, not an empty tab beside a working swarm."""
    import json

    proc = FakeProc()
    console = Console(runs=SwarmRuns(spawn=spawner(proc), exists=lambda r: True),
                      projects=scratch_store(tmp_path))
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


# --------------------------------------------------------------------------
# How a run ends, and whether the page can tell (#130)
# --------------------------------------------------------------------------


def ended(lines: tuple[str, ...], returncode: int, *, stop: bool = False) -> dict:
    """One run, fed a scripted ending, settled, and read back as the page reads it."""
    proc = FakeProc()
    runs = SwarmRuns(spawn=spawner(proc), exists=lambda r: True)
    job = runs.start({"objective": "x", "repo": "a/b"})
    for line in lines:
        proc.feed(line)
    if stop:
        runs.stop(job.id)          # sends SIGINT, which ends this fake at 130
    else:
        proc.finish(returncode)
    settle(job)
    return runs.status(job.id)


@pytest.mark.usefixtures("tokens")
def test_a_run_that_met_its_objective_says_so_rather_than_done():
    status = ended(("» objective met: every task is verified and merged",), 0)

    assert status["state"] == "done"
    assert status["progress"]["outcome"] == "met"


@pytest.mark.usefixtures("tokens")
def test_a_run_that_ran_out_of_cycles_is_not_reported_as_a_finished_project():
    """The ending `state` cannot express, and the reason `outcome` exists.

    `_report_outcome` exits 0 for a cap as well as for a met objective - the
    work is simply unfinished, which is not a failure - so a page reading the
    exit code alone tells the operator a half-built project is done.
    """
    status = ended(("» cycle 3: 2 live, 1 ready",
                    "» stopped after 3 cycle(s) with 2 live issue(s)"), 0)

    assert status["state"] == "done"
    assert status["progress"]["outcome"] == "capped"
    assert status["progress"]["met"] is False


@pytest.mark.usefixtures("tokens")
def test_a_run_that_stopped_short_is_a_failure_with_its_exit_code():
    status = ended(("» the objective is not met: the CLI has no `add` command",), 1)

    assert (status["state"], status["progress"]["outcome"]) == ("failed", "failed")
    assert status["returncode"] == 1


@pytest.mark.usefixtures("tokens")
def test_a_stopped_run_is_stopped_and_not_a_failure_whatever_it_exits():
    """130 is what a `SIGINT`ed run exits, and calling the operator's own Stop
    a failure is the one reading of that exit code that is never true."""
    status = ended((), 0, stop=True)

    assert (status["state"], status["progress"]["outcome"]) == ("stopped", "stopped")
    assert status["returncode"] == 130


def test_a_run_that_could_not_be_started_at_all_still_ends(tokens):
    """The console has to survive this ending too: no child, so no watcher
    thread will ever publish a verdict for this job."""
    def refuses(argv, **kwargs):
        raise OSError("no such file or directory: python")

    runs = SwarmRuns(spawn=refuses, exists=lambda r: True)
    with pytest.raises(SwarmRunError):
        runs.start({"objective": "x", "repo": "a/b"})

    job = next(iter(runs.jobs.values()))
    assert (job.state, job.progress["outcome"]) == ("failed", "failed")
    assert runs.live() is None


# --------------------------------------------------------------------------
# Stop, and what it is for
# --------------------------------------------------------------------------


def test_stop_disposes_every_container_the_run_spawned(monkeypatch):
    """The half of Stop that happens in the child, proven rather than assumed.

    `test_stop_is_sigint_so_the_run_disposes_its_containers` pins the signal;
    this pins what the signal is *worth*. `cli._loop` runs the reconcile loop
    inside `Reaper.guard()`, so the `KeyboardInterrupt` a `SIGINT` becomes
    unwinds out of `Reconciler.loop` and the guard's exit sweep removes this
    run's containers on the way past. A Stop that killed the process instead -
    `SIGKILL`, or an in-process thread abandoned - would leave every worker
    holding a clone, a disk and a token, and nothing here would notice.

    Driven through the real `Reaper` against `fixtures.docker.Daemon`, which
    parses the `--filter` arguments for real. That is what makes the second
    assertion mean something: the sweep removes the run's own containers and
    leaves the machine's alone.
    """
    import swarm.cli as cli
    from swarm.containers.manager import DockerCLI
    from swarm.run import Run

    from fixtures.docker import Container, Daemon

    monkeypatch.delenv("APIARY_CAPTURE", raising=False)
    run = Run.start("shahrestani-me/expense-tracker", "a CLI that tracks expenses")
    daemon = Daemon([
        Container("worker-one", run_id=run.id, issue=1, state="running"),
        Container("worker-two", run_id=run.id, issue=2),
        # Somebody's database, wearing no run label. Never this sweep's to touch.
        Container("postgres", name="db", image="postgres"),
    ])

    class Interrupted:
        """The loop, ended the way Stop ends it."""

        def loop(self, *, cycles=None):
            raise KeyboardInterrupt

    class Nothing:
        """Every collaborator `_loop` opens and closes around the loop."""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def summary(self) -> str:
            return "store: hermetic"

    monkeypatch.setattr("swarm.orchestrator.reconcile.Reconciler",
                        lambda **kwargs: Interrupted())
    monkeypatch.setattr("swarm.containers.manager.ContainerManager",
                        lambda **kwargs: SimpleNamespace(docker=DockerCLI(runner=daemon)))
    monkeypatch.setattr("swarm.store.SqliteTaskStore.open",
                        classmethod(lambda cls, repo: Nothing()))
    artifacts = Nothing()
    artifacts.worker_env = lambda: {}
    artifacts.mount_flags = lambda: []
    artifacts.log_sink = lambda handle: None
    artifacts.results_dir = "/var/apiary/results"
    artifacts.event = lambda name, **fields: {}
    artifacts.observed = lambda payload: {}
    monkeypatch.setattr(cli.RunArtifacts, "open", classmethod(lambda cls, r, **_: artifacts))

    code = cli._loop(
        SimpleNamespace(base_commit="", no_merge=True, no_goal_check=True,
                        dry_run=False, max_cycles=None),
        SimpleNamespace(run=run),
        source=SimpleNamespace(head_sha=lambda ref=None: "a" * 40),
    )

    assert code == 130                      # what `swarm run` exits on Ctrl-C
    assert daemon.ids == ["postgres"]       # both workers gone, the machine's untouched


# --------------------------------------------------------------------------
# The three lines an ending is read from, pinned to the code that prints them
# --------------------------------------------------------------------------


def test_the_ending_lines_match_what_the_run_actually_prints():
    """The one coupling in this module whose drift is silent *and* wrong.

    Every other line parsed here degrades to "no summary" when `cli.py` is
    reworded. These two do not: both endings they name exit 0, so a miss
    reports an unfinished project as a finished one. Asserting against a
    literal copied into a test would pin the copy - so this drives the real
    `_report_outcome`, and for the met verdict the real `GoalReport.summary()`
    underneath it, which is where the words actually come from.
    """
    import io
    from contextlib import redirect_stdout
    from types import SimpleNamespace

    import swarm.cli as cli
    from swarm.console_runs import _CAPPED_LINE, _MET_LINE

    def printed(reports) -> list[str]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli._report_outcome(reports)
        return buffer.getvalue().splitlines()

    # The ledger did not decide it. The count in the sentence is what
    # `_no_verdict` reads, so the group has to survive a reword too.
    capped = printed([SimpleNamespace(goal=None, live=2)])
    matched = [_CAPPED_LINE.search(line) for line in capped]
    assert any(matched), capped
    assert next(m for m in matched if m).group(1) == "2"

    # And the met verdict, through the object that composes the words. Built
    # rather than faked: pinning `_report_outcome`'s `»` prefix while leaving
    # `GoalReport.summary()` free is how "objective met" could be reworded to
    # "the objective is met" with this test still green and every met run on
    # the page pilled "done".
    from swarm.orchestrator.goal import Assessment, GoalReport

    met = GoalReport(
        assessment=Assessment(met=True, reason="everything is merged"),
        reason="everything is merged",
    )
    assert met.met                                   # the real object, really met
    assert _MET_LINE.search(f"» {met.summary()}")
    assert any(_MET_LINE.search(line) for line in printed([SimpleNamespace(goal=met)]))


@pytest.mark.usefixtures("tokens")
def test_a_model_sentence_about_the_objective_does_not_declare_it_met():
    """`met` used to be a bare substring test over merged stdout+stderr, and
    the goal judge prints a model-written `reason` into the cycle summary. One
    task goal reading "not the objective met by this task" was enough to pill a
    capped run "objective met"."""
    status = ended(("  · goal: not the objective met by this task, says the judge",
                    "» stopped after 2 cycle(s) with 1 live issue(s)"), 0)

    assert status["progress"]["met"] is False
    assert status["progress"]["outcome"] == "capped"


@pytest.mark.usefixtures("tokens")
def test_the_ending_survives_a_run_that_printed_more_than_the_log_cap():
    """The cap bounds storage, not parsing - and the lines that decide the
    ending are the last ones. A run verbose enough to hit 10 000 lines used to
    have its `» objective met:` thrown away with the rest, and every one of
    them was pilled "done"."""
    noise = tuple(f"  · noise {i}" for i in range(MAX_LINES + 5))
    status = ended(noise + ("» objective met: every task is verified and merged",), 0)

    assert status["progress"]["outcome"] == "met"
    assert any("output truncated" in line for line in status["lines"])


@pytest.mark.usefixtures("tokens")
def test_a_plan_that_simply_finished_is_not_reported_as_a_cap():
    """`_report_outcome` prints one line for two endings: the cap ran out, and
    the goal gate was off so nothing was ever asked to judge. Calling the
    second "cycle cap reached" tells the operator a cap was reached when none
    was set - the same wrong answer this field exists to remove, inverted."""
    gate_off = ended(("» goal gate: off; the run stops when the plan is exhausted",
                      "» cycle 4: 0 live, 0 ready",
                      "» stopped after 4 cycle(s) with 0 live issue(s)"), 0)
    capped = ended(("» cycle 4: 2 live, 1 ready",
                    "» stopped after 4 cycle(s) with 2 live issue(s)"), 0)

    assert gate_off["progress"]["outcome"] == "exhausted"
    assert capped["progress"]["outcome"] == "capped"


def test_the_page_can_name_every_ending_the_server_can_write():
    """The py/JS vocabulary agreement, pinned the way `test_console.py` pins the
    others - and pinned against `console_runs.OUTCOMES` rather than against a
    list retyped here, which would pass happily while a *seventh* ending added
    to `conclude` fell through the page's `OUTCOMES[...] || j.state` and read
    "done" on screen. Silently, which is the failure the field exists to
    remove."""
    from swarm.console import asset
    from swarm.console_runs import OUTCOMES

    table = asset("app.js").split("var OUTCOMES")[1].split("};")[0]
    for outcome in OUTCOMES:
        assert f"{outcome}:" in table, outcome

    # ...and every key in the table is one the server can actually write.
    named = {part.rsplit("{", 1)[-1].split(":")[0].strip()
             for part in table.split(",") if ":" in part}
    assert named == set(OUTCOMES)

    # And `OUTCOMES` itself is every word the endings assign, read out of the
    # source rather than trusted. Without this the tuple is a third list to
    # keep in step by hand, and a seventh ending added inside `conclude` alone
    # would satisfy everything above while rendering as "done" on the page.
    import re

    from swarm import console_runs

    body = pathlib.Path(console_runs.__file__).read_text()
    # Only the region that decides an ending - `conclude`, `_no_verdict` and
    # `fail`. Sweeping the whole module would collect any unrelated string
    # somebody adds later and fail with a diff that explains nothing.
    deciders = body[body.index("def conclude"):body.index("def to_dict")]
    written = set(re.findall(r'"outcome"\]\s*=\s*"([a-z]+)"', deciders))
    written |= set(re.findall(r'return "([a-z]+)" if .* else "([a-z]+)"', deciders)[0])
    assert written == set(OUTCOMES), written ^ set(OUTCOMES)


def test_the_view_that_draws_last_owns_the_run():
    """The one front-end fix with no other pin, and the bug behind it was found
    by driving the page rather than by reading it: two views of one run, polls
    arriving in pairs, and only the later chain cancellable. Both halves are
    asserted - the timer the sleeping chain holds, and the generation an
    in-flight request is checked against when it resolves."""
    from swarm.console import asset

    script = asset("app.js")

    # Anchored to its context, not matched loose: `clearTimeout(runTimer)`
    # also appears in `swarmFire`, so a bare substring test stayed green with
    # the one that closes this bug deleted.
    assert ("clearTimeout(runTimer);\n"
            "    runTimer = null;\n"
            "    var generation = ++runGeneration") in script
    assert "if (view.generation !== runGeneration) return" in script


def test_a_stop_that_lands_before_the_child_exists_is_not_lost(tokens):
    """`start` claims the slot under the lock but registers the process only
    after `spawn` returns. A Stop in that window used to return 200 having done
    nothing: the button greyed out, the child spawned a moment later, and the
    run the operator stopped kept its containers."""
    proc = FakeProc()
    runs = SwarmRuns(exists=lambda r: True)
    stopped: list[str] = []

    def spawn(argv, **kwargs):
        # The window, made deterministic: the job is published and `_running`
        # is claimed by now, and `_procs` is not.
        job_id = next(iter(runs.jobs))
        runs.stop(job_id)
        stopped.append(job_id)
        return proc

    runs.spawn = spawn
    job = runs.start({"objective": "x", "repo": "a/b"})
    settle(job, state="stopped")

    assert stopped and job.stop_requested
    assert proc.signals == [signal.SIGINT]      # honoured late, not dropped
    assert runs.status(job.id)["progress"]["outcome"] == "stopped"


def test_a_stop_racing_the_spawn_interrupts_exactly_once(tokens):
    """One press of Stop must be one `SIGINT`, whatever the threads do.

    Not a repeat-is-harmless case. `Reaper._on_signal` reads a *second*
    interrupt as a human who is done waiting: it restores the default
    disposition and chains immediately, raising `KeyboardInterrupt` inside the
    first sweep's own `for handle in doomed:` loop and abandoning the removals
    it had not reached. `guard()`'s exit sweep is a backstop, so nothing leaks
    for ever - but the handler sweep the reaper is built around is abandoned by
    a single press, which is the mechanism this ticket exists to protect.

    Two assertions, because the interesting one cannot be staged. The window
    was between registering the process and reading the stop flag; the fix puts
    both in one critical section, so a test that *held* `start` open inside it
    would simply deadlock the stopper against the lock - which is the fix
    working, not a test. So the first assertion is the invariant itself: the
    lock is held while `_procs` is written, which is what makes "`stop` finds a
    process, or `start` honours the flag - never both" true by construction.
    The second races the two for real and counts the signals.
    """
    proc = FakeProc()
    runs = SwarmRuns(spawn=lambda argv, **kwargs: proc, exists=lambda r: True)
    locked_during_registration: list[bool] = []

    class WatchesTheLock(dict):
        def __setitem__(self, key, value):
            # Non-reentrant, and this runs on `start`'s own thread: a refused
            # acquisition means `start` is already inside the critical section.
            free = runs._lock.acquire(blocking=False)
            if free:
                runs._lock.release()
            locked_during_registration.append(not free)
            super().__setitem__(key, value)

    runs._procs = WatchesTheLock()
    job = runs.start({"objective": "x", "repo": "a/b"})

    assert locked_during_registration == [True]

    # ...and the ordinary path end to end, which the invariant above is what
    # makes safe: one stop after the run is up, one signal.
    runs.stop(job.id)
    settle(job, state="stopped")

    assert proc.signals == [signal.SIGINT]
    assert runs.status(job.id)["progress"]["outcome"] == "stopped"


def test_a_cursor_past_the_log_is_not_rewound_into_a_replay():
    """`next` is paired with the slice it was returned beside. Deriving it from
    the whole log instead - which is what it did - rewinds a client whose
    cursor is past the end (a hand-written `?since=`, or one held across the
    `MAX_LINES` freeze) and replays the entire run at it."""
    job = RunJob(id="x", command="c", started=time.monotonic())
    job.absorb("one")

    assert job.to_dict(since=999)["next"] == 999
    assert job.to_dict(since=999)["lines"] == []
    assert job.to_dict(since=0)["next"] == 1


def test_a_status_snapshot_is_not_the_live_progress_dict():
    """The page's poll is serialised on the HTTP thread while `_watch` is still
    appending to `prs` and `issues`, and `json.dumps` over a structure somebody
    else is mutating raises rather than answering."""
    job = RunJob(id="x", command="c", started=time.monotonic())
    job.absorb("» cycle 1: opened PR #4 for add-store")

    snapshot = job.to_dict()

    assert snapshot["progress"]["prs"] == [4]
    assert snapshot["progress"] is not job.progress
    assert snapshot["progress"]["prs"] is not job.progress["prs"]


# --------------------------------------------------------------------------
# The terminal panel (#134): what the page draws when a run stops
# --------------------------------------------------------------------------


def test_the_ending_panel_is_read_from_the_artifacts_in_both_views():
    """A run that stops has to say so, and the console has two views that can
    be looking when it does: its own job, and a run it did not start. Both draw
    the panel from `/swarm/outcome` - the run's own summary - rather than from
    the job in memory, which is what makes the answer survive a page reload and
    exist at all for a run this process never spawned."""
    from swarm.console import asset

    script = asset("app.js")

    assert 'api("/swarm/outcome?run=" + encodeURIComponent(runId))' in script
    # The console's own run, at the moment it stops being "running"...
    assert 'drawOutcome(this, p.run_id, box);' in script
    # ...and the artifacts view, only once the summary proves it finished.
    assert 'if (b.state === "finished") drawOutcome(extView, b.run_id, box);' in script


def test_the_panel_quotes_the_run_and_never_hides_a_task_needing_a_person():
    """Two of the four criteria, in the page's own source. The sentence is
    `o.reason` - the run's, composed by `close_the_loop` and the goal gate -
    and the tasks needing a human get their own strip rather than a share of a
    total, because a total is where the one that must not scroll past goes."""
    from swarm.console import asset

    script = asset("app.js")
    panel = script.split("function outcomeCard")[1].split("\n  function drawOutcome")[0]

    assert 'el("p", "proposal", o.reason' in panel
    assert '"needs a human"' in panel
    assert "tasks.needs_human" in panel
    # Counts and clock, each named rather than summed into one number.
    for term in ('"tasks"', '"pull requests"', '"cycles"', '"wall clock"'):
        assert term in panel, term


def test_every_field_the_panel_reads_is_one_the_route_actually_sends(tmp_path):
    """The py/JS agreement for this payload, pinned mechanically. A page
    reading `o.walls` off a server sending `wall_s` renders an empty pill and
    says nothing about it - which is the silent half of the failure this whole
    panel exists to remove."""
    import re

    from swarm.console import asset
    from swarm.console_external import run_outcome

    from test_console_external import finished_run

    finished_run(tmp_path)
    payload = run_outcome(root=tmp_path)
    script = asset("app.js")
    panel = script.split("function outcomeCard")[1].split("\n  function drawOutcome")[0]
    panel += script.split("function issueRef")[1].split("\n  function outcomeCard")[0]

    read = set(re.findall(r"\bo\.([a-z_]+)", panel))

    assert read, "the panel reads nothing off the payload - has it been rewritten?"
    assert read <= set(payload), read - set(payload)


def test_a_cap_is_neither_a_success_nor_a_failure_on_the_panel():
    """The four endings have to render distinctly, and "distinctly" is not only
    the word: work left open is not a failure, and `exhausted` is not a
    disappointment - the goal gate was off, so the run did exactly what was
    planned."""
    from swarm.console import asset

    rule = asset("app.js").split("function outcomeClass")[1].split("}")[0]

    assert 'kind === "met" || kind === "exhausted" ? " ok"' in rule
    assert 'kind === "failed" ? " bad" : ""' in rule
