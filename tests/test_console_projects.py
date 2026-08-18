"""The projects store: what the console remembers across restarts.

Every test drives a `ProjectStore` on paths under `tmp_path` - never the real
`.swarm/projects.sqlite`, never the real run directories - and the endpoint
tests build a `Console` directly and call `render`, exactly as
`test_console.py` does. The run-start tests drive `SwarmRuns` through its
`spawn` seam with the same scripted fake process `test_console_run.py` uses,
so nothing here execs a subprocess or asks GitHub anything.

What is pinned: the schema exists after first use, upsert semantics (an empty
field is "no news", never an eraser), the recency ordering the selector will
render, the migration from recorded run artifacts (one project per repo,
latest run's values, local paths and corrupt files skipped, never fatal), the
two routes, and the bookkeeping side effect of a GitHub run starting.
"""

from __future__ import annotations

import json
import queue
import threading
import time

import pytest

from swarm.console import Console
from swarm.console_projects import ProjectError, ProjectStore
from swarm.console_runs import SwarmRuns, WORK_TOKEN_ENV

HOST = {"Host": "127.0.0.1:8117"}


def body(response):
    return json.loads(response.body)


@pytest.fixture
def store(tmp_path) -> ProjectStore:
    """A store whose database and runs root both live under `tmp_path`."""
    return ProjectStore(path=tmp_path / "projects.sqlite", runs_root=tmp_path / "runs")


def record_run(root, name, *, repo, started, objective="an objective",
               stack="", verify="", raw: str | None = None) -> None:
    """One fake run directory holding exactly what the migration reads."""
    directory = root / name
    directory.mkdir(parents=True)
    if raw is not None:
        (directory / "run.json").write_text(raw)
        return
    (directory / "run.json").write_text(json.dumps({
        "schema": 1, "run_id": name, "repo": repo, "objective": objective,
        "started_at": started, "stack": stack, "verify": verify,
    }))


# --------------------------------------------------------------------------
# The store itself
# --------------------------------------------------------------------------


def test_the_schema_is_created_on_first_use(store):
    """No file until an operation needs one: `Console` carries a store by
    default factory, and building a console must write nothing to disk."""
    assert not store.path.exists()

    assert store.list() == []
    assert store.path.exists()


def test_an_upsert_inserts_and_get_reads_it_back(store):
    stored = store.upsert("kamyar-finlex/wallet-tracker-service",
                          "track a wallet", "python", "python -m pytest -q")

    assert stored["repo"] == "kamyar-finlex/wallet-tracker-service"
    assert stored["objective"] == "track a wallet"
    assert stored["stack"] == "python"
    assert stored["verify"] == "python -m pytest -q"
    assert stored["created_at"]  # stamped at insert
    assert stored["last_run_at"] is None  # saving a project is not running it
    assert store.get("kamyar-finlex/wallet-tracker-service") == stored
    assert store.get("nobody/nothing") is None


def test_an_empty_field_is_no_news_not_an_eraser(store):
    """The run form's stack and verify are optional, so most run starts would
    otherwise blank out what the migration or a previous run wrote down."""
    store.upsert("a/b", "the real objective", "python", "pytest -q")

    store.upsert("a/b", "", "", "")

    kept = store.get("a/b")
    assert kept["objective"] == "the real objective"
    assert kept["stack"] == "python"
    assert kept["verify"] == "pytest -q"


def test_a_non_empty_field_does_update(store):
    store.upsert("a/b", "old objective", "python", "")

    store.upsert("a/b", "new objective", "", "npm test")

    row = store.get("a/b")
    assert row["objective"] == "new objective"
    assert row["stack"] == "python"
    assert row["verify"] == "npm test"


def test_created_at_survives_every_later_upsert(store):
    first = store.upsert("a/b", "x", created_at="2026-01-01T00:00:00+00:00")

    store.upsert("a/b", "y", created_at="2026-08-18T00:00:00+00:00")

    assert store.get("a/b")["created_at"] == first["created_at"]


def test_last_run_at_only_moves_forward(store):
    """A replayed migration must not rewind a project behind a run the console
    recorded a minute ago."""
    store.upsert("a/b", "x", last_run_at="2026-08-18T12:00:00+00:00")

    store.upsert("a/b", "x", last_run_at="2026-08-17T12:00:00+00:00")

    assert store.get("a/b")["last_run_at"] == "2026-08-18T12:00:00+00:00"


@pytest.mark.parametrize("bad", ["", "no-slash", "/Users/Kamyar/poc/demo",
                                 "a/b/c", "owner/", "owner name/repo"])
def test_a_repo_that_is_not_a_slug_is_refused_with_the_fix(store, bad):
    """The repo is about to become a primary key the page renders into links,
    and a local run's directory path is the trap this shape check exists for."""
    with pytest.raises(ProjectError) as caught:
        store.upsert(bad, "an objective")

    assert "owner/name" in str(caught.value)
    assert caught.value.fix


def test_projects_are_listed_most_recently_active_first(store):
    """The order the selector will render: ran recently beats ran long ago,
    ever ran beats never ran, and never-ran projects go newest first."""
    store.upsert("old/runner", "x", last_run_at="2026-08-17T10:00:00+00:00")
    store.upsert("new/runner", "x", last_run_at="2026-08-18T11:00:00+00:00")
    store.upsert("never/ran-early", "x", created_at="2026-08-01T00:00:00+00:00")
    store.upsert("never/ran-late", "x", created_at="2026-08-15T00:00:00+00:00")

    listed = [row["repo"] for row in store.list()]

    assert listed == ["new/runner", "old/runner", "never/ran-late", "never/ran-early"]


# --------------------------------------------------------------------------
# The migration: run artifacts become the first project list
# --------------------------------------------------------------------------


def test_the_migration_folds_the_recorded_runs_into_projects(tmp_path, store):
    """One project per repo; the latest run's objective, stack and verify;
    created from the earliest run and last active at the latest. This is the
    seeding that brings `kamyar-finlex/wallet-tracker-service` in as the
    first project on a console that never saved one."""
    runs = tmp_path / "runs"
    record_run(runs, "wallet-tracker-service-20260818-075556-6x6wj4",
               repo="kamyar-finlex/wallet-tracker-service",
               started="2026-08-18T07:55:56+00:00",
               objective="the first draft of the objective", stack="", verify="")
    record_run(runs, "wallet-tracker-service-20260818-113702-g5hz5j",
               repo="kamyar-finlex/wallet-tracker-service",
               started="2026-08-18T11:37:02+00:00",
               objective="I want a small tool that tracks a wallet",
               stack="python", verify="python -m pytest -q")
    record_run(runs, "apiary-sandbox-20260817-102003-rr9hkz",
               repo="kamyar-finlex/apiary-sandbox",
               started="2026-08-17T10:20:03+00:00",
               objective="extend the sandbox calculator")

    listed = store.list()

    assert [row["repo"] for row in listed] == [
        "kamyar-finlex/wallet-tracker-service",
        "kamyar-finlex/apiary-sandbox",
    ]
    wallet = listed[0]
    assert wallet["objective"] == "I want a small tool that tracks a wallet"
    assert wallet["stack"] == "python"
    assert wallet["verify"] == "python -m pytest -q"
    assert wallet["created_at"] == "2026-08-18T07:55:56+00:00"
    assert wallet["last_run_at"] == "2026-08-18T11:37:02+00:00"


def test_the_migration_skips_what_it_cannot_or_must_not_read(tmp_path, store):
    """A local run records a filesystem path in the repo field, a killed run
    can leave half a `run.json`, and neither may cost the console its list."""
    runs = tmp_path / "runs"
    record_run(runs, "good-20260818-120000-abcdef", repo="a/b",
               started="2026-08-18T12:00:00+00:00")
    record_run(runs, "local-20260818-120100-abcdef",
               repo="/Users/Kamyar/poc/wallet-local",
               started="2026-08-18T12:01:00+00:00")
    record_run(runs, "corrupt-20260818-120200-abcdef", repo="ignored",
               started="ignored", raw='{"repo": "c/d", "started_at":')
    (runs / "empty-20260818-120300-abcdef").mkdir()  # no run.json at all
    record_run(runs, "undated-20260818-120400-abcdef", repo="e/f",
               started="not a timestamp")

    listed = store.list()

    assert [row["repo"] for row in listed] == ["a/b"]


def test_a_missing_runs_root_seeds_nothing_and_breaks_nothing(store):
    assert store.list() == []


def test_the_migration_is_idempotent_across_store_instances(tmp_path, store):
    """"On every open" is the contract, so two consoles - or one restarted -
    against the same file must agree rather than accumulate."""
    runs = tmp_path / "runs"
    record_run(runs, "run-20260818-120000-abcdef", repo="a/b",
               started="2026-08-18T12:00:00+00:00", objective="from the run")
    first = store.list()

    again = ProjectStore(path=store.path, runs_root=runs)
    # And an operator's newer objective survives the replayed seed's older one
    # only where the seed has nothing newer to say - here it does not run
    # again for the same repo values, so the row is simply unchanged.
    assert again.list() == first
    assert len(again.list()) == 1


# --------------------------------------------------------------------------
# The prompt history: a project's runs are its prompts
# --------------------------------------------------------------------------


def test_a_projects_history_is_its_runs_newest_first(tmp_path, store):
    """What the page shows when a project is selected: every objective the
    repository has already been asked for. Derived from the run artifacts,
    never stored twice, so a run fired from a terminal is in it too."""
    runs = tmp_path / "runs"
    record_run(runs, "w-20260818-075556-aaaaaa", repo="kamyar-finlex/wallet-tracker-service",
               started="2026-08-18T07:55:56+00:00", objective="the first prompt")
    record_run(runs, "w-20260818-113702-bbbbbb", repo="kamyar-finlex/wallet-tracker-service",
               started="2026-08-18T11:37:02+00:00", objective="the follow-up prompt",
               stack="python", verify="pytest -q")
    record_run(runs, "s-20260817-102003-cccccc", repo="kamyar-finlex/apiary-sandbox",
               started="2026-08-17T10:20:03+00:00", objective="somebody else's prompt")
    (runs / "w-20260818-075556-aaaaaa" / "summary.json").write_text("{}")  # this run ended

    prompts = store.history("kamyar-finlex/wallet-tracker-service")

    assert [p["objective"] for p in prompts] == ["the follow-up prompt", "the first prompt"]
    assert prompts[0]["run_id"] == "w-20260818-113702-bbbbbb"
    assert prompts[0]["stack"] == "python" and prompts[0]["verify"] == "pytest -q"
    assert prompts[0]["finished"] is False  # no summary.json: live or killed
    assert prompts[1]["finished"] is True
    assert store.history("nobody/never-ran") == []


def test_the_history_route_answers_for_one_repo(tmp_path, store):
    runs = tmp_path / "runs"
    record_run(runs, "w-20260818-113702-bbbbbb", repo="a/b",
               started="2026-08-18T11:37:02+00:00", objective="the prompt")
    console = Console(projects=store)

    response = console.render("GET", "/projects/history?repo=a%2Fb", HOST)

    assert response.status == 200
    assert body(response)["repo"] == "a/b"
    assert [p["objective"] for p in body(response)["prompts"]] == ["the prompt"]


def test_the_history_route_refuses_a_non_slug_repo(store):
    """The repo arrives from a query string on its way to a directory scan."""
    console = Console(projects=store)

    response = console.render("GET", "/projects/history?repo=..%2F..%2Fetc", HOST)

    assert response.status == 400
    assert "owner/name" in body(response)["error"]


# --------------------------------------------------------------------------
# The routes
# --------------------------------------------------------------------------


def test_the_projects_route_lists_what_the_store_holds(store):
    store.upsert("a/b", "an objective", last_run_at="2026-08-18T12:00:00+00:00")
    console = Console(projects=store)

    response = console.render("GET", "/projects", HOST)

    assert response.status == 200
    projects = body(response)["projects"]
    assert [p["repo"] for p in projects] == ["a/b"]
    assert projects[0]["objective"] == "an objective"


def test_posting_a_project_upserts_and_returns_the_stored_row(store):
    console = Console(projects=store)

    response = console.render(
        "POST", "/projects", HOST,
        json.dumps({"repo": "a/b", "objective": "an objective",
                    "stack": "python", "verify": "pytest -q"}).encode(),
    )

    assert response.status == 200
    stored = body(response)
    assert stored["repo"] == "a/b"
    assert stored["verify"] == "pytest -q"
    assert store.get("a/b")["objective"] == "an objective"


def test_a_bad_repo_is_a_400_with_the_fix_attached(store):
    console = Console(projects=store)

    response = console.render(
        "POST", "/projects", HOST,
        json.dumps({"repo": "not-a-slug", "objective": "x"}).encode(),
    )

    assert response.status == 400
    assert "owner/name" in body(response)["error"]
    assert body(response)["fix"]
    assert store.list() == []


def test_an_empty_objective_is_refused_not_stored(store):
    """A selector entry with no objective selects nothing worth selecting."""
    console = Console(projects=store)

    response = console.render(
        "POST", "/projects", HOST,
        json.dumps({"repo": "a/b", "objective": "   "}).encode(),
    )

    assert response.status == 400
    assert "objective" in body(response)["error"]
    assert store.list() == []


def test_the_projects_routes_check_the_host_like_every_other_route(store):
    console = Console(projects=store)
    evil = {"Host": "attacker.example"}

    assert console.render("GET", "/projects", evil).status == 403
    assert console.render("POST", "/projects", evil, b"{}").status == 403


# --------------------------------------------------------------------------
# A starting run is a project - the fake-spawn seam from test_console_run.py
# --------------------------------------------------------------------------


class Script:
    """A stdout whose lines arrive when the test says so."""

    def __init__(self) -> None:
        self._lines: queue.Queue[str] = queue.Queue()

    def close(self) -> None:
        self._lines.put("")

    def readline(self) -> str:
        return self._lines.get()


class FakeProc:
    def __init__(self) -> None:
        self.stdout = Script()
        self.returncode = 0
        self._done = threading.Event()

    def finish(self) -> None:
        self.stdout.close()
        self._done.set()

    def wait(self) -> int:
        self._done.wait(timeout=5)
        return self.returncode


def settle(job) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if job.state != "running":
            return
        time.sleep(0.01)
    raise AssertionError(f"run never settled: state={job.state!r}")


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv(WORK_TOKEN_ENV, "github_pat_test")


def test_a_github_run_start_records_the_project(store, token):
    """The submitted values become the project the operator comes back to,
    with the moment of starting as its recency."""
    proc = FakeProc()
    console = Console(runs=SwarmRuns(spawn=lambda *a, **k: proc, exists=lambda r: True),
                      projects=store)

    started = console.render(
        "POST", "/swarm/start", HOST,
        json.dumps({"values": {"objective": "track a wallet", "repo": "a/b",
                    "stack": "python", "verify": "pytest -q"}}).encode(),
    )
    proc.finish()
    settle(console.runs.jobs[body(started)["id"]])

    assert started.status == 202
    project = store.get("a/b")
    assert project["objective"] == "track a wallet"
    assert project["stack"] == "python"
    assert project["verify"] == "pytest -q"
    assert project["last_run_at"]  # starting is being active


def test_a_local_run_start_records_no_project(store, token):
    """A local run's repo field is a directory path, not a project on GitHub -
    the same reason `RunJob.local` never builds a repo link."""
    proc = FakeProc()

    def never(repo: str) -> bool:
        raise AssertionError("a local run must not probe GitHub")

    console = Console(runs=SwarmRuns(spawn=lambda *a, **k: proc, exists=never),
                      projects=store)

    started = console.render(
        "POST", "/swarm/start", HOST,
        json.dumps({"values": {"objective": "x", "repo": "/Users/Kamyar/poc/demo",
                    "local": "1"}}).encode(),
    )
    proc.finish()
    settle(console.runs.jobs[body(started)["id"]])

    assert started.status == 202
    assert store.list() == []


def test_a_refused_start_records_no_project(store, token):
    console = Console(runs=SwarmRuns(spawn=lambda *a, **k: FakeProc(),
                                     exists=lambda r: True),
                      projects=store)

    refused = console.render("POST", "/swarm/start", HOST,
                             json.dumps({"values": {"objective": "x"}}).encode())

    assert refused.status == 400
    assert store.list() == []
