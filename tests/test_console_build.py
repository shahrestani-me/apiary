"""Start building: what the operator approved is what GitHub receives.

The console could always render a good decomposition and could always start a
run, and until #129 those were two unconnected facts. Starting a run from a
plan went through `swarm run --new <objective>`, which provisions a repository
and then calls `plan_node` on it - so the approval travelled as an objective
string, and an objective string is precisely the input whose answer varies. The
plan that ran was never the plan that had been read.

So the load-bearing test in this file is
`test_the_issues_written_are_the_tasks_that_were_displayed`, and its companion
`test_not_one_model_is_asked_anything`: the first proves the tasks survive the
trip intact, and the second proves there was no second opinion to survive.
Everything else here is the order of the refusals - which all have to land
before a repository exists, because a refusal afterwards is something a human
has to go and delete.

No token, no daemon, no socket. `Builder`'s seams take a recording fake in
place of `provision` and of the GitHub client, exactly as `tests/test_cli_run.py`
drives the entrypoint, and the assertions are on the **issue payloads** rather
than on the page.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import pytest

from swarm.console import DEFAULT_HOST, DEFAULT_PORT, Console, Job
from swarm.console_build import BUILD_SITE, BuildError, Builder, plan_from_result
from swarm.console_runs import SwarmRuns
from swarm.doctor import Check, Diagnosis
from swarm.greenfield.provision import ProvisionReport
from swarm.github.labels import LabelReport

from fixtures.procs import FakeProc, settle, spawner

HOST = {"Host": f"{DEFAULT_HOST}:{DEFAULT_PORT}"}

#: Both keys present. The token preflight checks presence and nothing else -
#: validity is GitHub's to judge - so a plausible pair is all a test needs.
ENV = {"GITHUB_TOKEN": "ghp-work", "APIARY_PROVISION_TOKEN": "github_pat_boot"}


@pytest.fixture(autouse=True)
def tokens(monkeypatch):
    """The *process* environment, which is a different one from `ENV`.

    `Builder` reads the mapping it was handed; the run a finished build starts
    reads `os.environ`, because that is what its child will inherit. Autouse
    so that the chain is exercised by every test in this file rather than by
    the handful written for it - a build whose run silently refused to start
    everywhere would leave `report["run"]` unasserted and unnoticed.
    """
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)

FORM = {"owner": "shahrestani-me", "name": "expense-tracker", "stack": "python",
        "public": "1", "objective": "a CLI that tracks expenses"}

#: The scaffold unticked - the operator's decomposition and nothing else.
PLAN_ONLY = dict(FORM, bootstrap="")

#: Two tasks, the second blocked on the first, both writable. The shape a
#: planner call actually returns, down to the `stack` key `_plan_run` carries
#: precisely so that this rebuild is lossless.
PLAN = {
    "reasoning": "storage first, then the command that uses it",
    "tasks": [
        {"id": "add-store", "goal": "Create the on-disk store with add() and all().",
         "files": ["src/store.py", "tests/test_store.py"], "depends_on": [], "stack": "python"},
        {"id": "add-cli", "goal": "Wire a CLI over the store with an `add` subcommand.",
         "files": ["src/cli.py", "tests/test_cli.py"], "depends_on": ["add-store"],
         "stack": "python"},
    ],
}


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


@dataclass
class FakeIssues:
    """The three calls `write_plan` makes, and every issue it created.

    `created` is the assertion surface: it is the literal payload that would
    have gone to GitHub, which is what the ticket's test notes ask to be
    checked instead of the HTML.
    """

    repo: str = "shahrestani-me/expense-tracker"
    issues: list[dict[str, Any]] = field(default_factory=list)
    created: list[dict[str, Any]] = field(default_factory=list)

    def list_issues(self, *, state: str = "open", **_: Any) -> list[dict[str, Any]]:
        return [dict(payload) for payload in self.issues]

    def get_issue(self, number: int) -> dict[str, Any]:
        for payload in self.issues:
            if payload["number"] == number:
                return dict(payload)
        raise AssertionError(f"nothing should ask for issue {number}")

    def create_issue(self, title: str, *, body: str | None = None,
                     labels: Iterable[str] | None = None, **_: Any) -> dict[str, Any]:
        payload = {"number": 100 + len(self.created), "title": title,
                   "body": body or "", "labels": [{"name": n} for n in (labels or [])]}
        self.created.append({"title": title, "body": body or "",
                             "labels": list(labels or []), "number": payload["number"]})
        self.issues.append(payload)
        return payload


@dataclass
class FakeProvisioner:
    """`provision`, minus GitHub. Records the plan it was handed.

    A repository is the one thing this suite can never really make: the boot
    key `security.assert_provision_token` demands does not exist here, and
    minting one is not a test's business. So acceptance criterion 1 is proved
    up to this line and no further - see the module docstring of
    `swarm.console_build`.
    """

    calls: list[Any] = field(default_factory=list)
    branch: str = "main"

    def __call__(self, plan: Any, *, assume_yes: bool = False, out: Any = None) -> Any:
        self.calls.append(plan)
        return ProvisionReport(
            repo=plan.full_name,
            html_url=f"https://github.com/{plan.full_name}",
            default_branch=self.branch,
            commit_sha="0" * 40,
            labels=LabelReport(repo=plan.full_name),
            protection=("required status checks",),
            verify_command=plan.verify_command,
        )


def healthy(_stacks) -> Diagnosis:
    return Diagnosis((Check.passed("docker.daemon", "reachable"),
                      Check.passed("image.python", "present"),))


def console_with(*, runs: SwarmRuns | None = None, **overrides):
    """A console whose every edge is a double, including the one #130 added.

    The `runs` seam is not optional in spirit: since a finished build starts
    the swarm on the repository it made, a console left with a real `SwarmRuns`
    would exec `python -m swarm.cli run` out of the test suite the moment the
    developer running it happened to have `GITHUB_TOKEN` exported. Every caller
    gets a `SwarmRuns` driven through its `spawn` seam by a scripted process,
    which is also what lets the argv the build produces be asserted.
    """
    provisioner, issues = FakeProvisioner(), FakeIssues()
    seams: dict[str, Any] = {
        "provisioner": provisioner,
        "client_for": lambda repo: issues,
        "preflight": healthy,
        "env": ENV,
        "out": lambda line: None,
    }
    seams.update(overrides)
    return (
        Console(builder=Builder(**seams), runs=runs if runs is not None else swarm_runs(FakeProc())),
        provisioner,
        issues,
    )


def swarm_runs(proc: FakeProc) -> SwarmRuns:
    """`SwarmRuns` with a scripted child and no question asked of GitHub.

    `exists` raises rather than answering, which is an assertion in every test
    that uses this: the build tells `start` the repository is there, and a
    probe for a repository this process created seconds ago is the round trip
    #130 removed - the one whose 404 would send the chained run down the
    greenfield branch and provision a second repository over the first.
    """
    def never(repo: str) -> bool:
        raise AssertionError(f"the chained run asked GitHub whether {repo} exists")

    return SwarmRuns(spawn=spawner(proc), exists=never)


def planned(console: Console, result: Any = None) -> str:
    """A finished planner call sitting in the console, as the page would have
    left it: the operator has read this and is about to press the button."""
    job = Job(id="planjob00000001", site="planner", started=time.monotonic(),
              state="done", result=PLAN if result is None else result)
    console.jobs[job.id] = job
    return job.id


def build(console: Console, plan_id: str, values: dict[str, Any] | None = None):
    """POST /swarm/build and wait for the job, exactly as the page does."""
    started = console.render(
        "POST", "/swarm/build", HOST,
        json.dumps({"plan": plan_id, "values": FORM if values is None else values}).encode(),
    )
    if started.status != 202:
        return started, None
    job_id = json.loads(started.body)["id"]
    for _ in range(500):
        job = json.loads(console.render("GET", f"/status?id={job_id}", HOST).body)
        if job["state"] != "running":
            return started, job
        time.sleep(0.01)
    raise AssertionError("the build never finished")


# --------------------------------------------------------------------------
# The claim the button rests on
# --------------------------------------------------------------------------


def test_the_issues_written_are_the_tasks_that_were_displayed():
    """Acceptance criterion 3, and the reason the ticket exists.

    Same ids, same goals, same files - checked against the payloads that would
    have reached GitHub, not against the card that rendered them.
    """
    console, provisioner, issues = console_with()

    _, job = build(console, planned(console))

    assert job["state"] == "done", job.get("error")
    assert len(provisioner.calls) == 1
    assert [c["number"] for c in issues.created] == [100, 101, 102]

    # The scaffold leads, and it is the only issue that is not on the screen.
    scaffold, *theirs = issues.created
    assert "id=bootstrap-the-project" in scaffold["body"]

    # Titles are the goal made legible; the id is the marker in the body, which
    # is what §2 of the contract makes identity. Both are checked, per task.
    for created, task in zip(theirs, PLAN["tasks"]):
        assert task["id"] in created["body"]
        assert task["goal"] in created["body"]
        assert created["title"] in task["goal"]
        for path in task["files"]:
            assert path in created["body"]
    # The second task blocks on the first, and by number rather than by name.
    assert "#101" in theirs[1]["body"]

    # And the page says the same thing, with somewhere to click.
    assert [i["task_id"] for i in job["result"]["issues"]] == [
        "bootstrap-the-project", "add-store", "add-cli"]
    assert job["result"]["html_url"] == "https://github.com/shahrestani-me/expense-tracker"
    assert [i["url"] for i in job["result"]["issues"]] == [
        "https://github.com/shahrestani-me/expense-tracker/issues/100",
        "https://github.com/shahrestani-me/expense-tracker/issues/101",
        "https://github.com/shahrestani-me/expense-tracker/issues/102",
    ]


def test_not_one_model_is_asked_anything(monkeypatch):
    """Acceptance criterion 5, asserted where it cannot be evaded.

    Every model call in this system is a `ChatOllama` built by `swarm.llm`, and
    both factories look the class up as a module global at call time - so this
    catches an inference reached through any import style, in the planner, the
    bootstrap, the judge or anywhere else the build path might wander.
    """
    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the build asked a model; the plan on screen was approved")

    monkeypatch.setattr("swarm.llm.ChatOllama", boom)
    console, _, issues = console_with()

    _, job = build(console, planned(console))

    assert job["state"] == "done", job.get("error")
    # Three, because the scaffold is on by default - so this also pins the
    # other call this action could have made and does not: `Bootstrap.for_prompt`
    # consults `choose_stack` only when the stack is blank, and it never is here.
    assert len(issues.created) == 3


def test_the_plan_travels_as_a_call_id_not_as_tasks_from_the_browser():
    """The structural half of the same claim: `/swarm/build` accepts no tasks.

    The body names the call whose answer is being approved, and the console
    writes what it returned. There is no field a re-post could put a different
    decomposition in.
    """
    console, _, issues = console_with()
    plan_id = planned(console)

    values = dict(FORM)
    values["tasks"] = [{"id": "sneak", "goal": "something else", "files": ["x.py"]}]
    _, job = build(console, plan_id, values)

    assert job["state"] == "done", job.get("error")
    assert [c["number"] for c in issues.created] == [100, 101, 102]
    assert not any("sneak" in c["title"] for c in issues.created)


# --------------------------------------------------------------------------
# A task that cannot be written is reported, never dropped
# --------------------------------------------------------------------------


def test_a_task_normalise_rejects_is_reported_on_the_page():
    """Acceptance criterion 4, whose failing direction is silence.

    Six issues appearing from eight tasks reads, on a page that says nothing,
    as six tasks having been planned - and the two that vanished are the ones
    nobody will ever go looking for.
    """
    plan = {"reasoning": "", "tasks": [
        dict(PLAN["tasks"][0]),
        {"id": "no-files-at-all", "goal": "Tidy things up.", "files": [],
         "depends_on": [], "stack": "python"},
    ]}
    console, _, issues = console_with()

    _, job = build(console, planned(console, plan), PLAN_ONLY)

    assert job["state"] == "done", job.get("error")
    assert [c["number"] for c in issues.created] == [100]
    assert job["result"]["rejected"] == [
        {"task_id": "no-files-at-all", "reason": "[Files] lists no files"}
    ]


def test_a_duplicate_id_is_reported_with_normalises_own_reason():
    """Two issues carrying one id is control-plane corruption, and `normalise`
    refuses the second claimant. The operator is told which task and why."""
    plan = {"reasoning": "", "tasks": [
        dict(PLAN["tasks"][0]),
        {"id": "add-store", "goal": "A second thing claiming the same id.",
         "files": ["src/other.py"], "depends_on": [], "stack": "python"},
    ]}
    console, _, issues = console_with()

    _, job = build(console, planned(console, plan), PLAN_ONLY)

    assert len(issues.created) == 1
    assert job["result"]["rejected"] == [
        {"task_id": "add-store", "reason": "a second task claims this id"}
    ]


def test_a_plan_no_task_of_which_can_be_written_creates_nothing():
    """A repository whose backlog would be empty is not worth the repository."""
    plan = {"reasoning": "", "tasks": [
        {"id": "", "goal": "", "files": [], "depends_on": [], "stack": None},
    ]}
    console, provisioner, issues = console_with()

    _, job = build(console, planned(console, plan))

    assert job["state"] == "error"
    assert provisioner.calls == []
    assert issues.created == []
    assert job["error"]["fix"]


def test_a_dependency_ring_refuses_before_the_repository_exists():
    """`write_plan` would refuse this too - but only once there is a real
    repository with a URL and a ruleset for a human to go and delete."""
    plan = {"reasoning": "", "tasks": [
        {"id": "a", "goal": "A.", "files": ["a.py"], "depends_on": ["b"], "stack": "python"},
        {"id": "b", "goal": "B.", "files": ["b.py"], "depends_on": ["a"], "stack": "python"},
    ]}
    console, provisioner, issues = console_with()

    _, job = build(console, planned(console, plan))

    assert job["state"] == "error"
    assert "ring" in job["error"]["message"]
    assert provisioner.calls == []
    assert issues.created == []


# --------------------------------------------------------------------------
# Nothing is created when preflight fails
# --------------------------------------------------------------------------


def test_preflight_failures_reach_the_page_with_doctors_own_fixes():
    """Acceptance criterion 2. `Check.__post_init__` refuses to construct a
    failing check without a fix, so the console has no business inventing
    one - it forwards what doctor said, per failing check."""
    def refuses(_stacks) -> Diagnosis:
        return Diagnosis((
            Check.failed("docker.daemon", "connection refused",
                         fix="start Docker Desktop, then: docker version"),
            Check.failed("image.python", "not on this daemon",
                         fix="docker build -f Dockerfile.worker -t apiary-worker:python ."),
        ))

    console, provisioner, issues = console_with(preflight=refuses)

    _, job = build(console, planned(console))

    assert job["state"] == "error"
    assert provisioner.calls == []
    assert issues.created == []
    assert [c["name"] for c in job["error"]["checks"]] == ["docker.daemon", "image.python"]
    assert job["error"]["checks"][0]["fix"] == "start Docker Desktop, then: docker version"
    assert job["error"]["checks"][1]["fix"].startswith("docker build")
    assert job["error"]["fix"] == "start Docker Desktop, then: docker version"


def test_a_skipped_check_is_not_a_failure():
    """A daemon that could not be asked is doctor's own third state, and it
    does not refuse a build - `Diagnosis.ok` counts failures only."""
    def skips(_stacks) -> Diagnosis:
        return Diagnosis((Check.skipped("image.python", "no daemon to ask"),))

    console, provisioner, _ = console_with(preflight=skips)

    _, job = build(console, planned(console))

    assert job["state"] == "done", job.get("error")
    assert len(provisioner.calls) == 1


@pytest.mark.parametrize("missing, expected", [
    ("GITHUB_TOKEN", "GITHUB_TOKEN is not set"),
    ("APIARY_PROVISION_TOKEN", "APIARY_PROVISION_TOKEN, which is not set"),
])
def test_a_missing_token_is_console_runs_own_refusal(missing, expected):
    """Reused rather than restated. Two descriptions of one missing variable is
    how the fix goes stale on the copy nobody edited, so the build calls
    `console_runs.assert_tokens` and shows what it says."""
    env = {k: v for k, v in ENV.items() if k != missing}
    console, provisioner, issues = console_with(env=env)

    _, job = build(console, planned(console))

    assert job["state"] == "error"
    assert expected in job["error"]["message"]
    assert job["error"]["fix"]
    assert provisioner.calls == []
    assert issues.created == []


def test_the_token_check_lands_before_docker_is_even_asked():
    """Order matters only in that both are free; this pins which one speaks
    first, so an operator with neither is not sent to build an image before
    being told about the token they also lack."""
    asked: list[Any] = []
    console, _, _ = console_with(env={}, preflight=lambda s: asked.append(s) or healthy(s))

    _, job = build(console, planned(console))

    assert job["state"] == "error"
    assert asked == []


# --------------------------------------------------------------------------
# Single flight - the same latch, not a second one
# --------------------------------------------------------------------------


def test_a_second_start_building_is_refused_while_one_is_in_flight():
    """Two builds racing would create two repositories from one plan."""
    console, _, _ = console_with()
    plan_id = planned(console)
    console._running = "already-building"
    console.jobs["already-building"] = Job(id="already-building", site="build",
                                           started=time.monotonic())

    response = console.render(
        "POST", "/swarm/build", HOST,
        json.dumps({"plan": plan_id, "values": FORM}).encode(),
    )

    assert response.status == 409
    payload = json.loads(response.body)
    assert "a build is already in flight" in payload["error"]
    assert "second repository" in payload["error"]
    assert payload["fix"]


def test_a_model_call_is_refused_while_a_build_is_in_flight():
    """The ticket asked for the refusal to work "the same way", and the same
    way means the same latch: a build claims `_running` like an inference, so
    the exclusion holds in both directions rather than in one."""
    console, _, _ = console_with()
    console._running = "already-building"
    console.jobs["already-building"] = Job(id="already-building", site="build",
                                           started=time.monotonic())

    response = console.render(
        "POST", "/run", HOST,
        json.dumps({"site": "stack", "values": {"brief": "x"}}).encode(),
    )

    assert response.status == 409
    assert "a build is already in flight" in json.loads(response.body)["error"]


def test_a_build_is_refused_while_a_model_call_is_in_flight():
    console, _, _ = console_with()
    plan_id = planned(console)
    console._running = "thinking"
    console.jobs["thinking"] = Job(id="thinking", site="planner", started=time.monotonic())

    response = console.render(
        "POST", "/swarm/build", HOST,
        json.dumps({"plan": plan_id, "values": FORM}).encode(),
    )

    assert response.status == 409
    assert "Ollama loads one model at a time" in json.loads(response.body)["error"]


def test_the_latch_is_released_when_a_build_refuses():
    """A refusal that left `_running` set would take the console down with it -
    every later call answered 409 until a restart."""
    console, _, _ = console_with(env={})

    build(console, planned(console))

    assert console._running == ""


# --------------------------------------------------------------------------
# What may be built
# --------------------------------------------------------------------------


def test_building_a_call_that_is_not_a_plan_is_refused():
    console, _, _ = console_with()
    console.jobs["stackjob00000001"] = Job(id="stackjob00000001", site="stack",
                                           started=time.monotonic(), state="done",
                                           result={"stack": "python"})

    started, _ = build(console, "stackjob00000001")

    assert started.status == 400
    assert "not a plan" in json.loads(started.body)["error"]


def test_building_a_plan_that_is_still_running_is_refused():
    console, _, _ = console_with()
    console.jobs["planjob00000002"] = Job(id="planjob00000002", site="planner",
                                          started=time.monotonic())

    started, _ = build(console, "planjob00000002")

    assert started.status == 400
    assert json.loads(started.body)["fix"]


def test_building_an_unknown_call_says_so():
    console, _, _ = console_with()

    started, _ = build(console, "deadbeefdeadbeef")

    assert started.status == 404
    assert "no plan to build" in json.loads(started.body)["error"]


@pytest.mark.parametrize("bad", ["../../etc/passwd", "", "PLAN", "a" * 80])
def test_a_traversing_plan_id_is_refused(bad):
    """The id becomes a dictionary key here and a path nowhere - but it is the
    same alphabet `validate_capture_id` guards everywhere else, and an id that
    is not one has no business reaching a lookup."""
    console, _, _ = console_with()

    started, _ = build(console, bad)

    assert started.status == 400


@pytest.mark.parametrize("values, expected", [
    ({"owner": "", "stack": "python"}, "needs an owner"),
    ({"owner": "me/them", "stack": "python"}, "needs an owner"),
    ({"owner": "me", "stack": "haskell"}, "unknown stack"),
])
def test_the_form_is_refused_field_by_field_before_anything_is_created(values, expected):
    console, provisioner, _ = console_with()
    payload = dict(FORM)
    payload.update(values)

    _, job = build(console, planned(console), payload)

    assert job["state"] == "error"
    assert expected in job["error"]["message"]
    assert job["error"]["fix"]
    assert provisioner.calls == []


# --------------------------------------------------------------------------
# What the repository ends up being
# --------------------------------------------------------------------------


def test_every_issue_is_verified_by_the_command_in_the_commit():
    """`cli._target`'s rule, kept: the `## Verify` the issues carry is the
    provisioner's report, not the form's field. A `## Verify` disagreeing with
    the required status check is a task that was red before a worker existed."""
    console, provisioner, issues = console_with()
    values = dict(FORM, verify="python -m pytest -q")

    _, job = build(console, planned(console), values)

    assert provisioner.calls[0].verify_command == "python -m pytest -q"
    assert job["result"]["verify_command"] == "python -m pytest -q"
    for created in issues.created:
        assert "python -m pytest -q" in created["body"]


def test_a_blank_verify_field_leaves_the_placeholder_gate_in_place():
    """And does not reject every task for an empty `## Verify`, which is what
    `normalise` does when it is handed one."""
    console, provisioner, issues = console_with()

    _, job = build(console, planned(console), dict(FORM, verify=""))

    assert job["state"] == "done", job.get("error")
    assert provisioner.calls[0].verify_command == "test -f README.md"
    assert len(issues.created) == 3
    assert job["result"]["rejected"] == []


def test_the_repository_is_generated_from_the_objective_the_plan_came_from():
    console, provisioner, _ = console_with()

    build(console, planned(console))

    assert provisioner.calls[0].prompt == "a CLI that tracks expenses"
    assert provisioner.calls[0].name == "expense-tracker"
    assert provisioner.calls[0].stack == "python"


def test_a_blank_name_is_derived_from_the_objective():
    console, provisioner, _ = console_with()

    build(console, planned(console), dict(FORM, name=""))

    assert provisioner.calls[0].name == "a-cli-that-tracks-expenses"


def test_the_public_checkbox_decides_visibility():
    console, provisioner, _ = console_with()

    build(console, planned(console), dict(FORM, public=""))

    assert provisioner.calls[0].private is True


def test_the_built_repository_becomes_a_project_the_operator_can_return_to(tmp_path):
    """Recorded by the console, as a started run is, so the new repository
    reaches the selector without anyone retyping its name."""
    from swarm.console_projects import ProjectStore

    console, _, _ = console_with()
    console.projects = ProjectStore(path=tmp_path / "projects.sqlite")

    _, job = build(console, planned(console))

    assert job["state"] == "done", job.get("error")
    stored = console.projects.list()
    assert [p["repo"] for p in stored] == ["shahrestani-me/expense-tracker"]


def test_a_build_carries_no_capture_from_the_call_before_it(tmp_path):
    """`_work` attaches the recorder's last record to every job it finishes,
    which is right when the job was a model call. A build is not, and its
    recorder's last record is the planner call from several minutes ago -
    rendered under a build, that is the most misleading card this page could
    draw given what the ticket is about."""
    from swarm.capture import Recorder

    console, _, _ = console_with()
    console.sink = Recorder.for_console(tmp_path / "console")

    _, job = build(console, planned(console))

    assert job["capture"] is None


# --------------------------------------------------------------------------
# The pieces, directly
# --------------------------------------------------------------------------


def test_a_plan_is_rebuilt_from_the_payload_the_page_was_drawn_from():
    plan = plan_from_result(PLAN)

    assert [task.id for task in plan.tasks] == ["add-store", "add-cli"]
    assert plan.tasks[1].depends_on == ["add-store"]
    # The field nothing renders, carried anyway: a task written back under a
    # different stack than the model chose is a worker in the wrong image.
    assert plan.tasks[0].stack == "python"


@pytest.mark.parametrize("result", [None, "a plan", {}, {"tasks": []}, {"tasks": "add-store"}])
def test_something_that_is_not_a_plan_is_refused_with_a_fix(result):
    with pytest.raises(BuildError) as caught:
        plan_from_result(result)

    assert caught.value.fix


def test_the_form_requires_a_stack_because_there_is_no_model_to_ask():
    """The swarm tab's stack field is optional because `Bootstrap.for_prompt`
    asks `choose_stack` when it is blank. Here there is nothing to ask, and the
    answer is needed twice before anything exists - by `doctor.preflight` for
    the worker image, and by the generated CI workflow for its toolchain."""
    stack = next(f for f in BUILD_SITE["fields"] if f["name"] == "stack")

    assert stack["value"] == "python"
    assert "python" in stack["label"] and "node" in stack["label"]


def test_the_build_form_is_served_under_its_own_key():
    """Not a fourth entry in `sites`: nothing that iterates model-call sites
    may pick up a form with no prompt behind it."""
    console, _, _ = console_with()

    payload = json.loads(console.render("GET", "/sites", HOST).body)

    assert payload["build"]["key"] == "build"
    assert "build" not in [site["key"] for site in payload["sites"]]


def test_the_planner_payload_carries_every_field_the_rebuild_needs():
    """`_plan_run`'s task dictionaries are the only source the build has. A
    field dropped on the way to the screen is a field written back wrong."""
    from swarm.console import SITES
    from swarm.state import PlannedTask

    class Stub:
        reasoning = "because"
        tasks = [PlannedTask(id="a", goal="A.", files=["a.py"], depends_on=[], stack="node")]

    import swarm.nodes.planner as planner_mod

    original = planner_mod.draft_plan
    planner_mod.draft_plan = lambda *a, **k: Stub()
    try:
        result = SITES["planner"].run({"objective": "x"})
    finally:
        planner_mod.draft_plan = original

    assert result["tasks"][0] == {"id": "a", "goal": "A.", "files": ["a.py"],
                                  "depends_on": [], "stack": "node"}
    assert plan_from_result(result).tasks[0].stack == "node"


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------


def test_the_button_lives_on_the_plan_it_acts_on():
    from swarm.console import asset

    script = asset("app.js")

    assert "Start building" in script
    assert 'api("/swarm/build"' in script
    assert "function buildCard" in script
    assert ".buildform" in asset("app.css")


def test_the_report_and_its_refusals_still_reach_the_dom_as_text():
    """Every string on that card - a repository name, a task id, an issue title,
    a rejection reason - is model output or is derived from it. The one sink on
    this page that executes a string is `href`, and it is parsed rather than
    trusted."""
    from swarm.console import asset

    script = asset("app.js")

    assert "innerHTML" not in script
    assert 'parsed.hostname !== "github.com"' in script
    assert 'parsed.protocol !== "https:"' in script
    assert "(r.rejected || []).forEach" in script     # criterion 4, on the page
    assert "(e.checks || []).forEach" in script       # doctor's fixes, one per check


# --------------------------------------------------------------------------
# The real preflight
# --------------------------------------------------------------------------


@pytest.mark.docker
def test_the_real_preflight_answers_for_this_host():
    """The seam replaced everywhere above, run once for real: `doctor.preflight`
    is what the build actually calls, and a signature that drifted would be
    invisible to every test that hands in a double."""
    from swarm import doctor

    diagnosis = doctor.preflight(["python"])

    assert diagnosis.checks
    for check in diagnosis.failures:
        assert check.fix, f"{check.name} failed without naming a fix"


def test_a_repository_that_was_created_is_named_even_when_its_issues_are_not():
    """The one failure that must not arrive as a traceback.

    By the time `write_plan` runs the repository is real, and a refusal that
    did not name it would leave the operator with something they cannot find
    and did not ask for - while pressing the button again would create a
    second one.
    """
    class Refuses:
        repo = "shahrestani-me/expense-tracker"

        def list_issues(self, **_):
            raise RuntimeError("403 Resource not accessible by personal access token")

    console, provisioner, _ = console_with(client_for=lambda repo: Refuses())

    _, job = build(console, planned(console))

    assert job["state"] == "error"
    assert len(provisioner.calls) == 1
    assert "https://github.com/shahrestani-me/expense-tracker" in job["error"]["message"]
    assert "issues could not be written" in job["error"]["message"].replace(
        "its issues", "issues")
    assert "issues: write" in job["error"]["fix"]
    # Not a traceback dump: the fix is the point, and it says what to do with
    # the repository that now exists rather than "try again".
    assert "delete it" in job["error"]["fix"]


# --------------------------------------------------------------------------
# The scaffold
# --------------------------------------------------------------------------


def test_the_scaffold_leads_and_every_task_blocks_on_it():
    """#101's project-scaffold issue, and the reason it is not optional in
    spirit: every task the model planned edits files that do not exist yet, so
    without it the first cycle dispatches all of them against an empty
    repository, each worker inventing its own idea of the project."""
    console, _, issues = console_with()

    _, job = build(console, planned(console))

    scaffold, *theirs = issues.created
    assert "id=bootstrap-the-project" in scaffold["body"]
    assert "swarm:ready" in scaffold["labels"]          # nothing blocks it
    # The section is always rendered; what matters is that it names no issue.
    assert "_none._" in scaffold["body"]
    for created in theirs:
        assert "swarm:blocked" in created["labels"]
        assert f"#{scaffold['number']}" in created["body"]


def test_unticking_the_scaffold_writes_the_decomposition_and_nothing_else():
    """The escape hatch, for a plan that already contains its own setup task."""
    console, _, issues = console_with()

    _, job = build(console, planned(console), PLAN_ONLY)

    assert [c["number"] for c in issues.created] == [100, 101]
    assert not any("bootstrap-the-project" in c["body"] for c in issues.created)
    assert [i["task_id"] for i in job["result"]["issues"]] == ["add-store", "add-cli"]


def test_a_caller_that_never_heard_of_the_field_still_gets_a_scaffold():
    """Absence means on here, and off two fields up on `public` - deliberately.

    An unticked `public` is the conservative direction: a private repository.
    An unticked scaffold is the broken one. So the field is the operator opting
    *out*, and anything that does not mention it gets what every other
    greenfield repository in this system gets.
    """
    console, _, issues = console_with()
    values = {k: v for k, v in FORM.items()}
    assert "bootstrap" not in values

    build(console, planned(console), values)

    assert "id=bootstrap-the-project" in issues.created[0]["body"]


def test_the_scaffold_cannot_stand_in_for_a_plan_that_is_entirely_unwritable():
    """A scaffold is always writable, so counting it toward "is there any work
    here" would let a plan whose every real task was rejected provision a
    repository containing nothing but its own setup issue - the empty backlog
    this guard refuses, wearing a disguise."""
    plan = {"reasoning": "", "tasks": [
        {"id": "", "goal": "", "files": [], "depends_on": [], "stack": None},
    ]}
    console, provisioner, issues = console_with()

    _, job = build(console, planned(console, plan))

    assert job["state"] == "error"
    assert provisioner.calls == []
    assert issues.created == []


def test_the_scaffold_is_the_only_issue_written_that_was_not_on_the_screen():
    """Criterion 3's boundary, stated as a test: the operator's tasks arrive
    unaltered except for the one dependency the label warned them about, and
    nothing else is added."""
    console, _, issues = console_with()

    _, job = build(console, planned(console))

    ids = [i["task_id"] for i in job["result"]["issues"]]
    assert set(ids) - {t["id"] for t in PLAN["tasks"]} == {"bootstrap-the-project"}


def test_the_checkbox_says_what_it_writes_and_what_it_changes():
    """The label is the disclosure - it is what makes the addition something
    the operator read rather than something done to them."""
    field = next(f for f in BUILD_SITE["fields"] if f["name"] == "bootstrap")

    assert field["value"] == "1"                       # on by default
    assert "blocked on it" in field["label"]           # what it changes
    assert "empty repository" in field["label"]        # why it is on
    assert "project scaffold" in BUILD_SITE["blurb"]   # and the blurb agrees


# --------------------------------------------------------------------------
# ...and then the swarm runs on it (#130)
# --------------------------------------------------------------------------


def test_a_finished_build_starts_the_swarm_on_the_repository_it_made():
    """#130's first criterion: one press, and cycles start.

    The command is asserted rather than the fact that something was spawned,
    because *which* command it is carries the whole ticket. `swarm run --repo`
    attaches to the repository that now exists and dispatches its backlog;
    `swarm run --new` would provision a second repository and ask the model to
    plan the objective again, which is the path #129 exists to have escaped.
    """
    proc = FakeProc()
    runs = swarm_runs(proc)
    console, _, _ = console_with(runs=runs)

    _, job = build(console, planned(console))

    assert job["state"] == "done", job.get("error")
    assert runs.spawn.argv[2:] == [                     # type: ignore[attr-defined]
        "-m", "swarm.cli", "run",
        "--repo", "shahrestani-me/expense-tracker",
        "--objective", "a CLI that tracks expenses",
        # The placeholder gate the provisioner committed, because the form left
        # the field blank - the same command every `## Verify` was written with.
        "--verify", "test -f README.md",
        "--stack", "python",
    ]
    # And the page is handed the run to follow, with the id `/swarm/status`
    # and `/swarm/stop` both take.
    assert job["result"]["run"]["id"] in console.runs.jobs
    assert job["result"]["run"]["state"] == "running"


def test_the_run_verifies_with_the_command_in_the_commit_not_the_form():
    """`Builder` already writes `report.verify_command` into every `## Verify`
    - the gate in the commit that now exists, which is not always the form's.
    A run told a different one would dispatch tasks that were red before a
    worker touched them."""
    proc = FakeProc()
    runs = swarm_runs(proc)
    console, provisioner, _ = console_with(runs=runs)
    provisioner.calls = []

    _, job = build(console, planned(console), dict(FORM, verify="pytest -x"))

    argv = runs.spawn.argv                              # type: ignore[attr-defined]
    assert argv[argv.index("--verify") + 1] == job["result"]["verify_command"] == "pytest -x"


def test_the_cap_and_the_merge_policy_travel_from_the_form_to_the_run():
    """The two fields that belong to the loop rather than to the repository.
    Without the cap there is no ending but "met" and "stopped", and the
    ticket's second ending is the cap."""
    proc = FakeProc()
    runs = swarm_runs(proc)
    console, _, _ = console_with(runs=runs)

    build(console, planned(console), dict(FORM, max_cycles="3", auto_merge=""))

    argv = runs.spawn.argv                              # type: ignore[attr-defined]
    assert argv[argv.index("--max-cycles") + 1] == "3"
    assert runs.spawn.env["APIARY_MERGE_ADMIN_OVERRIDE"] == "0"   # type: ignore[attr-defined]
    names = [f["name"] for f in BUILD_SITE["fields"]]
    assert "max_cycles" in names and "auto_merge" in names


def test_a_second_build_is_refused_while_the_swarm_it_started_is_live():
    """The refusal that has to arrive *before* a repository exists.

    `SwarmRuns.start` refuses a second run on its own - but only after this
    build has provisioned a repository and written a backlog into it, leaving
    both for a human to go and delete. So the gate is asked before anything is
    created, and it names the repository being built rather than saying that
    something, somewhere, is busy.
    """
    proc = FakeProc()
    console, provisioner, _ = console_with(runs=swarm_runs(proc))

    _, first = build(console, planned(console))
    assert first["state"] == "done", first.get("error")

    refused, _ = build(console, planned(console))       # the run is still live

    assert refused.status == 409
    body = json.loads(refused.body)
    assert "shahrestani-me/expense-tracker" in body["error"]
    assert "shahrestani-me/expense-tracker" in body["fix"]
    assert len(provisioner.calls) == 1                  # nothing was created twice

    # ...and once the run ends, the button works again.
    proc.finish(0)
    settle(console.runs.jobs[first["result"]["run"]["id"]])
    again, second = build(console, planned(console))
    assert again.status == 202 and second["state"] == "done", second.get("error")


def test_a_run_that_will_not_start_does_not_turn_a_finished_build_into_a_failure():
    """The repository and its issues are real by the time the loop is asked for.

    A build that reported itself failed because Docker was down would be
    describing the wrong thing, and would send the operator looking for a
    repository that is sitting there with its backlog written. So the report
    stands, and the refusal rides on it with the command that picks the work up.
    """
    def refuses(argv, **kwargs):
        raise OSError("no such file or directory: python")

    console, _, issues = console_with(runs=SwarmRuns(spawn=refuses, exists=lambda r: True))

    _, job = build(console, planned(console))

    assert job["state"] == "done"
    assert len(job["result"]["issues"]) == 3           # scaffold + the two tasks
    assert "run" not in job["result"]
    assert "could not start the run" in job["result"]["run_error"]["error"]
    assert "swarm run --repo shahrestani-me/expense-tracker" in job["result"]["run_error"]["fix"]


def test_a_build_that_refuses_starts_no_run_at_all():
    """Nothing created, nothing dispatched. The `_start_run` call sits after
    `Builder.run`, and a refusal from it must not reach the swarm."""
    console, provisioner, _ = console_with(
        runs=SwarmRuns(spawn=_never_spawn, exists=lambda r: True),
        preflight=lambda stacks: Diagnosis((
            Check.failed("image.python", "not built", fix="`swarm images build`"),)),
    )

    _, job = build(console, planned(console))

    assert job["state"] == "error"
    assert provisioner.calls == []


def _never_spawn(argv, **kwargs):
    raise AssertionError(f"a refused build spawned a run: {argv}")
