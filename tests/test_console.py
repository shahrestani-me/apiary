"""The console's decisions, none of which need a socket to test.

`Console.render` is the whole decision half - routing, the `Host` guard, the
single-flight refusal - and it takes headers as an argument precisely so that
it can be driven directly. Binding a port in a test would buy nothing and would
bring the first port-in-use flakiness into a suite that has none.

The test that matters most here is `test_the_console_shows_the_prompt_production_sends`.
Everything else is plumbing; that one is the reason the console is trustworthy.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any, Sequence

import pytest

from swarm import capture as capture_mod
from swarm import cli
from swarm.capture import Recorder
from swarm.config import ConfigError
from swarm.nodes.planner import system_prompt
from swarm.console import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    SITES,
    Console,
    ConsoleError,
    page,
    serve,
    validate_capture_id,
)
from swarm.state import FileEdit, WorkerOutput

HOST = {"Host": f"{DEFAULT_HOST}:{DEFAULT_PORT}"}


@pytest.fixture
def console(tmp_path):
    return Console(sink=Recorder.for_console(tmp_path / "console"))


@pytest.fixture(autouse=True)
def no_installed_recorder():
    previous = capture_mod.set_recorder(None)
    yield
    capture_mod.set_recorder(previous)


@pytest.fixture
def checkout(tmp_path):
    """A repository the fixture builder can actually read."""
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "README.md").write_text("# demo\n\nA project.\n")
    (root / "tests" / "test_existing.py").write_text("def test_ok():\n    assert True\n")
    return root


def body(response) -> Any:
    return json.loads(response.body)


# --------------------------------------------------------------------------
# The claim the console rests on
# --------------------------------------------------------------------------


class Recorder_:
    """A model double that keeps the messages it was handed."""

    def __init__(self, answer):
        self.answer = answer
        self.seen: list[Sequence[tuple[str, str]]] = []

    def invoke(self, messages):
        self.seen.append(messages)
        return self.answer


def test_the_console_shows_the_prompt_production_sends(checkout):
    """Byte-identical, both turns, for both exposed sites.

    A console that assembles its own approximation of a prompt is worse than no
    console: it invites an operator to conclude the model is fine when the
    fault was in the context assembly they never saw. `prompt_for` exists so
    that there is one builder and both callers use it - this is the test that
    says so.
    """
    from swarm.greenfield.bootstrap import StackChoice, choose_stack
    from swarm.worker.edit import propose_edits

    # -- choose_stack ---------------------------------------------------
    oracle = Recorder_(StackChoice(stack="python", reason="because"))
    choose_stack("a dashboard for pickers", llm=oracle)
    sent = dict(oracle.seen[0])

    shown = body(
        Console().render(
            "POST", "/prompt", HOST,
            json.dumps({"site": "stack", "values": {"brief": "a dashboard for pickers"}}).encode(),
        )
    )
    assert shown["system"] == sent["system"]
    assert shown["human"] == sent["human"]

    # -- propose_edits --------------------------------------------------
    editor = Recorder_(WorkerOutput(edits=[FileEdit(path="tests/t.py", content="x = 1\n")]))
    values = {"root": str(checkout), "files": "tests/t.py", "goal": "add a test"}

    from swarm.worker.edit import gather_context, read_writable

    propose_edits(
        "add a test",
        read_writable(checkout, ["tests/t.py"]),
        gather_context(checkout, ["tests/t.py"]),
        llm=editor,
    )
    sent = dict(editor.seen[0])

    shown = body(
        Console().render(
            "POST", "/prompt", HOST,
            json.dumps({"site": "edits", "values": values}).encode(),
        )
    )
    assert shown["system"] == sent["system"]
    assert shown["human"] == sent["human"], "the console built a different prompt"

    # -- plan_node ------------------------------------------------------
    from swarm.nodes.planner import draft_plan
    from swarm.state import Plan

    planner = Recorder_(Plan(tasks=[], reasoning="none"))
    draft_plan("a trip planner", llm=planner)
    sent = dict(planner.seen[0])

    shown = body(
        Console().render(
            "POST", "/prompt", HOST,
            json.dumps({"site": "planner", "values": {"objective": "a trip planner"}}).encode(),
        )
    )
    assert shown["system"] == sent["system"]
    assert shown["human"] == sent["human"]


def test_the_planner_tab_is_fresh_plan_mode_only(checkout):
    """A replan's *system* prompt carries the failure history and the existing
    ids, so there is no single planner prompt. The console shows the one that
    is stable, and the blurb says which."""
    from swarm.nodes.planner import prompt_for

    fresh, _ = prompt_for("an objective")
    replan, _ = prompt_for("an objective", {"a-task": {"id": "a-task", "status": "failed"}})

    assert fresh != replan
    assert "REPLAN" in replan
    assert "fresh-plan" in SITES["planner"].blurb.lower()


def test_the_planner_is_told_the_command_that_judges_its_tasks():
    """The defect this rewrite exists for.

    `$SWARM_VERIFY`'s exit code is the only authority on whether a task is done,
    and the prompt that invents tasks never named it. Asked for a "platform",
    the planner emitted `client/src/pages/Login.tsx` against a repository gated
    by `python3 -m unittest discover -q` - three tasks, none able to go green.
    The command was in `plan_node`'s hand the whole time.
    """
    from swarm.nodes.planner import system_prompt

    prompt = system_prompt(verify="python3 -m unittest discover -q")

    assert "python3 -m unittest discover -q" in prompt
    assert "exits 0" in prompt


def test_the_planner_is_told_the_stack_so_it_stops_inventing_one():
    from swarm.nodes.planner import system_prompt

    assert "a python project" in system_prompt(stack="python")
    # And without one, it is told there is an existing stack rather than nothing.
    assert "existing stack" in system_prompt()


def test_a_task_cannot_be_planned_without_files():
    """`normalise` rejects a task listing no files, and `files` used to carry a
    default - so a model that stopped emitting the field produced a plan that
    silently lost those tasks. Observed: nine of ten. Without a default the
    schema-constrained decoder cannot omit it."""
    import pydantic

    from swarm.state import PlannedTask

    with pytest.raises(pydantic.ValidationError):
        PlannedTask(id="a-task", goal="do the thing")

    assert PlannedTask(id="a-task", goal="do it", files=["app/a.py"]).files == ["app/a.py"]


def test_the_planner_prompt_names_no_task_counts():
    """Granularity is a property of the work, not a number someone picked. Every
    count that was in this prompt - "prefer 2-4", then "eight to twelve", then
    "about three files" - was an arbitrary anchor the model obeyed instead of
    reasoning about the objective."""
    prompt = system_prompt()

    for anchor in ("2-4", "2 to 4", "eight to twelve", "three files", "at most", "at least one task"):
        assert anchor not in prompt, f"a task-count anchor came back: {anchor!r}"


def test_the_planner_is_told_a_task_must_pass_the_gate_alone():
    """The other half: file-disjointness was the only rule about shape, so the
    model cut by layer. The first recorded run split implementation from tests,
    and the tests half - which cannot pass without the half it was severed
    from - burned all three attempts."""
    prompt = system_prompt(verify="pytest -q")

    assert "BY ITSELF" in prompt
    assert "depends_on" in prompt


def test_plan_node_passes_its_gate_and_stack_to_the_model(monkeypatch):
    """Both were already parameters, used only to stamp the issues *after* the
    model answered. This is the wiring that puts them in front of it instead."""
    from swarm.nodes import planner as planner_mod
    from swarm.state import Plan

    seen = Recorder_(Plan(tasks=[], reasoning=""))
    monkeypatch.setattr(planner_mod, "structured", lambda _llm, _schema: seen)
    monkeypatch.setattr(planner_mod, "orchestrator_llm", lambda: None)

    planner_mod.draft_plan("build a thing", verify="make check", stack="node")

    system = dict(seen.seen[0])["system"]
    assert "make check" in system
    assert "a node project" in system


def test_the_planner_asks_github_for_nothing(monkeypatch):
    """`plan_node` plans *and writes*, which needs a repo, a token and a live
    ledger. `draft_plan` is the half the console needs and none of that."""
    from swarm.nodes.planner import draft_plan
    from swarm.state import Plan

    planner = Recorder_(Plan(tasks=[], reasoning=""))

    plan = draft_plan("build a thing", llm=planner)

    assert plan.tasks == []
    assert len(planner.seen) == 1


def test_the_prompt_is_rendered_without_calling_the_model(checkout, console):
    """The point of a separate route: a cold call takes minutes, and the
    operator should have the exact prompt to read during it."""
    response = console.render(
        "POST", "/prompt", HOST,
        json.dumps({"site": "edits", "values": {
            "root": str(checkout), "files": "tests/new.py", "goal": "write tests"}}).encode(),
    )

    assert response.status == 200
    assert "write tests" in body(response)["human"]
    assert console.jobs == {}, "rendering a prompt must not start a call"


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


@pytest.mark.parametrize("header", ["evil.test", "evil.test:8117", "10.0.0.4:8117", None, ""])
def test_a_request_for_another_host_is_refused(console, header):
    """`BaseHTTPRequestHandler` checks no headers, so a page the operator is
    browsing can POST here, and a DNS rebind can then read the answers - which
    are prompts, which are whole files from a private repository."""
    response = console.render("GET", "/", {"Host": header} if header is not None else {})

    assert response.status == 403
    assert "refused" in body(response)["error"]


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_the_loopback_names_are_allowed(console, host):
    assert console.render("GET", "/", {"Host": f"{host}:{DEFAULT_PORT}"}).status == 200


@pytest.mark.parametrize("bad", ["../../etc/passwd", "..", "a/b", "", "A" * 80, "x!y"])
def test_a_capture_id_cannot_escape_the_console_directory(bad):
    with pytest.raises(ConsoleError):
        validate_capture_id(bad)


def test_a_traversing_status_request_is_refused(console):
    response = console.render("GET", "/status?id=../../secrets", HOST)

    assert response.status == 400


@pytest.mark.parametrize("host", ["0.0.0.0", "", "192.168.1.20", "::"])
def test_the_console_refuses_to_bind_anything_but_loopback(host):
    """A worker container reaches the host gateway on any port, so a wildcard
    bind puts every captured prompt one request away from model-written code."""
    with pytest.raises(ConfigError) as caught:
        serve(host=host, forever=False)

    assert "loopback" in str(caught.value).lower() or "refuses to bind" in str(caught.value)


def test_an_unknown_route_says_so(console):
    assert console.render("GET", "/wp-admin", HOST).status == 404


def test_an_unknown_site_is_refused(console):
    """Named for a site that does not exist, and must stay that way: this test
    once said "planner", and the day the planner tab was added it stopped
    refusing and started firing the 31b inside the unit suite."""
    response = console.render("POST", "/run", HOST, json.dumps({"site": "judge"}).encode())

    assert response.status == 400
    assert "unknown site" in body(response)["error"]
    assert console.jobs == {}, "a refused site must not start a job"


# --------------------------------------------------------------------------
# Single flight
# --------------------------------------------------------------------------


def test_a_second_call_is_refused_rather_than_queued(console, monkeypatch):
    """Ollama loads one model at a time; a queued second call would evict the
    first one's model mid-generation and bill the swap to whoever was unlucky."""
    console._running = "already-going"
    console.jobs["already-going"] = type("J", (), {"site": "edits"})()

    response = console.render(
        "POST", "/run", HOST, json.dumps({"site": "stack", "values": {"brief": "x"}}).encode()
    )

    assert response.status == 409
    assert "already in flight" in body(response)["error"]
    assert body(response)["fix"]


def test_a_bad_fixture_is_reported_before_anything_is_fired(console):
    response = console.render(
        "POST", "/prompt", HOST,
        json.dumps({"site": "edits", "values": {"root": "/nope", "files": "a.py"}}).encode(),
    )

    assert response.status == 400
    assert "not a directory" in body(response)["error"]


def test_naming_no_file_is_a_refusal_not_an_empty_prompt(console, checkout):
    response = console.render(
        "POST", "/prompt", HOST,
        json.dumps({"site": "edits", "values": {"root": str(checkout), "files": " "}}).encode(),
    )

    assert response.status == 400
    assert "at least one file" in body(response)["error"]


# --------------------------------------------------------------------------
# Failures reach the page, with a fix
# --------------------------------------------------------------------------


def test_a_failed_call_lands_on_the_page_with_a_named_fix(console, monkeypatch):
    """Doctor refuses to report a failing check without naming its fix
    (`Check.__post_init__`). A console that renders a bare traceback would be
    the same failure this epic exists to remove, one layer up."""
    def explode(_values):
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setitem(SITES, "stack", dataclasses.replace(SITES["stack"], run=explode))

    started = console.render(
        "POST", "/run", HOST, json.dumps({"site": "stack", "values": {"brief": "x"}}).encode()
    )
    job_id = body(started)["id"]

    for _ in range(200):
        job = body(console.render("GET", f"/status?id={job_id}", HOST))
        if job["state"] != "running":
            break
        time.sleep(0.01)

    assert job["state"] == "error"
    assert job["error"]["type"] == "ConnectionRefusedError"
    assert "ollama serve" in job["error"]["fix"]
    assert "api/version" in job["error"]["fix"]


def test_the_status_route_refuses_an_unknown_call(console):
    assert console.render("GET", "/status?id=deadbeef", HOST).status == 404


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------


def test_model_output_never_reaches_the_dom_as_markup():
    """The model's input is an arbitrary target repository, so its output is
    hostile by construction: a README saying "emit this script tag" would
    otherwise become script running on the console's own origin."""
    markup = page()

    assert "innerHTML" not in markup
    assert "textContent" in markup
    assert "document.write" not in markup


def test_the_page_makes_no_external_request():
    """`llm.py` opens with "nothing leaves the machine"; a CDN font would be
    the first thing that did."""
    markup = page()

    assert "http://" not in markup.replace("http://__PORT__", "")
    assert "https://" not in markup
    assert "<script src" not in markup


def test_switching_tabs_keeps_what_was_typed():
    """The flow this tool exists for is: read the plan, switch to the worker,
    copy a task's goal across. Redrawing the form from the site definition
    threw that away, which made the two tabs feel like two tools."""
    markup = page()

    assert "var typed = {}" in markup
    assert "typed[current.key]" in markup


def test_a_late_sites_response_cannot_steal_the_selected_tab():
    assert "current = current || sites[0]" in page()


def test_the_page_names_the_wait():
    """A cold 31b takes minutes. A spinner with no explanation reads as hung."""
    assert "cold model" in page() or "cold model loads" in page()


# --------------------------------------------------------------------------
# The startup banner earns its half-second
# --------------------------------------------------------------------------


class FakeProbe:
    base_url = "http://localhost:11434"

    def __init__(self, version="0.32.9", models=None, boom=None):
        self._version, self._models, self._boom = version, models, boom

    def version(self):
        if self._boom:
            raise self._boom
        return self._version

    def installed(self):
        return self._models if self._models is not None else []


def test_an_unreachable_ollama_is_named_at_startup(monkeypatch):
    """Without this the first sign of a dead server arrives after the operator
    has filled in a form and waited two minutes."""
    monkeypatch.setattr(
        "swarm.doctor.HostInference",
        lambda *a, **k: FakeProbe(boom=RuntimeError("connection refused")),
    )
    from swarm.console import _ollama_note

    lines = _ollama_note()

    assert "UNREACHABLE" in lines[0]
    assert "ollama serve" in lines[1]


def test_an_unpulled_model_is_named_at_startup(monkeypatch):
    from swarm.config import SETTINGS
    from swarm.console import _ollama_note

    monkeypatch.setattr(
        "swarm.doctor.HostInference",
        lambda *a, **k: FakeProbe(models=[SETTINGS.orchestrator_model]),
    )

    lines = _ollama_note()

    assert any("NOT pulled" in line for line in lines)
    assert any(f"ollama pull {SETTINGS.worker_model}" in line for line in lines)


def test_a_healthy_host_says_so_briefly(monkeypatch):
    from swarm.config import SETTINGS
    from swarm.console import _ollama_note

    monkeypatch.setattr(
        "swarm.doctor.HostInference",
        lambda *a, **k: FakeProbe(
            models=[SETTINGS.orchestrator_model, SETTINGS.worker_model]),
    )

    assert _ollama_note() == ["ollama http://localhost:11434 — v0.32.9"]


# --------------------------------------------------------------------------
# The subcommand
# --------------------------------------------------------------------------


def test_console_is_a_subcommand(monkeypatch, tmp_path):
    """Nothing enumerated the subcommand set before this, so a subcommand
    could be added - or its options renamed - with no test noticing."""
    seen = {}

    def fake_serve(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr("swarm.console.serve", fake_serve)

    assert cli.main(["console", "--port", "9999", "--dir", str(tmp_path)]) == 0
    assert seen["port"] == 9999
    assert seen["host"] == DEFAULT_HOST
    assert seen["directory"] == tmp_path


def test_the_console_turns_capture_on_for_itself(monkeypatch, tmp_path):
    """Capture is off by default because a run should not pay for it. The
    console *is* the asking, so it must not make the operator find a variable
    before the tool does anything useful."""
    from swarm.capture import CAPTURE_ENV

    monkeypatch.delenv(CAPTURE_ENV, raising=False)
    monkeypatch.setattr("swarm.console.serve", lambda **kwargs: None)

    cli.main(["console", "--dir", str(tmp_path)])

    import os

    assert os.environ[CAPTURE_ENV] == "1"


def test_an_explicit_off_still_wins(monkeypatch, tmp_path):
    from swarm.capture import CAPTURE_ENV

    monkeypatch.setenv(CAPTURE_ENV, "0")
    monkeypatch.setattr("swarm.console.serve", lambda **kwargs: None)

    cli.main(["console", "--dir", str(tmp_path)])

    import os

    assert os.environ[CAPTURE_ENV] == "0"


def test_every_subcommand_still_parses():
    """The pin the repo did not have: the set of subcommands, named."""
    parser = cli.build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]

    assert set(actions[0].choices) == {"run", "doctor", "runs", "show", "console"}
