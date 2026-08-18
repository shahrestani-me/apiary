"""The intake site: three plain questions in, a run-ready proposal out.

The tests that matter most are the founding-rule one - the prompt the console
shows for the intake tab is byte-identical to the prompt `propose_setup`
sends - and the fallback pair: a dead model must degrade into a proposal the
user can read and change, never into a traceback. Everything downstream of the
model is deterministic, so the rest is pinned as data: the brief's labelled
sections, the owner-resolution order, and the contract keys the front end
copies into the run form.

No model call and no network anywhere in here: model doubles are objects with
an `invoke`, and `urllib.request.urlopen` is stubbed wherever `resolve_owner`
could otherwise reach for it.
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from swarm import capture as capture_mod
from swarm import console_intake as intake
from swarm.capture import Recorder
from swarm.console import DEFAULT_HOST, DEFAULT_PORT, SITES, Console
from swarm.console_intake import (
    IntakeError,
    Setup,
    compose_brief,
    propose,
    propose_setup,
    prompt_for,
    resolve_owner,
)
from swarm.github.ledger import DEFAULT_STACK, KNOWN_STACKS
from swarm.greenfield.bootstrap import STACK_VERIFY

HOST = {"Host": f"{DEFAULT_HOST}:{DEFAULT_PORT}"}

ANSWERS = {
    "need": "Keep track of which meeting rooms are free right now.",
    "users": "Office staff, who need to see and book a room.",
    "done": "Book a room, then see it marked as taken.",
}


@pytest.fixture
def console(tmp_path):
    return Console(sink=Recorder.for_console(tmp_path / "console"))


@pytest.fixture(autouse=True)
def no_installed_recorder():
    previous = capture_mod.set_recorder(None)
    yield
    capture_mod.set_recorder(previous)


@pytest.fixture(autouse=True)
def no_owner_in_the_environment(monkeypatch):
    """Whatever the developer's shell exports must not decide a test."""
    for name in ("APIARY_OWNER", "GITHUB_REPOSITORY", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any test that reaches the real `/user` endpoint has already failed."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("a test tried to open a real network connection")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def body(response) -> Any:
    return json.loads(response.body)


class Oracle:
    """A model double that keeps the messages it was handed."""

    def __init__(self, answer):
        self.answer = answer
        self.seen: list[Sequence[tuple[str, str]]] = []

    def invoke(self, messages):
        self.seen.append(messages)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


# --------------------------------------------------------------------------
# The brief
# --------------------------------------------------------------------------


def test_the_brief_is_labelled_paragraphs_in_the_questions_order():
    brief = compose_brief(ANSWERS)

    sections = brief.split("\n\n")
    assert sections == [
        "The need: Keep track of which meeting rooms are free right now.",
        "Who will use it: Office staff, who need to see and book a room.",
        "How we will know it works: Book a room, then see it marked as taken.",
    ]


def test_an_empty_answer_is_skipped_not_rendered_as_an_empty_heading():
    brief = compose_brief({"need": "count the beans", "users": "  ", "done": ""})

    assert brief == "The need: count the beans"
    assert "Who will use it" not in brief


def test_a_blank_need_is_refused_with_the_fix_attached():
    """The one required answer. A brief with no need in it would plan nothing,
    two minutes and one model call later."""
    with pytest.raises(IntakeError) as caught:
        compose_brief({"need": "   ", "users": "everyone"})

    assert "cannot be skipped" in str(caught.value)
    assert "first answer" in caught.value.fix


def test_a_blank_need_is_a_400_on_the_prompt_route_not_a_fired_call(console):
    response = console.render(
        "POST", "/prompt", HOST,
        json.dumps({"site": "intake", "values": {"need": ""}}).encode(),
    )

    assert response.status == 400
    assert "cannot be skipped" in body(response)["error"]
    assert console.jobs == {}


# --------------------------------------------------------------------------
# The one model call
# --------------------------------------------------------------------------


def test_the_console_shows_the_prompt_propose_setup_sends(console):
    """The founding rule, applied to the fourth site: byte-identical, both
    turns, between what `/prompt` renders and what `propose_setup` fires."""
    oracle = Oracle(Setup(name="room-tracker", stack="react", reason="a page"))
    propose_setup(compose_brief(ANSWERS), llm=oracle)
    sent = dict(oracle.seen[0])

    shown = body(
        console.render(
            "POST", "/prompt", HOST,
            json.dumps({"site": "intake", "values": ANSWERS}).encode(),
        )
    )

    assert shown["system"] == sent["system"]
    assert shown["human"] == sent["human"], "the console built a different prompt"


def test_the_human_turn_is_the_brief_unchanged():
    system, human = prompt_for("The need: a bean counter")

    assert human == "The need: a bean counter"
    assert "python, node, react" in system


def test_a_clean_answer_becomes_the_setup():
    oracle = Oracle(Setup(name="room-tracker", stack="react", reason="a page people look at"))

    setup = propose_setup(compose_brief(ANSWERS), llm=oracle)

    assert setup.name == "room-tracker"
    assert setup.stack == "react"
    assert setup.reason == "a page people look at"
    assert len(oracle.seen) == 1


def test_a_model_name_with_spaces_in_it_is_slugified_not_trusted():
    """The schema description asks for kebab-case; a description is advice,
    and a repository name is not."""
    oracle = Oracle(Setup(name="Room Booker!", stack="python", reason=""))

    setup = propose_setup("The need: book rooms", llm=oracle)

    assert setup.name == "room-booker"


def test_a_dead_model_degrades_into_a_proposal_not_a_traceback(capsys):
    """`choose_stack`'s philosophy, whole: name from the brief's first line,
    the default stack, and a reason that says it is a fallback - because the
    silent half of a fallback is the part that gets people misdiagnosing."""
    oracle = Oracle(ConnectionRefusedError("connection refused"))

    setup = propose_setup(compose_brief(ANSWERS), llm=oracle)

    assert setup.name.startswith("keep-track-of-which-meeting-rooms")
    assert setup.stack == DEFAULT_STACK
    assert "fallback" in setup.reason
    assert "ConnectionRefusedError" in setup.reason
    assert "fell back" in capsys.readouterr().err


def test_an_answer_outside_the_stack_vocabulary_falls_back_too(capsys):
    oracle = Oracle(SimpleNamespace(name="bean-counter", stack="rust", reason="fast"))

    setup = propose_setup("The need: count beans", llm=oracle)

    assert setup.stack == DEFAULT_STACK
    assert setup.name == "bean-counter"
    assert "rust" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The owner
# --------------------------------------------------------------------------


def test_an_explicit_apiary_owner_beats_everything(monkeypatch):
    monkeypatch.setenv("APIARY_OWNER", "acme")
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone-else/whatever")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")

    assert resolve_owner() == "acme"


def test_the_owner_half_of_github_repository_is_second(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/some-repo")

    assert resolve_owner() == "acme"


def test_the_user_endpoint_is_last_and_read_with_the_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    seen = {}

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        seen["timeout"] = timeout
        return FakeResponse(json.dumps({"login": "octocat"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert resolve_owner() == "octocat"
    assert seen["url"] == "https://api.github.com/user"
    assert seen["auth"] == "Bearer ghp_x"
    assert seen["timeout"] == 10


def test_a_503_from_the_user_endpoint_falls_through_to_the_refusal(monkeypatch):
    """Fine-grained PATs 503 on `/user`, so any failure there must become the
    same refusal a missing token does - never a network traceback."""
    monkeypatch.setenv("GITHUB_TOKEN", "github_pat_x")

    def fail(*_args, **_kwargs):
        raise OSError("HTTP Error 503: Service Unavailable")

    monkeypatch.setattr(urllib.request, "urlopen", fail)

    with pytest.raises(IntakeError) as caught:
        resolve_owner()

    assert caught.value.fix == "export APIARY_OWNER=<your GitHub account>"


def test_no_owner_anywhere_is_a_refusal_that_names_the_fix():
    with pytest.raises(IntakeError) as caught:
        resolve_owner()

    assert "export APIARY_OWNER=<your GitHub account>" in caught.value.fix


# --------------------------------------------------------------------------
# The contract with the front end
# --------------------------------------------------------------------------


def test_the_proposal_carries_exactly_the_keys_the_run_form_reads(monkeypatch):
    """The front end copies these into the run form's fields by name, so the
    set is pinned as a set: a key renamed here is a form silently left empty."""
    monkeypatch.setenv("APIARY_OWNER", "acme")
    monkeypatch.setattr(
        intake, "propose_setup",
        lambda brief, **_kw: Setup(name="room-tracker", stack="react", reason="a page"),
    )

    proposal = propose(ANSWERS)

    assert set(proposal) == {
        "brief", "repo", "name", "stack", "verify", "public", "auto_merge", "reason",
    }
    assert proposal["repo"] == "acme/room-tracker"
    assert proposal["name"] == "room-tracker"
    assert proposal["stack"] == "react"
    assert proposal["verify"] == STACK_VERIFY["react"]
    assert proposal["public"] == "1"
    assert proposal["auto_merge"] == "1"
    assert proposal["reason"] == "a page"
    assert proposal["brief"] == compose_brief(ANSWERS)


def test_the_verify_command_is_the_stacks_falsified_default(monkeypatch):
    """Derived, never proposed: the gate a run with no `--verify` would use,
    for every stack the model may answer."""
    monkeypatch.setenv("APIARY_OWNER", "acme")
    for stack in sorted(KNOWN_STACKS):
        monkeypatch.setattr(
            intake, "propose_setup",
            lambda brief, _stack=stack, **_kw: Setup(name="a-tool", stack=_stack, reason=""),
        )
        assert propose(ANSWERS)["verify"] == STACK_VERIFY[stack]


def test_no_owner_refuses_before_the_model_is_consulted(monkeypatch):
    """The refusal is instant and the model call is minutes; a wait that ends
    in `export APIARY_OWNER` would be the console's founding complaint."""
    def must_not_be_called(*_args, **_kwargs):
        raise AssertionError("the model was consulted before the owner was resolved")

    monkeypatch.setattr(intake, "propose_setup", must_not_be_called)

    with pytest.raises(IntakeError):
        propose(ANSWERS)


# --------------------------------------------------------------------------
# The site, on the page
# --------------------------------------------------------------------------


def test_the_intake_site_is_served_with_its_three_questions(console):
    sites = body(console.render("GET", "/sites", HOST))["sites"]
    served = {site["key"]: site for site in sites}

    assert "intake" in served
    fields = served["intake"]["fields"]
    assert [f["name"] for f in fields] == ["need", "users", "done"]
    assert all(f["kind"] == "area" for f in fields)
    assert "non-developers" in served["intake"]["label"]
    assert "proposes" in served["intake"]["blurb"]


def test_the_blurb_says_nothing_is_created_until_the_run_is_fired():
    """The promise a business user needs before typing anything real."""
    assert "Nothing is created" in SITES["intake"].blurb


def test_a_fired_intake_goes_through_the_job_machinery(console, monkeypatch):
    monkeypatch.setattr(
        intake, "propose_setup",
        lambda brief, **_kw: Setup(name="room-tracker", stack="python", reason="a service"),
    )
    monkeypatch.setattr(intake, "resolve_owner", lambda: "acme")

    started = console.render(
        "POST", "/run", HOST,
        json.dumps({"site": "intake", "values": ANSWERS}).encode(),
    )
    assert started.status == 202
    job_id = body(started)["id"]

    for _ in range(200):
        job = body(console.render("GET", f"/status?id={job_id}", HOST))
        if job["state"] != "running":
            break
        time.sleep(0.01)

    assert job["state"] == "done"
    assert job["site"] == "intake"
    assert job["result"]["repo"] == "acme/room-tracker"
    assert job["result"]["verify"] == STACK_VERIFY["python"]
    assert console.jobs[job_id].state == "done"


def test_a_missing_owner_lands_on_the_page_with_its_own_fix(console, monkeypatch):
    """`IntakeError` carries a fix, and `_fix_for` must prefer it to pattern
    matching: "export APIARY_OWNER" beats a guess about Ollama."""
    monkeypatch.setattr(
        intake, "propose_setup",
        lambda brief, **_kw: Setup(name="a-tool", stack="python", reason=""),
    )

    started = console.render(
        "POST", "/run", HOST,
        json.dumps({"site": "intake", "values": ANSWERS}).encode(),
    )
    job_id = body(started)["id"]

    for _ in range(200):
        job = body(console.render("GET", f"/status?id={job_id}", HOST))
        if job["state"] != "running":
            break
        time.sleep(0.01)

    assert job["state"] == "error"
    assert job["error"]["type"] == "IntakeError"
    assert job["error"]["fix"] == "export APIARY_OWNER=<your GitHub account>"
