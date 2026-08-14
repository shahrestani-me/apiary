"""What capture must record, and the ways it could quietly stop recording it.

The load-bearing tests here are the ones that fail when capture becomes a
no-op. Capture attaches inside `llm.py`'s factories, and every existing double
in this suite is injected through an `llm=` / `oracle=` seam that never reaches
a factory - so nothing else in `tests/` exercises this code, and "the suite is
green" says nothing about whether a single record was ever written.

So the handler is driven directly, with the payload shapes LangChain actually
delivers. That is hermetic, needs no Ollama, and is the only level at which the
callback contract can be pinned in CI, which never runs the gated markers.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from swarm import capture as capture_mod
from swarm.capture import (
    CAPTURE_ENV,
    CAPTURE_MAX_CHARS_ENV,
    DEFAULT_MAX_CHARS,
    LLM_LOG_NAME,
    Capture,
    Recorder,
    Text,
    enabled,
    handler_for,
    max_chars,
)

SOURCE = Path(__file__).resolve().parent.parent / "src" / "swarm"


# --------------------------------------------------------------------------
# Doubles for what LangChain hands a callback
# --------------------------------------------------------------------------


class Message:
    """A `BaseMessage` as far as this handler is concerned."""

    def __init__(self, type_: str, content: str, *, metadata=None, usage=None):
        self.type = type_
        self.content = content
        self.response_metadata = metadata or {}
        self.usage_metadata = usage or {}


class Generation:
    def __init__(self, message: Message):
        self.message = message
        self.text = message.content


class Result:
    def __init__(self, message: Message):
        self.generations = [[Generation(message)]]


#: The shape `on_llm_end` really carries for ChatOllama: nanoseconds, and a
#: model name that the invocation params never had.
OLLAMA_META = {
    "model": "gemma4:31b",
    "total_duration": 13_401_000_000,
    "load_duration": 10_417_000_000,
    "done_reason": "stop",
}
OLLAMA_USAGE = {"input_tokens": 231, "output_tokens": 11}


def a_start(handler, run_id="run-1", *, system="be brief", human="the objective", schema="Plan"):
    handler.on_chat_model_start(
        {},
        [[Message("system", system), Message("human", human)]],
        run_id=run_id,
        invocation_params={"format": {"title": schema}, "_type": "chat-ollama"},
    )


def an_end(handler, run_id="run-1", *, text='{"tasks": []}'):
    handler.on_llm_end(
        Result(Message("ai", text, metadata=OLLAMA_META, usage=OLLAMA_USAGE)),
        run_id=run_id,
    )


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(CAPTURE_ENV, "1")


@pytest.fixture
def sink(tmp_path):
    return Recorder.for_console(tmp_path / "console")


@pytest.fixture(autouse=True)
def no_installed_recorder():
    """Nothing leaks between tests through the process-wide recorder."""
    previous = capture_mod.set_recorder(None)
    yield
    capture_mod.set_recorder(previous)


# --------------------------------------------------------------------------
# Off is off
# --------------------------------------------------------------------------


def test_capture_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv(CAPTURE_ENV, raising=False)
    assert enabled() is False
    assert handler_for(role="worker", model="m") is None


@pytest.mark.parametrize("value", ["0", "false", "no", "", "  "])
def test_the_flag_is_parsed_not_merely_present(monkeypatch, value):
    """`APIARY_CAPTURE=0` means off, not "non-empty, therefore on"."""
    monkeypatch.setenv(CAPTURE_ENV, value)
    assert enabled() is False


def test_a_process_with_capture_off_never_writes(on, sink, monkeypatch, tmp_path):
    monkeypatch.setenv(CAPTURE_ENV, "0")
    assert handler_for(role="worker", model="m", sink=sink) is None
    assert list((tmp_path / "console").glob("*")) == []


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_a_successful_call_records_prompt_response_and_ollamas_own_timings(on, sink):
    handler = handler_for(role="orchestrator", model="gemma4:31b", sink=sink)
    a_start(handler)
    an_end(handler)

    record = json.loads(sink.written[0].read_text())

    assert record["role"] == "orchestrator"
    assert record["model"] == "gemma4:31b"
    assert record["schema_name"] == "Plan"
    assert record["parsed_ok"] is True
    assert record["error"] is None
    assert [m["role"] for m in record["messages"]] == ["system", "human"]
    assert "the objective" in record["prompt"]["text"]
    assert record["response"]["text"] == '{"tasks": []}'
    # Nanoseconds are Ollama's unit and nobody reads nanoseconds.
    assert record["total_s"] == 13.401
    assert record["load_s"] == 10.417
    assert record["prompt_tokens"] == 231
    assert record["output_tokens"] == 11
    assert record["schema"] == 1


def test_the_record_says_the_host_was_swapping_not_thinking(on, sink):
    """`load_s` against `total_s` is the number `RunMetrics.swap_share` wanted.

    A measured call spent 10.4s of 13.4s loading the model. Recording only a
    wall clock would have reported a slow model rather than a cold one.
    """
    handler = handler_for(role="orchestrator", model="gemma4:31b", sink=sink)
    a_start(handler)
    an_end(handler)

    record = json.loads(sink.written[0].read_text())

    assert record["load_s"] / record["total_s"] > 0.75


# --------------------------------------------------------------------------
# The failure path - the reason this epic exists
# --------------------------------------------------------------------------


def test_a_failed_call_keeps_the_exception_type_and_the_prompt_that_caused_it(on, sink):
    """Today every site keeps `str(exc)` at best; `bootstrap` kept nothing."""
    handler = handler_for(role="worker", model="gemma4:26b", sink=sink)
    a_start(handler, human="build the thing")
    handler.on_llm_error(ConnectionRefusedError("connection refused"), run_id="run-1")

    record = json.loads(sink.written[0].read_text())

    assert record["error"] == {
        "type": "ConnectionRefusedError",
        "message": "connection refused",
    }
    assert record["parsed_ok"] is False
    assert "build the thing" in record["prompt"]["text"]
    assert record["model"] == "gemma4:26b"


def test_two_concurrent_calls_do_not_mix_their_prompts(on, sink):
    """LangChain pairs start/end by `run_id`, and so must the record."""
    handler = handler_for(role="worker", model="m", sink=sink)
    a_start(handler, "a", human="first")
    a_start(handler, "b", human="second")
    an_end(handler, "b", text="second answer")
    an_end(handler, "a", text="first answer")

    records = {
        json.loads(p.read_text())["response"]["text"]: json.loads(p.read_text())
        for p in sink.written
    }
    assert "first" in records["first answer"]["prompt"]["text"]
    assert "second" in records["second answer"]["prompt"]["text"]


# --------------------------------------------------------------------------
# Truncation, and the digest that survives it
# --------------------------------------------------------------------------


def test_a_capped_field_is_marked_and_still_hashes_the_whole_text():
    full = "x" * 100
    short = Text.of(full, cap=10)

    assert short.text == "x" * 10
    assert short.chars == 100
    assert short.truncated is True
    assert short.sha256 == Text.of(full).sha256, "the digest must cover the full text"


def test_a_console_capture_is_not_truncated(on, tmp_path):
    """The operator typed it; truncating their own prompt back at them is rude."""
    sink = Recorder.for_console(tmp_path / "console")
    handler = handler_for(role="worker", model="m", sink=sink)
    a_start(handler, human="q" * (DEFAULT_MAX_CHARS * 2))
    an_end(handler)

    record = json.loads(sink.written[0].read_text())
    assert record["prompt"]["truncated"] is False


def test_the_cap_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(CAPTURE_MAX_CHARS_ENV, "128")
    assert max_chars() == 128


def test_an_unreadable_cap_is_the_default_not_a_crash(monkeypatch):
    """This is read on the inference path; a typo must not end a run."""
    monkeypatch.setenv(CAPTURE_MAX_CHARS_ENV, "lots")
    assert max_chars() == DEFAULT_MAX_CHARS


# --------------------------------------------------------------------------
# A run's capture log
# --------------------------------------------------------------------------


def test_a_run_appends_every_call_to_one_log(on, tmp_path):
    sink = Recorder.for_run(tmp_path / "run")
    handler = handler_for(role="orchestrator", model="m", sink=sink)
    for index in range(3):
        a_start(handler, f"run-{index}", human=f"objective {index}")
        an_end(handler, f"run-{index}")

    lines = (tmp_path / "run" / LLM_LOG_NAME).read_text().strip().splitlines()

    assert len(lines) == 3
    assert [json.loads(line)["event"] for line in lines] == ["llm.call"] * 3
    assert "objective 2" in json.loads(lines[-1])["prompt"]["text"]


def test_run_capture_truncates_because_a_worker_prompt_carries_whole_files(on, tmp_path):
    sink = Recorder.for_run(tmp_path / "run", cap=64)
    handler = handler_for(role="worker", model="m", sink=sink)
    a_start(handler, human="f" * 5_000)
    an_end(handler)

    record = json.loads((tmp_path / "run" / LLM_LOG_NAME).read_text().strip())
    assert record["prompt"]["truncated"] is True
    assert record["prompt"]["chars"] > 5_000
    assert len(record["prompt"]["text"]) == 64


# --------------------------------------------------------------------------
# Redaction, and the guard that proves the writer did not bypass it
# --------------------------------------------------------------------------


CREDENTIAL = "a-very-secret-value-nobody-should-see"


def test_a_credential_in_a_prompt_does_not_reach_disk(on, monkeypatch, tmp_path):
    """The prompt is the highest-risk string in the system: a worker prompt is
    whole file bodies from a target repository. It must go through the same
    redactor every other writer in `artifacts.py` uses."""
    monkeypatch.setenv("GITHUB_TOKEN", CREDENTIAL)
    sink = Recorder.for_console(tmp_path / "console")
    handler = handler_for(role="worker", model="m", sink=sink)
    a_start(handler, human=f"deploy with {CREDENTIAL}")
    an_end(handler, text=f"sure, using {CREDENTIAL}")

    written = sink.written[0].read_text()

    assert CREDENTIAL not in written
    assert "***" in written


def test_the_redaction_test_is_not_a_writer_that_never_writes(on, tmp_path):
    """The control for the test above.

    Modelled on `test_the_audit_is_not_a_scanner_that_never_matches`: a capture
    path that silently wrote nothing would pass the redaction test trivially,
    and that is the exact way this feature rots.
    """
    sink = Recorder.for_console(tmp_path / "console")
    handler = handler_for(role="worker", model="m", sink=sink)
    a_start(handler, human="nothing secret here")
    an_end(handler)

    assert sink.written, "capture wrote no file at all"
    assert "nothing secret here" in sink.written[0].read_text()


def test_a_write_failure_is_reported_not_raised(on, tmp_path, capsys):
    """A full disk must not end a run, and LangChain swallowing the exception
    would make it end nothing and say nothing."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    sink = Recorder.for_console(blocked)

    assert sink.record(Capture(id="x")) is None
    assert "! capture:" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The factories actually accept it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factory", ["orchestrator_llm", "worker_llm"])
def test_a_model_can_be_built_with_the_handler_attached(on, factory):
    """The test this suite was missing, and the bug it let through.

    Every other test here drives the handler directly, so none of them ever
    constructs a `ChatOllama`. The first real call did - and failed, because
    `ChatOllama` is a pydantic model whose `callbacks` field validates as
    `list[BaseCallbackHandler]`: a duck-typed handler with all the right
    methods is refused at construction. Capture did not silently do nothing;
    the model could not be built at all.

    No network: constructing a `ChatOllama` connects to nothing.
    """
    import swarm.llm as llm_mod

    model = getattr(llm_mod, factory)()

    assert model.callbacks, "the factory attached no handler with capture on"
    assert type(model.callbacks[0]).__name__ == "_CaptureHandler"


@pytest.mark.parametrize("factory", ["orchestrator_llm", "worker_llm"])
def test_with_capture_off_the_factories_attach_nothing(monkeypatch, factory):
    monkeypatch.delenv(CAPTURE_ENV, raising=False)
    import swarm.llm as llm_mod

    assert getattr(llm_mod, factory)().callbacks is None


def test_a_structured_model_still_composes_with_the_handler_attached(on):
    """`structured()` wraps the model in a `RunnableSequence`. The handler must
    survive that composition, since every call site goes through it."""
    from pydantic import BaseModel

    import swarm.llm as llm_mod

    class Answer(BaseModel):
        value: int

    chain = llm_mod.structured(llm_mod.worker_llm(), Answer)

    assert chain is not None
    assert hasattr(chain, "invoke")


# --------------------------------------------------------------------------
# Source pins - the tenth call site problem
# --------------------------------------------------------------------------


def _calls_named(tree: ast.AST, name: str) -> bool:
    """Whether `name(...)` is actually *called* anywhere in this module.

    Parsed rather than grepped, deliberately. A text search also matches the
    prose in `capture.py` explaining why these two rules exist, and a pin that
    fails when someone documents it teaches people to delete the pin.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == name:
            return True
        if isinstance(func, ast.Attribute) and func.attr == name:
            return True
    return False


def _passes_keyword(tree: ast.AST, keyword: str) -> bool:
    return any(
        isinstance(node, ast.keyword) and node.arg == keyword for node in ast.walk(tree)
    )


def _modules():
    for path in sorted(SOURCE.rglob("*.py")):
        yield path.relative_to(SOURCE).as_posix(), ast.parse(path.read_text())


def test_only_llm_py_constructs_a_model():
    """A call site that built its own `ChatOllama` would be silently uncaptured.

    Same shape as the negative pin in `test_reconcile.py`: the guarantee here is
    "every model in this system comes from the two factories", and it is worth
    exactly as much as the test that says so.
    """
    offenders = [name for name, tree in _modules() if _calls_named(tree, "ChatOllama")]
    assert offenders == ["llm.py"]


def test_structured_output_never_asks_for_the_raw_response():
    """`include_raw=True` would change the return type at all nine call sites
    and convert parse failures into a field, so a run with capture on would
    fail differently from one with it off. The callback exists to avoid it."""
    offenders = [name for name, tree in _modules() if _passes_keyword(tree, "include_raw")]
    assert offenders == []


# --------------------------------------------------------------------------
# The end-to-end proof, against a real model
# --------------------------------------------------------------------------


@pytest.mark.ollama
def test_a_real_call_is_captured_end_to_end(on, tmp_path, monkeypatch):
    """Doctor's schema probe is the cheapest real model call in the system.

    Nothing below this line is faked: a real `ChatOllama`, a real constrained
    decode, and a real `load_duration` from the server.
    """
    from swarm.doctor import HostInference

    sink = Recorder.for_console(tmp_path / "console")
    capture_mod.set_recorder(sink)

    HostInference().schema_probe("orchestrator")

    record = json.loads(sink.written[0].read_text())
    assert record["parsed_ok"] is True
    assert record["prompt"]["chars"] > 0
    assert record["response"]["chars"] > 0
    assert record["total_s"] > 0
    assert record["schema_name"] == "Ping"
