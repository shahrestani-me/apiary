"""What was asked of a model, what came back, how long it took, and what broke.

Nothing in this system recorded any of that. The prompt is assembled inline in
the `.invoke(...)` argument at every call site and discarded; the raw response
is consumed by `with_structured_output`'s parser before any caller sees it; and
the exception is flattened to a string at nearly every site - at
`greenfield/bootstrap.py` it was not even bound to a name before a default was
returned, so "the model chose python" and "Ollama was not running" were the
same observation.

**The seam is the callback, not the return value.** `structured()` hands back a
`RunnableSequence` whose parser has already thrown the `AIMessage` away, so a
wrapper around it can see the prompt and the parsed object and nothing else -
no raw text, no timings. `with_structured_output(include_raw=True)` would keep
them, at the price of changing the return type at all nine call sites *and*
converting parse failures from raised exceptions into a `parsing_error` field,
so a run with capture on would fail differently from one with it off. A
`BaseCallbackHandler` attached to the `ChatOllama` constructor pays neither
price: `on_chat_model_start` carries the messages, `on_llm_end` carries Ollama's
own `total_duration` / `load_duration` / token counts, `on_llm_error` carries
the exception *and still lets it propagate*, and all three are paired by one
`run_id`.

So the handler is attached inside `llm.py`'s two factories, which are the only
places in `src/` that construct a `ChatOllama`. No call site changes, no
signature changes, and no test double changes: a double injected through an
`llm=` / `oracle=` seam never reaches the factory, and therefore never records
anything, which is correct - a fake model made no call.

**Failure here is never allowed to become failure there.** A capture write that
raises would turn a full disk into a failed run, and LangChain swallows handler
exceptions by default, which would turn it into silence instead. Both are worse
than the thing being recorded, so every entry point below catches, reports one
line on stderr, and returns.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_core.callbacks import BaseCallbackHandler

from .artifacts import EventLog, console_root, write_json, _default_redactor
from .config import _flag

__all__ = [
    "CAPTURE_ENV",
    "CAPTURE_MAX_CHARS_ENV",
    "DEFAULT_MAX_CHARS",
    "LLM_CALL",
    "LLM_LOG_NAME",
    "Capture",
    "Recorder",
    "Text",
    "enabled",
    "handler_for",
    "max_chars",
    "recorder",
    "set_recorder",
]

#: Off unless asked for. Capture writes whole prompts - and a worker prompt
#: embeds whole file bodies from the target repository - so this is data an
#: operator opts into for a debugging session, not a default cost every run
#: pays. Read from the environment at call time rather than from `SETTINGS`,
#: which is frozen at import and would make the flag untestable without
#: reloading the module.
CAPTURE_ENV = "APIARY_CAPTURE"

#: Per-field truncation for run capture. A worker prompt is `MAX_FILE_CHARS`
#: (20k) per writable file plus `CONTEXT_BUDGET_CHARS` (24k) of read-only
#: context, so an uncapped record is tens of kilobytes and a busy run is
#: megabytes into a directory nothing prunes.
CAPTURE_MAX_CHARS_ENV = "APIARY_CAPTURE_MAX_CHARS"
DEFAULT_MAX_CHARS = 8192

#: The event name in a run's capture log. A constant for the same reason
#: `CYCLE_FINISHED` is one: something reads it back.
LLM_CALL = "llm.call"

#: Sits beside `events.jsonl` in the run directory. JSON lines, because a run
#: makes many calls and they arrive one at a time; console sessions get one
#: file each instead, so that a leak audit naming a file names one capture
#: rather than naming a day.
LLM_LOG_NAME = "llm.jsonl"

#: Bumped when a field changes meaning, never when one is added. `.swarm/` is
#: gitignored, unpruned and kept forever, so a v2 reader will meet v1 files and
#: the stamp is the whole migration story.
SCHEMA_VERSION = 1


def enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether capture should record anything at all."""
    source = os.environ if env is None else env
    return _flag(source.get(CAPTURE_ENV)) is True


def max_chars(env: Mapping[str, str] | None = None) -> int:
    """The per-field cap, or 0 for "do not truncate".

    An unparseable value is the default rather than an error: this is read on
    the inference path, and a typo in an optional debugging variable must not
    be able to end a run.
    """
    source = os.environ if env is None else env
    raw = (source.get(CAPTURE_MAX_CHARS_ENV) or "").strip()
    if not raw:
        return DEFAULT_MAX_CHARS
    try:
        return max(int(raw), 0)
    except ValueError:
        return DEFAULT_MAX_CHARS


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Text:
    """A possibly-truncated string, with a digest of what it was before.

    The digest covers the **full** text, always. Two captures of the same
    prompt are then recognisably the same even when both were truncated, and a
    truncated record still answers "did the prompt change between attempt 1 and
    attempt 3?" - which is most of what anyone asks a capture file.
    """

    text: str
    chars: int
    sha256: str
    truncated: bool

    @classmethod
    def of(cls, value: Any, *, cap: int = 0) -> "Text":
        full = value if isinstance(value, str) else str(value)
        digest = hashlib.sha256(full.encode("utf-8", errors="replace")).hexdigest()
        if cap and len(full) > cap:
            return cls(text=full[:cap], chars=len(full), sha256=digest, truncated=True)
        return cls(text=full, chars=len(full), sha256=digest, truncated=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "chars": self.chars,
            "sha256": self.sha256,
            "truncated": self.truncated,
        }


@dataclass
class Capture:
    """One model call, from the messages that went out to whatever ended it."""

    id: str
    role: str = ""
    model: str = ""
    schema_name: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    prompt: Text | None = None
    response: Text | None = None
    parsed_ok: bool = False
    error: dict[str, str] | None = None
    total_s: float | None = None
    load_s: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "id": self.id,
            "role": self.role,
            "model": self.model,
            "schema_name": self.schema_name,
            "messages": self.messages,
            "parsed_ok": self.parsed_ok,
            "error": self.error,
            "total_s": self.total_s,
            "load_s": self.load_s,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }
        if self.prompt is not None:
            payload["prompt"] = self.prompt.to_dict()
        if self.response is not None:
            payload["response"] = self.response.to_dict()
        return payload


# --------------------------------------------------------------------------
# Where records go
# --------------------------------------------------------------------------


@dataclass
class Recorder:
    """Somewhere for finished captures to land, and a memory of the last one.

    `directory` is a run directory when a run is active and the console tree
    otherwise. `log` is set for the run case, where many calls append to one
    `llm.jsonl`; the console case writes one file per capture instead, which is
    what lets `swarm console` address a capture by id and what makes a leak
    audit's `path:line` point at a record rather than at a day's log.

    `last` exists for the console: it fires one call and immediately wants the
    record for it, and threading a return value back out through LangChain's
    callback machinery is not possible.
    """

    directory: Path
    log: EventLog | None = None
    cap: int = 0
    last: Capture | None = None
    written: list[Path] = field(default_factory=list)

    @classmethod
    def for_run(cls, run_directory: Path, *, cap: int | None = None) -> "Recorder":
        redactor = _default_redactor()
        return cls(
            directory=Path(run_directory),
            log=EventLog(Path(run_directory) / LLM_LOG_NAME, redact=redactor),
            cap=max_chars() if cap is None else cap,
        )

    @classmethod
    def for_console(cls, directory: Path | None = None, *, cap: int = 0) -> "Recorder":
        """A console recorder does not truncate: the operator typed the prompt."""
        return cls(directory=Path(directory) if directory else console_root(), cap=cap)

    def record(self, capture: Capture) -> Path | None:
        self.last = capture
        payload = capture.to_dict()
        try:
            if self.log is not None:
                self.log.emit(LLM_CALL, **payload)
                return self.log.path
            target = self.directory / f"{capture.id}.json"
            write_json(target, payload, redact=_default_redactor())
            self.written.append(target)
            return target
        except Exception as exc:  # noqa: BLE001 - a capture must never end a run
            print(f"! capture: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None


#: Process-wide, because nothing in `src/swarm/` imports threading, asyncio or
#: contextvars and the console is deliberately single-threaded. Kept behind two
#: functions so that the day any of that changes, the swap to a contextvar is
#: one edit here rather than an audit of every reader.
_RECORDER: Recorder | None = None


def recorder() -> Recorder | None:
    """The installed recorder, or a console-rooted default once capture is on.

    The default matters more than it looks. `swarm run` installs a run recorder
    so a run's calls land beside its other artifacts - but `swarm doctor`, a
    one-off script and anything else reaching a factory install nothing, and a
    capture feature that silently records nothing unless the caller remembered
    a setup step is a capture feature that will be believed and be empty.

    Created on first use rather than at import: with the flag off this is never
    reached, and no directory is created by a process that asked for nothing.
    """
    global _RECORDER
    if _RECORDER is None and enabled():
        _RECORDER = Recorder.for_console()
    return _RECORDER


def set_recorder(sink: Recorder | None) -> Recorder | None:
    """Install a recorder and return the one it replaced, for restoring."""
    global _RECORDER
    previous = _RECORDER
    _RECORDER = sink
    return previous


# --------------------------------------------------------------------------
# The handler
# --------------------------------------------------------------------------


def _messages(serialised: Any, prompts: Any) -> list[dict[str, str]]:
    """LangChain's message objects, flattened to role/content pairs.

    Defensive by construction: this runs on the inference path inside a
    callback whose exceptions LangChain swallows, so a shape it did not expect
    must degrade to a readable string rather than raise into silence.
    """
    rows: list[dict[str, str]] = []
    for batch in prompts or ():
        items: Sequence[Any] = batch if isinstance(batch, (list, tuple)) else [batch]
        for message in items:
            role = getattr(message, "type", None) or message.__class__.__name__
            content = getattr(message, "content", message)
            rows.append({"role": str(role), "content": content if isinstance(content, str) else str(content)})
    return rows


def _seconds(metadata: Mapping[str, Any], key: str) -> float | None:
    """Ollama reports durations in nanoseconds; nobody reads nanoseconds."""
    value = metadata.get(key)
    if not isinstance(value, (int, float)):
        return None
    return round(value / 1e9, 3)


class _CaptureHandler(BaseCallbackHandler):
    """The callback `ChatOllama` is constructed with.

    A real subclass, and that is not a style preference: `ChatOllama` is a
    pydantic model whose `callbacks` field validates as
    `list[BaseCallbackHandler]`, so a duck-typed object with the right methods
    is rejected at construction with a `ValidationError` - which surfaces as a
    model that cannot be built at all rather than as capture quietly not
    working. The cost is that importing this module imports langchain's
    callback base, which is why `llm.py` imports *this module* lazily and only
    when the flag is on.

    `role` and `model` are stamped by the factory rather than read from the
    callback: `invocation_params` carries neither, and the error path carries no
    response metadata at all, so a record built purely from what arrives would
    be unable to say which model failed.
    """

    def __init__(self, *, role: str, model: str, sink: Recorder | None = None) -> None:
        self.role = role
        self.model = model
        self._sink = sink
        self._open: dict[str, Capture] = {}

    # -- lifecycle ------------------------------------------------------

    @property
    def sink(self) -> Recorder | None:
        return self._sink if self._sink is not None else recorder()

    def _capture(self, run_id: Any) -> Capture:
        key = str(run_id)
        if key not in self._open:
            self._open[key] = Capture(id=key or uuid.uuid4().hex, role=self.role, model=self.model)
        return self._open[key]

    def _finish(self, run_id: Any) -> None:
        capture = self._open.pop(str(run_id), None)
        sink = self.sink
        if capture is not None and sink is not None:
            sink.record(capture)

    # -- callbacks ------------------------------------------------------

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        try:
            capture = self._capture(kwargs.get("run_id"))
            capture.messages = _messages(serialized, messages)
            cap = self.sink.cap if self.sink is not None else 0
            joined = "\n\n".join(row["content"] for row in capture.messages)
            capture.prompt = Text.of(joined, cap=cap)
            invocation = kwargs.get("invocation_params") or {}
            schema = invocation.get("format")
            if isinstance(schema, Mapping):
                capture.schema_name = str(schema.get("title") or "")
        except Exception as exc:  # noqa: BLE001 - never raise into the model call
            print(f"! capture: {type(exc).__name__}: {exc}", file=sys.stderr)

    #: A non-chat model would arrive here instead. Same treatment, so that a
    #: future call site using a completion model is not silently uncaptured.
    def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        try:
            capture = self._capture(kwargs.get("run_id"))
            capture.messages = [{"role": "human", "content": str(p)} for p in (prompts or ())]
            cap = self.sink.cap if self.sink is not None else 0
            capture.prompt = Text.of("\n\n".join(r["content"] for r in capture.messages), cap=cap)
        except Exception as exc:  # noqa: BLE001
            print(f"! capture: {type(exc).__name__}: {exc}", file=sys.stderr)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            capture = self._capture(kwargs.get("run_id"))
            cap = self.sink.cap if self.sink is not None else 0
            text, metadata, usage = "", {}, {}
            generations = getattr(response, "generations", None) or ()
            for batch in generations:
                for generation in batch if isinstance(batch, (list, tuple)) else [batch]:
                    message = getattr(generation, "message", None)
                    text = getattr(generation, "text", "") or str(getattr(message, "content", "") or "")
                    metadata = getattr(message, "response_metadata", None) or {}
                    usage = getattr(message, "usage_metadata", None) or {}
            capture.response = Text.of(text, cap=cap)
            capture.parsed_ok = True
            if isinstance(metadata, Mapping):
                capture.model = str(metadata.get("model") or capture.model)
                capture.total_s = _seconds(metadata, "total_duration")
                capture.load_s = _seconds(metadata, "load_duration")
            if isinstance(usage, Mapping):
                capture.prompt_tokens = usage.get("input_tokens")
                capture.output_tokens = usage.get("output_tokens")
        except Exception as exc:  # noqa: BLE001
            print(f"! capture: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            self._finish(kwargs.get("run_id"))

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        try:
            capture = self._capture(kwargs.get("run_id"))
            capture.parsed_ok = False
            capture.error = {"type": type(error).__name__, "message": str(error)}
        except Exception as exc:  # noqa: BLE001
            print(f"! capture: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            self._finish(kwargs.get("run_id"))


def handler_for(*, role: str, model: str, sink: Recorder | None = None) -> Any:
    """The callback to hand `ChatOllama(callbacks=[...])`, or `None` when off.

    Returning `None` rather than a do-nothing handler is what makes "capture is
    off" mean *nothing is attached*, which is the only version of off that
    cannot cost an inference call anything.
    """
    if not enabled():
        return None
    return _CaptureHandler(role=role, model=model, sink=sink)


def as_dict(capture: Capture) -> dict[str, Any]:
    """`Capture.to_dict`, for callers holding a dataclass they did not build."""
    return capture.to_dict() if isinstance(capture, Capture) else dataclasses.asdict(capture)
