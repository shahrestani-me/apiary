"""Model factory.

Everything talks to Ollama over HTTP. Nothing leaves the machine.
"""

from __future__ import annotations

from typing import TypeVar

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from .config import SETTINGS

T = TypeVar("T", bound=BaseModel)


def _callbacks(role: str, model: str) -> list | None:
    """Capture's hook, or nothing at all when `APIARY_CAPTURE` is unset.

    Imported here rather than at module scope so that a process with capture
    off never loads `swarm.capture` and never pays for LangChain's callback
    imports - and so that this module keeps its single dependency on `config`.

    This is the entire integration. Attaching at the constructor rather than
    around `structured()`'s return value is what lets a record hold the raw
    response and Ollama's own `load_duration`, both of which the structured
    output parser discards before any caller sees them; and it is why the two
    functions below keep their signatures, why no call site changed, and why
    every existing test double still bypasses capture without noticing it.
    """
    from .capture import handler_for  # noqa: PLC0415 - deliberately lazy, see above

    handler = handler_for(role=role, model=model)
    return [handler] if handler is not None else None


def orchestrator_llm() -> ChatOllama:
    """Small, deterministic. Used for planning and progress judgement."""
    return ChatOllama(
        model=SETTINGS.orchestrator_model,
        base_url=SETTINGS.ollama_base_url,
        temperature=SETTINGS.orchestrator_temperature,
        num_ctx=SETTINGS.orchestrator_num_ctx,
        callbacks=_callbacks("orchestrator", SETTINGS.orchestrator_model),
    )


def worker_llm() -> ChatOllama:
    """The code writer. Bigger is better here."""
    return ChatOllama(
        model=SETTINGS.worker_model,
        base_url=SETTINGS.ollama_base_url,
        temperature=SETTINGS.worker_temperature,
        num_ctx=SETTINGS.worker_num_ctx,
        callbacks=_callbacks("worker", SETTINGS.worker_model),
    )


def structured(llm: ChatOllama, schema: type[T]):
    """Force JSON matching `schema`.

    ChatOllama passes the JSON schema to Ollama's `format` parameter, which
    constrains decoding. This is what makes small models usable for
    orchestration - they cannot wander off-format even if they want to.
    """
    return llm.with_structured_output(schema)


def parse_failure(exc: BaseException) -> bool:
    """Did the structured boundary reject the model's reply as unparseable?

    Answered here rather than at the call site because the answer is a
    framework exception type, and ADR 0003 keeps every framework import inside
    this module: a caller that wants to retry a rejected reply (the worker
    does - an over-long prompt truncated by Ollama produces exactly this) asks
    the question without learning which framework did the rejecting.
    """
    from langchain_core.exceptions import OutputParserException  # noqa: PLC0415 - lazy, like `_callbacks`

    return isinstance(exc, OutputParserException)
