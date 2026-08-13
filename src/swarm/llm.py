"""Model factory.

Everything talks to Ollama over HTTP. Nothing leaves the machine.
"""

from __future__ import annotations

from typing import TypeVar

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from .config import SETTINGS

T = TypeVar("T", bound=BaseModel)


def orchestrator_llm() -> ChatOllama:
    """Small, deterministic. Used for planning and progress judgement."""
    return ChatOllama(
        model=SETTINGS.orchestrator_model,
        base_url=SETTINGS.ollama_base_url,
        temperature=SETTINGS.orchestrator_temperature,
        num_ctx=SETTINGS.orchestrator_num_ctx,
    )


def worker_llm() -> ChatOllama:
    """The code writer. Bigger is better here."""
    return ChatOllama(
        model=SETTINGS.worker_model,
        base_url=SETTINGS.ollama_base_url,
        temperature=SETTINGS.worker_temperature,
        num_ctx=SETTINGS.worker_num_ctx,
    )


def structured(llm: ChatOllama, schema: type[T]):
    """Force JSON matching `schema`.

    ChatOllama passes the JSON schema to Ollama's `format` parameter, which
    constrains decoding. This is what makes small models usable for
    orchestration - they cannot wander off-format even if they want to.
    """
    return llm.with_structured_output(schema)
