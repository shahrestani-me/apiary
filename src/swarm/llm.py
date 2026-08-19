"""Model factory.

Every model in this system is built here, from a `ModelSpec`, through a
registered provider. Ollama is registered first and is the default, so a clone
with nothing configured still talks to a local server and nothing leaves the
machine.

**Why a registry rather than an import.** ADR 0003 established that the
orchestration *framework* is a detail confined to one module. The provider is
the same kind of fact and used to be confined not at all: this module imported
`ChatOllama` at the top, both factories took no arguments, and `structured()`
was documented in terms of Ollama's `format` parameter. ADR 0006 is the
decision to treat the provider the way 0003 treats the framework, and this
module is where it is enforced.

**Why the spec carries an open option set.** The three providers registered
below need three genuinely different things to dial: Ollama needs a URL and no
credential at all, OpenAI needs an API key, and Bedrock needs an AWS profile
and a region. Giving `ModelSpec` a named field per provider would make every
new provider a change to the shared type, and would leave two of the fields
meaningless on any given spec - which is the shape that grows a `region` that
silently does nothing on OpenAI. So the four facts every provider has are
fields, and everything else is `options`, declared per provider and validated
against that declaration. A misspelled `regoin` is refused by name rather than
defaulting quietly to somewhere else in the world.

**`structured()` is the one function here that is not a passthrough.** Ollama's
`format` parameter constrains *decoding*, which is what makes small models
usable for orchestration - they cannot wander off-format even if they want to.
OpenAI's strict `json_schema` and Bedrock's `output_config` are different
mechanisms reaching the same guarantee by different routes, and they fail
differently: a violation surfaces as a parser rejecting a reply that was
actually produced, rather than as a decode that could not produce the wrong
shape in the first place. All three paths raise. None coerces. That is the
contract every caller here relies on, and there is a test per path that says so.

**`parse_failure()` is the other half of that contract.** `worker/edit.py`
retries a rejected reply exactly once, and it decides whether to by asking this
module. A provider whose rejection type is not named below is a provider whose
retry silently stops firing - the call fails on the first decode and escapes as
`EditError`, which reads like a broken model rather than a truncated one.

**Optional at runtime, present in dev.** `langchain-openai` and `langchain-aws`
are extras, not dependencies: `pip install apiary` still installs a local-only
system, which is what `README.md` and `pyproject.toml` promise. Their imports
are therefore lazy and inside the constructors, and a missing extra is a
`ConfigError` naming the install command rather than an `ImportError` at module
scope that would take `swarm --help` down with it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Callable, TypeVar

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from .config import SETTINGS, ConfigError

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "BEDROCK",
    "OLLAMA",
    "OPENAI",
    "PROVIDERS",
    "ModelSpec",
    "Option",
    "Provider",
    "orchestrator_llm",
    "parse_failure",
    "provider_for",
    "structured",
    "worker_llm",
]

#: Provider names. Constants because they are written in an environment
#: variable, saved in a file by the console and printed by doctor; three
#: spellings of the same string is how one of those quietly stops matching.
OLLAMA = "ollama"
OPENAI = "openai"
BEDROCK = "bedrock"


# --------------------------------------------------------------------------
# What a model is, written down
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Option:
    """One provider-specific setting, declared so a typo can be refused.

    `secret_env` marks an option whose value is the *name* of an environment
    variable holding a credential - never the credential. That distinction is
    what lets a whole spec be logged, saved to a file and rendered on a page
    with no redaction story of its own.
    """

    name: str
    default: str = ""
    doc: str = ""
    secret_env: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build one model, and nothing that is a secret.

    Frozen and comparable so that "did this run and that one use the same
    model?" is `==` rather than a reading exercise, and hashable so a caller
    can key a cache on one.

    `num_ctx` is a budget, not only a request. Ollama receives it as `num_ctx`
    and allocates a KV cache accordingly; a remote provider receives nothing
    and serves whatever window the model has. It is still carried for all
    three, because `worker/edit.py:prompt_budget` trims the prompt against this
    number, and a budget that silently stopped applying on a remote path would
    send an over-long prompt to a provider that bills by the token.
    """

    provider: str = OLLAMA
    model: str = ""
    temperature: float = 0.0
    num_ctx: int = 16384
    #: Provider-specific settings as sorted name/value pairs - a tuple rather
    #: than a mapping so the spec stays frozen, hashable and comparable, which
    #: a `dict` field would cost all three of.
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            known = ", ".join(sorted(PROVIDERS))
            raise ConfigError(
                f"unknown model provider {self.provider!r}; known providers: {known}"
            )
        declared = {option.name for option in PROVIDERS[self.provider].options}
        unknown = sorted({name for name, _ in self.options} - declared)
        if unknown:
            raise ConfigError(
                f"{self.provider} does not understand "
                f"{', '.join(repr(name) for name in unknown)}; it takes "
                f"{', '.join(sorted(declared)) or 'no options'}"
            )
        # Normalised on the way in, so that two specs written in a different
        # order are `==` and hash alike. `object.__setattr__` because frozen.
        object.__setattr__(self, "options", tuple(sorted(self.options)))

    # -- reading ---------------------------------------------------------

    def option(self, name: str) -> str:
        """One option's value, or the provider's declared default for it."""
        for key, value in self.options:
            if key == name:
                return value
        for declared in PROVIDERS[self.provider].options:
            if declared.name == name:
                return declared.default
        return ""

    @property
    def label(self) -> str:
        """`ollama:gemma4:31b`. What a log line and a console page show.

        Provider-qualified always, including for Ollama, because the whole
        point of this epic is that the bare model name stopped being an
        unambiguous answer to "which model did this run use".
        """
        return f"{self.provider}:{self.model}"

    @property
    def credential(self) -> str:
        """Where this spec's credential comes from, in the provider's own terms.

        A sentence rather than a value, and the three providers answer it in
        three shapes - no credential at all, the name of an environment
        variable, an AWS profile and region. Doctor (#267) and the console
        (#266) both need to *say* this, and neither should have to know which
        shape it is this time.
        """
        return PROVIDERS[self.provider].credential(self)

    # -- deriving --------------------------------------------------------

    def with_model(self, model: str) -> "ModelSpec":
        """The same spec pointed at a different model of the same provider."""
        return replace(self, model=model)

    def with_options(self, **values: str) -> "ModelSpec":
        """The same spec with options added or replaced."""
        merged = {name: value for name, value in self.options}
        merged.update({name: str(value) for name, value in values.items()})
        return replace(self, options=tuple(sorted(merged.items())))


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Provider:
    """One way to build a model, force a schema onto it, and read its refusals.

    `client_names` is how `structured()` and `parse_failure()` find their way
    back to a provider from a model they were merely handed. Matching on class
    *names* rather than on the classes themselves is deliberate: resolving the
    class would import the SDK, and asking "is this a ChatOpenAI?" about a
    `ChatOllama` must not install-check `langchain-openai` to answer no.
    """

    name: str
    build: Callable[["ModelSpec", list | None], Any]
    structured: Callable[[Any, type[BaseModel], "ModelSpec | None"], Any]
    #: Called lazily, and only when an exception needs classifying. Returns the
    #: exception types that mean "the reply was not the schema" - the ones
    #: `worker/edit.py` retries once. A refused socket, a missing credential
    #: and a model that declined are deliberately *not* here: they fail
    #: identically on a second decode, so retrying spends a whole-file
    #: generation to learn nothing.
    rejections: Callable[[], tuple[type[BaseException], ...]]
    client_names: tuple[str, ...]
    credential: Callable[["ModelSpec"], str]
    options: tuple[Option, ...] = ()


def _callbacks(role: str, model: str, provider: str = "") -> list | None:
    """Capture's hook, or nothing at all when `APIARY_CAPTURE` is unset.

    Imported here rather than at module scope so that a process with capture
    off never loads `swarm.capture` and never pays for LangChain's callback
    imports - and so that this module keeps its single dependency on `config`.

    This is the entire integration. Attaching at the constructor rather than
    around `structured()`'s return value is what lets a record hold the raw
    response and the provider's own timings, both of which the structured
    output parser discards before any caller sees them; and it is why the two
    factories below keep their signatures, why no call site changed, and why
    every existing test double still bypasses capture without noticing it.
    """
    from .capture import handler_for  # noqa: PLC0415 - deliberately lazy, see above

    handler = handler_for(role=role, model=model, provider=provider)
    return [handler] if handler is not None else None


def _missing(extra: str, package: str, exc: BaseException) -> ConfigError:
    """The message an operator gets instead of an `ImportError` traceback."""
    return ConfigError(
        f"the {extra} provider needs the `{package}` package, which apiary does not "
        f"install by default - it is local-first, and the default path needs no "
        f"remote SDK at all. Install it with `pip install 'apiary[{extra}]'` ({exc})"
    )


def _required_env(names: tuple[str, ...]) -> str:
    """The first of `names` that is set to something, or ""."""
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


# --- ollama ---------------------------------------------------------------


def _build_ollama(spec: ModelSpec, callbacks: list | None) -> Any:
    """`ChatOllama` is a module global, and is looked up as one at call time.

    Deliberately not lazy, unlike the two constructors below. `langchain-
    ollama` is a hard dependency, so importing it here buys nothing - and the
    global is a seam a test already uses: `test_console_build` patches
    `swarm.llm.ChatOllama` to prove an approved plan asks no model at all,
    which catches an inference reached through any import style anywhere in the
    build path. A lazy import would move that name out of reach and quietly
    retire the assertion.
    """
    return ChatOllama(
        model=spec.model,
        base_url=spec.option("base_url") or SETTINGS.ollama_base_url,
        temperature=spec.temperature,
        num_ctx=spec.num_ctx,
        callbacks=callbacks,
    )


def _structured_ollama(llm: Any, schema: type[BaseModel], spec: ModelSpec | None = None) -> Any:
    """Force JSON matching `schema`, by constraining the decoder.

    `ChatOllama` passes the JSON schema to Ollama's `format` parameter, which
    constrains decoding. This is what makes small models usable for
    orchestration - they cannot wander off-format even if they want to.
    """
    return llm.with_structured_output(schema)


def _rejections_langchain() -> tuple[type[BaseException], ...]:
    """LangChain's own "that reply was not the schema".

    Shared by Ollama and by Bedrock's `json_schema` path, both of which end in
    a `PydanticOutputParser`.
    """
    from langchain_core.exceptions import OutputParserException  # noqa: PLC0415 - lazy

    return (OutputParserException,)


def _credential_ollama(spec: ModelSpec) -> str:
    return "none - ollama is unauthenticated"


# --- openai ---------------------------------------------------------------

#: Where the OpenAI key is read from when a spec does not name somewhere else.
OPENAI_KEY_ENV = "OPENAI_API_KEY"


def _build_openai(spec: ModelSpec, callbacks: list | None) -> Any:
    try:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415 - lazy, and optional
    except ImportError as exc:  # pragma: no cover - only without the extra installed
        raise _missing(OPENAI, "langchain-openai", exc) from exc

    from pydantic import SecretStr  # noqa: PLC0415 - only the keyed path needs it

    name = spec.option("api_key_env") or OPENAI_KEY_ENV
    key = (os.environ.get(name) or "").strip()
    if not key:
        raise ConfigError(f"{name} is not set, and the openai provider cannot dial without it")

    # `num_ctx` is deliberately not sent. OpenAI serves the model's own window
    # and has no equivalent knob; the number still governs `prompt_budget`, so
    # an operator raising `SWARM_WORKER_CTX` on this provider is raising how
    # much prompt is *built*, which is the half that costs money.
    return ChatOpenAI(
        model=spec.model,
        temperature=spec.temperature,
        api_key=SecretStr(key),
        base_url=spec.option("base_url") or None,
        callbacks=callbacks,
    )


def _structured_openai(llm: Any, schema: type[BaseModel], spec: ModelSpec | None = None) -> Any:
    """Force JSON matching `schema`, by strict server-side schema adherence.

    `method="json_schema", strict=True` is the mechanism, and both arguments
    are passed explicitly rather than left to the library's defaults: they have
    moved before, and this is the guarantee the whole orchestrator rests on.
    Under it a violation is *raised* by the parser - the strict-outputs parser
    builds the model with `schema(**parsed)`, and a wrong shape is a pydantic
    `ValidationError` - which is the same observable contract the Ollama path
    gets from constrained decoding, reached the other way round.
    """
    return llm.with_structured_output(schema, method="json_schema", strict=True)


def _rejections_openai() -> tuple[type[BaseException], ...]:
    """What "the reply was not the schema" looks like on the OpenAI path.

    Two types. `ValidationError` is a parsed object that did not fit. And
    `LengthFinishReasonError` is the completion that stopped mid-token, which
    is one of the two failure classes #259 measured at roughly a 40% rate
    locally; it is exactly what the single retry exists for, and without it
    named here the remote path would give up on the failure most worth
    retrying.

    Two absences, both deliberate. `OpenAIRefusalError` is a decision, not a
    truncation, and a second identical decode buys another refusal at the price
    of a whole-file generation. And the parser's own bare `ValueError` - "no
    `parsed` field nor `refusal` field" - is *not* named, because
    `ValidationError` subclasses `ValueError` and naming the base would make
    every `ValueError` raised anywhere in this system a retryable parse
    failure. Losing one retry on a rare malformed envelope is much cheaper than
    retrying a `ConfigError` on a whole-file generation.
    """
    from pydantic import ValidationError  # noqa: PLC0415 - lazy, like the rest

    types: list[type[BaseException]] = [ValidationError]
    try:
        from openai import LengthFinishReasonError  # noqa: PLC0415 - optional extra
    except ImportError:  # pragma: no cover - only without the extra installed
        return tuple(types)
    types.append(LengthFinishReasonError)
    return tuple(types)


def _credential_openai(spec: ModelSpec) -> str:
    return f"${spec.option('api_key_env') or OPENAI_KEY_ENV}"


# --- bedrock --------------------------------------------------------------

#: The variables boto3 itself reads, in its own order. Named here so that "no
#: region" can be a `ConfigError` naming the three ways to fix it rather than a
#: `NoRegionError` from inside botocore.
AWS_REGION_ENV: tuple[str, ...] = ("AWS_REGION", "AWS_DEFAULT_REGION")
AWS_PROFILE_ENV: tuple[str, ...] = ("AWS_PROFILE", "AWS_DEFAULT_PROFILE")


def _build_bedrock(spec: ModelSpec, callbacks: list | None) -> Any:
    """A Bedrock model, reached through an AWS profile and a region.

    Neither is read from a key: Bedrock authenticates through the ordinary AWS
    credential chain, so the "credential" here is a profile name plus wherever
    boto3 finds its keys - an SSO cache, an instance role, an environment. That
    is a materially different credential shape from an API key, and #269 is
    where it gets examined before any of it reaches a worker container.

    Both options fall through to boto3's own environment when unset, which is
    what makes a machine already configured for AWS need no apiary
    configuration at all.
    """
    try:
        from langchain_aws import ChatBedrockConverse  # noqa: PLC0415 - lazy, and optional
    except ImportError as exc:  # pragma: no cover - only without the extra installed
        raise _missing(BEDROCK, "langchain-aws", exc) from exc

    region = spec.option("region") or _required_env(AWS_REGION_ENV)
    if not region:
        raise ConfigError(
            "bedrock needs a region, and neither the spec nor the environment names "
            f"one; set the spec's `region` option or export {' or '.join(AWS_REGION_ENV)}"
        )
    profile = spec.option("profile") or _required_env(AWS_PROFILE_ENV)

    try:
        return ChatBedrockConverse(
            model=spec.model,
            temperature=spec.temperature,
            region_name=region,
            credentials_profile_name=profile or None,
            callbacks=callbacks,
        )
    except Exception as exc:  # noqa: BLE001 - see below; every failure here is config
        # Bedrock resolves its credentials *at construction*, unlike the other
        # two providers, and `langchain-aws` reports a missing profile as a
        # pydantic `ValidationError` from inside its field validator. Two
        # reasons that must not escape as itself: it reads as a bug in apiary
        # rather than as an unconfigured profile, and `ValidationError` is one
        # of the types `parse_failure` treats as a retryable bad reply - so an
        # unbuildable client would be retried as though the model had answered.
        raise ConfigError(
            f"bedrock could not authenticate as {_credential_bedrock(spec)}: {exc}"
        ) from exc


def _structured_bedrock(llm: Any, schema: type[BaseModel], spec: ModelSpec | None = None) -> Any:
    """Force JSON matching `schema`, by Bedrock's structured-output config.

    `method` is passed explicitly and defaults to `json_schema` here, which is
    *not* `langchain-aws`'s own default - it ships `function_calling`, and a
    tool call the model chose not to make returns `None` rather than raising,
    which is precisely the silent coercion this module exists to prevent.

    It stays an option rather than a constant because not every model on
    Bedrock serves `json_schema`, and the spike that measures which is the one
    thing here that needs a live account. An operator whose model refuses it
    sets `method=function_calling` on the spec and gets the older behaviour
    without a code change.
    """
    method = (spec.option("method") if spec is not None else "") or "json_schema"
    return llm.with_structured_output(schema, method=method)


def _credential_bedrock(spec: ModelSpec) -> str:
    profile = spec.option("profile") or _required_env(AWS_PROFILE_ENV)
    region = spec.option("region") or _required_env(AWS_REGION_ENV) or "no region"
    return f"aws profile {profile!r} in {region}" if profile else f"the default AWS chain in {region}"


PROVIDERS: dict[str, Provider] = {
    OLLAMA: Provider(
        name=OLLAMA,
        build=_build_ollama,
        structured=_structured_ollama,
        rejections=_rejections_langchain,
        client_names=("ChatOllama",),
        credential=_credential_ollama,
        options=(
            Option("base_url", doc="where the server is; defaults to Settings.ollama_base_url"),
        ),
    ),
    OPENAI: Provider(
        name=OPENAI,
        build=_build_openai,
        structured=_structured_openai,
        rejections=_rejections_openai,
        client_names=("ChatOpenAI",),
        credential=_credential_openai,
        options=(
            Option("api_key_env", default=OPENAI_KEY_ENV, doc="which variable holds the key",
                   secret_env=True),
            Option("base_url", doc="an OpenAI-compatible endpoint other than OpenAI's own"),
        ),
    ),
    BEDROCK: Provider(
        name=BEDROCK,
        build=_build_bedrock,
        structured=_structured_bedrock,
        # Bedrock's `json_schema` path ends in a `PydanticOutputParser`, the
        # same component the Ollama path ends in, so it rejects the same way.
        rejections=_rejections_langchain,
        client_names=("ChatBedrockConverse",),
        credential=_credential_bedrock,
        options=(
            Option("profile", doc="the AWS profile to authenticate with"),
            Option("region", doc="the AWS region the model is served from"),
            Option("method", default="json_schema",
                   doc="how the schema is forced: json_schema, or function_calling"),
        ),
    ),
}


def provider_for(llm: Any) -> Provider:
    """Which provider built this model, answered from the object itself.

    Answered from the class name up the MRO rather than from a field on the
    model, because every client class here is a pydantic model that rejects
    attributes it does not declare - there is nowhere on them to stamp a spec.

    An unrecognised object is the Ollama provider, and that default is
    load-bearing rather than lazy: every test double in this suite is a plain
    object with `.invoke`, and they must keep composing exactly as they did
    before this module grew a registry.
    """
    names = {cls.__name__ for cls in type(llm).__mro__}
    for provider in PROVIDERS.values():
        if names & set(provider.client_names):
            return provider
    return PROVIDERS[OLLAMA]


# --------------------------------------------------------------------------
# The factories
# --------------------------------------------------------------------------


def _build(role: str, spec: ModelSpec | None) -> Any:
    """Resolve, announce once, build.

    `models` is imported here rather than at module scope because it imports
    *this* module for `ModelSpec` - and because the direction of the dependency
    is the point: resolution knows how a spec is chosen, this module knows how
    one is built, and only the second of those is allowed to import a provider
    SDK.
    """
    from .models import announce, resolve  # noqa: PLC0415 - see above

    resolution = resolve(role, spec)
    announce(role, resolution)
    chosen = resolution.spec
    return PROVIDERS[chosen.provider].build(
        chosen, _callbacks(role, chosen.model, chosen.provider)
    )


def orchestrator_llm(spec: ModelSpec | None = None):
    """Small, deterministic. Used for planning and progress judgement.

    `spec` is defaulted, so none of the existing call sites changed. It exists
    for the console firing one call at a chosen model (#266) - an argument
    rather than a mutable global, because `Settings` is frozen and the console
    runs inference on a background thread while serving HTTP, so a
    process-global written from a request handler would be a race against a run
    already in flight.
    """
    return _build("orchestrator", spec)


def worker_llm(spec: ModelSpec | None = None):
    """The code writer. Bigger is better here."""
    return _build("worker", spec)


def structured(llm: Any, schema: type[T], spec: ModelSpec | None = None):
    """Force JSON matching `schema`, by whichever mechanism this provider has.

    The mechanisms are genuinely different - see the module docstring - and this
    is the only place in the system that knows there is more than one.

    `spec` is a third defaulted argument rather than a change to the existing
    two, so the nine call sites and every test double stay exactly as they
    were. It carries the options a provider's structured path reads - Bedrock's
    `method` is the one that exists today - and its absence means "the
    provider's own default", which is what every current caller wants.
    """
    return provider_for(llm).structured(llm, schema, spec)


def parse_failure(exc: BaseException) -> bool:
    """Did the structured boundary reject the model's reply as unparseable?

    Answered here rather than at the call site because the answer is a set of
    framework and SDK exception types, and ADR 0003 keeps every such import
    inside this module: a caller that wants to retry a rejected reply (the
    worker does - an over-long prompt truncated by the server produces exactly
    this) asks the question without learning who did the rejecting.

    Asked of every provider rather than of the one that made the call, because
    the exception arrives at `worker/edit.py` with no model attached. A
    provider whose SDK is not importable contributes nothing and costs nothing,
    which is what keeps a local-only install from paying for a remote path's
    exception types.
    """
    for provider in PROVIDERS.values():
        try:
            types = provider.rejections()
        except ImportError:  # pragma: no cover - a provider whose SDK is absent
            continue
        if types and isinstance(exc, types):
            return True
    return False
