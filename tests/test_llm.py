"""ADR 0006, held still: the provider is a detail, and every path still raises.

Nothing here calls a model. Constructing a client opens no socket, and every
assertion below is about the object that comes back or about the parser it was
composed with - which is what lets the one property this epic actually rests on
be tested hermetically, on a machine with no key and no AWS account:

**a schema violation is raised, not silently coerced, on every path.**

That property is why `structured()` is the single function in `llm.py` that is
not a passthrough. Ollama reaches it by constraining the decoder, so a
malformed reply cannot be produced. OpenAI reaches it by strict server-side
schema adherence and a parser that constructs the pydantic model, so a
malformed reply is produced and then rejected. Bedrock reaches it through an
`output_config` and a `PydanticOutputParser` - but only under `json_schema`,
and the library's own default is `function_calling`, under which a tool call
the model declined to make returns `None`. Three mechanisms, three failure
modes, one observable contract - and the tests for it are written against the
*parser each path installs*, because that is where the difference lives and the
only part reachable without a server.

The second property, and the one that decays silently: `parse_failure()` knows
every path's rejection type. `worker/edit.py` retries a rejected reply exactly
once and asks this module whether to. A provider whose rejection type went
unnamed would not fail any test about retrying - the retry would simply stop
firing, and a truncated completion would read as a broken model.

**Why there is a `FakeBedrock` and no fake anything else.** Ollama and OpenAI
clients construct without reaching anything, so the real classes are used.
Bedrock resolves its credentials *at construction*, so a real client cannot be
built on a machine with no AWS configured - which is every CI runner this repo
has. The stub matches by class name, which is exactly how `provider_for`
dispatches, so what is under test is still the registry.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from swarm import llm as llm_mod
from swarm.config import ConfigError
from swarm.llm import BEDROCK, OLLAMA, OPENAI, ModelSpec


class Ping(BaseModel):
    """Two fields of different types, for the same reason `doctor.Ping` has
    two: `{"ok": "yes"}` is exactly what a model that ignored the constraint
    emits, and a single boolean is guessable."""

    ok: bool
    answer: int


@pytest.fixture
def key(monkeypatch):
    """A credential shaped like one, spent on nothing. `ChatOpenAI` refuses to
    construct without a key, and every test here stops before the socket."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")


# --------------------------------------------------------------------------
# The spec
# --------------------------------------------------------------------------


def test_a_spec_is_frozen_and_comparable():
    """`==` is how "did these two runs use the same model?" gets answered."""
    one = ModelSpec(provider=OLLAMA, model="gemma4:31b", num_ctx=16384)
    two = ModelSpec(provider=OLLAMA, model="gemma4:31b", num_ctx=16384)

    assert one == two
    with pytest.raises(Exception):
        one.model = "something-else"  # type: ignore[misc]


def test_a_spec_carries_the_name_of_a_credential_and_never_its_value():
    """The property that lets #265 save a spec to a file and #266 render one on
    a page, neither of which has a redaction story of its own."""
    spec = ModelSpec(provider=OPENAI, model="gpt-5.6-terra").with_options(
        api_key_env="OPENAI_API_KEY"
    )

    assert "OPENAI_API_KEY" in repr(spec)
    assert "sk-" not in repr(spec)


def test_options_are_normalised_so_two_ways_of_writing_one_spec_are_equal():
    """`==` is the whole point of the type; option order must not defeat it."""
    one = ModelSpec(provider=BEDROCK, model="gpt-5.6-luna").with_options(
        region="eu-west-1", profile="acme"
    )
    two = ModelSpec(provider=BEDROCK, model="gpt-5.6-luna").with_options(
        profile="acme", region="eu-west-1"
    )

    assert one == two
    assert {one, two} == {one}


def test_an_option_the_provider_does_not_declare_is_refused_by_name():
    """The failure this exists for is a typo that defaults quietly: a `regoin`
    silently ignored leaves a model served from somewhere else in the world,
    and nothing on the page or in the log would say so."""
    with pytest.raises(ConfigError) as caught:
        ModelSpec(provider=BEDROCK, model="gpt-5.6-luna", options=(("regoin", "eu-west-1"),))

    assert "regoin" in str(caught.value)
    assert "region" in str(caught.value)


def test_an_option_falls_back_to_the_providers_declared_default():
    assert ModelSpec(provider=BEDROCK, model="x").option("method") == "json_schema"
    assert ModelSpec(provider=OPENAI, model="x").option("api_key_env") == "OPENAI_API_KEY"


def test_a_label_is_provider_qualified_even_for_the_default():
    """The bare model name stopped being an unambiguous answer to "which model
    did this run use" the moment there was more than one provider."""
    assert ModelSpec(provider=OLLAMA, model="gemma4:26b").label == "ollama:gemma4:26b"
    assert ModelSpec(provider=OPENAI, model="gpt-5.6-terra").label == "openai:gpt-5.6-terra"
    assert ModelSpec(provider=BEDROCK, model="gpt-5.6-luna").label == "bedrock:gpt-5.6-luna"


def test_an_unknown_provider_is_refused_at_the_spec_and_names_the_known_ones():
    """A `ConfigError`, so `cli.main` renders it as one `!` line and exit 1 -
    the same treatment as every other setting an operator can fix by editing
    their environment."""
    with pytest.raises(ConfigError) as caught:
        ModelSpec(provider="vertex", model="anything")

    assert "vertex" in str(caught.value)
    assert "ollama" in str(caught.value) and "bedrock" in str(caught.value)


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


def test_ollama_is_registered_and_is_the_default():
    """The epic's non-goal, asserted: Ollama is not demoted to a fallback."""
    assert OLLAMA in llm_mod.PROVIDERS
    assert ModelSpec().provider == OLLAMA


@pytest.mark.parametrize("name", [OLLAMA, OPENAI, BEDROCK])
def test_every_provider_is_registered_behind_the_same_interface(name):
    """The claim ADR 0006 makes: a fourth provider is a registry entry, not a
    change to any shared type. It is only true if the interface is uniform."""
    provider = llm_mod.PROVIDERS[name]

    for attribute in ("build", "structured", "rejections", "credential"):
        assert callable(getattr(provider, attribute))
    assert provider.client_names


def test_each_provider_says_where_its_credential_comes_from_in_its_own_terms():
    """Three providers, three genuinely different credential shapes - none, an
    environment variable, an AWS profile and region. Doctor (#267) and the
    console (#266) both have to *say* this, and neither should have to know
    which shape it is this time."""
    assert "unauthenticated" in ModelSpec(provider=OLLAMA, model="x").credential
    assert ModelSpec(provider=OPENAI, model="x").credential == "$OPENAI_API_KEY"

    bedrock = ModelSpec(provider=BEDROCK, model="x").with_options(
        profile="acme", region="eu-west-1"
    )
    assert bedrock.credential == "aws profile 'acme' in eu-west-1"


def test_a_credential_summary_is_still_never_the_credential():
    """A profile name and a variable name are both safe to print. A key is not,
    and nothing here has one to leak."""
    for spec in (
        ModelSpec(provider=OPENAI, model="x"),
        ModelSpec(provider=BEDROCK, model="x").with_options(profile="acme", region="eu-west-1"),
    ):
        assert "sk-" not in spec.credential
        assert "AKIA" not in spec.credential


# --------------------------------------------------------------------------
# The factories
# --------------------------------------------------------------------------


def test_the_factories_still_build_ollama_from_settings_with_no_spec():
    """The defaulted argument changed none of the nine existing call sites, and
    a clone with nothing configured still runs fully local."""
    from swarm.config import SETTINGS

    orchestrator, worker = llm_mod.orchestrator_llm(), llm_mod.worker_llm()

    assert type(orchestrator).__name__ == "ChatOllama"
    assert orchestrator.model == SETTINGS.orchestrator_model
    assert worker.model == SETTINGS.worker_model
    assert orchestrator.base_url == SETTINGS.ollama_base_url


def test_a_spec_argument_builds_what_it_says(key):
    """The console firing one call at a chosen model, which is the thing this
    epic is for. An argument rather than a mutable global: `Settings` is frozen
    and the console runs inference on a background thread while serving HTTP."""
    built = llm_mod.orchestrator_llm(ModelSpec(provider=OPENAI, model="gpt-5.6-terra"))

    assert type(built).__name__ == "ChatOpenAI"
    assert built.model_name == "gpt-5.6-terra"


def test_a_spec_argument_does_not_change_the_next_call_without_one():
    """"Try it before I commit" must leave nothing behind - #266 keeps the two
    controls distinct, and this is the half of that which lives down here."""
    from swarm.config import SETTINGS

    llm_mod.orchestrator_llm(ModelSpec(provider=OLLAMA, model="something-else"))

    assert llm_mod.orchestrator_llm().model == SETTINGS.orchestrator_model


def test_a_remote_model_without_its_credential_is_refused_before_the_socket(monkeypatch):
    """"No key" is a `ConfigError` an operator can act on, not a 401 from a
    vendor arriving three steps into something. #267 makes doctor say the same
    thing before a run starts."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigError) as caught:
        llm_mod.worker_llm(ModelSpec(provider=OPENAI, model="gpt-5.6-terra"))

    assert "OPENAI_API_KEY" in str(caught.value)


# --------------------------------------------------------------------------
# structured() - the one function that is not a passthrough
# --------------------------------------------------------------------------


def _parser(chain):
    """The last step of the composed chain: the component the two paths differ
    in, and the only one reachable without a server."""
    return chain.steps[-1]


def test_the_ollama_path_raises_on_a_schema_violation_rather_than_coercing():
    chain = llm_mod.structured(llm_mod.orchestrator_llm(), Ping)

    from langchain_core.messages import AIMessage

    with pytest.raises(Exception) as caught:
        _parser(chain).invoke(AIMessage(content='{"ok": "yes", "answer": "seven"}'))

    assert llm_mod.parse_failure(caught.value), (
        "the ollama path rejected the reply with a type `parse_failure` does not "
        "know, so `worker/edit.py`'s single retry would stop firing"
    )


def test_the_openai_path_raises_on_a_schema_violation_rather_than_coercing(key):
    """The failure mode strict `json_schema` has and constrained decoding does
    not: the reply is produced, and then rejected when the parser constructs
    the model out of it."""
    chain = llm_mod.structured(
        llm_mod.orchestrator_llm(ModelSpec(provider=OPENAI, model="gpt-5.6-terra")), Ping
    )

    from langchain_core.messages import AIMessage

    violation = AIMessage(content="", additional_kwargs={"parsed": {"ok": "yes", "answer": "seven"}})
    with pytest.raises(ValidationError):
        _parser(chain).invoke(violation)


def test_the_openai_path_asks_for_strict_json_schema_explicitly(key):
    """Both arguments passed rather than left to the library's defaults: they
    have moved before - `method` defaulted to `function_calling` until
    recently - and this is the guarantee the whole orchestrator rests on."""
    chain = llm_mod.structured(
        llm_mod.orchestrator_llm(ModelSpec(provider=OPENAI, model="gpt-5.6-terra")), Ping
    )

    asked = chain.steps[0].kwargs["ls_structured_output_format"]["kwargs"]

    assert asked == {"method": "json_schema", "strict": True}


def test_an_unrecognised_model_composes_as_ollama_did():
    """Every test double in this suite is a plain object with `.invoke`. The
    fallback in `provider_for` is what keeps them composing exactly as before,
    and is load-bearing rather than lazy."""

    class Double:
        def with_structured_output(self, schema):
            return ("structured", schema)

    assert llm_mod.provider_for(Double()).name == OLLAMA
    assert llm_mod.structured(Double(), Ping) == ("structured", Ping)


# --------------------------------------------------------------------------
# parse_failure() - both frameworks' rejection types
# --------------------------------------------------------------------------


def test_parse_failure_knows_langchains_rejection():
    from langchain_core.exceptions import OutputParserException

    assert llm_mod.parse_failure(OutputParserException('Invalid json output: --- { "'))


def test_parse_failure_knows_a_strict_schema_violation():
    """The remote path's own "that was not the schema"."""
    with pytest.raises(ValidationError) as caught:
        Ping(ok="yes", answer="seven")  # type: ignore[arg-type]

    assert llm_mod.parse_failure(caught.value)


def test_parse_failure_knows_a_completion_that_stopped_mid_token():
    """One of the two failure classes #259 measured at roughly 40% locally, and
    the one the single retry is most worth spending on. Without this named, the
    remote path would give up on exactly the failure retrying fixes."""
    openai = pytest.importorskip("openai")

    truncated = openai.LengthFinishReasonError.__new__(openai.LengthFinishReasonError)

    assert llm_mod.parse_failure(truncated)


def test_a_refused_socket_is_still_never_a_parse_failure():
    """A refused socket, a missing model and a declined request fail the same
    on any number of tries, so they escape immediately - as before."""
    assert not llm_mod.parse_failure(RuntimeError("connection refused"))
    assert not llm_mod.parse_failure(ConfigError("OPENAI_API_KEY is not set"))


def test_a_model_that_declined_is_not_retried(key):
    """`OpenAIRefusalError` is deliberately absent from the rejection types. A
    refusal is a decision, not a truncation, and a second identical decode buys
    another refusal at the price of a whole-file generation."""
    from langchain_openai.chat_models.base import OpenAIRefusalError

    assert not llm_mod.parse_failure(OpenAIRefusalError("I can't help with that"))


# --------------------------------------------------------------------------
# Bedrock - the third provider, and the one with a different credential shape
# --------------------------------------------------------------------------


class FakeBedrock:
    """A stand-in whose *class name* is what `provider_for` matches on.

    Bedrock resolves credentials at construction, so a real client cannot be
    built on a machine with no AWS configured - which is every CI runner this
    repo has. Matching on the class name is what makes the dispatch testable
    anyway: this object is a `ChatBedrockConverse` as far as the registry is
    concerned, and the registry is what is under test.
    """

    def __init__(self) -> None:
        self.asked: dict = {}

    def with_structured_output(self, schema, **kwargs):
        self.asked = {"schema": schema, **kwargs}
        return self


FakeBedrock.__name__ = "ChatBedrockConverse"


def test_the_bedrock_path_forces_json_schema_rather_than_the_librarys_default():
    """`langchain-aws` ships `function_calling`, under which a tool call the
    model chose not to make returns `None` rather than raising - which is
    precisely the silent coercion this module exists to prevent."""
    client = FakeBedrock()

    llm_mod.structured(client, Ping)

    assert client.asked["method"] == "json_schema"


def test_the_bedrock_method_is_an_option_a_spec_can_override():
    """Not every model on Bedrock serves `json_schema`, and which ones do is
    the one question here that needs a live account. An operator whose model
    refuses it changes a spec, not this file."""
    client = FakeBedrock()
    spec = ModelSpec(provider=BEDROCK, model="gpt-5.6-luna").with_options(
        method="function_calling"
    )

    llm_mod.structured(client, Ping, spec)

    assert client.asked["method"] == "function_calling"


def test_bedrock_without_a_region_says_so_before_botocore_does(monkeypatch):
    """A `NoRegionError` from inside botocore moves the hour of confusion
    rather than removing it - doctor's own words about a check that reports on
    the wrong thing."""
    for name in llm_mod.AWS_REGION_ENV:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigError) as caught:
        llm_mod.worker_llm(ModelSpec(provider=BEDROCK, model="gpt-5.6-luna"))

    assert "region" in str(caught.value)
    assert "AWS_REGION" in str(caught.value)


def test_an_unusable_aws_profile_is_a_config_error_not_a_validation_error():
    """Two reasons this must not escape as `langchain-aws` raises it. It reads
    as a bug in apiary rather than as an unconfigured profile - and
    `ValidationError` is one of the types `parse_failure` treats as a
    retryable bad reply, so an unbuildable client would be retried as though
    the model had answered."""
    spec = ModelSpec(provider=BEDROCK, model="gpt-5.6-luna").with_options(
        profile="not-a-profile-on-any-machine", region="eu-west-1"
    )

    with pytest.raises(ConfigError) as caught:
        llm_mod.worker_llm(spec)

    assert not llm_mod.parse_failure(caught.value)
    assert "not-a-profile-on-any-machine" in str(caught.value)


def test_the_bedrock_json_schema_path_rejects_the_same_way_ollama_does():
    """Both end in a `PydanticOutputParser`, which is why one entry in the
    rejection table covers both - asserted rather than assumed, because the
    day `langchain-aws` swaps that parser is the day the worker's retry
    silently stops firing on this provider."""
    from langchain_core.output_parsers import PydanticOutputParser

    with pytest.raises(Exception) as caught:
        PydanticOutputParser(pydantic_object=Ping).invoke('{"ok": "yes", "answer": "seven"}')

    assert llm_mod.parse_failure(caught.value)
