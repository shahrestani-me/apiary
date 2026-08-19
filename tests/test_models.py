"""The four rungs, and the one that must never be climbed over.

The order is argument, environment, saved default, built-in. There is a test
per rung, and a test for the whole ladder in one - because the rungs are easy
to get individually right and collectively wrong, and the failure is silent:
every model still builds, just not the one anybody chose.

**The rung that matters is the environment beating the saved default.** A
worker container receives `SWARM_WORKER_MODEL` through
`containers.manager.INHERITED_ENV`, and a CI job exports it. Neither may be
overridden by whatever an operator last clicked in a console session. That is
asserted here against the real `INHERITED_ENV` rather than against a copy of
it, because a copy is a second thing to keep true.

Nothing here touches the real store: every test that reads or writes one hands
in a `ModelStore` under `tmp_path`, and every test that reads the environment
hands in a mapping. The two seams exist for this, and the default arguments are
what production uses.
"""

from __future__ import annotations

import json

import pytest

from swarm.config import (
    DEFAULT_ORCHESTRATOR_MODEL,
    DEFAULT_WORKER_MODEL,
    ConfigError,
)
from swarm.llm import BEDROCK, OLLAMA, OPENAI, ModelSpec
from swarm.models import (
    ARGUMENT,
    BUILT_IN,
    ENVIRONMENT,
    SAVED,
    ModelStore,
    announce,
    as_dict,
    forget_announcements,
    from_dict,
    parse_model,
    parse_options,
    resolve,
)


@pytest.fixture
def sink(tmp_path):
    """A store nobody else can see."""
    return ModelStore(path=tmp_path / "models.json")


@pytest.fixture
def bare():
    """An environment with none of the variables set."""
    return {}


LUNA = ModelSpec(provider=BEDROCK, model="gpt-5.6-luna").with_options(
    region="eu-west-1", profile="acme"
)


# --------------------------------------------------------------------------
# Reading a spec out of a string
# --------------------------------------------------------------------------


def test_a_bare_model_name_still_means_ollama():
    """The whole of this module's backward compatibility. Nobody's shell
    profile or compose file breaks."""
    assert parse_model("gemma4:31b") == (OLLAMA, "gemma4:31b")
    assert parse_model("llama3") == (OLLAMA, "llama3")


def test_a_registered_provider_prefix_qualifies_the_name():
    assert parse_model("bedrock:gpt-5.6-luna") == (BEDROCK, "gpt-5.6-luna")
    assert parse_model("openai:gpt-5.6-terra") == (OPENAI, "gpt-5.6-terra")
    assert parse_model("ollama:gemma4:31b") == (OLLAMA, "gemma4:31b")


def test_a_colon_in_an_ollama_tag_is_not_read_as_a_provider():
    """The ambiguity this rule exists for. "Split on the first colon" alone
    reads `gemma4` - the default worker model - as a provider name, so the
    prefix is *looked up* rather than assumed."""
    assert parse_model("gemma4:26b") == (OLLAMA, "gemma4:26b")
    assert parse_model("mistral:7b-instruct") == (OLLAMA, "mistral:7b-instruct")


def test_options_are_read_as_name_value_pairs():
    assert parse_options("region=eu-west-1,profile=acme") == (
        ("region", "eu-west-1"),
        ("profile", "acme"),
    )
    assert parse_options("") == ()
    assert parse_options("  region = eu-west-1 ,, ") == (("region", "eu-west-1"),)


def test_an_unreadable_option_is_refused_rather_than_shrugged_at():
    """An operator who wrote `region eu-west-1` has a model pointed somewhere
    they did not choose. All the value is in this happening before the call."""
    with pytest.raises(ConfigError) as caught:
        parse_options("region eu-west-1")

    assert "name=value" in str(caught.value)


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


def test_a_saved_spec_survives_a_fresh_process(sink, tmp_path):
    """The requirement in one line: a running process cannot rewrite its own
    environment, so the console needs somewhere a *later* process reads."""
    sink.save("worker", LUNA)

    reopened = ModelStore(path=tmp_path / "models.json")

    assert reopened.load("worker") == LUNA


def test_saving_one_role_leaves_the_other_alone(sink):
    sink.save("worker", LUNA)
    sink.save("orchestrator", ModelSpec(provider=OLLAMA, model="gemma4:31b"))

    assert sink.load("worker") == LUNA
    assert sink.load("orchestrator").model == "gemma4:31b"


def test_a_saved_default_can_be_cleared(sink):
    sink.save("worker", LUNA)
    sink.save("worker", None)

    assert sink.load("worker") is None


def test_a_saved_spec_holds_no_credential(sink):
    """ADR 0006's decision 2, asserted where the file actually gets written."""
    sink.save("worker", LUNA)

    written = sink.path.read_text()

    assert "acme" in written and "eu-west-1" in written
    assert "sk-" not in written and "AKIA" not in written


def test_an_unreadable_store_costs_one_resolution_its_rung_not_a_run(sink, capsys, bare):
    """This sits on the inference path. A half-written or hand-edited file must
    not be able to end a run - it degrades to "there is no saved default",
    which is a rung the ladder already knows how to skip."""
    sink.path.write_text("{ not json")

    assert sink.load("worker") is None
    assert resolve("worker", env=bare, sink=sink).source == BUILT_IN
    assert "! models:" in capsys.readouterr().err


def test_a_saved_spec_naming_an_unknown_provider_is_ignored_loudly(sink, capsys, bare):
    """A `models.json` written by a later version, read by an earlier one."""
    sink.path.write_text(json.dumps({"schema": 1, "roles": {"worker": {"provider": "vertex"}}}))

    assert sink.load("worker") is None
    assert "! models:" in capsys.readouterr().err


def test_a_spec_round_trips_through_json():
    assert from_dict(as_dict(LUNA)) == LUNA


def test_saving_an_unknown_role_is_refused(sink):
    with pytest.raises(ConfigError):
        sink.save("judge", LUNA)


# --------------------------------------------------------------------------
# The ladder, one rung at a time
# --------------------------------------------------------------------------


def test_rung_one_an_explicit_argument_wins_over_everything(sink):
    """The console firing one call at a chosen model. Not persisted, and it
    must beat an environment that names something else."""
    sink.save("worker", ModelSpec(provider=OLLAMA, model="saved"))
    env = {"SWARM_WORKER_MODEL": "from-the-environment"}

    got = resolve("worker", LUNA, env=env, sink=sink)

    assert got.spec == LUNA
    assert got.source == ARGUMENT


def test_rung_two_the_environment_wins_over_a_saved_default(sink):
    """**The decision to get right.** A container run or a CI job must not
    silently inherit whatever someone clicked in the console last week."""
    sink.save("worker", ModelSpec(provider=OLLAMA, model="clicked-last-week"))
    env = {"SWARM_WORKER_MODEL": "bedrock:gpt-5.6-luna", "SWARM_WORKER_MODEL_OPTIONS": "region=eu-west-1"}

    got = resolve("worker", env=env, sink=sink)

    assert got.spec.provider == BEDROCK
    assert got.spec.model == "gpt-5.6-luna"
    assert got.spec.option("region") == "eu-west-1"
    assert got.source == ENVIRONMENT
    assert got.detail == "SWARM_WORKER_MODEL"


def test_rung_three_a_saved_default_wins_over_the_built_in(sink, bare):
    sink.save("orchestrator", LUNA)

    got = resolve("orchestrator", env=bare, sink=sink)

    assert got.spec == LUNA
    assert got.source == SAVED
    assert str(sink.path) in got.detail


def test_rung_four_a_clone_with_nothing_configured_runs_fully_local(sink, bare):
    """The epic's non-goal, asserted: Ollama is not demoted to a fallback that
    nobody tests."""
    orchestrator = resolve("orchestrator", env=bare, sink=sink)
    worker = resolve("worker", env=bare, sink=sink)

    assert orchestrator.source == BUILT_IN and worker.source == BUILT_IN
    assert orchestrator.spec.provider == OLLAMA and worker.spec.provider == OLLAMA
    assert orchestrator.spec.model == DEFAULT_ORCHESTRATOR_MODEL
    assert worker.spec.model == DEFAULT_WORKER_MODEL


def test_the_whole_ladder_in_one(sink):
    """Each rung removed in turn, and the next one takes over. The rungs are
    easy to get individually right and collectively wrong."""
    sink.save("worker", ModelSpec(provider=OLLAMA, model="saved"))
    env = {"SWARM_WORKER_MODEL": "from-env"}

    assert resolve("worker", LUNA, env=env, sink=sink).source == ARGUMENT
    assert resolve("worker", env=env, sink=sink).spec.model == "from-env"
    assert resolve("worker", env={}, sink=sink).spec.model == "saved"

    sink.save("worker", None)
    assert resolve("worker", env={}, sink=sink).spec.model == DEFAULT_WORKER_MODEL


def test_a_bare_environment_name_still_resolves_to_ollama(sink, bare):
    """Backward compatibility, at the rung it actually matters on."""
    got = resolve("worker", env={"SWARM_WORKER_MODEL": "gemma4:26b"}, sink=sink)

    assert got.spec.provider == OLLAMA
    assert got.spec.model == "gemma4:26b"


def test_an_options_variable_alone_does_not_take_the_rung(sink):
    """It names no model. Applying it to a spec that came from the store would
    be the environment silently *editing* the saved default rather than
    replacing it - a third behaviour nobody asked for and nobody could predict
    from the documented order."""
    sink.save("worker", LUNA)

    got = resolve("worker", env={"SWARM_WORKER_MODEL_OPTIONS": "region=us-east-1"}, sink=sink)

    assert got.source == SAVED
    assert got.spec.option("region") == "eu-west-1"


def test_an_exported_context_window_still_outranks_a_saved_one(sink):
    """`SWARM_WORKER_CTX` is its own rung-2 variable. A saved spec carries a
    window an operator chose in the console; an exported one is the same
    operator choosing again, later, from a shell that outranks it."""
    sink.save("worker", ModelSpec(provider=OLLAMA, model="gemma4:26b", num_ctx=16384))

    got = resolve("worker", env={"SWARM_WORKER_CTX": "32768"}, sink=sink)

    assert got.source == SAVED
    assert got.spec.num_ctx == 32768


def test_an_unresolvable_role_is_refused(sink, bare):
    with pytest.raises(ConfigError):
        resolve("judge", env=bare, sink=sink)


# --------------------------------------------------------------------------
# The container path keeps winning by construction
# --------------------------------------------------------------------------


def test_a_worker_containers_inherited_environment_beats_a_saved_default(sink):
    """Asserted, not assumed - the ticket's own words.

    Read off the real `INHERITED_ENV` rather than a copy of it: a copy is a
    second thing to keep true, and this test exists precisely because the two
    could drift apart without anything else noticing.
    """
    from swarm.containers.manager import INHERITED_ENV

    assert "SWARM_WORKER_MODEL" in INHERITED_ENV
    assert "SWARM_WORKER_MODEL_OPTIONS" in INHERITED_ENV

    sink.save("worker", ModelSpec(provider=OLLAMA, model="clicked-in-the-console"))
    inside_the_container = {
        name: value
        for name, value in {
            "SWARM_WORKER_MODEL": "bedrock:gpt-5.6-luna",
            "SWARM_WORKER_MODEL_OPTIONS": "region=eu-west-1,profile=acme",
        }.items()
        if name in INHERITED_ENV
    }

    got = resolve("worker", env=inside_the_container, sink=sink)

    assert got.source == ENVIRONMENT
    assert (got.spec.provider, got.spec.model) == (BEDROCK, "gpt-5.6-luna")
    assert got.spec.options == LUNA.options
    # And the role's own temperature, which the environment rung does not carry
    # and must not lose: the worker writes whole files at 0.1, the orchestrator
    # emits schema-constrained JSON at 0.0, and that split predates this epic.
    assert got.spec.temperature == 0.1


# --------------------------------------------------------------------------
# Saying which rung won
# --------------------------------------------------------------------------


def test_the_resolution_is_announced_once_per_role(capsys, sink, bare):
    """Once, because this is called from the model factory and a run makes
    hundreds of calls."""
    forget_announcements()
    resolution = resolve("worker", env=bare, sink=sink)

    announce("worker", resolution)
    announce("worker", resolution)

    assert capsys.readouterr().out.count("worker model:") == 1


def test_the_announcement_names_the_model_and_the_rung(capsys, sink, bare):
    """"Which model did this run actually use" answerable from a log rather
    than by reasoning about four rungs and somebody's shell."""
    forget_announcements()
    sink.save("worker", LUNA)

    announce("worker", resolve("worker", env=bare, sink=sink))

    printed = capsys.readouterr().out
    assert "bedrock:gpt-5.6-luna" in printed
    assert SAVED in printed
    assert str(sink.path) in printed


def test_an_announcement_never_prints_a_credential(capsys, sink, bare):
    forget_announcements()
    sink.save("worker", LUNA)

    announce("worker", resolve("worker", env=bare, sink=sink))

    assert "sk-" not in capsys.readouterr().out
