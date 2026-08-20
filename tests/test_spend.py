"""What a run has spent, and the ceiling it stops at.

The assertion that matters most here is the first one: **a fully local run
reaches none of this.** Every safety rail in this system was shaped by
inference being free, and a spend ceiling that cost a local run anything -
overhead, a ledger, a surprise halt - would have made the default path pay for
a feature it does not use.

The second is the one that is easy to get subtly wrong: a model with no known
price still has its *tokens* counted, and its cost is reported as a floor
rather than as a confident `$0.00`. A zero that means "not priced" is a number
somebody will believe, and it is the same mistake #268 refused to make with
timings.
"""

from __future__ import annotations

import pytest

from swarm.config import ConfigError
from swarm.spend import (
    CEILING_TOKENS_ENV,
    CEILING_USD_ENV,
    PRICES_ENV,
    Price,
    Spend,
    accounts,
    ledger,
    prices,
    record,
    set_ledger,
)

TERRA = "openai:gpt-5.6-terra"
LUNA = "bedrock:gpt-5.6-luna"
LOCAL = "ollama:gemma4:26b"


@pytest.fixture(autouse=True)
def no_installed_ledger():
    """Nothing leaks between tests through the process-wide ledger."""
    previous = set_ledger(None)
    yield
    set_ledger(previous)


@pytest.fixture
def table():
    return {TERRA: Price(2.0, 12.0), LUNA: Price(0.2, 1.2)}


# --------------------------------------------------------------------------
# A local run is unaffected
# --------------------------------------------------------------------------


def test_a_local_call_is_never_accounted_and_builds_no_ledger():
    """"No accounting overhead worth measuring" is not a claim about speed - it
    is that nothing is constructed at all."""
    record("ollama", LOCAL, {"input_tokens": 10_000, "output_tokens": 5_000})

    assert ledger() is None


def test_no_accountant_is_attached_to_a_local_provider():
    assert accounts("ollama") is False
    assert accounts("openai") is True
    assert accounts("bedrock") is True


def test_a_local_only_run_has_no_ceiling_to_reach():
    """`exceeded()` is asked once per round. With no ledger there is nothing to
    ask, which is what keeps the local path free of the whole feature."""
    from swarm.orchestrator.goal import _spend_exceeded

    assert _spend_exceeded() is None


# --------------------------------------------------------------------------
# Accumulation
# --------------------------------------------------------------------------


def test_input_and_output_are_priced_separately(table):
    """#259 costed Terra at $2 in and $12 out. A worker that emits whole files
    spends most of its money on the second number, so averaging them would
    misprice exactly the role this epic exists for."""
    spend = Spend()

    spend.add(TERRA, 1_000_000, 1_000_000, table)

    assert spend.usd == pytest.approx(14.0)


def test_a_ledger_breaks_the_total_down_by_model(table):
    """"Which role spent it" is the question a mixed run raises, and the answer
    is not derivable from a single total."""
    spend = Spend()

    spend.add(TERRA, 1000, 500, table)
    spend.add(LUNA, 2000, 400, table)
    spend.add(TERRA, 1000, 500, table)

    assert spend.by_model[TERRA] == (2000, 1000)
    assert spend.by_model[LUNA] == (2000, 400)
    assert spend.tokens == 5400


def test_an_unpriced_model_still_has_its_tokens_counted(table):
    """The bug this design exists to avoid: a confident `$0.00` for a model
    nobody has a rate for."""
    spend = Spend()

    spend.add("bedrock:something-new", 1000, 1000, table)

    assert spend.tokens == 2000
    assert spend.usd == 0.0
    assert spend.unpriced_tokens == 2000
    assert spend.priced is False, "an unpriced model must mark the dollar figure a floor"


def test_a_call_that_reported_nothing_is_not_recorded(table):
    spend = Spend()

    spend.add(TERRA, 0, 0, table)

    assert spend.by_model == {}


# --------------------------------------------------------------------------
# The ceiling
# --------------------------------------------------------------------------


def test_the_ceiling_halts_rather_than_degrades(table):
    """It stops the run. It does not silently switch model, shrink the context
    or reduce parallelism - a run that quietly became a different run would be
    worse than one that stopped."""
    spend = Spend(ceiling_usd=1.0)

    spend.add(TERRA, 0, 100_000, table)   # $1.20

    assert spend.exceeded() is not None


def test_the_halt_reads_like_the_round_cap_it_sits_beside(table):
    """A first-class outcome an operator can read - not a crash and not a
    silent stop. It is rendered next to "the follow-up budget is spent after 8
    round(s)" and has to sound like it."""
    spend = Spend(ceiling_usd=1.0)
    spend.add(TERRA, 0, 100_000, table)

    reason = spend.exceeded()

    assert "spend ceiling" in reason
    assert "$1.20" in reason and "$1.00" in reason
    assert CEILING_USD_ENV in reason, "the halt must name the knob that raises it"


def test_a_run_under_the_ceiling_carries_on(table):
    spend = Spend(ceiling_usd=5.0)

    spend.add(LUNA, 100_000, 100_000, table)   # $0.14

    assert spend.exceeded() is None


def test_the_token_ceiling_binds_when_a_price_is_unknown(table):
    """A ceiling you cannot compute is not a ceiling. An unpriced model would
    otherwise accrue for ever against a dollar limit it can never reach."""
    spend = Spend(ceiling_usd=5.0, ceiling_tokens=1000)

    spend.add("bedrock:something-new", 600, 600, table)

    reason = spend.exceeded()
    assert reason is not None
    assert "token ceiling" in reason
    assert "floor rather than a total" in reason
    assert CEILING_TOKENS_ENV in reason


def test_a_ceiling_of_zero_is_off_rather_than_immediately_exceeded():
    """`>= 0` would halt every run before its first call. An operator who
    writes 0 means "no limit", which is the only reading that is useful."""
    spend = Spend(ceiling_usd=0.0, ceiling_tokens=0)

    spend.add(TERRA, 1_000_000, 1_000_000)

    assert spend.exceeded() is None


# --------------------------------------------------------------------------
# What an operator watches
# --------------------------------------------------------------------------


def test_the_running_total_is_announced_but_not_once_per_call(table, capsys):
    """A line per generation would bury the run's own output; none at all would
    leave the console's run view empty while the run is in flight."""
    spend = Spend(ceiling_usd=10.0, ceiling_tokens=0)

    for _ in range(20):
        spend.add(TERRA, 0, 10_000, table)   # $0.12 each, $2.40 in total

    printed = [line for line in capsys.readouterr().out.splitlines() if "spend:" in line]
    assert 1 <= len(printed) <= 4, printed


def test_the_announced_line_is_the_one_the_console_parses(table):
    """One format string, because something reads it back."""
    from swarm.console_runs import RunJob

    spend = Spend(ceiling_usd=5.0)
    spend.add(TERRA, 100_000, 100_000, table)

    job = RunJob(id="x", command=["swarm"], started=0.0)
    job.absorb(spend.line())

    assert job.progress["spend"]["usd"] == pytest.approx(round(spend.usd, 2), abs=0.01)
    assert job.progress["spend"]["tokens"] == spend.tokens


def test_the_console_is_told_when_the_figure_is_only_a_floor(table):
    """Carried through to the page rather than rounded away, for the same
    reason #268 refused to record a zero that meant "not reported"."""
    from swarm.console_runs import RunJob

    spend = Spend(ceiling_usd=5.0)
    spend.add("bedrock:something-new", 100_000, 100_000, table)

    job = RunJob(id="x", command=["swarm"], started=0.0)
    job.absorb(spend.line())

    assert job.progress["spend"]["floor"] is True


def test_a_local_run_leaves_the_console_with_nothing_to_show():
    """The pill appearing at all is the signal that a run is costing money."""
    from swarm.console_runs import RunJob

    job = RunJob(id="x", command=["swarm"], started=0.0)
    job.absorb("» cycle 3: two workers dispatched")

    assert job.progress["spend"] is None


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------


def test_a_rate_can_be_stated_for_a_model_the_table_does_not_carry():
    table = prices({PRICES_ENV: "bedrock:some-model=1.5/7.5"})

    assert table["bedrock:some-model"] == Price(1.5, 7.5)


def test_an_unreadable_rate_is_refused_rather_than_ignored():
    """Every other reader on the inference path in this codebase is forgiving
    and this one is not, on purpose: a mistyped rate that fell back to
    "unknown" would silently move the run from the dollar ceiling to the token
    one - the failure this module exists to prevent, arrived at by a typo."""
    for bad in ("bedrock:m=1.5", "bedrock:m", "bedrock:m=a/b"):
        with pytest.raises(ConfigError):
            prices({PRICES_ENV: bad})


def test_a_stated_rate_overrides_the_built_in_one():
    """Vendors change rates, and this file is not where a run should learn last
    quarter's price."""
    table = prices({PRICES_ENV: f"{TERRA}=1.0/2.0"})

    assert table[TERRA] == Price(1.0, 2.0)


# --------------------------------------------------------------------------
# Parallelism, which was sized against a machine that no longer binds
# --------------------------------------------------------------------------


def test_a_remote_worker_is_not_capped_by_ollamas_own_limits():
    """`OLLAMA_NUM_PARALLEL` describes a server this run is not calling.
    Reporting a cap "bound by inference slots" would send an operator to change
    a setting that cannot affect anything."""
    from swarm.orchestrator.dispatcher import Capacity

    local = Capacity(slots=8, provider="ollama")
    remote = Capacity(slots=8, provider="bedrock")

    assert "inference slots" in local.bounds
    assert "inference slots" not in remote.bounds


def test_the_summary_says_what_binds_instead():
    """Said out loud rather than left to be inferred from a missing bound."""
    from swarm.orchestrator.dispatcher import Capacity

    summary = Capacity(slots=8, provider="bedrock").summary()

    assert "quota and the spend ceiling" in summary
    assert "GPU memory" in summary


def test_container_memory_still_binds_on_a_remote_worker():
    """A worker is still a container on this host; only its model stopped
    being. Dropping this bound too would overcommit the machine."""
    from swarm.orchestrator.dispatcher import Capacity

    assert "container memory" in Capacity(slots=8, provider="bedrock", memory=1).bounds
