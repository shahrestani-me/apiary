"""What a run has spent, and the ceiling it stops at.

Every safety rail in this system is shaped by the fact that inference was free
and local. `max_workers_parallel` defaults to 2 because "Ollama allocates KV
cache as `OLLAMA_NUM_PARALLEL` x `num_ctx`, so raising this raises memory
linearly" (`config.py`), and `dispatcher.py` sizes concurrency against GPU
memory and `OLLAMA_MAX_LOADED_MODELS`. Remote, none of that binds: the
constraints become quota and money, and until this module nothing counted
either.

The existing round and attempt caps bound **effort**, which correlated with
cost while cost was zero. `max_total_attempts_per_task` deliberately renews a
task's budget when a retry fails *differently*, which is right for progress and
unbounded for spend. An unattended swarm looping on a paid endpoint is the
failure this module exists to prevent.

## Three decisions worth stating

**A local run pays nothing, including in overhead.** The accounting handler is
attached only for a paid provider, so an Ollama-only run reaches none of this -
no handler, no ledger, no ceiling. That is why the attachment lives in `llm.py`
beside capture's rather than being a wrapper around every call.

**It is not capture.** Capture records token counts too, and #268 made it do so
for every provider - but capture is off unless `APIARY_CAPTURE` is set, and a
spend ceiling that only worked while an operator was debugging would not be a
ceiling. This handler is attached on the *provider*, not on a flag.

**The halt is a first-class outcome, not an exception and not a silent stop.**
Raising from a callback would be swallowed by LangChain, and raising from the
call site would end a run in a traceback rather than a verdict. Instead the
ledger is *asked*, at the same point the round cap is asked, and answers in the
same shape - so "it stopped because it ran out of money" reads exactly like "it
stopped because it ran out of rounds", which is what an operator needs it to.

## What a price is, and what it is not

`PRICES` carries the published rates for the models this project has actually
been pointed at. A model that is not in the table still has its **tokens**
counted; only its cost is unknown, and the ledger says so rather than reporting
a confident `$0.00`. That distinction is the whole of `priced` below: a zero
that means "not priced" is a number somebody will believe, and it is the same
mistake #268 refused to make with timings.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping

from .config import ConfigError

__all__ = [
    "CEILING_TOKENS_ENV",
    "CEILING_USD_ENV",
    "DEFAULT_CEILING_TOKENS",
    "DEFAULT_CEILING_USD",
    "PRICES",
    "PRICES_ENV",
    "Price",
    "Spend",
    "accounts",
    "ledger",
    "prices",
    "record",
    "set_ledger",
]

#: USD per one million tokens, by `provider:model`. The two rates #259 costed
#: the epic against; anything else an operator adds through `PRICES_ENV`.
#:
#: Deliberately small. A price table that tried to be complete would be a table
#: that is wrong somewhere and believed everywhere - vendors change rates, and
#: this file is not where a run should learn last quarter's price. What it is
#: for is making the default configuration countable.
PRICES: dict[str, "Price"] = {}

#: `provider:model=in/out` pairs, comma separated. Where an operator states a
#: rate this file does not carry, in the same shape the table uses.
PRICES_ENV = "SWARM_MODEL_PRICES"

#: The ceiling, in dollars. A default rather than "off", because the whole
#: point is the unattended run, and an unattended run is exactly the one nobody
#: was there to set a limit for. Chosen against #259's own arithmetic: a full
#: run costed at roughly $2-3, so this stops the second one rather than the
#: first.
CEILING_USD_ENV = "SWARM_SPEND_CEILING_USD"
DEFAULT_CEILING_USD = 5.0

#: The other ceiling, and the one that binds when a model has no known price.
#: A ceiling you cannot compute is not a ceiling, so there are two, and
#: whichever is reached first halts the run.
CEILING_TOKENS_ENV = "SWARM_SPEND_CEILING_TOKENS"
DEFAULT_CEILING_TOKENS = 5_000_000

#: How often the running total is announced, as a fraction of the ceiling.
#: Every call would be one line per generation on a busy run; never would leave
#: the console's run view with nothing to show while the run is in flight,
#: which is the half of #270 an operator actually watches.
REPORT_STEP = 0.1

#: The line the console parses (`console_runs._SPEND_LINE`). One format string,
#: because something reads it back - the same contract `CYCLE_FINISHED` has.
SPEND_LINE = "» spend: {usd} · {tokens} tokens{ceiling}"


@dataclass(frozen=True)
class Price:
    """USD per one million tokens, in and out.

    Split, because they are not close: #259 costed Terra at $2 in and $12 out,
    and a worker that emits whole files spends most of its money on the second
    number. Averaging them would misprice exactly the role this epic exists for.
    """

    input_per_m: float
    output_per_m: float

    def usd(self, prompt_tokens: int, output_tokens: int) -> float:
        return (prompt_tokens * self.input_per_m + output_tokens * self.output_per_m) / 1_000_000


def _seed() -> dict[str, Price]:
    """The rates #259 costed the epic against."""
    return {
        "openai:gpt-5.6-terra": Price(2.0, 12.0),
        "openai:gpt-5.6-luna": Price(0.2, 1.2),
        "bedrock:gpt-5.6-terra": Price(2.0, 12.0),
        "bedrock:gpt-5.6-luna": Price(0.2, 1.2),
    }


PRICES.update(_seed())


def prices(env: Mapping[str, str] | None = None) -> dict[str, Price]:
    """The table, plus whatever `SWARM_MODEL_PRICES` states.

    Raises on an unreadable entry rather than ignoring it. Every other reader
    on the inference path in this codebase is forgiving, and this one is not on
    purpose: a mistyped rate that fell back to "unknown" would silently move
    the run from the dollar ceiling to the token one, which is the failure this
    module exists to prevent, arrived at by a typo.
    """
    source = os.environ if env is None else env
    table = dict(PRICES)
    for part in (source.get(PRICES_ENV) or "").split(","):
        item = part.strip()
        if not item:
            continue
        label, separator, rates = item.partition("=")
        rate_in, slash, rate_out = rates.partition("/")
        if not separator or not slash:
            raise ConfigError(
                f"{PRICES_ENV} entries are `provider:model=IN/OUT` in USD per million "
                f"tokens; could not read {item!r}"
            )
        try:
            table[label.strip()] = Price(float(rate_in), float(rate_out))
        except ValueError as exc:
            raise ConfigError(f"{PRICES_ENV}: {item!r} has a rate that is not a number") from exc
    return table


def _limit(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a number") from exc


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


@dataclass
class Spend:
    """What this run has spent, by model, and whether it may spend more.

    Not thread-safe by lock, and that is a considered choice rather than an
    oversight: the additions are `+=` on ints and floats under CPython, the
    dispatcher's workers make their model calls *inside containers* rather than
    in this process, and the cost of being one call behind is a ceiling that
    fires one generation late. A lock here would be protecting a number that is
    already approximate against a race that costs less than it does.
    """

    prompt_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    #: Tokens billed by a model with no known price. Kept apart from `usd`
    #: rather than folded in as zero - see the module docstring.
    unpriced_tokens: int = 0
    #: `provider:model` -> (prompt, output). What a per-model breakdown needs,
    #: and what makes "which role spent it" answerable.
    by_model: dict[str, tuple[int, int]] = field(default_factory=dict)
    ceiling_usd: float = field(default_factory=lambda: _limit(CEILING_USD_ENV, DEFAULT_CEILING_USD))
    ceiling_tokens: float = field(
        default_factory=lambda: _limit(CEILING_TOKENS_ENV, DEFAULT_CEILING_TOKENS)
    )
    #: How much had been announced when the last line was printed.
    _announced: float = 0.0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def priced(self) -> bool:
        """Whether every token counted here had a known rate.

        False means `usd` is a **floor**, not a total, and the token ceiling is
        what is actually holding the run.
        """
        return self.unpriced_tokens == 0

    def add(self, label: str, prompt_tokens: int, output_tokens: int,
            table: Mapping[str, Price] | None = None) -> None:
        """Record one call. `label` is a `ModelSpec.label`."""
        prompt_tokens, output_tokens = max(int(prompt_tokens), 0), max(int(output_tokens), 0)
        if not prompt_tokens and not output_tokens:
            return
        self.prompt_tokens += prompt_tokens
        self.output_tokens += output_tokens
        was = self.by_model.get(label, (0, 0))
        self.by_model[label] = (was[0] + prompt_tokens, was[1] + output_tokens)

        price = (prices() if table is None else table).get(label)
        if price is None:
            self.unpriced_tokens += prompt_tokens + output_tokens
        else:
            self.usd += price.usd(prompt_tokens, output_tokens)
        self._announce()

    # -- the verdict -----------------------------------------------------

    def exceeded(self) -> str | None:
        """Why this run must stop, or `None` to carry on.

        A sentence rather than a boolean, because it is rendered as a run's
        reason beside "exhausted after 8 round(s)" and has to read like one.
        """
        if self.ceiling_usd > 0 and self.usd >= self.ceiling_usd:
            return (
                f"the spend ceiling was reached: ${self.usd:.2f} of ${self.ceiling_usd:.2f} "
                f"({self.tokens:,} tokens). Raise {CEILING_USD_ENV} to continue"
            )
        if self.ceiling_tokens > 0 and self.tokens >= self.ceiling_tokens:
            unpriced = "" if self.priced else (
                f" - {self.unpriced_tokens:,} of them from a model with no known price, "
                f"so ${self.usd:.2f} is a floor rather than a total"
            )
            return (
                f"the token ceiling was reached: {self.tokens:,} of "
                f"{int(self.ceiling_tokens):,}{unpriced}. Raise {CEILING_TOKENS_ENV} to continue"
            )
        return None

    # -- what an operator watches ----------------------------------------

    def line(self) -> str:
        """One line, in the format `console_runs` parses back."""
        usd = f"${self.usd:.4f}" + ("" if self.priced else "+")
        ceiling = f" (ceiling ${self.ceiling_usd:.2f})" if self.ceiling_usd > 0 else ""
        return SPEND_LINE.format(usd=usd, tokens=f"{self.tokens:,}", ceiling=ceiling)

    def _announce(self) -> None:
        """Print on crossing each step of the ceiling, and never per call.

        A line per generation would bury the run's own output on a busy run;
        no line at all would leave the console's run view with nothing to show
        *while the run is in flight*, which is the half of this an operator
        actually watches.
        """
        reached = self.usd / self.ceiling_usd if self.ceiling_usd > 0 else 0.0
        if self.ceiling_tokens > 0:
            reached = max(reached, self.tokens / self.ceiling_tokens)
        if reached >= self._announced + REPORT_STEP:
            self._announced = reached - (reached % REPORT_STEP)
            print(self.line(), flush=True)

    def summary(self) -> dict[str, Any]:
        """The whole ledger, for a run's artifacts and the console."""
        return {
            "usd": round(self.usd, 4),
            "priced": self.priced,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "tokens": self.tokens,
            "unpriced_tokens": self.unpriced_tokens,
            "ceiling_usd": self.ceiling_usd,
            "ceiling_tokens": self.ceiling_tokens,
            "by_model": {label: {"prompt_tokens": p, "output_tokens": o}
                         for label, (p, o) in sorted(self.by_model.items())},
        }


#: Process-wide, and created on first paid call. Behind two functions for the
#: same reason `capture.recorder` is.
_LEDGER: Spend | None = None


def ledger() -> Spend | None:
    """This process's ledger, or `None` if nothing paid has been spent."""
    return _LEDGER


def set_ledger(replacement: Spend | None) -> Spend | None:
    """Install a ledger and return the one it replaced, for restoring."""
    global _LEDGER
    previous, _LEDGER = _LEDGER, replacement
    return previous


def _ensure() -> Spend:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = Spend()
    return _LEDGER


# --------------------------------------------------------------------------
# Recording a call
# --------------------------------------------------------------------------


def record(provider: str, label: str, usage: Mapping[str, Any] | None) -> None:
    """Add one call's billed tokens to this process's ledger.

    **Free for a local provider, in every sense**: it returns before touching
    anything, so an Ollama-only run never constructs a ledger.

    The *callback* that calls this lives in `llm.py`, not here, and that is
    ADR 0003 rather than taste: `BaseCallbackHandler` is a LangChain type, the
    framework's reach is deliberately countable at four modules, and adding a
    fifth to hold a class with one method would have widened it for nothing.
    This module stays arithmetic - which is also what makes it testable without
    constructing a model.

    `usage_metadata` is langchain's normalised shape and is the same three keys
    on every provider, which is why this needs none of #268's per-provider
    readers: token counts were the one thing that genuinely did not vary.
    """
    if provider == "ollama" or not isinstance(usage, Mapping):
        return
    try:
        _ensure().add(
            label,
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
        )
    except Exception as exc:  # noqa: BLE001 - never raise into the model call
        # Same rule as capture's: a failure here must not become a failure
        # there. The cost of a swallowed accounting error is an underestimate;
        # the cost of raising is a dead run.
        print(f"! spend: {type(exc).__name__}: {exc}", file=sys.stderr)


def accounts(provider: str) -> bool:
    """Whether calls to `provider` are worth attaching an accountant to."""
    return provider != "ollama"
