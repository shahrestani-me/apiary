"""The last question a run asks: was the *objective* met, or just the *plan*?

`docs/architecture-v2.md`'s loop ends when nothing is left in flight, and until
this module existed that was also where the run ended. The two are not the same
thing and the difference is the whole reason this file is here: the plan is one
model's first reading of an objective, written before a single line of the
repository existed, and a run that merges every task in it has proved that the
plan is finished - not that the objective is.

`swarm run --new "a trip planner"` shows the gap plainly. The planner emits
"implement the library" and "add a test suite", both merge, the ledger goes
quiet and the run exits reporting success - with no CLI, no way to actually plan
a trip, and an objective nobody would call met. The operator's only recourse is
to read the repository, notice what is missing, and start another run. That is
the manual step this module removes.

**This is not the replan, and the difference is which question failed.**
`replan.py` fires when a run has stopped *moving*: it rewrites the backlog,
because the arrangement of the tasks is what is wrong. This one fires when a run
has moved all the way to the end and arrived somewhere short of the objective:
it *appends*, because nothing about the finished work is wrong - there is simply
less of it than the objective asked for. So the write goes through
`planner.write_plan(retire_dropped=False)`, which creates the new issues and
leaves every existing one exactly as it is. A follow-up round that retired
anything would be closing work that already merged.

**Three grounds for refusing to ask the model at all**, and each of them is a
way that asking makes the run worse:

- **An empty ledger.** Nothing was planned, so nothing shipped, and the model
  would be asked to assess an objective against no evidence.
- **A task the swarm gave up on.** `swarm:failed` means the attempt budget is
  spent, or CI failed somewhere the worker may not edit, or nothing ever gated
  the pull request. Planning *more* work on top of that is stacking tasks onto a
  repository whose last known state is broken, and the follow-up would inherit
  the failure. The run stops and names the issue, which is what `checks.py`
  already decided a `swarm:failed` label means.
- **Work still in flight.** Called before the ledger is quiet, the assessment
  would judge an objective against a half-landed run. `Reconciler.loop` only
  calls it when nothing is live, and the guard is here as well because a
  future second caller must not be able to get this wrong quietly.

**A model that cannot be reached does not end the run and does not extend it.**
Same reading as `judge.Verdict.unresolved` and `docs/issue-contract.md` §4's
exit 2: an unreachable Ollama is an infrastructure fact, not an answer about the
objective. The run reports that it could not assess and stops, rather than
either declaring victory or planning follow-ups from a verdict nobody gave.

**The rounds are bounded, and the bound is small.** `MAX_ROUNDS` follow-up
rounds per run, defaulting to `replan.MAX_REPLANS`'s two, for the same reason
that one bounds replans: a model that has twice failed to close the gap it named
is not going to close it on the third pass, and each round costs a full
dispatch-verify-merge sweep of the swarm. What a third round would buy is a
tracker full of near-duplicate issues; what it costs is the operator's ability
to read it. When the budget runs out the run stops and prints what is still
missing, which is the sentence somebody needs in order to decide what to do
next.

**Nothing here judges code.** The model is asked whether the objective's goals
are covered by the goals that shipped, and its answer can only ever *add work*
to a tracker. It cannot merge, cannot pass a gate, and cannot mark anything
done - `## Verify` and CI remain the only authority on whether code is correct,
which `CONTRIBUTING.md` makes the first rule of this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from ..config import SETTINGS
from ..github.ledger import Ledger, LedgerEntry
from ..llm import orchestrator_llm, structured
from ..nodes.planner import (
    FOLLOWUP_SUFFIX,
    SYSTEM,
    IssueAction,
    PlanError,
    PlanReport,
    normalise,
    write_plan,
)
from ..run import TERMINAL_LABELS
from ..state import ObjectiveAssessment, Plan

#: How many follow-up rounds one run may plan for itself. See the docstring;
#: this is `replan.MAX_REPLANS`'s bound applied to the other end of the loop,
#: and it is deliberately not derived from it - a run that stalled twice and a
#: run that fell short twice are different failures with the same cheap answer.
MAX_ROUNDS = 2

#: How much of a shipped task's goal the model is shown. Enough to recognise
#: what the task did, short enough that twenty of them do not crowd the
#: objective out of the context of a 31B model.
GOAL_CHARS = 160

#: The refusals, as the sentences they are reported with. Strings rather than an
#: enum for `checks.py`'s reason: they are printed far more often than matched,
#: and "the swarm did nothing" and "the swarm decided against doing something"
#: are indistinguishable from the outside unless the reason travels.
EMPTY = "the ledger is empty; there is nothing to assess an objective against"
IN_FLIGHT = "work is still in flight; the objective is not assessed mid-run"
FAILED = "a task was abandoned; a human decides what happens before more is planned"
EXHAUSTED = "the follow-up budget is spent"
NO_TASKS = "the follow-up plan normalised to no writable task"
MET = "the objective is met"


class Oracle(Protocol):
    """The one model call `assess` makes. Injectable for the same reason
    `replan.Proposer` is: the tests must be able to drive every branch of this
    module without an Ollama and without a tracker."""

    def invoke(self, messages: Sequence[tuple[str, str]]) -> ObjectiveAssessment: ...


class Proposer(Protocol):
    """The one model call `follow_up` makes."""

    def invoke(self, messages: Sequence[tuple[str, str]]) -> Plan: ...


SYSTEM_ASSESS = """You decide whether a software objective has been met.

You are shown an objective and the list of work that has already shipped to the
default branch. Every shipped task passed its own verify command and CI.

Judge COVERAGE, not code quality: does the shipped work, taken together,
deliver everything the objective asks for? You cannot see the code and must not
guess at its quality - the test suite already answered that question.

Answer objective_met=false only when something the objective plainly asks for
has no shipped task behind it, and then list each such thing in `missing`, one
short line each. Do not invent polish, hardening or extra features the
objective did not ask for.

Return JSON only."""


# --------------------------------------------------------------------------
# Assessing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Assessment:
    """Whether the objective is met, and what is missing if it is not."""

    met: bool = False
    missing: tuple[str, ...] = ()
    reason: str = ""
    #: A model was asked. False on every arithmetic path, and worth reporting
    #: for `judge.Verdict.consulted`'s reason: it is one ~6.7 s model swap.
    consulted: bool = False
    #: A model was asked and could not answer. Not the same as "not met" - see
    #: the module docstring - and the caller must not plan from it.
    unresolved: bool = False
    #: The issues that stopped this being assessed at all, if any.
    abandoned: tuple[int, ...] = ()

    @property
    def actionable(self) -> bool:
        """May a follow-up round be planned from this? Only a real, negative answer."""
        return not self.met and not self.unresolved and not self.abandoned

    def summary(self) -> str:
        parts = [f"met={self.met}", "model" if self.consulted else "arithmetic"]
        if self.unresolved:
            parts.append("assessment unavailable")
        if self.abandoned:
            parts.append("abandoned: " + ", ".join(f"#{n}" for n in self.abandoned))
        if self.missing:
            parts.append(f"{len(self.missing)} gap(s)")
        return f"{', '.join(parts)}: {self.reason}"


def shipped(ledger: Ledger) -> tuple[LedgerEntry, ...]:
    """The tasks that reached `swarm:done`, in issue order.

    `swarm:done` and nothing else. An issue a human closed by hand is read as
    done by the reconciler and carries the label by the time this runs, so the
    two agree; an issue that is merely closed and never relabelled is not
    evidence that its work landed.
    """
    entries = (entry for entry in ledger.entries.values() if entry.state_label == "swarm:done")
    return tuple(sorted(entries, key=lambda entry: entry.number))


def abandoned(ledger: Ledger) -> tuple[LedgerEntry, ...]:
    """The tasks the swarm gave up on, in issue order."""
    entries = (entry for entry in ledger.entries.values() if entry.state_label == "swarm:failed")
    return tuple(sorted(entries, key=lambda entry: entry.number))


def live(ledger: Ledger) -> tuple[LedgerEntry, ...]:
    """Anything not terminal. `run.TERMINAL_LABELS` decides, as it does for resume."""
    entries = (
        entry for entry in ledger.entries.values() if entry.state_label not in TERMINAL_LABELS
    )
    return tuple(sorted(entries, key=lambda entry: entry.number))


def _catalogue(entries: Sequence[LedgerEntry]) -> str:
    """The shipped work, as the model sees it. Goals, not titles.

    The goal is the sentence the contract calls load-bearing (§1.1) and the
    title is whatever fitted in 72 characters, so an assessment of coverage
    reads the first and not the second.
    """
    lines = [
        f"- {entry.task_id} (#{entry.number}): {' '.join(entry.goal.split())[:GOAL_CHARS]}"
        for entry in entries
    ]
    return "\n".join(lines) or "- nothing"


def assess(
    objective: str,
    ledger: Ledger,
    *,
    oracle: Oracle | None = None,
) -> Assessment:
    """Has this objective been delivered? Arithmetic first, then one model call.

    The three arithmetic answers are the module docstring's refusals, in the
    order that costs least: a ledger with nothing in it, a ledger with work
    still running, and a ledger carrying a task the swarm abandoned. Only a run
    that finished everything it planned, cleanly, is worth a model swap.
    """
    if not ledger.entries:
        return Assessment(met=False, reason=EMPTY)

    running = live(ledger)
    if running:
        return Assessment(
            met=False,
            reason=f"{IN_FLIGHT}: {', '.join(f'#{entry.number}' for entry in running)}",
        )

    gave_up = abandoned(ledger)
    if gave_up:
        return Assessment(
            met=False,
            reason=FAILED,
            missing=tuple(f"#{entry.number} {entry.title}" for entry in gave_up),
            abandoned=tuple(entry.number for entry in gave_up),
        )

    landed = shipped(ledger)
    try:
        llm = oracle if oracle is not None else structured(orchestrator_llm(), ObjectiveAssessment)
        answer = llm.invoke(
            [
                ("system", SYSTEM_ASSESS),
                (
                    "human",
                    f"Objective:\n{objective}\n\nShipped:\n{_catalogue(landed)}",
                ),
            ]
        )
    except Exception as exc:  # noqa: BLE001 - any transport failure reads the same
        return Assessment(
            met=False,
            reason=f"the objective could not be assessed: {exc}",
            consulted=True,
            unresolved=True,
        )

    missing = tuple(line.strip() for line in answer.missing if line and line.strip())
    if answer.objective_met:
        return Assessment(met=True, reason=answer.reason or MET, consulted=True)
    return Assessment(
        met=False,
        missing=missing,
        # A model that says "not met" and names nothing has given the follow-up
        # planner nothing to work from. Reported as the reason so the refusal
        # that follows in `close_the_loop` is legible.
        reason=answer.reason or "the objective is not met, and no gap was named",
        consulted=True,
    )


# --------------------------------------------------------------------------
# Extending
# --------------------------------------------------------------------------


def propose(
    objective: str,
    ledger: Ledger,
    assessment: Assessment,
    *,
    proposer: Proposer | None = None,
) -> Plan:
    """Ask for the tasks that close the named gap. The planner's own prompt.

    `SYSTEM` plus `FOLLOWUP_SUFFIX`, imported rather than rewritten, for
    `replan.propose`'s reason: the hard rules in the planner's prompt are what
    make a plan writable at all, and a second copy of them here would drift from
    the one `_self_check` holds every draft to.
    """
    prompt = SYSTEM + FOLLOWUP_SUFFIX.format(
        shipped=_catalogue(shipped(ledger)),
        missing="\n".join(f"- {line}" for line in assessment.missing) or "- not stated",
    )
    llm = proposer if proposer is not None else structured(orchestrator_llm(), Plan)
    return llm.invoke([("system", prompt), ("human", f"Objective:\n{objective}")])


@dataclass(frozen=True)
class GoalReport:
    """What the end of a run concluded, and what it did about it."""

    assessment: Assessment
    extended: bool = False
    plan: PlanReport | None = None
    reason: str = ""
    rounds: int = 0

    @property
    def met(self) -> bool:
        return self.assessment.met

    @property
    def done(self) -> bool:
        """May the loop stop? Everything except a round that just added work."""
        return not self.extended

    @property
    def created(self) -> tuple[IssueAction, ...]:
        return () if self.plan is None else self.plan.created

    def summary(self) -> str:
        if self.extended:
            names = ", ".join(f"#{action.number}" for action in self.created)
            return (
                f"objective not met after round {self.rounds}: {self.reason}; "
                f"planned {len(self.created)} follow-up task(s) {names}"
            )
        if self.met:
            return f"objective met: {self.reason}"
        gaps = "; ".join(self.assessment.missing)
        detail = f" still missing: {gaps}" if gaps else ""
        return f"stopping without meeting the objective: {self.reason}.{detail}"


def close_the_loop(
    client: Any,
    ledger: Ledger,
    objective: str,
    *,
    rounds: int = 0,
    max_rounds: int = MAX_ROUNDS,
    verify: str | None = None,
    oracle: Oracle | None = None,
    proposer: Proposer | None = None,
    writer: Callable[..., PlanReport] = write_plan,
) -> GoalReport:
    """Assess, and extend the plan if the objective is not met yet.

    `ledger` is the read the caller already made this cycle, for
    `replan.replan`'s reason: re-listing the issues here would double the
    rate-limit cost of the one cycle that is otherwise entirely free, and the
    caller's copy is the one the loop decided to stop on.

    `writer` and the two model seams exist so the whole path is exercised
    without an Ollama and without a tracker; `write_plan` is the default and is
    called with `retire_dropped=False`, which is the single most important line
    in this module. A follow-up round must add issues and touch nothing else.
    """
    assessment = assess(objective, ledger, oracle=oracle)
    if assessment.met:
        return GoalReport(assessment, reason=assessment.reason or MET, rounds=rounds)
    if not assessment.actionable:
        return GoalReport(assessment, reason=assessment.reason, rounds=rounds)
    if not assessment.missing:
        # Nothing to decompose. Asking anyway produces a model's guess at what
        # its own previous answer meant, which is how a tracker fills up with
        # tasks nobody can trace to the objective.
        return GoalReport(assessment, reason=assessment.reason, rounds=rounds)
    if rounds >= max(int(max_rounds), 0):
        return GoalReport(assessment, reason=f"{EXHAUSTED} after {rounds} round(s)", rounds=rounds)

    try:
        plan = propose(objective, ledger, assessment, proposer=proposer)
    except Exception as exc:  # noqa: BLE001 - any transport failure reads the same
        # The stall's reading again: the model is unreachable, the run is not
        # wrong, and the round is not spent so a later caller may try again.
        return GoalReport(assessment, reason=f"the planner could not be reached: {exc}", rounds=rounds)

    # `write_plan`'s own normalisation, run first for `replan.replan`'s reason -
    # except that here the danger it guards against is different and smaller:
    # with `retire_dropped=False` an empty plan closes nothing. It is still
    # refused, because a round that writes nothing must not be charged to the
    # budget as though it had.
    drafts, _ = normalise(plan.tasks, verify=verify or SETTINGS.verify_command)
    if not drafts:
        return GoalReport(assessment, reason=NO_TASKS, rounds=rounds)

    try:
        written = writer(client, plan, ledger=ledger, verify=verify, retire_dropped=False)
    except PlanError as exc:
        return GoalReport(
            assessment, reason=f"the follow-up plan is not writable: {exc}", rounds=rounds
        )

    if not written.created:
        # Every draft was rejected by the planner's self-check, or every one of
        # them named a task that already exists. Either way the tracker did not
        # grow, and reporting this as an extension would send the loop round
        # again to find the same nothing.
        return GoalReport(
            assessment, plan=written, reason="the follow-up plan created no issue", rounds=rounds
        )

    return GoalReport(
        assessment,
        extended=True,
        plan=written,
        reason=assessment.reason,
        rounds=rounds + 1,
    )
