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
  the failure. So no model is asked - but the gate no longer simply resigns.
  A failed task whose issue is still open and whose hard total budget
  (`SWARM_MAX_TOTAL_ATTEMPTS`) is not spent is **revived** instead, through the
  same `planner.revive` the replan uses: back to `swarm:ready`, marker
  untouched, so the failure-signature arithmetic guards the retry. Observed
  live before this: a run merged everything else, arrived at the gate with one
  failed task sitting at attempt 5 of a 9 hard cap, and exited failed asking a
  human to do the one relabel the orchestrator knew was safe. When every
  failed task is closed (GitHub wins - a human or the planner shut it) or
  budget-spent, the failed work is settled and the gate asks the coverage
  question once anyway: a met answer closes the open failed leftovers as
  superseded ("at the end having failed ticket is wrong" - the user's rule,
  verbatim) and the run ends successfully; only an unmet one stops the run and
  names the issues, which is what `checks.py` already decided a `swarm:failed`
  label means.
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

from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol, Sequence

from ..config import SETTINGS
from ..github.ledger import Ledger, LedgerEntry
from ..github.readiness import resolve_states
from ..llm import orchestrator_llm, structured
from ..nodes.planner import (
    FOLLOWUP_SUFFIX,
    SYSTEM,
    IssueAction,
    PlanError,
    PlanReport,
    human_prompt,
    normalise,
    repository_files,
    retire_superseded,
    revive,
    write_plan,
)
from ..state import ObjectiveAssessment, Plan
from ..taskref import TaskRef
from .authority import Belief, state_of
from .derived import LANDED, NEEDS_HUMAN

#: `run.TERMINAL_LABELS` in the vocabulary the authority answers in, and the
#: same two facts: the work landed, or a human has to decide. Spelled out rather
#: than translated from that set so a reader of `live` can see both halves of
#: the branch without following an import into the label table.
TERMINAL_STATES = frozenset({LANDED, NEEDS_HUMAN})

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
    #: The tasks that stopped this being assessed at all, if any - named by
    #: ref, not by issue number. This is the module's one *cross-module*
    #: identity join: the tuple is built from the ledger in `assess` and
    #: matched back against it in `_revive_abandoned`, so the two sides must
    #: agree on what identity is. #142 made that `TaskRef` everywhere the
    #: internal model decides anything; this field was the residue. It is
    #: typed rather than merely conventional because the failure is silent:
    #: a set of ints tested against `entry.ref` matches nothing, revives
    #: nothing, and logs nothing about not having done so.
    abandoned: tuple[TaskRef, ...] = ()

    @property
    def actionable(self) -> bool:
        """May a follow-up round be planned from this? Only a real, negative answer."""
        return not self.met and not self.unresolved and not self.abandoned

    def summary(self) -> str:
        parts = [f"met={self.met}", "model" if self.consulted else "arithmetic"]
        if self.unresolved:
            parts.append("assessment unavailable")
        if self.abandoned:
            parts.append("abandoned: " + ", ".join(str(ref) for ref in self.abandoned))
        if self.missing:
            parts.append(f"{len(self.missing)} gap(s)")
        return f"{', '.join(parts)}: {self.reason}"


def shipped(ledger: Ledger, believed: Belief | None = None) -> tuple[LedgerEntry, ...]:
    """The tasks that landed, in issue order.

    Landing and nothing else. An issue a human closed by hand is read as done by
    the reconciler and carries the label by the time this runs, so the two
    agree; an issue that is merely closed and never relabelled is not evidence
    that its work landed.

    `believed` is the cycle's authority on state (#147), and `None` reads the
    label - the `__main__` dry run and `APIARY_STATE_SOURCE=labels`. This is the
    partition the whole gate is computed from, so a label edited mid-run put a
    task on the wrong side of the objective assessment: `swarm:done` typed onto
    unfinished work catalogued it to the model as shipped, and a `swarm:failed`
    typed onto merged work ended the run asking for a human about a task that
    had landed.
    """
    entries = (
        entry for entry in ledger.entries.values() if state_of(entry, believed) == LANDED
    )
    return tuple(sorted(entries, key=lambda entry: entry.ref))


def abandoned(ledger: Ledger, believed: Belief | None = None) -> tuple[LedgerEntry, ...]:
    """The tasks the swarm gave up on, in issue order. See `shipped` for `believed`."""
    entries = (
        entry for entry in ledger.entries.values() if state_of(entry, believed) == NEEDS_HUMAN
    )
    return tuple(sorted(entries, key=lambda entry: entry.ref))


def live(ledger: Ledger, believed: Belief | None = None) -> tuple[LedgerEntry, ...]:
    """Anything not terminal. `TERMINAL_STATES` decides, as `run.TERMINAL_LABELS`
    does for resume - the same two facts, and `believed=None` still reaches the
    labels through `authority.state_of`. See `shipped`.
    """
    entries = (
        entry
        for entry in ledger.entries.values()
        if state_of(entry, believed) not in TERMINAL_STATES
    )
    return tuple(sorted(entries, key=lambda entry: entry.ref))


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
    believed: Belief | None = None,
) -> Assessment:
    """Has this objective been delivered? Arithmetic first, then one model call.

    The three arithmetic answers are the module docstring's refusals, in the
    order that costs least: a ledger with nothing in it, a ledger with work
    still running, and a ledger carrying a task the swarm abandoned. Only a run
    that finished everything it planned, cleanly, is worth a model swap.

    All three partition the ledger through `believed` (#147), and `None` reads
    the label. The three refusals are the reason it matters here: each of them
    ends or extends a *run*, and a task on the wrong side of one is a run that
    stops early, resigns over work that landed, or asks a model to assess a
    catalogue that does not describe what shipped.
    """
    if not ledger.entries:
        return Assessment(met=False, reason=EMPTY)

    running = live(ledger, believed)
    if running:
        return Assessment(
            met=False,
            reason=f"{IN_FLIGHT}: {', '.join(f'#{entry.number}' for entry in running)}",
        )

    gave_up = abandoned(ledger, believed)
    if gave_up:
        return Assessment(
            met=False,
            reason=FAILED,
            missing=tuple(f"#{entry.number} {entry.title}" for entry in gave_up),
            abandoned=tuple(entry.ref for entry in gave_up),
        )

    landed = shipped(ledger, believed)
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
            reason=f"the objective could not be assessed: {type(exc).__name__}: {exc}",
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
    files: Sequence[str] | None = None,
) -> Plan:
    """Ask for the tasks that close the named gap. The planner's own prompt.

    `SYSTEM` plus `FOLLOWUP_SUFFIX`, imported rather than rewritten, for
    `replan.propose`'s reason: the hard rules in the planner's prompt are what
    make a plan writable at all, and a second copy of them here would drift from
    the one `_self_check` holds every draft to. The human turn is
    `planner.human_prompt`'s: `files` is the repository's listing when the
    caller could get one - and a follow-up round is where it earns the most,
    because by now the tree holds everything the shipped tasks built - while
    None sends the turn exactly as it always was.
    """
    prompt = SYSTEM + FOLLOWUP_SUFFIX.format(
        shipped=_catalogue(shipped(ledger)),
        missing="\n".join(f"- {line}" for line in assessment.missing) or "- not stated",
    )
    llm = proposer if proposer is not None else structured(orchestrator_llm(), Plan)
    return llm.invoke([("system", prompt), ("human", human_prompt(objective, files))])


@dataclass(frozen=True)
class GoalReport:
    """What the end of a run concluded, and what it did about it."""

    assessment: Assessment
    extended: bool = False
    plan: PlanReport | None = None
    reason: str = ""
    rounds: int = 0
    #: Failed tasks this gate returned to `swarm:ready` because their issue was
    #: open and their hard total budget was not spent. Non-empty means the run
    #: carries on - the revived work dispatches next cycle - which is why
    #: `done` reads it. Not counted against `rounds`: a revival plans nothing,
    #: and its bound is the attempt budget itself (`planner.revive`'s docstring
    #: - a revived issue is ready, not failed, so the gate cannot see it again
    #: without a fresh give-up in between, and every give-up burned attempts).
    revived: tuple[IssueAction, ...] = ()
    #: Failed, open tasks closed as `not_planned` because the objective was met
    #: without them. The other half of "at the end, a failed ticket is wrong":
    #: a run that delivered its objective must not leave a board wearing
    #: `swarm:failed` for work nothing needs. Set only on a met report, so it
    #: never keeps a loop alive - the run is over, successfully.
    superseded: tuple[IssueAction, ...] = ()

    @property
    def met(self) -> bool:
        return self.assessment.met

    @property
    def done(self) -> bool:
        """May the loop stop? Not after a round that added work, and not after
        a revival - both put dispatchable issues on the tracker, and stopping
        would strand them exactly as the pre-gate loop stranded follow-ups."""
        return not self.extended and not self.revived

    @property
    def created(self) -> tuple[IssueAction, ...]:
        return () if self.plan is None else self.plan.created

    def summary(self) -> str:
        if self.revived:
            return f"objective not met; {self.reason}; the run continues"
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


def _revive_abandoned(
    client: Any,
    ledger: Ledger,
    assessment: Assessment,
    *,
    max_attempts: int,
    max_total_attempts: int,
    believed: Belief | None = None,
) -> tuple[IssueAction, ...]:
    """Revive every abandoned task the orchestrator may safely retry. See
    `close_the_loop` for the rule; this is only its per-issue application.

    The selection is a join on `TaskRef`, matching the tuple `assess` built
    from this same ledger. Both sides are refs deliberately: a set miss
    returns rather than raises, so an identity join keyed on the wrong type
    revives nobody and reports nothing. That shape shipped green once already
    during #142 - a ref-keyed `Reconciler._results()` read with an int, which
    took the judgement path out with every test passing - and it is why the
    tests for this function assert on the join and not just on the revival.

    The open/closed read is one issue listing (`resolve_states`), and it is the
    load-bearing check: a closed failed issue was shut by a human or retired by
    the planner, and relabelling it would be the orchestrator arguing with the
    one input it never argues with. Missing and unreadable issues are treated
    as closed for the same reason - writing to an issue this cycle could not
    see is a guess, and `readiness._met` already established that unknown reads
    as the cautious answer. The budget check is *not* repeated here:
    `planner.revive` owns it and answers a spent task with a `retained` action
    and no writes, which this function simply does not count as a revival.
    """
    wanted = set(assessment.abandoned)
    # The same `believed` `assess` partitioned on, because this is the other
    # half of one join: an assessment built under the resolver against a set
    # rebuilt from the labels matches on whatever the two happen to agree
    # about, which is the silent-miss shape this function's docstring is about.
    entries = [entry for entry in abandoned(ledger, believed) if entry.ref in wanted]
    states = resolve_states(client, [entry.ref for entry in entries])
    actions: list[IssueAction] = []
    for entry in entries:
        state = states.get(entry.ref)
        if state is None or not state.exists or state.closed:
            continue
        action = revive(
            client,
            entry,
            max_attempts=max_attempts,
            max_total_attempts=max_total_attempts,
            because=(
                "the goal gate found the objective unmet and this task still has "
                "retry budget"
            ),
        )
        if action.kind == "revived":
            actions.append(action)
    return tuple(actions)


def _retire_superseded(
    client: Any, ledger: Ledger, believed: Belief | None = None
) -> tuple[IssueAction, ...]:
    """Close every failed, still-open task of a met objective as superseded.

    The user's rule, verbatim: "at the end having failed ticket is wrong". A
    met objective is the run declaring nothing further is needed, so a
    `swarm:failed` issue left open at that moment is a request for a human
    decision that has already been made - the coverage question was asked and
    answered without that task's work. Closed issues are skipped for
    `_revive_abandoned`'s reason (GitHub wins; unknown reads as closed), and
    the closure itself is `planner.retire_superseded`, shared with the
    replan's drop path so the two supersessions cannot drift apart. Bounded to
    failed tasks only: ready or blocked leftovers on a met objective are a
    different question, deliberately not answered here.
    """
    entries = abandoned(ledger, believed)
    if not entries:
        return ()
    states = resolve_states(client, [entry.ref for entry in entries])
    actions: list[IssueAction] = []
    for entry in entries:
        state = states.get(entry.ref)
        if state is None or not state.exists or state.closed:
            continue
        actions.append(
            retire_superseded(
                client,
                entry,
                because="the objective was met without this task",
                reason="swarm:failed with the objective met; closed as superseded",
            )
        )
    return tuple(actions)


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
    max_attempts: int = SETTINGS.max_attempts_per_task,
    max_total_attempts: int = SETTINGS.max_total_attempts_per_task,
    believed: Belief | None = None,
) -> GoalReport:
    """Assess, then revive what is revivable, then extend if a gap remains.

    `ledger` is the read the caller already made this cycle, for
    `replan.replan`'s reason: re-listing the issues here would double the
    rate-limit cost of the one cycle that is otherwise entirely free, and the
    caller's copy is the one the loop decided to stop on.

    `believed` is the same cycle's authority over that ledger (#147), and it is
    passed rather than re-derived for exactly the same reason: it is the belief
    the loop decided to stop on, already folded by everything this cycle wrote.
    One value threads the whole gate - the assessment, the revival join, the
    settled re-assessment and the supersession - because they are four readings
    of one partition and a second source anywhere in that chain would revive a
    task the assessment never named. `None` reads the label, which is the
    `__main__` dry run and `APIARY_STATE_SOURCE=labels`.

    `writer` and the two model seams exist so the whole path is exercised
    without an Ollama and without a tracker; `write_plan` is the default and is
    called with `retire_dropped=False`, which is the single most important line
    in this module. A follow-up round must add issues and touch nothing else.

    **The gate is the second caller of `planner.revive`** (the replan's write
    path is the first). An unmet assessment whose blockers are `swarm:failed`
    tasks used to end the run asking a human, even when the marker said budget
    remained - a run was observed to merge everything else and quit over one
    task at attempt 5 of a 9 hard cap. Now each failed, still-open issue with
    total budget remaining is returned to `swarm:ready` (marker untouched; the
    signature arithmetic guards the retry) and the run continues; closed
    issues are never touched - GitHub wins. `max_attempts`/`max_total_attempts`
    exist so the comment the revival posts names the caps the run actually
    enforces. A revive-fail-revive hot loop needs no counter here: a revived
    issue is ready, not failed, so this branch cannot see it again until a
    fresh give-up - which consumed attempts - puts it back, and the hard total
    cap ends that cycle (`planner.revive`).

    **When nothing is revivable, the failed work is settled, and the gate asks
    the coverage question once before resigning.** Every abandoned issue being
    closed or budget-spent means no retry is coming from anywhere, so "is the
    objective met without that work?" finally has a stable answer - and the
    body of the question is unchanged, because `assess` only ever catalogues
    `shipped()` for the model. A *met* answer closes the failed leftovers as
    `not_planned` with a superseded comment (`_retire_superseded`) and the run
    ends successfully; an unmet or unresolved answer keeps the old resignation
    verbatim - unmet, the abandoned issues named, the run ends failed, and no
    follow-up is planned on top of a repository whose last known state is
    broken.
    """
    assessment = assess(objective, ledger, oracle=oracle, believed=believed)
    if assessment.abandoned:
        revived = _revive_abandoned(
            client,
            ledger,
            assessment,
            max_attempts=max_attempts,
            max_total_attempts=max_total_attempts,
            believed=believed,
        )
        if revived:
            names = ", ".join(f"#{action.number}" for action in revived)
            return GoalReport(
                assessment,
                revived=revived,
                reason=(
                    f"revived {len(revived)} failed task(s) with budget remaining: {names}"
                ),
                rounds=rounds,
            )
        # Nothing revivable, so the failed work is *settled*: every abandoned
        # issue is closed (a human's or the planner's decision) or budget-spent
        # (the machinery's own bound). One question is still worth one model
        # swap before resigning: was the objective met without that work?
        # `assess` only ever shows the model what `shipped()` - so this is the
        # same coverage question the abandoned arithmetic was short-circuiting,
        # asked over the same catalogue. Only a *met* answer is acted on; an
        # unmet or unresolved one changes nothing, because planning follow-ups
        # on top of a repository whose last known state is broken is exactly
        # what the arithmetic refusal exists to prevent.
        gave_up = {entry.task_id for entry in abandoned(ledger, believed)}
        settled = replace(
            ledger,
            entries={
                task_id: entry
                for task_id, entry in ledger.entries.items()
                if task_id not in gave_up
            },
        )
        if settled.entries:
            final = assess(objective, settled, oracle=oracle, believed=believed)
            if final.met:
                assessment = final
    if assessment.met:
        superseded = _retire_superseded(client, ledger, believed)
        reason = assessment.reason or MET
        if superseded:
            names = ", ".join(f"#{action.number}" for action in superseded)
            reason = f"{reason}; closed {len(superseded)} superseded failed task(s): {names}"
        return GoalReport(assessment, superseded=superseded, reason=reason, rounds=rounds)
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
        # Best-effort by `repository_files`'s contract: a follow-up whose tree
        # read fails is planned exactly as it was before listings existed.
        plan = propose(
            objective, ledger, assessment, proposer=proposer, files=repository_files(client)
        )
    except Exception as exc:  # noqa: BLE001 - any transport failure reads the same
        # The stall's reading again: the model is unreachable, the run is not
        # wrong, and the round is not spent so a later caller may try again.
        return GoalReport(
            assessment,
            reason=f"the planner could not be reached: {type(exc).__name__}: {exc}",
            rounds=rounds,
        )

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
