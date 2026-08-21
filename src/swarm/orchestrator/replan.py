"""What a stall buys: one new decomposition, written over the old issues.

Step 5 of `docs/architecture-v2.md`'s loop ends "stall triggers a replan, which
rewrites issues rather than in-memory tasks". The judge (#24's other half)
decides *whether* a run has stopped making progress; this module is what
happens next, and it is the only place in v2 where the swarm rewrites its own
backlog.

**This is the most destructive operation in the system, so most of the module
is about refusing to perform it.** A replan closes issues and opens new ones.
Done on a run that was merely slow, it throws away work that was one merge from
landing; done on a run whose problem is not its plan, it produces a second plan
with the same problem and a tracker full of churn. `decide` is therefore the
first thing `replan` calls, and it says no on five distinct grounds - see the
constants below and `judge.Verdict.should_replan`.

**The write path is #10's, not a second copy of it.** `planner.write_plan`
already matches on the marker id (§2), updates an issue in place rather than
opening a duplicate, retires a dropped task as `not_planned` so readiness never
reads a cancellation as a satisfied dependency, and - the rule that matters
most here - **leaves a dropped task that is `swarm:claimed` or `swarm:review`
exactly where it is and reports it**. That decision belongs to the planner's
module docstring and is not re-taken here; this module reads the retentions out
of the `PlanReport` and reports them onwards, because a human is the only one
who should be deciding to abandon a container mid-edit. The same ownership
holds for the newer revival rule: a `swarm:failed` task the new plan *keeps*
is returned to `swarm:ready` by `write_plan` itself (budget intact - the
failure signature is the guard), and this module only surfaces it, because a
replan that rewrote the tracker around a failed task while leaving the task
itself dead was observed to re-stall the run until a human relabelled it.

**The one model call is the plan, never the decision.** Whether to replan is
arithmetic over the judge's verdict, and the verdict itself is arithmetic on
almost every cycle (#24). The model is asked exactly one question - "given that
this failed, decompose it differently" - and only once the arithmetic has
already concluded that asking is worth a ~6.7 s model swap.

**A plan with nothing writable in it is refused outright.** `write_plan`
retires every entry the new plan does not contain, so a model that answered
with no tasks - or with four tasks that all fail the planner's own
normalisation - would close the entire tracker in one call. That is the single
worst thing this module could do and it is one malformed model response away,
so the plan is put through `planner.normalise` first and refused if nothing
survives.

**There is deliberately no dry run and no `__main__` smoke test**, where every
other module in this package has both. `write_plan` cannot render a body
without the issue numbers that only exist after creation (planner's docstring),
and a "manual dry run" of a module whose job is to close issues is a manual dry
run against somebody's real backlog. The tests drive it through the same
`write_plan` seam with a fake client, which is the only way it is ever exercised
outside a real run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from ..config import SETTINGS
from ..github.client import GitHubClient
from ..github.ledger import Ledger
from ..llm import orchestrator_llm, structured
from ..nodes.judge import Blocker, Verdict
from ..nodes.planner import (
    REPLAN_SUFFIX,
    SYSTEM,
    IssueAction,
    PlanError,
    PlanReport,
    human_prompt,
    normalise,
    repository_files,
    write_plan,
)
from ..state import Plan
from .authority import Belief, state_of

#: How many times one run may rewrite its own plan. Two, because the second
#: replan is where the evidence changes character: a decomposition that stalled,
#: was replaced, and stalled again says the objective or the repository is the
#: problem, not the arrangement of the tasks - and a third rewrite spends the
#: model, the rate limit and a human's ability to read the tracker on the same
#: wrong answer. The run stops and says so instead.
MAX_REPLANS = 2

#: How much of a failure the model is shown per task. Enough to recognise the
#: error, not enough for four tasks' tracebacks to crowd out the objective in a
#: 16K context (`config.py`).
FAILURE_CHARS = 300

#: The reasons `decide` gives for refusing, spelled once so the report, the
#: tests and the log line cannot drift apart.
SATISFIED = "the run is satisfied; there is nothing to replan"
PROGRESSING = "the run is still making progress"
UNRESOLVED = "the judgement could not be made; the planner would call the same model"
NEEDS_HUMAN = (
    "every task still moving is blocked on files it may not edit; a replan cannot fix that"
)
TOO_SOON = "the stall budget is not spent yet"
EXHAUSTED = "this run has replanned as often as it may; the plan is not the problem"
NO_TASKS = "the model proposed no usable tasks; writing that would retire the whole tracker"


class Proposer(Protocol):
    """The one model call this module makes. See `propose`."""

    def invoke(self, messages: Sequence[tuple[str, str]]) -> Plan: ...


# --------------------------------------------------------------------------
# Deciding
# --------------------------------------------------------------------------


def _replannable(verdict: Verdict) -> tuple[str, ...]:
    """Task ids a new decomposition could still help. Empty means refuse.

    **The narrowing #293 asked for.** `Verdict.needs_human` is true when *any*
    task is failing on paths its `## Files` does not list, and that is a correct
    judgement about that task: re-cutting the plan cannot hand it a file it was
    never given, so a replan would produce a second plan with the same problem.
    What was wrong is the scope. The condition is per task and the refusal was
    per run, so one task the plan cannot help vetoed replanning all the tasks it
    could - and since a stalled run usually has exactly one such task, the veto
    was close to unconditional.

    So the refusal now needs both halves to be true: something is blocked on
    files it may not edit, *and* there is nothing else left that a replan could
    move. `blockers` names the first half; this names the second.

    Terminal tasks are excluded because a replan has nothing to offer a task
    that has already landed or already been given up on, and an observation this
    module could not read at all falls back to the old refusal - the whole point
    of the veto is that a replan is destructive, so an unreadable world is not
    the place to start guessing.
    """
    observation = verdict.observation
    if observation is None:
        return ()
    blocked = {blocker.task_id for blocker in verdict.blockers}
    return tuple(
        sorted(
            task_id
            for task_id, signal in observation.signals.items()
            if task_id not in blocked and not signal.terminal
        )
    )


def decide(
    verdict: Verdict,
    *,
    replans: int = 0,
    max_stalls: int = SETTINGS.max_stalls,
    max_replans: int = MAX_REPLANS,
) -> str:
    """Empty string to go ahead; otherwise the sentence explaining the refusal.

    A string rather than a bool because every caller wants the reason: it goes
    in the report, and "the swarm did nothing" and "the swarm decided against
    doing something" are indistinguishable from the outside otherwise - the
    same argument `dispatcher.Deferred` makes about a cycle that dispatched
    nothing.

    The first four grounds are the judge's, taken apart one at a time only so
    the report can name which of them fired; `should_replan` is then asked as
    the single authority on whether the stall budget is spent, so the two
    modules cannot end up disagreeing about it. The fifth ground is this
    module's own, and bounds how many times one run may rewrite itself.
    """
    if verdict.satisfied:
        return SATISFIED
    if verdict.unresolved:
        return UNRESOLVED
    if verdict.needs_human and not _replannable(verdict):
        return NEEDS_HUMAN
    if not verdict.stalled:
        return PROGRESSING
    if replans >= max(int(max_replans), 0):
        return EXHAUSTED
    if not verdict.should_replan(max_stalls=max_stalls):
        return TOO_SOON
    return ""


# --------------------------------------------------------------------------
# Proposing
# --------------------------------------------------------------------------


def brief(
    ledger: Ledger, verdict: Verdict, believed: Belief | None = None
) -> tuple[str, str]:
    """The two blocks `planner.REPLAN_SUFFIX` interpolates: failures, then ids.

    The failures come from what the workers actually wrote down - the judge's
    observation carries each task's latest failure text, read out of the
    artifacts directory - rather than from a status field. A model told only
    "add-retry-logic: failed" reliably re-emits the same plan; one shown the
    error has something to decompose around.

    Every id is listed, not only the failing ones, because `REPLAN_SUFFIX` asks
    the model to re-emit unchanged work under its exact existing id, and an id
    it never sees is an id it invents a replacement for - which is the second
    set of issues §2 exists to prevent.

    **What each id is annotated with is `believed`, the cycle's authority on
    state (#147), and this call site is the one that is not a decision.**
    Nothing branches on the annotation; it is a word in a prompt. It is here for
    the other half of that epic: `swarm:*` is a storage detail #141 and #140 are
    deleting, and a prompt is the worst place to leave one, because the model
    reads the vocabulary as the run's own and re-emits it. The authority answers
    in the internal words (`landed`, `needs-human`, `eligible`), which is what
    `lifecycle.scrub` already rewrites the event log into, so a replan brief and
    an `events.jsonl` line now describe a task the same way.

    `believed=None` prints the label verbatim, which is every caller outside
    `Reconciler.cycle` - the `__main__` dry run and `APIARY_STATE_SOURCE=labels`
    - and keeps the prompt they send byte-identical.
    """
    observation = verdict.observation
    signals = dict(observation.signals) if observation is not None else {}

    lines: list[str] = []
    for task_id, entry in sorted(ledger.entries.items(), key=lambda item: item[1].ref):
        signal = signals.get(task_id)
        evidence = signal.evidence if signal is not None else ""
        # One line per task: a traceback pasted verbatim would put lines
        # starting with `-` into a list the model reads as more tasks.
        flattened = " ".join(evidence.split())
        if not flattened:
            continue
        attempts = signal.attempts if signal is not None else entry.attempt
        lines.append(
            f"- {task_id} (#{entry.number}, attempt {attempts}): {flattened[:FAILURE_CHARS]}"
        )
    failures = "\n".join(lines) or f"- no task recorded an error; {verdict.reason}"

    # `entry.state_label` under `None` rather than `authority.state_of`, which
    # would translate it: this is a string in a prompt, so "unchanged" has to
    # mean the same characters, not the same fact.
    tracked = "\n".join(
        f"- {task_id} ({state_of(entry, believed)}): "
        f"{entry.goal[:120]}"
        for task_id, entry in sorted(ledger.entries.items(), key=lambda item: item[1].ref)
    ) or "- none"
    return failures, tracked


def propose(
    objective: str,
    ledger: Ledger,
    verdict: Verdict,
    *,
    proposer: Proposer | None = None,
    files: Sequence[str] | None = None,
    believed: Belief | None = None,
) -> Plan:
    """Ask for a different decomposition of the same objective.

    The prompt is the planner's own - `SYSTEM` plus `REPLAN_SUFFIX` - imported
    rather than rewritten, because the hard rules in it (disjoint `## Files`,
    re-emit under the existing id) are what make the resulting plan writable at
    all, and a second copy of them here would drift from the one the planner
    self-checks against. The human turn is `planner.human_prompt`'s for the
    same reason: `files` is the repository listing when the caller could get
    one, and None sends the turn exactly as it always was.
    """
    failures, tracked = brief(ledger, verdict, believed)
    prompt = SYSTEM + REPLAN_SUFFIX.format(failures=failures, existing=tracked)
    llm = proposer if proposer is not None else structured(orchestrator_llm(), Plan)
    return llm.invoke([("system", prompt), ("human", human_prompt(objective, files))])


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplanReport:
    """What one stall did to the tracker, or why it did nothing.

    `plan` is `write_plan`'s own report, kept whole rather than flattened: it
    is where the per-issue actions live, and a caller asking "what happened to
    the task my container is working on" needs the retention, not a count.
    """

    repo: str
    replanned: bool = False
    reason: str = ""
    plan: PlanReport | None = None
    #: The stall count the caller should carry into the next cycle. Reset to
    #: zero by a replan: the tracker it was counting stalls against no longer
    #: exists, and carrying the count forward would replan again on the very
    #: next stalled cycle, before the new plan had a chance to run.
    stalls: int = 0
    replans: int = 0
    blockers: tuple[Blocker, ...] = ()

    @property
    def retained(self) -> tuple[IssueAction, ...]:
        """Issues the write path refused to touch - in-flight work, mostly.

        #10 owns that decision (module docstring); this is the channel that
        carries it up to whoever is reading the run, because an in-flight task
        the new plan no longer wants is precisely the thing a human has to
        settle.
        """
        return () if self.plan is None else self.plan.retained

    @property
    def revived(self) -> tuple[IssueAction, ...]:
        """Failed tasks the new plan kept, returned to `swarm:ready`.

        The write path's decision too (`planner.write_plan`'s revival rule),
        surfaced here because a stall is exactly where it matters: the failed
        task whose chain blocks everything used to be "left alone" by a replan
        - observed live as a run that replanned, stayed 0-ready and re-stalled
        until a human relabelled the issue by hand.
        """
        return () if self.plan is None else self.plan.revived

    @property
    def numbers(self) -> tuple[int, ...]:
        """The issues the plan is now made of, ascending."""
        return () if self.plan is None else self.plan.numbers

    def summary(self) -> str:
        if not self.replanned:
            head = f"{self.repo}: no replan - {self.reason}"
            return f"{head}; {len(self.blockers)} task(s) need a human" if self.blockers else head
        detail = self.plan.summary() if self.plan is not None else "nothing written"
        return f"replanned after {self.reason}: {detail}"


# --------------------------------------------------------------------------
# Replanning
# --------------------------------------------------------------------------


def replan(
    client: GitHubClient | Any,
    ledger: Ledger,
    objective: str,
    verdict: Verdict,
    *,
    replans: int = 0,
    max_stalls: int = SETTINGS.max_stalls,
    max_replans: int = MAX_REPLANS,
    verify: str | None = None,
    proposer: Proposer | None = None,
    writer: Callable[..., PlanReport] = write_plan,
    believed: Belief | None = None,
) -> ReplanReport:
    """Decide, propose, write. Nothing is written unless all three succeed.

    `ledger` is the read the caller already made this cycle - the same argument
    `write_plan` makes for taking one: re-listing the issues here would double
    the rate-limit cost of a stalled cycle, and the caller's copy is the one the
    verdict was computed from, so a fresh read could describe a different run.

    `verify` is the run's repo-wide verify command, resolved once by whoever
    started the run (`cli._target`) and carried here so that a rewritten issue
    carries the same gate the original did. Defaulting it locally would mean a
    stall silently re-pointed every task in a generated repository at
    `SETTINGS.verify_command`, which that repository has no way to run.

    `believed` is that same cycle's authority on the ledger's states, and it goes
    to both halves of this function: the brief names each task with it (#198) and
    the write selects its revivals and retirements on it (#212). One value for
    both, because a prompt that called a task abandoned while the write path read
    it as landed would be two control planes inside one replan. `None` reads the
    labels, which is `__main__` and `APIARY_STATE_SOURCE=labels`.

    `proposer` and `writer` are the two seams, and they exist for one reason:
    **a replan rewrites a real issue tracker.** A test that exercised this
    function against GitHub would close issues in the repository the swarm is
    working from. So the model is injectable and the write is `write_plan` by
    default and a double in the tests, which is how the whole path is covered
    without a single API call.

    The report is deliberately not followed by a re-read. The next cycle reads
    the tracker at its top like every other cycle does, and `GitHub wins on any
    disagreement` (`docs/architecture-v2.md`) is only a rule that means anything
    if nothing here keeps a second copy of the answer.
    """
    repo = getattr(client, "repo", "")

    refusal = decide(
        verdict, replans=replans, max_stalls=max_stalls, max_replans=max_replans
    )
    if refusal:
        return ReplanReport(
            repo=repo,
            reason=refusal,
            stalls=verdict.stalls,
            replans=replans,
            blockers=verdict.blockers,
        )

    try:
        # `repository_files` is best-effort by contract: a replan whose tree
        # read fails plans exactly as it did before listings existed, because
        # a stalled run must never be made unfixable by a 502.
        plan = propose(
            objective,
            ledger,
            verdict,
            proposer=proposer,
            files=repository_files(client),
            believed=believed,
        )
    except Exception as exc:  # noqa: BLE001 - any transport failure reads the same
        # Same reading as an unresolved judgement: the model is unreachable, the
        # run is not wrong, and the stall count stands so the next cycle can try
        # again rather than starting the budget over.
        return ReplanReport(
            repo=repo,
            reason=f"the planner could not be reached: {type(exc).__name__}: {exc}",
            stalls=verdict.stalls,
            replans=replans,
            blockers=verdict.blockers,
        )

    # The write path's own normalisation, run before the write rather than
    # inside it: `write_plan` retires every entry the plan does not contain, so
    # a plan that normalises to nothing closes the whole tracker.
    drafts, _ = normalise(plan.tasks, verify=verify or SETTINGS.verify_command)
    if not drafts:
        return ReplanReport(
            repo=repo, reason=NO_TASKS, stalls=verdict.stalls, replans=replans
        )

    try:
        # `believed` goes to the write as well as to the brief (#212): the
        # planner's revival and retirement decisions are the two this ticket's
        # criterion is sharpest about, and a replan that annotated the prompt
        # with the authority's answer while the write still selected on the
        # label would be reading two control planes in one function.
        report = writer(client, plan, ledger=ledger, verify=verify, believed=believed)
    except PlanError as exc:
        # A ring in the proposed graph. `write_plan` orders the drafts before
        # its first write and raises having written nothing, so the tracker is
        # untouched and the run carries on with the plan it has.
        return ReplanReport(
            repo=repo,
            reason=f"the proposed plan is not writable: {exc}",
            stalls=verdict.stalls,
            replans=replans,
            blockers=verdict.blockers,
        )
    return ReplanReport(
        repo=repo,
        replanned=True,
        reason=verdict.reason or verdict.judgement.reason,
        plan=report,
        stalls=0,
        replans=replans + 1,
        blockers=verdict.blockers,
    )
