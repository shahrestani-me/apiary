"""One event per task transition, in apiary's own vocabulary (#141).

A cycle is minutes long, and `cycle.reconciled` - one summary sentence per
cycle - is all that survived of it. Every state worth watching (a task became
eligible, a container took it, its gate spoke, its pull request merged) starts
and ends *inside* a cycle, so an operator reading `events.jsonl` afterwards
could see that a run happened and not what happened in it. The reconciler
already computes each of these transitions; it just never announced them.

**This announces; it does not decide.** `lifecycle_events` is a pure projection
of a `CycleReport` that has already been computed, already been written to
GitHub and already been judged. Nothing in the loop reads it, nothing here
moves a label, and `cycle.reconciled` is untouched. That is why it is a
function over a finished report rather than a hook inside the loop, and why it
lives in its own module: `reconcile.py`'s docstring frames that file as the
body that *decides*, and the module #147 and #152 will gut is not where the
substrate they read from should sit.

**The vocabulary is apiary's own.** Every payload is keyed by the task ref
`Transition.task_id` already carries, and the states it names are ADR 0001's
internal ones. Not the issue number, and not the `swarm:*` label: the log is
append-only and read back (`RunMetrics.from_events`, `console_external`,
and #145's replay corpus), so a payload written in the label vocabulary would
be invalidated the day the rest of epic #140 removes the labels. That is the
whole reason #131 was retargeted into this ticket.

**Derived facts and applied writes are sourced differently, and the difference
is ADR 0001's.** `eligible`, `result`, `pr.opened` and `pr.checks` are facts
about the world - dependencies discharged, a record on disk, an open pull
request, a check set - so they are projected from what the cycle *read*, and
each carries a `once` key because the world keeps saying them until it changes.
`claimed`, `pr.merged`, `landed` and `needs_human` are consequences of writes,
so they are projected from what actually **landed** - `fold`'s rule, for
`fold`'s reason: a label write GitHub refused left the task where it was, and
announcing it would put a state in an append-only log that the control plane
never reached.

One consequence worth stating: a dry run announces the derived half and none of
the written half, because on a dry run the derived facts are true and no write
happened. `--dry-run` already records `run.started`, `cycle.started` and
`cycle.reconciled` into the same file, so this is the existing bargain, not a
new one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from ..artifacts import (
    PR_CHECKS,
    PR_MERGED,
    PR_OPENED,
    TASK_CLAIMED,
    TASK_ELIGIBLE,
    TASK_LANDED,
    TASK_NEEDS_HUMAN,
    TASK_RESULT,
)
from ..github.branches import parse_task_branch
from ..github.readiness import BLOCKED, READY
from ..github.refs import issue_number
from ..taskref import TaskRef
from ..worker.result import ResultRecord
from ..github.refs import pull_number
from .checks import CheckSet, PullState
from .dispatcher import CLAIMED, REVIEW
from .reconcile import DONE, FAILED, CycleReport, Transition

__all__ = [
    "INTERNAL_STATE",
    "LifecycleLog",
    "TaskEvent",
    "internal_state",
    "lifecycle_events",
    "scrub",
]

#: ADR 0001's internal workflow, keyed by the label that happens to store it
#: today. The states on the right are apiary's own and identical for every
#: customer; the six strings on the left are a storage detail that epic #140
#: removes. Everything announced here speaks the right-hand column, so a
#: recorded run stays readable once the left-hand one is gone.
#:
#: `blocked` is the sixth. ADR 0001's table names five because it is describing
#: the states a *running* task passes through, and a task waiting on a
#: dependency is not one of them - but readiness holds tasks there and prose
#: names it, so it is mapped explicitly rather than left to fall through the
#: label suffix.
INTERNAL_STATE = {
    READY: "eligible",
    BLOCKED: "blocked",
    CLAIMED: "claimed",
    REVIEW: "review",
    DONE: "landed",
    FAILED: "needs-human",
}

#: A branch name carries the task ref (`github/branches.py`), and for the
#: GitHub adapter a ref *is* an issue number - `apiary/%2312-attempt-1` spells
#: 12 as plainly as `swarm/issue-12` did. So prose quoting a branch still
#: smuggles the number into a payload that is meant to be joinable on the task
#: ref alone, and it is still translated on the way out, alongside any label
#: name. Matched loosely and parsed strictly: the regex finds the shape, and
#: `parse_task_branch` decides whether it really is one of ours.
_BRANCH_RE = re.compile(r"\bapiary/[A-Za-z0-9_%-]+-attempt-\d+\b")
_LABEL_RE = re.compile(r"\bswarm:([a-z_-]+)\b")


def internal_state(label: str) -> str:
    """The internal state a `swarm:*` label is storing. See `INTERNAL_STATE`."""
    return INTERNAL_STATE.get(label, label.split(":", 1)[-1])


def scrub(text: str, refs: Mapping[int, str]) -> str:
    """Rewrite a sentence written for a human into the internal vocabulary.

    Two things the reconciler and the merge gate legitimately put in a reason
    are the two this ticket keeps out of the log: the branch
    (`apiary/%2312-attempt-1`, which carries one) and the label
    (`swarm:ready`, which epic #140 deletes). A task ref the ledger cannot
    resolve becomes a neutral phrase rather than the number, because a number
    that survives *because* the ledger lost the task is the worst case, not the
    exempt one.

    The issue comment carrying the same sentence is not rewritten and still
    says `swarm:ready`, which is the form an operator can act on today. That is
    deliberate: the comment is for the human holding the issue, and this is for
    the reader reconstructing the run after the labels are gone.
    """

    def branch(match: re.Match[str]) -> str:
        parsed = parse_task_branch(match.group(0))
        if parsed is not None:
            try:
                number = issue_number(parsed.ref)
            except ValueError:
                # A ref this adapter did not mint. Nothing here can turn it
                # into a number, which is the correct outcome rather than a
                # failure: `refs` is keyed by issue number because the code
                # host is, and a foreign ref simply has no entry to find.
                return "another task in this run"
            return refs.get(number, "another task in this run")
        return "another task in this run"

    return _LABEL_RE.sub(
        lambda match: internal_state(match.group(0)), _BRANCH_RE.sub(branch, text)
    )


@dataclass(frozen=True)
class TaskEvent:
    """One thing that happened to one task, ready for `RunArtifacts.event`.

    `once` is the identity of the *occurrence*, for the events whose source is
    a standing fact rather than a change: the results directory still holds
    last cycle's record, the pull request is still open, the check set still
    says pending. It is the occurrence and not the task, so a genuinely new
    occurrence - a second attempt, a force-pushed head, a check set that moved
    - is announced again. `None` means the source only speaks when something
    landed, and a second announcement would take a second write to produce.
    """

    name: str
    fields: Mapping[str, Any]
    once: tuple[Any, ...] | None = None


def lifecycle_events(
    report: CycleReport,
    *,
    results: Mapping[TaskRef, ResultRecord] | None = None,
    pulls: Mapping[str, PullState] | None = None,
    checks: Mapping[TaskRef, CheckSet] | None = None,
) -> tuple[TaskEvent, ...]:
    """Project one finished cycle onto the per-task lifecycle. Pure.

    The three facts the cycle read and then discarded are passed rather than
    carried on the report: `CycleReport` is what `on_cycle` hands to whoever is
    watching, and widening it with data that only an announcement reads is how
    a future rule finds a decision to take after the plan was computed - the
    one property `plan_reconcile` being pure exists to guarantee.

    The order is the cycle's own: reconcile, recovery, the merge gate, then
    readiness and dispatch. So one task's log reads
    `eligible -> claimed -> result -> pr.opened -> pr.checks -> pr.merged ->
    landed` across the cycles that produced it, and a retry reads
    `result -> eligible -> claimed` in that order.
    """
    results = results or {}
    pulls = pulls or {}
    checks = checks or {}

    # Keyed by `TaskRef`, which is what everything the projection reads is
    # keyed by since #142 - the results directory, the container lookup, the
    # readiness graph and `Transition` itself. `slugs` is the one translation
    # this module makes, and it makes it in that direction on purpose: a
    # `TaskRef` is the *tracker's* handle (GitHub mints `#42`), and #141 is the
    # ticket that says a payload may not carry one.
    entries = {entry.ref: entry for entry in report.ledger.entries.values()}
    slugs = {ref: entry.task_id for ref, entry in entries.items() if entry.task_id}
    # And by issue number, because two things this reads are still spelled that
    # way: the merge gate's `Merge` rows (`checks.py` addresses pull requests
    # and issues, both of which GitHub numbers) and a branch name quoted in
    # prose. #142 retyped the internal model; the code-host half is
    # GitHub-shaped and ADR 0001 says it stays that way. The check sets are no
    # longer among them - #174 moved that map onto the ref, because the same
    # join in `plan_checks` was escalating healthy issues when it missed.
    by_issue = {entry.number: entry for entry in entries.values()}
    by_number = {number: entry.task_id for number, entry in by_issue.items() if entry.task_id}
    # Pull requests, keyed by the task ref inside each head branch rather than
    # by the name itself (#144). `LedgerEntry.branch` names a ticket's *current*
    # attempt, and this projection runs after the cycle folded its transitions -
    # so a failing check set moves the counter, and a name comparison would then
    # find no pull request for the very task whose checks just failed. The
    # `pr.checks failing` announcement is the one nobody can afford to lose.
    by_pull_ref = {
        parsed.ref: state
        for name, state in pulls.items()
        if (parsed := parse_task_branch(name)) is not None
    }
    events: list[TaskEvent] = []

    def emit(name: str, once: tuple[Any, ...] | None = None, **fields: Any) -> None:
        # `cycle` on every payload, because `cycle.reconciled` carries one and a
        # reader that had to bracket these between two of those would be doing
        # the join this ticket exists to remove.
        events.append(TaskEvent(name, {"cycle": report.index, **fields}, once=once))

    # 1. What the workers said. Read from the results directory rather than
    #    from a transition, because the outcome that moves no label - exit 0,
    #    whose `claimed -> review` belongs to the worker (#17) - is the one an
    #    operator is most often waiting for.
    for ref, record in sorted(results.items()):
        task = slugs.get(ref, "")
        if not task:
            continue
        emit(
            TASK_RESULT,
            # Not the attempt alone. Exit 2 does not consume one (§4), so two
            # consecutive infrastructure failures are two records at the same
            # attempt number written to the same filename - and a broken host
            # saying so three times is precisely what an operator needs to see.
            once=(task, record.attempt, _stamp(record)),
            task=task,
            attempt=record.attempt,
            exit_code=record.exit_code,
            outcome=record.outcome,
            duration_s=record.duration_s,
        )

    events += _landed_or_human(report.result.applied, slugs, by_number, report.index)
    if report.recovered is not None:
        # A claim with no container behind it, released or escalated at the cap.
        # `.result.applied` rather than the `.applied` shorthand, because that is
        # the attribute `Reconciler.cycle` folds from - one spelling, so a
        # recovery report shape that satisfies the loop satisfies this too.
        events += _landed_or_human(
            report.recovered.result.applied, slugs, by_number, report.index
        )

    # 2. The merge gate, in the order it ran: mergeability decides what may
    #    merge against the base as it is now, then checks merges it.
    for ref, entry in sorted(entries.items()):
        pull = by_pull_ref.get(entry.ref)
        task = slugs.get(ref, "")
        if pull is None or not task:
            continue
        # Un-minted for the payload, not carried as a `PullRef` (#185). Every
        # field here is written into a JSONL artifact through
        # `json.dumps(default=str)`, which would silently turn `101` into
        # `"#101"` and change the schema every reader of the run log parses.
        emit(
            PR_OPENED,
            once=(task, pull_number(pull.number)),
            task=task,
            pull=pull_number(pull.number),
            head_sha=pull.sha,
        )

    for ref, check_set in sorted(checks.items()):
        task = slugs.get(ref, "")
        pull = by_pull_ref.get(ref) if ref in entries else None
        if not task or pull is None:
            continue
        state = check_set.verdict
        # Which check decided it. One name rather than the three lists, because
        # the question this answers is "what is this pull request waiting on".
        #
        # **Verbatim, and it is the one field here that is.** The name was
        # written by whoever wrote the target repository's workflow, so it
        # cannot carry an apiary issue number, and it is precisely the string a
        # reader pastes into the CI UI to find the run. Scrubbing it would
        # silently rename a check somebody happened to call `swarm:lint`.
        names = check_set.failed or check_set.pending or check_set.succeeded or ("",)
        deciding = names[0]
        emit(
            PR_CHECKS,
            # The head as well as the pull request. A retry reuses the open PR
            # (`worker/pr.py`), so one number cycles pending -> failing ->
            # pending across attempts, and a key without the sha would announce
            # the first attempt's gate and silently swallow every later one.
            # `PullState.sha` can be empty - the listing is the source and a
            # payload without a head is a real answer - and the key then
            # degrades to the coarse form rather than failing.
            once=(task, pull_number(pull.number), pull.sha, state, deciding),
            task=task,
            pull=pull_number(pull.number),
            head_sha=pull.sha,
            state=state,
            check=deciding,
        )

    if report.mergeability is not None:
        events += _landed_or_human(report.mergeability.applied, slugs, by_number, report.index)

    gate = report.checks
    if gate is not None:
        landed = set(gate.merged)
        commits = gate.commit_by_issue
        for merge in gate.plan.merges:
            task = by_number.get(merge.number, "")
            if merge.number not in landed or not task:
                continue
            emit(
                PR_MERGED,
                task=task,
                pull=pull_number(merge.pull),
                merge_commit=commits.get(merge.number, ""),
            )
        events += _landed_or_human(gate.applied, slugs, by_number, report.index)

    # 3. Readiness and dispatch, last, because that is when they ran. A task
    #    that failed this cycle and went back to `swarm:ready` is claimed again
    #    in the next one, and the log should read in that order.
    if report.readiness is not None:
        for verdict in report.readiness.verdicts:
            entry = entries.get(verdict.ref)  # type: ignore[assignment]
            if not verdict.ready or not verdict.task_id or entry is None:
                continue
            emit(
                TASK_ELIGIBLE,
                # **Every ready verdict, not only the ones that moved a label.**
                # The planner creates a task whose blockers are already met
                # already labelled `swarm:ready` (`nodes/planner.py`), so the
                # root task of every plan - and "a run with one task", which is
                # #141's own criterion - never produces a readiness
                # *transition* at all. Eligibility is derived and recomputed
                # each cycle (ADR 0001), so it is projected from the verdict.
                #
                # The key is the task, and `LifecycleLog` forgets it when the
                # task is claimed - so this is once per *episode of being
                # eligible*, not once per run and not once per attempt. Keying
                # on the attempt looked right and was not: exit 2 consumes no
                # attempt (§4), so the re-dispatch is ready at the number
                # already announced, and a goal-gate revival resets nothing at
                # all. Both would have been silent.
                once=(verdict.task_id,),
                task=verdict.task_id,
                attempt=entry.attempt,
                # Every dependency this task declared, all of which are
                # satisfied - that is what made the verdict ready. Already task
                # refs: `ledger` resolves `## Blocked by` numbers when it loads
                # and drops the ones outside the run. Empty means it never had
                # any, which is the root of a plan.
                depends_on=list(entry.depends_on),
            )

    if report.dispatched is not None:
        for sent in report.dispatched.dispatched:
            if not sent.entry.task_id:
                continue
            emit(
                TASK_CLAIMED,
                task=sent.entry.task_id,
                attempt=sent.entry.attempt,
                container=sent.handle.id[:12],
                image=sent.handle.image or "",
            )

    return tuple(events)


def _stamp(record: ResultRecord) -> str:
    """What distinguishes two records that share an attempt number.

    Both timestamps are optional on `ResultRecord` - a record synthesised from
    a container log has neither - so for those the key degrades to the attempt
    alone and a second infrastructure failure at that attempt is announced
    once. The degraded case is the one the field cannot fix, not a bug in it.
    """
    return f"{record.started_at}/{record.finished_at}"


def _landed_or_human(
    transitions: Iterable[Transition],
    slugs: Mapping[TaskRef, str],
    by_number: Mapping[int, str],
    cycle: int,
) -> list[TaskEvent]:
    """`task.landed` and `task.needs_human`, from transitions that **landed**.

    Called for all four writers of a terminal label - the reconciler, the
    recovery sweep, mergeability and the check gate - because ADR 0001 makes
    `needs-human` the one state reported outbound and the one a customer's
    tracker cannot infer, and a substrate that hears it from only some of its
    producers is not one #147 can read from.

    §1.4's malformed-contract escalation is the deliberate exception: that
    issue never parsed, so it is not in `Ledger.entries` and has no task ref to
    key on. An event keyed on nothing is not a timeline entry, and inventing
    one from the issue number is the thing this whole module refuses to do.
    """
    events: list[TaskEvent] = []
    for transition in transitions:
        task = transition.task_id or slugs.get(transition.ref, "")
        if not task:
            continue
        if transition.to_label == DONE:
            events.append(
                TaskEvent(TASK_LANDED, {"cycle": cycle, "task": task}, once=(task,))
            )
        elif transition.to_label == FAILED:
            events.append(
                TaskEvent(
                    TASK_NEEDS_HUMAN,
                    {
                        "cycle": cycle,
                        "task": task,
                        "reason": scrub(transition.reason, by_number),
                        # §4's rule, reported rather than re-derived: exit 2
                        # never consumes an attempt, and neither does an
                        # escalation raised on a failure no attempt could fix.
                        # `attempt is None` is precisely "this transition wrote
                        # no counter". The counter itself is deliberately not
                        # carried: `Transition.attempt` is the value about to
                        # be written and `task.result.attempt` is the one that
                        # just ran, so two events about one attempt would
                        # disagree by one and a reader joining on the pair
                        # would mis-pair them.
                        "attempt_consumed": transition.attempt is not None,
                    },
                    # A task lands, or reaches a human, once. Four modules write
                    # a terminal label and they are disjoint today - but that is
                    # an invariant four files away, and the key makes it local.
                    once=(task,),
                )
            )
    return events


@dataclass
class LifecycleLog:
    """Announces a cycle's task events, once each, through whatever `emit` is.

    Run-scoped, for the same reason `Reconciler._infrastructure` is: "have I
    said this already" is a question about a sequence, and a restart announcing
    a task's standing facts once more is the harmless direction - the log is
    append-only and every payload carries the occurrence it belongs to.
    """

    _announced: set[tuple[Any, ...]] = field(default_factory=set, repr=False)

    def announce(
        self,
        report: CycleReport,
        *,
        emit: Callable[..., Any] | None = None,
        **facts: Any,
    ) -> tuple[TaskEvent, ...]:
        """Emit this cycle's fresh events, and return them."""
        fresh: list[TaskEvent] = []
        for event in lifecycle_events(report, **facts):
            if event.once is not None:
                key = (event.name, *event.once)
                if key in self._announced:
                    continue
                self._announced.add(key)
            # Claiming a task ends its episode of being eligible, so the next
            # time it is ready - a retry, a revival - is a new occurrence and
            # is announced again. Done here rather than in the projection
            # because it is the only thing in this module that remembers
            # anything, and `lifecycle_events` stays pure.
            if event.name == TASK_CLAIMED:
                self._announced.discard((TASK_ELIGIBLE, event.fields["task"]))
            fresh.append(event)
        if emit is not None:
            for event in fresh:
                emit(event.name, **event.fields)
        return tuple(fresh)
