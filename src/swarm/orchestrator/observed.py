"""Building a cycle's `Observation`, and recording it.

**Split out of `orchestrator/shadow.py` (#152 c1), and the split is the point.**
Two things lived in that module and only one of them is a migration instrument.

The *window* — `ShadowWindow`, `classify`, `ShadowReport` — runs the resolver
beside the label control plane and reports what disagrees. It exists to make
#147's cutover checkable and it dies with the labels it compares against, which
is `#152`'s AC5.

The *recorder* — this module — outlives all of it. `tests/fixtures/runs/README.md`
names the missing recorder as the whole reason #145's replay corpus is
synthesised, and a synthesised corpus proves the reducer self-consistent and
nothing about reality. Every real run that writes `observed.jsonl` is a run the
corpus harness can replay, and that value has nothing to do with labels.

Keeping them in one file meant the deletion of the first could not be done
without reading around the second. Now `shadow.py` imports from here, and the
step that removes the window deletes a file rather than dissecting one.

**Nothing here changed in the split.** The functions are the ones `shadow.py`
had, moved verbatim; `shadow.py` re-exports them so every existing caller and
every existing test is untouched. This module has no reference back to the
window, which is what makes the eventual deletion a one-line import removal.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..containers.manager import Handle
from ..github.branches import parse_task_branch, task_branch
from ..github.readiness import READY as READY_LABEL
from ..github.readiness import IssueState
from ..github.refs import issue_number, pull_number, task_ref
from ..taskref import TaskRef
from ..worker.result import ResultRecord, record_path
from .authority import revived_tasks
from .checks import PullState
from .derived import (
    AttemptFact,
    Budget,
    ContainerFact,
    Observation,
    PullFact,
    observe,
)
from .dispatcher import CLAIMED as CLAIMED_LABEL
from .lifecycle import INTERNAL_STATE, state_label
from .reconcile import CycleReport

__all__ = [
    "control_labels",
    "build_observation",
    "observation_for",
    "observed_line",
]


# --------------------------------------------------------------------------
# The two sides
# --------------------------------------------------------------------------


def control_labels(report: CycleReport) -> dict[str, str]:
    """The `swarm:*` label each task wore when this cycle finished. By task id.

    Five sources, in the order the cycle writes them, so the last word wins:

    1. `report.ledger`, which `cycle` has already folded with what `apply_plan`,
       the recovery sweep and the check gate **applied** - `fold`'s rule, and
       for `fold`'s reason: a label write GitHub refused left the task where it
       was, and crediting it would diff against a state the control plane never
       reached.
    2. mergeability, which is the fourth writer of a terminal label
       (`lifecycle._landed_or_human` names all four) and the one the cycle does
       **not** fold back: a pull request that will not rebase inside its update
       budget is escalated here, and a control map built without it would report
       every starved task as a divergence the resolver invented. Skipped for a
       task the check gate also wrote, because the gate runs after this one and
       step 1 already carries its answer.
    3. the readiness pass, which writes `swarm:ready` / `swarm:blocked` and
       whose result the cycle does not fold back into the ledger either. Its
       verdicts cover exactly the transitionable entries, so nothing here can
       overwrite a `claimed`, `review`, `done` or `failed` that step 1
       established.
    4. the dispatcher's claims, which are written after readiness and are also
       not folded.
    5. a dispatch that claimed and then failed to spawn. `DispatchFailure.claimed`
       is precisely "the label was written and no container is running under
       it", which is a claim the control plane is holding whether or not the
       spawn worked - and the case #35's recovery sweep exists for.

    And a sixth that is not a label writer in `cycle` at all: `planner.revive`,
    reached from step 5 through `replan` and from the goal gate, moves a task
    `swarm:failed -> swarm:ready` on GitHub. `_judge` runs *before* this window
    and nothing folds it either, so a map without it reports `needs-human` for a
    task the cycle left `ready`. Overlaid before readiness, because that is
    where it happens in the cycle and because a revived task is exactly the one
    readiness may then move again.

    Labels rather than internal states, because that is what the replay corpus
    records (`tests/fixtures/runs/README.md`: "the corpus records what the
    control plane actually held, so the day epic #140 removes the labels it is
    the translation that gets deleted and not the data"). The translation
    happens once, on the way into `diverge`.
    """
    labels: dict[str, str] = {}
    for entry in report.ledger.entries.values():
        if entry.task_id:
            labels[entry.task_id] = entry.state_label
    if report.mergeability is not None:
        gated = {
            transition.task_id
            for transition in getattr(report.checks, "applied", ()) or ()
            if transition.task_id
        }
        for transition in report.mergeability.applied:
            if transition.task_id and transition.task_id not in gated:
                # Translated *back* to a label on purpose. A `Transition` carries
                # a state since #152, and this function's whole subject is what
                # the label control plane was left holding - the map is compared
                # against the resolver's states by the caller, so keeping the
                # state here would compare the derived answer with itself and
                # report agreement that was never tested.
                labels[transition.task_id] = state_label(transition.to_state)
    for task in revived_tasks(report):
        labels[task] = READY_LABEL
    if report.readiness is not None:
        for verdict in report.readiness.verdicts:
            if verdict.task_id:
                labels[verdict.task_id] = verdict.label
    if report.dispatched is not None:
        slugs = {entry.ref: entry.task_id for entry in report.ledger.entries.values()}
        for item in report.dispatched.dispatched:
            if item.entry.task_id:
                labels[item.entry.task_id] = CLAIMED_LABEL
        for failure in report.dispatched.failed:
            # `DispatchFailure` carries the issue number rather than the entry,
            # so the ref is re-minted through the adapter and joined back to the
            # slug the ledger holds - `mergeability.py`'s rule, a task is a ref
            # and an API address is a number.
            task = slugs.get(task_ref(int(failure.number)), "")
            if failure.claimed and task:
                labels[task] = CLAIMED_LABEL
    return labels


def build_observation(
    *,
    cycle: int,
    entries: Iterable[Any],
    containers: Iterable[Handle] = (),
    pulls: Mapping[str, PullState] | None = None,
    results: Mapping[TaskRef, ResultRecord] | None = None,
    states: Mapping[TaskRef, IssueState] | None = None,
    budget: Budget | None = None,
    live_run_ids: Iterable[str] = (),
) -> Observation:
    """One `Observation` from raw cycle inputs. **Reads nothing.**

    Every argument is something `Reconciler.cycle` computed for its own reasons,
    which is what makes "shadowing adds no API call" a structural claim rather
    than a promise: there is no client here to call one with.

    **Raw inputs rather than a `CycleReport`**, with `observation_for` below as
    the adapter. All of these are read at the *top* of a cycle, so an
    observation can be built before the cycle decides anything - which is the
    shape #147 needs when the derived state becomes the thing decided *on*, and
    is a free property to keep now rather than a reshape later.

    `containers` is the **raw listing, one entry per container**, not
    `Reconciler._handles`' first-wins map. Two containers under one task is the
    double-spawn `dispatcher.release` is written about and one of the cases #146
    gives as a reason to shadow at all; a collapsed map cannot express it, and
    an exited container listed ahead of a running one would additionally make
    the resolver read not-claimed. `ContainerFact` is per-container by design.

    Two inputs are narrower than `derived.py` would like, and saying so is more
    useful than pretending otherwise:

    - **Branch names are the head refs of open pull requests, and nothing more.**
      A remote branch listing is not a call this cycle makes, and #146's
      acceptance criteria forbid adding one. `attempts_spent` takes the maximum
      of three lower bounds, so a missing source lowers the bound rather than
      corrupting it - and the pull request carries the same attempt the branch
      does, which is `derived.py`'s own argument for keeping them separate.
    - **Merged pull requests are absent.** `Snapshot.pull_requests` lists open
      ones only, deliberately, because a merge closes its issue through
      `Closes #<n>` and the issue listing already carries it. So `landed`
      arrives here through `TaskFact.closed_as_completed`, which is the second
      of the two paths `derived._landed` documents - and the reason
      `state_reason` is passed at all.
    """
    pulls = pulls or {}
    results = results or {}
    states = states or {}

    facts: list[PullFact] = []
    for branch, pull in pulls.items():
        parsed = parse_task_branch(str(branch))
        if parsed is None:
            # A pull request whose head this system did not mint - a human's,
            # against the same repository. Dropped exactly as
            # `lifecycle.lifecycle_events` and the corpus loader drop it.
            continue
        facts.append(
            PullFact(
                # Both sides are `PullRef` since #208, so the un-mint that used
                # to stand here (`pull_number(pull.number)`) is gone rather than
                # merely justified: this is orchestrator-to-orchestrator, not an
                # API boundary, and the only reason it un-minted was that
                # `PullFact.number` could not hold the type. Nothing left in
                # this hop can put a pull request's number where an issue's
                # belongs, which is what #185 was for.
                number=pull.number,
                ref=parsed.ref,
                attempt=parsed.attempt,
                draft=bool(pull.draft),
                head_sha=str(pull.sha or ""),
            )
        )

    return observe(
        cycle=cycle,
        entries=entries,
        branch_names=[str(name) for name in pulls],
        containers=[
            ContainerFact(
                id=handle.id,
                run_id=handle.run_id,
                ref=task_ref(int(handle.issue)),
                running=handle.running,
            )
            for handle in containers
            if handle.issue is not None
        ],
        pulls=facts,
        results=[
            AttemptFact(ref=ref, attempt=record.attempt, exit_code=record.exit_code)
            for ref, record in results.items()
        ],
        budget=budget,
        live_run_ids=live_run_ids,
        state_reasons={ref: state.state_reason for ref, state in states.items()},
    )


def observation_for(report: CycleReport, **facts: Any) -> Observation:
    """`build_observation` over a finished cycle. The two lines a report adds.

    A thin adapter and nothing more, so that the builder above stays usable
    before a cycle has decided anything: the ledger and the cycle index are the
    only things a `CycleReport` contributes, and both are read at the top of the
    cycle rather than produced by it.
    """
    return build_observation(
        cycle=report.index, entries=report.ledger.entries.values(), **facts
    )


# --------------------------------------------------------------------------
# The recorder
# --------------------------------------------------------------------------


def observed_line(
    observation: Observation,
    control: Mapping[str, str],
    *,
    result_names: Mapping[TaskRef, str] | None = None,
) -> dict[str, Any]:
    """One `observed.jsonl` line, in the shape `tests/fixtures/corpus.py` loads.

    A projection of an `Observation` that has already been built, so nothing
    here reads anything and nothing here can disagree with what was resolved.
    The one asymmetry is `results`: the corpus records the *file names* a cycle
    could see rather than the records themselves. `result_names` is those
    names, carried here from the read that produced `observation.results`
    (`Reconciler._results`) rather than looked up again - this function still
    reads nothing, which is what stops it disagreeing with what was resolved.

    **The name cannot be rebuilt from the record.** Since #177 `write_result`
    bumps the *filename* on a collision and leaves the record's `attempt`
    alone, so several records for one issue share an attempt and live under
    different names; `record_path(issue, attempt)` names the first of them
    whichever one this cycle actually read. That only changes a replayed fact
    when two records at one attempt disagree on their exit code - an
    infrastructure failure and a task failure at the same attempt number, where
    the difference is `AttemptFact.spends_budget` - but it is a fact about the
    run either way, and #230 is the loader half of the same seam.

    The rebuilt name is still the fallback, for a caller that has an
    `Observation` and no directory: a hand-written line and every test that
    predates the argument. It is exactly right whenever no two records share an
    attempt, which is every run that never hit infrastructure trouble.

    Labels rather than internal states in `control`, because that is the
    corpus's own decision and its reason is good: the day the labels go away it
    is the translation that gets deleted, not the recorded data.

    Tasks whose label is not one of the six are dropped rather than written.
    `load_corpus` refuses a label it cannot translate - correctly - and a
    recorder that could emit an unloadable line would produce corpus runs that
    fail at the door for a reason that has nothing to do with the resolver.
    """
    return {
        "cycle": observation.cycle,
        "tasks": [
            {
                "ref": str(fact.ref),
                "task_id": fact.task_id,
                "depends_on": [str(dep) for dep in fact.depends_on],
                "closed": fact.closed,
                "state_reason": fact.state_reason,
            }
            for fact in observation.tasks
        ],
        "branches": [task_branch(branch.ref, branch.attempt) for branch in observation.branches],
        "containers": [
            {
                "id": container.id,
                "run_id": container.run_id,
                "ref": None if container.ref is None else str(container.ref),
                "running": container.running,
            }
            for container in observation.containers
        ],
        "pulls": [
            {
                # Un-minted, because this one *is* a boundary: the corpus format
                # `tests/fixtures/runs/README.md` documents holds a JSON number,
                # every existing fixture carries one, and a `PullRef` would not
                # serialise. The loader mints it back (`fixtures/corpus.py`), so
                # the type lives inside the process and the file stays the file.
                "number": pull_number(pull.number),
                "head": task_branch(pull.ref, pull.attempt),
                "merged": pull.merged,
                "closed": pull.closed,
                "draft": pull.draft,
                "sha": pull.head_sha,
            }
            for pull in observation.pulls
        ],
        "results": [
            (result_names or {}).get(record.ref)
            or record_path("", _issue_of(record.ref), record.attempt).name
            for record in observation.results
        ],
        "budget": {
            "max_attempts": observation.budget.max_attempts,
            "max_total_attempts": observation.budget.max_total_attempts,
        },
        "live_run_ids": sorted(observation.live_run_ids),
        "control": {
            task: label
            for task, label in sorted(control.items())
            if label in INTERNAL_STATE
        },
    }


def _issue_of(ref: TaskRef) -> int:
    """The issue number a result file is named after. Through the adapter.

    `record_path` takes an `int` because the worker files its record under the
    number the code host gave the issue, and `github.refs.issue_number` is the
    only module allowed to turn a ref back into one (#142/#166). A ref this
    adapter did not mint has no result file under any name, and `0` produces a
    corpus line the loader rejects loudly rather than one that quietly points
    at the wrong record.
    """
    try:
        return issue_number(ref)
    except ValueError:
        return 0
