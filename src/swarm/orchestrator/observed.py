"""Building a cycle's `Observation`, and recording it.

**The window this was split out of is gone (#152).** #245 separated the two
things that lived in `orchestrator/shadow.py`, and said why the split was worth
doing on its own: only one of them was a migration instrument. The *window* -
`ShadowWindow`, `classify`, `ShadowReport` - ran the resolver beside the label
control plane to make #147's cutover checkable, and it died with the labels it
compared against. That is this commit. Because the split landed first, removing
it was deleting a file rather than dissecting one.

The *recorder* - this module - outlives all of it, for a reason that has nothing
to do with the cutover. `tests/fixtures/runs/README.md` names the missing
recorder as the whole reason #145's replay corpus is synthesised, and a
synthesised corpus proves the reducer self-consistent and nothing about reality.
Every real run that writes `observed.jsonl` is a run the corpus harness can
replay.

Four things live here, and none of them reads anything:

- `build_observation` turns the facts a cycle already computed into a
  `derived.Observation`. There is no client in this module, which is what makes
  "recording adds no API call" structural rather than a promise.
- `observation_for` is the same thing from a finished `CycleReport`.
- `observed_line` projects an `Observation` into one `observed.jsonl` line, in
  the shape `tests/fixtures/corpus.py` loads.
- `record_cycle` is the one call a cycle makes, and the only one with a policy in
  it: a cycle that could not see records nothing.

`control_labels` is the fifth and the one with a date on it. The corpus format
records what the control plane was left holding, as labels, and
`tests/fixtures/runs/README.md` documents it that way - so it stays until the
labels themselves go, which is the ticket that also decides what a `control`
field means once there is no control plane to have one. Recording it is not
believing it: nothing in a cycle reads this back.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

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
from .lifecycle import INTERNAL_STATE
from .reconcile import CycleReport

__all__ = [
    "control_labels",
    "build_observation",
    "observation_for",
    "observed_line",
    "record_cycle",
]


# --------------------------------------------------------------------------
# The two sides
# --------------------------------------------------------------------------


def control_labels(report: CycleReport) -> dict[str, str]:
    """What this cycle believed, by task id. **No longer a second opinion.**

    This assembled the `swarm:*` label each task wore when the cycle finished,
    from five writers in the order the cycle wrote them, so that a recorded run
    carried *both* sides of #147's comparison: the derived world, and the
    control plane the cycle had actually left behind. Replaying the two against
    each other is what `docs/recording-runs.md` §2 grades a run on.

    #152 removed the control plane, so there is no second side to record. What
    is written now is the cycle's own belief - which is the derived answer, and
    therefore agrees with the resolver by construction. **A run recorded from
    here on cannot be part of that gate**, and the runbook says so in as many
    words: "a run recorded after it carries an empty `control` and can never be
    part of this gate."

    The field is kept rather than dropped, and the belief is what fills it, for
    two reasons that are not the same:

    - `observed.jsonl` is append-only and read back. `tests/fixtures/corpus.py`
      parses `control`, and runs recorded *before* this ticket still hold a real
      control plane in it. Removing the key would make those runs unreadable,
      which is the archive this whole exercise exists to have produced.
    - What a reader of a *new* recording wants from that slot is "what did the
      orchestrator think", and the belief is exactly that. It is honest as a
      record and worthless as a comparison, which is a distinction this
      docstring has to carry because nothing in the data will.

    Labels rather than states was the old spelling, and it went with them: the
    values here are ADR 0001's internal states.
    """
    states = getattr(report.belief, "states", None) or {}
    return {task_id: state for task_id, state in states.items() if state}




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


# --------------------------------------------------------------------------
# The one call a cycle makes
# --------------------------------------------------------------------------


def record_cycle(
    report: CycleReport,
    *,
    record: Callable[..., Any] | None,
    containers: Iterable[Handle] = (),
    pulls: Mapping[str, PullState] | None = None,
    results: Mapping[TaskRef, ResultRecord] | None = None,
    result_names: Mapping[TaskRef, str] | None = None,
    states: Mapping[TaskRef, IssueState] | None = None,
    max_attempts: int = 3,
    max_total_attempts: int = 9,
    live_run_ids: Iterable[str] = (),
) -> bool:
    """Write one `observed.jsonl` line for a finished cycle.

    **`pulls=None` is "this cycle could not look", and it is not `{}`.**
    `checks.read_pulls` and `Snapshot.open_branches` both go to lengths to keep
    those apart. An empty mapping read as the answer would record every task in
    review as having no pull request, which is a corpus line that replays to a
    world that never existed - worse than no line at all, because a fixture is
    trusted. So a blind cycle records nothing and says so by returning `False`.

    **This may raise, and the caller is where that is handled.**
    `ShadowWindow.run` wrapped its whole body in one `except Exception`, and the
    argument for it survives the window: this code reads a dozen attributes off
    objects five other modules own, and the day one of them is renamed the right
    outcome is a recorder that stops and says so rather than a run that dies
    holding containers. What does not survive is putting the guard *here* - "have
    I already given up" is run-scoped, the run is the caller's, and a flag on a
    module-level function would be state shared between two `Reconciler`s. So
    `Reconciler._record_observed` owns both the `try` and the flag, and this
    function stays a projection.

    Returns whether a line was written, so a caller can tell "nothing to record"
    from "something went wrong" without reading its own stderr.
    """
    if record is None or pulls is None:
        return False
    observation = observation_for(
        report,
        containers=list(containers),
        pulls=pulls,
        results=results,
        states=states,
        budget=Budget(max_attempts=max_attempts, max_total_attempts=max_total_attempts),
        live_run_ids=live_run_ids,
    )
    record(observed_line(observation, control_labels(report), result_names=result_names))
    return True
