"""The replay corpus: one loader, and it cannot tell a real run from a made-up one.

#145 exists to answer ADR 0001's go/no-go - does derived state reproduce label
state - and the honest way to answer it is to replay real recorded runs through
`orchestrator/derived.py` and diff. **There are none, and this environment
cannot produce any.** `swarm run --new` refuses classic and OAuth tokens by
design (`security.assert_provision_token`), no fine-grained PAT is available
here, and `docs/demo-run.md` records the same wall from 2026-08-14. Four
implementers hit it independently before the maintainer wrote the constraint
into the ticket.

So every run under `runs/` is **synthesised**: an `events.jsonl` stream, a
results directory and a per-cycle observation built by hand to exercise one
transition each. That is a real test of the reducer and no test at all of
whether the reducer's model of reality matches reality, and nothing here
pretends otherwise.

**Which makes the format the deliverable, not the data.** This module is
written so that a genuinely recorded run drops in beside a synthesised one with
no code change: same directory shape, same loader, same assertions. The only
thing `corpus.json` says about provenance is one string in `origin`, and
**nothing in this file or in `test_derived.py` branches on it** - there is a
test that proves that by flipping the field and asserting the replay is
byte-identical. When somebody acquires a working credential, recording a run
and dropping it in `runs/` is the whole of the work, and it retires the largest
risk in epic #140.

## What a corpus run is

    tests/fixtures/runs/<slug>/
      corpus.json       the manifest: origin, what it exercises, expected divergences
      run.json          exactly what `RunArtifacts` writes at startup
      events.jsonl      exactly the #141 lifecycle log
      results/*.json    exactly `worker.result.ResultRecord.to_dict`
      observed.jsonl    one line per cycle - the world, and the control plane

Four of those five files are produced verbatim by a live run today. `read_run`
is called on every corpus directory in `load_corpus` for exactly that reason: a
corpus run that `swarm show` cannot read is a corpus run that has drifted from
the format it claims to be in, and the assertion is cheaper here than the
discovery is later.

`observed.jsonl` is the fifth and the one a recorder has to add. Its shape is
not new state: every field in it is something a cycle **already reads** for its
own reasons - `ContainerManager.find`, `checks.read_pulls`, the branch listing
`recovery` sweeps, `load_results`, and `Ledger.entries`. So the recorder #146
needs is a projection of a cycle's own inputs, not a second source of truth,
which is the property that makes "drop a real run in" a recording problem
rather than a design problem.

## The one field that is not the world

`control` - the `swarm:*` label each task wore at the end of that cycle. It is
in the corpus because it is the thing being diffed against, and it is
deliberately *not* reachable from an `Observation`: the loader hands the world
to `resolve` and the control plane to `diverge`, and the two never meet. See
`derived.py`'s sourcing invariant. `events.jsonl` is loaded and offered for the
same reason it is written - a human reconstructing the run - and the replay
assertions do not read the applied half of it.

## Why divergences are declared rather than forbidden

Three things turned out not to be derivable (`derived.py` names them: the
infrastructure ceiling, a renewed retry budget, a goal-gate revival). A corpus
that asserted "no divergence, ever" could only be committed by leaving those
cases out, which is precisely the tuning that would make the whole exercise
worthless. So each manifest declares the divergences its run should produce,
with a reason, and the harness asserts the set matches **exactly** - an
undeclared divergence fails, and a declared one that stops happening fails too.
The second half matters more than the first: it is what makes #147 notice the
day one of these becomes derivable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from swarm.artifacts import (
    CORPUS_MANIFEST_NAME,
    CORPUS_SCHEMA as ARTIFACTS_CORPUS_SCHEMA,
    EVENT_LOG_NAME,
    OBSERVED_LOG_NAME,
    RESULTS_DIR_NAME,
    read_events,
    read_run,
)
from swarm.github.branches import parse_task_branch
from swarm.github.refs import task_ref
from swarm.taskref import TaskRef
from swarm.orchestrator.derived import (
    AttemptFact,
    Budget,
    ContainerFact,
    Divergence,
    Observation,
    PullFact,
    TaskFact,
)
from swarm.orchestrator.lifecycle import INTERNAL_STATE
from swarm.worker.result import ResultRecord, record_path

#: Where the committed runs live. A directory rather than a roster, so a run
#: added tomorrow is replayed because it exists - the same ratchet
#: `pyproject.toml`'s `packages = ["swarm"]` buys for the type gate, and for the
#: same reason: a list somebody has to remember to update is a list that ends up
#: not naming the interesting case.
RUNS_ROOT = Path(__file__).resolve().parent / "runs"

#: Both names come from `swarm.artifacts` since #146, because a live run now
#: writes both: the shadow window records `observed.jsonl` every cycle and drops
#: the manifest beside it. Two spellings of these would be the seam along which
#: "a recorded run drops in with no code change" quietly stopped being true.
MANIFEST_NAME = CORPUS_MANIFEST_NAME
OBSERVED_NAME = OBSERVED_LOG_NAME

#: Bumped when a field in `observed.jsonl` changes meaning, never when one is
#: added - `artifacts.SCHEMA_VERSION`'s rule, and the loader below obeys the
#: other half of it by ignoring keys it does not know.
#:
#: **From `swarm.artifacts` since #146**, because the recorder there stamps the
#: number this loader checks. Two spellings would be worse than none: the check
#: below only refuses a number *greater* than this one, so a bump made on one
#: side alone leaves the recorder stamping the old number and this loader
#: silently reading new-meaning fields as old ones.
CORPUS_SCHEMA = ARTIFACTS_CORPUS_SCHEMA

#: The two provenances a run can have. `origin` is metadata and nothing branches
#: on it; both constants exist so that a recorded run can be *labelled* as one
#: without anybody having to guess the spelling, and so that
#: `test_derived.py`'s drop-in test has something real to flip.
SYNTHESISED = "synthesised"
RECORDED = "recorded"


class CorpusError(RuntimeError):
    """A corpus run is missing, malformed, or not in the format it claims."""


@dataclass(frozen=True)
class Cycle:
    """One cycle of a run: what the world was, and what the labels said.

    The two halves are separate attributes rather than one object on purpose.
    A caller has to reach for `control` by name to get at the control plane,
    which makes "the resolver read a label" a visible line in a diff rather than
    an attribute access that looks like every other one.
    """

    observation: Observation
    control: dict[str, str]

    @property
    def index(self) -> int:
        return self.observation.cycle


@dataclass(frozen=True)
class ExpectedDivergence:
    """A disagreement this run is *supposed* to produce, and why.

    `why` is prose for a human and is never matched on - `Divergence.key`
    explains why. What is matched on is the four-tuple, so a declaration that
    names the wrong cycle or the wrong pair of states fails as loudly as an
    undeclared one.
    """

    cycle: int
    task: str
    derived: str
    control: str
    why: str = ""

    @property
    def key(self) -> tuple[int, str, str, str]:
        return (self.cycle, self.task, self.derived, self.control)


@dataclass(frozen=True)
class CorpusRun:
    """One replayable run, loaded. Real or synthesised - nothing here knows.

    `events` is carried but not asserted on by the replay: it is the #141 log,
    half of which announces writes that landed, and reading that half would be
    the sourcing violation `derived.py` exists to avoid. It is here because a
    human debugging a divergence wants the timeline, and because a corpus whose
    event log had quietly stopped matching its observations would be a corpus
    nobody could trust - `test_derived.py` cross-checks the two.
    """

    name: str
    path: Path
    origin: str
    describes: str
    exercises: tuple[str, ...]
    cycles: tuple[Cycle, ...]
    expected: tuple[ExpectedDivergence, ...]
    events: tuple[dict[str, Any], ...]
    results: tuple[ResultRecord, ...]

    @property
    def expected_keys(self) -> set[tuple[int, str, str, str]]:
        return {one.key for one in self.expected}

    def reason_for(self, divergence: Divergence) -> str:
        for one in self.expected:
            if one.key == divergence.key:
                return one.why
        return ""

    def __str__(self) -> str:
        return f"{self.name} ({self.origin}, {len(self.cycles)} cycle(s))"


def corpus_runs(root: Path | None = None) -> tuple[CorpusRun, ...]:
    """Every committed run, in name order. The entry point the suite parametrises on."""
    base = RUNS_ROOT if root is None else root
    if not base.is_dir():
        raise CorpusError(f"{base} is not a directory")
    found = tuple(
        load_corpus(child)
        for child in sorted(base.iterdir())
        if child.is_dir() and (child / MANIFEST_NAME).is_file()
    )
    if not found:
        # An empty corpus would make every replay test pass by vacuum, which is
        # the one failure mode a harness must not have.
        raise CorpusError(f"no corpus runs under {base}")
    return found


def load_corpus(path: Path) -> CorpusRun:
    """Load one run directory. **The same loader for a recorded run.**

    `read_run` is called and its result discarded, which looks wasteful and is
    the point: it is `swarm show`'s reader, so a directory that passes here is a
    directory a live run could have written. A corpus that drifted into a shape
    only this loader understood would still replay green while having stopped
    being evidence about anything.
    """
    manifest = _read_json(path / MANIFEST_NAME)
    schema = int(manifest.get("schema", CORPUS_SCHEMA))
    if schema > CORPUS_SCHEMA:
        raise CorpusError(
            f"{path.name}: corpus schema {schema} is newer than this loader's {CORPUS_SCHEMA}"
        )
    try:
        read_run(path)
    except Exception as exc:  # noqa: BLE001 - the message is the whole value here
        raise CorpusError(f"{path.name}: not readable as a run directory: {exc}") from exc

    results = _load_results(path / RESULTS_DIR_NAME)
    default_run_id = str(_read_json(path / "run.json").get("run_id") or path.name)
    cycles = tuple(
        _cycle(line, results=results, default_run_id=default_run_id, where=path.name)
        for line in _read_lines(path / OBSERVED_NAME)
    )
    if not cycles:
        raise CorpusError(f"{path.name}: {OBSERVED_NAME} records no cycle")

    indexes = [cycle.index for cycle in cycles]
    if indexes != sorted(indexes) or len(set(indexes)) != len(indexes):
        # Out of order or repeated, either of which would make a divergence's
        # `cycle` field ambiguous - and the cycle number is half of what makes a
        # divergence nameable rather than a count.
        raise CorpusError(f"{path.name}: cycle indexes are not strictly increasing: {indexes}")

    return CorpusRun(
        name=path.name,
        path=path,
        origin=str(manifest.get("origin") or SYNTHESISED),
        describes=str(manifest.get("describes") or ""),
        exercises=tuple(str(one) for one in manifest.get("exercises") or ()),
        cycles=cycles,
        expected=tuple(
            ExpectedDivergence(
                cycle=int(one["cycle"]),
                task=str(one["task"]),
                derived=str(one["derived"]),
                control=str(one["control"]),
                why=str(one.get("why") or ""),
            )
            for one in manifest.get("expected_divergences") or ()
        ),
        events=read_events(path / EVENT_LOG_NAME),
        results=results,
    )


# --------------------------------------------------------------------------
# One line of observed.jsonl
# --------------------------------------------------------------------------


def _cycle(
    line: Mapping[str, Any],
    *,
    results: Sequence[ResultRecord],
    default_run_id: str,
    where: str,
) -> Cycle:
    """Turn one recorded observation into the object `resolve` takes.

    Everything that needs interpreting is interpreted **here**, by the same
    functions a live cycle would use, rather than being pre-chewed in the JSON:
    a pull request is recorded by its head branch name and joined to a task
    through `parse_task_branch` (#144's rule, and `mergeability.py`'s "a task is
    a ref; an API address is a number"), and a branch listing is recorded as the
    raw names a remote would hand back. A corpus that stored `{"ref": "#12",
    "attempt": 1}` instead would be recording the loader's answer rather than
    the code host's fact, and the join - which is the thing #174 found failing
    silently in production code - would never be exercised at all.
    """
    index = int(line["cycle"])
    tasks = tuple(
        TaskFact(
            ref=_ref(one["ref"]),
            task_id=str(one["task_id"]),
            depends_on=tuple(_ref(dep) for dep in one.get("depends_on") or ()),
            closed=bool(one.get("closed", False)),
            state_reason=one.get("state_reason"),
        )
        for one in line.get("tasks") or ()
    )
    containers = tuple(
        ContainerFact(
            id=str(one["id"]),
            run_id=str(one.get("run_id") or default_run_id),
            ref=None if one.get("ref") is None else _ref(one["ref"]),
            running=bool(one.get("running", False)),
        )
        for one in line.get("containers") or ()
    )

    pulls: list[PullFact] = []
    for one in line.get("pulls") or ():
        head = str(one.get("head") or "")
        branch = parse_task_branch(head)
        if branch is None:
            # A pull request whose head apiary did not mint. Dropped, exactly as
            # `lifecycle.lifecycle_events` drops it: a human's PR against the
            # same repository is not a task's review state, and a corpus that
            # could not record one could not test that it is ignored.
            continue
        pulls.append(
            PullFact(
                number=int(one["number"]),
                ref=branch.ref,
                attempt=branch.attempt,
                merged=bool(one.get("merged", False)),
                closed=bool(one.get("closed", False)),
                draft=bool(one.get("draft", False)),
                head_sha=str(one.get("sha") or ""),
            )
        )

    visible = frozenset(str(name) for name in line.get("results") or ())
    facts = tuple(
        AttemptFact(
            ref=task_ref(record.issue), attempt=record.attempt, exit_code=record.exit_code
        )
        for record in results
        if _result_name(record) in visible
    )
    missing = visible - {_result_name(record) for record in results}
    if missing:
        # A cycle claiming to have read a record the directory does not hold is
        # a corpus editing mistake that would otherwise show up as a mysterious
        # off-by-one in an attempt count several assertions later.
        raise CorpusError(f"{where} cycle {index}: no such result file(s): {sorted(missing)}")

    budget = line.get("budget") or {}
    observation = Observation(
        cycle=index,
        tasks=tasks,
        branches=tuple(
            branch
            for branch in (parse_task_branch(str(name)) for name in line.get("branches") or ())
            if branch is not None
        ),
        containers=containers,
        pulls=tuple(pulls),
        results=facts,
        budget=Budget(
            max_attempts=int(budget.get("max_attempts", 3)),
            max_total_attempts=int(budget.get("max_total_attempts", 9)),
        ),
        live_run_ids=frozenset(
            str(one) for one in line.get("live_run_ids", [default_run_id]) or ()
        ),
    )
    return Cycle(observation=observation, control=_control(line.get("control") or {}, where, index))


def _control(raw: Mapping[str, Any], where: str, index: int) -> dict[str, str]:
    """The `swarm:*` labels, translated into ADR 0001's vocabulary.

    Translated here and not in `derived.py`, which has never seen a label and
    must not learn: `lifecycle.INTERNAL_STATE` is the mapping and it belongs to
    whoever reads a label. Recording the label rather than the internal state is
    the honest direction - the corpus records what the control plane actually
    held, and the day epic #140 removes the labels this translation is what gets
    deleted, not the data.
    """
    states: dict[str, str] = {}
    for task, label in raw.items():
        text = str(label)
        if text not in INTERNAL_STATE:
            raise CorpusError(f"{where} cycle {index}: {text!r} is not a state label")
        states[str(task)] = INTERNAL_STATE[text]
    return states


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def _ref(value: Any) -> TaskRef:
    """A ref as the corpus spells it. Through the adapter, never constructed here.

    `github/refs.task_ref` is the only minter (#142/#166), so a corpus recording
    `"#12"` is parsed back to a number and re-minted rather than being handed to
    `TaskRef` directly. It costs a round trip and it means a corpus cannot smuggle
    in a ref shape the adapter would never produce.
    """
    text = str(value)
    if not text.startswith("#") or not text[1:].isdigit():
        raise CorpusError(f"{text!r} is not a ref this adapter mints")
    return task_ref(int(text[1:]))


def _result_name(record: ResultRecord) -> str:
    """`worker.result.record_path`'s name, never a second spelling of it.

    The recorder (`orchestrator/shadow.observed_line`) builds the same name the
    same way, so a rename in `worker/result.py` moves both sides at once."""
    return record_path("", record.issue, record.attempt).name


def _load_results(directory: Path) -> tuple[ResultRecord, ...]:
    if not directory.is_dir():
        return ()
    return tuple(
        ResultRecord.from_dict(_read_json(child)) for child in sorted(directory.glob("*.json"))
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CorpusError(f"{path} is missing") from exc
    except ValueError as exc:
        raise CorpusError(f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorpusError(f"{path}: expected an object")
    return payload


def _read_lines(path: Path) -> Iterator[dict[str, Any]]:
    """JSON lines, strictly.

    `artifacts.read_events` skips a line that does not parse, because the last
    line of a killed run's log is *expected* to be half-written. This one
    refuses, because a corpus is a committed file rather than a live append: a
    cycle silently dropped here is a cycle whose divergence silently stops being
    asserted, which is the harness failing open.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CorpusError(f"{path} is missing") from exc
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError as exc:
            raise CorpusError(f"{path}:{number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CorpusError(f"{path}:{number}: expected an object")
        yield payload


__all__ = [
    "CORPUS_SCHEMA",
    "MANIFEST_NAME",
    "OBSERVED_NAME",
    "RECORDED",
    "RUNS_ROOT",
    "SYNTHESISED",
    "CorpusError",
    "CorpusRun",
    "Cycle",
    "ExpectedDivergence",
    "corpus_runs",
    "load_corpus",
]
