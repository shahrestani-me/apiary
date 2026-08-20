"""#147's gate, replayed: ten recorded runs with no unexplained divergence.

`docs/recording-runs.md` §2 is the whole of this file. It asks three things of a
run before it counts, and each is asserted below.

**Why these are not in `tests/fixtures/runs/`.** That directory is the *reducer*
corpus, and it holds runs that **finished** - `test_derived`'s
`test_every_corpus_run_reaches_a_terminal_state_or_declares_why_not` requires the
last cycle's control plane to be entirely terminal, which is what makes it a fair
test of the reducer against a complete history.

These ten were stopped by `--max-cycles`, not by finishing. Putting them in that
directory would have meant relaxing that invariant, and the invariant is right:
a truncated run says nothing about whether the reducer reaches the end correctly.

They are still exactly what the gate asks for, because the gate is a different
question - "does derived state reproduce the control plane, cycle by cycle, on
real runs" - and a truncated recording answers it for every cycle it contains.
Two checks, two preconditions, kept apart rather than merged into one that is
weaker than either.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures.corpus import CorpusRun, load_corpus
from swarm.orchestrator.derived import diverge, resolve

RECORDED_ROOT = Path(__file__).parent / "fixtures" / "recorded"

RUNS: tuple[CorpusRun, ...] = tuple(
    load_corpus(child)
    for child in sorted(RECORDED_ROOT.iterdir())
    if child.is_dir() and (child / "corpus.json").is_file()
)


def test_there_are_ten_of_them() -> None:
    """#147's gate is *ten* consecutive runs. Nine is not the gate."""
    assert len(RUNS) == 10, f"the gate is ten recorded runs; found {len(RUNS)}"


@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_the_run_was_recorded_rather_than_synthesised(run: CorpusRun) -> None:
    """A synthesised corpus proves the reducer self-consistent and nothing about
    reality (`tests/fixtures/runs/README.md`). Replaying the wrong directory is
    the easiest way to believe you have evidence you do not."""
    assert run.origin == "recorded", f"{run.name} is {run.origin}, not a real run"


@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_the_run_measured_something(run: CorpusRun) -> None:
    """No cycles is a run that measured nothing. Unmeasured is not clean."""
    assert run.cycles, f"{run.name} has no cycles"


@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_every_divergence_is_explained(run: CorpusRun) -> None:
    """The gate itself: `unexplained=0`, not `divergences=0`.

    An *expected* divergence is evidence the model is right - ADR 0001 names
    three states no code-host fact can derive, and they diverge by construction.
    What must not exist is one nobody accounted for.
    """
    unexplained = [
        one
        for cycle in run.cycles
        for one in diverge(resolve(cycle.observation), cycle.control)
        if not run.reason_for(one)
    ]
    assert not unexplained, (
        f"{run.name}: {len(unexplained)} unexplained divergence(s), "
        f"first {unexplained[0]}"
    )


@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_the_control_plane_was_still_being_written(run: CorpusRun) -> None:
    """The deadline `docs/recording-runs.md` §2 calls "the real one".

    `control` is populated only while the labels are written. A run recorded
    after #152 carries an empty one and can never be part of this gate - so
    these ten are also a record of *when* they were taken, and a future run
    dropped in here with an empty control plane would be caught rather than
    counted.
    """
    assert any(cycle.control for cycle in run.cycles), (
        f"{run.name} has an empty control plane on every cycle: it was recorded "
        f"after the label writes were removed and proves nothing"
    )
