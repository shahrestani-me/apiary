"""The ratchet #168 bought, held still.

`#142` (merged as `#166`) retyped the internal model from issue numbers to an
opaque `TaskRef` across 24 files and passed all 1480 tests **while broken**:
`Reconciler._results()` became ref-keyed and `judge.Observation.of` still read
`results.get(entry.number)`. A dict miss returns rather than raises, so every
signal read as "no evidence" - arithmetic loop detection stopped working,
`needs_human` could never fire and `replan.brief` rendered empty. A human-style
review caught it; nothing mechanical did, because CI ran pytest and nothing
else.

Two things are asserted here, and they fail for different reasons.

**One: the checker still catches that defect.** `test_ref_keyed_map_read_with_an
_int_is_an_error` is the seam itself, in miniature, against the real `TaskRef` -
a `Mapping[TaskRef, ...]` read with `entry.number`. It is written as a pair, the
wrong key and the right one, because a checker that reports an error on both is
no better than one that reports it on neither. If somebody relaxes the settings
far enough that this stops failing, the gate is decorative and this test says so
rather than the next refactor saying so six months later.

**One and a half: the checker catches a *mis-sourcing*, not only a retype.**
#185. `Merge` carries `number` (the issue) beside `pull` (the pull request), and
while both were `int` a `Merge` built from the pull request's number twice was
well-typed: it minted a perfectly valid `TaskRef`, just one addressing a pull
request, so the refusal was filed under an identity no outcome answered to and
`swarm:done` went out for a merge that never happened. Nothing above caught that
- the assertion in the first group is about a *key type*, and both halves of
this one were `int`. `test_a_pull_requests_number_in_the_issues_place_is_an_error`
is that defect in miniature, and its control is the same construction sourced
correctly.

**Two: the exclusions cannot quietly become a blanket.** The backlog is excluded
two ways and no others - per-module `disable_error_code` lists in
`pyproject.toml`, and `# type: ignore[code]` at the site in the modules where a
type error is silent. Both are countable and both are meant to shrink. What
would kill the ratchet is not a large backlog, it is a cheap way to add to it:
an `ignore_errors`, a wildcard relaxation covering a whole package, a bare
`# type: ignore`, or a target list of files that a new module simply never joins.
Each of those has an assertion below.

The second group is deliberately structural rather than a snapshot. **There is no
error count and no file roster here**, because either would go stale the moment
any of the typed refactors still in flight (#144, #159, #170) lands, and a gate
that fails on unrelated work is a gate somebody deletes.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Packages where a type error is silent rather than loud - a dict miss returns.
#: Nothing may be switched off at module level inside these; see `pyproject.toml`.
PRIORITY = ("swarm.orchestrator", "swarm.github", "swarm.nodes", "swarm.taskref")


@pytest.fixture(scope="module")
def mypy_config() -> dict:
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["mypy"]
    return settings


def overrides(config: dict) -> list[dict]:
    return list(config.get("overrides", ()))


def modules_of(override: dict) -> list[str]:
    module = override["module"]
    return [module] if isinstance(module, str) else list(module)


# --------------------------------------------------------------------------
# One: the defect is still caught
# --------------------------------------------------------------------------


REPRODUCTION = '''\
from typing import Mapping

from swarm.taskref import TaskRef


def observe(results: Mapping[TaskRef, str], number: int, ref: TaskRef) -> None:
    results.get(number)  # the #142 defect
    results.get(ref)  # what #166 fixed it to
'''


@pytest.fixture(scope="module")
def reproduction_findings(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Run the project's own mypy settings over the seam, in miniature."""
    scratch = tmp_path_factory.mktemp("type-gate")
    module = scratch / "seam.py"
    module.write_text(REPRODUCTION)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--no-pretty",
            "--no-error-summary",
            "--no-incremental",
            # The project's settings, minus its target list: this checks one
            # throwaway file, not the package.
            "--config-file",
            str(ROOT / "pyproject.toml"),
            # The real `TaskRef`, without type-checking the package that
            # defines it - that is the `types` CI job's job, not this test's.
            "--follow-imports",
            "silent",
            str(module),
        ],
        capture_output=True,
        text=True,
        cwd=scratch,
        env={**os.environ, "MYPYPATH": str(SRC)},
    )
    assert result.returncode in (0, 1), result.stderr or result.stdout
    return result.stdout


def test_ref_keyed_map_read_with_an_int_is_an_error(reproduction_findings: str) -> None:
    """The exact shape that shipped green in #142: `results.get(entry.number)`."""
    flagged = [line for line in reproduction_findings.splitlines() if ":7: error:" in line]
    assert flagged, (
        "a Mapping[TaskRef, ...] read with an int key was not reported - the "
        "settings in [tool.mypy] no longer catch the #142 defect:\n"
        + reproduction_findings
    )


def test_the_same_read_with_a_ref_is_clean(reproduction_findings: str) -> None:
    """The control. An error on both keys would be noise, not a gate."""
    assert not [line for line in reproduction_findings.splitlines() if ":8: error:" in line], (
        "the corrected lookup was also flagged, so the finding above is not "
        "evidence of anything:\n" + reproduction_findings
    )


# --------------------------------------------------------------------------
# One and a half: a pull request's number is not an issue's
# --------------------------------------------------------------------------


#: The #185 defect and its control, in the records that carry both numberings.
#: Each construction is tagged, and the tags - not hard-coded line numbers - are
#: what the assertions below look up, so inserting a case cannot silently point
#: an assertion at its neighbour.
#:
#: Both directions are covered, and the comment has to say so because they are
#: not symmetrical to a reader. Most `WRONG` lines are sourced the way the bug
#: was - the pull request's number read into the field that holds the issue's -
#: but the `Mergeability` pair is the reverse, an issue's `int` into a `PullRef`
#: field. A newtype that rejected only one direction would let the other swap
#: through unremarked. Every `RIGHT` line is the same call sourced correctly: a
#: checker that flags both is worth nothing, which is why the pairs are here
#: rather than the errors alone.
MIS_SOURCING = """\
from swarm.orchestrator.checks import Merge, PullState
from swarm.orchestrator.mergeability import Decision, Mergeability


def build(issue: int, pull: PullState, facts: Mergeability) -> None:
    Merge(number=pull.number, pull=pull.number, branch=pull.branch)  # WRONG
    Merge(number=issue, pull=pull.number, branch=pull.branch)  # RIGHT

    Decision(number=facts.number, pull=facts.number, verdict="fresh")  # WRONG
    Decision(number=issue, pull=facts.number, verdict="fresh")  # RIGHT

    Mergeability(number=issue)  # WRONG
    Mergeability(number=pull.number)  # RIGHT
"""


def _tagged(tag: str) -> list[int]:
    """The 1-based line numbers of every construction carrying `tag`."""
    return [
        number
        for number, line in enumerate(MIS_SOURCING.splitlines(), start=1)
        if line.rstrip().endswith(f"# {tag}")
    ]


@pytest.fixture(scope="module")
def mis_sourcing_findings(tmp_path_factory: pytest.TempPathFactory) -> str:
    scratch = tmp_path_factory.mktemp("two-numberings")
    module = scratch / "numberings.py"
    module.write_text(MIS_SOURCING)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--no-pretty",
            "--no-error-summary",
            "--no-incremental",
            "--config-file",
            str(ROOT / "pyproject.toml"),
            "--follow-imports",
            "silent",
            str(module),
        ],
        capture_output=True,
        text=True,
        cwd=scratch,
        env={**os.environ, "MYPYPATH": str(SRC)},
    )
    assert result.returncode in (0, 1), result.stderr or result.stdout
    return result.stdout


def _lines_with_errors(findings: str, *, naming: str | None = None) -> set[int]:
    """Lines mypy reported an error on; optionally only errors naming a type.

    `naming` exists because "an error on this line" is not what the assertion
    below means. Rename `Merge.number` and mypy reports `call-arg: unexpected
    keyword argument` on every tagged line - the test would stay green while
    proving nothing at all about the two numberings, which is the one thing it
    is here to prove. Requiring the message to name the type makes the finding
    evidence rather than a coincidence of line numbers.

    The control passes no `naming`, deliberately: a correctly-sourced
    construction must produce no error of *any* kind, and narrowing that would
    weaken it.
    """
    return {
        int(match.group(1))
        for match in re.finditer(r"^[^:]+:(\d+): error:(.*)$", findings, flags=re.MULTILINE)
        if naming is None or naming in match.group(2)
    }


def test_a_pull_requests_number_in_the_issues_place_is_an_error(
    mis_sourcing_findings: str,
) -> None:
    """The shape that shipped green before #185, on all three records.

    Not "a guard raises later": the construction itself is rejected, which is
    the difference between a bug that is *caught* and one that cannot be
    written. #184's `UnresolvedJoin` still catches the version of this a type
    cannot - two records that genuinely disagree about which task a merge is
    for, because a human fetched the wrong number - and that is a different
    failure with a different fix.
    """
    wrong = _tagged("WRONG")
    assert wrong, "the reproduction lost its tags"
    flagged = _lines_with_errors(mis_sourcing_findings, naming="PullRef")
    missed = [number for number in wrong if number not in flagged]
    assert not missed, (
        "a pull request's number was accepted where an issue's belongs, on line(s) "
        f"{missed} of the reproduction - the two numberings are interchangeable "
        "to the checker again:\n" + (mis_sourcing_findings or "(mypy reported nothing)")
    )


def test_the_same_records_built_correctly_are_clean(mis_sourcing_findings: str) -> None:
    """The control. Errors on both spellings would make the finding above noise."""
    flagged = _lines_with_errors(mis_sourcing_findings)
    wrongly_flagged = [number for number in _tagged("RIGHT") if number in flagged]
    assert not wrongly_flagged, (
        f"the correctly-sourced construction was also flagged, on line(s) "
        f"{wrongly_flagged}, so the finding above is not evidence of anything:\n"
        + mis_sourcing_findings
    )


# --------------------------------------------------------------------------
# Two: the exclusions stay countable
# --------------------------------------------------------------------------


def test_the_target_is_a_package_not_a_list_of_files(mypy_config: dict) -> None:
    """A new module is checked because it exists, not because it was listed."""
    assert mypy_config.get("packages") == ["swarm"]
    assert "files" not in mypy_config, (
        "a file list is a roster somebody forgets to update, and the module "
        "they forget is the new one nobody has read yet"
    )


def test_nothing_switches_checking_off_wholesale(mypy_config: dict) -> None:
    for scope in [mypy_config, *overrides(mypy_config)]:
        assert not scope.get("ignore_errors"), modules_of(scope)
        assert scope.get("follow_imports", "normal") == "normal", modules_of(scope)
    assert mypy_config.get("ignore_missing_imports") is False, (
        "an untyped dependency should be named, not waved through globally"
    )


def test_relaxations_are_per_module_never_per_package(mypy_config: dict) -> None:
    """`disable_error_code` on a wildcard would relax modules not yet written."""
    for override in overrides(mypy_config):
        if "disable_error_code" not in override:
            continue
        for module in modules_of(override):
            assert "*" not in module, module


def test_the_silent_modules_disable_nothing(mypy_config: dict) -> None:
    """`orchestrator/`, `github/`, `nodes/`, `taskref` keep every error code.

    Their backlog is excluded line by line instead, so a *second* error of the
    same code in the same module still fails the build. This is the assertion
    that stops `arg-type` being switched off in `reconcile.py` - the module the
    #142 defect ran through - to clear two legacy findings.
    """
    for override in overrides(mypy_config):
        if "disable_error_code" not in override:
            continue
        for module in modules_of(override):
            assert not any(
                module == package or module.startswith(package + ".") for package in PRIORITY
            ), f"{module} may not relax an error code for the whole module"


def test_strict_equality_is_on(mypy_config: dict) -> None:
    """It is what reports `entry.number in wanted` once `wanted` is ref-keyed.

    The consumer half of #170's defect. No argument check sees it, because
    `in` accepts `object`.
    """
    assert mypy_config.get("strict_equality") is True


def test_every_suppression_in_src_names_its_code() -> None:
    """A bare `# type: ignore` is a blanket relaxation wearing a comment."""
    bare = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in sorted(SRC.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if re.search(r"#\s*type:\s*ignore(?!\[)", line)
    ]
    assert not bare, bare


def test_the_ignore_without_code_check_is_enabled(mypy_config: dict) -> None:
    """Which is what makes the assertion above true of new code as well."""
    assert "ignore-without-code" in mypy_config.get("enable_error_code", [])
