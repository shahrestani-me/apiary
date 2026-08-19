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
