"""Unit tests for the greenfield scaffold.

The centre of this file is `test_the_generated_suite_passes_from_a_bare_clone`:
it writes the scaffold into a temporary directory and runs the exact string the
generated CI workflow will run, in a subprocess, with nothing installed. That
is the acceptance criterion stated as a test - a scaffold whose declared verify
command does not actually pass hands every planned issue in the repository a
gate that was already red before anyone touched it.

Running it in a subprocess rather than importing the generated modules is the
whole point of the test. `python3 -m unittest discover` resolves the
interpreter, the working directory, `sys.path` and test discovery itself, and
every one of those is a way for a scaffold to look right in-process and fail in
the container. An assertion about the *contents* of the generated files would
pass for a project that does not import.

Nothing here reaches GitHub: the composition with `provision` is checked
through `ProvisionPlan.files()` and `verify_command`, which need no transport.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from swarm.greenfield.provision import CI_WORKFLOW_PATH, PLACEHOLDER_VERIFY, ProvisionPlan
from swarm.greenfield.scaffold import (
    INITIAL_VERSION,
    PYTHON,
    PYTHON_VERIFY,
    STACKS,
    Scaffold,
    ScaffoldedPlan,
    ScaffoldError,
    UnsupportedStack,
    choose_stack,
    main,
    module_name,
    scaffold_for,
)

OWNER = "shahrestani-me"
PROMPT = "a CLI that converts markdown tables to CSV"
NAME = "a-cli-that-converts-markdown-tables-to-csv"
PACKAGE = "a_cli_that_converts_markdown_tables_to_csv"


def run_verify(directory: Path, command: str = PYTHON_VERIFY) -> subprocess.CompletedProcess:
    """Run a verify command exactly as CI and the worker container will run it.

    `shell=True` from the repository root, nothing installed, and only the exit
    code believed - `docs/issue-contract.md` §1.3, and v1's `verifier.py` before
    it.
    """
    return subprocess.run(command, shell=True, cwd=directory, capture_output=True, text=True)


def written(tmp_path: Path, prompt: str = PROMPT, **kwargs: Any) -> Scaffold:
    scaffold = scaffold_for(prompt, **kwargs)
    scaffold.write(tmp_path)
    return scaffold


# --------------------------------------------------------------------------
# The generated suite actually runs - the acceptance criterion
# --------------------------------------------------------------------------


def test_the_generated_suite_passes_from_a_bare_clone(tmp_path: Path):
    scaffold = written(tmp_path)

    result = run_verify(tmp_path, scaffold.verify_command)

    assert result.returncode == 0, result.stderr
    # Guard against the failure mode the placeholder command exists to avoid:
    # a runner that collects nothing exits non-zero (5) for pytest and unittest
    # alike, but a suite that is discovered and empty would pass silently here
    # if the runner were ever changed.
    assert "Ran 1 test" in result.stderr


def test_the_entry_point_runs_as_a_module(tmp_path: Path):
    # `python3 -m <package>` is the other front door pyproject's console script
    # points at; a broken __main__.py is invisible to the test suite otherwise.
    scaffold = written(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", scaffold.package], cwd=tmp_path, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert PROMPT in result.stdout
    assert INITIAL_VERSION in result.stdout


def test_a_prompt_full_of_quoting_hazards_still_produces_a_project_that_runs(tmp_path: Path):
    # The prompt is free prose from a human and lands in Python source, a
    # docstring and a TOML value. A scaffold that does not import fails as the
    # first worker's problem, minutes later and far from the cause.
    hostile = 'a tool for "quoted" \\ backslashed \'\'\' triple-quoted # text'
    scaffold = written(tmp_path, hostile, name="hazard-tool")

    assert run_verify(tmp_path, scaffold.verify_command).returncode == 0
    assert tomllib.loads((tmp_path / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_verify_run_leaves_nothing_the_first_worker_would_commit(tmp_path: Path):
    # Verifying writes __pycache__ into the working tree, and a worker that
    # stages everything commits it on the first task unless .gitignore is there
    # from the initial commit.
    written(tmp_path)
    run_verify(tmp_path)

    assert list(tmp_path.rglob("__pycache__"))
    assert "__pycache__/" in (tmp_path / ".gitignore").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# The verify command itself
# --------------------------------------------------------------------------


def test_the_verify_command_installs_nothing():
    # The ruling this module exists to make. The command runs in every worker
    # container, on every attempt, and again in CI; an install step in it is an
    # install step on the price of every task in the repository, forever.
    assert not {"pip", "install", "npm", "poetry", "uv", "curl"} & set(PYTHON_VERIFY.split())


def test_the_verify_command_needs_only_the_stdlib():
    # No third-party test runner: the worker image installs apiary without its
    # dev extras, so pytest is not there to be run.
    assert PYTHON_VERIFY.split()[:3] == ["python3", "-m", "unittest"]
    assert "unittest" in sys.stdlib_module_names


def test_the_verify_command_says_python3_not_python():
    # The generated CI workflow has no setup-python step. `python3` is on PATH
    # on a bare GitHub runner and in python:3.12-slim; plain `python` is not
    # guaranteed on either.
    assert PYTHON_VERIFY.startswith("python3 ")


def test_the_verify_command_is_a_single_line_a_plan_will_accept():
    # docs/issue-contract.md §1.3, enforced by ProvisionPlan: one command, and
    # only its exit code is believed.
    ProvisionPlan.for_prompt(PROMPT, owner=OWNER, verify_command=PYTHON_VERIFY)
    assert len(PYTHON_VERIFY.splitlines()) == 1


def test_the_scaffold_replaces_the_placeholder_command():
    assert PYTHON_VERIFY != PLACEHOLDER_VERIFY


# --------------------------------------------------------------------------
# Choosing a stack
# --------------------------------------------------------------------------


def test_a_prompt_that_names_no_stack_gets_the_one_that_runs_for_free():
    assert choose_stack(PROMPT) is PYTHON
    assert choose_stack("a service that summarises log files") is PYTHON


def test_python_is_the_only_registered_stack():
    # Not an oversight: a second entry means a second toolchain in the worker
    # image, which #14 rules out on purpose.
    assert list(STACKS) == ["python"]


@pytest.mark.parametrize(
    "prompt, named",
    [
        ("a React dashboard for build metrics", "React"),
        ("a TypeScript library for parsing dates", "TypeScript"),
        ("a small tool written in Go", "Go"),
        ("a crate in Rust for diffing JSON", "Rust"),
        ("an API in C#", "C#"),
    ],
)
def test_a_prompt_naming_another_stack_is_refused_not_quietly_substituted(prompt: str, named: str):
    # Emitting a Python project for a prompt that asked for Go produces a
    # repository nobody wanted, and the swarm then plans a whole backlog
    # against it. Refusing says what is missing instead.
    with pytest.raises(UnsupportedStack, match=re.escape(named)):
        choose_stack(prompt)


def test_the_refusal_says_what_would_have_to_change():
    with pytest.raises(UnsupportedStack, match="worker image"):
        choose_stack("a CLI in Rust")


@pytest.mark.parametrize(
    "prompt",
    [
        "a tool to go through log files",
        "a service that lets users go back to an earlier version",
        "a c compiler explainer for beginners",
    ],
)
def test_ordinary_english_is_not_mistaken_for_a_stack(prompt: str):
    # "go" and "c" are words before they are languages. Refusing these would be
    # worse than useless.
    assert choose_stack(prompt) is PYTHON


def test_an_explicit_stack_overrides_the_prompt():
    # The refusal is about guessing, not about the caller's right to decide.
    assert scaffold_for("a dashboard in TypeScript", name="dash", stack=PYTHON).stack is PYTHON


# --------------------------------------------------------------------------
# The generated skeleton
# --------------------------------------------------------------------------


def test_the_skeleton_is_an_entry_point_a_manifest_and_one_real_test():
    assert sorted(scaffold_for(PROMPT).files()) == [
        ".gitignore",
        f"{PACKAGE}/__init__.py",
        f"{PACKAGE}/__main__.py",
        "pyproject.toml",
        "tests/__init__.py",
        f"tests/test_{PACKAGE}.py",
    ]


def test_the_package_sits_at_the_repository_root_not_under_src():
    # `python3 -m unittest` puts the working directory on sys.path and nothing
    # else. A src/ layout is unimportable until something installs it, which is
    # the install this scaffold exists to avoid.
    assert not [path for path in scaffold_for(PROMPT).files() if path.startswith("src/")]


def test_the_tests_directory_is_a_package():
    # Unittest discovery cannot enter a directory that is not importable; it
    # aborts with "Start directory is not importable", which reads like a
    # broken suite rather than a missing file.
    assert "tests/__init__.py" in scaffold_for(PROMPT).files()


def test_the_manifest_declares_no_dependencies():
    # The property that makes the zero-install verify command possible, checked
    # as data rather than as a substring.
    manifest = tomllib.loads(scaffold_for(PROMPT).files()["pyproject.toml"])

    assert manifest["project"]["dependencies"] == []
    assert manifest["project"]["name"] == NAME
    assert manifest["project"]["version"] == INITIAL_VERSION
    assert manifest["project"]["scripts"][NAME] == f"{PACKAGE}:main"


def test_the_generated_test_asserts_on_behaviour_rather_than_passing_by_default():
    # A suite that passes without executing anything reports a green gate on a
    # repository where nothing works.
    source = scaffold_for(PROMPT).files()[f"tests/test_{PACKAGE}.py"]

    assert f"import {PACKAGE}" in source
    assert f"{PACKAGE}.main()" in source
    assert "assertEqual(code, 0)" in source


def test_the_prompt_survives_into_the_generated_code():
    files = scaffold_for(PROMPT).files()
    assert PROMPT in files[f"{PACKAGE}/__init__.py"]
    assert PROMPT in files["pyproject.toml"]


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------


def test_the_module_name_is_the_repository_name_made_importable():
    assert module_name("markdown-to-csv") == "markdown_to_csv"
    assert module_name("Widget.Tools") == "widget_tools"


def test_a_name_starting_with_a_digit_is_still_importable():
    assert module_name("2048-solver").isidentifier()


def test_a_name_that_would_shadow_the_stdlib_is_moved_out_of_the_way():
    # The working directory is on sys.path during verify, so a package called
    # `csv` is imported instead of the real one - by the tests, and by parts of
    # unittest itself.
    assert module_name("csv") == "csv_app"
    assert module_name("json") == "json_app"
    assert module_name("tests") == "tests_app"


def test_a_prompt_with_no_letters_asks_for_an_explicit_name():
    with pytest.raises(ScaffoldError, match="could not derive"):
        scaffold_for("!!! ???")


def test_a_scaffold_needs_a_prompt():
    with pytest.raises(ScaffoldError, match="needs the prompt"):
        Scaffold(name="thing", package="thing", prompt="   ")


def test_a_package_name_that_is_not_importable_is_rejected_before_anything_is_written():
    with pytest.raises(ScaffoldError, match="importable module name"):
        Scaffold(name="thing", package="not a module", prompt=PROMPT)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def test_writing_creates_every_file_and_reports_them(tmp_path: Path):
    scaffold = scaffold_for(PROMPT)

    paths = scaffold.write(tmp_path)

    assert {path.relative_to(tmp_path).as_posix() for path in paths} == set(scaffold.files())
    assert all(path.is_file() for path in paths)


def test_writing_over_an_existing_file_writes_nothing_at_all(tmp_path: Path):
    # A half-written scaffold imports for neither the reader nor the runner,
    # and the reader has to work out which half is theirs.
    (tmp_path / "pyproject.toml").write_text("# hand-written\n", encoding="utf-8")

    with pytest.raises(ScaffoldError, match="refusing to overwrite"):
        scaffold_for(PROMPT).write(tmp_path)

    assert list(tmp_path.iterdir()) == [tmp_path / "pyproject.toml"]
    assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == "# hand-written\n"


# --------------------------------------------------------------------------
# Composition with provisioning
# --------------------------------------------------------------------------


def test_the_scaffold_rides_in_the_initial_commit():
    # The acceptance criterion is "before any worker has run". A scaffold added
    # in a second commit leaves the first CI run - the one the required status
    # check reports on - with no tests at all.
    plan = ScaffoldedPlan.for_prompt(PROMPT, owner=OWNER, year=2026)

    files = plan.files()

    assert {"README.md", "LICENSE", CI_WORKFLOW_PATH} <= set(files)
    assert set(scaffold_for(PROMPT).files()) <= set(files)


def test_a_provisioned_repo_verifies_with_the_scaffolds_command(tmp_path: Path):
    # End to end without GitHub: lay out exactly what the initial commit
    # carries and run exactly what the workflow will run.
    plan = ScaffoldedPlan.for_prompt(PROMPT, owner=OWNER, year=2026)
    for path, content in plan.files().items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    assert plan.verify_command == PYTHON_VERIFY
    assert f"run: {PYTHON_VERIFY}" in plan.files()[CI_WORKFLOW_PATH]
    assert PYTHON_VERIFY in plan.files()["README.md"]
    assert run_verify(tmp_path, plan.verify_command).returncode == 0


def test_the_scaffold_never_overwrites_a_provisioned_file():
    # Merging silently would let a stack rewrite the README documenting the
    # branch protection, or the workflow enforcing it.
    class Colliding(ScaffoldedPlan):
        def scaffold(self):
            return Scaffold(name=self.name, package="pkg", prompt=self.prompt, stack=_readme_stack())

    with pytest.raises(ScaffoldError, match="README.md"):
        Colliding.for_prompt(PROMPT, owner=OWNER).files()


def _readme_stack():
    from swarm.greenfield.scaffold import Stack

    return Stack(id="colliding", verify_command="true", build=lambda scaffold: {"README.md": "x"})


def test_correcting_the_default_branch_keeps_the_scaffold():
    # `provision` calls dataclasses.replace on the plan once GitHub reports the
    # real default branch name. A subclass that lost its identity there would
    # commit the scaffold on some accounts and not others.
    from dataclasses import replace

    plan = replace(ScaffoldedPlan.for_prompt(PROMPT, owner=OWNER), default_branch="trunk")

    assert isinstance(plan, ScaffoldedPlan)
    assert f"{PACKAGE}/__init__.py" in plan.files()


def test_an_unsupported_stack_refuses_before_the_repository_exists():
    # Planning is the last moment a refusal is free; afterwards there is a real
    # repository with a URL somebody may already have seen.
    with pytest.raises(UnsupportedStack):
        ScaffoldedPlan.for_prompt("a dashboard in TypeScript", owner=OWNER)


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def test_the_command_writes_the_scaffold_and_prints_the_verify_command(tmp_path: Path, capsys):
    code = main([PROMPT, "--into", str(tmp_path)])

    assert code == 0
    assert (tmp_path / f"{PACKAGE}/__init__.py").is_file()
    assert PYTHON_VERIFY in capsys.readouterr().out
    assert run_verify(tmp_path).returncode == 0


def test_the_command_refuses_rather_than_overwriting(tmp_path: Path, capsys):
    (tmp_path / "pyproject.toml").write_text("# hand-written\n", encoding="utf-8")

    code = main([PROMPT, "--into", str(tmp_path)])

    assert code == 1
    assert "refusing to overwrite" in capsys.readouterr().err


def test_the_command_reports_an_unsupported_stack_instead_of_raising(tmp_path: Path, capsys):
    code = main(["a dashboard in TypeScript", "--into", str(tmp_path)])

    assert code == 1
    assert "TypeScript" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
