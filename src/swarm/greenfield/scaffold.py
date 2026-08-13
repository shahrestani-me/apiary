"""Fill a fresh repository with the smallest project that can be verified.

`provision` (#25) creates a repository with a README, a LICENSE and a CI
workflow, and nothing to run. Its verify command is `test -f README.md` - a
placeholder chosen because it is the only thing that passes in an empty repo,
since `python -m pytest -q` exits 5 when it collects no tests and would open
every generated repository red. This module replaces that placeholder with a
real one:

    python -m swarm.greenfield.scaffold "a CLI that converts markdown tables to CSV" \\
        --into ./my-project

The output is an entry point, a dependency manifest, one real passing test and
the command that runs it. The planner then inherits that command as the default
`## Verify` for every issue it writes (`docs/issue-contract.md` §1.3), which is
the whole point: an empty repository has no test command, therefore no quality
gate, therefore no way to judge the first worker's output.

Four decisions here are load-bearing.

**The verify command must cost nothing to run.** It is not run once - it runs in
every worker container, on every attempt, and again in CI on every push. The
worker image deliberately bakes in no toolchain (#14, `Dockerfile.worker`), so
anything the command needs it has to install first, and a scaffold whose tests
begin with a dependency install adds that install to the price of every task in
the repository forever. So the command is `python3 -m unittest discover -q`:
stdlib test runner, stdlib assertions, no third-party package, no install step,
no network. Note `python3` and not `python` - the generated CI workflow has no
`setup-python` step, and the interpreter that is guaranteed to be on PATH both
on a GitHub runner and in `python:3.12-slim` is spelled `python3`.

**Only stacks the worker image can already run are supported, and today that is
Python.** The obvious generosity - detect "a React dashboard", emit a
`package.json`, set verify to `npm ci && npm test` - buys a scaffold that
violates the rule above on its first run. So `choose_stack` recognises a prompt
that names another stack and **refuses**, naming what is missing. Refusing is
the honest failure: quietly substituting a language nobody asked for produces a
repository the requester did not want, and the swarm then plans a whole backlog
against it. The `STACKS` registry is the seam for a second entry, on the day the
worker image grows a toolchain to justify one.

**Flat layout, not `src/`.** apiary itself uses `src/swarm`, which is the better
layout for a package that is installed before it is tested. This one is never
installed - the verify command runs `python3 -m unittest` straight from a fresh
clone, and `python3 -m` puts the working directory on `sys.path`, so a package
at the repository root imports and a package under `src/` does not. A scaffold
whose tests only pass after `pip install -e .` is the four-minute install again,
wearing a smaller coat.

**`tests/` ships an `__init__.py`.** Unittest discovery cannot enter a directory
that is not an importable package: it aborts with `Start directory is not
importable`, which reads like a broken test suite rather than a missing file.
pytest's rootdir magic hides this, and the scaffold cannot rely on pytest.

Nothing in this module touches the network. It writes files into a directory,
and `ScaffoldedPlan` hands its files to `provision` for the initial commit -
which is what makes the scaffold's test suite pass in CI **before any worker has
run**, rather than after some later commit adds it.
"""

from __future__ import annotations

import argparse
import keyword
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .provision import ProvisionPlan, slugify

# The verify command written into every generated Python repository. See the
# module docstring: stdlib only, no install, and `python3` rather than `python`
# because that is the spelling guaranteed on a bare GitHub runner.
PYTHON_VERIFY = "python3 -m unittest discover -q"

# What a generated project's version starts at. 0.0.0 rather than 0.1.0 on
# purpose: nothing has been released, and the first worker to ship something
# real is the one that gets to choose the first number.
INITIAL_VERSION = "0.0.0"

# Module names the generated package must not take. Shadowing a stdlib module
# from the repository root is a trap that surfaces far from its cause - the
# working directory is on `sys.path` during verify, so a package called `csv`
# is imported instead of the real one by every test in the repo, and by parts
# of unittest itself. `tests` is reserved for the same reason at closer range.
_RESERVED_MODULES = frozenset(sys.stdlib_module_names) | {"tests", "test"}


class ScaffoldError(ValueError):
    """The scaffold could not be generated, or could not be written."""


class UnsupportedStack(ScaffoldError):
    """The prompt asks for a stack the worker container cannot run.

    A distinct type because the caller's response differs: this is not a bug to
    retry, it is a request the swarm has to decline until the worker image
    carries that toolchain.
    """


# --------------------------------------------------------------------------
# Stacks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Stack:
    """One generatable project shape: what it emits and how it is verified.

    `verify_command` is the contract with everything downstream - CI runs it,
    the worker runs it, and the planner copies it into each issue's
    `## Verify`. It must be a single line whose exit code is the whole verdict
    (`docs/issue-contract.md` §1.3).
    """

    id: str
    verify_command: str
    build: Callable[["Scaffold"], dict[str, str]]


def _python_files(scaffold: "Scaffold") -> dict[str, str]:
    """The generated Python project, path -> text.

    Six files. Every one of them is load-bearing: remove the manifest and the
    project has no declared dependencies, remove `tests/__init__.py` and
    discovery aborts, remove `.gitignore` and the first worker commits the
    `__pycache__` its own verify run just created.
    """
    package = scaffold.package
    return {
        f"{package}/__init__.py": _package_init(scaffold),
        f"{package}/__main__.py": _package_main(scaffold),
        "tests/__init__.py": _tests_init(),
        f"tests/test_{package}.py": _test_module(scaffold),
        "pyproject.toml": _manifest(scaffold),
        ".gitignore": _gitignore(),
    }


PYTHON = Stack(id="python", verify_command=PYTHON_VERIFY, build=_python_files)

# The registry, and the seam. A second entry means a second toolchain in the
# worker image; see the module docstring before adding one.
STACKS: dict[str, Stack] = {PYTHON.id: PYTHON}
DEFAULT_STACK = PYTHON

# Stack names that are unmistakable in prose: nobody writes "typescript" or
# "cargo" by accident. Matched as whole words, case-insensitively.
_FOREIGN_WORDS = {
    "node": "Node.js",
    "nodejs": "Node.js",
    "npm": "Node.js",
    "pnpm": "Node.js",
    "yarn": "Node.js",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "react": "React",
    "svelte": "Svelte",
    "deno": "Deno",
    "bun": "Bun",
    "golang": "Go",
    "rust": "Rust",
    "cargo": "Rust",
    "java": "Java",
    "kotlin": "Kotlin",
    "ruby": "Ruby",
    "rails": "Ruby on Rails",
    "php": "PHP",
    "swift": "Swift",
    "elixir": "Elixir",
    "haskell": "Haskell",
    "scala": "Scala",
    "clojure": "Clojure",
    "dotnet": ".NET",
}

# Names that are also ordinary English, or ordinary punctuation. "a tool to go
# through logs" is not a request for Go, and refusing it would be worse than
# useless - so these count only when something says they name a language.
_FOREIGN_PHRASES = {
    "go": "Go",
    "c": "C",
    "c++": "C++",
    "c#": "C#",
    ".net": ".NET",
}

_WORD_RE = re.compile(r"[a-z0-9+#.]+")
_QUALIFIER_RE = re.compile(
    r"\b(?:in|using|with|written\s+in)\s+([a-z+#.]+)(?=$|[\s,.;:!?])",
    re.IGNORECASE,
)


def choose_stack(prompt: str) -> Stack:
    """Pick the stack to generate, or refuse to guess.

    Python is the default rather than a detection result: it is the only stack
    the worker image can run without an install, so a prompt that names no
    stack gets the one that works. A prompt that names a *different* one is a
    refusal, not a fallback - see the module docstring.
    """
    named = _foreign_stack(prompt)
    if named is not None:
        raise UnsupportedStack(
            f"the prompt asks for {named}, which the worker image cannot run: it bakes in "
            f"no toolchain on purpose (#14), so a {named} test suite would have to install "
            "one on every task. Generate this project outside the swarm, or add the "
            "toolchain to the worker image and register the stack in STACKS."
        )
    return DEFAULT_STACK


def _foreign_stack(prompt: str) -> str | None:
    """The stack this prompt names, if it names one that is not Python."""
    lowered = prompt.casefold()
    for word in _WORD_RE.findall(lowered):
        if word in _FOREIGN_WORDS:
            return _FOREIGN_WORDS[word]
    for candidate in _QUALIFIER_RE.findall(lowered):
        stripped = candidate.rstrip(".,;:")
        if stripped in _FOREIGN_PHRASES:
            return _FOREIGN_PHRASES[stripped]
        if stripped in _FOREIGN_WORDS:
            return _FOREIGN_WORDS[stripped]
    return None


# --------------------------------------------------------------------------
# The scaffold
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scaffold:
    """A skeleton decided in full before a single byte is written.

    Same split as `ProvisionPlan`: deciding is separate from acting, so the
    entire decision surface - file set, contents, verify command - is testable
    without a filesystem, and `write` has nothing left to choose.
    """

    name: str
    package: str
    prompt: str
    stack: Stack = DEFAULT_STACK

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ScaffoldError("a scaffold needs the prompt it is generated from")
        if not self.package.isidentifier() or keyword.iskeyword(self.package):
            raise ScaffoldError(f"{self.package!r} is not an importable module name")

    @property
    def verify_command(self) -> str:
        return self.stack.verify_command

    def files(self) -> dict[str, str]:
        """The generated skeleton, path -> text."""
        return self.stack.build(self)

    def write(self, directory: Path | str) -> tuple[Path, ...]:
        """Write the skeleton under `directory`, or write nothing at all.

        Every target is checked before the first one is created. A scaffold
        that stops halfway leaves a project that neither imports nor verifies,
        and the reader has to work out which half is theirs.
        """
        root = Path(directory)
        files = self.files()
        existing = sorted(path for path in files if (root / path).exists())
        if existing:
            raise ScaffoldError(
                f"refusing to overwrite {len(existing)} existing file(s) in {root}: "
                f"{', '.join(existing)}"
            )

        written = []
        for path, content in sorted(files.items()):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(target)
        return tuple(written)


def scaffold_for(
    prompt: str,
    *,
    name: str | None = None,
    stack: Stack | None = None,
) -> Scaffold:
    """Derive the whole skeleton from the prompt, choosing a stack if none is given."""
    derived = name or slugify(prompt)
    if not derived:
        raise ScaffoldError(
            f"could not derive a project name from {prompt!r}; pass one explicitly"
        )
    return Scaffold(
        name=derived,
        package=module_name(derived),
        prompt=prompt.strip(),
        stack=stack or choose_stack(prompt),
    )


def module_name(name: str) -> str:
    """Turn a repository name into an importable module name.

    `-` and `.` are legal in a repository name and illegal in an identifier, a
    leading digit is legal in neither position of an import, and a name that
    collides with a stdlib module or with `tests` gets `_app` appended rather
    than shadowing it - see `_RESERVED_MODULES`.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    if not cleaned:
        raise ScaffoldError(f"could not derive a module name from {name!r}")
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    if cleaned in _RESERVED_MODULES or keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_app"
    return cleaned


# --------------------------------------------------------------------------
# Generated file contents
# --------------------------------------------------------------------------
#
# The prompt is free prose from a human and reaches Python source, a docstring
# and a TOML value. It is never interpolated raw: `repr()` produces a valid
# Python literal for any string, `_toml_string` does the same for TOML. A
# prompt containing a quote, a backslash or a triple quote is not exotic - it
# is Tuesday - and a scaffold that does not import is worse than no scaffold,
# because the failure looks like the worker's fault.


def _package_init(scaffold: "Scaffold") -> str:
    return f'''\
"""{scaffold.package}: generated by apiary, and not yet a project.

`PROMPT` below is what was asked for. Everything in this file is a placeholder
whose only job is to give the test suite something real to assert on, so the
first worker has a gate to verify against. Replacing it is the point.
"""

NAME = {scaffold.name!r}
PROMPT = {scaffold.prompt!r}
__version__ = {INITIAL_VERSION!r}


def main() -> int:
    """Print what this project is meant to become. Returns a process exit code."""
    print(f"{{NAME}} {{__version__}}: {{PROMPT}}")
    return 0
'''


def _package_main(scaffold: "Scaffold") -> str:
    return f'''\
"""Entry point, so `python3 -m {scaffold.package}` runs the same code the tests do.

`pyproject.toml` points its console script at the same `main`; one behaviour
with two front doors, rather than two that can drift.
"""

from . import main

if __name__ == "__main__":
    raise SystemExit(main())
'''


def _tests_init() -> str:
    return f'''\
"""Makes `tests` a package, which unittest discovery requires.

Without this file `{PYTHON_VERIFY}` aborts with "Start directory is
not importable" - a message that reads like a broken suite rather than a
missing file. pytest would not need it; the verify command cannot assume pytest.
"""
'''


def _test_module(scaffold: "Scaffold") -> str:
    package = scaffold.package
    return f'''\
"""The first quality gate this repository has.

One test, and a real one: it runs the entry point and asserts on what came out,
so it fails if the entry point is broken. That matters more than its size - the
swarm judges every task by a command whose exit code is the whole verdict, and
a suite that passes without executing anything would report a green gate on a
repository where nothing works.

Stdlib `unittest`, not pytest, deliberately: this suite runs inside a worker
container that has no toolchain installed, so a third-party runner would have
to be fetched before the first assertion (see the scaffold's README section on
the verify command).
"""

import io
import unittest
from contextlib import redirect_stdout

import {package}


class EntryPointTests(unittest.TestCase):
    def test_running_it_succeeds_and_says_what_it_is(self):
        captured = io.StringIO()

        with redirect_stdout(captured):
            code = {package}.main()

        self.assertEqual(code, 0)
        self.assertIn({package}.__version__, captured.getvalue())
        self.assertIn({package}.NAME, captured.getvalue())


if __name__ == "__main__":
    unittest.main()
'''


def _manifest(scaffold: "Scaffold") -> str:
    return f"""\
[project]
name = {_toml_string(scaffold.name)}
version = {_toml_string(INITIAL_VERSION)}
description = {_toml_string(_one_line(scaffold.prompt))}
requires-python = ">=3.11"
# Empty, and worth keeping empty for as long as possible. The verify command
# runs from a fresh clone with nothing installed, in a container that carries
# no toolchain; the first dependency added here is also the first install added
# to the price of every task in this repository.
dependencies = []

[project.scripts]
{scaffold.name} = "{scaffold.package}:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["{scaffold.package}"]
"""


def _gitignore() -> str:
    return """\
# Running the verify command writes __pycache__ into the working tree, so a
# worker that stages everything commits it unless this file exists first.
__pycache__/
*.py[cod]

.venv/
build/
dist/
*.egg-info/
"""


def _one_line(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= 200 else collapsed[:197].rstrip() + "..."


def _toml_string(text: str) -> str:
    """A TOML basic string. Same job `repr` does for the Python literals above."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return f'"{escaped}"'


# --------------------------------------------------------------------------
# Composition with provisioning
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaffoldedPlan(ProvisionPlan):
    """A `ProvisionPlan` whose initial commit already contains the scaffold.

    The acceptance criterion is that the generated test suite passes in CI
    *before any worker has run*. Adding the scaffold in a second commit would
    leave the first commit - the one the required status check first reports on
    - with the placeholder command and no tests. So the scaffold rides in the
    same commit as the README, the LICENSE and the workflow.

    A subclass rather than an edit to `provision.py`, because that file is
    outside this issue's `## Files` contract. It costs nothing: `provision`
    reads the plan through `files()` and `verify_command`, both of which this
    class supplies, and it adds no fields, so `dataclasses.replace` - which
    `provision` uses to correct the default branch name - keeps the subclass.
    """

    @classmethod
    def for_prompt(cls, prompt: str, **overrides) -> "ScaffoldedPlan":
        # Chosen before anything is created, so an unsupported stack refuses at
        # planning time rather than after the repository exists.
        overrides.setdefault("verify_command", choose_stack(prompt).verify_command)
        return super().for_prompt(prompt, **overrides)  # type: ignore[return-value]

    def scaffold(self) -> Scaffold:
        return scaffold_for(self.prompt, name=self.name)

    def files(self) -> dict[str, str]:
        provisioned = super().files()
        generated = self.scaffold().files()
        clash = sorted(set(provisioned) & set(generated))
        if clash:
            # Merging silently would let a stack quietly rewrite the README that
            # documents the branch protection, or the workflow that enforces it.
            raise ScaffoldError(f"scaffold would overwrite provisioned file(s): {', '.join(clash)}")
        return {**provisioned, **generated}


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="swarm-scaffold",
        description="Generate the minimal project a swarm worker can verify against.",
    )
    parser.add_argument("prompt", help="what the project should be")
    parser.add_argument(
        "--into",
        default=".",
        type=Path,
        help="directory to write the scaffold into (default: the working directory)",
    )
    parser.add_argument("--name", default=None, help="project name (default: from the prompt)")
    parser.add_argument(
        "--stack",
        default=None,
        choices=sorted(STACKS),
        help="stack to generate (default: chosen from the prompt)",
    )
    args = parser.parse_args(argv)

    try:
        scaffold = scaffold_for(
            args.prompt,
            name=args.name,
            stack=STACKS[args.stack] if args.stack else None,
        )
        written = scaffold.write(args.into)
    except ScaffoldError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1

    for path in written:
        print(path)
    print()
    print(f"verify with: {scaffold.verify_command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
