"""The first issue of every greenfield plan: write the project itself.

`scaffold.py` emits six f-string templates and the greenfield path contains
**zero model calls** - `choose_stack` refuses 26 named technologies rather than
generating any of them. This module is how a generated project stops being a
template.

## Why this is a phase and not a `Stack.build` implementation

The obvious change is to swap `Stack.build` for a model call. It cannot be
done. A plan's `files()` runs in the **orchestrator** process, which:

- has no checkout and cannot get one, so there is nowhere for generated files
  to be written, verified or committed;
- reaches Docker through a socket proxy with `EXEC=0`, `BUILD=0` and
  `IMAGES=0`, so it cannot obtain one either;
- is required by `docs/architecture-v2.md` to stay boring - arithmetic and
  routing, with model intelligence in the workers.

**`Stack.build` is not the seam. The phase is.** So the bootstrap becomes issue
#1 of the plan and is executed by a worker like every other task: it clones,
asks the model for whole files through the same `WorkerOutput` schema, applies
them through the same guard rail, runs the same gate and opens the same pull
request. Nothing in the worker needs to know it is special.

The repository is still provisioned first, with `PLACEHOLDER_VERIFY`
(`test -f README.md`) green on the initial commit - it has to be, because the
required status check reports on that commit before any worker exists - and the
bootstrap's pull request replaces it.

## What this module actually decides

Two things, and both are decisions rather than generation:

**Which stack the prompt implies.** D5's defaults: something describing a user
interface is React web, and a service, CLI or library is Python. The model
answers, because "a dashboard for warehouse pickers" is a judgement and a word
list is what #87 is removing. `llm=None` reaches the real one, following
`edit.propose_edits`'s seam exactly, so tests need no Ollama.

**What the bootstrap task is allowed to write.** The `## Files` list is capped
and named here rather than left to the model, for the reason #88 measured: the
generated set has to be bounded by the declared file list, not by a context
window. Measured at 500-950 characters per file, so `MAX_BOOTSTRAP_FILES`
leaves comfortable headroom in `worker_num_ctx=16384` - #88 found peak
consumption at roughly half the window, and `done_reason: stop` on 22 of 22
generations.

## What this module does not decide

The gate. #102 owns proposing a verify command and proving it can go red, and
until it lands the bootstrap inherits the repository's command like any other
task. That matters more than it sounds: #88 measured `node --test` exiting **0
on a repository with no tests in it**, so a bootstrap that proposed its own
gate today could grade an empty generation green.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel, Field

from ..github.ledger import DEFAULT_STACK, KNOWN_STACKS
from ..llm import orchestrator_llm, structured
from ..state import PlannedTask

#: The task id the bootstrap always carries. Fixed rather than generated,
#: because a replan has to recognise the bootstrap it already created - the id
#: is identity (`docs/issue-contract.md` §2), and a bootstrap under a fresh id
#: on every replan would open a second one against a repository that already
#: has its project.
BOOTSTRAP_TASK_ID = "bootstrap-the-project"

#: How many files the bootstrap may declare.
#:
#: A cap, not a target. #88 measured this model at 500-950 characters per
#: generated file, with peak context consumption at roughly half of
#: `worker_num_ctx=16384` for a four-file project - so ten leaves real headroom
#: and twenty would not. The number that matters is not the window, though: it
#: is that the generated set is bounded by something a human wrote down.
#:
#: Raising `worker_num_ctx` is not the escape hatch it looks like. `config.py`
#: notes that a second context size spawns a second Ollama runner and doubles
#: resident memory, and #88 measured 32768 producing *worse* results than
#: 16384 rather than better.
MAX_BOOTSTRAP_FILES = 10

#: What each stack's bootstrap is asked for. Paths rather than prose, because
#: they become the `## Files` contract the worker is held to, and `apply_edits`
#: refuses anything outside it.
#:
#: Deliberately small. This is the first commit of a project, not the project:
#: every stack gets a manifest, one source file, one test file and a README,
#: and the swarm adds the rest by doing the work. `provision.py` already
#: commits the README and LICENSE, so neither appears here - a bootstrap that
#: redeclared them would have its edits refused as outside its file set.
BOOTSTRAP_FILES: dict[str, tuple[str, ...]] = {
    "python": (
        "pyproject.toml",
        "src/main.py",
        "tests/__init__.py",
        "tests/test_main.py",
    ),
    "node": (
        "package.json",
        "src/index.js",
        "test/index.test.js",
    ),
    "react": (
        "package.json",
        "index.html",
        "src/main.jsx",
        "src/App.jsx",
        "test/App.test.jsx",
    ),
}

#: The gate each stack falls back to, and what the model is shown as an example.
#:
#: **Every one of these must install nothing.** The command is not run once: it
#: runs in every worker container, on every attempt, and again in CI on every
#: push, so an install step in it is an install step on the price of every task
#: in the repository forever. It is also *impossible* - a worker's only route
#: out is the egress proxy's static allowlist, so an install is denied in under
#: a second (#90's `DENIED_EGRESS_SIGNATURES`). This was the sharpest statement
#: of the cost tension in the old scaffold and it survives here as a per-stack
#: rule.
#:
#: `python3` and not `python`: the generated workflow has a setup step only for
#: stacks that need one, and the interpreter guaranteed on a bare runner and in
#: `python:3.12-slim` is spelled `python3`.
#:
#: **The Node command is compound on purpose, and #88 measured why.**
#: `node --test` alone exits **0 on a repository with no tests in it** - so it
#: would grade an empty or partial generation green, and no flag fixes it. The
#: `test -n` guard is what makes the gate fail when there is nothing to run.
STACK_VERIFY: dict[str, str] = {
    "python": "python3 -m unittest discover -q",
    "node": 'test -n "$(ls test/*.test.js 2>/dev/null)" && node --test',
    # React web needs a transform `node --test` does not have, so its real gate
    # arrives with #106. Until then it inherits the placeholder rather than
    # claiming a command that cannot run.
    "react": "",
}

SYSTEM = """You choose which technology stack a software project should be built in.

Answer with exactly one of: python, node, react.

- react means React on the web: anything with a user interface, a dashboard,
  a page, a form, a visualisation, something a person looks at in a browser.
- node means a JavaScript program with no user interface.
- python means a service, an API, a CLI, a library, a script, a data pipeline -
  anything whose users are other programs or a terminal.

When the prompt does not say and could be either, prefer python.

Return JSON only."""


class StackChoice(BaseModel):
    """The one question this module asks a model.

    A schema rather than free text for the reason every other model call here
    uses one: `structured` passes it to Ollama's `format`, which constrains
    decoding, so an answer outside the three ids is not something to validate -
    it is something the decoder cannot emit.
    """

    stack: str = Field(
        description="one of: python, node, react. react means React on the web, not React Native"
    )
    reason: str = Field(default="", description="one short sentence")


class Oracle:
    """What `choose_stack` calls. `structured(...)` satisfies it.

    A `Protocol` in spirit, spelled as a class only so the docstring has
    somewhere to live: a bootstrap test that needs Ollama running to check that
    a dashboard resolves to React is a test that does not run.
    """

    def invoke(self, messages): ...  # pragma: no cover - shape only


def choose_stack(prompt: str, *, llm: object | None = None) -> str:
    """Which stack this prompt implies. D5's defaults, decided by the model.

    Falls back to `DEFAULT_STACK` rather than raising when the model is
    unreachable or answers with something outside the vocabulary. That is the
    right failure: Python is the stack whose image this host has always had,
    and a greenfield run that refused to start because Ollama hiccuped during a
    one-word classification would be worse than one that generated a Python
    project the operator can re-run with `--stack`.

    The refusal that *does* belong here is #103's, and it is about the host
    rather than the prompt: a stack with no image is refused by
    `StackImages.for_stack` before anything is claimed.
    """
    model = structured(orchestrator_llm(), StackChoice) if llm is None else llm
    try:
        answer = model.invoke([("system", SYSTEM), ("human", prompt)])
    except Exception:  # noqa: BLE001 - local model failures are varied
        return DEFAULT_STACK
    chosen = str(getattr(answer, "stack", "") or "").strip().casefold()
    return chosen if chosen in KNOWN_STACKS else DEFAULT_STACK


@dataclass(frozen=True)
class Bootstrap:
    """The bootstrap task, before it becomes a `PlannedTask`.

    A type rather than a bare function so the two things a caller needs - the
    task itself, and the id every other task must block on - travel together
    and cannot be derived twice with different answers.
    """

    prompt: str
    stack: str
    files: tuple[str, ...]

    @classmethod
    def for_prompt(cls, prompt: str, *, stack: str | None = None, llm: object | None = None) -> Bootstrap:
        """`stack` overrides the model, which is what `swarm run --stack` is."""
        resolved = (stack or "").strip().casefold() or choose_stack(prompt, llm=llm)
        if resolved not in KNOWN_STACKS:
            resolved = DEFAULT_STACK
        files = BOOTSTRAP_FILES.get(resolved, BOOTSTRAP_FILES[DEFAULT_STACK])
        return cls(prompt=prompt.strip(), stack=resolved, files=files[:MAX_BOOTSTRAP_FILES])

    @property
    def task(self) -> PlannedTask:
        """The bootstrap as the planner's own type, so nothing downstream has a
        second code path for it: it is normalised, rendered, round-tripped
        through `parse_contract` and written like any other task."""
        return PlannedTask(
            id=BOOTSTRAP_TASK_ID,
            goal=self.goal,
            files=list(self.files),
            depends_on=[],
            stack=self.stack,
        )

    @property
    def goal(self) -> str:
        """What the worker is asked for.

        The prompt is quoted rather than paraphrased. It is the only statement
        of what the project *is*, and a planner that summarised it would be
        deciding the product in the one place nobody reviews.
        """
        return (
            f"Create the initial {self.stack} project for: {self.prompt} "
            f"Write every file listed, complete and working, with no dependencies "
            f"beyond the language's standard library. The test file must contain at "
            f"least one real assertion about the code in this project - a suite that "
            f"passes because it tests nothing is worse than no suite."
        )


# --------------------------------------------------------------------------
# Falsifying the gate
# --------------------------------------------------------------------------
#
# `planner.py` refuses to let the model choose a verify command, because "a
# command guessed wrong is a gate that was red before any worker touched the
# task". Making the bootstrap model-generated reopens that: the model would
# write both the code and the gate that judges it.
#
# So a proposed command is not trusted, it is *falsified* - run against a
# deliberately broken copy of the tree and required to notice.


#: The gate's own wall clock, separate from `SWARM_VERIFY_TIMEOUT`.
#:
#: Distinct because they bound different things: `verify_timeout_s` is how long
#: a *task's* gate may take on a real repository, and this is how long a one-off
#: probe may take on a tree with four files in it. Sharing the number would mean
#: a repository that legitimately needs a five-minute suite also gets five
#: minutes per falsification run, twice, at provision time.
FALSIFY_TIMEOUT_S = 120

#: Commands refused before a container is created.
#:
#: Belt and braces: `true` and `exit 0` would be rejected by the decision table
#: anyway, since they pass on the mutated tree. They are named so the message a
#: human reads says *tautological* rather than "passed the mutation", and
#: because this is the cheapest possible check.
TAUTOLOGIES = frozenset({"true", ":", "exit 0", "/bin/true"})


class ProposedGate(BaseModel):
    """The verify command the bootstrap proposes for the repository it wrote.

    Separate from `WorkerOutput` on purpose: the files and the gate are two
    answers with two different fates. The files are applied if they land inside
    the declared set; the command is applied only if it survives falsification,
    and a rejected command must not take the files down with it.
    """

    command: str = Field(
        description=(
            "one shell command whose exit code is the whole verdict. It MUST fail "
            "when the project's source files are empty - a command that still passes "
            "on an empty project is not testing anything"
        )
    )
    reason: str = Field(default="", description="one sentence: what it runs")


@dataclass(frozen=True)
class Verdict:
    """What falsification decided, and why in words a human can act on."""

    command: str
    accepted: bool
    reason: str

    def __str__(self) -> str:
        state = "accepted" if self.accepted else "REFUSED"
        return f"gate {state}: {self.command!r} - {self.reason}"


def empty_declared(tree: Path, files: Sequence[str], into: Path) -> Path:
    """The mutation: copy the tree, then truncate every declared file.

    **Orchestrator-chosen and deterministic, never model-chosen.** A mutation
    the model picked would put the same model on both sides of the test, which
    is the failure this whole function exists to prevent.

    **Emptying, not deleting**, and the difference is load-bearing.
    `PLACEHOLDER_VERIFY` is `test -f README.md`: a falsifier that *deleted*
    files would see it go red and admit it as a real gate. Emptying leaves the
    file there, so `test -f` passes on the mutated tree too and the command is
    correctly refused. It is the permanent regression case this ticket names,
    and it stays caught only because of this choice.

    Only the *declared* files are touched. Everything provisioning committed -
    README, LICENSE, the workflow - is left alone, for the same reason.
    """
    shutil.copytree(tree, into, dirs_exist_ok=True)
    for relative in files:
        target = into / relative
        if target.is_file():
            target.write_text("", encoding="utf-8")
    return into


def falsify(
    command: str,
    tree: Path,
    files: Sequence[str],
    *,
    mutate: Callable[[Path, Sequence[str], Path], Path] = empty_declared,
    run: Callable[[str, Path], int] | None = None,
    workspace: Path | None = None,
) -> Verdict:
    """Accept a proposed gate only if it passes clean code and fails broken code.

    Two runs, and the decision is the pair:

    ==========  ==========  ===============================================
    clean       mutated     verdict
    ==========  ==========  ===============================================
    non-zero    -           refused: red before a worker touched it
    zero        zero        refused: tautological, proves nothing
    zero        non-zero    accepted
    ==========  ==========  ===============================================

    **Both dependencies are injectable and both have real defaults.** That is
    not a testing convenience, it is the safety boundary: `run` decides *where*
    model-proposed shell executes, and the only acceptable answer is a
    container. Provisioning runs in the orchestrator process, which holds
    `APIARY_PROVISION_TOKEN` - `administration` and `workflows` - and a
    `DOCKER_HOST` pointing at a proxy with `ALLOW_START=1`.
    `assert_unprivileged` only inspects argv that `ContainerManager.spawn`
    built, so a one-liner calling `docker` directly bypasses it entirely. One
    line of model-proposed shell there is host root *plus* the ability to
    rewrite the CI it is being falsified against. On the `.venv` path it is
    model-written shell on the developer's Mac.

    ## What this does not buy, stated rather than discovered

    It is a **shape** check with a known false-negative rate.

    - Emptying the declared files proves the command **reads** the code. It does
      not prove it **tests** it.
    - It cannot detect a **narrow** gate. `npm test` whose suite asserts
      `true === true`, or a command covering one file of nine, passes clean and
      fails emptied, and is admitted. Narrowness is exactly what makes CI green
      while the application is broken.
    - **Measured, and specific:** `node --test` exits 0 on a tree whose source
      and test files are empty *but whose `package.json` is intact*. It is
      caught today only because `BOOTSTRAP_FILES["node"]` declares the manifest,
      so emptying that breaks module resolution and the run goes red. That is a
      property of the file list, not of this function - a stack whose manifest
      were not declared would admit a vacuous gate.
    """
    stripped = command.strip()
    if not stripped or stripped.casefold() in TAUTOLOGIES:
        return Verdict(stripped, False, "tautological: it cannot fail, so it gates nothing")

    execute = run if run is not None else _container_run
    clean = execute(stripped, tree)
    if clean != 0:
        return Verdict(
            stripped,
            False,
            f"red on the code it was written for (exit {clean}); this gate would block "
            "every pull request before a worker touched the task",
        )

    root = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="apiary-falsify-"))
    broken = execute(stripped, mutate(tree, files, root / "mutated"))
    if broken == 0:
        return Verdict(
            stripped,
            False,
            "still passes with every declared file emptied, so it is not reading the "
            "code at all - see #88 for how a gate can be green on an empty project",
        )
    return Verdict(stripped, True, f"passes clean and fails emptied (exit {broken})")


def _container_run(command: str, tree: Path) -> int:
    """Run one proposed command in a container, and nothing else.

    Deliberately the only thing in this module that executes anything, and
    deliberately not a `subprocess` call: `test_nothing_under_greenfield_runs_a_shell`
    asserts there is no `subprocess` import and no `shell=True` anywhere under
    `src/swarm/greenfield/`, because this is the module a reader would most
    reasonably add one to.
    """
    from ..containers.manager import ContainerManager, StackImages
    from ..run import Run

    manager = ContainerManager(
        run=Run.start("apiary/falsify", "falsify a proposed gate"),
        image=StackImages().for_stack(DEFAULT_STACK),
        env={},
        timeout_s=FALSIFY_TIMEOUT_S,
        extra_flags=["--network", "none", "--volume", f"{Path(tree).resolve()}:/w:ro"],
    )
    handle = manager.spawn(
        0, "", entrypoint="/bin/sh", command=["-c", f"cd /w && {command}"]
    )
    try:
        return manager.wait(handle, timeout_s=FALSIFY_TIMEOUT_S)
    finally:
        manager.dispose(handle)


def choose_gate(
    proposed: str,
    tree: Path,
    files: Sequence[str],
    *,
    operator: str | None = None,
    fallback: str = "",
    **kwargs,
) -> Verdict:
    """Which command becomes the repository's gate.

    **An explicit `--verify` skips falsification entirely.** It is the escape
    hatch, and an escape hatch that can be refused is not one: a false rejection
    there has no recovery path, because the operator has already told the system
    the answer and the system would be arguing with them about their own
    repository. `cli._target` already treats `--verify` as authoritative over
    the scaffold's command for the same reason.

    A proposed command that fails falsification does **not** fail the run. The
    repository keeps `fallback` - the placeholder the initial commit was
    provisioned with - and the reason is printed. Aborting would destroy a
    generated project over a gate, and the project is the expensive part.
    """
    if operator:
        return Verdict(operator.strip(), True, "chosen by the operator with --verify; not falsified")
    verdict = falsify(proposed, tree, files, **kwargs)
    if verdict.accepted:
        return verdict
    return Verdict(fallback, False, f"keeping {fallback!r}: {verdict.reason}")
