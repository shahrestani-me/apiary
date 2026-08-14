"""The first issue of every greenfield plan: write the project itself.

`scaffold.py` emits six f-string templates and the greenfield path contains
**zero model calls** - `choose_stack` refuses 26 named technologies rather than
generating any of them. This module is how a generated project stops being a
template.

## Why this is a phase and not a `Stack.build` implementation

The obvious change is to swap `Stack.build` for a model call. It cannot be
done. `ScaffoldedPlan.files()` runs in the **orchestrator** process, which:

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

from dataclasses import dataclass

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
