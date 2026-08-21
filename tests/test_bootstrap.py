"""Tests for the bootstrap phase - the first issue of every greenfield plan.

Two properties carry this file.

**The bootstrap is a task, not a special case.** Everything downstream - the
contract parser, the readiness pass, the dispatcher, the worker - must be able
to treat it as an ordinary issue, because the whole point of making it a
*phase* rather than a `Stack.build` implementation is that no other module has
to learn about it. So the assertions below are mostly "it round-trips through
`parse_contract`" and "it is ordered and blocked like anything else".

**No model, anywhere.** `choose_stack` takes the same `llm=None` injection seam
as `edit.propose_edits`, and every test here passes a double. A bootstrap test
that needed Ollama running to check that a dashboard resolves to React is a
test that does not run - and it is not hypothetical: leaving the real call live
turned `test_cli_run.py` into a 118-second suite before the seam was used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest

from swarm.github.ledger import DEFAULT_STACK, KNOWN_STACKS, parse_contract
from swarm.config import SETTINGS
from swarm.greenfield.provision import CI_SETUP, PLACEHOLDER_VERIFY
from swarm.greenfield.stacks import STACK_RULE, REACT_TOOLCHAIN, package_names
from swarm.worker.edit import system_for
from swarm.greenfield.bootstrap import (
    BOOTSTRAP_FILES,
    STACK_VERIFY,
    FALSIFY_TIMEOUT_S,
    ProposedGate,
    Verdict,
    choose_gate,
    empty_declared,
    falsify,
    BOOTSTRAP_TASK_ID,
    MAX_BOOTSTRAP_FILES,
    Bootstrap,
    StackChoice,
    choose_stack,
)
from swarm.nodes.planner import Draft, normalise, order_drafts, with_bootstrap
from swarm.state import PlannedTask

VERIFY = "python -m pytest -q"
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Answers:
    """An oracle with a scripted answer, recording what it was asked."""

    stack: str = "python"
    reason: str = "because"
    asked: list[Any] = field(default_factory=list)

    def invoke(self, messages: Sequence[tuple[str, str]]) -> StackChoice:
        self.asked.append(messages)
        return StackChoice(stack=self.stack, reason=self.reason)


class Unreachable:
    """Ollama restarting, a socket refusing, a model that is not pulled."""

    def invoke(self, messages: Sequence[tuple[str, str]]) -> StackChoice:
        raise RuntimeError("connection refused")


class Nonsense:
    """A model answering outside the vocabulary. `format` should make this
    impossible; the fallback exists because "should" is not "cannot"."""

    def invoke(self, messages: Sequence[tuple[str, str]]) -> StackChoice:
        return StackChoice(stack="haskell")


# --------------------------------------------------------------------------
# Which stack the prompt implies (D5)
# --------------------------------------------------------------------------


def test_a_ui_prompt_resolves_to_react():
    oracle = Answers(stack="react")

    assert choose_stack("a dashboard for warehouse pickers", llm=oracle) == "react"
    # The prompt reaches the model, rather than a summary of it: "a dashboard
    # for warehouse pickers" is a judgement, and a word list is what #87 is
    # removing.
    assert "warehouse pickers" in oracle.asked[0][-1][1]


def test_a_service_prompt_resolves_to_python():
    assert choose_stack("an API that stores receipts", llm=Answers(stack="python")) == "python"


def test_react_means_react_web():
    """D5 was narrowed deliberately: React Native is out of #87's scope, and a
    model that answered "react" meaning Native would get a Node image."""
    assert "react" in KNOWN_STACKS
    assert BOOTSTRAP_FILES["react"] != BOOTSTRAP_FILES["node"]
    assert any(path.endswith(".jsx") for path in BOOTSTRAP_FILES["react"])


def test_an_unreachable_model_falls_back_rather_than_failing_the_run():
    """The right failure. Python is the stack whose image this host has always
    had, and a greenfield run that refused to start because Ollama hiccuped
    during a one-word classification would be worse than one the operator
    re-runs with `--stack`."""
    assert choose_stack("anything", llm=Unreachable()) == DEFAULT_STACK


def test_the_fallback_says_why_it_fell_back(capsys):
    """The control on the test above, and the whole point of this pair.

    The fallback was silent, and the `except` did not even bind the exception -
    so an operator who typed a React brief and got a Python scaffold could not
    tell a refused socket from a model that genuinely answered "python". The
    behaviour is unchanged; only the diagnosis is new.
    """
    choose_stack("a dashboard", llm=Unreachable())

    reported = capsys.readouterr().err
    assert "RuntimeError" in reported, "the exception type is what distinguishes the two cases"
    assert "connection refused" in reported


def test_an_answer_outside_the_vocabulary_falls_back():
    assert choose_stack("anything", llm=Nonsense()) == DEFAULT_STACK


def test_an_out_of_vocabulary_answer_reads_differently_from_an_unreachable_one(capsys):
    """Two fallbacks, two faults. This one says something about the model;
    the other says something about the host."""
    choose_stack("anything", llm=Nonsense())

    reported = capsys.readouterr().err
    assert "haskell" in reported
    assert "RuntimeError" not in reported


def test_an_explicit_stack_is_not_a_question_for_the_model():
    """`swarm run --stack node` is the operator saying what the repository is.
    Asking anyway would spend a model call to maybe disagree with them."""
    oracle = Answers(stack="react")

    bootstrap = Bootstrap.for_prompt("a dashboard", stack="node", llm=oracle)

    assert bootstrap.stack == "node"
    assert oracle.asked == []


# --------------------------------------------------------------------------
# What the bootstrap is allowed to write
# --------------------------------------------------------------------------


def test_the_generated_set_is_bounded_by_the_file_list():
    """#88's constraint, and the reason it is a list rather than a limit: the
    generated set has to be bounded by something a human wrote down, not by a
    context window."""
    for stack, files in BOOTSTRAP_FILES.items():
        assert files, stack
        assert len(files) <= MAX_BOOTSTRAP_FILES, stack


def test_every_declarable_stack_can_be_bootstrapped():
    assert set(BOOTSTRAP_FILES) == KNOWN_STACKS


def test_the_bootstrap_does_not_redeclare_what_provisioning_committed():
    """`provision.files()` writes README.md, LICENSE and the workflow in the
    initial commit. A bootstrap redeclaring them would have its edits refused
    by `apply_edits` as outside its own file set, and a worker may never write
    `.github/workflows/*` at all - one that can edit CI can edit its grader."""
    for stack, files in BOOTSTRAP_FILES.items():
        assert "README.md" not in files, stack
        assert "LICENSE" not in files, stack
        assert not any(path.startswith(".github/") for path in files), stack


def test_the_goal_quotes_the_prompt_rather_than_paraphrasing_it():
    """It is the only statement of what the project *is*, and a planner that
    summarised it would be deciding the product where nobody reviews it."""
    bootstrap = Bootstrap.for_prompt(
        "a CLI that converts markdown tables to CSV", stack="python"
    )

    assert "markdown tables to CSV" in bootstrap.goal


def test_the_goal_asks_for_a_suite_that_asserts_something():
    """#88 measured `node --test` exiting 0 on a repository with no tests in
    it, so "the gate passed" is not evidence a suite exists. Asking is weaker
    than #102's falsification and costs nothing."""
    assert "at least one real assertion" in Bootstrap.for_prompt("x", stack="node").goal


# --------------------------------------------------------------------------
# It is a task like any other
# --------------------------------------------------------------------------


def test_the_bootstrap_round_trips_through_the_contract_parser():
    """The property that keeps every other module ignorant of it."""
    bootstrap = Bootstrap.for_prompt("a markdown table tool", stack="node")

    drafts, rejected = normalise([bootstrap.task], verify=VERIFY)

    assert rejected == ()
    contract = parse_contract(7, drafts[0].body())
    assert contract.task_id == BOOTSTRAP_TASK_ID
    assert contract.files == bootstrap.files
    assert contract.stack == "node"


def test_every_other_task_is_blocked_on_the_bootstrap():
    """Without this the dispatcher runs three workers against an empty
    repository in the first cycle, each generating its own idea of the
    project."""
    bootstrap = Bootstrap.for_prompt("a tool", stack="python").task
    planned = [
        PlannedTask(id="add-csv", goal="g", files=["src/csv.py"]),
        PlannedTask(id="add-cli", goal="g", files=["src/cli.py"], depends_on=["add-csv"]),
    ]

    tasks = with_bootstrap(planned, bootstrap)

    assert tasks[0].id == BOOTSTRAP_TASK_ID
    assert all(BOOTSTRAP_TASK_ID in task.depends_on for task in tasks[1:])
    # The dependency the model asked for survives alongside the one we added.
    assert "add-csv" in tasks[2].depends_on


def test_the_bootstrap_is_ordered_first():
    bootstrap = Bootstrap.for_prompt("a tool", stack="python").task
    tasks = with_bootstrap([PlannedTask(id="later", goal="g", files=["a.py"])], bootstrap)

    drafts, _ = normalise(tasks, verify=VERIFY)

    assert [draft.task_id for draft in order_drafts(drafts)][0] == BOOTSTRAP_TASK_ID


def test_a_replan_that_re_emits_the_bootstrap_does_not_block_it_on_itself():
    """`order_drafts` refuses a self-edge, so this would fail the whole replan
    rather than the one task. The id is fixed precisely so a replan recognises
    the bootstrap it already created."""
    bootstrap = Bootstrap.for_prompt("a tool", stack="python").task

    tasks = with_bootstrap([bootstrap], bootstrap)

    assert len(tasks) == 1
    assert tasks[0].depends_on == []
    order_drafts(normalise(tasks, verify=VERIFY)[0])  # does not raise


def test_the_bootstrap_carries_the_stack_it_resolved():
    """#99 spawns the image from this, and #96 sets up the CI toolchain from
    it. A bootstrap that resolved react and declared python would generate a
    React app in a Python container."""
    assert Bootstrap.for_prompt("a dashboard", stack="react").task.stack == "react"


def test_a_task_the_model_planned_keeps_its_own_stack_out_of_it():
    """#87's non-goals: one repo, one stack. The plan's stack is the
    bootstrap's, and `write_plan` applies it to every task."""
    bootstrap = Bootstrap.for_prompt("a dashboard", stack="react")
    planned = [PlannedTask(id="later", goal="g", files=["src/x.jsx"], stack="python")]

    drafts, _ = normalise(
        with_bootstrap(planned, bootstrap.task), verify=VERIFY, stack=bootstrap.stack
    )

    assert {draft.stack for draft in drafts} == {"react"}


def test_no_test_here_reaches_a_model():
    """Stated as a test because it is the file's premise. Every oracle above is
    a double, and `choose_stack`'s default argument is the only path to a real
    one."""
    import inspect

    source = inspect.getsource(Bootstrap.for_prompt)

    assert "llm=llm" in source


# --------------------------------------------------------------------------
# Falsifying the gate (#102)
# --------------------------------------------------------------------------
#
# `planner.py:560-566` refuses to let the model choose a verify command,
# because "a command guessed wrong is a gate that was red before any worker
# touched the task". A model-generated bootstrap reopens that: the model would
# write both the code and the gate that judges it.
#
# `run` is scripted as an ordered list of exit codes, exactly as
# `fixtures/github.py`'s transport scripts responses. The whole decision table
# needs no container and no subprocess.


@dataclass
class Runs:
    """A scripted `run`. Each call takes the next exit code and records where."""

    codes: list[int]
    seen: list[tuple[str, Path]] = field(default_factory=list)

    def __call__(self, command: str, tree) -> int:
        self.seen.append((command, Path(tree)))
        return self.codes[len(self.seen) - 1]

    @property
    def trees(self) -> list[Path]:
        return [tree for _, tree in self.seen]


def a_tree(root: Path, files: Sequence[str] = ("src/main.py", "tests/test_main.py")) -> Path:
    tree = root / "clean"
    for relative in files:
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("real content\n")
    # Not declared, and therefore never mutated - the reason `test -f README.md`
    # stays refused.
    (tree / "README.md").write_text("# project\n")
    return tree


def test_a_gate_that_passes_clean_and_fails_broken_is_accepted(tmp_path):
    runs = Runs([0, 1])

    verdict = falsify("pytest -q", a_tree(tmp_path), ["src/main.py"], run=runs, workspace=tmp_path)

    assert verdict.accepted
    # Clean first, then the mutant - and the mutant is a different tree, not the
    # same one edited underneath the run that just passed.
    assert [tree.name for tree in runs.trees] == ["clean", "mutated"]


def test_neither_run_touches_the_tree_the_repository_will_keep(tmp_path):
    """The clean run used to execute against the original tree, mounted `:ro`.

    Both halves of that had to go. The mount cannot be read-only, because
    `vitest` writes a transformed config beside the real one and exits 1 if it
    cannot - a read-only probe does not refuse bad gates, it refuses whole
    stacks. And a writable mount of the *original* would hand one line of
    model-proposed shell the files that are about to become the initial
    project. So the probe runs on copies, and the original is never a volume.
    """
    tree = a_tree(tmp_path)
    runs = Runs([0, 1])

    falsify("pytest -q", tree, ["src/main.py"], run=runs, workspace=tmp_path)

    where = runs.trees
    assert [path.name for path in where] == ["clean", "mutated"]
    assert all(path != tree for path in where)
    # A copy, not an empty directory: the clean run has to see the real code.
    assert (where[0] / "src/main.py").read_text() == "real content\n"
    assert (tree / "src/main.py").read_text() == "real content\n"


def test_the_probe_is_confined_however_the_mount_changes(monkeypatch, tmp_path):
    """`--network none` was the *unpinned* half of the isolation pair.

    Deleting it passed the whole suite, docker-marked tests included, while
    three docstrings and `docs/security.md` rest the entire no-widened-egress
    argument on it - and now that `:ro` is gone it is the only thing standing
    between model-proposed shell and a package registry. Pinned here, together
    with the `HARDENING_FLAGS` the dispatcher has always applied and this probe
    did not, so an edit that widens either has to say so out loud.
    """
    from swarm.security import HARDENING_FLAGS
    import swarm.greenfield.bootstrap as module

    captured: dict[str, list[str]] = {}

    class Manager:
        def __init__(self, **kwargs):
            captured["flags"] = list(kwargs["extra_flags"])
            captured["image"] = kwargs["image"]
            captured["env"] = kwargs["env"]

        def spawn(self, *args, **kwargs):
            return object()

        def wait(self, handle, timeout_s=None):
            return 0

        def dispose(self, handle):
            return None

    monkeypatch.setattr("swarm.containers.manager.ContainerManager", Manager)
    module._container_run("echo hi", tmp_path, stack="python")

    flags = captured["flags"]
    assert flags[:2] == ["--network", "none"]
    for flag in HARDENING_FLAGS:
        assert flag in flags
    # Nothing that could carry a credential into model-proposed shell.
    assert captured["env"] == {}
    assert not any(":ro" in flag for flag in flags), "the mount is a copy, not read-only"


def test_the_probe_runs_in_the_stacks_own_image(monkeypatch, tmp_path):
    """React's gate is `vitest run` and `vitest` exists only in
    `apiary-worker-react`. Probing it in the Python image would refuse the
    stack's real gate as "red on the code it was written for" - the most
    misleading verdict this function can produce, because the command is fine
    and the container was wrong."""
    import swarm.greenfield.bootstrap as module

    seen: list[str] = []

    def fake(command, tree, *, stack):
        seen.append(stack)
        return 0 if not seen[1:] else 1

    monkeypatch.setattr(module, "_container_run", fake)

    verdict = falsify(
        "vitest run", a_tree(tmp_path), ["src/main.py"], stack="react", workspace=tmp_path
    )

    assert verdict.accepted
    assert seen == ["react", "react"]


def test_a_gate_that_is_red_on_its_own_code_is_refused(tmp_path):
    """The failure `planner.py` already refuses to risk: a gate that would
    block every pull request before a worker touched the task."""
    runs = Runs([1, 1])

    verdict = falsify("pytest -q", a_tree(tmp_path), ["src/main.py"], run=runs, workspace=tmp_path)

    assert not verdict.accepted
    assert "red on the code it was written for" in verdict.reason
    # And it never bothered running the mutation.
    assert len(runs.seen) == 1


def test_a_gate_that_survives_the_mutation_is_refused(tmp_path):
    runs = Runs([0, 0])

    verdict = falsify("pytest -q", a_tree(tmp_path), ["src/main.py"], run=runs, workspace=tmp_path)

    assert not verdict.accepted
    assert "not reading the code" in verdict.reason


@pytest.mark.parametrize("command", ["true", "exit 0", ":", "  true  ", "TRUE"])
def test_a_tautology_is_refused_without_creating_a_container(tmp_path, command):
    runs = Runs([])

    verdict = falsify(command, a_tree(tmp_path), ["src/main.py"], run=runs, workspace=tmp_path)

    assert not verdict.accepted
    assert "tautological" in verdict.reason
    assert runs.seen == []


def test_the_placeholder_verify_stays_refused(tmp_path):
    """The permanent regression case. `test -f README.md` is refused because
    the mutation **empties** the declared files rather than deleting the tree:
    README is not declared, so it survives, so the command passes on the mutant
    and is caught. A falsifier that deleted files would admit it."""
    tree = a_tree(tmp_path)

    def really_run(command: str, where) -> int:
        # The real semantics of `test -f README.md`, without a shell.
        return 0 if (Path(where) / "README.md").is_file() else 1

    verdict = falsify(
        PLACEHOLDER_VERIFY, tree, ["src/main.py"], run=really_run, workspace=tmp_path
    )

    assert not verdict.accepted
    assert "not reading the code" in verdict.reason


def test_the_mutation_empties_the_declared_files_and_nothing_else(tmp_path):
    tree = a_tree(tmp_path)

    mutated = empty_declared(tree, ["src/main.py"], tmp_path / "mutated")

    assert (mutated / "src/main.py").read_text() == ""
    # Still present, which is the point - not deleted.
    assert (mutated / "src/main.py").is_file()
    # Undeclared files are untouched, and so is the original tree.
    assert (mutated / "README.md").read_text() == "# project\n"
    assert (mutated / "tests/test_main.py").read_text() == "real content\n"
    assert (tree / "src/main.py").read_text() == "real content\n"


def test_a_declared_file_that_does_not_exist_is_not_an_error(tmp_path):
    """A bootstrap whose generation was partial - #88 measured 9% of them -
    must still be falsifiable rather than crashing the provisioning run."""
    mutated = empty_declared(a_tree(tmp_path), ["src/never-written.py"], tmp_path / "m")

    assert mutated.is_dir()


def test_the_mutation_is_ours_and_deterministic(tmp_path):
    """Never model-chosen: a mutation the model picked would put the same model
    on both sides of the test, which is the failure falsification exists to
    prevent. Deterministic, so two runs of one plan agree."""
    first = empty_declared(a_tree(tmp_path), ["src/main.py"], tmp_path / "a")
    second = empty_declared(a_tree(tmp_path), ["src/main.py"], tmp_path / "b")

    assert sorted(p.name for p in first.rglob("*")) == sorted(p.name for p in second.rglob("*"))


def test_a_refusal_is_a_verdict_and_never_an_exception(tmp_path):
    """A planning refusal with a printed reason. Raising would abort a
    provisioning run that has already created a repository."""
    verdict = falsify("x", a_tree(tmp_path), ["src/main.py"], run=Runs([1]), workspace=tmp_path)

    assert isinstance(verdict, Verdict)
    assert "REFUSED" in str(verdict)
    assert verdict.command in str(verdict)


def test_the_falsification_clock_is_not_the_verify_clock():
    """They bound different things. Sharing the number would give a repository
    with a legitimately slow suite five minutes per probe, twice, at provision
    time."""
    assert FALSIFY_TIMEOUT_S != SETTINGS.verify_timeout_s


def test_nothing_under_greenfield_runs_a_shell():
    """The highest-severity item in the epic, as an assertion.

    Provisioning runs in the orchestrator process, which holds the boot key
    (`administration` and `workflows`) and a `DOCKER_HOST` with
    `ALLOW_START=1`. `assert_unprivileged` only inspects argv that
    `ContainerManager.spawn` built, so a shell one-liner calling `docker`
    directly bypasses it. One line of model-proposed shell there is host root
    plus the ability to rewrite the CI it is being falsified against.

    Asserted over the **parse tree**, not the source text: the text contains
    the word `subprocess` in the docstring explaining why there is none, which
    is exactly how a grep-based version of this test fails.
    """
    import ast

    import swarm.greenfield as greenfield

    for path in Path(greenfield.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] in {"subprocess", "os"} and a.name == "subprocess" for a in node.names), path.name
            if isinstance(node, ast.ImportFrom):
                assert node.module != "subprocess", path.name
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "shell":
                        assert not (
                            isinstance(keyword.value, ast.Constant) and keyword.value.value
                        ), path.name
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                assert name not in {"system", "popen", "execv", "fork"}, f"{path.name}: {name}"


def test_the_proposed_gate_asks_for_one_that_can_fail():
    """The schema description is the prompt. #88 measured `node --test` exiting
    0 on a repository with no tests in it, so "write a test command" is not
    enough of an instruction."""
    described = ProposedGate.model_json_schema()["properties"]["command"]["description"]

    assert "MUST fail" in described
    assert "empty" in described


@pytest.mark.docker
def test_falsification_runs_the_real_command_in_a_real_container(tmp_path):
    """The one live test. Everything above scripts `run`; this proves the
    default actually executes, in a container, with no network, and returns an
    exit code the decision table can read.

    Skipped rather than failed when the image is not built - that is a manual
    host step by design (`BUILD=0`), so its absence is a fact about the machine.
    """
    from swarm.containers.manager import ContainerError, StackImages, build_hint, missing_image

    tree = tmp_path / "clean"
    (tree / "tests").mkdir(parents=True)
    (tree / "main.py").write_text("def add(a, b):\n    return a + b\n")
    (tree / "tests" / "__init__.py").write_text("")
    (tree / "tests" / "test_main.py").write_text(
        "import unittest\n"
        "from main import add\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(1, 2), 3)\n"
    )

    try:
        verdict = falsify(
            "python3 -m unittest discover -q",
            tree,
            ["main.py", "tests/test_main.py"],
            workspace=tmp_path,
        )
    except ContainerError as exc:
        image = StackImages().for_stack("python")
        if not missing_image(exc):
            raise
        pytest.skip(f"{image} is not built on this host: {build_hint(image)}")

    # Green on the code it was written for, red once that code is emptied.
    assert verdict.accepted, verdict.reason


@pytest.mark.docker
def test_the_react_gate_survives_falsification_in_a_real_container(tmp_path):
    """#106's whole claim, run rather than argued.

    A real JSX component rendered through `@testing-library/react` and jsdom,
    asserted on, in `apiary-worker-react` with **no network at all** - which is
    the thing the epic assumed impossible without widening the egress
    allowlist. The toolchain comes from `/node_modules` in the image; nothing
    here installs anything.

    It also covers the two mechanics that are invisible in a unit test:
    `/node_modules` resolving from a working directory that declares none (the
    reason `NODE_PATH` is not used - Node's ESM resolver ignores it), and the
    probe's mount being writable (`vitest` writes a transformed config beside
    the real one and exits 1 on `:ro`).
    """
    from swarm.containers.manager import ContainerError, StackImages, build_hint, missing_image

    tree = tmp_path / "clean"
    (tree / "src").mkdir(parents=True)
    (tree / "test").mkdir(parents=True)
    (tree / "package.json").write_text(
        '{"name": "probe", "version": "0.1.0", "private": true, "type": "module"}\n'
    )
    (tree / "vitest.config.js").write_text(
        "import { defineConfig } from 'vitest/config'\n"
        "import react from '@vitejs/plugin-react'\n\n"
        "export default defineConfig({\n"
        "  plugins: [react()],\n"
        "  test: { environment: 'jsdom', globals: true },\n"
        "})\n"
    )
    (tree / "src" / "App.jsx").write_text(
        "export default function App() {\n"
        "  return <h1>two</h1>\n"
        "}\n"
    )
    (tree / "test" / "App.test.jsx").write_text(
        "import '@testing-library/jest-dom/vitest'\n"
        "import { render, screen } from '@testing-library/react'\n"
        "import App from '../src/App.jsx'\n\n"
        "it('renders', () => {\n"
        "  render(<App />)\n"
        "  expect(screen.getByText('two')).toBeInTheDocument()\n"
        "})\n"
    )
    declared = ["package.json", "vitest.config.js", "src/App.jsx", "test/App.test.jsx"]

    try:
        verdict = falsify(
            STACK_VERIFY["react"], tree, declared, stack="react", workspace=tmp_path
        )
    except ContainerError as exc:
        image = StackImages().for_stack("react")
        if not missing_image(exc):
            raise
        pytest.skip(f"{image} is not built on this host: {build_hint(image)}")

    assert verdict.accepted, verdict.reason


def test_an_operators_verify_command_is_never_falsified(tmp_path):
    """The escape hatch, and an escape hatch that can be refused is not one: a
    false rejection has no recovery path, because the operator has already told
    the system the answer."""
    runs = Runs([])

    verdict = choose_gate(
        "pytest -q",
        a_tree(tmp_path),
        ["src/main.py"],
        stack="python",
        operator="make check",
        run=runs,
    )

    assert verdict.accepted
    assert verdict.command == "make check"
    assert runs.seen == []


def test_a_refused_gate_keeps_the_placeholder_rather_than_failing_the_run(tmp_path):
    """Aborting would destroy a generated project over a gate, and the project
    is the expensive part."""
    verdict = choose_gate(
        "true",
        a_tree(tmp_path),
        ["src/main.py"],
        stack="python",
        fallback=PLACEHOLDER_VERIFY,
        run=Runs([]),
        workspace=tmp_path,
    )

    assert not verdict.accepted
    assert verdict.command == PLACEHOLDER_VERIFY
    assert "tautological" in verdict.reason


def test_an_accepted_gate_replaces_the_placeholder(tmp_path):
    verdict = choose_gate(
        "pytest -q",
        a_tree(tmp_path),
        ["src/main.py"],
        stack="python",
        fallback=PLACEHOLDER_VERIFY,
        run=Runs([0, 1]),
        workspace=tmp_path,
    )

    assert verdict.accepted
    assert verdict.command == "pytest -q"


# --------------------------------------------------------------------------
# The per-stack gate rule, inherited from the deleted scaffold (#104)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stack", sorted(STACK_VERIFY))
def test_no_stacks_gate_installs_anything(stack: str):
    """`test_the_verify_command_installs_nothing`, kept as a per-stack rule.

    It was the sharpest statement of the cost tension in the old scaffold and
    it survives the templates' deletion. The command is not run once: it runs
    in every worker container, on every attempt, and again in CI on every push,
    so an install step in it is an install step on the price of every task in
    the repository forever.

    It is also *impossible*: a worker's only route out is the egress proxy's
    static allowlist, so an install is denied in under a second - which is why
    #90 has `DENIED_EGRESS_SIGNATURES` at all.
    """
    command = STACK_VERIFY[stack]

    assert not {"pip", "install", "npm", "poetry", "uv", "curl", "wget"} & set(command.split())


@pytest.mark.parametrize("stack", sorted(STACK_VERIFY))
def test_every_stacks_gate_is_a_single_line_a_contract_will_accept(stack: str):
    """`docs/issue-contract.md` §1.3: one command, and only its exit code is
    believed. "Run these in order, stopping on failure" is semantics nobody
    agreed to - write `&&`."""
    assert len(STACK_VERIFY[stack].splitlines()) <= 1


def test_the_python_gate_says_python3_not_python():
    """The generated workflow emits no setup step for Python, so the spelling
    guaranteed on a bare runner and in `python:3.12-slim` is `python3`."""
    assert STACK_VERIFY["python"].startswith("python3 ")


def test_the_node_gate_cannot_pass_on_a_project_with_no_tests():
    """#88 measured `node --test` exiting **0** on a repository with no tests
    in it, and no flag fixes it - so the bare command would grade an empty or
    partial generation green. The guard is what makes the gate able to fail."""
    command = STACK_VERIFY["node"]

    assert "node --test" in command
    assert command != "node --test"
    assert "test -n" in command


def test_every_declarable_stack_has_a_gate_entry():
    """Every stack, and every one of them non-empty.

    The wider house convention is 'empty says "considered", missing says
    "forgotten"' - `ledger.GENERATED_FILES` still uses it. `STACK_VERIFY` no
    longer does, and that is the assertion below: React's entry was empty until
    #106, because `node --test` cannot run JSX without a transform and
    inheriting the placeholder beat claiming a command that could not run. Now
    that every declarable stack has a gate that runs, an empty one is a stack
    whose gate somebody forgot."""
    assert set(STACK_VERIFY) == KNOWN_STACKS
    assert all(command for command in STACK_VERIFY.values())


def test_the_react_gate_needs_no_guard_where_the_node_one_did():
    """The opposite of #88's problem, measured in `apiary-worker-react` with
    `--network none`: working component and test **0**; component broken **1**;
    test files removed **1** ("No test files found"); every declared file
    emptied **1**. So `vitest run` survives #102's falsification on its own,
    where `node --test` needed `test -n "$(ls test/*.test.js)"` in front of it.

    Bare, rather than `npm test` or `npx vitest`, and that is the load-bearing
    part: the worker image puts `/node_modules/.bin` on `PATH` and the
    generated workflow puts `node_modules/.bin` on the runner's, so the same
    bytes run in both places. `npx` would work too and is worse - it is an
    installer, so its behaviour differs on the two sides of the egress fence.
    """
    assert STACK_VERIFY["react"] == "vitest run"


def test_the_react_bootstrap_declares_the_config_its_gate_needs():
    """`vitest run` cannot transform JSX without `@vitejs/plugin-react`, and
    `vitest.config.js` is where that is turned on. Undeclared, the bootstrap
    would write a project whose gate cannot parse its own source - and
    `apply_edits` would refuse the file if the model wrote it anyway."""
    assert "vitest.config.js" in BOOTSTRAP_FILES["react"]


def _npm_specs(text: str) -> set[str]:
    """The `name@version` arguments of the one `npm install` in `text`.

    Everything from `npm install` to the shell operator that ends it, minus the
    flags. Bounded that way rather than by scanning the whole file, because a
    Dockerfile is full of `@` - `worker@apiary.invalid` in the git identity
    would otherwise read as a package.

    Comments go first, and they have to: `Dockerfile.worker.react` explains at
    length why `npm install --prefix /` does not work, and prose about a
    command is not the command.
    """
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    # Checked, not assumed: a *second* `RUN npm install` layer would be
    # invisible to a parser that reads the first one and stops, which is
    # precisely the "package in the image that the workflow never installs"
    # drift the caller advertises catching.
    assert code.count("npm install") == 1, "expected exactly one npm install"
    body = code.split("npm install", 1)[1]
    for terminator in ("&&", "\n\n", "|"):
        body = body.split(terminator, 1)[0]
    return {
        word
        for word in body.split()
        if "@" in word.lstrip("@") and not word.startswith("-")
    }


def test_the_react_toolchain_is_pinned_identically_everywhere():
    """Three copies, none of which can import the others.

    The image installs the toolchain at build time; the generated workflow
    installs it on a runner, because a GitHub runner has no such image; and
    `stacks.REACT_TOOLCHAIN` is the constant both are checked against. Drift is
    silent and its symptom is a red CI run on a green worker, which is the one
    result that makes the whole gate untrustworthy.
    """
    dockerfile = (REPO_ROOT / "Dockerfile.worker.react").read_text(encoding="utf-8")
    ci = next(line for line in CI_SETUP["react"] if "npm install" in line)

    # Set equality in both directions, not a substring sweep: an image carrying
    # a package the workflow does not install is a green worker and a red
    # runner, which is the failure this test exists for.
    assert _npm_specs(dockerfile) == set(REACT_TOOLCHAIN)
    assert _npm_specs(ci) == set(REACT_TOOLCHAIN)


def test_the_react_worker_is_not_told_to_use_the_standard_library_only():
    """Every bootstrap used to be, and for React the instruction is impossible:
    react and react-dom *are* dependencies, so a model obeying it literally
    would write no React at all. The React text names the packages instead,
    because a worker has no route to a registry - anything outside that set is
    not slow to add, it is unobtainable.

    **Asserted on the system prompt, not the goal** (#293). The rule used to be
    spliced into the issue body, which made it an instruction on the one channel
    the orchestrator does not control - anyone with repository access can edit a
    work item. It is handed down at dispatch now; `system_for` is the whole of
    what the orchestrator says.
    """
    system = system_for("react")

    # The *python* rule, not the phrase. `SYSTEM` has always carried a general
    # "prefer the standard library when it suffices", which is a preference and
    # reaches every stack; what must never reach react is the flat prohibition,
    # because react and react-dom are dependencies by definition. Asserting on
    # the phrase alone passed only while the rule lived in a stack-specific goal.
    assert STACK_RULE["python"] not in system
    assert "no dependencies beyond" not in system
    assert "React Native" in system
    for name in ("react", "vitest", "@testing-library/react"):
        assert name in system
    # The version pins belong to the image and the workflow, not to a prompt: a
    # model asked to reproduce them would put them in `package.json` and be
    # graded on typing accuracy.
    assert "react@18" not in system


def test_a_scoped_package_survives_having_its_version_stripped():
    """`rpartition`, not `partition`: a scoped name starts with the same `@`
    the version is separated by, so splitting on the first one turns
    `@vitejs/plugin-react@4` into an empty string - and the prompt would name a
    package called nothing."""
    assert package_names(("react@18", "@vitejs/plugin-react@4")) == (
        "react",
        "@vitejs/plugin-react",
    )
    assert all(package_names())


def test_every_declarable_stack_says_what_it_may_depend_on():
    assert set(STACK_RULE) == KNOWN_STACKS


def test_the_react_rule_demands_the_jest_dom_registration_import():
    """Shipping the package supplies nothing on its own - `expect` learns
    `toBeInTheDocument` only once the registration module has run - and
    `toBeInTheDocument()` is what a model writes whether or not anything told
    it to. The package in the image without this line in the prompt is the
    failure the package is there to prevent.

    It must be an import in the test file. `setupFiles` was measured and does
    not work with the toolchain at `/`: Vite reads the resolved absolute path
    as a root-relative URL under the project root and fails to load a file that
    exists.
    """
    rule = STACK_RULE["react"]

    assert "@testing-library/jest-dom/vitest" in rule
    assert "toBeInTheDocument" in rule
    assert "setupFiles" not in rule


def test_the_react_rule_never_hangs_the_package_list_off_a_prohibition():
    """"do not add any others: react, react-dom, ..." binds the colon to the
    nearest clause, and a plausible reading is that React itself is forbidden -
    which would produce a package.json with no React in it."""
    rule = STACK_RULE["react"]
    before_list = rule.split(package_names()[0] + ",")[0]

    assert "already installed" in before_list
    assert "do not" not in before_list


def test_the_python_worker_still_gets_the_standard_library_rule():
    """The per-stack split must not have quietly relaxed the stacks it was not
    about, and neither must the move to the system prompt: a worker image has no
    installer reachable, so the rule is as load-bearing as it ever was."""
    assert "standard library" in system_for("python")


def test_the_task_goal_carries_no_behavioural_rules_at_all():
    """The property the move exists for (#293).

    `docs/issue-contract.md` says what a goal is - "one sentence: what must be
    true when this is done" - a specification. A work item is a document anyone
    with repository access can edit and its comments are a channel anyone can
    write to, so a rule that lives there is a rule set by whoever typed last.
    Instructions come from the orchestrator; the work item is data.
    """
    goal = Bootstrap(prompt="a beautiful to-do list", stack="react", files=()).goal

    for directive in ("TypeScript", "jsdom", "standard library", "@testing-library",
                      "do not add", "never .ts"):
        assert directive not in goal, f"the goal is carrying an instruction: {directive!r}"


def test_an_unknown_stack_is_told_nothing_it_cannot_act_on():
    """A `--repo` run against someone else's repository has no entry in the
    table, and inventing one would be the orchestrator asserting a constraint it
    has no evidence for."""
    assert system_for("cobol") == system_for(None)


def test_the_python_workflow_installs_the_gates_tool():
    """The generated workflow runs `python -m pytest -q` on a bare GitHub
    runner, and a bare runner has no pytest: the first greenfield python run
    passed its worker gate in a container that ships pytest and then failed
    the identical command in CI, on every PR, forever. The workflow ships the
    gate, so it ships the gate's tool - same reasoning, same package, as
    Dockerfile.worker."""
    steps = "\n".join(CI_SETUP["python"])

    assert "pip install" in steps and "pytest" in steps


def test_the_react_bootstrap_declares_the_stylesheet_it_will_import():
    """Observed live (#293): asked for "a beautiful list of to do", the react
    bootstrap emitted `import "./App.css"` from an undeclared path.

    It was obeying the rule it had - a worker may only *write* declared files, and
    it wrote none it should not have. The task was unwinnable anyway: vite cannot
    resolve the import, `test/App.test.jsx` never loads, `vitest run` reports zero
    tests, and three identical attempts later a human is asked about a CSS file.

    Two fixes, and this is the enabling half. `worker/edit.SYSTEM` forbids
    importing a path that cannot resolve, which is the general rule for every
    stack; for a stack whose briefs routinely ask for something that looks good,
    "do without styles" is a worse answer than one declared file.
    """
    files = BOOTSTRAP_FILES["react"]

    assert "src/App.css" in files
    assert len(files) <= MAX_BOOTSTRAP_FILES


def test_the_worker_is_told_that_an_unresolvable_import_fails_the_task():
    """The prompt said "edit ONLY the files listed as editable" and nothing about
    *importing* a path that is neither editable nor already there - so a model
    could follow its instructions exactly and still write a task that cannot
    pass. Stack-agnostic: `from .helpers import x` fails the same way."""
    from swarm.worker.edit import SYSTEM

    assert "Every relative import must resolve" in SYSTEM
    assert "zero tests" in SYSTEM
