"""ADR 0003, held still: the framework stays a detail, and runners say what they do.

Two assertions live here. They are unrelated in mechanism and identical in
motive - both properties are true on `main` today, both became true by accident
rather than decision, and both have already decayed once without anyone
noticing.

**One: no agent framework crosses into the working modules.** `orchestrator/`,
`github/`, `containers/`, `worker/`, `artifacts.py`, `security.py` and
`doctor.py` carry no framework import. That is what makes adding a Hermes
runner alongside the LangGraph one cost a module rather than a rewrite, and
until this file existed nothing protected it: `graph.py` sat vestigial through
the whole v2 rewrite and then went live again in #135, when `swarm local`
started running the v1 graph, without anybody deciding the framework should be
load-bearing again.

The permitted exceptions are named below with a reason each. **The list is the
point.** It makes the coupling countable, and widening it is then a deliberate
edit somebody has to justify in a review rather than an import nobody noticed.

**Two: every runner declares its capabilities, and the declaration matches the
code.** ADR 0003 decision 4. The declarations are in `src/swarm/runners.py`;
this file asserts them against the import graph, in both directions - a runner
claiming a sandbox it does not build fails, and so does one that quietly grew
one without saying so.

The failure this is really for is neither of those. It is the runner that
declares *nothing at all*, because that is precisely how `swarm local`
arrived: a second execution model with no container, no pull-request gate, no
merge queue and no egress policy, offered by the CLI as "a local checkout, no
GitHub", noticed by nobody through an entire architecture rewrite. So the
check is written against the parser's subcommands rather than against a list of
the two runners that exist - a third entry point fails the suite by existing
undeclared, which is the only moment at which anybody is guaranteed to be
looking.

**Static, over the import graph, and not a runtime probe.** Both assertions.
An import that fires on one code path is exactly the one a probe misses, and a
capability check that ran the runner would only ever see the path it happened
to take - a dry run that never dispatched would report no sandbox and be
believed.

**Why "reaches" is not the transitive closure.** The reach of a runner is
computed from its entry point through the functions it calls in its own
modules, and the modules those name. The full transitive closure of the import
graph was tried first and answers a different question: `nodes/judge.py`
imports one exit code from `worker/entrypoint.py`, which imports a regex from
`containers/manager.py`, so the closure of anything touching the v1 graph
contains `containers/` - and the closure would cheerfully report that `swarm
local` has a container sandbox. It does not. A capability is something a runner
*composes*, and composing it is what this measures.

The cost of that choice is a runner whose wiring lives in a helper module: it
must name that module in its declaration's `wiring`, or be read as not having
what the helper builds. That is a false failure with an obvious fix, in the
direction of under-claiming, which is the safe direction for a security
property.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pytest

import swarm
from swarm import cli
from swarm.runners import (
    CAPABILITIES,
    NOT_RUNNERS,
    RUNNERS,
    Runner,
    capability,
    capability_table,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
ADR = "docs/adr/0003-orchestration-framework-is-a-detail.md"


# --------------------------------------------------------------------------
# What counts as an agent framework
# --------------------------------------------------------------------------

#: Matched against the first dotted segment of an imported module, as itself or
#: as its `name_` prefix - so `langchain_ollama` and
#: `langgraph_checkpoint_sqlite` are caught without listing every distribution
#: a family ships. `pydantic` is deliberately absent and `pydantic_ai` is
#: deliberately present: the schema library is used everywhere and is not an
#: orchestration framework, and the agent framework built on it is.
#:
#: The list is not a complete census of the ecosystem and does not need to be.
#: It names the family apiary depends on, and the ones somebody reaching for a
#: second framework would plausibly reach for. Adopting one that is not here
#: means adding it here, which is the same deliberate edit the exception list
#: below asks for.
AGENT_FRAMEWORKS: tuple[str, ...] = (
    "langgraph",
    "langchain",
    "langsmith",
    "llama_index",
    "llamaindex",
    "autogen",
    "crewai",
    "semantic_kernel",
    "dspy",
    "haystack",
    "smolagents",
    "agno",
    "pydantic_ai",
    "atomic_agents",
    "griptape",
    "strands",
)


def framework_of(module: str) -> str | None:
    """Which framework `module` belongs to, if any."""
    head = module.split(".", 1)[0]
    for name in AGENT_FRAMEWORKS:
        if head == name or head.startswith(name + "_"):
            return name
    return None


# --------------------------------------------------------------------------
# The boundary
# --------------------------------------------------------------------------

#: ADR 0003 decision 1, module for module. These are the modules that carry the
#: system: the reconcile loop, the tracker adapters, the container layer, the
#: worker, the artifacts, the security policy and the preflight. A framework
#: import in any of them is the property gone.
GUARDED: tuple[str, ...] = (
    "swarm.orchestrator",
    "swarm.github",
    "swarm.containers",
    "swarm.worker",
    "swarm.artifacts",
    "swarm.security",
    "swarm.doctor",
)

#: The coupling that is allowed to exist, and why each one is here. Four
#: entries, which is the number ADR 0003 counted; a fifth is a decision, not an
#: import.
PERMITTED: dict[str, str] = {
    # The real one, and the only place a graph is built. It is `swarm local`'s
    # entire execution model, and ADR 0003's claim is that replacing the
    # framework means rewriting this module rather than the system.
    "swarm.graph": "builds the v1 StateGraph; the framework's one load-bearing use",
    # No framework import today - `Annotated[..., operator.add]` and
    # `Annotated[..., _merge_tasks]` are stdlib typing carrying LangGraph's
    # state-channel convention on types the v2 ledger also uses. Exempt because
    # the coupling is real even where the import is not, and because a reducer
    # that needed `langgraph.graph.add_messages` would belong here rather than
    # being a surprise.
    "swarm.state": "reducer annotations are LangGraph state channels, in stdlib typing",
    # 45 lines, LangChain rather than LangGraph, and ADR 0003 decision 2 keeps
    # it that size: the only module that knows a provider SDK exists.
    "swarm.llm": "the model client; the one module that knows a provider SDK exists",
    # LangChain's callback protocol, which is how prompts are recorded at all.
    # ADR 0003's consequences say this is rewritten under another framework and
    # the console feature goes with it - which is worth knowing before it grows.
    "swarm.capture": "prompt recording is written against LangChain's callback protocol",
}


def source_modules() -> dict[str, Path]:
    """Every module under `src/`, by dotted name."""
    found: dict[str, Path] = {}
    for path in sorted(SRC.rglob("*.py")):
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        found[".".join(parts)] = path
    return found


MODULES = source_modules()
TREES = {
    name: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for name, path in MODULES.items()
}


def absolute_imports(tree: ast.AST) -> list[tuple[str, int, str]]:
    """Every non-relative module a tree imports, as (imported, line, source).

    Function-level imports count. A framework import moved inside a function
    body to get under a boundary check is the thing the check exists to notice
    - and `cli.py`, `llm.py` and `_loop` all import from inside functions for
    reasons of their own, so a reader that only saw top-level statements would
    be wrong about this tree even before anybody tried to hide anything.
    """
    out: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names = ", ".join(alias.name for alias in node.names)
            out.append((node.module, node.lineno, f"from {node.module} import {names}"))
    return out


def guards(module: str) -> bool:
    return any(module == g or module.startswith(g + ".") for g in GUARDED)


def framework_imports_in(tree: ast.AST) -> list[tuple[str, int, str]]:
    """The rows of `absolute_imports` that belong to an agent framework."""
    return [row for row in absolute_imports(tree) if framework_of(row[0])]


def framework_imports(module: str) -> list[tuple[str, int, str]]:
    return framework_imports_in(TREES[module])


def test_the_working_modules_import_no_agent_framework():
    """ADR 0003 decision 1. The whole reason a second runner is cheap.

    The message names the module, the line and the import, because the fix is
    not "remove the import" - it is either to move the code that needs it into
    a runner, or to decide that the framework is load-bearing here and say so
    in the ADR. Both need to know exactly what crossed.
    """
    offences = [
        f"  {module} ({MODULES[module].relative_to(REPO)}:{line}): {text}"
        for module in sorted(MODULES)
        if guards(module)
        for _, line, text in framework_imports(module)
    ]

    assert not offences, (
        "an agent framework reached the modules that do the work:\n"
        + "\n".join(offences)
        + f"\n\n{ADR} decision 1: `orchestrator/`, `github/`, `containers/`, "
        "`worker/`, `artifacts.py`, `security.py` and `doctor.py` import no "
        "agent framework, so that a framework can be replaced - or run "
        "alongside another - without touching them. If the framework really "
        "does belong here, that is an edit to the ADR and to PERMITTED in this "
        "file, not an import."
    )


def test_the_framework_coupling_is_only_the_permitted_modules():
    """The exception list, asserted over the whole tree rather than the guarded
    part of it.

    ADR 0003's table counts the coupling that remains in four places, and its
    consequences claim that leaving LangGraph touches those four and nothing
    else. That is a statement about `src/swarm` entire, not about the guarded
    modules: a `langchain` import in `nodes/` or `console_intake.py` would
    falsify it just as thoroughly and would sit outside decision 1's list.

    So the coupling stays countable. Adding a fifth module here is allowed and
    is meant to be visible - it is one line in a review saying the framework
    now reaches somewhere new.
    """
    actual = {module for module in MODULES if framework_imports(module)}
    unexpected = sorted(actual - set(PERMITTED))

    assert not unexpected, (
        "these modules import an agent framework and are not on the permitted "
        "list:\n"
        + "\n".join(
            f"  {module}: " + "; ".join(text for _, _, text in framework_imports(module))
            for module in unexpected
        )
        + f"\n\nthe permitted four are {', '.join(sorted(PERMITTED))} - see "
        f"{ADR}. Widening this is a decision; make it one."
    )


def test_the_permitted_list_names_modules_that_exist():
    """A stale exemption is an exemption nobody re-examines.

    If `capture.py` is deleted or renamed, its entry here silently becomes a
    licence with no subject, and the next module to take its name inherits it.
    """
    missing = sorted(set(PERMITTED) - set(MODULES))
    assert not missing, (
        f"PERMITTED names modules that no longer exist: {missing}. Remove the "
        "entry, or point it at whatever replaced them."
    )


def test_the_permitted_reasons_are_written_down():
    """Decision 1 asks for a reason per exemption, not a list of names."""
    for module, reason in PERMITTED.items():
        assert len(reason.split()) >= 5, f"{module}'s exemption needs a reason"


def test_the_boundary_check_would_notice_an_import_that_crossed():
    """The check, proved non-vacuous.

    A boundary test that passes because its detector is broken looks exactly
    like a boundary test that passes because the boundary holds - and this one
    is expected to pass unchanged for a long time, which is the condition under
    which nobody would ever find out. So the detector is run against the
    imports it is meant to catch, including the two shapes that would otherwise
    slip: a `langchain_*` distribution nobody thought to list, and an import
    written inside a function body.
    """
    assert framework_of("langgraph.graph") == "langgraph"
    assert framework_of("langchain_ollama") == "langchain"
    assert framework_of("langgraph_checkpoint_sqlite") == "langgraph"
    assert framework_of("crewai.agents") == "crewai"
    # Not frameworks: the schema library every module uses, and the stdlib.
    assert framework_of("pydantic") is None
    assert framework_of("operator") is None

    smuggled = ast.parse(
        "from ..github.client import GitHubClient\n"
        "import os\n"
        "\n"
        "def apply_plan(state):\n"
        "    from langgraph.types import Send\n"
        "    return Send('worker', state)\n"
    )
    caught = framework_imports_in(smuggled)

    assert [row[0] for row in caught] == ["langgraph.types"], caught
    assert caught[0][1] == 5, "the offending line is named, or the message cannot be acted on"


def test_the_guarded_list_covers_the_modules_the_adr_names():
    """`GUARDED` is prefixes, and a prefix that matches nothing guards nothing.

    A rename - `orchestrator/` to `engine/`, say - would leave this list
    matching no module at all, and every assertion above would pass by having
    nothing to check.
    """
    for prefix in GUARDED:
        assert any(
            module == prefix or module.startswith(prefix + ".") for module in MODULES
        ), f"{prefix} matches no module under src/; the guard is empty"

    # And the reconcile loop in particular, because it is the module ADR 0003
    # calls the system: if this is not guarded, nothing else being guarded
    # matters.
    assert guards("swarm.orchestrator.reconcile")
    assert not guards("swarm.graph"), "graph.py is the exception, not the guard"


# --------------------------------------------------------------------------
# The import graph, per runner
# --------------------------------------------------------------------------


def package_of(module: str) -> str:
    """The package a relative import inside `module` is relative to."""
    return module if MODULES[module].name == "__init__.py" else module.rpartition(".")[0]


def resolve_relative(module: str, node: ast.ImportFrom) -> str:
    base = package_of(module).split(".")
    if node.level > 1:
        base = base[: len(base) - (node.level - 1)]
    tail = [node.module] if node.module else []
    return ".".join([*base, *tail])


def bound_names(module: str, nodes) -> dict[str, tuple[str, str]]:
    """name in this namespace -> (module it came from, symbol name there).

    The symbol name is what lets the walk follow a call into another module the
    same runner owns: `hermes_run` bound from `swarm.hermes` is the function
    `run` over there, and the reach continues from its body.
    """
    out: dict[str, tuple[str, str]] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out[alias.asname or alias.name.split(".")[0]] = (alias.name, "")
        elif isinstance(node, ast.ImportFrom):
            target = (
                resolve_relative(module, node) if node.level else (node.module or "")
            )
            for alias in node.names:
                out[alias.asname or alias.name] = (target, alias.name)
    return out


def top_level_functions(module: str) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in TREES[module].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def loaded_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def reached_modules(entrypoint: str, owns: tuple[str, ...]) -> set[str]:
    """Every module the code at `entrypoint` composes itself out of.

    The walk starts at one function and follows two edges: a name bound by an
    import, which records the module it came from, and a call to another
    top-level function in a module this runner owns, which continues the walk
    there. Imports written inside a function body count - `_loop` imports the
    container manager and the merge policy from inside itself, and a reader
    that only saw module-level imports would report `swarm run` as having
    neither.
    """
    module, _, function = entrypoint.partition(":")
    if module not in MODULES:
        raise AssertionError(f"{entrypoint}: no module {module} under src/")

    reached: set[str] = set()
    seen: set[tuple[str, str]] = set()
    queue = [(module, function)]
    while queue:
        here, name = queue.pop()
        if (here, name) in seen:
            continue
        seen.add((here, name))
        functions = top_level_functions(here)
        if name not in functions:
            raise AssertionError(
                f"{here}:{name} is declared as an entry point and does not exist"
            )
        body = functions[name]
        bindings = {
            **bound_names(here, TREES[here].body),
            **bound_names(
                here,
                [n for n in ast.walk(body) if isinstance(n, (ast.Import, ast.ImportFrom))],
            ),
        }
        for used in loaded_names(body):
            if used in bindings:
                target, symbol = bindings[used]
                reached.add(target)
                if target in owns and symbol:
                    queue.append((target, symbol))
            elif used in functions:
                queue.append((here, used))
    return reached


def provides(modules: set[str]) -> set[str]:
    """Which capabilities a set of reached modules amounts to."""
    return {
        entry.name
        for entry in CAPABILITIES
        if any(
            reached == prefix or reached.startswith(prefix + ".")
            for reached in modules
            for prefix in entry.modules
        )
    }


def reach_of(declaration: Runner) -> set[str]:
    return provides(reached_modules(declaration.entrypoint, declaration.owns))


def subcommands() -> set[str]:
    """Every `swarm <command>`, read off the parser rather than a list.

    Reading the parser is the whole mechanism: a third runner is a subcommand
    before it is anything else, so it appears here the moment it is added and
    fails the classification test until somebody has said what it can do.
    """
    parser = cli.build_parser()
    for action in parser._actions:  # noqa: SLF001 - argparse exposes no public reader
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("the CLI has no subcommands; this test cannot see runners")


# --------------------------------------------------------------------------
# The declarations
# --------------------------------------------------------------------------


def test_the_analysis_reads_the_tree_it_is_asserting_about():
    """Both assertions parse `src/` and import `swarm`, and they must be the
    same `swarm`.

    apiary is worked on in several git worktrees at once, and an editable
    install points at exactly one of them - so a suite run from a worktree can
    parse this checkout's source while importing another's declarations, and
    report on a tree nobody is looking at.
    """
    installed = Path(swarm.__file__).resolve().parent
    assert installed == (SRC / "swarm").resolve(), (
        f"the imported `swarm` is {installed}, and this file parses "
        f"{SRC / 'swarm'}. Run the suite against this checkout: "
        "`PYTHONPATH=src pytest -q`, or `pip install -e .` here."
    )


def test_every_subcommand_is_classified():
    """A new entry point meets `runners.py` before it reaches a user.

    This is the assertion the ticket exists for. `swarm local` became a second
    execution model with no sandbox and no gate, and the reason nobody caught
    it is that nothing anywhere had to be edited for it to appear. Now
    something does.

    Note what is *not* asserted: that the classification is honest. Somebody
    can file a runner under `NOT_RUNNERS` and describe it as a viewer. But that
    is a sentence written into a file that a reviewer reads, which is a
    different act from adding a subparser, and the test below about what a
    non-runner may compose makes the cheap version of that lie fail.
    """
    declared = {entry.command for entry in RUNNERS}
    excused = {entry.command for entry in NOT_RUNNERS}
    unclassified = sorted(subcommands() - declared - excused)

    assert not unclassified, (
        f"these subcommands are declared nowhere: {unclassified}\n"
        "Every entry point says what it can do. Add a `Runner` to "
        "src/swarm/runners.py with its capabilities, or a `NotARunner` saying "
        f"why it owns no execution model. {ADR}, decision 4: a runner declares "
        "its capabilities and a user's choice is presented in those terms - "
        "which cannot happen for a runner nobody wrote down."
    )


def test_nothing_is_classified_twice_or_invented():
    """The classification is a partition of the subcommands, both ways."""
    declared = [entry.command for entry in RUNNERS]
    excused = [entry.command for entry in NOT_RUNNERS]
    assert len(set(declared)) == len(declared), "a runner is declared twice"
    assert len(set(excused)) == len(excused), "a non-runner is listed twice"
    assert not set(declared) & set(excused), "a command is both a runner and not"

    stale = sorted((set(declared) | set(excused)) - subcommands())
    assert not stale, (
        f"these are declared and are not subcommands any more: {stale}. A "
        "declaration for a command that no longer exists is a claim nothing "
        "checks."
    )


def test_every_runner_declares_capabilities_that_mean_something():
    """A declaration is a structure, not a gesture.

    An empty capability set is a legitimate declaration - `swarm local`'s is
    empty and true - but it has to be the set on a record that exists, with an
    entry point that resolves and a summary written in capability terms. The
    failure mode being excluded is a `Runner` added to silence the test above
    with nothing in it.
    """
    known = {entry.name for entry in CAPABILITIES}
    for declaration in RUNNERS:
        assert declaration.module in MODULES, f"{declaration.entrypoint}: no such module"
        assert declaration.function in top_level_functions(declaration.module), (
            f"{declaration.entrypoint}: no such function"
        )
        unknown = sorted(declaration.capabilities - known)
        assert not unknown, (
            f"`swarm {declaration.command}` claims {unknown}, which is not a "
            f"capability. Known: {sorted(known)}. A new one is added to "
            "CAPABILITIES with the modules that provide it, so the claim stays "
            "checkable."
        )
        assert len(declaration.summary.split()) >= 5, (
            f"`swarm {declaration.command}` has no summary. It is what a user "
            "chooses between."
        )
        for extra in declaration.wiring:
            assert extra in MODULES, f"{declaration.command}: no module {extra}"


@pytest.mark.parametrize("declaration", RUNNERS, ids=lambda d: d.command)
def test_a_runner_reaches_exactly_what_it_declares(declaration: Runner):
    """The declaration against the import graph, in both directions.

    Overclaiming is the direction that matters: a runner offering a sandbox it
    does not build tells a user their code is contained when it is running on
    the host.

    Underclaiming fails too, which is not pedantry. It is the alarm for the
    reverse of #135 - a runner that quietly acquires a capability is a runner
    whose declaration has stopped describing it, and the next person to read
    the table is reading something that was true once. If the capability is
    real, the fix is one word in the declaration and a table that now says yes.
    """
    reached = reach_of(declaration)
    overclaimed = sorted(declaration.capabilities - reached)
    unclaimed = sorted(reached - declaration.capabilities)

    assert not overclaimed, (
        f"`swarm {declaration.command}` declares {overclaimed} and reaches "
        "none of the modules that provide it:\n"
        + "\n".join(f"  {name}: {capability(name).modules}" for name in overclaimed)
        + f"\n\nEither wire it up, or stop claiming it. If the capability is "
        "built in a module this runner owns, name that module in the "
        f"declaration's `wiring`. {ADR}, decision 4."
    )
    assert not unclaimed, (
        f"`swarm {declaration.command}` reaches {unclaimed} and does not "
        "declare it. A capability that arrived without the declaration moving "
        "is how this pair diverged in the first place - if the runner really "
        f"has it now, say so in src/swarm/runners.py. {ADR}, decision 4."
    )


@pytest.mark.parametrize("entry", NOT_RUNNERS, ids=lambda e: e.command)
def test_a_non_runner_composes_no_capability(entry):
    """The cheap version of a misfiled runner, caught.

    `NOT_RUNNERS` is where somebody would put a third runner to avoid writing
    its capabilities down. A subcommand that builds containers and merges pull
    requests is a runner whatever its entry in that list says, so the list is
    only usable for things that genuinely compose none of it.
    """
    reached = provides(reached_modules(entry.entrypoint, (entry.entrypoint.split(":")[0],)))
    assert not reached, (
        f"`swarm {entry.command}` is filed as not a runner - "
        f"\"{entry.why}\" - and composes {sorted(reached)}. Something that "
        "reaches the sandbox or the merge queue is an execution model. Declare "
        f"it as a `Runner` with its capabilities. {ADR}, decision 4."
    )


def test_swarm_local_records_the_gap_rather_than_closing_it():
    """The state of the world on the day this was written, pinned.

    `swarm local` has no container sandbox, no egress policy, no pull-request
    or CI gate and no merge queue: `nodes/verifier.py` runs the verify command
    with `shell=True` on the host, in a worktree of code a model has just
    written. This ticket documents that and does not fix it - whether the
    runner stays supported at all is the question ADR 0003 leaves open for the
    maintainer.

    What this asserts is that the *declaration* keeps saying so. Filling those
    four in without building them is the one edit that would make the table in
    `swarm local --help` lie, and the help is where somebody chooses this
    runner. The day the code changes, this test changes with it - deliberately,
    in the same commit, which is the point.
    """
    local = next(entry for entry in RUNNERS if entry.command == "local")
    absent = {"sandbox", "egress_policy", "quality_gate", "merge_queue"}

    assert not (local.capabilities & absent), (
        f"`swarm local` now declares {sorted(local.capabilities & absent)}. If "
        "the runner really gained them, this test is what you edit and the "
        "commit that does it is the one that closed ADR 0003's open question."
    )
    assert local.gap, "the gap is recorded in the declaration, for whoever reads it"


def test_the_local_help_says_what_the_declaration_says():
    """One vocabulary, two renderings - and the second is checked.

    #162 put a capability table in `swarm local --help`, by hand, at the same
    time as this test was being written to assert the same facts. Two copies of
    "does the local runner have a sandbox" is the divergence this is supposed
    to remove, so the help renders `capability_table` and this asserts that
    what a user reads is what the suite checked. `security.py` keeps the egress
    allowlist and the proxy config in step the same way.
    """
    parser = cli.build_parser()
    action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    help_text = action.choices["local"].format_help()

    run = next(entry for entry in RUNNERS if entry.command == "run")
    local = next(entry for entry in RUNNERS if entry.command == "local")
    for entry in CAPABILITIES:
        row = next(
            (line for line in help_text.splitlines() if line.strip().startswith(entry.label)),
            None,
        )
        assert row is not None, (
            f"`swarm local --help` does not name {entry.label!r}. The table is "
            "rendered from runners.py; a capability added there appears here."
        )
        columns = row.split()[-2:]
        assert columns == [
            "yes" if run.provides(entry.name) else "no",
            "yes" if local.provides(entry.name) else "no",
        ], f"the help's row for {entry.label!r} disagrees with the declaration"

    assert capability_table("run", "local") in help_text


def test_the_only_console_script_is_the_cli():
    """A runner can also arrive as its own entry point in `pyproject.toml`.

    `swarm = swarm.cli:main` is the one script, so every runner is a subcommand
    and the classification above sees all of them. A second script would route
    around that entirely, so it fails here instead.
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - the project requires 3.11
        pytest.skip("no tomllib")

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert scripts == {"swarm": "swarm.cli:main"}, (
        f"new console scripts: {sorted(set(scripts) - {'swarm'})}. If one of "
        "them starts a run, it is a runner and declares its capabilities in "
        f"src/swarm/runners.py; teach this test how to find it. {ADR}, "
        "decision 4."
    )


# --------------------------------------------------------------------------
# The checks, proved to fail
# --------------------------------------------------------------------------
#
# Everything above passes on `main` and is meant to keep passing for a long
# time. That is the condition under which a check that stopped working would
# never be noticed - so each one is also run against a case it must reject.


def test_an_overclaiming_runner_is_rejected():
    """A declaration that claims what the code does not compose.

    `_local` builds the v1 graph and reaches no container, so a declaration
    pointing at it and claiming a sandbox is exactly the lie the assertion
    exists for. Fabricated here rather than committed, because the point is
    that the *checker* refuses it.
    """
    lying = Runner(
        command="local",
        entrypoint="swarm.cli:_local",
        capabilities=frozenset({"sandbox", "merge_queue"}),
        summary="claims a container it never builds, and a queue it never joins",
    )

    reached = reach_of(lying)

    assert "sandbox" not in reached and "merge_queue" not in reached
    assert lying.capabilities - reached == {"sandbox", "merge_queue"}


def test_a_real_capability_is_seen_through_a_function_level_import():
    """And the same checker says yes when the code does compose it.

    Half of a two-sided check is worth nothing on its own: an analysis that
    reported "reaches nothing" for every input would pass the test above. This
    is the other half, and it goes through `_loop`, which imports the container
    manager and the merge policy from inside the function body - the case a
    module-level-imports-only reader would miss.
    """
    reached = reach_of(next(entry for entry in RUNNERS if entry.command == "run"))
    assert {"sandbox", "merge_queue", "quality_gate", "egress_policy"} <= reached


def test_an_undeclared_subcommand_is_rejected():
    """The third runner, simulated.

    The parser is asked for its subcommands, so this fabricates the answer: a
    subcommand nobody classified must leave the classification incomplete.
    Written this way because a test that only ever compares the two commands
    that exist to the two declarations that exist would pass forever while
    checking nothing - it would still pass with the parser emptied.

    The fabricated name is deliberately one nobody will ever ship, so that a
    real third runner called `hermes` does not quietly satisfy this instead of
    failing the test above.
    """
    declared = {entry.command for entry in RUNNERS}
    excused = {entry.command for entry in NOT_RUNNERS}
    undeclared = "a-runner-nobody-wrote-down"

    arrived = subcommands() | {undeclared}

    assert sorted(arrived - declared - excused) == [undeclared]
    # And the real parser is genuinely being read, rather than an empty set
    # that would make the line above true whatever happened.
    assert {"run", "local"} <= subcommands()


def test_a_runner_that_owns_more_than_one_module_is_followed():
    """The `wiring` escape hatch, exercised.

    A runner living in its own module and composing its sandbox in a helper is
    the shape a Hermes runner would plausibly have, and the walk has to follow
    into it or read the runner as having nothing. `cli.py` is used as the
    stand-in: `_run` is declared as if it were reached from `main`, so the walk
    has to cross one function boundary to find anything at all.
    """
    from_main = Runner(
        command="run",
        entrypoint="swarm.cli:main",
        capabilities=frozenset(),
        summary="the dispatcher, standing in for a runner that spans modules",
        wiring=("swarm.cli",),
    )

    # `main` itself imports nothing interesting; everything below is found by
    # following its call into `_run` and `_loop`.
    assert {"sandbox", "merge_queue", "tracker"} <= reach_of(from_main)
