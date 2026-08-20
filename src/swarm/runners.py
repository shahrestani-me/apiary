"""What each runner can do, written so a test can check it.

ADR 0003 decision 3 makes the *runner* the unit of execution: a top-level entry
point owning a complete execution model and composing the framework-free
modules beneath it. `swarm run` and `swarm local` are two; a Hermes runner
would be a third, and adding one is additive work.

Decision 4 is the rule this module exists for. **A runner declares what it
provides, and a user's choice is presented in those terms** - never as a
framework name, and never as a convenience ("a local checkout, no GitHub")
when the thing actually being chosen away is a container.

The pair that exists arrived at a violation of that by nobody deciding
anything: `graph.py` was vestigial through the whole v2 rewrite and became
live again in #135, and `swarm local` shipped with no sandbox, no pull-request
gate, no merge queue and no egress policy without that ever being written
down. So the declarations below are asserted rather than trusted -
`tests/test_framework_boundary.py` reads each runner's entry point, walks the
import graph from it, and fails when a declaration and the code disagree in
either direction.

**One vocabulary, two renderings.** This is `security.py`'s arrangement, for
`security.py`'s reason: the egress allowlist is a predicate and a tinyproxy
config file, and they cannot disagree because the second is generated from the
first. Here the capability set is a test's assertion and the table in `swarm
local --help`, and `capability_table` generates the second from the same
tuple. Two hand-maintained copies of "does the local runner have a sandbox"
diverge, and the divergence is the failure this is meant to remove: #162 wrote
that table by hand at the same time as this ticket was being written to check
it, which is the argument in miniature.

**What a declaration is not.** It grants nothing and switches nothing on. A
runner does not acquire a sandbox by listing one; listing one it does not
compose is a test failure. The declaration is a claim about the code, kept
next to the code, in the shape a machine can refute.
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = [
    "Capability",
    "CAPABILITIES",
    "Runner",
    "RUNNERS",
    "NotARunner",
    "NOT_RUNNERS",
    "capability",
    "runner",
    "capability_table",
]


@dataclass(frozen=True)
class Capability:
    """One thing a runner either provides or does not.

    `modules` is what makes the claim checkable: a capability is provided by
    reaching the code that implements it, so "sandbox" means composing
    something out of `containers/` and nothing else counts. Entries are module
    prefixes - `swarm.containers` matches `swarm.containers.manager`.
    """

    #: the name a declaration uses
    name: str
    #: what the CLI calls it in front of a human
    label: str
    #: module prefixes that provide it; reaching any one of them is the claim
    modules: tuple[str, ...]
    #: why this is the module that provides it, for whoever edits the tuple
    provided_by: str


#: The five capabilities a runner is described by. Order is the order the CLI
#: table prints, which is roughly worst-to-lose first.
CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        name="sandbox",
        label="container sandbox",
        modules=("swarm.containers",),
        provided_by=(
            "`containers/` is the only thing that puts model-written code in a "
            "container on an internal network. Running it on the host is not a "
            "weaker sandbox, it is none."
        ),
    ),
    Capability(
        name="egress_policy",
        label="egress policy",
        modules=("swarm.security",),
        provided_by=(
            "`security.py` owns the allowlist, the proxy configuration and the "
            "worker's create flags - the whole of what a worker may reach."
        ),
    ),
    Capability(
        name="quality_gate",
        label="pull request + CI",
        modules=("swarm.worker.pr", "swarm.orchestrator.checks"),
        provided_by=(
            "`worker/pr.py` publishes the work as a pull request and "
            "`orchestrator/checks.py` is the gate that judges it on neutral "
            "ground. A runner that merges a worktree locally has neither."
        ),
    ),
    Capability(
        name="merge_queue",
        label="merge queue",
        modules=("swarm.orchestrator.mergeability",),
        provided_by=(
            "`orchestrator/mergeability.py` decides what may land, updates "
            "branches under a budget, and serialises the merges."
        ),
    ),
    Capability(
        name="tracker",
        label="issue tracker",
        modules=("swarm.github",),
        provided_by=(
            "the tracker path: `github/` today, and whatever ADR 0001's MCP "
            "integration replaces it with in #151. Durable state that survives "
            "the process, which a run holding its plan in memory does not have."
        ),
    ),
)


@dataclass(frozen=True)
class Runner:
    """A top-level entry point owning a complete execution model.

    `entrypoint` is `module:function` - the function that owns the run, not the
    argument parsing in front of it. The test walks the call graph from there,
    through the functions the entry point calls in its own module, and reads
    which modules those name. That is deliberately narrower than the transitive
    closure of every import; `tests/test_framework_boundary.py` explains why
    the closure answers a different question.

    `wiring` names any *further* modules the runner composes itself out of, for
    a runner that does not fit in one file. A runner whose sandbox is built in
    a helper module lists that module here, or the test will read it as not
    having one.
    """

    #: the `swarm <command>` that starts it
    command: str
    #: `module:function` - where the run begins
    entrypoint: str
    #: capability names from `CAPABILITIES`; may be empty, but must be stated
    capabilities: frozenset[str]
    #: one line, in capability terms rather than framework or convenience terms
    summary: str
    #: modules composing this runner beyond its entry point's own module
    wiring: tuple[str, ...] = ()
    #: what a reader should know about the gap, when there is one
    gap: str = ""

    @property
    def module(self) -> str:
        return self.entrypoint.split(":", 1)[0]

    @property
    def function(self) -> str:
        return self.entrypoint.split(":", 1)[1]

    @property
    def owns(self) -> tuple[str, ...]:
        """Every module whose functions count as this runner's own code."""
        return (self.module, *self.wiring)

    def provides(self, name: str) -> bool:
        return name in self.capabilities


#: Every runner, declared. A new entry point that is not in here fails the
#: suite - an undeclared runner is the failure mode this exists to catch,
#: because an undeclared runner is exactly how `swarm local` arrived.
RUNNERS: tuple[Runner, ...] = (
    Runner(
        command="run",
        entrypoint="swarm.cli:_run",
        capabilities=frozenset(
            {"sandbox", "egress_policy", "quality_gate", "merge_queue", "tracker"}
        ),
        summary=(
            "workers in containers on a filtered network, work published as "
            "pull requests and judged by CI, merges through the queue, plan "
            "and state on the issue tracker"
        ),
    ),
    Runner(
        command="local",
        entrypoint="swarm.cli:_local",
        capabilities=frozenset(),
        summary=(
            "the v1 graph against a checkout on this machine: worktrees, host "
            "model calls, ordinary merges, and none of the defences above"
        ),
        # Not an omission, and not a thing this declaration is waiting to have
        # filled in. `nodes/verifier.py` runs the verify command with
        # `shell=True` on the host, in a worktree of code a model has just
        # written; there is no container to put it in, no proxy in front of it,
        # no pull request to gate it and no queue to merge it through. ADR 0003
        # leaves whether this runner stays supported to the maintainer, and
        # #162 states the gap where the runner is chosen. This records it where
        # a test can hold it still: closing the gap means writing the code
        # first, and the declaration follows the code rather than leading it.
        gap=(
            "no sandbox, no egress policy, no pull-request or CI gate and no "
            "merge queue: model-written code runs unconfined on the host "
            "(ADR 0003, decision 4; docs/security.md section 7)"
        ),
    ),
)


@dataclass(frozen=True)
class NotARunner:
    """A subcommand that is not an entry point for an execution model.

    Every subcommand is classified as one or the other, so that adding a third
    runner cannot be done without meeting this file. Listing something here is
    a claim too: the suite checks that a non-runner composes no capability,
    because a thing that builds containers and merges pull requests is a runner
    whatever its help string says.
    """

    command: str
    entrypoint: str
    why: str


#: The subcommands that read or check rather than execute.
NOT_RUNNERS: tuple[NotARunner, ...] = (
    NotARunner(
        command="reset",
        entrypoint="swarm.cli:_reset",
        why=(
            "one write to apiary's own task store; it dispatches nothing. The "
            "gesture ADR 0002 quotes - a human giving a stuck task another go - "
            "which ADR 0005 moved out of the issue marker and into the store"
        ),
    ),
    NotARunner(
        command="doctor",
        entrypoint="swarm.cli:_doctor",
        why="a read-only preflight; it checks preconditions and runs nothing",
    ),
    NotARunner(
        command="runs",
        entrypoint="swarm.cli:_runs",
        why="prints the run artifacts a previous run left behind",
    ),
    NotARunner(
        command="show",
        entrypoint="swarm.cli:_show",
        why="prints one run's summary from its artifacts",
    ),
    NotARunner(
        command="console",
        entrypoint="swarm.cli:_console",
        why=(
            "serves captured prompts and fires single model calls by hand; it "
            "dispatches no work and owns no execution model"
        ),
    ),
)


def capability(name: str) -> Capability:
    for entry in CAPABILITIES:
        if entry.name == name:
            return entry
    raise KeyError(
        f"{name!r} is not a capability. Known: "
        f"{', '.join(c.name for c in CAPABILITIES)}. Adding one means adding "
        "the modules that provide it, so the claim stays checkable."
    )


def runner(command: str) -> Runner:
    for entry in RUNNERS:
        if entry.command == command:
            return entry
    raise KeyError(
        f"no runner declares `swarm {command}`. Declare it in "
        "src/swarm/runners.py (ADR 0003, decision 4)."
    )


def capability_table(left: str, right: str, *, indent: str = "    ") -> str:
    """The comparison two runners are chosen between, rendered.

    `swarm local --help` prints this, which is the point: the table a human
    reads and the set a test asserts are one structure. A row that says "yes"
    is a row the suite has checked against the import graph.
    """
    lo, ro = runner(left), runner(right)
    names = (f"swarm {lo.command}", f"swarm {ro.command}")
    label_width = max(len(c.label) for c in CAPABILITIES) + 2
    left_width = max(len(names[0]), 3) + 4

    lines = [f"{indent}{'':{label_width}}{names[0]:{left_width}}{names[1]}"]
    for entry in CAPABILITIES:
        yes_no = ("yes" if lo.provides(entry.name) else "no",
                  "yes" if ro.provides(entry.name) else "no")
        lines.append(
            f"{indent}{entry.label:{label_width}}{yes_no[0]:{left_width}}{yes_no[1]}"
        )
    return "\n".join(lines)
