"""CLI entrypoint.

    swarm run --repo shahrestani-me/apiary --objective "add retry to the client"
    swarm run --new "a CLI that converts markdown tables to CSV" --owner me

Two modes, matching `docs/architecture-v2.md`: point it at a repository that
exists, or hand it a prompt and let #25 create one first.

**There is no `--resume`, and there is no sqlite checkpointer.** Both are gone
on purpose, and they are the same decision:

v1 checkpointed graph state to sqlite after every node so a crashed run could
be resumed by thread id. In v2 the durable state is the issues, and
`docs/architecture-v2.md` is explicit that "on any disagreement, GitHub wins" -
which is only a rule you need if two stores can disagree. A local checkpoint of
the plan is a second copy of the ledger written by a process that is meant to
hold no irreplaceable state, and the failure it invites is the nastiest kind:
the sqlite file says a task is running, GitHub says a human closed it, and
whichever the code happens to read first decides. So this entrypoint compiles
no graph and constructs no checkpointer. (`build_graph`'s `checkpointer`
argument still defaults to `None`; nothing in v2 passes one.)

Resume is therefore not a flag, it is the default. Re-invoking the same command
mints a new run id and adopts the issues the dead process left behind - see
`run.py`, which records why the id is new rather than reused. Nothing needs to
be remembered locally between invocations for that to work, which is the point.

**The verify command is decided here, and every issue inherits it.** A task's
`## Verify` is the repo-wide command, and this is the only place that knows
which repository is being worked in: `--new` takes the string the scaffold just
committed - the same one the generated workflow's required check runs - and
`--repo` takes `--verify`, falling back to `SETTINGS.verify_command`
(`SWARM_VERIFY`). It is passed down to `plan_node` explicitly rather than left
to default there, so the command in the CI workflow and the command in every
issue are one string that came from one place.

Nothing infers it by reading the target repository's CI. A workflow with a
matrix, four setup steps and a cache has no single line to lift, and a command
inferred wrong is a gate that was red before a worker touched the task - which
is exactly the failure this wiring exists to prevent. An existing repository
whose command is not the default says so with `--verify`, which is a sentence
the operator writes rather than one the swarm guesses.

What this command does today is mint the identity, attach to the ledger and run
the one reconcile step that exists (readiness, #11). Dispatch is #21 and
planning onto GitHub is #10; until those land it says so on stderr rather than
implying a swarm ran.

**Three more subcommands, and no logic behind any of them.** `swarm doctor`,
`swarm runs` and `swarm show <run-id>` are the preflight and the two artifact
readers, which existed as functions long before they were reachable by typing.
`doctor.py:62-64` and `artifacts.py:57-61` both say so and both left the wiring
here deliberately, because it is the module that owns the parser.

So the rule this file follows for them is: **decide nothing.** `doctor.main`
stays the only place that knows what a failed preflight prints and what it
exits with; `runs_text` and `show_text` stay the only places that know what a
run looks like written down. What is added here is argument parsing, one
`reversed()` - `list_runs` is documented oldest-first and a human asking "what
have I run" means the last one - and dispatch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from .capture import LLM_LOG_NAME, Recorder as CaptureRecorder
from .capture import enabled as capture_enabled
from .capture import set_recorder as capture_install
from .config import SETTINGS, ConfigError
from .console import DEFAULT_HOST as CONSOLE_HOST
from .console import DEFAULT_PORT as CONSOLE_PORT
from .doctor import DEFAULT_CI_REF
from .doctor import main as doctor_main
from .doctor import preflight
from .github.client import GitHubClient, GitHubError
from .github.ledger import DEFAULT_STACK, KNOWN_STACKS, LedgerError
from .github.readiness import DependencyCycleError, ReadinessError, apply_readiness
from .greenfield.bootstrap import Bootstrap
from .greenfield.provision import ProvisionPlan, provision
from .greenfield.scaffold import UnsupportedStack
from .security import EgressPolicy, worker_create_flags
from .artifacts import (
    ArtifactsError,
    RunArtifacts,
    list_runs,
    load_run,
    runs_text,
    show_text,
)
from .nodes.planner import plan_node
from .run import Attachment, RunError, start_run
from .runners import capability_table


# `swarm local`'s help is the place a user chooses this runner, so it is the
# place its missing capabilities are named. The command reads as a convenience
# ("a local checkout, no GitHub") and is in fact the one path that executes
# model-written code outside every defence `docs/security.md` argues for; a
# help string that says only "no GitHub" sells a security decision as a
# networking one.
LOCAL_DESCRIPTION = """\
The v1 graph against a local checkout: worktrees instead of issues, merges
instead of pull requests, host Ollama, no GitHub.

It is also the runner with no sandbox. `swarm run` executes model-written code
inside a container on a filtered network, and judges it on neutral ground in
CI. This command runs the verify command through a shell on this machine, in a
worktree of code a model has just written, with this shell's environment and
its network. None of docs/security.md applies to it.\
"""

# The rows are generated from `runners.py`, not written out here, and that is
# `security.py`'s arrangement for `security.py`'s reason: the egress allowlist
# is a predicate and a proxy config file and cannot disagree with itself
# because one is rendered from the other. "Does the local runner have a
# sandbox" is now answered in one place too. Every `yes` below is a claim
# `tests/test_framework_boundary.py` has checked against the import graph, so
# widening one means widening it where the code is - and failing there when
# the code does not back it.
LOCAL_CAPABILITIES = f"""\
what this runner gives up, against `swarm run`:

{capability_table("run", "local")}

Run it only against a repository, and on a machine, you would let an untrusted
script loose in - and pass --unsandboxed to say that you have. See
docs/security.md, "7. The local runner is outside all of it".\
"""

# The one sentence `--unsandboxed` exists to make somebody read. Kept next to
# the help text it repeats so the two cannot drift.
LOCAL_REFUSAL = (
    "swarm local has no sandbox: it runs the verify command through a shell on "
    "this host, in a worktree of model-written code, with this shell's "
    "environment and its network. No container, no egress filter, no CI gate. "
    "Pass --unsandboxed to accept that, or use `swarm run` for the sandboxed "
    "path (docs/security.md, section 7)."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swarm")
    # Subcommands from the start, even with one of them: #29 adds `swarm runs`
    # and `swarm show <run-id>`, and retrofitting subcommands onto a CLI whose
    # only form is `swarm <objective>` is a breaking change twice over.
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the swarm against a repository")
    run.add_argument("--repo", default=None, help="target repository, as owner/name")
    run.add_argument("--objective", default=None, help="what the swarm should accomplish; @path reads it from a file")
    run.add_argument(
        "--new",
        default=None,
        metavar="PROMPT",
        help="create a repository from this brief first, then run against it; @path reads it from a file",
    )
    run.add_argument("--owner", default=None, help="account to create --new under")
    run.add_argument(
        "--name", default=None, help="repository name for --new (default: slugified from the brief, which is rarely what you want once the brief is more than a phrase)"
    )
    run.add_argument(
        "--public", action="store_true", help="create --new public (default: private)"
    )
    run.add_argument("--yes", action="store_true", help="create --new without asking")
    run.add_argument(
        "--verify",
        default=None,
        help=(
            "command every planned issue carries as its ## Verify "
            f"(default: the scaffold's for --new, else {SETTINGS.verify_command!r}); "
            "with --new it is also what the generated CI workflow runs, which means "
            "it replaces the scaffold's own test suite as the gate"
        ),
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="read the ledger and report; write nothing to GitHub",
    )
    run.add_argument(
        "--plan-only",
        action="store_true",
        help="plan and compute readiness, then stop before dispatching anything",
    )
    run.add_argument(
        "--stack",
        default=None,
        choices=sorted(KNOWN_STACKS),
        help=(
            "the stack every planned task targets, written into each issue's "
            "## Stack (default: whatever the planner chooses, falling back to python). "
            "This is how a Node repository is reachable before #103 inverts the refusal"
        ),
    )
    run.add_argument(
        "--base-commit",
        default="",
        help="commit workers branch from (default: the base branch's head)",
    )
    run.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="stop after this many reconcile cycles (default: until the objective is met)",
    )
    run.add_argument(
        "--no-merge",
        action="store_true",
        help="open pull requests but never merge them; leave the review queue to a human",
    )
    run.add_argument(
        "--no-goal-check",
        action="store_true",
        help=(
            "stop when the plan is exhausted rather than asking whether the objective "
            "was met; the run does exactly the tasks that were planned and no more"
        ),
    )

    # `doctor`'s options mirror `doctor.build_parser` deliberately, and
    # `test_the_doctor_subcommand_accepts_every_option_doctor_itself_does` pins
    # that they stay mirrored - a flag added there and not here would leave
    # `python -m swarm.doctor` and `swarm doctor` quietly different commands.
    doctor = sub.add_parser("doctor", help="check every precondition; read-only")
    doctor.add_argument(
        "repo",
        nargs="?",
        default=None,
        help="target repository as owner/name (default: $GITHUB_REPOSITORY)",
    )
    doctor.add_argument(
        "--ci-ref",
        default=DEFAULT_CI_REF,
        help=f"ref whose check runs prove CI exists (default: {DEFAULT_CI_REF})",
    )
    doctor.add_argument(
        "--skip-schema",
        action="store_true",
        help="do not invoke the models; skips the only check that costs inference",
    )

    runs = sub.add_parser("runs", help="list recorded runs, newest first")
    runs.add_argument(
        "--root",
        default=None,
        help="artifacts directory to read (default: $APIARY_ARTIFACTS, else .swarm/runs)",
    )

    show = sub.add_parser("show", help="print one run's summary")
    show.add_argument("run_id", help="the run to print, as printed by `swarm runs`")
    show.add_argument(
        "--root",
        default=None,
        help="artifacts directory to read (default: $APIARY_ARTIFACTS, else .swarm/runs)",
    )

    local = sub.add_parser(
        "local",
        help=("run against a local checkout with no sandbox: model-written code "
              "is executed on this host, unconfined (needs --unsandboxed)"),
        description=LOCAL_DESCRIPTION,
        epilog=LOCAL_CAPABILITIES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    local.add_argument(
        "--repo",
        required=True,
        help="path to a local git repository (created and initialised if missing)",
    )
    local.add_argument(
        "--unsandboxed",
        action="store_true",
        help="accept that this run has no container, no egress filter and no CI "
             "gate; without it the command refuses to start",
    )
    local.add_argument(
        "--objective",
        required=True,
        help="what the swarm should accomplish; @path reads it from a file",
    )
    local.add_argument(
        "--verify",
        default=None,
        help=f"the gate every task must pass (default: {SETTINGS.verify_command!r})",
    )
    local.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help=f"dispatch rounds before giving up (default: {SETTINGS.max_rounds})",
    )

    console = sub.add_parser(
        "console",
        help="fire one model call by hand and read the answer, in a browser",
    )
    console.add_argument(
        "--port",
        type=int,
        default=CONSOLE_PORT,
        help=f"port to serve on (default: {CONSOLE_PORT}); 0 picks a free one",
    )
    console.add_argument(
        "--host",
        default=CONSOLE_HOST,
        help=(
            f"address to bind (default: {CONSOLE_HOST}). Loopback only - the console "
            "serves captured prompts, which are whole files from the repository under test"
        ),
    )
    console.add_argument(
        "--dir",
        default=None,
        help="where captures are written (default: $APIARY_CONSOLE_DIR, else .swarm/console)",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, client: GitHubClient | None = None) -> int:
    """`client` is the test seam, exactly as `provision`'s `target` is.

    The error handling below is shared by every subcommand on purpose: `show`
    of an id nobody ever ran is a typo, and a typo deserves one line naming
    what was not found rather than an `ArtifactsError` traceback.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "runs":
            return _runs(args)
        if args.command == "show":
            return _show(args)
        if args.command == "console":
            return _console(args)
        if args.command == "local":
            return _local(args, parser)
        return _run(args, parser, client=client)
    except DependencyCycleError as exc:
        # Its own branch because the fix is different from every other error
        # here: nothing is broken in the code or the token, somebody wrote a
        # ring of `## Blocked by` refs and no issue in it can ever run.
        print(f"! {exc}", file=sys.stderr)
        print("  break the cycle by editing one issue's ## Blocked by", file=sys.stderr)
        return 1
    except (
        ArtifactsError,
        RunError,
        GitHubError,
        LedgerError,
        ReadinessError,
        ValueError,
    ) as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1


def _refuse_unrunnable_stacks(stack: str | None) -> None:
    """Stop the run when a stack it needs has no image on this host.

    The stacks are the ones this invocation can actually reach: `--stack node`
    means Node, and no flag means whatever the planner may choose - so all of
    them, because refusing to name them would mean discovering a missing image
    on cycle four instead of before the first one.

    A **skipped** check passes. Through the socket proxy the image probe is
    denied by design, and doctor's inability to look is not evidence about the
    host - refusing there would make the containerised orchestrator unstartable.
    """
    diagnosis = preflight([stack] if stack else sorted(KNOWN_STACKS))
    if diagnosis.ok:
        return
    lines = [str(check) for check in diagnosis.failures]
    raise ConfigError(
        "this host cannot run every stack this run may need:\n  "
        + "\n  ".join(lines)
        + "\n  (swarm doctor for the whole preflight; SETUP.md step 4 builds the images)"
    )


def _doctor(args: argparse.Namespace) -> int:
    """Hand the arguments straight back to `doctor.main`, which owns the verdict.

    Re-serialising a namespace into an argv is not the prettiest thing in this
    file, and it is still the right shape: what a failed preflight prints, and
    what it exits with, is a decision `doctor.py` already made and documented,
    and the one thing this ticket must not do is make a second copy of it that
    can drift. Everything here is transport.
    """
    argv = [args.repo] if args.repo else []
    argv += ["--ci-ref", args.ci_ref]
    if args.skip_schema:
        argv.append("--skip-schema")
    return doctor_main(argv)


def _runs(args: argparse.Namespace) -> int:
    """Newest first.

    `list_runs` is documented oldest-first and stays that way - a human asking
    "what have I run" means the last one, and a chronological list is the right
    thing for every caller that is not a terminal. The reversal is this
    subcommand's opinion, so it lives at this call site.
    """
    print(runs_text(tuple(reversed(list_runs(args.root)))))
    return 0


def _console(args: argparse.Namespace) -> int:
    """Serve the console, with capture on whether or not the operator set it.

    `APIARY_CAPTURE` is off by default because a run should not pay for capture
    it did not ask for. The console *is* the asking: every call it fires exists
    to be read afterwards, so it turns capture on in its own process rather
    than making the operator discover a variable before the tool does anything
    useful. `setdefault`, so an explicit `APIARY_CAPTURE=0` still wins.
    """
    from .capture import CAPTURE_ENV
    from .console import serve

    os.environ.setdefault(CAPTURE_ENV, "1")
    serve(
        host=args.host,
        port=args.port,
        directory=Path(args.dir) if args.dir else None,
    )
    return 0


def _show(args: argparse.Namespace) -> int:
    """One run. An id with no directory raises `ArtifactsError` naming it."""
    print(show_text(load_run(args.run_id, args.root)))
    return 0


def ensure_local_repo(path: Path) -> Path:
    """The local run's target, created if missing - the same promise the
    console's repo field makes for GitHub, kept for a directory.

    `ensure_repo` (worktree.py) refuses a directory with no `.git`, and the
    graph's `base_branch` needs a commit to name, so a fresh directory gets
    both: `git init -b main` and one commit holding a README. The identity is
    passed per-command rather than written into the repo's config, because a
    demo repo should not end up impersonating whatever this shell's git
    identity is."""
    import subprocess

    repo = path.expanduser().resolve()
    if (repo / ".git").exists():
        return repo
    repo.mkdir(parents=True, exist_ok=True)
    identity = ["-c", "user.name=swarm", "-c", "user.email=swarm@localhost"]
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    readme = repo / "README.md"
    if not readme.exists():
        readme.write_text(f"# {repo.name}\n\nCreated by `swarm local`.\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", *identity, "commit", "-q", "-m", "init: swarm local target"],
                   cwd=repo, check=True)
    return repo


def _local(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """The v1 graph, end to end, with nothing but this machine.

    Worktree isolation, host model calls, the verify command's exit code as
    the only judge of done, and ordinary merges back to the base branch. No
    ledger: local runs exist for days when GitHub is down or a network is not
    wanted, so their only record is the repository itself.

    `Settings` is frozen and instantiated at import, so the two flags that
    must reach the graph's nodes are written onto the singleton with
    `object.__setattr__` - one documented mutation at the one entrypoint that
    owns the run, not a pattern. The alternative (reloading `config` and every
    module that imported from it) breaks whichever module reloaded first.

    **And it is unsandboxed.** `nodes/verifier.py` runs `verify_command` with
    `shell=True` on this host, in a worktree the model has just written into -
    so the whole of `docs/security.md` (container, egress filter, scrubbed
    verify environment, CI on neutral ground) is absent here, not weakened.
    That is a legitimate thing to want on a machine you are willing to spend,
    and it is not a thing anyone should get by reaching for the convenient
    subcommand: `--unsandboxed` is refused-by-default rather than
    warned-about, because a warning printed after the run has started is read
    once the code has already executed.
    """
    if not args.unsandboxed:
        # A `ConfigError` is a `ValueError`, so `main`'s handler renders this
        # as one line and exit 1, the same as every other precondition.
        raise ConfigError(LOCAL_REFUSAL)

    objective = _text(args.objective, "--objective", parser)
    repo = ensure_local_repo(Path(args.repo))
    object.__setattr__(SETTINGS, "repo_path", str(repo))
    if args.verify:
        object.__setattr__(SETTINGS, "verify_command", args.verify)
    if args.max_rounds:
        object.__setattr__(SETTINGS, "max_rounds", args.max_rounds)

    from .graph import build_graph

    print(f"» local run  repo {repo}  verify: {SETTINGS.verify_command}")
    print("» no GitHub: worktrees instead of issues, merges instead of pull requests")
    print("! unsandboxed: the verify command runs on this host, in a worktree of "
          "model-written code, with no container and no egress filter",
          file=sys.stderr)
    graph = build_graph()
    seen = 0
    final: dict = {}
    try:
        for state in graph.stream({"objective": objective},
                                  config={"recursion_limit": 300},
                                  stream_mode="values"):
            for line in state.get("events", [])[seen:]:
                print(f"  · {line}", flush=True)
            seen = len(state.get("events", []))
            final = state
    except KeyboardInterrupt:
        print("\n! interrupted; worktrees are left for inspection", file=sys.stderr)
        return 130

    tasks = final.get("tasks", {})
    verified = sum(1 for t in tasks.values() if t.get("status") == "verified")
    for task_id, task in sorted(tasks.items()):
        print(f"  {task.get('status', '?'):10} attempts={task.get('attempts', 0)}  {task_id}")
    print(f"» {verified}/{len(tasks)} task(s) verified and merged into the base branch")
    return 0 if tasks and verified == len(tasks) else 1


def _run(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, client: GitHubClient | None
) -> int:
    # Before the repository is resolved, and long before one is *created*: an
    # inverted timeout pair makes every container in the run die at the outer
    # cap with a reason naming the container rather than the gate, and there is
    # no point discovering that after `--new` has provisioned a repo. A
    # `ConfigError` is a `ValueError`, so `main`'s handler already renders it as
    # one line and exit 1. `swarm doctor` deliberately does not refuse - it is
    # the command that explains this one, with a fix hint.
    if conflict := SETTINGS.clock_conflict():
        raise ConfigError(conflict)

    # A stack whose image is not on this host cannot produce anything but
    # infrastructure failures, and `IMAGES=0`/`BUILD=0` mean the orchestrator
    # cannot fix that itself - so it is worth stopping for, before `_target`
    # creates a repository for `--new`. Only the image checks: a preflight that
    # refused a run over an unrelated `github.ci` verdict gets turned off.
    _refuse_unrunnable_stacks(args.stack)

    repo, objective, verify, bootstrap = _target(args, parser, client=client)

    attachment = start_run(
        repo,
        objective,
        source=client if client is not None else repo,
        adopt=not args.dry_run,
    )
    _report_run(attachment, dry_run=args.dry_run)

    source = client if client is not None else repo
    ledger = attachment.ledger

    if not attachment.resumed:
        # An empty ledger is where the planner writes issues. A dry run must not
        # write them, and saying so beats a run that silently does nothing on a
        # fresh repo - which reads as a bug in GitHub rather than as a choice.
        if args.dry_run:
            print("! nothing to attach to; --dry-run writes no plan", file=sys.stderr)
            return 0
        # The command is on this line because it is the one thing about a plan
        # that is chosen before the model is asked, and the one thing whose
        # being wrong makes every issue in the plan unrunnable.
        print(f"» planning from the objective; every task verifies with: {verify}")
        try:
            planned = plan_node(
                {"objective": objective}, source=source, verify=verify,
                stack=args.stack,
                # Only for `--new`. A repository that already exists has its
                # project; generating one over the top of it is the opposite
                # of what the operator asked for.
                bootstrap=bootstrap.task if bootstrap is not None else None,
            )
        except Exception as exc:  # noqa: BLE001 - local model failures are varied
            print(f"! planning failed: {exc}", file=sys.stderr)
            return 1
        for line in planned.get("events", ()):
            print(f"  · {line}")

        # Judge on the re-attach, not on `plan_node`'s own read. Both go to
        # GitHub, and GitHub is authoritative - but a read taken immediately
        # after a write can be served from its conditional cache and describe
        # the world as it was a moment earlier. Failing the run on that would
        # print "the planner produced nothing" directly under a line listing
        # the issues it had just created, which is exactly what it did.
        attachment = start_run(repo, objective, source=source, adopt=True)
        ledger = attachment.ledger
        if not ledger.entries:
            print("! the planner wrote nothing the ledger can read back",
                  file=sys.stderr)
            return 1

    if args.plan_only:
        plan = apply_readiness(source, ledger=ledger, dry_run=args.dry_run)
        for verdict in plan.verdicts:
            print(f"  · {verdict}")
        print()
        print(f"» {plan.summary()}")
        return 0

    return _loop(args, attachment, source=source, verify=verify)


def _loop(args, attachment: Attachment, *, source, verify: str = "") -> int:
    """Run cycles until the objective is met, the cap is hit, or Ctrl-C.

    Everything the loop needs is built here rather than inside `Reconciler`,
    because these are the objects with a lifetime: the fleet holds this run's
    containers, the reaper owns the signal handlers, and both have to be torn
    down whatever ends the run.

    **The policies are read here and nowhere else.** `MergePolicy.from_env` and
    `UpdatePolicy.from_env` both exist to be read once at the call site that
    starts a run - that is what their own docstrings say - and reading them here
    is also what lets the run *print* what it is about to do before it does it.
    A merge policy the operator cannot see is one they cannot have chosen, and
    the admin override's whole justification is that it is explicit.
    """
    from .containers.manager import INHERITED_ENV, ContainerManager
    from .containers.reaper import Reaper
    from .orchestrator.checks import MergePolicy
    from .orchestrator.mergeability import UpdateBudget, UpdatePolicy
    from .orchestrator.recovery import Recovery
    from .containers.manager import StackImages
    from .orchestrator.reconcile import InfrastructurePolicy, Reconciler

    run = attachment.run
    # Resolved once, and every collaborator gets the same object. `source` is a
    # client when a caller injected one and a repository *slug* otherwise, and
    # handing the slug to something expecting a client fails deep inside a
    # cycle - `Recovery` got the string here while `Reconciler` got a client,
    # and the run died three frames into `apply_plan` asking a `str` for
    # `get_issue`. One conversion, at the top, or this recurs per collaborator.
    github = source if not isinstance(source, str) else GitHubClient.from_env(source)

    # A context manager, and used as one: leaving through an exception is
    # recorded rather than swallowed, and the run summary is written on the way
    # out however the loop ends.
    # The stack and the gate go into `run.json` and `summary.json` here, at the
    # one place that knows both: #87's success signal is a query over
    # `.swarm/runs/*/summary.json` returning a non-Python run with merged PRs,
    # and no artifact recorded either until now. `DEFAULT_STACK` until #99
    # makes it vary - written down rather than left blank, because "python"
    # and "nobody recorded it" are different answers to that query.
    artifacts = RunArtifacts.open(run, stack=DEFAULT_STACK, verify=verify)
    # Capture, if the operator asked for it, lands beside the run's other
    # artifacts rather than in the console tree - one directory per run, and
    # `llm.jsonl` next to `events.jsonl`. Announced, because an optional
    # behaviour that nothing prints is one nobody remembers enabling, and this
    # one writes prompts that carry whole files from the repository under test.
    if capture_enabled():
        capture_install(CaptureRecorder.for_run(artifacts.path))
        print(f"» capture: ON -> {artifacts.path / LLM_LOG_NAME}")
    # Merged, not replaced. `ContainerManager` inherits GITHUB_TOKEN and
    # OLLAMA_HOST from this process when `env` is None, and passing an `env`
    # *overrides* that - so handing it only the artifacts variables shipped
    # workers with no token and no model host. The first worker to actually
    # start said "GITHUB_TOKEN is not set" and recorded it.
    inherited = {name: os.environ[name] for name in INHERITED_ENV if os.environ.get(name)}
    # A worker sits on an `internal: true` network with no default route - that
    # is the containment, not an accident - so it reaches GitHub only through
    # the egress proxy, and only if its HTTP client is told to. `proxy_env()`
    # exists for exactly this and nothing called it: the first worker with a
    # token still died on "Temporary failure in name resolution".
    fleet = ContainerManager(
        run=run,
        env={**inherited, **EgressPolicy().proxy_env(), **artifacts.worker_env()},
        extra_flags=[*artifacts.mount_flags(), *worker_create_flags()],
    )
    # `sink` is what puts a disposed container's logs in the run directory.
    # Without it every worker is destroyed with the only account of what it
    # did, and a failed run leaves an empty `logs/` - which is exactly what the
    # first real dispatch produced: a container spawned, disposed, and nothing
    # to say why it had finished in seconds.
    reaper = Reaper(run=run, docker=fleet.docker, sink=artifacts.log_sink)

    merge_policy = MergePolicy.from_env()
    update_policy = UpdatePolicy.from_env()
    infrastructure_policy = InfrastructurePolicy.from_env()
    images = StackImages.from_env()
    if args.no_merge:
        print("» merge policy: --no-merge; every pull request waits for a human")
    else:
        print(f"» {merge_policy.summary()}")
        print(f"» {update_policy.summary()}")
    print(f"» {infrastructure_policy.summary()}")
    print(f"» {images.summary()}")
    if args.no_goal_check:
        print("» goal gate: off; the run stops when the plan is exhausted")

    reconciler = Reconciler(
        run=run,
        client=github,
        # A commit, never a branch: a worker branching from whatever main
        # happened to be would verify against a tree nobody planned for. Empty
        # here meant the worker was dispatched with no base at all.
        base_commit=args.base_commit or github.head_sha(),
        fleet=fleet,
        # The *results* directory, which is the one a worker writes into and the
        # one `load_results` globs. Without it the reconciler never sees a
        # worker's exit code, and §4's retry rows - exit 1 costs an attempt,
        # exit 2 does not - never fire in a real run however well they are
        # tested.
        artifacts=artifacts.results_dir,
        recovery=Recovery(client=github, run=run, dry_run=args.dry_run),
        merge_gate=not args.no_merge,
        merge_policy=merge_policy,
        update_policy=update_policy,
        # Held here, so the per-pull-request update cap bounds the run rather
        # than the cycle: a budget constructed inside the cycle starts over
        # every fifteen seconds and therefore bounds nothing.
        update_budget=UpdateBudget(cap=update_policy.max_update_rounds),
        infrastructure_policy=infrastructure_policy,
        images=images,
        objective=run.objective,
        verify=verify,
        goal_gate=not args.no_goal_check,
        on_cycle=_report_cycle(artifacts),
        # The per-task lifecycle (#141), straight onto `events.jsonl` and
        # therefore through the same redactor as everything else in the run
        # directory. `_report_cycle` still writes `cycle.reconciled`, unchanged:
        # one is the cycle a human is watching, the other is the timeline a
        # reader reconstructs afterwards.
        events=artifacts.event,
        dry_run=args.dry_run,
    )

    with reaper.guard(), artifacts:
        # Containers a previous process left behind hold clones and disk, and
        # their run ids would otherwise be counted against this run's cap.
        swept = reaper.startup()
        if swept.reaped:
            print(f"» reaped {len(swept.reaped)} orphaned container(s)")
        try:
            reports = reconciler.loop(cycles=args.max_cycles)
        except KeyboardInterrupt:
            print("\n! interrupted; containers are being disposed", file=sys.stderr)
            return 130

    print()
    print(f"» artifacts in {artifacts.path}")
    return _report_outcome(reports)


def _report_cycle(artifacts: RunArtifacts):
    """Print each cycle as it happens, and file what the gates decided.

    Printing at the end of the run - which is what this used to do - means an
    operator watching a five-minute cycle sees nothing at all until it is over,
    and the decisions worth watching are exactly the ones that are still
    reversible: a pull request sitting at "no check runs yet, 120s into a 300s
    grace", an issue one attempt from `swarm:failed`, a merge GitHub refused.

    The same detail goes to `events.jsonl`, because the run directory is the
    only account that survives the terminal, and a merge that did not happen
    leaves no other trace: `RunArtifacts` records what the workers wrote and
    what the containers printed, and until now recorded nothing whatsoever
    about the orchestrator's own decisions.
    """

    def report(cycle) -> None:
        print(f"  · {cycle.summary()}")
        # One line per review issue: the check verdict, the mergeability
        # verdict, and what the gate did about it.
        if cycle.checks is not None:
            for outcome in cycle.checks.plan.outcomes:
                print(f"      {outcome}")
            for failure in cycle.checks.failures:
                print(f"    ! merge refused - {failure}", file=sys.stderr)
        if cycle.mergeability is not None:
            for failure in cycle.mergeability.failures:
                print(f"    ! {failure}", file=sys.stderr)
        artifacts.event(
            "cycle.reconciled",
            cycle=cycle.index,
            live=cycle.live,
            summary=cycle.summary(),
            merged=list(cycle.checks.merged) if cycle.checks is not None else [],
            gate=[str(o) for o in cycle.checks.plan.outcomes] if cycle.checks is not None else [],
            failures=(
                [str(f) for f in cycle.checks.failures] if cycle.checks is not None else []
            ),
            verdict=cycle.verdict.summary() if cycle.verdict is not None else "",
            goal=cycle.goal.summary() if cycle.goal is not None else "",
        )

    return report


def _report_outcome(reports) -> int:
    """The last word: was the objective met, and what is left if it was not.

    Non-zero when the run stopped short, because a swarm that gave up is a
    failed command - a shell script chaining `swarm run` must not read "I
    planned three things, abandoned one and stopped" as success. An empty run
    (`--max-cycles 0`) is neither met nor failed and reports nothing.
    """
    if not reports:
        return 0
    last = reports[-1]
    goal = last.goal
    if goal is None:
        # The cap or `until` ended this, not the ledger. Whatever is still open
        # is still open, and the next invocation attaches to it.
        print(f"» stopped after {len(reports)} cycle(s) with {last.live} live issue(s)")
        return 0
    print(f"» {goal.summary()}")
    if goal.met:
        return 0
    for line in goal.assessment.missing:
        print(f"  · still missing: {line}")
    return 1



#: Prefix that reads the value from a file instead of the command line.
#: A good objective is paragraphs, not a phrase - the planner decomposes it,
#: and "a trip planner" gives a 31B model nothing to decompose. Multi-paragraph
#: shell arguments are miserable to quote and impossible to keep in version
#: control, so both text-carrying flags accept `@path` as well as a literal.
FILE_PREFIX = "@"


def _text(value: str | None, flag: str, parser: argparse.ArgumentParser) -> str | None:
    """Resolve `@path` to the file's contents; pass anything else through.

    Errors through `parser` rather than raising, because a missing brief is a
    typo at the command line, and the greenfield path would otherwise discover
    it only after creating a repository.
    """
    if not value or not value.startswith(FILE_PREFIX):
        return value
    path = Path(value[len(FILE_PREFIX):]).expanduser()
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        parser.error(f"{flag} {value}: {exc}")
    if not text:
        parser.error(f"{flag} {value}: the file is empty")
    return text


def _target(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    client: GitHubClient | None,
) -> tuple[str, str, str, Bootstrap | None]:
    """Resolve the two modes down to `(repo, objective, verify, bootstrap)`.

    The greenfield branch creates a repository, so every way of asking for it
    ambiguously is refused *before* that happens rather than after.

    The verify command is resolved here and nowhere else, for the reasons in
    the module docstring. `--repo` has no scaffold to read it off, so it is
    `--verify` or v1's `SETTINGS.verify_command`; being wrong about an existing
    repository costs a red gate on every issue, which is loud, whereas
    inferring one from its CI would be wrong quietly.
    """
    args.new = _text(args.new, "--new", parser)
    args.objective = _text(args.objective, "--objective", parser)

    if args.new and args.repo:
        parser.error("--new creates a repository; it cannot also target --repo")
    if not args.new and not args.repo:
        parser.error("give --repo owner/name, or --new to create one")

    if not args.new:
        if not args.objective:
            parser.error("--repo needs an --objective")
        # No bootstrap: the repository exists and has whatever project it has.
        return args.repo, args.objective, args.verify or SETTINGS.verify_command, None

    if args.objective:
        # The prompt *is* the objective for a greenfield run, and two of them
        # would mean the repo was generated from one description and worked on
        # against another.
        parser.error("--new carries the objective; do not also pass --objective")
    if not args.owner:
        parser.error("--new needs --owner: an account to create the repository under")
    if args.dry_run:
        parser.error("--dry-run cannot create a repository; drop --new or --dry-run")

    # Which stack the prompt implies, decided before anything is created: the
    # generated CI workflow needs it (#96) and so does the image the first
    # worker runs in (#99). `--stack` is the operator overriding the model.
    bootstrap = Bootstrap.for_prompt(args.new, stack=args.stack)
    # The refusal, inverted (#103): this host either has an image for that
    # stack or it does not, and the answer is checked here - the last moment it
    # is free. Afterwards there is a real repository with a URL, a ruleset and a
    # backlog, and a refusal is something a human has to delete.
    _refuse_unrunnable_stacks(bootstrap.stack)

    # A plain `ProvisionPlan`: there is nothing to scaffold any more. #101 made
    # the project the first *issue* of the plan, so the initial commit is the
    # README, the LICENSE and a workflow whose gate is the placeholder - the
    # only command that passes on a repository with no code in it, which it has
    # to be, because the required status check reports on that commit before
    # any worker exists.
    plan = ProvisionPlan.for_prompt(
        args.new,
        owner=args.owner,
        name=args.name,
        private=not args.public,
        # So the generated workflow sets up the right toolchain (#96) and the
        # first worker runs in the right image (#99).
        stack=bootstrap.stack,
        # `--verify` is the operator's, and it is authoritative: #102 does not
        # falsify a command they chose, because an escape hatch that can be
        # refused is not one.
        **({"verify_command": args.verify} if args.verify else {}),
    )
    report = provision(plan, client, assume_yes=args.yes)
    print()
    print(report.summary())
    print()
    # The report's, not the plan's. What the issues must agree with is the
    # command in the commit that now exists.
    return report.repo, args.new, report.verify_command, bootstrap


def _report_run(attachment: Attachment, *, dry_run: bool) -> None:
    run = attachment.run
    note = "  (dry run: nothing will be written)" if dry_run else ""
    print(f"» {run.summary()}")
    print(f"» {attachment.summary()}{note}")
    if attachment.resumed:
        # Naming the previous process is impossible - it left no trace outside
        # GitHub, by design - so name what it left behind instead.
        print("  · resumed from the ledger, not from a local checkpoint; "
              "no issues were replanned")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
