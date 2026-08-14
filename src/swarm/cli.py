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

from .config import SETTINGS, ConfigError
from .doctor import DEFAULT_CI_REF
from .doctor import main as doctor_main
from .doctor import preflight
from .github.client import GitHubClient, GitHubError
from .github.ledger import DEFAULT_STACK, KNOWN_STACKS, LedgerError
from .github.readiness import DependencyCycleError, ReadinessError, apply_readiness
from .greenfield.bootstrap import Bootstrap
from .greenfield.provision import provision
from .greenfield.scaffold import ScaffoldedPlan
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


def _show(args: argparse.Namespace) -> int:
    """One run. An id with no directory raises `ArtifactsError` naming it."""
    print(show_text(load_run(args.run_id, args.root)))
    return 0


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

    # `ScaffoldedPlan`, not `ProvisionPlan`: it carries a generated project into
    # the *initial* commit, so the repository the first worker clones has a real
    # test suite and its workflow's required check runs that suite rather than
    # `test -f README.md`. A plain plan produces a repo with nothing to verify,
    # and then hands the planner a command with nothing to run.
    #
    # It also chooses the stack, which is what declines a prompt naming a
    # toolchain the worker image does not carry - at this point, before anything
    # irreversible has happened.
    # Which stack the prompt implies, decided before anything is created - the
    # generated CI workflow needs it (#96) and so does the image the first
    # worker runs in (#99). `--stack` is the operator overriding the model,
    # which is the right precedence: they know what the repository is for.
    bootstrap = Bootstrap.for_prompt(args.new, stack=args.stack)
    # The refusal, inverted: this host either has an image for the stack the
    # prompt implies or it does not, and the answer is checked here - the last
    # moment it is free. Afterwards there is a real repository with a URL, a
    # branch ruleset and a backlog, and a refusal is something a human has to
    # delete rather than something they read.
    _refuse_unrunnable_stacks(bootstrap.stack)
    plan = ScaffoldedPlan.for_prompt(
        args.new,
        owner=args.owner,
        name=args.name,
        private=not args.public,
        # So the generated workflow sets up the right toolchain (#96). The
        # scaffold's own stack choice is still Python-only; #104 deletes it.
        stack=bootstrap.stack,
        # Only when asked: `for_prompt` defaults this to the stack's command,
        # and passing None would override it with nothing.
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
