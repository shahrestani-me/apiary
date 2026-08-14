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

What this command does today is mint the identity, attach to the ledger and run
the one reconcile step that exists (readiness, #11). Dispatch is #21 and
planning onto GitHub is #10; until those land it says so on stderr rather than
implying a swarm ran.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .github.client import GitHubClient, GitHubError
from .github.ledger import LedgerError
from .github.readiness import DependencyCycleError, ReadinessError, apply_readiness
from .greenfield.provision import ProvisionPlan, provision
from .security import worker_create_flags
from .artifacts import RunArtifacts
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
    run.add_argument("--objective", default=None, help="what the swarm should accomplish")
    run.add_argument(
        "--new",
        default=None,
        metavar="PROMPT",
        help="create a repository from this prompt first, then run against it",
    )
    run.add_argument("--owner", default=None, help="account to create --new under")
    run.add_argument(
        "--name", default=None, help="repository name for --new (default: from the prompt)"
    )
    run.add_argument(
        "--public", action="store_true", help="create --new public (default: private)"
    )
    run.add_argument("--yes", action="store_true", help="create --new without asking")
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
        "--base-commit",
        default="",
        help="commit workers branch from (default: the base branch's head)",
    )
    run.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="stop after this many reconcile cycles (default: until the ledger finishes)",
    )
    run.add_argument(
        "--no-merge",
        action="store_true",
        help="open pull requests but never merge them; leave the review queue to a human",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, client: GitHubClient | None = None) -> int:
    """`client` is the test seam, exactly as `provision`'s `target` is."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args, parser, client=client)
    except DependencyCycleError as exc:
        # Its own branch because the fix is different from every other error
        # here: nothing is broken in the code or the token, somebody wrote a
        # ring of `## Blocked by` refs and no issue in it can ever run.
        print(f"! {exc}", file=sys.stderr)
        print("  break the cycle by editing one issue's ## Blocked by", file=sys.stderr)
        return 1
    except (RunError, GitHubError, LedgerError, ReadinessError, ValueError) as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1


def _run(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, client: GitHubClient | None
) -> int:
    repo, objective = _target(args, parser, client=client)

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
        print("» planning from the objective")
        try:
            planned = plan_node({"objective": objective}, source=source)
        except Exception as exc:  # noqa: BLE001 - local model failures are varied
            print(f"! planning failed: {exc}", file=sys.stderr)
            return 1
        for line in planned.get("events", ()):
            print(f"  · {line}")
        if not planned.get("tasks"):
            print("! the planner produced nothing to write", file=sys.stderr)
            return 1
        # `plan_node` already re-read the tracker after writing - "on any
        # disagreement, GitHub wins" only means something if nothing keeps a
        # second copy - so re-attach and take the ledger from that read.
        attachment = start_run(repo, objective, source=source, adopt=True)
        ledger = attachment.ledger

    if args.plan_only:
        plan = apply_readiness(source, ledger=ledger, dry_run=args.dry_run)
        for verdict in plan.verdicts:
            print(f"  · {verdict}")
        print()
        print(f"» {plan.summary()}")
        return 0

    return _loop(args, attachment, source=source)


def _loop(args, attachment: Attachment, *, source) -> int:
    """Run cycles until the ledger finishes, the cap is hit, or Ctrl-C.

    Everything the loop needs is built here rather than inside `Reconciler`,
    because these are the objects with a lifetime: the fleet holds this run's
    containers, the reaper owns the signal handlers, and both have to be torn
    down whatever ends the run.
    """
    from .containers.manager import ContainerManager
    from .containers.reaper import Reaper
    from .orchestrator.recovery import Recovery
    from .orchestrator.reconcile import Reconciler

    run = attachment.run
    artifacts = RunArtifacts.open(run)
    fleet = ContainerManager(
        run=run,
        env={**artifacts.worker_env()},
        extra_flags=[*artifacts.mount_flags(), *worker_create_flags()],
    )
    reaper = Reaper(run=run, docker=fleet.docker)
    reconciler = Reconciler(
        run=run,
        client=source if not isinstance(source, str) else GitHubClient.from_env(source),
        base_commit=args.base_commit or "",
        fleet=fleet,
        recovery=Recovery(client=source, run=run, dry_run=args.dry_run),
        merge_gate=not args.no_merge,
        dry_run=args.dry_run,
    )

    with reaper.guard():
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
        finally:
            artifacts.close()

    for report in reports:
        print(f"  · cycle {report.index}: {report.summary()}")
    print()
    print(f"» artifacts in {artifacts.directory}")
    return 0


def _target(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    client: GitHubClient | None,
) -> tuple[str, str]:
    """Resolve the two modes down to one `(repo, objective)` pair.

    The greenfield branch creates a repository, so every way of asking for it
    ambiguously is refused *before* that happens rather than after.
    """
    if args.new and args.repo:
        parser.error("--new creates a repository; it cannot also target --repo")
    if not args.new and not args.repo:
        parser.error("give --repo owner/name, or --new to create one")

    if not args.new:
        if not args.objective:
            parser.error("--repo needs an --objective")
        return args.repo, args.objective

    if args.objective:
        # The prompt *is* the objective for a greenfield run, and two of them
        # would mean the repo was generated from one description and worked on
        # against another.
        parser.error("--new carries the objective; do not also pass --objective")
    if not args.owner:
        parser.error("--new needs --owner: an account to create the repository under")
    if args.dry_run:
        parser.error("--dry-run cannot create a repository; drop --new or --dry-run")

    plan = ProvisionPlan.for_prompt(
        args.new,
        owner=args.owner,
        name=args.name,
        private=not args.public,
    )
    report = provision(plan, client, assume_yes=args.yes)
    print()
    print(report.summary())
    print()
    return report.repo, args.new


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
