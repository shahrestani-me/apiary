"""Start building: the plan on the screen becomes a repository and its issues.

## The gap this closes

The planner tab already renders a decomposition an operator can read task by
task. Until now the only thing to do with it was to retype the objective into
the swarm tab, and that path — `console_runs.build_argv`, which emits
`swarm run --new <objective> --owner ... --yes` — provisions the repository and
then **asks the model again** (`cli._run` calls `plan_node` on any repository
whose ledger is empty, which a repository created one second earlier always
is). So the plan that ran was never the plan that was read. It could not be:
the operator's approval travelled as an objective string, and an objective
string is exactly the input whose answer varies.

An approval that does not bind is worse than no approval step at all, because
it teaches the operator to trust a screen that is not load-bearing. This module
exists so that the plan shown is the plan written, and it earns that claim the
only way it can be earned — by never reaching a model. The tasks written here
are rebuilt from `Job.result`, which is the very payload the browser rendered;
there is no second inference, and `tests/test_console_build.py` fails the build
if a model server is so much as constructed.

## Why the whole decision happens before anything is created

`normalise` and `order_drafts` are pure, and both can refuse: a task with no
usable id or no files cannot become an issue, and a dependency ring cannot
become an ordering. `write_plan` runs them too, and refuses just as hard — but
by then `provision` has created a real repository with a URL, a ruleset and a
protected branch, and a refusal is something a human has to go and delete. So
they run here first, on the same tasks, and a plan that cannot be written is
refused while the network has still not been touched. The duplicated work is
one pass over a list of dictionaries; the thing it buys is that "nothing is
created when the plan is bad" is true rather than nearly true.

The same argument orders the two preflights. The token check (`console_runs`'s
own, reused rather than restated, so the operator reads one sentence about
`APIARY_PROVISION_TOKEN` and not two that disagree) and `doctor.preflight` both
answer questions that are free before provisioning and expensive after it.

## Why rejections are a field and not a log line

`normalise` rejecting a task is the failure mode this console is worst placed
to survive quietly: the operator watched eight tasks appear on the planner tab,
and if six issues are created they will read that as six tasks having been
planned. `BuildReport.rejected` is therefore a first-class part of the answer,
rendered on the page beside the issues that were written, carrying `normalise`'s
own reason for each. Silence is the failing direction here, not noise.

## The bootstrap task, and why it is a checkbox rather than an omission

Every greenfield repository this system has ever made leads with #101's
project-scaffold issue: `cli._target` hands `plan_node` a `Bootstrap`, and
`with_bootstrap` then blocks every other task on it. That is not decoration.
Every task the model planned edits files that do not exist yet, so without the
scaffold "the dispatcher would run three workers against an empty repository in
the first cycle, each generating its own idea of the project" - `with_bootstrap`'s
own words. A button that provisioned a repository and wrote a backlog whose
first cycle behaves that way would be a worse outcome than the retyping it
replaces.

The tension is real, though, and it is this module's central one: adding the
scaffold means writing an issue the operator did not read, and rewriting the
`## Blocked by` of every issue they did. The resolution is to *show* it rather
than to drop it. It is a checkbox, on by default, whose label states exactly
what it writes and what it changes about the rest - so the plan shown is still
the plan written, with the one addition sitting under the operator's cursor
when they press the button. Unchecked, they get the decomposition and nothing
else, which is the right answer when the plan already contains its own setup
task.

No model is asked either way: `Bootstrap.for_prompt` consults `choose_stack`
only when the stack is blank, and the stack is a required field here precisely
because there is nothing to ask.

## What this module does not do

It does not run anything, and after #130 that is a statement about this module
rather than about the button. Provisioning ends at a repository with a backlog;
`Console._build` then hands that repository to `SwarmRuns`, which supervises the
reconcile loop exactly as the swarm tab does - a child process running the real
`swarm run`, stopped with the `SIGINT` that lets `cli._loop` dispose its
containers. Keeping the two apart is what lets a run that will not start be
reported *on a build that succeeded*: the repository and its issues are real
either way, and a build that called itself failed because the loop could not
begin would be describing the wrong thing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .console_runs import PROVISION_TOKEN_ENV, SwarmRunError, assert_tokens
from .github.ledger import DEFAULT_STACK, KNOWN_STACKS
from .state import Plan

__all__ = [
    "BUILD_SITE",
    "BUILD_SITE_KEY",
    "BuildError",
    "BuildReport",
    "Builder",
    "RejectedTask",
    "WrittenIssue",
    "plan_from_result",
]

#: The `Job.site` a build claims. It is a site key like the model-call ones
#: purely so that the console's single-flight latch — one `_running`, one lock
#: — covers builds without growing a second concept; nothing that iterates
#: `SITES` ever sees it, because it is not in `SITES`.
BUILD_SITE_KEY = "build"


class BuildError(ValueError):
    """A refusal an operator can act on, in the console's `{error, fix}` shape.

    `checks` carries `doctor`'s failing checks through unchanged. Doctor is
    built so that a failing check cannot exist without naming its remedy
    (`Check.__post_init__` refuses to construct one), and re-describing those
    remedies here would be inventing a second, staler set of instructions for
    problems `doctor` already explains correctly.
    """

    def __init__(
        self,
        message: str,
        *,
        fix: str = "",
        checks: Sequence[Mapping[str, str]] = (),
    ) -> None:
        super().__init__(message)
        self.fix = fix
        self.checks = tuple(dict(check) for check in checks)


# --------------------------------------------------------------------------
# What a build produces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WrittenIssue:
    """One task that became one issue, and where to click to read it."""

    number: int
    task_id: str
    title: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {"number": self.number, "task_id": self.task_id,
                "title": self.title, "url": self.url}


@dataclass(frozen=True)
class RejectedTask:
    """One task the planner emitted that could not become an issue.

    `reason` is `normalise`'s, verbatim: "no usable task id", "a second task
    claims this id", "[Files] lists no files". Those sentences name the field
    the operator has to look at, which a paraphrase would not.
    """

    task_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "reason": self.reason}


@dataclass(frozen=True)
class BuildReport:
    """Everything that now exists, in the shape the page renders.

    `rejected` and `warnings` are beside `issues` rather than under it because
    a build that wrote six issues out of eight tasks is not a success with a
    footnote — it is two silently missing pieces of work unless the page says
    so.
    """

    repo: str
    html_url: str
    default_branch: str
    verify_command: str
    stack: str
    issues: tuple[WrittenIssue, ...] = ()
    rejected: tuple[RejectedTask, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "html_url": self.html_url,
            "default_branch": self.default_branch,
            "verify_command": self.verify_command,
            "stack": self.stack,
            "issues": [issue.to_dict() for issue in self.issues],
            "rejected": [task.to_dict() for task in self.rejected],
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        head = f"{self.repo}: {len(self.issues)} issue(s) written"
        return f"{head}, {len(self.rejected)} task(s) rejected" if self.rejected else head


# --------------------------------------------------------------------------
# The form
# --------------------------------------------------------------------------

#: Served under its own key on `/sites`, exactly as `SWARM_SITE` is and for the
#: same reason: nothing that iterates model-call sites may pick up a form with
#: no prompt behind it.
#:
#: The stack is **required** here although it is optional on the swarm tab, and
#: that difference is the whole ticket in one field. There, `Bootstrap.for_prompt`
#: asks `choose_stack` when the operator leaves it blank; here there is no
#: model to ask, and the answer is needed twice before anything is created —
#: `doctor.preflight` checks the worker image for it, and the generated CI
#: workflow sets up its toolchain.
BUILD_SITE: dict[str, Any] = {
    "key": "build",
    "kind": "build",
    "label": "Start building — this plan, as a repository, and then a running swarm",
    "blurb": (
        "Creates a new GitHub repository, writes the plan above into it as issues — "
        "the tasks on this screen, with these ids, these goals and these files — and then "
        "starts the swarm on it: one worker per ready issue, a pull request per task, "
        "cycle after cycle until the objective is met, the cap below is hit, or you press "
        "Stop. The model is not asked to plan again. Nothing is created until the token and "
        "image checks pass, and any task that cannot become an issue is listed here rather "
        "than dropped. The only issue written that is not on this screen is the project "
        "scaffold, and only while the box below is ticked."
    ),
    "fields": [
        {"name": "owner", "label": "Owner — the GitHub account or organisation to create it under",
         "kind": "text", "placeholder": "shahrestani-me", "value": ""},
        {"name": "name", "label": "Repository name (optional; derived from the objective when blank)",
         "kind": "text", "placeholder": "expense-tracker", "value": ""},
        {"name": "stack", "label": f"Stack — one of: {', '.join(sorted(KNOWN_STACKS))}",
         "kind": "text", "placeholder": DEFAULT_STACK, "value": DEFAULT_STACK},
        {"name": "public", "label": "Create it public — a free GitHub plan cannot put branch protection on a private repository",
         "kind": "check", "value": "1"},
        {"name": "bootstrap", "label": "Write the project-scaffold issue first — one extra issue that creates the initial project, with every task above blocked on it. Without it the first cycle dispatches every task against an empty repository, each worker inventing its own project.",
         "kind": "check", "value": "1"},
        {"name": "verify", "label": "Verify command (optional; default: the placeholder gate, which the first pull request replaces)",
         "kind": "text", "placeholder": "python -m pytest -q", "value": ""},
        # The two fields that belong to the *run*, not to the repository, and
        # they are here rather than on a second form because #130 made this one
        # button do both. They are the swarm tab's own two, verbatim: a build
        # that offered a different cap or a different merge policy from the one
        # `swarm run` offers would be a second set of defaults, which is the
        # thing the ticket says not to grow.
        {"name": "max_cycles", "label": "Stop after this many cycles (optional; default: until the objective is met)",
         "kind": "text", "placeholder": "", "value": ""},
        {"name": "auto_merge", "label": "Merge green pull requests automatically (admin override) — unchecked, every PR waits for a human",
         "kind": "check", "value": "1"},
    ],
}


def plan_from_result(result: Any) -> Plan:
    """Rebuild the plan the browser rendered, from the payload it rendered.

    Deliberately not "read a `Plan` the console kept aside": the claim being
    made is about what was *on the screen*, and the honest way to make it is to
    reconstruct from the same bytes the screen was drawn from. `_plan_run`'s
    task dictionaries carry every field `PlannedTask` needs, so this is a
    validation rather than a translation — a shape the console never produced
    is a refusal here, not a half-populated plan.
    """
    if not isinstance(result, Mapping):
        raise BuildError(
            "that call has no plan on it",
            fix="run the planner and press Start building on its answer",
        )
    tasks = result.get("tasks")
    if not isinstance(tasks, (list, tuple)) or not tasks:
        raise BuildError(
            "that plan has no tasks in it",
            fix="run the planner again; an empty decomposition is not something to build",
        )
    try:
        return Plan.model_validate({"tasks": list(tasks),
                                    "reasoning": str(result.get("reasoning") or "")})
    except Exception as exc:  # noqa: BLE001 - a pydantic error is the operator's answer
        raise BuildError(
            f"that plan is not in a shape that can be written: {exc}",
            fix="run the planner again and press Start building on the fresh answer",
        ) from exc


def _work_client(repo: str) -> Any:
    """A client for the issues, on the work key.

    `provision` authenticates as the boot key and this does not, on purpose:
    writing issues is what `GITHUB_TOKEN` is for, and reaching for the boot key
    because it happens to be in the environment would widen the credential that
    touches the tracker for no reason at all. See docs/security.md.
    """
    from .github.client import GitHubClient

    return GitHubClient.from_env(repo)


# --------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------


@dataclass
class Builder:
    """Provision, then write. Every edge of it is a seam, as `SwarmRuns`' is.

    The seams are where they are so that a test can drive `Console.render` end
    to end and assert the *issue payloads* — which is the acceptance criterion
    that can be checked on a laptop — without a token, a daemon or a socket.
    The one criterion that cannot be checked here is the first, "creates a
    repository": `security.assert_provision_token` accepts only a real
    fine-grained boot key, and minting one is not something a test suite gets
    to do.
    """

    #: `None` rather than the real callables, so importing this module does not
    #: drag in `greenfield` and `doctor` — and, for `env`, so that the value is
    #: read at build time. The console is long-lived, and an operator who
    #: exports the boot key into its shell should be refused for the reason
    #: that is true now, not for a snapshot taken at import.
    provisioner: Callable[..., Any] | None = None
    preflight: Callable[[Sequence[str]], Any] | None = None
    client_for: Callable[[str], Any] = _work_client
    env: Mapping[str, str] | None = None
    out: Callable[[str], None] = print

    def run(self, result: Any, values: Mapping[str, str]) -> BuildReport:
        """The whole action, in the order that makes each refusal free."""
        from .greenfield.bootstrap import Bootstrap
        from .greenfield.provision import PLACEHOLDER_VERIFY, ProvisionPlan
        from .nodes.planner import normalise, order_drafts, with_bootstrap, write_plan

        plan = plan_from_result(result)
        owner, name, stack, verify = _form(values)
        prompt = _prompt_of(values, plan)

        # No model: `Bootstrap.for_prompt` consults `choose_stack` only when the
        # stack is blank, and `_form` guarantees one. That is why the stack is a
        # required field on this form and an optional one on the swarm tab.
        # Absent means **on**, which is the opposite of what `public` does two
        # fields up, and the difference is deliberate. An unticked `public` is
        # the conservative direction: a private repository. An unticked
        # `bootstrap` is the broken one - a backlog whose first cycle dispatches
        # every task against an empty repository. The page always sends this key
        # (`buildValues` emits every field), so absence means a caller that has
        # not heard of the field, and such a caller should get what every other
        # greenfield repository in this system gets.
        scaffold = (
            None if values.get("bootstrap") == ""
            else Bootstrap.for_prompt(prompt, stack=stack).task
        )

        # Pure, and therefore first. A ring or an unwritable task refuses here
        # with nothing created anywhere. `write_plan` reaches the same verdicts
        # on the same tasks a moment later; the point is that this time being
        # wrong costs nothing. The gate string is the placeholder rather than
        # the form's, because the real one is not known until the commit that
        # carries it exists — and `normalise` rejects every task when it is
        # given an empty command, which would make a blank optional field look
        # like eight broken tasks.
        # `with_bootstrap` first, exactly as `write_plan` will do it: the ring
        # check and the rejection list have to be computed over the task set
        # that is actually written, or this pass would bless a plan the real one
        # refuses - which is the failure it exists to prevent.
        tasks = plan.tasks if scaffold is None else list(with_bootstrap(plan.tasks, scaffold))
        drafts, rejected = normalise(
            tasks, verify=verify or PLACEHOLDER_VERIFY, stack=stack
        )
        # The operator's own drafts, with the scaffold discounted. A scaffold is
        # always writable, so counting it here would let a plan whose every real
        # task was rejected provision a repository containing nothing but its
        # own setup issue - which is the empty backlog this guard exists to
        # refuse, wearing a disguise.
        theirs = [d for d in drafts if scaffold is None or d.task_id != scaffold.id]
        if not theirs:
            raise BuildError(
                "not one task in that plan can be written as an issue: "
                + "; ".join(f"{action.task_id} - {action.reason}" for action in rejected),
                fix="run the planner again; a repository with no work in it "
                    "is not worth creating",
            )
        try:
            order_drafts(drafts)
        except Exception as exc:  # noqa: BLE001 - PlanError, before anything exists
            raise BuildError(
                f"that plan's dependencies form a ring, so it has no order: {exc}",
                fix="run the planner again; a cycle cannot be written as "
                    "`## Blocked by` refs",
            ) from exc
        titles = {draft.task_id: draft.title for draft in drafts}

        # A blank field means "whatever the plan defaults to", which is the
        # placeholder gate - passing an empty string instead would be refused by
        # `ProvisionPlan.__post_init__`, correctly and unhelpfully.
        chosen: dict[str, Any] = {"verify_command": verify} if verify else {}
        try:
            provision_plan = ProvisionPlan.for_prompt(
                prompt,
                owner=owner,
                name=name or None,
                private=values.get("public") != "1",
                stack=stack,
                **chosen,
            )
        except ValueError as exc:
            # `ProvisionPlan` validates the owner, the name and the command in
            # `__post_init__`, and an underivable name in `for_prompt`. Rendered
            # as this page's one-line refusal rather than as a 500 with a
            # traceback where the fix should be.
            raise BuildError(str(exc),
                             fix="type a repository name explicitly") from exc

        self._preflight(stack)
        report = self._provision(provision_plan)

        # `report.verify_command`, not the form's: what every `## Verify` must
        # agree with is the command in the commit that now exists. That is
        # `cli._target`'s reasoning and the same trap either way — a `## Verify`
        # disagreeing with the required status check is a task that was red
        # before a worker touched it.
        try:
            written = write_plan(
                self.client_for(report.repo),
                plan,
                verify=report.verify_command,
                stack=stack,
                # `write_plan` applies `with_bootstrap` itself. The plan handed
                # over is still the operator's, unmodified - which keeps the
                # scaffold a documented addition rather than an edit to what
                # they read.
                bootstrap=scaffold,
            )
        except Exception as exc:  # noqa: BLE001 - the repository outlives this
            # The one failure that must not arrive as a traceback. By this line
            # the repository is real, and a refusal that did not name it would
            # leave the operator with something they cannot find and did not
            # ask for - the same reasoning `provision` gives for reporting what
            # exists rather than pretending an abort undid it. Pressing the
            # button again would create a *second* repository, so the fix says
            # what to do with the first.
            raise BuildError(
                f"{report.repo} was created at {report.html_url}, but its issues "
                f"could not be written: {type(exc).__name__}: {exc}",
                fix=f"the repository exists and is empty of work - fix the cause and "
                    f"run `swarm run --repo {report.repo} --objective ...`, or delete it "
                    f"before pressing Start building again; a GITHUB_TOKEN without "
                    f"`issues: write` fails exactly here",
            ) from exc
        issues = tuple(
            WrittenIssue(
                number=action.number,
                task_id=action.task_id,
                title=titles.get(action.task_id, action.task_id),
                # Built from the repository slug the provisioner reported, never
                # from anything the model wrote: task ids and goals are model
                # output and have no business being interpolated into a URL.
                url=f"https://github.com/{report.repo}/issues/{action.number}",
            )
            for action in written.created
            if action.number is not None
        )
        return BuildReport(
            repo=report.repo,
            html_url=report.html_url,
            default_branch=report.default_branch,
            verify_command=report.verify_command,
            stack=stack,
            issues=issues,
            # `write_plan`'s rejections, not the pre-pass's. Both reach the same
            # verdicts on the same tasks, and the authority should be the pass
            # that actually wrote the issues.
            rejected=tuple(
                RejectedTask(action.task_id, action.reason) for action in written.rejected
            ),
            warnings=tuple(written.warnings),
        )

    # -- the two refusals that must land before anything exists -----------

    def _preflight(self, stack: str) -> None:
        """Tokens, then Docker and the image. Neither writes anything.

        The token half is `console_runs`' own check, called rather than
        restated: an operator who has hit it once from the swarm tab should
        read the same sentence and the same fix here, and two descriptions of
        one missing variable is how a fix goes stale.
        """
        from . import doctor

        try:
            assert_tokens(os.environ if self.env is None else self.env, greenfield=True)
        except SwarmRunError as exc:
            raise BuildError(str(exc), fix=exc.fix) from exc

        diagnosis = (self.preflight or doctor.preflight)([stack])
        if diagnosis.ok:
            return
        failures = diagnosis.failures
        raise BuildError(
            "preflight refused: "
            + "; ".join(f"{check.name} - {check.detail}" for check in failures),
            # The first failure's fix as the headline, every failure's in
            # `checks`: the `{error, fix}` shape every other refusal on this
            # page uses has room for one, and the first is the one to act on —
            # a daemon that is down is *why* the image check could not run.
            fix=failures[0].fix,
            checks=[{"name": c.name, "detail": c.detail, "fix": c.fix} for c in failures],
        )

    def _provision(self, plan: Any) -> Any:
        """`assume_yes`, because the operator pressing the button is the consent.

        There is no terminal for `confirm_on_terminal` to ask on, and a console
        that fell through to `_refuse_unattended` would refuse every build with
        a message about stdin. Same reasoning `build_argv` writes down for the
        `--yes` it passes.
        """
        from .greenfield.provision import ProvisionAborted, ProvisionError, provision

        try:
            return (self.provisioner or provision)(plan, assume_yes=True, out=self.out)
        except ProvisionAborted as exc:
            raise BuildError(str(exc), fix="press Start building again") from exc
        except ProvisionError as exc:
            # The repository may be half-created by now — `provision` says so
            # itself and names what exists — so the fix points at the thing to
            # go and look at, rather than at "try again", which would create a
            # second one.
            raise BuildError(
                f"the repository could not be created: {exc}",
                fix=f"look for a partly-created repository under that account before "
                    f"firing again; a {PROVISION_TOKEN_ENV} without `administration` "
                    f"fails exactly here",
            ) from exc


def _prompt_of(values: Mapping[str, str], plan: Plan) -> str:
    """What the repository is generated from: the objective the plan came from.

    Falls back to the plan's own reasoning, then to its first goal, because
    `ProvisionPlan` will not build without a prompt and a build refused for a
    missing hidden field would be unfixable from the page.
    """
    for candidate in ((values.get("objective") or ""), plan.reasoning,
                      plan.tasks[0].goal if plan.tasks else ""):
        if candidate.strip():
            return candidate.strip()
    raise BuildError(
        "there is nothing to generate a repository from",
        fix="type the objective the plan was drafted from",
    )


def _form(values: Mapping[str, str]) -> tuple[str, str, str, str]:
    """The four answers the form must supply, refused one at a time.

    Every refusal here is free — nothing has been created and nothing has been
    read — so they are all checked before the first network call rather than
    discovered by `ProvisionPlan.__post_init__` after `provision` has started.
    """
    owner = (values.get("owner") or "").strip()
    if not owner or "/" in owner:
        raise BuildError(
            f"a repository needs an owner, and {owner!r} is not one",
            fix="a single GitHub login or organisation, e.g. shahrestani-me",
        )
    name = (values.get("name") or "").strip()
    stack = (values.get("stack") or "").strip().casefold() or DEFAULT_STACK
    if stack not in KNOWN_STACKS:
        raise BuildError(
            f"unknown stack {stack!r}",
            fix=f"one of: {', '.join(sorted(KNOWN_STACKS))}",
        )
    # Collapsed rather than merely stripped: `ProvisionPlan` refuses a
    # multi-line command, and a textarea that picked up a trailing newline
    # would be refused for something the operator cannot see.
    verify = " ".join((values.get("verify") or "").split())
    return owner, name, stack, verify
