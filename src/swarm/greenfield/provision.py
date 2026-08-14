"""Turn a prompt into a repository the swarm can immediately work in.

This is the "prompt in, project out" entry point of
`docs/architecture-v2.md`'s greenfield mode. One command creates the
repository, pushes a single initial commit carrying a README, a LICENSE and a
CI workflow, provisions the `swarm:*` labels (#8) and protects the default
branch:

    GITHUB_TOKEN=... python -m swarm.greenfield.provision \\
        --owner shahrestani-me "a CLI that converts markdown tables to CSV"

Four decisions here are load-bearing.

**Confirm before creating.** Creating a repository is outward-facing and this
tool cannot undo it - the repo carries issues, PRs and a URL other people may
already have seen by the time anyone notices it was a mistake. So `provision`
prints exactly what it is about to create and refuses to proceed without an
explicit go-ahead: a human typing the repository name, or the caller passing
`assume_yes`. There is no third path. When stdin is not a terminal and no flag
was passed, it aborts **before opening a single request**, because "nobody was
watching" is the situation the confirmation exists for, not an exemption from
it.

**Protection the swarm can actually satisfy.** The obvious move is to mirror
this repository's own ruleset, which requires a code-owner review. Applied to a
repository nobody is watching, that would make #23's merge path depend on the
admin override for every single PR, and a demo whose every merge bypasses its
own gate proves nothing about the gate. So the generated ruleset is **required
status checks plus no force-push and no deletion, and no required reviews** -
rules an unattended swarm passes by doing the work, not by being an admin.
Required review stays available as a documented opt-in (`require_reviews`) for
repos that do have a human in the loop, and the generated README says which
shape was chosen and why.

**One commit, built through the git data API.** The workflow has to be *in*
the first commit - "CI passes on the initial commit" is this issue's acceptance
criterion, and a repository whose first commit has no CI has an unverified
commit at the root of its history forever.

That cannot be done from nothing: GitHub answers `POST /git/blobs` on a
repository with no commits with `409: Git Repository is empty`, so there is no
way to write the first commit through the data API. So the repository is
created *with* `auto_init`, which gives it a commit the data API will accept,
and the parentless blob-tree-commit built here then replaces it by
force-updating the ref. The throwaway commit is unreachable and collected; what
remains is a one-commit history with the workflow in it.

**Protection is applied last.** A ruleset with required status checks rejects a
push to the default branch that has not passed them, and creating
`refs/heads/<default>` is a push. Protecting first would block the very commit
the protection exists to guard. Afterwards the swarm never pushes to the
default branch again - it opens PRs and merges them (#17, #23).

Like `labels.py`, this module reaches GitHub through `GitHubClient`'s request
plumbing rather than opening its own: the one-door rule in `client.py` is the
point of that module. Repo creation, the git data API and rulesets are not on
the client's public surface (it exposes the endpoints the *reconcile* loop
dispatches through), and adding them there is outside this issue's file
contract, so the calls below use the client's package-internal helpers and
inherit its retry, pagination and error handling.

There is deliberately **no live smoke test** in this module, where `client.py`
and `labels.py` both have one. Theirs read or converge; this one creates a
repository, and a module whose smoke test litters an account with `apiary-test-
*` repos teaches its readers the wrong habit. `python -m swarm.greenfield.provision`
is the command, and it asks first.
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from ..github.client import GitHubClient, GitHubError, GitHubHTTPError
from ..security import PROVISION_TOKEN_ENV, assert_provision_token
from ..github.labels import SWARM_LABELS, LabelReport, ensure_labels

# The ruleset this module writes. Named, so a second run can recognise its own
# work rather than stacking a duplicate ruleset beside it.
RULESET_NAME = "apiary-swarm"

# The required status check context, the CI job id and the CI job name are all
# this one string on purpose. GitHub matches a required status check against
# the *name* a check run reports, and a required check that never reports is
# indistinguishable from one that is still running: the branch is protected
# against everything, forever. Rename this in one place or not at all.
CHECK_NAME = "verify"

CI_WORKFLOW_PATH = ".github/workflows/ci.yml"

# The verify command written into a repo that has no code in it yet.
#
# It has to pass on the initial commit - that is the acceptance criterion - and
# the obvious `python -m pytest -q` does not: pytest exits 5 when it collects no
# tests, so the first CI run of every generated repo would be red and the
# required check would block the first PR.
#
# Nothing the swarm creates uses this any more: `cli` provisions a
# a plan carrying the stack the prompt implies (#101/#103), whose command is
# is what the planner writes into every issue's `## Verify`. It remains the
# default of a bare `ProvisionPlan` because a plan with no scaffold in it has
# nothing else that could pass, and `python -m swarm.greenfield.provision` can
# still be asked for one.
PLACEHOLDER_VERIFY = "test -f README.md"

# GitHub's own rule for a repository name.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_MAX_SLUG = 48

_LABEL_COUNT = len(SWARM_LABELS)


class ProvisionError(GitHubError):
    """Provisioning could not complete. The message says how far it got."""


class ProvisionAborted(ProvisionError):
    """Nobody said yes, so nothing was created.

    A distinct type because the caller's response differs: an abort is the
    safety rail working, not a fault to report or retry.
    """


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvisionPlan:
    """Everything that will exist afterwards, decided before anything is sent.

    Separating the plan from the act is what makes the confirmation honest: the
    text a human approves is rendered from this object, and `provision` sends
    nothing that is not on it. It is also what lets the whole decision surface -
    names, visibility, file contents, ruleset shape - be tested without a
    transport at all.
    """

    owner: str
    name: str
    prompt: str
    description: str
    license_holder: str
    year: int
    private: bool = True
    # What the default branch is expected to be called. GitHub decides for
    # real - the account has a preference - so `provision` corrects this from
    # the creation response before generating any file that names it.
    default_branch: str = "main"
    verify_command: str = PLACEHOLDER_VERIFY
    require_reviews: bool = False
    #: Which stack's toolchain the generated workflow sets up. A plain string
    #: rather than a `Stack`, because the only thing this module does with it
    #: is look it up in `CI_SETUP` - and importing `scaffold.py` for a key
    #: would make the module that generates the *repository* depend on the one
    #: that generates the *project inside it*, which is the wrong way round.
    #:
    #: Defaulting to python is today's behaviour written down: `choose_stack`
    #: returns `DEFAULT_STACK` unconditionally, so nothing can yet pass
    #: anything else. #99 and #101 are what make this field carry information.
    stack: str = "python"

    def __post_init__(self) -> None:
        if not self.owner or "/" in self.owner:
            raise ValueError(f"owner must be a single login, got {self.owner!r}")
        if not _NAME_RE.match(self.name) or self.name in {".", ".."}:
            raise ValueError(f"{self.name!r} is not a usable repository name")
        if not self.prompt.strip():
            raise ValueError("a greenfield repo needs a prompt to be generated from")
        # Same rule as docs/issue-contract.md §1.3: one command, exit code
        # believed. "Run these in order, stopping on failure" is semantics
        # nobody agreed to - write `&&`.
        if not self.verify_command.strip():
            raise ValueError("verify command must not be empty")
        if len(self.verify_command.strip().splitlines()) > 1:
            raise ValueError("verify command must be a single line; use && to chain")

    @classmethod
    def for_prompt(
        cls,
        prompt: str,
        *,
        owner: str,
        name: str | None = None,
        description: str | None = None,
        license_holder: str | None = None,
        year: int | None = None,
        **overrides: Any,
    ) -> ProvisionPlan:
        """Derive a complete plan from the prompt, before asking anyone anything."""
        derived = name or slugify(prompt)
        if not derived:
            raise ValueError(
                f"could not derive a repository name from {prompt!r}; pass one explicitly"
            )
        return cls(
            owner=owner,
            name=derived,
            prompt=prompt.strip(),
            description=description if description is not None else _one_line(prompt),
            license_holder=license_holder or owner,
            year=year or datetime.date.today().year,
            **overrides,
        )

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def visibility(self) -> str:
        return "private" if self.private else "public"

    def files(self) -> dict[str, str]:
        """The contents of the initial commit, path -> text.

        Three files and no more. A generated project's first commit is not the
        place for a code of conduct, an issue template and a changelog nobody
        has anything to put in; #26 adds the scaffold, and the swarm adds the
        rest by doing the work.
        """
        return {
            "README.md": _readme(self),
            "LICENSE": _mit_license(self.license_holder, self.year),
            # The branch, not a constant: `provision` corrects `default_branch`
            # from the creation response before generating any file, so the
            # push filter names the branch GitHub actually made rather than the
            # one this plan guessed.
            CI_WORKFLOW_PATH: _ci_workflow(
                self.verify_command, stack=self.stack, branch=self.default_branch
            ),
        }

    def rules(self) -> list[dict[str, Any]]:
        """The ruleset rules, in the order they are sent.

        `strict_required_status_checks_policy` is False deliberately. Strict
        means "the branch must be up to date with the base before merging",
        which turns every merge into a rebase of every other open PR - and the
        swarm runs several at once by design. Keeping the base moving under
        open PRs is #34's problem, not something to solve by serialising the
        entire swarm.
        """
        rules: list[dict[str, Any]] = [
            {"type": "deletion"},
            {"type": "non_fast_forward"},
            {
                "type": "required_status_checks",
                "parameters": {
                    "strict_required_status_checks_policy": False,
                    "required_status_checks": [{"context": CHECK_NAME}],
                },
            },
        ]
        if self.require_reviews:
            # The opt-in, for a repo with a human in the loop. An unattended
            # swarm cannot satisfy this rule - it has nobody to approve its
            # PRs - so turning it on means merges wait for a person, which is
            # exactly what someone turning it on is asking for.
            rules.append(
                {
                    "type": "pull_request",
                    "parameters": {
                        "required_approving_review_count": 1,
                        "dismiss_stale_reviews_on_push": True,
                        "require_code_owner_review": False,
                        "require_last_push_approval": False,
                        "required_review_thread_resolution": False,
                    },
                }
            )
        return rules

    def ruleset(self) -> dict[str, Any]:
        return {
            "name": RULESET_NAME,
            "target": "branch",
            "enforcement": "active",
            # `~DEFAULT_BRANCH` rather than the branch name: if the default
            # branch is ever renamed, a ruleset naming the old one silently
            # protects nothing.
            "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
            # No standing bypass, not even for admins - a gate with a permanent
            # exception is not a gate, and the whole point of this ticket is a
            # gate the swarm passes by doing the work. Repairing a broken gate
            # means editing the ruleset, which is visible, rather than
            # force-pushing, which is not.
            "bypass_actors": [],
            "rules": self.rules(),
        }

    def preview(self) -> str:
        """What a human is being asked to approve.

        Every irreversible consequence is on this screen: the exact full name
        that will exist, whether the world can see it, and what will be in it.
        """
        lines = [
            "About to create a new GitHub repository. This cannot be undone from here.",
            "",
            f"  repository   {self.full_name}  ({self.visibility})",
            f"  description  {self.description or '(none)'}",
            f"  prompt       {self.prompt}",
            f"  branch       {self.default_branch}, protected by ruleset {RULESET_NAME!r}",
        ]
        for rule in self.rules():
            lines.append(f"                 - {rule['type']}")
        if not self.require_reviews:
            lines.append("                 - (no required reviews; see the generated README)")
        lines.append(f"  verify       {self.verify_command}")
        lines.append("  initial commit")
        for path, content in sorted(self.files().items()):
            lines.append(f"                 - {path}  ({len(content.encode('utf-8'))} bytes)")
        lines.append(f"  labels       the {_LABEL_COUNT} swarm:* state labels")
        return "\n".join(lines)


def slugify(text: str, *, max_length: int = _MAX_SLUG) -> str:
    """A repository name from free prose. Empty when the prompt has no letters."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    if len(slug) > max_length:
        # Cut on a word boundary so the name reads like words rather than
        # ending mid-syllable; fall back to the hard cut for one long token.
        head = slug[:max_length].rsplit("-", 1)[0]
        slug = head or slug[:max_length]
    return slug.strip("-")


def _one_line(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= 200 else collapsed[:197].rstrip() + "..."


# --------------------------------------------------------------------------
# Confirmation
# --------------------------------------------------------------------------


def confirm_on_terminal(
    plan: ProvisionPlan,
    *,
    ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> bool:
    """Show the plan and require the repository name to be typed back.

    Not "y/n". A y/n prompt in a terminal is muscle memory - the reflex fires
    before the screen is read, which is precisely the failure this guard is
    for. The repository name is not muscle memory, and it is also the one
    string the human most needs to have actually looked at, since it is what
    will exist afterwards.
    """
    out(plan.preview())
    out("")
    try:
        answer = ask(f"Type {plan.name!r} to create it, anything else to abort: ")
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip() == plan.name


def _refuse_unattended(plan: ProvisionPlan) -> bool:
    raise ProvisionAborted(
        f"refusing to create {plan.full_name} unattended: stdin is not a terminal "
        "and no explicit go-ahead was given (pass --yes / assume_yes=True)"
    )


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvisionReport:
    """What now exists, in the shape a caller can log or hand to a human."""

    repo: str
    html_url: str
    default_branch: str
    commit_sha: str
    labels: LabelReport
    protection: tuple[str, ...]
    #: The command the workflow in that commit actually runs. Reported rather
    #: than left for the caller to re-derive from its own copy of the plan: the
    #: planner writes this same string into every issue's `## Verify`, and a
    #: `## Verify` that disagrees with the required status check is a task whose
    #: gate was red before a worker touched it.
    verify_command: str = PLACEHOLDER_VERIFY

    def summary(self) -> str:
        return (
            f"{self.repo}: created at {self.html_url}\n"
            f"  {self.default_branch} @ {self.commit_sha[:7]}, "
            f"protected by {', '.join(self.protection) or 'nothing'}\n"
            f"  verified by {self.verify_command}\n"
            f"  {self.labels.summary()}"
        )


# --------------------------------------------------------------------------
# Provisioning
# --------------------------------------------------------------------------


def _boot_client(repo: str) -> GitHubClient:
    """A client authenticated as the boot key.

    Falls back to nothing: if `APIARY_PROVISION_TOKEN` is unset, this raises
    rather than quietly reaching for `GITHUB_TOKEN`. The fallback is the whole
    bug - it would work on the first try, produce a repository, and leave the
    operator believing one token is enough right up until a worker holds
    `workflows` and can rewrite the CI that judges it.
    """
    token = os.environ.get(PROVISION_TOKEN_ENV)
    assert_provision_token(token)
    return GitHubClient(repo, token or "")


def provision(
    plan: ProvisionPlan,
    target: GitHubClient | None = None,
    *,
    assume_yes: bool = False,
    confirm: Callable[[ProvisionPlan], bool] | None = None,
    out: Callable[[str], None] = print,
) -> ProvisionReport:
    """Create the repository described by `plan`, after someone approves it.

    `target` is a `GitHubClient` for the repo about to exist, or None to build
    one from the environment. Raises `ProvisionAborted` if the go-ahead does
    not arrive, and `ProvisionError` naming what already exists if a later step
    fails - the repo is real from step two onward and pretending otherwise
    would lose its URL.
    """
    if not _approved(plan, assume_yes=assume_yes, confirm=confirm):
        raise ProvisionAborted(f"aborted; {plan.full_name} was not created")

    # Nothing above this line touches the network. That is the guarantee the
    # confirmation is worth anything at all: an abort cannot have created
    # something already.
    # The boot key, not the work key. Creating a repository needs
    # `administration` and pushing the CI workflow needs `workflows`, and both
    # are permissions `security.FORBIDDEN_PERMISSIONS` refuses to let a worker
    # hold - so this step authenticates as a credential that never enters a
    # container. See docs/security.md.
    client = target if target is not None else _boot_client(plan.full_name)

    repo = _create_repository(client, plan)
    html_url = repo.get("html_url") or f"https://github.com/{plan.full_name}"
    out(f"created {plan.full_name} ({plan.visibility}) at {html_url}")

    # GitHub picks the default branch name from the account's preference, which
    # need not be `main`. Pushing to the name we assumed would create a second
    # branch and leave the default one empty - and the generated README would
    # document protection on a branch that does not exist.
    branch = repo.get("default_branch") or plan.default_branch
    plan = replace(plan, default_branch=branch)

    commit_sha = _push_initial_commit(client, plan, branch, html_url)
    out(f"pushed the initial commit to {branch} @ {commit_sha[:7]}")

    labels = ensure_labels(client)
    out(labels.summary())

    protection = _apply_protection(client, plan, html_url)
    out(f"protected {branch} with: {', '.join(protection)}")

    return ProvisionReport(
        repo=plan.full_name,
        html_url=html_url,
        default_branch=branch,
        commit_sha=commit_sha,
        labels=labels,
        protection=protection,
        verify_command=plan.verify_command,
    )


def _approved(
    plan: ProvisionPlan,
    *,
    assume_yes: bool,
    confirm: Callable[[ProvisionPlan], bool] | None,
) -> bool:
    if assume_yes:
        return True
    if confirm is not None:
        return confirm(plan)
    if sys.stdin is not None and sys.stdin.isatty():
        return confirm_on_terminal(plan)
    return _refuse_unattended(plan)


def _create_repository(client: GitHubClient, plan: ProvisionPlan) -> dict[str, Any]:
    payload = {
        "name": plan.name,
        "description": plan.description,
        "private": plan.private,
        # True, and then thrown away. GitHub refuses `POST /git/blobs` on a
        # repository with no commits at all - "409: Git Repository is empty" -
        # so the blob-tree-commit-ref path cannot bootstrap one. `auto_init`
        # gives the repository a commit so that API works; the parentless
        # commit below then *replaces* it by force-updating the ref, leaving a
        # history one commit long that contains the workflow. The acceptance
        # criterion is kept: the first commit anyone ever sees has CI in it.
        "auto_init": True,
        # Issues are the ledger (`docs/architecture-v2.md`). A repo with issues
        # disabled cannot host the swarm at all.
        "has_issues": True,
        "has_wiki": False,
        "has_projects": False,
        # Squash only, matching the merge method #23 uses. Offering three merge
        # methods on a repo where exactly one caller ever merges is three ways
        # for the history to disagree with itself.
        "allow_squash_merge": True,
        "allow_merge_commit": False,
        "allow_rebase_merge": False,
        # Every task branch is `swarm/issue-<n>` and is dead the moment it
        # merges; without this the branch list grows by one per task forever.
        "delete_branch_on_merge": True,
    }
    path = "/user/repos" if _owner_is_viewer(client, plan.owner) else f"/orgs/{plan.owner}/repos"
    try:
        return client._send_json("POST", client._url(path), payload)
    except GitHubHTTPError as exc:
        if exc.status == 422:
            # Overwhelmingly "name already exists on this account". Nothing has
            # been written to it and nothing will be: adopting an existing repo
            # is a different operation with different consent behind it.
            raise ProvisionError(
                f"{plan.full_name} was not created and nothing was modified: {exc}"
            ) from exc
        raise


def _owner_is_viewer(client: GitHubClient, owner: str) -> bool:
    """Is `owner` the token's own account, or an organisation?

    The two take different creation endpoints, and guessing wrong answers 404 -
    an error that reads like "no such org" whether or not the org exists. One
    cheap call removes the ambiguity.
    """
    viewer = client._get(client._url("/user")) or {}
    return str(viewer.get("login", "")).casefold() == owner.casefold()


def _push_initial_commit(
    client: GitHubClient,
    plan: ProvisionPlan,
    branch: str,
    html_url: str,
) -> str:
    """Write every generated file as one parentless commit, then create the ref."""
    base = f"/repos/{client.repo}/git"
    try:
        tree = []
        for path, content in sorted(plan.files().items()):
            blob = client._send_json(
                "POST", client._url(f"{base}/blobs"), {"content": content, "encoding": "utf-8"}
            )
            tree.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})

        tree_sha = client._send_json("POST", client._url(f"{base}/trees"), {"tree": tree})["sha"]
        commit = client._send_json(
            "POST",
            client._url(f"{base}/commits"),
            {"message": _commit_message(plan), "tree": tree_sha, "parents": []},
        )
        # `auto_init` already created this ref, so the ordinary case is a
        # force-update that discards its commit. POST first anyway: a caller
        # that reached here on a repository without the ref (an adopted empty
        # one, a non-default branch) still gets it created.
        try:
            client._send_json(
                "POST",
                client._url(f"{base}/refs"),
                {"ref": f"refs/heads/{branch}", "sha": commit["sha"]},
            )
        except GitHubHTTPError as exc:
            if exc.status != 422:
                raise
            client._send_json(
                "PATCH",
                client._url(f"{base}/refs/heads/{branch}"),
                {"sha": commit["sha"], "force": True},
            )
    except GitHubHTTPError as exc:
        raise ProvisionError(f"{html_url} exists but is empty: {_explain(exc)}") from exc
    return str(commit["sha"])


def _commit_message(plan: ProvisionPlan) -> str:
    return (
        f"Initial commit\n\n"
        f"Repository generated by apiary from the prompt:\n\n    {plan.prompt}\n"
    )


def _apply_protection(
    client: GitHubClient, plan: ProvisionPlan, html_url: str
) -> tuple[str, ...]:
    """Create the branch ruleset. See `ProvisionPlan.rules` for the shape."""
    try:
        client._send_json(
            "POST", client._url(f"/repos/{client.repo}/rulesets"), plan.ruleset()
        )
    except GitHubHTTPError as exc:
        raise ProvisionError(
            f"{html_url} exists with its initial commit and labels, but is UNPROTECTED: "
            f"{_explain(exc)}"
        ) from exc
    return tuple(rule["type"] for rule in plan.rules())


def _explain(exc: GitHubHTTPError) -> str:
    """Add the sentence a bare GitHub error leaves the reader to guess.

    Both of these cost a real debugging session the first time: a fine-grained
    token is perfectly capable of creating a repository and yet forbidden from
    putting a workflow file in it, and rulesets on private repositories are a
    paid feature. Neither error says so.
    """
    message = str(exc)
    if exc.status == 403 and "workflow" in message.lower():
        return (
            f"{message} - the token needs the `workflows` permission to commit "
            f"{CI_WORKFLOW_PATH}"
        )
    if exc.status in {403, 404} and "/rulesets" in exc.url:
        return (
            f"{message} - rulesets on private repositories need a paid plan; "
            "create the repo public instead, or apply protection by hand"
        )
    return message


# --------------------------------------------------------------------------
# Generated file contents
# --------------------------------------------------------------------------


#: The setup step each stack's toolchain needs, keyed by stack id.
#:
#: **Python emits nothing, and that is the whole reason the default is safe.**
#: `ubuntu-latest` ships a Python, so today's generated workflow has no setup
#: step at all - and the steps list it produces after this change is the one it
#: produced before, unchanged.
#:
#: Node is the second entry because it is #87's first non-Python stack, and it
#: is where the caching matters: `cache: npm` keys on the lockfile and turns a
#: cold install into a warm one. `ubuntu-latest` ships a Node too, but not a
#: pinned one, and an unpinned runtime is a gate that changes under the repo.
#:
#: A stack with no entry gets no setup step rather than an error. The refusal
#: for a stack this host cannot run belongs to #103, at planning time, before a
#: repository exists - not here, where the only thing left to do about it is
#: write a broken workflow.
CI_SETUP: dict[str, tuple[str, ...]] = {
    "python": (),
    "node": (
        "- uses: actions/setup-node@v4",
        "  with:",
        "    node-version: '22'",
        "    cache: npm",
    ),
    "react": (
        "- uses: actions/setup-node@v4",
        "  with:",
        "    node-version: '22'",
        "    cache: npm",
    ),
}

#: How deep a step sits under `steps:`. One place, because the whole failure
#: mode this function has is an indent that YAML accepts and GitHub ignores.
_STEP_INDENT = " " * 6


def _ci_workflow(verify_command: str, *, stack: str = "python", branch: str = "main") -> str:
    """The workflow whose one job is the required status check.

    Three decisions worth stating, because two of them replace earlier ones.

    **`push` is filtered to the default branch.** It used to be unfiltered, and
    the argument was that the initial commit lands directly on the default
    branch and must produce a check run there. That is still true and the
    filter still allows it - what the filter removes is the *second* run every
    PR push paid for, once for `push` on `swarm/issue-<n>` and once for
    `pull_request`. Roughly 20 issues x ~1.3 attempts x 2 runs is ~52 CI runs
    per objective, each with a cold install, on billed private repositories.

    **The setup step is per stack.** `ubuntu-latest` ships a Python, which is
    why there was no setup step and why the Python command has to be spelled
    `python3`. Nothing else the swarm can generate is so lucky. `CI_SETUP` is
    the table; a stack that is not in it emits nothing, and refusing a stack
    this host cannot run is #103's job, at planning time.

    **The command is a block scalar.** `run: {command}` interpolated inline is
    valid YAML right up until the command contains `: `, a `#`, or a quote -
    and #101 makes that command model-generated. An invalid workflow does not
    fail loudly: the required check never reports, and **every PR blocks
    forever** while looking exactly like a check that is still running.
    """
    # Each setup line carries its own newline, so a stack with no setup step
    # leaves the steps list byte-for-byte as it was rather than a blank line
    # where a block used to be.
    steps = "".join(f"{_STEP_INDENT}{line}\n" for line in CI_SETUP.get(stack, ()))
    return f"""\
name: ci

on:
  push:
    branches: [{branch}]
  pull_request:

jobs:
  # This job id and its name are the required status check context in the
  # {RULESET_NAME} ruleset. Rename one without the others and the default
  # branch waits forever for a check that never reports.
  {CHECK_NAME}:
    name: {CHECK_NAME}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
{steps}
      - name: {CHECK_NAME}
        run: |
          {verify_command}
"""


def _readme(plan: ProvisionPlan) -> str:
    reviews = (
        "**required reviews are on**, so merges wait for a human approval"
        if plan.require_reviews
        else "**no required reviews**"
    )
    # The reasoning is written down either way. Which shape was chosen is only
    # half the story; a reader who disagrees needs to know what the other half
    # costs before switching.
    rationale = (
        """\
Requiring a review was chosen deliberately here, which means this repository
needs a human. Nobody approves the swarm's PRs but you, and until you do,
nothing merges. That is the right trade when someone is watching and the wrong
one when nobody is: on an unattended repository the same rule would force the
admin override on every merge, and a gate bypassed every time measures nothing.
"""
        if plan.require_reviews
        else f"""\
The missing rule is the interesting one, so here is the reasoning. Mirroring
apiary's own ruleset would require a code-owner review on every PR. Nobody is
watching this repository, so no review would ever arrive, and the swarm's merge
step would have to use the admin override every single time. A gate that is
bypassed on every merge is not a gate - it measures nothing, and it hides the
one signal worth having, which is whether CI actually passed.

So the protection here is the shape an unattended swarm can satisfy by doing
the work: write the code, pass the checks, merge. Nothing needs an override,
which means an override appearing in the log is a real signal that something
went wrong.

To opt in on a repository that does have a human in the loop, add a
`pull_request` rule with `required_approving_review_count: 1` to the
`{RULESET_NAME}` ruleset, or re-provision with `--require-reviews`. Be clear
about the consequence: an unattended swarm has nobody to approve its PRs, so
from that moment nothing merges without a person.
"""
    )
    # Which of the two this repository got is a fact about it, so the README
    # states the one that is true here rather than describing both. A generated
    # file that documents a command the workflow does not run is the first thing
    # a reader trusts and the last thing anyone re-checks.
    verify_note = (
        """\
CI runs exactly this command and only its exit code is believed. It is a
placeholder: there is no code here to test yet, and it passes on the initial
commit so the required check has something to report. Whoever writes the first
test replaces it here and in the workflow.
"""
        if plan.verify_command == PLACEHOLDER_VERIFY
        else """\
CI runs exactly this command and only its exit code is believed, and every task
the swarm plans carries the same command as its own `## Verify`. It is one
string in three places - this file, the workflow, and each issue - because
three that can disagree is a repository where the answer depends on which one
you happened to read.
"""
    )
    return f"""\
# {plan.name}

Generated by [apiary](https://github.com/shahrestani-me/apiary) from one prompt:

> {plan.prompt}

No human has written code here yet. Everything below describes the starting
conditions the swarm works against, not a finished project.

## Verify

```
{plan.verify_command}
```

{verify_note}
## Branch protection

`{plan.default_branch}` is protected by a repository ruleset named `{RULESET_NAME}`:

- **required status checks** - `{CHECK_NAME}` must pass before anything merges;
- **no force-push and no deletion** of the default branch;
- **no bypass actors**, including admins;
- and {reviews}.

{rationale}
Because there are no bypass actors, repairing a broken gate means editing the
ruleset - visible in the audit log - rather than force-pushing over it.
"""


def _mit_license(holder: str, year: int) -> str:
    return f"""\
MIT License

Copyright (c) {year} {holder}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="swarm-provision",
        description="Create a new GitHub repository from a prompt, ready for the swarm.",
    )
    parser.add_argument("prompt", help="what the project should be")
    parser.add_argument("--owner", required=True, help="user or organisation to create it under")
    parser.add_argument("--name", default=None, help="repository name (default: from the prompt)")
    parser.add_argument("--public", action="store_true", help="create it public (default: private)")
    parser.add_argument(
        "--verify",
        default=PLACEHOLDER_VERIFY,
        help=f"command CI runs, written into the workflow (default: {PLACEHOLDER_VERIFY!r})",
    )
    parser.add_argument(
        "--require-reviews",
        action="store_true",
        help="also require one approving review; only for repos with a human in the loop",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="create it without asking; the only way to run this unattended",
    )
    args = parser.parse_args(argv)

    try:
        plan = ProvisionPlan.for_prompt(
            args.prompt,
            owner=args.owner,
            name=args.name,
            private=not args.public,
            verify_command=args.verify,
            require_reviews=args.require_reviews,
        )
        report = provision(plan, assume_yes=args.yes)
    except ProvisionAborted as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    except (GitHubError, ValueError) as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1

    print()
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
