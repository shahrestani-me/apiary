"""Unit tests for greenfield provisioning, against a mocked GitHub.

**Nothing here creates a repository.** Every test drives `FakeGitHub`, an
in-memory account that stores repos, blobs, refs, labels and rulesets and hands
them back. That is not only about speed: the thing this module does is
irreversible and outward-facing, so a test suite that reached the real API
would leave a trail of half-provisioned repos behind every red run, and the one
behaviour most worth testing - "it refuses to create anything without a
go-ahead" - is only meaningful if a failure of it costs nothing.

`FakeGitHub` enforces two of GitHub's own rules rather than merely recording
calls, because both are ordering traps that a call-order assertion would let
through on a different but equally wrong implementation:

- a repository name that already exists answers 422 and stays untouched;
- a ref push to a branch already carrying a required-status-check rule is
  rejected, which is what makes "protect after the first commit, not before"
  a property of the code instead of a comment in it.

It is written to be lifted into the shared fixture set (#31), alongside
`test_github_client.py`'s `FakeTransport` and `test_labels.py`'s `FakeRepo`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from swarm.github.client import GitHubClient, Response
from swarm.github.labels import SWARM_LABELS
from swarm.greenfield.stacks import REACT_TOOLCHAIN
from swarm.greenfield.provision import (
    CHECK_NAME,
    CI_SETUP,
    CI_WORKFLOW_PATH,
    PLACEHOLDER_VERIFY,
    RULESET_NAME,
    ProvisionAborted,
    ProvisionError,
    ProvisionPlan,
    confirm_on_terminal,
    main,
    provision,
    slugify,
)

API = "https://api.github.com"
OWNER = "shahrestani-me"
PROMPT = "a CLI that converts markdown tables to CSV"
NAME = "a-cli-that-converts-markdown-tables-to-csv"


def plan_for(**overrides: Any) -> ProvisionPlan:
    overrides.setdefault("owner", OWNER)
    overrides.setdefault("year", 2026)
    return ProvisionPlan.for_prompt(PROMPT, **overrides)


# --------------------------------------------------------------------------
# A GitHub account with just enough API to provision into (#31)
# --------------------------------------------------------------------------


def response(status: int, payload: Any = None, **headers: str) -> Response:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    return Response(status, headers, body)


def _sha(payload: Any) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class FakeGitHub:
    """An account that stores what it is told to create and serves it back."""

    login: str = OWNER
    default_branch: str = "main"
    repos: dict[str, dict[str, Any]] = field(default_factory=dict)
    blobs: dict[str, str] = field(default_factory=dict)
    trees: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    commits: dict[str, dict[str, Any]] = field(default_factory=dict)
    refs: dict[str, str] = field(default_factory=dict)
    labels: dict[str, dict[str, str]] = field(default_factory=dict)
    rulesets: list[dict[str, Any]] = field(default_factory=list)
    requests: list[tuple[str, str, Any]] = field(default_factory=list)

    # --- introspection the tests read ---------------------------------

    @property
    def paths(self) -> list[str]:
        return [path for _, path, _ in self.requests]

    def posted(self, path: str) -> list[Any]:
        return [
            payload
            for method, sent, payload in self.requests
            if method == "POST" and sent.endswith(path)
        ]

    def files(self) -> dict[str, str]:
        """The tree of the one commit, path -> content."""
        tree = self.trees[next(iter(self.commits.values()))["tree"]]
        return {entry["path"]: self.blobs[entry["sha"]] for entry in tree}

    # --- routing ------------------------------------------------------

    def send(self, method, url, headers, body, timeout):
        split = urllib.parse.urlsplit(url)
        payload = json.loads(body.decode("utf-8")) if body else None
        self.requests.append((method, split.path, payload))
        parts = split.path.strip("/").split("/")

        if method == "GET" and parts == ["user"]:
            return response(200, {"login": self.login})
        if method == "POST" and parts == ["user", "repos"]:
            return self.create_repo(self.login, payload)
        if method == "POST" and len(parts) == 3 and (parts[0], parts[2]) == ("orgs", "repos"):
            return self.create_repo(parts[1], payload)

        if parts[0] == "repos" and len(parts) >= 4:
            repo = f"{parts[1]}/{parts[2]}"
            if repo not in self.repos:
                return response(404, {"message": "Not Found"})
            return self._repo_request(method, repo, parts[3:], split, payload)

        raise AssertionError(f"unexpected request {method} {split.path}")

    def _repo_request(self, method, repo, rest, split, payload) -> Response:
        if method == "POST" and rest == ["git", "blobs"]:
            return self.create_blob(payload)
        if method == "POST" and rest == ["git", "trees"]:
            return self.create_tree(payload)
        if method == "POST" and rest == ["git", "commits"]:
            return self.create_commit(payload)
        if method == "POST" and rest == ["git", "refs"]:
            return self.create_ref(repo, payload)
        if method == "PATCH" and rest[:3] == ["git", "refs", "heads"]:
            return self.update_ref(repo, "/".join(rest[3:]), payload)
        if rest == ["labels"]:
            if method == "GET":
                return response(200, list(self.labels.values()))
            if method == "POST":
                return self.create_label(payload)
        if method == "POST" and rest == ["rulesets"]:
            return self.create_ruleset(payload)
        raise AssertionError(f"unexpected request {method} {split.path}")

    # --- handlers, overridable one by one to inject a failure ---------

    def create_repo(self, owner: str, payload: Any) -> Response:
        full = f"{owner}/{payload['name']}"
        if full in self.repos:
            return response(
                422,
                {
                    "message": "Repository creation failed.",
                    "errors": [
                        {
                            "resource": "Repository",
                            "code": "custom",
                            "field": "name",
                            "message": "name already exists on this account",
                        }
                    ],
                },
            )
        self.repos[full] = dict(
            payload,
            full_name=full,
            default_branch=self.default_branch,
            html_url=f"https://github.com/{full}",
        )
        if payload.get("auto_init"):
            # GitHub writes its own initial commit, which means the default
            # branch already exists before anything else is pushed. Modelled
            # because the real thing does it: without this the fake accepted a
            # `POST /git/refs` the API would reject, and the suite could not
            # see the force-update the provisioner has to perform.
            self.refs[f"refs/heads/{self.default_branch}"] = _sha("auto-init")
        return response(201, self.repos[full])

    def create_blob(self, payload: Any) -> Response:
        sha = _sha(payload["content"])
        self.blobs[sha] = payload["content"]
        return response(201, {"sha": sha})

    def create_tree(self, payload: Any) -> Response:
        sha = _sha(payload["tree"])
        self.trees[sha] = payload["tree"]
        return response(201, {"sha": sha})

    def create_commit(self, payload: Any) -> Response:
        sha = _sha(payload)
        self.commits[sha] = dict(payload)
        return response(201, {"sha": sha, **payload})

    def update_ref(self, repo: str, branch: str, payload: Any) -> Response:
        ref = f"refs/heads/{branch}"
        if ref not in self.refs:
            return response(422, {"message": "Reference does not exist"})
        if not payload.get("force"):
            return response(422, {"message": "Update is not a fast forward"})
        self.refs[ref] = payload["sha"]
        return response(200, {"ref": ref, "object": {"sha": payload["sha"]}})

    def create_ref(self, repo: str, payload: Any) -> Response:
        branch = payload["ref"].removeprefix("refs/heads/")
        if payload["ref"] in self.refs:
            # What GitHub answers when `auto_init` already made this branch.
            return response(422, {"message": "Reference already exists"})
        # GitHub's own ordering rule: a branch carrying a required status check
        # rejects a push that has not passed it, and creating the ref is a
        # push. Provisioning that protects before it commits deadlocks here.
        if branch == self.repos[repo]["default_branch"] and self._checks_required():
            return response(
                422,
                {"message": f"required status check {CHECK_NAME!r} is expected"},
            )
        self.refs[payload["ref"]] = payload["sha"]
        return response(201, {"ref": payload["ref"], "object": {"sha": payload["sha"]}})

    def create_label(self, payload: Any) -> Response:
        self.labels[payload["name"].casefold()] = dict(payload)
        return response(201, self.labels[payload["name"].casefold()])

    def create_ruleset(self, payload: Any) -> Response:
        self.rulesets.append(dict(payload))
        return response(201, dict(payload, id=len(self.rulesets)))

    def _checks_required(self) -> bool:
        return any(
            rule["type"] == "required_status_checks"
            for ruleset in self.rulesets
            for rule in ruleset["rules"]
        )


def client_for(fake: FakeGitHub, repo: str) -> GitHubClient:
    return GitHubClient(repo, "test-token", api_url=API, transport=fake)


def provision_into(fake: FakeGitHub, plan: ProvisionPlan | None = None, **kwargs: Any):
    plan = plan or plan_for()
    kwargs.setdefault("assume_yes", True)
    kwargs.setdefault("out", lambda line: None)
    return provision(plan, client_for(fake, plan.full_name), **kwargs)


# --------------------------------------------------------------------------
# The plan, which needs no transport at all
# --------------------------------------------------------------------------


def test_the_repository_name_is_derived_from_the_prompt():
    assert plan_for().name == NAME
    assert plan_for().full_name == f"{OWNER}/{NAME}"


def test_a_long_prompt_is_cut_on_a_word_boundary():
    name = slugify("a tool for converting arbitrarily long descriptions into short names")
    assert len(name) <= 48
    assert not name.endswith("-")
    assert name == "a-tool-for-converting-arbitrarily-long"


def test_punctuation_and_case_do_not_reach_the_name():
    # GitHub accepts [A-Za-z0-9._-] only; anything else is a 422 at creation.
    assert slugify("Markdown --> CSV, please!") == "markdown-csv-please"


def test_a_prompt_with_no_letters_asks_for_an_explicit_name():
    # Better to stop here than to invent a name nobody chose for a repository
    # nobody can rename back.
    with pytest.raises(ValueError, match="could not derive"):
        ProvisionPlan.for_prompt("!!! ???", owner=OWNER)


def test_a_name_github_would_reject_is_rejected_locally():
    with pytest.raises(ValueError, match="usable repository name"):
        plan_for(name="not a repo name")
    with pytest.raises(ValueError, match="usable repository name"):
        plan_for(name="..")


def test_the_verify_command_must_be_a_single_line():
    # docs/issue-contract.md §1.3: one command, and only its exit code is
    # believed. "Run these in order" is semantics nobody agreed to.
    with pytest.raises(ValueError, match="single line"):
        plan_for(verify_command="pip install -e .\npytest -q")
    with pytest.raises(ValueError, match="must not be empty"):
        plan_for(verify_command="   ")


# --------------------------------------------------------------------------
# The initial commit's contents
# --------------------------------------------------------------------------


def test_the_initial_commit_carries_a_readme_a_license_and_ci():
    assert sorted(plan_for().files()) == [CI_WORKFLOW_PATH, "LICENSE", "README.md"]


def test_the_default_verify_command_passes_on_the_initial_commit(tmp_path: Path):
    # The acceptance criterion, checked without GitHub: lay the generated files
    # out and run the command CI will run. `python -m pytest -q` would exit 5
    # here - no tests collected - and every generated repo would open red.
    plan = plan_for()
    for path, content in plan.files().items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    result = subprocess.run(plan.verify_command, shell=True, cwd=tmp_path)

    assert result.returncode == 0


def test_the_workflow_is_valid_yaml_that_runs_on_push():
    # The workflow is built by string formatting, where one wrong indent
    # produces a file GitHub silently ignores - no check run, and a required
    # check that never reports blocks the branch forever.
    yaml = pytest.importorskip("yaml")
    document = yaml.safe_load(plan_for().files()[CI_WORKFLOW_PATH])

    # `on` is YAML 1.1's boolean true, which is why the key is not the string.
    triggers = document.get("on", document.get(True))
    assert set(triggers) == {"push", "pull_request"}
    # Filtered to the default branch, where it used to be unfiltered. The
    # initial commit lands straight on that branch and still produces a check
    # run there; what the filter removes is the second run every PR push paid
    # for, once for `push` on `swarm/issue-<n>` and once for `pull_request`.
    assert triggers["push"] == {"branches": ["main"]}


def test_a_pull_request_push_produces_one_run_and_not_two():
    """~20 issues x ~1.3 attempts x 2 runs is ~52 CI runs per objective, each
    with a cold install, on a billed private repo."""
    yaml = pytest.importorskip("yaml")
    triggers = yaml.safe_load(plan_for().files()[CI_WORKFLOW_PATH])
    triggers = triggers.get("on", triggers.get(True))

    assert triggers["push"]["branches"] == ["main"]
    assert "swarm/issue-7" not in triggers["push"]["branches"]


def test_the_push_filter_names_the_branch_github_actually_made():
    """`provision` corrects `default_branch` from the creation response before
    generating any file, because the account has a preference and this plan is
    only guessing. A filter naming the guess would never fire."""
    yaml = pytest.importorskip("yaml")
    plan = dataclasses.replace(plan_for(), default_branch="trunk")

    document = yaml.safe_load(plan.files()[CI_WORKFLOW_PATH])
    triggers = document.get("on", document.get(True))

    assert triggers["push"] == {"branches": ["trunk"]}


def test_the_ci_job_name_is_the_required_status_check_context():
    # A required check whose name no job reports blocks the branch forever, and
    # looks exactly like a check that is still running.
    yaml = pytest.importorskip("yaml")
    plan = plan_for()
    jobs = yaml.safe_load(plan.files()[CI_WORKFLOW_PATH])["jobs"]
    checks = _status_check_rule(plan)["parameters"]["required_status_checks"]

    assert list(jobs) == [CHECK_NAME]
    assert jobs[CHECK_NAME]["name"] == CHECK_NAME
    assert [check["context"] for check in checks] == [CHECK_NAME]


def test_the_workflow_runs_the_plans_verify_command():
    """Read back through the parser, not as a substring.

    The command is a block scalar now, so `run: cargo test` no longer appears
    in the text - and a substring assertion could not have caught the reason
    for the change anyway.
    """
    yaml = pytest.importorskip("yaml")
    workflow = plan_for(verify_command="cargo test").files()[CI_WORKFLOW_PATH]

    steps = yaml.safe_load(workflow)["jobs"][CHECK_NAME]["steps"]

    assert steps[-1]["run"].strip() == "cargo test"


@pytest.mark.parametrize(
    "command",
    [
        'echo "a: b" && pytest -q',          # a mapping-looking colon
        "pytest -q  # every test",            # a comment marker
        "pytest -q -k 'not slow'",            # single quotes
        'pytest -q -k "not slow"',            # double quotes
        "go test ./... # all packages",
    ],
)
def test_an_awkward_verify_command_still_yields_valid_yaml(command):
    """The reason the block scalar matters more than it looks.

    #101 makes this command model-generated. Interpolated inline, a stray `: `
    yields an invalid workflow - and an invalid workflow does not fail loudly:
    the required check never reports, and **every PR blocks forever** while
    looking exactly like a check that is still running.
    """
    yaml = pytest.importorskip("yaml")
    workflow = plan_for(verify_command=command).files()[CI_WORKFLOW_PATH]

    steps = yaml.safe_load(workflow)["jobs"][CHECK_NAME]["steps"]

    assert steps[-1]["run"].strip() == command


def test_python_generates_no_setup_step():
    """`ubuntu-latest` ships a Python, which is why there was never a setup
    step and why the command has to be spelled `python3`. The steps list is
    what it was before this ticket."""
    yaml = pytest.importorskip("yaml")

    steps = yaml.safe_load(plan_for().files()[CI_WORKFLOW_PATH])["jobs"][CHECK_NAME]["steps"]

    assert [step.get("uses") for step in steps] == ["actions/checkout@v4", None]


def test_node_sets_up_a_pinned_toolchain():
    """Not because `ubuntu-latest` lacks Node, but because it lacks a *pinned*
    Node - and an unpinned runtime is a gate that changes under the repo."""
    yaml = pytest.importorskip("yaml")
    plan = dataclasses.replace(plan_for(verify_command="npm test"), stack="node")

    steps = yaml.safe_load(plan.files()[CI_WORKFLOW_PATH])["jobs"][CHECK_NAME]["steps"]
    setup = steps[1]

    assert setup["uses"] == "actions/setup-node@v4"
    assert setup["with"]["node-version"]
    # Parsed, not grepped: `test_provision` already prefers this, and a
    # substring assertion cannot catch a setup step indented one space wrong -
    # which is valid YAML that GitHub reads as part of the previous step.
    assert steps[0]["uses"] == "actions/checkout@v4"


@pytest.mark.parametrize("stack", sorted(CI_SETUP))
def test_no_generated_workflow_asks_to_cache_a_lockfile_it_cannot_produce(stack: str):
    """`cache: npm` fails the *step*, not just the cache.

    `actions/setup-node` resolves that option by hashing a lockfile and errors
    with "Dependencies lock file is not found" when there is none - so the job
    goes red before the gate runs. Neither JS stack has one: Node's gate
    installs nothing, so nothing ever writes a lockfile, and React's cannot,
    because writing one needs the registry a worker is denied. It was a cache
    key for a file this design cannot produce.
    """
    yaml = pytest.importorskip("yaml")
    plan = dataclasses.replace(plan_for(verify_command="true"), stack=stack)

    steps = yaml.safe_load(plan.files()[CI_WORKFLOW_PATH])["jobs"][CHECK_NAME]["steps"]

    assert not any("cache" in (step.get("with") or {}) for step in steps), stack


def test_react_installs_the_image_toolchain_and_puts_it_on_the_path():
    """The seam between two worlds: a worker gets its toolchain from
    `Dockerfile.worker.react`, a GitHub runner has no such image.

    The `GITHUB_PATH` step is what keeps the gate one command rather than two.
    `run:` steps do not put `node_modules/.bin` on `PATH`, the worker image
    does, and `STACK_VERIFY["react"]` is a bare `vitest run` - so without it the
    identical command is "not found" on the runner and green in the worker.
    """
    yaml = pytest.importorskip("yaml")
    plan = dataclasses.replace(plan_for(verify_command="vitest run"), stack="react")

    steps = yaml.safe_load(plan.files()[CI_WORKFLOW_PATH])["jobs"][CHECK_NAME]["steps"]

    assert steps[0]["uses"] == "actions/checkout@v4"
    assert steps[1]["uses"] == "actions/setup-node@v4"
    install = steps[2]["run"]
    assert install.startswith("npm install ")
    for spec in REACT_TOOLCHAIN:
        assert f" {spec}" in install, spec
    assert "node_modules/.bin" in steps[3]["run"]
    assert "$GITHUB_PATH" in steps[3]["run"]
    # And the gate itself is still the last step, unchanged by any of it.
    assert steps[-1]["run"].strip() == "vitest run"


def test_a_stack_with_no_entry_emits_no_setup_step_rather_than_failing():
    """Refusing a stack this host cannot run is #103's job, at planning time,
    before a repository exists. Here the only thing left to do about it would
    be to write a broken workflow."""
    yaml = pytest.importorskip("yaml")
    plan = dataclasses.replace(plan_for(), stack="haskell")

    steps = yaml.safe_load(plan.files()[CI_WORKFLOW_PATH])["jobs"][CHECK_NAME]["steps"]

    assert [step.get("uses") for step in steps] == ["actions/checkout@v4", None]


def test_the_license_carries_the_owner_and_the_year():
    license_text = plan_for(license_holder="Kamyar Shahrestani", year=2026).files()["LICENSE"]
    assert license_text.startswith("MIT License")
    assert "Copyright (c) 2026 Kamyar Shahrestani" in license_text


def test_the_readme_records_the_prompt_and_the_verify_command():
    readme = plan_for().files()["README.md"]
    assert PROMPT in readme
    assert PLACEHOLDER_VERIFY in readme
    assert "placeholder" in readme


def test_the_readme_of_a_scaffolded_repo_does_not_call_its_command_a_placeholder():
    # The placeholder wording is true of a repo with nothing in it and a lie
    # about one whose command runs a real suite - and a generated file is the
    # first thing a reader trusts and the last thing anyone re-checks.
    readme = plan_for(verify_command="python3 -m unittest discover -q").files()["README.md"]

    assert "placeholder" not in readme
    assert "`## Verify`" in readme


def test_the_readme_says_which_protection_shape_was_chosen_and_why():
    # Required by the issue: the generated repo has to explain its own gate,
    # because "no required reviews" looks like an oversight otherwise.
    readme = plan_for().files()["README.md"]
    assert "no required reviews" in readme
    assert "admin override" in readme
    assert "--require-reviews" in readme


def test_the_readme_of_a_reviewed_repo_does_not_offer_the_opt_in_twice():
    readme = plan_for(require_reviews=True).files()["README.md"]
    assert "required reviews are on" in readme
    assert "--require-reviews" not in readme


# --------------------------------------------------------------------------
# The protection shape - the ruling this issue exists to make
# --------------------------------------------------------------------------


def _status_check_rule(plan: ProvisionPlan) -> dict[str, Any]:
    return next(rule for rule in plan.rules() if rule["type"] == "required_status_checks")


def test_protection_is_status_checks_plus_no_force_push_and_no_deletion():
    assert [rule["type"] for rule in plan_for().rules()] == [
        "deletion",
        "non_fast_forward",
        "required_status_checks",
    ]


def test_no_required_reviews_by_default():
    # The whole ruling: an unattended repo with a required review would need
    # the admin override on every merge, and a gate bypassed every time
    # measures nothing.
    assert not [rule for rule in plan_for().rules() if rule["type"] == "pull_request"]


def test_required_reviews_are_available_as_a_documented_opt_in():
    rules = {rule["type"]: rule for rule in plan_for(require_reviews=True).rules()}
    assert rules["pull_request"]["parameters"]["required_approving_review_count"] == 1
    # Still not code-owner review: that is apiary's own shape and needs a
    # CODEOWNERS file the generated repo does not have.
    assert rules["pull_request"]["parameters"]["require_code_owner_review"] is False


def test_status_checks_are_not_strict():
    # Strict means "up to date with the base before merging", which serialises
    # a swarm that runs several PRs at once. Keeping a moving base mergeable
    # is #34's job.
    parameters = _status_check_rule(plan_for())["parameters"]
    assert parameters["strict_required_status_checks_policy"] is False


def test_the_ruleset_targets_the_default_branch_and_has_no_bypass():
    ruleset = plan_for().ruleset()
    assert ruleset["name"] == RULESET_NAME
    assert ruleset["enforcement"] == "active"
    assert ruleset["conditions"]["ref_name"]["include"] == ["~DEFAULT_BRANCH"]
    assert ruleset["bypass_actors"] == []


# --------------------------------------------------------------------------
# Confirm before creating
# --------------------------------------------------------------------------


def test_a_refused_confirmation_sends_nothing_at_all():
    # The claim is not "it stops", it is "it stops before anything exists".
    fake = FakeGitHub()
    with pytest.raises(ProvisionAborted):
        provision_into(fake, assume_yes=False, confirm=lambda plan: False)

    assert fake.requests == []
    assert fake.repos == {}


def test_an_unattended_run_without_the_flag_refuses(monkeypatch):
    # No terminal to ask and no explicit go-ahead is exactly the situation the
    # confirmation exists for - not an exemption from it.
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    fake = FakeGitHub()

    with pytest.raises(ProvisionAborted, match="unattended"):
        provision(plan_for(), client_for(fake, f"{OWNER}/{NAME}"), out=lambda line: None)

    assert fake.requests == []


def test_assume_yes_is_the_only_way_through_without_being_asked():
    fake = FakeGitHub()
    asked: list[ProvisionPlan] = []

    provision_into(fake, assume_yes=True, confirm=lambda plan: asked.append(plan) or True)

    assert asked == []
    assert f"{OWNER}/{NAME}" in fake.repos


def test_the_terminal_confirmation_requires_the_repository_name():
    # "y" is muscle memory and fires before the screen is read; the repo name
    # is not, and it is the string most worth having looked at.
    plan = plan_for()
    printed: list[str] = []

    assert confirm_on_terminal(plan, ask=lambda prompt: "y", out=printed.append) is False
    assert confirm_on_terminal(plan, ask=lambda prompt: "yes", out=printed.append) is False
    assert confirm_on_terminal(plan, ask=lambda prompt: f"  {plan.name} ", out=printed.append)


def test_an_interrupted_confirmation_is_a_refusal():
    def interrupt(prompt: str) -> str:
        raise KeyboardInterrupt

    assert confirm_on_terminal(plan_for(), ask=interrupt, out=lambda line: None) is False


def test_the_preview_names_every_irreversible_consequence():
    preview = plan_for().preview()
    assert f"{OWNER}/{NAME}" in preview
    assert "private" in preview
    assert "cannot be undone" in preview
    assert CI_WORKFLOW_PATH in preview
    assert "no required reviews" in preview


def test_the_preview_says_public_out_loud():
    assert "public" in plan_for(private=False).preview()


# --------------------------------------------------------------------------
# Creating the repository
# --------------------------------------------------------------------------


def test_a_repo_under_the_token_owner_goes_to_the_user_endpoint():
    fake = FakeGitHub(login=OWNER)
    provision_into(fake)
    assert "/user/repos" in fake.paths


def test_a_repo_under_an_organisation_goes_to_the_org_endpoint():
    # Guessing wrong answers 404, which reads like "no such org" whether or not
    # the org exists.
    fake = FakeGitHub(login="kamyarshahrestani")
    provision_into(fake)
    assert f"/orgs/{OWNER}/repos" in fake.paths
    assert "/user/repos" not in fake.paths


def test_the_repo_is_created_with_issues_on_and_one_merge_method():
    fake = FakeGitHub()
    provision_into(fake)
    created = fake.repos[f"{OWNER}/{NAME}"]

    assert created["has_issues"] is True          # issues are the ledger
    # True, and then discarded: GitHub refuses the git data API on a
    # repository with no commits, so `auto_init` is the only way to bootstrap
    # one - and the parentless commit then force-replaces it. See
    # `test_the_history_is_one_commit_and_it_carries_the_workflow`.
    assert created["auto_init"] is True
    assert created["private"] is True
    assert created["allow_squash_merge"] is True
    assert created["allow_merge_commit"] is False
    assert created["allow_rebase_merge"] is False
    assert created["delete_branch_on_merge"] is True


def test_the_initial_commit_is_one_parentless_commit_with_every_file():
    fake = FakeGitHub()
    report = provision_into(fake)

    assert len(fake.commits) == 1
    commit = fake.commits[report.commit_sha]
    assert commit["parents"] == []
    assert PROMPT in commit["message"]
    assert fake.files() == plan_for().files()
    assert fake.refs == {"refs/heads/main": report.commit_sha}


def test_the_accounts_own_default_branch_name_is_respected():
    # Pushing to an assumed `main` would leave the real default branch empty
    # and the repository looking broken - and the README would document
    # protection on a branch nobody has.
    fake = FakeGitHub(default_branch="trunk")
    report = provision_into(fake)

    assert report.default_branch == "trunk"
    assert list(fake.refs) == ["refs/heads/trunk"]
    assert "`trunk` is protected" in fake.files()["README.md"]


def test_the_report_carries_the_command_the_committed_workflow_runs():
    # The caller's next act is to plan issues whose `## Verify` must be this
    # exact string. Reporting it means the caller never has to re-derive it
    # from its own copy of the plan, which is one more place it could differ
    # from what is actually in the repository.
    fake = FakeGitHub()
    plan = plan_for(verify_command="python3 -m unittest discover -q")

    report = provision_into(fake, plan)

    assert report.verify_command == plan.verify_command
    # Parsed rather than grepped: the command is a block scalar now, so
    # `run: <command>` no longer appears in the text.
    yaml = pytest.importorskip("yaml")
    steps = yaml.safe_load(fake.files()[CI_WORKFLOW_PATH])["jobs"][CHECK_NAME]["steps"]
    assert steps[-1]["run"].strip() == report.verify_command
    assert report.verify_command in report.summary()


def test_the_swarm_labels_are_provisioned():
    fake = FakeGitHub()
    report = provision_into(fake)

    assert set(fake.labels) == {spec.name.casefold() for spec in SWARM_LABELS}
    assert report.labels.changed is True


def test_protection_is_applied_after_the_commit_not_before():
    # The fake rejects a ref push to a branch that already requires a status
    # check, exactly as GitHub does - so protecting first does not merely look
    # wrong here, it fails.
    fake = FakeGitHub()
    report = provision_into(fake)

    assert fake.paths.index(f"/repos/{report.repo}/git/refs") < fake.paths.index(
        f"/repos/{report.repo}/rulesets"
    )
    assert fake.rulesets[0]["name"] == RULESET_NAME
    assert report.protection == ("deletion", "non_fast_forward", "required_status_checks")


def test_nothing_that_pre_existed_is_ever_deleted_or_overwritten():
    """Provisioning may only add, or replace something it made this run.

    A DELETE or a PUT would mean it can damage what was already there. The one
    PATCH is the exception that proves the rule: it force-updates the default
    branch, and that branch exists only because `auto_init` created it seconds
    earlier in this same call - the throwaway commit that made the git data
    API usable at all. It never touches a ref the run did not create.
    """
    fake = FakeGitHub()
    provision_into(fake)

    methods = {method for method, _, _ in fake.requests}
    assert "DELETE" not in methods
    assert "PUT" not in methods

    patched = [(method, path) for method, path, _ in fake.requests if method == "PATCH"]
    assert all(path.endswith(f"/git/refs/heads/{fake.default_branch}") for _, path in patched)


def test_the_report_summarises_what_now_exists():
    summary = provision_into(FakeGitHub()).summary()
    assert f"https://github.com/{OWNER}/{NAME}" in summary
    assert "non_fast_forward" in summary


# --------------------------------------------------------------------------
# Failures, and what they leave behind
# --------------------------------------------------------------------------


def test_an_existing_repository_is_reported_and_left_alone():
    # Adopting a repo somebody else made is a different operation with
    # different consent behind it. Not this command's call to make.
    fake = FakeGitHub()
    fake.repos[f"{OWNER}/{NAME}"] = {"full_name": f"{OWNER}/{NAME}", "default_branch": "main"}

    with pytest.raises(ProvisionError, match="nothing was modified"):
        provision_into(fake)

    assert fake.commits == {} and fake.refs == {} and fake.rulesets == []


def test_a_token_without_workflow_permission_is_told_what_is_missing():
    # A fine-grained token may create repositories and still be forbidden from
    # committing a file under .github/workflows. The bare error does not say so.
    class NoWorkflows(FakeGitHub):
        def create_blob(self, payload):
            if "actions/checkout" in payload["content"]:
                return response(
                    403,
                    {"message": "refusing to allow an integration to create workflow `ci.yml`"},
                )
            return super().create_blob(payload)

    with pytest.raises(ProvisionError, match="`workflows` permission"):
        provision_into(NoWorkflows())


def test_a_failed_ruleset_says_the_repo_exists_and_is_unprotected():
    # The repo is real by then; swallowing this would leave an unprotected repo
    # reported as a success, and raising without the URL would lose it.
    class NoRulesets(FakeGitHub):
        def create_ruleset(self, payload):
            return response(403, {"message": "Upgrade to GitHub Team to use rulesets"})

    fake = NoRulesets()
    with pytest.raises(ProvisionError) as exc:
        provision_into(fake)

    assert "UNPROTECTED" in str(exc.value)
    assert f"https://github.com/{OWNER}/{NAME}" in str(exc.value)
    assert "paid plan" in str(exc.value)
    assert len(fake.commits) == 1          # the commit is not rolled back, and says so


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def test_the_command_refuses_to_run_unattended_without_yes(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    code = main([PROMPT, "--owner", OWNER])

    assert code == 1
    assert "unattended" in capsys.readouterr().err


def test_the_command_needs_an_owner():
    with pytest.raises(SystemExit):
        main([PROMPT])


def test_the_history_is_one_commit_and_it_carries_the_workflow():
    """The acceptance criterion, kept despite an API that forbids the obvious route.

    `auto_init` gives the repository a commit so `POST /git/blobs` is accepted
    at all - without it GitHub answers `409: Git Repository is empty`, which is
    how this surfaced: a repository created, and then nothing written to it.
    The commit built here has no parents and force-replaces the ref, so what
    survives is a one-commit history containing the workflow rather than a
    generated README with an unverified commit under it.
    """
    fake = FakeGitHub()
    provision_into(fake)

    commits = [r for r in fake.requests if r[0] == "POST" and r[1].endswith("/git/commits")]
    assert commits, "no commit was built"
    assert commits[-1][2]["parents"] == [], "the commit must be parentless"

    refs = [r for r in fake.requests if "/git/refs" in r[1]]
    assert refs, "the branch was never pointed at the commit"
    forced = [r for r in refs if r[0] == "PATCH"]
    assert forced, "auto_init's ref must be force-replaced, not left in place"
    assert forced[-1][2]["force"] is True
