"""#151, held still: the tracker is reached over MCP, and the code host is not.

**The assertion that matters here is negative.** "Intake, comment and create go
through the MCP client" is easy to demonstrate and easy to demonstrate falsely -
a test that watches an MCP call happen proves the call happened, not that the
old one stopped. So the load-bearing tests in this file are the ones that say
what *did not* happen:

1. **A client that raises on every tracker endpoint survives a whole tracker
   path.** `Refuser` below has `list_issues`, `create_issue`,
   `create_issue_comment` and `list_issue_comments`, and every one of them fails
   the test. The ledger is still read, a comment is still posted, an issue is
   still filed - through `mcp.TrackerView` - and the calls that left apiary are
   asserted to be exactly the tools the contract named, with exactly the
   arguments it built.

2. **No module on the tracker path names a tracker endpoint the view does not
   route.** Static, over the AST of `orchestrator/`, `nodes/` and `github/`,
   because the regression this is really for is not today's code: it is the
   ticket six months from now that adds `client.list_milestones(...)` to a
   reconcile rule and silently reaches GitHub past a configured tracker. A
   runtime probe misses it unless the test happens to take that path.

   `GitHubClient`'s whole issue surface is *partitioned* across four lists -
   routed, the label plane #152 removes, the code host, and one entry that is
   nobody's - so a method added to that client tomorrow fails this file until
   somebody classifies it. That is the same shape, and the same motive, as
   `test_framework_boundary`'s "every subcommand is classified".

3. **Nothing in the tracker path tests which server is configured.** Also
   static. This is #151's second acceptance criterion and the reason the ticket
   exists at all: a seam with one implementation and one `if` for that
   implementation is not a seam, and the `if` is the easiest thing in this
   change to add in a hurry.

The positive tests are here too, and they are the cheap ones: what the argument
dict looks like, which shapes of tool result read, what happens when a server
says no. They exist so that a failure in the negative tests is legible - if
`intake` were simply broken, several of these would fail first and say so.

**`list_tree` has a test of its own** (the sixth acceptance criterion). It looks
like a tracker read, it is a git-trees call that landed in #161 so the planner
can see the repository it plans against, and routing it would be a category
error rather than a bug.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pytest

from fixtures.mcp import ENDPOINT, FakeMcpServer
from fixtures.mcp import client as mcp_client
from swarm.github.client import GitHubClient
from swarm.github.ledger import load_ledger, render_marker
from swarm.github.readiness import BLOCKED, READY, resolve_states
from swarm.github.refs import task_ref
from swarm.mcp.contract import PROFILES, ContractError, parse_tracker
from swarm.mcp.tracker import (
    CODE_HOST,
    LABEL_PLANE,
    TRACKER_ENDPOINTS,
    NoSuchCapability,
    Tracker,
    TrackerError,
    TrackerView,
    view_for,
)
from swarm.orchestrator.dispatcher import Capacity
from swarm.orchestrator.reconcile import Reconciler, Snapshot, post_comment
from swarm.run import Run
from swarm.store import STORE_DIR_ENV, SqliteTaskStore

SOURCE = Path(__file__).resolve().parents[1] / "src" / "swarm"

#: The slug the doubles agree on. Only ever printed and used as a store key.
SLUG = "acme/widgets"
RUN_ID = "apiary-20260819-101500-t7k2mq"
BASE_COMMIT = "9f2c1ab3d4e5f60718293a4b5c6d7e8f90a1b2c3"


@pytest.fixture(autouse=True)
def store_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`test_reconcile.store_root`'s reason, and it is per-module so it repeats.

    A test that forgot to redirect the root would open the *operator's* store at
    `.swarm/store`, read a real project's retry budgets and write test judgments
    into them. Nothing would fail; the next real run would believe something
    untrue about its own history.
    """
    root = tmp_path / "store"
    monkeypatch.setenv(STORE_DIR_ENV, str(root))
    return root


@dataclass
class Judged:
    """A scripted judge. Inert, and above all *injected*.

    `test_reconcile.Judged`'s reason: a cycle that changed nothing while nothing
    is in flight consults a model, and on a host with Ollama running a test that
    reached the real oracle would not fail - it would quietly spend a 31B
    inference and pass slowly. No test in this file can reach a model.
    """

    asked: list[Any] = field(default_factory=list)

    def invoke(self, messages: Any) -> Any:
        from swarm.nodes.judge import ProgressJudgement

        self.asked.append(messages)
        return ProgressJudgement(
            request_satisfied=False,
            progress_being_made=True,
            in_loop=False,
            reason="the test's judge",
        )


# --------------------------------------------------------------------------
# The doubles
# --------------------------------------------------------------------------


@dataclass
class Refuser:
    """A code host whose *tracker* endpoints are all failures of the test.

    The point of the whole file. Everything a code host is actually for answers
    normally, so a path that reaches this object for a pull request or a commit
    is not disturbed, and a path that reaches it for an issue fails loudly at the
    line that did it rather than a layer away.

    The label calls answer too, and deliberately: they are `LABEL_PLANE`, they
    are still the code host's until #152, and a double that refused them would
    be asserting a property this ticket does not have.
    """

    labels: dict[int, list[str]] = field(default_factory=dict)
    fetched: list[int] = field(default_factory=list)
    patched: list[int] = field(default_factory=list)
    repo: str = "acme/widgets"

    # --- the tracker endpoints: never, by any path -----------------------

    def list_issues(self, **kwargs: Any) -> Any:
        pytest.fail(f"intake reached the code host: list_issues({kwargs})")

    def create_issue(self, *args: Any, **kwargs: Any) -> Any:
        pytest.fail(f"create reached the code host: create_issue({args}, {kwargs})")

    def create_issue_comment(self, *args: Any, **kwargs: Any) -> Any:
        pytest.fail(f"comment reached the code host: create_issue_comment({args})")

    def list_issue_comments(self, number: int) -> Any:
        pytest.fail(f"a tracker read reached the code host: list_issue_comments({number})")

    # --- the label plane: still the code host's, until #152 ---------------

    def get_issue(self, number: int) -> dict[str, Any]:
        """On `LABEL_PLANE`, and the interesting entry on it.

        Every caller left in the orchestrator is the attempt marker's
        read-modify-write, which §5 requires to be a *fresh* read. That is not
        servable from a cached listing and not servable by a three-capability
        contract either, so it goes direct - and a refusal here would not move
        the marker onto MCP, it would stop the attempt counter being written at
        all. `mcp.LABEL_PLANE` carries the argument at length.
        """
        self.fetched.append(number)
        return {"number": number, "body": "<!-- apiary task=t attempt=1 -->"}

    def add_labels(self, number: int, labels: Iterable[str]) -> Any:
        self.labels.setdefault(number, []).extend(labels)
        return []

    def remove_label(self, number: int, label: str) -> bool:
        return True

    def update_issue(self, number: int, **kwargs: Any) -> dict[str, Any]:
        self.patched.append(number)
        return {"number": number}

    # --- the code host: unremarkable ------------------------------------

    def head_sha(self, ref: str | None = None) -> str:
        return "a" * 40

    def list_tree(self, ref: str | None = None) -> list[str]:
        return ["README.md", "src/app.py"]

    def list_pull_requests(self, *, state: str = "open") -> list[dict[str, Any]]:
        """Empty, but *present*: a cycle that cannot list these is a blind one.

        `Snapshot.open_branches` returns `None` when the method is missing, and
        every rule that needs pull-request state then declines to fire - so a
        double without this would run a cycle that skipped most of itself and
        report a pass for it.
        """
        return []


def issue(number: int, *, label: str = READY, task_id: str | None = None) -> dict[str, Any]:
    """One item as a GitHub MCP server hands it back: the API's own issue JSON.

    The payloads intake returns are still read by the GitHub adapter above
    (`ledger.load_ledger`), which is the pre-existing shape and the follow-up
    epic's to normalise. #151 changes where they came from, not what they are.
    """
    task_id = task_id or f"task-{number}"
    return {
        "number": number,
        "title": f"issue {number}",
        "state": "open",
        "state_reason": None,
        "labels": [{"name": label}],
        "body": "\n".join(
            [
                render_marker(task_id, 0),
                "",
                "## Goal",
                "Do the thing.",
                "",
                "## Files",
                f"- src/{task_id}.py",
                "",
                "## Verify",
                "python -m pytest -q",
                "",
                "## Blocked by",
                "-",
            ]
        ),
    }


#: A `github` block wearing an HTTP endpoint. The profile itself selects the
#: *local stdio* server, which is right for a real run and wrong for a test - it
#: would spawn `github-mcp-server` as a subprocess - so the endpoint is
#: overridden, which `_merge` supports precisely because pointing a profile at a
#: locally proxied server is a legitimate thing to configure. Everything under
#: test is unchanged by it: the tool names, the pinned `method: create`, the
#: `issue_number` field map and the `number` ref rule are all the profile's.
GITHUB_BLOCK = f"""
tracker:
  mcp: github
  endpoint: {ENDPOINT}
  args:
    owner: acme
    repo: widgets
  intake:
    args:
      labels: [swarm:ready]
"""


def tracker(server: FakeMcpServer, *, block: str = GITHUB_BLOCK, token: str = "t") -> Tracker:
    contract = parse_tracker(block, source="tracker.yaml")
    return Tracker(contract=contract, client=mcp_client(server, token=token))


def github_server(**results: Any) -> FakeMcpServer:
    return FakeMcpServer(
        tools=("list_issues", "add_issue_comment", "issue_write"),
        name="github-mcp-server",
        results=dict(results),
    )


# --------------------------------------------------------------------------
# 1. The negative assertion, at run time
# --------------------------------------------------------------------------


def test_the_ledger_is_read_through_mcp_and_the_code_host_is_not_asked():
    server = github_server(list_issues=[issue(4), issue(7)])
    view = TrackerView(Refuser(), tracker(server))

    ledger = load_ledger(Snapshot(view), adopt=False)  # type: ignore[arg-type]

    assert sorted(str(entry.ref) for entry in ledger.entries.values()) == ["#4", "#7"]
    # The call that left apiary, and its whole argument dict: the scope constants
    # from the block's shared `args`, the label filter from `intake.args`, and
    # nothing apiary added of its own.
    assert server.calls == [
        ("list_issues", {"owner": "acme", "repo": "widgets", "labels": ["swarm:ready"]})
    ]


def test_a_cycles_readers_share_one_intake_call():
    """`Snapshot`'s whole reason, and it survives the tracker underneath it.

    Three readers want the listing - the ledger, `resolve_states`, and the
    per-issue state fold - and a tracker read is a third party's rate limit now
    rather than only GitHub's, so the count matters more rather than less.
    """
    server = github_server(list_issues=[issue(4), issue(7)])
    snapshot = Snapshot(TrackerView(Refuser(), tracker(server)))  # type: ignore[arg-type]

    ledger = load_ledger(snapshot, adopt=False)  # type: ignore[arg-type]
    resolve_states(snapshot, [entry.ref for entry in ledger.entries.values()])  # type: ignore[arg-type]
    snapshot.states()

    assert len(server.arguments_for("list_issues")) == 1


def test_a_comment_goes_through_mcp_with_the_contracts_own_ref_argument():
    server = github_server()
    view = TrackerView(Refuser(), tracker(server))

    assert post_comment(view, 7, "apiary: needs a human") is True

    # `issue_number`, not `ref`: the field map in the `github` profile is the
    # thing being spent here, and ADR 0004 exists because a ref is a request
    # argument whose name diverges and not only a response field.
    assert server.calls == [
        (
            "add_issue_comment",
            {
                "owner": "acme",
                "repo": "widgets",
                "issue_number": 7,
                "body": "apiary: needs a human",
            },
        )
    ]


def test_create_goes_through_mcp_with_the_pinned_discriminator():
    """#143's single hardest fact, spent rather than described.

    `issue_write` fulfils create *and* update, so naming the tool is not enough
    and `method: "create"` has to reach the server. A contract carrying only a
    tool name could not make this call at all.
    """
    server = github_server(issue_write={"number": 11, "title": "new work"})
    view = TrackerView(Refuser(), tracker(server))

    filed = view.create_issue("new work", body="do it", labels=["swarm:ready"])

    assert filed["number"] == 11
    assert server.calls == [
        (
            "issue_write",
            {
                "owner": "acme",
                "repo": "widgets",
                "method": "create",
                "title": "new work",
                "body": "do it",
                "labels": ["swarm:ready"],
            },
        )
    ]


def test_create_is_one_call_and_carries_its_labels_with_it():
    """`planner._create`'s crash property, preserved across the seam.

    Create-then-label would leave an issue with no state label if the process
    died between the two, and §3 reads a state-labelless issue as outside the
    ledger entirely - work that exists in the tracker and that nothing will ever
    look at again. So `labels` travels in the create call, and this asserts the
    call count rather than only the arguments.
    """
    server = github_server(issue_write={"number": 11})
    view = TrackerView(Refuser(), tracker(server))

    view.create_issue("new work", body="do it", labels=["swarm:blocked"])

    assert [name for name, _ in server.calls] == ["issue_write"]


def test_the_label_plane_still_reaches_the_code_host():
    """Honest about what #151 does not do. #152 is the ticket that removes it.

    The `swarm:*` labels and the issue-body marker are apiary's own vocabulary,
    which ADR 0001 forbids writing into a customer's tracker at all - so they
    are not routed *and* not kept: they are deleted, one ticket later. Giving
    them a slot in the capability contract in the meantime would be building the
    thing being removed.
    """
    host = Refuser()
    view = TrackerView(host, tracker(github_server()))

    view.add_labels(7, ["swarm:claimed"])
    view.remove_label(7, "swarm:ready")
    view.get_issue(7)
    view.update_issue(7, body="marker")

    assert host.labels == {7: ["swarm:claimed"]}
    assert host.fetched == [7]
    assert host.patched == [7]
    assert "get_issue" in LABEL_PLANE


# --------------------------------------------------------------------------
# 2. The negative assertion, statically
# --------------------------------------------------------------------------

#: `GitHubClient`'s methods that are questions about a work item, and therefore
#: the universe this file partitions. Everything here is either routed by
#: `TrackerView`, on `LABEL_PLANE` awaiting #152, or nobody's.
#:
#: `list_milestones` is the third case and the one this list exists for. A
#: milestone is the customer's own workflow, which ADR 0001 decision 2 forbids
#: apiary from modelling, so it gets no capability and no route - and nothing
#: calls it, which the scan below is what proves.
NOBODYS: dict[str, str] = {
    "list_milestones": (
        "a milestone is the customer's workflow; ADR 0001 decision 2 forbids "
        "apiary modelling it, so it has no capability and no caller"
    ),
}

#: Not questions about a work item at all: constructing a client and dropping
#: its conditional-request cache.
MECHANICS: tuple[str, ...] = ("from_env", "invalidate_cache")

#: Where a call on the tracker path may appear. `github/client.py` is excluded
#: because it *defines* these methods, and `github/labels.py` because
#: provisioning the six `swarm:*` labels into a repository is the label plane's
#: own installer and dies with it in #152.
SCANNED = ("orchestrator", "nodes", "github")
NOT_SCANNED = ("client.py", "labels.py")


def _modules() -> list[Path]:
    found: list[Path] = []
    for package in SCANNED:
        for path in sorted((SOURCE / package).glob("*.py")):
            if path.name not in NOT_SCANNED:
                found.append(path)
    assert found, f"nothing to scan under {SOURCE}"
    return found


def _named(path: Path) -> set[str]:
    """Every attribute called, and every string literal, in one module.

    Both halves are needed and the second is not paranoia: `reconcile.py` reaches
    its comment method as `getattr(client, COMMENT_METHOD)`, where
    `COMMENT_METHOD` is a module constant - so an AST scan that looked only at
    `x.create_issue_comment(...)` would find the one call this whole ticket is
    about nowhere at all.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def _issue_surface() -> set[str]:
    return set(TRACKER_ENDPOINTS) | set(LABEL_PLANE) | set(NOBODYS)


def test_every_github_client_method_is_classified():
    """A method added to that client has to be sorted, not silently direct.

    The failure this is for is a later ticket growing `GitHubClient.close_issue`,
    calling it from a reconcile rule, and reaching GitHub past a configured
    tracker - with every test in this file still green, because none of them
    knew to look for it.
    """
    public = {
        name
        for name, value in vars(GitHubClient).items()
        if not name.startswith("_") and (callable(value) or isinstance(value, classmethod))
    }
    classified = _issue_surface() | set(CODE_HOST) | set(MECHANICS)

    assert public - classified == set(), (
        "GitHubClient grew a method nobody classified. Decide whether it is a "
        "tracker capability (route it in mcp/tracker.py and add it to "
        "TRACKER_ENDPOINTS), the label plane's (LABEL_PLANE, and #152 deletes "
        "it), the code host's (CODE_HOST, with the reason), or nobody's (NOBODYS "
        "in this file)."
    )
    assert classified - public - {"list_issue_comments"} <= public


def test_no_module_on_the_tracker_path_names_an_unrouted_tracker_endpoint():
    """#151's first acceptance criterion, as a scan rather than as a hope.

    Every name any of these modules reaches for that is a question about a work
    item must be one `TrackerView` routes, or one of the three the label plane
    still owns. A `list_milestones` here - or a `close_issue` a year from now -
    fails this line, which is the only moment anybody is guaranteed to be
    looking.
    """
    routed = set(TRACKER_ENDPOINTS) | set(LABEL_PLANE)
    unrouted: dict[str, set[str]] = {}
    for path in _modules():
        reached = _named(path) & _issue_surface()
        if reached - routed:
            unrouted[path.name] = reached - routed

    assert unrouted == {}, (
        f"these modules reach a work item past mcp.TrackerView: {unrouted}. Route "
        f"the call in mcp/tracker.py, or say why it is not a tracker question."
    )


def test_the_scan_would_notice_a_call_that_bypassed_the_view():
    """The scan is only worth having if it fails on the thing it is for."""
    module = ast.parse("def rule(client):\n    return client.list_milestones(state='open')\n")
    names: set[str] = {
        node.func.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert names & _issue_surface() == {"list_milestones"}
    assert not names & (set(TRACKER_ENDPOINTS) | set(LABEL_PLANE))


def test_the_scan_reads_the_tree_it_is_asserting_about():
    """Guard against a glob that quietly matches nothing (`test_framework_boundary`'s)."""
    names = {path.name for path in _modules()}
    assert {"reconcile.py", "planner.py", "readiness.py", "ledger.py"} <= names
    assert "client.py" not in names


# --------------------------------------------------------------------------
# 3. Nothing tests which server is configured
# --------------------------------------------------------------------------


def test_the_tracker_path_never_branches_on_which_server_this_is():
    """#151's second acceptance criterion. The easiest thing here to get wrong.

    Named profiles are data (`PROFILES`, `INADMISSIBLE_INTAKE`) and the code that
    spends them may not know their names. So: no module of the tracker path may
    contain a profile name as a literal, and none may compare against `.mcp`.
    """
    offenders: dict[str, list[str]] = {}
    for path in [SOURCE / "mcp" / "tracker.py", *_modules()]:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for side in (node.left, *node.comparators):
                    if isinstance(side, ast.Attribute) and side.attr == "mcp":
                        found.append("compares against contract.mcp")
                    if isinstance(side, ast.Constant) and side.value in PROFILES:
                        found.append(f"compares against {side.value!r}")
            # A dict or set keyed by a profile name would be the same branch
            # wearing a lookup, which is how `INADMISSIBLE_INTAKE` is allowed to
            # exist in `contract.py` and why that file is not scanned here: it is
            # the module that *holds* the data.
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                if node.slice.value in PROFILES:
                    found.append(f"looks up {node.slice.value!r}")
        if found:
            offenders[path.name] = found

    assert offenders == {}, (
        f"the tracker path decides something from which server is configured: "
        f"{offenders}. Per-server facts belong in PROFILES, which is data."
    )


# --------------------------------------------------------------------------
# The sixth acceptance criterion
# --------------------------------------------------------------------------


def test_list_tree_stays_on_the_direct_path():
    """It reads like a tracker call and is a git-trees read (#161, and #151's AC6).

    The planner sees the repository it is planning against through this. Routing
    it through a tracker would ask a task system for a source tree.
    """
    assert "list_tree" not in TRACKER_ENDPOINTS
    assert "list_tree" in CODE_HOST
    assert "git-trees" in CODE_HOST["list_tree"]

    host = Refuser()
    view = TrackerView(host, tracker(github_server()))
    assert view.list_tree() == ["README.md", "src/app.py"]


def test_the_code_host_calls_are_the_code_hosts():
    """AC5: pull requests, checks and merges still go direct, and say why."""
    for name, reason in CODE_HOST.items():
        assert reason, f"{name} is on the direct path with no reason written down"
        assert name not in TRACKER_ENDPOINTS


# --------------------------------------------------------------------------
# The fourth acceptance criterion
# --------------------------------------------------------------------------


def test_the_github_tracker_needs_no_credential_the_code_host_does_not_hold():
    """AC4, and #143's credential-overlap finding as a test rather than a claim.

    `issues: write` is already in `security.REQUIRED_PERMISSIONS`, the local
    stdio server takes a fine-grained PAT from its own environment, and
    `api.github.com` is already on the egress allowlist. So the GitHub tracker is
    the same token in a second use: no new variable, no header, no new hole.
    """
    from swarm.mcp.contract import parse_tracker as _parse

    contract = _parse("tracker: { mcp: github, args: { owner: a, repo: b } }")

    assert contract.auth.value_env == "GITHUB_TOKEN"
    # Into the server's own environment, never a header: the credential reaches
    # a subprocess and no request apiary makes carries it.
    assert contract.auth.server_env == "GITHUB_PERSONAL_ACCESS_TOKEN"
    assert contract.is_stdio
    assert contract.command == ("github-mcp-server", "stdio")


def test_the_shipped_github_intake_is_not_the_semantic_one():
    """ADR 0004 decision 4: `search_issues` is a non-deterministic control input."""
    assert PROFILES["github"]["intake"]["tool"] == "list_issues"
    with pytest.raises(ContractError, match="semantic"):
        parse_tracker("tracker: { mcp: github, intake: { tool: search_issues } }")


# --------------------------------------------------------------------------
# What the seam refuses, and why each refusal exists
# --------------------------------------------------------------------------


def test_intake_refuses_a_filter_rather_than_translating_it():
    """ADR 0004 decision 2: the filters are the server's own parameter names.

    Translating `state="open"` into whatever this tracker calls it is the adapter
    this epic deletes, written one keyword at a time. Dropping it silently would
    answer an "open issues" question with every item in the project.
    """
    view = TrackerView(Refuser(), tracker(github_server(list_issues=[])))

    with pytest.raises(NoSuchCapability, match="intake.args"):
        view.list_issues(state="open")
    with pytest.raises(NoSuchCapability, match="intake.args"):
        view.list_issues(state="all", labels=["swarm:ready"])


def test_intake_is_the_whole_answer_and_readiness_stops_fetching():
    """The closed capability set, expressed as a rule rather than as a refusal.

    `resolve_states` falls back to one fetch per ref the listing did not carry.
    There is nothing to fetch *with* over MCP, so the fallback is removed rather
    than redirected: an item intake did not list is one apiary does not act on.
    `Refuser` proves the removal - not that the fetch failed, but that nothing
    made it.
    """
    host = Refuser()
    server = github_server(list_issues=[issue(4)])
    snapshot = Snapshot(TrackerView(host, tracker(server)))  # type: ignore[arg-type]

    states = resolve_states(snapshot, [task_ref(4), task_ref(101)])  # type: ignore[arg-type]

    assert states[task_ref(4)].exists
    assert not states[task_ref(101)].exists
    assert host.fetched == []


def test_a_plain_client_still_fetches_the_ref_the_listing_missed():
    """The direct path is unchanged: the probe is absent, so the fallback runs.

    `issues=[]` stands in for the listing, so this measures the *fallback* rather
    than reaching `Refuser.list_issues`, which fails the test by design.
    """
    host = Refuser()
    states = resolve_states(host, [task_ref(101)], issues=[])  # type: ignore[arg-type]

    assert host.fetched == [101]
    assert states[task_ref(101)].exists


def test_reading_an_items_comments_is_refused_rather_than_served_directly():
    """#148 stopped the only caller; a reader that comes back must be loud."""
    view = TrackerView(Refuser(), tracker(github_server()))
    with pytest.raises(NoSuchCapability, match="#148"):
        view.list_issue_comments(7)


def test_create_refuses_a_field_that_belongs_in_the_block():
    """A keyword dropped here would be a field the caller believed it had set."""
    view = TrackerView(Refuser(), tracker(github_server(issue_write={"number": 1})))
    with pytest.raises(NoSuchCapability, match="create.args"):
        view.create_issue("t", body="b", milestone=3)


# --------------------------------------------------------------------------
# Reading what a server returned
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param([{"number": 4}], id="a bare array"),
        pytest.param({"items": [{"number": 4}]}, id="an object wrapping one array"),
        pytest.param({"issues": [{"number": 4}], "totalCount": 1}, id="one array beside a scalar"),
    ],
)
def test_intake_reads_both_shapes_a_server_may_answer_in(payload: Any):
    """Structural, never a table of key names.

    A rule matching `items`/`issues`/`nodes`/`data` is a rule that grows an entry
    per tracker - the per-tracker code this epic exists to delete, spelled as a
    dictionary.
    """
    server = github_server(list_issues=payload)
    assert tracker(server).intake() == [{"number": 4}]


def test_intake_reads_a_structured_answer_as_well_as_a_json_text_block():
    """Servers differ on whether they publish an `outputSchema`. Both must read."""
    server = github_server(list_issues=[{"number": 4}])
    server.structured = True
    assert tracker(server).intake() == [{"number": 4}]


def test_two_arrays_are_refused_rather_than_guessed_at():
    """A ledger built from the wrong list reads as a repository where nothing is planned."""
    server = github_server(list_issues={"issues": [{"number": 4}], "pulls": [{"number": 9}]})
    with pytest.raises(TrackerError, match="2 array field"):
        tracker(server).intake()


def test_an_unreadable_answer_is_refused_rather_than_read_as_nothing():
    """"Empty" and "unreadable" are different answers, and one of them empties the ledger."""
    server = github_server()
    server.results["list_issues"] = None
    with pytest.raises(TrackerError, match="carries no list"):
        tracker(server).intake()


def test_a_server_that_reports_the_call_failed_is_a_tracker_error():
    """`isError` is how a real server says a write did not happen.

    A caller that only handled JSON-RPC errors would post comments nobody
    received and record the task as reported.
    """
    server = github_server()
    server.tool_errors["add_issue_comment"] = "no such issue"
    with pytest.raises(TrackerError, match="no such issue"):
        tracker(server).comment(7, "hello")


def test_a_comment_the_tracker_could_not_carry_does_not_end_the_cycle(capsys):
    """`post_comment`'s discipline, which is why `TrackerError` is not an `McpError`.

    A comment is an explanation, never a prerequisite. The text is printed so
    the reason for a `swarm:failed` label is not lost entirely.
    """
    server = github_server()
    server.tool_errors["add_issue_comment"] = "no such issue"
    view = TrackerView(Refuser(), tracker(server))

    assert post_comment(view, 7, "apiary: needs a human") is False
    assert "comment on #7 failed" in capsys.readouterr().err


def test_a_misconfigured_ref_rule_reads_as_a_tracker_failure_at_the_call_site():
    """A block that cannot be honoured must not read differently from a dead server.

    Not at the call site that is only trying to post a comment, at least: both
    are "the tracker did not take this", both are survivable, and a
    `ContractError` escaping `post_comment` would take the cycle with it.
    """
    server = github_server(list_issues=[{"title": "no number here"}])
    contract = parse_tracker(GITHUB_BLOCK, source="tracker.yaml")
    items = Tracker(contract=contract, client=mcp_client(server, token="t")).intake()

    with pytest.raises(ContractError, match="intake.ref names 'number'"):
        contract.task_ref(items[0])


# --------------------------------------------------------------------------
# A whole cycle, on the MCP path
# --------------------------------------------------------------------------


def _reconciler(view: TrackerView) -> Reconciler:
    """A cycle with every model seam scripted and both write gates off.

    The merge gate and the goal gate are off because neither is what this file
    measures and both would need a pull request and an objective to say anything.
    What is left is the whole read-reconcile-readiness spine, which is where every
    tracker call in a cycle is made.
    """
    return Reconciler(
        run=Run.start(SLUG, "reconcile the ledger", run_id=RUN_ID),
        client=view,  # type: ignore[arg-type]
        store=SqliteTaskStore.open(SLUG),
        base_commit=BASE_COMMIT,
        capacity=Capacity(slots=3, configured=2),
        oracle=Judged(),
        goal_gate=False,
        merge_gate=False,
        sleep=lambda _seconds: None,
    )


def test_a_whole_cycle_runs_with_the_code_hosts_issue_endpoints_refused():
    """The strongest form of the negative assertion available without a credential.

    Not a call into one collaborator: `Reconciler.cycle` end to end - the ledger
    read, the reconcile rules, the belief, readiness, the label writes - against
    a code host whose `list_issues`, `create_issue`, `create_issue_comment` and
    `list_issue_comments` all fail the test. If any rule anywhere in a cycle
    reaches past `mcp.TrackerView` for a work item, this is the test that says so,
    and it says which call it was.

    #151's fifth acceptance criterion - five consecutive greenfield runs through
    this path - **cannot be met in the apiary development environment** and is
    not closed by this test. `swarm run --new` refuses classic and OAuth tokens
    by design (`security.assert_provision_token`), no fine-grained PAT is
    available here, and `docs/demo-run.md` records the same wall from
    2026-08-14. This is the verification that does not need a credential: the
    code path is exercised whole, against a server, with the direct path booby
    trapped.
    """
    host = Refuser()
    # One ready and one blocked-with-nothing-to-wait-for, so readiness has a
    # relabel to write: a cycle that only read would not prove the label plane
    # still reaches the code host from inside one.
    server = github_server(list_issues=[issue(4), issue(7, label=BLOCKED)])
    view = TrackerView(host, tracker(server))

    report = _reconciler(view).cycle()

    # The cycle saw both tasks, and it saw them through the tracker.
    assert {str(entry.ref) for entry in report.ledger.entries.values()} == {"#4", "#7"}
    assert [name for name, _ in server.calls] == ["list_issues"]
    # It got all the way to readiness, which is the last step of a cycle that
    # reads the ledger - so nothing in between declined to run.
    assert report.readiness is not None
    assert {str(verdict.ref) for verdict in report.readiness.verdicts} == {"#4", "#7"}
    # And the label plane wrote, through the view, to the code host. #152 is the
    # ticket that removes these two calls rather than routes them.
    assert host.labels == {7: [READY]}


def test_five_consecutive_cycles_stay_on_the_mcp_path():
    """One intake call per cycle, five cycles, and the direct path never used.

    The acceptance criterion this stands in for asks for five consecutive
    *greenfield runs*, which needs a credential this environment does not have.
    What it can assert is the property those runs would be measuring: the path
    does not drift after the first pass - no rule caches its way onto the direct
    path, and no cycle spends two intake calls where it spent one.
    """
    host = Refuser()
    server = github_server(list_issues=[issue(4)])
    reconciler = _reconciler(TrackerView(host, tracker(server)))

    for _ in range(5):
        reconciler.cycle()

    assert len(server.arguments_for("list_issues")) == 5
    assert {name for name, _ in server.calls} == {"list_issues"}


# --------------------------------------------------------------------------
# The one construction site
# --------------------------------------------------------------------------


def test_an_installation_with_no_tracker_block_gets_the_client_it_always_got():
    """`None` is a normal answer, not a failure. apiary runs on labels until #152.

    The whole no-flag-day property is this line: a tree with no tracker block is
    unchanged by #151, so nothing in the epic's fourth movement had to land as a
    cutover somebody schedules.
    """
    host = Refuser()
    chosen, tracker_ = view_for(host, env={})

    assert chosen is host
    assert tracker_ is None


def test_a_configured_block_wraps_the_client_once_for_the_whole_run(tmp_path: Path):
    """And the wrap is what every collaborator is then handed.

    Constructed, not connected: a stdio server is spawned on first use, so this
    asserts the wiring without starting `github-mcp-server`.
    """
    block = tmp_path / "tracker.yaml"
    block.write_text("tracker: { mcp: github, args: { owner: acme, repo: widgets } }")
    host = Refuser()

    chosen, tracker_ = view_for(host, path=str(block), env={"GITHUB_TOKEN": "ghs_x"})

    assert isinstance(chosen, TrackerView)
    assert chosen.client is host
    assert tracker_ is not None and tracker_.contract.mcp == "github"
    # Every code-host call still reaches the client it wrapped.
    assert chosen.list_tree() == ["README.md", "src/app.py"]


def test_a_named_block_that_is_not_there_is_a_refusal_and_not_a_silent_fallback(tmp_path: Path):
    """The misconfiguration that would otherwise present as a tracker being ignored."""
    with pytest.raises(ContractError, match="no such file"):
        view_for(Refuser(), path=str(tmp_path / "absent.yaml"), env={})
