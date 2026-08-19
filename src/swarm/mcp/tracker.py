"""The three tracker calls, made through the MCP client. **The only place they are made.**

`client.py` can call any tool and knows what none of them mean; `contract.py`
says which tool means what. This module is the third piece and the one #151 is
about: it spends the contract, so that intake, comment and create leave apiary
over MCP rather than over `GitHubClient`'s issue endpoints.

**Why a client-shaped wrapper rather than a new interface.** A cycle already
passes one object where a `GitHubClient` is expected - `reconcile.Snapshot`,
which caches the issue listing and delegates everything else - and every
tracker-shaped call in a cycle goes through it: `load_ledger` reads the
listing, `readiness.resolve_states` reads it again, `post_comment` probes it for
a comment method, and `goal`/`replan` reach `planner._create` still holding it.
So the cheapest true seam is another object of that shape, underneath it:
`TrackerView` overrides exactly the calls the contract covers and delegates the
rest to the code host. Nothing above it changes signature, which is what makes
"the tracker path is the MCP path" a property of one wrapper rather than a
promise spread over nine modules.

**The code host is untouched, and that is a decision rather than an oversight.**
Pull requests, check runs, merges, branches and `list_tree` are GitHub because
GitHub *is* the code host in this design, not because they were left behind.
`list_tree` is the one that looks like a tracker read and is not: it landed in
#161 so the planner can see the repository it plans against, and it is a
git-trees call. `CODE_HOST` below is the list, and it exists so that the
question "is this call the tracker's?" has a written answer.

## What this module may not do, and what enforces it

**Nothing here tests which server is configured.** No `if contract.mcp ==
"github"`, no branch on a tool name, no per-tracker special case - that is
#151's second acceptance criterion and the whole reason the ticket exists,
because a seam with one implementation and one `if` for it is not a seam.
Everything server-specific is in `PROFILES` and in the customer's block, which
are data. `tests/test_tracker_path.py` asserts the absence statically, because
a property this easy to violate in a hurry needs something other than good
intentions.

**Nothing here parses `intake.args`.** The filters are the server's own
parameter names, forwarded verbatim (ADR 0004 decision 2). A caller asking for
`list_issues(state="open")` is therefore *refused* rather than translated: the
argument dict is the contract's, and a view that quietly mapped apiary's idea of
"open" onto whatever this server calls it would be the adapter this epic
deletes, written one keyword at a time.

**Nothing here mints a task ref.** Identity is `TaskRef`, minted by the adapter
that spells it (`github/refs.py`, #142), and the payloads intake returns still
go through that adapter. The contract's `ref` rule is used where a *call* needs
the item's own id back - which is why `Capability.arguments` passes it through
untouched and this module does not stringify it.

## The two gaps this ticket found, both named rather than papered over

**1. Intake is one page.** The #143 spike's own note - "pagination diverges, so
the client owns pagination rather than passing it through" - has no slot in
ADR 0004's capability shape, which is `{tool, args, fields, ref}` and carries
nothing about paging. So `intake()` makes one call, and a project with more
tracker items than one page truncates its ledger *silently*, which for a ledger
is the worst available failure: a task nothing lists is a task nothing ever
looks at again. `perPage` belongs in the customer's `intake.args` until the
contract grows a paging rule, and `PAGING_GAP` below is the sentence a reader
needs at the moment they hit it.

**2. There is no "fetch one item" capability, by decision.** ADR 0004 closes
the set at three, and `readiness.resolve_states` has a `get_issue` fallback for
refs the listing did not carry. So that fallback is *removed* on this path
rather than redirected - `INTAKE_IS_AUTHORITATIVE` is the property readiness
probes for - because intake's answer is the whole answer and an item intake did
not list is one apiary does not act on. The consequence is written down in
`resolve_states`: a `## Blocked by` line naming a pull request resolves to
*missing* here where the direct path identified it.

`get_issue` itself stays on the direct path, and `LABEL_PLANE` says why at
length: every remaining caller is the attempt marker's read-modify-write, which
§5 requires to be a fresh read and which #152 deletes.

Manual check against a configured tracker - one intake call, writes nothing:

    APIARY_TRACKER_CONFIG=.swarm/tracker.yaml python -m swarm.mcp.tracker
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..config import SETTINGS, TRACKER_CONFIG_ENV
from .client import McpClient, McpError
from .contract import (
    COMMENT,
    CREATE,
    INTAKE,
    ContractError,
    TrackerContract,
    client_for,
    load_tracker,
)

__all__ = [
    "CODE_HOST",
    "INTAKE_IS_AUTHORITATIVE",
    "LABEL_PLANE",
    "PAGING_GAP",
    "TRACKER_ENDPOINTS",
    "NoSuchCapability",
    "Tracker",
    "TrackerError",
    "TrackerView",
    "main",
    "tracker_for",
    "view_for",
]


#: The client methods this view answers, and therefore the whole tracker
#: surface as far as the orchestrator is concerned. Named as data because
#: `tests/test_tracker_path.py` asserts over it: any tracker-shaped call the
#: orchestrator makes that is *not* on this list is a call that reaches
#: `GitHubClient` unrouted, which is exactly #151's negative assertion and
#: exactly the regression a later ticket adds without noticing.
#:
#: `list_issue_comments` is on the list and answered with a *refusal* for that
#: reason. The orchestrator has no caller for it; the worker still reads retry
#: feedback with it (`worker/entrypoint.fetch_feedback`) and does so with its own
#: client, in its own container, holding no tracker credential by design - so a
#: caller appearing on this side must be refused rather than quietly served by
#: the code host behind a configured tracker's back.
TRACKER_ENDPOINTS: tuple[str, ...] = (
    "list_issues",
    "create_issue",
    "create_issue_comment",
    "list_issue_comments",
)

#: What stays on the direct path, with the reason each one is not a tracker
#: read. This is the sixth acceptance criterion of #151 written as data, and
#: `list_tree` is the entry it exists for: it *looks* like a tracker read, it
#: arrived in #161 so the planner can see the repository it is planning
#: against, and routing a git-trees call through a tracker would be a category
#: error.
CODE_HOST: dict[str, str] = {
    "get_pull_request": "pull requests are the code host's",
    "list_pull_requests": "pull requests are the code host's",
    "create_pull_request": "pull requests are the code host's",
    "update_pull_request": "pull requests are the code host's",
    "merge_pull_request": "the merge is the code host's",
    "list_check_runs": "check runs are the code host's",
    "list_workflow_runs": "check runs are the code host's",
    "delete_branch": "branches are the code host's",
    "head_sha": "a commit is the code host's",
    "get_repo": "the repository is the code host's",
    "list_tree": "a git-trees read: the planner seeing the repository it plans against (#161)",
}

#: The label calls and the issue-body `PATCH`. Not tracker capabilities and
#: deliberately not routed: they are the `swarm:*` control plane, which ADR 0001
#: forbids writing into a customer's tracker at all and #152 removes. Routing
#: them would mean giving apiary's own vocabulary a place in the capability
#: contract days before it is deleted - and `CANONICAL_FIELDS` is closed for
#: precisely that reason.
#:
#: Public, and named as data, because it is the honest half of #151's first
#: acceptance criterion: three capabilities go through the MCP client and *these
#: calls still do not*, because they are not tracker capabilities and have one
#: ticket left to live. `tests/test_tracker_path.py` partitions `GitHubClient`'s
#: whole issue surface across this list, `TRACKER_ENDPOINTS`, `CODE_HOST` and one
#: entry that is nobody's - so a method added to that client tomorrow has to be
#: classified rather than quietly joining the direct path.
#:
#: **`get_issue` is on this list, and finding out why is what the static scan in
#: that file was for.** It reads like a tracker read, and its one caller in the
#: orchestrator is the marker's read-modify-write: `reconcile.bump_attempt`
#: fetches the issue to rewrite the `<!-- apiary ... -->` marker in its body and
#: put it back. `docs/issue-contract.md` §5 *requires* that read to be fresh,
#: because a human editing in between would otherwise have their edit
#: overwritten by a patch built from a stale copy - so it cannot be served from
#: the cycle's cached listing, and the three-capability contract has nothing to
#: serve it with either. Routing it to a refusal would therefore not move the
#: marker onto MCP; it would stop the attempt counter being written at all,
#: which grants every failing task an unbounded retry budget. It goes direct,
#: with the rest of the plane it belongs to.
#:
#: **One caller, and it used to be three.** `checks._patch_body` and
#: `mergeability._patch_body` each did the same fetch-rewrite-put in order to
#: carry a CI failure or a conflict alongside the counter. #152 deleted those
#: blocks - nothing read them - and with the extra payload gone both functions
#: *were* `bump_attempt`, so they collapsed into it. The classification above is
#: unchanged and its reason is now narrower and stronger: the read this list
#: exists to protect happens in one function, so the ticket that removes the
#: marker removes it in one place.
#:
#: The read `resolve_states` used it for - genuinely a tracker read - is handled
#: instead by `INTAKE_IS_AUTHORITATIVE`, which removes the call rather than
#: redirecting it.
LABEL_PLANE: tuple[str, ...] = ("add_labels", "remove_label", "get_issue", "update_issue")

#: The attribute `readiness.resolve_states` probes for, and `TrackerView` sets.
#:
#: Its fallback exists because a `## Blocked by` ref may name something the
#: listing did not carry, and on the direct path one extra fetch settles it. On
#: the MCP path there is nothing to fetch *with*, so the honest answer is not a
#: different call but a different rule: **intake's answer is the whole answer**,
#: and an item intake did not list is one apiary does not act on. Naming that as
#: a property of the source, rather than teaching readiness what a tracker is,
#: is what keeps `github/` free of any opinion about MCP.
INTAKE_IS_AUTHORITATIVE = "intake_is_authoritative"

#: Named so the sentence appears once. See the module docstring, gap 1.
PAGING_GAP = (
    "intake is one call and the capability contract carries no paging rule "
    "(ADR 0004 is {tool, args, fields, ref}), so a tracker holding more items "
    "than one page returns a partial ledger. Raise the page size in "
    "intake.args - `perPage` on GitHub, `limit` on Linear - until the contract "
    "grows one."
)


class TrackerError(RuntimeError):
    """A tracker call that did not happen, in the shape its callers already catch.

    Deliberately *not* an `McpError`. Every site that survives a failed comment
    catches `GitHubError` today (`reconcile.post_comment`,
    `planner._post_comment`), and those sites are the reason a `ContractError`
    explanation reaches an issue rather than a traceback reaching the operator.
    They now catch this alongside it, so one type covers "the tracker refused"
    however the tracker is reached, and a transport exception cannot end a cycle
    that was only trying to explain itself.
    """


class NoSuchCapability(TrackerError):
    """A call the contract has no capability for. See the module docstring, gap 2.

    Its own type because one caller *handles* it - `readiness.resolve_states`
    reads it as "intake did not list this ref" - and a caller that had to match
    on a message would be a caller that breaks when the message improves.
    """


# --------------------------------------------------------------------------
# The three calls
# --------------------------------------------------------------------------


@dataclass
class Tracker:
    """One organization's task system, reached over MCP. Three verbs, no opinions.

    Holds the contract and a client and adds nothing of its own: each method is
    "build the argument dict the contract describes, call the tool it names,
    read the items out of what came back". The absence of anything else here is
    the load-bearing part - see the module docstring on what this module may not
    do.
    """

    contract: TrackerContract
    client: McpClient

    # --- intake ----------------------------------------------------------

    def intake(self) -> list[dict[str, Any]]:
        """Every item the configured filters select, as the server returned them.

        **No arguments.** The filters are `intake.args` in the customer's block
        and are forwarded verbatim; a caller that could pass one would be a
        caller translating apiary's vocabulary into this server's, which is the
        adapter ADR 0001 deletes. One call - see `PAGING_GAP`.

        The payloads are handed on untouched. They are still read by the GitHub
        adapter above (`ledger.load_ledger`, `IssueState.from_payload`), which
        is the pre-existing shape and the follow-up epic's to normalise; what
        #151 changes is where they came from.
        """
        return [dict(item) for item in _items(self._call(INTAKE))]

    # --- comment ---------------------------------------------------------

    def comment(self, ref: Any, body: str) -> None:
        """Post one comment against the item `ref` identifies.

        `ref` is the item's id **in the server's own spelling** - the value the
        contract's `ref` rule reads out of an intake payload, not apiary's
        `TaskRef`. It is passed through untouched, because `issue_number` is an
        integer on GitHub and `issueId` is a uuid on Linear and a view that
        helpfully rendered both as strings would break the first one
        (`Capability.arguments`).
        """
        self._call(COMMENT, {"ref": ref, "body": body})

    # --- create ----------------------------------------------------------

    def create(
        self, title: str, *, body: str = "", labels: Sequence[str] = ()
    ) -> dict[str, Any]:
        """File one work item and return it as the server described it.

        One call, never create-then-label: `planner._create` chose that on
        purpose, because a crash between the two leaves an issue with no state
        label, which `docs/issue-contract.md` §3 reads as outside the ledger
        entirely - work that exists in the tracker and that nothing will ever
        look at again.

        Which is why `labels` is passed as an argument named `labels` and is
        **not** a canonical field. ADR 0004 closes apiary's vocabulary at
        `ref`, `title` and `body`, and it is closed for a reason that applies
        here exactly: the `swarm:*` labels are apiary's own vocabulary, ADR 0001
        forbids writing them into a customer's tracker, and #152 deletes them.
        Giving them a mappable slot in the contract would be building the thing
        being removed. Identity-named and pass-through means the argument
        reaches a server that spells it `labels` and reaches no server that does
        not - which is the correct blast radius for a value that has days to
        live.
        """
        values: dict[str, Any] = {"title": title, "body": body}
        if labels:
            values["labels"] = list(labels)
        return _item(self._call(CREATE, values))

    # --- the one call all three make -------------------------------------

    def _call(self, capability: str, values: Mapping[str, Any] | None = None) -> Any:
        """Spend one capability, and turn any transport failure into `TrackerError`.

        The single point at which this module talks to `client.py`, so the
        translation of an MCP failure into the shape callers already handle
        happens once. `ContractError` is translated too: a capability whose
        `ref` rule cannot be honoured is a misconfigured block, and a
        misconfigured block must not read differently from an unreachable
        server at the call site that is only trying to post a comment.
        """
        capability_ = self.contract.capability(capability)
        try:
            result = self.client.call_tool(
                capability_.tool, self.contract.arguments(capability, values)
            )
        except (McpError, ContractError) as exc:
            raise TrackerError(
                f"{self.contract.source}: {capability} through {capability_.tool} on "
                f"{self.contract.endpoint} did not happen: {exc}"
            ) from exc
        return result.structured if result.structured is not None else _decode(result, capability_.tool)

    # --- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Let go of the transport. A stdio server is a subprocess; it must end."""
        self.client.close()

    def summary(self) -> str:
        """One line for the operator, at the moment a run chooses this path."""
        return (
            f"tracker: {self.contract.mcp} over MCP at {self.contract.endpoint} "
            f"({self.contract.source})"
        )


# --------------------------------------------------------------------------
# Reading what a server returned
# --------------------------------------------------------------------------


def _decode(result: Any, tool: str) -> Any:
    """A tool result's text as JSON, for a server that publishes no `outputSchema`.

    `structuredContent` is preferred by the caller and this is the fallback,
    because it is the common case: an MCP server that answers in one text block
    of JSON is not doing anything unusual, and refusing to read it would make
    every such tracker unusable over a field the spec leaves optional.
    """
    text = (result.text or "").strip()
    if not text:
        raise TrackerError(
            f"{tool} returned no structured content and no text. apiary cannot tell "
            f"an empty answer from an unreadable one, and a tracker read that "
            f"guessed 'nothing' would empty the ledger."
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise TrackerError(
            f"{tool} returned neither structured content nor JSON text ({exc}). "
            f"apiary reads a tool result as the server's own JSON and parses nothing "
            f"else; the first 200 characters were: {text[:200]!r}"
        ) from exc


def _items(payload: Any) -> list[Mapping[str, Any]]:
    """The list of items inside whatever shape a server wrapped them in.

    Two shapes, and the rule is structural rather than a table of key names: a
    bare array is the items, and an object carrying exactly one array is an
    envelope around them. Both servers in scope answer in one of those, and a
    rule that matched on `items`/`issues`/`nodes`/`data` would be a rule that
    has to grow an entry per tracker - which is the per-tracker code this epic
    exists to delete, spelled as a dictionary.

    An object with two arrays is refused rather than guessed at. The failure of
    guessing is a ledger built from the wrong list, which reads as a repository
    where nothing is planned.
    """
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        arrays = {
            key: value
            for key, value in payload.items()
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        }
        if len(arrays) == 1:
            return _items(next(iter(arrays.values())))
        raise TrackerError(
            f"intake returned an object with {len(arrays)} array field(s) "
            f"({', '.join(sorted(arrays)) or 'none'}), so which one holds the work "
            f"items is a guess. apiary reads a bare array, or an object wrapping "
            f"exactly one."
        )
    raise TrackerError(
        f"intake returned {type(payload).__name__}, which carries no list of work "
        f"items. apiary reads a bare array, or an object wrapping exactly one."
    )


def _item(payload: Any) -> dict[str, Any]:
    """The one item a write returned - the created work item, as the server has it.

    A server that answers a create with the object is the common case; one that
    wraps it in a single-element array is read the same way, for `_items`'
    reason. Anything else is refused, because the caller needs the new item's
    ref out of this and a create whose answer could not be read is a work item
    that exists and that apiary cannot address.
    """
    if isinstance(payload, Mapping):
        return dict(payload)
    items = _items(payload)
    if len(items) == 1:
        return dict(items[0])
    raise TrackerError(
        f"create returned {len(items)} item(s) rather than the one it filed, so the "
        f"new work item's ref cannot be read back. The item may exist; check the "
        f"tracker before retrying."
    )


# --------------------------------------------------------------------------
# The view a cycle holds
# --------------------------------------------------------------------------


class TrackerView:
    """A client-shaped object whose tracker calls are the tracker's. **The seam.**

    `reconcile.Snapshot`'s idiom, one layer down and for the same reason: the
    calls this object *changes* are written out, everything else falls through
    to the code host untouched. Wrapping rather than replacing is what makes
    #151 a change of one construction site instead of a signature change
    through `reconcile`, `readiness`, `checks`, `mergeability`, `goal`,
    `replan`, `planner` and `recovery` - none of which learn that a tracker
    exists.

    Construct one per run, in `cli`, and hand it wherever a `GitHubClient` went.
    `Snapshot` then sits *above* it, so a cycle still costs one intake call and
    still shares it with every reader in the cycle.
    """

    def __init__(self, client: Any, tracker: Tracker) -> None:
        self.client = client
        self.tracker = tracker

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"TrackerView({self.client!r}, {self.tracker.contract.mcp})"

    # --- intake ----------------------------------------------------------

    def list_issues(self, *, state: str = "open", **kwargs: Any) -> list[dict[str, Any]]:
        """Intake. `state="all"` only, and every other filter is refused.

        Refused rather than translated, and refused rather than ignored. The
        filters live in `intake.args` and are the server's own parameter names
        (ADR 0004 decision 2), so there is nowhere for a keyword from here to
        go: mapping `state="open"` onto whatever this tracker calls it is the
        adapter this epic deletes, and dropping it silently would answer an
        "open issues" question with every item in the project.

        `state="all"` is the one accepted spelling because it is the only one a
        cycle asks for and because it asks for no narrowing at all - the whole
        intake answer, which is exactly what the configured filters already
        describe. It is a claim about what apiary wants, not a value forwarded
        anywhere.
        """
        if state != "all" or kwargs:
            raise NoSuchCapability(
                f"intake through {self.tracker.contract.mcp} takes its filters from "
                f"intake.args in {self.tracker.contract.source}, so list_issues"
                f"(state={state!r}"
                + (f", {', '.join(sorted(kwargs))}" if kwargs else "")
                + ") cannot be honoured: apiary does not translate its own vocabulary "
                "into a tracker's parameter names (ADR 0004). Put the filter in the "
                "block and re-check with:\n"
                f"    python -m swarm.mcp.contract {self.tracker.contract.source}"
            )
        return self.tracker.intake()

    #: Read by `readiness.resolve_states` in place of its per-ref fetch. See
    #: `INTAKE_IS_AUTHORITATIVE` for why the call is removed rather than routed.
    intake_is_authoritative = True

    # --- comment ---------------------------------------------------------

    def create_issue_comment(self, number: Any, body: str) -> dict[str, Any]:
        """Comment. `number` is the item's own id, passed through untouched.

        The name and the positional shape are `GitHubClient`'s, because this
        object stands where one stood and `reconcile.post_comment` probes for
        exactly this attribute. What arrives is whatever the caller was holding
        - an issue number today, because the ledger read it out of an intake
        payload - and it reaches the server under the name the contract's `ref`
        rule gives it.

        Returns an empty mapping rather than the comment: no caller reads one,
        and inventing a shape for a value nobody uses is how a seam acquires an
        opinion.
        """
        self.tracker.comment(number, body)
        return {}

    def list_issue_comments(self, number: Any) -> list[dict[str, Any]]:
        """Refused, deliberately. #148 stopped the only caller.

        On `TRACKER_ENDPOINTS` and unimplemented for the reason given there: a
        reader that comes back must be refused loudly rather than served by the
        code host behind the tracker's back.
        """
        raise NoSuchCapability(
            f"reading an item's comments is not one of the three capabilities ADR 0004 "
            f"defines ({INTAKE}, {COMMENT}, {CREATE}). The worker stopped reading them "
            f"in #148; nothing in apiary needs this."
        )

    # --- create ----------------------------------------------------------

    def create_issue(
        self,
        title: str,
        *,
        body: str = "",
        labels: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create. `GitHubClient.create_issue`'s signature, because it stands there.

        Extra keywords are refused rather than dropped: everything a create can
        carry beyond title, body and labels - assignees, a milestone, an issue
        type - is a constant that belongs in `create.args` in the customer's
        block, and a keyword silently discarded here would be a field the caller
        believed it had set.
        """
        if kwargs:
            raise NoSuchCapability(
                f"create through {self.tracker.contract.mcp} takes title, body and "
                f"labels; {', '.join(sorted(kwargs))} would have to be a constant in "
                f"create.args in {self.tracker.contract.source}."
            )
        return self.tracker.create(title, body=body, labels=list(labels or ()))

    # --- everything else is the code host's ------------------------------

    def __getattr__(self, name: str) -> Any:
        """Pull requests, checks, merges, branches, `list_tree` - the direct path.

        `Snapshot.__getattr__`'s shape and its recursion guard, for its reasons.
        The label calls and the issue-body `PATCH` fall through here too and are
        listed in `LABEL_PLANE` with why: they are the `swarm:*` control plane,
        not a tracker capability, and #152 removes them rather than routing
        them.
        """
        if name.startswith("_") or name in ("client", "tracker"):
            raise AttributeError(name)
        return getattr(self.client, name)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def tracker_for(
    contract: TrackerContract,
    *,
    env: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> Tracker:
    """A `Tracker` on the server this contract names.

    `env` defaults to the process environment rather than to nothing, which is
    the opposite of `client_for`'s default and deliberately so: that function's
    caller is `doctor`, which wants to probe reachability with no credential at
    all, and this function's caller is a run, which cannot start without one.
    """
    environment = dict(os.environ if env is None else env)
    return Tracker(contract=contract, client=client_for(contract, env=environment, **kwargs))


def view_for(
    client: Any,
    *,
    path: str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[Any, Tracker | None]:
    """`(what to hand every collaborator, the tracker or None)`.

    The one construction site, so that "is this run on the MCP path?" is decided
    once and announced once. `None` is a normal answer and not a failure: apiary
    still runs on the label control plane until #152, so an installation with no
    tracker block gets the client it always got, and `cli` prints which of the
    two it is. A block that is *named* and missing raises, because that is a
    misconfiguration that would otherwise present as a tracker quietly not being
    consulted (`load_tracker`).
    """
    contract = load_tracker(path, env=env)
    if contract is None:
        return client, None
    tracker = tracker_for(contract, env=env)
    return TrackerView(client, tracker), tracker


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """One intake call against the configured tracker, and what it returned.

    Reads; writes nothing. `python -m swarm.mcp.contract` answers "is this block
    right", and this answers the question after it - "does the server this block
    names actually hand back work items" - which is the one an operator has at
    the moment a run plans nothing.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        contract = load_tracker(args[0] if args else None)
        if contract is None:
            print(
                f"no tracker configured: {SETTINGS.tracker_config} does not exist and "
                f"{TRACKER_CONFIG_ENV} is unset, so this run would read and write the "
                f"tracker directly (the label control plane, until #152):\n"
                f"    export {TRACKER_CONFIG_ENV}=/path/to/tracker.yaml",
                file=sys.stderr,
            )
            return 1
        tracker = tracker_for(contract)
        try:
            items = tracker.intake()
        finally:
            tracker.close()
    except (ContractError, TrackerError) as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1
    print(tracker.summary())
    print(f"  intake  {contract.intake.tool} -> {len(items)} item(s)")
    for item in items:
        try:
            ref = contract.task_ref(item)
        except ContractError as exc:
            print(f"  ! {exc}", file=sys.stderr)
            return 1
        print(f"    {ref}  {str(item.get('title') or '')[:72]}")
    if items:
        print(f"  note: {PAGING_GAP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
