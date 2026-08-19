"""Which tool on the customer's MCP server fulfils which of apiary's needs.

`client.py` can call any tool on any server and knows what none of them mean.
This module is the other half: a per-organization block of configuration that
says *this* server calls intake `list_issues`, posts a comment with
`add_issue_comment`, and files a ticket with `issue_write`. ADR 0001's whole
claim - that supporting a new tracker costs configuration rather than an
adapter - is true only if that mapping is data. Here it is data.

**And it has to fail at load.** A misconfigured tracker does not present as a
misconfigured tracker. It presents as a run that planned nothing, or claimed
nothing, an hour after it started, because the first cycle that needed the
capability got a `KeyError` or a tool name the server never heard of. Every
refusal below therefore names the field and the fix, and
`python -m swarm.mcp.contract .swarm/tracker.yaml` re-asks the question without
starting a run.

## What #143 corrected, and why the shape here is not ADR 0001's

ADR 0001 sketched `{ tool, args }` per capability. The spike
(`docs/plans/spike-143-mcp-tool-shapes.md`) probed the live servers and found
that shape **cannot make a single real call**:

- GitHub fused create and update behind a discriminator, so `create` must pin
  `method: "create"` as an argument. That one needs no new field - `args` was
  always a static dict - but a contract that carried only a tool name could not
  express it.
- The issue body is `body` on GitHub and `description` on Linear, and no
  argument dict written ahead of time can hold a value the caller supplies at
  the moment of the call. That needs a **field map**.
- Nothing in the sketch says which response field is the durable task ref, and
  the derived state machine that is the point of the whole ADR is built on it:
  `apiary/<ref>-attempt-2` is a branch name. That needs a **ref rule**.

So: two fields per capability, `fields` and `ref`, and the residual divergence
between the two priority trackers is then statable in full - `method: "create"`,
`body` vs `description`, `owner`+`repo` vs `teamId`, and two ref rules. Three of
those four are static values in `args`.

## Three properties this module is built to keep, all three load-bearing

**`args` is opaque and stays opaque.** It is forwarded to the server verbatim
and nothing here parses it, validates it against a tool's published
`inputSchema`, or knows what a key means. That is not laziness: Jira's intake is
`searchJiraIssuesUsingJql`, whose one argument is a free-text query DSL, and its
create needs `cloudId`, `projectKey` and `issueTypeName` - three constants with
no GitHub or Linear counterpart. An `args` narrowed to the parameter names of
the two trackers we happen to support first is an `args` that cannot be widened
to Jira without reopening the design. #143 names this as the single most
important constraint on the page, and it is the reason the validation below
checks the *shape* of `args` and never its contents.

**The field map exists on every capability, and `body` is never assumed.**
GitHub and Linear both spell a comment's text `body`, which is exactly the
coincidence that would tempt a two-tracker design to hard-code it. Jira spells
it `commentBody`. The left-hand side of a field map is apiary's own vocabulary
and is closed - `ref`, `title`, `body`, and nothing else, so a typo is caught
here rather than by a server that quietly ignores an unknown argument. The
right-hand side is the server's and is never checked against anything.

**`auth.scheme` is explicit rather than implied.** Both priority trackers take
a bearer token, and Jira takes HTTP Basic. A contract that assumed Bearer would
read identically and be wrong in a way that only shows up as a 401 from the
third tracker anybody tries.

## Where a credential comes from, and where it does not

#143 probed the authorization-server metadata of every tracker apiary targets
and found `client_credentials` on none of them: there is no machine-to-machine
OAuth flow to drive. So the orchestrator holds a **pre-minted static
credential**, named by `auth.value_env`, minted by the operator out of band. The
contract names the variable; it never obtains the value, and `auth.mint` carries
the command that does, because that command differs per server and a `doctor`
run that says "401" without saying "run this" has moved the confusion rather
than removed it.

Two deliveries, because the two priority trackers need two. A remote server gets
a header. A **local stdio** server gets an environment variable in its own
process - that is the GitHub path, where `github-mcp-server` reads
`GITHUB_PERSONAL_ACCESS_TOKEN` itself, apiary's existing fine-grained PAT is
passed through unchanged, `security.assert_scoped_token`'s guarantee survives,
and no egress hole is opened at all. Note that the spike's recommended GitHub
block shows a `header:` while its recommendation #3 selects the stdio server;
`auth.server_env` is that inconsistency resolved in favour of the recommendation.

## What is deliberately not here

Any per-server *behaviour*. A block that names two fields has no branches, no
error handling and no lifecycle, and adding a tracker still touches no Python.
The bar for reversing that, stated in #143 so it is recognisable when it
arrives: a tracker that needs a two-call create, or a write that must be retried
differently, has run out of declarative contract and reopens the adapter
question. Neither priority tracker needs one, and neither does Jira.

Validate a block without starting anything - reads one file, opens no socket:

    python -m swarm.mcp.contract .swarm/tracker.yaml
"""

from __future__ import annotations

import base64
import difflib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import SETTINGS, TRACKER_CONFIG_ENV, ConfigError, Settings
from .client import STDIO_SCHEME, TRACKER_TOKEN_ENV, McpClient

__all__ = [
    "COMMENT",
    "CREATE",
    "INTAKE",
    "CAPABILITIES",
    "CANONICAL_FIELDS",
    "SCHEMES",
    "PROFILES",
    "INADMISSIBLE_INTAKE",
    "Auth",
    "Capability",
    "ContractError",
    "TrackerContract",
    "client_for",
    "load_tracker",
    "main",
    "parse_tracker",
]


#: The three needs the reconcile loop has of a task system, and the only three.
#: "Which items may the agent pick up", "post the PR link or flag needs-human",
#: "open a work item". Everything else a tracker can do is the customer's
#: workflow, which ADR 0001 decision 2 forbids apiary from modelling at all.
INTAKE = "intake"
COMMENT = "comment"
CREATE = "create"
CAPABILITIES: tuple[str, ...] = (INTAKE, COMMENT, CREATE)

#: apiary's own vocabulary - the left-hand side of a field map, and a closed
#: set. `ref` is on the list because identifying *which* item a call is about
#: diverges as hard as the body field does: GitHub's comment takes
#: `issue_number`, Linear's takes `issueId`, Jira's takes `issueIdOrKey`. A
#: design that mapped only the body would have had to grow a second mechanism
#: for that, and it would have grown it per tracker.
FIELD_REF = "ref"
FIELD_TITLE = "title"
FIELD_BODY = "body"
CANONICAL_FIELDS: tuple[str, ...] = (FIELD_REF, FIELD_TITLE, FIELD_BODY)

#: How the credential reaches an HTTP server. `basic` is here for Jira, whose
#: API-token mode is `base64(email:api_token)` and is not Bearer; the value of
#: `auth.value_env` is then the whole `email:token` pair, encoded here, because
#: apiary holding two halves of somebody's credential buys nothing.
SCHEMES: tuple[str, ...] = ("bearer", "basic", "raw")

#: What a task ref may look like. It is not a display string: ADR 0001 derives
#: execution state from branches named `apiary/<ref>-attempt-2`, so a ref
#: carrying a slash or a space produces a branch nobody meant and a state
#: machine that reads it back wrong. GitHub's `number` and Linear's
#: `identifier` both pass; a `title` configured by mistake does not, which is
#: the misconfiguration this pattern exists to catch.
_BRANCH_SAFE = re.compile(r"\A[A-Za-z0-9._-]+\Z")

#: The top-level keys a tracker block may carry. Unknown keys are refused
#: rather than ignored: `comments:` for `comment:` is a block that loads
#: cleanly, validates, and then fails on the first cycle that needs to report
#: anything - which is the exact failure mode this module exists to move.
_BLOCK_KEYS: tuple[str, ...] = ("mcp", "endpoint", "command", "auth", "args", *CAPABILITIES)
_CAPABILITY_KEYS: tuple[str, ...] = ("tool", "args", "fields", "ref")
_AUTH_KEYS: tuple[str, ...] = ("value_env", "header", "scheme", "server_env", "mint")


class ContractError(ConfigError):
    """A tracker block that cannot be honoured as written.

    A `ConfigError`, so `cli.main`'s existing handler renders it as one `!`
    line and exit 1 - the same treatment every other refusal an operator can
    fix by editing gets. `doctor` catches it instead and reports it as a
    verdict, because a preflight that died of the thing it was asked to measure
    would be the least useful moment for a traceback.
    """


# --------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Auth:
    """Where the credential is, and how it reaches the server.

    Never *what* it is. #143 found no `client_credentials` grant on any tracker
    apiary targets, so there is no flow to drive and nothing to refresh: an
    operator mints a token out of band and exports it, and the contract holds
    the name of the variable. `mint` is the command that does the minting,
    carried here rather than known by `doctor`, because it differs per server
    and it is the whole content of a useful 401.
    """

    value_env: str = TRACKER_TOKEN_ENV
    header: str = "Authorization"
    scheme: str = "bearer"
    #: The variable a *locally spawned* server reads its own credential from.
    #: Set for a stdio contract and meaningless for a remote one.
    server_env: str | None = None
    mint: str = ""

    def credential(self, env: Mapping[str, str]) -> str:
        """The credential, or `""` if the operator has not exported one."""
        return (env.get(self.value_env) or "").strip()

    def header_value(self, credential: str) -> str:
        """`Bearer abc` / `Basic <base64>` / the naked key, per `scheme`."""
        if self.scheme == "basic":
            return "Basic " + base64.b64encode(credential.encode("utf-8")).decode("ascii")
        if self.scheme == "raw":
            return credential
        return f"Bearer {credential}"

    def absent_fix(self) -> str:
        """The sentence a check prints when the variable is not set."""
        mint = f"\n    {self.mint}" if self.mint else ""
        return (
            f"export {self.value_env}=...  - the orchestrator holds a pre-minted static "
            f"credential and drives no OAuth flow (#143), so there is nothing to fall "
            f"back on{mint}"
        )


@dataclass(frozen=True)
class Capability:
    """One need, and the call that meets it on this server.

    `args` is the static half of the call and is **opaque**: forwarded to the
    server exactly as written, never parsed, and never checked against the
    tool's published schema. `fields` renames apiary's canonical fields to this
    server's spelling for the half that is supplied per call. `ref` names the
    response field that carries the durable task ref.
    """

    name: str
    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)
    fields: Mapping[str, str] = field(default_factory=dict)
    ref: str | None = None

    def field_name(self, canonical: str) -> str:
        """This server's spelling of one of apiary's fields.

        Identity when unmapped, which is what makes the two-tracker config
        short: GitHub and Linear both call a comment's text `body`, so neither
        block says so. Jira's `commentBody` is one line in Jira's block, and
        nothing above had to anticipate it.
        """
        return self.fields.get(canonical, canonical)

    def arguments(self, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The static arguments, plus `values` under this server's field names.

        Values are passed through untouched - no coercion, no stringification.
        GitHub's `issue_number` is an integer and Linear's `issueId` is a uuid;
        a contract that helpfully rendered both as strings would break the
        first one.
        """
        call = dict(self.args)
        for canonical, value in (values or {}).items():
            call[self.field_name(canonical)] = value
        return call


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------

#: The two priority trackers, shipped so that a customer's whole block is the
#: handful of constants only they can know. #143's recommendation #2: this is
#: what keeps "a new tracker costs configuration" true in practice rather than
#: on paper, because the alternative - every customer transcribing five tool
#: names out of a vendor's docs - is an adapter written in YAML by somebody
#: with less context than we have.
#:
#: A profile is a partial block, merged *under* the customer's own, so any
#: value here can be overridden by naming it. Nothing in this table is code.
PROFILES: dict[str, Mapping[str, Any]] = {
    # Local stdio, not the remote server, and #143 gives the reason: the remote
    # one at api.githubcopilot.com advertises the classic OAuth scopes -
    # `ghp_`/`gho_`/`ghu_`/`ghr_` - that `security.assert_scoped_token` refuses
    # outright. The local binary takes apiary's existing fine-grained PAT from
    # its own environment and talks to api.github.com, which is already on the
    # egress allowlist. Same credential, second use, no new hole.
    "github": {
        "command": ["github-mcp-server", "stdio"],
        "auth": {
            "value_env": "GITHUB_TOKEN",
            "server_env": "GITHUB_PERSONAL_ACCESS_TOKEN",
            "mint": (
                "mint a fine-grained PAT with issues:write at "
                "https://github.com/settings/personal-access-tokens"
            ),
        },
        # Label filters, and deliberately not `search_issues`. See
        # INADMISSIBLE_INTAKE below - that tool is a semantic match, and this
        # is a control loop.
        "intake": {"tool": "list_issues", "ref": "number"},
        "comment": {"tool": "add_issue_comment", "fields": {"ref": "issue_number"}},
        # The single hardest fact in #143's survey: create and update are one
        # tool behind a discriminator, so naming the tool is not enough.
        "create": {"tool": "issue_write", "args": {"method": "create"}},
    },
    "linear": {
        "endpoint": "https://mcp.linear.app/mcp",
        "auth": {
            "value_env": "APIARY_LINEAR_TOKEN",
            "mint": "mint an API key at https://linear.app/settings/api",
        },
        # `identifier` rather than the uuid: ENG-123 is branch-safe and stable,
        # and a uuid in a branch name is a ref nobody can read back.
        "intake": {"tool": "list_issues", "ref": "identifier"},
        "comment": {"tool": "create_comment", "fields": {"ref": "issueId"}},
        "create": {"tool": "create_issue", "fields": {"body": "description"}},
    },
}

#: Tools that must never be intake on a given server, and why. Data keyed by
#: the server the block names, rather than a branch in the validator, because
#: the moment this becomes an `if profile == "github"` the repository has grown
#: the per-tracker code ADR 0001 exists to delete.
#:
#: GitHub's `search_issues` is documented as natural-language semantic
#: matching: the server runs an inference to decide what matches. ADR 0001
#: requires the reconcile loop to be deterministic, and gives the measurement
#: behind that requirement - roughly 40% of `propose_edits` calls on this host
#: emit broken output, and unlike bad code a bad tracker write is not caught by
#: CI. An intake query whose result set can change without the query changing
#: is a non-deterministic control-loop input.
#:
#: Note this is a rule about one tool on one server, never about a name: Jira's
#: `searchJiraIssuesUsingJql` is a search *and* deterministic, and must stay
#: available when Jira arrives.
INADMISSIBLE_INTAKE: dict[str, dict[str, str]] = {
    "github": {
        "search_issues": (
            "the GitHub MCP server documents `search_issues` as natural-language "
            "semantic matching, so the same query can return a different set on the "
            "next cycle. ADR 0001 requires the reconcile loop to be deterministic"
        ),
    },
}


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackerContract:
    """One organization's task system, as configuration.

    Constructed through `from_mapping` rather than directly, because every
    interesting thing about this type is a refusal: the constructor holds a
    validated block, and `from_mapping` is where a block earns that.
    """

    mcp: str
    endpoint: str
    command: tuple[str, ...] = ()
    auth: Auth = field(default_factory=Auth)
    capabilities: Mapping[str, Capability] = field(default_factory=dict)
    #: Where this came from, quoted in every message. A block that is wrong is
    #: wrong in a file, and naming the file is most of the fix.
    source: str = "<built-in>"

    # --- reading it -------------------------------------------------------

    @property
    def is_stdio(self) -> bool:
        return self.endpoint.lower().startswith(STDIO_SCHEME)

    @property
    def intake(self) -> Capability:
        return self.capabilities[INTAKE]

    @property
    def comment(self) -> Capability:
        return self.capabilities[COMMENT]

    @property
    def create(self) -> Capability:
        return self.capabilities[CREATE]

    def capability(self, name: str) -> Capability:
        try:
            return self.capabilities[name]
        except KeyError:
            raise ContractError(
                f"{self.source}: no capability named {name!r}; this contract has "
                f"{', '.join(sorted(self.capabilities))}"
            ) from None

    @property
    def tools(self) -> tuple[str, ...]:
        """Every tool name this contract requires the server to have.

        Deduplicated and sorted, because a server is asked once whether a name
        exists however many capabilities named it - GitHub's `issue_write`
        fulfils create today and would fulfil an update tomorrow.
        """
        return tuple(sorted({cap.tool for cap in self.capabilities.values()}))

    def arguments(self, capability: str, values: Mapping[str, Any] | None = None) -> dict:
        """The argument dict for one call. `client.call_tool(c.tool, ...)` takes it."""
        return self.capability(capability).arguments(values)

    def ref_rule(self, capability: str = INTAKE) -> str:
        """Which response field carries the task ref, for one capability.

        A capability that does not restate it inherits intake's, because the
        ref rule is a fact about the server rather than about the call: the
        issue GitHub's create returns carries its `number` in the same field
        the one intake listed does.
        """
        rule = self.capability(capability).ref
        return rule or self.intake.ref or ""

    def task_ref(self, item: Mapping[str, Any], capability: str = INTAKE) -> str:
        """The durable task ref out of one item the server returned.

        Refuses rather than returns anything that cannot be half of a branch
        name. ADR 0001 derives claimed and review state from
        `apiary/<ref>-attempt-2`, so a ref with a slash in it is not a cosmetic
        problem - it is a branch that parses back as a different task.
        """
        rule = self.ref_rule(capability)
        if rule not in item:
            raise ContractError(
                f"{self.source}: {capability}.ref names {rule!r}, and the item the server "
                f"returned has no such field (it has: "
                f"{', '.join(sorted(map(str, item))) or 'nothing'}). "
                f"Name the field this server puts the issue's durable identifier in, and "
                f"re-check with:\n    python -m swarm.mcp.contract {self.source}"
            )
        value = str(item[rule])
        if not _BRANCH_SAFE.match(value):
            raise ContractError(
                f"{self.source}: {capability}.ref names {rule!r}, whose value {value!r} cannot "
                f"be part of a branch name - apiary derives execution state from branches "
                f"called apiary/<ref>-attempt-N (ADR 0001). Name a stable identifier field "
                f"instead: `number` on GitHub, `identifier` on Linear.\n"
                f"    python -m swarm.mcp.contract {self.source}"
            )
        return value

    # --- building it ------------------------------------------------------

    @classmethod
    def from_mapping(cls, block: Any, *, source: str = "<built-in>") -> TrackerContract:
        """Validate one tracker block and return the contract it describes.

        Every refusal names the field and the fix. The one thing never
        inspected is the *content* of an `args` dict: see the module docstring
        on why an `args` that understood GitHub's parameter names would be an
        `args` Jira could not be added to.
        """
        if not isinstance(block, Mapping):
            raise ContractError(
                f"{source}: the tracker block must be a mapping of settings, not "
                f"{type(block).__name__}. It looks like:\n"
                f"    tracker:\n"
                f"      mcp: github\n"
                f"      args: {{ owner: shahrestani-me, repo: apiary }}"
            )

        _refuse_unknown(block, _BLOCK_KEYS, source, "tracker")
        name = str(block.get("mcp") or "").strip()
        if not name:
            raise ContractError(
                f"{source}: tracker.mcp is missing. It names the server this "
                f"organization's task system is reached through, and selects a built-in "
                f"profile when it is one of {', '.join(sorted(PROFILES))}:\n"
                f"    mcp: github"
            )

        common = block.get("args")
        if common is not None and not isinstance(common, Mapping):
            raise ContractError(
                f"{source}: tracker.args must be a mapping of the arguments every "
                f"capability shares, not {type(common).__name__}. It is the short way to "
                f"say the constants that scope a whole tracker:\n"
                f"    args: {{ owner: shahrestani-me, repo: apiary }}"
            )

        merged = _merge(PROFILES.get(name, {}), block)
        endpoint, command = _endpoint_of(merged, name, source)
        auth = _auth_of(merged.get("auth"), source, stdio=bool(command))
        capabilities = {
            capability: _capability_of(capability, merged.get(capability), name, source)
            for capability in CAPABILITIES
        }
        return cls(
            mcp=name,
            endpoint=endpoint,
            command=command,
            auth=auth,
            capabilities=capabilities,
            source=source,
        )

    def describe(self) -> str:
        """The resolved block, for `python -m swarm.mcp.contract` and doctor output."""
        lines = [
            f"tracker: {self.mcp}   ({self.source})",
            f"  endpoint  {self.endpoint}",
            f"  credential {self.auth.value_env}"
            + (f" -> {self.auth.server_env} in the server's own environment"
               if self.auth.server_env else f" -> {self.auth.header}: {self.auth.scheme}"),
        ]
        for name in CAPABILITIES:
            cap = self.capabilities[name]
            parts = [f"  {name:<8}  {cap.tool}"]
            if cap.args:
                parts.append(f"args={json.dumps(dict(cap.args), sort_keys=True)}")
            if cap.fields:
                parts.append(f"fields={json.dumps(dict(cap.fields), sort_keys=True)}")
            if cap.ref:
                parts.append(f"ref={cap.ref}")
            lines.append("  ".join(parts))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _refuse_unknown(
    block: Mapping[str, Any], allowed: Sequence[str], source: str, where: str
) -> None:
    """Refuse a key nobody reads, and guess what it was meant to be."""
    unknown = [str(key) for key in block if str(key) not in allowed]
    if not unknown:
        return
    guesses = {
        key: near for key in unknown if (near := _closest(key, allowed))
    }
    hint = "".join(f"\n    {key}: did you mean {near}?" for key, near in guesses.items())
    raise ContractError(
        f"{source}: {where} has a key nothing reads: {', '.join(sorted(unknown))}. "
        f"Allowed here: {', '.join(allowed)}.{hint}\n"
        f"    python -m swarm.mcp.contract {source}"
    )


def _closest(key: str, allowed: Sequence[str]) -> str | None:
    """The allowed key `key` is one edit away from being, if there is one."""
    matches = difflib.get_close_matches(key, list(allowed), n=1, cutoff=0.7)
    return matches[0] if matches else None


def _merge(profile: Mapping[str, Any], block: Mapping[str, Any]) -> dict[str, Any]:
    """A built-in profile with the customer's block laid over it.

    Two levels, and no deeper. Capability sub-mappings merge key by key so that
    overriding `create.tool` does not silently discard the profile's
    `method: create`; `args` merges key by key inside that, so a block naming
    `owner` and `repo` keeps whatever constants the profile pinned. A top-level
    `args` is the common case made short - `owner` and `repo` are needed by all
    three GitHub capabilities and nobody should write them three times - and it
    goes underneath a capability's own, which wins.

    A sub-mapping that is not a mapping is carried through untouched rather
    than skipped. Merging is not the place to have an opinion about a malformed
    block, and a merge that quietly dropped `args: [1, 2]` would hand the
    validator a block that was never written and report nothing at all.
    """
    merged: dict[str, Any] = {
        key: value for key, value in profile.items() if key not in CAPABILITIES
    }
    merged.update({key: value for key, value in block.items() if key not in CAPABILITIES})

    if isinstance(profile.get("auth"), Mapping) and isinstance(block.get("auth"), Mapping):
        merged["auth"] = {**profile["auth"], **block["auth"]}

    # `endpoint` and `command` are alternatives, so a block naming one discards
    # the profile's other rather than colliding with it. Pointing the Linear
    # profile at a locally proxied server is a legitimate thing to configure,
    # and reporting it as "you named both" would be a refusal of something
    # nobody wrote.
    if "command" in block and "endpoint" not in block:
        merged.pop("endpoint", None)
    if "endpoint" in block and "command" not in block:
        merged.pop("command", None)

    common = block.get("args") if isinstance(block.get("args"), Mapping) else {}
    for capability in CAPABILITIES:
        base = profile.get(capability)
        over = block.get(capability)
        written = over if over is not None else base
        if written is not None and not isinstance(written, Mapping):
            # Leave whatever was written in place so the capability validator is
            # the one that reports it. Dropping it here would leave the profile's
            # own value standing and validate a block nobody wrote.
            merged[capability] = written
            continue
        if written is None:
            continue
        base = base if isinstance(base, Mapping) else {}
        over = over if isinstance(over, Mapping) else {}
        combined: dict[str, Any] = {**base, **over}
        combined.update(_sub_merge("args", base, over, common))
        combined.update(_sub_merge("fields", base, over, {}))
        merged[capability] = combined
    return merged


def _sub_merge(
    key: str, base: Mapping[str, Any], over: Mapping[str, Any], common: Mapping[str, Any]
) -> dict[str, Any]:
    """`{key: merged}`, or `{}` when neither side has one - or the bad value."""
    for source in (over, base):
        value = source.get(key)
        if value is not None and not isinstance(value, Mapping):
            return {key: value}
    merged = {
        **(base.get(key) or {}),
        **common,
        **(over.get(key) or {}),
    }
    return {key: merged} if merged or key in base or key in over else {}


def _endpoint_of(block: Mapping[str, Any], name: str, source: str) -> tuple[str, tuple[str, ...]]:
    """`(endpoint, command)` - exactly one of the two ways to reach a server.

    A remote server is a URL and a local one is a subprocess, and the endpoint
    string is what every error message and every doctor line quotes, so a local
    server gets the `stdio://` label `client.py` already reads rather than a
    `None` that prints as nothing.
    """
    endpoint = str(block.get("endpoint") or "").strip()
    raw_command = block.get("command")
    if raw_command is not None and not (
        isinstance(raw_command, Sequence) and not isinstance(raw_command, (str, bytes))
    ):
        raise ContractError(
            f"{source}: tracker.command must be the argv of a local server, as a list:\n"
            f"    command: [github-mcp-server, stdio]"
        )
    command = tuple(str(part) for part in (raw_command or ()))

    if endpoint and command:
        raise ContractError(
            f"{source}: tracker names both an endpoint ({endpoint}) and a command "
            f"({' '.join(command)}), which are the two alternative ways to reach a server. "
            f"Keep the one this organization uses and delete the other."
        )
    if command:
        return f"{STDIO_SCHEME}{command[0]}", command
    if not endpoint:
        raise ContractError(
            f"{source}: tracker names no endpoint and no command, so there is nothing to "
            f"reach{_profile_hint(name)}. A remote server is a URL and a local one is an "
            f"argv:\n"
            f"    endpoint: https://mcp.linear.app/mcp\n"
            f"    command: [github-mcp-server, stdio]"
        )
    if not endpoint.lower().startswith(("http://", "https://")):
        raise ContractError(
            f"{source}: tracker.endpoint is {endpoint!r}, which is not a URL. A remote MCP "
            f"server is http(s); a local one is named by `command:` instead:\n"
            f"    endpoint: https://mcp.linear.app/mcp"
        )
    return endpoint, ()


def _auth_of(block: Any, source: str, *, stdio: bool) -> Auth:
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise ContractError(
            f"{source}: tracker.auth must be a mapping, not {type(block).__name__}:\n"
            f"    auth: {{ value_env: APIARY_LINEAR_TOKEN, scheme: bearer }}"
        )
    _refuse_unknown(block, _AUTH_KEYS, source, "tracker.auth")

    value_env = str(block.get("value_env") or "").strip()
    if not value_env:
        raise ContractError(
            f"{source}: tracker.auth.value_env is missing. It names the environment "
            f"variable holding the pre-minted credential - apiary never obtains one, "
            f"because #143 found no machine-to-machine grant on any tracker it targets:\n"
            f"    auth: {{ value_env: APIARY_LINEAR_TOKEN }}"
        )
    scheme = str(block.get("scheme") or "bearer").strip().lower()
    if scheme not in SCHEMES:
        raise ContractError(
            f"{source}: tracker.auth.scheme is {scheme!r}; it must be one of "
            f"{', '.join(SCHEMES)}. It is spelled out rather than assumed because Jira's "
            f"API-token mode is HTTP Basic and both other trackers are Bearer - a "
            f"contract that assumed one would be wrong silently."
        )
    server_env = block.get("server_env")
    if stdio and not server_env:
        raise ContractError(
            f"{source}: this tracker is a locally spawned server, which reads its own "
            f"credential from its own environment rather than from a header. Name the "
            f"variable it reads:\n"
            f"    auth: {{ value_env: {value_env}, server_env: GITHUB_PERSONAL_ACCESS_TOKEN }}"
        )
    return Auth(
        value_env=value_env,
        header=str(block.get("header") or "Authorization"),
        scheme=scheme,
        server_env=str(server_env) if server_env else None,
        mint=str(block.get("mint") or ""),
    )


def _capability_of(name: str, block: Any, mcp: str, source: str) -> Capability:
    if block is None:
        raise ContractError(
            f"{source}: tracker.{name} is missing. apiary needs three things of a task "
            f"system and no more - {', '.join(CAPABILITIES)} - and a run reaches the "
            f"missing one an hour in{_profile_hint(mcp)}:\n"
            f"    {name}: {{ tool: list_issues }}"
        )
    if not isinstance(block, Mapping):
        raise ContractError(
            f"{source}: tracker.{name} must be a mapping naming the tool that fulfils it, "
            f"not {type(block).__name__}:\n"
            f"    {name}: {{ tool: list_issues, args: {{ labels: [agent-ready] }} }}"
        )
    _refuse_unknown(block, _CAPABILITY_KEYS, source, f"tracker.{name}")

    tool = str(block.get("tool") or "").strip()
    if not tool:
        raise ContractError(
            f"{source}: tracker.{name}.tool is missing. It is the whole point of the "
            f"block - which tool on this server fulfils this need{_profile_hint(mcp)}:\n"
            f"    {name}: {{ tool: list_issues }}"
        )
    if name == INTAKE and tool in INADMISSIBLE_INTAKE.get(mcp, {}):
        raise ContractError(
            f"{source}: tracker.intake.tool is {tool!r}, which cannot be intake on "
            f"{mcp}: {INADMISSIBLE_INTAKE[mcp][tool]}. Use a deterministic filter "
            f"instead:\n"
            f"    intake: {{ tool: list_issues, args: {{ labels: [agent-ready] }} }}"
        )

    args = block.get("args") or {}
    if not isinstance(args, Mapping):
        raise ContractError(
            f"{source}: tracker.{name}.args must be a mapping of arguments, not "
            f"{type(args).__name__}. apiary forwards it to the server untouched and "
            f"parses nothing in it, so it is whatever that tool's parameters are:\n"
            f"    {name}: {{ tool: {tool}, args: {{ owner: shahrestani-me, repo: apiary }} }}"
        )
    if any(not isinstance(key, str) for key in args):
        raise ContractError(
            f"{source}: tracker.{name}.args has a key that is not a name. Arguments are "
            f"passed through as a JSON object, whose keys are strings."
        )

    fields = block.get("fields") or {}
    if not isinstance(fields, Mapping):
        raise ContractError(
            f"{source}: tracker.{name}.fields must map apiary's field names to this "
            f"server's, not {type(fields).__name__}:\n"
            f"    {name}: {{ tool: {tool}, fields: {{ body: description }} }}"
        )
    unknown = [str(key) for key in fields if str(key) not in CANONICAL_FIELDS]
    if unknown:
        raise ContractError(
            f"{source}: tracker.{name}.fields maps {', '.join(sorted(unknown))}, which is "
            f"not one of apiary's fields. The left-hand side is apiary's vocabulary and is "
            f"closed - {', '.join(CANONICAL_FIELDS)} - and the right-hand side is this "
            f"server's and is never checked:\n"
            f"    {name}: {{ tool: {tool}, fields: {{ body: description }} }}"
        )
    if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
        raise ContractError(
            f"{source}: tracker.{name}.fields must name a field on the right-hand side:\n"
            f"    {name}: {{ tool: {tool}, fields: {{ body: description }} }}"
        )

    ref = block.get("ref")
    if ref is not None and (not isinstance(ref, str) or not ref.strip()):
        raise ContractError(
            f"{source}: tracker.{name}.ref must name the response field carrying the task "
            f"ref:\n    {name}: {{ tool: {tool}, ref: number }}"
        )
    if name == INTAKE and not ref:
        raise ContractError(
            f"{source}: tracker.intake.ref is missing. It names which field of a returned "
            f"item is the durable task ref, and apiary derives every execution state from "
            f"it: a branch called apiary/<ref>-attempt-N is what `claimed` and `review` "
            f"are read back out of (ADR 0001). There is no common answer to guess - "
            f"GitHub's is `number`, Linear's is `identifier`{_profile_hint(mcp)}:\n"
            f"    intake: {{ tool: {tool}, ref: number }}"
        )

    return Capability(
        name=name,
        tool=tool,
        args=dict(args),
        fields={str(key): str(value) for key, value in fields.items()},
        ref=str(ref).strip() if ref else None,
    )


def _profile_hint(name: str) -> str:
    """"; `mcp: gtihub` matches no built-in profile" - or nothing.

    Worth a sentence because the whole failure mode of a typo'd `mcp:` is that
    every *other* error the block produces is a consequence of it: a profile
    that did not apply leaves six fields missing, and reporting the six is
    reporting the symptom six times.
    """
    if name in PROFILES:
        return ""
    return (
        f"; `mcp: {name}` matches no built-in profile, so nothing was filled in for you "
        f"(the built-in ones are {', '.join(sorted(PROFILES))})"
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_tracker(
    path: str | Path | None = None,
    *,
    settings: Settings = SETTINGS,
    env: Mapping[str, str] | None = None,
) -> TrackerContract | None:
    """The configured contract, or `None` if this installation has no tracker.

    `None` and a refusal are different answers and the difference is deliberate.
    apiary today still runs on the label control plane (#152 removes it), so an
    installation with no tracker file is a normal one and must not fail a
    preflight. An installation that *named* a file meant it, so a named file
    that is not there is an error rather than a silent fallback to nothing -
    that is the misconfiguration that would otherwise present as a tracker
    quietly not being consulted.
    """
    environment = dict(os.environ if env is None else env)
    named = str(path) if path else environment.get(TRACKER_CONFIG_ENV, "").strip()
    location = Path(named or settings.tracker_config)

    if not location.is_file():
        if named:
            raise ContractError(
                f"{location}: no such file, and it is where {TRACKER_CONFIG_ENV} says the "
                f"tracker block is. Write it, or unset the variable:\n"
                f"    export {TRACKER_CONFIG_ENV}={location}"
            )
        return None

    return parse_tracker(_read(location), source=str(location))


def parse_tracker(text: str, *, source: str = "<string>") -> TrackerContract:
    """Parse and validate one tracker document.

    The document may be the block itself or a file with `tracker:` at the top,
    because ADR 0001 and #143 both write the second and a file holding one
    setting reads better as the first.
    """
    document = _parse(text, source)
    if isinstance(document, Mapping) and "tracker" in document:
        document = document["tracker"]
    return TrackerContract.from_mapping(document, source=source)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"{path}: cannot be read ({exc}). Check the path and its permissions.")


def _parse(text: str, source: str) -> Any:
    """YAML, which is also how a JSON file parses.

    Imported here rather than at module scope so that a missing PyYAML is a
    sentence about one file rather than an ImportError that takes `swarm
    doctor` down on the run where the operator has least confidence that
    anything works.
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover - a declared dependency
        raise ContractError(
            f"{source}: reading a tracker block needs PyYAML, which is a declared "
            f"dependency of this package and is not importable here:\n"
            f"    pip install -e '.[dev]'"
        ) from None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ContractError(
            f"{source}: is not valid YAML ({exc}). Indentation is the usual cause; "
            f"re-check with:\n    python -m swarm.mcp.contract {source}"
        ) from None


# --------------------------------------------------------------------------
# Reaching the server the contract names
# --------------------------------------------------------------------------


def client_for(
    contract: TrackerContract,
    *,
    env: Mapping[str, str] | None = None,
    require_credential: bool = True,
    **kwargs: Any,
) -> McpClient:
    """An `McpClient` wired the way this contract says.

    `require_credential=False` is `doctor`'s, and it is the reason this
    argument exists at all: a probe that refused to build a client without a
    credential could not tell "the server is unreachable" from "you have not
    exported a token yet", and those are different sentences with different
    fixes. Without one, the client is built with no authorization at all, the
    server answers 401, and reachability is proven separately from authority.
    """
    environment = dict(env if env is not None else {})
    credential = contract.auth.credential(environment)
    if require_credential and not credential:
        raise ContractError(
            f"{contract.source}: {contract.auth.value_env} is not set, and it is where "
            f"this tracker's credential lives.\n    {contract.auth.absent_fix()}"
        )

    if contract.is_stdio:
        # The credential is handed to the subprocess, not to a header, and
        # `over_stdio` passes through only the variables named here - so the
        # orchestrator's other secrets do not become a third-party binary's.
        server_env = {}
        if credential and contract.auth.server_env:
            server_env[contract.auth.server_env] = credential
        return McpClient.over_stdio(
            contract.command,
            env=server_env,
            token_env=contract.auth.value_env,
            **kwargs,
        )

    headers = {}
    if credential:
        headers[contract.auth.header] = contract.auth.header_value(credential)
    return McpClient(
        contract.endpoint,
        None,
        token_env=contract.auth.value_env,
        headers=headers,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a tracker block and print what it resolved to. Opens no socket.

    The command every refusal in this module names. It answers the question an
    operator actually has - "is this block right?" - without starting a run, and
    it prints the *merged* contract, so a block that leant on a built-in profile
    shows what the profile filled in.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        contract = load_tracker(args[0]) if args else load_tracker()
        if contract is None:
            print(
                f"no tracker configured: {SETTINGS.tracker_config} does not exist, and "
                f"{TRACKER_CONFIG_ENV} is unset. apiary reaches a task system through the "
                f"customer's own MCP server (ADR 0001), so this is configuration:\n"
                f"    export {TRACKER_CONFIG_ENV}=/path/to/tracker.yaml"
            )
            return 1
    except ContractError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1
    print(contract.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
