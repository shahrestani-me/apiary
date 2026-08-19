# ADR 0004 — the tracker capability contract, corrected

Status: **proposed**
Date: 2026-08-19
Amends: `docs/adr/0001-task-systems-are-integrations.md`, "Configuration, not code"

## The one-line summary

**A capability needs four things, not one: which tool, which arguments, what
the server calls apiary's fields — and that the task ref is one of those
fields.** ADR 0001 specified only the first two, and with them no real call can
be made against either priority tracker.

## Context

ADR 0001 decided that task systems are reached through a per-organization MCP
server, with a capability contract naming which tool fulfils intake, comment and
create. It sketched that contract from first principles, before any server had
been examined:

```yaml
comment: { tool: create_comment }
create:  { tool: create_issue }
```

The #143 spike then examined them. The thesis survived — a few declarative lines
per tracker is still config rather than the adapter this epic exists to delete —
but the shape did not. Two defects, both in the same section, and both such that
an implementer following ADR 0001 would produce something that cannot make a
single call.

Scope has also narrowed since: **GitHub and Linear are the priority, Jira is
deferred.** That turns out to matter for the second defect.

## The defects

**1. `{tool, args}` cannot make the calls.** A capability carrying only a tool
name cannot invoke GitHub's `issue_write`, which needs `method: "create"` pinned
as an argument, and cannot write an issue body to a server that calls the field
`description`. Two fields are missing from every capability:

- a **field map**, from apiary's canonical field names to this server's
- a **`ref` rule**, naming which response field is the durable, branch-safe task ref

**And a third thing, which the spike missed and #150 found while implementing
it.** The spike treated the ref purely as a *response* field. It is also a
*request argument whose name diverges*: GitHub's comment tool takes
`issue_number`, Linear's takes `issueId`, Jira's takes `issueIdOrKey`. So the
spike's own Linear example — `comment: { tool: create_comment }` — cannot make
the call either. It convicted ADR 0001 of exactly the defect it then repeated.

The fix is not a fifth field but a wider vocabulary: **`ref` is one of apiary's
canonical field names**, so the existing field map covers it —
`fields: { ref: issue_number }`.

The scope constants — `owner`+`repo`, `teamId` — need no new machinery. They
belong in the existing `args`, which is already a static dict.

**2. The `filter:` intake example is not a real parameter**, and the sentence
above it mis-describes what intake is:

```yaml
intake: { tool: list_issues, args: { filter: "label=agent-ready" } }
```

Neither GitHub's nor Linear's `list_issues` has a `filter` parameter; both take
discrete typed filters. ADR 0001's surrounding claim — that intake "stays in
their language", a query the customer wrote — is **true of Jira alone**. With
Jira deferred, the accurate statement is narrower and, usefully, stronger.

## Decision

**1. A capability is `{tool, args, fields, ref}`.** `args` carries the static
constants including any `method` discriminator; `fields` maps apiary's canonical
names onto the server's; `ref` names the response field carrying the durable
task ref.

**2. Intake args are the server's own typed filter parameters, forwarded
verbatim and never parsed by apiary.** The no-parsing guarantee ADR 0001 wanted
survives intact. The "customer's own query language" framing does not, and is
withdrawn.

**3. `intake.args` stays an opaque pass-through dict.** This is the one place
where Jira's deferral must not become Jira's exclusion. JQL is a single opaque
string, so the moment `intake.args` is narrowed to a typed schema validated
against GitHub's and Linear's parameter names, Jira cannot be added without
reopening the design. Two similar trackers make that narrowing look free. It is
not.

**4. GitHub intake uses label filters. `search_issues` is inadmissible.** It is
now natural-language semantic matching, which makes it a non-deterministic input
to the reconcile loop — ruled out by ADR 0001's own boundary between
deterministic and model-driven MCP, applied to a call it did not know was
semantic.

## The residual divergence, in full

Two trackers, and this is the entire list of what differs:

| | GitHub | Linear |
|---|---|---|
| create tool | `issue_write`, `method: "create"` pinned | `create_issue` |
| body field | `body` | `description` |
| comment's ref argument | `issue_number` | `issueId` |
| scope | `owner` + `repo` | `teamId` |
| ref rule | `number` | `identifier` |

**Five rows, not the four an earlier draft of this ADR claimed.** The comment
row is the one that was missing, and it is missing from the spike for the same
reason: a ref reads naturally as something a server *returns*, so nobody checks
whether it is also something the call *takes*.

Five rows is still config rather than code, which is the point. But the
correction is worth keeping visible, because both the ADR it amends and the
spike that corrected that ADR made the same omission independently.

## Consequences

- ADR 0001's "Configuration, not code" block is superseded by this one. The
  decision it belongs to — that trackers are configured rather than adapted —
  is unchanged.
- #150 implements this shape. The spike (`docs/plans/spike-143-mcp-tool-shapes.md`)
  carries the per-tracker detail and the seven-row Jira constraint table.
- **The comment-body trap closes itself.** GitHub and Linear both call the
  comment field `body`, so a two-tracker implementation was expected to
  hard-code it and lock out Jira's `commentBody`. Because comment now needs a
  field map on *both* priority trackers for its ref argument, the map exists
  before anyone is tempted — the divergence that was going to be missed forces
  the machinery that prevents it. Worth noting as luck rather than design.
- **GitHub's credential goes in the server's environment, not a header.** The
  spike contradicted itself here: its example block passes `Authorization:
  bearer`, while its own recommendation selects the local stdio server, which
  takes its credential from its own environment and never sees a header.
  Resolved in favour of the recommendation, as `auth.server_env`.
- **`ref` belongs under `intake`**, with other capabilities inheriting it. It is
  a fact about the server rather than about a call — GitHub's create returns
  `number` in the same field intake lists.

## What this does not change

That the tracker is reached over MCP, that organization workflow is never
modelled, and that apiary ships no tracker integration code. The spike's own
verdict on ADR 0001 was that it under-costed this block by roughly threefold,
not that its thesis was wrong — and the block is four rows.
