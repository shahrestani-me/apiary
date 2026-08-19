# Spike 143 — MCP tool shapes across trackers, and the headless auth story

Spike for #143. Epic #140. Timeboxed; the deliverable is a decision, not code.
Evidence gathered 2026-08-19 against the live servers and current vendor docs.

## The two verdicts, up front

**Q1 — does one capability contract cover Linear, Jira and GitHub?**
The *capability set* does. **The contract as written in ADR 0001 does not.**
`{ tool, args }` with opaque pass-through is not sufficient for any of the three
servers, and it is not close for two of them. ADR 0001 must be amended before
#150 implements it. The amendment is small and it stays configuration — see
[Recommendation](#recommendation).

**Q2 — headless auth.**
All three servers can be authorized in an unattended run, and none of them can
be authorized by the flow apiary would reach for first. No server here supports
the `client_credentials` grant — verified against their live metadata — so there
is no machine-to-machine OAuth path. **The orchestrator must never drive an
authorization-code flow. It must be handed a static, pre-minted credential and
send it as a header.** That is a supported, vendor-documented mode on all three.

---

## Part 1 — tool shapes

### Method

Tool names and schemas were read from primary sources: the GitHub MCP server's
live tool schemas as served to this session, the vendor docs for Linear and
Atlassian, and the Atlassian MCP repository. Auth behaviour was probed directly
against the three production endpoints (`tools/list` with no credential, and the
`.well-known` OAuth metadata). Raw probe output is quoted in Part 2.

### The three capabilities, per server

#### intake — "which items may the agent pick up?"

| | tool | how the query is expressed | required scope args |
|---|---|---|---|
| GitHub | `list_issues` | discrete typed filters: `labels[]`, `state`, `since`, `orderBy`, `direction`, `field_filters[]` | `owner`, `repo` |
| GitHub (alt) | `search_issues` | `query` — **natural-language semantic matching**, not GitHub search syntax | none (owner/repo optional) |
| Linear | `list_issues` | discrete typed filters: `teamId`, `stateId`, `assigneeId`, label, `updatedAt`, `limit`, `orderBy` | none — the token scopes the workspace |
| Jira | `searchJiraIssuesUsingJql` | `jql` — a real free-text query DSL | `cloudId` |

This table is the whole finding.

ADR 0001 says intake "stays in their language" and gives
`intake: { tool: list_issues, args: { filter: "label=agent-ready" } }`. That
example is not implementable on either server it appears to describe:

- **Linear's official `list_issues` has no `filter` parameter.** It exposes a
  curated set of discrete typed filters and deliberately does not surface
  Linear's GraphQL filter language. There is no string a customer can write that
  apiary passes through unparsed.
- **GitHub's `list_issues` has no `filter` parameter either**, and it *requires*
  `owner` and `repo` on every call. A customer's saved GitHub search string has
  nowhere to go.
- **Only Jira has the thing ADR 0001 assumed exists.** JQL is exactly the
  opaque-customer-query story, and it works.

There is a second, sharper problem hiding in the GitHub row. GitHub's
`search_issues` is documented as *"natural-language semantic matching"* — the
server runs an inference to decide what matches. ADR 0001 is explicit that the
reconcile loop is deterministic, and gives the reason: on this host ~40% of
`propose_edits` calls emit broken output, and a bad tracker write is not caught
by CI. **A semantic intake query is a non-deterministic control-loop input and
must not be used**, which leaves `list_issues` with label filters as the only
admissible GitHub intake — a narrower thing than the ADR promises.

#### comment — "post the PR link / flag needs-human"

| | tool | arguments |
|---|---|---|
| GitHub | `add_issue_comment` | `owner`, `repo`, `issue_number` (int), `body` |
| Linear | `create_comment` | issue id, `body` |
| Jira | `addCommentToJiraIssue` | `cloudId`, `issueIdOrKey` (`"PROJ-123"`), `commentBody` |

This is the closest of the three capabilities and it still does not line up:
the body field is `body`, `body`, `commentBody`; the issue is identified by an
integer plus two path components, a workspace-unique id, and a project key
string plus a `cloudId`.

#### create — "open a work item"

| | tool | arguments |
|---|---|---|
| GitHub | `issue_write` with **`method: "create"`** | `owner`, `repo`, `title`, `body`, `labels[]`, `assignees[]`, `milestone`, `type` |
| Linear | `create_issue` | `title`, `teamId` (**required**), `description`, `priority`, `stateId`, `assigneeId`, `labelIds[]` |
| Jira | `createJiraIssue` | `cloudId`, `projectKey`, `issueTypeName`, `summary`, `description` |

Three separate ways this breaks a tool-name-only contract:

1. **GitHub fused create and update behind a discriminator.** `create: { tool: create_issue }` cannot express `issue_write` — naming the tool is not
   enough, you must also pin `method: "create"` as a constant argument. A
   contract that carries only a name cannot call the current GitHub MCP server
   at all.
2. **The title/body field names differ three ways**: `title`/`body`,
   `title`/`description`, `summary`/`description`.
3. **Each server demands a different mandatory scope constant** that apiary
   does not otherwise possess: `owner`+`repo`, `teamId`, `cloudId`+`projectKey`.
   Jira additionally requires `issueTypeName`, which has no analogue on the
   other two.

### The finding the ADR did not anticipate: there is no common task ref

| | task identity | branch-safe? |
|---|---|---|
| GitHub | `(owner, repo, issue_number: int)` | only after composing |
| Linear | issue id (uuid), plus a human identifier like `ENG-123` | uuid is not; identifier is |
| Jira | `issueIdOrKey` — `"PROJ-123"` | yes |

ADR 0001 derives execution state from branch names shaped
`apiary/<ref>-attempt-2`, and derives `review` from a PR that *references the
task*. Both need a ref that is a single, stable, branch-safe token. GitHub
issues do not supply one — the number alone is only unique within a repository,
and apiary's own dependency graph is `int`-keyed today (`github/readiness.py`).
**The contract needs a `ref` rule per server**, saying which response field is
the durable task ref. This is not optional; it is load-bearing for the derived
state machine that is the point of the whole ADR.

Pagination diverges too — GitHub offers `page`/`perPage` *and* an `after`
cursor, Linear offers `before`/`after` cursors, Jira offers `nextPageToken` —
so the client must own pagination rather than passing it through.

### Verdict on Q1

**One contract, yes. One contract of the shape ADR 0001 specifies, no.**

The three capabilities are genuinely the right three; nothing in the survey
suggests apiary needs a fourth, and nothing suggests a capability is missing on
any server. What fails is the assumption that a capability reduces to a tool
name plus arguments the customer wrote. In practice every call needs three
things the ADR's YAML has no slot for: constants apiary must inject on every
call (`owner`/`repo`, `cloudId`, `teamId`, `method: "create"`), a mapping from
apiary's fields to this server's field names, and a rule for extracting the task
ref from the response.

Calling that a "translation layer" and therefore the adapter this epic exists to
delete would be the wrong read. The distinction that matters is **fixed-shape
declarative mapping versus per-server executable code**. Six lines of YAML per
server that name fields and constants is not a Python adapter: it has no branches,
no error handling, no lifecycle, and adding a fourth tracker still touches no
code. The honest statement is that ADR 0001 under-costed the config block by a
factor of about three, not that its thesis is wrong.

---

## Part 2 — the headless auth story

### The symptom, and why it happens

The known failure — interactively-authorized MCP servers simply absent in
headless runs — is not an MCP bug and not a bug at all. An interactive client
completes an OAuth 2.1 authorization-code flow in a browser and caches the token
in *that client's* per-user credential store. A cron or CI process is a
different process, usually a different user, often a different container. It
finds no cached token, cannot open a browser, and the correct behaviour is
exactly what is observed: the server is unavailable.

apiary is more exposed to this than a CLI is, because the orchestrator is a
long-lived unattended Python process (#149) and there is no human to prompt.

### What each server actually supports — probed directly

All three advertise OAuth 2.1 protected-resource metadata and challenge with
`WWW-Authenticate: Bearer` on an unauthenticated `tools/list`:

```
https://mcp.linear.app/mcp        401  Bearer realm="OAuth", error="invalid_token"
https://mcp.atlassian.com/v1/mcp  401  Bearer realm="OAuth", error="invalid_token"
https://api.githubcopilot.com/mcp/ 401 Bearer error="invalid_request"
```

The decisive evidence is the authorization-server metadata:

```
mcp.linear.app     grant_types_supported: authorization_code, refresh_token,
                                          urn:ietf:params:oauth:grant-type:jwt-bearer
mcp.atlassian.com  grant_types_supported: authorization_code, refresh_token
github.com/login/oauth
                   grant_types_supported: authorization_code, refresh_token
```

**Not one of them offers `client_credentials`.** There is no
machine-to-machine OAuth grant available on any of the three trackers apiary
targets. Any design that has the orchestrator "authenticate to the MCP server"
is designing a flow that does not exist.

What each vendor *does* support for unattended use:

| server | headless mechanism | header | notes |
|---|---|---|---|
| **GitHub** | GitHub App installation token, or a PAT | `Authorization: Bearer <token>` | The server's own docs name GitHub App auth as the mode "for non-interactive stdio deployments". A **local stdio** server binary also exists, taking `GITHUB_PERSONAL_ACCESS_TOKEN` from the environment — no network OAuth at all. |
| **Linear** | Linear API key, or an OAuth app-actor token | `Authorization: Bearer <key>` | Linear's docs state the MCP server accepts a bearer token or API key directly, bypassing the interactive flow. App-actor tokens attribute writes to the app rather than a person, which is what apiary wants. |
| **Jira / Atlassian** | scoped personal API token, HTTP Basic | `Authorization: Basic base64(email:api_token)` | Announced explicitly for "CI/CD pipelines, scheduled jobs, or backend services" and "cron jobs, agents, platform workers". Two catches: it must be a **scoped** token, not a classic one, and it is **off by default and requires org-admin enablement**. Cloud only — no Data Center or Server. |

### Verdict on Q2

**Named mechanism: a pre-minted static credential supplied to the orchestrator
as an environment-sourced secret and sent as an HTTP header on every MCP
request. The orchestrator performs no OAuth flow, opens no browser, and holds
no refresh token.** Unattended runs are supported on all three servers under
this mechanism, and unsupported under any other.

Consequences apiary must accept:

- **The credential is a config input, not something apiary obtains.** The
  tracker block names an env var; the operator mints the token out of band.
  This preserves the ADR's win — apiary never holds a Jira token in its own
  store — while making the unattended path real.
- **`doctor` must prove authorization before a cycle needs it**, which #150
  already requires. The check is a `tools/list` against the configured endpoint.
  A 401 must produce the named failure and the command that mints the token —
  and the three commands differ, so doctor's fix text is per-server.
- **A 401 is not retryable.** #149's rule — back off on 5xx and rate limits,
  fail fast on 4xx — is exactly right here, and 401 is the case that will
  actually occur: tokens expire and Atlassian's API-token mode can be revoked by
  an admin at the org level. Retrying a 401 turns a five-second config error
  into a backoff-length stall.
- **Atlassian is the one that can be switched off underneath apiary.** Because
  API-token auth is admin-gated and off by default, "the customer runs Jira" is
  not sufficient — the org must have enabled it. That belongs in doctor's output
  and in whatever onboarding text #150 produces.
- **Egress.** Three hosts to allow: `mcp.linear.app`, `mcp.atlassian.com`,
  `api.githubcopilot.com`. Per ADR 0001 and #149, `APIARY_EGRESS_ALLOW` is inert
  — the real edits are `security.EGRESS_ALLOWLIST` and `compose.yaml`, which
  `tests/test_security.py` asserts agree. Only the configured tracker's host
  should be opened, not all three.
- **Prefer a local stdio server where the vendor ships one.** GitHub does. It
  removes an egress hole, removes a network dependency from the reconcile loop's
  critical path, and takes its credential from the environment — the shape
  apiary already handles. Linear and Atlassian ship remote-only, so the client
  must support both transports regardless.

---

## Recommendation

**Amend ADR 0001 before #150 is implemented.** Three changes:

**1. Widen the capability contract from `{tool, args}` to four fields.**

```yaml
tracker:
  mcp: linear
  auth: { header: Authorization, value_env: APIARY_TRACKER_TOKEN, scheme: bearer }
  const: { }                                  # merged into every call
  intake:
    tool: list_issues
    args: { teamId: "...", stateId: "..." }   # this server's own filter params
    ref:  "identifier"                        # response field that is the task ref
  comment:
    tool: create_comment
    fields: { task: issueId, body: body }     # apiary's field -> this server's
  create:
    tool: create_issue
    fields: { title: title, body: description }
```

The same block for GitHub carries `const: { owner: ..., repo: ... }` and
`create.args: { method: create }`; for Jira it carries `const: { cloudId: ... }`
and an intake `args: { jql: "..." }`. Still declarative, still no code per
tracker, and now actually able to make the calls.

**2. Ship the three profiles in-repo** as defaults a customer selects and
overrides, rather than making every customer rediscover that `createJiraIssue`
wants `summary`. `tracker: { profile: jira, const: { cloudId: ..., projectKey: ... } }`
should be the whole config in the common case. This is what keeps "a new tracker
costs configuration" true in practice and not just on paper.

**3. Replace the "intake stays in their language" claim with the accurate one.**
Intake is opaque pass-through *on Jira*, where JQL exists. On Linear and GitHub
it is a set of typed filter arguments that apiary forwards without interpreting
— still no parsing by apiary, but not a customer-authored query string. And
GitHub's `search_issues` is off-limits for the reconcile loop because it is
semantic.

**On auth, no amendment is needed — only a commitment**: the orchestrator is a
bearer-token client and nothing else. Add to #150's doctor check that a missing
or rejected credential names the per-server minting command, and add to #149
that 401 and 403 fail fast and loud.

The bar for reversing this: if a fourth tracker were to need a *behaviour*
rather than a field mapping — a two-call create, or a write that must be
retried differently — the declarative contract has run out and the adapter
question genuinely reopens. Nothing in these three requires that today.

## Sources

- GitHub MCP server tool schemas, read live from the server in this session; and
  <https://github.com/github/github-mcp-server> (README, auth modes)
- <https://linear.app/docs/mcp> — endpoint, transport, OAuth 2.1 + bearer/API key
- <https://blog.fiberplane.com/blog/mcp-server-analysis-linear/> — official Linear tool list and `list_issues` parameter analysis
- <https://github.com/atlassian/atlassian-mcp-server> — Jira tool names and `cloudId` requirement
- <https://support.atlassian.com/rovo/docs/getting-started-with-the-atlassian-remote-mcp-server/> — endpoint, OAuth 2.1, Cloud-only
- Atlassian Community, "Announcing authentication via API token for Atlassian Rovo MCP Server" — Basic-auth header, scoped tokens, CI/cron framing, admin gating
- <https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/298> — the open proposal for non-interactive OAuth flows in MCP; confirms no standard mechanism exists yet
- Live probes of `mcp.linear.app`, `mcp.atlassian.com`, `api.githubcopilot.com`:
  `WWW-Authenticate` challenges and `.well-known` authorization-server metadata
