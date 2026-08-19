# Spike 143 — MCP tool shapes across trackers, and the headless auth story

Spike for #143. Epic #140. Timeboxed; the deliverable is a decision, not code.
Evidence gathered 2026-08-19 against the live servers and current vendor docs.

**Scope.** The maintainer has set the near-term priority at **GitHub and Linear;
Jira is deferred.** Parts 1 and 2 decide for those two. Jira is not discarded —
it is recorded in [Must not be precluded](#must-not-be-precluded--jira) as a
design constraint, because it is the awkward one and designing for exactly two
similar trackers is how you get an abstraction that fits exactly two.

## The two verdicts, up front

**Q1 — does one capability contract cover GitHub and Linear?**
Yes, and more comfortably than the three-tracker survey suggested. But
**ADR 0001's contract as written still cannot make the calls** — not because the
trackers are far apart, but because of two specific defects listed in
[What in ADR 0001 is wrong](#what-in-adr-0001-is-wrong). The amendment needed is
two fields plus the intake paragraph. That is smaller than the three-tracker
version of this document concluded, and it is still required before #150.

**Q2 — headless auth.**
Both servers can be authorized in an unattended run. **`client_credentials` is
absent from both** — verified against live metadata — so the conventional
machine-to-machine grant does not exist here. Linear does advertise
`urn:ietf:params:oauth:grant-type:jwt-bearer`, which is a real assertion-based
M2M grant; it is **available and deliberately declined**, with reasons below.
The recommendation is a pre-minted static credential sent as a header, chosen as
a trade-off rather than for lack of an alternative. For GitHub the credential is
one **apiary already holds**.

---

## Part 1 — tool shapes

### Method

Tool names and schemas were read from primary sources: the GitHub MCP server's
live tool schemas as served to this session, and the Linear vendor docs plus a
published analysis of the live `mcp.linear.app` tool list. Auth behaviour was
probed directly against the production endpoints; the exact URLs and responses
are quoted in Part 2 so the claims are reproducible.

### The three capabilities, for the two in-scope trackers

#### intake — "which items may the agent pick up?"

| | tool | how the query is expressed | required scope args |
|---|---|---|---|
| GitHub | `list_issues` | discrete typed filters: `labels[]`, `state`, `since`, `orderBy`, `direction`, `field_filters[]` | `owner`, `repo` |
| Linear | `list_issues` | discrete typed filters: `teamId`, `stateId`, `assigneeId`, label, `updatedAt`, `limit`, `orderBy` | none — the token scopes the workspace |

**With Jira deferred, intake is the most uniform of the three capabilities, not
the least.** That inverts the conclusion the three-tracker survey reached, and
it is worth being explicit about why.

The original finding was that a customer-authored opaque query string — the
thing ADR 0001 promised when it said intake "stays in their language" — exists
on Jira and nowhere else. JQL is a genuine query DSL; `list_issues` on GitHub
and on Linear is a fixed set of typed filter parameters. That finding stands
unchanged. What changes is its consequence: with the one tracker that *has* a
query language out of the near-term path, both remaining trackers express intake
the same way, as **a dict of typed filter arguments that apiary forwards
verbatim and never parses**.

So the contract carries *less* here, not more. `intake.args` is a static YAML
dict passed through untouched, and that is true of both servers. Note this is
still not "the customer's own query language" — it is the server's own filter
parameters — so the ADR's wording is wrong even though its mechanism survives.

One exclusion applies to GitHub only. Its `search_issues` tool is documented as
*"natural-language semantic matching"* — the server runs an inference to decide
what matches. ADR 0001 is explicit that the reconcile loop is deterministic, and
gives the reason: on this host ~40% of `propose_edits` calls emit broken output,
and a bad tracker write is not caught by CI. **A semantic intake query is a
non-deterministic control-loop input and must not be used**, which leaves
`list_issues` with label filters as the only admissible GitHub intake.

#### comment — "post the PR link / flag needs-human"

| | tool | arguments |
|---|---|---|
| GitHub | `add_issue_comment` | `owner`, `repo`, `issue_number` (int), `body` |
| Linear | `create_comment` | issue id, `body` |

**The body field is `body` on both.** For these two trackers, comment needs no
field mapping at all — only the scope constants and the task-ref shape differ.

#### create — "open a work item"

| | tool | arguments |
|---|---|---|
| GitHub | `issue_write` with **`method: "create"`** | `owner`, `repo`, `title`, `body`, `labels[]`, `assignees[]`, `milestone`, `type` |
| Linear | `create_issue` | `title`, `teamId` (**required**), `description`, `priority`, `stateId`, `assigneeId`, `labelIds[]` |

Two divergences survive the rescope:

1. **GitHub fused create and update behind a discriminator.**
   `create: { tool: create_issue }` cannot express `issue_write` — naming the
   tool is not enough, the call must also pin `method: "create"`. A contract
   that carries only a tool name cannot call the current GitHub MCP server at
   all. This is the single hardest fact in the survey and it is not going away.
2. **The body field differs on create**: GitHub `body`, Linear `description`.
   `title` matches. This is the entire field-mapping surface for two trackers —
   one field, on one capability, on one server.

### The finding the ADR did not anticipate: there is no common task ref

| | task identity | branch-safe? |
|---|---|---|
| GitHub | `(owner, repo, issue_number: int)` | only after composing |
| Linear | issue id (uuid), plus a human identifier like `ENG-123` | uuid is not; identifier is |

ADR 0001 derives execution state from branch names shaped
`apiary/<ref>-attempt-2`, and derives `review` from a PR that *references the
task*. Both need a ref that is a single, stable, branch-safe token. GitHub does
not supply one — the number alone is unique only within a repository, and
apiary's dependency graph is `int`-keyed today (`github/readiness.py`). Linear
supplies `identifier`, which is exactly right, and a uuid, which is not.

**The contract needs a `ref` rule per server** naming which response field is
the durable task ref. This is load-bearing for the derived state machine that is
the point of the whole ADR, and it survives the rescope untouched.

Pagination diverges too — GitHub offers `page`/`perPage` *and* an `after`
cursor, Linear offers `before`/`after` cursors — so the client owns pagination
rather than passing it through.

### Verdict on Q1

**One contract, yes — and for two trackers the residual divergence is small
enough to state exhaustively:** `method: "create"` on GitHub, `body` vs
`description` on create, `owner`+`repo` vs `teamId` as scope constants, and two
different task-ref rules. That is four items, three of which are static values.

ADR 0001's `args` field is *already* a static YAML dict, so the scope constants
need no new machinery — they simply go in `args`, once the fiction that `args`
is the customer's query language is dropped. `method: "create"` goes there too.
What genuinely has no slot in the current contract is the body field name and
the ref rule. **Two fields.**

Calling that a translation layer, and therefore the adapter this epic exists to
delete, would be the wrong read. The distinction that matters is
**fixed-shape declarative mapping versus per-server executable code**: a block
that names two fields has no branches, no error handling and no lifecycle, and
adding a tracker still touches no Python. The honest statement is that ADR 0001
under-costed the block by two fields and mis-described its `args`, not that its
thesis is wrong.

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

apiary is more exposed than a CLI is, because the orchestrator is a long-lived
unattended process (#149) and there is no human to prompt.

### Exactly what was probed

Both servers challenge an unauthenticated `tools/list` with
`WWW-Authenticate: Bearer`:

```
POST https://mcp.linear.app/mcp          401  Bearer realm="OAuth", error="invalid_token"
POST https://api.githubcopilot.com/mcp/  401  Bearer error="invalid_request"
```

**Naming the metadata endpoints, since one of them is not where you would look.**
Each 401 carries a `resource_metadata` URL; that document names the
authorization server; the AS metadata then lives under *that* origin:

| step | GitHub | Linear |
|---|---|---|
| protected-resource metadata | `https://api.githubcopilot.com/.well-known/oauth-protected-resource/mcp/` → 200 | `https://mcp.linear.app/.well-known/oauth-protected-resource/mcp` → 200 |
| it names the AS as | `https://github.com/login/oauth` | `https://mcp.linear.app` |
| AS metadata | `https://github.com/.well-known/oauth-authorization-server/login/oauth` → **200** | `https://mcp.linear.app/.well-known/oauth-authorization-server` → **200** |

`https://api.githubcopilot.com/mcp/.well-known/oauth-authorization-server`
returns **401** — it sits behind the same auth wall as the MCP endpoint and is
not the right place to look. GitHub's authorization server is a different
origin, and the protected-resource document is what tells you so. Re-probed and
confirmed.

The grant lists, verbatim:

```
github.com/login/oauth   grant_types_supported: ["authorization_code", "refresh_token"]
mcp.linear.app           grant_types_supported: ["authorization_code", "refresh_token",
                                                 "urn:ietf:params:oauth:grant-type:jwt-bearer"]
```

### What holds, and what an earlier draft overstated

**Holds — `client_credentials` is absent from both.** Neither server offers the
conventional machine-to-machine grant. Any design that has the orchestrator
exchange a client id and secret for a token is designing a flow that does not
exist on either tracker.

**Corrected — "no machine-to-machine path exists at all" was wrong for Linear.**
`urn:ietf:params:oauth:grant-type:jwt-bearer` (RFC 7523) is a genuine
assertion-based grant: a client presents a signed JWT and receives an access
token, with no browser and no user present. It is a real option and #149 should
know it exists.

**Why it is nonetheless declined for now**, recorded so the choice is
re-openable rather than forgotten:

- RFC 7523 requires a **pre-established trust relationship** — the assertion is
  signed with a key the authorization server already recognises. That is a
  registration handshake apiary would have to negotiate per customer, versus an
  API key which is one line of config.
- **Linear does not document jwt-bearer as a public integration path.** It is
  advertised in metadata but absent from the published OAuth and MCP docs, whose
  stated non-interactive answer is a bearer API key. The plausible fit for the
  advertised grant is Linear's enterprise-managed Okta/SAML configuration, which
  is not apiary's deployment shape.
- **Uniformity has real value here.** The whole premise of ADR 0001 is that
  trackers are interchangeable. One credential mechanism across every profile
  keeps `doctor`'s check, #149's failure classification, and the config block
  identical per tracker. Adopting jwt-bearer for Linear alone buys a marginally
  better credential lifetime at the cost of a second auth code path — the first
  thing that would make the tracker seam Linear-shaped.

So the recommendation below is a **trade-off, not a necessity**. If Linear
credential rotation becomes a real operational burden, jwt-bearer is the
documented way out and this paragraph is the reason it was not taken first.

### Supported unattended mechanisms

| server | mechanism | header |
|---|---|---|
| **GitHub** | GitHub App installation token, or a PAT. A **local stdio** server binary also exists, taking `GITHUB_PERSONAL_ACCESS_TOKEN` from the environment — no network OAuth at all. Its own docs name GitHub App auth as the mode "for non-interactive stdio deployments". | `Authorization: Bearer <token>` |
| **Linear** | Linear API key, or an OAuth app-actor token. Linear's docs state the MCP server accepts a bearer token or API key directly, bypassing the interactive flow. App-actor tokens attribute writes to the app rather than a person, which is what apiary wants. | `Authorization: Bearer <key>` |

### The GitHub tracker credential is one apiary already holds

The code host is GitHub and stays GitHub under ADR 0001, so the question is
whether the tracker profile needs a *second* credential. **It does not, and the
scopes already overlap — checked against `security.py`, not assumed.**

`swarm.security.REQUIRED_PERMISSIONS` is:

```python
{"contents": "write", "pull_requests": "write", "issues": "write", ...}
```

`issues: write` is **already in the required set**, justified in
`docs/security.md` as "read the contract, write `swarm:*` labels and comments".
ADR 0001 deletes the labels, but the permission that survives is exactly the one
the tracker profile needs: read issues for intake, write issues for comment and
create. **Same token, second use, zero new provisioning.**

There is a catch, and it points at which GitHub MCP server to use:

- `swarm.security.assert_scoped_token` **refuses** the `ghp_` / `gho_` / `ghu_`
  / `ghr_` prefixes — classic and OAuth tokens — on the goal-sentence grounds
  that their scopes are "verbs, not repositories" and so reach every repository
  the account can reach. It accepts only `github_pat_` (fine-grained) and `ghs_`
  (app).
- The **remote** server at `api.githubcopilot.com` advertises its
  `scopes_supported` as classic OAuth scopes (`repo`, `read:org`, `project`, …)
  — the token family apiary's own security module exists to reject.

**Therefore: use the local stdio `github-mcp-server` for the GitHub profile.**
It takes its credential from the environment, so apiary's existing fine-grained
PAT is passed through unchanged and `assert_scoped_token`'s guarantee is
preserved; and it talks to `api.github.com`, which is **already in
`security.EGRESS_ALLOWLIST`**. The GitHub tracker profile therefore needs no new
credential *and* no new egress hole. Only Linear adds one host,
`mcp.linear.app`, and `APIARY_EGRESS_ALLOW` is inert — the real edits are
`security.EGRESS_ALLOWLIST` and `compose.yaml`, which `tests/test_security.py`
asserts agree.

Deferring Jira removes the worst part of the auth story outright: Atlassian's
API-token mode is **admin-gated and off by default**, so it could be switched
off underneath a running apiary by someone who never heard of apiary. Nothing in
the GitHub or Linear path has that property.

### Verdict on Q2

**Named mechanism: a pre-minted static credential, supplied to the orchestrator
as an environment-sourced secret and sent as an HTTP header on every MCP
request. The orchestrator drives no OAuth flow, opens no browser, and holds no
refresh token.** Unattended runs are supported on both in-scope servers under
this mechanism. It is chosen for uniformity and simplicity over Linear's
jwt-bearer grant, not because Linear lacks an alternative.

Consequences apiary must accept:

- **The credential is a config input, not something apiary obtains.** The
  tracker block names an env var; the operator mints the token out of band. For
  GitHub, the env var is one that already exists.
- **`doctor` must prove authorization before a cycle needs it**, which #150
  already requires. The check is a `tools/list` against the configured endpoint.
  A 401 must produce the named failure and the command that mints the token, and
  those commands differ per server.
- **A 401 is not retryable.** #149's rule — back off on 5xx and rate limits,
  fail fast on 4xx — is load-bearing rather than theoretical, because expiry is
  the failure that will actually occur. `docs/security.md` already advises the
  shortest tolerable expiry on fine-grained PATs, which makes 401 a routine
  event rather than an exceptional one.

---

## What in ADR 0001 is wrong

ADR 0001 is merged and published; its amendment is being handled separately.
This section exists so that amendment can be written from a precise list rather
than from a re-reading of this whole document. Two defects, both in
**"Configuration, not code"**:

**Defect 1 — the contract shape `{ tool, args }` cannot make the calls.**

```yaml
comment: { tool: create_comment }
create:  { tool: create_issue }
```

A capability that carries only a tool name cannot call `issue_write`, which
needs `method: "create"` pinned as an argument, and cannot write an issue body
to a server that calls the field `description`. Two fields are missing from
every capability: a **field map** from apiary's canonical fields to this
server's names, and a **`ref` rule** naming which response field is the durable,
branch-safe task ref. The scope constants (`owner`/`repo`, `teamId`) need no new
field — they belong in the existing `args`, which is already a static dict.

**Defect 2 — the `filter:` intake example is not a real parameter, and the
sentence above it mis-describes what intake is.**

```yaml
intake: { tool: list_issues, args: { filter: "label=agent-ready" } }
```

Neither GitHub's nor Linear's `list_issues` has a `filter` parameter; both take
discrete typed filters (`labels[]`/`state` and `teamId`/`stateId`/`assigneeId`).
The surrounding claim — that intake "stays in their language", a JQL string or a
Linear filter the customer wrote — is true of Jira alone. The accurate statement
for the two in-scope trackers is that **intake args are the server's own typed
filter parameters, forwarded verbatim and never parsed by apiary**. The
no-parsing guarantee survives intact; the "customer's own query language" framing
does not.

One further line to add rather than correct: GitHub's `search_issues` is
semantic and is therefore inadmissible in the reconcile loop, for the same
determinism reason the ADR already gives in "Deterministic and model-driven MCP
are different call sites".

## Recommendation

**1. Amend the capability contract by two fields.**

```yaml
tracker:
  mcp: github
  auth: { header: Authorization, scheme: bearer, value_env: GITHUB_TOKEN }
  intake:
    tool: list_issues
    args: { owner: shahrestani-me, repo: apiary, labels: [agent-ready] }
    ref:  number                                   # response field -> task ref
  comment:
    tool: add_issue_comment
    args: { owner: shahrestani-me, repo: apiary }
  create:
    tool: issue_write
    args:   { owner: shahrestani-me, repo: apiary, method: create }
    fields: { body: body }                         # apiary field -> server field
```

and the same block for Linear:

```yaml
tracker:
  mcp: linear
  auth: { header: Authorization, scheme: bearer, value_env: APIARY_LINEAR_TOKEN }
  intake:  { tool: list_issues,   args: { teamId: "...", stateId: "..." }, ref: identifier }
  comment: { tool: create_comment }
  create:  { tool: create_issue,  args: { teamId: "..." }, fields: { body: description } }
```

Declarative, no per-tracker code, and now actually able to make the calls.

**2. Ship both profiles in-repo** as defaults a customer selects and overrides,
so `tracker: { profile: github, args: { owner: ..., repo: ... } }` is the whole
config in the common case. This is what keeps "a new tracker costs
configuration" true in practice rather than on paper.

**3. Use the local stdio GitHub MCP server, not the remote one.** It preserves
`assert_scoped_token`'s fine-grained-only guarantee, reuses the existing
`GITHUB_TOKEN`, and needs no egress change. The client must support both stdio
and streamable HTTP regardless, because Linear ships remote-only.

**4. On auth, commit rather than amend:** the orchestrator is a bearer-token
client and nothing else. #150's doctor check names the per-server minting
command on a 401; #149 fails fast on 4xx.

The bar for reversing this: if a tracker were to need a *behaviour* rather than
a field mapping — a two-call create, or a write that must be retried differently
— the declarative contract has run out and the adapter question genuinely
reopens. Neither in-scope tracker requires that, and neither does Jira.

## Must not be precluded — Jira

Jira is deferred, not discarded. It is the most different of the three workflow
models and the one most likely to break a contract designed against two similar
trackers, so its known requirements are recorded here as constraints on the
design above. Nothing in the recommendation should make any of these
unreachable without a rewrite.

| Requirement | What it constrains |
|---|---|
| `searchJiraIssuesUsingJql` takes **`jql`, a genuine free-text query DSL** | `intake.args` must stay an opaque pass-through dict. If it is ever narrowed to a typed schema validated against GitHub's and Linear's parameter names, Jira cannot be added without reopening it. This is the single most important constraint on this page. |
| Every call requires **`cloudId`**; create additionally requires **`projectKey`** and **`issueTypeName`** | The `args` dict must accept arbitrary per-capability constants, including ones with no analogue on the other two. `issueTypeName` in particular has no GitHub or Linear counterpart. |
| Fields are **`summary`/`description`**, and comment is **`commentBody`** | The field map must cover the comment capability too, not only create. GitHub and Linear both use `body` for comments, so a two-tracker design would be tempted to hard-code it. Do not. |
| Task ref is **`issueIdOrKey`** (`"PROJ-123"`) | Already satisfied by the per-server `ref` rule. Jira is the easy case here. |
| Auth is **HTTP Basic**, `base64(email:api_token)` — not Bearer | The `auth` block must carry a **scheme**, not assume Bearer. The recommendation above already has `scheme:`, and this is the reason it is there rather than implied. |
| The API-token mode is **admin-gated and off by default**, Cloud only — no Data Center or Server | `doctor` must be able to report "the mechanism is disabled for this org" as distinct from "the credential is wrong". A two-tracker doctor would only ever need the second. |
| Pagination is **`nextPageToken`**, a third scheme | Confirms the client must own pagination rather than pass it through. |

The cheapest insurance is that table's right-hand column taken as a whole: keep
`args` opaque, keep `auth.scheme` explicit, keep the field map available on
every capability, and keep pagination inside the client. All four are already in
the recommendation, and three of them are justified only by Jira.

## Sources

- GitHub MCP server tool schemas, read live from the server in this session; and
  <https://github.com/github/github-mcp-server> — auth modes, local stdio server,
  GitHub App for non-interactive deployments
- <https://linear.app/docs/mcp> — endpoint, transport, OAuth 2.1 with dynamic
  client registration, bearer token / API key alternative
- <https://blog.fiberplane.com/blog/mcp-server-analysis-linear/> — official
  Linear tool list and `list_issues` parameter analysis
- <https://datatracker.ietf.org/doc/html/rfc7523> — the JWT assertion grant
  Linear advertises
- <https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/298>
  — open proposal for non-interactive OAuth flows in MCP; confirms no standard
  mechanism exists yet
- `src/swarm/security.py` (`REQUIRED_PERMISSIONS`, `assert_scoped_token`,
  `EGRESS_ALLOWLIST`) and `docs/security.md` §"Minimum scopes" — the
  credential-overlap finding
- Live probes of `mcp.linear.app` and `api.githubcopilot.com`: `WWW-Authenticate`
  challenges, protected-resource metadata, and authorization-server metadata at
  the endpoints tabulated above
- Jira constraints retained from the wider survey:
  <https://github.com/atlassian/atlassian-mcp-server>,
  <https://support.atlassian.com/rovo/docs/getting-started-with-the-atlassian-remote-mcp-server/>,
  and the Atlassian Community announcement of API-token authentication
