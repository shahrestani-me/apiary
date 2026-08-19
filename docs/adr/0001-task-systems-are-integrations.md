# ADR 0001 — Task systems are an integration, not a subsystem

Status: **proposed**
Date: 2026-08-18
Amends: `docs/architecture-v2.md` ("GitHub is the database", "Labels are the
protocol"), `docs/issue-contract.md` §3–§4

## The one-line summary

**apiary is a developer agent, not a task manager.** It reaches whatever task
system a customer already runs through that customer's own MCP server, and it
never models, stores or renders their workflow.

## Context

v2 put the control plane on GitHub, and for a single-repo swarm on one machine
that was the right call: state survives the process, a human can intervene, and
nothing needed inventing. The goal has since changed. apiary is meant to improve
an organization's SDLC, and organizations do not share a task system — one
customer runs Linear, the next runs Jira, the third runs GitHub Issues, and each
has transitions somebody configured for reasons apiary will never see.

Under that goal, one decision does not survive: **two different state machines
are currently fused into one.**

| | Owner | Scope |
|---|---|---|
| Agent execution state — eligible, claimed, review, landed | apiary | private, internal, identical for every customer |
| Organization workflow — Backlog, In Progress, QA, UAT, Done | the customer | arbitrary, per-team, already exists |

apiary stores the first one *inside* the second. `github/labels.py` provisions
six `swarm:*` labels into the target repository and `orchestrator/reconcile.py`
uses them as working memory — `Snapshot`, `_label_names`, `fold`,
`rewrite_marker`, `bump_attempt`.

Three things follow, and all three are disqualifying for a multi-customer tool:

- **It writes apiary's vocabulary into a customer's tracker.** Six labels the
  customer did not choose, provisioned on every run.
- **It assumes their transitions.** `docs/issue-contract.md` §4 is a transition
  table. It is apiary's table, applied to their work items.
- **It reimplements a workflow engine.** Linear and Jira *are* workflow engines.
  Both already move issues on branch-created, PR-opened and PR-merged, driven by
  rules each customer wrote. There is nothing here worth rebuilding.

## Decision

**1. apiary ships no tracker integration code.** Task systems are reached
through a per-organization MCP server named in configuration, running with
credentials the customer authorizes. Linear MCP, Jira MCP and GitHub MCP are
maintained by their vendors; apiary holds a config field, not an adapter.

**2. Organization workflow is never modelled, stored, or rendered.** apiary does
not know whether a customer calls it "In Review" or "QA", and must not learn.
Where a work item sits in *their* process is theirs, and their existing
integration already moves it.

**3. Agent execution state is apiary's own** and lives where apiary already has
legitimate integration: the code host and the run artifact directory. Never in
the customer's tracker.

**4. The GitHub integration narrows to the code host** — branches, pull
requests, CI, merge. That integration is wanted and stays. GitHub Issues becomes
one tracker among several, reached like any other.

### Intake stays in their language

"Which items may the agent pick up?" is a query the customer writes in their own
system — a JQL string, a Linear filter, a label. apiary runs it and does not
parse it. Dependencies are read natively too: Jira has `blocks`, Linear has
`blocked by`, and neither needs a `## Blocked by` markdown section.

### Configuration, not code

```yaml
tracker:
  mcp: linear
  intake:  { tool: list_issues,    args: { filter: "label=agent-ready" } }
  comment: { tool: create_comment }
  create:  { tool: create_issue }
```

A capability contract naming which tool fulfils which need. Per customer this is
a few lines. Per customer today it would be a Python adapter.

## The internal workflow — the only one apiary owns

These are execution states of a fleet, and every one of them is **derived, not
stored**:

| State | Derived from |
|---|---|
| `eligible` | dependencies discharged and not yet landed — recomputed each cycle |
| `claimed` | a live worker container carries `apiary.run` and the task ref |
| `review` | an open pull request references the task |
| `landed` | that pull request merged |
| `needs-human` | attempts exhausted, or the infrastructure cap hit |

The difference from a board is not cosmetic. "In Progress" is a claim about
somebody's week. `claimed` is a claim about a container that exists right now
and is falsifiable with `docker ps`. Nothing here needs a durable write, which
is why removing the label store costs no crash-safety: a branch named
`apiary/<ref>-attempt-2` carries the ref and the counter, an open PR *is* review
state, and a merge *is* done. The code host was already holding this.

Only `needs-human` is reported outbound, because it is the one state the
customer's own integration cannot infer.

### Three of these are not derivable from the code host alone (#145)

The table above says "derived from" and names the code host and the containers.
Building the resolver found that incomplete in three specific ways, and they are
recorded here because #146's shadow window will surface them as divergences and
somebody has to know in advance which divergences are *expected*.

**The infrastructure ceiling is not derivable at all.** `needs-human` is listed
above as "attempts exhausted, or the infrastructure cap hit". The second half is
false. `infrastructure_streaks` counts *transitions*, and exit 2 deliberately
does not bump the attempt counter — so N consecutive mechanical failures write
the *same* result filename, and the results directory cannot tell one from
three. A run at the cap reads as `eligible` from the artifacts while the label
says `swarm:failed`.

**A renewed per-blocker budget is not derivable from the code host.**
`_retry_or_give_up` gives up on `streak`, not `attempt`, and the blocker
signature is an ADR 0002 store judgment. A task at attempt 3 against a cap of 3
looks spent from the branch *and* the pull request, so a second code-host source
does not rescue it.

**A goal-gate revival is not derivable while it is in flight.** `planner.revive`
"deliberately resets nothing", so the counter reads spent while the label reads
ready. It converges on merge, which is what the `landed > needs-human`
precedence exists for — but it diverges for the cycles in between.

The honest statement is therefore narrower than the one above: **the five
lifecycle states are derived from the code host, the containers, the run
artifacts, and apiary's own store.** ADR 0002's store is not an addition beside
the derivation; two of the five states need it. That does not weaken the
decision — the store holds only apiary's own judgments, so nothing here is a
fact the tracker owns — but a reader who took the original table literally would
build a resolver that reads three states wrong.

## Deterministic and model-driven MCP are different call sites

MCP is a protocol, not a model. Plain code can call an MCP tool, and the
distinction is load-bearing here:

- **Programmatic** — the reconcile loop: fetch intake, post the PR link, flag
  `needs-human`. apiary decides when and with which arguments; MCP is the
  transport that makes it tracker-agnostic. Deterministic.
- **Model-driven** — a planner turning a prompt into tickets, where a model
  choosing the call is the point.

The loop stays programmatic, and this is not fastidiousness. Measured on this
host, roughly 40% of `propose_edits` calls emit broken output. A model deciding
whether to call `create_issue` inside a control loop will duplicate tickets,
skip reports and invent keys — and unlike bad code, a bad tracker write is not
caught by CI.

## What the console shows

The internal workflow above, and nothing else. Which worker holds which task,
attempt number, cycle, container state, why a run stalled. No tracker shows
this, which is the whole reason it exists here: Jira can say an item is "In
Progress"; it cannot say worker 3 is on attempt 2 of cycle 7, blocked on an
inference that took 163 seconds.

A console rendering Backlog / In Progress / Done would be reimplementing the
customer's board, which decision 2 forbids.

## Consequences

**Removed**

- `github/labels.py` — provisioning `swarm:*` into customer repositories
- the six state labels and the §4 transition table
- `resolve_state_label`, `LabelRepair`, the exactly-one-state-label repair —
  pathologies that exist only because GitHub labels are a set and can be zero or two
- most of the 22 tracker-write call sites; `Snapshot` / `_label_names` / much of `fold`
- `## Blocked by` parsing wherever the tracker has native relations

**Kept**

- dependency-ordered scheduling, the dispatcher, the workers
- `orchestrator/mergeability.py`, the CI and merge half of `orchestrator/checks.py`,
  `worker/pr.py` — all code-host, all legitimate
- `artifacts.py` and `runs` / `show` — the fleet ops view

**New**

- an MCP client in the orchestrator, with the retry discipline `github/client.py`
  already reasons through: backoff on 5xx and rate limiting, fail fast on 4xx
- per-organization credential handling. This is an improvement — apiary never
  holds a Jira token — but it needs a headless story, since interactively
  authorized MCP servers are absent in cron and CI runs
- an egress hole for MCP endpoints. `security.py` generates the proxy allowlist
  from a single tuple deliberately; note that `APIARY_EGRESS_ALLOW` is documented
  in four places and read by none, so the real edit is `compose.yaml`

## What this invalidates upstream

`docs/architecture-v2.md` says "GitHub is the database" and "Labels are the
protocol". Both were sound for the problem that document was solving — the
orchestrator holds no irreplaceable state, and a human can see the run. Those
needs remain; what changes is that they were paid for by borrowing the
customer's tracker. The code host settles the same debt without touching anyone's
workflow.

`docs/issue-contract.md` §3 and §4 become a description of the GitHub Issues
adapter, not of the system.
