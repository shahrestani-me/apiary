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
false — though not for the reason given here until #217, which was that N
consecutive mechanical failures write the *same* result filename. They do not,
and have not since #177: `write_result` never replaces an existing record, it
bumps the filename on collision, so two mechanical failures at one attempt write
two files, and since #218 `latest` orders records by what they say rather than
by how the directory lists them.

What is not derivable is the *count*. `summarise_dir(...).latest` hands back
**one record per task** — the newest — so the second mechanical failure displaces
the first in that map rather than adding to it, and a streak is not a thing the
map has room for. That is why `infrastructure_streaks` counts *transitions*: a
transition fires once per verdict a cycle actually acted on, which is a fact
about what the orchestrator did rather than one the results directory holds. A
run at the cap reads as `eligible` from the artifacts while the label says
`swarm:failed`.

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

### And three more that the shadow window found (#146)

Wiring the resolver into the live cycle added three divergence classes to the
three above. None of them changes the decision; all three are things a reader of
a divergence log has to be told, because they look identical to a real one.

**Two are the cycle acting after it reads, not a disagreement about a fact.** A
cycle samples the world once, at the top, and then writes: the merge gate merges
a pull request that was open when the listing was taken, and the dispatcher
claims and spawns after the container listing. So on the cycle a task lands, the
control plane says `landed` and the derived side — reading the pre-merge
listing — says `review`; on the cycle a task is dispatched, the same happens for
`claimed`. Both converge on the next cycle. They could only be removed by
feeding the cycle's own writes back into the observation, which is the sourcing
violation the whole exercise exists to avoid, so they are named instead.

**One is `claimed` being liveness rather than existence.** `#187` gave
`Handle` the container's state so that an *exited* worker's container stops
reading as a claim, which it had to for a shadow window to be worth running at
all. The other edge of that is the create-to-start gap: `docker ps --all` lists
a container from the instant `docker create` returns, and until `docker start`
takes effect it reads `created`, so the resolver says not-claimed while the
label says claimed. `dispatcher.release` takes the opposite reading of the same
window and is right to — it is deciding whether to *act*, and "any container
blocks a release" is the only reading that cannot produce two workers — but a
resolver that decides nothing should read what is true rather than what is safe.

**And one finding that is a gap rather than a limit.** A work item a human
closed *as not planned* escalates to `needs-human` through
`reconcile._closed_verdict`, and the resolver had no rule for it even though
`TaskFact.state_reason` carries the fact. Unlike the three above it is
derivable, and #147 derived it (`derived.TaskFact.abandoned`). The
classification is kept and should now never fire: if it does, `state_reasons`
has stopped reaching the observation, which is a defect in the wiring and
belongs in the log under a name rather than in the unexplained count.

### What the orchestrator does about the three, now that the resolver decides (#147)

#147 made the derived value authoritative: `github/readiness.py`,
`orchestrator/dispatcher.py`, `orchestrator/reconcile.py` and the merge gate
(`checks.plan_checks`, `mergeability.run_mergeability`) take their state from
`orchestrator/authority.py` rather than from a `swarm:*` label. Making a
value decisive does not make it complete, and this section is the answer to the
question that follows: **what does the orchestrator do about the three states
above, when the thing that cannot see them is the thing deciding?**

The short answer is ADR 0002's sentence, applied literally. The authority is the
resolver **plus apiary's own store and its own run-scoped counters**, and the
join is one module. Each of the three, and its cost:

**The infrastructure ceiling stays in the orchestrator's own counter.** A task
whose streak has reached `APIARY_MAX_INFRASTRUCTURE` is `needs-human` whatever
the resolver says. The counter was always run-scoped; what the label added was
that its *consequence* survived a restart, and it no longer does — a resumed run
gives a mechanically-broken task another `cap` free failures before escalating
again. Exit 2 consumes no attempt, so that costs cycles rather than budget.

**The retry budget is read from the store, in both directions.** The resolver's
`needs-human` is arithmetic over code-host evidence and is now *advisory*:
`_retry_or_give_up` gives up on `streak`, so the store decides, with ADR 0002's
own fallback (a missing judgment reads as the attempt counter — absence
escalates and never grants). The second direction is the one that was not
obvious and is the more dangerous: a task apiary has given up on leaves **no
code-host evidence at all** once its process is gone. Results live in the run
directory and a run directory is per run; the observation takes branch names off
*open* pull requests, because a remote branch listing is a call no cycle makes.
So a failed task resolves to `eligible` from scratch on the next process, and
`swarm:failed` was what carried the verdict across the restart. The store
carries it now.

**A revival is recorded as what it is — one granted attempt.**
`planner.revive` "deliberately resets nothing", so a revived task reads spent
from every source there is; under the resolver it would be re-escalated the
instant it was revived and the goal gate could never unstick a run. The grant is
held run-scoped and lapses the moment the attempt it granted is spent, where the
streak revive never reset caps the task again. A restart forgets a pending
revival, which is the safe direction — a forgotten revival escalates, it never
grants a budget — and it is the opposite of what the label did.

Two further findings, both from making the value decide rather than observe:

**`plan_reconcile` is incremental and the resolver is absolute.** The label was
quietly doing double duty as the *previous* state, and two of reconcile's rules
are edge-triggered: "a claimed task whose worker finished" (the worker exits, no
pull request exists, the resolver correctly says `eligible`) and "a task in
review whose pull request was closed unmerged" (a closed pull request is not in
the listing at all). Waiting for `claimed` or `review` would have silently
removed the retry engine and forgiven every rejected pull request. The
orchestrator therefore carries its previous belief, in the register
`_infrastructure` and the update budget already occupy. A task a process has
never seen is seeded from the label, which is the one place a label still
reaches a decision and is documented as such; #152 moves that seam to the store.

**`landed` is terminal within a run, and the world stops showing it.** A merged
pull request leaves the open listing, so once `Closes #<n>` has been honoured
the only remaining evidence is the work item being closed as completed — and two
ordinary things take that evidence away. `checks._decide_passed` writes
`swarm:done` *before* GitHub has processed the closing keyword, and a human can
reopen a finished issue. In both the resolver reads `eligible` and the
dispatcher would put a worker back on code that is already on the default
branch. So the belief ratchets: once a run has seen a task land it stays landed,
and a fresh process seeds that from the label along with the previous belief.

**`claimed` is liveness, and a dispatcher has to read existence.** The resolver
reads `docker ps --format {{.State}}` and is right to — that is what "a live
worker container" means, and #187 exists because reading existence held every
task in `claimed` from the moment its worker exited. But `dispatcher.release`
takes the opposite reading *because it is deciding whether to act*, and dispatch
is the other half of that decision: a container that exists blocks a spawn, or
two workers can end up over one file set. The label carried existence for free.

And one of the three above the shadow window found is now closed rather than
classified: a work item closed **as not planned** is derivable, and
`derived.TaskFact.abandoned` derives it.

**The cutover shipped before its own gate.** #146's criterion was ten
consecutive greenfield runs with zero unexplained divergences, and no credential
in this environment can run one — the same wall the corpus README and
`docs/demo-run.md` hit. `APIARY_STATE_SOURCE=labels` restores the pre-#147
behaviour completely and is the guarantee that replaced it. That is weaker than a
clean window, and it is recorded here rather than left for whoever reads the
first real run.

### What a clean shadow window is evidence of

Worth stating before #152 is decided on one. `reconcile.plan_reconcile` computes
a cycle's label writes from the same issue listing, results directory and
container listing the resolver reads, so **for a task the cycle relabelled the
two sides were handed the same observation**: their agreeing shows two reducers
implementing one set of rules, which is the claim "the label is a redundant
cache" and is what this ADR needs — but it is not the resolver tracking a world
nobody told it about.

The independent comparisons are the tasks a cycle did *not* write, where the
label is the accumulation of many earlier cycles over earlier observations and
the derived state is one absolute reading of now. A cycle writes at most a small
constant number of labels out of a ledger of N, so the independent share tends to
`(N − O(1)) / N` — but it is counted rather than assumed:
`orchestrator/shadow.py` reports it per cycle and `swarm show` totals it. A clean
window with a small independent count is a weak result.

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
