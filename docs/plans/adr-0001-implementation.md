# ADR 0001 implementation — replacing the label control plane

Plan for `docs/adr/0001-task-systems-are-integrations.md`, amended by ADR 0002 and 0003.
Status: **published as epic #140 with 16 tickets.**
Revised 2026-08-19, after #135–#138 and #154–#156 landed.

## Problem

The six `swarm:*` labels are the control plane: provisioned into target
repositories by `github/labels.py`, driven by the transition table in
`docs/issue-contract.md` §4, and used as working memory by
`orchestrator/reconcile.py`. ADR 0001 removes them. The removal spans ~22
tracker-write call sites across 8 modules, plus `github/readiness.py`'s
int-keyed dependency graph — while apiary runs on itself, on that control plane.

## Outcome

apiary reaches any task system through a per-organization MCP server named in
config, writes nothing of its own vocabulary into a customer's tracker, and
derives its internal state from the code host and run artifacts. GitHub Issues
goes through the same MCP path as everything else, with no special case — which
is what proves the seam is not GitHub-shaped.

## Non-goals

- Linear and Jira adapters. Follow-up epic once the seam is proven.
- Native dependency relations (Jira `blocks`, Linear `blocked by`). Follow-up.
- Any console UI for configuring MCP — already out of scope in epic #128.
- Changing what the reconciler *decides*. This replaces where state lives, not
  the scheduling policy.

## Decisions taken

| Decision | Choice | Why | Reversible? |
|---|---|---|---|
| Relationship to epic #128 | Absorb #131 as the first ticket, retargeted | It is already "announcement only" and already writes to `events.jsonl` — the ADR's derived-state substrate, ticketed before the ADR existed | Yes |
| Migration shape | Shadow, prove, then cut over | 988 tests pass today against label semantics; the only honest proof that derived state reproduces them is running both and diffing | Yes |
| ~~Self-hosting~~ | **Withdrawn** | apiary has never run against its own repository — zero swarm-minted branches or PRs, and the one recorded attempt (`docs/demo-run.md`) died before planning. Verification is against a scratch greenfield target, which is what the swarm actually does | — |
| Proof bar | GitHub Issues through MCP, Linear later | Same code path with no special case proves the seam; a second tracker then costs config, not migration | Yes |

## Approach

Four movements, in order:

1. **Make internal state observable** — per-task lifecycle events carrying the
   task slug (which already exists as `Transition.task_id`), and branch names
   carrying ref plus attempt. Nothing changes behaviour.
2. **Prove derived state reproduces label state** — a resolver computing
   eligible / claimed / review / landed / needs-human from code host,
   containers and artifacts; validated offline against recorded runs, then
   shadowed live with divergence logged as an event. **This is the go/no-go.**
3. **Cut over** — readers move to derived state behind a fallback flag; the
   worker stops touching the tracker entirely.
4. **Route the tracker through MCP** — programmatic client with the retry
   discipline `github/client.py` already reasons through, a capability-contract
   config, and GitHub Issues proving it. Then delete the label control plane.

## Risks

| Risk | Likelihood | Impact | Mitigation / early warning |
|---|---|---|---|
| Derived state cannot reproduce label state in some case | Medium | Fatal to the ADR | T4/T5 are an explicit go/no-go before anything is deleted |
| Cutover misbehaves in ways 988 tests don't catch | Medium | Blocks apiary's own development | `APIARY_STATE_SOURCE=labels` fallback flag, removed only in #152 |
| Replay corpus is inadequate | High | #145 proves nothing | Corpus recorded *after* #141 — today's `events.jsonl` is cycle-level only — and from greenfield runs, since apiary does not run on itself |
| Nothing dogfoods the control plane daily | High | A derived-state bug surfaces on a demo, not on our own work | Direct consequence of withdrawing self-hosting; #146's shadow window is ten runs rather than two |
| MCP server absent or unauthorized in headless runs | Medium | Runs stall | S1 spike decides the headless auth story before #149 |
| Worker containers gain a new egress destination | Low | Security regression | Tracker MCP is orchestrator-only; T7 removes the worker's tracker access entirely |

## Open questions

| Question | Blocks | Owner |
|---|---|---|
| Do Linear/Jira MCP tool shapes fit one capability contract, or need per-server mapping? | #150 | S1 spike |
| Headless/cron auth for interactively-authorized MCP servers | #149 | S1 spike |
| Does `swarm show` print terminal labels anywhere that outlives T11? | #152 | #141 |

## Tickets

Published as epic #140.

| # | Title | Type | Size | Depends on |
|---|---|---|---|---|
| #141 | Per-task lifecycle events in the internal vocabulary (retargets #131) | feat | M | — |
| #142 | `TaskRef` replaces issue numbers through the internal model | refactor | M | — |
| #143 | Spike: MCP tool-shape survey and headless auth story | spike | S | — |
| #144 | Branch names carry task ref and attempt | feat | S | #142 |
| #145 | Derived-state resolver and offline replay equivalence suite | feat | M | #141, #144 |
| #146 | Shadow the resolver in the live cycle; divergence is an event | feat | M | #145 |
| #147 | Flip readiness, dispatcher and reconcile to derived state | feat | M | #146 |
| #148 | The worker stops writing to the tracker | refactor | S | #147 |
| #149 | MCP client in the orchestrator, with retry discipline and egress | feat | M | #143 |
| #150 | Tracker capability-contract config, with a doctor check | feat | M | #143 |
| #151 | GitHub Issues through the MCP path — the self-host proof | feat | M | #147, #149, #150 |
| #152 | Remove the label control plane | refactor | M | #148, #151 |
| #153 | Amend architecture-v2.md and issue-contract.md; mark ADR accepted | docs | S | #152 |
| #158 | The console board reads derived state, not labels | feat | M | #147 |
| #159 | apiary's own task store, holding the failure record (ADR 0002) | feat | M | #142 |
| #160 | The working modules stay framework-free (ADR 0003) | test | S | — |

**Critical path:** T1 → T4 → T5 → T6 → T7 → T11 → T12
**Parallel at the start:** #141, #142, S1
**Go/no-go gate:** T4 + T5. If derived state cannot reproduce label state, the
ADR needs revisiting before anything is deleted.

## Refinements to existing issues (not new tickets)

- **#131** — retargeted by T1; event payloads carry the task slug and derived
  state names rather than issue number and terminal label.
- **#132, #133, #134** — console tickets; vocabulary updated to the internal
  state names once T1 lands. Data source already correct (`events.jsonl`).

## Follow-ups (not in this epic)

- Linear MCP adapter and its OAuth setup
- Jira MCP adapter
- Native dependency relations, replacing `## Blocked by` parsing
- Retiring `github/client.py`'s issue endpoints once nothing calls them
