# ADR 0002 — apiary owns a thin per-project task store

Status: **accepted** — built in #159
Date: 2026-08-19
Amends: `docs/adr/0001-task-systems-are-integrations.md`

## The one-line summary

**Derived state was not enough.** apiary holds a small per-project store of the
judgments only it makes — and nothing else.

## Context

ADR 0001 said agent execution state is "derived, not stored": five lifecycle
states recomputed each cycle from the code host and the containers. That holds
for those five, because each is a fact about something external — an open pull
request *is* review, a live container *is* claimed.

It does not hold for everything, and the proof arrived two days later. #154–#156
added a failure signature (`blocker`), a consecutive-failure count (`streak`) and
a per-blocker retry budget. None of those is a fact about the code host, a
container, or the tracker. They are apiary's own judgments about its own
execution.

They were stored in the customer's issue body, riding the same `PATCH` as the
attempt counter (`docs/issue-contract.md` §5).

That is not carelessness. It is the predictable consequence of removing apiary's
store without giving it one back: the issue body was the only writable place
left. ADR 0001 closed a door and left no other, so the next piece of genuinely
internal state went straight back through it.

## Decision

**1. apiary owns a per-project task store**, holding only what apiary itself
judges and no external system is authoritative for:

| Holds | Never holds |
|---|---|
| task ref, attempt, blocker signature, streak, renewal budget | goal, files, verify command, title, description, dependencies |
| apiary's judgments about its own execution | anything the tracker owns |

**2. The five lifecycle states stay derived.** Storing a fact that can be read
from the code host creates drift and buys nothing.

**3. The store never mirrors the tracker.** This is the discipline that keeps it
from becoming a second issue tracker, and it is the whole reason the split above
is worth writing down. Two stores of the same fact is a sync problem; ADR 0001
inherited the absence of one from `architecture-v2.md`'s "GitHub is the database"
and that property is worth keeping. Because the store holds only fields the
tracker never had, there is nothing to reconcile.

**4. The backend is swappable, and starts local.** A local store — SQLite or
files under the run artifacts root — is correct for proof-of-concept work. A
production deployment switches to an organization database. Design the seam now;
build only the local implementation until something needs the other.

## Consequences

- The failure record moves out of the issue body. The body `PATCH` that carries
  attempt, blocker and streak is a tracker write, and it joins the removal list.
- `docs/issue-contract.md` §5 — "the counter is a single `PATCH` of the body" —
  describes a mechanism that is going away, and is demoted with §3 and §4.
- The store is a new durable artefact on a system whose orchestrator was designed
  to hold none. That is a real cost and it is accepted here deliberately: the
  alternative, demonstrated by #154–#156, is that the state lands in somebody
  else's database instead.
- A run started on a different host does not see a local store. Acceptable while
  the store is local and runs are single-host; it is the constraint that decides
  when the organization backend stops being optional.

## Alternatives considered

**Derive the failure record from run artifacts.** The results directory already
holds each attempt's testimony, so the signature could be recomputed. Rejected:
deriving a judgment from its own log is strained, it makes every consumer re-run
the classifier, and a signature that changes when the classifier changes is not a
signature.

**Leave it in the issue body.** Rejected — it is the thing ADR 0001 exists to
stop, and it puts apiary's retry accounting in a field a customer can edit.

**Put it in the code host** — a branch name, a PR comment. Rejected: encoding a
growing record into a ref name repeats the mistake, and comments are the
customer's surface as much as labels were.

## As built (#159)

Two things in the decision above did not survive contact with the code, and
both are recorded here rather than quietly diverged from.

**The attempt counter did not move.** Decision 1's table lists `attempt` among
the things the store holds, and the store does hold one — but as the *stamp*
saying which attempt a judgment was taken at, not as the counter's residence.
The counter stays in the issue marker. The worker reads it there: a worker is a
container with no socket and no view of the host filesystem beyond
`/artifacts`, its only picture of the task is the issue body it fetched, and it
derives both its branch name (`apiary/<ref>-attempt-<n>`) and its result
filename from that number. Moving it would mean threading it through
`ContainerManager.spawn`, the container environment and the worker entrypoint
before a branch could be named — three modules this ticket does not own, and
the one number `github/branches.py`, `orchestrator/recovery.py` and
`worker/result.py` all key on.

That leaves one authority per fact, which is the property decision 3 is really
protecting: the tracker owns the counter, the store owns the judgment about it,
and neither holds a copy of the other's. When the two disagree the tracker
wins and the judgment is dropped as stale, because a signature is a claim about
one attempt and a moved counter means it is no longer that attempt.

**The store is not under the run artifacts root.** Decision 4 says "SQLite or
files under the run artifacts root". A run directory is per *run*, so a store
inside one is empty at the start of every run — and an empty store reads as
"every task is on attempt 0 with no blocker", which grants every task a fresh
retry budget on every invocation. That is precisely the silent failure the
ticket's acceptance criteria forbid, so the store is per *project* and lives in
its own root (`APIARY_STORE_DIR`, default `.swarm/store`), a sibling of the
artifacts tree deriving nothing from it — the rule `artifacts.CONSOLE_ROOT_ENV`
already states.

**And the reason given above for the per-project store is not the reason it is
safe.** The paragraph before this one says an empty store "reads as 'every task
is on attempt 0 with no blocker'". The second half is true; the first half is
not, because the counter never moved into the store — it stays in the issue
marker, so `max_total_attempts_per_task` still bites on an empty store.

What actually protects the budget is two pieces of arithmetic in
`orchestrator/reconcile.py`, and both deserve naming because a later reader
tidying either one would open the hole this ADR only *thought* it had closed:

- `previous_streak = entry.attempt if entry.streak is None else entry.streak`.
  A missing judgment falls back to the attempt counter, and `streak` only ever
  increments alongside `attempt` and resets on renewal, so `streak <= attempt`
  always holds. The fallback is therefore the **largest** streak consistent
  with the counter: absence gives up sooner, never later. Simplifying that
  fallback to `0` is the change that would make the ADR's stated fear real.
- `renewed = bool(entry.blocker) and sig != entry.blocker`. Renewal — the only
  thing that can *extend* a budget — is gated on a blocker being **present**. A
  missing judgment yields an empty blocker, so a miss can never renew.

The decision stands; the argument for it was weaker than the code. Measured on
a task at attempt 5 against a cap of 3 with no store row: the join returns
`blocker='' streak=None renewals=0` and the verdict is `swarm:failed`. The safe
direction, reached by a mechanism this document had not identified.

SQLite won over a directory of JSON files on one requirement: a corrupt store
has to be recognisable as corrupt. A directory degrades a file at a time, and a
reader cannot tell "this task has no judgment yet" from "this task's judgment
is the file that got truncated" — the two are the same empty answer, and one of
them unbounds a retry budget.
