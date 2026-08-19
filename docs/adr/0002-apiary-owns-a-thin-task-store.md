# ADR 0002 — apiary owns a thin per-project task store

Status: **proposed**
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
