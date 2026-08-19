"""apiary's own judgments about its own execution, and nowhere else's facts.

`docs/adr/0002-apiary-owns-a-thin-task-store.md` is this package, and the ADR
before it is why the package has to justify itself line by line.

**What happened.** ADR 0001 said agent execution state is derived, not stored:
five lifecycle states recomputed each cycle from the code host and the
containers. That holds for those five, because each is a fact about something
external - an open pull request *is* review, a live container *is* claimed. Two
days later #154-#156 added a failure signature, a consecutive-failure count and
a per-blocker retry budget, and none of the three is a fact about anything
external. They are apiary's judgments about its own execution: they cannot be
derived, they must survive the process, and ADR 0001 had left nowhere to put
them. So they went into the customer's issue body, riding the same `PATCH` as
the attempt counter. That was not carelessness; it was the only writable place
left. This package is the place that should have existed.

**The discipline, which is the whole design.** The store holds a task ref,
apiary's attempt stamp, the blocker signature, the streak and the renewal
count - and refuses to hold a goal, a file list, a verify command, a title, a
description or a dependency. Those belong to the tracker, and the moment the
store carries one of them there are two records of one fact and a sync problem
to write, which is exactly what `architecture-v2.md`'s "GitHub is the database"
was buying its way out of. Because the store holds only fields the tracker
never had, **there is nothing to reconcile** - and that property is not a
happy accident to be preserved by good intentions. `TaskJudgement` is a closed
five-field record and the seam takes nothing else, so adding a tracker field
here is a schema change somebody has to argue for in review.

**What is deliberately *not* here: the attempt counter's home.** The ADR's
table lists `attempt` among the things the store holds, and it does hold one -
but as the stamp saying *which attempt a judgment was recorded at*, not as the
counter's residence. The counter itself stays in the issue marker
(`docs/issue-contract.md` §5), because the worker reads it: a worker is a
container with no socket, no host filesystem beyond `/artifacts`, and its only
view of the task is the issue body it fetches. It derives its branch name and
its result filename from that number. Moving the counter here would mean
passing it down through `spawn`, the container environment and the entrypoint
before the worker could name a branch - a change to three modules the ticket
does not own, and a change to the one number `github/branches.py`,
`orchestrator/recovery.py` and `worker/result.py` all key on. So the counter
is read where it always was, and the stamp is how a judgment recognises that
somebody moved it underneath (see `TaskJudgement.matches`).

That leaves exactly one authority per fact, which is the property that matters:
the tracker owns the counter, the store owns the judgment about it, and neither
one holds a copy of the other's.

**The seam.** `TaskStore` is three methods - read everything, write one row,
close - and `SqliteTaskStore` is the one implementation. Three methods is not
minimalism for its own sake: `read()` returns the whole project in one call
because a cycle wants every task's judgment at once and an organization
database replacing this seam wants one round trip rather than one per issue,
which is the same budget argument `reconcile.Snapshot` makes about the issue
listing. Building the remote implementation is out of scope until something
needs it (ADR 0002, decision 4); what is in scope is that swapping it touches
no caller.

**Failing loudly is a feature.** A store that cannot be read must never be
answered with an empty one. An empty store reads as "every task is on attempt
0 with no blocker", which grants every task in the ledger a fresh retry budget
- silently, at the exact moment something is already wrong. So a file that is
not a database, or carries a schema this build does not know, or belongs to a
different project, raises `StoreCorrupt` naming the path and the fix, and the
run stops. A store that is simply *absent* is the ordinary first run and is
created; the seeding path in `github/ledger.py` fills it from whatever the
legacy markers still say, so upgrading a repository mid-flight does not reset
anybody's budget either.
"""

from __future__ import annotations

from .base import (
    SCHEMA_VERSION,
    StoreCorrupt,
    StoreError,
    StoreMissing,
    TaskJudgement,
    TaskStore,
    record_judgement,
)
from .sqlite import (
    DEFAULT_STORE_DIR,
    STORE_DIR_ENV,
    SqliteTaskStore,
    store_path,
    store_root,
)

__all__ = [
    "DEFAULT_STORE_DIR",
    "SCHEMA_VERSION",
    "STORE_DIR_ENV",
    "SqliteTaskStore",
    "StoreCorrupt",
    "StoreError",
    "StoreMissing",
    "TaskJudgement",
    "TaskStore",
    "record_judgement",
    "store_path",
    "store_root",
]
