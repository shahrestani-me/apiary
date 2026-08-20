"""Turn a fixture's declared states into the `Belief` production now requires.

Every test in this suite said what state a task was in by giving its `entry()`
helper a label, and the orchestrator read it back off `LedgerEntry.state_label`.
#152 removed that field: a state is derived per cycle and lives on the cycle's
`Belief`, so `authority.state_of` raises without one rather than falling back to
a label that no longer exists.

The declaration is still the right thing for a fixture to make - it is what the
test is *about* - so it is stashed on `LedgerEntry.labels` and turned into a
belief here. One place, so a test that does not care about the authority reads
exactly as it did before.
"""

from __future__ import annotations

from typing import Any

from swarm.orchestrator.authority import Belief


def fixture_belief(book: Any, **kwargs: Any) -> Belief:
    """The belief a ledger's declared states imply."""
    entries = getattr(book, "entries", book)
    states = {
        task_id: next(iter(getattr(item, "labels", ()) or ()), "")
        for task_id, item in entries.items()
    }
    kwargs.setdefault("states", states)
    kwargs.setdefault("stored", dict(states))
    kwargs.setdefault("previous", dict(states))
    kwargs.setdefault("refs", {task_id: item.ref for task_id, item in entries.items()})
    return Belief(**kwargs)
