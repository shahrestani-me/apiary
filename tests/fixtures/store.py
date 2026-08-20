"""A store that says when it was written, so crash-ordering stays assertable.

`docs/adr/0002-apiary-owns-a-thin-task-store.md` fixes an order - the judgment
is persisted *before* the label that re-readies the task, so a crash between
them costs an attempt rather than granting a free one. Until ADR 0005 both
writes went to the same client and the order was visible in one log; the
judgment now goes to a different object entirely, and an assertion that only
checked the label would no longer be checking the property at all.

So the spy appends to the *client's* log rather than keeping its own. One
sequence, one `index()` comparison, and the test reads the way it did before.
"""

from __future__ import annotations

from typing import Any

from swarm.store import SqliteTaskStore, TaskJudgement


class RecordingStore:
    """A real store that also narrates its writes into `log`."""

    def __init__(self, repo: str, log: list[str]) -> None:
        self._store = SqliteTaskStore.open(repo)
        self._log = log

    def read(self) -> Any:
        return self._store.read()

    def write(self, judgement: TaskJudgement) -> None:
        self._log.append(f"store {judgement.ref} attempt={judgement.attempt}")
        self._store.write(judgement)

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "RecordingStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
