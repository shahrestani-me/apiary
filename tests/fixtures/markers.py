"""Identity markers in the shape older builds wrote them.

`swarm.github.ledger.render_marker` emits `id=` and `attempt=` and nothing
else: #159 moved the failure signature into apiary's own task store, so no code
path writes `blocker=` or `streak=` into an issue body any more
(`docs/adr/0002-apiary-owns-a-thin-task-store.md`).

The *parser* still reads them, and must - every issue in a repository that ran
an older build carries them, and a build that stopped reading them would answer
"no previous blocker" for all of those tasks at once, which hands each of them
a fresh retry budget on the first cycle after the upgrade. That back-compat is
the thing these tests pin, and pinning it needs a way to write the old form.

So it lives here rather than in `render_marker`: a renderer that can still emit
a field nothing writes is a renderer somebody eventually writes with. A helper
in `tests/fixtures/` cannot be reached from `src/`, which is exactly the
guarantee wanted.
"""

from __future__ import annotations


def legacy_marker(
    task_id: str,
    attempt: int = 0,
    *,
    blocker: str = "",
    streak: int | None = None,
) -> str:
    """The marker as builds before #159 wrote it, signature fields and all.

    With neither optional field this is byte-for-byte what `render_marker`
    produces today, which is what makes it usable for "an old body round-trips
    unchanged" as well as for "an old body with a signature still parses".
    """
    fields = [f"id={task_id}", f"attempt={attempt}"]
    if blocker:
        fields.append(f"blocker={blocker}")
    if streak is not None:
        fields.append(f"streak={streak}")
    return f"<!-- apiary:task {' '.join(fields)} -->"
