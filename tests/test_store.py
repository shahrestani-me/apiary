"""apiary's own task store: what it holds, what it refuses, and how it fails.

Three groups, and they fail for different reasons.

**The discipline.** `docs/adr/0002-apiary-owns-a-thin-task-store.md` is one
sentence long where it matters: the store holds only apiary's own judgments and
never mirrors the tracker. That is a property nothing enforces at runtime -
`TaskJudgement` is a dataclass and dataclasses accept whatever fields somebody
adds - so it is asserted here as a field roster. A test that fails when a
`goal` or a `files` appears is the only mechanical guard between this package
and a second issue tracker.

**The failure modes.** A store that cannot be trusted must stop the run. Every
plausible recovery from a corrupt store - start empty, drop the bad rows,
rebuild from the tracker - answers "what is this task's retry budget?" with "a
fresh one", for every task at once, and does it silently. The tests below pin
each refusal *and* the fact that the bad file is still there afterwards, because
a store that raises and then quietly recreates itself has the same effect as one
that never raised.

**The behaviour that must not have moved.** #154-#156's retry arithmetic is the
thing this ticket was forbidden to change. Its own tests
(`tests/test_reconcile.py`) still assert it directly on `plan_reconcile` and did
not need rewriting, which is the strongest evidence available; what is added
here is the round trip they cannot see - a failure recorded through the store,
read back through `load_ledger`, and judged the same way it was before the
record lived anywhere else.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import pytest

from fixtures.markers import legacy_marker
from swarm.github.branches import task_branch
from swarm.github.ledger import (
    Ledger,
    attempt_floor,
    load_ledger,
    render_marker,
    seed_attempt_floor,
)
from swarm.github.refs import task_ref as ref
from swarm.orchestrator.dispatcher import CLAIMED
from swarm.orchestrator.reconcile import (
    FAILED,
    READY,
    apply_plan,
    plan_reconcile,
    signature,
)
from swarm.store import (
    DEFAULT_STORE_DIR,
    SCHEMA_VERSION,
    STORE_DIR_ENV,
    SqliteTaskStore,
    StoreCorrupt,
    StoreError,
    StoreMissing,
    TaskJudgement,
    TaskStore,
    record_judgement,
    store_path,
    store_root,
)
from swarm.taskref import TaskRef
from swarm.worker.result import ResultRecord

REPO = "shahrestani-me/apiary"
OTHER = "shahrestani-me/hive"

#: A failure with a stable signature, and a different one. Two traceback tails
#: rather than two arbitrary strings, because `signature` is only as
#: deterministic as its input is realistic.
IMPORT_FAILURE = "ModuleNotFoundError: No module named 'sqlalchemy'\n"
ASSERT_FAILURE = "E   AssertionError: assert 3 == 4\n"


@pytest.fixture(autouse=True)
def store_root_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every store this module opens lands under `tmp_path`.

    Autouse for `tests/test_reconcile.py`'s reason: a test that forgot to
    redirect would read and write the operator's real store, and nothing would
    fail until a later real run believed something untrue about its history.
    """
    root = tmp_path / "store"
    monkeypatch.setenv(STORE_DIR_ENV, str(root))
    return root


def body(task_id: str, *, attempt: int = 0, marker: str | None = None) -> str:
    return "\n".join(
        [
            marker if marker is not None else render_marker(task_id, attempt),
            "",
            "## Goal",
            "Do the thing.",
            "",
            "## Files",
            f"- src/{task_id}.py",
            "",
            "## Verify",
            "python -m pytest -q",
            "",
            "## Blocked by",
            "_none._",
        ]
    )


class FakeClient:
    """The three calls `load_ledger` and `apply_plan` make. No HTTP anywhere."""

    def __init__(self, issues: Mapping[int, dict[str, Any]]) -> None:
        self.issues = dict(issues)
        self.log: list[str] = []

    def list_issues(self, *, state: str = "open", **_kwargs: Any) -> list[dict[str, Any]]:
        return [dict(payload) for payload in self.issues.values()]

    def get_issue(self, number: int) -> dict[str, Any]:
        return dict(self.issues[number])

    def update_issue(self, number: int, **fields: Any) -> dict[str, Any]:
        self.log.append(f"update_issue #{number}")
        self.issues[number].update(fields)
        return dict(self.issues[number])

    def add_labels(self, number: int, labels: list[str]) -> None:
        self.log.append(f"+{labels[0]} #{number}")
        self.issues[number]["labels"] = [
            *self.issues[number].get("labels", ()),
            *({"name": name} for name in labels),
        ]

    def remove_label(self, number: int, label: str) -> None:
        self.log.append(f"-{label} #{number}")
        self.issues[number]["labels"] = [
            item for item in self.issues[number].get("labels", ()) if item["name"] != label
        ]


def issue(number: int, *, label: str, task_id: str, marker: str | None = None) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"issue {number}",
        "state": "open",
        "state_reason": None,
        "labels": [{"name": label}],
        "body": body(task_id, marker=marker),
    }


def record(number: int, *, attempt: int, verify_output: str) -> ResultRecord:
    return ResultRecord(
        run_id="apiary-20260819-090000-k3f9qz",
        issue=number,
        attempt=attempt,
        exit_code=1,
        reason="the verify command failed",
        verify_output=verify_output,
    )


# --------------------------------------------------------------------------
# The discipline: only apiary's own judgments
# --------------------------------------------------------------------------


#: Everything the tracker owns, in the ADR's own words. A field named here
#: appearing on `TaskJudgement` is not a bug in this test.
TRACKER_OWNED = frozenset(
    {
        "goal",
        "files",
        "verify",
        "verify_command",
        "title",
        "description",
        "body",
        "dependencies",
        "blocked_by",
        "depends_on",
        "labels",
        "state_label",
        "status",
        "number",
        "assignee",
        "milestone",
    }
)


def test_the_record_holds_exactly_apiarys_own_judgments_and_nothing_else():
    """ADR 0002's table, as an assertion.

    The store is worth having only because there is nothing to reconcile
    between it and the tracker, and that is true only while it holds fields the
    tracker never had. The moment one of `TRACKER_OWNED` appears here there are
    two records of one fact, and the sync problem is somebody's next ticket.
    """
    held = {field.name for field in dataclasses.fields(TaskJudgement)}

    assert held == {"ref", "attempt", "blocker", "streak", "renewals", "updated_at"}
    assert not held & TRACKER_OWNED


def test_the_seam_is_three_methods_so_a_remote_backend_can_satisfy_it():
    """A seam that leaked a path, a cursor or a query would not be swappable,
    which is the only thing ADR 0002 asks of it before the organization
    database exists."""
    surface = {name for name in dir(TaskStore) if not name.startswith("_")}

    assert surface == {"read", "write", "close"}


def test_an_in_memory_backend_satisfies_the_seam_and_no_caller_notices():
    """The swap ADR 0002 defers, done in miniature. `load_ledger` is handed a
    store that shares nothing with SQLite but the three methods, and produces
    the same ledger - which is the acceptance criterion "swapping it touches no
    caller", checked rather than asserted in prose."""

    class Memory:
        def __init__(self) -> None:
            self.held: dict[TaskRef, TaskJudgement] = {}

        def read(self) -> Mapping[TaskRef, TaskJudgement]:
            return dict(self.held)

        def write(self, judgement: TaskJudgement) -> None:
            self.held[judgement.ref] = judgement

        def close(self) -> None:
            return None

    memory = Memory()
    assert isinstance(memory, TaskStore)
    memory.write(TaskJudgement(ref=ref(4), attempt=2, blocker="ab12cd34ef", streak=2))
    client = FakeClient({4: issue(4, label=READY, task_id="task-4", marker=render_marker("task-4", 2))})

    ledger = load_ledger(client, adopt=False, store=memory)

    entry = ledger.entries["task-4"]
    assert (entry.attempt, entry.blocker, entry.streak) == (2, "ab12cd34ef", 2)


# --------------------------------------------------------------------------
# The local backend
# --------------------------------------------------------------------------


def test_a_judgment_round_trips(store_root_dir: Path):
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(7), attempt=3, blocker="ab12cd34ef", streak=2, renewals=1))

    with SqliteTaskStore.open(REPO) as store:
        held = store.read()[ref(7)]

    assert (held.ref, held.attempt, held.blocker, held.streak, held.renewals) == (
        ref(7),
        3,
        "ab12cd34ef",
        2,
        1,
    )


def test_writing_the_same_task_twice_replaces_rather_than_accumulates():
    """One judgment per task, always the latest. The history of *attempts* is
    the artifacts directory's job (`worker/result.py` keeps one file per
    attempt); this answers only "what does apiary currently believe"."""
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(7), attempt=1, blocker="aaaaaaaaaa", streak=1))
        store.write(TaskJudgement(ref=ref(7), attempt=2, blocker="bbbbbbbbbb", streak=1))

        held = store.read()

    assert len(held) == 1
    assert (held[ref(7)].attempt, held[ref(7)].blocker) == (2, "bbbbbbbbbb")


def test_an_absent_streak_survives_the_round_trip_as_absent():
    """`None` and `0` mean different things downstream - "the row does not say"
    falls back to the attempt counter, and `0` does not - so a column that
    turned one into the other would change the arithmetic silently."""
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(7), attempt=1, blocker="ab12cd34ef"))

        assert store.read()[ref(7)].streak is None


def test_the_store_is_per_project_not_per_run(store_root_dir: Path):
    """The reason it is not under the run artifacts directory. A per-run store
    is empty at the start of every run, and an empty store reads as a fresh
    retry budget for every task - a bound that bounds nothing."""
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(7), attempt=2, blocker="ab12cd34ef", streak=2))
    with SqliteTaskStore.open(OTHER) as other:
        assert other.read() == {}

    assert store_path(REPO) != store_path(OTHER)
    assert store_path(REPO).parent == store_root_dir


def test_the_store_root_is_its_own_setting_and_not_derived_from_the_artifacts_root(
    monkeypatch: pytest.MonkeyPatch,
):
    """`artifacts.CONSOLE_ROOT_ENV`'s rule: an operator who moved runs has said
    nothing about where anything else goes."""
    monkeypatch.delenv(STORE_DIR_ENV, raising=False)
    monkeypatch.setenv("APIARY_ARTIFACTS_DIR", "/var/apiary/runs")

    assert store_root() == Path(DEFAULT_STORE_DIR)


def test_reading_a_closed_store_says_so_rather_than_answering_empty():
    store = SqliteTaskStore.open(REPO)
    store.close()
    store.close()  # idempotent: every exit path calls it

    with pytest.raises(StoreError, match="closed"):
        store.read()


# --------------------------------------------------------------------------
# Missing, and corrupt
# --------------------------------------------------------------------------


def test_a_missing_store_a_caller_required_fails_with_a_named_fix():
    with pytest.raises(StoreMissing) as raised:
        SqliteTaskStore.open(REPO, create=False)

    message = str(raised.value)
    assert STORE_DIR_ENV in message
    assert str(store_path(REPO)) in message


def test_a_first_run_creates_its_store_rather_than_refusing_to_start():
    """The one case that must not be an error: a project apiary has never run
    legitimately has no store, and the seeding path below is what stops that
    from resetting anything."""
    with SqliteTaskStore.open(REPO) as store:
        assert store.read() == {}
        assert store.path.exists()


def test_a_corrupt_store_refuses_to_open_and_names_the_fix():
    target = store_path(REPO)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"this is not a database, it is a text file\n" * 40)

    with pytest.raises(StoreCorrupt) as raised:
        SqliteTaskStore.open(REPO)

    message = str(raised.value)
    assert str(target) in message
    assert "Move it aside" in message


def test_a_corrupt_store_is_not_quietly_replaced_with_an_empty_one():
    """The assertion the acceptance criterion is really about. A refusal that
    then recreates the file has exactly the effect of never refusing: the next
    run reads an empty store, every task looks like attempt 0 with no blocker,
    and a task that has already burned its budget retries forever."""
    target = store_path(REPO)
    target.parent.mkdir(parents=True, exist_ok=True)
    original = b"corrupt, but somebody's history\n" * 40
    target.write_bytes(original)

    with pytest.raises(StoreCorrupt):
        SqliteTaskStore.open(REPO)

    assert target.read_bytes() == original


def test_a_store_written_by_a_newer_build_is_refused_rather_than_misread():
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(7), attempt=2, blocker="ab12cd34ef", streak=2))
    connection = sqlite3.connect(str(store_path(REPO)))
    connection.execute("UPDATE meta SET value = ? WHERE key = 'schema'", (str(SCHEMA_VERSION + 1),))
    connection.commit()
    connection.close()

    with pytest.raises(StoreCorrupt, match="newer apiary"):
        SqliteTaskStore.open(REPO)


def test_another_projects_store_is_refused_rather_than_spent():
    """Task refs are not comparable across projects, so `#7` in one project's
    store is not `#7` in another's - and reading it as one would spend a budget
    that belongs to work nobody in this run has heard of."""
    with SqliteTaskStore.open(OTHER) as store:
        store.write(TaskJudgement(ref=ref(7), attempt=2, blocker="ab12cd34ef", streak=2))
    store_path(OTHER).rename(store_path(REPO))

    with pytest.raises(StoreCorrupt, match="holds judgments for"):
        SqliteTaskStore.open(REPO)


def test_a_foreign_sqlite_database_is_refused_rather_than_read_as_empty():
    """It would open cleanly, hold no `judgement` rows, and the tempting
    reading of that is "an empty store" - which is a fresh budget for
    everything."""
    target = store_path(REPO)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target))
    connection.execute("CREATE TABLE somebody_elses (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    with pytest.raises(StoreCorrupt, match="not an apiary task store"):
        SqliteTaskStore.open(REPO)


def test_a_row_with_no_task_ref_is_refused_rather_than_keyed_on_a_blank():
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(7), attempt=1))
    connection = sqlite3.connect(str(store_path(REPO)))
    connection.execute("UPDATE judgement SET ref = '' WHERE ref = '#7'")
    connection.commit()
    connection.close()

    with SqliteTaskStore.open(REPO) as store:
        with pytest.raises(StoreCorrupt, match="no task ref"):
            store.read()


def test_a_project_with_no_filesystem_safe_name_is_refused():
    with pytest.raises(StoreError, match="filesystem-safe"):
        store_path("///")


# --------------------------------------------------------------------------
# The join with the tracker
# --------------------------------------------------------------------------


def test_a_task_the_store_has_never_judged_keeps_the_legacy_marker_record():
    """The upgrade path. Every issue in a repository that ran an older build
    carries `blocker=`/`streak=` in its body, and a build that stopped reading
    them would hand all of those tasks a fresh budget on the first cycle."""
    client = FakeClient(
        {
            4: issue(
                4,
                label=READY,
                task_id="task-4",
                marker=legacy_marker("task-4", 2, blocker="ab12cd34ef", streak=2),
            )
        }
    )

    with SqliteTaskStore.open(REPO) as store:
        ledger = load_ledger(client, adopt=False, store=store)

    entry = ledger.entries["task-4"]
    assert (entry.attempt, entry.blocker, entry.streak) == (2, "ab12cd34ef", 2)


def test_a_judgment_about_this_attempt_wins_over_the_body():
    """Once the store has ruled, it is authoritative for the signature - the
    body does not carry one any more, and a residue from before the upgrade
    must not outrank what apiary has since decided."""
    client = FakeClient(
        {
            4: issue(
                4,
                label=READY,
                task_id="task-4",
                marker=legacy_marker("task-4", 2, blocker="ab12cd34ef", streak=2),
            )
        }
    )

    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(4), attempt=2, blocker="ffffffffff", streak=1, renewals=1))
        ledger = load_ledger(client, adopt=False, store=store)

    entry = ledger.entries["task-4"]
    assert (entry.blocker, entry.streak, entry.renewals) == ("ffffffffff", 1, 1)


def test_the_stored_counter_wins_over_a_marker_that_still_carries_one():
    """The inversion ADR 0005 made, and the case that used to prove the old rule.

    This test read the other way until the counter moved: the tracker held it,
    the store held a claim *about* it, and a marker saying `attempt=0` under a
    judgment stamped at 3 meant somebody had reset the counter - so the counter
    stood and the signature was dropped as stale.

    The store owns the counter now, so there is no second opinion to lose to. A
    marker left saying `attempt=0` is a fossil of a body apiary no longer
    writes, and reading it would hand the task a fresh budget on the strength of
    a field nothing maintains. The row wins whole - counter, signature, streak
    and renewals together, because they were written in one act."""
    client = FakeClient(
        {4: issue(4, label=READY, task_id="task-4", marker=render_marker("task-4", 0))}
    )

    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(4), attempt=3, blocker="ab12cd34ef", streak=3, renewals=2))
        ledger = load_ledger(client, adopt=False, store=store)

    entry = ledger.entries["task-4"]
    assert (entry.attempt, entry.blocker, entry.streak, entry.renewals) == (
        3,
        "ab12cd34ef",
        3,
        2,
    )


def test_a_loader_with_no_store_reads_exactly_what_it_read_before():
    """`load_tasks`, the console and every read-only caller pass no store, and
    must keep seeing the tracker's own facts rather than an exception."""
    client = FakeClient(
        {
            4: issue(
                4,
                label=READY,
                task_id="task-4",
                marker=legacy_marker("task-4", 1, blocker="ab12cd34ef", streak=1),
            )
        }
    )

    ledger = load_ledger(client, adopt=False)

    assert ledger.entries["task-4"].blocker == "ab12cd34ef"


def test_record_judgement_without_a_store_writes_nothing_rather_than_raising():
    record_judgement(None, ref(4), 1, blocker="ab12cd34ef", streak=1)


# --------------------------------------------------------------------------
# #154-#156's arithmetic, over the store
# --------------------------------------------------------------------------


def cycle(client: FakeClient, store: SqliteTaskStore, verify_output: str) -> Ledger:
    """One observe-and-write pass: read the ledger, judge a failed worker, write.

    Deliberately assembled from the real functions rather than from
    `Reconciler`, so what is exercised is the counter-and-judgment path and not
    a container, a pull request or a model.
    """
    ledger = load_ledger(client, adopt=False, store=store)
    entry = next(iter(ledger.entries.values()))
    plan = plan_reconcile(
        ledger,
        results={entry.ref: record(entry.number, attempt=entry.attempt, verify_output=verify_output)},
        max_attempts=3,
    )
    apply_plan(client, plan, store=store)
    # The label the transition wrote is what the next `load_ledger` reads, and
    # `FakeClient` already applied it; re-reading is the point.
    return ledger


def test_the_same_failure_three_times_still_gives_up_at_the_cap():
    """#154-#156's give-up, end to end over the store. The arithmetic itself is
    asserted directly in `tests/test_reconcile.py` and did not move; this is
    the round trip through persistence those tests cannot see."""
    client = FakeClient({4: issue(4, label=CLAIMED, task_id="task-4")})

    with SqliteTaskStore.open(REPO) as store:
        for _ in range(3):
            client.issues[4]["labels"] = [{"name": CLAIMED}]
            cycle(client, store, IMPORT_FAILURE)

        held = store.read()[ref(4)]

    assert {label["name"] for label in client.issues[4]["labels"]} == {FAILED}
    assert (held.attempt, held.streak, held.renewals) == (3, 3, 0)
    assert held.blocker == signature(IMPORT_FAILURE)


def test_a_failure_that_changes_renews_the_budget_and_the_store_counts_it():
    """The live defect #155 fixed: a task capped on one blocker, fixed by a
    human, then failing on a *new* one and given up anyway. The renewal is what
    stops that, and the store is now where the evidence for it lives."""
    client = FakeClient({4: issue(4, label=CLAIMED, task_id="task-4")})

    with SqliteTaskStore.open(REPO) as store:
        for _ in range(2):
            client.issues[4]["labels"] = [{"name": CLAIMED}]
            cycle(client, store, IMPORT_FAILURE)
        client.issues[4]["labels"] = [{"name": CLAIMED}]
        cycle(client, store, ASSERT_FAILURE)

        held = store.read()[ref(4)]

    # Still ready: the third failure was a different one, so the per-blocker
    # streak restarted rather than the task being given up.
    assert {label["name"] for label in client.issues[4]["labels"]} == {READY}
    assert (held.attempt, held.streak, held.renewals) == (3, 1, 1)
    assert held.blocker == signature(ASSERT_FAILURE)


def test_nothing_apiary_judges_reaches_the_issue_body_any_more():
    """The acceptance criterion, as a grep - and it grew a third field.

    #159 put `blocker` and `streak` in the store and left the counter in the
    marker, so this asserted the first two were absent and the third still
    present. ADR 0005 moved the counter too, so `attempt=` joins them: after a
    consumed attempt the body is byte-identical to what it was, and the `0` its
    marker still carries is a fossil the loader ignores in favour of the store.

    That is the whole of `docs/issue-contract.md` §5 gone - the body `PATCH` had
    no other payload left."""
    client = FakeClient({4: issue(4, label=CLAIMED, task_id="task-4")})

    with SqliteTaskStore.open(REPO) as store:
        cycle(client, store, IMPORT_FAILURE)
        held = store.read()[ref(4)]

    text = client.issues[4]["body"]
    assert "blocker=" not in text
    assert "streak=" not in text
    # The counter, too: the consumed attempt is in the store and the body was
    # never touched, so the marker still reads whatever it read at creation.
    assert "attempt=1" not in text
    # And it really was consumed - an absent write is only the criterion if the
    # number landed somewhere.
    assert held.attempt == 1


# --------------------------------------------------------------------------
# The floor under the counter (ADR 0005)
# --------------------------------------------------------------------------


def test_the_floor_takes_the_furthest_attempt_each_task_pushed():
    """One branch per attempt since #144, so the furthest is the lower bound."""
    assert attempt_floor(
        [
            task_branch(ref(4), 0),
            task_branch(ref(4), 2),
            task_branch(ref(4), 1),
            task_branch(ref(7), 3),
        ]
    ) == {ref(4): 2, ref(7): 3}


def test_the_floor_ignores_branches_apiary_did_not_mint():
    """`build_observation`'s discipline: a branch this system did not create
    says nothing about a task, so it is dropped rather than counted."""
    assert attempt_floor(["main", "fix/typo", "swarm/issue-4", "release-1.2"]) == {}


def test_seeding_gives_an_unjudged_task_the_floor_its_branches_imply():
    with SqliteTaskStore.open(REPO) as store:
        seeded = seed_attempt_floor(store, [task_branch(ref(4), 2)])
        held = store.read()[ref(4)]

    assert seeded == (ref(4),)
    assert held.attempt == 2


def test_a_seeded_row_leaves_the_streak_absent_rather_than_zero():
    """The line ADR 0002 warns by name not to simplify.

    `previous_streak = entry.attempt if entry.streak is None else entry.streak`
    falls back to the counter, which is the largest streak consistent with it -
    so absence gives up sooner and never later. Writing `0` here would look
    tidier and would silently hand every seeded task its budget back.
    """
    with SqliteTaskStore.open(REPO) as store:
        seed_attempt_floor(store, [task_branch(ref(4), 2)])
        held = store.read()[ref(4)]

    assert held.streak is None
    assert held.blocker == ""
    assert held.renewals == 0


def test_seeding_never_overwrites_a_task_the_store_has_already_judged():
    """A floor is under a counter apiary wrote, not an adjudicator of it.

    The branch listing is a *lower* bound reconstructed from the code host; the
    row is what this project actually decided. A seed that overwrote it would
    lose a signature and a renewal count to a number that is only ever <= the
    truth.
    """
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(4), attempt=3, blocker="ab12cd34ef", streak=3, renewals=1))
        seeded = seed_attempt_floor(store, [task_branch(ref(4), 1)])
        held = store.read()[ref(4)]

    assert seeded == ()
    assert (held.attempt, held.blocker, held.streak, held.renewals) == (3, "ab12cd34ef", 3, 1)


def test_a_task_whose_only_branch_is_attempt_zero_is_not_seeded():
    """Attempt 0 is the floor every task already has, so a row saying so is
    noise - and a row is also the thing that stops the marker's legacy fields
    being read on an upgrade."""
    with SqliteTaskStore.open(REPO) as store:
        seeded = seed_attempt_floor(store, [task_branch(ref(4), 0)])
        held = store.read()

    assert seeded == ()
    assert held == {}


def test_seeding_without_a_store_writes_nothing_rather_than_raising():
    assert seed_attempt_floor(None, [task_branch(ref(4), 2)]) == ()


def test_a_seeded_floor_is_what_the_ledger_then_reads_as_the_counter():
    """End to end: a wiped store plus a branch listing rebuilds the budget.

    This is the whole point of the floor. Without it the same ledger read would
    report `attempt=0` for a task that has already burned three attempts, and
    `max_attempts` would bound nothing.
    """
    client = FakeClient(
        {4: issue(4, label=READY, task_id="task-4", marker=render_marker("task-4", 0))}
    )

    with SqliteTaskStore.open(REPO) as store:
        seed_attempt_floor(store, [task_branch(ref(4), 3)])
        ledger = load_ledger(client, adopt=False, store=store)

    assert ledger.entries["task-4"].attempt == 3


# --------------------------------------------------------------------------
# `swarm reset` - the gesture the issue marker used to be (ADR 0005 decision 4)
# --------------------------------------------------------------------------


def reset(*argv: str) -> int:
    import swarm.cli as cli

    return cli.main(["reset", *argv])


def test_reset_gives_a_capped_task_its_budget_back():
    with SqliteTaskStore.open(REPO) as store:
        store.write(
            TaskJudgement(ref=ref(4), attempt=3, blocker="ab12cd34ef", streak=3, renewals=2)
        )

    assert reset(str(ref(4)), "--repo", REPO, "--yes") == 0

    with SqliteTaskStore.open(REPO) as store:
        held = store.read()[ref(4)]
    assert (held.attempt, held.blocker, held.streak) == (0, "", None)


def test_reset_keeps_the_renewal_count():
    """A history of the task, not a claim about one attempt - and nothing
    branches on it, so keeping it costs nothing and losing it costs the one
    number a human reading a capped task actually wants."""
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(4), attempt=3, blocker="ab", streak=3, renewals=4))

    reset(str(ref(4)), "--repo", REPO, "--yes")

    with SqliteTaskStore.open(REPO) as store:
        assert store.read()[ref(4)].renewals == 4


def test_a_reset_survives_the_next_runs_floor_seeding():
    """The reason a reset writes a row rather than deleting one.

    `seed_attempt_floor` seeds only tasks the store has never judged, so a
    deleted row would be refilled from the branch listing at the next startup -
    putting the task straight back under the cap the human just lifted. The row
    is what makes the reset stick.
    """
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(4), attempt=3, blocker="ab", streak=3))

    reset(str(ref(4)), "--repo", REPO, "--yes")

    with SqliteTaskStore.open(REPO) as store:
        seeded = seed_attempt_floor(store, [task_branch(ref(4), 3)])
        held = store.read()[ref(4)]

    assert seeded == ()
    assert held.attempt == 0


def test_reset_writes_nothing_when_the_task_has_no_judgment():
    """Already a full budget, so there is nothing to give back - and a row
    written here would shadow the marker for no reason."""
    assert reset(str(ref(4)), "--repo", REPO, "--yes") == 0
    with SqliteTaskStore.open(REPO) as store:
        assert store.read() == {}


def test_reset_declined_at_the_prompt_writes_nothing(monkeypatch):
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(4), attempt=3, blocker="ab", streak=3))

    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert reset(str(ref(4)), "--repo", REPO) == 1

    with SqliteTaskStore.open(REPO) as store:
        assert store.read()[ref(4)].attempt == 3


def test_reset_with_no_answer_available_declines(monkeypatch):
    """A pipe or a CI job. "Nobody is there" reads as no; `--yes` is how an
    unattended caller says yes deliberately."""
    def no_terminal(_prompt):
        raise EOFError

    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(4), attempt=3, blocker="ab", streak=3))

    monkeypatch.setattr("builtins.input", no_terminal)
    assert reset(str(ref(4)), "--repo", REPO) == 1

    with SqliteTaskStore.open(REPO) as store:
        assert store.read()[ref(4)].attempt == 3


def test_reset_can_write_a_counter_other_than_zero():
    """A human who knows two of the three attempts were the broken environment
    and one was a real failure."""
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(4), attempt=3, blocker="ab", streak=3))

    reset(str(ref(4)), "--repo", REPO, "--attempt", "1", "--yes")

    with SqliteTaskStore.open(REPO) as store:
        assert store.read()[ref(4)].attempt == 1


def test_reset_takes_the_ref_verbatim_so_a_non_github_tracker_works():
    """No format is invented here (`_reset`). The store's keys are refs in the
    tracker's own spelling, and Linear's are `ENG-123` rather than numbers - a
    command that parsed an issue number would be the GitHub adapter leaking into
    a place epic #140 is removing it from."""
    linear = TaskRef("ENG-123")
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=linear, attempt=3, blocker="ab", streak=3))

    assert reset("ENG-123", "--repo", REPO, "--yes") == 0

    with SqliteTaskStore.open(REPO) as store:
        assert store.read()[linear].attempt == 0


def test_reset_names_what_the_store_holds_when_the_ref_does_not_match():
    """"Nothing to reset" and "you typed the wrong spelling" look identical
    otherwise, and the second is the likely one against a tracker whose refs are
    not numbers."""
    with SqliteTaskStore.open(REPO) as store:
        store.write(TaskJudgement(ref=ref(4), attempt=3, blocker="ab", streak=3))

    assert reset("ENG-999", "--repo", REPO, "--yes") == 0

    with SqliteTaskStore.open(REPO) as store:
        assert store.read()[ref(4)].attempt == 3


@pytest.mark.parametrize("bad", ["", "  "])
def test_reset_refuses_an_empty_ref(bad):
    with pytest.raises(SystemExit) as exc:
        reset(bad, "--repo", REPO, "--yes")
    assert exc.value.code == 2


def test_reset_refuses_a_negative_counter():
    with pytest.raises(SystemExit) as exc:
        reset(str(ref(4)), "--repo", REPO, "--attempt", "-1", "--yes")
    assert exc.value.code == 2
