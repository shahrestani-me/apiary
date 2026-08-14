"""Unit tests for the issue-contract parser and the ledger loader.

Every case in `docs/issue-contract.md` §7 appears here, plus the two things
that table takes for granted: that the `TaskRecord` the loader emits is the one
v1 built in memory, and that `depends_on` is a graph of task ids rather than of
issue numbers.

No network and no token. `FakeClient` replays issue payloads shaped like
GitHub's and records the writes, which is the whole surface `load_ledger`
touches; #31 will lift it into the shared fixture set.

The fixtures marked "real" are copied verbatim from this repository's own
backlog. That backlog is the corpus the contract doc names: it has hand-written
issues with no markers, prose outside every section, and #11's code-span trap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence
from unittest import mock

import pytest

from swarm.github import ledger as ledger_module
from swarm.github.ledger import (
    DEFAULT_STACK,
    KNOWN_STACKS,
    LABEL_PRECEDENCE,
    REQUIRED_SECTIONS,
    STATUS_BY_LABEL,
    ContractError,
    DuplicateTaskIdError,
    load_ledger,
    load_tasks,
    parse_contract,
    render_marker,
    resolve_state_label,
    slugify,
)


# --------------------------------------------------------------------------
# Fixtures and helpers (#31 will lift these into the shared set)
# --------------------------------------------------------------------------


@dataclass
class FakeClient:
    """The two calls the loader makes, and a record of every write."""

    issues: list[dict[str, Any]]
    patched: list[tuple[int, str]] = field(default_factory=list)

    def list_issues(self, *, state: str = "open", **_: Any) -> list[dict[str, Any]]:
        self.last_state = state
        return [dict(issue) for issue in self.issues]

    def update_issue(self, number: int, *, body: str | None = None, **_: Any) -> dict[str, Any]:
        assert body is not None, "the loader only ever patches bodies"
        self.patched.append((number, body))
        return {"number": number, "body": body}


def issue(
    number: int,
    body: str,
    *,
    title: str = "A task",
    labels: Sequence[str] = ("swarm:ready",),
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "body": body,
        "labels": [{"name": name} for name in labels],
    }


def contract_body(
    *,
    marker: str | None = "<!-- apiary:task id=add-retry-logic attempt=0 -->",
    goal: str = "`http_client.get` retries idempotent requests three times.",
    files: str = "- src/swarm/github/client.py\n- tests/test_client_retry.py",
    verify: str = "python -m pytest -q tests/test_client_retry.py",
    blocked_by: str = "- #7",
    stack: str | None = None,
) -> str:
    """The canonical body of §6, with one section swapped out per test.

    `stack=None` omits `## Stack` entirely, which is what a Python task's body
    looks like and what every body written before #98 looks like. Passing the
    empty string writes the heading with nothing under it, which is its own
    error case.
    """
    head = f"{marker}\n\n" if marker else ""
    tail = "" if stack is None else f"\n## Stack\n{stack}\n"
    return (
        f"{head}## Goal\n{goal}\n\n"
        f"## Files\n{files}\n\n"
        f"## Verify\n{verify}\n\n"
        f"## Blocked by\n{blocked_by}\n{tail}"
    )


# The real body of https://github.com/shahrestani-me/apiary/issues/11, verbatim.
# Its `## Goal` sentence carries the literal string `## Blocked by` inside a
# code span - the trap that motivated the line-anchored heading rule.
REAL_ISSUE_11 = """\
> Design: [`docs/architecture-v2.md`](https://github.com/shahrestani-me/apiary/blob/main/docs/architecture-v2.md). Read the "Three constraints" section before starting.

## Goal
An issue is labelled `swarm:ready` exactly when every `## Blocked by` reference is closed, and a dependency cycle fails the run loudly instead of hanging it.

## Files
- src/swarm/github/readiness.py
- tests/test_readiness.py

## Verify
python -m pytest -q tests/test_readiness.py

## Blocked by
- #9

---

This replaces v1's in-memory `depends_on` set arithmetic ([`graph.py`](https://github.com/shahrestani-me/apiary/blob/main/src/swarm/graph.py) `fan_out`).

**Scope**
- Parse `Blocked by` refs, resolve their state, relabel `swarm:blocked` <-> `swarm:ready`.
- **Detect dependency cycles and fail loudly.**
- Treat a reference to a non-existent or cross-repo issue as an error, not as satisfied.
"""

# The real body of issue #6, whose `## Blocked by` is prose rather than a list.
REAL_ISSUE_6 = """\
> Design: [`docs/architecture-v2.md`](https://github.com/shahrestani-me/apiary/blob/main/docs/architecture-v2.md).

## Goal
`docs/issue-contract.md` pins down the body schema, the identity rule, the status mapping and the label state machine.

## Files
- docs/issue-contract.md

## Verify
test -f docs/issue-contract.md

## Blocked by
_none — this is the root of the dependency graph._

---

**Why it is first:** every other module reads or writes this.
"""

# A meta-issue quoting the schema inside a fence: every heading below sits at
# column 0 and must still open no section.
QUOTES_THE_SCHEMA = """\
<!-- apiary:task id=document-the-schema attempt=0 -->

## Goal
Document the body schema, which looks like this:

```markdown
<!-- apiary:task id=the-example attempt=3 -->

## Goal
One sentence.

## Files
- src/example.py

## Verify
false

## Blocked by
- #999
```

## Files
- docs/issue-contract.md

## Verify
test -f docs/issue-contract.md

## Blocked by
- #7
"""


# --------------------------------------------------------------------------
# Section finding: the two ways the obvious parser fails
# --------------------------------------------------------------------------


def test_code_span_in_goal_does_not_open_a_section():
    """#11's real body: one dependency parsed, not zero.

    A substring search for "## Blocked by" cuts this body inside the goal
    sentence and reports the issue as unblocked - ready when it was not.
    """
    contract = parse_contract(11, REAL_ISSUE_11)

    assert contract.blocked_by == (9,)
    assert contract.goal.startswith("An issue is labelled `swarm:ready`")
    assert contract.files == ("src/swarm/github/readiness.py", "tests/test_readiness.py")
    assert contract.verify == "python -m pytest -q tests/test_readiness.py"


def test_fenced_headings_open_no_section():
    contract = parse_contract(1, QUOTES_THE_SCHEMA)

    # The fenced copy's `## Files`, `## Verify` and `## Blocked by` are opaque
    # text inside `## Goal`, so none of them is a repeat and none of them wins.
    assert contract.files == ("docs/issue-contract.md",)
    assert contract.verify == "test -f docs/issue-contract.md"
    assert contract.blocked_by == (7,)
    assert "One sentence." in contract.goal


def test_fenced_marker_is_not_identity():
    """The quoted example's id and attempt belong to the example, not the issue."""
    contract = parse_contract(1, QUOTES_THE_SCHEMA)

    assert contract.task_id == "document-the-schema"
    assert contract.attempt == 0


def test_a_body_that_is_only_a_fenced_schema_is_malformed():
    body = "## Goal\nExplain the schema.\n\n```\n## Files\n- a.py\n\n## Verify\ntrue\n\n## Blocked by\n```\n"

    with pytest.raises(ContractError) as caught:
        parse_contract(2, body)

    assert caught.value.section == "Files"


def test_a_heading_with_trailing_content_is_not_a_section():
    """`## Verify` and `## Verify (updated)` are not the same heading.

    The looser reading would let a human's aside redefine the gate; the strict
    one says the contract is missing, which is the failure they can see.
    """
    with pytest.raises(ContractError) as caught:
        parse_contract(3, contract_body().replace("## Verify\n", "## Verify (updated)\n"))

    assert caught.value.section == "Verify"
    assert "missing" in caught.value.reason


def test_prose_outside_the_sections_is_ignored():
    body = (
        "> Design: docs/architecture-v2.md.\n\nSome preamble nobody parses.\n\n"
        + contract_body()
        + "\n---\n\nRationale for humans, with a stray #42 in it.\n"
    )

    contract = parse_contract(3, body)

    assert contract.blocked_by == (7,)
    assert contract.goal == "`http_client.get` retries idempotent requests three times."


def test_unrecognised_heading_ends_the_previous_section():
    """`## Notes` terminates `## Verify` instead of being swallowed into it.

    Swallowed, the note would read as a second command and the issue would be
    rejected - so this test also proves the section did in fact end.
    """
    body = (
        "## Goal\nShip it.\n\n## Files\n- src/thing.py\n\n"
        "## Verify\npython -m pytest -q tests/test_thing.py\n\n"
        "## Notes\nRemember to rebase before merging.\n\n"
        "## Blocked by\n- #12\n"
    )

    contract = parse_contract(4, body)

    assert contract.verify == "python -m pytest -q tests/test_thing.py"
    assert contract.blocked_by == (12,)


# --------------------------------------------------------------------------
# Per-section rules
# --------------------------------------------------------------------------


def test_blocked_by_prose_means_no_dependencies():
    assert parse_contract(6, REAL_ISSUE_6).blocked_by == ()


def test_reference_outside_a_list_item_is_malformed():
    """The shape of a dependency that gets silently dropped."""
    with pytest.raises(ContractError) as caught:
        parse_contract(5, contract_body(blocked_by="Blocked on #12 landing first."))

    assert caught.value.section == "Blocked by"
    assert caught.value.number == 5


def test_cross_repository_reference_is_malformed():
    with pytest.raises(ContractError) as caught:
        parse_contract(5, contract_body(blocked_by="- shahrestani-me/other#12"))

    assert caught.value.section == "Blocked by"
    assert "cross-repository" in caught.value.reason


def test_missing_verify_is_malformed():
    body = "## Goal\nShip it.\n\n## Files\n- src/thing.py\n\n## Blocked by\n- #12\n"

    with pytest.raises(ContractError) as caught:
        parse_contract(7, body)

    assert caught.value.section == "Verify"


def test_two_verify_sections_are_malformed():
    body = contract_body() + "\n## Verify\npython -m pytest -q tests/test_other.py\n"

    with pytest.raises(ContractError) as caught:
        parse_contract(8, body)

    assert caught.value.section == "Verify"
    assert "more than once" in caught.value.reason


def test_fenced_verify_equals_the_bare_form():
    fenced = parse_contract(9, contract_body(verify="```sh\npython -m pytest -q tests/t.py\n```"))
    bare = parse_contract(9, contract_body(verify="python -m pytest -q tests/t.py"))

    assert fenced.verify == bare.verify == "python -m pytest -q tests/t.py"


def test_multi_line_verify_is_malformed():
    with pytest.raises(ContractError) as caught:
        parse_contract(10, contract_body(verify="pip install -e .\npython -m pytest -q"))

    assert caught.value.section == "Verify"


@pytest.mark.parametrize(
    "path",
    ["- src/**/*.py", "- ../outside/thing.py", "- /etc/passwd", "- src/thing_[12].py"],
)
def test_unsafe_or_glob_paths_are_malformed(path):
    with pytest.raises(ContractError) as caught:
        parse_contract(11, contract_body(files=path))

    assert caught.value.section == "Files"


def test_file_paths_are_normalised():
    contract = parse_contract(12, contract_body(files="- `./src/thing.py`\n- tests/test_thing.py"))

    assert contract.files == ("src/thing.py", "tests/test_thing.py")


def test_empty_sections_are_malformed():
    for section, body in (
        ("Goal", contract_body(goal="")),
        ("Files", contract_body(files="")),
    ):
        with pytest.raises(ContractError) as caught:
            parse_contract(13, body)
        assert caught.value.section == section


def test_goal_collapses_newlines():
    contract = parse_contract(14, contract_body(goal="One sentence,\nspread over\ntwo lines."))

    assert contract.goal == "One sentence, spread over two lines."


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def test_marker_supplies_id_and_attempt():
    contract = parse_contract(15, contract_body(marker=render_marker("add-retry-logic", 2)))

    assert (contract.task_id, contract.attempt) == ("add-retry-logic", 2)


def test_missing_attempt_field_reads_as_zero():
    contract = parse_contract(16, contract_body(marker="<!-- apiary:task id=add-retry-logic -->"))

    assert contract.attempt == 0


def test_unparseable_attempt_field_reads_as_zero():
    contract = parse_contract(
        17, contract_body(marker="<!-- apiary:task id=add-retry-logic attempt=soon -->")
    )

    assert contract.attempt == 0


def test_a_marker_id_that_is_not_a_slug_is_malformed():
    with pytest.raises(ContractError) as caught:
        parse_contract(18, contract_body(marker="<!-- apiary:task id=Add_Retry_Logic -->"))

    assert caught.value.section == "marker"


def test_slugify_matches_planned_task_id_shape():
    assert slugify("Read the task ledger from GitHub issues!") == "read-the-task-ledger-from-github-issues"
    assert len(slugify("x" * 200)) <= 64


# --------------------------------------------------------------------------
# Labels → status
# --------------------------------------------------------------------------


def test_every_state_label_has_a_status():
    assert set(STATUS_BY_LABEL) == set(LABEL_PRECEDENCE)


@pytest.mark.parametrize(
    "label,status",
    [
        ("swarm:ready", "pending"),
        ("swarm:blocked", "pending"),
        ("swarm:claimed", "running"),
        ("swarm:review", "running"),
        ("swarm:done", "verified"),
        ("swarm:failed", "abandoned"),
    ],
)
def test_label_maps_to_the_documented_status(label, status):
    """§3's table, both of its counter-intuitive rows included.

    `review` is `running` because a PR in review can still fail CI - mapping it
    to `verified` would let the judge call a run finished while its output sits
    in open PRs. `failed` is `abandoned` because v2's `failed` label means
    "attempts exhausted, needs a human", which is v1's `abandoned` exactly.
    """
    client = FakeClient([issue(20, contract_body(), labels=[label])])

    entry = load_ledger(client).entries["add-retry-logic"]

    assert entry.status == status
    assert entry.state_label == label


def test_two_state_labels_are_repaired_by_precedence():
    client = FakeClient(
        [issue(21, contract_body(), labels=["swarm:claimed", "swarm:done", "area/ops"])]
    )

    ledger = load_ledger(client)
    entry = ledger.entries["add-retry-logic"]

    assert (entry.state_label, entry.status) == ("swarm:done", "verified")
    assert [(r.number, r.kept, r.removed) for r in ledger.repairs] == [
        (21, "swarm:done", ("swarm:claimed",))
    ]
    # Reporting only: removing the losing label is the reconciler's write.
    assert client.patched == []


def test_routing_labels_are_not_state_labels():
    assert resolve_state_label(1, ["area/control-plane", "size/M"]) == (None, None)


# --------------------------------------------------------------------------
# Ledger membership
# --------------------------------------------------------------------------


def test_an_issue_without_a_swarm_label_is_not_in_the_ledger():
    """This repository's own backlog is the live example."""
    client = FakeClient(
        [
            issue(11, REAL_ISSUE_11, title="Compute readiness", labels=["area/control-plane"]),
            issue(6, REAL_ISSUE_6, title="Define the issue contract", labels=["size/S"]),
        ]
    )

    ledger = load_ledger(client)

    assert ledger.entries == {}
    assert ledger.errors == ()
    assert ledger.ignored == (6, 11)
    assert client.patched == []


def test_an_unparseable_issue_is_reported_and_never_dispatched():
    """§1.4: one bad hand-written issue must not stop a cycle."""
    client = FakeClient(
        [
            issue(30, "no contract here at all", labels=["swarm:ready"]),
            issue(31, contract_body(), labels=["swarm:ready"]),
        ]
    )

    ledger = load_ledger(client)

    assert list(ledger.entries) == ["add-retry-logic"]
    assert [(e.number, e.section) for e in ledger.errors] == [(30, "Goal")]
    # No partial record: the malformed issue contributes nothing at all.
    assert 30 not in ledger.by_number


def test_load_tasks_raises_on_a_malformed_issue_by_default():
    client = FakeClient([issue(30, "no contract here at all", labels=["swarm:ready"])])

    with pytest.raises(ContractError):
        load_tasks(client)

    assert load_tasks(client, strict=False) == {}


def test_closed_done_issues_stay_in_the_ledger():
    client = FakeClient([issue(32, contract_body(), labels=["swarm:done"])])

    load_ledger(client)

    assert client.last_state == "all"


# --------------------------------------------------------------------------
# Adoption
# --------------------------------------------------------------------------


def test_an_issue_without_a_marker_is_adopted():
    body = contract_body(marker=None)
    client = FakeClient([issue(40, body, title="Compute readiness!", labels=["swarm:ready"])])

    ledger = load_ledger(client)
    entry = ledger.entries["compute-readiness"]

    assert entry.adopted is True
    assert entry.attempt == 0
    # One invisible line, prepended, with every other byte preserved.
    assert client.patched == [(40, f"{render_marker('compute-readiness', 0)}\n\n{body}")]


def test_adoption_can_be_suppressed_for_a_read_only_caller():
    client = FakeClient([issue(41, contract_body(marker=None), labels=["swarm:ready"])])

    ledger = load_ledger(client, adopt=False)

    assert list(ledger.entries) == ["a-task"]
    assert client.patched == []


def test_adopted_ids_that_collide_get_the_issue_number():
    client = FakeClient(
        [
            issue(42, contract_body(marker=None), title="A task", labels=["swarm:ready"]),
            issue(43, contract_body(marker=None), title="A task", labels=["swarm:ready"]),
        ]
    )

    ledger = load_ledger(client)

    assert sorted(ledger.entries) == ["a-task", "a-task-43"]


def test_two_issues_with_the_same_marker_id_abort_the_cycle():
    client = FakeClient(
        [
            issue(50, contract_body(), labels=["swarm:ready"]),
            issue(51, contract_body(), labels=["swarm:claimed"]),
        ]
    )

    with pytest.raises(DuplicateTaskIdError) as caught:
        load_ledger(client)

    assert caught.value.numbers == (50, 51)
    assert "50" in str(caught.value) and "51" in str(caught.value)


# --------------------------------------------------------------------------
# The projection onto v1's TaskRecord
# --------------------------------------------------------------------------


def test_task_records_have_the_v1_shape():
    client = FakeClient(
        [
            issue(
                60,
                contract_body(
                    marker=render_marker("add-retry-logic", 2), blocked_by="- #61"
                ),
                labels=["swarm:blocked"],
            ),
            issue(
                61,
                contract_body(
                    marker=render_marker("add-the-client"),
                    files="- src/swarm/github/client.py",
                    blocked_by="_none._",
                ),
                labels=["swarm:done"],
            ),
        ]
    )

    tasks = load_tasks(client)

    assert tasks["add-retry-logic"] == {
        "id": "add-retry-logic",
        "goal": "`http_client.get` retries idempotent requests three times.",
        "files": ["src/swarm/github/client.py", "tests/test_client_retry.py"],
        # Task ids, not issue numbers: v1's graph intersects this against the
        # set of verified task ids, and an integer would never match.
        "depends_on": ["add-the-client"],
        "status": "pending",
        "attempts": 2,
        # Addressing, so the branch name uses the number.
        "branch": "swarm/issue-60",
    }
    assert tasks["add-the-client"]["status"] == "verified"


def test_dependencies_outside_the_ledger_are_dropped_from_the_projection():
    """`depends_on` is a task-id graph, so an unledgered ref has no id to name.

    The ref itself survives on the entry, which is what readiness (#11) reads -
    it resolves the issue's state, and "closed as not planned" is a fact the
    v1 projection cannot represent at all.
    """
    client = FakeClient([issue(70, contract_body(blocked_by="- #999"), labels=["swarm:ready"])])

    ledger = load_ledger(client)
    entry = ledger.entries["add-retry-logic"]

    assert entry.blocked_by == (999,)
    assert entry.depends_on == ()
    assert ledger.tasks["add-retry-logic"]["depends_on"] == []


def test_the_verify_command_survives_only_on_the_entry():
    """`TaskRecord` has nowhere to put it, which is why `LedgerEntry` exists."""
    client = FakeClient([issue(80, contract_body(), labels=["swarm:review"])])

    ledger = load_ledger(client)

    assert ledger.entries["add-retry-logic"].verify == "python -m pytest -q tests/test_client_retry.py"
    assert "verify" not in ledger.tasks["add-retry-logic"]


# --------------------------------------------------------------------------
# The optional fifth section (#98)
# --------------------------------------------------------------------------
#
# `SECTIONS` used to be one tuple doing two jobs - it built the known-heading
# regex *and* it was the required-section loop - which is why "exactly four
# sections" read as immovable. There was no way to say "recognised but
# optional".


def test_a_body_with_no_stack_parses_exactly_as_before():
    """The compatibility floor. Every issue written before this section existed
    is a Python one, and none of them may start failing to parse."""
    contract = parse_contract(7, contract_body())

    assert contract.stack is None
    assert contract.goal and contract.files and contract.verify


def test_a_body_with_no_stack_defaults_to_python_on_the_entry():
    """`TaskContract.stack` records what the body said; `LedgerEntry.stack`
    records the answer. Keeping them distinct is what makes "the body did not
    declare" answerable at all."""
    client = FakeClient([issue(7, contract_body())])

    entry = load_ledger(client).entries["add-retry-logic"]

    assert entry.stack == DEFAULT_STACK == "python"


@pytest.mark.parametrize("declared", sorted(KNOWN_STACKS))
def test_a_declared_stack_is_parsed_and_resolved(declared):
    contract = parse_contract(7, contract_body(stack=declared))

    assert contract.stack == declared


#: The known-heading regex as it was before `## Stack` existed. Restoring it is
#: the whole simulation of an older orchestrator: it is the only thing about the
#: parser that differed.
PRE_STACK_HEADING_RE = re.compile(rf"^## ({'|'.join(REQUIRED_SECTIONS)})[ \t]*$")


def _stack_at(position: str) -> str:
    """The same contract with `## Stack` in one of three places."""
    goal, files = "## Goal\ng\n", "## Files\n- src/a.py\n"
    verify, blocked = "## Verify\npytest -q\n", "## Blocked by\n- #7\n"
    stack = "## Stack\nnode\n"
    order = {
        "last": (goal, files, verify, blocked, stack),
        "middle": (goal, files, stack, verify, blocked),
        "second": (goal, stack, files, verify, blocked),
    }[position]
    return "\n".join(order)


@pytest.mark.parametrize("position", ["last", "middle", "second"])
def test_a_body_with_a_stack_parses_on_the_pre_change_parser(position):
    """**The property that makes this change safe to deploy**, and it holds
    more strongly than the plan for #98 assumed.

    The plan said `## Stack` had to be emitted last or compatibility "stops
    being true". It does not: `_split_sections` treats an unrecognised ATX
    heading as a terminator, and any *later recognised* heading opens a new
    section - so an older parser reads all four required sections identically
    wherever the fifth one sits, and simply discards it.

    Parametrised over three placements rather than asserting the one, because
    the useful guarantee is the general one: an issue a newer planner wrote
    cannot break an orchestrator that has not been redeployed.
    """
    text = _stack_at(position)

    with mock.patch.object(ledger_module, "_KNOWN_HEADING_RE", PRE_STACK_HEADING_RE):
        before = parse_contract(7, text)

    after = parse_contract(7, text)

    assert (before.goal, before.files, before.verify, before.blocked_by) == (
        after.goal,
        after.files,
        after.verify,
        after.blocked_by,
    )
    assert before.blocked_by == (7,)
    assert before.stack is None and after.stack == "node"


def test_the_planner_still_writes_the_stack_last():
    """Position is a readability decision, not a compatibility one — but it is
    still a decision, and `render_body` is where it is made. Pinned so the
    canonical example in §6 and the bodies actually written cannot diverge."""
    from swarm.nodes.planner import render_body

    rendered = render_body(
        "add-retry-logic", goal="g", files=["src/a.py"], verify="pytest -q",
        blocked_by=[7], stack="node",
    )

    assert rendered.index("## Stack") > rendered.index("## Blocked by")


def test_a_python_task_writes_no_stack_section_at_all():
    """So a Python plan's bodies are byte-for-byte what they were, and adding
    this section costs nothing to a repository that never uses it."""
    from swarm.nodes.planner import render_body

    rendered = render_body("add-retry-logic", goal="g", files=["src/a.py"], verify="pytest -q")

    assert "## Stack" not in rendered
    assert parse_contract(7, rendered).stack is None


def test_an_unknown_stack_is_refused_rather_than_defaulted():
    """A silent default is the failure worth preventing: `## Stack` `rust` in a
    repo with no Rust image would become a Python container that fails its gate
    three times, and the message would name the tests."""
    with pytest.raises(ContractError) as raised:
        parse_contract(7, contract_body(stack="rust"))

    assert raised.value.section == "Stack"
    assert "rust" in str(raised.value)
    assert "python" in str(raised.value)


def test_an_empty_stack_section_is_refused():
    """Somebody wrote the heading meaning to say something. Reading that as "no
    opinion" discards their intent silently - the same call `## Verify` makes."""
    with pytest.raises(ContractError) as raised:
        parse_contract(7, contract_body(stack=""))

    assert raised.value.section == "Stack"


def test_two_stacks_are_refused():
    text = contract_body(stack="node\npython")

    with pytest.raises(ContractError) as raised:
        parse_contract(7, text)

    assert raised.value.section == "Stack"


def test_a_stack_is_read_case_insensitively_and_unbackticked():
    """A human writes `Node`; a model writes `` `node` ``. Neither meant a
    different stack, and refusing them would be pedantry with a `ContractError`
    attached."""
    assert parse_contract(7, contract_body(stack="Node")).stack == "node"
    assert parse_contract(7, contract_body(stack="`react`")).stack == "react"


def test_every_generatable_stack_is_a_declarable_one():
    """The drift pin.

    `KNOWN_STACKS` deliberately does not import `greenfield.scaffold.STACKS` -
    that registry is about generating a project, this vocabulary is about what
    an issue may declare, and `github/` depending on `greenfield/` would be the
    wrong way round. Two lists mean drift unless something notices, and a stack
    the scaffolder can emit but the parser refuses is a repository whose own
    planner cannot write an issue for it.
    """
    from swarm.greenfield.scaffold import STACKS

    assert set(STACKS) <= KNOWN_STACKS
