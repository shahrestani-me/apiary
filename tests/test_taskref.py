"""Unit tests for `TaskRef` and the GitHub adapter that mints it (#142).

Three properties, and each of them is load-bearing somewhere else: the order
is what keeps `find_cycle` naming the same ring twice, `str` is what keeps
every message a human reads unchanged by the retype, and the round trip is the
only route back to an issue number - which is what makes "opaque above the
adapter" enforceable rather than a convention.
"""

from __future__ import annotations

import pytest

from swarm.github.refs import issue_number, task_ref
from swarm.taskref import TaskRef


# --------------------------------------------------------------------------
# The type
# --------------------------------------------------------------------------


def test_a_ref_prints_as_the_value_its_adapter_chose():
    """`f"{ref}"` is how every existing message keeps its wording."""
    assert str(task_ref(42)) == "#42"
    assert f"{TaskRef('ENG-123')}" == "ENG-123"


def test_refs_are_hashable_and_compare_by_value():
    assert task_ref(42) == task_ref(42)
    assert task_ref(42) != task_ref(43)
    assert len({task_ref(42), task_ref(42), task_ref(43)}) == 2


@pytest.mark.parametrize("value", ["", " ", "#42 ", " #42"])
def test_an_empty_or_padded_ref_is_refused_at_construction(value):
    """The alternative is a graph node nothing resolves, found much later."""
    with pytest.raises(ValueError):
        TaskRef(value)


# --------------------------------------------------------------------------
# Ordering - the property `find_cycle` depends on
# --------------------------------------------------------------------------


def test_digit_runs_sort_numerically_not_lexicographically():
    """`#9` before `#10`, exactly as the `int` key this replaced did.

    Lexicographic order would put `#10` first and quietly change which ring
    `find_cycle` reports for a graph nobody edited.
    """
    numbers = [10, 9, 100, 2, 31]
    assert sorted(task_ref(n) for n in numbers) == [task_ref(n) for n in sorted(numbers)]


def test_the_same_order_holds_for_a_tracker_whose_ids_are_not_numbers():
    refs = [TaskRef("ENG-10"), TaskRef("ENG-2"), TaskRef("ENG-9")]
    assert sorted(refs) == [TaskRef("ENG-2"), TaskRef("ENG-9"), TaskRef("ENG-10")]


def test_refs_from_different_trackers_still_have_a_total_order():
    """A dictionary of mixed refs must be sortable, not raise halfway through."""
    mixed = [TaskRef("ENG-2"), task_ref(2), TaskRef("PROJ-2")]
    assert sorted(mixed) == sorted(reversed(mixed))


def test_two_refs_that_look_alike_numerically_are_still_ordered():
    """Zero padding is where a natural key stops being a strict weak ordering.

    `PROJ-007` and `PROJ-7` share every numeric chunk, so without a tiebreaker
    they compare neither equal nor less-than in either direction and `sorted()`
    silently falls back to insertion order - which is exactly the
    non-determinism `find_cycle` promises it does not have. Unreachable through
    `task_ref` (`int()` normalises), reachable for any tracker that pads.
    """
    padded, bare = TaskRef("PROJ-007"), TaskRef("PROJ-7")

    assert padded != bare
    assert (padded < bare) != (bare < padded), "neither is ordered before the other"
    assert sorted([padded, bare]) == sorted([bare, padded])


def test_a_ref_is_not_ordered_against_anything_else():
    with pytest.raises(TypeError):
        _ = task_ref(1) < 1


# --------------------------------------------------------------------------
# The adapter boundary
# --------------------------------------------------------------------------


def test_a_github_ref_round_trips_through_the_adapter():
    assert issue_number(task_ref(7)) == 7


def test_a_ref_another_adapter_minted_is_refused_rather_than_guessed():
    """Addressing some unrelated issue in this repository is the worse failure."""
    with pytest.raises(ValueError, match="not minted by the GitHub adapter"):
        issue_number(TaskRef("ENG-123"))


# --------------------------------------------------------------------------
# The Docker-safe form the container layer labels with
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, token",
    [
        ("#42", "42"),
        ("#9", "9"),
        ("ENG-123", "ENG-123"),
        ("PROJ_7", "PROJ_7"),
        ("probe", "probe"),
        ("a b/c", "abc"),
    ],
)
def test_label_value_keeps_only_what_docker_allows_in_a_name(value, token):
    assert TaskRef(value).label_value == token


def test_a_github_ref_labels_a_container_exactly_as_the_issue_number_did():
    """Why moving `containers/` onto `label_value` broke no wire format.

    The label was `apiary.issue=42` and the container was named
    `...-issue-42-...` when both came from an `int`. They still are, because a
    GitHub ref reduced to Docker's character set *is* its number. This is the
    property that made the change safe to ship without a compatibility window,
    so it is asserted rather than left to be rediscovered.
    """
    assert all(task_ref(n).label_value == str(n) for n in (1, 9, 42, 1234))


def test_a_ref_with_no_docker_safe_form_is_refused_rather_than_labelled():
    """Unreachable for any real tracker, and a `docker create` failure if not.

    Blaming the container layer for a ref it was handed is the confusing
    version of this error, so it is raised where the ref is.
    """
    with pytest.raises(ValueError, match="no Docker-safe form"):
        TaskRef("#/#").label_value
