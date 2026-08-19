"""Unit tests for `TaskRef` and the GitHub adapter that mints it (#142).

Three properties, and each of them is load-bearing somewhere else: the order
is what keeps `find_cycle` naming the same ring twice, `str` is what keeps
every message a human reads unchanged by the retype, and the round trip is the
only route back to an issue number - which is what makes "opaque above the
adapter" enforceable rather than a convention.

`PullRef` is tested here too, beside its twin rather than in a file of its own,
because every one of its cases is "the same rule for the other numbering" and
splitting them is how the two drift. Its guards arrived with #196 and its
ordering with #208 - the property `derived.py` was blocked on, whose reason is
`TaskRef`'s reason applied to a different caller: not the same ring twice, the
same pull request twice.
"""

from __future__ import annotations

import pytest

from swarm.github.refs import issue_number, pull_number, pull_ref, task_ref
from swarm.taskref import PullRef, TaskRef


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


@pytest.mark.parametrize("value", ["", " ", "#42 ", " #42"])
def test_an_empty_or_padded_pull_ref_is_refused_at_construction(value):
    """`PullRef` shipped in #185 with this branch never executed by a test.

    The same guard as its twin above, and it matters for the same reason: a
    padded value reaches `PUT /pulls/{n}/merge` as a URL nobody can read back,
    and the failure surfaces at the API rather than at the construction.
    """
    with pytest.raises(ValueError):
        PullRef(value)


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
# `PullRef`'s ordering - the property `derived.py` was blocked on (#208)
# --------------------------------------------------------------------------


def test_pull_refs_sort_numerically_over_a_mixed_set():
    """A stable sort over several, not two refs that happen to compare.

    `derived._landed` and `derived._merged_pull` both run `sorted()` over one
    cycle's pull requests and have to agree about which merge a task landed
    through, so the property under test is the whole sequence: `#9` before `#10`
    for every element, not just for the pair somebody thought to check.
    """
    numbers = [101, 9, 1004, 10, 2]
    assert sorted(pull_ref(n) for n in numbers) == [pull_ref(n) for n in sorted(numbers)]


def test_two_pull_refs_that_look_alike_numerically_are_still_ordered():
    """The zero-padding case, for the twin of `TaskRef`'s reason.

    `_natural_key` exists partly because `#007` and `#7` share every numeric
    chunk, so without the value as a tiebreaker they compare neither equal nor
    less-than in either direction and `sorted()` falls back to insertion order.

    Unreachable through `pull_ref`, which `int()`-normalises, and GitHub is the
    only code host ADR 0001 admits - so unlike `TaskRef`'s version of this case
    there is no second spelling waiting to arrive. It is asserted anyway because
    the tiebreaker is *shared*: somebody tidying `_natural_key` for the type
    whose minter cannot produce a padded value breaks the type whose can.
    """
    padded, bare = PullRef("#007"), PullRef("#7")

    assert padded != bare
    assert (padded < bare) != (bare < padded), "neither is ordered before the other"
    assert sorted([padded, bare]) == sorted([bare, padded])


def test_a_pull_ref_orders_inside_the_tuple_open_pull_compares():
    """`derived._open_pull` compares `(attempt, number)`, so the second element
    is only reached when the attempts are equal - and before #208 reaching it
    was a `TypeError` rather than a wrong answer. Two open pull requests on one
    attempt means a worker published twice for one dispatch; the newer one wins.
    """
    assert (1, pull_ref(70)) < (1, pull_ref(71))
    assert (2, pull_ref(70)) > (1, pull_ref(71))


def test_a_pull_ref_is_not_ordered_against_a_task_ref():
    """The no-base-class choice, held at the comparison too.

    Assignment across the two is rejected by mypy; a comparison would be the
    hole that reopens the shared vocabulary at runtime, and `sorted()` over a
    list holding both is never something a caller means. Both directions,
    because `total_ordering` fills in the reflected operators and a one-sided
    guard would let one of them through.
    """
    with pytest.raises(TypeError):
        _ = pull_ref(1) < task_ref(1)
    with pytest.raises(TypeError):
        _ = task_ref(1) < pull_ref(1)
    with pytest.raises(TypeError):
        _ = pull_ref(1) < 1


# --------------------------------------------------------------------------
# The adapter boundary
# --------------------------------------------------------------------------


def test_a_github_ref_round_trips_through_the_adapter():
    assert issue_number(task_ref(7)) == 7


def test_a_ref_another_adapter_minted_is_refused_rather_than_guessed():
    """Addressing some unrelated issue in this repository is the worse failure."""
    with pytest.raises(ValueError, match="not minted by the GitHub adapter"):
        issue_number(TaskRef("ENG-123"))


def test_a_pull_ref_another_adapter_minted_is_refused_rather_than_guessed():
    """The twin of the case above, and dead in the suite until now.

    Un-minting a pull ref is ordinary rather than a smell - ADR 0001 keeps the
    code host GitHub-shaped - but "ordinary" is not "unchecked": a value some
    other adapter spelled would address an unrelated pull request in this
    repository, which is the same worse-failure this guard exists for.
    """
    with pytest.raises(ValueError, match="not minted by the GitHub adapter"):
        pull_number(PullRef("ENG-123"))


def test_a_pull_ref_round_trips_through_its_minter():
    """The property the type exists for: it can be handed back to the API."""
    assert pull_number(pull_ref(7)) == 7


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
