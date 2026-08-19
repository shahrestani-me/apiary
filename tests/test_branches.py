"""The branch name, and the one property everything downstream rests on.

ADR 0001 makes agent execution state derived rather than stored, and #144 is
the ticket that puts the two facts a crash destroys into the branch name. That
only buys anything if the name is *lossless*: recovery reads a name off a remote
with no ledger in hand, so a `(ref, attempt)` that does not come back exactly as
it went in is a task the orchestrator cannot re-associate with its work.

So the round trip is the subject, asserted over the refs that are awkward rather
than the one that is convenient - long, unicode, leading-digit, punctuation-only,
`ENG-123`, and refs that contain the separator this format splits on. Alongside
it, the two refusals: a name from before this ticket is unrecognised rather than
guessed at, and a ref too long to be a git ref name is refused at mint time by
the module that built it.

`git check-ref-format` is run against every name these tests mint. The
encoding's whole job is to produce something git accepts, and that is not a
claim a restatement of git's rules in Python can make - only git can. It needs
no marker: the suite already shells out to git for `fixtures/repo.py`.
"""

from __future__ import annotations

import subprocess

import pytest

from swarm.github.branches import (
    ATTEMPT_SEPARATOR,
    BRANCH_PREFIX,
    MAX_COMPONENT_BYTES,
    TaskBranch,
    parse_task_branch,
    task_branch,
)
from swarm.github.refs import task_ref
from swarm.taskref import TaskRef

#: One per awkward shape, with the reason each is awkward. Every tracker apiary
#: might ever speak to is somewhere in this list.
AWKWARD_REFS = [
    "#42",  # GitHub, whose spelling is not a git-safe token at all
    "#4242",
    "ENG-123",  # Linear: already safe, and must survive unchanged
    "PROJ-7",  # Jira
    "7",  # a bare number, from a tracker with no sigil
    "0",
    "a" * 200,  # a long slug, still inside git's limit
    "add-retry-logic-to-the-github-client",
    "TASK/17",  # a separator that would otherwise make a second path component
    "tache-numéro-9",  # non-ASCII, one byte over ASCII
    "課題-42",  # non-ASCII, three bytes per character
    "task attempt 3",  # spaces, which git refuses outright
    "-leading-dash",  # a name git commands would read as an option
    "_leading-underscore",
    "%not-an-escape%",  # the escape character itself, unescaped
    "ref~with^every:forbidden?char*",
    "ends-with-a-dot.",  # git refuses a component ending in `.`
    "has..double..dots",
    "ref-attempt-3",  # collides with this format's own separator
    "-attempt-9",
    "ENG-123-attempt-2-attempt-4",
]


def check_ref_format(name: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "check-ref-format", f"refs/heads/{name}"],
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------
# The round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", AWKWARD_REFS)
@pytest.mark.parametrize("attempt", [0, 1, 9, 10, 137])
def test_a_branch_name_round_trips_to_the_pair_it_was_minted_for(value, attempt):
    """The acceptance criterion, and the reason the encoding is not
    `TaskRef.label_value`: that one collapses refs that differ only in
    characters Docker forbids, and two tasks recovery cannot tell apart is
    exactly what a derived control plane cannot afford."""
    parsed = parse_task_branch(task_branch(TaskRef(value), attempt))

    assert parsed == TaskBranch(ref=TaskRef(value), attempt=attempt)


def test_refs_that_differ_only_in_unsafe_characters_do_not_collapse():
    """The property `label_value` deliberately does not have. `#42` and `42`
    are one Docker token and must be two branches, or a recovery sweep on a
    tracker with a sigil would hand one task's work to another."""
    names = {task_branch(TaskRef(value), 0) for value in ("#42", "42", "%2342")}

    assert len(names) == 3


def test_a_ref_that_is_already_git_safe_is_left_alone_and_reads_as_itself():
    """Readability is not a nicety here: a human greps `git branch --list` with
    this, and #142 gave up injectivity for exactly this property. Trackers whose
    ids need no escaping should pay nothing for the ones that do."""
    assert task_branch(TaskRef("ENG-123"), 2) == f"{BRANCH_PREFIX}ENG-123{ATTEMPT_SEPARATOR}2"


def test_a_ref_containing_the_separator_splits_at_the_last_one():
    """`-attempt-` is spelled out rather than a bare `-` so that a ref ending in
    `-3` is not read as attempt 3, but a ref may still contain the whole
    separator. The attempt is the anchored, right-hand end of the name."""
    parsed = parse_task_branch(task_branch(TaskRef("ENG-attempt-3"), 5))

    assert parsed == TaskBranch(ref=TaskRef("ENG-attempt-3"), attempt=5)


def test_the_github_adapters_own_ref_survives_the_trip():
    """The one that matters today, spelled through the adapter rather than by
    hand: nothing in this module knows a GitHub ref looks like `#42`."""
    parsed = parse_task_branch(task_branch(task_ref(4242), 1))

    assert parsed is not None and parsed.ref == task_ref(4242)


def test_a_task_branch_prints_as_the_name_it_was_parsed_from():
    name = task_branch(task_ref(7), 3)

    assert str(parse_task_branch(name)) == name


# --------------------------------------------------------------------------
# What git will accept
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", AWKWARD_REFS)
def test_every_minted_name_is_one_git_will_accept(value):
    """The claim the encoding exists to make, checked against git itself rather
    than against a restatement of git's rules. A name this refuses is a task
    whose push fails after the gate has already passed - the most expensive
    moment in a run to discover a naming bug."""
    result = check_ref_format(task_branch(TaskRef(value), 0))

    assert result.returncode == 0, f"git refused {task_branch(TaskRef(value), 0)!r}"


def test_a_ref_too_long_to_be_a_branch_is_refused_where_it_is_minted():
    """Refused here, by the module that built the name, rather than three layers
    away in a git error at the end of a finished task."""
    with pytest.raises(ValueError, match="over git's"):
        task_branch(TaskRef("x" * MAX_COMPONENT_BYTES), 0)


def test_a_negative_attempt_is_a_caller_bug_and_says_so():
    """It would encode as `-1`, put a second dash beside the separator, and
    parse back as something else - a silent round-trip failure."""
    with pytest.raises(ValueError, match="negative"):
        task_branch(task_ref(7), -1)


# --------------------------------------------------------------------------
# Everything else on the remote
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "swarm/issue-7",  # minted before this ticket
        "swarm/issue-7-attempt-0",  # the old prefix with the new suffix
        "main",
        "",
        "apiary/",
        "apiary/ENG-123",  # ours, but with no attempt in it
        "apiary/ENG-123-attempt-",
        "apiary/ENG-123-attempt-x",
        "apiary/-attempt-0",  # an empty ref, which `TaskRef` refuses
        "apiary/%FF-attempt-0",  # an escape that is not valid UTF-8
        "fix/typo",
        "renovate/urllib3-2.x",
    ],
)
def test_a_branch_this_scheme_did_not_mint_is_unrecognised_rather_than_guessed(name):
    """The acceptance criterion about legacy branches, widened to everything
    else a remote holds. A remote is not apiary's alone: it carries branches
    from before #144, a human's work, and whatever a bot pushed. Parsing has to
    answer "not mine" for all of them, because the alternative - one stray
    branch raising - turns a recovery sweep into a crash."""
    assert parse_task_branch(name) is None
