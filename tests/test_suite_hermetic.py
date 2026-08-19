"""The suite writes nowhere but `tmp_path`, held still.

The `.swarm/` roots default to *relative* paths - `.swarm/runs`,
`.swarm/console`, `.swarm/store` - so on a developer's machine they resolve
under whatever directory pytest was invoked from, which is the repository, and
the repository is where that developer's real runs and real projects database
live. CI has none of it, so nothing here fails there either way. That
asymmetry is the whole problem: the suite could read and write a developer's
state for months and stay green in the only place anybody looks.

It did. `Console.projects` is a `ProjectStore` by default factory, a build
records the repository it built, and the console tests built three - so
`expense-tracker` and `a-cli-that-tracks-expenses` sat in one developer's
`.swarm/projects.sqlite` next to a real project, and the one console test that
asserted the *contents* of the store failed on that machine and nowhere else.

`conftest.hermetic_roots` is the fix and this file is what keeps it. The
assertions are deliberately about the roots rather than about any one store:
the next default-constructed thing to reach for `artifacts_root()` should
inherit the guarantee without anybody remembering this file exists.
"""

from __future__ import annotations

from pathlib import Path

from swarm.artifacts import artifacts_root, console_root
from swarm.console_projects import ProjectStore
from swarm.store.sqlite import store_root


def test_every_swarm_root_lands_under_the_test_temporary_directory(tmp_path):
    for root in (artifacts_root(), console_root(), store_root()):
        assert root.is_absolute(), f"{root} is relative: it follows the cwd"
        assert root.is_relative_to(tmp_path), f"{root} escapes {tmp_path}"


def test_a_store_nobody_seamed_still_cannot_reach_the_real_database(tmp_path):
    """The failure mode this is really for: not the store a test configured,
    but the one it never mentioned, carried in by `Console()`'s default."""
    assert ProjectStore().path.is_relative_to(tmp_path)


def test_the_repository_checkout_is_not_a_root(tmp_path):
    """A guard against fixing this by pointing the roots at some *other* fixed
    place: `.swarm/` beside the source tree is exactly what must stop being
    reachable, whatever pytest's working directory happens to be."""
    checkout = Path(__file__).resolve().parent.parent
    for root in (artifacts_root(), console_root(), store_root()):
        # Resolved, not taken as given: an unguarded root is *relative*, and a
        # relative path is trivially "not under the checkout" while in fact
        # naming a directory inside it.
        assert not root.resolve().is_relative_to(checkout)
