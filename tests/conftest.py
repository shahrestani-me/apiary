"""Shared fixtures, and the markers that keep the default suite hermetic.

Two jobs.

**The doubles.** `fake_github` and `scratch_repo` expose `fixtures/github.py`
and `fixtures/repo.py` as pytest fixtures. Anything that talks to the GitHub
API or to a git remote should reach for one of those rather than growing a
sixth private version of it.

**The roots.** `hermetic_roots` is autouse: every test runs with the three
`.swarm/` roots - runs, console sessions and the task store - pointed at its
own `tmp_path`. Passing explicit paths remains the right thing for a test that
cares (see `ProjectStore`'s docstring on its seams); this is the floor beneath
that, for the default-constructed store nobody remembered to seam. Without it
`Console()` carries a `ProjectStore` addressed at the developer's real
`.swarm/projects.sqlite`, and the console tests read *and write* it - which is
how three fixture repositories ended up in one developer's database while CI,
having no `.swarm/` at all, stayed green.

**The gate.** Three markers - `docker`, `network` and `ollama` - name the three
things a laptop may have and CI does not. They are deselected by default, so
`pytest -q` is the hermetic suite everywhere, and enabled per-marker with
`--with-docker`, `--with-network`, `--with-ollama`, `--with-all`, or
`APIARY_TEST_MARKERS=docker,ollama` (or `all`). Naming a marker in `-m` enables
it too, so `pytest -m docker` means what it looks like it means instead of
collecting nothing.

The count of what was hidden is printed on every run, including under `-q`.
That is the point of preferring a gate to a silent skip: a test that runs
nowhere is worse than one that fails, and the only way to notice it is to be
told how many there are. CI prints the line into the job summary.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

import pytest

from fixtures.github import client as _github_client
from fixtures.repo import ScratchRepo

#: marker -> (what the machine must have, how to turn it on)
GATED_MARKERS: dict[str, tuple[str, str]] = {
    "docker": ("a reachable Docker daemon", "--with-docker"),
    "network": ("outbound network access and a GITHUB_TOKEN", "--with-network"),
    "ollama": ("an Ollama server on the host", "--with-ollama"),
}

MARKER_ENV_VAR = "APIARY_TEST_MARKERS"

_GATE = pytest.StashKey[dict]()


# --------------------------------------------------------------------------
# Markers
# --------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("apiary", "apiary gated markers")
    for marker, (needs, flag) in GATED_MARKERS.items():
        group.addoption(
            flag,
            action="store_true",
            default=False,
            help=f"also run tests marked {marker!r}; they need {needs}",
        )
    group.addoption(
        "--with-all",
        action="store_true",
        default=False,
        help="also run every gated test: " + ", ".join(GATED_MARKERS),
    )


def pytest_configure(config: pytest.Config) -> None:
    for marker, (needs, flag) in GATED_MARKERS.items():
        config.addinivalue_line(
            "markers",
            f"{marker}: needs {needs}; deselected unless {flag} is given",
        )


def enabled_markers(config: pytest.Config) -> set[str]:
    """Which gated markers this invocation asked for."""
    if config.getoption("--with-all"):
        return set(GATED_MARKERS)

    requested = {
        token.strip().lower()
        for token in os.environ.get(MARKER_ENV_VAR, "").split(",")
        if token.strip()
    }
    if "all" in requested:
        return set(GATED_MARKERS)

    enabled = requested & set(GATED_MARKERS)
    enabled |= {m for m in GATED_MARKERS if config.getoption(GATED_MARKERS[m][1])}
    # An explicit `-m` naming a marker is a request for it. `-m "not docker"`
    # enables it too and then pytest's own expression deselects it again, which
    # is the same outcome by a longer route.
    markexpr = config.getoption("markexpr", default="") or ""
    enabled |= {m for m in GATED_MARKERS if re.search(rf"\b{m}\b", markexpr)}
    return enabled


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    enabled = enabled_markers(config)
    deselected: list[pytest.Item] = []
    kept: list[pytest.Item] = []
    counts = dict.fromkeys(GATED_MARKERS, 0)

    for item in items:
        blocked = [m for m in GATED_MARKERS if m not in enabled and item.get_closest_marker(m)]
        if blocked:
            deselected.append(item)
            for marker in blocked:
                counts[marker] += 1
        else:
            kept.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept

    config.stash[_GATE] = {"enabled": enabled, "deselected": counts}


def pytest_report_header(config: pytest.Config) -> str:
    enabled = enabled_markers(config)
    on = ", ".join(m for m in GATED_MARKERS if m in enabled) or "none"
    off = ", ".join(m for m in GATED_MARKERS if m not in enabled) or "none"
    return f"apiary gated markers: enabled {on}; deselected {off}"


def pytest_terminal_summary(terminalreporter: Any) -> None:
    gate = terminalreporter.config.stash.get(_GATE, None)
    if gate is None:  # collection never ran
        return

    enabled, counts = gate["enabled"], gate["deselected"]
    summary = ", ".join(
        f"{marker} enabled" if marker in enabled else f"{marker} {counts[marker]} deselected"
        for marker in GATED_MARKERS
    )
    terminalreporter.write_line(f"apiary gated markers: {summary}")
    if any(counts[m] for m in counts if m not in enabled):
        flags = " ".join(GATED_MARKERS[m][1] for m in GATED_MARKERS if m not in enabled)
        terminalreporter.write_line(f"apiary: run them with {flags}, or --with-all")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


#: The three roots a test must never write to on a developer's machine. Each
#: is its own setting on purpose - see `test_store.py` on why the store root is
#: not derived from the artifacts root - so each is redirected by name.
HERMETIC_ROOT_ENV_VARS = {
    "APIARY_ARTIFACTS_DIR": "runs",
    "APIARY_CONSOLE_DIR": "console",
    "APIARY_STORE_DIR": "store",
}


@pytest.fixture(autouse=True)
def hermetic_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point every `.swarm/` root at this test's `tmp_path`.

    The defaults are relative (`.swarm/runs`), so "the developer's real state"
    is whatever sits under the directory pytest happens to be run from - the
    repository itself, usually, holding real runs and a real projects
    database. A test that constructs its own paths never noticed; a test that
    let a default through wrote into that database and passed anyway.

    A test that wants the default back deletes the variable itself
    (`monkeypatch.delenv(..., raising=False)`), which is what the two tests
    asserting the defaults already do - this fixture runs first, and theirs
    wins.
    """
    root = tmp_path / ".swarm"
    for var, leaf in HERMETIC_ROOT_ENV_VARS.items():
        monkeypatch.setenv(var, str(root / leaf))
    return root


@pytest.fixture()
def fake_github() -> Callable[..., tuple[Any, Any, list[float]]]:
    """Build a `GitHubClient` on a scripted transport.

    Returns the factory rather than a client, because the script is what each
    test differs by:

        gh, transport, slept = fake_github(response(200, {"number": 7}))
    """
    return _github_client


@pytest.fixture()
def scratch_repo(tmp_path: Path) -> ScratchRepo:
    """A seeded git repo with a bare `origin`, both under `tmp_path`."""
    return ScratchRepo.create(tmp_path)
