"""The recorder is separable from the window, and stays that way.

`orchestrator/observed.py` was split out of `orchestrator/shadow.py` in #152 c1
so that removing the window is a file deletion rather than a dissection. That is
only true while the dependency runs one way, and "one way" is not a property a
reader can check by looking at two files of a thousand lines each.

So it is asserted, from two directions: an `ast` pass over the file, which sees
every import it declares at any depth, and a real import with the window
poisoned, which sees the ones it reaches through another module. Each catches
something the other cannot; the docstrings below say which, because a pair of
tests that look redundant is a pair somebody deletes one of.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "swarm" / "orchestrator"


def _imported_modules(path: Path) -> set[str]:
    """Every module name this file imports, relative names included."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_the_recorder_does_not_import_the_window():
    """The dependency runs one way, and #152's last step depends on it.

    `shadow.py` imports the recorder; the recorder must not import back. When
    the window goes, `observed.py` has to keep working with the file deleted -
    so an import in this direction is not a style question, it is the difference
    between deleting a module and unpicking one.
    """
    assert "shadow" not in _imported_modules(SOURCE / "observed.py")


def test_the_recorder_imports_without_the_window_present():
    """The other half, and it is a different half than it first looks.

    `ast.walk` above descends into function bodies, so it already catches a
    deferred `from .shadow import ...` - which is the form this dependency would
    most plausibly come back in, since `shadow.py` uses that style itself to
    break its own cycles. Verified by writing one and watching the test above go
    red; this one stays green, because the function is never called.

    What this catches instead is a **transitive** import-time dependency: the
    recorder importing some third module that imports the window. No pass over
    this one file can see that, and the deletion in #152's last step fails just
    as hard on it. So the module is imported for real with
    `orchestrator.shadow` poisoned, which is the only way to ask "would this
    still work with that file gone" and get an honest answer.
    """
    hidden = {
        name: module
        for name, module in sys.modules.items()
        if name.endswith("orchestrator.observed") or name.endswith("orchestrator.shadow")
    }
    for name in hidden:
        del sys.modules[name]
    sys.modules["swarm.orchestrator.shadow"] = None  # type: ignore[assignment]
    try:
        import importlib

        module = importlib.import_module("swarm.orchestrator.observed")
        # And it is the real thing, not an empty module that imported cleanly
        # because there was nothing in it to fail.
        assert callable(module.observed_line)
        assert callable(module.build_observation)
    finally:
        del sys.modules["swarm.orchestrator.shadow"]
        sys.modules.pop("swarm.orchestrator.observed", None)
        sys.modules.update(hidden)
