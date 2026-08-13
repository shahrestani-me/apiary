"""Turning a goal into files on disk - and refusing everything else.

Ported from v1's [`nodes/worker.py`](../nodes/worker.py), which read the files
a task declared, asked the model for whole-file replacements in a forced
schema, and wrote back the ones it was allowed to write. That shape survives
v2 unchanged: local models under ~10B are unreliable at multi-turn tool use,
and a runaway agent loop costs wall-clock time even when tokens are free.

What is new here is the second file set.

## The readable context set

**Decision: the worker gets a readable-but-not-writable context set, derived
from the checkout.**

v1 handed the model only the files it was allowed to edit. On a toy target that
is fine. On a real repository it means the model writes code having never seen
the project's conventions, its neighbouring modules or its README - a ceiling
on output quality that no amount of retrying lifts, because every retry shows
the model exactly as little as the last one did. This is not speculative: a run
of this very backlog produced tickets whose implementations could not see the
modules they had to call.

So `gather_context` collects, deterministically and within a character budget:

- the project's front matter at the repository root - the README, the
  contributing guide, the dependency manifest - which is where conventions are
  stated in prose;
- the **siblings** of every writable file, i.e. the other files in the same
  directories, which is where conventions are stated in code.

Two properties of that set are load-bearing.

**Readable is not writable.** `apply_edits` authorises writes against the
declared `## Files` set and nothing else, so a context file is exactly as
unwritable as any other path in the repository. That matters beyond this
module: the dispatcher ([#21](https://github.com/shahrestani-me/apiary/issues/21))
serialises tasks whose `## Files` sets intersect, and if reading a file implied
a claim on it, two tasks that merely read the same README could never run
concurrently. The overlap rule guards writes, and only writes.

**It is derived, not declared.** `docs/issue-contract.md` §1 fixes the body
schema at four sections, and a `## Context` section would mean growing
`swarm.github.ledger`'s parser and `LedgerEntry` - a public API in a module
this ticket may not touch. Deriving the set from the checkout needs no contract
change and no cross-module reach-around. The cost is that a task cannot point
at a distant module it happens to depend on, which is the follow-up: declare
the set in the contract, keep this derivation as the default when the section
is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..llm import structured, worker_llm
from ..state import FileEdit, WorkerOutput

SYSTEM = """You are a careful software engineer working alone on ONE task.

You are given a goal, the complete current contents of the files you may edit,
and some read-only context from the same repository so that your code matches
how the project is already written.

Return the COMPLETE new contents of every file you change - never a diff,
never a fragment, never "... rest unchanged".

Edit ONLY the files listed as editable. The read-only context is there to be
imitated, not modified: an edit to any other path is discarded and the task
fails. If a file should be created, include it with its full contents. Keep
changes minimal and focused on the goal."""

#: Per-file truncation, as in v1. A file past this is nearly always generated
#: or vendored, and spending the window on it starves the files that matter.
MAX_FILE_CHARS = 20_000

#: Total budget for the readable set. `Settings.worker_num_ctx` is 16K tokens
#: and the writable files plus the answer have to fit in it too, so the context
#: is capped in characters rather than allowed to grow with the directory.
CONTEXT_BUDGET_CHARS = 24_000

#: Root files that state a project's conventions in prose. Order is the order
#: they are offered to the model, best first.
CONTEXT_ROOT_FILES = (
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    "CONTRIBUTING.md",
    "AGENTS.md",
    "CLAUDE.md",
    "pyproject.toml",
    "package.json",
    "go.mod",
    "Cargo.toml",
)

#: Suffixes worth reading. A binary read as UTF-8 is noise that costs budget,
#: and an allow-list fails closed where a deny-list fails open.
CONTEXT_SUFFIXES = frozenset(
    {
        ".c", ".cfg", ".cpp", ".cs", ".css", ".go", ".h", ".ini", ".java", ".js",
        ".json", ".jsx", ".kt", ".md", ".mjs", ".php", ".pl", ".py", ".rb", ".rs",
        ".rst", ".sh", ".sql", ".swift", ".toml", ".ts", ".tsx", ".txt", ".yaml",
        ".yml",
    }
)

#: Directories that never teach anything: build output, dependency trees, caches.
CONTEXT_SKIP_DIRS = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build", ".mypy_cache"}
)


class EditError(RuntimeError):
    """The model never produced usable output.

    Infrastructure, not task failure: an unreachable or broken Ollama is the
    same kind of problem for every task on the host, and letting it consume the
    attempt budget would burn every issue's retries before anyone noticed
    (`docs/issue-contract.md` §4).
    """


@dataclass(frozen=True)
class SourceFile:
    """One file as the model will see it. `text is None` means it is not there yet."""

    path: str
    text: str | None
    truncated: bool = False

    def render(self, role: str) -> str:
        if self.text is None:
            return f"--- {self.path} ({role}) --- (does not exist yet)"
        suffix = "\n... [truncated]" if self.truncated else ""
        return f"--- {self.path} ({role}) ---\n{self.text}{suffix}"


@dataclass(frozen=True)
class Applied:
    """What reached the disk, and what was turned away and why.

    Refusals are kept rather than counted: they are the evidence that the guard
    rail fired, they belong in the container log #15 captures, and "the model
    tried to edit a file it was not given" is the single most useful line in a
    post-mortem of a bad task.
    """

    written: tuple[str, ...] = ()
    refused: tuple[tuple[str, str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.written)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _read(root: Path, relative: str) -> SourceFile:
    target = root / relative
    if not target.is_file():
        return SourceFile(relative, None)
    text = target.read_text(encoding="utf-8", errors="replace")
    return SourceFile(relative, text[:MAX_FILE_CHARS], truncated=len(text) > MAX_FILE_CHARS)


def read_writable(root: Path, files: Sequence[str]) -> tuple[SourceFile, ...]:
    """The declared `## Files`, in the order the contract lists them.

    A path that does not exist is still offered, as "does not exist yet": most
    tasks in this backlog create files, and silently omitting them would read
    to the model as "not part of this task".
    """
    return tuple(_read(root, relative) for relative in files)


def gather_context(
    root: Path,
    writable: Sequence[str],
    *,
    budget: int = CONTEXT_BUDGET_CHARS,
) -> tuple[SourceFile, ...]:
    """The readable set: project front matter, then the writable files' siblings.

    Deterministic - sorted, budgeted, and identical for two runs of the same
    task against the same commit. A context set that varied between attempts
    would make a flaky task impossible to tell from a flaky model.
    """
    root = root.resolve()
    claimed = {_normalise(path) for path in writable}
    candidates: list[str] = []

    for name in CONTEXT_ROOT_FILES:
        if (root / name).is_file():
            candidates.append(name)

    for directory in sorted({str(Path(_normalise(path)).parent) for path in writable}):
        folder = root / directory if directory != "." else root
        if not folder.is_dir():
            continue
        for child in sorted(folder.iterdir(), key=lambda path: path.name):
            if not child.is_file() or child.name.startswith("."):
                continue
            if child.suffix not in CONTEXT_SUFFIXES:
                continue
            relative = child.relative_to(root).as_posix()
            if any(part in CONTEXT_SKIP_DIRS for part in child.relative_to(root).parts):
                continue
            candidates.append(relative)

    gathered: list[SourceFile] = []
    spent = 0
    for relative in dict.fromkeys(candidates):
        # The writable files are shown in full elsewhere in the same prompt;
        # repeating them here would spend the budget on what the model already
        # has and blur the one distinction that matters.
        if relative in claimed:
            continue
        source = _read(root, relative)
        if source.text is None or not source.text.strip():
            continue
        if spent + len(source.text) > budget:
            continue
        gathered.append(source)
        spent += len(source.text)
    return tuple(gathered)


def build_prompt(
    goal: str,
    writable: Sequence[SourceFile],
    readable: Sequence[SourceFile] = (),
) -> str:
    """The human turn: the goal, the write set, then the read-only set.

    The write set is named twice - once as a list of paths, once as file
    bodies - because the whole failure mode this prompt guards against is the
    model helpfully "fixing" a neighbouring file it was only shown for context.
    """
    editable = ", ".join(source.path for source in writable) or "(none listed)"
    blocks = [
        f"Task goal:\n{goal}",
        f"Files you may edit: {editable}",
        "Current contents:\n" + "\n\n".join(source.render("editable") for source in writable),
    ]
    if readable:
        blocks.append(
            "Read-only context from the same repository - match its conventions, "
            "do not edit it:\n"
            + "\n\n".join(source.render("read-only context") for source in readable)
        )
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Generating
# --------------------------------------------------------------------------


def propose_edits(
    goal: str,
    writable: Sequence[SourceFile],
    readable: Sequence[SourceFile] = (),
    *,
    llm=None,
) -> WorkerOutput:
    """Ask the worker model for whole-file replacements, schema-forced.

    `llm` is the test seam - anything with `.invoke(messages)` returning a
    `WorkerOutput`. The default reaches the host's Ollama, which is why every
    test that uses it carries the `ollama` marker.
    """
    model = structured(worker_llm(), WorkerOutput) if llm is None else llm
    try:
        return model.invoke([("system", SYSTEM), ("human", build_prompt(goal, writable, readable))])
    except Exception as exc:  # noqa: BLE001 - local model failures are varied
        raise EditError(f"model call failed: {exc}") from exc


# --------------------------------------------------------------------------
# Writing, and the guard rail
# --------------------------------------------------------------------------


def _normalise(path: str) -> str:
    """`./src/thing.py` and `src\\thing.py` both mean `src/thing.py`.

    `Path.as_posix` drops the `./` and collapses repeated separators; it does
    **not** resolve `..`, which is the point - a traversal has to stay visible
    to the containment check rather than being tidied away here.
    """
    return Path(path.strip().replace("\\", "/")).as_posix()


def apply_edits(root: Path, edits: Iterable[FileEdit], allowed: Sequence[str]) -> Applied:
    """Write the edits the task is authorised to make. Refuse the rest.

    Two checks, and neither is redundant:

    **Membership in the declared set.** This is v1's guard rail, kept verbatim
    in spirit: a task may write the files its contract lists and no others.
    Matching is exact, deliberately *unlike* the dispatcher's case-insensitive
    overlap comparison (`docs/issue-contract.md` §1.3) - that one folds case so
    that two tasks on a case-insensitive filesystem are never allowed to
    collide, whereas folding case here would authorise writing `src/Thing.py`
    when `src/thing.py` was declared, which on Linux is a second file nobody
    granted.

    **Containment.** The path-traversal check, also from v1, tightened from a
    string prefix to a real path comparison after resolution: `startswith` is
    satisfied by a sibling directory whose name merely begins with the
    checkout's, and `resolve` is what catches a symlink pointing out of the
    tree. The contract parser already rejects `..` in `## Files`, but this code
    is applying paths chosen by a language model, and the guard has to hold
    when the input is hostile rather than merely wrong.
    """
    root = root.resolve()
    permitted = {_normalise(path) for path in allowed}
    written: list[str] = []
    refused: list[tuple[str, str]] = []

    for edit in edits:
        path = _normalise(edit.path)
        if not path or path == ".":
            refused.append((edit.path, "empty path"))
            continue
        if path not in permitted:
            refused.append((edit.path, "not in the declared file set"))
            continue
        target = (root / path).resolve()
        if root not in target.parents:
            refused.append((edit.path, "resolves outside the checkout"))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit.content, encoding="utf-8")
        written.append(path)

    return Applied(written=tuple(written), refused=tuple(refused))
