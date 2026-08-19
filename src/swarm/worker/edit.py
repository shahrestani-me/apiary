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

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..config import SETTINGS
from ..llm import parse_failure, structured, worker_llm
from ..state import FileEdit, WorkerOutput

SYSTEM = """You are a careful software engineer working alone on ONE task.

You are given a goal, the complete current contents of the files you may edit,
and some read-only context from the same repository so that your code matches
how the project is already written.

Return the COMPLETE new contents of every file you change - never a diff,
never a fragment, never "... rest unchanged".

Edit ONLY the files listed as editable. The read-only context is there to be
imitated, not modified: an edit to any other path is discarded and the task
fails. If a file should be created, include it with its full contents. To
DELETE a file, return it with empty content - removing an obsolete file is
often the right edit for a cleanup goal. Keep changes minimal and focused on
the goal.

Third-party packages exist for your code ONLY if they are declared in
requirements.txt, which is installed before the verify command runs (when the
operator has allowed the package index) - you may always edit requirements.txt,
listed or not. Prefer the standard library when it suffices."""

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

#: Files that are *generated* rather than written, and enormous. A lockfile
#: pins a resolved dependency graph; it teaches a model nothing that the
#: manifest beside it does not teach better, and it is routinely three orders
#: of magnitude larger.
#:
#: The cost is not hypothetical and it is not once. `MAX_FILE_CHARS` is 20,000
#: and `CONTEXT_BUDGET_CHARS` is 24,000, so **one** truncated lockfile spends
#: 83% of the readable budget - and `package-lock.json` sorts before
#: `package.json`, so it spends it first, on every task in that repository
#: forever. A measured Expo lockfile is 16,347 lines.
#:
#: Two of these leak today and the rest are pre-emptive, which is deliberate:
#: `CONTEXT_SUFFIXES` is an allow-list, so only `package-lock.json` (`.json`)
#: and `pnpm-lock.yaml` (`.yaml`) are reachable right now. `.lock` and `.sum`
#: are not readable suffixes *yet*. Naming all of them here means adding
#: `.lock` to the allow-list later cannot silently reopen this, and it puts the
#: reason in one place instead of splitting it across two constants.
#:
#: This skip is **ambient only**. A lockfile named in a task's `## Files` is
#: read in full by `read_writable`, which does not consult this set - a task
#: told to work on a file must be shown it.
CONTEXT_SKIP_FILES = frozenset(
    {
        "package-lock.json",  # npm; `.json`, so reachable today
        "npm-shrinkwrap.json",  # npm, publishable variant of the above
        "pnpm-lock.yaml",  # pnpm; `.yaml`, so reachable today
        "yarn.lock",  # yarn
        "bun.lock",  # bun, text form
        "Cargo.lock",  # cargo
        "go.sum",  # go module checksums
        "poetry.lock",  # poetry
        "uv.lock",  # uv
        "Pipfile.lock",  # pipenv
        "Gemfile.lock",  # bundler
        "composer.lock",  # composer
    }
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
    """What reached the disk, what left it, and what was turned away and why.

    Refusals are kept rather than counted: they are the evidence that the guard
    rail fired, they belong in the container log #15 captures, and "the model
    tried to edit a file it was not given" is the single most useful line in a
    post-mortem of a bad task.

    `deleted` is separate from `written` rather than folded in, because every
    downstream consumer treats the two differently: the syntax gate parses
    written files and must not look for deleted ones, the collection audit
    demands written test files be collected while a deleted test file has
    nothing left to collect, and a PR body that listed a deletion as a write
    would tell a reviewer the opposite of what happened.
    """

    written: tuple[str, ...] = ()
    refused: tuple[tuple[str, str], ...] = ()
    deleted: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.written or self.deleted)


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

    `CONTEXT_SKIP_FILES` is consulted here and only here, which is what makes
    the skip ambient: `read_writable` reads a declared `## Files` entry whatever
    its name, so a task told to work on a lockfile is still shown it.
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
            if child.name in CONTEXT_SKIP_FILES:
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


def prompt_for(
    goal: str,
    writable: Sequence[SourceFile],
    readable: Sequence[SourceFile] = (),
) -> tuple[str, str]:
    """The exact `(system, human)` pair `propose_edits` sends.

    Extracted so that `swarm console` shows the prompt production sends rather
    than a reconstruction of it. A console that assembles its own approximation
    is worse than no console: it invites an operator to conclude the model is
    fine when the fault was in the context this function built - a truncated
    lockfile eating the budget, say, so the sibling file carrying the
    convention never made it in.
    """
    return SYSTEM, build_prompt(goal, writable, readable)


# --------------------------------------------------------------------------
# Fitting the prompt to the context window
# --------------------------------------------------------------------------
#
# `CONTEXT_BUDGET_CHARS` bounds the readable set against a *typical* task, and
# a size-L task walked straight past it: twelve writable files, eighteen
# context files and a folded-in failure report add up to a prompt far larger
# than `worker_num_ctx`, and Ollama does not refuse an over-long prompt - it
# silently truncates the FRONT of it, which is where the system instructions
# are. The model then answers without ever having been told to emit the
# schema, langchain's parser rejects the junk, and the failure is classified
# as infrastructure - three of those trip the escalation, for a task whose
# real problem was arithmetic this module could have done up front.

#: The crude rule for turning characters into tokens: four characters per
#: token. Crude on purpose - the real tokenizer lives in the model and varies
#: by model, and shipping one here would pin this module to a vocabulary. Code
#: tokenises denser than prose, so /4 overestimates tokens slightly, which is
#: the safe direction for a budget check.
CHARS_PER_TOKEN = 4

#: The share of the context window reserved for the model's ANSWER. A quarter:
#: the reply replays the full contents of every file it *changes*, so the
#: honest worst case is a mirror of the whole writable set - but reserving
#: that would halve the usable window for every task to protect the rare one
#: that rewrites everything, and a task large enough to hit this reserve is a
#: task the pinned failure below already tells the planner to split.
RESPONSE_RESERVE_DIVISOR = 4

#: Tokens the chat template and the structured-output scaffolding spend around
#: the two turns this module builds: role markers, message framing, and the
#: JSON-schema grammar constraint. Small and flat, so a constant.
TEMPLATE_OVERHEAD_TOKENS = 64


def estimate_tokens(text: str) -> int:
    """~How many tokens `text` costs, by the chars/4 rule. See `CHARS_PER_TOKEN`."""
    return len(text) // CHARS_PER_TOKEN + 1


def prompt_budget(num_ctx: int) -> int:
    """The tokens the prompt may spend: the window minus the answer's reserve."""
    return num_ctx - num_ctx // RESPONSE_RESERVE_DIVISOR - TEMPLATE_OVERHEAD_TOKENS


def _prompt_tokens(
    goal: str, writable: Sequence[SourceFile], readable: Sequence[SourceFile]
) -> int:
    """The estimated cost of the exact `(system, human)` pair `propose_edits` sends."""
    system, human = prompt_for(goal, writable, readable)
    return estimate_tokens(system) + estimate_tokens(human)


def fit_context(
    goal: str,
    writable: Sequence[SourceFile],
    readable: Sequence[SourceFile],
    *,
    num_ctx: int | None = None,
) -> tuple[tuple[SourceFile, ...], str | None]:
    """Make the prompt fit `num_ctx`, or say honestly that it cannot.

    Returns `(readable, None)` when the prompt fits as offered, a trimmed
    readable set when dropping context files makes it fit, and `((), failure)`
    when the goal and the writable set alone overflow the window - the one
    case no amount of trimming can fix, because a task must be shown the files
    it is told to edit (`read_writable`'s rule).

    The failure is a returned string, not an exception, in this module's usual
    shape (`syntax_failure`, `install_dependencies`): it is an *outcome* - the
    task as planned does not fit this worker - and its first line is pinned
    because `orchestrator.reconcile.diagnose` matches it, turning "three
    parse errors that looked like infrastructure" into "split the task".

    Context files are dropped whole, least useful first. The keep order is:
    files sharing a directory with a writable file first (they are the
    conventions the edit must match - the reason `gather_context` collects
    siblings at all), then shorter files first (a short neighbour teaches the
    house style at a fraction of the cost of a long one), path as the
    deterministic tie-break. Survivors keep `gather_context`'s original
    presentation order. Every trim is printed to stdout, so the container log
    tells the operator what the model was *not* shown.

    `num_ctx` defaults to `SETTINGS.worker_num_ctx` - the same value
    `worker_llm()` passes to Ollama, so the budget and the window cannot
    drift apart. One divergence, stated: `swarm console` renders `prompt_for`
    without this fitting pass, so on an over-budget task the console shows
    the prompt as built, not as trimmed.
    """
    num_ctx = SETTINGS.worker_num_ctx if num_ctx is None else num_ctx
    budget = prompt_budget(num_ctx)

    base = _prompt_tokens(goal, writable, ())
    if base > budget:
        return (), (
            f"the task is too large for the worker's context window (~{base} tokens "
            f"against a budget of {budget}; SWARM_WORKER_CTX={num_ctx}), and the plan "
            "should split it into tasks with smaller file sets. The goal and the "
            "declared files alone overflow the window before any read-only context "
            "is added, so a retry with the same file set will overflow identically."
        )

    total = _prompt_tokens(goal, writable, readable)
    if total <= budget:
        return tuple(readable), None

    directories = {Path(_normalise(source.path)).parent for source in writable}

    def keep_rank(source: SourceFile) -> tuple[int, int, str]:
        sibling = Path(_normalise(source.path)).parent in directories
        return (0 if sibling else 1, len(source.text or ""), source.path)

    ranked = sorted(readable, key=keep_rank)
    dropped = 0
    # Token cost is order-independent (the rendered blocks are joined, not
    # nested), so counting against `ranked` answers for the prompt that will
    # actually be sent in the original order.
    while ranked and _prompt_tokens(goal, writable, ranked) > budget:
        ranked.pop()
        dropped += 1
    kept_paths = {source.path for source in ranked}
    print(f"  · context trimmed: dropped {dropped} file(s) (~{total - budget} tokens over budget)")
    return tuple(source for source in readable if source.path in kept_paths), None


#: The one line appended for the parse retry. A statement of the constraint,
#: not a paraphrase of the schema - the schema is already enforced by Ollama's
#: grammar-constrained decoding; what the retry buys is a second decode.
RETRY_INSTRUCTION = (
    "your previous reply was not valid JSON for the schema; "
    "reply with ONLY the JSON object"
)


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

    A reply the parser rejects is retried ONCE, with `RETRY_INSTRUCTION`
    appended as a further human turn, before it escapes as `EditError`. Once
    and never a loop: the schema is grammar-enforced at the decoder, so a
    parse failure means the model is emitting junk under a format constraint -
    a truncated prompt, a broken runner - and a model that does it twice is
    broken in a way more turns will not fix, while each extra turn costs a
    whole-file generation of wall-clock time. Only the parse failure is
    retried; a refused socket or a missing model fails the same on any number
    of tries and escapes immediately, as before.
    """
    model = structured(worker_llm(), WorkerOutput) if llm is None else llm
    system, human = prompt_for(goal, writable, readable)
    messages = [("system", system), ("human", human)]
    try:
        return model.invoke(messages)
    except Exception as exc:  # noqa: BLE001 - local model failures are varied
        if not parse_failure(exc):
            # The type as well as the message. `str(exc)` alone turns a refused
            # socket, a missing model and a schema the server rejected into
            # three sentences that read the same in a result file.
            raise EditError(f"model call failed: {type(exc).__name__}: {exc}") from exc
        print(f"  · the reply was not valid JSON ({type(exc).__name__}); retrying once")
        try:
            return model.invoke([*messages, ("human", RETRY_INSTRUCTION)])
        except Exception as retry_exc:  # noqa: BLE001 - same varied failures
            raise EditError(
                f"model call failed: {type(retry_exc).__name__}: {retry_exc}"
            ) from retry_exc


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

    **Empty content means delete.** An edit whose content is empty - or
    whitespace-only, since a file of two spaces is never what anyone intended -
    removes the file instead of writing it. This overloads the one schema the
    model already emits rather than adding a `deleted: bool` field to
    `FileEdit`, and the trade was weighed rather than assumed: a new field
    changes the structured-decoding schema for every model call, needs every
    old record and prompt to be reconsidered, and buys expressiveness for
    exactly one case - an intentionally empty source file - that this tool has
    no business producing anyway. Nothing in this codebase treats an empty
    write as meaningful (an empty generated file was, before this, simply a
    smell that slipped through), and a task that truly needs an empty file has
    an honest workaround: a single comment line. What the overload buys is the
    thing three live attempts proved impossible: a cleanup task can now
    actually remove the files its goal names, instead of emptying them and
    failing the collection audit for it.

    A deletion is held to the same two checks as a write - the model may only
    delete what it was allowed to edit - and deleting a file that is not there
    is a refusal, not a no-op: a model asking to remove a file that does not
    exist is confused about the tree it was shown, and that confusion belongs
    in the log next to the other refusals. After a deletion, parent directories
    left empty are pruned up to (never including) the checkout root, because a
    commit cannot express an empty directory and leaving one behind would make
    the working tree disagree with what the PR says happened.

    Edits are applied in order and the ledger reports *net effect* per path: a
    file written and then deleted in the same output ends deleted if it existed
    before this batch, and ends as nothing at all if the batch itself created
    it - reporting a path both written and deleted would hand `commit_edits` a
    path `git add` may find nothing to stage for.

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
    deleted: list[str] = []
    #: Whether each path existed before this batch first touched it - recorded
    #: at first touch, because it is what decides whether a later deletion is a
    #: real deletion or merely undoes this batch's own write.
    before: dict[str, bool] = {}

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

        if not edit.content.strip():
            # The delete branch. Whitespace-only counts: no file of two spaces
            # was ever intended, and treating it as a write would let a model's
            # stray newline quietly produce the emptied-not-removed file this
            # feature exists to end.
            if path not in before:
                before[path] = target.is_file()
            if not target.is_file():
                refused.append((edit.path, "deletes a file that does not exist"))
                continue
            target.unlink()
            if path in written:
                written.remove(path)
            if path in deleted:
                deleted.remove(path)
            if before[path]:
                deleted.append(path)
            _prune_empty_dirs(root, target.parent)
            continue

        if path not in before:
            before[path] = target.is_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit.content, encoding="utf-8")
        if path in deleted:
            deleted.remove(path)
        if path not in written:
            written.append(path)

    return Applied(written=tuple(written), refused=tuple(refused), deleted=tuple(deleted))


def _prune_empty_dirs(root: Path, directory: Path) -> None:
    """Remove directories a deletion emptied, up to and never including `root`.

    git cannot commit an empty directory, so one left behind exists only in the
    working tree - invisible to the PR, and a lie the next reader of the tree
    has to notice. Walking stops at the first non-empty parent (an occupied
    directory is somebody else's), at anything that is not a real directory,
    and always before the checkout root itself, which must survive even a task
    that deleted the last file in the repository.
    """
    while directory != root and root in directory.parents:
        if not directory.is_dir() or directory.is_symlink() or any(directory.iterdir()):
            return
        directory.rmdir()
        directory = directory.parent


def syntax_failure(root: Path, written: Sequence[str]) -> str | None:
    """Do this attempt's written Python files at least *parse*? Text if not.

    The verify command was supposed to make this question redundant, and a
    real generated repository proved it does not: a merged test file carried a
    literal model thought-leak (`amount=3.5 far, ... # wait, typo`) and a
    full-width Unicode full stop - both SyntaxErrors, both green through the
    gate, because the repo's pytest `testpaths` never collected the file. A
    test suite only parses the files it runs, so "the gate passed" and "the
    files parse" are independent claims, and this is the cheap one checked
    first.

    `ast.parse` rather than `py_compile`: no subprocess, no `.pyc` dropped
    into a tree whose commit stages exactly the declared paths, and the raised
    `SyntaxError` carries the filename, line and offending text this function
    exists to report. The check is *only* a parse - it proves nothing about
    imports or behaviour, which stay the verify command's job.

    Scoped to what this attempt wrote, `.py` files only: non-Python files and
    non-Python stacks pass through untouched, and pre-existing debt elsewhere
    in the repository is not this task's to answer for. Every broken file is
    reported, not just the first, because the text becomes the retry comment
    and a retry told about one error per attempt burns the budget one line at
    a time.

    Returns `None` when everything parses, and the failure text otherwise -
    an outcome for the caller to fold into the run as a failed gate, in the
    module's usual shape (`install_dependencies` returns the same way, for the
    same reason: the text is the next attempt's feedback, not an exception).
    The first line of each finding is pinned - `python syntax error in <path>,
    line <n>: <msg>` - because `orchestrator.reconcile.diagnose` matches it.
    """
    failures: list[str] = []
    for relative in written:
        if not relative.endswith(".py"):
            continue
        target = root / relative
        if not target.is_file():
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(text, filename=relative)
        except SyntaxError as exc:
            quoted = (exc.text or "").strip()
            shown = f"\n    {quoted}" if quoted else ""
            failures.append(
                f"python syntax error in {relative}, line {exc.lineno}: {exc.msg}{shown}"
            )
        except ValueError as exc:
            # `ast.parse` refuses a NUL byte with ValueError rather than
            # SyntaxError. Same verdict: the file cannot be parsed.
            failures.append(f"python syntax error in {relative}: {exc}")
    if not failures:
        return None
    return (
        "\n".join(failures)
        + "\n\nThe verify command was not run: a file that does not parse fails "
        "every test that imports it, and proves nothing in a suite that never "
        "collects it. Rewrite the file so the quoted line parses."
    )
