"""Run identity, and what "resume" means once the ledger lives on GitHub.

`docs/architecture-v2.md` promises that the orchestrator "holds no
irreplaceable state" and that `apiary.run=<id>` labels every worker container.
Neither sentence says where the id comes from, and nothing says what became of
v1's `--resume <thread-id>`. This module answers both, because #15, #20 and #29
all consume the answer and none of them may invent their own.

**A run id names one orchestrator process, not one piece of work.** The work is
identified by its issue - `docs/issue-contract.md` §2 - and the issues are the
checkpoint. A run id identifies the attempt: which containers belong to which
process, and which artifacts directory (#29) a cycle writes into.

**Decision: re-invoking the same command mints a NEW run id, and the new run
adopts the in-flight ledger.** Ids are never reused, never reserved and never
garbage collected, so `apiary.run=<id>` means exactly one process forever.

The tidier-looking alternative - derive the id from the repo so a restart
recovers the same string - breaks two consumers at once:

- The orphan reaper (#20) sweeps containers by `apiary.run=<id>` and removes
  the ones whose run is no longer live. Reuse makes the containers the killed
  orchestrator left behind indistinguishable from the ones the new process just
  spawned; it would either reap its own live workers or adopt a zombie as its
  own, and both failures look like a healthy run from the outside.
- Artifacts (#29) are one directory per run id. Two invocations sharing a
  directory interleave their logs, and the single most useful fact about a bad
  run - that it was killed at 14:31 and restarted at 14:33 - is exactly the one
  that gets flattened away.

Nothing durable is keyed by the run id, which is precisely why it is free to
change. Continuity is the ledger's job: `attach` reads GitHub, finds the issues
a previous process left mid-flight, and reports that this run continues them.
Replanning happens only when there is nothing live to continue, so re-invoking
after a crash neither replans from zero nor duplicates issues. What a human
follows across restarts is the repository and its `swarm:*` issues, and #29's
`swarm runs` groups run ids by repo for that reason.

**The id's shape is dictated by its four consumers, not by taste.** It is a
Docker label value (#15), a `docker ps --filter` argument (#20), a directory
name (#29) and something a human types into `swarm show` (#29). So:

    <repo-name>-<YYYYMMDD>-<HHMMSS>-<suffix>
    apiary-20260814-142530-k3f9qz

Lowercase and `[a-z0-9-]` only - the development host's filesystem is
case-insensitive, so two ids differing only in case would be one directory
while being two labels. UTC, so ids from either side of a DST change still sort
chronologically, and so a lexicographic sort of one repo's ids is a
chronological one. A six-character suffix from a 33-character alphabet makes
two runs of the same repo in the same second collide with probability 1/33**6;
without it, a crash-restart loop tight enough to fit inside one second would
silently share a run id, which is the one thing this whole decision forbids.

`validate_run_id` exists because the id arrives back from a human on the way to
becoming a filesystem path, and `../../etc` is a valid string.
"""

from __future__ import annotations

import datetime as dt
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .github.client import GitHubClient
from .github.ledger import Ledger, LedgerEntry, load_ledger

# The container label key. #15 writes it at spawn, #20 filters on it at reap,
# and #29 names the artifacts directory after its value; one constant so a
# rename cannot leave the three disagreeing.
RUN_LABEL = "apiary.run"

# 64 is comfortably inside every consumer's limit (a filesystem component is
# 255, a Docker label value is unbounded) and short enough to read in a
# `docker ps` column without wrapping.
MAX_RUN_ID_LENGTH = 64
SUFFIX_LENGTH = 6

# Crockford-ish: no `l`, `1`, `o` or `0`, because these ids get read off a
# terminal and typed back into `swarm show`.
SUFFIX_ALPHABET = "abcdefghijkmnopqrstuvwxyz23456789"

# `-<YYYYMMDD>-<HHMMSS>-<suffix>` is a fixed 23 characters, so the repo-derived
# prefix gets whatever is left.
_STAMP_LENGTH = len("-20260814-142530-") + SUFFIX_LENGTH
MAX_PREFIX_LENGTH = MAX_RUN_ID_LENGTH - _STAMP_LENGTH

# Anchored, lowercase, no leading or trailing hyphen. `.` is excluded outright
# rather than merely refusing `.` and `..`: a run id is also a shell word in
# `docker ps --filter`, and a dot buys nothing here.
_RUN_ID_RE = re.compile(rf"^[a-z0-9](?:[a-z0-9-]{{0,{MAX_RUN_ID_LENGTH - 2}}}[a-z0-9])?$")

# `owner/name`, the only repo shape v2 accepts. One task, one repo.
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

# Which state labels mean a run is still in flight. Straight off
# `docs/issue-contract.md` §3's label set, and read from the label rather than
# from `TaskStatus`, because the projection is lossy and §3 says the label set
# is the authority.
TERMINAL_LABELS = frozenset({"swarm:done", "swarm:failed"})

RunMode = Literal["plan", "attach"]


class RunError(RuntimeError):
    """A run could not be identified or started."""


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def repo_prefix(repo: str) -> str:
    """The human-legible half of a run id, derived from `owner/name`.

    The name only, not the owner: nearly every run of a given installation is
    under one account, so the owner is length that buys no distinction, and the
    full `owner/name` lives on `Run.repo` for anything that needs to be exact.
    """
    if not _REPO_RE.match(repo or ""):
        raise RunError(f"repo must be 'owner/name', got {repo!r}")
    name = repo.split("/", 1)[1]
    prefix = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    if len(prefix) > MAX_PREFIX_LENGTH:
        prefix = prefix[:MAX_PREFIX_LENGTH].rstrip("-")
    # A repository can legitimately be named `...` or `___`, and a prefix of
    # nothing would produce an id starting with the timestamp's hyphen.
    return prefix or "run"


def new_run_id(repo: str, *, now: dt.datetime | None = None, suffix: str | None = None) -> str:
    """Mint an id for one orchestrator process. Never reused - see the docstring.

    `now` and `suffix` are injection points for tests only; nothing in the
    system passes them, because an id a caller can choose is an id two
    processes can choose the same way.
    """
    moment = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    tail = suffix if suffix is not None else _random_suffix()
    run_id = f"{repo_prefix(repo)}-{moment:%Y%m%d}-{moment:%H%M%S}-{tail}"
    # Belt and braces: the components are all constrained above, so a failure
    # here is a bug in this function rather than bad input, and it is much
    # cheaper to catch now than as a container label GitHub-side or a directory
    # traversal in #29.
    return validate_run_id(run_id)


def _random_suffix() -> str:
    """`secrets`, not `random`, for independence from any seeded global state.

    Not a secret - it is a tiebreaker. But `random` is a shared, seedable
    module, and a test or a library that seeds it would quietly make every run
    id in the process identical.
    """
    return "".join(secrets.choice(SUFFIX_ALPHABET) for _ in range(SUFFIX_LENGTH))


def validate_run_id(value: str) -> str:
    """Return `value` if it is a usable run id, else raise `RunError`.

    Called on every id that arrives from outside - a `--run-id` flag, a
    `swarm show` argument, a label read back off a container - because the next
    thing that happens to it is being joined onto an artifacts path.
    """
    if not isinstance(value, str) or not _RUN_ID_RE.match(value):
        raise RunError(
            f"{value!r} is not a run id: expected lowercase [a-z0-9-], "
            f"1-{MAX_RUN_ID_LENGTH} characters, not starting or ending with '-'"
        )
    return value


@dataclass(frozen=True)
class Run:
    """One orchestrator process's identity, fixed at startup.

    Deliberately small and inert. Everything else about a run - which
    containers are up, which issues are claimed - is derived from Docker and
    GitHub on demand, because `docs/architecture-v2.md` makes holding it here
    the thing that stops being true the moment the process dies.
    """

    id: str
    repo: str
    objective: str
    started_at: dt.datetime

    @classmethod
    def start(
        cls,
        repo: str,
        objective: str,
        *,
        now: dt.datetime | None = None,
        run_id: str | None = None,
    ) -> Run:
        moment = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
        if not objective.strip():
            raise RunError("a run needs an objective")
        # `run_id` is for rehydrating a `Run` from a container label or an
        # artifacts directory (#20, #29), never for starting a second process
        # under an id that is already in use.
        return cls(
            id=validate_run_id(run_id) if run_id else new_run_id(repo, now=moment),
            repo=_checked_repo(repo),
            objective=objective.strip(),
            started_at=moment,
        )

    @property
    def container_labels(self) -> dict[str, str]:
        """Labels every container this run spawns carries (#15).

        `apiary.issue=<n>` is added per container by the lifecycle manager;
        this is the part that is constant for the whole run and is what #20
        filters on.
        """
        return {RUN_LABEL: self.id}

    def artifacts_dir(self, root: str | Path) -> Path:
        """This run's directory under an artifacts root (#29 owns the contents).

        The id is validated at construction and contains no separator, so the
        join cannot escape `root` - which is the reason `validate_run_id`
        refuses `.` at all rather than only refusing `..`.
        """
        return Path(root) / self.id

    def summary(self) -> str:
        return f"run {self.id}  repo {self.repo}  objective: {self.objective}"


def _checked_repo(repo: str) -> str:
    repo_prefix(repo)  # raises RunError on anything that is not `owner/name`
    return repo


# --------------------------------------------------------------------------
# Attaching to an in-flight ledger
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Attachment:
    """What a fresh run found waiting for it on GitHub.

    Holds the ledger it read rather than a summary of it, so the caller does
    not pay for a second read to do the work: `apply_readiness` takes a
    `Ledger`, and re-listing the issues to hand it one would double the
    rate-limit cost of every cycle's first act.
    """

    run: Run
    ledger: Ledger
    live: tuple[LedgerEntry, ...]

    @property
    def mode(self) -> RunMode:
        """`attach` continues the issues already there; `plan` writes new ones.

        This is the whole resume rule in one expression. A ledger with anything
        live in it is a run somebody started and did not finish, and planning
        over it would fork a second set of issues for the same objective -
        precisely the duplication `docs/issue-contract.md` §2 rejects the
        issue-number-as-identity scheme for.
        """
        return "attach" if self.live else "plan"

    @property
    def resumed(self) -> bool:
        return self.mode == "attach"

    @property
    def counts(self) -> dict[str, int]:
        """Live entries per state label, for the line a human reads on startup."""
        tally: dict[str, int] = {}
        for entry in self.live:
            tally[entry.state_label] = tally.get(entry.state_label, 0) + 1
        return tally

    def summary(self) -> str:
        if not self.resumed:
            return "no live issues; this run plans from the objective"
        detail = ", ".join(f"{count} {label}" for label, count in sorted(self.counts.items()))
        return f"attached to {len(self.live)} live issue(s): {detail}"


def live_entries(ledger: Ledger) -> tuple[LedgerEntry, ...]:
    """The entries a previous process left unfinished, in issue order.

    `swarm:done` and `swarm:failed` are terminal, so an entirely finished
    ledger is not something to attach to - a fresh objective against a repo
    whose last run completed is a new run's work, not a resumption of the old
    one's.
    """
    unfinished = (
        entry for entry in ledger.entries.values() if entry.state_label not in TERMINAL_LABELS
    )
    return tuple(sorted(unfinished, key=lambda entry: entry.number))


def attach(run: Run, ledger: Ledger) -> Attachment:
    """Decide whether this run continues an existing ledger. Pure."""
    return Attachment(run=run, ledger=ledger, live=live_entries(ledger))


def start_run(
    repo: str,
    objective: str,
    *,
    source: GitHubClient | str | None = None,
    now: dt.datetime | None = None,
    adopt: bool = True,
) -> Attachment:
    """Mint an identity, read the ledger, and report what this run continues.

    `adopt=False` for a dry run: adoption (`docs/issue-contract.md` §2) writes
    a marker onto every hand-written issue, and a command that promised to
    change nothing must not do that either.
    """
    run = Run.start(repo, objective, now=now)
    ledger = load_ledger(source if source is not None else repo, adopt=adopt)
    return attach(run, ledger)
