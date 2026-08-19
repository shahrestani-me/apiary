# The issue contract

Status: **specification.** Every v2 module that reads or writes the ledger reads
this document; none of them may invent its own answer.

[`docs/architecture-v2.md`](architecture-v2.md) says GitHub is the database. That
only works if there is exactly one definition of what an issue *means*. This
document is that definition: how a body is parsed, what identifies a task, how
v1's `TaskStatus` relates to v2's labels, who is allowed to move an issue between
states, and where the attempt counter lives.

Where this document and the architecture doc disagree, this one wins — the
architecture doc is design intent, this is the contract the code implements. One
such disagreement exists and is called out in [§5](#5-the-attempt-counter).

---

## 1. The body schema

The rendered form, which is also what the planner emits:

```markdown
<!-- apiary:task id=add-retry-logic attempt=0 -->

## Goal
One sentence: what must be true when this is done.

## Files
- src/thing.py
- tests/test_thing.py

## Verify
python -m pytest -q tests/test_thing.py

## Blocked by
- #12
```

Those four sections are required. A fifth, `## Stack`, is optional and is
written **last**, after `## Blocked by`:

```markdown
## Stack
node
```

The marker comment is required on issues the planner writes and is adopted onto
issues it did not — see [§2](#2-task-identity). Order is not significant to the
parser, which locates sections rather than walking them in sequence — with one
exception, and it is the reason `## Stack` goes last.

**Backward compatibility.** [§1.1](#11-what-counts-as-a-section-heading) makes
any unrecognised ATX heading a section *terminator* whose content is dropped,
and any later recognised heading opens a new section. So an orchestrator that
has never heard of `## Stack` reads the four required sections **identically**,
wherever the fifth one sits — it sees the trailing heading, ends the section
before it, and discards what follows. That is what lets an issue carrying
`## Stack` be read by an older orchestrator, and it is pinned for three
placements by corpus rows in [§7](#7-test-corpus).

**Why last, then.** Not compatibility — readability. It keeps the four
canonical sections contiguous and in the order [§6](#6-canonical-example)
documents, so the optional section reads as an appendix rather than as an
interruption of the contract.

### 1.1 What counts as a section heading

This is the single most important paragraph in the document, because the obvious
implementation is wrong and fails silently.

A section heading is **a whole line that matches `## <Name>` exactly** — no
leading whitespace, no trailing content, only optional trailing spaces:

```
^## (Goal|Files|Verify|Blocked by)[ \t]*$
```

and which is **not inside a fenced code block**. Both halves are load-bearing:

- **Line-anchored, not substring.** Issue #11's `## Goal` sentence contains the
  literal string `` `## Blocked by` `` inside a code span. A parser that sections
  on the first *occurrence* of the heading text — `body.find("## Blocked by")`,
  or an unanchored `re.split` — cuts that issue at the wrong offset and reports
  it as having no dependencies. It looked ready when it was blocked. Anchoring to
  line boundaries fixes this for free, because a line containing a code span
  starts with a backtick and cannot match. This exact issue body belongs in the
  test corpus ([§7](#7-test-corpus)).
- **Fence-aware.** The line anchor alone is not enough. `docs/architecture-v2.md`
  contains a fenced block whose contents are `## Goal`, `## Files`, `## Verify`
  and `## Blocked by` at column 0, and any issue that quotes the schema — a
  planner explaining itself, a human writing a meta-issue — carries the same
  shape. Those lines match the anchor and must still not open a section. The
  parser tracks ` ``` ` and `~~~` fences, respecting fence length and info
  strings, and treats every line between them as opaque text.

Code *spans* need no special handling beyond the line anchor: a section heading
occupies a whole line, and a whole line wrapped in backticks does not match it.
Fences do need handling. Implementations must not conflate the two problems.

### 1.2 Section extent

A section's content runs from the line after its heading to the first of:

- any subsequent unfenced ATX heading, `#` through `######` — not just the four
  known ones, so an unrecognised `## Notes` ends the previous section rather than
  being swallowed into it;
- an unfenced thematic break on its own line (`---`, `***`, `___`);
- the end of the body.

Everything outside the four known sections is **tolerated and ignored**. The v2
backlog has a `> Design: …` blockquote before `## Goal` and several paragraphs of
rationale after the thematic break; both are prose for humans and neither is
contract. Tolerating them is not laxity — it is what lets a human write a real
issue instead of filling in a form.

Unrecognised headings are ignored. A **repeated** known heading is malformed: two
`## Verify` sections mean two candidate commands and no rule for choosing.

### 1.3 Per-section rules

| Section | Content | Parsed to | Malformed when |
|---|---|---|---|
| `## Goal` | free prose, newlines collapsed to single spaces | `TaskRecord.goal: str` | empty |
| `## Files` | markdown list, one repo-relative path per item | `TaskRecord.files: list[str]` | empty; any path absolute, containing `..`, or containing a glob metacharacter |
| `## Verify` | one shell command, bare or in a fence | `str` on the task's contract record | empty; more than one non-empty command line |
| `## Blocked by` | markdown list of `- #N`, or no list at all | `list[int]` of issue numbers | a `#N` appears outside a list item; a ref is cross-repo (`owner/repo#N`) |
| `## Stack` *(optional)* | one stack id: `python`, `node` or `react` | `str \| None` on the contract; resolved to `python` on the ledger entry | present but empty; more than one id; an id outside the known set |

Notes on each:

**Files.** A path the *verify command* produces — a lockfile — must **not** be
listed here. Those are the **generated set**, described below, and an overlap
is a `ContractError`. Surrounding backticks are stripped, `./` prefixes removed. Globs are
rejected rather than expanded: the dispatcher decides concurrency by intersecting
`## Files` sets ([#21](https://github.com/shahrestani-me/apiary/issues/21)), and a
glob has no set semantics without a filesystem to resolve it against — the wrong
place to discover that two tasks overlap is inside a running container. Overlap
comparison is **case-insensitive**, because the development host's filesystem is,
and `src/Thing.py` and `src/thing.py` are one file there.

**Verify.** Both a bare line and a fenced single line are accepted; the fence is
stripped and the result is one string. Multi-line is rejected because "run these
in order, stopping on failure" is a semantics nobody agreed to — write `&&`. The
command runs with `shell=True` from the repository root inside the worker
container, and, exactly as in v1's
[`verifier.py`](../src/swarm/nodes/verifier.py), **only its exit code is
believed.**

**Blocked by.** A section with no list items means no dependencies; the backlog's
`_none — this is the root of the dependency graph._` is that case and parses to
an empty list. But a `#N` in a non-list line inside the section is malformed
rather than ignored, because that is precisely the shape of a dependency that
gets silently dropped. Cross-repo refs are an error, not a satisfied dependency
([#11](https://github.com/shahrestani-me/apiary/issues/11)); one task, one repo.

### 1.4 Malformed bodies reject loudly

A parse failure raises `ContractError` naming the issue number, the section and
the reason. The parser **never returns a partial record** — no defaulting a
missing `## Verify` to the repo-wide command, no treating an unparseable
`## Blocked by` as empty. A silently mis-parsed contract is worse than a failed
one: it produces a task that runs, passes the wrong gate, and merges.

The loader ([#9](https://github.com/shahrestani-me/apiary/issues/9)) raises;
policy lives in the orchestrator, and the policy is: label the offending issue
`swarm:failed`, post the `ContractError` text as an issue comment, continue the
cycle with the remaining issues. One bad hand-written issue must not stop a run,
but it must never be dispatched either.

Issues carrying no `swarm:*` label are not parsed at all. Humans use the tracker
too — this repository's own backlog is the example — and an unlabelled issue is
simply not part of the ledger.

---

## 2. Task identity

**Decision: identity is a slug in an HTML comment marker at the top of the body.
The issue number is a handle, not the key.**

```
<!-- apiary:task id=add-retry-logic attempt=0 -->
```

The id matches `^[a-z0-9]+(-[a-z0-9]+)*$`, at most 64 characters — the same shape
`PlannedTask.id` already has in [`state.py`](../src/swarm/state.py).

### Why not the issue number

Because two tickets asserted incompatible things and the combination duplicates
the ledger on every replan.
[#9](https://github.com/shahrestani-me/apiary/issues/9) originally said the issue
number *is* the task id;
[#10](https://github.com/shahrestani-me/apiary/issues/10) requires replanning to
match tasks by id so it updates issues instead of opening new ones. A replan
re-invokes the planner, which emits `PlannedTask.id` — a model-generated
kebab-case slug that can never equal an integer. Matching would find nothing
every time, and every stall would fork a complete second set of issues.

### Why a marker and not a title prefix or a body section

A title prefix is user-visible and users edit titles. The architecture doc names
"a human can retitle, re-scope, close or reprioritize a task mid-run" as a
*feature* of putting the ledger on GitHub; identity that lives in the title turns
that feature into ledger corruption.

A `## Task id` section would be a fifth required heading in a contract that is
also meant to be hand-writable, and it renders as noise inviting exactly the edit
that breaks it. The HTML comment is invisible in rendered markdown, survives
retitles, relabels and body edits below it, and is preserved verbatim by GitHub's
editor.

### Authority, and adoption

- The **marker id is authoritative for identity** — it is the ledger key, the
  thing replanning matches on, the thing that survives.
- The **issue number is authoritative for addressing** — API calls, `## Blocked
  by` refs, container labels (`apiary.issue=<n>`). Refs never use the slug: a
  slug is not resolvable by GitHub and would not close an issue or cross-link.
- **Branch names carry the task ref and the attempt**, not the number:
  `apiary/<ref>-attempt-<n>` (`src/swarm/github/branches.py`, #144). The ref is
  percent-encoded into something git accepts and reads back out losslessly, so
  an orchestrator that lost its memory can reconstruct what was in flight from
  the code host alone. GitHub's `#42` encodes as `%2342`; a tracker whose ids
  are already git-safe keeps them verbatim.
- They cannot "disagree" about the same fact because they answer different
  questions. What can go wrong is **two issues carrying the same id**, and that
  is control-plane corruption: the cycle aborts with an error naming both issue
  numbers. Picking one silently risks two containers editing the same files.

An issue that has a `swarm:*` label but no marker was written by a human. The
loader **adopts** it: it derives an id by slugifying the title, appends the
marker to the body, and from that moment the id is stable. Collisions get the
issue number appended. Adoption is a one-time write of one invisible line, which
is what makes this repository's own backlog usable as a live fixture. **A marker
is never rewritten after it exists**, and an id is never reassigned to different
work.

---

## 3. Status mapping, both directions

v1 `TaskStatus` is `pending | running | verified | failed | abandoned`
([`state.py`](../src/swarm/state.py)). The v2 state labels are `swarm:ready |
blocked | claimed | review | done | failed`. This is not a bijection in either
direction, and pretending otherwise breaks the judge.

### Labels → `TaskStatus` (what the loader computes)

| Label | `TaskStatus` | Why |
|---|---|---|
| `swarm:ready` | `pending` | planned, not started |
| `swarm:blocked` | `pending` | also planned and not started; the difference is derived from `## Blocked by`, not stored |
| `swarm:claimed` | `running` | a container holds it |
| `swarm:review` | `running` | **decision below** |
| `swarm:done` | `verified` | PR merged; the work is in the base branch |
| `swarm:failed` | `abandoned` | attempts exhausted, a human is needed |

**`swarm:review` maps to `running`, not `verified`.** The tempting reading is
that the worker ran the verify command and passed, so the task is verified. But
v1's `verified` fed `verified_count`, which the judge uses to decide the
objective is satisfied, and v1 integrated a verified task immediately. In v2 the
integration step is the merge, and a PR in review can still fail CI, be closed
unmerged, or need a rebase. Mapping `review` to `verified` would let the judge
declare a run finished while its output sits in open PRs. Completion is the
merge.

**`swarm:failed` maps to `abandoned`, not `failed`.** v2's `failed` label means
"attempts exhausted, needs a human" — v1's `abandoned` exactly. v1's `failed`
meant "this attempt failed, another is available", and **v2 has no label for
it**: a retryable failure is persisted as `swarm:ready` with an incremented
attempt counter. That state is transient and never sits in the ledger between
cycles.

### `TaskStatus` → labels (what the writers apply)

| `TaskStatus` | Label | Condition |
|---|---|---|
| `pending` | `swarm:blocked` | any `## Blocked by` ref not closed-as-completed |
| `pending` | `swarm:ready` | otherwise |
| `running` | `swarm:claimed` | dispatched, no PR yet |
| `running` | `swarm:review` | PR open |
| `verified` | `swarm:done` | PR merged |
| `failed` | `swarm:ready` | attempt counter incremented; no dedicated label |
| `abandoned` | `swarm:failed` | |

### The rule that follows

Both directions lose information: `ready`/`blocked` collapse into one status, and
`claimed`/`review` collapse into another. Therefore:

**The label set is authoritative. `TaskStatus` is a lossy projection maintained
for v1 code, and no v2 component may make a decision that depends on information
the projection dropped.** A component that needs to know whether a PR is open
reads `swarm:review`, never `status == "running"`.

### Exactly one state label

An issue carries exactly one of the six state labels. Zero or two is a fault, and
the most likely cause is a human relabelling mid-run — which
[#22](https://github.com/shahrestani-me/apiary/issues/22) requires the system to
survive. The reconciler repairs it rather than crashing, by furthest-along-wins
precedence:

```
done  >  failed  >  review  >  claimed  >  blocked  >  ready
```

The losing labels are removed, the repair is logged and commented on the issue.
The precedence direction is deliberate: a human adding `swarm:done` or
`swarm:failed` to a claimed issue means "stop", and stopping is the safe reading.
An issue with **no** state label is treated as outside the ledger — see
[§1.4](#14-malformed-bodies-reject-loudly) — not as ready.

Routing labels (`area/*`, `size/*`) are orthogonal, planner-assigned, and never
read by the state machine.

---

## 4. The label state machine

Every legal transition, and the one component permitted to make it. Without the
writer column the dispatcher, the worker, the reconciler and a human all write
labels and fight.

| From | To | Trigger | Writer |
|---|---|---|---|
| — | `ready` | issue created with all deps met | planner (#10) |
| — | `blocked` | issue created with an unmet dep | planner (#10) |
| `blocked` | `ready` | every `## Blocked by` ref closed as completed | readiness (#11) |
| `ready` | `blocked` | a dep reopened, or a replan added one | readiness (#11) |
| `ready` | `claimed` | picked for dispatch, **before** the container is spawned | dispatcher (#21) |
| `claimed` | `review` | branch pushed and PR opened (worker exit 0) | worker (#17) |
| `claimed` | `ready` | worker exit 1, attempts remain | reconciler (#22) |
| `claimed` | `failed` | worker exit 1, attempts exhausted | reconciler (#22) |
| `claimed` | `ready` | worker exit 2 (infrastructure) — attempt **not** consumed | reconciler (#22) |
| `claimed` | `ready` | stale claim, no live container at startup — attempt consumed | recovery (#35) |
| `claimed` | `review` | stale claim but an open PR exists (worker died after pushing) | recovery (#35) |
| `review` | `done` | PR merged | reconciler (#23) |
| `review` | `ready` | checks failed with attempts remaining, or PR closed unmerged | reconciler (#23) |
| `review` | `failed` | checks failed, attempts exhausted | reconciler (#23) |
| any | `failed` | malformed contract, or a human giving up | reconciler, human |
| any | `ready` | human reset, attempt counter cleared by hand | human |
| `done` | — | terminal within a run; a reopened issue is new work with a new id | nobody |

Rules the table does not show:

- **The worker writes exactly one label, `swarm:review`, and never touches any
  other.** It never writes the attempt counter and never marks itself done. The
  container runs LLM-generated code with a push token
  ([architecture-v2 §3](architecture-v2.md)); every label it does not need is
  scope it does not get.
- The worker writing that label at all is a deliberate trade: it knows the PR
  exists at the instant it exists, and having the orchestrator discover it by
  polling widens the window where a finished task looks claimed. The cost is that
  a worker crashing between `git push` and the label leaves a claimed issue with
  an open PR, which is why the recovery row above exists.
- `ready → claimed` happens **before** the spawn. A crash in that window leaves a
  claim with no container, recoverable by #35. The reverse order loses issues
  outright.
- Infrastructure failures (exit 2) do not consume the attempt budget. A broken
  Ollama would otherwise burn every task's budget before anyone noticed
  ([#18](https://github.com/shahrestani-me/apiary/issues/18)); repeated infra
  failures halt the run instead.
- A human may make any transition. The system's response to a human edit is to
  reconcile, never to overwrite: GitHub wins.

---

## 5. The attempt counter

**Decision: the counter is a field in the identity marker, not a label.**
`swarm:attempt/1..3` is removed from the protocol.

```
<!-- apiary:task id=add-retry-logic attempt=2 -->
```

This supersedes the `swarm:attempt/1..3` row in architecture-v2's label table.
[#8](https://github.com/shahrestani-me/apiary/issues/8) must therefore **not**
create attempt labels, and the awkward collision it anticipated — "create on
demand" versus "leave existing labels alone" — disappears rather than needing a
namespace carve-out.

### Why not a label

Changing `swarm:attempt/1` to `swarm:attempt/2` is two API calls with no
transaction around them, and both crash windows are bad: crash after the removal
and the issue reads as attempt 0, so it retries forever; crash after the addition
and it carries two counters with no rule for reading them. A counter whose whole
purpose is to bound retries must not have a failure mode that unbounds them.

The marker field is a single `PATCH` of the issue body: it either applied or it
did not, and both outcomes are legible. It also costs no extra API call to read,
which matters for [#22](https://github.com/shahrestani-me/apiary/issues/22)'s
budget — the body is already in the response the loader fetched.

### The write rule

- Only the reconciler and recovery write the counter. The worker never does.
- The write **re-reads the body immediately before patching and preserves every
  byte outside the marker line.** A human editing prose while the orchestrator
  bumps a counter must not lose their edit; GitHub's last-write-wins gives no
  help here.
- **The increment is persisted before the re-dispatch**, i.e. before
  `swarm:claimed` is applied. A crash between the two costs an attempt rather
  than granting a free one.
- That ordering, plus #35 bumping on stale-claim recovery, means the counter can
  over-count by one. This is deliberate: **the counter is an upper bound on
  attempts made, never a lower bound.** Over-counting gives up early and puts a
  human in front of the problem; under-counting loops forever while looking
  healthy.
- A missing or unparseable `attempt=` field reads as `0` — the adoption case in
  [§2](#2-task-identity), where a human wrote the issue by hand.
- The cap is `Settings.max_attempts_per_task` ([`config.py`](../src/swarm/config.py)),
  unchanged from v1 — but it bounds *one blocker*, not the task; see below.

### The failure signature: the budget is per blocker

**The signature is not in the body. It is in apiary's own store**
([ADR 0002](adr/0002-apiary-owns-a-thin-task-store.md), #159). The marker
carried it for two days and that was the wrong place: a failure signature is
apiary's judgment about its own execution, not a fact the customer's tracker
owns, and [ADR 0001](adr/0001-task-systems-are-integrations.md) exists to stop
apiary writing its vocabulary into somebody else's issues. `render_marker`
therefore emits `id=` and `attempt=` and nothing else. A body written by an
older build still carries `blocker=` and `streak=`; the parser still reads
them, because a build that stopped would answer "no previous blocker" for every
task in an upgraded repository at once and hand each of them a fresh budget,
and they leave the body the first time anything rewrites the marker.

What the store holds per task is a task ref, the attempt the judgment was taken
at, a short deterministic signature of the failure the last consumed attempt
died on (`reconcile.signature`: the diagnosis when one was recognised,
otherwise the normalised exception line — paths, line numbers and addresses
stripped), how many consecutive attempts have failed with that signature, and
how many times the budget has been renewed. It holds nothing the tracker owns,
which is why there is nothing to reconcile between the two.

The arithmetic is unchanged. The give-up test runs on the **streak**: the same
failure repeating burns the budget down as it always did, while a *different*
failure than the last recorded one is proof the previous blocker is gone, so
the streak restarts at 1 and the retry is granted even when `attempt` has
reached `max_attempts_per_task`. Renewal is bounded by
`Settings.max_total_attempts_per_task` (`SWARM_MAX_TOTAL_ATTEMPTS`, default
three full per-blocker budgets) on the monotonic `attempt` itself, so a task
that keeps failing in new ways still ends.

The write rule above extends across the split rather than being weakened by it:
the judgment is recorded, then the counter is patched, then the label goes back
to `swarm:ready`. The two were never one transaction — the counter and the
label were already two calls — and the guarantee comes from the order, so a
crash anywhere in the sequence costs an attempt with its signature recorded
rather than granting a retry that forgot what it was retrying. A task the store
has never judged reads as "no previous blocker recorded" and behaves exactly as
this section always specified; a writer that consumes an attempt with nothing
to sign (a stale claim, a failed check run) records the judgment without a
signature, which falls back to the same arithmetic.

If the store's stamped attempt and the marker's counter disagree, somebody
edited the counter — a human resetting it after fixing the environment is the
documented case — and the tracker wins, exactly as it wins everywhere else. The
counter stands as the body has it and the signature is dropped as a statement
about an attempt that no longer exists, which is the safe direction: it can
only give up early.

The **counter itself stays in the marker**, and that is not an oversight. The
worker reads it: a worker is a container with no socket and no view of the
host, it derives its branch name and its result filename from that number, and
it has no path to apiary's store. One authority per fact — the tracker owns the
counter, the store owns the judgment about it, and neither holds a copy of the
other's.

---

## 6. Canonical example

A complete issue as the planner writes it, exercising every rule:

```markdown
<!-- apiary:task id=add-retry-logic attempt=0 -->

> Design: docs/architecture-v2.md.

## Goal
`http_client.get` retries idempotent requests three times with backoff.

## Files
- src/swarm/github/client.py
- tests/test_client_retry.py

## Verify
python -m pytest -q tests/test_client_retry.py

## Blocked by
- #7

## Stack
python

---

Anything below the thematic break is prose for humans and is not parsed.
```

`## Stack` is last, and on a Python task the planner omits it entirely — it is
shown here because this example exercises every rule. Labels on that issue: one
state label, plus `area/control-plane` and `size/S`.

---

## 6b. Generated files

A third category, between "the task's files" and "everything else": paths the
**verify command produces** that the worker commits if they appear.

It is **not part of the issue body.** It is a per-stack constant,
`swarm.github.ledger.GENERATED_FILES`, keyed by the `## Stack` above:

| Stack | Generated |
|---|---|
| `python` | — |
| `node` | `package-lock.json` |
| `react` | `package-lock.json` |

### Why this is not just a wider `## Files`

**A lockfile cannot be declared, because it cannot be written.** The worker's
edit protocol demands "the COMPLETE new contents of every file you change —
never a diff", and a measured Expo lockfile is 16,347 lines: roughly 180k
tokens against a 16,384-token window. It also carries SHA-512 integrity hashes,
which cannot be produced by generation at any context size. A task that
declared one would burn its attempts on a truncated file.

**And the staging rule must not simply be loosened.** `commit_edits` stages
exactly the declared paths, which is what keeps `node_modules`, caches and
everything else a verify command drops in the tree out of the pull request.
`git add -A` after a verify run is precisely the change this category exists to
avoid making.

So the set is *named by the system, per stack*, rather than declared per task.
That also means model output cannot widen what gets committed — which is the
one property the staging rule exists to guarantee.

### Rules

- Absent is normal. Most tasks add no dependency; a generated path the gate did
  not produce is simply not committed, and the task succeeds.
- The two sets are **disjoint**. A `## Files` entry naming a generated path is a
  `ContractError` on `## Files`, naming the path and saying it will be committed
  if it appears.
- Staging happens **after** the gate runs, because that is the only moment the
  file exists.
- A path that resolves outside the repository — a symlink — is skipped.

Without this, a PR that adds a dependency carries the manifest and no lock, CI
re-runs the command on neutral ground, and `npm ci` fails. "Add a dependency"
is unimplementable.

**Note, since #106: the generated stacks do not exercise this yet.** A worker
has no route to a registry, so nothing a generated project's gate runs can
*produce* a lockfile in the first place — React's toolchain arrives in the
image and the generated workflow uses `npm install`, never `npm ci`. So both
JS rows above are a permission held open for a repository that brings its own
installing gate, not a description of what a bootstrapped repo commits today.
See `greenfield/stacks.py` and `docs/security.md` §3.

---

## 7. Test corpus

Shared fixtures live with
[#31](https://github.com/shahrestani-me/apiary/issues/31); these cases are
mandatory for [#9](https://github.com/shahrestani-me/apiary/issues/9) and
[#11](https://github.com/shahrestani-me/apiary/issues/11).

| Case | Expected |
|---|---|
| Issue #11's real body — `` `## Blocked by` `` inside a code span in the `## Goal` sentence | one dependency parsed, not zero |
| A body quoting the schema inside a fence, with `## Goal` at column 0 | the fenced headings open no section |
| `## Blocked by` containing only `_none — …_` | empty dependency list |
| `#12` on a non-list line in `## Blocked by` | `ContractError` |
| Cross-repo ref `owner/repo#12` | `ContractError` |
| Missing `## Verify` | `ContractError`, no partial record |
| Two `## Verify` sections | `ContractError` |
| Fenced single-line `## Verify` | same string as the bare form |
| A path with a glob, or `..`, in `## Files` | `ContractError` |
| Prose before `## Goal` and after `---` | ignored, parse succeeds |
| An unrecognised `## Notes` heading between sections | ends the previous section, content ignored |
| Issue with no `swarm:*` label | not in the ledger |
| Issue with a `swarm:*` label and no marker | adopted, id slugified from the title |
| Two issues with the same marker id | cycle aborts, both numbers named |
| Issue carrying both `swarm:claimed` and `swarm:done` | repaired to `done`, logged |
| `swarm:review` issue | `TaskStatus == "running"` |
| `swarm:failed` issue | `TaskStatus == "abandoned"` |
| A body with no `## Stack` | parses exactly as before; the entry's stack is `python` |
| A body **with** `## Stack` — last, middle or second — read by the **pre-`## Stack`** parser | all four required sections parse identically; the fifth is discarded |
| `## Stack` naming an id outside the known set | `ContractError`, never a silent default |
| `## Stack` present but empty | `ContractError` |
| `## Files` naming a path the stack generates | `ContractError`, naming the path |
| A generated path the verify command did not produce | commit succeeds without it |
| A generated path that is a symlink out of the repository | skipped, not staged |

This repository's own backlog (#6–#35) is a real corpus: it has a diamond in its
dependency graph, hand-written issues with no markers, prose outside every
section, and the code-span trap in #11.
