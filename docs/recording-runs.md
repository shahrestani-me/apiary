# Recording real runs — the operator's runbook

**Who this is for:** the person holding the two fine-grained PATs. Everything
below needs a credential that cannot be minted from inside a coding session, so
this document exists to be handed to a human and executed by them.

**What it unblocks:** #152 — deleting the label control plane — and #153 behind
it. Every code prerequisite for both is merged. A PR deleting the labels would
pass CI today. What does not exist is the evidence that deleting them is safe.

---

## 1. Why this is the last thing standing

#152 is the point of no return, and it is worth being precise about why.

Today the labels are still written and still compared. That buys two things:
`APIARY_STATE_SOURCE=labels` restores the old behaviour completely, and the
shadow window has something to diff the resolver against. Delete the writes and
**both vanish at once** — a resolver mistake stops being a divergence somebody
can see and becomes ordinary wrong behaviour with no second opinion to catch it.

The proof that the resolver is right does not exist yet:

- **#145 built the replay corpus, and it is synthesised.** Its own README says
  so: a green replay proves the reducer is self-consistent and proves nothing
  about whether the reducer's model of reality matches reality.
- **#146 built the shadow window — the real instrument — and it has never seen
  a real run.**
- **#195 wired the recorder**, so `observed.jsonl` plus a manifest now land in
  every run directory. A recorded run is a `cp -r` from being a corpus run.

Everything is built. Nothing has been pointed at reality.

## 2. What "done" looks like

#147's gate: **ten consecutive runs with zero unexplained divergences.**

Two things make that harder to satisfy honestly than it looks, and both are
already instrumented — read them before deciding a clean run is a good one.

**A clean window with a small independent count is a weak result.**
`plan_reconcile` computes a cycle's labels from the same listings the resolver
reads. So agreement on a task the cycle itself just wrote proves only that two
reducers implement one rule set. `ShadowReport.independent` counts the
comparisons that are genuinely independent — the tasks the cycle did *not*
write — and `swarm show` prints it. A run reporting `40 task-cycle(s) compared
… 3 of them independent` is close to no evidence at all.

**Zero divergences is ambiguous unless the window actually ran.** An events log
with no `state.divergence` line in it is either a clean shadowed run or a run
with the flag off. That is why every shadowed cycle emits `state.shadow`, and
why `swarm show` distinguishes three cases explicitly:

```
derived shadow: not run (APIARY_DERIVED_SHADOW off, or a run from before #146)
derived shadow: ran for 6 cycle(s) and compared no tasks - this run is unmeasured, not clean
derived shadow: 40 task-cycle(s) compared over 6 cycle(s); 31 of them independent of this cycle's own writes
  2 divergence(s), 0 unexplained (dispatched-this-cycle 2)
```

Only the third shape counts toward the ten.

## 3. Preconditions

Two **distinct** fine-grained PATs. `swarm doctor` fails the run if they are
equal.

```bash
export GITHUB_TOKEN=github_pat_...            # contents:write, pull_requests:write, issues:write, metadata:read
export APIARY_PROVISION_TOKEN=github_pat_...  # administration:write, contents:write, workflows:write, issues:write, metadata:read
```

Classic and OAuth tokens are refused by design in
`security.assert_provision_token`, and that refusal is correct — a classic PAT
here reaches every repository the account owns while holding `administration`.
Do not work around it.

Leave both shadow-related environment variables **unset**:

| | default | what setting it would do |
|---|---|---|
| `APIARY_DERIVED_SHADOW` | on | off means the run records nothing and counts for nothing |
| `APIARY_STATE_SOURCE` | `derived` | `labels` is the escape hatch; a run on it measures nothing |

Then the images, and a read-only preflight:

```bash
docker build -f Dockerfile.worker      -t apiary-worker      .
docker build -f Dockerfile.worker.node -t apiary-worker-node .
swarm doctor
```

Before a `--new` run, the one recorded attempt (`docs/demo-run.md`, 2026-08-14)
saw **15 checks, 12 ok, 1 failed, 2 not attempted** — the one failure being
`github.repo`, "no target repository", because the repository is what the run is
about to create. Treat the shape rather than the exact counts as the
expectation; checks have been added since.

## 4. The credential wall, and the cheap way past it

**Read this before running anything, because the obvious order costs ten trips
to a browser and the right order costs one.**

`docs/demo-run.md` §4 records the structural problem. A fine-grained PAT's
repository selection is fixed when the token is minted. The work key must name
the target repository. The target repository does not exist until the run
creates it. So `swarm run --new` **always** stops immediately after
provisioning, when `start_run` makes its first read with the work key and gets
a 404. That is where every greenfield run stops, and it is a property of the
design rather than of any particular host.

The only ways through are to widen the work key to every repository the account
owns — which is the account-wide reach `security.py` exists to prevent — or to
have a human add the new repository to the work key between provisioning and
dispatch.

**Take the second path, but batch it.** Provisioning is complete and durable
before the run dies: the repository, the initial commit, the six `swarm:*`
labels and branch protection are all in place, and it takes about six seconds.
So provision all ten first, add all ten to the token in one visit, then run all
ten.

### Phase 1 — provision ten repositories (~1 minute total)

Each command creates a repository and then exits 1, printing nothing. **That is
the expected outcome here, not a failure.** See §8 on why it prints nothing.

```bash
for i in $(seq 1 10); do
  swarm run --new "@briefs/brief-$i.txt" \
    --owner shahrestani-me \
    --name apiary-corpus-$i \
    --yes
done
```

`--yes` skips `provision`'s confirmation prompt, which is the only interlock on
an action its own docstring says it cannot undo. Skipping it across ten
repositories in a loop is worth a deliberate decision rather than a habit —
that decision is yours, and this document is where it is being asked for.

### Phase 2 — one browser visit

1. Open the **work** token at <https://github.com/settings/personal-access-tokens>
2. Add all ten `apiary-corpus-*` repositories to its selected repositories
3. Save

Verify before continuing, so a bad save is caught now rather than ten runs from
now:

```bash
for i in $(seq 1 10); do
  gh api "repos/shahrestani-me/apiary-corpus-$i" --silent \
    && echo "$i ok" || echo "$i STILL 404"
done
```

That must use the **work** key, not the boot key — the boot key answers 200
either way, which is exactly the confusion §4 of `demo-run.md` documents.

### Phase 3 — run them

```bash
swarm run --repo shahrestani-me/apiary-corpus-$i \
  --objective "@briefs/brief-$i.txt" \
  --verify "<the right command for this repo's stack>"
```

**Pass `--verify` deliberately every time.** The default for a `--repo` run is
`python -m pytest -q`, which on a repository the planner typed as `react` fails
in every worker for a reason that has nothing to do with the model's output —
and a run whose every task fails mechanically is a run that tells you about the
verify command rather than about the resolver.

## 5. Choosing the ten briefs — the part that decides whether the gate means anything

Ten runs of the same easy brief would very likely produce ten clean windows, and
would be close to worthless. The divergence classes most worth exercising are
the ones that only appear when things go **wrong**:

| kind | what produces it |
|---|---|
| `infrastructure-ceiling` | mechanical failures — a missing image, a dead daemon, a pulled network |
| `budget-renewed` | a task that exhausts its retries and gets a renewed per-blocker budget |
| `revived` | `planner.revive` on a task that was given up on |
| `dispatched-this-cycle` | ordinary, and the one you will see most |
| `merged-this-cycle` | the merge gate landing something after the world was read |

So aim for a spread: a few briefs that should succeed, at least one large enough
to need several cycles and a rebase, and at least one run where you deliberately
break the infrastructure mid-run (stop the Docker daemon for a cycle, or remove
`apiary-worker-node` before a node task dispatches). **A deliberately broken run
is worth more here than a clean one**, because the infrastructure ceiling is one
of ADR 0001's three states that is not derivable at all, and it is the one the
overlays exist to handle.

Bigger plans also help the number that matters: the independent share tends to
`(N - O(1)) / N` and rises with plan size, so a ten-task plan produces far
stronger evidence per cycle than a two-task one.

## 6. After each run — triage before you keep it

```bash
swarm runs                 # newest first
swarm show <run-id>
```

Read the shadow block against §2. Then:

- **Unexplained divergences?** That is a finding, and possibly the most valuable
  output of this whole exercise — it is the resolver being wrong about reality,
  which is precisely what no synthesised corpus can show. Keep the run, capture
  the output, and do **not** try to make it go away.
- **`compared no tasks`?** Unmeasured, not clean. Does not count toward the ten.
- **Clean, but `independent` is low?** Weak. Keep it, note the number, and do
  not count it as a strong result.

To commit a run to the corpus:

```bash
cp -r .swarm/runs/<run-id> tests/fixtures/runs/08-<a-name-for-it>
pytest tests/test_derived.py -q
```

## 7. What a machine must not do for you

The recorder writes `corpus.json` with `origin: "recorded"` and
**`expected_divergences` empty on purpose.** The harness fails on an undeclared
divergence, so a recorded run *refuses to pass* until a human has written the
argument for each divergence it produced.

That is deliberate and it is the one part of this nobody should automate. Do not
let anything — including me — fill in `expected_divergences` to make the suite
go green. `describes` and `exercises` are also for a human: the recorder writes
the objective and an empty list.

If you hand the run directories back with the divergences unexplained, that is
the correct state to hand them back in. Writing the arguments is a review
conversation, not a chore.

## 8. Known rough edges you will hit

None of these are new, all are recorded, and knowing them in advance saves an
hour of debugging something already understood.

- **A failed run before `start_run` is invisible to the tooling.** Run artifacts
  are created by `start_run`, so a run dying earlier leaves nothing in
  `.swarm/runs` and `swarm runs` / `swarm show` can say nothing about it. Every
  phase-1 provisioning exit is one of these.
- **It exits 1 while printing nothing.** `cli.py`'s shared handler prints
  `! {exc}` and the exception carries an empty message, so a run that stops with
  no stated reason looks exactly like one that finished.
- **The greenfield CI gate is a placeholder and stays one.** Generated CI runs
  `test -f README.md`, which passes on every commit that could ever be pushed,
  including one deleting the entire application. Combined with the admin
  override, a merged PR in a greenfield repository carries a green check
  certifying that a file nobody touched still exists.

  For *this* exercise that is tolerable, and it is worth saying why: the shadow
  window compares derived state against label state, and both are computed from
  the same run regardless of whether the code is any good. A worthless verify
  gate does not bias the comparison. It does bias the **mix** of states reached
  — which is what §5 is about — and any figure reported as a merge count from
  these runs has not been earned.
- **The stack choice is single-valued and silent.** `choose_stack` asks a model
  what the brief implies and takes one answer; a brief asking for a backend and
  a frontend keeps one half, and nothing reports that as a narrowing. Write
  briefs that name one stack, unless narrowing is what you want to observe.

## 9. What to hand back

For each run:

- the run directory, `.swarm/runs/<run-id>` — all five files
- the `swarm show <run-id>` output, verbatim, including the shadow block
- one line on what the run was meant to do and anything you did to it mid-flight
  (stopped the daemon, edited a token, killed a container)

Ten of those, with the independent counts visible, is what #152 has been waiting
for. Nine clean runs and one with an unexplained divergence is a **better**
outcome than ten clean ones, because it is the first time this system has been
told something about itself it did not already assume.
