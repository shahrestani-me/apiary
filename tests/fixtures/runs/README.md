# The replay corpus

One directory per run. Every one committed here is **synthesised**; the format
is written so that a genuinely recorded run drops in beside them with no code
change, and `tests/fixtures/corpus.py` is the single loader for both.

## Why these are synthesised, and what that costs

#145 is epic #140's go/no-go: nothing is deleted until derived state is proven
to reproduce label state. The honest proof is a replay of real recorded runs.
There are none, and the session that built this could not make any —
`swarm run --new` refuses classic and OAuth tokens by design
(`security.assert_provision_token`), no fine-grained PAT was available, and
`docs/demo-run.md` records the same wall from 2026-08-14. Four implementers hit
it independently.

So the streams below were built by hand. **A green replay proves the reducer is
self-consistent. It proves nothing about whether the reducer's model of reality
matches reality.** #146's shadow window is where that question gets answered,
and it needs at least one genuine run through it.

Anyone who acquires a working credential should record runs and re-run this
suite against them. That is the cheapest retirement of the largest risk in the
epic, and it costs no code: record, drop the directory in here, set
`"origin": "recorded"`, run `pytest tests/test_derived.py`.

## The directory

    tests/fixtures/runs/<slug>/
      corpus.json       the manifest
      run.json          exactly what `RunArtifacts` writes at startup
      events.jsonl      exactly the #141 lifecycle log
      results/*.json    exactly `worker.result.ResultRecord.to_dict`
      observed.jsonl    one line per cycle — the world, and the control plane

Four of the five are produced verbatim by a live run today, and `load_corpus`
calls `artifacts.read_run` on every directory to keep it that way: a corpus run
that `swarm show` cannot read has drifted from the format it claims to be in.

`observed.jsonl` is the one file a recorder has to add, and it is not new state.
Every field is something a cycle **already reads** for its own reasons —
`ContainerManager.find`, `checks.read_pulls`, the branch listing `recovery`
sweeps, `worker.result.load_results`, `Ledger.entries`. So recording one is a
projection of a cycle's own inputs, not a second source of truth.

### `observed.jsonl`, one line per cycle

```json
{
  "cycle": 3,
  "tasks": [
    {"ref": "#11", "task_id": "core", "depends_on": [], "closed": false, "state_reason": null}
  ],
  "branches": ["apiary/%2311-attempt-0"],
  "containers": [{"id": "c0refeed0001", "run_id": "…", "ref": "#11", "running": false}],
  "pulls": [{"number": 101, "head": "apiary/%2311-attempt-0", "merged": false,
             "closed": false, "draft": false, "sha": "000b0c0ffee"}],
  "results": ["issue-11-attempt-0.json"],
  "budget": {"max_attempts": 3, "max_total_attempts": 9},
  "live_run_ids": ["…"],
  "control": {"core": "swarm:review"}
}
```

| Key | What it records | Notes |
|---|---|---|
| `cycle` | the reconcile cycle index | strictly increasing; the loader refuses otherwise |
| `tasks` | identity and declared dependencies | **no state and no label** — see below |
| `branches` | the raw names a remote listing returned | parsed by `parse_task_branch`; anything apiary did not mint is dropped |
| `containers` | one `docker ps --all` row, reduced | `running` is the field `Handle` does not carry today |
| `pulls` | open and merged pull requests, by **head branch** | joined to a task through the branch name (#144), never through `Closes #n` |
| `results` | which result files this cycle could see | names must exist under `results/`, or the load fails |
| `budget` | the caps the run was configured with | operator setting, not a fact about the world |
| `live_run_ids` | which runs' containers hold claims | defaults to this run's id; `recovery.py`'s rule |
| `control` | the `swarm:*` label each task wore | **the thing being diffed against** |

Everything except `control` is the world. `control` is the control plane, and
the loader keeps them in separate attributes so that "the resolver read a label"
would be a visible line in a diff rather than an attribute access that looks
like every other one. `orchestrator/derived.py` has never seen a `swarm:*`
string; `lifecycle.INTERNAL_STATE` does the translation, in the loader, because
that mapping belongs to whoever *reads* a label.

Recording the label rather than the internal state is deliberate: the corpus
records what the control plane actually held, so the day epic #140 removes the
labels it is the translation that gets deleted and not the data.

### `corpus.json`

```json
{
  "schema": 1,
  "origin": "synthesised",
  "describes": "prose: what this run is and why it is interesting",
  "exercises": ["a verify failure", "an attempt consumed"],
  "expected_divergences": [
    {"cycle": 7, "task": "bootstrap", "derived": "eligible", "control": "needs-human",
     "why": "the argument for why derived state cannot reproduce the label here"}
  ]
}
```

`origin` is metadata. **Nothing branches on it** — `test_derived.py` proves that
by copying a run, flipping the field to `recorded`, and asserting the replay is
identical.

## Divergences are declared, not forbidden

Three things turned out not to be derivable, and a corpus asserting "no
divergence, ever" could only have been committed by leaving all three out —
which is exactly the tuning that would make the exercise worthless. So each
manifest declares the divergences its run should produce, and the harness
asserts the set matches **exactly**: an undeclared divergence fails, and a
declared one that stops happening fails too.

The second direction is the one that will earn its keep. The day one of these
becomes derivable, `test_derived.py` says so rather than silently agreeing.

Every `why` must be a real argument — the suite refuses a declaration shorter
than a sentence, because an empty one is a disagreement somebody silenced.

## The runs

| Run | What it exercises | Divergences |
|---|---|---|
| `01-happy-path-chain` | all six states, one dependency edge | none |
| `02-verify-failure-retry` | exit 1, an attempt consumed, a retry that lands | none |
| `03-interrupted-orchestrator` | a claim written, the process killed before the spawn | 1 — the stale claim |
| `04-container-died-mid-flight` | a container that vanished, leaving no testimony | 1 — the stale claim |
| `05-infrastructure-exit-2` | three exit 2s onto one filename, into the infra ceiling | 1 — the ceiling is not derivable |
| `06-goal-gate-revival` | the budget spent, then `planner.revive` | 3 — the revival is not derivable |
| `07-renewed-retry-budget` | a blocker signature changing, renewing the budget | 3 — `streak` is not derivable |

Runs 03 and 04 look identical from the code host on purpose: in one the spawn
never happened and in the other it happened and left nothing, and **nothing in
the branches, the containers or the artifacts distinguishes them**. Both are
cases where the derived state is right and the label is the stale one, which
makes them the strongest evidence for ADR 0001 in this corpus.

Runs 05, 06 and 07 are the opposite: the label is right and the derived state
cannot get there, because the fact that decided it is a judgment apiary made
about its own execution and `docs/adr/0002-apiary-owns-a-thin-task-store.md`
puts those in apiary's own store rather than on the code host.

## Adding a run

Write the five files. Nothing needs registering — `corpus_runs()` reads the
directory, so a run added tomorrow is replayed because it exists. Then run:

    pytest tests/test_derived.py -q

If it reports an undeclared divergence, do not add a declaration until you can
write the argument for it. The declaration is the argument; the four-tuple is
just its index.
