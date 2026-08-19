# ADR 0005 — the attempt counter moves to the store

Status: **proposed**
Date: 2026-08-19
Amends: `docs/adr/0002-apiary-owns-a-thin-task-store.md`

## The one-line summary

**The counter follows the judgment into the store, and the branch listing takes
over the job the issue marker was doing.** ADR 0002 left the counter in the
tracker for a reason that #239 has since discharged — but the reason it gave was
not the reason it was *safe*, and the safety has to be rebuilt somewhere else
before the marker can go.

## Context

ADR 0002 decision 3 kept the attempt counter in the issue marker while moving
`blocker`, `streak` and `renewals` into apiary's own store. Its stated reason was
mechanical:

> Moving it would mean threading it through `ContainerManager.spawn`, the
> container environment and the worker entrypoint before a branch could be named
> — three modules this ticket does not own.

**#239 did exactly that threading**, for #152's sake: `ContainerManager.spawn`
takes an `attempt=`, it travels as `APIARY_ATTEMPT`, and
`worker.entrypoint.dispatched_attempt` prefers what it was told over what the
marker says. The stated reason is discharged. That is what makes this amendment
possible; it is not what makes it correct.

### The reason it was safe is a different reason, and ADR 0002 says so itself

ADR 0002 has a passage most of the way down that a reader looking only at
decision 3 will miss. It corrects its own earlier account of why a per-project
store is safe:

> **And the reason given above for the per-project store is not the reason it is
> safe.** […] an empty store "reads as 'every task is on attempt 0 with no
> blocker'". The second half is true; the first half is not, because the counter
> never moved into the store — it stays in the issue marker, so
> `max_total_attempts_per_task` still bites on an empty store.

It then names the two pieces of arithmetic that actually hold the budget
together, and warns about one of them by name:

> - `previous_streak = entry.attempt if entry.streak is None else entry.streak`.
>   […] The fallback is therefore the **largest** streak consistent with the
>   counter: absence gives up sooner, never later. **Simplifying that fallback to
>   `0` is the change that would make the ADR's stated fear real.**
> - `renewed = bool(entry.blocker) and sig != entry.blocker`. Renewal […] is
>   gated on a blocker being **present**. A missing judgment yields an empty
>   blocker, so a miss can never renew.

The second protection survives this amendment untouched: it is about `blocker`,
which is already in the store, and a miss still cannot renew.

**The first protection does not survive, and it does not survive quietly.**
Nobody has to simplify the fallback to `0` for the hole to open. The fallback is
safe because `entry.attempt` comes from somewhere a store wipe cannot reach —
the customer's tracker. Move the counter into the store and `entry.attempt`
becomes a reading *of the store*, so on an empty store the fallback evaluates to
`0` by itself, with the line unchanged and every test still passing. The warning
ADR 0002 wrote is about an edit; the danger this amendment introduces needs no
edit at all.

That is the whole of what this ADR has to answer.

### Why the obvious answer is not enough

`derived._attempts_spent` already reconstructs a lower bound on the counter from
the code host: results that spent budget, branches, and pull requests, folded
with `max`. It looks like the replacement, and on its own it is not one.
`Observation.branches` carries **the head refs of open pull requests only** —
`tests/fixtures/runs/README.md` says so, and gives the reason: a remote branch
listing is not a call the cycle makes, and #146 forbade adding one.

So a task whose pull requests are all closed or merged contributes no branch,
and `_attempts_spent` reads `0`. On a populated store that is harmless, because
the store answers first. On an empty one it is the fresh budget for every task
that ADR 0002 forbids by name.

## Decision

**1. The store owns the attempt counter.** It is apiary's own judgment about its
own execution, which is ADR 0002's own test for what belongs there, and after
#152 the tracker holds nothing of apiary's to hold it in. `LedgerEntry.attempt`
becomes a reading of the store rather than of the issue body.

**2. The durable substrate for the counter is the branch listing.** The marker's
value was never the marker; it was that the number lived somewhere a store wipe
could not reach. After #152 the only externally durable record of how far a task
got is the branch names `apiary/<ref>-attempt-<n>` that #144 put there, on the
code host, outliving both the store and the run.

So `_attempts_spent` is fed a **full** task-branch listing rather than the head
refs of open pull requests.

**3. That listing is `recovery`'s, not a new call.** `orchestrator/recovery.py`
already lists remote branches every cycle and already parses them with
`in_flight` into the `TaskBranch` values `Observation.branches` holds. Threading
the listing it already has into the observation adds **no API call**, which is
what keeps #146's constraint intact. Anyone closing this gap by adding a listing
call to the observation would be reintroducing exactly what #146 refused.

**4. The human-resets-the-counter affordance moves rather than being lost.**
ADR 0002 quotes it as a real workflow: a person edits the marker to give a stuck
task another go. It is not acceptable for that to become "no longer possible"
by omission. It becomes an operation on the store, and #153's documentation
amendment is where the new gesture is written down.

## Consequences

**The budget's floor becomes a branch-name fact.** With decision 2 the lower
bound survives a store wipe, a run directory deletion and a machine change,
because it is reconstructed from refs on the code host. It is a *lower* bound,
as it was before: a task whose branch was deleted reads lower than the truth and
gets a retry it may not have earned. That is the same direction the old fallback
erred in, and it is the safe one — `_attempts_spent`'s docstring already argues
that over-counting gives up too early and under-counting merely retries.

**Two lines must not be tidied, and one of them is new.** ADR 0002 named
`previous_streak`'s fallback; that warning now transfers to decision 2's
listing. If a later change narrows `Observation.branches` back to open pull
requests — which is what it holds today, and which looks like a harmless
optimisation because a cycle rarely needs more — the budget floor silently drops
to zero on an empty store. `renewed`'s blocker gate is unchanged and still
load-bearing.

**Reconcile's oldest rule needs restating, not overriding.** "GitHub wins, every
cycle, on every disagreement" was written when the tracker held the counter.
After this amendment the code host still wins on the facts it owns — issues,
pull requests, branches, containers — and the store owns apiary's judgments,
which is ADR 0002 decision 3's "one authority per fact" applied to a counter that
has changed residence. `ledger._judged` inverts: today the tracker's counter
decides whether a stored signature is stale, and afterwards the stored counter is
the counter, with the branch listing as its floor rather than its adjudicator.

**This is the change that makes an empty store dangerous for the first time.**
Worth stating plainly because ADR 0002's readers were told the opposite and were
right at the time. Until now a wiped store cost a stale signature. After this it
costs the budget, unless decision 2 is in place first — so decision 2 lands with
decision 1 or before it, never after.

## Evidence

**This amendment should not be implemented before the ten recorded runs**
(`docs/recording-runs.md`), and the reason is stronger than caution.

`_attempts_spent` under-reading the budget is not a hypothetical to be reasoned
about — it is the exact shape of thing the derived-state comparison detects. A
task whose counter the resolver reads lower than the control plane does shows up
as a divergence on the run where it happens, on precisely the tasks that matter.
The runs would tell us whether decision 2's floor is sufficient in practice or
whether branch deletion makes it leakier than the argument above predicts, and
they can only tell us that while the labels are still written.

**The instrument is the recorder, not the window** — which is a correction to this
section rather than a change to its conclusion. This ADR was written while the
shadow window still resolved beside each live cycle and emitted
`state.divergence`; #244 deleted it. The comparison survives because a line of
`observed.jsonl` carries both sides — the derived world and the `control` labels
— so divergences are computed from the recording afterwards, which is what
`docs/recording-runs.md` now instructs. Two consequences worth having in writing:

- The runs no longer have to be measured *as they happen*. A recording made today
  can be replayed against a resolver written next month, which is strictly better
  than a live window for an amendment that changes what `_attempts_spent` reads.
- What is genuinely lost is the *independence* count, which came from the
  `CycleReport` and is not in the recording. So "the floor is sufficient in
  practice" is a judgement over the recorded divergences rather than a ratio
  somebody can quote.

So the sequencing is not a preference:

1. The ten runs. The labels must still be written when they are *recorded*; the
   window no longer has to exist.
2. This amendment implemented, decision 2 first.
3. #152 c3 — the label writes and `APIARY_STATE_SOURCE`, together. (The window
   was the third item here and is already gone, merged as #244; that changed the
   deadline from "before the window goes" to "before c3 goes", and nothing else.)

An implementation that lands before step 1 forecloses the only measurement that
would have checked it.

## What this does not change

`blocker`, `streak` and `renewals` stay exactly where ADR 0002 put them. The
store's `attempt` stops being only a stamp on a judgment and becomes the counter,
which is a change of role for a field that already exists rather than a new
column. And ADR 0002's decision 4 — the store is per project and not under the
run artifacts root — becomes *more* load-bearing, not less: it was already the
thing standing between a wiped store and a fresh budget, and after this it is the
only structural thing doing so.
