# ADR 0003 — the orchestration framework is a detail, not the architecture

Status: **proposed**
Date: 2026-08-19

## The one-line summary

**apiary's orchestration is its own loop.** No agent framework's types appear in
the modules that do the work — so a framework can be replaced, or run alongside
another, without touching them.

## Context

apiary declares five dependencies and four are LangChain-ecosystem:
`langgraph`, `langgraph-checkpoint-sqlite`, `langchain-ollama`,
`langchain-core`. A reader would reasonably conclude the system is a LangGraph
application.

It mostly is not, and that happened by accident rather than design. v2 replaced
the graph with a reconciler, and the modules that carry the system —
`orchestrator/`, `github/`, `containers/`, `worker/`, `artifacts.py`,
`security.py`, `doctor.py` — contain no framework import at all. `Reconciler` is
a plain loop over plan-then-apply. That is the valuable property, and nothing
currently protects it.

The coupling that remains is in four places, and they are not equal:

| Where | What | Weight |
|---|---|---|
| `graph.py` | `StateGraph`, `START`/`END`, `Send` | The real one — and it is live again |
| `state.py` | `Annotated[..., _merge_tasks]`, `Annotated[..., operator.add]` | LangGraph state channels, on types the v2 ledger also uses |
| `llm.py` | `ChatOllama`, `.with_structured_output()` | 45 lines, LangChain not LangGraph |
| `capture.py` | `BaseCallbackHandler` | LangChain's callback protocol |

`graph.py` was vestigial until #135 revived it: `swarm local` runs the v1 graph
against a local checkout — worktrees instead of issues, merges instead of pull
requests, no GitHub at all. So there are now **two execution models** in the
tree, and the older one is the framework-shaped one.

## Decision

**1. No framework type crosses into the working modules.** `orchestrator/`,
`github/`, `containers/`, `worker/`, `artifacts.py`, `security.py` and
`doctor.py` import no agent framework, and a test asserts it. This is already
true; the decision is to keep it true on purpose.

**2. The model client is a narrow interface.** `llm.py` is 45 lines and is the
only place that knows a provider SDK exists. Structured output — the one thing
apiary genuinely relies on — is named in apiary's own terms, not
`with_structured_output`'s.

**3. The runner is the unit of execution, and runners may coexist.** A runner is
a top-level entry point owning a complete execution model and composing the
framework-free modules beneath it. `swarm run` and `swarm local` are two of them
today. A Hermes runner would be a third, and adding one is additive work that
touches no existing runner.

The framework is an implementation detail *of a runner*, which is what makes
"support LangGraph and Hermes at once" cost nothing structurally: they never meet.

**4. A runner declares its capabilities, and a user's choice is presented in those
terms — never as a framework name.** This is the rule the existing pair already
violates, which is why it is written down.

`swarm run` and `swarm local` did not diverge in implementation. They diverged in
what they can do:

| | `swarm run` | `swarm local` |
|---|---|---|
| Container sandbox | yes | **no** |
| Pull request and CI gate | yes | **no** |
| Merge queue | yes | **no** |
| Egress policy | yes | **no** |

`nodes/verifier.py` runs the verify command with `shell=True` on the host, in a
worktree of model-generated code — so the local runner bypasses the entire
argument of `docs/security.md`, while the CLI offers it as "a local checkout, no
GitHub". A convenience framing on a security decision.

So: whatever a user is choosing between, it is capabilities. Somebody selecting
"Hermes" must not thereby be selecting "no container isolation" without being
told. A runner that drops a capability says so where the choice is made.

**5. No unifying framework abstraction.** Rejected, and this is narrower than it
sounds: what is rejected is an adapter interface that makes frameworks
interchangeable *inside one runner*. That is building a second framework to avoid
depending on the first — the mistake ADR 0001 identified in the tracker adapters —
and such an interface can only expose the intersection of what the frameworks do,
which is less than either.

Coexisting runners need no such layer, which is the point. Decision 3 gets the
multi-framework outcome by composition; this decision refuses to get it by
indirection.

## Consequences

- Switching to Hermes, or to no framework at all, touches `graph.py`, `llm.py`,
  `capture.py` and the `Annotated` reducers in `state.py`. It does not touch the
  reconciler, the dispatcher, the check gate, the merge queue, the container
  layer, or any adapter.
- `capture.py`'s prompt recording is written against LangChain's callback
  protocol. Under a different framework it is rewritten, and the console feature
  it feeds goes with it. Worth knowing before it grows.
- The import-boundary test is the load-bearing artefact of this ADR. Without it
  this is a preference, and preferences decay — as `graph.py` demonstrated by
  becoming live again without anyone deciding it should.
- Every runner is a surface to test and maintain. Two exist and their capability
  gap went unnoticed and undocumented; a third makes that arithmetic worse, not
  better. Coexistence is cheap structurally and is not free operationally.
- `swarm local`'s missing sandbox needs documenting or gating either way, and is
  not blocked on the open question below.

## What this does not claim

That apiary should leave LangGraph, or that adopting another framework means
retiring it. It is a reasonable dependency, runners may use different ones at the
same time, and nothing here argues against either. The claim is only that the
decision stay cheap to revisit, which today it accidentally is and tomorrow it
accidentally might not be.

## Still open

Whether `swarm local` is a supported runner or is retired. Decision 4 says that if
it stays, its capability gap is stated where the choice is made; it does not
decide whether it stays. That is the maintainer's call.
