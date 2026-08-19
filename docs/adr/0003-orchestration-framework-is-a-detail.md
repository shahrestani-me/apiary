# ADR 0003 — the orchestration framework is a detail, not the architecture

Status: **proposed**
Date: 2026-08-19

## The one-line summary

**apiary's orchestration is its own loop.** No agent framework's types appear in
the modules that do the work, so replacing LangGraph costs a module rather than a
rewrite.

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

**3. `swarm local` is where the framework lives, and its fate is decided
explicitly rather than by neglect.** Either it is a supported second execution
model, in which case it is documented as the one LangGraph-dependent surface, or
it is retired. What it must not be is an undecided path that quietly makes the
framework load-bearing again — which is exactly what happened between the v2
rewrite and #135.

**4. No framework abstraction layer.** Explicitly rejected. Writing an adapter so
that LangGraph or Hermes could be swapped is building a second framework to avoid
depending on the first, and it is the same mistake ADR 0001 identified in the
tracker adapters. The protection is the *absence* of coupling in the working
modules, verified by a test — not an indirection that stands between them and
anything.

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

## What this does not claim

That apiary should leave LangGraph. It is a reasonable dependency and nothing
here argues against it. The claim is only that the decision stay cheap to
revisit, which today it accidentally is and tomorrow it accidentally might not be.
