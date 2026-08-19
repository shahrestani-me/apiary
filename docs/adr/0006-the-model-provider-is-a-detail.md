# ADR 0006 — the model provider is a detail, not the default

Status: **proposed**
Date: 2026-08-20

## The one-line summary

**A provider is a registry entry.** Which model serves a role — and whether it
runs on this machine at all — is a `ModelSpec` resolved at call time, not an
import at the top of `llm.py`. Ollama stays the default, and stays covered.

## Context

ADR 0003 decided the orchestration *framework* is a detail confined to one
module, and a test holds it there. The provider is the same kind of fact and was
confined not at all:

- `llm.py` imported `ChatOllama` at module scope and named it in both factories.
- The factories took no arguments, so the only way to change a model was an
  environment variable and a restart.
- `structured()` was documented purely in terms of Ollama's `format` parameter —
  "which is what makes small models usable for orchestration - they cannot
  wander off-format even if they want to."

That last one is the reason this is not simply "add a provider". Every
orchestrator call in this system is a `structured()` call, and Ollama's
guarantee comes from constraining the *decoder*. A remote provider reaching the
same guarantee does so by a different mechanism with a different failure mode:
the malformed reply is produced, and then rejected. Both raise, which is the
property callers depend on — but they raise different types, and
`worker/edit.py` retries a rejected reply exactly once based on what
`parse_failure()` recognises. A provider whose rejection type went unnamed would
not fail a test. Its retry would just stop firing.

The forcing function is #259: `propose_edits` emits corrupt output at roughly a
**40% rate** against `gemma4:26b`, in two classes — mid-token truncation, and
model chatter emitted as code. At that rate `SWARM_MAX_ATTEMPTS=3` often burns
every attempt on one task, so the swarm's headline capability is throttled by
generation quality rather than by any orchestration this repo has built.

## Decision

**1. `ModelSpec` is the unit, and it is data.** Provider, model, temperature,
context budget, and an open set of provider-specific options. Frozen, hashable,
comparable and cheap to log — so "did these two runs use the same model?" is
`==`, and a spec can be saved to a file (#265) and rendered on a page (#266)
with no redaction story of its own.

**2. A spec never holds a credential — only the means of finding one.** An
option may name an environment variable, or an AWS profile. Neither is a secret,
which is what makes decision 1's "cheap to log" true rather than aspirational.

**3. The option set is open, and validated against the provider's declaration.**
The three providers registered today need three genuinely different things to
dial:

| Provider | Endpoint | Credential | Schema mechanism |
|---|---|---|---|
| `ollama` | a URL | none — unauthenticated | `format`, constrained decoding |
| `openai` | the SDK's own | `$OPENAI_API_KEY` | strict `json_schema` |
| `bedrock` | a region | an AWS profile, through boto3's chain | `output_config` + `json_schema` |

A named `ModelSpec` field per provider would make every new provider a change to
a shared type, and would leave most fields meaningless on any given spec — which
is the shape that grows a `region` silently doing nothing on OpenAI. So the four
facts every provider has are fields, and the rest are options declared by the
provider. An option it does not declare is refused **by name**: a typo'd
`regoin` that defaulted quietly would leave a model served from somewhere else
in the world with nothing on the page or in the log to say so.

**4. `structured()` dispatches; everything else is a passthrough.** It is the
one function in the module that knows there is more than one mechanism. Two
consequences are written into the code rather than left to library defaults:

- OpenAI is asked for `method="json_schema", strict=True` **explicitly**. That
  default has moved before — it was `function_calling` until recently — and this
  is the guarantee the whole orchestrator rests on.
- Bedrock is asked for `json_schema`, which is **not** `langchain-aws`'s own
  default. Under its `function_calling` default, a tool call the model declined
  to make returns `None` rather than raising: silent coercion, which is the one
  outcome this module exists to prevent. It stays an *option* because not every
  model on Bedrock serves `json_schema`, and which ones do is a question only a
  live account answers.

**5. Remote SDKs are extras, not dependencies.** `pip install apiary` installs a
local-only system, which is what `README.md` and `pyproject.toml` promise.
`langchain-openai` and `langchain-aws` are imported inside their constructors,
and a missing one is a `ConfigError` naming the install command — not an
`ImportError` at module scope, which would take `swarm --help` down with it.
Both are in the `dev` extra, because a path the suite never runs is a guarantee
nobody holds.

**6. Ollama stays the default and stays covered.** Not a fallback nobody tests.
A clone with nothing configured runs fully local, `ModelSpec()` is an Ollama
spec, and the whole existing suite passes unchanged — which is the actual proof
the seam did not leak.

## What this rejects

**A provider adapter interface that normalises the mechanisms.** Rejected for
the same reason ADR 0003 decision 5 rejects a framework adapter, and ADR 0001
rejects tracker adapters: such a layer can only expose the intersection of what
the providers do. The intersection here is "returns JSON", which is exactly the
guarantee that is *not* good enough — constrained decoding and strict schema are
different strengths, and flattening them would hide which one a run got.

**Reading the provider from a mutable global.** `Settings` is `frozen=True` and
the console runs inference on a background thread while serving HTTP. A
process-global written from a request handler is a race against a run already in
flight, so the seam is an optional, defaulted *argument* — which also changed
none of the nine existing call sites.

## Consequences

- Adding a fourth provider is a registry entry plus an extra. It touches no
  shared type and no call site. That claim is asserted, not asserted-ish:
  `tests/test_llm.py` parametrises the interface over every registered provider.
- `tests/test_capture.py`'s "only `llm.py` constructs a model" guard now reads
  its list of client classes off the registry, and gained a second direction —
  a registered provider whose constructor is never called would otherwise have
  satisfied the original guard for free.
- `doctor.py`'s four `ollama.*` checks are now checks about a provider that may
  not be the configured one (#267), and `capture.py` reads Ollama's nanosecond
  `total_duration` spelling and would record blanks against the other two
  (#268). Both are follow-on work this ADR creates.
- A worker container holding a *model-provider* credential is a new security
  question, and it is specific to the worker: it executes model-generated code,
  and today needs no model credential at all because Ollama is unauthenticated.
  #269 decides it. Nothing in this ADR grants it.
- `dispatcher.py` sizes parallelism against GPU memory and
  `OLLAMA_MAX_LOADED_MODELS`. On a remote provider neither binds, and the
  constraint becomes quota and money, which nothing here counts (#270).

## What this does not claim

That remote is better, or that anyone should switch. #259 is the evidence gate
on whether the remote half is trustworthy at all, and it is unrun: it needs a
live account, and this ADR deliberately does not wait on one, because the seam
is worth having even if the answer is no-go. What the seam buys either way is
that the question becomes answerable by an operator on a page instead of by a
maintainer with a rebuild.

## Still open

- **Whether strict schema adherence actually holds** on the chosen remote model,
  across a replay of the same prompts `gemma4:26b` corrupts, and what a
  violation looks like when it happens. That is #259, and it gates any
  recommendation to *use* the remote path — not the existence of the seam.
- **Whether `gemma4:26b` at a raised `num_ctx` recovers any of the 40%.** If
  much of the corruption is prompt truncation rather than model quality, the
  cheapest fix is a local setting and no provider at all.
