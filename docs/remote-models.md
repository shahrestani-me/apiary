# Running a remote model — Bedrock, and how the models compare

**Who this is for:** anyone deciding which model serves `orchestrator_model` or
`worker_model`, or replaying a `propose_edits` prompt against two models to see
which one is wrong.

**What it is not:** a change of default. Ollama is still the default and still
covered (ADR 0006); `pip install apiary` installs a local-only system, and the
remote providers are extras a default install does not download. Everything here
is opt-in.

---

## 1. Why this document exists

#259 asked whether a remote model fixes the corruption `gemma4:26b` emits from
`propose_edits` — mid-token truncation, and model chatter written out as code.
The answer turned out to be yes, and the more useful finding was that it does
not take a frontier model to get it.

That answer is only worth having if it is re-checkable. So this document carries
two things: the commands to point a role at Bedrock, and a table of what each
model actually scored. **The table is meant to grow** — §5 is the procedure for
adding a row.

---

## 2. Install the extra

`langchain-aws` and `langchain-openai` are extras, not dependencies. Their
imports are lazy and inside the constructors, so a missing extra is a
`ConfigError` naming the install command rather than an `ImportError` that would
take `swarm --help` down with it.

```bash
pip install -e ".[bedrock]"     # AWS Bedrock
pip install -e ".[openai]"      # the direct OpenAI API
```

## 3. Point a role at it

A **bare model name still means Ollama**, exactly as it always has. A provider
prefix is what opts in:

```bash
export SWARM_WORKER_MODEL="bedrock:eu.anthropic.claude-haiku-4-5-20251001-v1:0"
export SWARM_WORKER_MODEL_OPTIONS="profile=preprod,region=eu-west-1"
```

`SWARM_ORCHESTRATOR_MODEL` / `SWARM_ORCHESTRATOR_MODEL_OPTIONS` are the same
shape, and the two roles are resolved independently — they need not be the same
model, or even the same provider.

Bedrock declares three options: `profile`, `region`, and `method`
(`json_schema`, the default, or `function_calling` — how the schema is forced).
An option a provider does not declare is refused by name, so a misspelled
`regoin` is an error rather than a silent default.

Bedrock authenticates through the ordinary AWS credential chain, so the
"credential" is a profile name plus wherever boto3 finds its keys. Both options
fall through to boto3's own environment when unset, which is why a machine
already configured for AWS needs no apiary configuration at all.

## 4. Prove it before you spend anything

```bash
swarm doctor
```

Four checks answer this, and they are deliberately separate — each one fails for
a different reason:

| check | what it proves |
|---|---|
| `model.target` | which model each role resolved to, and from where |
| `model.reachable` | the endpoint answered at all |
| `model.available` | the model is in the catalogue — or that listing is a separate permission |
| `model.schema` | **one real call returned the requested schema** |

`model.schema` is the one that matters, because every call in this system is a
`structured()` call. A model that cannot be constrained cannot hold either role.
Run doctor first; it is much cheaper than a failed replay.

A real failure, for reference — the endpoint is fine and the *model* is not:

```
ok    model.reachable   bedrock reachable as aws profile 'preprod' in eu-west-1
FAIL  model.schema      AccessDeniedException: Model access is denied due to IAM
                        user or service role is not authorized to perform the
                        required AWS Marketplace actions
                        (aws-marketplace:ViewSubscriptions, aws-marketplace:Subscribe)
```

`SWARM_SPEND_CEILING_USD` defaults to `5.00` and halts a run that spends it.
Only paid providers accrue; a fully local run never constructs a ledger.

## 5. Comparing two models on the same prompt

`swarm console` is the instrument. It turns capture on for itself, is
single-flight, and the `propose_edits` tab takes a **per-call model override
that persists nothing** — so a like-for-like comparison against the local
default needs no configuration at all.

```bash
export APIARY_CAPTURE=1
swarm console
```

Then, on the `propose_edits — the worker` tab:

1. Fill in **Repository checkout**, **Files it may edit**, and **Goal**.
2. Leave the model override empty for the baseline — that resolves to whatever
   `SWARM_WORKER_MODEL` says, a bare name meaning Ollama.
3. Run it, then press **Keep for comparison** to pin the answer.
4. Set the override and run the same prompt again.

The override is two fields:

| field | value |
|---|---|
| model | `bedrock:eu.anthropic.claude-haiku-4-5-20251001-v1:0` |
| options | `profile=preprod,region=eu-west-1` |

Only `propose_edits` and `choose_stack` accept an override. The other sites build
their models several layers down, and the console refuses rather than accepting a
setting that would not take effect.

A cold local call takes one to three minutes. A Bedrock call takes seconds —
which is itself one of the findings below.

## 6. What the models scored

10 whole-file `propose_edits` goals over one small Python library, each replayed
twice. Every reply was scored three ways: did it satisfy the `WorkerOutput`
schema, did every edited `.py` file parse, and did `pytest` pass after applying
the edits to a clean copy.

| model | provider | n | schema held | green | corrupt | median |
|---|---|---|---|---|---|---|
| `gemma4:26b` @ `num_ctx=16384` | ollama | 12 | 12/12 | 8/12 (66%) | **1/12** | 127 s |
| `claude-haiku-4-5` | bedrock | 20 | 20/20 | 18/20 (90%) | **0/20** | 4.3 s |
| `claude-sonnet-4-5` | bedrock | 20 | 20/20 | 20/20 (100%) | **0/20** | 7.5 s |
| `gpt-5.6-luna` | bedrock | — | not run | — | — | — |
| `gemma4:26b` @ `num_ctx=65536` | ollama | — | not run | — | — | — |

**"Corrupt" means the two classes #259 names**, and nothing else: output that
Python's parser rejects, or model chatter emitted as source. A file that parses
and is simply wrong is a *semantic* failure — counted against `green`, not
against `corrupt`. That distinction is the point. Only Python's parser objects to
corruption; the same text in Markdown, JSON or a comment passes `$SWARM_VERIFY`
silently.

### What this says

**Remote eliminates the corruption class.** Zero occurrences in 40 remote calls.
The local failure reproduced verbatim in kind:

```
tests/test_validate.py:15  invalid decimal literal
trip = Trip("", date(20ng, 5, 1), [S
```

`date(20ng, 5, 1)` is the same mid-token corruption as #259's
`date(20int(2023, 5, 15))`. Chatter-as-code appeared in no configuration.

**Strict schema adherence held, 40/40.** This was the open question, and the one
`llm.py` warns about in as many words: Ollama's `format` constrains the *decoder*
so a model cannot wander off-format, while Bedrock's `json_schema` produces a
reply and then rejects it. Different mechanism, different failure mode — and in
40 calls it never failed, so **there is still no observed example of what a
violation looks like on this provider**, and `parse_failure()`'s Bedrock
rejection path remains untested against a real rejection.

**A frontier tier is not required.** Haiku matched Sonnet on corruption (0/20
each) and cost a fraction. Haiku's only misses were the same task twice — a
semantic error, asserting a field name the model invented. That is the practical
recommendation: `claude-haiku-4-5` for both roles, with `claude-sonnet-4-5` as
the upgrade if worker quality ever proves limiting.

**Retry policy follows from the failure class.** `propose_edits` retries a
rejected reply exactly once. That is right for corruption and close to useless
for a semantic error, which a second decode at temperature 0.1 will reproduce.
Remote shifts the remaining failures from the first kind to the second.

### Caveats, so the numbers are not read as more than they are

- **The 40% was not reproduced.** `gemma4:26b` corrupted 1/12 (8%) here. #259's
  figure came from a `trip-planner#2` fixture that is not in this repository, so
  a new corpus was written for the replay and it is evidently easier. The failure
  *class* reproduced; the *rate* did not. Treat the local number as a floor.
- **Unequal samples.** The local pass is n=12 against the remote n=20; it was
  stopped once the remote result was unambiguous.
- **The corpus is new, not captured.** #259 states a captured corpus exists on
  `main`. It does not — every capture record on the machine used was an
  orchestrator call (`Plan`, `StackChoice`, `ProgressJudgement`,
  `ObjectiveAssessment`), and none was a `WorkerOutput`.
- **`num_ctx` is a request, not a fact, on a remote provider.** Ollama receives
  it and allocates a KV cache; Bedrock serves whatever window the model has. It
  is still carried for all providers because `prompt_budget` trims against it.

### Adding a row

The table is a record, not a conclusion. To add a model:

1. `swarm doctor` with the model configured — `model.schema` must pass first.
2. Replay the same goals through `swarm console`'s `propose_edits` tab, using the
   per-call override so nothing is persisted (§5).
3. Score each reply three ways — schema held, every edited `.py` file parses,
   `pytest` green after applying the edits to a clean copy.
4. Record `n` honestly, and keep `corrupt` to the two classes above.

Rows for `gpt-5.6-luna` and for `gemma4:26b` at a raised `num_ctx` are the two
that would most change the picture. The second needs no credential at all, and is
worth doing first: if a larger local window recovers much of the gap, the
cheapest fix is one environment variable and no provider at all.

## 7. Bedrock gotchas worth knowing before you file a ticket

**The OpenAI models have no EU-resident path.** On the `bedrock-runtime`
endpoint, for every EU region, the GPT-5.6 model cards show In-Region
unsupported, EU Geo (EU CRIS) unsupported, and Global cross-region the only
option. The only Geo inference IDs that exist anywhere are `us.openai.*` and
`in.openai.*` — `eu.openai.*` is not a provisioned profile and returns
`ValidationException: The provided model identifier is invalid`. Global CRIS
routes to any commercial region worldwide, which AWS documents as the option for
"when there are no residency constraints". If you are bound to EU-only
processing, the OpenAI models on Bedrock are not currently available to you at
any permission level.

**Listing the catalogue is a separate permission from calling a model.** This is
why `model.available` reports "not listable" rather than failing, and why
`model.schema` proves the configured model with one real call instead of trusting
a catalogue lookup.

**Marketplace-gated models fail as `AccessDenied`, not as "not enabled".** Some
third-party models on Bedrock require an AWS Marketplace subscription, and a role
without `aws-marketplace:ViewSubscriptions` / `aws-marketplace:Subscribe` cannot
establish one. The error names the missing Marketplace actions, which reads like
a broken model rather than a missing subscription. A single `converse` call is
the cheap way to confirm access before anyone closes a ticket saying it is
enabled.

**Anthropic models on Bedrock take an `eu.` prefix, not `global.`** —
`eu.anthropic.claude-haiku-4-5-20251001-v1:0` is EU-resident;
`global.anthropic.…` is not, and may not be subscribed.

**The builder passes no `max_tokens`.** `_build_bedrock` constructs
`ChatBedrockConverse` without one, so a whole-file `WorkerOutput` relies on the
model's default output limit. It did not bite on the corpus above, but a reply
truncated by that limit would surface as a parse failure — indistinguishable
from the corruption this document is about.
