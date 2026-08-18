"""Three plain-language questions, and the technical setup proposed from them.

The console's other sites assume their operator can name a checkout path, a
verify command, an `owner/name` repository. A business user cannot, and should
not have to: what they know is what the tool must do, who will use it, and how
they would check it works. This module is the translation layer - it asks
exactly those three questions, composes the answers into one planner-ready
brief, and makes **one** schema-forced model call to propose the two things a
run genuinely needs a decision on: a repository name and a stack. Everything
else in the run form - the verify command, the owner, visibility, the merge
policy - is derived deterministically, because a derived value is one a
non-developer cannot get wrong.

**One model call, mirroring `bootstrap.choose_stack` exactly.** Same
`structured(orchestrator_llm(), Setup)` seam, same `llm=None` injection point
so tests need no Ollama, and the same `prompt_for` rule the console was founded
on: this module exports the exact `(system, human)` pair it sends, and the
console's prompt tab calls it rather than approximating it. The human turn *is*
the composed brief, unchanged - which also means the brief the model saw is the
brief the swarm run will be handed, one artefact with two readers.

**A failed model call falls back rather than raising**, for `choose_stack`'s
documented reason: a proposal that refused to appear because Ollama hiccuped
during a one-line classification would be worse than a default the user can
read and change. The fallback name is `slugify` of the brief's first line, the
stack is `DEFAULT_STACK`, and the reason *says* it is a fallback - the silent
half of a fallback is the part `choose_stack` learned to stop doing.

**The owner is resolved, never asked for.** A business user does not know
which GitHub account repositories go under, and a form field for it would be
answered wrong. `resolve_owner` tries, in order: `APIARY_OWNER` (the explicit
override), the owner half of `GITHUB_REPOSITORY` (free inside Actions), and
GitHub's `/user` endpoint with the token the run needs anyway. `/user` is
known to 503 for fine-grained PATs, so *any* failure there falls through to a
refusal that names the fix - it must not become a raised traceback on the page.

**Nothing is created here.** `propose` returns a dict; the repository, the
issues and the workers only exist after the operator fires the proposal
through the swarm tab. The dict's keys - brief, repo, name, stack, verify,
public, auto_merge, reason - are a contract with the front end, which feeds
them into the run form's fields by those names.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Mapping

from pydantic import BaseModel, Field

from .github.ledger import DEFAULT_STACK, KNOWN_STACKS, slugify

__all__ = [
    "QUESTIONS",
    "IntakeError",
    "Setup",
    "compose_brief",
    "prompt_for",
    "propose",
    "propose_setup",
    "resolve_owner",
]


class IntakeError(ValueError):
    """A refusal a business user can fix, with the fix attached.

    `ValueError`-family like `ConsoleError` and `SwarmRunError`, and carrying
    a `fix` like the latter, so the page renders it as advice rather than as
    a traceback.
    """

    def __init__(self, message: str, *, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix


# --------------------------------------------------------------------------
# The questions, and the brief they become
# --------------------------------------------------------------------------

#: The three questions, as data rather than as `console.Field` instances:
#: `console.py` builds its `Field`s from these, and importing the dataclass
#: here would make the two modules import each other.
#:
#: Three, and in this order, on purpose. "What should it do" is the only
#: required one - it is the objective, and a planner can work from it alone.
#: Users and the done-check sharpen the plan when present, so they are asked,
#: but an empty answer must not block a person who only knows the first.
QUESTIONS: tuple[dict[str, str], ...] = (
    {
        "name": "need",
        "label": "What should this tool do? Describe it the way you would to a colleague.",
        "kind": "area",
        "placeholder": "e.g. keep track of which meeting rooms are free right now",
    },
    {
        "name": "users",
        "label": "Who will use it, and what must they be able to do?",
        "kind": "area",
        "placeholder": "optional - e.g. office staff, who need to see and book a room",
    },
    {
        "name": "done",
        "label": "How will you know it works? What would you check first?",
        "kind": "area",
        "placeholder": "optional - e.g. book a room, then see it marked as taken",
    },
)

#: How each answer is labelled in the brief. Plain sentences rather than field
#: names, because the brief's readers are the planner model and, later, a human
#: scrolling a GitHub issue - neither should meet a key called `done`.
LABELS: dict[str, str] = {
    "need": "The need",
    "users": "Who will use it",
    "done": "How we will know it works",
}


def compose_brief(values: Mapping[str, str]) -> str:
    """The three answers as one planner-ready brief: labelled paragraphs,
    empty answers skipped rather than rendered as empty headings.

    The composed text is the single artefact downstream - `propose_setup`'s
    human turn and the objective the fired run plans from - so it is composed
    once, here, and never re-derived from the raw answers.
    """
    need = str(values.get("need") or "").strip()
    if not need:
        raise IntakeError(
            "say what the tool should do - it is the one question that cannot be skipped",
            fix="fill in the first answer; one or two plain sentences are enough",
        )
    sections = [f"{LABELS['need']}: {need}"]
    for key in ("users", "done"):
        answer = str(values.get(key) or "").strip()
        if answer:
            sections.append(f"{LABELS[key]}: {answer}")
    return "\n\n".join(sections)


# --------------------------------------------------------------------------
# The one model call
# --------------------------------------------------------------------------


class Setup(BaseModel):
    """The two decisions a run needs that a business user cannot make.

    A schema rather than free text for `StackChoice`'s reason: `structured`
    hands it to Ollama's `format`, which constrains decoding, so a name with
    spaces in it or a fourth stack is not something to validate away - it is
    something the decoder cannot emit. (The name is still slugified on the way
    out, because a schema description is advice and a repository name is not.)
    """

    name: str = Field(
        description=(
            "a short kebab-case repository name: two to four lowercase words "
            "joined by hyphens, naming what the tool is, e.g. room-booker"
        )
    )
    stack: str = Field(
        description="one of: python, node, react. react means React on the web, not React Native"
    )
    reason: str = Field(
        default="", description="one short sentence a non-developer can read"
    )


SYSTEM = """You turn a plain-language description of a tool into a technical setup.

Propose:
- name: a short kebab-case repository name - two to four lowercase words joined
  by hyphens, naming what the tool is, e.g. room-booker.
- stack: exactly one of python, node, react.
  - react means React on the web: anything with a user interface, a dashboard,
    a page, a form, a visualisation, something a person looks at in a browser.
  - node means a JavaScript program with no user interface.
  - python means a service, an API, a CLI, a library, a script, a data
    pipeline - anything whose users are other programs or a terminal.
  When the description does not say and could be either, prefer python.
- reason: one short sentence, readable by someone who does not program.

Return JSON only."""


def prompt_for(brief: str) -> tuple[str, str]:
    """The exact `(system, human)` pair `propose_setup` sends.

    The human turn is the composed brief, unchanged - `bootstrap.prompt_for`'s
    property, and the one that lets the console's prompt tab show precisely
    what will be fired rather than an approximation of it.
    """
    return SYSTEM, brief


def _fallback_name(brief: str) -> str:
    """A repository name derived from the brief when the model cannot be asked.

    The brief's first line minus its label, slugified in `PlannedTask.id`'s
    alphabet and cut short - a repository name, not a sentence.
    """
    first = brief.strip().splitlines()[0] if brief.strip() else ""
    prefix = f"{LABELS['need']}:"
    if first.startswith(prefix):
        first = first[len(prefix):]
    return slugify(first, limit=40) or "new-project"


def propose_setup(brief: str, *, llm: object | None = None) -> Setup:
    """Name, stack and a one-line reason for this brief. One model call.

    `llm=None` reaches the real orchestrator through the same
    `structured(orchestrator_llm(), Setup)` seam `choose_stack` uses, so tests
    inject a double and need no Ollama. Failures fall back rather than raise -
    `choose_stack`'s philosophy, applied whole: named on stderr, defaulted to
    something the user can read and change, and the reason on the page *says*
    it is a fallback so a dead Ollama and a real proposal are distinguishable.
    """
    model = llm
    if model is None:
        from .llm import orchestrator_llm, structured

        model = structured(orchestrator_llm(), Setup)

    system, human = prompt_for(brief)
    try:
        answer = model.invoke([("system", system), ("human", human)])
    except Exception as exc:  # noqa: BLE001 - local model failures are varied
        print(
            f"! setup proposal fell back after {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return Setup(
            name=_fallback_name(brief),
            stack=DEFAULT_STACK,
            reason=(
                f"fallback: the model could not be asked ({type(exc).__name__}), "
                f"so the name comes from your first sentence and the stack "
                f"defaults to {DEFAULT_STACK}"
            ),
        )

    name = slugify(str(getattr(answer, "name", "") or ""), limit=40) or _fallback_name(brief)
    stack = str(getattr(answer, "stack", "") or "").strip().casefold()
    reason = str(getattr(answer, "reason", "") or "").strip()
    if stack not in KNOWN_STACKS:
        # `choose_stack`'s second fallback: the model answered, and answered
        # outside the vocabulary `format` was supposed to constrain it to.
        print(
            f"! setup proposal fell back to {DEFAULT_STACK}: the model answered "
            f"{stack or '(nothing)'!r}, which is not one of {sorted(KNOWN_STACKS)}",
            file=sys.stderr,
        )
        stack = DEFAULT_STACK
    return Setup(name=name, stack=stack, reason=reason)


# --------------------------------------------------------------------------
# The owner, resolved rather than asked for
# --------------------------------------------------------------------------


def resolve_owner() -> str:
    """The GitHub account new repositories go under.

    In order: `APIARY_OWNER`, because an explicit answer beats every guess;
    the owner half of `GITHUB_REPOSITORY`, which Actions sets for free; and
    finally GitHub's `/user` with the token a run needs anyway. The last step
    is best-effort on purpose - `/user` returns 503 for fine-grained PATs, and
    a token can be org-scoped with no user behind it - so *any* failure there
    falls through to the refusal, which names the one-line fix instead of
    surfacing a network traceback to someone who cannot act on one.
    """
    owner = os.environ.get("APIARY_OWNER", "").strip()
    if owner:
        return owner

    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if "/" in repository:
        head = repository.split("/", 1)[0].strip()
        if head:
            return head

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        request = urllib.request.Request(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "apiary-console",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                login = str(json.load(response).get("login") or "").strip()
            if login:
                return login
        except Exception:  # noqa: BLE001 - fine-grained PATs 503 here; fall through
            pass

    raise IntakeError(
        "cannot tell which GitHub account new repositories should go under",
        fix="export APIARY_OWNER=<your GitHub account>",
    )


# --------------------------------------------------------------------------
# The site's run
# --------------------------------------------------------------------------


def propose(values: Mapping[str, str]) -> dict[str, str]:
    """The intake site's `run`: everything the swarm-run form needs, by name.

    The keys are a contract with the front end - it copies them into the run
    form's fields verbatim - so they change together with `app.js` or not at
    all. `verify` comes from `STACK_VERIFY` because the stack's falsified
    default gate is exactly what a run with no `--verify` would get; `public`
    and `auto_merge` are "1" because a person who cannot name a stack is not
    the person to quiz about repository visibility or merge policy, and both
    stay editable on the run form for whoever is.

    The owner is resolved *before* the model is consulted: its refusal is
    instant and the model call is not, and a two-minute wait that ends in
    "export APIARY_OWNER" would be the console's founding complaint, reborn.
    """
    brief = compose_brief(values)
    owner = resolve_owner()
    setup = propose_setup(brief)

    from .greenfield.bootstrap import STACK_VERIFY

    return {
        "brief": brief,
        "repo": f"{owner}/{setup.name}",
        "name": setup.name,
        "stack": setup.stack,
        "verify": STACK_VERIFY[setup.stack],
        "public": "1",
        "auto_merge": "1",
        "reason": setup.reason,
    }
