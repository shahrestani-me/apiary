"""Which model a role actually gets, and where that answer came from.

"Model" here means the LLM, as it does everywhere else in this tree.

`llm.py` knows how to *build* a model from a `ModelSpec` (ADR 0006). This
module decides *which* spec, from four sources in a fixed order, and says out
loud which one won. The two are separate because they fail differently: a bad
spec is a `ConfigError` an operator fixes in their environment, and a bad build
is a provider refusing a credential.

## The order, highest first

1. **An explicit `spec=` argument.** The console firing one call at a chosen
   model. Not persisted, and affects nothing else.
2. **`SWARM_*` environment.** CI, `swarm run`, and every worker container.
3. **The console-saved default.** A file, because a running process cannot
   rewrite its own environment.
4. **Built-in Ollama defaults.** Zero-config still means local.

**Environment above the saved default is the decision to get right.** A worker
container receives `SWARM_WORKER_MODEL` through `containers.manager.
INHERITED_ENV`, and a CI job sets it explicitly; neither may be silently
overridden by whatever an operator last clicked in the console. Putting the
environment above the file makes the existing container path keep winning *by
construction* rather than by anybody remembering to check.

**Ollama stays rank 4.** A clone with nothing configured runs fully local, and
`SWARM_ORCHESTRATOR_MODEL=gemma4:31b` still means what it has always meant.

## Why the store is a file and not `Settings`

`Settings` is `frozen=True` and `SETTINGS` is built from the environment at
import (`config.py:547`). A console that wants to change the default therefore
cannot write an environment variable and cannot mutate the singleton - it needs
somewhere to put the answer that survives a restart and is read at *resolution*
time rather than at import. One small JSON document, beside `projects.sqlite`,
which is the precedent `console_projects.py` set for exactly this.

## Saying which rung won

Every resolution carries its source, and the first one per role in a process is
printed. "Which model did this run actually use, and why" is then answerable
from a log rather than by reasoning about four rungs and someone's shell - and
the rung that is hardest to guess from the outside, a saved default from a
console session last week, is precisely the one that would otherwise be
invisible.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .artifacts import ArtifactsError, artifacts_root, write_json
from .config import (
    DEFAULT_ORCHESTRATOR_MODEL,
    DEFAULT_ORCHESTRATOR_NUM_CTX,
    DEFAULT_WORKER_MODEL,
    DEFAULT_WORKER_NUM_CTX,
    SETTINGS,
    ConfigError,
    Settings,
)
from .llm import PROVIDERS, ModelSpec

__all__ = [
    "ARGUMENT",
    "BUILT_IN",
    "ENVIRONMENT",
    "ROLES",
    "SAVED",
    "ModelStore",
    "Resolution",
    "as_dict",
    "from_dict",
    "parse_model",
    "parse_options",
    "resolve",
    "store",
]

#: The two roles, and the environment variables each reads. A table rather than
#: an f-string over the role name, so that the variables an operator exports are
#: greppable in the source they are read from.
ROLES: dict[str, dict[str, str]] = {
    "orchestrator": {
        "model": "SWARM_ORCHESTRATOR_MODEL",
        "options": "SWARM_ORCHESTRATOR_MODEL_OPTIONS",
        "num_ctx": "SWARM_ORCH_CTX",
    },
    "worker": {
        "model": "SWARM_WORKER_MODEL",
        "options": "SWARM_WORKER_MODEL_OPTIONS",
        "num_ctx": "SWARM_WORKER_CTX",
    },
}

#: The four rungs, named. Constants because they are printed, stored in a
#: capture record and asserted on by name.
ARGUMENT = "argument"
ENVIRONMENT = "environment"
SAVED = "saved default"
BUILT_IN = "built-in"

#: Beside `projects.sqlite`, for the same reason that lives where it does.
STORE_NAME = "models.json"

#: Bumped when a field changes meaning, never when one is added - the same
#: contract `capture.SCHEMA_VERSION` carries, and for the same reason: this
#: file is written once and read by every later version.
STORE_SCHEMA = 1


# --------------------------------------------------------------------------
# Reading a spec out of a string
# --------------------------------------------------------------------------


def parse_model(text: str) -> tuple[str, str]:
    """`bedrock:gpt-5.6-luna` -> `(bedrock, gpt-5.6-luna)`. And `gemma4:31b` -> ollama.

    **A bare name still means Ollama**, which is the whole of this module's
    backward compatibility: nobody's shell profile or compose file breaks.

    The ambiguity is real and is resolved by looking the prefix up rather than
    by guessing. Ollama model names contain colons - `gemma4:31b` is the
    default - so "split on the first colon" alone would read `gemma4` as a
    provider. The rule is therefore: a prefix that names a *registered
    provider* is a qualifier, and anything else is part of the model name.
    `ollama:gemma4:31b` is accepted too, for an operator who would rather be
    explicit than rely on that rule.
    """
    raw = (text or "").strip()
    if not raw:
        return "", ""
    prefix, separator, rest = raw.partition(":")
    if separator and prefix in PROVIDERS and rest.strip():
        return prefix, rest.strip()
    return "ollama", raw


def parse_options(text: str) -> tuple[tuple[str, str], ...]:
    """`region=eu-west-1,profile=acme` -> the pairs a `ModelSpec` takes.

    A refusal rather than a shrug on anything unparseable. An operator who
    wrote `region eu-west-1` has a model pointed somewhere they did not choose,
    and the value of the check is entirely in it happening before the call.
    """
    pairs: list[tuple[str, str]] = []
    for part in (text or "").split(","):
        item = part.strip()
        if not item:
            continue
        name, separator, value = item.partition("=")
        if not separator or not name.strip():
            raise ConfigError(
                f"model options are `name=value`, comma separated; could not read {item!r}"
            )
        pairs.append((name.strip(), value.strip()))
    return tuple(pairs)


# --------------------------------------------------------------------------
# Reading and writing a spec as JSON
# --------------------------------------------------------------------------


def as_dict(spec: ModelSpec) -> dict[str, Any]:
    """A spec as plain JSON. Safe to write: a spec holds no secret (ADR 0006)."""
    return {
        "provider": spec.provider,
        "model": spec.model,
        "temperature": spec.temperature,
        "num_ctx": spec.num_ctx,
        "options": {name: value for name, value in spec.options},
    }


def from_dict(payload: Mapping[str, Any]) -> ModelSpec:
    """The inverse, refusing anything it cannot honour as written.

    Every field is coerced defensively because this reads a file an operator
    can edit by hand, and because a `models.json` written by a later version
    will one day be read by an earlier one.
    """
    options = payload.get("options") or {}
    if not isinstance(options, Mapping):
        raise ConfigError(f"a saved model's options must be an object, got {type(options).__name__}")
    try:
        return ModelSpec(
            provider=str(payload.get("provider") or "ollama"),
            model=str(payload.get("model") or ""),
            temperature=float(payload.get("temperature", 0.0)),
            num_ctx=int(payload.get("num_ctx", DEFAULT_WORKER_NUM_CTX)),
            options=tuple((str(name), str(value)) for name, value in options.items()),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"a saved model could not be read: {exc}") from exc


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


@dataclass
class ModelStore:
    """The console-saved default per role: one small JSON document.

    `path` is the test seam, exactly as `ProjectStore.path` is - a test hands in
    a path under `tmp_path` and never touches the real file.

    **A read never raises.** This sits on the resolution path, which is on the
    inference path, and a hand-edited or half-written file must cost one
    resolution its rung rather than costing a run. A write does raise: the
    console asked for something and needs to know it did not happen.
    """

    path: Path = field(default_factory=lambda: artifacts_root().parent / STORE_NAME)

    # -- reading ---------------------------------------------------------

    def _document(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            print(f"! models: {self.path} is unreadable ({exc}); ignoring it", file=sys.stderr)
            return {}
        return payload if isinstance(payload, dict) else {}

    def load(self, role: str) -> ModelSpec | None:
        """The saved default for `role`, or `None` if there is not one."""
        saved = self._document().get("roles", {})
        entry = saved.get(role) if isinstance(saved, Mapping) else None
        if not isinstance(entry, Mapping):
            return None
        try:
            return from_dict(entry)
        except ConfigError as exc:
            print(f"! models: the saved {role} model is unusable ({exc}); ignoring it",
                  file=sys.stderr)
            return None

    def all(self) -> dict[str, ModelSpec]:
        """Every saved default, for the console's page."""
        return {role: spec for role in ROLES if (spec := self.load(role)) is not None}

    # -- writing ---------------------------------------------------------

    def save(self, role: str, spec: ModelSpec | None) -> None:
        """Set, or with `None` clear, the saved default for one role.

        Read-modify-write of the whole document rather than a partial update:
        it holds two entries, and the alternative is a merge story for a file
        that will never be big enough to need one.
        """
        if role not in ROLES:
            raise ConfigError(f"unknown role {role!r}; the roles are {', '.join(ROLES)}")
        document = self._document()
        roles = dict(document.get("roles") or {})
        if spec is None:
            roles.pop(role, None)
        else:
            roles[role] = as_dict(spec)
        try:
            write_json(self.path, {"schema": STORE_SCHEMA, "roles": roles})
        except ArtifactsError as exc:
            raise ConfigError(f"could not save the {role} model: {exc}") from exc


#: Process-wide, and created on first use. Kept behind a function for the same
#: reason `capture.recorder` is: the day this needs to be per-thread, the swap
#: is one edit here rather than an audit of every reader.
_STORE: ModelStore | None = None


def store(replacement: ModelStore | None = None) -> ModelStore:
    """The installed store, installing `replacement` first if one is given."""
    global _STORE
    if replacement is not None:
        _STORE = replacement
    if _STORE is None:
        _STORE = ModelStore()
    return _STORE


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    """A spec, and which rung produced it."""

    spec: ModelSpec
    source: str
    #: What to look at to change it: a variable name, a file path, or "".
    detail: str = ""

    @property
    def line(self) -> str:
        """One line, and the answer to "which model did this run use"."""
        where = f" ({self.detail})" if self.detail else ""
        return f"{self.spec.label} from {self.source}{where}"


def _built_in(role: str, settings: Settings | None = None) -> ModelSpec:
    """Rung 4. Ollama, out of `Settings` - which *is* the built-in answer.

    Reading `Settings` rather than the `DEFAULT_*` constants directly is what
    keeps this rung honest. `Settings` folds the environment into its own
    fields, so on the only path that reaches rung 4 - no `SWARM_*_MODEL` set,
    nothing saved - it holds exactly those constants. What it buys is that a
    caller holding a *different* `Settings` gets that one: `swarm doctor` pins
    its settings so a test can provoke a verdict without breaking the machine
    it runs on, and a rung that ignored them would report on a model the doctor
    was not asked about.
    """
    chosen = SETTINGS if settings is None else settings
    if role == "orchestrator":
        return ModelSpec(
            provider="ollama",
            model=chosen.orchestrator_model or DEFAULT_ORCHESTRATOR_MODEL,
            temperature=chosen.orchestrator_temperature,
            num_ctx=chosen.orchestrator_num_ctx or DEFAULT_ORCHESTRATOR_NUM_CTX,
        )
    return ModelSpec(
        provider="ollama",
        model=chosen.worker_model or DEFAULT_WORKER_MODEL,
        temperature=chosen.worker_temperature,
        num_ctx=chosen.worker_num_ctx or DEFAULT_WORKER_NUM_CTX,
    )


def _num_ctx(role: str, env: Mapping[str, str], fallback: int) -> int:
    """`SWARM_*_CTX` if it is *set*, otherwise whatever the rung carried.

    Presence rather than value, because the two questions differ: a saved spec
    carries a window an operator chose in the console, and an exported
    `SWARM_WORKER_CTX` is the same operator choosing again, later, from a shell
    that outranks it. An unparseable value is the fallback rather than an
    error - this is on the inference path, and `Settings` reads the same
    variable the same forgiving way.
    """
    raw = (env.get(ROLES[role]["num_ctx"]) or "").strip()
    if not raw:
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def _from_environment(
    role: str, env: Mapping[str, str], settings: Settings | None = None
) -> Resolution | None:
    """Rung 2, or `None` if this role's model variable is not set.

    Only the *model* variable decides whether this rung applies. An options
    variable alone is not enough: it names no model, and applying it to a spec
    that came from the store would be the environment silently editing the
    saved default rather than replacing it - which is a third behaviour nobody
    asked for and nobody could predict from the documented order.
    """
    names = ROLES[role]
    raw = (env.get(names["model"]) or "").strip()
    if not raw:
        return None
    provider, model = parse_model(raw)
    defaults = _built_in(role, settings)
    spec = ModelSpec(
        provider=provider,
        model=model,
        temperature=defaults.temperature,
        num_ctx=_num_ctx(role, env, defaults.num_ctx),
        options=parse_options(env.get(names["options"], "")),
    )
    return Resolution(spec=spec, source=ENVIRONMENT, detail=names["model"])


def resolve(
    role: str,
    spec: ModelSpec | None = None,
    *,
    env: Mapping[str, str] | None = None,
    sink: ModelStore | None = None,
    settings: Settings | None = None,
) -> Resolution:
    """Which model `role` gets, and why. The four rungs, highest first.

    `env`, `sink` and `settings` are the test seams. All three default to the
    real thing, so every production caller passes none of them.
    """
    if role not in ROLES:
        raise ConfigError(f"unknown role {role!r}; the roles are {', '.join(ROLES)}")
    environment = os.environ if env is None else env

    if spec is not None:
        return Resolution(spec=spec, source=ARGUMENT)

    from_env = _from_environment(role, environment, settings)
    if from_env is not None:
        return from_env

    saved = (sink or store()).load(role)
    if saved is not None:
        # The window is re-read even here: see `_num_ctx`.
        window = _num_ctx(role, environment, saved.num_ctx)
        return Resolution(
            spec=saved if window == saved.num_ctx else ModelSpec(
                provider=saved.provider,
                model=saved.model,
                temperature=saved.temperature,
                num_ctx=window,
                options=saved.options,
            ),
            source=SAVED,
            detail=str((sink or store()).path),
        )

    return Resolution(spec=_built_in(role, settings), source=BUILT_IN)


#: Roles already announced in this process. A set rather than a flag because
#: the two roles resolve independently and an operator wants to see both.
_ANNOUNCED: set[str] = set()


def announce(role: str, resolution: Resolution) -> None:
    """Print the resolution once per role per process.

    Once, because this is called from the model factory and a run makes
    hundreds of calls. On stdout, beside every other line a run prints, rather
    than through a logger this project does not have.
    """
    if role in _ANNOUNCED:
        return
    _ANNOUNCED.add(role)
    print(f"· {role} model: {resolution.line}")


def forget_announcements() -> None:
    """Test seam: let the next `announce` print again."""
    _ANNOUNCED.clear()
