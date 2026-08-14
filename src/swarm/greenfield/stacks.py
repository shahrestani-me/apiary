"""Per-stack facts that more than one generator needs, and that no artefact owns.

Almost every per-stack table in this codebase lives with the thing that reads
it - `BOOTSTRAP_FILES` and `STACK_VERIFY` with the project generator,
`CI_SETUP` with the repository generator, `DEFAULT_STACK_IMAGES` with the
dispatcher, `GENERATED_FILES` with the ledger. That is the convention and it is
a good one.

This module is for the exception: a fact that **three** artefacts have to
agree on and that none of them can be the source of truth for, because one of
them is not Python.

`provision.py`'s note on why `ProvisionPlan.stack` is a plain string is the
constraint that produced this file: the module that generates the *repository*
must not import the module that generates the *project inside it*. So the
shared constant cannot live in `bootstrap.py`, and duplicating it into
`provision.py` would be exactly the drift the test exists to catch.
"""

from __future__ import annotations

#: The React toolchain, pinned, in the one place it is written down.
#:
#: **Two copies have to agree, and only one of the two directions needs a
#: test.** `greenfield.provision.CI_SETUP["react"]` interpolates this tuple, so
#: the generated workflow cannot drift from it by construction.
#: `Dockerfile.worker.react` cannot import Python and therefore repeats the
#: versions, which is the copy
#: `test_the_react_toolchain_is_pinned_identically_everywhere` exists for. Same
#: shape as `security.py` / `compose.yaml` / `test_security.py`, for the same
#: reason: the drift is silent, and the failure it produces is a red CI run on
#: a green worker - the one result that makes the whole gate untrustworthy.
#:
#: **Why the workflow installs at all**, rather than consuming the image: the
#: worker gets its toolchain from its image and a GitHub runner cannot. That is
#: the residual gap #106 could not close - `npm ci` needs a lockfile, producing
#: a lockfile needs the registry a worker is denied (docs/security.md §3), and
#: #105 shipped the mechanism for *committing* a generated lockfile, not for
#: generating one without a network. So CI runs `npm install` and resolves
#: inside these ranges independently.
#:
#: Major-only pins (`react@18`, not `react@18.3.1`), because a lockfile is what
#: pins a patch version and there is not one. A tighter spelling here would
#: imply a reproducibility this design does not have, and docs/security.md
#: states the residual drift rather than hiding it.
#:
#: `@testing-library/jest-dom` is in the set for a reason worth naming: it is
#: the assertion vocabulary every React example on the internet uses, so a
#: model writes `toBeInTheDocument()` whether or not the prompt allows it, and
#: a missing matcher is a red gate on a project that is otherwise correct.
REACT_TOOLCHAIN: tuple[str, ...] = (
    "react@18",
    "react-dom@18",
    "vitest@2",
    "@vitejs/plugin-react@4",
    "jsdom@25",
    "@testing-library/react@16",
    "@testing-library/jest-dom@6",
)


def package_names(specs: tuple[str, ...] = REACT_TOOLCHAIN) -> tuple[str, ...]:
    """`react@18` -> `react`, and `@vitejs/plugin-react@4` -> `@vitejs/plugin-react`.

    `rpartition`, not `partition`: a scoped package name starts with the same
    `@` the version is separated by, and splitting on the first one turns every
    scoped package into an empty string.
    """
    return tuple(spec.rpartition("@")[0] for spec in specs)
