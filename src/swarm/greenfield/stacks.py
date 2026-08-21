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
#: Everything a stack's bootstrap must be told beyond "write these files",
#: spliced into the goal the worker is handed.
#:
#: **Named for the stack and not for dependencies**, because for one stack it
#: is only about dependencies and for another it is not. It began as a single
#: sentence - "no dependencies beyond the language's standard library" - which
#: is the correct instruction for Python and for `node --test` and an
#: impossible one for React: react and react-dom *are* dependencies, and a
#: model obeying the rule literally would write no React at all. Whatever the
#: next stack needs said belongs here, whether or not it is about packages.
#:
#: React's entry names the packages, because that list is the whole contract
#: with the image: a worker has no route to a registry (docs/security.md §3),
#: so a package outside this set is not slow to add, it is unobtainable.
#:
#: **The `@testing-library/jest-dom` import line is not decoration.**
#: Installing that package supplies nothing on its own - `expect` learns
#: `toBeInTheDocument` only once the registration module has run - and
#: `toBeInTheDocument()` is what a model writes whether or not anything told it
#: to. So the package is in the image *and* the prompt demands the import. One
#: without the other produces exactly the failure the package is there to
#: prevent, on a project that is otherwise correct.
#:
#: **It has to be an import in the test file; `setupFiles` does not work.**
#: Measured: `setupFiles: ["@testing-library/jest-dom/vitest"]` resolves to
#: `/node_modules/@testing-library/jest-dom/dist/vitest.mjs`, which Vite then
#: reads as a root-*relative URL* under the project root and fails to load -
#: "Does the file exist?", about a file that does. It is the second consequence
#: of putting the toolchain at `/` (see `Dockerfile.worker.react` on why
#: `NODE_PATH` was not an option either): a bare specifier inside a source file
#: resolves by walking parent directories and works, an absolute path handed to
#: Vite's config does not.
#:
#: The package list hangs off "already installed", never off a prohibition.
#: Written the other way - "do not add any others: react, react-dom, ..." - the
#: colon binds to the nearest clause, and a plausible reading is that React
#: itself is the forbidden thing.
STACK_RULE: dict[str, str] = {
    "python": "Use no dependencies beyond the language's standard library.",
    "node": "Use no dependencies beyond the language's standard library.",
    "react": (
        "This is React on the web, not React Native. "
        "These packages are already installed and are the only ones available: "
        + ", ".join(package_names())
        + ". There is no network, so do not import or declare anything else. "
        "package.json must set \"type\": \"module\" and list exactly those "
        "packages. vitest.config.js must export a config that uses the "
        "@vitejs/plugin-react plugin and sets test.environment to \"jsdom\" and "
        "test.globals to true. Every test file must begin with the line "
        "import \"@testing-library/jest-dom/vitest\"; - without it, matchers "
        "such as toBeInTheDocument() do not exist - and must render components "
        "with @testing-library/react. "
        # Said outright, because "these are the only packages available" was not
        # read as forbidding it (#293): a react run was planned with "Define a
        # TypeScript interface for Todo items" and "Initialize a React project
        # with Vite, TypeScript, and Vitest", neither of which any worker in
        # this image could ever satisfy. A prohibition costs one sentence; an
        # unbuildable plan costs the whole run.
        "There is no TypeScript and no build step: write plain JavaScript and "
        "JSX in .js and .jsx files only, never .ts or .tsx, and do not add a "
        "bundler, a compiler or a type checker."
    ),
}
