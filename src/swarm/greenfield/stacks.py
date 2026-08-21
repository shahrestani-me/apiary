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

import json

#: `bootstrap.STACK_VERIFY["react"]`, repeated because the import may only go
#: the other way (see the module docstring), and pinned by
#: `test_the_react_manifests_test_script_is_the_gate`.
STACK_VERIFY_REACT = "vitest run"

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
#: Major-only pins (`react@19`, not `react@19.2.8`), because a lockfile is what
#: pins a patch version and there is not one. A tighter spelling here would
#: imply a reproducibility this design does not have, and docs/security.md
#: states the residual drift rather than hiding it. `lucide-react@0` is the one
#: that looks wrong and is not: it has never left 0.x, so the major *is* zero.
#:
#: **Everything here must be in the worker image, because a worker cannot
#: install.** Its only route out is the egress proxy's static allowlist and no
#: registry is on it, so this list is not a suggestion the model resolves - it is
#: the entire universe of packages a generated project may import. That is why it
#: stays short, and why every addition costs an image rebuild.
#:
#: ## What #295 added, and why each one
#:
#: The stack generated plain JSX with a hand-written stylesheet, which is not how
#: React is written any more - so a brief asking for something that looks good
#: got whatever CSS the model invented that day. The target is what Lovable
#: produces: TypeScript, Tailwind, and components in the shadcn/ui idiom, which
#: is also the idiom current models write best because it is what the ecosystem
#: they learned from is full of.
#:
#: **TypeScript and `@types/*`.** Both were already *reachable* - vite transpiles
#: `.tsx` through esbuild, measured green before this landed - and neither was
#: offered, so the generator kept emitting 2021. The types packages are listed
#: explicitly because their absence is the one kind that is invisible until a
#: human opens the project: vite does not need them and `tsc` is not in the gate.
#:
#: **Tailwind 4, not 3.** Its config is CSS-first: `@import "tailwindcss";` in one
#: stylesheet replaces `tailwind.config.js`, `postcss.config.js` and
#: `autoprefixer` - three files a model gets subtly wrong, for one line it
#: cannot. `@tailwindcss/vite` is what makes that work under vite and vitest.
#:
#: **shadcn/ui is not here and cannot be.** It is not a package; it is a CLI that
#: copies component source into a project, and running it needs the registry a
#: worker is denied. What *is* installable is everything its components are built
#: from: `class-variance-authority` for variants, `clsx` and `tailwind-merge` for
#: the `cn` helper every shadcn file imports, the Radix primitives its interactive
#: components wrap, and `lucide-react` for icons. So the stack ships the
#: vocabulary and the generator writes the components. That is a bet, and the
#: defensible one: a model asked for a button in a project that already has
#: `src/lib/utils.ts` and cva writes the shadcn shape without being told to.
#:
#: Three Radix primitives, not thirty. `react-slot` is load-bearing - `asChild`
#: is in every shadcn component signature - and dialog and label cover the two
#: interactions a generated form actually needs. The rest are one image rebuild
#: away, and thirty unused packages is a slower pull for every task in the
#: repository forever.
#:
#: `@testing-library/jest-dom` is in the set for a reason worth naming: it is the
#: assertion vocabulary every React example on the internet uses, so a model
#: writes `toBeInTheDocument()` whether or not the prompt allows it, and a missing
#: matcher is a red gate on a project that is otherwise correct.
#: `@testing-library/user-event` is here for the same reason one layer up - a
#: generated test clicks things, and `fireEvent` is not what the docs show.
REACT_TOOLCHAIN: tuple[str, ...] = (
    "react@19",
    "react-dom@19",
    "@types/react@19",
    "@types/react-dom@19",
    "typescript@5",
    "vite@6",
    "@vitejs/plugin-react@4",
    "tailwindcss@4",
    "@tailwindcss/vite@4",
    "class-variance-authority@0.7",
    "clsx@2",
    "tailwind-merge@3",
    "lucide-react@0",
    "@radix-ui/react-slot@1",
    "@radix-ui/react-dialog@1",
    "@radix-ui/react-label@2",
    "vitest@2",
    "jsdom@25",
    "@testing-library/react@16",
    "@testing-library/jest-dom@6",
    "@testing-library/user-event@14",
)


#: Which of `REACT_TOOLCHAIN` a built application actually ships, as opposed to
#: what only builds or tests it. The split is what makes `dependencies` and
#: `devDependencies` mean anything: `npm ci --omit=dev` on a deploy must still
#: get react, the class-merging helpers every generated component imports, and
#: the Radix primitives they render - and must not need vite, vitest, jsdom, the
#: type stubs or the compiler.
#:
#: `tailwindcss` is deliberately on the dev side: Tailwind 4 runs as a build
#: plugin and its output is the CSS that ships, not a runtime import.
RUNTIME_PACKAGES: frozenset[str] = frozenset(
    {
        "react",
        "react-dom",
        "class-variance-authority",
        "clsx",
        "tailwind-merge",
        "lucide-react",
        "@radix-ui/react-slot",
        "@radix-ui/react-dialog",
        "@radix-ui/react-label",
    }
)


def react_manifest(name: str, specs: tuple[str, ...] = REACT_TOOLCHAIN) -> str:
    """The `package.json` for a generated React project, rendered from the pins.

    **Why apiary writes this file rather than asking a model to.** It used to be
    in `bootstrap.BOOTSTRAP_FILES` with the stack rule telling the worker to
    "list exactly those packages", which is a prompt asking a model to echo
    twenty-one exact version pins back. Measured on run
    `to-do-react-generated-app-20260821-151111-95rff7`: the model wrote seven of
    them, at `react@^18`, `vitest@^1` and `jsdom@^23` - its priors, not the
    list - and left out `@testing-library/user-event` entirely. A later task
    imported `user-event`, the worker resolved it from the toolchain baked into
    its image and passed, and CI - which installs what the project declares -
    could not resolve it and went red. The worker wrote correct code and was
    told it had failed.

    Nothing about that is a model being careless: a version pin is not a
    judgment, apiary knows the list exactly, and a generated file cannot drift
    from `REACT_TOOLCHAIN` the way a prompt's output can. What the model is
    still asked to do is the part that *is* a judgment - adding a dependency the
    project needs - and it does that by editing this file, which is an ordinary
    editable source file once it exists.

    `"type": "module"` because a modern React project is ESM and vite's config
    is loaded as one; `test` is spelled the same as `STACK_VERIFY["react"]` so a
    human running `npm test` runs the gate rather than something adjacent to it.
    """
    runtime: dict[str, str] = {}
    development: dict[str, str] = {}
    for spec in specs:
        package, _, version = spec.rpartition("@")
        target = runtime if package in RUNTIME_PACKAGES else development
        target[package] = f"^{version}"
    manifest = {
        "name": name,
        "private": True,
        "version": "0.0.0",
        "type": "module",
        "scripts": {
            "dev": "vite",
            "build": "vite build",
            "preview": "vite preview",
            "test": STACK_VERIFY_REACT,
        },
        "dependencies": dict(sorted(runtime.items())),
        "devDependencies": dict(sorted(development.items())),
    }
    # Sorted keys off, so the order above is the order a human reads; a trailing
    # newline because every other generated file has one and git prefers it.
    return json.dumps(manifest, indent=2) + "\n"


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
        "This is React on the web, not React Native, and it is written the way "
        "the ecosystem writes React today: TypeScript, Tailwind utility classes, "
        "and components in the shadcn/ui idiom. "
        #
        # The package list hangs off "already installed", never off a
        # prohibition. Written the other way - "do not add any others: react,
        # react-dom, ..." - the colon binds to the nearest clause, and a
        # plausible reading is that React itself is the forbidden thing.
        "These packages are already declared in `package.json` and installed: "
        + ", ".join(package_names())
        + ". If the work genuinely needs another package, add it to the "
        "`dependencies` or `devDependencies` of `package.json` - your gate and "
        "CI both install from that file, so declaring it there is what makes it "
        "available in both. Do not run a CLI that writes source for you: there "
        "is no `npx shadcn`; write the component yourself. "
        #
        # TypeScript, stated as what is actually absent. #293 is why: this said
        # "never .ts or .tsx", which was false - vite transpiles both through
        # esbuild - and which outranked tasks whose `## Files` declared `.tsx`,
        # so every edit was refused for being undeclared.
        "Write TypeScript: `.ts` and `.tsx`, with real prop types and no `any`. "
        "It is transpiled by vite through esbuild and never type-checked - there "
        "is no `tsc` in the gate - so type errors will not be reported and are "
        "yours to avoid. "
        #
        # Tailwind 4 is CSS-first, which removes the three config files a model
        # gets subtly wrong. Saying so is what stops it writing them anyway.
        "Style with Tailwind utility classes in `className`. Tailwind 4 needs no "
        "config file: one stylesheet contains `@import \"tailwindcss\";` and the "
        "vite config uses the `@tailwindcss/vite` plugin. Never write a "
        "`tailwind.config.js`, a `postcss.config.js` or a plain `.css` file of "
        "your own rules, and never import a stylesheet that is not in your file "
        "list. "
        #
        # The shadcn idiom, named concretely enough to reproduce. This is the
        # "Lovable pattern" half of #295: the packages alone do not produce it.
        "Build components in the shadcn/ui idiom. Merge classes with the `cn` "
        "helper from `src/lib/utils.ts` - `twMerge(clsx(inputs))` - so a caller's "
        "`className` can override yours. Express variants with `cva` from "
        "class-variance-authority and type the props with `VariantProps`. Accept "
        "`asChild` through Radix's `Slot` where a component might need to render "
        "as something else. Use `lucide-react` for icons. Prefer semantic, "
        "accessible markup - a real `<button>`, a `<label>` tied to its input - "
        "over a styled `<div>`, because the tests below query by role. "
        #
        # The gate's mechanics, unchanged and still load-bearing.
        "`package.json` already exists, sets \"type\": \"module\" and declares "
        "the packages above; you do not need to create or rewrite it. The vite "
        "config must register the `@vitejs/plugin-react` and "
        "`@tailwindcss/vite` plugins and set `test.environment` to \"jsdom\" and "
        "`test.globals` to true. Every test file must begin with the line "
        "import \"@testing-library/jest-dom/vitest\"; - without it, matchers "
        "such as toBeInTheDocument() do not exist - must render with "
        "@testing-library/react, and should query by role or label rather than "
        "by test id."
    ),
}
