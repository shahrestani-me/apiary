"""Real test-runner failure output, one sample per language.

Kept as fixtures rather than inline strings for the reason #93 gives: these are
claims about what four other tools print, made by someone reading their docs,
and they will be wrong somewhere. A fixture can be corrected against reality by
pasting over it; a regex buried in a parametrise list cannot.

Every sample is trimmed to the lines that carry a path, and every path is one a
worker could plausibly have been given in a `## Files` set. Nothing here is
absolute, and nothing goes through `node_modules` or `site-packages` - those
belong to `test_a_frame_through_a_dependency_is_not_this_tasks_fault`, which is
about the filter rather than about the language.
"""

from __future__ import annotations

#: pytest. The shape everything else here was measured against.
PYTEST = """\
=================================== FAILURES ===================================
_________________________________ test_adds ____________________________________

    def test_adds():
>       assert add(1, 2) == 4
E       assert 3 == 4

tests/test_calc.py:12: AssertionError
=========================== short test summary info ============================
FAILED tests/test_calc.py::test_adds - assert 3 == 4
"""

#: `node --test`, the default spec reporter. Stdlib since Node 18, which is why
#: #87 chose it as the first non-Python stack: no dependencies, no lockfile, no
#: registry.
NODE_TEST = """\
✖ adds numbers (1.234ms)
  AssertionError [ERR_ASSERTION]: Expected values to be strictly equal:

  3 !== 4

      at TestContext.<anonymous> (test/calc.test.js:5:10)
      at Test.runInAsyncScope (node:async_hooks:203:9)

ℹ tests 1
ℹ fail 1
"""

#: vitest. Names the file twice, in two different shapes, which is why it is a
#: good control for both the summary and the location pattern.
VITEST = """\
 FAIL  src/calc.test.ts > adds numbers
AssertionError: expected 3 to be 4 // Object.is equality

- Expected
+ Received

- 4
+ 3

 ❯ src/calc.test.ts:5:19
      3|
      4| test('adds numbers', () => {
      5|   expect(add(1, 2)).toBe(4)

Test Files  1 failed (1)
"""

#: jest. The summary line only - jest's stack frames are parenthesised the same
#: way node's are.
JEST = """\
FAIL src/calc.test.js
  ● adds numbers

    expect(received).toBe(expected)

      at Object.<anonymous> (src/calc.test.js:5:25)

Tests:       1 failed, 1 total
"""

#: `go test`. The one sample whose path has **no directory**: a package's tests
#: run in that package's directory, so the file is a bare name. `_test.go` is
#: mandatory in Go and is what makes accepting it safe.
GO_TEST = """\
--- FAIL: TestAdd (0.00s)
    calc_test.go:12: Add(1, 2) = 3, want 4
FAIL
FAIL\texample.com/m/internal/calc\t0.002s
"""

#: A failed Go *compile*, which is the other half of `go test` and does carry a
#: qualified path. This is the sample the agreement test uses, because it is the
#: one both extractors can see.
GO_BUILD = """\
# example.com/m/internal/calc
./internal/calc/calc.go:12:2: undefined: helper
FAIL\texample.com/m/internal/calc [build failed]
"""

#: `cargo test`. The panic location is the only path in it, which makes it the
#: sharpest test of the location pattern.
CARGO_TEST = """\
running 1 test
test tests::adds ... FAILED

failures:

---- tests::adds stdout ----
thread 'tests::adds' panicked at src/lib.rs:12:9:
assertion `left == right` failed
  left: 3
 right: 4

failures:
    tests::adds

test result: FAILED. 0 passed; 1 failed; 0 ignored
"""

#: What each sample must yield. Written out rather than derived, so a change to
#: the patterns that quietly stops seeing a language fails here.
EXPECTED: dict[str, tuple[str, ...]] = {
    "pytest": ("tests/test_calc.py",),
    "node": ("test/calc.test.js",),
    "vitest": ("src/calc.test.ts",),
    "jest": ("src/calc.test.js",),
    "go": ("calc_test.go",),
    "go-build": ("internal/calc/calc.go",),
    "cargo": ("src/lib.rs",),
}

SAMPLES: dict[str, str] = {
    "pytest": PYTEST,
    "node": NODE_TEST,
    "vitest": VITEST,
    "jest": JEST,
    "go": GO_TEST,
    "go-build": GO_BUILD,
    "cargo": CARGO_TEST,
}
