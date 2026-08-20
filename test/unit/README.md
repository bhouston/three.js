## Setup

- Execute `npm install` from the root folder

## Run

- `npm run test-unit` — runs the full suite once (both the plain-Node lane and
  the real-Chromium lane, see below).
- `npm run test-unit-watch` — same, but in watch mode: only the tests
  affected by what you changed re-run, live, as you edit.
- `npm run test-unit-node` / `npm run test-unit-browser` — run just one lane.

Tests live under `test/unit/src/` (mirrors `src/`) and `test/unit/addons/`
(mirrors `examples/jsm/`). A file runs in one of two lanes, chosen by its
name:

- `X.js` — the default. Runs in plain Node (via [Vitest](https://vitest.dev)).
  Fast, no browser involved.
- `X.browser.js` — runs in a real headless Chromium (via Vitest's
  [browser mode](https://vitest.dev/guide/browser/), using Playwright).
  Reserved for tests that genuinely need a DOM/canvas/GPU API Node doesn't
  have (e.g. `createImageBitmap`).

See `VITEST_MIGRATION.md` in this folder for the full set of conventions and
the QUnit → Vitest conversion playbook (kept for reference / future test
files).

See [Installation](https://threejs.org/docs/#manual/introduction/Installation) for more information.

## Troubleshooting

An error that says "No test found in suite" for a file with no failures
listed usually means the file's `describe()` block has zero `test()` calls
in it — either a mistake, or (for an intentionally-empty placeholder) add a
`test.todo( 'no tests yet' )` inside it.

An error that indicates "no tests were found" for the whole run usually means
an import statement couldn't be resolved — check for a typo in the import
path, or in the `@src`/`@test-utils` alias usage.

## Debugging

To debug a test, add `debugger;` to the test code and run Vitest with
`--inspect-brk`, or use your editor's Vitest integration to run/debug a
single test directly.
