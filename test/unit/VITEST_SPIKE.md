# Vitest spike (QUnit -> Vitest migration)

Status: **spike / proof of concept**. Nothing existing was touched — the QUnit
suite (`npm run test-unit`, `npm run test-unit-addons`) and CI still run
exactly as before. This is purely additive: new config + two converted files.

## What's here

- `vitest.config.js` (repo root) — two vitest "projects":
  - **`unit-node`** — plain Node, matches `test/unit/src/**/*.js` other than
    the exclusions below. This is the default lane. Fast (no browser boot),
    watch-mode-friendly.
  - **`unit-browser`** — real Chromium via Playwright (vitest browser mode),
    matches `test/unit/src/**/*.browser.js`.
  - Both exclude the old `*.tests.js` QUnit files, and the node lane also
    excludes `*.browser.js` so a file only runs in one lane.
  - `resolve.alias` defines `@src` -> `src/` and `@test-utils` ->
    `test/unit/utils/`, so imports don't need `../../../../`.
- `test/unit/vitest-setup.js` — `expect()` matchers with the same names/
  semantics as the custom QUnit asserts in `test/unit/utils/qunit-utils.js`
  (`toNumEqual`, `toEqualKey`, `toSmartEqual`), so converting a file that uses
  them is a mechanical rename, not a rewrite.
- Two converted files, proving both lanes end-to-end:
  - `test/unit/src/math/Vector3.js` (node lane) — 48 tests, runs in
    ~190ms cold (incl. Vite transform + import), vs. the QUnit version which
    only runs as part of booting all of `UnitTests.html` in headless Chrome.
  - `test/unit/src/loaders/ImageBitmapLoader.browser.js` (browser lane) —
    5 tests, runs in real Chromium in ~800ms cold. Notably this version is
    *shorter* than the QUnit original: the QUnit/Node version has to suppress
    a `createImageBitmap() not supported` console warning on every test
    because Node doesn't have `createImageBitmap`; running in a real browser,
    that workaround simply isn't needed anymore.

Run them:
```
npm run test-unit-vitest          # both lanes
npm run test-unit-vitest-node     # node lane only
npm run test-unit-vitest-browser  # browser lane only
npm run test-unit-vitest-watch    # watch mode, both lanes
```

## Naming and file conventions

- **Lane** is chosen by filename suffix, no separate allowlist to maintain:
  `*.browser.js` -> browser lane, everything else -> node lane. Reserve the
  browser lane for tests that genuinely exercise `document`,
  `canvas`/`Image`/`createImageBitmap`, a real WebGL/WebGPU context, or
  similar — not just for files that import a class that touches those things
  in a code path the test doesn't exercise.
- **No `.test.js`/`.tests.js` suffix** — these already live under
  `test/unit/src/`, so the plain `.js` name matching the class under test
  (`Vector3.js`, `ImageBitmapLoader.browser.js`) is enough. The old QUnit
  files keep the `.tests.js` suffix so the two can coexist during migration.
- **Imports use the `@src`/`@test-utils` aliases**, not relative
  `../../../../` paths — the alias makes a converted file's imports correct
  regardless of how deeply nested it is, which matters once conversion is
  spread across many folders by different people/agents.
- **Don't add comments that just restate the next line** (`// Instancing`
  above `test( 'Instancing', ...)`, or a per-file comment re-explaining that
  `.browser.js` means "runs in a real browser" — that's already established
  by the naming convention, once, here). Keep comments that add information
  the code doesn't already say (e.g. "ensure that it is a true copy" above an
  assertion that isn't self-evidently about copy semantics).

## Findings from surveying all 234 existing `*.tests.js` files

Grepped every test file for `document.`, `window.`, `new Image(`,
`navigator.`, `createElementNS`, `HTMLCanvasElement`, `OffscreenCanvas`, and
separately for tests that import renderer/canvas-ish classes
(`WebGLRenderer`, `CanvasTexture`, `WebGLExtensions`, etc.) to see if they
exercise real GPU/DOM paths.

**Result: only 2 of 234 files actually need a browser today:**

| File | Needs browser? | Why |
|---|---|---|
| `audio/AudioContext.tests.js` | No — already has a Node fallback | Test file itself mocks `global.window.AudioContext` when `window` is undefined |
| `loaders/ImageBitmapLoader.tests.js` | **Yes** | Exercises real `createImageBitmap()` and `canvas.toDataURL()` |

Everything else that looked suspicious turned out not to need it:
- `renderers/WebGLRenderer.tests.js` is an empty stub (no body).
- `textures/CanvasTexture.tests.js` only checks inheritance/instancing, never
  calls `getContext()` or draws.
- `renderers/webgl/WebGLExtensions.tests.js` uses a hand-rolled mock GL
  context object, not a real one.

Real GPU rendering correctness is covered by the separate e2e screenshot
suite (`test/e2e/`), not the unit tests — so this isn't a gap the migration
introduces, it's how the suite already divides responsibility.

**Practical implication: the `unit-browser` project will likely stay small.**
Budget for it accordingly — most segments below should convert entirely to
the node lane, with maybe a file or two per segment moved to
`*.browser.js` after inspection, not before.

## File counts by segment (for delegating conversion work)

| Segment (`test/unit/src/...`) | Files | Expected browser-lane files |
|---|---:|---|
| `renderers/` (incl. `webgl/`, `shaders/`) | 30 | 0 (verify `webgl/` subfolder individually) |
| `math/` | 27 | 0 |
| `geometries/` | 21 | 0 |
| `extras/` | 21 | 0 |
| `materials/` | 18 | 0 |
| `core/` | 17 | 0 |
| `loaders/` | 16 | 1 (`ImageBitmapLoader`) |
| `lights/` | 14 | 0 |
| `animation/` | 14 | 0 |
| `objects/` | 13 | 0 |
| `helpers/` | 13 | 0 |
| `textures/` | 12 | 0 |
| `cameras/` | 6 | 0 |
| `audio/` | 5 | 0 (has Node fallback already) |
| `scenes/` | 3 | 0 |
| `nodes/` | 2 | 0 |
| top-level (`utils.tests.js`, `constants.tests.js`) | 2 | 0 |

`test/unit/addons/` (the addons suite, separate CI job) hasn't been surveyed
yet — same process applies before delegating it.

## Conversion playbook (for a per-segment agent)

For each `X.tests.js` in the assigned segment:

1. Create `X.js` next to it (do **not** delete or edit the `.tests.js`
   original, and do **not** remove its `import` line from
   `test/unit/three.source.unit.js` / `three.addons.unit.js` — the QUnit
   suite must keep passing untouched until a segment is fully verified and
   the team decides to cut over).
2. Mechanical rewrites:
   - `import { describe, test, expect } from 'vitest';` at the top.
   - Rewrite `../../../../src/...` imports to `@src/...`, and
     `../../utils/...` imports to `@test-utils/...`.
   - `QUnit.module( 'X', () => {` → `describe( 'X', () => {`; drop the
     `export default` on the outermost one.
   - `QUnit.test( 'name', ( assert ) => {` → `test( 'name', () => {` (drop
     the `assert` param).
   - `assert.ok( EXPR, 'msg' )` → `expect( EXPR ).toBeTruthy()`
   - `assert.strictEqual( A, B, 'msg' )` / `assert.equal( A, B, 'msg' )` →
     `expect( A ).toBe( B )`
   - `assert.deepEqual( A, B, 'msg' )` → `expect( A ).toEqual( B )`
   - `assert.notDeepEqual( A, B, 'msg' )` → `expect( A ).not.toEqual( B )`
   - `assert.notEqual( A, B, 'msg' )` → `expect( A ).not.toBe( B )`
   - `assert.throws( fn, /regex/, 'msg' )` → `expect( fn ).toThrow( /regex/ )`
   - `assert.numEqual/equalKey/smartEqual(...)` → `expect(...).toNumEqual/
     toEqualKey/toSmartEqual(...)` (matchers already provided by
     `test/unit/vitest-setup.js`, no logic changes needed)
   - Drop section comments that just restate the test name right below them
     (`// Instancing` above `test( 'Instancing', ...)`), and drop any
     console-suppression workaround that existed only because Node lacked a
     browser API you're now running against for real.
3. If (and only if) the test genuinely exercises a DOM/canvas/GPU API Node
   doesn't have: name it `X.browser.js` instead (see the `ImageBitmapLoader`
   example above).
4. Run `npx vitest run <file>` and diff test counts/names against the QUnit
   original to make sure nothing was silently dropped.
5. `npx eslint <file>` — should be clean with no config changes needed
   (`test/unit/**` is already covered by `npm run lint-test`).

None of this touches `src/`, `build/`, or CI — it's purely additive until a
segment is verified and the team chooses to retire the QUnit versions.
